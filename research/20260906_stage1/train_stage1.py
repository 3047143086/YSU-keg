"""Train BRIDGE evidence controls without reading official evaluation labels."""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['global', 'local', 'shuffled'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    root = Path(__file__).resolve().parent
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'result.json').exists():
        raise RuntimeError('This run already has a result')
    import numpy as np
    import torch
    from torch.nn import functional as F
    from stage1_core import EvidenceMatcher, atomic_json, leave_one_out_profile, sha256

    spec = json.loads((root / 'experiment.json').read_text())
    split = json.loads((root / 'split_manifest.json').read_text())
    snapshot = root / 'bridge_src'
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    from common.trainer import Trainer
    from diagnose_ecli import scalarize_hyperparameters
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.ecli import per_user_ranking_metrics
    from utils.utils import init_seed

    data = run / 'data' / 'baby'
    data.mkdir(parents=True, exist_ok=False)
    for name in ('image_feat.npy', 'text_feat.npy'):
        (data / name).symlink_to(root / 'features' / name)
    (data / 'baby.inter').symlink_to(root / 'split' / 'baby.inter')
    config = Config('BRIDGE', 'baby', {
        'data_path': str(data.parent) + '/', 'seed': [args.seed], 'gpu_id': args.gpu_id,
        'epochs': spec['epochs'], 'learning_rate': [spec['learning_rate']],
        'train_batch_size': spec['train_batch_size'], 'eval_batch_size': spec['eval_batch_size'],
        'save_recommended_topk': False, 'topk': [10, 20], 'metrics': ['Recall', 'NDCG'],
        'valid_metric': 'NDCG@20',
    })
    scalarize_hyperparameters(config)
    config['seed'] = args.seed
    if not (config['use_topk_correction'] and config['correction_scope'] == 'topk'
            and config['behavior_eval_topk'] > 0 and config['use_behavior_correction']
            and config['behavior_weight'] != 0):
        raise RuntimeError('BRIDGE configuration bypasses the evidence inference hook')
    init_seed(args.seed)
    torch.set_num_threads(4)
    dataset = RecDataset(config)
    dataset.item_num = split['items']
    dataset.user_num = split['users']
    train_dataset, dev_dataset, empty_test = dataset.split()
    if len(empty_test):
        raise RuntimeError('Stage-one dataset must not contain official test data')
    print(str(train_dataset), str(dev_dataset), flush=True)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    dev = EvalDataLoader(config, dev_dataset, additional_dataset=train_dataset,
                         batch_size=config['eval_batch_size'])
    train.pretrain_setup()
    arrays = {name: np.load(root / 'features' / (name + '.npy')) for name in
              ('global_bank', 'global_mask', 'local_bank', 'local_mask')}

    class EvidenceBRIDGE(BRIDGE):
        def __init__(self):
            super().__init__(config, train)
            self.evidence_inference_calls = 0
            self.matcher = EvidenceMatcher(arrays['global_bank'], arrays['global_mask'],
                                           arrays['local_bank'], arrays['local_mask'], args.variant,
                                           spec['shuffle_seed'])
            features = torch.tensor(arrays['global_bank'].reshape(self.n_items, -1), dtype=torch.float32)
            sums = torch.zeros((self.n_users, features.shape[1]), dtype=torch.float32)
            user_ids = torch.tensor(train_dataset.df.userID.to_numpy(), dtype=torch.long)
            item_ids = torch.tensor(train_dataset.df.itemID.to_numpy(), dtype=torch.long)
            sums.index_add_(0, user_ids, features[item_ids])
            counts = torch.bincount(user_ids, minlength=self.n_users).float()
            self.register_buffer('content_features', features, persistent=False)
            self.register_buffer('history_sums', sums, persistent=False)
            self.register_buffer('history_counts', counts, persistent=False)

        def profile(self, users, excluded_items=None):
            return leave_one_out_profile(self.history_sums, self.history_counts,
                                         self.content_features, users, excluded_items)

        def _route_pair_scores(self, users, pos_items, neg_items, pos_base, neg_base,
                               user_emb, item_emb_all, pos_candidate_mask=None, neg_candidate_mask=None):
            # Both alternatives use the same history with the target removed.
            profile = self.profile(users, pos_items)
            pos = self.matcher(profile, pos_items)
            neg = self.matcher(profile, neg_items)
            auxiliary = spec['content_aux_weight'] * F.softplus(-(pos - neg)).mean()
            pos_mask = 1.0 if pos_candidate_mask is None else pos_candidate_mask
            neg_mask = 1.0 if neg_candidate_mask is None else neg_candidate_mask
            return (pos_base + spec['content_scale'] * pos * pos_mask,
                    neg_base + spec['content_scale'] * neg * neg_mask, auxiliary)

        def _route_candidate_scores(self, users, candidate_idx, candidate_base, user_emb, item_emb_all):
            self.evidence_inference_calls += 1
            return candidate_base + spec['content_scale'] * self.matcher(self.profile(users), candidate_idx)

    started = time.monotonic()
    model = EvidenceBRIDGE().to(config['device'])
    trainer = Trainer(config, model)
    config_record = {key: str(value) if isinstance(value, torch.device) else value
                     for key, value in config.final_config_dict.items()}
    atomic_json(run / 'config.json', {'variant': args.variant, 'seed': args.seed,
                'config': config_record, 'experiment': spec, 'split': split,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'matcher_parameters': sum(p.numel() for p in model.matcher.parameters()),
                'code_sha256': {name: sha256(root / name) for name in ('train_stage1.py', 'stage1_core.py')},
                'official_test_evaluated': False})

    def checked_loss(interaction):
        loss = model.calculate_loss(interaction)
        if not torch.isfinite(loss).all():
            raise RuntimeError('Nonfinite training loss')
        return loss

    @torch.no_grad()
    def evaluate(limit=None):
        model.eval()
        ranking = []
        for index, interaction in enumerate(dev):
            scores = model.full_sort_predict(interaction)
            if not torch.isfinite(scores).all():
                raise RuntimeError('Nonfinite prediction before masking')
            hidden = interaction[1]
            scores[hidden[0], hidden[1]] = -torch.inf
            ranking.append(torch.topk(scores, 20, dim=-1).indices.cpu().numpy())
            if limit is not None and index + 1 >= limit:
                return {'batches_checked': index + 1}, None, None
        ranking = np.concatenate(ranking)
        positives = dev.get_eval_items()
        lengths = np.asarray(dev.get_eval_len_list())
        if len(positives) != len(ranking):
            raise RuntimeError('Evaluation user order mismatch')
        hits = np.asarray([np.isin(row, truth) for row, truth in zip(ranking, positives)])
        columns, metrics = [], {}
        for k in (10, 20):
            recall, ndcg = per_user_ranking_metrics(hits, lengths, k)
            columns.extend((recall, ndcg))
            metrics['recall@' + str(k)] = float(recall.mean())
            metrics['ndcg@' + str(k)] = float(ndcg.mean())
        return metrics, np.stack(columns, axis=1), dev.get_eval_users().cpu().numpy()

    if args.smoke:
        before = model.matcher.scorer[-1].weight.detach().clone()
        model.pre_epoch_processing()
        trainer._train_epoch(itertools.islice(train, 2), 0, loss_func=checked_loss)
        if torch.equal(before, model.matcher.scorer[-1].weight.detach()):
            raise RuntimeError('Matcher parameters did not update in the smoke test')
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Nonfinite gradient: ' + name)
        metrics, _, _ = evaluate(limit=1)
        if not model.evidence_inference_calls:
            raise RuntimeError('Evidence inference hook was not called')
        metrics['evidence_inference_calls'] = model.evidence_inference_calls
        metrics['matcher_parameter_update_verified'] = True
        atomic_json(run / 'result.json', {'status': 'smoke_passed', 'not_a_performance_result': True,
                    'checks': metrics, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
        print('SMOKE_PASSED', flush=True)
        return

    best, stale, best_record = -float('inf'), 0, None
    for epoch in range(1, spec['epochs'] + 1):
        epoch_started = time.monotonic()
        model.pre_epoch_processing()
        loss_sum, loss_batches = trainer._train_epoch(train, epoch - 1, loss_func=checked_loss)
        trainer.lr_scheduler.step()
        model.post_epoch_processing()
        mean_loss = float(sum(loss_sum) if isinstance(loss_sum, tuple) else loss_sum) / len(loss_batches)
        if not np.isfinite(mean_loss):
            raise RuntimeError('Nonfinite epoch loss')
        progress = {'status': 'training', 'variant': args.variant, 'seed': args.seed,
                    'epoch': epoch, 'mean_train_loss': mean_loss,
                    'epoch_train_seconds': time.monotonic() - epoch_started,
                    'elapsed_seconds': time.monotonic() - started,
                    'official_test_evaluated': False}
        del loss_batches
        if epoch == 1 or epoch % spec['eval_every'] == 0:
            metrics, per_user, users = evaluate()
            progress['dev_metrics'] = metrics
            if metrics['ndcg@20'] > best:
                best, stale = metrics['ndcg@20'], 0
                best_record = {'epoch': epoch, 'metrics': metrics}
                temporary = run / 'best.pth.tmp'
                torch.save(model.state_dict(), temporary)
                temporary.replace(run / 'best.pth')
                np.savez_compressed(run / 'best_per_user.npz', users=users, metrics=per_user)
                atomic_json(run / 'best.json', best_record)
            else:
                stale += 1
        progress['best'] = best_record
        progress['stale_evaluations'] = stale
        atomic_json(run / 'progress.json', progress)
        with (run / 'epochs.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(progress) + '\n')
        print(json.dumps(progress), flush=True)
        if stale >= spec['patience_evaluations']:
            break
    result = {'status': 'completed', 'variant': args.variant, 'seed': args.seed,
              'best': best_record, 'epochs': epoch, 'elapsed_seconds': time.monotonic() - started,
              'official_test_evaluated': False, 'official_validation_evaluated': False,
              'stage': 'information_screen_not_full_RAVE',
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_sha256': sha256(run / 'best.pth')}
    atomic_json(run / 'result.json', result)
    atomic_json(run / 'progress.json', result)
    print('RUN_COMPLETED', json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1])
            if output.is_dir():
                (output / 'failure.json').write_text(json.dumps({'error': str(exc)}))
        raise
