"""Fresh, single-seed BRIDGE training with validation-only checkpoint selection."""
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, default=str, allow_nan=False)


class Selection:
    """Match the legacy validation rounding, strict improvement, and patience."""
    def __init__(self, patience=20):
        if patience < 0:
            raise ValueError('patience must be nonnegative')
        self.patience, self.bad_epochs = patience, 0
        self.best_score, self.best_epoch = -1.0, None

    def update(self, score, epoch):
        if not math.isfinite(float(score)):
            raise FloatingPointError('Non-finite validation selection score')
        score = round(float(score), 4)
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs > self.patience


def training_loop(trainer, train, valid, max_epochs, patience, loss_func, save, emit):
    """Keep this orchestration independent of Torch so the protocol can be tested."""
    selection = Selection(patience)
    for epoch in range(max_epochs):
        start = time.monotonic()
        trainer.model.pre_epoch_processing()
        losses, _ = trainer._train_epoch(train, epoch, loss_func=loss_func)
        components = losses if isinstance(losses, (tuple, list)) else [losses]
        losses = [float(value) for value in components]
        if not all(math.isfinite(value) for value in losses):
            raise FloatingPointError('Non-finite epoch loss at epoch ' + str(epoch))
        trainer.lr_scheduler.step()
        post_info = trainer.model.post_epoch_processing()
        diagnostics = None
        if hasattr(trainer.model, 'train_diagnostics'):
            diagnostics = trainer.model.train_diagnostics()
            # Diagnostics must be real JSON values; do not stringify tensors or NaN.
            json.dumps(diagnostics, allow_nan=False)
        metrics = {key.lower(): float(value) for key, value in trainer.evaluate(valid).items()}
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError('Non-finite validation metrics')
        improved, stop = selection.update(metrics['recall@20'], epoch)
        if improved:
            save(epoch, selection.best_score, metrics)
        emit({'epoch': epoch, 'loss_components': losses, 'validation': metrics,
              'improved': improved, 'best_epoch': selection.best_epoch,
              'best_score': selection.best_score, 'bad_epochs': selection.bad_epochs,
              'early_stop': stop, 'elapsed_seconds': time.monotonic() - start,
              'learning_rate': trainer.optimizer.param_groups[0]['lr'], 'post_info': post_info,
              'train_diagnostics': diagnostics})
        if stop:
            break
    if selection.best_epoch is None:
        raise RuntimeError('No validation checkpoint was selected')
    return selection, epoch + 1


def split_train_valid(dataset, config):
    """Preserve the supplied split, without constructing the test dataset."""
    field = config['inter_splitting_label']
    train_df = dataset.df.loc[dataset.df[field] == 0].drop(columns=[field]).copy()
    valid_df = dataset.df.loc[dataset.df[field] == 1].drop(columns=[field]).copy()
    if config['filter_out_cod_start_users']:
        valid_df = valid_df.loc[valid_df[dataset.uid_field].isin(train_df[dataset.uid_field])].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError('Empty train/validation split')
    train, valid = dataset.copy(train_df), dataset.copy(valid_df)
    str(train), str(valid)  # Legacy dataloader requires __str__ to initialize inter_num.
    return train, valid


def scalarize(config):
    for key in config['hyper_parameters']:
        value = config[key]
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError('A single frozen value is required for ' + key)
            config[key] = value[0]


def assert_baseline(model):
    for key in ('use_activity_simplex', 'use_behavior_proto', 'use_private_residual',
                'use_residual_gate', 'use_user_graph'):
        if getattr(model, key):
            raise RuntimeError('Unexpected baseline branch: ' + key)
    if model.msr_weight != 0:
        raise RuntimeError('Baseline requires msr_weight=0')


def final_validation(model, valid, output):
    import numpy as np
    import torch
    from utils.ecli import per_user_ranking_metrics
    model.eval()
    rankings = []
    with torch.no_grad():
        for interaction in valid:
            scores = model.full_sort_predict(interaction)
            mask = interaction[1]
            scores[mask[0], mask[1]] = -1e10  # Match Trainer.evaluate exactly.
            rankings.append(torch.topk(scores, k=50, dim=1).indices.cpu().numpy())
    ranking = np.concatenate(rankings)
    truth, lengths = valid.get_eval_items(), np.asarray(valid.get_eval_len_list())
    if len(ranking) != len(truth) or (lengths <= 0).any():
        raise RuntimeError('Invalid validation user order or relevance count')
    hits = np.array([np.isin(items, positives) for items, positives in zip(ranking, truth)])
    per_user = {'user_ids': np.asarray(valid.get_eval_users()), 'positive_lengths': lengths}
    metrics = {}
    for cutoff in (5, 10, 20, 50):
        recall, ndcg = per_user_ranking_metrics(hits, lengths, cutoff)
        for name, values in (('recall', recall), ('ndcg', ndcg)):
            per_user[name + '_' + str(cutoff)] = values
            metrics[name + '@' + str(cutoff)] = float(values.mean())
    np.savez_compressed(output / 'per_user_validation.npz', **per_user)
    return metrics, len(ranking)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bridge-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('baby', 'sports', 'elec'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--gpu-id', default='1')
    parser.add_argument('--cuda-memory-fraction', type=float, default=0.55)
    parser.add_argument('--plugin-module')
    parser.add_argument('--plugin-class')
    parser.add_argument('--plugin-config', type=Path)
    parser.add_argument('--max-epochs', type=int, default=300)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if bool(args.plugin_module) != bool(args.plugin_class) or (args.plugin_config and not args.plugin_module):
        parser.error('plugin module/class are required together; plugin config requires both')
    if args.smoke:
        args.max_epochs = 1
    if not 1 <= args.max_epochs <= 300:
        parser.error('max-epochs must be between 1 and 300')
    if args.train_batch_size is not None and args.train_batch_size <= 0:
        parser.error('train-batch-size must be positive')
    if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('learning-rate must be finite and positive')
    if not math.isfinite(args.cuda_memory_fraction) or not 0 < args.cuda_memory_fraction <= 1:
        parser.error('cuda-memory-fraction must be finite and in (0, 1]')
    root, output, bundle = args.bridge_root.resolve(), args.output.resolve(), Path(__file__).resolve().parent
    plugin_path = args.plugin_config.resolve() if args.plugin_config else None
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id) if args.device == 'cuda' else ''
    os.chdir(root / 'src')
    sys.path[:0] = [str(bundle), str(root / 'src')]
    import numpy as np
    import torch
    from common.trainer import Trainer
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed
    torch.set_num_threads(4)
    config = Config('BRIDGE', args.dataset, {'use_gpu': args.device == 'cuda', 'gpu_id': args.gpu_id,
                    'cuda_memory_fraction': args.cuda_memory_fraction,
                    'seed': [args.seed], 'epochs': args.max_epochs, 'stopping_step': 20,
                    'eval_step': 1, 'eval_batch_size': 64, 'save_recommended_topk': False,
                    'req_training': True, 'valid_metric': 'Recall@20'})
    scalarize(config)
    plugin_config = json.loads(plugin_path.read_text()) if plugin_path else {}
    # Plugins may add controls, but cannot silently change any established baseline setting.
    if not isinstance(plugin_config, dict) or set(plugin_config) & set(config.final_config_dict):
        raise ValueError('Plugin configuration must contain only new model-specific keys')
    config.final_config_dict.update(plugin_config)
    if args.train_batch_size is not None:
        config['train_batch_size'] = args.train_batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    if str(config['device']) != args.device:
        raise RuntimeError('Requested device is unavailable: ' + args.device)
    if args.device == 'cuda':
        # CUDA_VISIBLE_DEVICES selects the physical GPU; its logical index is 0.
        # Torch 1.12 rejects an unindexed torch.device('cuda') in this API.
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device=0)
    if config['feature_ablation'] != 'all' or not config['use_item_graph']:
        raise RuntimeError('This experiment requires original features and the existing item graph')
    data = Path(config['data_path'] + args.dataset).resolve()
    graph = data / ('mm_adj_' + str(config['knn_k']) + '.pt')
    if not graph.is_file():
        raise RuntimeError('Existing graph cache required; shared data must not be modified')
    paths = [Path(__file__).resolve(), graph]
    paths += [root / 'src' / p for p in ('models/bridge.py', 'models/dual_frequency_encoder.py',
              'common/trainer.py', 'common/abstract_recommender.py', 'utils/configurator.py',
              'utils/dataset.py', 'utils/dataloader.py', 'utils/ecli.py', 'utils/utils.py',
              'utils/topk_evaluator.py', 'utils/metrics.py', 'configs/overall.yaml',
              'configs/model/BRIDGE.yaml', 'configs/dataset/' + args.dataset + '.yaml')]
    paths += [data / config[key] for key in ('inter_file_name', 'vision_feature_file', 'text_feature_file')]
    optional = [data / config[key] for key in ('user_emb_file', 'user_graph_dict_file')]
    paths += [path for path in optional if path.exists()]
    absent = [path for path in optional if not path.exists()]
    model_class = BRIDGE
    if args.plugin_module:
        module = importlib.import_module(args.plugin_module)
        if Path(module.__file__).resolve().parent != bundle:
            raise RuntimeError('Plugin must be located beside this runner')
        model_class = getattr(module, args.plugin_class)
        if not issubclass(model_class, BRIDGE):
            raise TypeError('Plugin class must subclass BRIDGE')
        paths.append(Path(module.__file__).resolve())
    if plugin_path:
        paths.append(plugin_path)
    if args.plugin_module:
        paths += list(bundle.glob('*.py'))
        if 'evidence_cache' in plugin_config:
            paths.append(Path(plugin_config['evidence_cache']).resolve())
    inputs = {str(path): digest(path) for path in paths}
    write_json(output / 'inputs.json', {'sha256': inputs, 'absent': list(map(str, absent))})
    write_json(output / 'config.json', {'effective_config': config.final_config_dict,
               'arguments': vars(args), 'command': sys.argv, 'fresh_initialization': True,
               'test_evaluated': False, 'test_loader_constructed': False,
               'budget_overridden': args.max_epochs != 300 or args.train_batch_size is not None
                                    or args.learning_rate is not None,
               'torch_version': torch.__version__, 'numpy_version': np.__version__})
    init_seed(args.seed)
    dataset = RecDataset(config)
    train_dataset, valid_dataset = split_train_valid(dataset, config)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    train.pretrain_setup()
    model = model_class(config, train).to(config['device'])
    if not args.plugin_module:
        assert_baseline(model)
    trainer = Trainer(config, model)
    original_predict = model.full_sort_predict

    def checked_predict(interaction):
        scores = original_predict(interaction)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Non-finite unmasked validation scores')
        return scores

    model.full_sort_predict = checked_predict

    def checked_loss(interaction):
        losses = model.calculate_loss(interaction)
        for value in losses if isinstance(losses, tuple) else [losses]:
            if not torch.isfinite(value).all():
                raise FloatingPointError('Non-finite training loss before backward')
        return losses

    def save(epoch, score, metrics):
        torch.save({'model': model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                    'scheduler': trainer.lr_scheduler.state_dict(), 'epoch': epoch,
                    'selection_score': score, 'validation': metrics,
                    'config': config.final_config_dict, 'inputs_sha256': inputs}, output / 'best.pending.pt')
        os.replace(output / 'best.pending.pt', output / 'best.pt')

    with (output / 'progress.jsonl').open('x', encoding='utf-8') as log:
        def emit(row):
            row['unix'] = time.time()
            line = json.dumps(row, default=str, allow_nan=False)
            log.write(line + '\n')
            log.flush()
            print(line, flush=True)
        selection, epochs = training_loop(trainer, train, valid, args.max_epochs, 20, checked_loss, save, emit)
    # Avoid duplicating optimizer state on the GPU during final confirmation.
    best = torch.load(output / 'best.pt', map_location='cpu')
    model.load_state_dict(best['model'], strict=True)
    del best
    metrics, users = final_validation(model, valid, output)
    if abs(metrics['recall@20'] - selection.best_score) > 0.000051:
        raise RuntimeError('Best-checkpoint validation does not reproduce its rounded selection score')
    if any(digest(path) != sha for path, sha in inputs.items()) or any(path.exists() for path in absent):
        raise RuntimeError('Training inputs changed during the run')
    write_json(output / 'result.json', {'status': 'completed', 'dataset': args.dataset, 'seed': args.seed,
               'model': args.plugin_class or 'BRIDGE', 'best_epoch': selection.best_epoch,
               'epochs_completed': epochs, 'selection_metric': 'rounded_4_validation_recall@20',
               'selection_score': selection.best_score, 'metrics': metrics, 'validation_users': users,
               'train_interactions': len(train_dataset), 'validation_interactions': len(valid_dataset),
               'test_evaluated': False, 'test_loader_constructed': False, 'inputs_unchanged': True,
               'fresh_initialization': True, 'checkpoint_sha256': digest(output / 'best.pt'),
               'per_user_sha256': digest(output / 'per_user_validation.npz'),
               'elapsed_seconds': time.monotonic() - start})
    print('COMPLETED ' + str(output), flush=True)


if __name__ == '__main__':
    main()
