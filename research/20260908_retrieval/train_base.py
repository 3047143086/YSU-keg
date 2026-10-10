"""Train a support-only BRIDGE checkpoint and export retrieval-screen inputs."""

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
                         encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ranking_metrics(ranking, targets):
    import numpy as np
    ranking = np.asarray(ranking)
    targets = np.asarray(targets)
    if ranking.ndim != 2 or targets.shape != (len(ranking),) or np.any(targets < 0):
        raise ValueError('Ranking and single-target arrays do not align')
    hits = ranking == targets[:, None]
    columns, metrics = [], {}
    for k in (10, 20):
        recall = hits[:, :k].sum(axis=1).astype(np.float64)
        ndcg = (hits[:, :k] / np.log2(np.arange(2, min(k, ranking.shape[1]) + 2))).sum(axis=1)
        columns.extend((recall, ndcg))
        metrics['recall@' + str(k)] = float(recall.mean())
        metrics['ndcg@' + str(k)] = float(ndcg.mean())
    return metrics, np.stack(columns, axis=1)


def validate_and_profile(support_users, support_items, target_users, target_items,
                         roles, features, n_users, n_items):
    import numpy as np
    support_users = np.asarray(support_users, dtype=np.int64)
    support_items = np.asarray(support_items, dtype=np.int64)
    target_users = np.asarray(target_users, dtype=np.int64)
    target_items = np.asarray(target_items, dtype=np.int64)
    if roles.shape != (n_users,) or not np.isin(roles, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid role vector')
    for users, items in ((support_users, support_items), (target_users, target_items)):
        if users.shape != items.shape or not len(users):
            raise ValueError('Empty or unaligned interaction arrays')
        if users.min() < 0 or users.max() >= n_users or items.min() < 0 or items.max() >= n_items:
            raise ValueError('Interaction outside declared catalog')
    if len(np.unique(target_users)) != len(target_users):
        raise ValueError('Expected exactly one held-out target per eligible user')
    targets = np.full(n_users, -1, dtype=np.int64)
    targets[target_users] = target_items
    if not np.array_equal(roles >= 0, targets >= 0):
        raise ValueError('Roles and eligible target users disagree')
    if not all(np.any(roles == role) for role in (0, 1, 2)):
        raise ValueError('Each role needs at least one eligible user')
    support_mask = np.zeros((n_users, n_items), dtype=np.bool_)
    support_mask[support_users, support_items] = True
    if support_mask[target_users, target_items].any():
        raise ValueError('Held-out target leaked into support')
    counts = support_mask.sum(axis=1)
    if np.any(counts[target_users] == 0):
        raise ValueError('An eligible user has no support history')
    if np.any(n_items - counts < min(200, n_items)):
        raise ValueError('Not enough unobserved items to export the candidate budget')
    features = np.asarray(features, dtype=np.float32)
    if features.shape != (n_items, 2, 512) or not np.isfinite(features).all():
        raise ValueError('Expected finite global_bank [items, 2, 512]')
    flat_features = features.reshape(n_items, 1024)
    sums = np.zeros((n_users, 1024), dtype=np.float32)
    # Collapse duplicate support pairs, so the profile and mask use the same history.
    for user in np.flatnonzero(counts):
        sums[user] = flat_features[support_mask[user]].mean(axis=0)
    sums /= np.maximum(np.linalg.norm(sums, axis=1, keepdims=True), 1e-12)
    return support_mask, targets, sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu-id', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[variable] = '4'
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parent
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'result.json').exists() or (run / 'config.json').exists():
        raise RuntimeError('Refusing to overwrite an existing base run')
    started = time.monotonic()
    atomic_json(run / 'progress.json', {'status': 'initializing', 'variant': 'base',
                                      'seed': args.seed, 'official_test_evaluated': False})
    import numpy as np
    import torch

    spec = json.loads((root / 'experiment.json').read_text(encoding='utf-8'))
    manifest = json.loads((root / 'split_manifest.json').read_text(encoding='utf-8'))
    n_users, n_items = int(manifest['users']), int(manifest['items'])
    if n_items < 200:
        raise ValueError('The screen contract requires 200 candidates')
    roles = np.load(root / 'roles.npy', allow_pickle=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Expected an assigned CUDA GPU; refusing an accidental CPU run')
    torch.cuda.set_per_process_memory_fraction(float(spec['gpu_memory_fraction']), 0)
    snapshot = root / 'bridge_src'
    os.chdir(snapshot)
    sys.path.insert(0, str(snapshot))
    from common.trainer import Trainer
    from diagnose_ecli import scalarize_hyperparameters
    from models.bridge import BRIDGE
    from models.dual_frequency_encoder import DualFrequencyEncoder
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed

    data = run / 'data' / 'baby'
    data.mkdir(parents=True, exist_ok=False)
    for name in ('image_feat.npy', 'text_feat.npy'):
        (data / name).symlink_to(root / 'features' / name)
    (data / 'baby.inter').symlink_to(root / 'split' / 'baby.inter')
    config = Config('BRIDGE', 'baby', {
        'data_path': str(data.parent) + '/', 'seed': [args.seed], 'gpu_id': args.gpu_id,
        'epochs': spec['base_epochs'], 'learning_rate': [spec['base_learning_rate']],
        'train_batch_size': spec['base_train_batch_size'],
        'eval_batch_size': spec.get('base_eval_batch_size', 64),
        'save_recommended_topk': False, 'topk': [10, 20], 'metrics': ['Recall', 'NDCG'],
        'valid_metric': 'NDCG@20', 'checkpoint_dir': str(run / 'checkpoints'),
    })
    scalarize_hyperparameters(config)
    config['seed'] = args.seed
    init_seed(args.seed)
    dataset = RecDataset(config)
    dataset.item_num, dataset.user_num = n_items, n_users
    train_dataset, heldout_dataset, empty_test = dataset.split()
    if len(empty_test) or len(train_dataset) == 0 or len(heldout_dataset) == 0:
        raise RuntimeError('Expected support and one internal holdout, without test labels')
    uid, iid = dataset.uid_field, dataset.iid_field
    support_mask, targets, history = validate_and_profile(
        train_dataset.df[uid].to_numpy(), train_dataset.df[iid].to_numpy(),
        heldout_dataset.df[uid].to_numpy(), heldout_dataset.df[iid].to_numpy(), roles,
        np.load(root / 'features' / 'global_bank.npy', allow_pickle=False), n_users, n_items)
    calibration_frame = heldout_dataset.df.loc[
        roles[heldout_dataset.df[uid].to_numpy(dtype=np.int64)] == 1].copy()
    calibration_dataset = heldout_dataset.copy(calibration_frame)
    # Legacy RecDataset otherwise creates this required loader field only in __str__.
    train_dataset.inter_num = len(train_dataset)
    calibration_dataset.inter_num = len(calibration_dataset)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    calibration = EvalDataLoader(config, calibration_dataset, additional_dataset=train_dataset,
                                 batch_size=config['eval_batch_size'])
    if not np.all(roles[calibration.get_eval_users().numpy()] == 1):
        raise RuntimeError('Checkpoint calibration contains a non-calibration user')
    train.pretrain_setup()
    model = BRIDGE(config, train).to(config['device'])
    trainer = Trainer(config, model)
    source_paths = {
        'train_base.py': root / 'train_base.py', 'experiment.json': root / 'experiment.json',
        'split_manifest.json': root / 'split_manifest.json', 'roles.npy': root / 'roles.npy',
        'split/baby.inter': root / 'split' / 'baby.inter',
        'features/global_bank.npy': root / 'features' / 'global_bank.npy',
        'features/image_feat.npy': root / 'features' / 'image_feat.npy',
        'features/text_feat.npy': root / 'features' / 'text_feat.npy',
    }
    for source in sorted(snapshot.rglob('*')):
        if source.is_file() and source.suffix in ('.py', '.yaml', '.yml'):
            source_paths['bridge_src/' + source.relative_to(snapshot).as_posix()] = source
    hashes = {name: sha256(path) for name, path in source_paths.items()}
    atomic_json(run / 'config.json', {
        'variant': 'base', 'seed': args.seed, 'experiment': spec, 'split': manifest,
        'config': {key: str(value) if isinstance(value, torch.device) else value
                   for key, value in config.final_config_dict.items()},
        'source_sha256': hashes, 'checkpoint_selection_role': 1,
        'support_interactions': int(support_mask.sum()),
        'role_counts': {str(role): int((roles == role).sum()) for role in (-1, 0, 1, 2)},
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'official_validation_evaluated': False, 'official_test_evaluated': False,
    })

    def checked_loss(interaction):
        loss = model.calculate_loss(interaction)
        terms = loss if isinstance(loss, tuple) else (loss,)
        if not all(torch.isfinite(term).all() for term in terms):
            raise RuntimeError('Nonfinite training loss')
        return loss

    @torch.no_grad()
    def evaluate(limit=None):
        model.eval()
        rankings, users_seen = [], []
        for batch_index, interaction in enumerate(calibration):
            scores = model.full_sort_predict(interaction)
            if not torch.isfinite(scores).all():
                raise RuntimeError('Nonfinite base prediction before masking')
            hidden = interaction[1]
            scores[hidden[0], hidden[1]] = -torch.inf
            rankings.append(torch.topk(scores, 20, dim=1).indices.cpu().numpy())
            users_seen.append(interaction[0].cpu().numpy())
            if limit is not None and batch_index + 1 >= limit:
                return {'batches_checked': batch_index + 1}, None, None
        users_seen = np.concatenate(users_seen)
        if not np.array_equal(users_seen, calibration.get_eval_users().numpy()):
            raise RuntimeError('Calibration user order mismatch')
        if not np.all(roles[users_seen] == 1):
            raise RuntimeError('Only role1 may select the base checkpoint')
        metrics, per_user = ranking_metrics(np.concatenate(rankings), targets[users_seen])
        return metrics, per_user, users_seen

    if args.smoke:
        probe = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        before = probe.detach().clone()
        model.pre_epoch_processing()
        loss_sum, loss_batches = trainer._train_epoch(itertools.islice(train, 2), 0,
                                                      loss_func=checked_loss)
        if len(loss_batches) != 2 or torch.equal(before, probe.detach()):
            raise RuntimeError('Two smoke batches did not update base parameters')
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise RuntimeError('Nonfinite gradient: ' + name)
        checks, _, _ = evaluate(limit=1)
        checks['train_batches_checked'] = 2
        checks['base_parameter_update_verified'] = True
        result = {'status': 'smoke_passed', 'variant': 'base', 'seed': args.seed,
                  'not_a_performance_result': True, 'checks': checks,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                  'elapsed_seconds': time.monotonic() - started}
        atomic_json(run / 'result.json', result)
        atomic_json(run / 'progress.json', result)
        print('BASE_SMOKE_PASSED', json.dumps(result), flush=True)
        return

    best, stale, best_record = -float('inf'), 0, None
    for epoch in range(1, int(spec['base_epochs']) + 1):
        epoch_started = time.monotonic()
        model.pre_epoch_processing()
        loss_sum, loss_batches = trainer._train_epoch(train, epoch - 1, loss_func=checked_loss)
        trainer.lr_scheduler.step()
        model.post_epoch_processing()
        if not len(loss_batches):
            raise RuntimeError('The base epoch had no training batches')
        mean_loss = float(sum(loss_sum) if isinstance(loss_sum, tuple) else loss_sum) / len(loss_batches)
        del loss_batches
        if not np.isfinite(mean_loss):
            raise RuntimeError('Nonfinite epoch mean loss')
        progress = {'status': 'training', 'variant': 'base', 'seed': args.seed,
                    'epoch': epoch, 'mean_train_loss': mean_loss,
                    'epoch_train_seconds': time.monotonic() - epoch_started,
                    'elapsed_seconds': time.monotonic() - started,
                    'checkpoint_selection_role': 1, 'role2_evaluated': False,
                    'official_validation_evaluated': False, 'official_test_evaluated': False}
        if epoch == 1 or epoch % int(spec['base_eval_every']) == 0:
            metrics, per_user, users = evaluate()
            progress['calibration_metrics'] = metrics
            if metrics['ndcg@20'] > best:
                best, stale = metrics['ndcg@20'], 0
                best_record = {'epoch': epoch, 'calibration_metrics': metrics}
                temporary = run / 'best.pth.tmp'
                torch.save(model.state_dict(), temporary)
                temporary.replace(run / 'best.pth')
                np.savez_compressed(run / 'best_calibration_per_user.npz', users=users, metrics=per_user)
                atomic_json(run / 'best.json', best_record)
            else:
                stale += 1
        progress['best'] = best_record
        progress['stale_evaluations'] = stale
        atomic_json(run / 'progress.json', progress)
        with (run / 'epochs.jsonl').open('a', encoding='utf-8') as output:
            output.write(json.dumps(progress, allow_nan=False) + '\n')
        print(json.dumps(progress), flush=True)
        if stale >= int(spec['base_patience_evaluations']):
            break
    if best_record is None:
        raise RuntimeError('No selected base checkpoint')
    model.load_state_dict(torch.load(run / 'best.pth', map_location=config['device']))
    model.eval()
    progress = {'status': 'exporting', 'variant': 'base', 'seed': args.seed,
                'exported_users': 0, 'total_users': n_users, 'role2_evaluated': False}
    atomic_json(run / 'progress.json', progress)
    with torch.no_grad():
        raw_user, item_emb, _, _, _ = DualFrequencyEncoder.forward(model)
        native_user, _, _ = model._apply_behavior_proto(raw_user, item_emb)
        base_user = native_user.cpu().numpy().astype(np.float32)
        if not np.isfinite(base_user).all() or base_user.shape[0] != n_users:
            raise RuntimeError('Invalid native user embeddings')
        np.save(run / 'base_user.npy', base_user, allow_pickle=False)
        del raw_user, item_emb, native_user
        score_path = run / 'base_scores.npy.tmp'
        scores_export = np.lib.format.open_memmap(score_path, mode='w+', dtype=np.float32,
                                                 shape=(n_users, n_items))
        top200 = np.empty((n_users, 200), dtype=np.int64)
        batch_size = int(config['eval_batch_size'])
        for first in range(0, n_users, batch_size):
            last = min(first + batch_size, n_users)
            users = torch.arange(first, last, device=config['device'])
            scores = model.full_sort_predict([users])
            if scores.shape != (last - first, n_items) or not torch.isfinite(scores).all():
                raise RuntimeError('Nonfinite or malformed full-catalog export')
            scores_export[first:last] = scores.cpu().numpy()
            mask = torch.from_numpy(support_mask[first:last]).to(config['device'])
            scores.masked_fill_(mask, -torch.inf)
            top200[first:last] = torch.topk(scores, 200, dim=1).indices.cpu().numpy()
            progress['exported_users'] = last
            if first == 0 or last == n_users or first % (batch_size * 20) == 0:
                progress['elapsed_seconds'] = time.monotonic() - started
                atomic_json(run / 'progress.json', progress)
        scores_export.flush()
        del scores_export
        score_path.replace(run / 'base_scores.npy')
    if support_mask[np.arange(n_users)[:, None], top200].any():
        raise RuntimeError('Support item appeared in exported candidate ranking')
    calibration_users = np.flatnonzero(roles == 1)
    exported_calibration, _ = ranking_metrics(top200[calibration_users, :20],
                                               targets[calibration_users])
    for name, value in best_record['calibration_metrics'].items():
        if abs(exported_calibration[name] - value) > 1e-8:
            raise RuntimeError('Exported checkpoint does not reproduce calibration: ' + name)
    for name, value in (('base_top200', top200), ('history', history),
                        ('support_mask', support_mask), ('targets', targets), ('roles', roles)):
        np.save(run / (name + '.npy'), value, allow_pickle=False)
    artifact_names = ('base_scores', 'base_top200', 'base_user', 'history',
                      'support_mask', 'targets', 'roles')
    result = {'status': 'completed', 'variant': 'base', 'seed': args.seed,
              'best': best_record, 'epochs': epoch, 'selected_budget': 200,
              'calibration_metrics': best_record['calibration_metrics'],
              'exported_calibration_verified': True,
              'dev_metrics': None, 'role2_evaluated': False,
              'elapsed_seconds': time.monotonic() - started,
              'checkpoint_sha256': sha256(run / 'best.pth'),
              'artifacts': {name: {'path': str(run / (name + '.npy')),
                                   'sha256': sha256(run / (name + '.npy'))}
                            for name in artifact_names},
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
              'official_validation_evaluated': False, 'official_test_evaluated': False,
              'stage': 'support_only_base_for_internal_retrieval_screen'}
    atomic_json(run / 'result.json', result)
    atomic_json(run / 'progress.json', result)
    print('BASE_COMPLETED', json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if '--output' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--output') + 1]).resolve()
            if output.is_dir():
                failure = {'status': 'failed', 'variant': 'base', 'error': str(exc),
                           'traceback': traceback.format_exc()}
                atomic_json(output / 'failure.json', failure)
                atomic_json(output / 'progress.json', failure)
        raise
