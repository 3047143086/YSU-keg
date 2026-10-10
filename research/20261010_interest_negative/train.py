"""Fresh official MSCA, private negative sampling and validation-only selection.

The original carrier model, Trainer and evaluator are not edited. The small
selection/split orchestration follows the already audited October 9 runner.
"""
import argparse
import hashlib
import json
import math
import os
import pickle
import random
import sys
import time
import traceback
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    os.replace(str(temporary), str(path))


def protected_paths(root):
    manifest = json.loads((root / 'carrier_manifest.json').read_text())
    paths = {str(root / 'carrier/src' / name): sha
             for name, sha in manifest['carrier_sha256'].items()}
    for name, row in manifest['inputs'].items():
        if row['exists']:
            paths[str(root / 'carrier/data' / name)] = row['sha256']
            paths[row['path']] = row['sha256']
    source = json.loads((root / 'source_protection.json').read_text())
    license_row = source.get('license', {})
    if license_row.get('exists'):
        paths[license_row['path']] = license_row['sha256']
        paths[str(root / 'carrier/LICENSE')] = license_row['sha256']
    for name, sha in source['source_sha256'].items():
        paths[str(Path(source['source_root']) / 'src' / name)] = sha
    deployment = json.loads((root / 'deployment.json').read_text())
    for name in ('carrier_manifest', 'source_protection'):
        if digest(root / (name + '.json')) != deployment[name + '_sha256']:
            raise RuntimeError('Frozen deployment manifest differs: ' + name)
    # Deployment helper freezes every submitted text/source byte.
    for name, sha in deployment['bundle_sha256'].items():
        paths[str(root / name)] = sha
    paths[str(root / 'carrier_manifest.json')] = digest(root / 'carrier_manifest.json')
    paths[str(root / 'source_protection.json')] = digest(root / 'source_protection.json')
    return paths


def verify(paths):
    for name, expected in paths.items():
        if digest(name) != expected:
            raise RuntimeError('Protected path changed: ' + name)


class Selection:
    def __init__(self):
        self.best_score, self.best_epoch, self.bad_epochs = -1., None, 0

    def update(self, score, epoch):
        if not math.isfinite(float(score)):
            raise FloatingPointError('Nonfinite validation score')
        score = round(float(score), 4)
        improved = score > self.best_score
        if improved:
            self.best_score, self.best_epoch, self.bad_epochs = score, epoch, 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs > 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=('baby', 'sports'))
    parser.add_argument('--mode', choices=('uniform', 'semantic', 'shuffled'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--prepare-assets', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.prepare_assets and (args.mode is None or args.output is None):
        parser.error('Training needs mode and output')
    root = Path(__file__).resolve().parent
    assets_dir = args.assets.resolve()
    output = args.output.resolve() if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.prepare_assets else '1'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '4'
    os.chdir(root / 'carrier/src')
    sys.path[:0] = [str(root), str(root / 'carrier/src')]
    import numpy as np
    # The inspected official MMRec evaluator uses the removed np.float alias.
    # This compatibility alias leaves its formula and source bytes unchanged.
    np_float_alias_needed = not hasattr(np, 'float')
    if np_float_alias_needed:
        np.float = float
    import scipy.sparse as sp
    import torch
    from common.trainer import Trainer
    from models.msca import MSCA
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.utils import init_seed
    from sampling import InterestNegativeSampler, assemble_group_assets, build_group_assets
    torch.set_num_threads(4)
    seed = 2026
    domain = {'baby': (2, .4, .005, 3e-7), 'sports': (3, .3, .005, 5e-8)}[args.dataset]
    settings = {'data_path': str(root / 'carrier/data') + '/', 'gpu_id': '1',
                'use_gpu': not args.prepare_assets, 'seed': seed, 'epochs': 1 if args.smoke else 300,
                'stopping_step': 20, 'eval_step': 1, 'train_batch_size': 2048,
                'eval_batch_size': 64, 'learning_rate': .001, 'learner': 'adam',
                'learning_rate_scheduler': [1., 50], 'weight_decay': 0.,
                'save_recommended_topk': False, 'req_training': True, 'valid_metric': 'Recall@20',
                'checkpoint_dir': str(output / 'unused_legacy_checkpoints') if output else str(assets_dir / 'unused'),
                'n_layers': domain[0], 'fusion_coeff': domain[1], 'cl_weight': domain[2], 'reg_weight': domain[3]}
    config = Config('MSCA', args.dataset, settings, mg=False)
    for key in config['hyper_parameters']:
        if isinstance(config[key], (tuple, list)):
            if len(config[key]) != 1:
                raise ValueError('Unexpected parameter grid: ' + key)
            config[key] = config[key][0]
    if config['use_full_sampling'] or not config['use_neg_sampling'] or config['use_neighborhood_loss']:
        raise RuntimeError('Unsupported sampling branch')
    paths = protected_paths(root)
    verify(paths)
    dataset = RecDataset(config)
    # Never call RecDataset.split(): it constructs the test dataset as well.
    split = config['inter_splitting_label']
    train_df = dataset.df.loc[dataset.df[split] == 0].drop(columns=[split]).copy()
    train_data = dataset.copy(train_df)
    str(train_data)
    loader = TrainDataLoader(config, train_data, batch_size=2048, shuffle=True)
    csr = loader.inter_matrix(form='csr')
    csr.sort_indices()
    if args.prepare_assets:
        if torch.cuda.is_initialized():
            raise RuntimeError('CPU preparation initialized CUDA')
        assets_dir.mkdir(parents=True, exist_ok=False)
        start = time.monotonic()
        data = root / 'carrier/data' / args.dataset
        image = np.load(data / config['vision_feature_file'], mmap_mode='r')
        text = np.load(data / config['text_feature_file'], mmap_mode='r')
        # One backend is picked once before any candidate metric and persisted.
        assets = build_group_assets(csr, image, text, seed=seed, n_groups=8, kmeans_backend='auto')
        np.savez_compressed(assets_dir / 'arrays.npz', labels=assets.labels,
                            shuffled_labels=assets.shuffled_labels, universe=assets.universe,
                            permutation=assets.control_user_permutation,
                            **{'pool_' + str(i): p for i, p in enumerate(assets.pools)})
        write_json(assets_dir / 'diagnostics.json', assets.diagnostics)
        verify(paths)
        write_json(assets_dir / 'preparation.json', {'status': 'completed',
                   'elapsed_seconds': time.monotonic() - start, 'cuda_initialized': False,
                   'array_sha256': digest(assets_dir / 'arrays.npz'),
                   'diagnostics_sha256': digest(assets_dir / 'diagnostics.json'),
                   'protected_count': len(paths), 'dataset': args.dataset})
        print('TRAIN-only groups prepared; no model training or evaluation', flush=True)
        return
    start = time.monotonic()
    progress = {'status': 'initializing', 'dataset': args.dataset, 'mode': args.mode,
                'seed': seed, 'smoke': args.smoke, 'finite_optimizer_updates': 0,
                'completed_epochs': 0, 'test_loader_constructed': False, 'test_evaluated': False}
    write_json(output / 'progress.json', progress)
    def emit(value):
        progress.update(value)
        progress['elapsed_seconds'] = time.monotonic() - start
        write_json(output / 'progress.json', progress)
    stage = 'initialization'
    sampler = None
    try:
        if str(config['device']) != 'cuda':
            raise RuntimeError('GPU 1 is unavailable')
        torch.cuda.set_per_process_memory_fraction(.55, device=0)
        init_seed(seed)
        arr = np.load(assets_dir / 'arrays.npz', allow_pickle=False)
        diagnostics = json.loads((assets_dir / 'diagnostics.json').read_text())
        preparation = json.loads((assets_dir / 'preparation.json').read_text())
        if (preparation.get('status') != 'completed' or preparation.get('dataset') != args.dataset
                or preparation.get('cuda_initialized') is not False
                or preparation.get('array_sha256') != digest(assets_dir / 'arrays.npz')
                or preparation.get('diagnostics_sha256') != digest(assets_dir / 'diagnostics.json')):
            raise RuntimeError('CPU group-asset preparation receipt does not match')
        assets = assemble_group_assets(csr, arr['labels'], seed=seed, n_groups=8,
                                      grouping_diagnostics=diagnostics.get('grouping'))
        if assets.diagnostics != diagnostics:
            raise RuntimeError('Cached TRAIN group assets differ')
        for name, actual in [('shuffled_labels', assets.shuffled_labels), ('universe', assets.universe),
                             ('permutation', assets.control_user_permutation)]:
            if not np.array_equal(arr[name], actual):
                raise RuntimeError('Cached array differs: ' + name)
        for i, pool in enumerate(assets.pools):
            if not np.array_equal(pool, arr['pool_' + str(i)]):
                raise RuntimeError('Cached fixed pool differs')
        paths[str(assets_dir / 'arrays.npz')] = digest(assets_dir / 'arrays.npz')
        paths[str(assets_dir / 'diagnostics.json')] = digest(assets_dir / 'diagnostics.json')
        paths[str(assets_dir / 'preparation.json')] = digest(assets_dir / 'preparation.json')
        sampler = InterestNegativeSampler(assets, mode=args.mode, seed=seed)
        # Keep the loader's positive ordering; replace only the negative draw.
        loader._sample_neg_ids = lambda users: torch.from_numpy(sampler.sample(np.asarray(users, dtype=np.int64)))
        init_seed(seed)
        loader.pretrain_setup()
        model = MSCA(config, loader).to(config['device'])
        init_hash = hashlib.sha256()
        for name, value in model.state_dict().items():
            init_hash.update(name.encode())
            init_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        initial_sha = init_hash.hexdigest()
        trainer = Trainer(config, model, mg=False)
        if trainer.clip_grad_norm:
            raise RuntimeError('Unexpected gradient clipping')
        valid_df = dataset.df.loc[dataset.df[split] == 1].drop(columns=[split]).copy()
        if config['filter_out_cod_start_users']:
            valid_df = valid_df.loc[valid_df[dataset.uid_field].isin(train_df[dataset.uid_field])].copy()
        valid_data = dataset.copy(valid_df)
        str(valid_data)
        valid = EvalDataLoader(config, valid_data, additional_dataset=train_data, batch_size=64, shuffle=False)
        write_json(output / 'config.json', config.final_config_dict)
        write_json(output / 'protected_paths.json', paths)
        write_json(output / 'environment.json', {'python': sys.version, 'torch': torch.__version__,
                   'numpy': np.__version__, 'physical_gpu': 1, 'cpu_threads': 4,
                   'numpy_float_compatibility_alias': np_float_alias_needed,
                   'cuda_memory_fraction': .55, 'initial_state_sha256': initial_sha,
                   'group_assets_sha256': digest(assets_dir / 'arrays.npz'),
                   'source_revision': json.loads((root / 'source_revision.json').read_text())})
        selection = Selection()
        epoch_hashes = []
        full_epoch_hashes = []
        current_hash = hashlib.sha256()
        full_hash = hashlib.sha256()
        current_batch = 0
        original_loss = model.calculate_loss
        original_step = trainer.optimizer.step
        def finite(tensor, context):
            if not bool(torch.isfinite(tensor).all().item()):
                raise FloatingPointError(context)
        original_predict = model.full_sort_predict
        def guarded_predict(interaction):
            scores = original_predict(interaction)
            finite(scores, 'Nonfinite validation scores before ranking')
            return scores
        model.full_sort_predict = guarded_predict
        def guarded_loss(interaction):
            nonlocal stage, current_batch
            stage = 'forward'
            current_hash.update(interaction[:2].detach().cpu().contiguous().numpy().tobytes())
            full_hash.update(interaction.detach().cpu().contiguous().numpy().tobytes())
            value = original_loss(interaction)
            finite(value, 'Nonfinite loss before backward')
            stage = 'backward'
            return value
        def guarded_step(*step_args, **step_kwargs):
            nonlocal stage, current_batch
            stage = 'pre_optimizer'
            for name, param in model.named_parameters():
                if param.grad is not None:
                    finite(param.grad, 'Nonfinite gradient: ' + name)
            stage = 'optimizer'
            result = original_step(*step_args, **step_kwargs)
            stage = 'post_optimizer'
            for name, param in model.named_parameters():
                finite(param, 'Nonfinite parameter: ' + name)
            for state in trainer.optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        finite(value, 'Nonfinite Adam state: ' + key)
            progress['finite_optimizer_updates'] += 1
            current_batch += 1
            if progress['finite_optimizer_updates'] == 1 or progress['finite_optimizer_updates'] % 50 == 0:
                emit({'status': 'training', 'batch': current_batch, 'last_finite_loss': float(last_loss[0])})
            return result
        last_loss = [0.]
        def tracked_loss(interaction):
            value = guarded_loss(interaction)
            last_loss[0] = value.detach().item()
            return value
        model.calculate_loss = tracked_loss
        trainer.optimizer.step = guarded_step
        for epoch in range(config['epochs']):
            epoch_start = time.monotonic()
            progress['epoch'] = epoch
            current_batch = 0
            current_hash = hashlib.sha256()
            full_hash = hashlib.sha256()
            model.pre_epoch_processing()
            losses, _ = trainer._train_epoch(loader, epoch)
            if not math.isfinite(float(losses)):
                raise FloatingPointError('Nonfinite epoch loss')
            trainer.lr_scheduler.step()
            model.post_epoch_processing()
            stage = 'validation'
            with torch.no_grad():
                metrics = {key.lower(): float(value) for key, value in trainer.evaluate(valid).items()}
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite validation')
            improved, stop = selection.update(metrics['recall@20'], epoch)
            epoch_hashes.append(current_hash.hexdigest())
            full_epoch_hashes.append(full_hash.hexdigest())
            if improved:
                torch.save({'state_dict': model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
                            'epoch': epoch, 'metrics': metrics, 'initial_sha256': initial_sha}, output / 'best.pt')
                with (output / 'best_rng.pkl').open('wb') as stream:
                    pickle.dump({'python': random.getstate(), 'numpy': np.random.get_state(),
                                 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                                 'sampler': sampler.rng.get_state(), 'loader_df': loader.dataset.df,
                                 'loader_all_items': loader.all_items}, stream, protocol=4)
            record = {'epoch': epoch, 'loss': float(losses), 'validation': metrics,
                      'improved': improved, 'best_epoch': selection.best_epoch,
                      'best_score': selection.best_score, 'bad_epochs': selection.bad_epochs,
                      'epoch_seconds': time.monotonic() - epoch_start,
                      'positive_order_sha256': current_hash.hexdigest(),
                      'full_interaction_sha256': full_hash.hexdigest(),
                      'finite_optimizer_updates': progress['finite_optimizer_updates']}
            with (output / 'epochs.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            emit({'status': 'training', 'completed_epochs': epoch + 1, 'best_epoch': selection.best_epoch,
                  'best_validation': metrics if improved else progress.get('best_validation'),
                  'last_validation': metrics, 'early_stop': stop})
            print(json.dumps(record), flush=True)
            if stop:
                break
        stage = 'final_validation'
        checkpoint = torch.load(output / 'best.pt', map_location=config['device'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        rankings = []
        with torch.no_grad():
            for interaction in valid:
                scores = model.full_sort_predict(interaction)
                finite(scores, 'Nonfinite final scores')
                mask = interaction[1]
                scores[mask[0], mask[1]] = -1e10
                rankings.append(torch.topk(scores, k=50, dim=1).indices.cpu().numpy())
        ranking = np.concatenate(rankings)
        truth = valid.get_eval_items()
        lengths = np.asarray(valid.get_eval_len_list())
        if len(ranking) != len(truth) or np.any(lengths <= 0):
            raise RuntimeError('Invalid relevance/user order')
        hits = np.asarray([np.isin(items, positives) for items, positives in zip(ranking, truth)])
        per_user = {'user_ids': np.asarray(valid.get_eval_users()), 'positive_lengths': lengths}
        metrics = {}
        for cutoff in (5, 10, 20, 50):
            recall = hits[:, :cutoff].sum(1) / lengths
            weights = 1. / np.log2(np.arange(2, cutoff + 2))
            dcg = (hits[:, :cutoff] * weights).sum(1)
            ideal = np.cumsum(weights)[np.minimum(lengths, cutoff) - 1]
            ndcg = dcg / ideal
            for name, values in (('recall', recall), ('ndcg', ndcg)):
                if not np.all(np.isfinite(values)):
                    raise FloatingPointError('Nonfinite per-user metrics')
                metrics[name + '@' + str(cutoff)] = float(values.mean())
                per_user[name + '_' + str(cutoff)] = values
                if abs(round(float(values.mean()), 4) - checkpoint['metrics'][name + '@' + str(cutoff)]) > .00010001:
                    raise RuntimeError('Final metrics disagree with official evaluator')
        np.savez_compressed(output / 'per_user_validation.npz', **per_user)
        stage = 'integrity'
        verify(paths)
        result = {**progress, 'status': 'completed', 'metrics': metrics,
                  'test_evaluated': False,
                  'completed_epochs': len(epoch_hashes), 'best_epoch': selection.best_epoch,
                  'initial_state_sha256': initial_sha, 'positive_epoch_hashes': epoch_hashes,
                  'full_train_epoch_hashes': full_epoch_hashes,
                  'validation_users': len(lengths), 'validation_positives': int(lengths.sum()),
                  'protected_paths_match': True, 'protected_count': len(paths),
                  'group_assets_sha256': digest(assets_dir / 'arrays.npz'),
                  'group_diagnostics_sha256': digest(assets_dir / 'diagnostics.json'),
                  'checkpoint_sha256': digest(output / 'best.pt'),
                  'per_user_sha256': digest(output / 'per_user_validation.npz'),
                  'sampler': sampler.stats(), 'elapsed_seconds': time.monotonic() - start,
                  'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(0),
                  'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(0)}
        write_json(output / 'result.json', result)
        emit({'status': 'completed'})
    except Exception as exc:
        intact = None
        try:
            verify(paths)
            intact = True
        except Exception:
            intact = False
        failure = {**progress, 'status': 'technical_failure', 'stage': stage,
                   'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
                   'protected_paths_match': intact, 'elapsed_seconds': time.monotonic() - start,
                   'sampler': sampler.stats() if sampler else None}
        write_json(output / 'failure.json', failure)
        emit({'status': 'technical_failure', 'stage': stage, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
