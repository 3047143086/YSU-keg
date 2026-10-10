"""Validation-only confirmation of existing weights; does not train or select a model."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('baby', 'sports'), required=True)
    parser.add_argument('--bridge-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(str(output))
    root = args.bridge_root.resolve()
    bundle = Path(__file__).resolve().parent
    audit = json.loads((bundle / 'input_manifest.json').read_text())
    historical = json.loads((bundle / (args.dataset + '_historical.json')).read_text())
    checkpoint = Path(historical['checkpoint'])
    inputs = audit['checkpoints'][args.dataset]['inputs']
    tracked = {root / path: row['sha256'] for path, row in audit['files'].items()}
    tracked.update({Path(row['path']): row['sha256'] for row in inputs.values() if row['exists']})
    tracked.update({checkpoint: historical['sha256'][str(checkpoint)]})
    for path, expected in tracked.items():
        if digest(path) != expected:
            raise RuntimeError('Input drift: ' + str(path))
    for row in inputs.values():
        if not row['exists'] and Path(row['path']).exists():
            raise RuntimeError('Previously absent input appeared: ' + row['path'])
    # Record loader/metric/config dependencies omitted from the Sep5 receipt.
    for relative in ('src/utils/configurator.py', 'src/utils/dataset.py',
                     'src/utils/dataloader.py', 'src/utils/utils.py',
                     'src/common/abstract_recommender.py',
                     'src/configs/dataset/' + args.dataset + '.yaml'):
        path = root / relative
        tracked[path] = digest(path)
    if not inputs['mm_adj_10.pt']['exists']:
        raise RuntimeError('Existing graph cache required; do not build into shared data')

    os.chdir(root / 'src')
    sys.path.insert(0, str(root / 'src'))
    from diagnose_ecli import scalarize_hyperparameters
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.ecli import per_user_ranking_metrics
    from utils.utils import init_seed

    start = time.monotonic()
    torch.set_num_threads(4)
    config = Config('BRIDGE', args.dataset, {
        'use_gpu': False, 'seed': [2026], 'eval_batch_size': 64,
        'save_recommended_topk': False,
    })
    scalarize_hyperparameters(config)
    config['seed'] = 2026
    init_seed(2026)
    # Verify actual branch behavior after construction as well as configuration.
    dataset = RecDataset(config)
    train_dataset, valid_dataset, _ = dataset.split()
    str(train_dataset)
    str(valid_dataset)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    model = BRIDGE(config, train).to(config['device'])
    if model.use_activity_simplex or model.msr_weight != 0 or model.use_user_graph:
        raise RuntimeError('Unexpected optional branch enabled')
    state = torch.load(checkpoint, map_location=config['device'])
    mismatch = model.load_state_dict(state, strict=False)
    # Legacy encoder declares an unused Parameter via .to(device). CPU registers
    # it, while GPU checkpoints omit it. It is never consumed by scoring.
    if (root / 'src/models/dual_frequency_encoder.py').read_text().count('result_embed') != 1:
        raise RuntimeError('Re-audit legacy result_embed before allowing a missing key')
    if set(mismatch.missing_keys) - {'user_activity_z', 'result_embed'} or set(mismatch.unexpected_keys) - {
        'id_preference', 'v_preference', 't_preference'
    }:
        raise RuntimeError('Unexpected checkpoint keys: ' + str(mismatch))
    del state
    model.eval()
    print('MODEL_READY ' + args.dataset, flush=True)
    batches = []
    for index, interaction in enumerate(valid):
        scores = model.full_sort_predict(interaction)
        if not torch.isfinite(scores).all():
            raise RuntimeError('Non-finite unmasked scores')
        mask = interaction[1]
        scores[mask[0], mask[1]] = -torch.inf
        batches.append(torch.topk(scores, k=min(200, model.n_items), dim=1).indices.cpu().numpy())
        if index % 50 == 0:
            print('validation_batch={} elapsed={:.1f}'.format(index, time.monotonic() - start), flush=True)
    ranking = np.concatenate(batches)
    positives = valid.get_eval_items()
    lengths = np.asarray(valid.get_eval_len_list())
    if len(ranking) != len(positives) or (lengths <= 0).any():
        raise RuntimeError('Invalid validation ordering or relevance counts')
    hits = np.array([np.isin(items, truth) for items, truth in zip(ranking, positives)])
    metrics = {}
    for cutoff in (10, 20, 50, 100, 200):
        recall, ndcg = per_user_ranking_metrics(hits, lengths, cutoff)
        metrics['recall@' + str(cutoff)] = float(recall.mean())
        metrics['ndcg@' + str(cutoff)] = float(ndcg.mean())
    deltas = {key: metrics[key] - historical['metrics'][key] for key in metrics}
    # Same absolute tolerance as the historical preflight, allows CPU/GPU rank ties.
    matched = all(abs(deltas[key]) < 0.00015 for key in ('recall@20', 'ndcg@20'))
    unchanged = all(digest(path) == expected for path, expected in tracked.items())
    record = {
        'purpose': 'saved_checkpoint_validation_confirmation_only',
        'dataset': args.dataset, 'seed': 2026, 'device': str(config['device']),
        'checked_unix': time.time(), 'checkpoint': str(checkpoint),
        'training_performed': False, 'test_evaluated': False, 'new_method_evaluated': False,
        'users': len(ranking), 'items': model.n_items,
        'metrics': metrics, 'delta_from_20260905': deltas,
        'historical_match_tolerance_absolute': 0.00015,
        'historical_baseline_reproduced': matched, 'inputs_unchanged': unchanged,
        'scoring_interface': 'BRIDGE.full_sort_predict; all items; mask training positives',
        'effective_config': config.final_config_dict,
        'checkpoint_key_compatibility': {'missing': mismatch.missing_keys, 'unexpected': mismatch.unexpected_keys},
        'legacy_cpu_compatibility': 'result_embed is declared once and unused in scoring; CPU registers it, historical GPU weights omit it',
        'sha256': {str(path): value for path, value in tracked.items()},
        'script_sha256': digest(__file__), 'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'elapsed_seconds': time.monotonic() - start,
    }
    with output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, default=str)
    print(json.dumps({'dataset': args.dataset, 'metrics': metrics, 'matched': matched,
                      'inputs_unchanged': unchanged}), flush=True)
    if not matched or not unchanged:
        raise RuntimeError('Confirmation failed; inspect receipt before using baseline')


if __name__ == '__main__':
    main()
