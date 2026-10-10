"""Reproduce a saved BRIDGE baseline and quantify validation shortlist coverage.

This is an engineering diagnostic, not a proposed-method performance result.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bridge-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=['baby', 'sports'], default='baby')
    parser.add_argument('--gpu-id', type=int, default=0)
    args = parser.parse_args()
    output_path = args.output.resolve()
    checkpoint = args.checkpoint.resolve()
    root = args.bridge_root.resolve()
    os.chdir(root / 'src')
    sys.path.insert(0, str(root / 'src'))

    from diagnose_ecli import bridge_scores, cache_bridge_embeddings, scalarize_hyperparameters
    from models.bridge import BRIDGE
    from utils.configurator import Config
    from utils.dataloader import EvalDataLoader, TrainDataLoader
    from utils.dataset import RecDataset
    from utils.ecli import per_user_ranking_metrics
    from utils.utils import init_seed

    start = time.monotonic()
    config = Config('BRIDGE', args.dataset, {
        'gpu_id': args.gpu_id, 'seed': [2026], 'eval_batch_size': 64,
        'save_recommended_topk': False,
    })
    scalarize_hyperparameters(config)
    config['seed'] = 2026
    init_seed(2026)
    torch.set_num_threads(4)
    dataset = RecDataset(config)
    train_dataset, valid_dataset, _ = dataset.split()
    str(train_dataset)
    str(valid_dataset)
    train = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=64)
    model = BRIDGE(config, train).to(config['device'])
    state = torch.load(checkpoint, map_location=config['device'])
    mismatch = model.load_state_dict(state, strict=False)
    if set(mismatch.missing_keys) - {'user_activity_z'} or set(mismatch.unexpected_keys) - {
        'id_preference', 'v_preference', 't_preference'
    }:
        raise RuntimeError(f'Unsupported checkpoint mismatch: {mismatch}')
    model.eval()
    cached = cache_bridge_embeddings(model)
    batches = []
    max_error = None
    for index, interaction in enumerate(valid):
        scores = model.full_sort_predict(interaction)
        if index == 0:
            cached_scores = bridge_scores(model, interaction[0], cached)
            max_error = float((cached_scores - scores).abs().max().cpu())
            print(f'cached_score_audit_max_error={max_error}; evaluation_uses_original_interface', flush=True)
        mask = interaction[1]
        scores[mask[0], mask[1]] = -torch.inf
        batches.append(torch.topk(scores, k=min(200, model.n_items), dim=1).indices.cpu().numpy())
        if index % 50 == 0:
            print(f'validation_batch={index}', flush=True)

    ranking = np.concatenate(batches, axis=0)
    positives = valid.get_eval_items()
    lengths = np.asarray(valid.get_eval_len_list())
    if len(positives) != len(ranking) or np.any(lengths <= 0):
        raise RuntimeError('Validation user ordering or relevance counts are invalid')
    hits = np.array([np.isin(items, truth) for items, truth in zip(ranking, positives)])
    metrics = {}
    for cutoff in (10, 20, 50, 100, 200):
        recall, ndcg = per_user_ranking_metrics(hits, lengths, cutoff)
        metrics[f'recall@{cutoff}'] = float(recall.mean())
        metrics[f'ndcg@{cutoff}'] = float(ndcg.mean())

    # Relevance-sorted shortlists are a diagnostic bound, never a model result.
    available = hits.sum(axis=1)
    ideal_hits = np.arange(20)[None, :] < np.minimum(available, 20)[:, None]
    oracle_recall, oracle_ndcg = per_user_ranking_metrics(ideal_hits, lengths, 20)
    expected = {'baby': (0.1055593, 0.0492366), 'sports': (0.1198, 0.0567)}[args.dataset]
    reproduced = abs(metrics['recall@20'] - expected[0]) < 0.00015 and abs(
        metrics['ndcg@20'] - expected[1]) < 0.00015
    frame = train_dataset.df
    degrees = frame.groupby(train_dataset.uid_field).size()
    user_ids = valid.get_eval_users().numpy()
    user_degrees = degrees.reindex(user_ids, fill_value=0).to_numpy()
    strata = {}
    for name, selected in {
        'history_1_to_5': user_degrees <= 5,
        'history_6_to_15': (user_degrees > 5) & (user_degrees <= 15),
        'history_over_15': user_degrees > 15,
    }.items():
        if selected.any():
            r20, n20 = per_user_ranking_metrics(hits[selected], lengths[selected], 20)
            strata[name] = {'users': int(selected.sum()), 'recall@20': float(r20.mean()),
                            'ndcg@20': float(n20.mean()),
                            'candidate_recall@200': float((available[selected] / lengths[selected]).mean())}
    hashes = {}
    for path in [checkpoint, Path(dataset.dataset_path) / config['inter_file_name'],
                 root / 'src/models/bridge.py', root / 'src/models/dual_frequency_encoder.py',
                 root / 'src/configs/model/BRIDGE.yaml']:
        hashes[str(path)] = digest(path)
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    record = {
        'purpose': 'baseline_reproduction_and_candidate_coverage_only',
        'new_method_evaluated': False, 'test_evaluated': False,
        'dataset': args.dataset, 'seed': 2026, 'users': len(ranking),
        'items': model.n_items, 'checkpoint': str(checkpoint),
        'metrics': metrics, 'historical_baseline_reproduced': bool(reproduced),
        'cached_vs_direct_max_abs_error': max_error,
        'scoring_interface': 'BRIDGE.full_sort_predict; cached scores not used for evaluation',
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'oracle_diagnostic_not_model_performance': {
            'recall@20': float(oracle_recall.mean()), 'ndcg@20': float(oracle_ndcg.mean()),
            'fraction_users_with_any_positive_in_top200': float((available > 0).mean()),
        },
        'activity_strata': strata, 'sha256': hashes, 'git_commit': git.stdout.strip(),
        'diagnostic_script_sha256': digest(Path(__file__).resolve()),
        'elapsed_seconds': time.monotonic() - start,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps(record, indent=2), flush=True)
    if not reproduced:
        raise RuntimeError('Historical baseline mismatch; stop before method experiments')


if __name__ == '__main__':
    main()
