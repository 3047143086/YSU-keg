"""Validation gates for candidate supervision and its joint-update control."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

PRIMARY = ('recall@20', 'ndcg@20')
SHORTLIST = ('recall@10', 'ndcg@10')
ARMS = ('baseline', 'candidate', 'joint')
DATASETS = ('baby', 'sports')
REMOTE_ROOT = '/data/sunzixuan/candidate_gate_20261009_v1'
TERMINAL_SUCCESS = 'pilot_passed_needs_multiseed_and_multimodal_attribution'


def validate_metrics(metrics):
    if not isinstance(metrics, dict) or set(metrics) - set(ARMS):
        raise ValueError('Expected only baseline, candidate and joint metric arms')
    for arm, values in metrics.items():
        if not isinstance(values, dict):
            raise ValueError('Expected a metric mapping for ' + arm)
        for key in PRIMARY + SHORTLIST:
            value = values.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('Invalid metric for ' + arm + ':' + key)
    if 'baseline' in metrics:
        for key in PRIMARY:
            if metrics['baseline'][key] <= 0:
                raise ValueError('Positive baseline required for a relative gain')


def decide(metrics):
    validate_metrics(metrics)
    missing = [arm for arm in ARMS if arm not in metrics]
    if missing:
        return {'status': 'pending', 'missing': missing}
    relative = {key: metrics['candidate'][key] / metrics['baseline'][key] - 1 for key in PRIMARY}
    shortlist = {key: metrics['candidate'][key] - metrics['baseline'][key] for key in SHORTLIST}
    control = {key: metrics['candidate'][key] - metrics['joint'][key] for key in PRIMARY}
    passed = (all(value >= .005 for value in relative.values())
              and all(value >= -.0002 for value in shortlist.values())
              and all(value > 0 for value in control.values()))
    return {'status': 'keep' if passed else 'discard', 'relative_vs_baseline': relative,
            'shortlist_absolute_delta': shortlist, 'absolute_vs_joint': control,
            'evidence_level': 'single-seed exploratory validation',
            'tested_mechanism': 'candidate-negative supervision assigned to the gate only'}


def finish(root, state, outputs, source_revision=None):
    root = Path(root)
    if set(outputs) - set(DATASETS):
        raise ValueError('Unexpected dataset in formal outputs')
    decisions, rows, metrics = {}, [], {}
    for dataset in DATASETS:
        metrics[dataset] = {}
        for arm, path in outputs.get(dataset, {}).items():
            if arm not in ARMS:
                raise ValueError('Unexpected formal arm: ' + arm)
            path = Path(path)
            result = json.loads((path / 'result.json').read_text())
            if result.get('status') != 'completed' or result.get('test_evaluated') is not False:
                raise RuntimeError('Cannot report incomplete or test-selected results')
            metrics[dataset][arm] = result['metrics']
        decisions[dataset] = decide(metrics[dataset])
        for arm, result_metrics in metrics[dataset].items():
            memory = 'NA'
            status = 'control'
            if arm == 'candidate':
                status = decisions[dataset]['status']
            samples = [job.get('peak_sampled_gpu_memory_mib') for job in state.get('jobs', [])
                       if job['name'] == dataset + '_' + arm + '_s2026']
            if samples and samples[0] is not None:
                memory = '{:.6f}'.format(samples[0] / 1024.)
            rows.append([source_revision or '-', result_metrics['recall@20'], memory,
                         status, dataset + ' ' + arm + ' seed 2026'])
    if state.get('status') == 'technical_failure':
        rows.append([source_revision or '-', 'NA', 'NA', 'crash', state.get('error', 'technical failure')])
    metric_success = all(row['status'] == 'keep' for row in decisions.values())
    success = (metric_success and
               state.get('status') == TERMINAL_SUCCESS)
    with (root / 'results.tsv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['commit', 'validation_recall20', 'sampled_peak_gpu_gib', 'status', 'description'])
        writer.writerows(rows)
    report = {'checked_unix': time.time(), 'pipeline_status': state.get('status'),
              'stage': state.get('stage'), 'decisions': decisions, 'metrics': metrics,
              'pilot_success': success, 'test_evaluated': False,
              'resource_memory_is_sampled_not_exact_peak': True, 'source_revision': source_revision,
              'new_completed_formal_arms': sum(arm in ('candidate', 'joint') for ds in metrics.values() for arm in ds),
              'evidence_level': 'single-seed exploratory validation',
              'tested_mechanism': 'candidate-negative supervision assigned to the gate only',
              'novelty_established': False, 'generalizable_performance_claim': False}
    with (root / 'research_report.json').open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    root = Path(__file__).resolve().parent
    record = json.loads((root / 'latest_status.json').read_text())
    if record['root'] != REMOTE_ROOT:
        raise RuntimeError('Status snapshot is from another experiment')
    completed, outputs = {}, {}
    for name, row in record.get('jobs', {}).items():
        if not name.startswith('runs/') or row.get('result', {}).get('status') != 'completed':
            continue
        dataset, arm, _ = name.split('/')[-1].split('_')
        if dataset not in DATASETS or arm not in ARMS:
            raise ValueError('Unexpected formal result in status snapshot: ' + name)
        if row['result'].get('test_evaluated') is not False:
            raise RuntimeError('Cannot report a test-selected status snapshot')
        completed.setdefault(dataset, {})[arm] = row['result']['metrics']
    for dataset in DATASETS:
        outputs[dataset] = decide(completed.get(dataset, {}))
    print(json.dumps({'snapshot_unix': record['checked_unix'],
                      'state': record['artifacts'].get('pipeline_status.json', {}),
                      'decisions': outputs}, indent=2))


if __name__ == '__main__':
    main()
