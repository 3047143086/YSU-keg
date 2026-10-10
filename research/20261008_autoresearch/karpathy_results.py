"""Generate the Karpathy-style scientific ledger from completed validation evidence."""
import argparse
import csv
import json
from pathlib import Path

from verify import inspect

ROOT = Path(__file__).resolve().parent


def decide(completed, dataset):
    arms = {arm: completed.get('runs/{}_{}_s2026'.format(dataset, arm))
            for arm in ('baseline', 'edge', 'history')}
    if not all(arms.values()):
        return {'status': 'pending', 'missing': [arm for arm, row in arms.items() if not row]}
    base, edge, history = [arms[arm]['metrics'] for arm in ('baseline', 'edge', 'history')]
    if any(base[key] <= 0 for key in ('recall@20', 'ndcg@20')):
        raise ValueError('Relative gain is undefined for a zero baseline')
    gains = {key: edge[key] / base[key] - 1 for key in ('recall@20', 'ndcg@20')}
    guards = {key: edge[key] - base[key] for key in ('recall@10', 'ndcg@10')}
    history_delta = {key: edge[key] - history[key] for key in ('recall@20', 'ndcg@20')}
    passed = (all(value >= .005 for value in gains.values()) and
              all(value >= -.0002 for value in guards.values()) and
              all(value > 0 for value in history_delta.values()))
    return {'status': 'keep' if passed else 'discard', 'relative_gain': gains,
            'shortlist_delta': guards, 'history_delta': history_delta,
            'evidence_level': 'single-seed exploratory validation, not final test confirmation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('baseline', 'candidate'))
    parser.add_argument('--dataset', choices=('baby', 'sports'), default='baby')
    args = parser.parse_args()
    snapshot = ROOT / 'latest_status.json'
    if not snapshot.exists():
        parser.error('Fetch current status with manage.py status --brief first')
    report = inspect(snapshot)
    if args.metric:
        arm = 'edge' if args.metric == 'candidate' else 'baseline'
        row = report['completed'].get('runs/{}_{}_s2026'.format(args.dataset, arm))
        if row is None:
            parser.error('Requested metric is unavailable: this formal arm has not completed')
        print(row['metrics']['recall@20'])
        return
    decisions = {dataset: decide(report['completed'], dataset) for dataset in ('baby', 'sports')}
    for dataset, screen in report['scientific_screens'].items():
        if decisions[dataset]['status'] == 'pending':
            raise ValueError('Screen exists without all completed comparison arms')
        if screen['passed'] != (decisions[dataset]['status'] == 'keep'):
            raise ValueError('Recomputed verdict disagrees with the remote screen')
    report.update(upstream='https://github.com/karpathy/autoresearch',
                  primary_metric='validation_recall20', direction='higher_is_better',
                  decisions=decisions, memory_note='NA: peak VRAM was not reported by the frozen runner')
    rows = [('-', 'NA', 'NA', 'crash',
             'Historical soft_v1 baseline: non-finite loss at epoch 39; not a scientific result')]
    for dataset in ('baby', 'sports'):
        baseline = report['completed'].get('runs/{}_baseline_s2026'.format(dataset))
        if baseline:
            rows.append(('0babcdb', baseline['metrics']['recall@20'], 'NA', 'keep',
                         dataset + ' seed2026 freshly trained baseline'))
        if decisions[dataset]['status'] != 'pending':
            candidate = report['completed']['runs/{}_edge_s2026'.format(dataset)]
            rows.append(('0babcdb', candidate['metrics']['recall@20'], 'NA',
                         decisions[dataset]['status'], dataset + ' seed2026 alpha0.5; all guards checked'))
    if report['pipeline_status'] == 'technical_failure':
        error = str(report.get('error')).replace('\t', ' ').replace('\n', ' ')
        rows.append(('0babcdb', 'NA', 'NA', 'crash', 'Recovery: ' + error))
    with (ROOT / 'results.tsv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['commit', 'validation_recall20', 'memory_gb', 'status', 'description'])
        writer.writerows(rows)
    (ROOT / 'research_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('checked_utc', 'pipeline_status', 'stage',
                     'completed_formal_arms', 'decisions')}, indent=2))


if __name__ == '__main__':
    main()
