"""Read a fetched snapshot without network access or held-out-label evaluation."""
import argparse
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPECTED = '/data/sunzixuan/target_isolated_20261008_autoresearch_v1'
SOURCE = '/data/sunzixuan/target_isolated_20260911_soft_v1'


def inspect(path):
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('root') not in (EXPECTED, SOURCE):
        raise ValueError('Snapshot belongs to another experiment')
    state = record.get('artifacts', {}).get('pipeline_status.json', {})
    completed = {}
    for name, job in record.get('jobs', {}).items():
        if not name.startswith('runs/'):
            continue
        result = job.get('result', {})
        if result.get('status') != 'completed':
            continue
        if result.get('test_evaluated') is not False or result.get('inputs_unchanged') is not True:
            raise ValueError('Invalid completed-run protocol: ' + name)
        metrics = result.get('metrics', {})
        for key in ('recall@10', 'ndcg@10', 'recall@20', 'ndcg@20'):
            value = metrics.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid completed-run metric: ' + name + ':' + key)
        completed[name] = {'metrics': metrics, 'reused_from': job.get('reused_from')}
    return {'snapshot': str(path), 'remote_root': record['root'],
            'checked_utc': datetime.datetime.fromtimestamp(record['checked_unix'],
                datetime.timezone.utc).isoformat(), 'pipeline_status': state.get('status'),
            'stage': state.get('stage'), 'completed_formal_arms': len(completed),
            'completed': completed, 'scientific_screens': state.get('screens', {}),
            'error': state.get('error'), 'live_query': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    path = ROOT / 'latest_status.json'
    if not path.exists():
        path = ROOT / 'previous_status.json'
    result = inspect(path)
    print(json.dumps(result, indent=2) if args.report else result['completed_formal_arms'])
