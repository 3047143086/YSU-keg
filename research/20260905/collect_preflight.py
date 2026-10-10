"""Collect isolated diagnostics, preserving structured results and job status."""

import json
from pathlib import Path

from remote_session import Session


def main():
    source = '''
import json
from pathlib import Path
records = {}
for dataset, directory in {
    'baby': '/data/sunzixuan/rave_20260905_preflight_v2',
    'sports': '/data/sunzixuan/rave_20260905_sports_preflight_v2',
}.items():
    root = Path(directory)
    if not root.exists():
        records[dataset] = {'status': 'not_started'}
        continue
    receipt = json.loads((root / 'launch.json').read_text())
    process = Path('/proc') / str(receipt['pid'])
    record = {'launch': receipt, 'process_exists': process.exists()}
    if process.exists():
        record['process_state'] = (process / 'stat').read_text().split(') ', 1)[1].split()[0]
    result = root / (dataset + '.json')
    record['stdout_tail'] = (root / 'stdout.log').read_text()[-4000:]
    if result.exists():
        record['result'] = json.loads(result.read_text())
    records[dataset] = record
print(json.dumps(records))
'''
    session = Session('213')
    try:
        code, output = session.run_python(source, 'python', 60)
        if code:
            raise RuntimeError(output)
        records, _ = json.JSONDecoder().raw_decode(output.lstrip())
        destination = Path(__file__).resolve().parent
        (destination / 'preflight_status.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        for dataset, record in records.items():
            if 'result' in record:
                (destination / (dataset + '_baseline.json')).write_text(
                    json.dumps(record['result'], indent=2), encoding='utf-8')
                print(dataset, 'process_exists=', record['process_exists'],
                      'reproduced=', record['result']['historical_baseline_reproduced'],
                      'metrics=', record['result']['metrics'])
            else:
                print(dataset, json.dumps(record))
    finally:
        session.close()


if __name__ == '__main__':
    main()
