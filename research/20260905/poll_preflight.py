"""Read the status and results of this session's bounded diagnostic."""

import json
from pathlib import Path

root = Path('/data/sunzixuan/rave_20260905_preflight_v2')
receipt = json.loads((root / 'launch.json').read_text())
process = Path('/proc') / str(receipt['pid'])
print('PROCESS_EXISTS', process.exists())
if process.exists():
    print('PROCESS_STAT', (process / 'stat').read_text().split(') ', 1)[1].split()[0])
result = root / 'baby.json'
if result.exists():
    print('RESULT_JSON_START')
    print(result.read_text())
    print('RESULT_JSON_END')
else:
    print('STDOUT_TAIL')
    print((root / 'stdout.log').read_text()[-6000:])
