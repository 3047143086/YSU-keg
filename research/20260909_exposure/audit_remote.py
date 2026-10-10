"""Read-only audit of available interaction metadata on server 213."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '20260905'))
from remote_session import Session

REMOTE_DATA = '/data/sunzixuan/expirement/LOBSTER/data'
REMOTE_PREPARED = '/data/sunzixuan/rcmqr_20260908_screen_v2'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'

SOURCE = r'''
import json, os, re
from pathlib import Path
roots = [Path(%r), Path(%r)]
records = []
keywords = re.compile(r'(time|timestamp|expos|impress|display|view|click|skip|shown|rank|position|missing|image|text|label|rating)', re.I)
for root in roots:
    if not root.exists():
        continue
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
            continue
        item = {'root': str(root), 'path': str(path), 'size': path.stat().st_size}
        if path.suffix.lower() in ('.inter', '.tsv', '.csv', '.txt', '.json', '.yaml', '.yml'):
            try:
                sample = path.read_text(errors='replace')[:65536]
                lines = sample.splitlines()[:5]
                item['sample_lines'] = lines
                item['keyword_lines'] = [line[:500] for line in lines if keywords.search(line)]
                item['keywords_in_sample'] = sorted(set(keywords.findall(sample)))[:50]
            except Exception as error:
                item['read_error'] = str(error)
        records.append(item)
print(json.dumps({'roots': [str(r) for r in roots], 'files': records}, indent=2))
''' % (REMOTE_DATA, REMOTE_PREPARED)


if __name__ == '__main__':
    session = Session('213')
    try:
        code, output = session.run_python(SOURCE, PYTHON, 120)
        print(output)
        raise SystemExit(code)
    finally:
        session.close()
