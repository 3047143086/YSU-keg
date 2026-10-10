"""Construct a strict per-user chronological split on server 213."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '20260905'))
from remote_session import Session

PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
REMOTE_OUT = '/data/sunzixuan/temporal_20260909_probe'
SOURCE = r'''
import json, hashlib, os, shutil, time
from pathlib import Path
import pandas as pd
source = Path('/data/sunzixuan/expirement/LOBSTER/data/baby/baby.inter')
out = Path(%r)
out.mkdir(exist_ok=True)
frame = pd.read_csv(source, sep='\t')
frame = frame.sort_values(['userID','timestamp','itemID'], kind='mergesort').copy()
counts = frame.groupby('userID').size()
if counts.min() < 3:
    raise RuntimeError('Every user needs at least train, validation, and test events')
frame['chronological_label'] = 0
last = frame.groupby('userID').tail(1).index
penultimate = frame.drop(last).groupby('userID').tail(1).index
frame.loc[penultimate, 'chronological_label'] = 1
frame.loc[last, 'chronological_label'] = 2
result = frame[['userID','itemID','rating','timestamp','chronological_label']]
result.to_csv(out / 'baby_chronological.inter', sep='\t', index=False)
train = result[result.chronological_label == 0]
cal = result[result.chronological_label == 1]
test = result[result.chronological_label == 2]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()
manifest = {
  'source': str(source), 'source_sha256': sha(source),
  'output': str(out / 'baby_chronological.inter'), 'output_sha256': sha(out / 'baby_chronological.inter'),
  'users': int(result.userID.nunique()), 'items': int(result.itemID.nunique()),
  'rows': int(len(result)), 'train_rows': int(len(train)), 'cal_rows': int(len(cal)), 'test_rows': int(len(test)),
  'train_min_per_user': int(train.groupby('userID').size().min()),
  'future_test_by_construction': True,
  'test_after_train_fraction': float((test.timestamp.to_numpy() > train.groupby('userID').timestamp.max().reindex(test.userID).to_numpy()).mean()),
  'cal_after_train_fraction': float((cal.timestamp.to_numpy() > train.groupby('userID').timestamp.max().reindex(cal.userID).to_numpy()).mean()),
  'tie_rule': 'timestamp ascending, itemID ascending; last item is test and penultimate is calibration',
  'ratings_are_not_used_as_labels': True,
  'official_test_evaluated': False,
  'purpose': 'data preparation probe; no performance claim'
}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
'''

session = Session('213')
try:
    code, output = session.run_python(SOURCE % REMOTE_OUT, PYTHON, 180)
    print(output)
    raise SystemExit(code)
finally:
    session.close()
