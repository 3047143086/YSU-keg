"""Read-only temporal/rating statistics for deciding the next experiment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '20260905'))
from remote_session import Session

PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'
SOURCE = r'''
import json
from pathlib import Path
import numpy as np
import pandas as pd
path = Path('/data/sunzixuan/expirement/LOBSTER/data/baby/baby.inter')
frame = pd.read_csv(path, sep='\t')
frame['timestamp'] = pd.to_numeric(frame['timestamp'])
frame['rating'] = pd.to_numeric(frame['rating'])
user_order = frame.sort_values(['userID','timestamp']).groupby('userID')['timestamp'].agg(['min','max','count'])
user_label = frame.groupby('userID')['x_label'].agg(['sum','count'])
out = {
  'path': str(path), 'rows': int(len(frame)), 'columns': frame.columns.tolist(),
  'users': int(frame.userID.nunique()), 'items': int(frame.itemID.nunique()),
  'timestamp_min': int(frame.timestamp.min()), 'timestamp_max': int(frame.timestamp.max()),
  'unique_timestamps': int(frame.timestamp.nunique()),
  'timestamp_label_counts': frame.groupby('x_label').timestamp.agg(['min','max','count']).to_dict('index'),
  'rating_label_means': frame.groupby('x_label').rating.mean().to_dict(),
  'label_counts': frame.x_label.value_counts().sort_index().to_dict(),
  'users_with_both_labels': int((user_label['sum'] > 0).sum()),
  'user_history_count_quantiles': np.quantile(user_order['count'], [0,.25,.5,.75,1]).tolist(),
  'user_time_span_days_quantiles': (np.quantile(user_order['max']-user_order['min'], [0,.25,.5,.75,1]) / 86400).tolist(),
  'chronological_order_ties_fraction': float((frame.groupby(['userID','timestamp']).size() > 1).mean()),
}
support_max = frame[frame.x_label == 0].groupby('userID').timestamp.max()
held = frame[frame.x_label != 0].copy()
held['support_max'] = held.userID.map(support_max)
out['held_after_support_fraction'] = float((held.timestamp > held.support_max).mean())
out['held_same_or_before_support_fraction'] = float((held.timestamp <= held.support_max).mean())
out['held_timestamp_gap_days_quantiles'] = (np.quantile(held.timestamp - held.support_max, [0,.25,.5,.75,1]) / 86400).tolist()
out['per_user_first_held_after_support_fraction'] = float((held.groupby('userID').first().timestamp > held.groupby('userID').first().support_max).mean())
print(json.dumps(out, indent=2))
'''

session = Session('213')
try:
    code, output = session.run_python(SOURCE, PYTHON, 120)
    print(output)
    raise SystemExit(code)
finally:
    session.close()
