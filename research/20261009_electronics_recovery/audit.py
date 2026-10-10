"""Read-only structural audit of the recovered host213 Electronics data."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/20260905"))
from remote_session import Session
from recover import TARGET_ROOT, PYTHON


def main():
    source = r'''
from pathlib import Path
import csv, json, numpy as np
root = Path(%r)
features = {}
for name in ('image_feat.npy', 'text_feat.npy'):
    array = np.load(root / name, mmap_mode='r', allow_pickle=False)
    finite = all(np.isfinite(array[start:start + 256]).all()
                 for start in range(0, array.shape[0], 256))
    features[name] = {'shape': list(array.shape), 'dtype': str(array.dtype),
                      'all_finite': bool(finite)}
    if not finite: raise RuntimeError('Non-finite feature: ' + name)
users, items = set(), set()
split_counts = {}
rows = 0
with (root / 'elec.inter').open('r', encoding='utf-8', newline='') as stream:
    reader = csv.DictReader(stream, delimiter='\t')
    fields = reader.fieldnames
    if fields != ['userID', 'itemID', 'rating', 'timestamp', 'x_label']:
        raise RuntimeError('Unexpected interaction schema: ' + repr(fields))
    for row in reader:
        rows += 1
        user, item, label = int(row['userID']), int(row['itemID']), int(row['x_label'])
        if user < 0 or item < 0 or label not in (0, 1, 2):
            raise RuntimeError('Invalid interaction ID or split label')
        users.add(user); items.add(item)
        split_counts[label] = split_counts.get(label, 0) + 1
if len(users) != max(users) + 1 or len(items) != max(items) + 1:
    raise RuntimeError('IDs are not contiguous zero-based ranges')
if any(row['shape'][0] != len(items) for row in features.values()):
    raise RuntimeError('Feature rows differ from item ID range')
print(json.dumps({'root': str(root), 'schema': fields, 'interaction_rows': rows,
                  'user_count': len(users), 'item_count': len(items),
                  'split_counts': split_counts, 'features': features,
                  'graph_cache_exists': (root / 'mm_adj_10.pt').exists(),
                  'id_ranges_contiguous_zero_based': True,
                  'audit_read_only': True, 'training_started': False}, sort_keys=True))
''' % TARGET_ROOT
    session = Session("213")
    try:
        code, output = session.run_python(source, PYTHON, 240)
    finally:
        session.close()
    if code:
        raise RuntimeError("Target structure audit failed: " + output[-2000:])
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    (Path(__file__).resolve().parent / "target_structure.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

