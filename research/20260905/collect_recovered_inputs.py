"""Collect input audit and an inspectable sample without copying full datasets."""

import base64
import json
from pathlib import Path

from remote_session import Session


def main():
    source = '''
import base64, io, json, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
root = Path('/data/sunzixuan/rave_20260905_inputs/baby')
audit = json.loads((root / 'mapping_audit.json').read_text())
mapping = json.loads((root / 'item_id_to_asin.json').read_text())
asin_to_item = {asin: int(item_id) for item_id, asin in mapping.items()}
metadata = {row['item_id']: row for row in map(json.loads, (root / 'items.jsonl').read_text().splitlines())}
dtype = np.dtype([('asin', 'S10'), ('features', '<f4', (4096,))])
byte_limit = dtype.itemsize * 256
url = 'https://snap.stanford.edu/data/amazon/productGraph/image_features/categoryFiles/image_features_Baby.b'
request = urllib.request.Request(url, headers={'Range': 'bytes=0-%d' % (byte_limit - 1)})
with urllib.request.urlopen(request, timeout=30) as response:
    payload = response.read(byte_limit)
records = np.frombuffer(payload[:len(payload) // dtype.itemsize * dtype.itemsize], dtype=dtype)
matrix = np.load('/data/sunzixuan/expirement/LOBSTER/data/baby/image_feat.npy', mmap_mode='r')
checked = []
mismatched = []
for row in records:
    asin = row['asin'].decode('ascii')
    if asin in asin_to_item:
        item_id = asin_to_item[asin]
        if np.array_equal(np.asarray(matrix[item_id], dtype='<f4'), row['features']):
            checked.append(item_id)
        else:
            mismatched.append(item_id)
audit['independent_feature_alignment_sample'] = {'source_records_read': len(records),
    'matching_mapped_items': checked, 'mismatched_mapped_items': mismatched,
    'scope': 'sampled original visual vectors, not all catalog items'}
if mismatched or not checked:
    raise RuntimeError('Independent vector alignment check failed')
canvas = Image.new('RGB', (1000, 520), 'white')
draw = ImageDraw.Draw(canvas)
samples = []
for position, sample in enumerate(audit['image_access_sample']):
    item_id = sample['item_id']
    x, y = (position % 4) * 250, (position // 4) * 260
    if 'error' not in sample:
        path = root / ('sample_%05d.%s' % (item_id, sample['format'].lower()))
        product = Image.open(path).convert('RGB')
        product.thumbnail((238, 205))
        canvas.paste(product, (x + (250 - product.width) // 2, y + (210 - product.height) // 2))
    title = str(metadata[item_id].get('title') or '').encode('ascii', 'replace').decode('ascii')
    draw.text((x + 5, y + 212), str(item_id) + ' / ' + sample['asin'], fill='black')
    draw.text((x + 5, y + 230), title[:38], fill='black')
    draw.text((x + 5, y + 244), title[38:76], fill='black')
    samples.append({'item_id': item_id, 'asin': sample['asin'], 'title': metadata[item_id].get('title')})
buffer = io.BytesIO()
canvas.save(buffer, format='JPEG', quality=90)
print(json.dumps({'audit': audit, 'mapping': mapping, 'samples': samples,
                  'contact_sheet_base64': base64.b64encode(buffer.getvalue()).decode('ascii')}))
'''
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 100)
        if code:
            raise RuntimeError(output)
        record, _ = json.JSONDecoder().raw_decode(output.lstrip())
        root = Path(__file__).resolve().parent
        for key, filename in [('audit', 'mapping_audit.json'), ('mapping', 'baby_item_id_to_asin.json'),
                              ('samples', 'recovered_samples.json')]:
            (root / filename).write_text(json.dumps(record[key], indent=2), encoding='utf-8')
        (root / 'recovered_samples.jpg').write_bytes(base64.b64decode(record['contact_sheet_base64']))
        print('mapped_items', record['audit']['mapped_items'])
        print('coverage', record['audit']['metadata_coverage'])
        print('alignment', record['audit']['independent_feature_alignment_sample'])
        print('contact_sheet', root / 'recovered_samples.jpg')
    finally:
        session.close()


if __name__ == '__main__':
    main()
