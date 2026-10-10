"""Inspect input schemas and raw-content availability without evaluating test."""

import json
import os
from pathlib import Path

import numpy as np


roots = [Path('/data/sunzixuan/expirement'), Path('/data/sunzixuan/bridge_official/data')]
for root in roots:
    if not root.exists():
        continue
    print('DATA_ROOT', str(root), 'resolved', str(root.resolve()))
    for current, directories, files in os.walk(root):
        depth = len(Path(current).relative_to(root).parts)
        directories[:] = [d for d in directories if not d.startswith('.') and d not in
                          {'saved_models', 'log', 'logs', '__pycache__', 'cache', 'env', 'node_modules'}]
        if depth >= 4:
            directories[:] = []
        for filename in files:
            if (filename in {'inter.json', 'README.md'} and '/data/' in current) or any(
                word in filename.lower() for word in ('asin', 'mapping', 'metadata', 'raw', 'clip', 'title')
            ):
                path = Path(current) / filename
                print('POTENTIAL_INPUT', str(path), path.stat().st_size)

for domain in ['baby', 'sports']:
    root = Path('/data/sunzixuan/expirement/LOBSTER/data') / domain
    for name in ['image_feat.npy', 'text_feat.npy']:
        matrix = np.load(root / name, mmap_mode='r')
        print('FEATURE', domain, name, matrix.shape, str(matrix.dtype))
    info = json.loads((root / 'inter.json').read_text())
    print('INTER_METADATA_TYPE', domain, type(info).__name__)
    if isinstance(info, dict):
        print('INTER_METADATA_KEYS', list(info)[:20])
        for key, value in list(info.items())[:8]:
            print('INTER_METADATA_FIELD', key, type(value).__name__,
                  len(value) if isinstance(value, (dict, list, str)) else value)
    with (root / (domain + '.inter')).open() as handle:
        print('INTER_HEADER', handle.readline().strip())

for domain in ['baby', 'sports']:
    root = Path('/data/sunzixuan/bridge_official/src/saved_models/BRIDGE') / domain
    for path in root.glob('*/model.pth'):
        print('CHECKPOINT', str(path), path.stat().st_size)

for relative in ['src/configs/overall.yaml', 'src/utils/dataloader.py']:
    path = Path('/data/sunzixuan/bridge_official') / relative
    if relative.endswith('.yaml'):
        print('OVERALL_CONFIG', path.read_text())
