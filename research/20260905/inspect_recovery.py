"""Read input-recovery progress and available feature-extraction dependencies."""

import importlib.metadata
import json
from pathlib import Path


root = Path('/data/sunzixuan/rave_20260905_inputs/baby')
print('RECOVERY_FILES', json.dumps({path.name: path.stat().st_size for path in root.iterdir()}))
for name in ('mapping_audit.json', 'mapping_audit_incomplete.json'):
    if (root / name).exists():
        print('AUDIT', (root / name).read_text())
packages = {}
for name in ('torch', 'torchvision', 'transformers', 'sentence-transformers', 'open-clip-torch', 'clip', 'Pillow'):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print('PACKAGES', json.dumps(packages))
cache = Path('/data/sunzixuan/.cache/huggingface/hub')
print('MODEL_CACHE', json.dumps([path.name for path in cache.iterdir() if path.is_dir()] if cache.exists() else []))
