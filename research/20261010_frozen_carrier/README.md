# Frozen original BRIDGE carrier

This directory contains the 35 source/configuration files actually used by
the frozen BRIDGE baseline carrier. They were read from the isolated
Electronics v1 carrier and verified byte for byte against the previously
declared carrier SHA256 manifest. `manifest.json` records all 35 hashes and
file sizes. Original bytes and line endings are preserved; the editable
workspace BRIDGE tree may have different line endings or later changes.

This is the original baseline snapshot, not a new method or evidence of a
successful publication experiment. It includes the baseline implementation
and optional historical branches/configuration files because the complete
35-file carrier is preserved. Enabling those optional branches changes the
algorithm and requires a separate declared protocol.

The original BRIDGE license is retained in `LICENSE` (GPL-3.0). No data,
features, graph caches, model weights, checkpoints, or credentials are
included. Source export was read-only on the server. `export.py` records the
read/transfer/verification procedure and uses the local research connection
helper; that authentication helper is deliberately not bundled here.

To verify the downloaded snapshot locally:

```python
import hashlib, json
from pathlib import Path
root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())
assert manifest['file_count'] == 35
for name, row in manifest['files'].items():
    value = (root / 'src' / name).read_bytes()
    assert len(value) == row['bytes']
    assert hashlib.sha256(value).hexdigest() == row['sha256']
```
