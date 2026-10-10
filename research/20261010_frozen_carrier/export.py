"""Read and verify only the 35 frozen BRIDGE source/configuration files."""
import base64
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
WORKSPACE = LOCAL.parents[1]
EXPECTED = WORKSPACE / 'research/20261009_electronics_baseline/input_audit.json'
REMOTE_ROOT = '/data/sunzixuan/electronics_baseline_20261009_v1/carrier/src'
PYTHON = '/data/sunzixuan/miniconda3/envs/MMRec/bin/python'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    declared = json.loads(EXPECTED.read_text(encoding='utf-8'))['carrier_files']
    if len(declared) != 35:
        raise RuntimeError('Expected exactly 35 source/configuration paths')
    expected = {name: row['sha256'] for name, row in declared.items()}
    for name in expected:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise RuntimeError('Unsafe manifest path')
    source = r'''
import base64,hashlib,json
from pathlib import Path
root=Path(%r)
expected=%r
files={}
for name,wanted in expected.items():
    path=root/name
    value=path.read_bytes()
    if len(value)>8*1024*1024:raise RuntimeError('Unexpected large source file: '+name)
    digest=hashlib.sha256(value).hexdigest()
    if digest!=wanted:raise RuntimeError('Frozen source mismatch: '+name)
    files[name]={'sha256':digest,'bytes':len(value),'base64':base64.b64encode(value).decode('ascii')}
print(json.dumps({'root':str(root),'files':files},allow_nan=False))
''' % (REMOTE_ROOT, expected)
    sys.path.insert(0, str(WORKSPACE / 'research/20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(source, PYTHON, 180)
        if code:
            raise RuntimeError('Read-only remote export failed: ' + output[-1500:])
    finally:
        session.close()
    packet = json.JSONDecoder().raw_decode(output.lstrip())[0]
    if set(packet['files']) != set(expected):
        raise RuntimeError('Remote source path coverage differs')
    content = {}
    for name, row in packet['files'].items():
        value = base64.b64decode(row['base64'], validate=True)
        if len(value) != row['bytes'] or sha(value) != expected[name] or row['sha256'] != expected[name]:
            raise RuntimeError('Transferred source mismatch: ' + name)
        content[name] = value
    # Every transfer is checked before writing any frozen source byte.
    for name, value in content.items():
        target = LOCAL / 'src' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != value:
            raise RuntimeError('Existing export differs; refusing overwrite: ' + name)
        if not target.exists():
            target.write_bytes(value)
    observed = {name: sha((LOCAL / 'src' / name).read_bytes()) for name in expected}
    if observed != expected:
        raise RuntimeError('Local frozen source verification failed')
    license_source = WORKSPACE / 'baselines/BRIDGE-official/LICENSE'
    license_target = LOCAL / 'LICENSE'
    if license_target.exists() and license_target.read_bytes() != license_source.read_bytes():
        raise RuntimeError('Existing license differs')
    if not license_target.exists():
        shutil.copyfile(license_source, license_target)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'remote_source_root': REMOTE_ROOT, 'expected_manifest': str(EXPECTED.relative_to(WORKSPACE)),
                'expected_manifest_sha256': sha(EXPECTED.read_bytes()),
                'purpose': 'original frozen baseline source snapshot; not a new method',
                'file_count': 35, 'all_sha256_match': True, 'line_endings_preserved': True,
                'files': {name: {'sha256': expected[name], 'bytes': len(content[name])} for name in sorted(expected)},
                'license_source': str(license_source.relative_to(WORKSPACE)),
                'license_sha256': sha(license_source.read_bytes()),
                'data_downloaded': False, 'weights_downloaded': False,
                'credentials_downloaded': False, 'remote_files_modified': False}
    (LOCAL / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'status': 'completed', 'source_files': 35, 'all_sha256_match': True,
                      'bytes': sum(map(len, content.values())), 'license_sha256': manifest['license_sha256'],
                      'manifest': str(LOCAL / 'manifest.json')}, indent=2))


if __name__ == '__main__':
    main()
