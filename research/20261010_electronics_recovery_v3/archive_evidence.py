"""Read only the explicitly preregistered small audit receipts and source files."""
import base64
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'prerequisites.json').read_bytes())
    wanted = {}
    for role, entry in manifest['entries'].items():
        folder = HERE / 'receipts' / 'exact' / role
        folder.mkdir(parents=True, exist_ok=True)
        for field in ('path', 'source_config_path'):
            source = entry[field]
            if source.startswith('/data/sunzixuan/'):
                wanted[source] = str((folder / Path(source).name).relative_to(HERE)).replace('\\', '/')
        for name in entry['payload_sha256']:
            source = str(Path(entry['source_config_path']).parent / name).replace('\\', '/')
            wanted[source] = str((folder / name).relative_to(HERE)).replace('\\', '/')
    source = "import base64,json\nfrom pathlib import Path\nwanted=" + repr(wanted) + "\nprint(json.dumps({p:base64.b64encode(Path(p).read_bytes()).decode() for p in wanted}))"
    sys.path.insert(0, str(HERE.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 120)
        if code:
            raise RuntimeError('Whitelisted evidence read failed')
        received = json.JSONDecoder().raw_decode(output.lstrip())[0]
        if set(received) != set(wanted):
            raise RuntimeError('Read-only archive coverage changed')
        for remote, encoded in received.items():
            target = HERE / wanted[remote]
            content = base64.b64decode(encoded)
            if target.exists() and target.read_bytes() != content:
                raise RuntimeError('Existing archive differs: ' + str(target.relative_to(HERE)))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        for role, entry in manifest['entries'].items():
            folder = 'receipts/exact/' + role + '/'
            for field in ('path', 'source_config_path'):
                remote = entry[field]
                if remote.startswith('/data/sunzixuan/'):
                    target = HERE / folder / Path(remote).name
                    target.write_bytes(base64.b64decode(received[remote]))
            for name in entry['payload_sha256']:
                remote = str(Path(entry['source_config_path']).parent / name).replace('\\', '/')
                target = HERE / folder / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(base64.b64decode(received[remote]))
            entry['local_source_config_path'] = folder + Path(entry['source_config_path']).name
            if entry['path'].startswith('/data/sunzixuan/'):
                entry['local_path'] = folder + Path(entry['path']).name
            else:
                entry['local_path'] = entry['path']
            for field in ('path', 'source_config_path'):
                expected = entry['sha256' if field == 'path' else 'source_config_sha256']
                local = entry['local_path' if field == 'path' else 'local_source_config_path']
                if hashlib.sha256((HERE / local).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Exact original evidence hash mismatch: ' + role + '/' + field)
            for name, expected in entry['payload_sha256'].items():
                if hashlib.sha256((HERE / folder / name).read_bytes()).hexdigest() != expected:
                    raise RuntimeError('Exact original payload mismatch: ' + role + '/' + name)
        (HERE / 'prerequisites.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        print(json.dumps({'status': 'archived_exact', 'files_read': len(wanted),
                          'training_started': False, 'recovery_deployed': False}))
    finally:
        session.close()


if __name__ == '__main__':
    main()
