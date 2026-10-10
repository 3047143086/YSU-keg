"""Export the experiment workspace without data, credentials, or nested Git stores."""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {'.py', '.md', '.json', '.tsv', '.yaml', '.yml', '.toml', '.txt', '.patch'}
SKIP_PARTS = {'.git', '__pycache__', 'data', 'runs', '.venv', 'paper_figures_template'}
SKIP_NAMES = {'SERVER_CONNECTION.md', 'recover.py', 'remote_audit.txt'}
SKIP_PATHS = {'research/20260905/RESEARCH_PROPOSAL.md'}
SENSITIVE = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'(?i)(?:authorization\s*[:=]\s*[\"\']?bearer\s+)[A-Za-z0-9_.-]{20,}'),
    re.compile(r'(?i)(?:password|api_key|access_token|client_secret)\s*[:=]\s*[\"\'][^\"\'\n]{8,}[\"\']'),
    re.compile(r'https?://[^\s/]*(?:feishu\.cn|larksuite\.com|doubao\.com)/(?:docx|wiki|sheets|base|drive|file|slides)/[^\s)\]<>]+', re.IGNORECASE),
]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def check_sharing(content, label):
    text = content.decode('utf-8-sig', errors='replace')
    if any(pattern.search(text) for pattern in SENSITIVE):
        raise SystemExit('Potential credential or private sharing link; review required: ' + label)


def revision(directory):
    # Inspect only actual nested repositories, never inherit the export repo.
    if not (directory / '.git').is_dir():
        return None
    result = subprocess.run(['git', '-C', str(directory), 'rev-parse', '--verify', 'HEAD'],
                            text=True, capture_output=True)
    head = result.stdout.strip() if result.returncode == 0 else None
    changed = subprocess.run(['git', '-C', str(directory), 'status', '--porcelain'],
                             text=True, capture_output=True)
    return {'commit': head,
            'working_tree_has_changes': bool(changed.stdout.strip()) if changed.returncode == 0 else None,
            'git_metadata_readable': result.returncode == 0 and changed.returncode == 0}


def allowed(path, workspace):
    rel = path.relative_to(workspace)
    if rel.as_posix() in SKIP_PATHS:
        return False
    if any(part in SKIP_PARTS for part in rel.parts) or path.name in SKIP_NAMES:
        return False
    # Include only declared epoch-progress receipts in the small status archives.
    # Do not enable arbitrary JSONL datasets or raw training directories.
    status_progress = (len(rel.parts) == 6
                       and rel.parts[:2] == ('research', '20261010_electronics_recovery_status')
                       and rel.parts[3] == 'seeds'
                       and rel.parts[4] in {'elec_projector_recovery_s' + str(seed) for seed in range(2026, 2031)}
                       and path.name == 'progress.jsonl')
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {'LICENSE', 'LICENSE.txt'} and not status_progress:
        return False
    return path.stat().st_size <= 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if workspace == REPO or workspace in REPO.parents and REPO.parent != workspace:
        raise SystemExit('Expected the experiment workspace containing this export repository')
    for name in ('.gitattributes', '.gitignore', 'README.md', 'SOURCE_MANIFEST.json', 'tools/sync_snapshot.py'):
        path = REPO / name
        if path.exists():
            check_sharing(path.read_bytes(), name)
    roots = [workspace / 'research', workspace / 'result']
    for name in ('PRAVI', 'SiMGR2025', 'baselines/BRIDGE-official'):
        base = workspace / name
        roots.extend(base / child for child in ('src', 'docs', 'tests'))
        roots.extend(base.glob('*.md'))
        roots.extend(base.glob('LICENSE*'))
        roots.extend(base.glob('*requirements*.txt'))
        roots.extend(base.glob('environment.yaml'))
    sources = {}
    for root in roots:
        if not root.exists():
            continue
        for path in ([root] if root.is_file() else root.rglob('*')):
            if not path.is_file() or path.is_symlink() or not allowed(path, workspace):
                continue
            content = path.read_bytes()
            rel = path.relative_to(workspace).as_posix()
            check_sharing(content, rel)
            sources[rel] = (path, content)
    old_manifest = REPO / 'SOURCE_MANIFEST.json'
    previous = json.loads(old_manifest.read_text(encoding='utf-8')) if old_manifest.exists() else {'files': {}}
    # Only remove previously exported files, never an unlisted repository file.
    for rel in previous['files'].keys() - sources.keys():
        target = (REPO / rel).resolve()
        if REPO not in target.parents or '.git' in target.relative_to(REPO).parts:
            raise SystemExit('Unsafe manifest path')
        if target.is_file():
            target.unlink()
    entries = {}
    for rel, (source, content) in sorted(sources.items()):
        target = REPO / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != content:
            shutil.copyfile(source, target)
        if target.read_bytes() != content:
            raise RuntimeError('Export differs from source: ' + rel)
        entries[rel] = {'sha256': digest(content), 'bytes': len(content)}
    provenance = {}
    for directory in sorted((workspace / 'research').iterdir()):
        if directory.is_dir():
            value = revision(directory)
            if value:
                provenance[directory.relative_to(workspace).as_posix()] = value
    manifest = {'format': 1, 'description': 'Exact source snapshot; private assets excluded',
                'source_repositories': provenance, 'files': entries}
    with old_manifest.open('w', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'files': len(entries), 'bytes': sum(row['bytes'] for row in entries.values())}))


if __name__ == '__main__':
    main()
