"""Download the user-selected upstream using the Codex installer's ZIP validation."""
import hashlib
import json
import runpy
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
installer = Path('C:/Users/zixua/.codex/skills/.system/skill-installer/scripts')
sys.path.insert(0, str(installer))
helpers = runpy.run_path(str(installer / 'install-skill-from-github.py'))
commit = json.loads(helpers['_request'](
    'https://api.github.com/repos/karpathy/autoresearch/commits/master'))['sha']
destination = root / 'upstream'
destination.mkdir(exist_ok=True)
downloaded = Path(helpers['_download_repo_zip']('karpathy', 'autoresearch', commit, str(destination)))
record = {'repository': 'https://github.com/karpathy/autoresearch', 'commit': commit,
          'local_checkout': str(downloaded), 'installation_kind': 'local Codex adapter for upstream program.md',
          'hashes': {name: hashlib.sha256((downloaded / name).read_bytes()).hexdigest()
                     for name in ('README.md', 'program.md', 'prepare.py', 'train.py', 'pyproject.toml')}}
(root / 'skill-source.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps(record, indent=2))
