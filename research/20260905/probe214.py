import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_session import Session

source = r'''
import json
from pathlib import Path
base=Path('/data/sunzixuan')
out=[]
for p in sorted(base.iterdir()):
    if p.is_dir() and any(x in p.name.lower() for x in ('sonic','simgr','mmrec','lobster','official')):
        out.append({'path':str(p), 'entries':[x.name for x in list(p.iterdir())[:20]]})
print(json.dumps(out, indent=2))
'''
s=Session('214')
try:
    code,out=s.run_python(source,'/data/sunzixuan/miniconda3/envs/MMRec/bin/python',60)
    print(out); raise SystemExit(code)
finally: s.close()
