"""Archive small exact recovery receipts and verify owned process identities."""
import argparse
import base64
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = '/data/sunzixuan/electronics_baseline_20261010_recovery_v3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not args.tag.replace('_', '').replace('-', '').isalnum():
        parser.error('Use a simple unique archive tag')
    target = HERE / args.tag
    if target.exists():
        raise RuntimeError('Preserve prior status snapshots; choose a new tag')
    source = r'''
import base64,hashlib,json,subprocess
from pathlib import Path
root=Path(%r)
names=['deployment.json','source_revision.json','test_receipt.json','launch.json','pipeline_status.json','results.tsv']
state=json.loads((root/'pipeline_status.json').read_text()) if (root/'pipeline_status.json').exists() else {}
for seed in range(2026,2031):
    folder='runs/elec_projector_recovery_s'+str(seed)+'/'
    names += [folder+name for name in ['config.json','failure.json','numerics_receipt.json','result.json','progress.jsonl']]
def owned(entry):
    try:
        actual=Path('/proc',str(entry['pid']),'cmdline').read_bytes().split(b'\0')[:-1]
        return actual==[value.encode() for value in entry['command']]
    except (KeyError,FileNotFoundError):
        return False
launch=json.loads((root/'launch.json').read_text()) if (root/'launch.json').exists() else {}
files={}
for name in names:
    path=root/name
    if path.is_file():
        if path.stat().st_size>4*1024*1024:raise RuntimeError('Unexpected large receipt: '+name)
        content=path.read_bytes()
        files[name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content),'base64':base64.b64encode(content).decode('ascii')}
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
logs={}
for path in (root/'logs').glob('*.log') if (root/'logs').exists() else []:
    with path.open('rb') as stream:
        stream.seek(max(0,path.stat().st_size-4000));logs[path.name]=stream.read().decode(errors='replace')
print(json.dumps({'root':str(root),'files':files,'queue_alive_matching_command':owned(launch),
    'active_job_alive_matching_command':owned(state.get('active_job') or {}),
    'gpu_process_query_succeeded':gpu.returncode==0,'gpu_compute_processes':gpu.stdout.strip(),
    'pipeline_summary':{key:state.get(key) for key in ['status','stage','current_seed','completed_seeds','seed_states','active_job','error']},
    'owned_log_tails':logs}))
''' % REMOTE
    sys.path.insert(0, str(HERE.parent / '20260905'))
    from remote_session import Session
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 120)
        if code:
            raise RuntimeError('Read-only status capture failed')
        receipt = json.JSONDecoder().raw_decode(output.lstrip())[0]
    finally:
        session.close()
    target.mkdir()
    for name, row in receipt['files'].items():
        content = base64.b64decode(row.pop('base64'))
        if hashlib.sha256(content).hexdigest() != row['sha256'] or len(content) != row['bytes']:
            raise RuntimeError('Transferred receipt differs: ' + name)
        archive_name = 'seeds/' + name[len('runs/'):] if name.startswith('runs/') else name
        row['archive_path'] = archive_name
        destination = target / archive_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    receipt['captured_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (target / 'capture_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: receipt[key] for key in ('queue_alive_matching_command',
        'active_job_alive_matching_command', 'pipeline_summary', 'owned_log_tails')}, indent=2))


if __name__ == '__main__':
    main()
