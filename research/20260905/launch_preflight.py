"""Upload and launch an isolated diagnostic on an idle GPU, without retraining."""

import argparse
import base64
from pathlib import Path

from remote_session import Session


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['baby', 'sports'], default='baby')
    args = parser.parse_args()
    run_root = {'baby': '/data/sunzixuan/rave_20260905_preflight_v2',
                'sports': '/data/sunzixuan/rave_20260905_sports_preflight_v2'}[args.dataset]
    checkpoint_name = {'baby': 'bridge_baby_seed2026_screen',
                       'sports': 'bridge_sports_s2026_confirm'}[args.dataset]
    checkpoint = ('/data/sunzixuan/bridge_official/src/saved_models/BRIDGE/'
                  + args.dataset + '/' + checkpoint_name + '/model.pth')
    script = Path(__file__).with_name('preflight.py').read_bytes()
    payload = base64.b64encode(script).decode('ascii')
    source = f'''
import base64, json, os, subprocess
from pathlib import Path
root = Path({run_root!r})
if root.exists():
    raise RuntimeError('Diagnostic directory exists; inspect it before any retry')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
available = []
for line in gpu.strip().splitlines():
    index, memory, utilization = [int(value.strip()) for value in line.split(',')]
    if memory < 500 and utilization == 0:
        available.append(index)
if not available:
    raise RuntimeError('No idle GPU found; no task launched')
root.mkdir()
(root / 'preflight.py').write_bytes(base64.b64decode({payload!r}))
command = ['/data/sunzixuan/miniconda3/envs/MMRec/bin/python', '-u', str(root / 'preflight.py'),
 '--bridge-root', '/data/sunzixuan/bridge_official',
 '--checkpoint', {checkpoint!r},
 '--dataset', {args.dataset!r}, '--gpu-id', str(available[0]), '--output', str(root / '{args.dataset}.json')]
environment = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
with (root / 'stdout.log').open('xb') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment)
receipt = {{'pid': process.pid, 'root': str(root), 'gpu': available[0], 'command': command}}
(root / 'launch.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
'''
    session = Session('213')
    try:
        code, output = session.run_python(source, 'python', 60)
        print(output)
        receipt_file = 'launch_receipt_' + args.dataset + '_v2.txt'
        Path(__file__).with_name(receipt_file).write_text(output + '\n', encoding='utf-8')
        raise SystemExit(code)
    finally:
        session.close()


if __name__ == '__main__':
    main()
