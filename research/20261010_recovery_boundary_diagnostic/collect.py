"""Run the independent CPU-only read diagnostic and preserve its exact JSON bytes."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / '20260905'))
from remote_session import Session


def main():
    source = (ROOT / 'remote_diagnostic.py').read_text(encoding='utf-8')
    receipt_path = ROOT / 'remote_receipt.json'
    if receipt_path.exists():
        raise RuntimeError('This diagnostic receipt already exists; do not overwrite it')
    client_started = datetime.now(timezone.utc).isoformat()
    session = Session('213')
    try:
        code, output = session.run_python(source, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 900)
        if code:
            raise RuntimeError('CPU diagnostic failed: ' + output[-2000:])
        raw = output.split('RECOVERY_BOUNDARY_DIAGNOSTIC_JSON\n', 1)[1]
        parsed, end = json.JSONDecoder().raw_decode(raw)
        exact = raw[:end].encode('utf-8')
        with receipt_path.open('xb') as stream:
            stream.write(exact)
        metadata = {'client_transport_started_utc': client_started,
            'client_captured_utc': datetime.now(timezone.utc).isoformat(), 'clock_offset_measured': False,
            'remote_receipt_sha256': hashlib.sha256(exact).hexdigest(), 'remote_receipt_bytes': len(exact),
            'remote_script_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest()}
        with (ROOT / 'capture_metadata.json').open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, allow_nan=False)
        print(json.dumps({'status': parsed['status'], 'saved_capture_count': parsed['saved_capture_count'],
            'failure_record': parsed['failure_record'], 'elapsed_seconds': parsed['elapsed_seconds'],
            'packet_sha256': parsed['packet_sha256_before'], 'receipt_sha256': hashlib.sha256(exact).hexdigest(),
            'all_protected_hashes_match': parsed['all_protected_hashes_match'],
            'boundary_rows': [{'capture_index': row['capture_index'], 'shape': row['shape'],
                               'boundary_rows': row.get('boundary_rows')} for row in parsed['captures']]}, indent=2))
    finally:
        session.close()


if __name__ == '__main__':
    main()
