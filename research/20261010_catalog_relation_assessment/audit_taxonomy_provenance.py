"""Compare explicit categories in an existing raw catalog and its exported rows."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

LOCAL = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL.parent / '20260905'))
from remote_session import Session

REMOTE = r'''
import ast,gzip,hashlib,json,time
from pathlib import Path
from collections import Counter
root=Path('/data/sunzixuan/rave_20260905_inputs/baby')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
items=root/'items.jsonl'
raw=root/'meta_Baby.json.gz'
certificate=root/'mapping_audit.json'
record={'checked_unix':time.time(),'gpu_initialized':False,'reviews_read':False,'interactions_read':False,
        'raw_metadata_path':str(raw),'exported_metadata_path':str(items),'certificate_path':str(certificate)}
if not raw.is_file():
 record['status']='blocked_missing_raw_metadata'
else:
 exported={row['asin']:row.get('categories') for row in map(json.loads,items.read_text().splitlines())}
 audit=json.loads(certificate.read_text())
 record.update(raw_sha256=sha(raw),exported_sha256=sha(items),certificate_sha256=sha(certificate))
 expected=audit.get('downloads',{}).get('metadata',{}).get('sha256')
 record['raw_hash_matches_mapping_certificate']=record['raw_sha256']==expected
 matched=equal=duplicates=0
 seen=set();structures=Counter();pathlengths=Counter();signatures=Counter()
 with gzip.open(raw,'rt',encoding='utf-8') as stream:
  for line in stream:
   if not line.strip():continue
   try:row=json.loads(line)
   except json.JSONDecodeError:row=ast.literal_eval(line)
   asin=row.get('asin')
   if asin not in exported:continue
   if asin in seen:duplicates+=1;continue
   seen.add(asin);matched+=1
   categories=row.get('categories')
   equal+=categories==exported[asin]
   structures[type(categories).__name__]+=1
   if isinstance(categories,list):
    for path in categories:
     if isinstance(path,list):pathlengths[len(path)]+=1
   signatures[json.dumps(categories,sort_keys=True)]+=1
 record.update(status='completed',selected_asins=len(exported),matched_asins=matched,
               raw_exported_categories_equal_count=equal,raw_exported_category_mismatch_count=matched-equal,
               duplicate_selected_asins=duplicates,raw_category_signature_count=len(signatures),
               raw_selected_path_length_counts=dict(pathlengths),raw_category_schema_counts=dict(structures),
               raw_category_frequency_histogram=dict(Counter(signatures.values())))
print(json.dumps(record,indent=2,allow_nan=False))
'''


def main():
    git = ['git', '-c', 'safe.directory=' + LOCAL.as_posix()]
    commit = subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=str(LOCAL), text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'], cwd=str(LOCAL), text=True).strip()
    if dirty:
        raise RuntimeError('Commit provenance diagnostic before execution')
    path = Path(__file__)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    compile(REMOTE, 'taxonomy_provenance_remote', 'exec')
    session = Session('213')
    try:
        code, output = session.run_python(REMOTE, '/data/sunzixuan/miniconda3/envs/MMRec/bin/python', 240)
    finally:
        session.close()
    if code:
        raise RuntimeError(output[-2000:])
    record, _ = json.JSONDecoder().raw_decode(output.lstrip())
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if before != after:
        raise RuntimeError('Provenance source changed during audit')
    record.update(source_commit=commit,source_sha256=before,post_source_sha256=after)
    (LOCAL/'taxonomy_provenance_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__ == '__main__':
    main()
