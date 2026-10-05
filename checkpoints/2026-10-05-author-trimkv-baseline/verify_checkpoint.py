"""Read-only package hashes and ZIP members; never imports archived research."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile
root=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
m=json.loads((root/'CHECKPOINT_MANIFEST.json').read_text())
for row in m['files_except_this_manifest']:
 p=PurePosixPath(row['file']);assert not p.is_absolute() and '..' not in p.parts
 b=(root/row['file']).read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['file']
s=json.loads((root/'SOURCE_SELECTION_MANIFEST.json').read_text())
with zipfile.ZipFile(root/'TRIMKV_PhaseA_Baseline_Source_2026-10-05.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==s['source_member_count']
 for row in s['files']:
  p=PurePosixPath(row['path']);assert not p.is_absolute() and '..' not in p.parts
  b=z.read(row['path']);assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['path']
print('PASS: checkpoint hashes and all source ZIP members; no research run or CI claim')
