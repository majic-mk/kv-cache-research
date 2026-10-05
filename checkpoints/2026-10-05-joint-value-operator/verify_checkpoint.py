"""Read-only package integrity check. No research code is executed."""
from pathlib import Path, PurePosixPath
import hashlib, json, zipfile
root=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
m=json.loads((root/'CHECKPOINT_MANIFEST.json').read_text())
for row in m['files_except_this_manifest']:
    b=(root/row['file']).read_bytes()
    assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['file']
s=json.loads((root/'SOURCE_SELECTION_MANIFEST.json').read_text())
with zipfile.ZipFile(root/'Joint_Value_Operator_Source_2026-10-05.zip') as z:
    assert z.testzip() is None
    assert len(z.namelist())==len(set(z.namelist()))==s['source_member_count']
    for row in s['files']:
        p=PurePosixPath(row['path']);assert not p.is_absolute() and '..' not in p.parts
        b=z.read(row['path']);assert len(b)==row['bytes'] and sha(b)==row['sha256'],row['path']
print('PASS: checkpoint file hashes and all source ZIP members; no research run')
