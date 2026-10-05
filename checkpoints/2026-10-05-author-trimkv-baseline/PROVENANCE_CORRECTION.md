# Additive provenance correction

This corrects the earlier completion packet without changing it or any original evidence.

The worker’s post-run verification checked frozen pins, then ran a plain `python -` import of the v4 decoder without `-B` or `sys.dont_write_bytecode`. At14:43:27UTC, after the numerical run ended14:40:49, Python rewrote the derived `__pycache__/evidence_codec.cpython-312.pyc` file. The earlier statement that all pins remained unchanged after verification was too broad, and the verification was not fully read-only.

Frozen cache SHA256:2378944d3a76fa0f5b3eae296b5a0ce2a233302a6ae6e3c125778c0b54d1a55b

Post-run cache SHA256:02945da0a1b2c4cad3b0bddfbb3739e637b7eba0d82b5739d0c2283355b11af4

Exact original bytes still exist in the v3 cache. The v4 preparation copied that incidental derived cache and later pinned it. Its stored source timestamp was13:50:42, whereas the unchanged v4 source timestamp was14:18:37. The standard CPython timestamp check would reject it; `-B` alone suppresses writes, not reads. No loader trace was recorded during the run, so this usage assessment is explicitly based on source/header inspection.

Read-only inspection using `python -I -S -B` compared all27 code objects: all fields/constants match except `co_filename` changing from the v3 source path to v4. The source SHA remains03ff37ede9adb969eadc11c59ac5ac9e3aa3b96912b8c8656de1feca0f596d69. Original result, trace, guard, resource samples and completion-report hashes were rechecked unchanged. No numerical rerun or cache restoration was performed.

The restricted Phase A conclusion must cite the independent terminal audit’s disclosed cache exception. The original reports and both cache states remain preserved; this addendum does not silently repair or replace history. PROVENANCE_CORRECTION.json contains the exact route, chronology, hashes and source validation evidence.
