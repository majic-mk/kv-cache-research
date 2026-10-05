# Addendum to the original Phase A completion packet

The restricted Phase A numerical result remains accepted, with one explicit terminal provenance exception. This additive note supersedes any implication in the original completion packet that every frozen file remained unchanged after evidence verification. It does not alter the original completion report or receipts.

The numerical v4 run completed at14:40:49 UTC under-B. At14:43:27, a later `python -` decoder import, without bytecode-write suppression, rewrote the copied derived file `v4/__pycache__/evidence_codec.cpython-312.pyc`.

Frozen expected SHA256: `2378944d3a76fa0f5b3eae296b5a0ce2a233302a6ae6e3c125778c0b54d1a55b`

Observed post-run SHA256: `02945da0a1b2c4cad3b0bddfbb3739e637b7eba0d82b5739d0c2283355b11af4`

Exact frozen-before bytes remain in the original v3 cache. All27 code objects are identical except embedded source filenames changing from v3 to v4; the header source timestamp changed as well. The authoritative Python source remains SHA256 `03ff37ede9adb969eadc11c59ac5ac9e3aa3b96912b8c8656de1feca0f596d69`. All171 other repository pins, both stdlib pins and numerical output hashes match. No substantive source or numerical output was changed.

The initial failed audit, both original cache paths, exact before/after snapshots and detailed comparisons are preserved. The independent final audit loaded only verified source bytes, with bytecode writes disabled, and accepted all1,621 events,1,949 tensors,9,705 chunks,8 groups,59 negative controls and resource/cleanup evidence. Its status is `PASS_RESTRICTED_PHASE_A_WITH_DISCLOSED_POSTRUN_CACHE_EXCEPTION`.

Use `TERMINAL_AUDIT_REPORT.md`, `INDEPENDENT_AUDIT_FINAL.json`, `CACHE_PROVENANCE.json`, `CACHE_WRITE_CONFIRMATION.json` and `AUDIT_COST_LEDGER.json` alongside the unchanged original completion packet. Do not state that all frozen files remained unchanged. No numerical rerun, model/GPU work or Phase B authorization follows from this exception.
