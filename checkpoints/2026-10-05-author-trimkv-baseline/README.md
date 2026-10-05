# Restricted TRIM-KV Phase A baseline, 2026-10-05

**Final result: PASS for restricted author-cache semantics and tiny current-HF CPU integration, with a disclosed post-run derived-bytecode exception.** This is a small baseline engineering foundation. It is not a trained-model quality result, a new compression method, or a publication result.

The separately authorized v4 run passed all **8 protocol groups and 59 negative controls**. Independent terminal inspection verified the exact same-run trace: **1,621 events, 1,949 tensor descriptors and 9,705 unique chunks**, including shapes, dtypes, logical order, raw hashes, finite flags and the complete frozen inventory. No numerical experiment was repeated for this checkpoint.

## Result and audit

- [Final independent audit](INDEPENDENT_AUDIT_FINAL.json) is the terminal decision
- [Original completion report](PHASE_A_COMPLETION_REPORT.md) and [completion receipt](VERIFIED_COMPLETION.json) are preserved verbatim as historical records
- [Same-run result](PHASE_A_RESULT.json) and [guard receipt](PHASE_A_GUARD_RECEIPT.json) bind the execution evidence
- [Derived-cache provenance](CACHE_PROVENANCE.json) and the [terminal report](TERMINAL_AUDIT_REPORT.md) disclose the exception and supersede any earlier unqualified claim that every frozen file remained unchanged

The terminal audit checked 172 frozen repository entries: 171 still matched byte-for-byte. One generated evidence_codec.pyc changed after execution during post-run verification. Original frozen bytes were recovered by exact identity from the preserved v3 cache. The source hash was unchanged; recursive code-object comparison found unchanged executable contents, with source-path and header timestamp metadata differences. Both cache identities, the initial failed audit and subsequent resolution are retained in the full archive. This is **not an all-files-unchanged result**; no old evidence was repaired or overwritten to hide the discrepancy.

Across 136 FP32 continuous controls, maximum absolute error was 3.814697e-6 and maximum reported RMS error was 6.880806e-7; all original elementwise tolerances passed. All 136 BF16 continuous controls were equal in this tiny run. Same-primitive comparisons used canonical logical bytes, preserving signed zero.

Guard wall time was 6.787702 s; current run aggregate CPU was 8.176754 s. The guard conservatively charged cumulative CPU of 41.769319 s against the 100 s gate. Peak sampled guard-plus-child RSS was 384,397,312 bytes; a sampled-memory observation is not a hard kernel tree reservation. The largest sample gap was 27.548 ms against the 25 ms target. Terminal process-identity and scratch checks passed. Read-only terminal audit and archival costs are separate from the original run accounting.

## Preserved failures

1. v1 failed native import because the original write whitelist blocked tempfile
2. v2 stopped on repeated JSON evidence volume; its incomplete inner gzip is retained as written
3. v3 stopped at the singleton-stride dtype-view test helper; its valid 406-event prefix remains partial evidence
4. v4 passed after the independently reviewed narrow layout-helper fix; cases, fixtures, policy and tolerances were not retuned

The original failures, logs, consumed claims and reviews remain separate and recoverable. Missing trailing bytes from a stopped attempt were never invented.

## Source and restoration

The [source ZIP](TRIMKV_PhaseA_Baseline_Source_2026-10-05.zip) contains unchanged authored source/protocol/test files, chronological reviews and required small attributed author reference snapshots. [Source selection hashes](SOURCE_SELECTION_MANIFEST.json) identify every ZIP member. Run `python verify_checkpoint.py` for a standard-library-only integrity check; it does not execute archived research.

The complete raw/partial traces, arrays, resource logs and cache-exception artifacts are preserved in the [full evidence archive](TRIM-KV_restricted_PhaseA_engineering_archive_20261005.tar.gz) beside this source package. The same 14,454,096-byte archive was saved as private Library item libfile_3ebc5db3bed081919366ddb8934000f2, version 0. Library download verification is blocked by repeated HTTP502 responses, so Library remote bytes are not claimed verified. This private GitHub copy is the fallback delivery. [Exact identities and SHA256](EVIDENCE_ARCHIVE_REFERENCE.json), the [355-file restoration map](RESTORATION_MAP.json), the [original archive manifest](FULL_ARCHIVE_MANIFEST.json), and [external dependency pins](EXTERNAL_DEPENDENCIES.json) provide restoration coordinates. See [restoration and scope](RESTORE_AND_SCOPE.md). No sharing permissions changed.

Earlier preparation/status strings remain historical. Later Phase B source preparation is excluded from this checkpoint. Phase B needs a separately prepared runner, resource admission, review and execution authorization.

## Limits

This pass concerns unchanged author class bodies plus a restricted synthetic CPU adapter/reference route. It does not establish trained 4B accuracy, real long-history memory fit, full author-program execution, CUDA/FA2 parity, cross-backend tie identity, throughput, a novel method or publication readiness. No weights or installed runtime are bundled. Source/archive integrity is not a CI pass or an unchanged portable rerun. All 135 prior remote repository files, including root README, are preserved byte-for-byte.

## Separate later source-only screen

The [headwise no-copy source screen](POST_ARCHIVE_HEADWISE_SOURCE_SCREEN.md) and [source ledger](POST_ARCHIVE_HEADWISE_SOURCES.tsv) were finalized after the full-archive freeze. They are included separately in this GitHub checkpoint and are not members of that archive or the 158-member source ZIP. This is a source-only no-go finding, with no experiment, promising-method claim or execution authorization.

The [archival placement exception](ARCHIVAL_PLACEMENT_EXCEPTION.json) allows a bounded real-file workspace copy solely for private delivery. It changes no completed experiment gate and grants no future execution-limit change; copies are retained.
