# CPU engineering and diagnostics through attempt06

Frozen private source checkpoint, 2026-10-04. This adds a compact archive and provenance without modifying earlier project files or rewriting Git history. It is an archival source delivery, not a full expanded worktree/history sync or CI pass.

## Current scientific status

- Earlier negative results remain preserved: first24 found no qualifying harm; MuSiQue Full6 stopped at baseline competence (1/6).
- File-backed paging and process-local trim engineering successfully executed the fixed CPU workload within the later reviewed resource bounds, with recorded native-prefix/hash checks. This is bounded engineering evidence, not a cache-compression quality or GPU benefit.
- Original Full attempt04 failed the unchanged 64-token/no-EOS completion gate.
- The separate author-budget calibration (attempt05) reached natural EOS, but its complete answer still failed independent semantic review: it assigned the 5–2 record to June16 instead of June30 and made an unsupported abstention. Its engineering pass does not reverse that failure.
- Attempt06 succeeded on one separately declared query-only diagnostic: the complete answer reported 5–2 and June30, 2023, 08:14. It used 131 model calls, 50 selections and 49 emitted/fed tokens, ending naturally at EOS. The complete query was reformulated and added a date request, so this shows sensitivity to that full query change on one history, not an isolated temporal-scope mechanism.

The original pilot remains failed and closed. No no-crop, compressed/eviction or GPU comparison was run, and no novel-method claim is established. TRIM-KV Appendix B.3 already studies sequential chunked-prefill eviction; an offline-only prior-work characterization is not supported.

## Files and restoration

- `KV_CPU_Engineering_Through_Attempt06_Source_20261004.zip`: 346 unchanged source, configuration, test, documentation and provenance files, plus bounded terminal/semantic summaries, selected from cumulative CPU Library v9 and the separate frozen attempt06 packet
- `SOURCE_SELECTION_MANIFEST.json`: every selected byte hash and every exclusion; the two historical roots stay separate
- `CHECKPOINT_MANIFEST.json`: compact ZIP identity, source-package identities and scope
- `RESTORE.md` and `OFFICIAL_ASSET_PINS.json`: verification, exact official asset hashes, and pinned private Library identities for missing raw evidence

Read [RESTORE.md](RESTORE.md) before combining snapshots. Weights, runtime environments, credentials, platform-operation records and large raw tensors/logits are excluded. Earlier checkpoint source remains under `../2026-10-04-source/`. Historical pending/pre-run strings are preserved as history; the attempt06 terminal status and this README describe the latest frozen outcome. Restoring files does not authorize another run.

Packaging performed ZIP/per-file hash verification only. No project tests, models, GPU jobs, paid API calls or rentals were executed for this upload.
