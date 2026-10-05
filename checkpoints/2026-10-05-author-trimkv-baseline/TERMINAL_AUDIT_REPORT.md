# Independent terminal audit: restricted Phase A parity passes

2026-10-05 UTC. **ACCEPTED: restricted author-cache/gate semantics and tiny current-HF CPU integration, with the explicit post-run derived-bytecode exception below.** This is an actual completed synthetic parity result, not a trained4B result or Phase B authorization.

Run: `/tmp/trimkv_phase_a_author_parity_20261005_attempt4`

Admitted v4 manifest SHA256: `173a928ab6c852a705a0a71738f98cec3c197f8a8a9831b1c05a2d16a6ba2150`

Independent final audit: `INDEPENDENT_AUDIT_FINAL.json`, SHA256 `e6c36ad7a2dd2d4aa3fb8b59e6672d2816d1c8271bcb9ddc6b4a95129e1d5491`

## Terminal result

- All8 original semantic groups passed
- All59 declared negative controls were detected; their trace names/verdicts exactly match result.json
- Complete1,621-event inventory verified against the frozen independent event contract
- All1,949 typed arrays and9,705 shared chunks verified through frame, row, reconstructed-array and typed-content hashes, shape/dtype, finite flags and final stream commitment
- All272 continuous-error records match result.json and retain the original prospective tolerances
- Same-run source/protocol/runtime/manifest/output/claim binding verified
- Guard accepted resources; owned process identities were absent; scratch was sealed and terminal/live scratch was empty
- Previous failed runs and separate codec/helper checks remain preserved

The parent observed launcher exit0. The immutable guard receipt independently records child exit0, no stop reason and accepted completion. The independent bounded stdlib audit exited0. No Torch, Transformers, NumPy, model or numerical fixture was imported or rerun by this terminal audit.

## Exact output identities

| Artifact | SHA256 |
|---|---|
| result.json |`e6818865f68cd616928b90d038a13e5640f214283b6671346b65964caf8a0520`|
| guard_receipt.json |`8d41ca213f57a18454e93b01b920792923f4101cd5a7f873d0e14c6a37828571`|
| trace.tkv3 |`5434e0f4d5240db16a53f301ba5570c9ad05eb05b1f27cf17e8c54c457a20acd`|

The trace is4,314,510 bytes, comfortably below the23MB trace partition and the independent21,350,347-byte complete-suite upper bound. All original run files were rehashed at audit completion. The detailed per-file inventory is in the independent receipt. Final logical run output including the persistent claim is5,017,984 bytes; this includes the terminal guard receipt, unlike its earlier pre-receipt total.

The original implementation completion packet remains at `research/reconstructed/author_trimkv_phase_a_completion_20261005`. Its `VERIFIED_COMPLETION.json` SHA256 is `a4f4ebb4a9cf7d899c5af9f1f4fce3bf9df9a9de8dba0cc121381d58e7d7da90`. It is preserved as history. Any implication that every frozen file remained unchanged after post-run verification is superseded by the specific exception in this report and `POST_RUN_BYTECODE_ADDENDUM.md`.

## What the pass establishes

The executed reference used the unchanged complete author `TrimKVCache` and `RetentionGate` source spans. The independent adapter, scalar score/gather oracle and actual current native Qwen3 decoder integration passed their fixed FP32/BF16 cases. The original positions/shared clock, literal N+1-p age, buffer32 cadence, per-head membership/order, paired gathers and post-global-call prune placement are covered by the preserved suite.

The tiny two-layer integration includes lower-layer compressed outputs feeding upper-layer gates, normalized-hidden gate inputs, Q/K normalization and fixed RoPE, native two-value cache ABI, explicit retained-plus-current causal visibility and no-crop/native controls. Deliberately wrong clock/age/buffer/prune timing, raw or wrong-dimensional gate inputs, history partition/freeze, position rebasing, stale upper-layer inputs, top-left masks, GQA mapping and unsupported routes all had detecting controls.

The source/dtype/shape and exact same-primitive comparisons remain strict bitwise checks. The v4 canonical-storage helper fixes a singleton-stride implementation error without changing the numerical policy or tolerances; its separate24-case/18-reconstruction check remains pinned. No failed comparison was converted into a tolerant one.

The separately tolerant CPU attention/head comparisons have136 records per dtype:

- FP32 maximum recorded absolute error:3.814697265625e-6; maximum recorded RMS error:6.88080604049901e-7
- BF16 maximum recorded absolute and RMS errors:0 in these136 recorded comparisons

The FP32 number is accepted under the unchanged combined elementwise atol2e-6/rtol2e-5 criterion, not a standalone maximum-absolute threshold. BF16 retained atol0.03125/rtol0.02. These observations do not establish all-shape, all-input or cross-backend bitwise parity. Native topk tie behavior remains pinned-backend-only. CPU eager/query-tiled parity is not FlashAttention2 parity.

## Resource and cleanup audit

All261 resource samples were checked against the original envelope. Recorded results:

- Guard wall:6.787701620s
- Own guard+child CPU:8.176754s, as recorded in the guard's aggregate field
- Conservative cumulative CPU charged by the guard:41.769319s, including its prior-run/check reserves
- Peak sampled guard+child RSS:384,397,312 bytes
- Sum of separate guard/child OS RSS high-water marks:387,313,664 bytes, a conservative upper bound for those two process peaks rather than a simultaneous measurement
- Largest sample interval:0.027548131s, versus the25ms target
- Minimum observed host available memory:5,626,736,640 bytes; every sample also retained the continuing1.5GB floor after its outstanding tmpfs reserve
- Minimum observed output/source free space exceeded the5GB floor

The120s wall,90s child hard CPU,5s guard hard CPU,100s cumulative CPU,1GiB RSS, one-thread,32MB output and no-GPU/network/install/payload limits were unchanged. Both child stdout/stderr logs are empty. There were no sampled unexpected descendants. The guard's fresh PID/start-time cleanup scan found no owned process remaining; the independent audit also observed the shared scratch directory empty. The child scratch receipt correctly precedes atexit; terminal cleanup evidence, not those earlier pre-cleanup flags, determines acceptance.

The running guard's sample/high-water and cooperative output accounting limitations remain. Operation timers and logical-byte figures cover instrumented regions and overlap; they are not an end-to-end performance benchmark or measured DRAM traffic.

## Explicit post-run derived-bytecode exception

The first independent audit stopped before decoder import because one frozen repository entry had drifted:

`author_trimkv_phase_a_implementation_v4_20261005/__pycache__/evidence_codec.cpython-312.pyc`

- Frozen-before SHA256:`2378944d3a76fa0f5b3eae296b5a0ce2a233302a6ae6e3c125778c0b54d1a55b`
- Observed-after SHA256:`02945da0a1b2c4cad3b0bddfbb3739e637b7eba0d82b5739d0c2283355b11af4`
- Rewrite time:14:43:27.268977 UTC, after the main guard's14:40:49 completion

**171/172 repository manifest entries match exactly at terminal review; the one derived cache is an explicit exception. Both external stdlib pins match. Do not claim all frozen files remained unchanged.** Authoritative codec source SHA256 remains `03ff37ede9adb969eadc11c59ac5ac9e3aa3b96912b8c8656de1feca0f596d69`, and all substantive numerical/source/output pins match.

V4 preparation had copied a v3 cache whose bytes became a manifest entry. Exact frozen-before bytes still exist at the original v3 cache path. That cache's timestamp header refers to the earlier v3 source mtime, so it was timestamp-invalid for the later v4 source. The main run used-B; Python could compile the pinned source but could not rewrite that cache. A subsequent ordinary `python -` decoder verification imported without-B or other bytecode-write suppression and rewrote the derived cache after its initial pin checks. This execution route and chronology were confirmed.

Static marshal inspection, without executing either cache, compared all27 code objects. Every code/constant/name/argument/flag/stack/line-table/exception-table attribute matches except the v3→v4 source filename; the cache header's source timestamp also changed. There is no changed numerical or decoder operation. The exact before/current bytes, timestamps, headers and all27 comparison records are retained in `CACHE_PROVENANCE.json`, plus `CACHE_BEFORE_MATCHING_FROZEN.pyc` and `CACHE_AFTER_POSTRUN.pyc`. Neither original cache was restored, overwritten or deleted.

The parent explicitly accepted adjudication as a derived-bytecode exception. The final independent decoder was compiled directly from the verified source bytes under-I -S -B, with model imports/network/subprocesses and original-file writes denied. It does not rely on either cache. This supports the restricted actual parity verdict while retaining the provenance deviation visibly.

## Independent audit history and separate CPU

All audit receipts are preserved:

1. `INDEPENDENT_AUDIT.json`: stopped on the real derived-cache hash drift before decoder import;0.037301s process CPU
2. `INDEPENDENT_AUDIT_RESOLVED.json`: completed source-bound decoding, then stopped because the auditor referenced an obsolete resource-field name;1.029249s CPU. This was an audit-script schema mistake, not a model/guard failure
3. `INDEPENDENT_AUDIT_FINAL.json`: corrected that field name and completed all decoder/source/resource/provenance checks;1.226972s CPU,1.165986s wall,27,156,480-byte OS maximum RSS

Independent reported audit-process CPU totals2.293522s, separate from the numerical run. The implementation's earlier exhaustive verification separately reported0.905617s. `AUDIT_COST_LEDGER.json` records these values and hashes. Receipt-time CPU counters are not an end-to-end benchmark; no numerical tests were rerun by these read-only corrections.

## Remaining Phase B gates

Phase A's restricted source/actual-semantics and tiny CPU attention/head gates are now satisfied. Tokenizer-only preparation separately froze the two original Oracle records, history/question lengths6476/50 and3959/41, M960 and buffer32. Both history and complete question calls trigger global compression; no online history repartitioning is justified.

Phase B still requires its own reviewed implementation, exact trained base/gate payload identity checks, real-shape CPU/pager/attention/head integration and input-specific physical memory/CPU/logical-read/output admission before loading weights. It needs a fresh guard and live preflight, and separate actual-model authorization. The original worst-case800-selection×4-arm schedule exceeds the4TB logical-read allowance; resource censoring must remain explicit rather than shortening or replacing selected records.

A prospective source-preparation amendment replacing full vocabulary-vector evidence with hashes/top2 was separately accepted for preparation. It is a material, non-lossless Phase B evidence change and must remain disclosed in its eventual reviewed execution contract; this Phase A pass does not approve or validate it. The current Phase B source work stays execution-locked pending its own gates.

No trained4B quality, full author-loader execution, long-prefill fit, GPU/FA2 result, throughput, general accuracy, novelty or publication-level result follows. No model/GPU execution or Phase B launch is authorized by this report.
