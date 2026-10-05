# Actual Stage B result: exact proposal fails its usefulness gates

One real Stage B run was separately root-authorized at 2026-10-05T11:04:18Z after clean Stage A capture and weight provenance. It completed with launcher exit 0 and a valid, complete population. The exploratory usefulness gate failed. This report supersedes source-preparation-only status descriptions without changing any frozen source or prior receipt.

## Primary registered comparison

Metric: unweighted mean of six layerwise holdout SSE / same-layer nearest SSE. Lower is better. Each layer contains 16 adjacent queries from one passage; these are correlated local-operator observations.

| Arm | Mean layerwise nearest-normalized holdout SSE |
|---|---:|
| nearest | 1.000000000 |
| headwise_cd | 0.828324749 |
| group_joint_cd | 0.821914459 |
| full_joint_cd | 0.808808818 |
| guarded_token_gptq | 0.966594180 |
| projected_block_gptq | 0.804669366 |

Full-joint reduced this metric 19.12% relative to nearest, 2.36% relative to headwise and 1.59% relative to group-joint. It was 0.51% worse than projected-block. Thus it did not deliver the required 10% advantage over every strong required control.

- Full-joint beat headwise only in layers 0 and 35: 2/6, below the required 4/6
- The query-level 1.25×nearest quality cap failed once: layer 7, position 520, SSE 0.02114292260835871 versus nearest 0.013716359108188997, ratio 1.5414383978716244
- All same-cap encoder row constraints passed; this query failure is a held-out operator-quality failure, not a V-row-cap violation or resource failure
- All 36 layer/arm encodings completed and were locked before the first holdout payload read; no arm or layer was dropped
- This closes this exact proposal at its frozen stopping gates. No rerun, retuning, new model inference, GPU run or budget extension followed

## Work coverage

The common upper cap was 1,300,000,000 source-defined MAC-equivalent units per layer/encoder. This does not assert equal CPU time or equal total processor work. Each arm retained the required first complete sweep; additional complete channel-block/group units used the remaining budget.

| Arm | Charged units per layer | Full sweeps | Complete channel-block/group units |
|---|---:|---:|---:|
| headwise_cd | 1,299,791,872 | 2 | 534 |
| group_joint_cd | 1,298,661,376 | 1 | 460 |
| full_joint_cd | 1,298,661,376 | 1 | 460 |
| guarded_token_gptq | 440,598,528 | 1 | token trajectory |
| projected_block_gptq | 1,296,711,328 | 1 | 263 |

First-sweep code checkpoints are saved, but no first-sweep quality comparison or checkpoint selection was made. Ordinary raw-token GPTQ was also evaluated as a separate diagnostic trajectory and remained ineligible for same-cap superiority. Its row-cap violation counts by layers 0, 7, 14, 21, 28, 35 were 677, 937, 966, 950, 928, 976. Projected raw results are proposals along the accepted trajectory, not a separate raw optimizer.

## Resource and I/O outcome

- Launcher exit 0; guard termination null; owned group fully reaped and absent
- Aggregate CPU including source/terminal finalization: 37.932891 seconds, cap 240
- Guard wall 35.347983 seconds, cap 300; final launcher checks also passed
- Sampled tree RSS 208,621,568 bytes; child high-water 195,977,216 bytes, cap 536,870,912
- Minimum observed host available 6,159,855,616 bytes, floor 2,000,000,000
- Final run directory 185,308,476 bytes including FINAL_MANIFEST.json, cap 268,435,456
- Exactly 251,658,240 selected W payload bytes consumed: six 20 MiB segments once encoding and once evaluation, plus separately counted headers
- Child pre-result file-read counter 651,233,361 bytes includes source/tensor/output-hash/code reads. Parent cumulative read counter is 185,847,945 bytes, including 595,972 prelaunch bytes despite its post_cleanup_hash_read_bytes label. Total instrumented run reads are 837,081,306 bytes; prelaunch bytes must not be added again
- Full-shard provenance was a separate verified 8,044,982,000-byte cryptographic pass, not hidden within selected-segment I/O
- Physical shared calibration acquisition was counted once, standalone acquisition was charged to each bounded encoder, and evaluation/recomputed calibration work was separately recorded

Cgroup ceiling visibility remained unknown under fresh packet-specific root acceptance. Host availability and sampled RSS do not prove cgroup capacity. CPU/logical work is not a serving-latency or GPU-speed result.

## Scope and retained evidence

This is FP64 local operator error on exact captured BF16 Q/K/V and signed W_o, with fixed teacher-forced reference trajectory and full causal prefix/suffix denominators. It does not measure generated answers, autoregressive error propagation, task quality or generalization beyond this passage.

Per-layer/per-query calibration and holdout metrics include head/group/full SSE, signed cross terms and full signed all-head error vectors. Original output vectors, all codes/grids/paths/first checkpoints, raw and guard decisions, work ledgers, byte records and cleanup receipts are retained.

The 716,800B V-prefix /1,765,376B KV-prefix figures are hypothetical packed-format bytes versus1,048,576B /2,097,152B original BF16. The evidence uses uint8 code arrays; this is not a native serving-memory measurement.

Primary artifacts: run/result.json, run/ENCODING_LOCK.json, run/guard_result.json, run/FINAL_MANIFEST.json. Source manifest remains d9ad52fb6ec8aa585939fe9f9fc8c6d921a3588f5b0429b290e992a9fb0d740b. Independent terminal audit confirmed valid execution and failed usefulness gates: all 36 records and artifact hashes agree, no prelock holdout reads occurred, and signed decomposition discrepancy was only 1.39e-17. No source change or rerun was requested.
