# Earlier resource screens: selected non-security conclusions

2026-10-05. This is a selective summary of the earlier CPU resource-management and scheduling screens. It contains no newly executed experiment.

## Capacity analysis

No eviction/admission topic was promoted. The residual question was efficient multi-capacity certification of a shared active-and-reusable KV pool, including active references, decode growth, delayed materialization and completion. A distinct algorithm and useful real-trace gap were not established. Adding active-byte peaks to an LRU estimate is insufficient because active and reusable blocks overlap and capacities can change which requests overlap.

Strong controls remain independent real-scheduler simulation at each capacity and an elementary active-block-union plus LRU shadow account. [KVSET's pinned analysis](https://github.com/llc-kc/kv_cache_capacity_estimator/blob/a90dc298151a8e117aa37e3c30fc0686de5225fa/src/kv_capacity_estimator/analysis.py) is a storage-reuse baseline, not an established active-HBM predictor. [SGLang's simulator](https://github.com/sgl-project/sglang/blob/59799a368793b9f795baf59b067233a65ad8e38e/tools/sglang-simulator/README.md) supplies an existing scheduler substrate, but its I/O and payload abstractions limit what a CPU run can establish. No simulator was installed or benchmarked in that screen.

## Trace and scheduling caveats

The [Qwen Bailian trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/tree/5f7439c51ec248a0c585f7d90a41a6f57773b912) exposes per-file relative clocks, parent/chat IDs, lengths and hashes. Its four trace types must not be merged into one invented global timeline. Context growth is not automatically append-only. One inspected [CacheWise sample](https://github.com/cachewise-project/cachewise-coding-traces/blob/181c435a090d328d00bbbee4c8eeb27d32f3abd2/parsed_traces/project_001/session_0001/events.json) repeats a logical call identity with differing annotations. That is a parser/schema caution, not a corpus duplicate-rate result.

Concurrent cold-prefix coalescing already has direct scheduler controls and pending-load-sharing proposals. Low-headroom live representation conversion still lacked a verified production transition or residual that beats ordinary page-wise conversion, bounded scratch, copy-on-write, or deferral. Neither became an admitted performance method. These summaries do not reopen prior frozen/closed experiments.

Source reports used for this selective summary: cpu_resource_bottleneck_screen_20261005/REPORT.md and the first two performance/resource sections of new_direction_screen_20261005/REPORT.md. Their broader branches are not part of this source checkpoint.
