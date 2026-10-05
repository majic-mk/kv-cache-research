# Restricted Phase A verification passed

The single separately authorized v4 run exited0. All eight protocol groups and59 deliberate negative controls passed. The exact same-run typed trace was exhaustively decoded:1,621 events,1,949 tensor descriptors and9,705 unique chunks; all lengths, shapes, dtypes, raw hashes, finite flags, logical order and the independently frozen full inventory verified. Guard/source/result/output bindings were checked. No trained-model or Phase B execution occurred.

Implementation: research/reconstructed/author_trimkv_phase_a_implementation_v4_20261005

Frozen manifest:173a928ab6c852a705a0a71738f98cec3c197f8a8a9831b1c05a2d16a6ba2150

Evidence remains in /tmp/trimkv_phase_a_author_parity_20261005_attempt4. It was not copied onto the workspace, whose disk margin above its5GB floor is smaller than the full5MB evidence packet. The small completion receipt records every artifact hash and location without moving or deleting originals.

## What the pass establishes

- Unchanged author TrimKVCache/RetentionGate class bodies agree with the independent adapter; scalar selection/gather checks validate separated cases
- Correct threshold/buffer32, original absolute positions, age2 convention, global post-call pruning, per-head paired gathers and current-token visibility/eviction
- Normalized-hidden gate input, fixed default RoPE/QK norms, two-layer closed-loop CPU integration and current HF ABI handling
- Full/native versus gate-loaded no-crop, FP32/BF16 cache/gate/output checks, tiny tiled-attention/GQA/mask and last-only-head controls
- All deliberately wrong age/clock/prune timing/boundary/gate/position/closed-loop/mask/head-routing and unsupported-route controls were detected

Exact same-primitive comparisons used canonical logical bytes, preserving signs of zero. Continuous controls used the original tolerances. Across136 FP32 controls max absolute error was3.814697e-6 and max RMS6.880806e-7; all elementwise atol+rtol conditions passed. All136 BF16 continuous controls were equal in this tiny run. This is a restricted CPU-port smoke gate, not a cross-backend or FA2 claim.

## Resource and cleanup evidence

- Guard wall6.787702s; childCPU6.646863s, guardCPU1.529997s
- Current run aggregateCPU8.176754s; conservatively charged cumulative41.769319s versus100s gate
- Tracked measured cumulative in guard28.928499s; exhaustive read-only final decode added0.905617s separately
- Peak sampled guard+tree RSS384,397,312B; child OS maxRSS369,750,016B
- Maximum observed sample gap27.548ms against25ms target
- Minimum observed host available5,626,736,640B; output disk free5,062,578,176B; workspace free5,003,780,096B
- Trace4,314,510B; all guarded new logical evidence including claim5,017,984B, below32MB
- Guard cleanup proved owned process identities absent; fresh post-run identity scan agreed. Scratch was checked empty by guard and independently reread empty

RSS remains sampled with OS high-water supplements, not a hard kernel tree-memory reservation. Complete resource samples and all prior failures remain preserved.

## Preserved earlier outcomes

Attempt1 failed native import because the original write whitelist blocked tempfile. Attempt2 stopped on repeated JSON evidence volume. Attempt3 stopped on a singleton-stride dtype-view test-helper error. Each is preserved as a failed infrastructure attempt. Narrow versioned fixes were independently reviewed and separately authorized, with codec/helper checks, without changing policy, fixture arrays, case IDs or tolerances.

This result establishes a faithful small baseline foundation only. It does not establish trained4B accuracy, real long-history memory fit, full author-program execution, FA2 parity, throughput, novelty or a paper result. Phase B requires its own prepared runner, resource admission, review and execution approval.
