# Synthetic shared-prefix scheduling diagnostic

**This is an exact arithmetic scheduling toy, not an LLM, GPU, latency, or quality experiment.** No model, dataset, CUDA, external package, network request, or paid compute is used. Durations are arbitrary units chosen for a counterexample, not hardware measurements.

## Reproduce

From the repository root, using Node.js and its standard library:

```sh
node research/diagnostics/reproduce_shared_prefix.cjs > /tmp/shared_prefix_enumeration.json
cmp research/diagnostics/shared_prefix_enumeration.json /tmp/shared_prefix_enumeration.json
```

The executable runs assertions before producing JSON. A successful exit and `tests.status = "PASS"` mean the stated schedule invariants were checked. The committed JSON includes all 52 valid resource-order/mode schedules, not only a summary.

`shared_prefix_original.js` preserves the original exact-arithmetic computation. The executable provides its two host functions, `store` and `text`, runs it in a Node VM, and independently validates its results. The only additions to that original source are provenance comments.

## Model and assumptions

- One serial PCIe resource P and one serial GPU resource G can overlap
- Every request arrives at time 0
- Each prefix is restored exactly once: loading occupies P, recomputation occupies G
- Once ready, a prefix is shared by its requests without additional copy cost
- Every request then needs a non-preemptive unit-duration GPU suffix
- Fixed known durations, adequate memory, no batching, preemption, cancellation, stochastic timing, or cache eviction
- Completion equal to a deadline is successful
- Objective: maximize requests completing by their deadline; use makespan only as a tie-break

| Prefix | Load on P | Recompute on G | Requests | Suffix/request | Deadline/request |
|---|---:|---:|---:|---:|---:|
| A | 4 | 10 | 2 | 1 | 6 |
| B | 2 | 3 | 1 | 1 | 4 |

## Enumeration completeness within this toy

For each of four mode assignments, enumerate every ordering of tasks on P and G. Add precedence edges from each prefix to its request suffixes and between consecutive tasks on each resource. Discard cyclic graphs. Compute earliest start times for every remaining directed acyclic graph.

Every legal non-preemptive schedule induces one of these resource orders. For a fixed resource order, moving tasks to their earliest precedence-respecting start cannot make any completion later. Consequently an earliest-start representative suffices to maximize on-time completions here. This argument depends on the assumptions above and does not justify an oracle for a real serving system.

The 52 valid representatives split into 12 load/load, 12 A-load/B-recompute, 8 A-recompute/B-load, and 20 recompute/recompute. These are distinct mode/resource-order descriptions, not a count of all possible continuous start-time schedules.

## Results and interpretation

- Singleflight plus EDF loads B during 0–2 and A during 2–6. B completes at 3, A's requests at 7 and 8: 1/3 on time
- The best all-load ordering restores A first and completes A's requests at 5 and 6, then B at 7: 2/3
- The oracle loads A on P during 0–4, recomputes B on G during 0–3, and runs suffixes B during 3–4 and A during 4–6: 3/3
- Two schedules achieve 3/3, differing only in which A suffix runs first
- A strong simple baseline reaches the same oracle: A cannot satisfy its deadline by recomputation, so reserve its load; B is feasible by recomputation, which can overlap A's transfer

For comparison only, without shared-prefix singleflight and with request order B, A1, A2, independent earliest-finish mode selection loads each prefix copy: P intervals B=0–2, A1=2–6, A2=6–10; G suffix completions are 3, 7, 11. This policy also meets only 1/3 deadlines. Duplicate-copy schedules are not part of the shared-state enumeration and this comparison is not used in the oracle claim.

**Negative novelty result:** resource coupling defeats per-request/EDF choices in this toy, but a compulsory-load reservation rule eliminates the gap. This does not warrant a new scheduler claim. A related gap exists without sharing, so the toy also fails to show that prefix sharing itself is the novel difficulty.

## Headroom kill criterion

Before a new scheduler is implemented, compare against singleflight, deadline feasibility checks, compulsory-resource reservation, and short-horizon joint mode/order lookahead. If a strong simple baseline is close to the oracle on the intended workload family, reject the proposed algorithmic direction. A counterexample against FIFO/EDF alone, or retuning a predictor margin, is insufficient. Any eventual GPU work must independently establish realistic durations, memory constraints, overlap limits, and end-to-end benefit.

## Relevant prior-art sources checked separately

These sources motivate the skeptical baselines; none supplied or measured the toy durations.

- [Strata](https://arxiv.org/html/2508.18572v1): shared-miss deferral and compute/I/O-balanced scheduling
- [Multi-tier dynamic storage](https://doi.org/10.1007/s40747-025-02200-4): load/partial-load/recompute selection
- [Mooncake TENT](https://kvcache-ai.github.io/Mooncake/design/tent/deadline-scheduling.html): deadline ordering and predicted infeasible-transfer fallback

Source-check cutoff: October 3, 2026. This directory is separate from the model-evaluation harness.
