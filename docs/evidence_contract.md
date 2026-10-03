# Evidence contract, schema v1

## Manifests

`RunManifest.create` requires a repository root, evidence kind, artifact pins,
configuration and seed. `model` and `dataset` are required artifact roles; add
separate tokenizer, engine, author implementation and evaluator roles where they
are independently versioned. Each pin has `identifier`, `revision_kind`, and
`revision`. Allowed kinds are full Git commit (40/64 lowercase hex) and content
SHA-256 (64 lowercase hex). Existence and ownership of a claimed revision must be
verified by the experiment adapter/operator; syntax is not that verification.

`manifest.json` contains `manifest` and `manifest_sha256`. The SHA-256 covers UTF-8
canonical JSON with sorted string keys, compact separators and no nonfinite
numbers. Duplicate JSON keys are rejected. The frozen Python object returns
detached payload copies. A file save is write-once with atomic hard-link
publication. Existing paths are never replaced, including under racing writers.

Source capture records the current commit, hashes for tracked/staged/unstaged
diffs and status, untracked file/symlink hashes, and a source snapshot hash. It
never collects remote URLs or arbitrary environment variables. Ignored files are
not captured, so model/data/input artifacts must be pinned independently.
Submodules and assume-unchanged/skip-worktree index entries are rejected in v1.
Do not mutate source during snapshotting or execution. Dirty fixture manifests
do not contain enough bytes to restore edits; preserve the checkout separately.

Manifest creation is not execution. `real_model` requires a clean committed
checkout and rejects explicitly fixture/simulation-prefixed artifacts. A future
adapter must collect actual runtime logs, actual pinned inputs, hardware, drivers,
numerical precision, attention backend, seed behavior and deterministic settings.
No adapter for a real model is included yet.

## JSONL records

Every line is one observation. Empty lines are errors. Required top-level keys:

- `schema_version`, `run_id`, `manifest_sha256`, `evidence_kind`
- `phase`: `warmup` or `measurement`
- `method`, `unit_id`, `cluster_id`, nonnegative integer `replicate` and `order_index`
- `workload`, `workload_sha256`: a nonempty JSON object and its checksum
- `status`: `ok` or `error`; nullable `error` and `output_sha256`
- `timing`, `shared_setup`, `metrics`, `resources`, `execution`

One result file is one run, one manifest and one evidence kind. The observation
key is phase/method/unit/replicate. A successful row has a checksum of its actual
output artifact and no error. Failed rows have an error and no successful-output
checksum. Selected failures stop paired analysis; they are never silently dropped.
Custom metrics are finite numbers and must have documented units and direction.

### Pairing, contexts and bootstrap

`unit_id` identifies a matched input/question/trace observation. `replicate` is a
repeat or paired seed. `cluster_id` identifies the independent context/trace.
All questions and repeats sharing one source document or cache belong to the same
cluster. An input cannot change clusters across methods or repeats. Use distinct
unit IDs for different questions within that context. The harness cannot infer
hidden dependence or train/test contamination from these labels.

Within each unit, average matched replicates. Within each cluster, average units.
Weight independent clusters equally, then resample whole clusters with replacement
using a local seeded Python RNG. This estimates a context-weighted mean difference,
not a request-weighted serving average. The distinction is reported in summaries.
One cluster receives no confidence interval; fewer than ten carry a small-sample
warning. Intervals are unadjusted descriptive percentile intervals, not a causal
or publication-significance guarantee. Use separate multiplicity/power analysis
when designing an actual study.

Baseline and candidate need exactly the same measured unit/replicate keys and
workload checksums. Encode actual context/input hashes, target generation budget,
load trace, arrival process, cache regime and evaluation conditions in `workload`.
Keep method-specific actions outside it. Backend/device/synchronization and timing
accounting must match. Execution-order indices differ within a pair; randomize or
counterbalance them and record the realized order. Warmups are excluded and counted.

Each summary records a hash of all canonical input records sorted by observation
identity, including warmups and unselected methods. The CPU completion marker also
hashes the actual JSONL bytes. A manifest hash alone cannot identify measurements.

### Fair time and shared work

All timing quantities are seconds and must be finite and nonnegative. Every phase
must be present, with explicit zero when unused:

`queue_wait_s`, `preprocessing_s`, `scoring_s`, `host_copy_s`,
`host_to_device_copy_s`, `device_to_host_copy_s`, `cache_read_s`, `cache_write_s`,
`recompute_s`, `prefill_s`, `decode_s`, `postprocessing_s`, `synchronization_s`,
`other_s`, plus `end_to_end_s` and `accounting_mode`.

- `serial_exclusive`: phases are nonoverlapping parts of one end-to-end wall
  interval. Their sum cannot exceed it; untagged overhead remains included
- `overlapping`: phase intervals may overlap. Do not add them to infer wall time.
  Each phase is bounded by the enclosing wall interval
- End-to-end begins at the relevant request arrival/work boundary and ends after
  the actual output and required synchronization. It includes host work, queueing
  and copies. A future serving adapter must define these boundaries explicitly
- `SerialTimer` measures CPU wall time only. It does not automatically synchronize
  CUDA. GPU rows must declare synchronized boundaries; the adapter must implement
  and audit that claim

`shared_setup` describes one separately measured setup job with `group_id`,
`amortization_count`, `preprocessing_s`, `scoring_s`, `copies_s`, and `other_s`.
Its categories are exclusive elapsed costs, not overlapping kernel sums. Every
row consuming the same setup must repeat the same total setup costs. The scope of
a setup group is phase/method/group ID. Its denominator must equal the actual
number of rows in that scope; imaginary future requests cannot lower measured cost.
Use a new group ID for a repeated setup job. `none` means no shared cost, all zero.
Do not include that same work in request phase timing as well.

Default analysis metric:

`total_cost_s = end_to_end_s + sum(shared setup costs) / observed amortization_count`

This is attributed cost per observation. It is not the same as online latency,
batch makespan, or tail-latency/throughput estimation. A serving experiment needs
those separately measured endpoints and its own trace-level estimators. If
`end_to_end_s` is selected with nonzero shared work, the summary warns that it
excludes setup. Report both cold/one-shot and realized-reuse costs when relevant.

### Logical budgets are not physical memory

`resources` has the following nullable, nonnegative integer quantities:

- `logical_kept_tokens`: logical retained-token budget/count
- `logical_kv_bytes`: nominal bytes implied by the logical budget, with dtype and
  layer/head assumptions documented by the adapter
- `physical_device_kv_allocated_bytes`: actual live KV tensor/buffer storage on the
  measured device; avoid double-counting shared storage/views
- `physical_kv_stored_bytes`: actual serialized/offloaded KV artifact size; record
  the storage tier and compression/container overhead in the measurement method
- `peak_device_allocated_bytes`: allocator peak for the full measured execution
- `peak_device_reserved_bytes`: allocator reserved-memory peak for that execution

Also required: `representation` (`unknown`, `full_dense`, `masked_dense`, or
`physically_compacted`) and a nonempty `measurement_method`. Non-null quantities
cannot use `not_measured`. Null means unknown/unmeasured, never zero.

An attention mask or logical kept-token count is insufficient evidence of memory
savings. Masked dense tensors may retain the full allocation. Only measured
physical storage/allocator data can support physical-memory claims. Peaks include
temporary scoring/preprocessing allocations and should be reset around documented
boundaries. This schema does not implement allocator instrumentation for you.

## Costs and staged execution

The existing CSV ledger has timestamp, provider, resource, decimal amount,
currency, status and evidence. `ledger` audits it without changing bytes. Amounts
are grouped by currency and status, and are never silently converted or interpreted
as approval. Statuses: no_external_purchase, estimate, committed, settled, cancelled.
Do not record the same purchase as both committed and settled and add them as two
purchases; the v1 audit deliberately does not infer invoice identity.

The current CPU pilot is zero external spend. The GPU template has zero authorized
hours/spend, null source pins, and a list of gates. Its proposed sample sizes are
planning placeholders, not a power calculation or permission to rent hardware.
