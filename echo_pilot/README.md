# EchoPress baseline diagnostic pilot

**State: CPU contract/mock tests only. GPU adapter, weights, model inference, CUDA numerical parity, baseline quality and speed have not been run.** This package is a prepared experiment, not evidence for the regional score-transfer hypothesis.

## Scope and immutable backend

- Author repository: https://github.com/ljwljwljwljw/kvpress
- Exact commit: `39748e9c8301944128eb2e681bfb074e9a2de860`
- The tree reports 0.5.5 but includes post-release fixes. Installing only `kvpress==0.5.5` is rejected.
- Model/tokenizer first pilot: `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, BF16, a single visible CUDA device, one loaded model, batch size one.
- `torch_eager` virtual scoring, SDPA attention, author score storage precision, head-scoped virtual-to-exact calibration, no boundary guards, ordinary KVzip rather than KVzip+.
- The experiment reuses the author-KVPress chat pathway across every arm. LongBench LCC/RepoBench-P originally uses raw code completion; this is explicitly a **LongBench prompt adaptation**, not reproduction of its original raw-completion protocol. Exact replication of paper quality remains unverified.

Four mandatory arms:

1. `fullkv`: reference without eviction
2. `echo`: author EchoPress scoring, first exact replay and calibration
3. `kvzip`: author KVzip context-token partition
4. `kvzip_echo_partition`: exact KVzip reconstruction using EchoPress's first/later chunk boundaries and prompts, an explicitly named diagnostic adaptation

Full KV runs once per context; compressed methods rebuild independently for each eviction ratio. This favors clarity over minimizing pilot compute. No regional map, learned scorer, anchor policy, no-calibration ablation or float32-storage intervention is implemented yet. Full-reconstruction score tensors are diagnostic reference data, never deployable oracle inputs.

## Honest systems accounting

The pinned attention patch replaces evicted keys with fake keys during attention. It retains the original number of KV tensors and can raise if a nullifying hyperplane is not found. **It does not implement physical compaction, real KV-byte savings or a speedup.** A failure is recorded, not silently replaced by another masking implementation.

The runner records both the tensor view's KV payload and distinct backing-storage bytes. Sliced replay caches can retain a larger allocation than their visible shape, so storage can grow rather than stay equal. Logical retained pairs are never relabeled physical bytes. CUDA allocated/reserved and stage peaks are reported, plus per-query cache cloning and model-loading footprints. CPU RAM is not fully measured; captured diagnostic tensor bytes are a partial CPU-memory account.

Timing fields separately cover input/H2D setup, press/prefill/scoring, post-score capture/report, total builder excluding disk I/O, tensor serialization and the full arm including query/result I/O. GPU-stage boundaries are CUDA-synchronized; the full arm excludes its own timing-summary write and no fsync time is claimed. Hooks and CPU score copies are included in their corresponding measured interval. They are cold-run diagnostic timings without warmup or isolated-kernel claims. Replay pass count, complete replay input tokens (including prompts/suffix/postfix), and a dense causal attention-pair proxy are captured. The proxy is not measured FLOPs or compute matching.

## Correctness boundaries

- Compression receives only a `PreparedContext`; questions and labels are rejected in its schema. Questions are separate local inputs. The scoring API cannot inspect them.
- A fresh full context prefill is made for every build. Before each question the frozen context cache is deep-copied with disjoint storage and the build's mask is reinstalled. This is necessary because the author attention patch changes cached keys in-place. Separate clones also isolate metadata.
- All arms share exact token IDs, greedy generation, sink count and recent-token protection. `recent_tokens=0` preserves author defaults; a nonzero value is explicitly a common-policy adaptation, and its protected pairs consume the same total budget.
- Eviction is the author's floor-truncated global pair budget and two-stage torch top-k. Impossible protected budgets fail. Actual mask counts and protection are asserted. Global/layer cutoff ties, layer/head and layer/head/region retention are recorded; no stable tie-breaking equivalence is assumed.
- Context truncation is either `reject` or explicit right truncation. The default real-data generator uses reject. Removed ranges, original/retained lengths and token hashes are recorded; regions are clipped to exactly the retained span. Questions and generated lengths are never silently truncated.
- The author reconstructs with its own tokenizer loading that lacks a revision. A scoped adapter replaces only the two press modules' tokenizer factories with the already immutable-revision tokenizer. Prefix/suffix IDs and the canonical chat-template hash must match the frozen tokenized input contract.
- Empty bodies, singleton first anchors when mapping is required, insufficient first-replay budget, nonfinite scores, invalid masks and positional overflows fail. Scaled/dynamic RoPE, sliding-window caches, remote model code, quantized caches, multiple devices and other architectures are outside this first pilot.
- Greedy decoding stops if the first predicted token is EOS. This corrects a small upstream pipeline behavior consistently for all arms and is documented rather than called bit-identical pipeline generation.

## Inputs and strict schema

`python -m echo_pilot run --config CONFIG.json` validates without importing torch, Transformers or KVPress. Paths in CONFIG are relative to that file. Unknown fields are rejected to catch accidental configuration drift. Inputs are local JSONL; no dataset loader scripts are executed.

Context record fields:

- `context_id`, `source_id`: nonempty identifiers
- `context_token_ids`: complete frozen context segment, including the model's chat prefix
- `regions`: contiguous, nonoverlapping half-open token spans covering the post-prefix body; each has `region_id`, `label`, `start`, `end`

Question record fields:

- `context_id`, globally unique `question_id`
- `question_token_ids`: the future query segment including the exact frozen assistant/chat suffix; explicit answer prefixes are currently unsupported
- Optional `reference_texts` enables only raw string exact match after generation. Official benchmark scoring belongs in `data_prep.metrics`; the prepared LongBench inputs keep labels in a separate file

Dataset/model/tokenizer revisions and the local-origin preprocessing revision must be exact 40-character commits. Preprocessing content SHA-256 is authoritative across different source-transfer histories. Dataset files, full runtime package inventory and chat template are SHA-256 frozen. Fake IDs/tokens used in tests are labeled `fixture_only`; the inference entry point refuses fixture data.

## Future smoke workflow (not executed)

Read `INSTALL.md` first. After an actual environment has been installed and preprocessing code has an actual local Git commit:

```bash
python -m echo_pilot freeze-environment --output packages.lock.json
python -m echo_pilot make-config --packages-lock packages.lock.json \
  --preprocessing-revision ACTUAL_COMMIT_SHA --task lcc \
  --preset smoke8k --split development --output configs/pilot-lcc-smoke.json
python -m echo_pilot run --config configs/pilot-lcc-smoke.json
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m echo_pilot preflight --config configs/pilot-lcc-smoke.json
```

`make-config` verifies preprocessing source bytes against the data manifest and records the original local-history commit as provenance. The commit value is operator-supplied local-origin provenance; its syntax is checked, while preprocessing content SHA-256 is verified. It does not require that commit to be an ancestor of a source-only transferred repository. Actual inference separately requires a clean committed checkout and records its actual HEAD through the shared evidence manifest. It defaults to **development only** and cannot combine development with heldout. `smoke8k` produces 2 LCC development contexts or 1 RepoBench-P development context, four arms at 50% eviction. It is a small compatibility smoke, not a statistically adequate research result. Use `--task repobench-p` for the separate 1-context smoke.

Only on an approved/provisioned GPU, after successful preflight:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m echo_pilot run --config configs/pilot-lcc-smoke.json \
  --output outputs/pilot-lcc-smoke --execute
```

No weights/tokenizers are downloaded by any runner command. They must have been separately staged at the exact revisions. Use `--preset pilot16k --split development` next only if smoke compatibility is established. Heldout requires an explicit `--split heldout`; don't inspect it for adaptive tuning. The prepared data manifest remains the authority for selected IDs and exclusion rules.

## Outputs

Fresh output directory required; no overwrite/resume or hidden retries.

- Frozen configuration, input plan, full package inventory, verified source hashes and resolved model/RoPE configuration
- `model_load.json`: parameter bytes and model-load device memory/time
- Per-build JSON: context fingerprint, truncation, chunks and replay token IDs, actual forward lengths, instrumentation-inclusive time/memory, allocation/tie counts
- Per-build `scores_masks.pt`: native-dtype score arrays before/after protection; Echo virtual author-storage arrays and pre-storage float32 chunk scores; first virtual/exact anchors where calibration occurs; exact masks and keep tensors. These are own-produced tensor containers; use `torch.load(..., weights_only=True)` to inspect
- `results.jsonl`: every raw generated token sequence, decoded/special-token text, context/question/build IDs, query clone and generation time/memory
- SHA-256 artifact manifest plus `COMPLETE.json` only after success; `FAILED.json` marks incomplete experiments

The outputs may contain dataset code/text and tokenized source. Preserve the source's research-use and redistribution restrictions; they are not intended for Git publication.

## Validation and remaining gates

```bash
python -m unittest discover -s echo_pilot/tests -v
python -m compileall -q echo_pilot
```

Synthetic/fixture tests check strict pins, hashes, query separation, exact truncation, budgets, clone disjointness, storage accounting, missing-dependency errors, dry-run isolation and explicit execution gating. They do **not** validate real CUDA hooks, numerical parity, exact tokenization/model behavior, real physical allocation, downstream quality or runtime compatibility. No model result is claimed.

Source files needed beyond the earlier audit are preserved with Apache-2.0 SPDX headers under `vendor_audit/`; each has verified Git blob and SHA-256 provenance. See the original license in `research/sources/echopress_39748e9/LICENSE`.
