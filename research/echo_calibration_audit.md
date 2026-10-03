# EchoPress calibration-transfer audit and falsification specification

Audit date: 2026-10-03 UTC. Evidence: author code, primary papers, public release history, and synthetic CPU tests. **No model inference, GPU, weights download, paid API, or remote write was performed. No positive research result is established.**

## Decision

Keep a narrow **diagnostic hypothesis**, not an accepted method: a request-level score map fitted on one region may transfer poorly to another. This is directly testable, but code/prose/table heterogeneity alone does not prove score-scale error. Different virtual-score distributions may be correct changes in importance. First show that an exact regional-marginal oracle improves downstream outcomes without changing virtual within-region rankings. If only a ranking oracle helps, reject calibration as the mechanism.

The complex region-conditioned/shrunk policy should not be built before the first/random/uniform-anchor controls. Uniform coverage, ordinary recency protection, precision/tie handling, or an existing learned scorer may eliminate the opportunity. None has been empirically ruled out. Matching scoring partitions removes a major confound; it does not by itself validate quantile transfer.

**Prominent systems limitation:** this KVPress implementation masks evicted indices in attention. It does not physically free their KV tensors. Logical retained-pair count is an accuracy experiment budget, not measured memory savings. Actual bytes/latency require an independently validated compaction backend.

## 1. Baseline availability, immutable pins, and provenance

- Author repository: [ljwljwljwljw/kvpress](https://github.com/ljwljwljwljw/kvpress), public and readable through the connected GitHub read API
- Audited `echo-press` tip: [39748e9c8301944128eb2e681bfb074e9a2de860](https://github.com/ljwljwljwljw/kvpress/commit/39748e9c8301944128eb2e681bfb074e9a2de860), committed 2026-10-02; do not pin only the mutable branch or version string
- Tree: `7fb6af0b3212770127d6c8e334d34e8ad31dcee6`
- Upstream parent: [327ed6f316cfadcff37a3e644a817f9ff58b8848](https://github.com/NVIDIA/kvpress/commit/327ed6f316cfadcff37a3e644a817f9ff58b8848), also NVIDIA main when checked
- Code license: [Apache-2.0](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/LICENSE), including SPDX headers in the added implementation. This does not establish licenses for model weights, datasets, or other repositories
- The package reports 0.5.5, but it is **not identical to the 0.5.5 release**. [Release tag](https://github.com/NVIDIA/kvpress/releases/tag/v0.5.5) resolves to `a13a1da31ab2b92671d9e26994d5bdbd69d9bd61`; EchoPress's upstream parent is five commits ahead
- Selected source snapshot: `research/sources/echopress_39748e9/`. `provenance.json` lists exact URLs, Git blob IDs, local SHA-256s, and verified Git blob equality for 14 retrieved files. Full repository/dependencies were not installed

The [EchoPress paper](https://arxiv.org/html/2610.00412v1) is dated 2026-09-30. It motivates first-chunk reconstruction by larger prefill/reconstruction state mismatch there. It reports code-completion regressions against KVzip at aggressive eviction, an appropriate natural starting point. Its accuracy path uses KVPress/SDPA; its latency path uses an original-KVzip/FlashAttention implementation with a different chunk convention and physical compaction. Those paper timings are not measurements of our proposed change. The paper's calibration ablation tests direction and layer/head granularity, not equal-budget alternative-region anchors. Its claim that an exact first chunk fits a 2,048-token reconstruction-input budget agrees with the default code, subject to the small-budget edge case below.

## 2. Code-derived mathematical specification

Primary sources: [echo_press.py](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/echo_press.py), [kvzip_press.py](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/kvzip_press.py), and [Triton kernels](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/echo_press_kernels.py). The equations below are an implementation reading, not new empirical claims.

### 2.1 Exact reconstruction target

Let original cached context length be n including the chat prefix; target chunk C=[a,b), reconstruction input contain P prompt/suffix tokens plus m=b-a context tokens, and G query heads share each KV head. Reconstruction runs against the full original cache, so its hidden states see all original context. For score calculation only, select original sink positions S=[0,min(4,a)), original chunk keys C, and the repeated-input keys with causal masking.

For layer l, KV head h, original cached key i in C:

    exact[l,h,i] = max over g,u of exp(q_replay[g,u] dot k_original[i]/sqrt(d)) / Z[g,u]

    Z[g,u] = sum over v in S∪C of exp(q_replay[g,u] dot k_original[v]/sqrt(d))
           + sum over v<=u in repeat input of exp(q_replay[g,u] dot k_replay[v]/sqrt(d))

The prompt query rows participate in that maximum. Reconstruction KV additions are removed per layer after scoring. This is KVzip's attention score, **not downstream evidence importance ground truth**. `kvzip_plus_normalization=False` by default; enabling it adds hidden-state/value-output weighting and changes the target.

### 2.2 Virtual scorer

During the one full-context prefill, capture projected queries per layer, apply Qwen3 query normalization if applicable, and apply the prefill position embeddings. Original keys already contain their position embeddings. For each chunk use offset Δ=n+P-a, obtain shifted q'=R_Δ q, and compute:

    virtual[l,h,i] = max over g,j in C of exp(q'[g,j] dot k_original[i]/sqrt(d)) / Z_virtual[g,j]

    Z_virtual[g,j] = sum over v in S∪C of exp(q'[g,j] dot k_original[v]/sqrt(d))
                   + sum over v in C,v<=j of exp(q[g,j] dot k_original[v]/sqrt(d))

The second term implements a causal virtual copy using unchanged q-k products; its shared rotation cancels. Prompt rows and columns are omitted from this score. This preserves causal prefill representations, not full-context replay representations. No downstream question is consulted. Queries are released one layer at a time after scalar scores are calculated.

### 2.3 Chunk boundaries and actual input budget

`prepare()` at lines 147–175 uses context after the chat prefix. Let q0 be the token count of the first repeat prompt and s the chat suffix count:

    first_context_capacity = max(1, 2048 - q0 - s)
    actual_first_input = min(remaining_context, first_context_capacity) + q0 + s

Later virtual chunks contain up to 2,048 **context** tokens. Their positional offset P includes the second prompt, the preceding chunk's last min(8,previous_size) context tokens, and the suffix. Although not executed, these prompt tokens affect virtual RoPE offset. Obtain counts from the exact tokenizer revision; do not hard-code the smoke-test examples as real model counts.

The `max(1,...)` means a custom chunk budget smaller than prompt overhead can be exceeded. An empty post-prefix context leaves no first chunk. Guard these unsupported configurations in any integration. For normal long contexts the first pass fits the nominal budget.

### 2.4 Quantile transfer, exactly as implemented

For each layer/head independently, sort first-chunk virtual anchor vector x and exact anchor vector y **separately**, yielding x_(k), y_(k). This is marginal quantile matching, not a paired regression of exact on virtual token scores. Token correspondence is discarded when fitting the curve.

At evaluation value z:

    zc = min(max(z,x_(1)),x_(m))
    j = clamp(left_searchsorted(x_sorted,zc),1,m-1)  # zero-based implementation
    w = clamp((zc-x[j-1])/max(x[j]-x[j-1],1e-12),0,1)
    M(z) = y[j-1] + w*(y[j]-y[j-1])

Apply M only to later virtual scores; retain first-chunk exact scores. Default scope `head` means L×H_kv separate rows; optional `layer` pools all heads within each layer, and `global` pools all layers/heads. `exact_to_virtual` is an ablation, not the default. The first-chunk scalar map does not reverse within-head later-token order except by creating ties. Region-conditioned maps can change **between-region** order and budget allocation but cannot fix rank inversions within a region.

Important precision details: `_v0` keeps first virtual scores from the float32 scorer, while `score_val` uses model dtype. Exact anchors and later virtual scores have therefore passed through model-dtype storage before map-time float32 conversion. The mapped output is cast back again. The torch path materializes up to 1,024 query rows, normalizes in float32, and can use torch.compile. The Triton path is a different two-pass kernel. These should agree within measured tolerances before either is used for mechanism claims; we have not run their GPU parity test.

### 2.5 Eviction semantics

`KVzipPress.compress_post()` promotes initial sinks and evicts floor(r×L×H_kv×n) pairs globally by score; it first counts removals per layer, then finds per-layer bottom-k. `layerwise=False` is the baseline. The final sink promotion is max_score+1, regardless of initialization. `guard_tokens=0` by default. The code records `masked_key_indices`; source tests explicitly assert cache sequence length remains n.

Ties can make selected token identities backend-dependent. Record exact mask counts, tied-cutoff multiplicity, per-layer/head/region allocation, and repeated-run differences. Protected-sink claims are impossible when the requested retained budget is below the number of sink pairs; reject such configurations rather than relying on comments.

## 3. Audit hazards and required artifact controls

1. **Repeated quantile knots:** left searchsorted is intentional baseline behavior. x=[1,2,2], y=[10,20,30] maps both z=2 and z>2 to 20, not 30. All equal x maps to the lowest y. With nonduplicate endpoints, values outside support clip to endpoint y. Do not silently replace this with NumPy's duplicate handling or tie averaging
2. **Singleton anchors:** `_quantile_map` assumes at least two knots; upstream gather is invalid at m=1. Our diagnostic explicitly rejects this case unless no mapping is needed
3. **Precision is a confound:** compare author storage to a common float32-score-storage control across every arm. A gain disappearing after this correction is an implementation/precision issue, not proof of heterogeneous transfer
4. **First-region privilege:** moving the exact anchor also moves which scores are exact. Separate anchor-location effects from map-location effects with diagnostic-only crossed interventions: common exact replacement, vary fitted map; then common map, vary exact replacement
5. **Scoring partitions matter:** smaller chunks change both the maximum's query set and the normalization key set. Four 512-token replays are not samples from the same score definition as one 2,048-token replay. For the coverage experiment compare first/random/uniform sets of the same number and length of microchunks, with one common virtual partition, prompt construction, and exact-score target. Retain author EchoPress as a separate original baseline; report any change due to rechunking alone
6. **Compute matching matters:** record total replay input tokens including each prompt/suffix/postfix, number of passes, model-width projection/MLP work, and attention work. For replay length t over n original keys, a dense causal pair-count proxy is nt+t(t+1)/2. Equal sum(t) leaves different sum(t²) and launch overhead. Match a measured build-cost envelope, not just “4×512=2048”
7. **Single-prepass restriction:** the implementation asserts batch size one and a single full-context prefill, requires separate q_proj, and rejects Gemma3. Do not claim support for packed batches, streaming/paged prefill, or arbitrary architectures
8. **Position/configuration scope:** pin tokenizer, chat template, RoPE settings, model revision, attention backend, dtype, and dependencies. Inspect any dynamic/scaled RoPE changes separately; the constant-offset rotation must match actual embeddings
9. **No hidden future query:** compress distinct context once; answer questions from separate copies of its frozen compressed cache. Keep gold answers, all future question text, and full-KV outcomes outside anchor choice/hyperparameter decisions

## 4. Release fixes already included

[v0.5.5 release notes](https://github.com/NVIDIA/kvpress/releases/tag/v0.5.5) include float32 Knorm scoring to avoid low-precision ties, corrected logits-to-keep handling, KVzap/Transformers-5 checkpoint loading, cache-position phase detection, minimum budgets, random dtype/device handling, nested-answer parsing, and safer evaluation/checkpoint loading. EchoPress's parent already contains these. The [five-commit comparison](https://github.com/NVIDIA/kvpress/compare/a13a1da31ab2b92671d9e26994d5bdbd69d9bd61...327ed6f316cfadcff37a3e644a817f9ff58b8848) additionally includes the LooGLE invalid-prediction denominator fix, query-state helper refactor, MergingPress eligible-set rank fix, DropKV, and a KVgrad default change.

Use the pinned EchoPress tree or a documented equivalent integration for all arms. Merely installing `kvpress==0.5.5` and copying EchoPress may miss its newer `get_query_states` dependency. Do not compare corrected candidate metrics to stale baseline metrics, claim known release fixes as novelty, or interpret Knorm's already-fixed tie problem as evidence of an EchoPress result.

## 5. Nearest primary prior, beyond abstracts

- [EchoPress source](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/echo_press.py): directly owns per-request, layer/head quantile mapping from reconstruction. Any contribution must be about demonstrated transfer failure and a useful budget-matched resolution, not “calibrate pruning scores”
- [Compactor §§3.4–3.5](https://arxiv.org/html/2507.08143v1): blends standardized noncausal-attention and leverage scores, then fits an offline two-parameter context-NLL/retention curve using question/ground-truth-answer likelihood ratios. Inference chooses a retention rate from a quality tolerance. Its “context calibration” concerns achievable compression level, not first-region virtual-to-exact score transport. Its [KVPress source](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/compactor_press.py) documents departures from the paper, so label which baseline is used
- [KVzap §3.8 and Appendix A](https://arxiv.org/html/2601.07891v1): model-specific hidden-state predictors of KVzip+ scores, thresholded eviction, with a recency window. It explicitly discusses short-training-context distribution shift. Its code-completion ablation shows recency-window protection can be decisive. This is a mandatory simple explanation/baseline before attributing EchoPress code regressions to calibration
- [Fast KVzip §3 and implementation](https://arxiv.org/html/2601.17668v1): reconstruction-distilled gating on hidden states; published algorithm includes protected recent tokens and eviction between prefill chunks. A common-backend, full-prefill fixed-budget variant is an adaptation, not the untouched native deployment
- [Expected Attention §2.2](https://arxiv.org/html/2510.00636v1): moment-based future-query approximation with value-output weighting and adaptive head budgets. It is a low-overhead training-free comparator and adjacent evidence that distributional score models are established
- [BACON §§4.2–4.4](https://arxiv.org/html/2606.14782v3): boundary attention evidence filtered by local/interlayer structure and variance-matched to an observation-window score. This is query-aware multimodal/window calibration rather than reusable query-agnostic regional reconstruction, but blocks broad novelty claims about “regional evidence calibration” or “calibrating scores across layers/heads”

This focused search did not identify the exact equal-reconstruction-budget regional quantile-transfer experiment. That is **not** an exhaustive novelty clearance. Shrinkage and stratified sampling are generic tools; their addition alone is weak novelty.

## 6. Fair baseline matrix

### Mandatory order

1. Full KV, paper-faithful EchoPress, and KVzip under one accuracy backend and identical context/template. Add common-score-partition KVzip to distinguish chunk-definition effects
2. Author calibration versus no calibration and common float32 storage. Record sink/recency handling and tie counts
3. One exact first anchor versus one random contiguous anchor of the same context length and matched prompt convention/cost. Use multiple predetermined seeds and log prompt tokens
4. Common microchunk partition: first four, random four without replacement, uniformly spaced four, and document-boundary stratified four. Keep number of passes, per-pass replay shape, virtual partitions and map granularity identical. Use pooled per-layer/head maps first. Do not concatenate disjoint snippets into an artificial single replay and call its target exact KVzip
5. Only if necessary: regional conditional maps from those SAME anchors, with shrinkage toward pooled map. Use fixed content labels/document boundaries or prefill-only features, never gold/question-dependent region labels
6. KVzap-MLP (and linear if useful), Fast KVzip, ExpectedAttention/Compactor, start+recent, and random selection at comparable logical/physical budget

### KVzap details that must not be omitted

The [predictor implementation](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/kvzap_press.py) loads model-specific `nvidia/KVzap-{linear|mlp}-{base_model}` weights and predicts per-head scores from hidden states; the [DMS wrapper](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/dms_press.py) defaults to a final-128-token protected window and threshold selection. The released training uses KVzip+ targets, so raw KVzap scores need not agree with plain KVzip. No predictor was downloaded in this audit.

Evaluate (a) native threshold policy as an operating curve with actual retained counts and (b) an explicitly named global fixed-budget adaptation with original recency protection. Also give EchoPress a matched recency protection control, with those protected pairs deducted from the same total budget. Turning off KVzap's recent window to make it look weaker is unfair. Native per-head `ScorerPress` top-k is not equivalent to global cross-layer top-k. On code tasks this distinction is especially consequential. If KVzap dominates on quality/build cost, reject an expensive calibration mechanism even if it improves EchoPress.

### Optional candidate mathematical form, unvalidated

Let M_pool be a per-layer/head map fitted on all selected anchors; M_r a map on anchors assigned to region r. Use

    M_shrunk,r(x) = (1-w_r) M_pool(x) + w_r M_r(x),  w_r = n_r/(n_r+lambda)

with shared fixed lambda selected on a separate development set. Both maps share target semantics and input domain; convex combination preserves monotonicity. Absent/insufficient regional anchors must fall back to pooled map. A region with no exact anchors cannot acquire an exact regional map from its virtual-score histogram alone. Such a map would be an oracle leak or an additional learned assumption. Do not force all region histograms equal: genuine importance differs across regions. This optional candidate is **not implemented or validated** here.

## 7. Falsification protocol and stopping rules

### Stage A: source/math validation, now completed narrowly

Run isolated NumPy tests for sorted-marginal interpolation, ties, clipping, head separation, empty/nonfinite checks, first-budget counting, region-map/rank oracles, and logical global selection. These establish arithmetic/diagnostic behavior only. Torch baseline, tokenizer, RoPE, CUDA kernel, cache-hook and downstream parity remain untested.

### Stage B: mechanism existence, requires separately authorized model compute

Start with natural LongBench code-completion contexts and a natural multi-document/code-documentation collection; reserve controlled mixed prose/code/table permutations as labeled diagnostics. Use 32–64 contexts as a cheap screening set on one supported 8B BF16 model at 8K–16K. Freeze context IDs/tokenizations before observing results. Include homogeneous and matched-order controls; each permutation gets its own full-KV reference. Quality uncertainty resamples **contexts**, not questions as if independent.

From full exact reconstruction scores on an analysis subset, construct four diagnostics:

- Baseline EchoPress
- Scale-map oracle: fit a true marginal map separately per layer/head/region and retain virtual within-region order; same first-anchor exact override. Full reconstruction labels are allowed only for this diagnostic
- Rank-only oracle: reorder baseline regional score histograms by exact within-region rank, preserving each region's histogram/allocation absent cutoff ties
- Full exact score reference

Evaluate both mask/allocation change and downstream answer/code metrics at 50/75/90% eviction. Report rank agreement, support clipping, duplicate knots/cutoff ties, threshold-crossing disagreement, natural-workload mean, later-region/worst-group quality, and preserved evidence separately. A marginal distribution plot or score correlation is not sufficient evidence.

**Kill** if a meaningful natural quality deficit cannot be reproduced, if regional scale correction does not recover it while rank correction does, or if it disappears with fair float32/recency/partition controls. Full exact reconstruction can itself underperform full KV; matching KVzip is not a correctness guarantee.

### Stage C: deployment-feasible anchors

Only after Stage B passes, run the mandatory anchor matrix. Split development and held-out contexts before selecting any rule. Count every exact forward and prompt; hold logical retained pairs fixed first, then validate physical representation costs. For unequal measured build cost, compare a Pareto frontier or tune to a prespecified cost cap; do not relabel token matching as compute matching.

**Kill the complex mechanism** if pooled uniformly spaced anchors match it within paired uncertainty. A simple upstream robustness patch may remain worthwhile but does not automatically support a substantial research claim. Also kill if ordinary last-128 protection or a cheap existing scorer explains the gain.

### Stage D: usefulness/generalization

Project-chosen gates, not literature facts: aim for at least 2 absolute points of held-out downstream improvement at matched bytes/build-cost envelope, or a meaningful byte reduction at reference quality, without more than 1 point of natural-workload mean regression. These thresholds need context-paired uncertainty and adequate sample size before a conclusion. Confirm on a second architecture before claiming generality. Do not tune repeatedly on the same holdout.

For reusable caches, report cache-build cost and per-question serving cost separately, then amortized totals for fixed reuse counts. Include predictor storage, scalar-score buffers, temporary reconstruction states, full original cache needed during calibration, compaction scratch, positions/offsets, metadata and peak allocated/reserved device memory. No service-throughput claim follows from this audit.

## 8. Delivered CPU diagnostics and limits

Directory: `research/diagnostics/echo_calibration/`

- `reference.py`: float32 NumPy quantile map, first-chunk calibration, token-budget layout, replay cost proxy, scale/rank oracles, logical mask/selection reports, clipping/tie diagnostics
- `test_reference.py`: 29 synthetic unit tests
- `synthetic_smoke.py` and `synthetic_results.json`: constructed scale-shift case corrected by the scale oracle and constructed rank-failure case corrected only by the rank oracle
- `test_run.txt`: successful test transcript
- `README.md`: scope, commands, differences from author runtime

Test run: `python -m unittest discover -s research/diagnostics/echo_calibration -p 'test_*.py' -v`, **29 passed**. The initial run exposed only a float64-versus-float32 exact-equality test-fixture mismatch, corrected by making the fixture float32. No source behavior was changed to make the test pass.

The toy results are constructed counterexamples validating diagnostic separation, **not evidence that EchoPress fails on real contexts or that a deployable policy improves it**. No synthetic accuracy number should appear as a model benchmark. Source author tests were read but not executed; torch/transformers are unavailable in this environment and were not installed. Stable NumPy tie-breaking is documented, not claimed equivalent to torch.topk.
