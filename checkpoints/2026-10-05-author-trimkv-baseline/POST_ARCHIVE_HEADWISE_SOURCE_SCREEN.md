# Exact-policy headwise cache representation screen

2026-10-05 UTC. **NO-GO as a new research mechanism.** Source-only screen tied to the released static TRIM-KV baseline. No model/tensor imports, numerical tests, trained payload reads, installations, GPU/AutoDL operations, or external writes. Only existing source text, public primary papers/code and scalar arithmetic were inspected. Report and source ledger are small files in /tmp.

## Decision

Stable per-head K/V slots plus a chronological logical-to-physical index vector can preserve the mathematical static TRIM policy and avoid survivor K/V copies. It is ordinary indexed sparse attention combined with a free list. The surviving difference from prior work is the chronological enumeration requirement, which does not itself constitute a differentiated mechanism. Existing work already supplies per-head memory management, victim-slot reuse, and eviction-proportional relocation. A kernel can enumerate an ordinary ordered index array; that is an engineering port, not a new cache method.

Do not divert the faithful trained CPU baseline to implement or benchmark this as a novelty candidate. No differentiated mechanism survives strongly enough to justify a new falsifier experiment. This screen does not establish that the engineering optimization is slow or useless; it rejects the current research claim and makes no measured speed assertion.

## Actual baseline and two separate copy costs

Reviewed the two requested reports under research/reconstructed and re-read the pinned author's source:
- ngocbh/trimkv commit 49d2a2951ba12e6431a3986b418ee791b209d7c2
- src/trimkv/cache_utils.py local SHA256 bb3013d5ff53a8264218b48bafa513c608ad3919d2c6d4172fb248a8b656cae5
- Qwen3-4B base config local SHA256 5beea1a4a34c62782bfb2f911c606741a3bab8f92d80a118fa053c28af12e8ba

The static cache stores fixed log_beta and original absolute positions with every row. After each original whole-model call it tests R >= M+32, forms FP32 log_beta*(N+1-position), selects top M independently per KV head, sorts the selected physical indices ascending and gathers K/V/gates/IDs. Chronology follows inductively from chronological append plus this sorted gather. The newest Q1 row has age 2. No protected recency, sink or within-call prune can be added.

Crucially, update() also calls torch.cat on K, V, gates and IDs **on every append**, copying the old cache. An implementation replacing both cat and prune gathers cannot attribute its entire gain to a new eviction representation. Ordinary preallocated contiguous M+32 storage with in-place append is a mandatory stronger control. The original oversized prefill/question calls still require temporary capacity beyond M+32 before their single post-call prune.

Sources: [append](https://github.com/ngocbh/trimkv/blob/49d2a2951ba12e6431a3986b418ee791b209d7c2/src/trimkv/cache_utils.py#L134-L175), [prune](https://github.com/ngocbh/trimkv/blob/49d2a2951ba12e6431a3986b418ee791b209d7c2/src/trimkv/cache_utils.py#L199-L247).

The SAME author file already includes per-layer/per-head block tables, free-block indices and in-place append in PagedCache, followed by gather-and-repack in PagedTrimKVCache. It is not an exact-policy replacement: its default pooled fixed budget chooses globally across layers/heads, and its physical block default is 256. Its backing tensor pool grows geometrically and releasing blocks returns them to its internal pool, not necessarily to the system allocator. Changing only its selection to independent per-head M is an ordinary port.

Sources: [pool and head tables](https://github.com/ngocbh/trimkv/blob/49d2a2951ba12e6431a3986b418ee791b209d7c2/src/trimkv/cache_utils.py#L538-L650), [append](https://github.com/ngocbh/trimkv/blob/49d2a2951ba12e6431a3986b418ee791b209d7c2/src/trimkv/cache_utils.py#L663-L738), [pooled select and compact](https://github.com/ngocbh/trimkv/blob/49d2a2951ba12e6431a3986b418ee791b209d7c2/src/trimkv/cache_utils.py#L886-L1032).

## Closest actual implementations, pinned

### HeadKV, ICLR 2025

FYYFU/HeadKV at **0862a0955fe82e9ff611d59541918e02c5def625** (2025-03-10).
- DynamicCacheSplitHeadFlatten.update invokes update_flatten_view separately for K and V
- Its CUDA implementation allocates a new flattened tensor and copies every old head segment before inserting the new row
- ReasonSnapKVCluster gathers per-head selected prompt rows then concatenates the recent window and heads; the selected portion is in score order, not TRIM chronological order
- The inspected Llama integration expands GQA K/V to query heads before compression and uses variable-length FlashAttention. Reusing its reported storage numbers for Qwen's eight physical KV heads would be wrong

This is direct evidence that HeadKV's released layout is not zero-copy, but its inefficiency is not evidence of novelty.
[cache/selection](https://github.com/FYYFU/HeadKV/blob/0862a0955fe82e9ff611d59541918e02c5def625/headkv/snapkv_utils.py#L15-L58),
[CUDA copy](https://github.com/FYYFU/HeadKV/blob/0862a0955fe82e9ff611d59541918e02c5def625/csrc/csrc/cuda_api.cu#L12-L78),
[GQA expansion](https://github.com/FYYFU/HeadKV/blob/0862a0955fe82e9ff611d59541918e02c5def625/headkv/adaptive_llama_hijack.py#L235-L307),
[Reason selection](https://github.com/FYYFU/HeadKV/blob/0862a0955fe82e9ff611d59541918e02c5def625/headkv/snapkv_utils.py#L418-L490).

### Ada-KV, NeurIPS 2025

FFY0/AdaKV at **04497abac4c1a58426f3daf1014578990e225cc5** (2025-11-26).
- The same flattened-head update allocates and copies old rows
- Source has explicit GQA and non-GQA selection routes; GQA groups scores and retains one K/V representation per group, while non-GQA repeats K/V
- Both selection routes gather retained head rows and concatenate them; no stable-slot chronological indirection is implemented in these inspected paths

Head-adaptive budgets are already established, but they differ from fixed M separately for every TRIM KV head.
[cache](https://github.com/FFY0/AdaKV/blob/04497abac4c1a58426f3daf1014578990e225cc5/adaptive_snapkv/monkeypatch/snapkv_utils.py#L14-L62),
[GQA route and gather](https://github.com/FFY0/AdaKV/blob/04497abac4c1a58426f3daf1014578990e225cc5/adaptive_snapkv/monkeypatch/snapkv_utils.py#L372-L502),
[CUDA copy](https://github.com/FFY0/AdaKV/blob/04497abac4c1a58426f3daf1014578990e225cc5/csrc/csrc/cuda_api.cu#L12-L80).

### KV-Compress: stronger old control than full survivor gather

IsaacRe/vllm-kvcompress at **5e2a639447aeda377ddfaefdb300fa011bf1cad3** (2024-10-28).
- Slot mapping is [token, KV head]; block tables are [layer, sequence, KV head, block]
- single_tier_schedule_cache_moves_kernel scans at most the evicted-count tail positions, moving live tail rows into earlier evicted holes, then frees tail blocks
- Thus for a 32-row per-head eviction, at most 32 surviving rows per head need relocating, versus M in the author TRIM gather
- This scrambles physical chronological order. Applying its block-aligned selection wholesale also changes TRIM's policy. Using its movement idea with exact TRIM victims and a chronological indirection vector is an ordinary composition

[head-specific append](https://github.com/IsaacRe/vllm-kvcompress/blob/5e2a639447aeda377ddfaefdb300fa011bf1cad3/csrc/kvcompress_cache_kernels.cu#L29-L87),
[move schedule](https://github.com/IsaacRe/vllm-kvcompress/blob/5e2a639447aeda377ddfaefdb300fa011bf1cad3/csrc/kvcompress_eviction_kernels.cu#L223-L289),
[move execution](https://github.com/IsaacRe/vllm-kvcompress/blob/5e2a639447aeda377ddfaefdb300fa011bf1cad3/csrc/kvcompress_eviction_kernels.cu#L364-L433).

### DiffKV, SOSP 2025: actual per-head victim-slot reuse

zyqCSL/DiffKV at **c747e6d0fbebd846486b6ee9d3727a6f3b039a6b** (2025-10-11).
- GPU allocator and attention explicitly have per-request/layer/KV-head block tables
- Decode compression selects a victim in the high/low precision head cache and writes the newly appended row into that victim's slot; it can also requantize one victim into another region
- Hence "per-head slots avoid copying all survivors" already has actual author CUDA, not only a paging analogy
- It changes eviction policy/precision and visits physical high/low-precision regions, so it is not a numerically faithful TRIM drop-in

[replacement cases](https://github.com/zyqCSL/DiffKV/blob/c747e6d0fbebd846486b6ee9d3727a6f3b039a6b/csrc/cache_kernels.cu#L1001-L1066),
[headwise tables](https://github.com/zyqCSL/DiffKV/blob/c747e6d0fbebd846486b6ee9d3727a6f3b039a6b/csrc/mem_mgt_kernels.cu#L150-L282),
[indexed attention](https://github.com/zyqCSL/DiffKV/blob/c747e6d0fbebd846486b6ee9d3727a6f3b039a6b/csrc/attention/sparse_attention_kernels.cu#L127-L165).

### ThinKV / ThinkKV, ICLR 2026: directly anticipates the no-gather pitch

Primary paper **arXiv:2510.01290v2, 2026-05-07**, sections 5.2 and C.3. Its CT system marks evicted slots, reuses those slots for later tokens and deliberately leaves physical order unsorted, relying on mathematical attention permutation invariance. This directly precedes the proposed no-survivor-copy motivation. I did not locate a public author CT implementation linked by the paper, so this is paper evidence, not a verified code pin. The paper reports little batch-one gain and emphasizes larger-batch throughput; its headline speedups do not transfer to this batch-one 4B CPU case.

[system and ordering](https://arxiv.org/html/2510.01290v2#S5.SS2), [batch-size qualification](https://arxiv.org/html/2510.01290v2#S6.SS2).

### R-KV author serving updates, 2026: inspect semantics rather than labels

Zefan-Cai/R-KV at **6715468b9872442be72e5c97322e4d9c9a2abf55** (2026-07-20).
- FlashInfer compressor preserves per-KV-head token choice. Its engine stacks current K/V, calls the compressor, and writes all survivors back to the region prefix. This remains a gather/repack control
- SGLang integration explicitly reduces scores across heads and layers to one global per-token selection because its req_to_token mapping is shared. It then relocates survivors to prefix slots and frees the tail using page_size=1
- That SGLang route is a material policy change and must not be described as exact headwise TRIM just because it serves compressed KV

[FlashInfer contract](https://github.com/Zefan-Cai/R-KV/blob/6715468b9872442be72e5c97322e4d9c9a2abf55/FlashInfer/rkv/compressor.py#L110-L152),
[actual writes](https://github.com/Zefan-Cai/R-KV/blob/6715468b9872442be72e5c97322e4d9c9a2abf55/FlashInfer/rkv/engine.py#L426-L451),
[SGLang policy change](https://github.com/Zefan-Cai/R-KV/blob/6715468b9872442be72e5c97322e4d9c9a2abf55/SGLang/docs/IMPLEMENTATION.md#L34-L48),
[physical compaction](https://github.com/Zefan-Cai/R-KV/blob/6715468b9872442be72e5c97322e4d9c9a2abf55/SGLang/docs/IMPLEMENTATION.md#L130-L142).

### RazorAttention: actual vendor surface, limited code visibility

The ICLR2025 paper keeps full retrieval-head history and sink/local history plus compensation for other heads, a different policy. Huawei's July2025 primary deployment article identifies the real production route in MindIE/MindStudio and CANN, rather than a guessed GitHub repo. CANN **8.1.RC1** documents per-head block tables [num_tokens*kv_heads, max_blocks] and one-head K/V blocks [num_blocks, block_size,1,D] plus razorOffset.

Inspectable vendor code: mindspore-ai/golden-stick at **58619f044a2c9a1c8e02708d3cb5ca9b0bc6e08f** (2026-06-15), mindspore_gs/sequence_compress/razor_attention/razor_attention.py, Git blob **1141d4c80dd2b8acf7203fbbe8f256ef8b45bd93**. This module calibrates/selects echo and induction heads; it is not the runtime cache allocator. The production operator's internals were not publicly verified here, so no claim about its exact copy count or chronology follows.

[paper](https://openreview.net/pdf?id=tkiZQlL04w),
[primary deployment article](https://www.hiascend.com/developer/techArticles/20250709-1),
[versioned operator contract](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/ascendtbapi/ascendtb_01_0193.html),
[calibration code](https://github.com/mindspore-ai/golden-stick/blob/58619f044a2c9a1c8e02708d3cb5ca9b0bc6e08f/mindspore_gs/sequence_compress/razor_attention/razor_attention.py#L127-L189).

## Source-derived traffic: batch 1, actual Qwen3-4B geometry

36 layers, 8 KV heads, D128, BF16 K/V. One token-position across all heads/layers is S=36*8*128*2(K,V)*2 bytes=147,456 bytes=144 KiB. "32k" below means the author's exact 32,768 budget. Logical traffic counts one read and one write for each copied byte, not measured DRAM/HBM traffic. Caching, write allocation, allocator behavior, GQA kernel reuse, and paging can change physical traffic.

| Quantity | M=960 | M=32768 |
|---|---:|---:|
| Retained K+V | 135 MiB | 4608 MiB = 4.5 GiB |
| K+V at M+32 | 139.5 MiB | 4612.5 MiB |
| Survivor gather read+write/event: 2SM | 270 MiB | 9216 MiB = 9 GiB |
| Gather amortized over 32 Q1 appends | 8.4375 MiB/token | 288 MiB/token |
| Existing cat read+write, cycle mean: 2S(M+16.5) | 274.640625 MiB/token | 9220.640625 MiB/token |
| One ideal K+V attention read, cycle mean | 137.3203125 MiB/token | 4610.3203125 MiB/token |
| One int32 chronological slot vector/head | 1.0546875 MiB | 36 MiB |
| Existing BF16 gate + int64 ID at M | 2.63671875 MiB | 90 MiB |

The cat rows count both input reads and output writes, including new input rows; old-survivor-only traffic is slightly lower. A normal preallocation removes this repeated old-cache copy, without novel eviction logic. Metadata gather adds about 1.95% to K/V gather bytes under these dtypes; score/topk/sort costs are additional.

At most32 moved survivors/head in an eviction-proportional hole-filling control means at most9 MiB total K/V read+write per event, or0.28125 MiB/token, before metadata. Pure stable slots avoid even those moves but charge indirect loads every attention.

Compared only with the stipulated approximately8,000,000,000 weight bytes streamed per output token, eliminating the prune gather saves **0.111%** at M960 and **3.775%** at M32768. Relative to weights plus one ideal cache read plus gather, the ideal traffic fractions are about0.109% and2.30%, after ordinary preallocation. These are transfer-budget comparisons, not speedup bounds for all architectures: launch/selection overhead can matter, whereas matmul, disk/weight paging or other costs can dominate.

At either large-M limit the gather overhead is only approximately2/32=6.25% of one attention K/V read. A modest loss in indexed-attention efficiency can consume that budget. ThinKV's high-batch results are not evidence otherwise.

32GB does not make this a memory necessity at M32768: approximately8GB weights plus4.83GB KV, approximately95MB trained gates, existing metadata and a36MiB index are roughly13GB before activations/runtime/scratch. This is capacity arithmetic only, not admission of the original dense-history attention. BF16 weights resident in32GB can still be streamed through memory bandwidth every token. Oversized original calls and dense score/activation materialization need separate peak-memory admission.

## Required controls and exactness caveats

1. **Preallocated contiguous cache + original topM/sort/gather.** Removes author cat copies while preserving policy and logical layout. Separate one-time conversion cost after oversized calls.
2. **Complementary bottom(R-M), usually bottom32, + ascending survivor scan/gather.** Preserves the score-defined keep set if there is no boundary tie; reduces selection/sort output size. At oversized calls R-M need not32, so choose the smaller side. It does not eliminate K/V compaction.
3. **Existing head-specific page/slot tables with ordered indices**, and ordinary sorted-index gather before an unchanged dense attention. The latter recovers order but pays a gather every token, defeating the target.
4. **Eviction-proportional tail-to-hole movement**, maintaining the same logical chronology via an index table. Already supported by the KV-Compress movement mechanism.
5. **Buffer128 versus32 only as altered-policy control.** At fixed M, adds96 positions of capacity=13.5MiB across this model; gather amortization decreases fourfold, but prune timing, residency, attention, gates and future keep sets change. Holding peak capacity constant instead requires reducing M and also changes policy.
6. **Tombstones** without reuse/reclamation do not bound physical capacity. Reusing per-head slots inside an M+32 pool bounds steady-state decode storage; it does not automatically shrink an oversized prefill allocation. With block allocation, scattered survivors can pin nearly all pages. Budget means physical payload, reserved blocks, masks/indices, allocator slack and conversion scratch, not just active rows.

"Exact" has three different obligations:
- Policy/state: identical clocks, all original call boundaries, irreversible removal, native FP32 score expression and exact ordered K/V/gate/ID state
- Mathematical attention: same keys/values, original RoPE/IDs, GQA mapping and causal visibility. Physical reordering is harmless only with matching masks; Q>1 cannot infer causality from recycled slot number
- Numerical execution: chronological enumeration alone does not fix softmax reduction tree, tile boundaries, accumulation, FMA use or backend rounding. An indirect kernel may be mathematically equal yet not bitwise equal. If that changes a later hidden state/gate near a cutoff, closed-loop keep sets may differ

Native torch.topk does not promise stable tie membership. Complementary bottom-k, a custom heap, or changing score-array order can select different members at an equal cutoff. To claim exact author output rather than mathematical top-M equivalence, preserve native tie behavior (e.g. route ambiguous cutoffs through the original oracle); inventing an ID tie-breaker is a policy change. Index sorting restores order only after membership is decided.

## Fixed age-affine score maintenance

Algebraically each old score is b_i*N + b_i*(1-p_i), so kinetic/parametric heaps are standard prior art, including insert/delete-min under increasing time; no new name is justified. [Kaplan, Tarjan and Tsioutsiouliklis, Faster Kinetic Heaps](https://www.cs.princeton.edu/courses/archive/fall03/cs528/handouts/faster%20kinetic%20heaps.pdf), [Fonseca and de Figueiredo, kinetic heap analysis](https://pageperso.lis-lab.fr/guilherme.fonseca/kh.pdf).

There is no source evidence it beats complementary bottom32 here. Scores are recomputed only once per32 Q1 appends: about8,928 and295,200 scalar score multiplications per generated token at the two budgets, across all288 heads. This is metadata work beside billions of weight bytes and attention arithmetic. Maintaining all crossing events between observed prune times is potentially wasted work; insertion/deletion and heap/event metadata add overhead. Algebraically distributed affine evaluation can also round differently from the author's FP32 b_i*(integer_age) expression; ties and discrete evaluation times require care. Reject as an additional candidate in this screen, without claiming a measured comparison.

## Scope and stopping point

No test is proposed because no substantive differentiator survived the source comparison. Continue the already-authorized faithful CPU baseline. If profiling that baseline later shows a real bottleneck, ordinary preallocation and complementary selection are legitimate engineering improvements after baseline parity; their usefulness would not establish a new research method.

The search was focused, not an exhaustive novelty theorem. No public CT kernel pin or Razor production kernel internals were verified. Exact inspected Git commits and blobs are in /tmp/trimkv_headwise_sources_20261005.tsv. No source checkout, large source copy or model download was created.

