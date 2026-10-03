# EchoPress CPU arithmetic audit

This isolated directory is a synthetic pre-GPU diagnostic. It does not load a model, download weights, use a GPU, call paid services, modify the shared harness, or implement physical KV compaction. It depends only on already-installed NumPy and the Python standard library.

Source baseline: [EchoPress at 39748e9c8301944128eb2e681bfb074e9a2de860](https://github.com/ljwljwljwljw/kvpress/blob/39748e9c8301944128eb2e681bfb074e9a2de860/kvpress/presses/echo_press.py#L287-L314), Apache-2.0. The source snapshot and its license/provenance are in `../../sources/echopress_39748e9/`. The full audit is `../../echo_calibration_audit.md`.

From the `kv_research_20261003` root:

```sh
python -m unittest discover -s research/diagnostics/echo_calibration -p 'test_*.py' -v
python research/diagnostics/echo_calibration/synthetic_smoke.py
```

## What is faithful

- Independently sort source/target marginals, left-searchsorted intervals, clamp denominator to 1e-12, interpolate and clip, preserve duplicate-knot behavior
- Float32 map arithmetic and per-layer/head row isolation
- Exact first-chunk replacement and later-only forward calibration
- Author first replay token formula and later prompt/postfix/suffix lengths
- Global floor(total_pairs × eviction_ratio) logical eviction count

## What is deliberately not claimed

- No executed PyTorch equivalence: torch and transformers were not installed
- No reproduction of BF16 score storage or low-precision QK products, CUDA/Triton reductions, model hooks, tokenizer output, RoPE implementation, attention patch, or generated answers
- Logical masks do not save physical KV bytes
- Stable flat-index tie-breaking differs from torch.topk's unspecified tie order; tied-cutoff results need separate analysis
- Empty/nonfinite inputs, singleton anchors, and impossible protected-sink budgets raise explicit errors. Upstream does not provide all these checks; these are diagnostic guardrails
- Cost ledger is a dense causal query-key pair proxy, not wall time or measured FLOPs

The scale oracle uses full exact reconstruction marginals and cannot be deployed at the stated anchor budget. The rank oracle uses exact rankings and is also diagnostic only. Both must use identical region labels/partitions, and callers must preserve the same first exact chunk when diagnosing EchoPress. Conditional/shrinkage candidate policies are not implemented.

All fixtures and saved results are explicitly synthetic. Their purpose is to distinguish scale-transfer error from within-region rank error. They establish no real-model failure or improvement and confer no novelty claim.
