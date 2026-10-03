"""CPU-only EchoPress calibration audit utilities, NOT a model implementation.

Quantile map follows EchoPress commit 39748e9c8301944128eb2e681bfb074e9a2de860,
echo_press.py:287-314 (Apache-2.0). NumPy is the only third-party dependency.
Default float32 arithmetic mirrors map-time dtype; upstream BF16 storage and
Triton/CUDA matmuls are NOT reproduced. No model loading or network access.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _array(x, name, ndim=None):
    a = np.asarray(x, dtype=np.float32)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must contain only finite values")
    return a


def quantile_map(source, target, values):
    """Independent rowwise sorted-marginal map with upstream left-tie behavior.

    Arrays are [rows, knots] and [rows, evaluation_points]. Values are clipped,
    never extrapolated. Duplicates are NOT collapsed or averaged. Empty/singleton
    anchors and nonfinite inputs are rejected explicitly (upstream has no such
    contract and singleton anchors reach an invalid gather). Empty values work.
    """
    source, target, values = (_array(x, n, 2) for x, n in
                              ((source, "source"), (target, "target"), (values, "values")))
    if source.shape != target.shape or source.shape[0] != values.shape[0]:
        raise ValueError("source/target shapes and all row counts must match")
    if source.shape[0] == 0 or source.shape[1] < 2:
        raise ValueError("at least one row and two anchor knots are required")
    vs, ts = np.sort(source, axis=-1), np.sort(target, axis=-1)
    x = np.minimum(np.maximum(values, vs[:, :1]), vs[:, -1:])
    idx = np.stack([np.searchsorted(row, val, side="left") for row, val in zip(vs, x)])
    idx = np.clip(idx, 1, vs.shape[-1] - 1)
    x0, x1 = np.take_along_axis(vs, idx - 1, 1), np.take_along_axis(vs, idx, 1)
    y0, y1 = np.take_along_axis(ts, idx - 1, 1), np.take_along_axis(ts, idx, 1)
    w = np.clip((x - x0) / np.maximum(x1 - x0, np.float32(1e-12)), 0, 1)
    return y0 + w * (y1 - y0)


def calibrate_first_chunk(virtual, exact_first, start=0, scope="head"):
    """Virtual scores [layers, kv_heads, tokens]; preserve exact anchor slice.

    No model-dtype rounding, protected-sink override, or eviction occurs here.
    Prefix positions before start stay unchanged; anchors have no future queries.
    """
    v, e = _array(virtual, "virtual", 3), _array(exact_first, "exact_first", 3)
    if v.shape[:2] != e.shape[:2] or min(v.shape) == 0 or e.shape[-1] == 0:
        raise ValueError("nonempty matching layer/head dimensions are required")
    if not isinstance(start, (int, np.integer)) or start < 0 or start + e.shape[-1] > v.shape[-1]:
        raise ValueError("invalid first-chunk bounds")
    rows = {"global": 1, "layer": v.shape[0], "head": v.shape[0] * v.shape[1]}.get(scope)
    if rows is None:
        raise ValueError("scope must be global, layer, or head")
    out, end = v.copy(), start + e.shape[-1]
    out[..., start:end] = e
    if end < v.shape[-1]:
        mapped = quantile_map(v[..., start:end].reshape(rows, -1), e.reshape(rows, -1),
                              v[..., end:].reshape(rows, -1))
        out[..., end:] = mapped.reshape(v[..., end:].shape)
    return out


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    prompt_tokens: int

    @property
    def context_tokens(self):
        return self.end - self.start

    @property
    def repeat_input_tokens(self):
        return self.context_tokens + self.prompt_tokens


def echo_chunk_layout(context_length, prefix_length, first_prompt_tokens,
                      later_prompt_tokens, suffix_tokens, chunk_size=2048,
                      prev_postfix_size=8):
    """Token-count version of upstream prepare(), including max(1, budget-P).

    Prompt counts must be measured with the actual tokenizer. Later prompt count
    excludes suffix and preceding-chunk postfix. context_length includes prefix.
    Empty contexts and invalid counts are rejected; overly small positive budgets
    intentionally return an over-budget first pass, exactly as upstream max(1,...).
    """
    args = (context_length, prefix_length, first_prompt_tokens, later_prompt_tokens,
            suffix_tokens, chunk_size, prev_postfix_size)
    if any(isinstance(a, bool) or not isinstance(a, (int, np.integer)) for a in args):
        raise ValueError("all lengths must be integers")
    if min(args) < 0 or chunk_size == 0 or context_length <= prefix_length:
        raise ValueError("positive chunk size and nonempty post-prefix context are required")
    first_size = max(1, chunk_size - first_prompt_tokens - suffix_tokens)
    start, previous_size, chunks = prefix_length, 0, []
    while start < context_length:
        first = not chunks
        size = min(first_size if first else chunk_size, context_length - start)
        p = first_prompt_tokens if first else later_prompt_tokens + min(prev_postfix_size, previous_size)
        chunks.append(Chunk(start, start + size, p + suffix_tokens))
        start, previous_size = start + size, size
    return chunks


def reconstruction_ledger(chunks, context_length):
    """Dimensionless dense causal attention pair count, NOT measured FLOPs/time.

    For each independent replay with t input tokens over n original keys, count
    n*t+t*(t+1)/2 accessible query/key pairs. Multiply by architecture/operation
    factors for FLOPs; projection/MLP token count and extra launch costs separate.
    """
    if context_length < 0:
        raise ValueError("context length must be nonnegative")
    sizes = [c.repeat_input_tokens for c in chunks]
    return {"passes": len(sizes), "reconstruction_input_tokens": sum(sizes),
            "context_anchor_tokens": sum(c.context_tokens for c in chunks),
            "dense_causal_attention_pairs": sum(context_length*t+t*(t+1)//2 for t in sizes),
            "pass_input_tokens": sizes}


def _regions(regions, n_tokens):
    a = np.asarray(regions)
    if a.ndim != 1 or a.shape[0] != n_tokens or a.dtype.kind not in "iu":
        raise ValueError("regions must be one integer ID per token")
    if a.size == 0:
        raise ValueError("regions cannot be empty")
    return a


def scale_map_oracle(virtual, exact, regions):
    """DIAGNOSTIC ONLY: exact regional marginals, virtual regional ordering.

    Full exact reconstruction is required. This does not repair within-region
    rank inversions; duplicate virtual knots retain author tie behavior. Every
    region needs at least two tokens. Does not preserve an exact first chunk
    automatically: caller must overwrite the same exact-anchor slice in all arms.
    """
    v, e = _array(virtual, "virtual", 3), _array(exact, "exact", 3)
    if v.shape != e.shape or min(v.shape) == 0:
        raise ValueError("scores must be matching nonempty [layers, heads, tokens]")
    r, out = _regions(regions, v.shape[-1]), np.empty_like(v)
    for region in np.unique(r):
        mask = r == region
        a, b = v[..., mask], e[..., mask]
        out[..., mask] = quantile_map(a.reshape(-1, mask.sum()), b.reshape(-1, mask.sum()),
                                      a.reshape(-1, mask.sum())).reshape(a.shape)
    return out


def rank_only_oracle(calibrated, exact, regions):
    """DIAGNOSTIC ONLY: exact ordering, unchanged per-layer/head/region histogram.

    Stable token-index order resolves exact ties solely for reproducibility.
    Global allocation counts remain unchanged absent cutoff ties; tied cutoff
    membership is not an identified causal effect and must be separately logged.
    """
    c, e = _array(calibrated, "calibrated", 3), _array(exact, "exact", 3)
    if c.shape != e.shape or min(c.shape) == 0:
        raise ValueError("scores must be matching nonempty [layers, heads, tokens]")
    r, out = _regions(regions, c.shape[-1]), np.empty_like(c)
    for region in np.unique(r):
        mask = r == region
        a, b = c[..., mask], e[..., mask]
        order = np.argsort(b, axis=-1, kind="stable")
        reordered = np.empty_like(a)
        np.put_along_axis(reordered, order, np.sort(a, axis=-1), axis=-1)
        out[..., mask] = reordered
    return out


def global_keep_mask(scores, eviction_ratio, n_sink=0):
    """Logical selection only: does NOT reduce allocated KV memory.

    Upstream total floor(N*r) eviction count and sink promotion; ties use stable
    flat-index ordering here, unlike torch.topk's unspecified tie behavior.
    Reject impossible protected budgets rather than silently evicting sinks.
    """
    s = _array(scores, "scores", 3).copy()
    if min(s.shape) == 0 or not np.isfinite(eviction_ratio) or not 0 <= eviction_ratio < 1:
        raise ValueError("nonempty scores and eviction ratio in [0,1) required")
    if not isinstance(n_sink, (int, np.integer)) or not 0 <= n_sink <= s.shape[-1]:
        raise ValueError("invalid n_sink")
    n_pruned = math.floor(s.size * eviction_ratio)
    if s.size - n_pruned < s.shape[0] * s.shape[1] * n_sink:
        raise ValueError("retained budget cannot preserve all sink pairs")
    s[..., :n_sink] = s.max() + np.float32(1)
    mask = np.ones(s.size, dtype=bool)
    mask[np.argsort(s.reshape(-1), kind="stable")[:n_pruned]] = False
    return mask.reshape(s.shape)


def selection_report(scores, exact, regions, eviction_ratio, n_sink=0):
    """Compare logical selected pairs with KVzip score reference, not task truth."""
    s, e = _array(scores, "scores", 3), _array(exact, "exact", 3)
    if s.shape != e.shape:
        raise ValueError("score shapes must match")
    r = _regions(regions, s.shape[-1])
    keep, reference = global_keep_mask(s, eviction_ratio, n_sink), global_keep_mask(e, eviction_ratio, n_sink)
    return {"logical_pairs_retained": int(keep.sum()), "total_pairs": int(keep.size),
            "retention_disagreement_fraction": float(np.mean(keep != reference)),
            "exact_selected_pair_recall": float(np.sum(keep & reference) / reference.sum()),
            "mean_absolute_score_error": float(np.mean(np.abs(s-e))),
            "retained_pairs_by_region": {str(x): int(keep[..., r == x].sum()) for x in np.unique(r)},
            "reference_pairs_by_region": {str(x): int(reference[..., r == x].sum()) for x in np.unique(r)},
            "physical_memory_measured": False}


def anchor_diagnostics(source, values):
    """Per-row support clipping and duplicate counts before applying a map."""
    s, v = _array(source, "source", 2), _array(values, "values", 2)
    if s.shape[0] != v.shape[0] or min(s.shape) == 0 or v.shape[1] == 0:
        raise ValueError("matching nonempty rows and values required")
    return {"below_support_fraction": np.mean(v < s.min(-1, keepdims=True), axis=-1).tolist(),
            "above_support_fraction": np.mean(v > s.max(-1, keepdims=True), axis=-1).tolist(),
            "duplicate_anchor_fraction": [1 - len(np.unique(row))/len(row) for row in s]}
