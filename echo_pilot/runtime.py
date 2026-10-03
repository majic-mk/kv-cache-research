"""Optional GPU adapter for the audited author source; never imported by dry-run.

All timings include diagnostic capture overhead. Masking changes neither cache shape
nor the amount of KV payload; replay slices can even retain larger backing storage.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
import os
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import BACKEND_COMMIT
from .contracts import ContractError, canonical_hash, require, selection_budget, sha256_file


class DependencyError(RuntimeError):
    pass


def deterministic_environment(config):
    value = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if config["runtime"]["deterministic_algorithms"]:
        require(value in (":4096:8", ":16:8"),
            "Deterministic CUDA matmul requires CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before launching Python")
    return {"CUBLAS_WORKSPACE_CONFIG": value,
            "deterministic_algorithms": config["runtime"]["deterministic_algorithms"], "tf32_allowed": False}


def load_dependencies():
    try:
        import torch
        import transformers
        import kvpress
    except (ImportError, OSError) as exc:
        raise DependencyError("GPU runtime dependencies are missing or incompatible. Follow echo_pilot/INSTALL.md in a separate approved GPU environment; dry-run needs only Python. Original error type: " + type(exc).__name__) from exc
    return SimpleNamespace(torch=torch, transformers=transformers, kvpress=kvpress)


def verify_environment(deps, lock: dict) -> dict:
    require(platform.python_version() == lock["python_version"], "Python does not match the frozen packages lock")
    normalize = lambda name: re.sub(r"[-_.]+", "-", name).lower()
    installed = {normalize(item.metadata["Name"]): item.version for item in metadata.distributions() if item.metadata["Name"]}
    expected = {normalize(package): version for package, version in lock["packages"].items()}
    require(len(expected) == len(lock["packages"]), "Lock contains duplicate normalized package names")
    require(installed == expected, "Complete installed package inventory differs from the frozen lock; refreeze only before observing outcomes")
    direct = metadata.distribution("kvpress").read_text("direct_url.json")
    require(direct is not None, "kvpress must be installed from the exact Git commit (PEP 610 metadata required)")
    origin = json.loads(direct)
    require(origin.get("vcs_info", {}).get("commit_id") == BACKEND_COMMIT, "Installed kvpress Git commit differs from audited pin")
    source_root = Path(deps.kvpress.__file__).resolve().parent.parent
    audit_root = Path(__file__).resolve().parents[1] / "research/sources/echopress_39748e9"
    source_hashes = {}
    provenance = json.loads((audit_root / "provenance.json").read_text())
    for entry in provenance["files"]:
        if entry["path"].startswith("kvpress/"):
            path = source_root / entry["path"]
            require(path.is_file() and sha256_file(path) == entry["local_sha256"], "Installed source differs from audited file: " + entry["path"])
            source_hashes[entry["path"]] = entry["local_sha256"]
    for name in ("kvpress/presses/base_press.py", "kvpress/attention_patch.py", "kvpress/__init__.py", "kvpress/utils.py"):
        audit = Path(__file__).parent / "vendor_audit" / name.replace("/", "__")
        expected = sha256_file(audit)
        require(sha256_file(source_root / name) == expected, "Installed source differs from audited file: " + name)
        source_hashes[name] = expected
    require(deps.torch.cuda.is_available(), "CUDA is unavailable; no inference was attempted")
    return {"python_version": platform.python_version(), "packages": dict(sorted(installed.items())),
            "backend_commit": BACKEND_COMMIT, "verified_source_sha256": source_hashes,
            "cuda_runtime": deps.torch.version.cuda, "cudnn": deps.torch.backends.cudnn.version(),
            "device_name": deps.torch.cuda.get_device_name(0), "device_capability": list(deps.torch.cuda.get_device_capability(0))}


def cache_bytes(cache) -> dict:
    """Report tensor-view payload and distinct backing storage separately."""
    storage = {}
    payload = 0
    shapes = []
    for layer in cache.layers:
        shapes.append([list(layer.keys.shape), list(layer.values.shape)])
        for tensor in (layer.keys, layer.values):
            payload += tensor.numel() * tensor.element_size()
            backing = tensor.untyped_storage()
            storage[(str(tensor.device), backing.data_ptr())] = backing.nbytes()
    return {"kv_tensor_payload_bytes": payload, "kv_unique_storage_bytes": sum(storage.values()),
            "kv_shapes": shapes, "compacted": False}


def independent_cache(cache):
    """Deep-copy metadata and tensors: KVPress mutates cached keys while masking."""
    clone = copy.deepcopy(cache)
    for source, destination in zip(cache.layers, clone.layers, strict=True):
        for name in ("keys", "values"):
            a, b = getattr(source, name), getattr(destination, name)
            require(a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr(), "Cache clone aliases frozen KV storage")
    return clone


def clear_masks(model):
    for layer in model.model.layers:
        layer.self_attn.masked_key_indices = None


def snapshot_masks(model):
    return [None if layer.self_attn.masked_key_indices is None else
            tuple(t.detach().cpu().clone() for t in layer.self_attn.masked_key_indices)
            for layer in model.model.layers]


def restore_masks(model, masks):
    for layer, mask in zip(model.model.layers, masks, strict=True):
        layer.self_attn.masked_key_indices = None if mask is None else tuple(t.clone() for t in mask)


def memory_snapshot(torch):
    free, total = torch.cuda.mem_get_info(0)
    return {"allocated_bytes": torch.cuda.memory_allocated(0), "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "device_free_bytes": free, "device_total_bytes": total}


def synchronized_time(torch):
    torch.cuda.synchronize(0)
    return time.perf_counter()


@contextmanager
def pinned_press_tokenizer(tokenizer):
    # Author __call__ loads AutoTokenizer without revision. Scope the replacement to
    # those two modules and return the already revision-verified tokenizer only.
    proxy = SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
    echo_module = importlib.import_module("kvpress.presses.echo_press")
    kvzip_module = importlib.import_module("kvpress.presses.kvzip_press")
    with patch.object(echo_module, "AutoTokenizer", proxy), patch.object(kvzip_module, "AutoTokenizer", proxy):
        yield


def make_press(deps, arm: str, ratio: float, config: dict, capture: dict):
    torch = deps.torch
    EchoPress, KVzipPress = deps.kvpress.EchoPress, deps.kvpress.KVzipPress
    selection = config["selection"]
    base_class = EchoPress if arm == "echo" else KVzipPress

    class InstrumentedPress(base_class):
        def prepare(self, model, tokenizer, prev_postfix_size=8):
            if arm == "kvzip_echo_partition":
                pairs = EchoPress.prepare(self, model, tokenizer, prev_postfix_size)
            else:
                pairs = super().prepare(model, tokenizer, prev_postfix_size)
            require(bool(pairs), "No nonempty reconstruction chunks")
            if arm in ("echo", "kvzip_echo_partition"):
                require(pairs[0][1].shape[1] <= self.chunk_size, "First replay exceeds chunk budget because prompt overhead leaves no capacity")
            if arm == "echo" and len(pairs) > 1:
                require(pairs[0][0].shape[1] >= 2, "Author quantile mapping does not support singleton anchors")
            cursor = self.prefix_length
            capture["chunks"] = []
            for chunk, repeat in pairs:
                size, replay_length = chunk.shape[1], repeat.shape[1]
                capture["chunks"].append({"start": cursor, "end": cursor + size,
                    "context_tokens": size, "replay_input_tokens": replay_length,
                    "prompt_suffix_postfix_tokens": replay_length - size,
                    "replay_input_token_ids": repeat[0].tolist(),
                    "executed_exact_replay": arm != "echo" or cursor == self.prefix_length})
                cursor += size
            return pairs

        def _chunk_scores(self, module, q, keys, start, end, n_prompt):
            scores = super()._chunk_scores(module, q, keys, start, end, n_prompt)
            capture.setdefault("virtual_scores_float32_by_chunk", {})[f"layer{int(module.layer_idx)}:{start}:{end}"] = scores.detach().cpu().clone()
            return scores

        def _prefill_hook(self, module, args, kwargs, output):
            super()._prefill_hook(module, args, kwargs, output)
            idx = int(module.layer_idx)
            capture.setdefault("virtual_scores_author_storage", {})[str(idx)] = self.score_val[idx].detach().cpu().clone()
            capture.setdefault("first_virtual_anchor_float32", {})[str(idx)] = self._v0[idx].detach().cpu().clone()

        def _apply_per_sample_calibration(self, start, end):
            capture["first_exact_anchor_author_storage"] = self.score_val[..., start:end].detach().cpu().clone()
            return super()._apply_per_sample_calibration(start, end)

        def compress_post(self, model):
            require(bool(torch.isfinite(self.score_val).all()), "Nonfinite reconstruction scores")
            capture["scores_before_protection"] = self.score_val.detach().cpu().clone()
            n_layer, batch, heads, length = self.score_val.shape
            require(batch == 1, "Only batch size one is supported")
            capture["budget"] = selection_budget(n_layer, heads, length, self.compression_ratio,
                                                   self.n_sink, selection["recent_tokens"])
            # Default recent_tokens=0 is author-faithful; nonzero is an explicitly
            # labeled common-policy adaptation for both compressed algorithms.
            recent = selection["recent_tokens"]
            if recent:
                self.score_val[..., max(0, length - recent):] = self.score_val.amax() + 1.0
            super().compress_post(model)
            capture["scores_used_for_selection"] = self.score_val.detach().cpu().clone()

    kwargs = {"compression_ratio": ratio, "layerwise": False, "n_sink": selection["n_sink"],
              "chunk_size": selection["chunk_size"], "kvzip_plus_normalization": False}
    if arm == "echo":
        kwargs.update(exact_first_chunk=True, guard_tokens=0, score_calibration=True,
                      calibration_direction="virtual_to_exact", calibration_scope="head",
                      scoring_backend=selection["scoring_backend"])
    return InstrumentedPress(**kwargs)


def mask_report(torch, masks, scores, context, budget):
    length = len(context.token_ids)
    keep = torch.ones_like(scores, dtype=torch.bool)
    per_layer_head = []
    per_region = {r["region_id"]: 0 for r in context.regions}
    cutoff_ties = []
    removed_total = 0
    for layer_idx, mask in enumerate(masks):
        if mask is not None:
            batch, head, seq = mask
            require(len(batch) == len(set(zip(batch.tolist(), head.tolist(), seq.tolist()))), "Duplicate masked coordinates")
            require(bool(((seq >= 0) & (seq < length)).all()), "Mask extends past frozen context")
            keep[layer_idx, batch, head, seq] = False
            removed_total += len(seq)
        per_layer_head.append(keep[layer_idx, 0].sum(-1).tolist())
        removed_scores = scores[layer_idx][~keep[layer_idx]]
        if removed_scores.numel():
            cutoff = removed_scores.max()
            equal = scores[layer_idx] == cutoff
            cutoff_ties.append({"layer": layer_idx, "cutoff": float(cutoff), "equal_count": int(equal.sum()),
                                "equal_retained": int((equal & keep[layer_idx]).sum()), "equal_pruned": int((equal & ~keep[layer_idx]).sum())})
    require(removed_total == budget["pruned_pairs"], "Actual mask does not match exact author budget")
    for region in context.regions:
        per_region[region["region_id"]] = {"retained_pairs": int(keep[..., region["start"]:region["end"]].sum()),
            "retained_per_layer_head": keep[:, 0, :, region["start"]:region["end"]].sum(-1).tolist()}
    global_pruned = scores[~keep]
    global_ties = None
    if global_pruned.numel():
        cutoff = global_pruned.max()
        equal = scores == cutoff
        global_ties = {"cutoff": float(cutoff), "equal_count": int(equal.sum()),
                       "equal_retained": int((equal & keep).sum()), "equal_pruned": int((equal & ~keep).sum())}
    return {"retained_per_layer_head": per_layer_head, "retained_per_region": per_region,
            "global_cutoff_ties": global_ties,
            "retained_prefix_pairs": int(keep[..., :context.prefix_length].sum()),
            "actual_pruned_pairs": removed_total, "actual_retained_pairs": int(keep.sum()),
            "layer_cutoff_ties": cutoff_ties}, keep


def greedy_answer(torch, model, tokenizer, query_ids, cache, context_length, max_new_tokens):
    """Shared generation for every arm. Unlike upstream pipeline, stop on first EOS."""
    ids = torch.tensor([query_ids], dtype=torch.long, device=model.device)
    positions = torch.arange(context_length, context_length + ids.shape[1], device=model.device).unsqueeze(0)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    generated = []
    for step in range(max_new_tokens):
        result = model(input_ids=ids, past_key_values=cache, position_ids=positions, logits_to_keep=1)
        token = int(result.logits[0, -1].argmax().item())
        generated.append(token)
        if token in eos:
            break
        ids = torch.tensor([[token]], dtype=torch.long, device=model.device)
        positions = positions[:, -1:] + 1
    return {"generated_token_ids": generated,
            "text": str(tokenizer.decode(generated, skip_special_tokens=True)),
            "text_with_special_tokens": str(tokenizer.decode(generated, skip_special_tokens=False)),
            "stopped_on_eos": bool(generated and generated[-1] in eos),
            "generated_token_count": len(generated)}


def validate_template(tokenizer, config):
    template = config["template"]
    require(canonical_hash(tokenizer.chat_template) == template["chat_template_sha256"], "Tokenizer chat template differs from frozen SHA-256")
    if tokenizer.chat_template is None:
        prefix, suffix = "", "\n"
    else:
        dummy = "dummy context"
        separator = "\n" + "#" * len(dummy)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": dummy + separator}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pieces = rendered.split(separator)
        require(len(pieces) == 2 and pieces[0].count(dummy) == 1, "Chat-template extraction is ambiguous")
        prefix, suffix = pieces[0].split(dummy)[0], pieces[1]
    require(tokenizer.encode(prefix, add_special_tokens=False) == template["prefix_token_ids"], "Frozen prefix IDs differ from author template extraction")
    require(tokenizer.encode(suffix, add_special_tokens=False) == template["suffix_token_ids"], "Frozen suffix IDs differ from author template extraction")


def validate_model_config(model_config, contexts, questions, config, tokenizer):
    require(model_config.model_type in ("qwen3", "llama"), "First pilot supports only Qwen3/Llama full-attention causal models")
    require(not getattr(model_config, "sliding_window", None) and not getattr(model_config, "use_sliding_window", False), "Sliding-window cache layouts are outside this pilot")
    for rope in (getattr(model_config, "rope_scaling", None), getattr(model_config, "rope_parameters", None)):
        require(not rope or rope.get("rope_type", rope.get("type", "default")) == "default", "Scaled/dynamic RoPE requires a separate composition-parity validation")
    require(getattr(model_config, "_commit_hash", None) == config["model"]["revision"], "Resolved model config revision differs from immutable pin")
    max_positions = model_config.max_position_embeddings
    overhead = max(len(tokenizer.encode(p, add_special_tokens=False)) for p in (
        "\n\nRepeat the previous context exactly.", "\n\nRepeat the part of the previous context exactly, starting with")) + 8 + len(config["template"]["suffix_token_ids"])
    for context in contexts:
        n = len(context.token_ids)
        require(max(context.token_ids) < model_config.vocab_size, "Context token ID exceeds model vocabulary")
        require(n + config["selection"]["chunk_size"] + overhead <= max_positions, "Context plus reconstruction replay exceeds model positional capacity")
        for ratio in config["selection"]["eviction_ratios"]:
            selection_budget(model_config.num_hidden_layers, model_config.num_key_value_heads, n, ratio,
                             config["selection"]["n_sink"], config["selection"]["recent_tokens"])
        for query in questions[context.context_id]:
            require(max(query["question_token_ids"]) < model_config.vocab_size, "Question token ID exceeds model vocabulary")
            require(n + len(query["question_token_ids"]) + config["generation"]["max_new_tokens"] <= max_positions, "Context plus query/output exceeds positional capacity; questions are never silently truncated")


def split_capture(capture):
    """Keep tensors in their native dtype, scalars/IDs in JSON."""
    keys = {"scores_before_protection", "scores_used_for_selection", "virtual_scores_author_storage",
            "first_virtual_anchor_float32", "first_exact_anchor_author_storage", "virtual_scores_float32_by_chunk"}
    return {k: v for k, v in capture.items() if k not in keys}, {k: v for k, v in capture.items() if k in keys}


def build_cache(deps, model, tokenizer, context, config, arm, ratio):
    torch = deps.torch
    torch.cuda.reset_peak_memory_stats(0)
    builder_started = synchronized_time(torch)
    clear_masks(model)
    cache = deps.transformers.DynamicCache()
    ids = torch.tensor([context.token_ids], dtype=torch.long, device=model.device)
    input_ready = synchronized_time(torch)
    capture = {}
    forward_lengths = []
    hook = model.model.register_forward_pre_hook(lambda module, args, kwargs: forward_lengths.append(int(kwargs["input_ids"].shape[1])), with_kwargs=True)
    before = memory_snapshot(torch)
    started = synchronized_time(torch)
    try:
        press = None if arm == "fullkv" else make_press(deps, arm, ratio, config, capture)
        with pinned_press_tokenizer(tokenizer), (press(model) if press else nullcontext()):
            before_prefill = synchronized_time(torch)
            model.model(input_ids=ids, past_key_values=cache, use_cache=True)
            after_prefill = synchronized_time(torch)
        ended = synchronized_time(torch)
    finally:
        hook.remove()
    after = memory_snapshot(torch)
    masks = snapshot_masks(model)
    require(all(cache.get_seq_length(i) == len(context.token_ids) for i in range(len(model.model.layers))), "Scoring did not restore every layer to original cache length")
    dimensions = model.config
    expected_bytes = 2 * dimensions.num_hidden_layers * dimensions.num_key_value_heads * len(context.token_ids) * model.model.layers[0].self_attn.head_dim * next(model.parameters()).element_size()
    physical = cache_bytes(cache)
    require(physical["kv_tensor_payload_bytes"] == expected_bytes, "Unexpected KV dtype/shape/layout")
    metadata_capture, tensor_capture = split_capture(capture)
    if arm != "fullkv":
        allocation, keep = mask_report(torch, masks, tensor_capture["scores_used_for_selection"], context, capture["budget"])
        require(bool(keep[..., :config["selection"]["n_sink"]].all()), "Sink token was evicted")
        recent = config["selection"]["recent_tokens"]
        require(not recent or bool(keep[..., -recent:].all()), "Protected recent token was evicted")
        tensor_capture["logical_keep_mask"] = keep
    else:
        require(all(mask is None for mask in masks), "Full KV accidentally inherited a compressed mask")
        total = dimensions.num_hidden_layers * dimensions.num_key_value_heads * len(context.token_ids)
        allocation = {"actual_pruned_pairs": 0, "actual_retained_pairs": total}
    tensor_capture["masked_key_indices"] = masks
    replays = [c for c in metadata_capture.get("chunks", []) if c["executed_exact_replay"]]
    require(forward_lengths == [len(context.token_ids)] + [c["replay_input_tokens"] for c in replays], "Observed forward calls differ from declared replay plan")
    report_finished = synchronized_time(torch)
    stats = {"instrumented_prefill_scoring_interval_seconds": ended - started,
        "input_setup_including_h2d_seconds": input_ready - builder_started,
        "post_scoring_capture_report_seconds": report_finished - ended,
        "total_builder_wall_seconds_excluding_disk_io": report_finished - builder_started,
        "instrumented_prefill_forward_seconds": after_prefill - before_prefill,
        "instrumented_post_prefill_seconds": ended - after_prefill,
        "timing_scope": "Separate CUDA-synchronized input setup, press/prefill/scoring, post-score reporting intervals; total builder excludes disk I/O; cold-run, no speedup or amortized-cost claim",
        "memory_before_build": before, "memory_after_build": after, **physical,
        "physical_kv_payload_reduction_bytes": 0, "physical_memory_savings_demonstrated": False,
        "diagnostic_tensor_cpu_bytes": sum_tensor_bytes(tensor_capture),
        "forward_input_lengths": forward_lengths, "exact_replay_passes": len(replays),
        "exact_replay_input_tokens": sum(c["replay_input_tokens"] for c in replays),
        "replay_dense_causal_attention_pairs_proxy": sum(len(context.token_ids) * c["replay_input_tokens"] + c["replay_input_tokens"] * (c["replay_input_tokens"] + 1) // 2 for c in replays),
        "metadata": metadata_capture, "allocation": allocation}
    return cache, masks, tensor_capture, stats


def sum_tensor_bytes(value):
    if hasattr(value, "numel"):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(sum_tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(sum_tensor_bytes(v) for v in value)
    return 0


def preflight(config, contexts, questions, lock):
    deterministic_settings = deterministic_environment(config)
    deps = load_dependencies()
    environment = verify_environment(deps, lock)
    spec = config["model"]
    tokenizer = deps.transformers.AutoTokenizer.from_pretrained(spec["tokenizer_id"], revision=spec["tokenizer_revision"], local_files_only=True, trust_remote_code=False)
    validate_template(tokenizer, config)
    model_config = deps.transformers.AutoConfig.from_pretrained(spec["id"], revision=spec["revision"], local_files_only=True, trust_remote_code=False)
    validate_model_config(model_config, contexts, questions, config, tokenizer)
    return {"status": "runtime_preflight_passed_no_weights_loaded_no_inference", "environment": environment,
            "deterministic_settings": deterministic_settings,
            "max_position_embeddings": model_config.max_position_embeddings,
            "warning": "Checkpoint completeness, cache hooks, CUDA numerical parity and actual peak memory remain untested"}


def execute(config, contexts, questions, lock, output: Path):
    require(config["dataset"]["evidence_kind"] != "fixture_only", "Fixture-only data cannot be run as a real-model experiment")
    deterministic_settings = deterministic_environment(config)
    deps = load_dependencies()
    torch, hf = deps.torch, deps.transformers
    environment = verify_environment(deps, lock)
    environment["deterministic_settings"] = deterministic_settings
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["runtime"]["seed"])
    torch.cuda.manual_seed_all(config["runtime"]["seed"])
    torch.use_deterministic_algorithms(config["runtime"]["deterministic_algorithms"])
    spec = config["model"]
    tokenizer = hf.AutoTokenizer.from_pretrained(spec["tokenizer_id"], revision=spec["tokenizer_revision"], local_files_only=True, trust_remote_code=False)
    validate_template(tokenizer, config)
    model_config = hf.AutoConfig.from_pretrained(spec["id"], revision=spec["revision"], local_files_only=True, trust_remote_code=False)
    validate_model_config(model_config, contexts, questions, config, tokenizer)
    write_json(output / "environment.json", environment)
    # Record real resolved config, including RoPE and maximum positions, before load.
    resolved = model_config.to_dict()
    resolved["_name_or_path"] = spec["id"]
    write_json(output / "resolved_model_config.json", resolved)
    torch.cuda.reset_peak_memory_stats(0)
    load_started = synchronized_time(torch)
    model = hf.AutoModelForCausalLM.from_pretrained(spec["id"], revision=spec["revision"], config=model_config,
        torch_dtype=getattr(torch, spec["dtype"]), attn_implementation="sdpa", local_files_only=True, trust_remote_code=False)
    model.to(config["runtime"]["device"])
    model.eval()
    load_ended = synchronized_time(torch)
    write_json(output / "model_load.json", {"wall_seconds": load_ended - load_started, "device_memory": memory_snapshot(torch),
        "model_parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "note": "CPU RAM and external process GPU allocations are not separately attributed"})
    require(model.config._attn_implementation == "sdpa", "Loaded model attention backend is not SDPA")
    try:
        with torch.inference_mode(), (output / "results.jsonl").open("x") as results:
            for ci, context in enumerate(contexts):
                for arm in config["arms"]:
                    ratios = [0.0] if arm == "fullkv" else config["selection"]["eviction_ratios"]
                    for ri, ratio in enumerate(ratios):
                        build_id = f"c{ci:05d}-{arm}-r{ri:02d}"
                        arm_started = synchronized_time(torch)
                        folder = output / build_id
                        folder.mkdir()
                        base_cache, masks, capture, stats = build_cache(deps, model, tokenizer, context, config, arm, ratio)
                        tensor_write_started = time.perf_counter()
                        torch.save(capture, folder / "scores_masks.pt")
                        stats["tensor_file_serialization_wall_seconds"] = time.perf_counter() - tensor_write_started
                        stats["serialization_note"] = "OS-buffered writes; JSON/result I/O included only in arm total; no fsync/durability timing"
                        write_json(folder / "build.json", {"context_id": context.context_id, "context_fingerprint": context.fingerprint(),
                            "arm": arm, "eviction_ratio": ratio, "common_recent_tokens": config["selection"]["recent_tokens"],
                            "quality_backend": "author-kvpress-sdpa-fake-key-masking", "truncation": context.truncation,
                            "regions": context.regions, **stats})
                        del capture
                        for query in questions[context.context_id]:
                            torch.cuda.reset_peak_memory_stats(0)
                            clone_started = synchronized_time(torch)
                            query_cache = independent_cache(base_cache)
                            restore_masks(model, masks)
                            clone_ended = synchronized_time(torch)
                            answer = greedy_answer(torch, model, tokenizer, query["question_token_ids"], query_cache,
                                                   len(context.token_ids), config["generation"]["max_new_tokens"])
                            query_ended = synchronized_time(torch)
                            require(all(base_cache.get_seq_length(i) == len(context.token_ids) for i in range(len(base_cache.layers))), "Frozen cache metadata changed")
                            row = {"build_id": build_id, "context_id": context.context_id, "question_id": query["question_id"],
                                "arm": arm, "eviction_ratio": ratio, "evidence_kind": config["dataset"]["evidence_kind"],
                                "query_token_ids_sha256": canonical_hash(query["question_token_ids"]),
                                "cache_clone_seconds": clone_ended - clone_started, "query_generation_seconds": query_ended - clone_ended,
                                "query_peak_memory": memory_snapshot(torch), "query_cache": cache_bytes(query_cache), **answer}
                            if "reference_texts" in query:
                                row["raw_string_exact_match"] = answer["text"] in query["reference_texts"]
                                row["metric_note"] = "Raw exact match only, not official dataset quality metric"
                            results.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                            results.flush()
                            del query_cache
                        del base_cache, masks
                        clear_masks(model)
                        arm_finished = synchronized_time(torch)
                        write_json(folder / "arm_timing.json", {"arm_wall_seconds": arm_finished - arm_started,
                            "scope": "Folder creation, total builder, tensor/build JSON writes, all query clones/generation, result writes/flush, cleanup; excludes this summary write; no fsync",
                            "speedup_or_amortized_cost_claim": False})
    finally:
        clear_masks(model)
        del model
    return environment


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
