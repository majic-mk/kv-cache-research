"""Strict, dependency-free contracts and query-blind context preparation."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import BACKEND_COMMIT


class ContractError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def exact_keys(value: dict, required: set, optional: set, where: str) -> None:
    require(isinstance(value, dict), f"{where} must be an object")
    require(required <= set(value), f"{where} missing fields: {sorted(required - set(value))}")
    require(set(value) <= required | optional, f"{where} unknown fields: {sorted(set(value) - required - optional)}")


def positive_int(value: Any, where: str, minimum: int = 1) -> None:
    require(type(value) is int and value >= minimum, f"{where} must be an integer >= {minimum}")


def hex_pin(value: Any, length: int, where: str) -> None:
    require(isinstance(value, str) and bool(re.fullmatch(f"[0-9a-f]{{{length}}}", value)),
            f"{where} must be an immutable {length}-character lowercase hex pin")


def tokens(value: Any, where: str, allow_empty: bool = False) -> None:
    require(isinstance(value, list) and (allow_empty or len(value) > 0), f"{where} must be a token-ID list")
    require(all(type(x) is int and x >= 0 for x in value), f"{where} needs nonnegative integer token IDs")


@dataclass(frozen=True)
class PreparedContext:
    context_id: str
    source_id: str
    token_ids: tuple[int, ...]
    prefix_length: int
    regions: tuple[dict, ...]
    original_tokens: int
    truncation: dict

    def fingerprint(self) -> str:
        return canonical_hash({"context_id": self.context_id, "token_ids": self.token_ids,
                               "regions": self.regions, "truncation": self.truncation})


def validate_config(config: dict) -> None:
    exact_keys(config, {"schema_version", "backend_commit", "model", "dataset", "selection",
                       "generation", "runtime", "template", "arms", "truncation"}, set(), "config")
    require(config["schema_version"] == 1, "schema_version must be 1")
    require(config["backend_commit"] == BACKEND_COMMIT, "Use the audited EchoPress backend commit, not kvpress==0.5.5")
    model = config["model"]
    exact_keys(model, {"id", "revision", "tokenizer_id", "tokenizer_revision", "dtype"}, set(), "model")
    for name in ("id", "tokenizer_id"):
        require(isinstance(model[name], str) and bool(re.fullmatch(r"[\w.-]+/[\w.-]+", model[name])), f"model.{name} must be a Hugging Face repository ID")
    for name in ("revision", "tokenizer_revision"):
        hex_pin(model[name], 40, "model." + name)
    require(model["dtype"] in ("bfloat16", "float16", "float32"), "Unsupported model dtype")
    template = config["template"]
    exact_keys(template, {"chat_template_sha256", "prefix_token_ids", "suffix_token_ids", "enable_thinking", "prompt_mode"}, set(), "template")
    require(template["prompt_mode"] == "chat_adapted", "First pilot implements chat_adapted only; official LongBench raw completion is a separate reproduction gap")
    hex_pin(template["chat_template_sha256"], 64, "template.chat_template_sha256")
    tokens(template["prefix_token_ids"], "template.prefix_token_ids", True)
    tokens(template["suffix_token_ids"], "template.suffix_token_ids")
    require(template["enable_thinking"] is False, "Author baseline extracts its template with enable_thinking=False")
    ds = config["dataset"]
    exact_keys(ds, {"source_id", "source_revision", "split", "preprocessing_revision", "preprocessing_sha256", "preprocessing_revision_scope", "contexts_file",
                    "contexts_sha256", "questions_file", "questions_sha256", "license", "evidence_kind"}, set(), "dataset")
    for name in ("source_revision", "preprocessing_revision"):
        hex_pin(ds[name], 40, "dataset." + name)
    require(ds["preprocessing_revision_scope"] == "local_origin", "preprocessing_revision records local-origin provenance, not necessarily the run checkout ancestry")
    for name in ("contexts_sha256", "questions_sha256", "preprocessing_sha256"):
        hex_pin(ds[name], 64, "dataset." + name)
    for name in ("source_id", "split", "license", "contexts_file", "questions_file"):
        require(isinstance(ds[name], str) and bool(ds[name].strip()), "dataset." + name + " is required")
    require(ds["evidence_kind"] in ("natural", "controlled_diagnostic", "fixture_only"), "Unknown dataset evidence_kind")
    selection = config["selection"]
    exact_keys(selection, {"eviction_ratios", "n_sink", "recent_tokens", "chunk_size", "scoring_backend"}, set(), "selection")
    for name in ("n_sink", "recent_tokens"):
        positive_int(selection[name], "selection." + name, 0)
    positive_int(selection["chunk_size"], "selection.chunk_size", 2)
    ratios = selection["eviction_ratios"]
    require(isinstance(ratios, list) and ratios and all(type(x) in (int, float) and math.isfinite(x) and 0 < x < 1 for x in ratios), "eviction_ratios must be nonempty and strictly between zero and one")
    require(len(set(ratios)) == len(ratios), "Duplicate eviction ratios")
    require(selection["scoring_backend"] == "torch_eager", "Pilot fixes EchoPress scoring_backend=torch_eager; Triton parity is not established")
    require(isinstance(config["arms"], list) and set(config["arms"]) == {"fullkv", "echo", "kvzip", "kvzip_echo_partition"} and len(config["arms"]) == 4,
            "All four baseline arms must occur once: fullkv, echo, kvzip, kvzip_echo_partition")
    gen = config["generation"]
    exact_keys(gen, {"max_new_tokens", "decoding"}, set(), "generation")
    positive_int(gen["max_new_tokens"], "generation.max_new_tokens")
    require(gen["decoding"] == "greedy", "Only deterministic greedy decoding is implemented")
    truncation = config["truncation"]
    exact_keys(truncation, {"policy", "max_context_tokens"}, set(), "truncation")
    require(truncation["policy"] in ("reject", "right"), "truncation.policy must be reject or right")
    positive_int(truncation["max_context_tokens"], "truncation.max_context_tokens")
    require(truncation["max_context_tokens"] > len(template["prefix_token_ids"]), "Truncation would remove the entire context body")
    runtime = config["runtime"]
    exact_keys(runtime, {"device", "attention_backend", "seed", "local_files_only", "packages_lock_file",
                         "packages_lock_sha256", "capture_scores", "deterministic_algorithms"}, set(), "runtime")
    require(runtime["device"] == "cuda:0", "Pilot uses one model and one visible CUDA device (cuda:0)")
    require(runtime["attention_backend"] == "sdpa", "All arms must use the same SDPA masking backend")
    positive_int(runtime["seed"], "runtime.seed", 0)
    require(runtime["local_files_only"] is True, "Execution never downloads model/tokenizer files; provision separately")
    require(runtime["capture_scores"] is True, "This is an instrumented diagnostic runner: score capture is required")
    require(type(runtime["deterministic_algorithms"]) is bool, "deterministic_algorithms must be boolean")
    hex_pin(runtime["packages_lock_sha256"], 64, "runtime.packages_lock_sha256")
    require(isinstance(runtime["packages_lock_file"], str) and runtime["packages_lock_file"], "packages_lock_file is required")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def verified_path(root: Path, filename: str, expected_sha256: str) -> Path:
    path = (root / filename).resolve()
    require(path.is_file(), f"Missing local input: {filename}. Prepare pinned data/assets first; no downloads occur.")
    require(sha256_file(path) == expected_sha256, f"SHA-256 mismatch: {filename}")
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            require(bool(line.strip()), f"{path.name}:{line_number}: empty lines are not accepted")
            try:
                row = json.loads(line)
            except ValueError as exc:
                raise ContractError(f"{path.name}:{line_number}: invalid JSON") from exc
            require(isinstance(row, dict), f"{path.name}:{line_number}: expected an object")
            rows.append(row)
    require(bool(rows), f"{path.name} is empty")
    return rows


def prepare_context(row: dict, config: dict) -> PreparedContext:
    # Reject question/gold fields structurally. No query or metric data reaches this function.
    exact_keys(row, {"context_id", "source_id", "context_token_ids", "regions"}, set(), "context record")
    for name in ("context_id", "source_id"):
        require(isinstance(row[name], str) and bool(row[name]), f"{name} must be nonempty")
    ids = row["context_token_ids"]
    tokens(ids, "context_token_ids")
    prefix = config["template"]["prefix_token_ids"]
    require(ids[:len(prefix)] == prefix, "Context does not start with the frozen chat prefix")
    require(len(ids) > len(prefix), "Empty post-prefix context is unsupported")
    regions = row["regions"]
    require(isinstance(regions, list) and bool(regions), "Regions must partition the context body")
    cursor = len(prefix)
    seen = set()
    for region in regions:
        exact_keys(region, {"region_id", "label", "start", "end"}, set(), "region")
        require(isinstance(region["region_id"], str) and region["region_id"] and region["region_id"] not in seen, "Region IDs must be unique and nonempty")
        require(isinstance(region["label"], str) and bool(region["label"]), "Region label required")
        positive_int(region["start"], "region.start", 0)
        positive_int(region["end"], "region.end")
        require(region["start"] == cursor and region["end"] > cursor, "Regions must be contiguous, non-overlapping, half-open token spans")
        cursor = region["end"]
        seen.add(region["region_id"])
    require(cursor == len(ids), "Regions must cover the entire original post-prefix token sequence")
    limit = config["truncation"]["max_context_tokens"]
    policy = config["truncation"]["policy"]
    require(len(ids) <= limit or policy == "right", "Context exceeds max_context_tokens under reject policy")
    kept = min(len(ids), limit)
    clipped = tuple({**r, "end": min(r["end"], kept)} for r in regions if r["start"] < kept)
    return PreparedContext(row["context_id"], row["source_id"], tuple(ids[:kept]), len(prefix), clipped, len(ids),
                           {"policy": policy, "original_tokens": len(ids), "retained_tokens": kept,
                            "removed_range": [kept, len(ids)] if kept < len(ids) else None,
                            "original_token_ids_sha256": canonical_hash(ids),
                            "retained_token_ids_sha256": canonical_hash(ids[:kept])})


def validate_questions(rows: list[dict], context_ids: set[str], suffix: list[int]) -> dict[str, list[dict]]:
    grouped = {key: [] for key in context_ids}
    ids = set()
    for row in rows:
        exact_keys(row, {"context_id", "question_id", "question_token_ids"}, {"reference_texts"}, "question record")
        require(row["context_id"] in context_ids, "Question points to an unknown context")
        require(isinstance(row["question_id"], str) and row["question_id"] and row["question_id"] not in ids, "Question IDs must be globally unique and nonempty")
        tokens(row["question_token_ids"], "question_token_ids")
        require(row["question_token_ids"][-len(suffix):] == suffix, "Question tokens must include the exact frozen chat suffix; answer prefixes unsupported")
        if "reference_texts" in row:
            require(isinstance(row["reference_texts"], list) and row["reference_texts"] and all(isinstance(x, str) for x in row["reference_texts"]), "reference_texts must be a nonempty string list")
        ids.add(row["question_id"])
        grouped[row["context_id"]].append(row)
    require(all(grouped.values()), "Every context must have at least one question")
    return grouped


def selection_budget(n_layers: int, n_heads: int, n_tokens: int, ratio: float, n_sink: int, recent: int) -> dict:
    total = n_layers * n_heads * n_tokens
    pruned = int(total * ratio)  # exactly the author's truncation, not round()
    protected_tokens = len(set(range(min(n_sink, n_tokens))) | set(range(max(0, n_tokens - recent), n_tokens)))
    protected = protected_tokens * n_layers * n_heads
    require(total - pruned >= protected, "Requested retained budget is below protected sink/recency pairs")
    return {"total_pairs": total, "pruned_pairs": pruned, "retained_pairs": total - pruned,
            "protected_pairs": protected}


def summarize_plan(config: dict, contexts: list[PreparedContext], questions: dict) -> dict:
    return {"schema_version": 1, "status": "validated_not_executed", "evidence_kind": config["dataset"]["evidence_kind"],
            "model": config["model"], "backend_commit": BACKEND_COMMIT, "config_sha256": canonical_hash(config),
            "context_count": len(contexts), "question_count": sum(map(len, questions.values())),
            "cache_builds": len(contexts) * (1 + 3 * len(config["selection"]["eviction_ratios"])),
            "context_fingerprints": [{"context_id": x.context_id, "sha256": x.fingerprint(), "truncation": x.truncation} for x in contexts],
            "limitations": ["No model or GPU execution in dry-run", "Quality-only masking backend; no memory reduction or speedup claim",
                            "Author KVzip and EchoPress use different score partitions; fourth arm controls that confound",
                            "No regional calibration candidate or oracle is deployed", "Tokenizer/asset/runtime compatibility still requires execution preflight"]}


def validate_inputs(config: dict, root: Path) -> tuple[list[PreparedContext], dict, dict]:
    ds = config["dataset"]
    context_path = verified_path(root, ds["contexts_file"], ds["contexts_sha256"])
    question_path = verified_path(root, ds["questions_file"], ds["questions_sha256"])
    runtime = config["runtime"]
    lock_path = verified_path(root, runtime["packages_lock_file"], runtime["packages_lock_sha256"])
    lock = json.loads(lock_path.read_text())
    exact_keys(lock, {"python_version", "packages", "backend_commit"}, set(), "packages lock")
    require(lock["backend_commit"] == BACKEND_COMMIT, "Environment lock backend mismatch")
    require(isinstance(lock["packages"], dict) and all(isinstance(k, str) and isinstance(v, str) and v for k, v in lock["packages"].items()), "Lock packages must map names to exact versions")
    require({"torch", "transformers", "kvpress", "numpy", "peft"} <= set(lock["packages"]), "Environment lock is missing required packages")
    require(isinstance(lock["python_version"], str) and re.fullmatch(r"3\.\d+\.\d+", lock["python_version"]), "Lock requires exact Python x.y.z")
    contexts = [prepare_context(row, config) for row in read_jsonl(context_path)]
    context_ids = {row.context_id for row in contexts}
    require(len(context_ids) == len(contexts), "Duplicate context_id")
    questions = validate_questions(read_jsonl(question_path), context_ids, config["template"]["suffix_token_ids"])
    return contexts, questions, lock
