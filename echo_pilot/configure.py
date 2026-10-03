"""Create a frozen runnable config from prepared real data and a real environment lock."""
import json
import os
import subprocess
from pathlib import Path

from . import BACKEND_COMMIT
from .contracts import hex_pin, require, sha256_file, validate_config, validate_inputs


MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


def make_config(repo: Path, manifest_path: Path, packages_lock: Path, preprocessing_revision: str,
                task: str, split: str, preset: str, output: Path) -> dict:
    require(task in ("lcc", "repobench-p"), "Unknown task")
    require(split in ("development", "heldout"), "Choose exactly one split; combined tuning/holdout inputs are prohibited")
    require(preset in ("smoke8k", "pilot16k"), "Choose smoke8k or pilot16k")
    require(preset != "smoke8k" or split == "development", "Smoke uses development contexts only")
    require(not output.exists(), "Refusing to overwrite frozen config")
    hex_pin(preprocessing_revision, 40, "preprocessing_revision")
    manifest = json.loads(manifest_path.read_text())
    # Byte identity survives source-only transfer to a different Git history.
    # This commit is local-origin provenance; the run separately pins its clean HEAD.
    require(sha256_file(repo / "data_prep/prepare_longbench.py") == manifest["preprocessor_sha256"],
            "Current preprocessor does not match the prepared-data manifest")
    require(manifest["tokenizer_revision"] == MODEL_REVISION, "Prepared tokenizer revision differs from pilot model tokenizer pin")
    artifacts = manifest["presets"][preset]["artifacts"].get(task + "." + split)
    require(artifacts is not None and artifacts["contexts"]["rows"] > 0, "No prepared records for this task/split/preset")
    root = output.resolve().parent
    source_path = lambda record: os.path.relpath((repo / record["path"]).resolve(), root)
    config = {"schema_version": 1, "backend_commit": BACKEND_COMMIT,
        "model": {"id": "Qwen/Qwen3-8B", "revision": MODEL_REVISION, "tokenizer_id": "Qwen/Qwen3-8B", "tokenizer_revision": MODEL_REVISION, "dtype": "bfloat16"},
        "dataset": {"source_id": "zai-org/LongBench/" + task, "source_revision": manifest["source_revision"],
                    "preprocessing_revision": preprocessing_revision, "preprocessing_sha256": manifest["preprocessor_sha256"],
                    "preprocessing_revision_scope": "local_origin", "split": split,
                    "contexts_file": source_path(artifacts["contexts"]), "contexts_sha256": artifacts["contexts"]["sha256"],
                    "questions_file": source_path(artifacts["questions"]), "questions_sha256": artifacts["questions"]["sha256"],
                    "license": "Private local research; redistribution clearance not asserted; see configs/data/longbench_sources.lock.json", "evidence_kind": "natural"},
        "template": {**manifest["template"], "prompt_mode": "chat_adapted"},
        "selection": {"eviction_ratios": [0.5] if preset == "smoke8k" else [0.5, 0.75, 0.9], "n_sink": 4, "recent_tokens": 0,
                      "chunk_size": 2048, "scoring_backend": "torch_eager"},
        "generation": {"max_new_tokens": 64, "decoding": "greedy"},
        "runtime": {"device": "cuda:0", "attention_backend": "sdpa", "seed": 20261003,
                    "local_files_only": True, "packages_lock_file": os.path.relpath(packages_lock.resolve(), root),
                    "packages_lock_sha256": sha256_file(packages_lock), "capture_scores": True, "deterministic_algorithms": True},
        "arms": ["fullkv", "echo", "kvzip", "kvzip_echo_partition"],
        "truncation": {"policy": "reject", "max_context_tokens": 8192 if preset == "smoke8k" else 16384}}
    validate_config(config)
    validate_inputs(config, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return {"status": "config_frozen_not_executed", "config_sha256": sha256_file(output),
            "contexts": artifacts["contexts"]["rows"], "preset": preset, "split": split,
            "data_manifest_sha256": sha256_file(manifest_path)}
