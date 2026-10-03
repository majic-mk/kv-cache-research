"""Dependency-free validation by default; inference requires an explicit switch."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import BACKEND_COMMIT
from .contracts import ContractError, canonical_hash, load_config, sha256_file, summarize_plan, validate_inputs


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def code_manifest():
    base = Path(__file__).parent
    return {str(path.relative_to(base)): sha256_file(path) for path in sorted(base.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in (".py", ".md", ".json", ".in")}


def run(config_path: Path, output: Path | None, execute: bool):
    config = load_config(config_path)
    contexts, questions, lock = validate_inputs(config, config_path.parent)
    plan = summarize_plan(config, contexts, questions)
    if not execute:
        if output:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output / "dry_run.json", plan)
            write_json(output / "frozen_config.json", config)
        return plan
    if output is None:
        raise ContractError("--output is required for --execute")
    if config["dataset"]["evidence_kind"] == "fixture_only":
        # Create a clearly failed invocation record without ever entering real_model gates.
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "FAILED.json", {"status": "failed", "error_type": "ContractError", "reason": "fixture_only"})
        raise ContractError("Fixture-only data cannot be run as a real-model experiment")
    from kv_harness.manifest import RunManifest
    repo = Path(__file__).resolve().parents[1]
    if sha256_file(repo / "data_prep/prepare_longbench.py") != config["dataset"]["preprocessing_sha256"]:
        raise ContractError("Preprocessing source bytes differ from frozen data preparation")
    evidence_manifest = RunManifest.create(repo=repo, evidence_kind="real_model", seed=config["runtime"]["seed"], config=config,
        artifacts={"model": {"identifier": config["model"]["id"], "revision_kind": "git_commit", "revision": config["model"]["revision"]},
                   "dataset": {"identifier": config["dataset"]["source_id"], "revision_kind": "git_commit", "revision": config["dataset"]["source_revision"]},
                   "tokenized_contexts": {"identifier": "frozen-context-jsonl", "revision_kind": "sha256", "revision": config["dataset"]["contexts_sha256"]},
                   "tokenized_questions": {"identifier": "frozen-question-jsonl", "revision_kind": "sha256", "revision": config["dataset"]["questions_sha256"]}})
    # Fresh directory only. Partial failures never acquire a COMPLETE marker.
    output.mkdir(parents=True, exist_ok=False)
    evidence_manifest.save(output / "evidence_manifest.json")
    write_json(output / "frozen_config.json", config)
    write_json(output / "input_plan.json", plan)
    write_json(output / "packages.lock.json", lock)
    manifest = {"schema_version": 1, "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": canonical_hash(config), "code_sha256": code_manifest(),
        "backend_commit": BACKEND_COMMIT, "runtime_stage": "not_yet_executed",
        "claim_scope": "quality-only instrumented masking backend; no physical-memory savings or serving-speedup proof"}
    write_json(output / "manifest.json", manifest)
    try:
        from .runtime import execute as run_runtime
        run_runtime(config, contexts, questions, lock, output)
    except Exception as exc:
        # Save no credential-bearing environment variables or private cache paths.
        write_json(output / "FAILED.json", {"status": "failed", "error_type": type(exc).__name__,
                                            "completed_utc": datetime.now(timezone.utc).isoformat(),
                                            "interpretation": "Partial outputs, if any, are not a complete experiment"})
        raise
    manifest.update(status="completed", runtime_stage="executed", completed_utc=datetime.now(timezone.utc).isoformat())
    manifest["artifact_sha256"] = {str(path.relative_to(output)): sha256_file(path) for path in sorted(output.rglob("*"))
                                   if path.is_file() and path.name not in ("manifest.json", "COMPLETE.json")}
    write_json(output / "manifest.json", manifest)
    write_json(output / "COMPLETE.json", {"manifest_sha256": sha256_file(output / "manifest.json"), "status": "completed"})
    return {"status": "completed", "output": str(output)}


def freeze_environment(output: Path):
    if output.exists():
        raise ContractError("Refusing to overwrite an existing environment lock")
    packages = {re.sub(r"[-_.]+", "-", item.metadata["Name"]).lower(): item.version for item in metadata.distributions() if item.metadata["Name"]}
    if not {"torch", "transformers", "kvpress", "numpy", "peft"} <= set(packages):
        raise ContractError("Cannot freeze a GPU profile before all runtime packages are installed")
    direct = metadata.distribution("kvpress").read_text("direct_url.json")
    if not direct or json.loads(direct).get("vcs_info", {}).get("commit_id") != BACKEND_COMMIT:
        raise ContractError("kvpress must have PEP 610 metadata for the audited Git commit")
    # Do not store pip freeze direct URLs, which can contain credentials.
    lock = {"python_version": platform.python_version(), "packages": dict(sorted(packages.items())), "backend_commit": BACKEND_COMMIT}
    write_json(output, lock)
    return {"status": "environment_frozen", "sha256": sha256_file(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EchoPress quality-only pilot. Default operation is no-inference validation.")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("run")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    validate.add_argument("--execute", action="store_true", help="Explicitly authorize this invocation to perform local GPU inference; never downloads assets")
    preflight = sub.add_parser("preflight", help="Check pinned installed dependencies/CUDA/local tokenizer+config; no weights or inference")
    preflight.add_argument("--config", type=Path, required=True)
    generate = sub.add_parser("make-config")
    generate.add_argument("--repo", type=Path, default=Path("."))
    generate.add_argument("--data-manifest", type=Path, default=Path("configs/data/longbench_pilot.manifest.json"))
    generate.add_argument("--packages-lock", type=Path, required=True)
    generate.add_argument("--preprocessing-revision", required=True)
    generate.add_argument("--task", choices=("lcc", "repobench-p"), required=True)
    generate.add_argument("--split", choices=("development", "heldout"), default="development")
    generate.add_argument("--preset", choices=("smoke8k", "pilot16k"), default="smoke8k")
    generate.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze-environment")
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run(args.config.resolve(), args.output, args.execute)
        elif args.command == "freeze-environment":
            result = freeze_environment(args.output)
        elif args.command == "make-config":
            from .configure import make_config
            result = make_config(args.repo.resolve(), args.data_manifest.resolve(), args.packages_lock.resolve(),
                args.preprocessing_revision, args.task, args.split, args.preset, args.output.resolve())
        else:
            from .runtime import preflight as check_runtime
            config = load_config(args.config.resolve())
            contexts, questions, lock = validate_inputs(config, args.config.resolve().parent)
            result = check_runtime(config, contexts, questions, lock)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ContractError, OSError, ValueError) as exc:
        print(f"echo-pilot: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"echo-pilot runtime blocked: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
