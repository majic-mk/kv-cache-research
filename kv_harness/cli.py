"""Local, dependency-free command line. There is no GPU/rental launch command."""

from __future__ import annotations

import argparse
import sys

from .common import ValidationError, canonical_json, read_json, require_keys, write_new
from .fixtures import run_cpu_fixture
from .ledger import audit_ledger
from .manifest import RunManifest
from .results import load_results
from .statistics import paired_summary


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fixture = commands.add_parser("cpu-fixture", help="Run labeled NONEXPERIMENTAL CPU primitives")
    fixture.add_argument("--repo", default=".")
    fixture.add_argument("--output", required=True, help="New directory; must not already exist")
    fixture.add_argument("--units", type=int, default=4)
    fixture.add_argument("--replicates", type=int, default=2)
    fixture.add_argument("--warmups", type=int, default=1)
    fixture.add_argument("--seed", type=int, default=7)
    manifest = commands.add_parser("manifest", help="Snapshot pins, git source and environment without running a model")
    manifest.add_argument("--repo", default=".")
    manifest.add_argument("--spec", required=True, help="JSON: evidence_kind, artifacts, config, seed, hardware")
    manifest.add_argument("--output", required=True)
    validate = commands.add_parser("validate", help="Validate manifest checksum and all JSONL records")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--results", required=True)
    analyze = commands.add_parser("analyze", help="Strictly paired, unit-clustered descriptive summary")
    analyze.add_argument("--manifest", required=True)
    analyze.add_argument("--results", required=True)
    analyze.add_argument("--baseline", required=True)
    analyze.add_argument("--candidate", required=True)
    analyze.add_argument("--metric", default="total_cost_s")
    analyze.add_argument("--direction", choices=("lower", "higher"))
    analyze.add_argument("--confidence", type=float, default=0.95)
    analyze.add_argument("--bootstrap-samples", type=int, default=10000)
    analyze.add_argument("--seed", type=int, default=0)
    analyze.add_argument("--allow-nonexperimental", action="store_true")
    analyze.add_argument("--output", help="Optional new JSON output file; existing files cannot be replaced")
    ledger = commands.add_parser("ledger", help="Read-only CSV cost ledger audit")
    ledger.add_argument("--path", default="records/cost_ledger.csv")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "cpu-fixture":
            result = run_cpu_fixture(repo=args.repo, output=args.output, units=args.units,
                replicates=args.replicates, warmups=args.warmups, seed=args.seed)
        elif args.command == "manifest":
            spec = read_json(args.spec)
            require_keys(spec, {"evidence_kind", "artifacts", "config", "seed", "hardware"}, "manifest spec")
            manifest = RunManifest.create(repo=args.repo, **spec)
            manifest.save(args.output)
            result = {"run_id": manifest.run_id, "manifest_sha256": manifest.sha256,
                      "evidence_kind": manifest.payload["evidence_kind"], "execution_performed": False}
        elif args.command in {"validate", "analyze"}:
            manifest = RunManifest.load(args.manifest)
            rows = load_results(args.results, manifest)
            if args.command == "validate":
                result = {"valid": True, "rows": len(rows), "evidence_kind": manifest.payload["evidence_kind"],
                          "manifest_sha256": manifest.sha256,
                          "note": "Schema and checksum validation does not prove model execution or scientific correctness."}
            else:
                if args.metric.startswith("metrics.") and args.direction is None:
                    raise ValidationError("Custom metrics require an explicit --direction lower|higher")
                result = paired_summary(rows, baseline=args.baseline, candidate=args.candidate,
                    metric=args.metric, direction=args.direction or "lower", confidence=args.confidence,
                    bootstrap_samples=args.bootstrap_samples, seed=args.seed,
                    allow_nonexperimental=args.allow_nonexperimental)
                if args.output:
                    write_new(args.output, canonical_json(result) + "\n")
        else:
            result = audit_ledger(args.path)
        print(canonical_json(result))
        return 0
    except (ValidationError, OSError, UnicodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
