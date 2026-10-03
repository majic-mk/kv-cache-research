"""Actual CPU primitive timings for plumbing tests; NEVER model experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

from .common import ValidationError, canonical_json, digest, integer, sha256_bytes, write_new
from .manifest import RunManifest
from .results import save_results, unmeasured_resources, zero_setup
from .timing import SerialTimer


def _run_primitive(workload: dict[str, Any]) -> tuple[dict[str, Any], str, float]:
    with SerialTimer() as timer:
        with timer.section("preprocessing_s"):
            data = json.dumps(workload, sort_keys=True).encode("utf-8")
        with timer.section("host_copy_s"):
            copied = bytes(bytearray(data))
        with timer.section("scoring_s"):
            checksum = hashlib.sha256(copied).hexdigest()
            score = sum(copied) / max(1, len(copied))
    return timer.snapshot(), checksum, score


def run_cpu_fixture(*, repo: str | Path, output: str | Path, units: int = 4,
                    replicates: int = 2, warmups: int = 1, seed: int = 7) -> dict[str, Any]:
    integer(units, "units", minimum=1)
    integer(replicates, "replicates", minimum=1)
    integer(warmups, "warmups")
    integer(seed, "seed")
    if units * replicates > 10000 or warmups > 100:
        raise ValidationError("Fixture limit: 10,000 unit-replicates and 100 warmups")
    output = Path(output)
    # Reserve one new directory so a failed or concurrent run can never overwrite evidence.
    output.mkdir(parents=True, exist_ok=False)
    model_description = b"NONEXPERIMENTAL sha256 and byte-copy CPU primitive v1"
    dataset_description = canonical_json({"fixture": "generated-identical-workloads-v1", "units": units,
                                          "replicates": replicates, "seed": seed}).encode()
    manifest = RunManifest.create(repo=repo, evidence_kind="cpu_fixture", seed=seed,
        artifacts={"model": {"identifier": "fixture:no-model-sha256-primitive",
                              "revision_kind": "sha256", "revision": sha256_bytes(model_description)},
                   "dataset": {"identifier": "fixture:generated-in-memory",
                                "revision_kind": "sha256", "revision": sha256_bytes(dataset_description)}},
        config={"purpose": "NONEXPERIMENTAL plumbing check", "units": units, "replicates": replicates,
                "warmups": warmups, "methods": ["fixture_baseline", "fixture_candidate"],
                "method_semantics": "Both execute exactly the same CPU primitive. No KV policy."},
        hardware={"device": "cpu", "accelerator": "none_used"})
    manifest.save(output / "manifest.json")
    rng = random.Random(seed)
    rows = []
    schedule = [("warmup", f"warmup-{index:04d}", 0) for index in range(warmups)]
    schedule += [("measurement", f"unit-{unit:04d}", replicate)
                 for unit in range(units) for replicate in range(replicates)]
    for phase, unit_id, replicate in schedule:
        workload = {"fixture": "generated-identical-workloads-v1", "unit": unit_id,
                    "seed": seed, "load": "single_cpu_operation", "payload": unit_id * 16}
        methods = ["fixture_baseline", "fixture_candidate"]
        rng.shuffle(methods)
        for order, method in enumerate(methods):
            timing, checksum, score = _run_primitive(workload)
            rows.append({"schema_version": 1, "run_id": manifest.run_id, "manifest_sha256": manifest.sha256,
                "evidence_kind": "cpu_fixture", "phase": phase, "method": method,
                "unit_id": unit_id, "cluster_id": unit_id, "replicate": replicate, "order_index": order,
                "workload": workload, "workload_sha256": digest(workload), "status": "ok", "error": None,
                "output_sha256": checksum, "timing": timing, "shared_setup": zero_setup(),
                "metrics": {"fixture_byte_mean": score},
                "resources": unmeasured_resources(),
                "execution": {"device_type": "cpu", "backend": "python-stdlib-fixture", "synchronized": True}})
    save_results(output / "results.jsonl", rows, manifest)
    completion = {"status": "complete", "evidence_kind": "cpu_fixture", "experimental_evidence": False,
                  "run_id": manifest.run_id, "manifest_sha256": manifest.sha256, "rows": len(rows),
                  "results_file_sha256": sha256_bytes((output / "results.jsonl").read_bytes()),
                  "warning": "No model, dataset download, GPU, paid API, or KV policy was used."}
    write_new(output / "COMPLETE.json", canonical_json(completion) + "\n")
    return completion
