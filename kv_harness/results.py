"""Strict JSONL measurements and explicit, comparable cost accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .common import (ValidationError, canonical_json, choice, digest, finite_number, integer,
                     is_hex, nonempty_string, parse_json, require_keys, write_new)
from .manifest import EVIDENCE_KINDS, RunManifest

PHASE_COSTS = ("queue_wait_s", "preprocessing_s", "scoring_s", "host_copy_s", "host_to_device_copy_s",
               "device_to_host_copy_s", "cache_read_s", "cache_write_s", "recompute_s",
               "prefill_s", "decode_s", "postprocessing_s", "synchronization_s", "other_s")
SHARED_COSTS = ("preprocessing_s", "scoring_s", "copies_s", "other_s")
MEMORY_FIELDS = ("logical_kept_tokens", "logical_kv_bytes", "physical_device_kv_allocated_bytes",
                 "physical_kv_stored_bytes", "peak_device_allocated_bytes", "peak_device_reserved_bytes")
RESULT_KEYS = {"schema_version", "run_id", "manifest_sha256", "evidence_kind", "phase",
               "method", "unit_id", "cluster_id", "replicate", "order_index", "workload", "workload_sha256",
               "status", "error", "output_sha256", "timing", "shared_setup", "metrics",
               "resources", "execution"}


def zero_timing() -> dict[str, Any]:
    return {"accounting_mode": "serial_exclusive", "end_to_end_s": 0.0,
            **{name: 0.0 for name in PHASE_COSTS}}


def zero_setup() -> dict[str, Any]:
    return {"group_id": "none", "amortization_count": 1, **{name: 0.0 for name in SHARED_COSTS}}


def unmeasured_resources() -> dict[str, Any]:
    return {**{name: None for name in MEMORY_FIELDS}, "representation": "unknown",
            "measurement_method": "not_measured"}


def total_cost_s(row: dict[str, Any]) -> float:
    """Request wall time plus attributed shared work; never just GPU kernel time."""
    setup = row["shared_setup"]
    return row["timing"]["end_to_end_s"] + sum(setup[name] for name in SHARED_COSTS) / setup["amortization_count"]


def validate_result(row: Any, manifest: RunManifest | None = None) -> None:
    require_keys(row, RESULT_KEYS, "result")
    if type(row["schema_version"]) is not int or row["schema_version"] != 1:
        raise ValidationError("Unsupported result schema_version")
    for name in ("run_id", "method", "unit_id", "cluster_id"):
        nonempty_string(row[name], name)
    for name in ("manifest_sha256", "workload_sha256"):
        if not is_hex(row[name]):
            raise ValidationError(f"Invalid {name}")
    choice(row["evidence_kind"], EVIDENCE_KINDS, "evidence_kind")
    choice(row["phase"], {"warmup", "measurement"}, "phase")
    for name in ("replicate", "order_index"):
        integer(row[name], name)
    if not isinstance(row["workload"], dict) or not row["workload"]:
        raise ValidationError("workload must be a nonempty JSON object defining matched inputs and load")
    if row["workload_sha256"] != digest(row["workload"]):
        raise ValidationError("Workload checksum mismatch")
    choice(row["status"], {"ok", "error"}, "status")
    if row["status"] == "ok":
        if row["error"] is not None or not is_hex(row["output_sha256"]):
            raise ValidationError("Successful rows require output SHA-256 and null error")
    else:
        nonempty_string(row["error"], "error")
        if row["output_sha256"] is not None:
            raise ValidationError("Failed rows must not claim an output checksum")
    timing = row["timing"]
    require_keys(timing, {*PHASE_COSTS, "end_to_end_s", "accounting_mode"}, "timing")
    choice(timing["accounting_mode"], {"serial_exclusive", "overlapping"}, "timing.accounting_mode")
    elapsed = finite_number(timing["end_to_end_s"], "end_to_end_s", minimum=0)
    tolerance = max(1e-9, elapsed * 1e-6)
    for name in PHASE_COSTS:
        value = finite_number(timing[name], f"timing.{name}", minimum=0)
        if value > elapsed + tolerance:
            raise ValidationError(f"timing.{name} exceeds end-to-end wall time")
    if timing["accounting_mode"] == "serial_exclusive" and sum(timing[name] for name in PHASE_COSTS) > elapsed + tolerance:
        raise ValidationError("Exclusive phases exceed end-to-end wall time; double-counted work?")
    setup = row["shared_setup"]
    require_keys(setup, {*SHARED_COSTS, "group_id", "amortization_count"}, "shared_setup")
    nonempty_string(setup["group_id"], "shared_setup.group_id")
    integer(setup["amortization_count"], "shared_setup.amortization_count", minimum=1)
    for name in SHARED_COSTS:
        finite_number(setup[name], f"shared_setup.{name}", minimum=0)
    if setup["group_id"] == "none" and any(setup[name] for name in SHARED_COSTS):
        raise ValidationError("Nonzero shared costs need a named setup group")
    if not isinstance(row["metrics"], dict):
        raise ValidationError("metrics must be an object")
    for name, value in row["metrics"].items():
        nonempty_string(name, "metric name")
        finite_number(value, f"metrics.{name}")
    require_keys(row["resources"], {*MEMORY_FIELDS, "representation", "measurement_method"}, "resources")
    for name in MEMORY_FIELDS:
        value = row["resources"][name]
        if value is not None:
            integer(value, f"resources.{name}")
    choice(row["resources"]["representation"], {"unknown", "full_dense", "masked_dense", "physically_compacted"}, "resources.representation")
    nonempty_string(row["resources"]["measurement_method"], "resources.measurement_method")
    if any(row["resources"][name] is not None for name in MEMORY_FIELDS) and row["resources"]["measurement_method"] == "not_measured":
        raise ValidationError("Reported memory/token budgets need a measurement method")
    require_keys(row["execution"], {"device_type", "backend", "synchronized"}, "execution")
    execution = row["execution"]
    choice(execution["device_type"], {"cpu", "gpu", "simulated"}, "execution.device_type")
    if not isinstance(execution["synchronized"], bool):
        raise ValidationError("Invalid execution device or synchronization flag")
    nonempty_string(execution["backend"], "execution.backend")
    if row["evidence_kind"] == "cpu_fixture" and execution["device_type"] != "cpu":
        raise ValidationError("cpu_fixture rows must use the CPU")
    if row["evidence_kind"] == "simulation" and execution["device_type"] != "simulated":
        raise ValidationError("simulation rows require a simulated device label")
    if row["evidence_kind"] == "real_model":
        if execution["device_type"] == "simulated":
            raise ValidationError("Simulated execution cannot be real_model")
        if execution["device_type"] == "gpu" and not execution["synchronized"]:
            raise ValidationError("GPU wall-time records must declare synchronized timing boundaries")
    if manifest is not None:
        expected = manifest.payload
        if row["run_id"] != expected["run_id"] or row["manifest_sha256"] != manifest.sha256:
            raise ValidationError("Result does not match the supplied immutable manifest")
        if row["evidence_kind"] != expected["evidence_kind"]:
            raise ValidationError("Result evidence_kind differs from manifest")
    canonical_json(row)


def validate_records(rows: Iterable[dict[str, Any]], manifest: RunManifest | None = None) -> list[dict[str, Any]]:
    rows = list(rows)
    if not rows:
        raise ValidationError("No result records")
    seen: set[tuple[Any, ...]] = set()
    identity: tuple[str, str, str] | None = None
    setup_groups: dict[tuple[str, str, str], tuple[str, int, int]] = {}
    unit_clusters: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        try:
            validate_result(row, manifest)
            current = (row["run_id"], row["manifest_sha256"], row["evidence_kind"])
            if identity is not None and identity != current:
                raise ValidationError("Mixed manifests, runs or evidence kinds are forbidden in one result file")
            identity = current
            key = (row["phase"], row["method"], row["unit_id"], row["replicate"])
            if key in seen:
                raise ValidationError(f"Duplicate observation {key}")
            seen.add(key)
            if row["unit_id"] in unit_clusters and unit_clusters[row["unit_id"]] != row["cluster_id"]:
                raise ValidationError("An observed unit cannot move between independent clusters")
            unit_clusters[row["unit_id"]] = row["cluster_id"]
            if row["shared_setup"]["group_id"] != "none":
                group_key = (row["phase"], row["method"], row["shared_setup"]["group_id"])
                signature = canonical_json(row["shared_setup"])
                previous = setup_groups.get(group_key)
                if previous is not None and previous[0] != signature:
                    raise ValidationError("Shared setup group has inconsistent costs or denominator")
                setup_groups[group_key] = (signature, (previous[1] if previous else 0) + 1,
                                           row["shared_setup"]["amortization_count"])
        except (ValidationError, TypeError) as exc:
            raise ValidationError(f"Record {index}: {exc}") from exc
    for group, (_, observed_count, denominator) in setup_groups.items():
        if observed_count != denominator:
            raise ValidationError(f"Shared setup group {group} declares {denominator} reuses, but contains {observed_count} observations")
    return rows


def load_results(path: str | Path, manifest: RunManifest | None = None) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValidationError(f"Line {line_number}: blank JSONL records are forbidden")
            try:
                rows.append(parse_json(line))
            except ValidationError as exc:
                raise ValidationError(f"Line {line_number}: {exc}") from exc
    return validate_records(rows, manifest)


def save_results(path: str | Path, rows: Iterable[dict[str, Any]], manifest: RunManifest) -> Path:
    validated = validate_records(rows, manifest)
    return write_new(path, "".join(canonical_json(row) + "\n" for row in validated))
