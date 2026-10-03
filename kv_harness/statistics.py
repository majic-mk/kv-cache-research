"""Paired, cluster-level percentile bootstrap; no independence claim is automatic."""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Any

from .common import ValidationError, choice, digest, finite_number, integer, nonempty_string
from .results import total_cost_s, validate_records


def _quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _mean(values: Any) -> float:
    values = list(values)
    # Scaling before summation avoids overflow when a representable mean exists.
    try:
        return finite_number(math.fsum(value / len(values) for value in values), "aggregate mean")
    except OverflowError as exc:
        raise ValidationError("Numerical overflow in aggregate mean") from exc


def _value(row: dict[str, Any], metric: str) -> float:
    if metric == "total_cost_s":
        value = total_cost_s(row)
    elif metric == "end_to_end_s":
        value = row["timing"][metric]
    elif metric.startswith("metrics.") and metric[8:] in row["metrics"]:
        value = row["metrics"][metric[8:]]
    else:
        raise ValidationError(f"Metric {metric!r} is absent or unsupported")
    return finite_number(value, metric)


def paired_summary(rows: list[dict[str, Any]], *, baseline: str, candidate: str,
                   metric: str = "total_cost_s", direction: str = "lower",
                   confidence: float = 0.95, bootstrap_samples: int = 10000,
                   seed: int = 0, allow_nonexperimental: bool = False) -> dict[str, Any]:
    rows = validate_records(rows)
    nonempty_string(baseline, "baseline")
    nonempty_string(candidate, "candidate")
    nonempty_string(metric, "metric")
    if baseline == candidate:
        raise ValidationError("Baseline and candidate must differ")
    choice(direction, {"lower", "higher"}, "direction")
    finite_number(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValidationError("confidence must be between 0 and 1")
    integer(bootstrap_samples, "bootstrap_samples", minimum=100)
    if bootstrap_samples > 1_000_000:
        raise ValidationError("bootstrap_samples exceeds safety limit of 1,000,000")
    integer(seed, "seed")
    evidence_kind = rows[0]["evidence_kind"]
    if evidence_kind != "real_model" and not allow_nonexperimental:
        raise ValidationError("Nonexperimental evidence requires explicit allow_nonexperimental=True")
    selected = [row for row in rows if row["phase"] == "measurement" and row["method"] in {baseline, candidate}]
    if any(row["status"] != "ok" for row in selected):
        raise ValidationError("Selected methods have failed trials; refusing success-only survivor analysis")
    by_method: dict[str, dict[tuple[str, int], dict[str, Any]]] = {baseline: {}, candidate: {}}
    for row in selected:
        by_method[row["method"]][(row["unit_id"], row["replicate"])] = row
    baseline_keys, candidate_keys = set(by_method[baseline]), set(by_method[candidate])
    if not baseline_keys or not candidate_keys:
        raise ValidationError("Both methods need measured observations")
    if baseline_keys != candidate_keys:
        raise ValidationError(f"Incomplete pairing: missing candidate={sorted(baseline_keys - candidate_keys)}, "
                              f"missing baseline={sorted(candidate_keys - baseline_keys)}")
    units: dict[str, list[tuple[float, float]]] = defaultdict(list)
    unit_cluster_ids: dict[str, str] = {}
    candidate_first = 0
    for key in sorted(baseline_keys):
        left, right = by_method[baseline][key], by_method[candidate][key]
        if left["workload_sha256"] != right["workload_sha256"]:
            raise ValidationError(f"Mismatched inputs or load for pair {key}")
        if left["timing"]["accounting_mode"] != right["timing"]["accounting_mode"]:
            raise ValidationError(f"Mismatched timing accounting for pair {key}")
        if left["order_index"] == right["order_index"]:
            raise ValidationError(f"Methods share an execution order index for pair {key}")
        if left["execution"] != right["execution"]:
            raise ValidationError(f"Mismatched backend/device/synchronization for pair {key}")
        candidate_first += right["order_index"] < left["order_index"]
        units[key[0]].append((_value(left, metric), _value(right, metric)))
        unit_cluster_ids[key[0]] = left["cluster_id"]
    clusters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for unit_id, values in sorted(units.items()):
        clusters[unit_cluster_ids[unit_id]].append((_mean(item[0] for item in values),
                                                   _mean(item[1] for item in values)))
    cluster_values = [(_mean(item[0] for item in values), _mean(item[1] for item in values))
                      for _, values in sorted(clusters.items())]
    deltas = [finite_number(right - left, "paired delta") for left, right in cluster_values]
    baseline_mean = _mean(left for left, _ in cluster_values)
    candidate_mean = _mean(right for _, right in cluster_values)
    delta_mean = _mean(deltas)
    count = len(deltas)
    interval: list[float] | None = None
    if count >= 2:
        rng = random.Random(seed)
        draws = sorted(_mean(deltas[rng.randrange(count)] for _ in range(count))
                       for _ in range(bootstrap_samples))
        tail = (1 - confidence) / 2
        interval = [_quantile(draws, tail), _quantile(draws, 1 - tail)]
    warnings = ["Cluster IDs must identify independent contexts/traces; questions sharing a cache must share a cluster ID.",
                "Percentile bootstrap is descriptive; no multiple-comparison correction or causal claim is supplied."]
    if evidence_kind != "real_model":
        warnings.insert(0, "NONEXPERIMENTAL: CPU fixtures/simulation are not model or GPU validation.")
    if count < 10:
        warnings.append("Fewer than 10 independent clusters: interval stability is limited.")
    if count == 1:
        warnings.append("One independent cluster: confidence interval withheld.")
    if candidate_first in {0, len(baseline_keys)}:
        warnings.append("One method always ran first; order effects may confound timing.")
    if metric == "end_to_end_s" and any(total_cost_s(row) > row["timing"]["end_to_end_s"] for row in selected):
        warnings.append("This metric excludes nonzero shared setup; inspect total_cost_s before claiming a speedup.")
    relative_delta = None if baseline_mean == 0 else finite_number(
        delta_mean / abs(baseline_mean) * 100, "relative delta percent")
    return {"schema_version": 1, "run_id": rows[0]["run_id"],
            "records_sha256": digest(sorted(rows, key=lambda row: (row["phase"], row["method"], row["unit_id"], row["replicate"]))),
            "manifest_sha256": rows[0]["manifest_sha256"], "evidence_kind": evidence_kind,
            "experimental_evidence": evidence_kind == "real_model",
            "baseline": baseline, "candidate": candidate, "metric": metric, "direction": direction,
            "paired_observations": len(baseline_keys), "paired_units": len(units), "independent_clusters": count,
            "replicates_per_unit": {unit: len(values) for unit, values in sorted(units.items())},
            "units_per_cluster": {cluster: len(values) for cluster, values in sorted(clusters.items())},
            "weighting": "equal cluster weight; mean over units per cluster, then matched replicates per unit",
            "baseline_mean": baseline_mean, "candidate_mean": candidate_mean,
            "mean_candidate_minus_baseline": delta_mean,
            "relative_delta_percent": relative_delta,
            "mean_improvement": -delta_mean if direction == "lower" else delta_mean,
            "delta_confidence_interval": interval, "confidence": confidence,
            "bootstrap_method": "paired context/trace-cluster percentile, linear quantiles",
            "bootstrap_samples": bootstrap_samples if interval is not None else 0,
            "bootstrap_seed": seed, "candidate_first_pairs": candidate_first,
            "warmup_rows_excluded": sum(row["phase"] == "warmup" and row["method"] in {baseline, candidate} for row in rows),
            "warnings": warnings}
