from copy import deepcopy
import unittest

from kv_harness.common import ValidationError, digest
from kv_harness.statistics import paired_summary
from tests.helpers import pairs, record


def summarize(rows, **kwargs):
    return paired_summary(rows, baseline="base", candidate="candidate", bootstrap_samples=500,
                          allow_nonexperimental=True, **kwargs)


class StatisticsTests(unittest.TestCase):
    def test_constant_paired_effect(self):
        result = summarize(pairs([-3, -3, -3], repeats=2))
        self.assertEqual(result["paired_observations"], 6)
        self.assertEqual(result["independent_clusters"], 3)
        self.assertEqual(result["mean_candidate_minus_baseline"], -3)
        self.assertEqual(result["delta_confidence_interval"], [-3, -3])
        self.assertFalse(result["experimental_evidence"])

    def test_deterministic_bootstrap_and_input_order(self):
        rows = pairs([-3, 1, -2, 4])
        one = summarize(rows, seed=23)
        self.assertEqual(one, summarize(rows, seed=23))
        self.assertEqual(one, summarize(list(reversed(rows)), seed=23))

    def test_questions_sharing_context_are_one_cluster(self):
        rows = pairs([-2, -2, -2, -2, -2])
        for row in rows:
            row["cluster_id"] = "one-shared-cache-context"
        result = summarize(rows)
        self.assertEqual(result["paired_units"], 5)
        self.assertEqual(result["independent_clusters"], 1)
        self.assertIsNone(result["delta_confidence_interval"])
        self.assertEqual(result["bootstrap_samples"], 0)

    def test_equal_cluster_weight_and_nested_replicates(self):
        rows = pairs([-2, -2, -2, 4], repeats=3)
        for row in rows:
            row["cluster_id"] = "context-a" if row["unit_id"] != "u3" else "context-b"
        result = summarize(rows)
        self.assertEqual(result["paired_observations"], 12)
        self.assertEqual(result["independent_clusters"], 2)
        self.assertEqual(result["mean_candidate_minus_baseline"], 1)
        self.assertEqual(result["units_per_cluster"], {"context-a": 3, "context-b": 1})

    def test_one_unit_cannot_move_clusters(self):
        rows = pairs([-2], repeats=2)
        rows[-1]["cluster_id"] = "other-context"
        with self.assertRaisesRegex(ValidationError, "move between"):
            summarize(rows)

    def test_missing_pairs_fail(self):
        with self.assertRaisesRegex(ValidationError, "Incomplete pairing"):
            summarize(pairs([-2, -1])[:-1])

    def test_mismatched_workload_fails(self):
        rows = pairs([-1])
        rows[1]["workload"]["load"] = "different"
        rows[1]["workload_sha256"] = digest(rows[1]["workload"])
        with self.assertRaisesRegex(ValidationError, "Mismatched inputs"):
            summarize(rows)

    def test_failed_trials_not_dropped(self):
        rows = pairs([-1, -2])
        rows[1].update(status="error", error="out of memory", output_sha256=None)
        with self.assertRaisesRegex(ValidationError, "failed trials"):
            summarize(rows)

    def test_warmups_excluded_but_counted(self):
        rows = pairs([-1, -1])
        warmup = record(value=999)
        warmup["phase"] = "warmup"
        result = summarize(rows + [warmup])
        self.assertEqual(result["warmup_rows_excluded"], 1)
        self.assertEqual(result["mean_candidate_minus_baseline"], -1)

    def test_nonexperimental_requires_opt_in(self):
        with self.assertRaisesRegex(ValidationError, "Nonexperimental"):
            paired_summary(pairs([-1]), baseline="base", candidate="candidate")

    def test_denominator_must_equal_observed_reuse(self):
        rows = pairs([-2, -2])
        for row in rows:
            row["shared_setup"].update(group_id="test-setup", scoring_s=100, amortization_count=1_000_000_000)
        with self.assertRaisesRegex(ValidationError, "declares"):
            summarize(rows)

    def test_accounted_shared_work_changes_effect(self):
        rows = pairs([-2, -2])
        for row in rows:
            if row["method"] == "candidate":
                row["shared_setup"].update(group_id="candidate-setup", scoring_s=6, amortization_count=2)
        full = summarize(rows)
        online = summarize(rows, metric="end_to_end_s")
        self.assertEqual(full["mean_candidate_minus_baseline"], 1)
        self.assertEqual(online["mean_candidate_minus_baseline"], -2)
        self.assertTrue(any("excludes nonzero" in warning for warning in online["warnings"]))

    def test_summary_hash_identifies_records(self):
        one, two = summarize(pairs([-2, -2])), summarize(pairs([1, 1]))
        self.assertEqual(one["manifest_sha256"], two["manifest_sha256"])
        self.assertNotEqual(one["records_sha256"], two["records_sha256"])

    def test_extreme_finite_metrics_do_not_overflow_mean(self):
        rows = pairs([0, 0])
        for row in rows:
            row["metrics"]["quality"] = 1e308
        result = summarize(rows, metric="metrics.quality", direction="higher")
        self.assertEqual(result["baseline_mean"], 1e308)
        self.assertEqual(result["mean_candidate_minus_baseline"], 0)

    def test_invalid_bootstrap_parameters(self):
        for kwargs in ({"confidence": 1}, {"confidence": float("nan")}, {"seed": -1}, {"direction": "sideways"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValidationError):
                summarize(pairs([-1, -1]), **kwargs)


if __name__ == "__main__":
    unittest.main()
