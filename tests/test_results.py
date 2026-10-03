from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from kv_harness.common import ValidationError, canonical_json, digest
from kv_harness.results import load_results, save_results, total_cost_s, validate_records, validate_result
from tests.helpers import record, test_manifest


class ResultsTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            save_results(path, [record()], test_manifest())
            self.assertEqual(load_results(path, test_manifest()), [record()])

    def test_timing_and_workload_validation(self):
        mutations = [lambda row: row["timing"].pop("scoring_s"),
                     lambda row: row["timing"].update(end_to_end_s=-1),
                     lambda row: row["timing"].update(scoring_s=True),
                     lambda row: row["timing"].update(scoring_s=float("nan")),
                     lambda row: row["timing"].update(scoring_s=8, preprocessing_s=8),
                     lambda row: row["workload"].update(tampered=True),
                     lambda row: row.update(replicate=True),
                     lambda row: row.update(status=[]),
                     lambda row: row.update(extra_field=True)]
        for index, mutate in enumerate(mutations):
            value = record()
            mutate(value)
            with self.subTest(index=index), self.assertRaises(ValidationError):
                validate_result(value)

    def test_explicit_overlap_is_allowed(self):
        row = record()
        row["timing"].update(accounting_mode="overlapping", prefill_s=8, cache_read_s=8)
        validate_result(row)

    def test_shared_cost_is_inclusive_and_denominator_positive(self):
        row = record()
        row["shared_setup"].update(group_id="setup-a", preprocessing_s=12, scoring_s=4, copies_s=2, other_s=2,
                                   amortization_count=4)
        validate_result(row)
        self.assertEqual(total_cost_s(row), 15)
        row["shared_setup"]["amortization_count"] = 0
        with self.assertRaises(ValidationError):
            validate_result(row)

    def test_mixed_and_duplicate_records_rejected(self):
        with self.assertRaisesRegex(ValidationError, "Duplicate"):
            validate_records([record(), record()])
        other = record(unit="u1")
        other["run_id"] = "different"
        with self.assertRaisesRegex(ValidationError, "Mixed"):
            validate_records([record(), other])
        with self.assertRaisesRegex(ValidationError, "manifest"):
            validate_result(other, test_manifest())

    def test_nonexperimental_cannot_change_evidence_kind(self):
        row = record()
        row["evidence_kind"] = "real_model"
        with self.assertRaisesRegex(ValidationError, "differs from manifest"):
            validate_result(row, test_manifest())

    def test_gpu_requires_synchronized_boundaries(self):
        row = record(manifest=test_manifest("real_model"))
        row["execution"].update(device_type="gpu", synchronized=False)
        with self.assertRaisesRegex(ValidationError, "synchronized"):
            validate_result(row)

    def test_memory_logical_budget_is_not_physical_memory(self):
        row = record()
        row["resources"].update(logical_kept_tokens=64, logical_kv_bytes=1024,
                                representation="masked_dense", measurement_method="test-vector-budget")
        validate_result(row)
        self.assertIsNone(row["resources"]["physical_device_kv_allocated_bytes"])
        self.assertIsNone(row["resources"]["peak_device_reserved_bytes"])
        row["resources"]["measurement_method"] = "not_measured"
        with self.assertRaisesRegex(ValidationError, "measurement method"):
            validate_result(row)

    def test_inconsistent_shared_group(self):
        one, two = record(unit="u1"), record(unit="u2")
        for row in (one, two):
            row["shared_setup"].update(group_id="same", copies_s=2)
        two["shared_setup"]["copies_s"] = 3
        with self.assertRaisesRegex(ValidationError, "inconsistent"):
            validate_records([one, two])

    def test_blank_duplicate_key_and_nonfinite_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            for text in ("\n", '{"a":1,"a":2}\n', '{"a":NaN}\n'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValidationError):
                    load_results(path)
        with self.assertRaisesRegex(ValidationError, "No result"):
            validate_records([])


if __name__ == "__main__":
    unittest.main()
