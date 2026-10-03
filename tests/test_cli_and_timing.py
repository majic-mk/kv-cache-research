from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from kv_harness.cli import main
from kv_harness.common import ValidationError, parse_json, read_json, sha256_bytes
from kv_harness.fixtures import run_cpu_fixture
from kv_harness.ledger import audit_ledger
from kv_harness.manifest import RunManifest
from kv_harness.results import load_results
from kv_harness.timing import SerialTimer
from tests.helpers import init_repo


class TimingTests(unittest.TestCase):
    def test_serial_phases_and_outer_overhead(self):
        with SerialTimer() as timer:
            with timer.section("scoring_s"):
                sum(range(100))
            with timer.section("host_copy_s"):
                bytes(bytearray(b"abc"))
        timing = timer.snapshot()
        self.assertGreater(timing["end_to_end_s"], 0)
        self.assertGreaterEqual(timing["end_to_end_s"], timing["scoring_s"] + timing["host_copy_s"])

    def test_nested_phases_and_reuse_rejected(self):
        timer = SerialTimer()
        with self.assertRaises(RuntimeError):
            timer.snapshot()
        with timer:
            with timer.section("scoring_s"):
                with self.assertRaises(RuntimeError):
                    with timer.section("prefill_s"):
                        pass
        with self.assertRaises(RuntimeError):
            with timer:
                pass
        with self.assertRaises(RuntimeError):
            with timer.section("scoring_s"):
                pass

    def test_exception_preserves_measured_cost(self):
        timer = SerialTimer()
        with self.assertRaisesRegex(ValueError, "fixture failure"):
            with timer:
                with timer.section("scoring_s"):
                    raise ValueError("fixture failure")
        self.assertGreater(timer.snapshot()["scoring_s"], 0)


class LedgerTests(unittest.TestCase):
    def test_existing_initialization_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.csv"
            text = ("timestamp_utc,provider,resource,amount,currency,status,evidence\n"
                    "2026-10-03T06:30:00Z,none,initialization,0,CNY,no_external_purchase,no_charge\n")
            path.write_text(text)
            self.assertEqual(audit_ledger(path)["totals_by_currency_and_status"]["CNY"]["settled"], "0")
            self.assertEqual(path.read_text(), text)

    def test_invalid_cost_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "costs.csv"
            header = "timestamp_utc,provider,resource,amount,currency,status,evidence\n"
            for amount, currency, status in (("NaN", "CNY", "settled"), ("1", "CNY", "no_external_purchase"),
                                             ("-1", "USD", "settled"), ("1", "usd", "settled")):
                path.write_text(header + f"2026-10-03T00:00:00Z,test,test,{amount},{currency},{status},test\n")
                with self.subTest(amount=amount, currency=currency, status=status), self.assertRaises(ValidationError):
                    audit_ledger(path)


class CLITests(unittest.TestCase):
    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_fixture_cli_validation_analysis_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_repo(root)
            output = root / "outputs" / "fixture"
            status, stdout, _ = self.invoke(["cpu-fixture", "--repo", str(root), "--output", str(output)])
            self.assertEqual(status, 0)
            completion = parse_json(stdout)
            self.assertFalse(completion["experimental_evidence"])
            self.assertEqual(completion["results_file_sha256"], sha256_bytes((output / "results.jsonl").read_bytes()))
            manifest_path, results_path = str(output / "manifest.json"), str(output / "results.jsonl")
            args = ["--manifest", manifest_path, "--results", results_path]
            self.assertEqual(self.invoke(["validate", *args])[0], 0)
            analyze = ["analyze", *args, "--baseline", "fixture_baseline", "--candidate", "fixture_candidate",
                       "--bootstrap-samples", "100"]
            status, _, stderr = self.invoke(analyze)
            self.assertEqual(status, 2)
            self.assertIn("Nonexperimental", stderr)
            status, stdout, _ = self.invoke(analyze + ["--allow-nonexperimental"])
            self.assertEqual(status, 0)
            self.assertEqual(parse_json(stdout)["independent_clusters"], 4)
            self.assertEqual(self.invoke(["cpu-fixture", "--repo", str(root), "--output", str(output)])[0], 2)
            rows = load_results(results_path, RunManifest.load(manifest_path))
            checksum_pairs = {}
            for row in rows:
                checksum_pairs.setdefault((row["phase"], row["unit_id"], row["replicate"]), set()).add(row["output_sha256"])
            self.assertTrue(all(len(checksums) == 1 for checksums in checksum_pairs.values()))

    def test_invalid_utf8_is_clean_cli_error(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory) / "bad.json"
            spec.write_bytes(b"\xff")
            status, _, stderr = self.invoke(["manifest", "--spec", str(spec), "--output", str(Path(directory) / "out.json")])
            self.assertEqual(status, 2)
            self.assertIn("UTF-8", stderr)

    def test_custom_metric_requires_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            init_repo(root)
            output = root / "outputs" / "fixture"
            run_cpu_fixture(repo=root, output=output, units=1, replicates=1)
            status, _, stderr = self.invoke(["analyze", "--manifest", str(output / "manifest.json"),
                "--results", str(output / "results.jsonl"), "--baseline", "fixture_baseline", "--candidate",
                "fixture_candidate", "--metric", "metrics.fixture_byte_mean", "--allow-nonexperimental"])
            self.assertEqual(status, 2)
            self.assertIn("explicit --direction", stderr)


if __name__ == "__main__":
    unittest.main()
