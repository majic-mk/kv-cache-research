"""Synthetic FIXTURE-ONLY tests; none are model accuracy or GPU evidence."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from echo_pilot import BACKEND_COMMIT
from echo_pilot.cli import run
from echo_pilot.contracts import (ContractError, canonical_hash, prepare_context, selection_budget,
                                 sha256_file, validate_config, validate_inputs, validate_questions)
from echo_pilot.runtime import DependencyError, cache_bytes, independent_cache, load_dependencies, clear_masks, restore_masks


def fixture_config():
    return {"schema_version": 1, "backend_commit": BACKEND_COMMIT,
        "model": {"id": "fixture/model", "revision": "a" * 40, "tokenizer_id": "fixture/model", "tokenizer_revision": "a" * 40, "dtype": "bfloat16"},
        "dataset": {"source_id": "fixture-only", "source_revision": "b" * 40, "preprocessing_revision": "c" * 40, "preprocessing_sha256": "c" * 64, "preprocessing_revision_scope": "local_origin",
                    "split": "fixture", "license": "fixture", "contexts_file": "contexts.jsonl", "contexts_sha256": "d" * 64,
                    "questions_file": "questions.jsonl", "questions_sha256": "e" * 64, "evidence_kind": "fixture_only"},
        "template": {"prompt_mode": "chat_adapted", "chat_template_sha256": "f" * 64, "prefix_token_ids": [1, 2], "suffix_token_ids": [9], "enable_thinking": False},
        "selection": {"eviction_ratios": [0.5, 0.75, 0.9], "n_sink": 1, "recent_tokens": 0, "chunk_size": 2048, "scoring_backend": "torch_eager"},
        "generation": {"max_new_tokens": 64, "decoding": "greedy"},
        "runtime": {"device": "cuda:0", "attention_backend": "sdpa", "seed": 17, "local_files_only": True,
                    "packages_lock_file": "packages.json", "packages_lock_sha256": "f" * 64,
                    "capture_scores": True, "deterministic_algorithms": True},
        "arms": ["fullkv", "echo", "kvzip", "kvzip_echo_partition"],
        "truncation": {"policy": "reject", "max_context_tokens": 16}}


def fixture_context():
    return {"context_id": "fixture-context", "source_id": "fixture-source", "context_token_ids": list(range(1, 13)),
            "regions": [{"region_id": "a", "label": "fixture", "start": 2, "end": 7},
                        {"region_id": "b", "label": "fixture", "start": 7, "end": 12}]}


def fixture_files(folder):
    config = fixture_config()
    (folder / "contexts.jsonl").write_text(json.dumps(fixture_context()) + "\n")
    (folder / "questions.jsonl").write_text(json.dumps({"context_id": "fixture-context", "question_id": "fixture-q", "question_token_ids": [8, 9]}) + "\n")
    (folder / "packages.json").write_text(json.dumps({"python_version": "3.12.14", "backend_commit": BACKEND_COMMIT,
        "packages": {"torch": "fixture", "transformers": "fixture", "kvpress": "fixture", "numpy": "fixture", "peft": "fixture"}}))
    for kind in ("contexts", "questions"):
        config["dataset"][kind + "_sha256"] = sha256_file(folder / (kind + ".jsonl"))
    config["runtime"]["packages_lock_sha256"] = sha256_file(folder / "packages.json")
    (folder / "config.json").write_text(json.dumps(config))
    return config


class ContractTests(unittest.TestCase):
    def test_valid_fixture_config(self):
        validate_config(fixture_config())

    def test_mutable_revision_rejected(self):
        config = fixture_config(); config["model"]["revision"] = "main"
        with self.assertRaisesRegex(ContractError, "immutable"):
            validate_config(config)

    def test_release_version_instead_of_commit_rejected(self):
        config = fixture_config(); config["backend_commit"] = "0.5.5"
        with self.assertRaises(ContractError): validate_config(config)

    def test_unknown_knob_rejected(self):
        config = fixture_config(); config["selection"]["oracle"] = True
        with self.assertRaisesRegex(ContractError, "unknown"): validate_config(config)

    def test_nonfinite_ratio_rejected(self):
        config = fixture_config(); config["selection"]["eviction_ratios"] = [float("nan")]
        with self.assertRaises(ContractError): validate_config(config)

    def test_download_mode_rejected(self):
        config = fixture_config(); config["runtime"]["local_files_only"] = False
        with self.assertRaises(ContractError): validate_config(config)

    def test_future_query_in_context_schema_rejected(self):
        row = fixture_context(); row["question_token_ids"] = [9]
        with self.assertRaisesRegex(ContractError, "unknown"): prepare_context(row, fixture_config())

    def test_region_gap_rejected(self):
        row = fixture_context(); row["regions"][1]["start"] = 8
        with self.assertRaisesRegex(ContractError, "contiguous"): prepare_context(row, fixture_config())

    def test_missing_prefix_rejected(self):
        row = fixture_context(); row["context_token_ids"][0] = 0
        with self.assertRaisesRegex(ContractError, "prefix"): prepare_context(row, fixture_config())

    def test_truncation_reject_is_default(self):
        config = fixture_config(); config["truncation"]["max_context_tokens"] = 9
        with self.assertRaisesRegex(ContractError, "exceeds"): prepare_context(fixture_context(), config)

    def test_exact_right_truncation_and_region_clip(self):
        config = fixture_config(); config["truncation"] = {"policy": "right", "max_context_tokens": 9}
        result = prepare_context(fixture_context(), config)
        self.assertEqual(result.token_ids, tuple(range(1, 10)))
        self.assertEqual(result.regions[-1]["end"], 9)
        self.assertEqual(result.truncation["removed_range"], [9, 12])
        self.assertEqual(result.prefix_length, 2)

    def test_boolean_token_rejected(self):
        row = fixture_context(); row["context_token_ids"][2] = True
        with self.assertRaises(ContractError): prepare_context(row, fixture_config())

    def test_questions_change_cannot_change_context_hash(self):
        context = prepare_context(fixture_context(), fixture_config())
        initial = context.fingerprint()
        for question in ([8, 9], [1, 1, 1, 9]):
            validate_questions([{"context_id": context.context_id, "question_id": "q", "question_token_ids": question}], {context.context_id}, [9])
            self.assertEqual(context.fingerprint(), initial)

    def test_missing_question_suffix_rejected(self):
        with self.assertRaisesRegex(ContractError, "suffix"):
            validate_questions([{"context_id": "c", "question_id": "q", "question_token_ids": [8]}], {"c"}, [9])

    def test_duplicate_question_id_rejected(self):
        row = {"context_id": "c", "question_id": "q", "question_token_ids": [9]}
        with self.assertRaisesRegex(ContractError, "unique"): validate_questions([row, row], {"c"}, [9])

    def test_exact_floor_eviction_budget(self):
        result = selection_budget(2, 3, 11, 0.75, 1, 0)
        self.assertEqual(result["pruned_pairs"], 49)
        self.assertEqual(result["retained_pairs"], 17)

    def test_protection_overlap_not_double_counted(self):
        result = selection_budget(1, 1, 10, 0.0, 6, 6)
        self.assertEqual(result["protected_pairs"], 10)

    def test_impossible_sink_recency_budget_rejected(self):
        with self.assertRaisesRegex(ContractError, "protected"):
            selection_budget(2, 3, 12, 0.9, 4, 4)

    def test_dry_run_does_not_import_ml_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_files(root)
            code = "from echo_pilot.cli import run; from pathlib import Path; import sys; run(Path(sys.argv[1]), None, False); assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules; assert 'kvpress' not in sys.modules"
            subprocess.run([sys.executable, "-c", code, str(root / "config.json")], check=True)

    def test_dry_run_evidence_marked_fixture_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_files(root)
            result = run(root / "config.json", root / "out", False)
            self.assertEqual(result["status"], "validated_not_executed")
            self.assertEqual(result["evidence_kind"], "fixture_only")
            self.assertEqual(result["cache_builds"], 10)
            self.assertFalse((root / "out/COMPLETE.json").exists())

    def test_input_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = fixture_files(root)
            (root / "questions.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ContractError, "SHA-256"): validate_inputs(config, root)

    def test_fixture_data_cannot_execute_as_real_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_files(root)
            with self.assertRaisesRegex(ContractError, "Fixture-only"):
                run(root / "config.json", root / "out", True)
            self.assertTrue((root / "out/FAILED.json").is_file())
            self.assertFalse((root / "out/COMPLETE.json").exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture_files(root); (root / "out").mkdir()
            with self.assertRaises(FileExistsError): run(root / "config.json", root / "out", False)

    def test_optional_import_error_is_actionable(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(DependencyError, "INSTALL.md"): load_dependencies()


class FixtureStorage:
    def __init__(self, capacity): self.capacity = capacity
    def data_ptr(self): return id(self)
    def nbytes(self): return self.capacity


class FixtureTensor:
    """Fixture-only object exposing storage accounting; not an ML tensor."""
    def __init__(self, data, storage=None):
        self.data = list(data); self.storage = storage or FixtureStorage(len(data) * 4)
        self.shape = (1, 1, len(data), 1); self.device = "fixture-cpu"
    def numel(self): return len(self.data)
    def element_size(self): return 4
    def untyped_storage(self): return self.storage
    def clone(self): return FixtureTensor(self.data)


class FixtureLayer:
    def __init__(self): self.keys = FixtureTensor([1, 2]); self.values = FixtureTensor([3, 4])


class FixtureCache:
    def __init__(self): self.layers = [FixtureLayer()]; self.metadata = {"counter": [2]}


class CacheIsolationFixtureTests(unittest.TestCase):
    def test_independent_query_clones_preserve_frozen_cache(self):
        source = FixtureCache(); a = independent_cache(source); b = independent_cache(source)
        a.layers[0].keys.data[0] = -999
        a.metadata["counter"][0] = 99
        self.assertEqual(source.layers[0].keys.data, [1, 2])
        self.assertEqual(b.layers[0].keys.data, [1, 2])
        self.assertEqual(source.metadata["counter"], [2])

    def test_aliasing_deepcopy_rejected(self):
        source = FixtureCache()
        with patch("echo_pilot.runtime.copy.deepcopy", return_value=source):
            with self.assertRaisesRegex(ContractError, "aliases"): independent_cache(source)

    def test_physical_storage_not_logical_payload(self):
        cache = FixtureCache()
        cache.layers[0].keys.storage = FixtureStorage(128)
        info = cache_bytes(cache)
        self.assertEqual(info["kv_tensor_payload_bytes"], 16)
        self.assertEqual(info["kv_unique_storage_bytes"], 136)
        self.assertFalse(info["compacted"])

    def test_shared_backing_storage_counted_once(self):
        cache = FixtureCache(); cache.layers[0].values.storage = cache.layers[0].keys.storage
        self.assertEqual(cache_bytes(cache)["kv_unique_storage_bytes"], 8)



class FixtureVector:
    """Fixture-only minimal IDs/positions object; no model or tensor computation."""
    def __init__(self, values): self.values = list(values); self.shape = (1, len(self.values))
    def unsqueeze(self, dim): return self
    def __getitem__(self, item): return FixtureVector(self.values[item[-1] if isinstance(item, tuple) else item])
    def __add__(self, scalar): return FixtureVector([x + scalar for x in self.values])


class FixtureTorch:
    long = "fixture-long"
    @staticmethod
    def tensor(values, **kwargs): return FixtureVector(values[0])
    @staticmethod
    def arange(start, end, **kwargs): return FixtureVector(range(start, end))


class FixtureLogits:
    def __init__(self, value): self.value = value
    def __getitem__(self, item): return self
    def argmax(self): return self
    def item(self): return self.value


class GenerationFixtureTests(unittest.TestCase):
    def generate(self, outputs, limit=5, eos=9):
        from types import SimpleNamespace
        from echo_pilot.runtime import greedy_answer
        class Model:
            device = "fixture-cpu"
            generation_config = SimpleNamespace(eos_token_id=eos)
            def __init__(self): self.calls = []
            def __call__(self, **kwargs):
                self.calls.append({"ids": kwargs["input_ids"].values, "positions": kwargs["position_ids"].values,
                                   "cache": kwargs["past_key_values"]})
                return SimpleNamespace(logits=FixtureLogits(outputs[len(self.calls) - 1]))
        model = Model()
        tokenizer = SimpleNamespace(decode=lambda values, **kwargs: str(values))
        cache = object()
        answer = greedy_answer(FixtureTorch, model, tokenizer, [5, 6], cache, 100, limit)
        return answer, model.calls

    def test_first_eos_stops_without_extra_forward(self):
        answer, calls = self.generate([9])
        self.assertEqual(answer["generated_token_ids"], [9])
        self.assertTrue(answer["stopped_on_eos"])
        self.assertEqual(len(calls), 1)

    def test_greedy_absolute_positions_and_one_token_decode(self):
        answer, calls = self.generate([3, 4, 9])
        self.assertEqual(answer["generated_token_ids"], [3, 4, 9])
        self.assertEqual([x["positions"] for x in calls], [[100, 101], [102], [103]])
        self.assertEqual([x["ids"] for x in calls], [[5, 6], [3], [4]])
        self.assertTrue(all(x["cache"] is calls[0]["cache"] for x in calls))

    def test_output_limit_exact(self):
        answer, calls = self.generate([3, 4], limit=2)
        self.assertEqual(len(calls), 2)
        self.assertFalse(answer["stopped_on_eos"])

    def test_multiple_eos_ids(self):
        answer, calls = self.generate([7], eos=[7, 9])
        self.assertTrue(answer["stopped_on_eos"])
        self.assertEqual(len(calls), 1)


class RuntimeGateFixtureTests(unittest.TestCase):
    def test_extra_unlocked_package_rejected(self):
        from types import SimpleNamespace
        from echo_pilot.runtime import verify_environment
        lock = {"python_version": "3.12.14", "packages": {"torch": "fixture"}}
        dist = [SimpleNamespace(metadata={"Name": "torch"}, version="fixture"),
                SimpleNamespace(metadata={"Name": "unlocked-transitive"}, version="fixture")]
        with patch("echo_pilot.runtime.platform.python_version", return_value="3.12.14"), patch("echo_pilot.runtime.metadata.distributions", return_value=dist):
            with self.assertRaisesRegex(ContractError, "Complete installed package inventory"):
                verify_environment(None, lock)

    def test_normalized_duplicate_packages_rejected(self):
        from types import SimpleNamespace
        from echo_pilot.runtime import verify_environment
        lock = {"python_version": "3.12.14", "packages": {"a_b": "fixture", "a-b": "fixture"}}
        dist = [SimpleNamespace(metadata={"Name": "a-b"}, version="fixture")]
        with patch("echo_pilot.runtime.platform.python_version", return_value="3.12.14"), patch("echo_pilot.runtime.metadata.distributions", return_value=dist):
            with self.assertRaisesRegex(ContractError, "duplicate normalized"):
                verify_environment(None, lock)


class DeterministicEnvironmentTests(unittest.TestCase):
    def test_absent_cublas_config_rejected_before_gpu_import(self):
        from echo_pilot.runtime import deterministic_environment
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ContractError, "before launching Python"):
                deterministic_environment(fixture_config())

    def test_invalid_cublas_config_rejected(self):
        from echo_pilot.runtime import deterministic_environment
        with patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": "invalid"}, clear=True):
            with self.assertRaises(ContractError): deterministic_environment(fixture_config())

    def test_valid_cublas_config_recorded(self):
        from echo_pilot.runtime import deterministic_environment
        for value in (":4096:8", ":16:8"):
            with patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": value}, clear=True):
                self.assertEqual(deterministic_environment(fixture_config())["CUBLAS_WORKSPACE_CONFIG"], value)

    def test_nondeterministic_explicit_mode_does_not_require_cublas(self):
        from echo_pilot.runtime import deterministic_environment
        config = fixture_config(); config["runtime"]["deterministic_algorithms"] = False
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(deterministic_environment(config)["CUBLAS_WORKSPACE_CONFIG"])

if __name__ == "__main__": unittest.main()
