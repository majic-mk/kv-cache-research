from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

from kv_harness.common import ValidationError, canonical_json, digest, parse_json, read_json, write_new
from kv_harness.manifest import RunManifest, capture_environment, capture_source
from tests.helpers import git, init_repo, manifest_payload, test_manifest


class CommonTests(unittest.TestCase):
    def test_canonical_order(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '[Infinity]', '1e999', 'bad'):
            with self.subTest(text=text), self.assertRaises(ValidationError):
                parse_json(text)
        for value in ({1: "bad"}, (1, 2), {"a": float("inf")}, {"bad": "\ud800"}):
            with self.subTest(value=repr(value)), self.assertRaises(ValidationError):
                canonical_json(value)

    def test_write_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            write_new(path, "first")
            with self.assertRaises(FileExistsError):
                write_new(path, "second")
            self.assertEqual(path.read_text(), "first")
            self.assertEqual(list(Path(directory).glob(".publish-*")), [])


class ManifestTests(unittest.TestCase):
    def test_roundtrip_tamper_and_immutability(self):
        manifest = test_manifest()
        payload = manifest.payload
        payload["config"]["tampered"] = True
        self.assertNotIn("tampered", manifest.payload["config"])
        with self.assertRaises(FrozenInstanceError):
            manifest._canonical = "bad"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest.save(path)
            self.assertEqual(RunManifest.load(path), manifest)
            envelope = read_json(path)
            envelope["manifest"]["seed"] = 42
            bad_path = Path(directory) / "tampered.json"
            bad_path.write_text(canonical_json(envelope))
            with self.assertRaisesRegex(ValidationError, "checksum"):
                RunManifest.load(bad_path)

    def test_reject_branch_revision(self):
        payload = manifest_payload()
        payload["artifacts"]["model"]["revision"] = "main"
        with self.assertRaisesRegex(ValidationError, "immutable"):
            RunManifest(canonical_json(payload))

    def test_reject_fixture_labeled_real(self):
        payload = manifest_payload("real_model")
        payload["artifacts"]["model"]["identifier"] = "fixture:primitive"
        with self.assertRaisesRegex(ValidationError, "cannot be labeled"):
            RunManifest(canonical_json(payload))

    def test_reject_malformed_enum_and_utc(self):
        for key, value in (("evidence_kind", []), ("seed", True), ("created_at_utc", "2026-10-03T00:00:00")):
            payload = manifest_payload()
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(ValidationError):
                RunManifest(canonical_json(payload))

    def test_source_snapshot_tamper(self):
        payload = manifest_payload()
        payload["source"]["dirty"] = True
        with self.assertRaisesRegex(ValidationError, "snapshot checksum"):
            RunManifest(canonical_json(payload))

    def test_git_tracks_staged_unstaged_and_untracked(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_repo(repo)
            clean = capture_source(repo)
            self.assertFalse(clean["dirty"])
            (repo / "source.txt").write_text("staged\n")
            git(repo, "add", "source.txt")
            (repo / "source.txt").write_text("unstaged\n")
            (repo / "new.txt").write_text("untracked")
            changed = capture_source(repo)
            self.assertTrue(changed["dirty"])
            self.assertNotEqual(changed["staged_diff_sha256"], clean["staged_diff_sha256"])
            self.assertNotEqual(changed["unstaged_diff_sha256"], clean["unstaged_diff_sha256"])
            self.assertIn("new.txt", changed["untracked"])
            self.assertNotEqual(changed["snapshot_sha256"], clean["snapshot_sha256"])
            (repo / "new.txt").write_text("different")
            self.assertNotEqual(capture_source(repo)["untracked"], changed["untracked"])

    def test_real_runs_require_clean_committed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_repo(repo)
            payload = manifest_payload("real_model")
            arguments = dict(repo=repo, evidence_kind="real_model", artifacts=payload["artifacts"], config={}, seed=0)
            manifest = RunManifest.create(**arguments)
            self.assertFalse(manifest.payload["source"]["dirty"])
            (repo / "untracked.txt").write_text("change")
            with self.assertRaisesRegex(ValidationError, "clean"):
                RunManifest.create(**arguments)

    def test_hidden_index_flags_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_repo(repo)
            for enable, disable in (("--assume-unchanged", "--no-assume-unchanged"),
                                    ("--skip-worktree", "--no-skip-worktree")):
                git(repo, "update-index", enable, "source.txt")
                with self.subTest(flag=enable), self.assertRaisesRegex(ValidationError, "index flags"):
                    capture_source(repo)
                git(repo, "update-index", disable, "source.txt")

    def test_submodules_are_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            init_repo(repo)
            revision = git(repo, "rev-parse", "HEAD")
            git(repo, "update-index", "--add", "--cacheinfo", f"160000,{revision},module")
            git(repo, "config", "submodule.module.ignore", "all")
            with self.assertRaisesRegex(ValidationError, "Submodules"):
                capture_source(repo)

    def test_environment_no_secret_capture(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {"HARNESS_TEST_SECRET": "never-record-me"}):
            self.assertNotIn("never-record-me", canonical_json(capture_environment()))


if __name__ == "__main__":
    unittest.main()
