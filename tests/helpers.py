from copy import deepcopy
from pathlib import Path
import subprocess

from kv_harness.common import canonical_json, digest, sha256_bytes
from kv_harness.manifest import RunManifest
from kv_harness.results import unmeasured_resources, zero_setup, zero_timing


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "-c", "user.name=Harness test",
        "-c", "user.email=fixture@localhost", *args], stderr=subprocess.DEVNULL).decode().strip()


def init_repo(repo: Path) -> None:
    git(repo, "init", "-b", "main")
    (repo / "source.txt").write_text("original\n")
    git(repo, "add", "source.txt")
    git(repo, "commit", "-m", "CPU test fixture")


def manifest_payload(kind: str = "cpu_fixture") -> dict:
    source = {"git_revision": "a" * 40, "dirty": False, "tracked_diff_sha256": sha256_bytes(b""),
              "staged_diff_sha256": sha256_bytes(b""), "unstaged_diff_sha256": sha256_bytes(b""),
              "status_sha256": sha256_bytes(b""), "untracked": {}, "submodule_revisions": []}
    source["snapshot_sha256"] = digest(source)
    return {"schema_version": 1, "run_id": "test-run", "created_at_utc": "2026-10-03T00:00:00Z",
            "evidence_kind": kind, "seed": 7, "source": source,
            "artifacts": {name: {"identifier": "test-" + name, "revision_kind": "git_commit", "revision": "b" * 40}
                          for name in ("model", "dataset")},
            "environment": {"python_version": "3.12", "python_implementation": "CPython", "platform": "test",
                            "machine": "test", "harness_version": "test", "packages": [], "hardware": {}},
            "config": {"fixture_only": True}}


def test_manifest(kind: str = "cpu_fixture") -> RunManifest:
    return RunManifest(canonical_json(manifest_payload(kind)))


def record(method: str = "base", unit: str = "u0", replicate: int = 0,
           value: float = 10.0, manifest: RunManifest | None = None) -> dict:
    manifest = manifest or test_manifest()
    workload = {"fixture": True, "unit": unit, "input_digest": digest(unit)}
    timing = zero_timing()
    timing["end_to_end_s"] = value
    return {"schema_version": 1, "run_id": manifest.run_id, "manifest_sha256": manifest.sha256,
            "evidence_kind": manifest.payload["evidence_kind"], "phase": "measurement", "method": method,
            "unit_id": unit, "cluster_id": unit, "replicate": replicate, "order_index": 0 if method == "base" else 1,
            "workload": workload, "workload_sha256": digest(workload), "status": "ok", "error": None,
            "output_sha256": "c" * 64, "timing": timing, "shared_setup": zero_setup(),
            "metrics": {"quality": value}, "resources": unmeasured_resources(),
            "execution": {"device_type": "cpu", "backend": "test", "synchronized": True}}


def pairs(deltas: list[float], *, repeats: int = 1) -> list[dict]:
    result = []
    for unit, delta in enumerate(deltas):
        for replicate in range(repeats):
            result += [record("base", f"u{unit}", replicate, 10),
                       record("candidate", f"u{unit}", replicate, 10 + delta)]
    return deepcopy(result)
