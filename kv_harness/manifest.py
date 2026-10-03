"""Immutable run manifests with input, source and environment provenance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
from typing import Any
from uuid import uuid4

from . import __version__
from .common import (ValidationError, canonical_json, choice, digest, integer, is_hex,
                     nonempty_string, parse_json, read_json, require_keys,
                     sha256_bytes, write_new)

EVIDENCE_KINDS = {"real_model", "cpu_fixture", "simulation"}
MANIFEST_KEYS = {"schema_version", "run_id", "created_at_utc", "evidence_kind",
                 "seed", "source", "artifacts", "environment", "config"}


def _git(repo: Path, *args: str, required: bool = True) -> bytes:
    process = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if required and process.returncode:
        raise ValidationError(f"git {' '.join(args)} failed: {process.stderr.decode().strip()}")
    return process.stdout if process.returncode == 0 else b""


def capture_source(repo: str | Path) -> dict[str, Any]:
    repo = Path(repo).resolve()
    root = Path(_git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    if root != repo:
        raise ValidationError("Source must be the git root, not a directory inside another project")
    index_entries = _git(repo, "ls-files", "--stage", "-z").split(b"\0")
    if any(entry.startswith(b"160000 ") for entry in index_entries):
        raise ValidationError("Submodules are unsupported in manifest v1; snapshot a standalone checkout")
    flags = _git(repo, "ls-files", "-v", "-z").split(b"\0")
    if any(entry and (entry[:1].islower() or entry[:1] == b"S") for entry in flags):
        raise ValidationError("assume-unchanged/skip-worktree index flags can hide edits and are unsupported")
    revision = _git(repo, "rev-parse", "--verify", "HEAD", required=False).decode().strip() or None
    diff = (_git(repo, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv")
            if revision else _git(repo, "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv"))
    # Index and worktree can cancel relative to HEAD, so hash each diff separately too.
    staged = _git(repo, "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv")
    unstaged = _git(repo, "diff", "--binary", "--no-ext-diff", "--no-textconv")
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    untracked: dict[str, dict[str, str]] = {}
    for raw_name in _git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw_name:
            continue
        name = os.fsdecode(raw_name)
        path = repo / name
        content = os.fsencode(os.readlink(path)) if path.is_symlink() else path.read_bytes()
        untracked[name] = {"kind": "symlink" if path.is_symlink() else "file",
                           "sha256": sha256_bytes(content)}
    snapshot = {"git_revision": revision, "dirty": bool(status),
                "tracked_diff_sha256": sha256_bytes(diff),
                "staged_diff_sha256": sha256_bytes(staged),
                "unstaged_diff_sha256": sha256_bytes(unstaged),
                "status_sha256": sha256_bytes(status),
                "untracked": untracked,
                "submodule_revisions": _git(repo, "submodule", "status", "--recursive").decode().splitlines()}
    snapshot["snapshot_sha256"] = digest(snapshot)
    return snapshot


def capture_environment(*, hardware: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect versions, never credentials, arbitrary environment variables or remote URLs."""
    if hardware is not None and not isinstance(hardware, dict):
        raise ValidationError("hardware must be an object")
    packages = sorted((distribution.metadata.get("Name", "unknown"), distribution.version)
                      for distribution in importlib.metadata.distributions())
    return {"python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "harness_version": __version__,
            "packages": [{"name": name, "version": version} for name, version in packages],
            "hardware": hardware or {"accelerator": "not_inspected"}}


def validate_manifest(payload: Any) -> None:
    require_keys(payload, MANIFEST_KEYS, "manifest")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValidationError("Unsupported manifest schema_version")
    nonempty_string(payload["run_id"], "run_id")
    choice(payload["evidence_kind"], EVIDENCE_KINDS, "evidence_kind")
    integer(payload["seed"], "seed")
    try:
        timestamp = datetime.fromisoformat(payload["created_at_utc"].replace("Z", "+00:00"))
        if timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
            raise ValueError("timestamp must include UTC offset")
    except (TypeError, AttributeError, ValueError) as exc:
        raise ValidationError("created_at_utc must be a timezone-aware UTC ISO timestamp") from exc
    source = payload["source"]
    require_keys(source, {"git_revision", "dirty", "tracked_diff_sha256", "staged_diff_sha256",
                         "unstaged_diff_sha256", "status_sha256", "untracked",
                         "submodule_revisions", "snapshot_sha256"}, "source")
    if source["git_revision"] is not None and not is_hex(source["git_revision"], (40, 64)):
        raise ValidationError("Invalid git revision")
    if not isinstance(source["dirty"], bool):
        raise ValidationError("source.dirty must be boolean")
    for key in ("tracked_diff_sha256", "staged_diff_sha256", "unstaged_diff_sha256",
                "status_sha256", "snapshot_sha256"):
        if not is_hex(source[key]):
            raise ValidationError(f"Invalid source.{key}")
    if not isinstance(source["untracked"], dict):
        raise ValidationError("source.untracked must be an object")
    for name, entry in source["untracked"].items():
        nonempty_string(name, "untracked path")
        require_keys(entry, {"kind", "sha256"}, "untracked entry")
        choice(entry["kind"], {"file", "symlink"}, "untracked.kind")
        if not is_hex(entry["sha256"]):
            raise ValidationError("Invalid untracked entry")
    if not isinstance(source["submodule_revisions"], list) or not all(
            isinstance(value, str) for value in source["submodule_revisions"]):
        raise ValidationError("submodule_revisions must be an array of strings")
    if digest({key: value for key, value in source.items() if key != "snapshot_sha256"}) != source["snapshot_sha256"]:
        raise ValidationError("Source snapshot checksum mismatch")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or not {"model", "dataset"} <= artifacts.keys():
        raise ValidationError("Pinned artifacts must include model and dataset")
    for role, artifact in artifacts.items():
        nonempty_string(role, "artifact role")
        require_keys(artifact, {"identifier", "revision_kind", "revision"}, f"artifact {role}")
        nonempty_string(artifact["identifier"], f"artifact {role}.identifier")
        kind = choice(artifact["revision_kind"], {"git_commit", "sha256"}, "revision_kind")
        if not is_hex(
                artifact["revision"], (40, 64) if kind == "git_commit" else (64,)):
            raise ValidationError(f"Artifact {role} needs an immutable commit or content SHA-256, not a branch/tag")
        if payload["evidence_kind"] == "real_model" and artifact["identifier"].startswith(("fixture:", "simulation:")):
            raise ValidationError("Fixture/simulation artifacts cannot be labeled real_model")
    if payload["evidence_kind"] == "real_model" and (source["git_revision"] is None or source["dirty"]):
        raise ValidationError("real_model runs require a committed, clean source tree")
    environment = payload["environment"]
    require_keys(environment, {"python_version", "python_implementation", "platform", "machine",
                               "harness_version", "packages", "hardware"}, "environment")
    for name in ("python_version", "python_implementation", "platform", "machine", "harness_version"):
        nonempty_string(environment[name], f"environment.{name}")
    if not isinstance(environment["packages"], list) or not isinstance(environment["hardware"], dict):
        raise ValidationError("Environment needs package list and hardware object")
    for package in environment["packages"]:
        require_keys(package, {"name", "version"}, "package")
        nonempty_string(package["name"], "package.name")
        nonempty_string(package["version"], "package.version")
    if not isinstance(payload["config"], dict):
        raise ValidationError("config must be an object")
    canonical_json(payload)


@dataclass(frozen=True)
class RunManifest:
    """Canonical JSON inside a frozen value; payload access returns a detached copy."""

    _canonical: str

    def __post_init__(self) -> None:
        payload = parse_json(self._canonical)
        validate_manifest(payload)
        object.__setattr__(self, "_canonical", canonical_json(payload))

    @property
    def payload(self) -> dict[str, Any]:
        return parse_json(self._canonical)

    @property
    def sha256(self) -> str:
        return sha256_bytes(self._canonical.encode("utf-8"))

    @property
    def run_id(self) -> str:
        return self.payload["run_id"]

    @classmethod
    def create(cls, *, repo: str | Path, evidence_kind: str, artifacts: dict[str, Any],
               config: dict[str, Any], seed: int,
               hardware: dict[str, Any] | None = None) -> "RunManifest":
        return cls(canonical_json({"schema_version": 1, "run_id": str(uuid4()),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(),
                   "evidence_kind": evidence_kind, "seed": seed,
                   "source": capture_source(repo), "artifacts": artifacts,
                   "environment": capture_environment(hardware=hardware), "config": config}))

    def save(self, path: str | Path) -> Path:
        return write_new(path, canonical_json({"manifest_sha256": self.sha256,
                                              "manifest": self.payload}) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        envelope = read_json(path)
        require_keys(envelope, {"manifest_sha256", "manifest"}, "manifest envelope")
        manifest = cls(canonical_json(envelope["manifest"]))
        if manifest.sha256 != envelope["manifest_sha256"]:
            raise ValidationError("Manifest checksum mismatch")
        return manifest
