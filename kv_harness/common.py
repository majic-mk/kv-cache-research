"""Strict JSON and content-addressed, no-overwrite file primitives."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


class ValidationError(ValueError):
    """Input fails the versioned evidence contract."""


def canonical_json(value: Any) -> str:
    def check(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValidationError("JSON object keys must be strings")
                check(key)
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeError as exc:
                raise ValidationError("JSON strings must be valid UTF-8") from exc
        elif item is not None and type(item) not in (bool, int, float):
            raise ValidationError(f"Unsupported JSON type: {type(item).__name__}")
    check(value)
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"Not canonical JSON: {exc}") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ValidationError(f"Non-finite JSON value: {value}")))
        canonical_json(value)  # Includes 1e999 and malformed Unicode escapes.
        return value
    except json.JSONDecodeError as exc:
        raise ValidationError(f"Invalid JSON: {exc}") from exc


def read_json(path: str | Path) -> Any:
    try:
        return parse_json(Path(path).read_text(encoding="utf-8"))
    except UnicodeError as exc:
        raise ValidationError("JSON input must be valid UTF-8") from exc


def require_keys(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    missing, extra = keys - value.keys(), value.keys() - keys
    if missing or extra:
        raise ValidationError(f"{name} fields: missing={sorted(missing)}, extra={sorted(extra)}")


def nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{name} must be a nonempty string")
    return value


def choice(value: Any, values: set[str], name: str) -> str:
    nonempty_string(value, name)
    if value not in values:
        raise ValidationError(f"{name} must be one of {sorted(values)}")
    return value


def finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValidationError(f"{name} is too large") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValidationError(f"{name} must be finite and >= {minimum}")
    return number


def integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{name} must be an integer >= {minimum}")
    return value


def is_hex(value: Any, lengths: tuple[int, ...] = (64,)) -> bool:
    return (isinstance(value, str) and len(value) in lengths
            and all(char in "0123456789abcdef" for char in value))


def write_new(path: str | Path, content: str) -> Path:
    """Publish a complete file atomically; never replace an existing pathname.

    Hard-link publication prevents a concurrent writer from being overwritten.
    This is write-once by convention, not a malicious-tamper security boundary.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".publish-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return path
