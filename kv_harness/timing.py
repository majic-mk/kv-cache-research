"""Synchronous serial wall-clock instrumentation for adapters and CPU fixtures."""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

from .results import PHASE_COSTS, zero_timing


class SerialTimer:
    """CPU wall-clock only. GPU adapters must synchronize at documented boundaries.

    Disjoint named phases are subsets of the outer wall interval. Untagged Python,
    scheduling and instrumentation overhead remains included in end_to_end_s.
    """

    def __init__(self) -> None:
        self._values = zero_timing()
        self._start: int | None = None
        self._end: int | None = None
        self._active: str | None = None

    def __enter__(self) -> "SerialTimer":
        if self._start is not None:
            raise RuntimeError("Timers are single-use")
        self._start = time.perf_counter_ns()
        return self

    def __exit__(self, *exception: object) -> None:
        self._end = time.perf_counter_ns()

    @contextmanager
    def section(self, name: str) -> Iterator[None]:
        if name not in PHASE_COSTS:
            raise ValueError(f"Unknown phase: {name}")
        if self._start is None or self._end is not None:
            raise RuntimeError("Phase must be inside an active timer")
        if self._active is not None:
            raise RuntimeError("Nested/overlapping phases cannot use SerialTimer")
        self._active = name
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self._values[name] += (time.perf_counter_ns() - start) / 1_000_000_000
            self._active = None

    def snapshot(self) -> dict[str, float | str]:
        if self._start is None or self._end is None:
            raise RuntimeError("Timer must finish before snapshot")
        return {**self._values, "end_to_end_s": (self._end - self._start) / 1_000_000_000}
