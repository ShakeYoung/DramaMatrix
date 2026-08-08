"""Cross-scene limited concurrency for Agnes rendering (G1).

Shots are grouped into scene segments (consecutive shots sharing a scene_id).
Within a segment, shots MUST stay serial (tail-frame chain). Across segments,
rendering MAY run concurrently up to DRAMAMATRIX_MAX_IN_FLIGHT (default 1).

A shared CapacityThrottle records any queue_full / rate-limit event; once
degraded, in-flight workers keep finishing but no NEW submission starts until a
successful create resets the flag. This prevents thundering-herd against an
already-saturated Agnes queue.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from typing import Callable, Optional


def max_in_flight() -> int:
    """Configured cross-scene concurrency cap. 1 = fully serial (default)."""
    try:
        value = int(os.getenv("DRAMAMATRIX_MAX_IN_FLIGHT", "1"))
    except ValueError:
        value = 1
    return max(1, value)


@dataclass
class CapacityThrottle:
    """Shared throttle flag + lock for cross-scene rendering (G1).

    - `degraded` is set by any worker that sees a queue_full/rate-limit event.
    - `acquire_create()` blocks new submissions while degraded (returns False).
    - A successful create calls `report_success()` to clear the flag, allowing
      the next submission. This auto-recovers capacity without a manual reset.
    - All counters/guards are lock-protected for thread safety.
    """

    degraded: bool = False
    in_flight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def acquire_create(self) -> bool:
        """Reserve a creation slot. Returns False if currently degraded."""
        with self._lock:
            if self.degraded:
                return False
            self.in_flight += 1
            return True

    def report_success(self) -> None:
        with self._lock:
            self.degraded = False

    def report_queue_full(self) -> None:
        with self._lock:
            self.degraded = True

    def release_create(self) -> None:
        with self._lock:
            self.in_flight = max(0, self.in_flight - 1)


def run_segments(
    segments: list,
    segment_runner: Callable[[int], Optional[object]],
    workers: int,
) -> list:
    """Run scene segments with a bounded worker pool (G1).

    Each segment is independent (scene-boundary clears the tail-frame chain),
    so segments are safe to run in parallel. `segment_runner(index)` renders one
    segment serially and returns its outcome (or None). With workers==1 this is
    equivalent to the previous fully-serial behavior.

    Returns the per-segment outcomes in segment order.
    """
    if workers <= 1 or len(segments) <= 1:
        return [segment_runner(i) for i in range(len(segments))]
    outcomes: list = [None] * len(segments)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_index: dict[Future, int] = {
            pool.submit(segment_runner, i): i for i in range(len(segments))
        }
        for future in future_to_index:
            future_to_index[future]  # noop, keep mapping
        for future, index in future_to_index.items():
            outcomes[index] = future.result()
    return outcomes
