"""Agnes usage tracking and cost guardrail (阶段4).

Each created Agnes task / downloaded shot is recorded to an in-memory tracker
and (optionally) flushed to a CSV report in the output directory. A budget cap
can hard-stop new submissions once estimated usage exceeds the configured limit.

The module is intentionally side-effect-light and thread-free so tests can
cover it without a real API.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from src.agnes_video import output_root


@dataclass
class AgnesUsageRecord:
    project_id: str
    ep_key: str
    shot_id: str
    task_id: str
    frames: int
    width: int
    height: int
    created_at_unix: Optional[float] = None


@dataclass
class CostTracker:
    """Accumulates Agnes usage and enforces a submission budget guardrail."""

    max_creates: int = 0  # 0 => unlimited
    records: list = field(default_factory=list)

    @classmethod
    def from_environment(cls) -> "CostTracker":
        return cls(
            max_creates=int(os.getenv("DRAMAMATRIX_MAX_AGNES_CREATES", "0") or 0),
        )

    @property
    def create_count(self) -> int:
        return len(self.records)

    def record_create(self, **kwargs) -> None:
        self.records.append(AgnesUsageRecord(**kwargs))

    def budget_exhausted(self) -> bool:
        return self.max_creates > 0 and self.create_count >= self.max_creates

    def write_report(self, directory: Optional[Path] = None) -> Path:
        """Flush records to report.csv under the output root (or given dir)."""
        directory = directory or output_root()
        directory.mkdir(parents=True, exist_ok=True)
        report_path = directory / "agnes_usage_report.csv"
        fieldnames = list(AgnesUsageRecord.__dataclass_fields__.keys())
        with report_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        return report_path