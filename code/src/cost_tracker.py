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
    """Accumulates Agnes usage and enforces a submission budget guardrail.

    The budget is DURABLE across runs (review F8): the create_count is seeded
    from the persisted usage report at construction, and the report is written
    in append mode so prior usage is never overwritten/lost on restart.
    """

    max_creates: int = 0  # 0 => unlimited
    records: list = field(default_factory=list)
    report_path: Optional[Path] = None
    prior_count: int = 0  # usage persisted by previous runs

    def __post_init__(self) -> None:
        # Durable budget (F8): when constructed with a report_path, auto-seed
        # prior_count from the persisted file so the limit holds across restarts,
        # regardless of construction path (direct or from_environment()).
        if self.report_path:
            self.prior_count = self._count_existing_rows(self.report_path)

    @classmethod
    def from_environment(cls) -> "CostTracker":
        report_path = output_root() / "agnes_usage_report.csv"
        return cls(
            max_creates=int(os.getenv("DRAMAMATRIX_MAX_AGNES_CREATES", "0") or 0),
            report_path=report_path,
            prior_count=cls._count_existing_rows(report_path),
        )

    @staticmethod
    def _count_existing_rows(report_path: Path) -> int:
        if not report_path.is_file():
            return 0
        try:
            with report_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return sum(1 for _ in reader)
        except (OSError, csv.Error):
            return 0

    @property
    def create_count(self) -> int:
        # real records from this run + persisted count from prior runs
        return len(self.records) + self.prior_count

    def record_create(self, **kwargs) -> None:
        self.records.append(AgnesUsageRecord(**kwargs))

    def budget_exhausted(self) -> bool:
        return self.max_creates > 0 and self.create_count >= self.max_creates

    def write_report(self, directory: Optional[Path] = None) -> Path:
        """Persist usage to report.csv in APPEND mode so history is retained."""
        directory = directory or output_root()
        directory.mkdir(parents=True, exist_ok=True)
        report_path = self.report_path or (directory / "agnes_usage_report.csv")
        fieldnames = list(AgnesUsageRecord.__dataclass_fields__.keys())
        write_header = not report_path.exists() or report_path.stat().st_size == 0
        with report_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        return report_path