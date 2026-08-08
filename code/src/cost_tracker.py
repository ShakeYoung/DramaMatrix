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
import time
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
    # F3：排队/渲染/下载耗时与重绘次数（扩展字段）
    queue_wait_seconds: Optional[float] = None
    render_seconds: Optional[float] = None
    download_seconds: Optional[float] = None
    redraw_count: int = 0
    # E5：记录使用哪个视频供应商
    provider: Optional[str] = None


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
    project_id: str = ""

    def __post_init__(self) -> None:
        # Durable budget (F8): when constructed with a report_path, auto-seed
        # prior_count from the persisted file so the limit holds across restarts,
        # regardless of construction path (direct or from_environment()).
        if self.project_id:
            try:
                from src.db import db_count_agnes_usage

                database_count = db_count_agnes_usage(self.project_id)
                csv_count = (
                    self._count_existing_rows(self.report_path, project_id=self.project_id)
                    if self.report_path
                    else 0
                )
                # Upgrade compatibility: old runs only wrote CSV. Taking the
                # larger count preserves their budget without double-counting
                # once both stores contain the same records.
                self.prior_count = max(database_count, csv_count)
            except Exception:
                if self.report_path:
                    self.prior_count = self._count_existing_rows(
                        self.report_path, project_id=self.project_id
                    )
        elif self.report_path:
            self.prior_count = self._count_existing_rows(self.report_path)

    @classmethod
    def from_environment(cls, project_id: str = "") -> "CostTracker":
        report_path = output_root() / "agnes_usage_report.csv"
        return cls(
            max_creates=int(os.getenv("DRAMAMATRIX_MAX_AGNES_CREATES", "0") or 0),
            report_path=report_path,
            project_id=project_id,
        )

    @staticmethod
    def _count_existing_rows(report_path: Path, project_id: str = "") -> int:
        if not report_path.is_file():
            return 0
        try:
            with report_path.open(encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                return sum(
                    1
                    for row in reader
                    if not project_id or row.get("project_id") == project_id
                )
        except (OSError, csv.Error):
            return 0

    @property
    def create_count(self) -> int:
        # real records from this run + persisted count from prior runs
        return len(self.records) + self.prior_count

    def record_create(self, **kwargs) -> None:
        kwargs.setdefault("created_at_unix", time.time())
        record = AgnesUsageRecord(**kwargs)
        if any(existing.task_id == record.task_id for existing in self.records):
            return
        if self.project_id:
            try:
                from src.db import db_record_agnes_usage

                if not db_record_agnes_usage(**asdict(record)):
                    # The task was already durably accounted for by a prior run.
                    return
            except Exception as exc:
                # 资产快照仍由 Agent5 保存；CSV 会在节点结束时作为第二份审计记录。
                print(f"⚠️ Agnes 用量写入 SQLite 失败，将保留 CSV 报表：{exc}")
        self.records.append(record)

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
