"""Failure report generation (E4).

Scans the persisted project state for terminal/blocked episodes and produces a
JSON report listing which shots failed/missing, the status, and a suggested
next action — for manual disposition instead of silent --resume loops.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agnes_video import episode_output_dir

# Map terminal/blocked episode status -> suggested disposition.
_DISPOSITION = {
    "render_failed": "检查 Agnes 配置/预算后重试，或人工删除该镜资产后 --resume",
    "render_partial": "解除镜头上限后 --resume 继续渲染剩余镜头",
    "waiting_for_agnes_capacity": "等待队列恢复（next_retry_at 已持久化）后 --resume",
    "waiting_for_connectivity": "网络恢复后 --resume 续跑",
    "submission_uncertain": "到 Agnes 控制台核对任务后人工确认",
    "storyboard_blocked": "修正分镜（数量/生成失败）后重置状态",
    "awaiting_review": "人工审阅 review.json 并标记后重跑（请用 review_approver）",
    "director_rejected": "分镜需重写，检查 QC/内容策略反馈",
    "editing_failed": "检查 ffmpeg/素材完整性",
    "growth_failed": "检查投流切片导出相关 ffmpeg/路径",
}


def build_failure_report(state: dict[str, Any]) -> dict[str, Any]:
    """Scan episodes and summarize blocked/terminal shots with dispositions."""
    report: dict[str, Any] = {
        "project_id": state.get("project_id"),
        "system_status": state.get("system_status"),
        "blocked_episodes": [],
    }
    blocked_statuses = set(_DISPOSITION.keys())
    for ep_key, ep in (state.get("episodes") or {}).items():
        if ep.status not in blocked_statuses:
            continue
        entry = {
            "ep_key": ep_key,
            "status": ep.status,
            "disposition": _DISPOSITION.get(ep.status, "人工检查"),
            "missing_shots": [],
            "blocked_shots": [],
        }
        # Missing/completed vs pending shot assets.
        available = {a.shot_id for a in ep.video_assets if a.local_path}
        for shot in ep.storyboard_data:
            if shot.shot_id not in available:
                entry["missing_shots"].append(shot.shot_id)
        for fb in ep.feedback_log:
            if fb.reason_code in {"QC_REDRAW", "AGNES_RENDER_FAILED", "GROWTH_EXPORT_FAILED", "FFMPEG_EDIT_FAILED"}:
                entry["blocked_shots"].append({"reason": fb.message})
        report["blocked_episodes"].append(entry)
    return report


def write_failure_report(state: dict[str, Any], output_dir: Path | None = None) -> Path | None:
    """Write failure_report.json next to the project output; return its path."""
    report = build_failure_report(state)
    if not report["blocked_episodes"]:
        return None
    project_id = state.get("project_id")
    base = output_dir or (episode_output_dir(project_id, "report") if project_id else Path.cwd())
    base.mkdir(parents=True, exist_ok=True)
    path = base / "failure_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path