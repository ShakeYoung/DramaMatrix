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
    "awaiting_review": "后台模式：完成 review.json 决策后用同一项目续跑；前台可使用 interactive 模式",
    "director_rejected": "分镜需重写，检查 QC/内容策略反馈",
    "editing_failed": "检查 ffmpeg/素材完整性后 --resume（已支持从 Agent 6 断点重试）",
    "growth_failed": "检查投流切片导出相关 ffmpeg/路径/交付包完整性后 --resume（已支持从 Agent 7 断点重试）",
    "awaiting_episode_review": "整集验收：核对 episode_review.json（对白完整性/字幕对齐/溢出标记）后 python -m src.episode_review <project> <ep> approve|rework",
}

# R1：系统级阻塞（无 episode 级状态）的修复指引。
_SYSTEM_DISPOSITION = {
    "blocked_on_source": "production 模式缺少真实书源：配置 DRAMAMATRIX_LOCAL_NOVEL_DIR 后 --resume；演示用途设 DRAMAMATRIX_RUN_MODE=demo",
    "blocked_on_text_model": "评审模型调用失败：配置文本模型密钥（OPENAI_API_KEY 或 TEXT_MODEL_*）后 --resume",
    "blocked_on_script": "编剧模型调用失败：配置文本模型密钥后 --resume（production 模式不使用固定回退剧情）",
    "blocked_on_missing_character_bible": "角色圣经为空：检查文本模型与 Agent3 角色抽取；确认可接受无一致性约束后设 DRAMAMATRIX_ALLOW_NO_CHARACTERS=1（demo 模式自动放行）",
    "waiting_for_market_data": "等待真实投放数据：配置 DRAMAMATRIX_ANALYTICS_IMPORT（平台导出 CSV/JSON）后 --resume；演示用途设 DRAMAMATRIX_RUN_MODE=demo",
    "waiting_for_episode_review": "整集验收等待人工核片：查看 episode_review.json 后 python -m src.episode_review <project> <ep> approve|rework",
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
    # R1：blocked_on_source/blocked_on_text_model/blocked_on_script/
    # waiting_for_market_data 等系统级阻塞没有 episode 级状态，单独记录。
    system_status = state.get("system_status", "") or ""
    if not report["blocked_episodes"] and system_status.startswith(("blocked_", "waiting_", "failed")):
        report["system_block"] = {
            "status": system_status,
            "disposition": _SYSTEM_DISPOSITION.get(
                system_status, "检查运行配置/密钥后 --resume，或查阅运行日志定位"
            ),
        }
    return report


def write_failure_report(state: dict[str, Any], output_dir: Path | None = None) -> Path | None:
    """Write failure_report.json next to the project output; return its path."""
    report = build_failure_report(state)
    if not report["blocked_episodes"] and not report.get("system_block"):
        return None
    project_id = state.get("project_id")
    base = output_dir or (episode_output_dir(project_id, "report") if project_id else Path.cwd())
    base.mkdir(parents=True, exist_ok=True)
    path = base / "failure_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
