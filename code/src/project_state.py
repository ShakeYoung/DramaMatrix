"""Project-state construction and safe restoration from SQLite snapshots."""

from __future__ import annotations

from typing import Any

from src.state import CharacterSheet, EpisodeState, EvaluationReport, MarketFeedback


def new_project_state(project_id: str) -> dict[str, Any]:
    return {
        "project_id": project_id,
        "meta_info": {"source_title": "待定", "genre_tags": []},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {},
        "task_cycle": 1,
        "scout_attempts": 0,
        "characters": [],
        "run_context": None,
        "system_status": "starting",
    }


def restore_project_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Rehydrate Pydantic values needed by agents from a JSON snapshot."""
    state = snapshot["state"]
    state["episodes"] = {
        key: EpisodeState.model_validate(value)
        for key, value in state.get("episodes", {}).items()
    }
    report = state.get("source_material", {}).get("report")
    if isinstance(report, dict):
        state["source_material"]["report"] = EvaluationReport.model_validate(report)
    feedback = state.get("market_feedback")
    if isinstance(feedback, dict):
        state["market_feedback"] = MarketFeedback.model_validate(feedback)
    # 恢复角色一致性表（阶段3/F4）：否则后续 render_character_block 访问
    # ch.name 会对 dict 抛 AttributeError。
    if isinstance(state.get("characters"), list):
        state["characters"] = [
            CharacterSheet.model_validate(ch) if isinstance(ch, dict) else ch
            for ch in state["characters"]
        ]
    elif state.get("characters") is None:
        state["characters"] = []
    return state
