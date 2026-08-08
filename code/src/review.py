"""Manual review checkpoint (E1).

Generates a file-based review manifest per episode listing each shot with its
thumbnail path, QC result, issues, duration, and hash. The human reviews the
manifest and marks each shot .approve / .redraw / .delete; a re-run honors those
marks before proceeding to Agent6.

The design is file-based (JSON + a sidecar decisions dict) so it needs no new
dependencies and works for a single-operator studio.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agnes_video import episode_output_dir, sha256_file
from src.state import EpisodeState


def review_mode_enabled() -> bool:
    import os
    return os.getenv("DRAMAMATRIX_REVIEW_MODE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _review_dir(project_id: str, ep_key: str) -> Path:
    directory = episode_output_dir(project_id, ep_key) / "review"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def build_review_manifest(project_id: str, ep_key: str, ep_state: EpisodeState) -> dict[str, Any]:
    """Build a review manifest from the rendered shot assets."""
    assets_by_shot = {a.shot_id: a for a in ep_state.video_assets}
    shots: list[dict[str, Any]] = []
    for shot in ep_state.storyboard_data:
        asset = assets_by_shot.get(shot.shot_id)
        shots.append({
            "shot_id": shot.shot_id,
            "scene_id": shot.scene_id,
            "thumbnail": f"{ep_key}_{shot.shot_id}_head.png",  # head frame thumb
            "actual_duration": asset.actual_duration if asset else None,
            "sha256": asset.sha256 if asset else None,
            "local_path": asset.local_path if asset else None,
            "status": asset.status if asset else "missing",
            "qc_issues": [fb.message for fb in ep_state.feedback_log if shot.shot_id in (fb.message or "") and fb.reason_code == "QC_REDRAW"],
        })
    manifest = {
        "project_id": project_id,
        "ep_key": ep_key,
        "generated_at": __import__("time").time(),
        "shots": shots,
    }
    return manifest


def write_review_manifest(project_id: str, ep_key: str, ep_state: EpisodeState) -> Path:
    """Write the review manifest + an initially-empty decisions dict."""
    directory = _review_dir(project_id, ep_key)
    manifest = build_review_manifest(project_id, ep_key, ep_state)
    manifest_path = directory / "review.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    decisions_path = directory / "decisions.json"
    if not decisions_path.exists():
        decisions_path.write_text("{}", encoding="utf-8")
    return manifest_path


def load_decisions(project_id: str, ep_key: str) -> dict[str, str]:
    """Load per-shot decisions {shot_id: 'approve'|'redraw'|'delete'|''}."""
    decisions_path = _review_dir(project_id, ep_key) / "decisions.json"
    if not decisions_path.exists():
        return {}
    try:
        data = json.loads(decisions_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def all_decided(project_id: str, ep_key: str, ep_state: EpisodeState) -> bool:
    """Whether every shot has an explicit approve/redraw/delete decision."""
    decisions = load_decisions(project_id, ep_key)
    shot_ids = {s.shot_id for s in ep_state.storyboard_data}
    if not shot_ids:
        return True
    return shot_ids.issubset(set(decisions.keys()))


def pending_shot_ids(project_id: str, ep_key: str, ep_state: EpisodeState, decision: str | None = None) -> list[str]:
    """Shots matching a decision (or all decided shots if decision is None)."""
    decisions = load_decisions(project_id, ep_key)
    if decision is None:
        return [s.shot_id for s in ep_state.storyboard_data if s.shot_id in decisions]
    return [s.shot_id for s in ep_state.storyboard_data if decisions.get(s.shot_id) == decision]