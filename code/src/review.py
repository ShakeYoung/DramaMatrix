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
import os
import sys
from pathlib import Path
from typing import Any

from src.agnes_video import episode_output_dir, sha256_file
from src.state import EpisodeState


VALID_DECISIONS = {"approve", "redraw", "delete"}


def review_mode() -> str:
    """Return ``interactive``, ``background`` or ``off``.

    Legacy boolean values remain supported: truthy values map to background
    review (safe for nohup), while falsy values disable the checkpoint.
    """
    value = os.getenv("DRAMAMATRIX_REVIEW_MODE", "background").strip().lower()
    if value in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if value in {"interactive", "foreground", "terminal"}:
        return "interactive"
    return "background"


def review_mode_enabled() -> bool:
    return review_mode() != "off"


def interactive_review_available() -> bool:
    """Whether this process owns an interactive terminal.

    nohup redirects stdin from ``/dev/null``; prompting in that situation would
    hang or raise EOFError, so interactive mode degrades to background review.
    """
    return bool(getattr(sys.stdin, "isatty", lambda: False)())


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
        thumbnail = None
        if asset and asset.local_path:
            video_path = Path(asset.local_path)
            head_path = video_path.with_name(f"{video_path.stem}_head.png")
            if head_path.is_file():
                thumbnail = str(head_path)
        shots.append({
            "shot_id": shot.shot_id,
            "scene_id": shot.scene_id,
            "thumbnail": thumbnail,
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
        if not isinstance(data, dict):
            return {}
        return {
            str(shot_id): str(decision).strip().lower()
            for shot_id, decision in data.items()
            if str(decision).strip().lower() in VALID_DECISIONS
        }
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


def save_decisions(project_id: str, ep_key: str, decisions: dict[str, str]) -> Path:
    """Persist validated decisions and return the sidecar path."""
    path = _review_dir(project_id, ep_key) / "decisions.json"
    clean = {
        str(shot_id): str(decision).strip().lower()
        for shot_id, decision in decisions.items()
        if str(decision).strip().lower() in VALID_DECISIONS
    }
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_review_decisions(
    project_id: str,
    ep_key: str,
    ep_state: EpisodeState,
) -> str:
    """Apply a complete review and return the resulting episode status.

    State transitions are deliberately centralized here so foreground and
    background review have identical behaviour:

    - all remaining shots approved/deleted -> ``video_generated``;
    - one or more shots requested for redraw -> ``storyboard_done``;
    - every shot deleted -> ``render_failed``;
    - incomplete decisions -> ``awaiting_review``.
    """
    if not all_decided(project_id, ep_key, ep_state):
        ep_state.status = "awaiting_review"
        return ep_state.status

    decisions = load_decisions(project_id, ep_key)
    redraw_ids = {sid for sid, decision in decisions.items() if decision == "redraw"}
    delete_ids = {sid for sid, decision in decisions.items() if decision == "delete"}

    if delete_ids:
        ep_state.storyboard_data = [
            shot for shot in ep_state.storyboard_data if shot.shot_id not in delete_ids
        ]
        ep_state.video_assets = [
            asset for asset in ep_state.video_assets if asset.shot_id not in delete_ids
        ]

    if redraw_ids:
        ep_state.video_assets = [
            asset for asset in ep_state.video_assets if asset.shot_id not in redraw_ids
        ]
        # A redrawn shot must be reviewed again; keep approvals for untouched
        # shots and consume the redraw decisions now.
        save_decisions(
            project_id,
            ep_key,
            {sid: decision for sid, decision in decisions.items() if sid not in redraw_ids},
        )

    if not ep_state.storyboard_data:
        ep_state.status = "render_failed"
    elif redraw_ids:
        ep_state.status = "storyboard_done"
    else:
        ep_state.status = "video_generated"
    return ep_state.status
