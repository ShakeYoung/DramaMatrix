"""Shot continuity helpers (P1-A / P1-C).

Provides end_state normalization across consecutive shots and the wiring for
conditional (chain) generation: per-scene reference images, last-frame
extraction from the previous shot, and per-scene seed selection.

Continuity is enforced as a SOFT constraint: instead of failing when an LLM
produces inconsistent start/end states, we normalize shot[i].end_state to match
shot[i+1].start_state and record warnings. Strong equality would fail too often.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from src.state import ShotBoundaryState, ShotStoryboard


@dataclass
class ContinuityWarning:
    shot_id: str
    field: str
    message: str


def _boundary_from_shot(shot: ShotStoryboard) -> ShotBoundaryState:
    """Build a boundary state from a shot's flat continuity fields."""
    return ShotBoundaryState(
        pose=shot.start_pose or "",
        subject_position=shot.subject_position or "",
        gaze_direction=shot.gaze_direction or "",
        light_direction=shot.light_direction or shot.start_state.light_direction if shot.start_state else (shot.light_direction or ""),
        color_temperature=shot.color_temperature or (shot.start_state.color_temperature if shot.start_state else ""),
    )


def normalize_continuity(
    shots: Sequence[ShotStoryboard],
) -> list[ContinuityWarning]:
    """Normalize consecutive-shot boundary states in place (P1-A).

    For each adjacent pair, set shot[i].end_state = shot[i+1].start_state so the
    editor/generator sees a consistent handoff. Missing scene_id is also filled
    by propagating the previous shot's scene_id. Returns collected warnings
    instead of raising — LLM output is too noisy for hard constraints.
    """
    warnings: list[ContinuityWarning] = []
    if not shots:
        return warnings

    # Backfill start_state from flat fields when absent.
    for shot in shots:
        if shot.start_state is None:
            shot.start_state = _boundary_from_shot(shot)
        if not shot.scene_id:
            warnings.append(ContinuityWarning(shot.shot_id, "scene_id", "缺失 scene_id，将沿用上一镜场景"))

    # Propagate scene_id forward when missing.
    last_scene: Optional[str] = None
    for shot in shots:
        if shot.scene_id:
            last_scene = shot.scene_id
        elif last_scene:
            shot.scene_id = last_scene

    # Normalize end_state of each shot to the next shot's start_state.
    for i in range(len(shots) - 1):
        current = shots[i]
        nxt = shots[i + 1]
        nxt_start = nxt.start_state or _boundary_from_shot(nxt)
        nxt.start_state = nxt_start
        current.end_state = nxt_start.model_copy()
        current.previous_shot_id = shots[i - 1].shot_id if i > 0 else None
        # If two adjacent shots share a scene but disagree on key env fields, warn.
        if current.scene_id and nxt.scene_id and current.scene_id == nxt.scene_id:
            for field_name in ("light_direction", "color_temperature", "time_of_day"):
                a = getattr(current, field_name, None)
                b = getattr(nxt, field_name, None)
                if a and b and a != b:
                    warnings.append(
                        ContinuityWarning(
                            current.shot_id,
                            field_name,
                            f"同场景 {current.scene_id} 相邻镜 {field_name} 不一致：{a} vs {b}（已按下一镜归一化）",
                        )
                    )

    # Last shot: end_state falls back to its own start_state.
    last = shots[-1]
    last.previous_shot_id = shots[-2].shot_id if len(shots) > 1 else None
    if last.end_state is None:
        last.end_state = (last.start_state or _boundary_from_shot(last)).model_copy()
    return warnings


def conditional_generation_enabled() -> bool:
    """Whether chain/conditional generation (first-frame passing) is on (P1-C)."""
    return os.getenv("DRAMAMATRIX_CONDITIONAL_GENERATION", "0").strip().lower() in {"1", "true", "yes", "on"}


def scene_seed(base: int, scene_id: Optional[str], scene_index: int) -> int:
    """Stable per-scene seed: same scene => same seed across its shots (P1-C)."""
    if not scene_id:
        return base + scene_index
    # Deterministic hash of scene_id folded into the seed space.
    h = sum(ord(c) for c in scene_id)
    return base + (h % 1000) + scene_index * 0  # scene-stable; index not added so all shots in scene share


def group_by_scene(shots: Sequence[ShotStoryboard]) -> list[list[ShotStoryboard]]:
    """Group consecutive shots sharing a scene_id (P1-C)."""
    groups: list[list[ShotStoryboard]] = []
    current: list[ShotStoryboard] = []
    current_scene: Optional[str] = None
    for shot in shots:
        if shot.scene_id != current_scene:
            if current:
                groups.append(current)
            current = [shot]
            current_scene = shot.scene_id
        else:
            current.append(shot)
    if current:
        groups.append(current)
    return groups


def prepare_shot_reference(
    shot: ShotStoryboard,
    scene_references: dict[str, str],
    previous_last_frame: Optional[str],
) -> dict[str, Optional[str]]:
    """Decide the conditional inputs for a shot (P1-C).

    - First shot of a scene: use the scene's fixed reference image (image_url).
    - Subsequent shot in the same scene: use the previous shot's last frame
      (image_url pointing at the extracted frame) so identity/wardrobe carry over.
    Returns {'image_url': ..., 'last_frame_url': ...} (any may be None).
    """
    result = {"image_url": None, "last_frame_url": None}
    scene_id = shot.scene_id
    if previous_last_frame:
        # Chain from the previous shot's tail frame.
        result["image_url"] = previous_last_frame
    elif scene_id and scene_id in scene_references:
        result["image_url"] = scene_references[scene_id]
    return result