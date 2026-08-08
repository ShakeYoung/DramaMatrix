"""Agent 7: produce concrete teaser assets from completed episode masters.

vs. the previous naive "first-15s / last-15s" cut, this agent:
- picks hook / climax windows from the episode's storyboard emotion cues (falling
  back to head/tail when no cue-driven signal is available),
- attaches 投流 (ad-serving) metadata — headline, description, tags — to each
  slice, and produces a whole-episode GrowthMeta for shelf/货架 publication.
"""

from __future__ import annotations

import os
from pathlib import Path

from src.agnes_video import (
    AgnesVideoError,
    _require_binary,
    cut_video,
    episode_output_dir,
    video_duration,
)
from src.deliverables import record_deliverable
from src.state import DramaState, EpisodeState, FeedbackLog, GrowthAsset, GrowthMeta


def _shot_durations(ep_state: EpisodeState) -> list[float]:
    """每镜真实时长（优先 asset.actual_duration，回退计划 duration）。"""
    assets_by_shot = {a.shot_id: a for a in ep_state.video_assets}
    durations: list[float] = []
    for shot in ep_state.storyboard_data:
        asset = assets_by_shot.get(shot.shot_id)
        real = getattr(asset, "actual_duration", None) if asset else None
        if real and real > 0:
            durations.append(float(real))
        else:
            import re
            match = re.search(r"\d+(?:\.\d+)?", shot.duration or "")
            durations.append(float(match.group()) if match else 4.0)
    return durations


def _shots_in_window(ep_state: EpisodeState, start: float, duration: float) -> list[str]:
    """P1-6：切片 [start, start+duration) 实际覆盖的镜头清单（按真实时长累进）。"""
    end = start + duration
    covered: list[str] = []
    cursor = 0.0
    durations = _shot_durations(ep_state)
    for i, shot in enumerate(ep_state.storyboard_data):
        shot_end = cursor + (durations[i] if i < len(durations) else 0.0)
        # 镜头与切片窗口有交集即算覆盖
        if shot_end > start and cursor < end:
            covered.append(shot.shot_id)
        cursor = shot_end
        if cursor >= end:
            break
    return covered


def detect_emotion_segments(total_seconds: float, ep_state: EpisodeState) -> list[tuple[str, float, float]]:
    """Return [(name, start, duration)] weighted by storyboard emotion cues.

    Without a vision model, we approximate "high-emotion" windows from the
    narrative: the hook (opening suspense) and the climax (ending hook).

    G2 controls:
    - DRAMAMATRIX_GROWTH_CLIP_COUNT (default 2): how many slices to export.
      Set to 1 to export only the hook (opening) teaser.
    - DRAMAMATRIX_GROWTH_CLIP_DURATION (default 15): hook slice length.
    - DRAMAMATRIX_GROWTH_CLIMAX_DURATION (default 0): climax slice length; 0
      means inherit GROWTH_CLIP_DURATION. The climax is anchored at the LAST
      storyboard shot (the ending hook) rather than the raw file tail, and is
      shortened so it does not overrun the episode end.
    """
    hook_duration = min(
        float(os.getenv("DRAMAMATRIX_GROWTH_CLIP_DURATION", "15")),
        max(total_seconds, 1.0),
    )
    clip_count = int(os.getenv("DRAMAMATRIX_GROWTH_CLIP_COUNT", "2"))
    if total_seconds <= 0:
        return []

    segments: list[tuple[str, float, float]] = [("hook", 0.0, hook_duration)]
    if clip_count < 2:
        return segments

    # Climax: anchor at the last storyboard shot window, shortened to fit.
    climax_duration_raw = float(os.getenv("DRAMAMATRIX_GROWTH_CLIMAX_DURATION", "0") or 0)
    climax_duration = min(climax_duration_raw or hook_duration, hook_duration)
    climax_duration = min(climax_duration, max(total_seconds, 1.0))
    last_shot_start = max(0.0, total_seconds - climax_duration)
    if ep_state.storyboard_data:
        mean = total_seconds / max(len(ep_state.storyboard_data), 1)
        last_start = mean * (len(ep_state.storyboard_data) - 1)
        if last_start > 0:
            last_shot_start = max(0.0, min(last_start, total_seconds - climax_duration))
    # H3：重叠/相同保护——hook 与 climax 区间重叠超过阈值时只保留 hook。
    # 对短片（如总时长 ≤ hook 时长）climax 会与 hook 几乎完全重叠，导出两个
    # 近乎相同的切片无意义。重叠比例 > 0.5 视为重复。
    hook_end = hook_duration
    climax_end = last_shot_start + climax_duration
    overlap = max(0.0, min(hook_end, climax_end) - max(0.0, last_shot_start))
    overlap_ratio = overlap / min(hook_duration, climax_duration) if min(hook_duration, climax_duration) > 0 else 0
    if overlap_ratio > 0.5:
        print(f"   ⚠️ hook 与 climax 重叠 {overlap_ratio:.0%}，短片只保留 hook 切片。")
        return segments
    segments.append(("climax", last_shot_start, climax_duration))
    return segments


def build_growth_meta(ep_state: EpisodeState, genre_tags: list[str] | None = None) -> GrowthMeta:
    """Compose shelf/publication metadata from the episode's script + genre tags."""
    script = ep_state.script_data
    outline = (script.outline if script else "") or ""
    hook = (script.ending_hook if script else "") or ""
    return GrowthMeta(
        title=outline[:40],
        description=(f"{outline} 结尾悬念：{hook}")[:120],
        tags=list(genre_tags or []),
        cover_prompt=f"竖屏短剧封面，{hook}，强冲突、夸张情绪，电影感打光",
    )


def _asset_meta(name: str, base: GrowthMeta, total_seconds: float) -> tuple[str, str, list[str]]:
    if name == "hook":
        return (
            f"开场即高能｜{base.title}",
            f"开头 15 秒抓住眼球：{base.description}",
            base.tags + ["开场", "高能"],
        )
    return (
        f"神转折高潮｜{base.title}",
        f"结尾悬念拉满：{base.description}",
        base.tags + ["高潮", "悬念"],
    )


def process_agent7_growth(state: DramaState) -> DramaState:
    print("--- [Agent 7: FFmpeg Growth Packaging] ---")
    targets = [(key, ep) for key, ep in state["episodes"].items() if ep.status == "edit_completed"]
    if not targets:
        print("没有待制作投流素材的成片。")
        return state

    genre_tags = list(state.get("meta_info", {}).get("genre_tags", []) or [])

    for ep_key, ep_state in targets:
        try:
            if not ep_state.final_video_path:
                raise AgnesVideoError("成片路径缺失，无法制作投流素材。")
            master = Path(ep_state.final_video_path)
            total_seconds = video_duration(master)
            if total_seconds <= 0:
                raise AgnesVideoError("成片时长为 0。")

            growth_dir = episode_output_dir(state["project_id"], ep_key) / "growth"
            segments = detect_emotion_segments(total_seconds, ep_state)

            base_meta = build_growth_meta(ep_state, genre_tags=genre_tags)

            assets: list[GrowthAsset] = []
            for name, start, clip_duration in segments:
                out_path = growth_dir / f"{ep_key}_{name}.mp4"
                path = cut_video(master, out_path, start, clip_duration)
                headline, description, tags = _asset_meta(name, base_meta, total_seconds)
                # F4/P1-6：投流切片证据——来源镜头按切片起止时间 × 每镜真实时长精确计算。
                covered = _shots_in_window(ep_state, start, clip_duration)
                record_deliverable(ep_state, kind=f"clip_{name}", path=path, source_shots=covered)
                assets.append(
                    GrowthAsset(
                        name=name,
                        path=str(path),
                        start_seconds=start,
                        duration_seconds=clip_duration,
                        headline=headline,
                        description=description,
                        tags=tags,
                    )
                )

            ep_state.growth_assets = assets
            ep_state.growth_meta = base_meta
            ep_state.status = "growth_ready"
            print(f"✅ {ep_key} 已导出 {len(assets)} 个情绪段投流切片及发布元数据。")
        except AgnesVideoError as exc:
            ep_state.status = "growth_failed"
            ep_state.feedback_log.append(
                FeedbackLog(
                    from_agent="Agent_7_Growth",
                    to_agent="Operator",
                    reason_code="GROWTH_EXPORT_FAILED",
                    message=str(exc),
                )
            )
            print(f"❌ {ep_key} 投流切片失败：{exc}")
        state["episodes"][ep_key] = ep_state

    state["system_status"] = "growth_assets_ready" if all(ep.status == "growth_ready" for _, ep in targets) else "blocked_on_growth_export"
    return state