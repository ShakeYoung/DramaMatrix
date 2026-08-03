"""Agent 7: produce concrete teaser assets from completed episode masters."""

from pathlib import Path

from src.agnes_video import AgnesVideoError, cut_video, episode_output_dir, video_duration
from src.state import DramaState, EpisodeState, FeedbackLog, GrowthAsset


def process_agent7_growth(state: DramaState) -> DramaState:
    print("--- [Agent 7: FFmpeg Growth Packaging] ---")
    targets = [(key, ep) for key, ep in state["episodes"].items() if ep.status == "edit_completed"]
    if not targets:
        print("没有待制作投流素材的成片。")
        return state

    for ep_key, ep_state in targets:
        try:
            if not ep_state.final_video_path:
                raise AgnesVideoError("成片路径缺失，无法制作投流素材。")
            master = Path(ep_state.final_video_path)
            total_seconds = video_duration(master)
            if total_seconds <= 0:
                raise AgnesVideoError("成片时长为 0。")
            clip_duration = min(15.0, total_seconds)
            growth_dir = episode_output_dir(state["project_id"], ep_key) / "growth"
            hook_path = cut_video(master, growth_dir / f"{ep_key}_hook.mp4", 0.0, clip_duration)
            climax_start = max(0.0, total_seconds - clip_duration)
            climax_path = cut_video(master, growth_dir / f"{ep_key}_climax.mp4", climax_start, clip_duration)
            ep_state.growth_assets = [
                GrowthAsset(name="hook", path=str(hook_path), start_seconds=0.0, duration_seconds=clip_duration),
                GrowthAsset(name="climax", path=str(climax_path), start_seconds=climax_start, duration_seconds=clip_duration),
            ]
            ep_state.status = "growth_ready"
            print(f"✅ {ep_key} 已导出 hook 与 climax 两个投流版本。")
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
