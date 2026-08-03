"""Agent 6: concatenate downloaded Agnes shots into episode masters with FFmpeg."""

from pathlib import Path

from src.agnes_video import AgnesVideoError, concat_videos, episode_output_dir
from src.state import DramaState, EpisodeState, FeedbackLog


def process_agent6_editor(state: DramaState) -> DramaState:
    print("--- [Agent 6: FFmpeg Episode Editor] ---")
    targets = [(key, ep) for key, ep in state["episodes"].items() if ep.status == "video_generated"]
    if not targets:
        print("没有已下载、待合成的剧集。")
        return state

    for ep_key, ep_state in targets:
        try:
            inputs = [Path(asset.local_path) for asset in ep_state.video_assets if asset.local_path]
            output = episode_output_dir(state["project_id"], ep_key) / f"{ep_key}_master.mp4"
            final_video = concat_videos(inputs, output)
            ep_state.final_video_path = str(final_video)
            ep_state.status = "edit_completed"
            print(f"✅ {ep_key} 已合成为 {final_video}")
        except AgnesVideoError as exc:
            ep_state.status = "editing_failed"
            ep_state.feedback_log.append(
                FeedbackLog(
                    from_agent="Agent_6_Editor",
                    to_agent="Operator",
                    reason_code="FFMPEG_EDIT_FAILED",
                    message=str(exc),
                )
            )
            print(f"❌ {ep_key} 合成失败：{exc}")
        state["episodes"][ep_key] = ep_state

    state["system_status"] = "episodes_edited" if all(ep.status == "edit_completed" for _, ep in targets) else "blocked_on_editing"
    return state
