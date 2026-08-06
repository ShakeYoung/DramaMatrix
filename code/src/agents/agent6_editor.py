"""Agent 6: concatenate downloaded Agnes shots into episode masters with FFmpeg,
then attach a voiceover track (TTS) produced from the shots' dialogue.
"""

from pathlib import Path

from src.agnes_video import AgnesVideoError, concat_videos, episode_output_dir
from src.state import DramaState, EpisodeState, FeedbackLog
from src.subtitles import build_ass_track, burn_subtitles
from src.tts import build_voiceover, mix_audio_into_video, tts_provider_url


def process_agent6_editor(state: DramaState) -> DramaState:
    print("--- [Agent 6: FFmpeg Episode Editor] ---")
    targets = [(key, ep) for key, ep in state["episodes"].items() if ep.status == "video_generated"]
    if not targets:
        print("没有已下载、待合成的剧集。")
        return state

    for ep_key, ep_state in targets:
        try:
            inputs = [Path(asset.local_path) for asset in ep_state.video_assets if asset.local_path]
            ep_dir = episode_output_dir(state["project_id"], ep_key)
            output = ep_dir / f"{ep_key}_master.mp4"
            final_video = concat_videos(inputs, output)

            # T5: build a voiceover track from the storyboard dialogue and mix
            # it into the master. If TTS is unavailable it degrades to keeping
            # the master's (usually silent) audio.
            voiceover = _apply_voiceover(ep_state, ep_dir)
            if voiceover:
                muxed = mix_audio_into_video(final_video, Path(voiceover), ep_dir / f"{ep_key}_voiced.mp4")
                ep_state.audio_track = str(voiceover)
                final_video = muxed

            # T6: burn in "大字报" subtitles for the shot dialogue windows.
            subtitled = _apply_subtitles(ep_state, final_video, ep_dir)
            if subtitled is not None:
                ep_state.subtitle_track = str(subtitled[1])
                final_video = subtitled[0]

            ep_state.final_video_path = str(final_video)
            ep_state.status = "edit_completed"
            print(f"✅ {ep_key} 已合成为 {final_video}"
                  + ("（含配音音轨）" if voiceover else "（无声/保持原音轨）"))
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


def _apply_voiceover(ep_state: EpisodeState, ep_dir: Path):
    """Build a voiceover track for the shots; return its path or None."""
    if not tts_provider_url():
        # No TTS endpoint configured: skip voiceover entirely (keep master audio).
        return None
    dialogue_segments = [
        ((shot.dialogue or "").strip(), _shot_seconds(shot.duration))
        for shot in ep_state.storyboard_data
    ]
    if not any(text for text, _ in dialogue_segments):
        return None
    result = build_voiceover(dialogue_segments, ep_dir / "audio")
    return result.audio_path


def _shot_seconds(duration: str) -> float:
    import re
    match = re.search(r"\d+(?:\.\d+)?", duration or "")
    return float(match.group()) if match else 4.0


def _apply_subtitles(ep_state: EpisodeState, video_path: Path, ep_dir: Path):
    """Assemble and burn shot-dialogue subtitles; return (burned_video, ass_path) or None."""
    segments = []  # (text, start_seconds, duration_seconds)
    cursor = 0.0
    for shot in ep_state.storyboard_data:
        duration = _shot_seconds(shot.duration)
        text = (shot.dialogue or "").strip()
        segments.append((text, cursor, duration))
        cursor += duration
    if not any(text for text, _, _ in segments):
        return None
    ass_path = build_ass_track(segments, ep_dir / f"{ep_state.script_data.ep_id if ep_state.script_data else 'ep'}_subs.ass")
    if not ass_path.exists():
        return None
    out = ep_dir / f"{Path(video_path).name}"
    burned = burn_subtitles(video_path, ass_path, out)
    if burned == video_path:
        # No ffmpeg / subtitles disabled: fall back to video unchanged.
        return None
    return burned, ass_path