"""Agent 6: concatenate downloaded Agnes shots into episode masters with FFmpeg,
then attach a voiceover track (TTS) produced from the shots' dialogue.
"""

from pathlib import Path

from src.agnes_video import AgnesVideoError, concat_videos, episode_output_dir, has_audio_stream
from src.state import DramaState, EpisodeState, FeedbackLog
from src.subtitles import build_ass_track, burn_subtitles
from src.tts import build_voiceover, mix_audio_into_video


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

            # H4：若成片已含音频（Agnes 原生语音成功），保留原音轨，跳过独立 TTS；
            # 仅在无声时才用独立 TTS 兜底。
            voiceover = None
            if has_audio_stream(final_video):
                print(f"   {ep_key} 成片已含音频（疑似 Agnes 原生语音），保留原音轨。")
            else:
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


def _aligned_durations(ep_state: EpisodeState) -> list[float]:
    """V3：返回每镜真实时长（优先 actual_duration），避免多镜累积漂移。

    按 storyboard 顺序对齐 video_assets；当某镜无真实时长时回退到计划 duration。
    """
    assets_by_shot = {a.shot_id: a for a in ep_state.video_assets}
    durations: list[float] = []
    for shot in ep_state.storyboard_data:
        asset = assets_by_shot.get(shot.shot_id)
        real = getattr(asset, "actual_duration", None) if asset else None
        durations.append(float(real) if real and real > 0 else _shot_seconds(shot.duration))
    return durations


def _apply_voiceover(ep_state: EpisodeState, ep_dir: Path):
    """Build an independent TTS voiceover track; return its path or None (G4b).

    Used as the fallback when Agnes native voice is off or produced a silent
    clip. Returns None when no TTS provider is configured (the master keeps its
    own audio, which may carry Agnes-generated speech if AGNES_VOICE is on).
    """
    from src.tts import tts_provider
    if not tts_provider():
        return None
    durations = _aligned_durations(ep_state)
    dialogue_segments = [
        ((shot.dialogue or "").strip(), durations[i])
        for i, shot in enumerate(ep_state.storyboard_data)
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
    # V3：以真实时长（actual_duration）驱动字幕时间窗，避免多镜后漂移。
    durations = _aligned_durations(ep_state)
    segments = []  # (text, start_seconds, duration_seconds)
    cursor = 0.0
    for i, shot in enumerate(ep_state.storyboard_data):
        duration = durations[i]
        text = (shot.dialogue or "").strip()
        segments.append((text, cursor, duration))
        cursor += duration
    if not any(text for text, _, _ in segments):
        return None
    subs_name = f"{ep_state.script_data.ep_id if ep_state.script_data else 'ep'}_subs"
    ass_path = build_ass_track(segments, ep_dir / f"{subs_name}.ass")
    if not ass_path.exists():
        return None
    # Use a DISTINCT output name: FFmpeg cannot overwrite its own input in-place
    # (review F2). Success is only returned for the burned copy; the caller
    # promotes it to final_video_path.
    out = ep_dir / f"{subs_name}_subtitled.mp4"
    burned = burn_subtitles(video_path, ass_path, out)
    if burned == video_path:
        # No ffmpeg / subtitles disabled: fall back to video unchanged.
        return None
    return burned, ass_path