"""Agent 6: concatenate downloaded Agnes shots into episode masters with FFmpeg,
then attach a voiceover track (TTS) produced from the shots' dialogue.

R2（可靠成片）：
- 对白时间表：逐句携带说话人（shot.speaker）选择角色音色，合成后实测时长
  按时间表排布（禁止截断，详见 src/tts.py）。
- 字幕对齐：有配音时字幕时间窗用语音的真实起止时间，而非镜头窗口。
- 整集验收：DRAMAMATRIX_EPISODE_REVIEW=1（默认）时合成完成进入
  awaiting_episode_review 暂停，人工核片后 approve 进入投流 / rework 重合成。
"""

import os
from pathlib import Path

from src.agnes_video import (
    AgnesVideoError,
    concat_videos,
    episode_output_dir,
    has_dialogue_stream,
)
from src.deliverables import record_deliverable
from src.state import DramaState, EpisodeState, FeedbackLog
from src.subtitles import build_ass_track, burn_subtitles
from src.tts import build_voiceover, mix_audio_into_video, mix_with_bgm_and_normalize


def _episode_review_targets(project_id: str, state: DramaState) -> list[tuple[str, EpisodeState]]:
    """video_generated 的集，以及被标记 rework 的待重合成集。"""
    targets = []
    for key, ep in state["episodes"].items():
        if ep.status == "video_generated":
            targets.append((key, ep))
        elif ep.status == "awaiting_episode_review":
            try:
                from src.episode_review import episode_rework

                if episode_rework(project_id, key):
                    targets.append((key, ep))
            except Exception:
                pass
    return targets


def process_agent6_editor(state: DramaState) -> DramaState:
    print("--- [Agent 6: FFmpeg Episode Editor] ---")
    project_id = state["project_id"]
    targets = _episode_review_targets(project_id, state)
    if not targets:
        print("没有已下载、待合成的剧集。")
        return state

    from src.episode_review import episode_review_enabled, write_episode_review_manifest

    results = []
    for ep_key, ep_state in targets:
        try:
            inputs = [Path(asset.local_path) for asset in ep_state.video_assets if asset.local_path]
            source_shots = [asset.shot_id for asset in ep_state.video_assets if asset.local_path]
            ep_dir = episode_output_dir(project_id, ep_key)
            output = ep_dir / f"{ep_key}_master.mp4"
            final_video = concat_videos(inputs, output)
            # F4：master 证据
            record_deliverable(ep_state, kind="master", path=final_video, source_shots=source_shots)

            # H4/E3/R2：仅当明确检测到对白轨（True）才保留原音轨；检测不确定
            # （None，含"只有一条音轨"的情况——纯 BGM/环境音无法区分）或确认
            # 无声时都应用独立 TTS，不再凭"单音轨"跳过配音。
            voiceover = None
            tts_result = None
            has_dialogue = has_dialogue_stream(final_video)
            if has_dialogue is True:
                print(f"   {ep_key} 明确检测到对白轨（语音编解码/音轨标记），保留原音轨。")
            else:
                reason = "未检测到对白轨" if has_dialogue is False else "对白轨无法确认（单音轨不判定为对白）"
                print(f"   {ep_key} {reason}，启用独立 TTS 配音。")
                voiceover, tts_result = _apply_voiceover(ep_state, ep_dir)
            if voiceover:
                # E3：混音时叠加 BGM（若配置）并做 loudnorm 响度标准化。
                muxed = mix_audio_into_video(final_video, Path(voiceover), ep_dir / f"{ep_key}_voiced.mp4")
                bgm_path = os.getenv("DRAMAMATRIX_BGM_PATH", "").strip()
                if bgm_path and Path(bgm_path).is_file():
                    muxed = mix_with_bgm_and_normalize(
                        final_video, Path(voiceover), Path(bgm_path),
                        ep_dir / f"{ep_key}_voiced_bgm.mp4",
                    )
                ep_state.audio_track = str(voiceover)
                final_video = muxed
                # F4：配音版证据
                record_deliverable(ep_state, kind="voiced", path=muxed, source_shots=source_shots)
                _record_tts_cost(project_id, ep_key, ep_state, tts_result)

            # R2：对白时间表报告——整集验收核对"台词是否被截断/缺失"的证据。
            ep_state.dialogue_report = _build_dialogue_report(
                ep_state, tts_result, native=bool(has_dialogue is True)
            )

            # T6/R2: 字幕时间窗优先用语音真实起止（有配音时），否则回退镜头窗口。
            line_timings = [l for l in (tts_result.lines if tts_result else []) if l.synthesized]
            subtitled = _apply_subtitles(ep_state, final_video, ep_dir, line_timings=line_timings)
            if subtitled is not None:
                ep_state.subtitle_track = str(subtitled[1])
                final_video = subtitled[0]
                # F4：字幕版证据
                record_deliverable(ep_state, kind="subtitled", path=final_video, source_shots=source_shots)

            ep_state.final_video_path = str(final_video)
            results.append(_finish_episode(project_id, ep_key, ep_state, voiceover))
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
            results.append(ep_state.status)
            print(f"❌ {ep_key} 合成失败：{exc}")
        state["episodes"][ep_key] = ep_state

    if all(status == "edit_completed" for status in results):
        state["system_status"] = "episodes_edited"
    elif any(status == "awaiting_episode_review" for status in results):
        # waiting_ 前缀：退出码 2 + 故障报告，提醒人工核片。
        state["system_status"] = "waiting_for_episode_review"
    else:
        state["system_status"] = "blocked_on_editing"
    return state


def _finish_episode(project_id: str, ep_key: str, ep_state: EpisodeState, voiceover) -> str:
    """合成成功后的收尾：默认进入整集验收暂停，关闭门禁则直接 edit_completed。"""
    from src.episode_review import episode_review_enabled, write_episode_review_manifest

    if episode_review_enabled():
        manifest = write_episode_review_manifest(project_id, ep_key, ep_state)
        ep_state.status = "awaiting_episode_review"
        print(f"✅ {ep_key} 已合成，进入整集验收暂停。")
        print(f"   验收清单：{manifest}")
        print(f"   核片后执行：python -m src.episode_review {project_id} {ep_key} approve|rework --note ...")
    else:
        ep_state.status = "edit_completed"
        print(f"✅ {ep_key} 已合成为 {ep_state.final_video_path}"
              + ("（含配音音轨）" if voiceover else "（无声/保持原音轨）"))
    return ep_state.status


def _build_dialogue_report(ep_state: EpisodeState, tts_result, native: bool) -> dict:
    """R2：对白时间表报告（存入 EpisodeState，随快照持久化）。"""
    if tts_result is None:
        return {"native_dialogue": native, "tts_applied": False, "lines_total": 0,
                "lines_synthesized": 0, "lines": []}
    lines = [
        {
            "index": l.index,
            "role": l.role,
            "text": l.text,
            "synthesized": l.synthesized,
            "start": round(l.start, 3),
            "end": round(l.end, 3),
            "speed_ratio": l.speed_ratio,
            "overflow": l.overflow,
            "unmeasured": l.unmeasured,
        }
        for l in tts_result.lines
    ]
    total = sum(1 for shot in ep_state.storyboard_data if (shot.dialogue or "").strip())
    return {
        "native_dialogue": native,
        "tts_applied": True,
        "lines_total": total,
        "lines_synthesized": tts_result.segments_built,
        "overflow_count": sum(1 for l in lines if l["overflow"]),
        "unmeasured_count": sum(1 for l in lines if l["unmeasured"]),
        "lines": lines,
    }


def _record_tts_cost(project_id: str, ep_key: str, ep_state: EpisodeState, tts_result) -> None:
    """R3：TTS 费用落账（价目表未配置则零费用、不落账）。"""
    try:
        from src import cost_ledger

        price = cost_ledger.tts_price_per_1k_chars()
        if price <= 0 or tts_result is None:
            return
        chars = sum(len(l.text) for l in tts_result.lines if l.synthesized)
        if chars <= 0:
            return
        amount = round(chars / 1000.0 * price, 4)
        cost_ledger.record_cost_event(
            project_id=project_id,
            ep_key=ep_key,
            stage="tts",
            kind="confirmed",
            amount=amount,
            provider="tts",
            ref_id=f"tts:{ep_key}:v{ep_state.storyboard_version}",
            note=f"{chars} chars",
        )
    except Exception as exc:  # noqa: BLE001 - 记账失败不阻断合成
        print(f"   ⚠️ TTS 费用落账失败（不阻断）：{exc}")


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
    """Build an independent TTS voiceover track; return (path, TTSResult) or (None, None).

    R2：对白段携带说话人（shot.speaker）——tts_voice 按
    DRAMAMATRIX_TTS_VOICE_MAP 选择角色音色；时间表与溢出记录见 tts.py。
    """
    from src.tts import tts_provider
    if not tts_provider():
        return None, None
    durations = _aligned_durations(ep_state)
    dialogue_segments = [
        ((shot.dialogue or "").strip(), durations[i], (shot.speaker or "").strip() or None)
        for i, shot in enumerate(ep_state.storyboard_data)
    ]
    if not any(text for text, _, _ in dialogue_segments):
        return None, None
    result = build_voiceover(dialogue_segments, ep_dir / "audio")
    return result.audio_path, result


def _shot_seconds(duration: str) -> float:
    import re
    match = re.search(r"\d+(?:\.\d+)?", duration or "")
    return float(match.group()) if match else 4.0


def _apply_subtitles(ep_state: EpisodeState, video_path: Path, ep_dir: Path, line_timings=None):
    """Assemble and burn shot-dialogue subtitles; return (burned_video, ass_path) or None.

    R2：line_timings（TTS 逐句真实起止）非空时，字幕窗口=语音实际窗口；
    否则回退到按镜头真实时长累进的窗口（原生对白/无配音场景）。
    """
    if line_timings:
        segments = [
            (l.text, l.start, max(l.end - l.start, 0.5))
            for l in line_timings
            if (l.text or "").strip()
        ]
    else:
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
