"""Text-to-speech voiceover + mixing helpers for the voiceover pipeline.

The module is provider-neutral: an external TTS HTTP service is invoked only if
`DRAMAMATRIX_TTS_ENABLED` is truthy and a provider URL is configured. When TTS
is disabled/unconfigured or fails, it degrades to an empty (silent) track so the
main pipeline still produces an audible file, marking the episode as voiceless.

All heavy lifting is shelled out to ffmpeg and is mock-friendly for CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.agnes_video import AgnesVideoError, _require_binary, video_duration


def tts_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_TTS_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def tts_provider_url() -> str:
    return os.getenv("DRAMAMATRIX_TTS_URL", "").strip()


def tts_provider() -> str:
    """Which TTS backend to use: 'edge' (edge-tts), 'openai', or '' (none)."""
    return os.getenv("DRAMAMATRIX_TTS_PROVIDER", "").strip().lower()


def agnes_voice_enabled() -> bool:
    """Whether to request Agnes native voice/narration in create_video (G4a).

    Off by default — Agnes audio support is unverified. When on, the dialogue is
    passed as voice_prompt/narration; Agent6 still verifies the output has audio
    and falls back to independent TTS if Agnes produced a silent clip.
    """
    return os.getenv("DRAMAMATRIX_AGNES_VOICE", "0").strip().lower() in {"1", "true", "yes", "on"}


def tts_voice(role: str | None = None) -> str:
    """Voice id for the chosen TTS provider (E3 role-aware).

    Supports DRAMAMATRIX_TTS_VOICE_MAP (JSON: {"男主": "zh-CN-YunjianNeural", ...})
    for per-role voices. Falls back to the global DRAMAMATRIX_TTS_VOICE.
    """
    import json as _json
    global_voice = os.getenv("DRAMAMATRIX_TTS_VOICE", "zh-CN-XiaoxiaoNeural").strip()
    raw = os.getenv("DRAMAMATRIX_TTS_VOICE_MAP", "").strip()
    if role and raw:
        try:
            mapping = _json.loads(raw)
            if isinstance(mapping, dict) and mapping.get(role):
                return str(mapping[role])
        except (_json.JSONDecodeError, AttributeError):
            pass
    return global_voice


@dataclass(frozen=True)
class TTSLine:
    """R2 对白时间表：单句对白在成片音轨上的真实排布。

    start/end 为该句语音在音轨上的实际起止秒；speed_ratio 为变速倍率
    （1.0=原速）；overflow=True 表示即使变速到上限仍超出所属镜头窗口
    （语音保留完整不截断，后续句顺延）；unmeasured=True 表示无法探测
    合成时长，退回旧的"按时长裁齐"行为（需人工复核）。
    """
    index: int
    role: Optional[str]
    text: str
    synthesized: bool
    start: float = 0.0
    end: float = 0.0
    speed_ratio: float = 1.0
    overflow: bool = False
    unmeasured: bool = False


@dataclass(frozen=True)
class TTSResult:
    """Outcome of a per-episode voiceover pass."""
    audio_path: Optional[str]  # final mixed audio path (None => silent)
    voiceover: bool  # whether real voiceover was produced
    segments_built: int  # number of dialogue clips that were synthesized
    lines: list = field(default_factory=list)  # list[TTSLine] 逐句时间表（R2）


def synthesize_line(text: str, destination: Path, role: str | None = None) -> Optional[Path]:
    """Synthesize a single dialogue line to an audio file (G4b / E3 role voice).

    Provider is selected by DRAMAMATRIX_TTS_PROVIDER:
    - 'edge': edge-tts (free, no key). Requires the `edge-tts` package.
    - 'openai': OpenAI-compatible TTS via DRAMAMATRIX_TTS_URL + key.
    Returns the produced audio path, or None if no provider is configured /
    synthesis fails (callers degrade to keeping the source audio).
    """
    if not text.strip():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    provider = tts_provider()
    try:
        if provider == "edge":
            return _synthesize_edge(text, destination, role=role)
        if provider == "openai":
            return _synthesize_openai(text, destination, role=role)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"   [TTS] 合成失败（{provider}）：{exc}")
        return None
    return None


def _synthesize_edge(text: str, destination: Path, role: str | None = None) -> Optional[Path]:
    """edge-tts synthesis (free, Microsoft Edge online TTS)."""
    import asyncio
    import edge_tts  # type: ignore[import-not-found]
    voice = tts_voice(role)

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(destination))

    asyncio.run(_run())
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def _synthesize_openai(text: str, destination: Path, role: str | None = None) -> Optional[Path]:
    """OpenAI-compatible TTS via configured URL + key."""
    import requests
    url = tts_provider_url() or "https://api.openai.com/v1/audio/speech"
    api_key = os.getenv("DRAMAMATRIX_TTS_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("DRAMAMATRIX_TTS_MODEL", "tts-1")
    voice = os.getenv("DRAMAMATRIX_TTS_OPENAI_VOICE", "alloy")
    if role:
        voice = tts_voice(role) or voice
    response = requests.post(
        url,
        json={"model": model, "input": text, "voice": voice},
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=60,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def _audio_duration(path: Path) -> Optional[float]:
    """Probe an audio file's duration in seconds via ffprobe (R2)."""
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    ffprobe = _shutil.which("ffprobe")
    if not ffprobe or not path.is_file():
        return None
    command = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "json", str(path),
    ]
    try:
        probe = _subprocess.run(command, check=True, capture_output=True, text=True)
        info = _json.loads(probe.stdout or "{}")
        value = float((info.get("format") or {}).get("duration") or 0.0)
        return value if value > 0 else None
    except (ValueError, OSError, _subprocess.CalledProcessError, _json.JSONDecodeError):
        return None


def _max_tts_speed() -> float:
    """R2：对白变速上限（超过则保留完整语音并标记溢出，绝不截断）。"""
    try:
        return min(max(float(os.getenv("DRAMAMATRIX_TTS_MAX_SPEED", "1.35")), 1.0), 2.0)
    except ValueError:
        return 1.35


def build_voiceover(
    dialogue_segments: list[tuple],
    destination_dir: Path,
    probe_duration=None,
) -> TTSResult:
    """Build a voiceover track timed to each shot (G4b / H1 / R2).

    dialogue_segments: [(dialogue_text, shot_duration_seconds[, role])...]
    R2 禁止截断：先逐句合成并实测时长（ffprobe），再按时间表排布——
    - 语句落在自己的镜头窗口内则原速放置；
    - 超出窗口时按 natural/slot 变速压缩（上限 DRAMAMATRIX_TTS_MAX_SPEED，
      默认 1.35）；到上限仍放不下则保留完整语音、标记 overflow、后续句顺延；
    - 无对白的镜头生成等长静音；音轨总长不少于视频总长（尾部补静音）。
    每句的起止时间/变速/溢出通过 TTSResult.lines 返回，供字幕对齐与整集验收。
    """
    if not tts_enabled():
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)
    provider = tts_provider()
    if not provider:
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)
    if probe_duration is None:
        probe_duration = _audio_duration

    destination_dir.mkdir(parents=True, exist_ok=True)
    max_speed = _max_tts_speed()
    lines: list[TTSLine] = []
    placed: list[tuple[TTSLine, Path]] = []  # (line, prepared clip)
    segments_built = 0

    # 镜头起点（时间表锚）
    shot_starts: list[float] = []
    cursor = 0.0
    for seg in dialogue_segments:
        shot_starts.append(cursor)
        cursor += max(float(seg[1]), 0.5)
    total_duration = cursor

    for idx, seg in enumerate(dialogue_segments):
        if len(seg) >= 3:
            text, duration, role = seg[0], seg[1], seg[2]
        else:
            text, duration = seg[0], seg[1]
            role = None
        slot = max(float(duration), 0.5)
        text = (text or "").strip()
        if not text:
            continue  # 无对白：不留静音占位（由最终拼装时的间隙静音覆盖）
        raw_clip = destination_dir / f"line_{idx:03d}.mp3"
        if not synthesize_line(text, raw_clip, role=role):
            lines.append(TTSLine(index=idx, role=role, text=text, synthesized=False))
            continue
        segments_built += 1
        natural = probe_duration(raw_clip)
        if natural is None:
            # 无法实测时长：退回旧的按时长裁齐（截断）行为并显式标记，
            # 供整集验收人工复核（缺 ffprobe 环境）。
            timed = _fit_clip_to_duration(raw_clip, slot, destination_dir / f"timed_{idx:03d}.m4a")
            if not timed:
                lines.append(TTSLine(index=idx, role=role, text=text, synthesized=False))
                continue
            prev_end = next((l.end for l in reversed(lines) if l.synthesized), 0.0)
            start = max(shot_starts[idx], prev_end)
            line = TTSLine(index=idx, role=role, text=text, synthesized=True,
                           start=start, end=start + slot, unmeasured=True)
            lines.append(line)
            placed.append((line, timed))
            print(f"   [TTS] 第 {idx} 句无法探测时长，退回裁齐（unmeasured，需人工复核）。")
            continue
        # 时间表：不早于本镜头起点，也不与上一句重叠。
        prev_end = next((l.end for l in reversed(lines) if l.synthesized), 0.0)
        start = max(shot_starts[idx], prev_end)
        speed = 1.0
        if natural > slot + 0.05:
            speed = min(natural / slot, max_speed)
        effective = natural / speed
        overflow = (start + effective) > shot_starts[idx] + slot + 0.05
        if overflow:
            print(f"   [TTS] 第 {idx} 句语音 {natural:.2f}s 超出镜头窗口 {slot:.2f}s"
                  f"（变速 x{speed:.2f} 后仍溢出，保留完整语音并顺延后续对白）。")
        line = TTSLine(index=idx, role=role, text=text, synthesized=True,
                       start=start, end=start + effective,
                       speed_ratio=round(speed, 3), overflow=overflow)
        lines.append(line)
        prepared = _prepare_clip(raw_clip, speed, destination_dir / f"timed_{idx:03d}.m4a")
        if not prepared:
            lines[-1] = TTSLine(index=idx, role=role, text=text, synthesized=False)
            segments_built -= 1
            continue
        placed.append((line, prepared))

    if not placed:
        return TTSResult(audio_path=None, voiceover=False, segments_built=0, lines=lines)

    # 拼装：句间间隙与首尾补静音，总长不少于视频时长。
    pieces: list[Path] = []
    timeline_cursor = 0.0
    for line, clip in placed:
        gap = line.start - timeline_cursor
        if gap > 0.01:
            silent = _make_silent_track(gap, destination_dir / f"gap_{len(pieces):03d}.m4a")
            if not silent:
                return TTSResult(audio_path=None, voiceover=False, segments_built=0, lines=lines)
            pieces.append(silent)
        pieces.append(clip)
        timeline_cursor = line.end
    if total_duration > timeline_cursor + 0.01:
        tail = _make_silent_track(total_duration - timeline_cursor, destination_dir / "tail.m4a")
        if tail:
            pieces.append(tail)

    voiceover_path = destination_dir / "voiceover.m4a"
    if _concat_audio(pieces, voiceover_path):
        for clip in pieces:
            clip.unlink(missing_ok=True)
        return TTSResult(
            audio_path=str(voiceover_path), voiceover=True,
            segments_built=segments_built, lines=lines,
        )
    return TTSResult(audio_path=None, voiceover=False, segments_built=0, lines=lines)


def _prepare_clip(clip: Path, speed_ratio: float, destination: Path) -> Optional[Path]:
    """R2：把合成语音统一重编码为 aac；speed_ratio>1 时施加 atempo 变速。"""
    if speed_ratio <= 1.0 + 1e-6:
        filter_expr = None
    else:
        # atempo 单段有效范围 0.5–100，1.35 内无需链式拆分。
        filter_expr = f"atempo={speed_ratio:.4f}"
    try:
        import subprocess
        ffmpeg = _require_binary("ffmpeg")
    except (AgnesVideoError, ImportError):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-i", str(clip)]
    if filter_expr:
        command += ["-filter:a", filter_expr]
    command += ["-c:a", "aac", "-b:a", "128k", str(destination)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def _fit_clip_to_duration(clip: Path, target_seconds: float, destination: Path) -> Optional[Path]:
    """Pad/trim an audio clip to exactly target_seconds (H1).

    Uses ffmpeg apad/atrim so each dialogue line occupies exactly its shot
    window; the next line starts at the next shot boundary, not immediately.
    """
    try:
        import subprocess
        ffmpeg = _require_binary("ffmpeg")
    except (AgnesVideoError, ImportError):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-i", str(clip),
        "-filter:a", f"atrim=0:{target_seconds:.3f},asetpts=N/SR/TB,apad=whole_dur={target_seconds:.3f}",
        "-c:a", "aac", "-b:a", "128k", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return None
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def _concat_audio(clips: list[Path], destination: Path) -> bool:
    """Concatenate audio clips via ffmpeg into a single track."""
    try:
        import subprocess
        ffmpeg = _require_binary("ffmpeg")
    except (AgnesVideoError, ImportError):
        return False
    list_file = destination.with_suffix(".concat.txt")
    list_file.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips) + "\n",
        encoding="utf-8",
    )
    command = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:a", "aac", "-b:a", "128k", str(destination),
    ]
    try:
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return False
    finally:
        list_file.unlink(missing_ok=True)
    return destination.is_file() and destination.stat().st_size > 0


def _make_silent_track(duration_seconds: float, destination: Path) -> Optional[Path]:
    """Generate a silent PCM/WAV track of the given duration via ffmpeg."""
    try:
        ffmpeg = _require_binary("ffmpeg")
    except AgnesVideoError:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-t", f"{duration_seconds:.3f}",
        "-c:a", "aac", "-b:a", "128k", str(destination),
    ]
    try:
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return destination


def mix_audio_into_video(video_path: Path, audio_path: Optional[Path], destination: Path) -> Path:
    """Mux an audio track into a video file. When audio_path is None, the video
    is copied as-is (already carries its own audio or stays silent).

    This is intentionally a thin wrapper so tests can mock the media call.
    """
    if audio_path is None:
        # No voiceover: keep the source file (Agnes shots are usually silent;
        # if they carry audio we preserve it).
        destination.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(video_path, destination)
        return destination

    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # H1：去掉 -shortest——以视频时长为准，音频短于视频时尾部自然静音，
    # 不截断视频。voiceover 已按镜头时长对齐补齐，通常与视频等长。
    command = [
        ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", str(destination),
    ]
    try:
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AgnesVideoError(f"FFmpeg 混音失败: {exc.stderr[-1200:]}") from exc
    return destination


def mix_with_bgm_and_normalize(
    video_path: Path,
    voiceover_path: Path,
    bgm_path: Path,
    destination: Path,
) -> Path:
    """Mix voiceover + BGM and apply loudnorm loudness normalization (E3).

    Audio pipeline: voiceover (dialogue) mixed with BGM bed, then loudnorm
    standardizes integrated loudness. Returns `destination`, or the input video
    unchanged if ffmpeg is unavailable (degrade gracefully).
    """
    import subprocess
    try:
        ffmpeg = _require_binary("ffmpeg")
    except AgnesVideoError:
        return video_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    # -af: mix voiceover (input 1) with BGM (input 2); BGM volume lowered to bed.
    # filter='amix=inputs=2:duration=first' keeps dialogue first-channel priority.
    command = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-i", str(voiceover_path),
        "-i", str(bgm_path),
        "-filter_complex",
        "[1:a]volume=1.0[vo];[2:a]volume=0.15[bgm];[vo][bgm]amix=inputs=2:duration=first:normalize=0[mix];"
        "[mix]loudnorm=I=-16:TP=-1.5:LRA=11[outa]",
        "-map", "0:v", "-map", "[outa]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"   ⚠️ BGM 混音/响度标准化失败（保留配音版）：{exc.stderr[-300:]}")
        import shutil
        shutil.copyfile(video_path, destination)
    return destination