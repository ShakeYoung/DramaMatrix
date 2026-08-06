"""Subtitle generation and burn-in ("大字报") for vertical short dramas.

The dialogue of each storyboard shot is turned into an ASS subtitle track at the
shot's time window, then burned into the video with the ASS filter via ffmpeg.
The ASS style uses a large, high-contrast, centered font sized for 9:16 content,
which reads well on mobile.

Everything is configurable and mock-friendly: with no ffmpeg the burn step
returns the source unchanged, and subtitle assembly is pure string work.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.agnes_video import AgnesVideoError, _require_binary


def subtitles_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_SUBTITLES_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _seconds(value: str) -> float:
    import re
    match = re.search(r"\d+(?:\.\d+)?", value or "")
    return float(match.group()) if match else 4.0


def build_ass_track(
    dialogue_segments: list[tuple[str, float, float]],
    destination: Path,
    play_res_x: int = 720,
    play_res_y: int = 1280,
) -> Path:
    """Write an ASS subtitle track.

    dialogue_segments: [(text, start_seconds, duration_seconds)...]
    Uses a big centered Bold style typical of "大字报" short-drama overlays.
    """
    font_size = int(os.getenv("DRAMAMATRIX_SUBTITLE_FONT_SIZE", "72"))
    primary = os.getenv("DRAMAMATRIX_SUBTITLE_COLOR", "&H00FFFFFF")
    outline = os.getenv("DRAMAMATRIX_SUBTITLE_OUTLINE", "&H00000000")
    destination.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans CJK SC,{font_size},{primary},&H000000FF,{outline},&H64000000,-1,0,0,0,100,100,0,0,1,3,2,2,48,48,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []
    for text, start, duration in dialogue_segments:
        text = (text or "").strip()
        if not text:
            continue
        end = start + max(duration, 0.3)
        # escape ASS newline/comma-sensitive chars minimally
        safe = text.replace("\\", "\\\\").replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(start)},{_ass_timestamp(end)},Default,,0,0,0,,{safe}"
        )

    destination.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _ass_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    hundredths = int(round((seconds - int(seconds)) * 100))
    if hundredths == 100:
        secs += 1
        hundredths = 0
    return f"{hours}:{minutes:02d}:{secs:02d}.{hundredths:02d}"


def burn_subtitles(
    video_path: Path,
    ass_path: Path,
    destination: Path,
) -> Path:
    """Burn the ASS track into the video. Without ffmpeg, returns input unchanged."""
    if not subtitles_enabled():
        return video_path
    try:
        ffmpeg = _require_binary("ffmpeg")
    except AgnesVideoError:
        return video_path

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vf", f"ass={ass_path}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "copy",
        "-movflags", "+faststart", str(destination),
    ]
    try:
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AgnesVideoError(f"FFmpeg 字幕烧录失败: {exc.stderr[-1200:]}") from exc
    return destination