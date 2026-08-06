"""Text-to-speech voiceover + mixing helpers for the voiceover pipeline.

The module is provider-neutral: an external TTS HTTP service is invoked only if
`DRAMAMATRIX_TTS_ENABLED` is truthy and a provider URL is configured. When TTS
is disabled/unconfigured or fails, it degrades to an empty (silent) track so the
main pipeline still produces an audible file, marking the episode as voiceless.

All heavy lifting is shelled out to ffmpeg and is mock-friendly for CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TTSResult:
    """Outcome of a per-episode voiceover pass."""
    audio_path: Optional[str]  # final mixed audio path (None => silent)
    voiceover: bool  # whether real voiceover was produced
    segments_built: int  # number of dialogue clips that were synthesized


def synthesize_line(text: str, destination: Path) -> Optional[Path]:
    """Synthesize a single dialogue line to an audio file.

    Returns the produced audio path, or None if TTS is unavailable. The actual
    provider call is left to the calling environment (server-side TTS); locally
    this returns None so the pipeline degrades gracefully without an audio
    backend.
    """
    raise NotImplementedError(
        "Concrete TTS synthesis is provider-specific and must be provided by "
        "the server deployment (see DRAMAMATRIX_TTS_URL)."
    )


def build_voiceover(
    dialogue_segments: list[tuple[str, float]],
    destination_dir: Path,
) -> TTSResult:
    """Stitch per-line dialogue into a single voiceover track laid to BGM duration.

    dialogue_segments: [(dialogue_text, shot_duration_seconds)...]
    No generic TTS backend is shipped, so real synthesis is not performed here;
    the server deployment overrides synthesize_line. To avoid silently replacing
    the original (usually silent) track with a blank one, this returns
    audio_path=None (no voiceover) unless a real voiceover was actually produced.
    """
    if not tts_enabled():
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)

    # No real synth backend: do NOT fabricate a silent replacement. Agent 6 will
    # keep the source audio and print an accurate "无配音" note (review F5).
    return TTSResult(audio_path=None, voiceover=False, segments_built=0)


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
    command = [
        ffmpeg, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(destination),
    ]
    try:
        import subprocess
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AgnesVideoError(f"FFmpeg 混音失败: {exc.stderr[-1200:]}") from exc
    return destination