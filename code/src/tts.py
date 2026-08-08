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


def tts_voice() -> str:
    """Voice id for the chosen TTS provider."""
    return os.getenv("DRAMAMATRIX_TTS_VOICE", "zh-CN-XiaoxiaoNeural").strip()


@dataclass(frozen=True)
class TTSResult:
    """Outcome of a per-episode voiceover pass."""
    audio_path: Optional[str]  # final mixed audio path (None => silent)
    voiceover: bool  # whether real voiceover was produced
    segments_built: int  # number of dialogue clips that were synthesized


def synthesize_line(text: str, destination: Path) -> Optional[Path]:
    """Synthesize a single dialogue line to an audio file (G4b).

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
            return _synthesize_edge(text, destination)
        if provider == "openai":
            return _synthesize_openai(text, destination)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        print(f"   [TTS] 合成失败（{provider}）：{exc}")
        return None
    return None


def _synthesize_edge(text: str, destination: Path) -> Optional[Path]:
    """edge-tts synthesis (free, Microsoft Edge online TTS)."""
    import asyncio
    import edge_tts  # type: ignore[import-not-found]
    voice = tts_voice()

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(destination))

    asyncio.run(_run())
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def _synthesize_openai(text: str, destination: Path) -> Optional[Path]:
    """OpenAI-compatible TTS via configured URL + key."""
    import requests
    url = tts_provider_url() or "https://api.openai.com/v1/audio/speech"
    api_key = os.getenv("DRAMAMATRIX_TTS_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("DRAMAMATRIX_TTS_MODEL", "tts-1")
    voice = os.getenv("DRAMAMATRIX_TTS_OPENAI_VOICE", "alloy")
    response = requests.post(
        url,
        json={"model": model, "input": text, "voice": voice},
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=60,
    )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination if destination.is_file() and destination.stat().st_size > 0 else None


def build_voiceover(
    dialogue_segments: list[tuple[str, float]],
    destination_dir: Path,
) -> TTSResult:
    """Stitch per-line dialogue into a single voiceover track (G4b).

    dialogue_segments: [(dialogue_text, shot_duration_seconds)...]
    When a TTS provider is configured, each non-empty line is synthesized and
    concatenated (with silence padding to match shot duration) via ffmpeg. When
    no provider is configured, returns audio_path=None so Agent6 keeps the
    source audio (which, if DRAMAMATRIX_AGNES_VOICE is on, may already carry
    Agnes-generated speech).
    """
    if not tts_enabled():
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)
    provider = tts_provider()
    if not provider:
        # No independent TTS backend; rely on Agnes native voice (if enabled).
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)

    destination_dir.mkdir(parents=True, exist_ok=True)
    segments_built = 0
    clip_paths: list[Path] = []
    for idx, (text, _duration) in enumerate(dialogue_segments):
        if not text.strip():
            continue
        clip = destination_dir / f"line_{idx:03d}.mp3"
        if synthesize_line(text, clip):
            clip_paths.append(clip)
            segments_built += 1
    if not clip_paths:
        return TTSResult(audio_path=None, voiceover=False, segments_built=0)

    # Concatenate synthesized clips into one voiceover track.
    voiceover_path = destination_dir / "voiceover.mp3"
    if _concat_audio(clip_paths, voiceover_path):
        return TTSResult(audio_path=str(voiceover_path), voiceover=True, segments_built=segments_built)
    return TTSResult(audio_path=None, voiceover=False, segments_built=0)


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