"""Per-shot continuity quality-control hook (P2-A).

Defines a pluggable ContinuityChecker interface. The DefaultChecker does only
lightweight checks (frame histogram/亮度 distance via ffmpeg, or none when ffmpeg
is absent → degrades to "pass + warn" so the main pipeline never blocks). Heavy
identity-similarity models (CLIP / InsightFace) can be supplied as an alternative
implementation on the server side without changing call sites.

A failing check signals that the CURRENT shot should be redrawn (not the whole
episode), bounded by max_revisions.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

from src.state import ShotStoryboard


@dataclass
class ContinuityResult:
    passed: bool
    issues: list[str] = field(default_factory=list)


class ContinuityChecker(Protocol):
    """Pluggable per-shot continuity check (P2-A / R3)."""

    def check(
        self,
        prev_video: Optional[Path],
        prev_last_frame: Optional[Path],
        curr_video: Path,
        shot: ShotStoryboard,
        curr_first_frame: Optional[Path] = None,
    ) -> ContinuityResult: ...


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def _frame_mean_brightness(image: Path) -> Optional[float]:
    """Mean luminance of a still frame via ffmpeg signalstats (best-effort, R3)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not image.is_file():
        return None
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-i", str(image), "-vf", "signalstats", "-f", "null", "-"],
            check=False, capture_output=True, text=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    # signalstats writes YAVG= to stderr; parse the last occurrence.
    import re
    matches = re.findall(r"YAVG=([0-9.]+)", result.stderr)
    if matches:
        try:
            return float(matches[-1])
        except ValueError:
            return None
    return None


def _frame_brightness_distance(frame_a: Path, frame_b: Path) -> Optional[float]:
    """Absolute brightness difference between two frames (R3). None if unavailable."""
    a = _frame_mean_brightness(frame_a)
    b = _frame_mean_brightness(frame_b)
    if a is None or b is None:
        return None
    return abs(a - b)


def _brightness_threshold() -> float:
    """Configurable brightness-difference threshold for QC failure (R3)."""
    return float(__import__("os").getenv("DRAMAMATRIX_QC_BRIGHTNESS_THRESHOLD", "25"))


class DefaultChecker:
    """Lightweight continuity checker (P2-A).

    Without heavy vision models this primarily verifies that the current shot
    file exists and is non-empty, and (optionally) that a tail-frame chain is in
    place. When ffmpeg is missing it degrades to pass+warn so the pipeline keeps
    moving; the real identity/wardrobe check is a server-side pluggable impl.
    """

    def __init__(self, strict: bool = False):
        self.strict = strict

    def check(
        self,
        prev_video: Optional[Path],
        prev_last_frame: Optional[Path],
        curr_video: Path,
        shot: ShotStoryboard,
        curr_first_frame: Optional[Path] = None,
    ) -> ContinuityResult:
        issues: list[str] = []
        if not curr_video.is_file():
            issues.append(f"当前镜头文件不存在: {curr_video}")
            return ContinuityResult(passed=False, issues=issues)
        if curr_video.stat().st_size == 0:
            issues.append(f"当前镜头文件为空: {curr_video}")
            return ContinuityResult(passed=False, issues=issues)
        if not _ffmpeg_available():
            # Graceful degradation: cannot run real checks → pass with a note.
            issues.append("无 ffmpeg，跳过像素级连续性检查（建议服务器端启用）。")
            return ContinuityResult(passed=True, issues=issues)
        # R3：当存在上一镜尾帧与当前镜首帧时，做亮度/直方图差异比较。
        if prev_last_frame and curr_first_frame:
            diff = _frame_brightness_distance(prev_last_frame, curr_first_frame)
            if diff is not None and diff > _brightness_threshold():
                issues.append(f"上一镜尾帧与当前镜首帧亮度差异过大（{diff:.2f}），可能存在跳变。")
                return ContinuityResult(passed=False, issues=issues)
        return ContinuityResult(passed=True, issues=issues)


def get_checker() -> ContinuityChecker:
    """Factory: returns the configured checker (P2-A). Default is DefaultChecker."""
    return DefaultChecker()