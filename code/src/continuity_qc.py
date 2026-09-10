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
    metrics: dict[str, float] = field(default_factory=dict)


class ContinuityChecker(Protocol):
    """Pluggable per-shot continuity check (P2-A / R3 / W4)."""

    def check(
        self,
        prev_video: Optional[Path],
        prev_last_frame: Optional[Path],
        curr_video: Path,
        shot: ShotStoryboard,
        curr_first_frame: Optional[Path] = None,
        reference_frame: Optional[Path] = None,
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
            [
                ffmpeg,
                "-hide_banner",
                "-i",
                str(image),
                "-vf",
                "signalstats,metadata=print",
                "-f",
                "null",
                "-",
            ],
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
    return float(__import__("os").getenv("DRAMAMATRIX_QC_BRIGHTNESS_THRESHOLD", "45"))


# ---------------- U3：结构级视觉相似度（dHash 感知哈希） ----------------
# 亮度差只能发现曝光跳变；dHash 捕捉空间梯度结构，能进一步发现构图/主体
# 漂移（如角色突然消失、机位错位）。经 ffmpeg 解码为 size×size 灰度图后
# 纯 Python 计算位哈希，不引入重依赖（CLIP/InsightFace 仍可作为可插拔
# 实现替换整个 checker）。默认只告警不拦截；DRAMAMATRIX_QC_SIMILARITY_GATE=1
# 开启硬门禁（低于阈值判不合格走重绘）。

_HASH_SIZE = 16
_HASH_BITS = _HASH_SIZE * (_HASH_SIZE - 1)


def _frame_gray_bytes(image: Path, size: int = _HASH_SIZE) -> Optional[bytes]:
    """Decode a still frame into a size×size grayscale bitmap via ffmpeg rawvideo."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not image.is_file():
        return None
    try:
        result = subprocess.run(
            [
                ffmpeg, "-hide_banner", "-loglevel", "error",
                "-i", str(image),
                "-vf", f"scale={size}:{size}",
                "-f", "rawvideo", "-pix_fmt", "gray", "-",
            ],
            check=False, capture_output=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    data = result.stdout
    return data if len(data) >= size * size else None


def perceptual_hash(gray: bytes, size: int = _HASH_SIZE) -> int:
    """dHash：逐行比较相邻像素亮度（左>右 置位），输出 size*(size-1) 位整数。"""
    bits = 0
    for row in range(size):
        base = row * size
        for col in range(size - 1):
            bits <<= 1
            if gray[base + col] > gray[base + col + 1]:
                bits |= 1
    return bits


def hash_bit_count() -> int:
    return _HASH_BITS


def frame_similarity(frame_a: Path, frame_b: Path) -> Optional[float]:
    """Structural similarity in [0,1] between two still frames (1 = identical)."""
    gray_a = _frame_gray_bytes(frame_a)
    gray_b = _frame_gray_bytes(frame_b)
    if gray_a is None or gray_b is None:
        return None
    distance = bin(perceptual_hash(gray_a) ^ perceptual_hash(gray_b)).count("1")
    return 1.0 - distance / _HASH_BITS


def _similarity_threshold() -> float:
    import os
    return float(os.getenv("DRAMAMATRIX_QC_SIMILARITY_THRESHOLD", "0.55"))


def _similarity_gate_enabled() -> bool:
    import os
    return os.getenv("DRAMAMATRIX_QC_SIMILARITY_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}


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
        reference_frame: Optional[Path] = None,
    ) -> ContinuityResult:
        issues: list[str] = []
        metrics: dict[str, float] = {}
        if not curr_video.is_file():
            issues.append(f"当前镜头文件不存在: {curr_video}")
            return ContinuityResult(passed=False, issues=issues, metrics=metrics)
        if curr_video.stat().st_size == 0:
            issues.append(f"当前镜头文件为空: {curr_video}")
            return ContinuityResult(passed=False, issues=issues, metrics=metrics)
        if not _ffmpeg_available():
            # W4：像素级检查不可用不挡身份质检——判官走多模态 API，不依赖本地 ffmpeg。
            gate_failed, identity_issue = self._identity_check(reference_frame, curr_first_frame, metrics)
            if identity_issue:
                issues.append(identity_issue)
            if gate_failed:
                return ContinuityResult(passed=False, issues=issues, metrics=metrics)
            # Graceful degradation: cannot run real checks → pass with a note.
            issues.append("无 ffmpeg，跳过像素级连续性检查（建议服务器端启用）。")
            return ContinuityResult(passed=not self.strict, issues=issues, metrics=metrics)
        # R3：当存在上一镜尾帧与当前镜首帧时，做亮度/直方图差异比较。
        if prev_last_frame and curr_first_frame:
            diff = _frame_brightness_distance(prev_last_frame, curr_first_frame)
            threshold = _brightness_threshold()
            # P0-2：阈值必须写入 metrics，否则 Agent5 落库的 threshold 恒为 NULL。
            metrics["threshold"] = threshold
            if diff is None:
                issues.append("无法计算上一镜尾帧与当前镜首帧的亮度差异。")
                if self.strict:
                    return ContinuityResult(passed=False, issues=issues, metrics=metrics)
            else:
                metrics["brightness_diff"] = diff
            if diff is not None and diff > threshold:
                issues.append(f"上一镜尾帧与当前镜首帧亮度差异过大（{diff:.2f}），可能存在跳变。")
                return ContinuityResult(passed=False, issues=issues, metrics=metrics)
            # U3：结构级相似度（dHash）。默认告警；开 GATE 后作为硬门禁。
            similarity = frame_similarity(prev_last_frame, curr_first_frame)
            if similarity is not None:
                similarity_threshold = _similarity_threshold()
                metrics["frame_similarity"] = similarity
                metrics["similarity_threshold"] = similarity_threshold
                if similarity < similarity_threshold:
                    message = (
                        f"上一镜尾帧与当前镜首帧结构相似度过低（{similarity:.3f} < "
                        f"{similarity_threshold:.3f}），可能存在画面跳变或角色/场景漂移。"
                    )
                    if _similarity_gate_enabled():
                        issues.append(message)
                        return ContinuityResult(passed=False, issues=issues, metrics=metrics)
                    issues.append(f"告警：{message}")
        elif prev_last_frame and not curr_first_frame:
            issues.append("当前镜首帧提取失败，无法执行跨镜质检。")
            if self.strict:
                return ContinuityResult(passed=False, issues=issues, metrics=metrics)
        # W4：角色身份一致性（Vision-LLM 判官，参考图 vs 当前首帧）。
        if reference_frame and curr_first_frame:
            gate_failed, identity_issue = self._identity_check(reference_frame, curr_first_frame, metrics)
            if identity_issue:
                issues.append(identity_issue)
            if gate_failed:
                return ContinuityResult(passed=False, issues=issues, metrics=metrics)
        return ContinuityResult(passed=True, issues=issues, metrics=metrics)

    def _identity_check(
        self, reference_frame: Optional[Path], curr_first_frame: Optional[Path],
        metrics: dict[str, float],
    ) -> tuple[bool, Optional[str]]:
        """身份质检（W4）。Returns (gate_failed, issue_text)。

        issue_text 为 None 表示跳过（未启用/判官失败/达标）；否则是需要写入
        issues 的告警或拦截文案，gate_failed=True 时该文案即为拦截原因。
        """
        if not reference_frame or not curr_first_frame:
            return False, None
        from src.identity_qc import identity_gate_enabled, identity_threshold, judge_identity

        verdict = judge_identity(reference_frame, curr_first_frame)
        if verdict is None:
            return False, None
        threshold = identity_threshold()
        metrics["identity_score"] = verdict["score"]
        metrics["identity_threshold"] = threshold
        if verdict["score"] >= threshold:
            return False, None
        message = (
            f"角色身份一致性低于阈值（{verdict['score']:.0f} < {threshold:.0f}）："
            f"{verdict['reasons']}"
        )
        if identity_gate_enabled():
            return True, message
        return False, f"告警：{message}"


def get_checker() -> ContinuityChecker:
    """Factory: returns the configured checker (P2-A). Default is DefaultChecker."""
    import os

    strict = os.getenv("DRAMAMATRIX_QC_STRICT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return DefaultChecker(strict=strict)
