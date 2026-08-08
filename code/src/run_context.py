"""Run context capture for reproducibility (V4).

Captures git commit SHA, non-sensitive env config, Python/dependency versions,
and ffmpeg version at the start of each run. Stored in DramaState.run_context
and persisted with every state snapshot + state_history row, so future runs can
explain whether differences stem from code, config, parameters, or model.
"""

from __future__ import annotations

import os
import platform
import subprocess
import time
from typing import Any
from uuid import uuid4


# Whitelist of non-sensitive env vars to snapshot (NO keys/tokens/secrets).
_CONFIG_ENV_WHITELIST = [
    "DRAMAMATRIX_MAX_CYCLES",
    "DRAMAMATRIX_MAX_SCOUT_ATTEMPTS",
    "DRAMAMATRIX_CONDITIONAL_GENERATION",
    "DRAMAMATRIX_MAX_IN_FLIGHT",
    "DRAMAMATRIX_MIN_SHOTS_PER_EPISODE",
    "DRAMAMATRIX_MAX_SHOTS_PER_EPISODE",
    "DRAMAMATRIX_MAX_TOTAL_SHOTS",
    "DRAMAMATRIX_ALLOW_MOCK_STORYBOARD",
    "DRAMAMATRIX_AGNES_VOICE",
    "DRAMAMATRIX_TTS_ENABLED",
    "DRAMAMATRIX_TTS_PROVIDER",
    "DRAMAMATRIX_TTS_VOICE",
    "DRAMAMATRIX_TTS_MODEL",
    "DRAMAMATRIX_SUBTITLES_ENABLED",
    "DRAMAMATRIX_QC_STRICT",
    "DRAMAMATRIX_QC_BRIGHTNESS_THRESHOLD",
    "DRAMAMATRIX_GROWTH_CLIP_COUNT",
    "DRAMAMATRIX_GROWTH_CLIP_DURATION",
    "DRAMAMATRIX_GROWTH_CLIMAX_DURATION",
    "AGNES_VIDEO_MODEL",
    "AGNES_VIDEO_WIDTH",
    "AGNES_VIDEO_HEIGHT",
    "AGNES_VIDEO_FRAME_RATE",
    "AGNES_MAX_REVISIONS",
    "AGNES_POST_RETRY_ATTEMPTS",
    "AGNES_QUEUE_RETRY_MAX_SECONDS",
    "AGNES_PREFLIGHT_CACHE_SECONDS",
    "AGNES_MAX_SHOTS_PER_EPISODE",
    # F6：补全实验关键参数
    "DRAMAMATRIX_TEST_MODE",
    "DRAMAMATRIX_MAX_EPISODES",
    "DRAMAMATRIX_TARGET_SHOTS_PER_EPISODE",
    "DRAMAMATRIX_MAX_AGNES_CREATES",
    "DRAMAMATRIX_OUTPUT_DIR",
    "DRAMAMATRIX_PROJECT_ID",
    "AGNES_IMAGE_FIELD",
    "AGNES_IMAGE_URL_FIELD",
    "AGNES_LAST_FRAME_FIELD",
    "AGNES_NARRATION_FIELD",
]


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return "unknown"


def _ffmpeg_version() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.splitlines()[0] if result.stdout else "unknown"
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return "not-found"


def _dependency_versions() -> dict[str, str]:
    """Versions of key runtime deps, best-effort."""
    versions: dict[str, str] = {}
    for pkg in ("langgraph", "langchain_openai", "langchain_core", "pydantic", "requests"):
        try:
            import importlib.metadata as md
            versions[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            versions[pkg] = "unknown"
    try:
        import edge_tts
        versions["edge_tts"] = getattr(edge_tts, "__version__", "installed")
    except Exception:  # noqa: BLE001
        versions["edge_tts"] = "not-installed"
    return versions


def capture_run_context() -> dict[str, Any]:
    """Capture the full run context for reproducibility (V4).

    Excludes all credentials (any env var whose name contains KEY/TOKEN/SECRET
    is never included, even if whitelisted by mistake).
    """
    config: dict[str, str] = {}
    for name in _CONFIG_ENV_WHITELIST:
        value = os.getenv(name)
        if value is not None and not _looks_sensitive(name):
            config[name] = value
    return {
        # F6：UUID 避免秒级时间戳碰撞（同一秒多项目/多运行会冲突）
        "run_id": uuid4().hex,
        "captured_at": time.time(),
        "git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "ffmpeg_version": _ffmpeg_version(),
        "dependencies": _dependency_versions(),
        "config": config,
    }


def _looks_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
