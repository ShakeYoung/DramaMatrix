"""Video generation provider abstraction & factory (E5).

Decouples the pipeline from a single Agnes dependency so a vendor can be
selected via DRAMAMATRIX_VIDEO_PROVIDER (default 'agnes'), with a DummyProvider
for dev/test, and a stable extension point for future providers (clone/kai/…).

Real third-party adapters are NOT fabricated here — they require live API keys
and are added as `DRAMAMATRIX_VIDEO_PROVIDER` values when available.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from src.agnes_video import (
    AgnesConfigurationError,
    AgnesVideoClient,
    AgnesVideoSettings,
)


class VideoProvider(ABC):
    """Minimal contract every video provider must satisfy (E5)."""

    name: str = "base"

    @abstractmethod
    def preflight(self) -> None: ...

    @abstractmethod
    def create(self, *, prompt: str, negative_prompt: str, duration: str, seed: int,
               image_url: Optional[str] = None, narration: Optional[str] = None) -> dict:
        """Submit one video task; return a dict with video_id/task_id."""

    @abstractmethod
    def wait(self, video_id: str, task_id: Optional[str] = None) -> dict:
        """Poll until complete; return result with metadata.url."""

    @abstractmethod
    def download(self, remote_url: str, destination: Path) -> Path: ...


class AgnesProvider(VideoProvider):
    """Adapter over the existing AgnesVideoClient (E5)."""

    name = "agnes"

    def __init__(self) -> None:
        settings = AgnesVideoSettings.from_environment()
        self._client = AgnesVideoClient(settings)

    def preflight(self) -> None:
        self._client.preflight()

    def create(self, *, prompt, negative_prompt, duration, seed, image_url=None, narration=None) -> dict:
        return self._client.create_video(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            seed=seed,
            image_url=image_url,
            narration=narration,
        )

    def wait(self, video_id: str, task_id: Optional[str] = None) -> dict:
        return self._client.wait_for_video(video_id, task_id)

    def download(self, remote_url: str, destination: Path) -> Path:
        return self._client.download_video(remote_url, destination)


class DummyProvider(VideoProvider):
    """Dev/test provider that fakes a round-trip without any network (E5)."""

    name = "dummy"

    def preflight(self) -> None:
        print("[DummyProvider] preflight ok")

    def create(self, *, prompt, negative_prompt, duration, seed, image_url=None, narration=None) -> dict:
        vid = f"dummy_{int(time.time() * 1000)}"
        print(f"[DummyProvider] create {vid}")
        return {"video_id": vid, "task_id": vid}

    def wait(self, video_id: str, task_id: Optional[str] = None) -> dict:
        return {"status": "completed", "metadata": {"url": f"http://dummy/{video_id}.mp4"}}

    def download(self, remote_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"dummy-video")
        return destination


def configured_provider_name() -> str:
    return os.getenv("DRAMAMATRIX_VIDEO_PROVIDER", "agnes").strip().lower()


def get_video_provider() -> VideoProvider:
    """Factory: return the configured provider (E5)."""
    name = configured_provider_name()
    if name == "dummy":
        return DummyProvider()
    if name in {"agnes", "ag", ""}:
        return AgnesProvider()
    # Unknown provider → fail loudly rather than silently fall back to Agnes,
    # so a mistyped DRAMAMATRIX_VIDEO_PROVIDER is caught early.
    raise AgnesConfigurationError(
        f"未知的视频 provider：{name}。可选：agnes / dummy（其它第三方预留扩展）。"
    )