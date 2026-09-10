"""Video generation provider abstraction & factory (E5 / U1).

Decouples the pipeline from a single Agnes dependency so a vendor can be
selected via DRAMAMATRIX_VIDEO_PROVIDER (default 'agnes'), with a DummyProvider
for dev/test, and a stable extension point for future providers (sora/…).

U1 wires this abstraction into Agent 5's production path: the director now
speaks ONLY to VideoProvider (create/wait/download/render_profile) and catches
the neutral exceptions from src.provider_errors, so adding a vendor means
adding one class here — zero changes in agent5_director.py.

Real third-party adapters are NOT fabricated here — they require live API keys
and are added as `DRAMAMATRIX_VIDEO_PROVIDER` values when available.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.provider_errors import ProviderConfigurationError

from src.agnes_video import AgnesVideoClient, AgnesVideoSettings


@dataclass(frozen=True)
class RenderProfile:
    """Render constraints the pipeline needs from whichever provider is active.

    Kept deliberately small: these feed usage accounting (frames/size), QC
    redraw bounds and shot caps — none of them are Agnes-specific.
    """

    model: str
    width: int = 720
    height: int = 1280
    frame_rate: int = 24
    max_revisions: int = 2
    max_shots_per_episode: int = 0


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    try:
        return default if value in (None, "") else int(value)
    except ValueError:
        return default


class VideoProvider(ABC):
    """Minimal contract every video provider must satisfy (E5 / U1)."""

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

    @abstractmethod
    def render_profile(self) -> RenderProfile:
        """Render constraints (model/size/fps/limits) for accounting & QC."""


class AgnesProvider(VideoProvider):
    """Adapter over the existing AgnesVideoClient (E5 / U1).

    `client`/`settings` may be injected — Agent 5 passes its module-resolved
    AgnesVideoClient/AgnesVideoSettings so legacy patch points keep working;
    the factory path constructs them from the environment.
    """

    name = "agnes"

    def __init__(self, client: Optional[AgnesVideoClient] = None,
                 settings: Optional[AgnesVideoSettings] = None) -> None:
        self._settings = settings or AgnesVideoSettings.from_environment()
        self._client = client or AgnesVideoClient(self._settings)

    def preflight(self) -> None:
        self._client.preflight()

    def create(self, *, prompt, negative_prompt, duration, seed, image_url=None, narration=None) -> dict:
        # 只透传非空的可选字段，保持旧直连调用契约（缺省即不出现），
        # 便于按 kwargs 断言"是否携带参考图"的既有测试与调试习惯。
        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            seed=seed,
        )
        if image_url:
            kwargs["image_url"] = image_url
        if narration:
            kwargs["narration"] = narration
        return self._client.create_video(**kwargs)

    def wait(self, video_id: str, task_id: Optional[str] = None) -> dict:
        return self._client.wait_for_video(video_id, task_id)

    def download(self, remote_url: str, destination: Path) -> Path:
        return self._client.download_video(remote_url, destination)

    def render_profile(self) -> RenderProfile:
        s = self._settings
        return RenderProfile(
            model=s.model,
            width=s.width,
            height=s.height,
            frame_rate=s.frame_rate,
            max_revisions=s.max_revisions,
            max_shots_per_episode=s.max_shots_per_episode,
        )


class DummyProvider(VideoProvider):
    """Dev/test provider that fakes a round-trip without any network (E5).

    U1：随抽象接线产品化——DRAMAMATRIX_VIDEO_PROVIDER=dummy 现在可以端到端
    跑通整条流水线（不要求 AGNES_API_KEY），便于离线演示与集成测试。
    """

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

    def render_profile(self) -> RenderProfile:
        return RenderProfile(
            model="dummy-video-v1",
            width=_env_int("AGNES_VIDEO_WIDTH", 720),
            height=_env_int("AGNES_VIDEO_HEIGHT", 1280),
            frame_rate=_env_int("AGNES_VIDEO_FRAME_RATE", 24),
            max_revisions=_env_int("AGNES_MAX_REVISIONS", 2),
            max_shots_per_episode=_env_int("AGNES_MAX_SHOTS_PER_EPISODE", 0),
        )


def configured_provider_name() -> str:
    return os.getenv("DRAMAMATRIX_VIDEO_PROVIDER", "agnes").strip().lower()


def get_video_provider() -> VideoProvider:
    """Factory: return the configured provider (E5 / U1)."""
    name = configured_provider_name()
    if name == "dummy":
        return DummyProvider()
    if name in {"agnes", "ag", ""}:
        return AgnesProvider()
    # Unknown provider → fail loudly rather than silently fall back to Agnes,
    # so a mistyped DRAMAMATRIX_VIDEO_PROVIDER is caught early.
    raise ProviderConfigurationError(
        f"未知的视频 provider：{name}。可选：agnes / dummy（其它第三方预留扩展）。"
    )
