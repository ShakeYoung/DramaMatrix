"""Minimal Agnes Video V2.0 client and local media helpers.

The module uses the documented asynchronous API:
POST /v1/videos -> GET /agnesapi?video_id=<id> -> metadata.url.
No credentials are written to disk or included in persisted project state.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


class AgnesVideoError(RuntimeError):
    """An Agnes API, download, or local-media error."""


class AgnesConfigurationError(AgnesVideoError):
    """The local Agnes configuration is incomplete or invalid."""


class AgnesTaskFailed(AgnesVideoError):
    """The remote video task reached the failed terminal state."""


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value in (None, "") else int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value in (None, "") else float(value)


@dataclass(frozen=True)
class AgnesVideoSettings:
    api_key: str
    base_url: str = "https://apihub.agnes-ai.com/v1"
    model: str = "agnes-video-v2.0"
    width: int = 720
    height: int = 1280
    frame_rate: int = 24
    request_timeout_seconds: int = 60
    poll_interval_seconds: float = 5.0
    max_poll_seconds: int = 900
    max_revisions: int = 2
    max_shots_per_episode: int = 0

    @classmethod
    def from_environment(cls) -> "AgnesVideoSettings":
        return cls(
            api_key=os.getenv("AGNES_API_KEY", "").strip(),
            base_url=os.getenv("AGNES_API_BASE_URL", "https://apihub.agnes-ai.com/v1").rstrip("/"),
            model=os.getenv("AGNES_VIDEO_MODEL", "agnes-video-v2.0"),
            width=_int_env("AGNES_VIDEO_WIDTH", 720),
            height=_int_env("AGNES_VIDEO_HEIGHT", 1280),
            frame_rate=_int_env("AGNES_VIDEO_FRAME_RATE", 24),
            request_timeout_seconds=_int_env("AGNES_REQUEST_TIMEOUT_SECONDS", 60),
            poll_interval_seconds=_float_env("AGNES_POLL_INTERVAL_SECONDS", 5.0),
            max_poll_seconds=_int_env("AGNES_MAX_POLL_SECONDS", 900),
            max_revisions=_int_env("AGNES_MAX_REVISIONS", 2),
            max_shots_per_episode=_int_env("AGNES_MAX_SHOTS_PER_EPISODE", 0),
        )

    def validate(self) -> None:
        if not self.api_key:
            raise AgnesConfigurationError("缺少 AGNES_API_KEY；请在 .env 中配置，切勿写入源码。")
        if self.model != "agnes-video-v2.0":
            raise AgnesConfigurationError("AGNES_VIDEO_MODEL 必须为 agnes-video-v2.0。")
        if not self.base_url.startswith(("https://", "http://")):
            raise AgnesConfigurationError("AGNES_API_BASE_URL 必须是完整的 http(s) URL。")
        if not (1 <= self.frame_rate <= 60):
            raise AgnesConfigurationError("AGNES_VIDEO_FRAME_RATE 必须在 1–60 之间。")
        if self.width <= 0 or self.height <= 0:
            raise AgnesConfigurationError("AGNES_VIDEO_WIDTH 和 AGNES_VIDEO_HEIGHT 必须为正整数。")


def safe_component(value: str) -> str:
    """Return a portable file-name component without allowing path traversal."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "asset"


def output_root() -> Path:
    configured = os.getenv("DRAMAMATRIX_OUTPUT_DIR", "").strip()
    root = Path(configured) if configured else Path(__file__).resolve().parents[1] / "outputs"
    return root.resolve()


def episode_output_dir(project_id: str, episode_id: str) -> Path:
    directory = output_root() / safe_component(project_id) / safe_component(episode_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def frames_for_duration(duration: str, frame_rate: int) -> int:
    """Convert a storyboard duration to a valid Agnes 8n+1 frame count."""
    match = re.search(r"\d+(?:\.\d+)?", duration or "")
    seconds = float(match.group()) if match else 5.0
    target_frames = max(9, round(seconds * frame_rate))
    valid_frames = 8 * round((target_frames - 1) / 8) + 1
    return max(9, min(441, valid_frames))


class AgnesVideoClient:
    def __init__(self, settings: AgnesVideoSettings):
        settings.validate()
        self.settings = settings

    @property
    def result_base_url(self) -> str:
        parsed = urlparse(self.settings.base_url)
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:600]
            raise AgnesVideoError(f"Agnes API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise AgnesVideoError(f"无法连接 Agnes API: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AgnesVideoError("Agnes API 请求超时。") from exc
        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AgnesVideoError("Agnes API 返回了非 JSON 响应。") from exc
        if not isinstance(result, dict):
            raise AgnesVideoError("Agnes API 返回格式无效。")
        return result

    def create_video(
        self,
        prompt: str,
        negative_prompt: str,
        duration: str,
        seed: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.settings.model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": self.settings.width,
            "height": self.settings.height,
            "num_frames": frames_for_duration(duration, self.settings.frame_rate),
            "frame_rate": self.settings.frame_rate,
            "seed": seed,
        }
        result = self._request_json("POST", f"{self.settings.base_url}/videos", payload)
        if not (result.get("video_id") or result.get("task_id") or result.get("id")):
            raise AgnesVideoError("创建任务响应未包含 video_id、task_id 或 id。")
        return result

    def get_video(self, video_id: str) -> dict[str, Any]:
        query = urlencode({"video_id": video_id, "model_name": self.settings.model})
        return self._request_json("GET", f"{self.result_base_url}/agnesapi?{query}")

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Compatibility endpoint for responses that do not include a video_id."""
        return self._request_json("GET", f"{self.settings.base_url}/videos/{task_id}")

    def wait_for_video(self, video_id: Optional[str], task_id: Optional[str] = None) -> dict[str, Any]:
        if not video_id and not task_id:
            raise AgnesVideoError("轮询任务需要 video_id 或 task_id。")
        deadline = time.monotonic() + self.settings.max_poll_seconds
        while time.monotonic() < deadline:
            result = self.get_video(video_id) if video_id else self.get_task(str(task_id))
            status = str(result.get("status", "")).lower()
            if status == "completed":
                return result
            if status == "failed":
                error = result.get("error") or "远端视频任务失败。"
                raise AgnesTaskFailed(str(error))
            if status not in {"queued", "in_progress", ""}:
                raise AgnesVideoError(f"Agnes 返回未知任务状态: {status}")
            time.sleep(self.settings.poll_interval_seconds)
        raise AgnesVideoError(f"等待视频任务 {video_id or task_id} 超时。")

    def download_video(self, remote_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        request = Request(remote_url, method="GET")
        try:
            with urlopen(request, timeout=self.settings.request_timeout_seconds) as response:
                with temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
        except (HTTPError, URLError, TimeoutError) as exc:
            temporary.unlink(missing_ok=True)
            raise AgnesVideoError(f"无法下载 Agnes 视频成品: {exc}") from exc
        if temporary.stat().st_size == 0:
            temporary.unlink(missing_ok=True)
            raise AgnesVideoError("下载的视频文件为空。")
        temporary.replace(destination)
        return destination


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AgnesConfigurationError(f"未找到 {name}；请安装 FFmpeg 并确保 {name} 在 PATH 中。")
    return path


def concat_videos(inputs: list[Path], destination: Path) -> Path:
    if not inputs:
        raise AgnesVideoError("没有可合成的视频素材。")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise AgnesVideoError(f"以下分镜视频不存在: {', '.join(missing)}")
    ffmpeg = _require_binary("ffmpeg")
    list_file = destination.with_suffix(".concat.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{path.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in inputs]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-movflags", "+faststart", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AgnesVideoError(f"FFmpeg 合成失败: {exc.stderr[-1200:]}") from exc
    finally:
        list_file.unlink(missing_ok=True)
    return destination


def video_duration(path: Path) -> float:
    ffprobe = _require_binary("ffprobe")
    command = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise AgnesVideoError(f"无法读取视频时长: {path}") from exc


def cut_video(source: Path, destination: Path, start_seconds: float, duration_seconds: float) -> Path:
    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-ss", f"{start_seconds:.3f}", "-i", str(source),
        "-t", f"{duration_seconds:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise AgnesVideoError(f"FFmpeg 切片失败: {exc.stderr[-1200:]}") from exc
    return destination
