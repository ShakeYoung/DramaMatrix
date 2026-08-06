"""Agnes Video V2.0 client and local media helpers.

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
from urllib.parse import urlencode, urlparse, urlunparse

import requests


class AgnesVideoError(RuntimeError):
    """An Agnes API, download, or local-media error."""


class AgnesConfigurationError(AgnesVideoError):
    """The local Agnes configuration is incomplete or invalid."""


class AgnesTaskFailed(AgnesVideoError):
    """The remote video task reached the failed terminal state."""


class AgnesContentPolicyViolation(AgnesVideoError):
    """Agnes rejected the prompt or existing task for content policy reasons."""


class AgnesConnectionError(AgnesVideoError):
    """A network error occurred during an idempotent Agnes operation."""


class AgnesSubmissionUncertain(AgnesVideoError):
    """A create request timed out after it may have reached Agnes."""


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
    connect_timeout_seconds: int = 10
    preflight_timeout_seconds: int = 15
    poll_interval_seconds: float = 5.0
    max_poll_seconds: int = 900
    max_revisions: int = 2
    max_shots_per_episode: int = 0
    get_retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    retry_max_delay_seconds: float = 30.0
    proxy_url: str = ""

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
            connect_timeout_seconds=_int_env("AGNES_CONNECT_TIMEOUT_SECONDS", 10),
            preflight_timeout_seconds=_int_env("AGNES_PREFLIGHT_TIMEOUT_SECONDS", 15),
            poll_interval_seconds=_float_env("AGNES_POLL_INTERVAL_SECONDS", 5.0),
            max_poll_seconds=_int_env("AGNES_MAX_POLL_SECONDS", 900),
            max_revisions=_int_env("AGNES_MAX_REVISIONS", 2),
            max_shots_per_episode=_int_env("AGNES_MAX_SHOTS_PER_EPISODE", 0),
            get_retry_attempts=_int_env("AGNES_GET_RETRY_ATTEMPTS", 3),
            retry_delay_seconds=_float_env("AGNES_RETRY_DELAY_SECONDS", 2.0),
            retry_max_delay_seconds=_float_env("AGNES_RETRY_MAX_DELAY_SECONDS", 30.0),
            proxy_url=_proxy_url_from_environment(),
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
        if self.get_retry_attempts < 0 or self.retry_delay_seconds < 0:
            raise AgnesConfigurationError("Agnes 重试次数和重试间隔不能为负数。")
        if min(
            self.request_timeout_seconds,
            self.connect_timeout_seconds,
            self.preflight_timeout_seconds,
        ) <= 0:
            raise AgnesConfigurationError("Agnes 连接、请求和预检超时必须为正数。")
        if self.retry_max_delay_seconds < self.retry_delay_seconds:
            raise AgnesConfigurationError("AGNES_RETRY_MAX_DELAY_SECONDS 不能小于初始重试间隔。")
        if self.proxy_url:
            parsed_proxy = urlparse(self.proxy_url)
            if parsed_proxy.scheme not in {"http", "https"} or not parsed_proxy.hostname:
                raise AgnesConfigurationError("AGNES_PROXY_URL 必须是完整的 http(s) URL。")


def _proxy_url_from_environment() -> str:
    value = os.getenv("AGNES_PROXY_URL")
    if value is None:
        value = os.getenv("DRAMAMATRIX_PROXY_URL", "")
    value = value.strip()
    return "" if value.lower() in {"", "direct", "none", "off"} else value


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


def shot_output_dir(project_id: str, episode_id: str, version: int = 1) -> Path:
    """Directory for a specific storyboard version's rendered shots.

    Versioning (P0-A) isolates shots from different storyboard rewrites so that
    a recovery rewrite does not mix new shots with stale s01..s09 files.
    """
    base = episode_output_dir(project_id, episode_id) / "shots"
    directory = base / f"v{max(1, int(version))}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def purge_shot_versions_except(project_id: str, episode_id: str, keep_version: int) -> None:
    """Delete every shots/v{N} directory except the one to keep (P0-A).

    Called on storyboard recovery so that 'clear video_assets in state' is
    aligned with 'clear stale files on disk' — no old/new shot mixing.
    """
    base = episode_output_dir(project_id, episode_id) / "shots"
    if not base.is_dir():
        return
    keep = f"v{max(1, int(keep_version))}"
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("v") and child.name != keep:
            shutil.rmtree(child, ignore_errors=True)


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
        self.session = requests.Session()
        # Conflicting upper/lower-case proxy variables are common in long-lived
        # shells. Agnes uses only the explicit project setting.
        self.session.trust_env = False
        if settings.proxy_url:
            self.session.proxies.update({"http": settings.proxy_url, "https": settings.proxy_url})
        self.session.headers.update(
            {
                "Authorization": f"Bearer {settings.api_key}",
                "Accept": "application/json",
                "User-Agent": "DramaMatrix/1.0",
            }
        )
        # Separate session for media downloads. `metadata.url` usually points at
        # an external CDN, NOT the Agnes API host; sending the API key there
        # would leak credentials to a third party. This session deliberately
        # carries no Authorization header (security, see review F1).
        self.download_session = requests.Session()
        self.download_session.trust_env = False
        if settings.proxy_url:
            self.download_session.proxies.update(
                {"http": settings.proxy_url, "https": settings.proxy_url}
            )
        self.download_session.headers.update({"User-Agent": "DramaMatrix/1.0"})

    @property
    def result_base_url(self) -> str:
        parsed = urlparse(self.settings.base_url)
        return urlunparse((parsed.scheme, parsed.netloc, "", "", "", "")).rstrip("/")

    @property
    def route_description(self) -> str:
        if not self.settings.proxy_url:
            return "direct"
        parsed = urlparse(self.settings.proxy_url)
        return f"proxy {parsed.hostname}:{parsed.port or 80}"

    @property
    def request_timeout(self) -> tuple[int, int]:
        return (
            self.settings.connect_timeout_seconds,
            self.settings.request_timeout_seconds,
        )

    def preflight(self) -> None:
        """Verify a real HTTPS path and credentials without creating a video task."""
        url = f"{self.settings.base_url}/models"
        try:
            response = self.session.get(
                url,
                timeout=(
                    self.settings.connect_timeout_seconds,
                    self.settings.preflight_timeout_seconds,
                ),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise AgnesConnectionError(
                "Agnes 预检失败（preflight）：无法通过 "
                f"{self.route_description} 连接 {urlparse(url).hostname}:443；"
                "请配置可用且支持远程 DNS 的 AGNES_PROXY_URL。"
                f"原始错误: {exc}"
            ) from exc
        if response.status_code in {401, 403}:
            raise AgnesConfigurationError(
                f"Agnes 预检失败（preflight）：API Key 被拒绝（HTTP {response.status_code}）。"
            )
        if response.status_code >= 500:
            raise AgnesConnectionError(
                f"Agnes 预检失败（preflight）：网关返回 HTTP {response.status_code}，请稍后重试。"
            )
        if not (200 <= response.status_code < 300):
            # 404/429/3xx redirects are NOT a healthy sign for /models.
            raise AgnesConnectionError(
                f"Agnes 预检失败（preflight）：端点返回 HTTP {response.status_code}，"
                "预期为 2xx。请检查 base_url 与网关状态。"
            )
        print(
            f"✅ Agnes HTTPS 预检通过：{self.route_description} -> "
            f"{urlparse(self.settings.base_url).hostname}"
        )

    def _retry_delay(self, attempt: int, response: Optional[requests.Response] = None) -> float:
        if response is not None:
            retry_after = response.headers.get("Retry-After", "").strip()
            try:
                return min(float(retry_after), self.settings.retry_max_delay_seconds)
            except ValueError:
                pass
        return min(
            self.settings.retry_delay_seconds * (2**attempt),
            self.settings.retry_max_delay_seconds,
        )

    def _request_json(
        self,
        method: str,
        url: str,
        payload: Optional[Mapping[str, Any]] = None,
        stage: str = "request",
    ) -> dict[str, Any]:
        retry_count = self.settings.get_retry_attempts if method.upper() == "GET" else 0
        for attempt in range(retry_count + 1):
            try:
                response = self.session.request(
                    method,
                    url,
                    json=payload,
                    timeout=self.request_timeout,
                )
                detail = response.text[:600]
                if response.status_code in {429, 500, 502, 503, 504} and attempt < retry_count:
                    delay = self._retry_delay(attempt, response)
                    print(
                        f"   Agnes {stage} 返回 HTTP {response.status_code}，"
                        f"{delay:g} 秒后进行第 {attempt + 1}/{retry_count} 次重试。"
                    )
                    time.sleep(delay)
                    continue
                if "content_policy_violation" in detail:
                    raise AgnesContentPolicyViolation(
                        f"Agnes {stage} HTTP {response.status_code}: {detail}"
                    )
                if not response.ok:
                    raise AgnesVideoError(
                        f"Agnes {stage} HTTP {response.status_code}: {detail}"
                    )
                result = response.json()
                if not isinstance(result, dict):
                    raise AgnesVideoError(f"Agnes {stage} 返回格式无效。")
                return result
            except AgnesVideoError:
                raise
            except (requests.Timeout, requests.ConnectionError) as exc:
                if method.upper() == "POST":
                    raise AgnesSubmissionUncertain(
                        "Agnes 创建任务失败（create）：请求结果未知。远端可能已收到任务，"
                        "为防止重复提交，本项目不会自动重试；请先在 Agnes 控制台核对。"
                        f"原始错误: {exc}"
                    ) from exc
                if attempt < retry_count:
                    delay = self._retry_delay(attempt)
                    print(
                        f"   Agnes {stage} 网络中断，{delay:g} 秒后进行第 "
                        f"{attempt + 1}/{retry_count} 次重试：{exc}"
                    )
                    time.sleep(delay)
                    continue
                raise AgnesConnectionError(
                    f"Agnes {stage} 失败：通过 {self.route_description} 连接 "
                    f"{urlparse(url).hostname} 时网络中断: {exc}"
                ) from exc
            except (requests.RequestException, json.JSONDecodeError) as exc:
                raise AgnesVideoError(f"Agnes {stage} 响应处理失败: {exc}") from exc
        raise AssertionError("unreachable")

    def create_video(
        self,
        prompt: str,
        negative_prompt: str,
        duration: str,
        seed: int,
        image: Optional[str] = None,
        image_url: Optional[str] = None,
        last_frame: Optional[str] = None,
        last_frame_url: Optional[str] = None,
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
        # P1-B：首帧/参考图与关键帧尾帧条件输入（OpenAI 兼容假设，字段名可配）。
        # 仅当传值时注入；字段名经 env 可配置，便于服务器端按真实接口联调校准。
        image_field = os.getenv("AGNES_IMAGE_FIELD", "image")
        image_url_field = os.getenv("AGNES_IMAGE_URL_FIELD", "image_url")
        last_frame_field = os.getenv("AGNES_LAST_FRAME_FIELD", "last_frame")
        last_frame_url_field = os.getenv("AGNES_LAST_FRAME_URL_FIELD", "last_frame_url")
        if image:
            payload[image_field] = image
        if image_url:
            payload[image_url_field] = image_url
        if last_frame:
            payload[last_frame_field] = last_frame
        if last_frame_url:
            payload[last_frame_url_field] = last_frame_url
        result = self._request_json(
            "POST", f"{self.settings.base_url}/videos", payload, stage="create"
        )
        if not (result.get("video_id") or result.get("task_id") or result.get("id")):
            raise AgnesVideoError("创建任务响应未包含 video_id、task_id 或 id。")
        return result

    def get_video(self, video_id: str) -> dict[str, Any]:
        query = urlencode({"video_id": video_id, "model_name": self.settings.model})
        return self._request_json(
            "GET", f"{self.result_base_url}/agnesapi?{query}", stage="poll"
        )

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Compatibility endpoint for responses that do not include a video_id."""
        return self._request_json(
            "GET", f"{self.settings.base_url}/videos/{task_id}", stage="poll"
        )

    def wait_for_video(self, video_id: Optional[str], task_id: Optional[str] = None) -> dict[str, Any]:
        if not video_id and not task_id:
            raise AgnesVideoError("轮询任务需要 video_id 或 task_id。")
        deadline = time.monotonic() + self.settings.max_poll_seconds
        while time.monotonic() < deadline:
            result = self.get_video(video_id) if video_id else self.get_task(str(task_id))
            status = str(result.get("status", "")).lower()
            if status == "completed":
                # `/agnesapi` can report completion a little before it exposes
                # `metadata.url`.  The task endpoint is then the authoritative
                # fallback (and may already contain the finished asset URL).
                if (result.get("metadata") or {}).get("url") or not task_id:
                    return result
                detail = self.get_task(str(task_id))
                detail_status = str(detail.get("status", "")).lower()
                if detail_status == "failed":
                    error = detail.get("error") or "远端视频任务失败。"
                    raise AgnesTaskFailed(str(error))
                if detail_status == "completed" and (detail.get("metadata") or {}).get("url"):
                    return detail
                # The completion status and asset URL are eventually consistent;
                # keep polling instead of failing this entire episode early.
                time.sleep(self.settings.poll_interval_seconds)
                continue
            if status == "failed":
                error = result.get("error") or "远端视频任务失败。"
                raise AgnesTaskFailed(str(error))
            if status not in {"queued", "pending", "in_progress", "processing", "running", ""}:
                raise AgnesVideoError(f"Agnes 返回未知任务状态: {status}")
            time.sleep(self.settings.poll_interval_seconds)
        raise AgnesVideoError(f"等待视频任务 {video_id or task_id} 超时。")

    def download_video(self, remote_url: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(self.settings.get_retry_attempts + 1):
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with self.download_session.get(
                    remote_url,
                    headers=headers,
                    timeout=self.request_timeout,
                    stream=True,
                ) as response:
                    if response.status_code == 416 and offset:
                        temporary.unlink(missing_ok=True)
                        raise requests.ConnectionError("远端不接受已有断点，已重置临时文件")
                    response.raise_for_status()
                    mode = "ab" if offset and response.status_code == 206 else "wb"
                    with temporary.open(mode) as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                if temporary.exists() and temporary.stat().st_size > 0:
                    temporary.replace(destination)
                    return destination
                raise AgnesVideoError("下载的视频文件为空。")
            except (requests.Timeout, requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as exc:
                if attempt < self.settings.get_retry_attempts:
                    delay = self._retry_delay(attempt)
                    retained_bytes = temporary.stat().st_size if temporary.exists() else 0
                    print(
                        f"   Agnes download 网络中断，已保留 {retained_bytes} 字节断点；"
                        f"{delay:g} 秒后进行第 {attempt + 1}/"
                        f"{self.settings.get_retry_attempts} 次重试：{exc}"
                    )
                    time.sleep(delay)
                    continue
                raise AgnesConnectionError(
                    f"Agnes download 失败：已保留临时文件 {temporary}，下次将断点续传: {exc}"
                ) from exc
            except requests.HTTPError as exc:
                raise AgnesVideoError(
                    f"Agnes download HTTP {exc.response.status_code}: {remote_url}"
                ) from exc
        raise AssertionError("unreachable")


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise AgnesConfigurationError(f"未找到 {name}；请安装 FFmpeg 并确保 {name} 在 PATH 中。")
    return path


def trim_unstable_frames(path: Path, workdir: Path, head: Optional[int] = None, tail: Optional[int] = None) -> Path:
    """Trim unstable head/tail frames from a shot (P2-B).

    AI-generated clips often have jittery first/last frames; cropping them per
    shot before concat avoids wobble at cut points. head/tail default to env
    DRAMAMATRIX_TRIM_HEAD / DRAMAMATRIX_TRIM_TAIL (default 0 = no trim, so the
    behavior is unchanged unless explicitly enabled). Without ffmpeg the input
    is returned unchanged.
    """
    head = int(os.getenv("DRAMAMATRIX_TRIM_HEAD", "0")) if head is None else head
    tail = int(os.getenv("DRAMAMATRIX_TRIM_TAIL", "0")) if tail is None else tail
    if head <= 0 and tail <= 0:
        return path
    try:
        ffmpeg = _require_binary("ffmpeg")
        ffprobe = _require_binary("ffprobe")
    except AgnesConfigurationError:
        return path
    # Read fps to convert frame counts to seconds.
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True, text=True,
        )
        frac = probe.stdout.strip().split("/")
        fps = float(frac[0]) / float(frac[1]) if len(frac) == 2 and float(frac[1]) else 24.0
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError):
        fps = 24.0
    start = head / fps
    out = workdir / f"trim_{path.name}"
    command = [ffmpeg, "-y", "-ss", f"{start:.3f}", "-i", str(path)]
    if tail > 0:
        # -sseof-style tail trim via tduration; approximate by trimming tail seconds.
        command += ["-t", f"max(0, duration-{(head + tail) / fps})"]
    command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(out)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        return path
    return out if out.is_file() else path


def concat_videos(inputs: list[Path], destination: Path) -> Path:
    if not inputs:
        raise AgnesVideoError("没有可合成的视频素材。")
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise AgnesVideoError(f"以下分镜视频不存在: {', '.join(missing)}")
    ffmpeg = _require_binary("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # 逐镜归一化：统一分辨率、帧率、编码与像素格式，避免镜头参数不一致
    # 导致的拼接失败或镜头间抖动。若无 ffmpeg（测试/降级）则直接复制返回。
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        destination.write_bytes(inputs[0].read_bytes())
        return destination

    # P2-B：拼接前裁掉每段头尾不稳定帧（默认 0 不裁，可配开启）。
    trimmed = [trim_unstable_frames(path, destination.parent) for path in inputs]
    normalized = _normalize_shots(ffmpeg, trimmed, destination.parent)
    list_file = destination.with_suffix(".concat.txt")
    lines = [
        f"file '{path.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"
        for path in normalized
    ]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # P2-B：同场景硬切（concat demuxer 即硬切），不统一加交叉淡化，避免人物漂移时的双脸/重影。
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
        for path in normalized:
            path.unlink(missing_ok=True)
    return destination


def _normalize_shots(ffmpeg: str, inputs: list[Path], workdir: Path) -> list[Path]:
    """Re-encode each shot to a common 720x1280@24 profile before concat.

    Env vars AGNES_VIDEO_WIDTH / AGNES_VIDEO_HEIGHT / AGNES_VIDEO_FRAME_RATE
    define the target; defaults match the Agnes generation settings.
    """
    width = _int_env("AGNES_VIDEO_WIDTH", 720)
    height = _int_env("AGNES_VIDEO_HEIGHT", 1280)
    fps = _int_env("AGNES_VIDEO_FRAME_RATE", 24)
    normalized: list[Path] = []
    for index, source in enumerate(inputs):
        out = workdir / f"norm_{index:03d}.mp4"
        scale_spec = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        command = [
            ffmpeg, "-y", "-i", str(source),
            "-vf", f"{scale_spec},fps={fps}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "44100",
            "-movflags", "+faststart", str(out),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise AgnesVideoError(f"FFmpeg 分镜归一化失败({source}): {exc.stderr[-1200:]}") from exc
        normalized.append(out)
    return normalized


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


def extract_last_frame(video_path: Path, destination: Path) -> Optional[Path]:
    """Extract the last stable frame of a shot to use as the next shot's first
    frame (P1-C). Returns None when ffmpeg is unavailable so callers can degrade
    to plain text-to-video instead of failing the whole episode.
    """
    try:
        ffmpeg = _require_binary("ffmpeg")
    except AgnesVideoError:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-y", "-sseof", "-0.3", "-i", str(video_path),
        "-frames:v", "1", "-q:v", "2", str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"   ⚠️ 提取尾帧失败，将退回纯文生视频：{exc.stderr[-300:]}")
        return None
    if destination.is_file():
        return destination
    return None
