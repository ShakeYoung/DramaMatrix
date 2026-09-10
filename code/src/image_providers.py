"""Image generation provider abstraction (U2).

角色/场景参考图此前的 reference_image_prompt 字段只生成不消费；本模块给出
图像生成抽象，让参考图链路（生成 → 落盘 → 哈希 → 注入视频任务）可以真正
跑通。与视频侧一致：中立异常（provider_errors）+ 工厂 + Dummy 实现供测试。

配置：
- DRAMAMATRIX_IMAGE_PROVIDER: off（默认，保持旧行为）/ dummy / openai
- openai（OpenAI 兼容 /images/generations）：
  IMAGE_MODEL_BASE_URL（回落 TEXT_MODEL_BASE_URL / OPENAI_BASE_URL）
  IMAGE_MODEL_API_KEY（回落 TEXT_MODEL_API_KEY / OPENAI_API_KEY）
  IMAGE_MODEL_NAME（默认 gpt-image-1）、IMAGE_SIZE（默认 1024x1024）
"""

from __future__ import annotations

import base64
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

from src.provider_errors import ProviderConfigurationError, ProviderConnectionError, ProviderError

# 1x1 灰度 PNG：Dummy 输出需要是"真实图片文件"，保证哈希/缩略图/文件检查可用。
_MINIMAL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class ImageProvider(ABC):
    """Minimal contract every image provider must satisfy (U2)."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, destination: Path) -> Path:
        """Generate one image for `prompt` and write it to `destination`."""


class DummyImageProvider(ImageProvider):
    """Dev/test provider：写入内置 1x1 PNG，无任何网络请求。"""

    name = "dummy"

    def generate(self, prompt: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(_MINIMAL_PNG)
        print(f"[DummyImageProvider] 已生成占位参考图 -> {destination.name}")
        return destination


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


class OpenAICompatImageProvider(ImageProvider):
    """OpenAI 兼容 /v1/images/generations 适配（b64_json 优先，URL 兜底下载）。"""

    name = "openai"

    def __init__(self) -> None:
        self.base_url = _first_env("IMAGE_MODEL_BASE_URL", "TEXT_MODEL_BASE_URL", "OPENAI_BASE_URL").rstrip("/")
        self.api_key = _first_env("IMAGE_MODEL_API_KEY", "TEXT_MODEL_API_KEY", "OPENAI_API_KEY")
        self.model = _first_env("IMAGE_MODEL_NAME", default="gpt-image-1")
        self.size = _first_env("IMAGE_SIZE", default="1024x1024")
        self.timeout = float(_first_env("IMAGE_TIMEOUT_SECONDS", default="120"))
        if not self.api_key:
            raise ProviderConfigurationError(
                "启用 openai 图像供应商需要 IMAGE_MODEL_API_KEY（或 TEXT_MODEL_API_KEY / OPENAI_API_KEY）。"
            )
        if not self.base_url:
            raise ProviderConfigurationError(
                "启用 openai 图像供应商需要 IMAGE_MODEL_BASE_URL（或 TEXT_MODEL_BASE_URL / OPENAI_BASE_URL）。"
            )

    def generate(self, prompt: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.post(
                f"{self.base_url}/images/generations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "prompt": prompt, "size": self.size, "n": 1},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderConnectionError(f"图像生成请求失败（{self.model}）：{exc}") from exc
        if not response.ok:
            raise ProviderError(f"图像生成返回 HTTP {response.status_code}: {response.text[:400]}")
        try:
            payload = response.json()
            item = (payload.get("data") or [])[0]
        except (ValueError, IndexError, AttributeError) as exc:
            raise ProviderError(f"图像生成响应格式无效：{exc}") from exc
        b64 = item.get("b64_json")
        if b64:
            destination.write_bytes(base64.b64decode(b64))
            return destination
        url = item.get("url")
        if url:
            try:
                image = requests.get(url, timeout=self.timeout)
                image.raise_for_status()
            except requests.RequestException as exc:
                raise ProviderConnectionError(f"参考图下载失败：{exc}") from exc
            destination.write_bytes(image.content)
            return destination
        raise ProviderError("图像生成响应未包含 b64_json 或 url。")


def configured_image_provider_name() -> str:
    return os.getenv("DRAMAMATRIX_IMAGE_PROVIDER", "").strip().lower()


def get_image_provider() -> Optional[ImageProvider]:
    """Factory：返回配置的图像供应商；off/未配置返回 None（保持旧行为）。"""
    name = configured_image_provider_name()
    if name in {"", "off", "none", "disabled"}:
        return None
    if name == "dummy":
        return DummyImageProvider()
    if name in {"openai", "openai-compat", "openai_compat"}:
        return OpenAICompatImageProvider()
    raise ProviderConfigurationError(
        f"未知的图像 provider：{name}。可选：off / dummy / openai。"
    )
