"""Configuration boundary for OpenAI and OpenAI-compatible text-model APIs.

Providers such as DeepSeek, Qwen, vLLM, LiteLLM, and many self-hosted gateways
can be selected with TEXT_MODEL_BASE_URL as long as they expose the OpenAI chat
completions protocol expected by langchain-openai.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


class TextModelConfigurationError(RuntimeError):
    """The configured text-model provider cannot be used safely."""


@dataclass(frozen=True)
class TextModelSettings:
    api_key: str
    model: str
    base_url: str | None = None

    @classmethod
    def from_environment(cls) -> "TextModelSettings":
        """Prefer provider-neutral names while preserving existing OpenAI settings."""
        return cls(
            api_key=(os.getenv("TEXT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip(),
            model=(os.getenv("TEXT_MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4o").strip(),
            base_url=(
                os.getenv("TEXT_MODEL_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or os.getenv("OPENAI_API_BASE")
                or ""
            ).strip().rstrip("/")
            or None,
        )

    def validate(self) -> None:
        if not self.api_key:
            raise TextModelConfigurationError(
                "缺少 TEXT_MODEL_API_KEY（或兼容的 OPENAI_API_KEY）。"
            )
        if not self.model:
            raise TextModelConfigurationError(
                "缺少 TEXT_MODEL_NAME（或兼容的 OPENAI_MODEL）。"
            )
        if self.base_url:
            parsed = urlparse(self.base_url)
            if parsed.scheme not in {"https", "http"} or not parsed.netloc:
                raise TextModelConfigurationError(
                    "TEXT_MODEL_BASE_URL 必须是完整 URL，例如 https://api.example.com/v1。"
                )

    def chat_openai_kwargs(self, temperature: float) -> dict[str, Any]:
        self.validate()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
            "temperature": temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


def has_text_model_credentials() -> bool:
    return bool((os.getenv("TEXT_MODEL_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip())


def create_text_model(temperature: float = 0.7):
    """Create a ChatOpenAI client against OpenAI or a compatible provider URL."""
    settings = TextModelSettings.from_environment()
    kwargs = settings.chat_openai_kwargs(temperature)
    # Deferred import keeps environment/configuration tests independent of LangChain.
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)
