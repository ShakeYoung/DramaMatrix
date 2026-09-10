"""Vision-LLM identity consistency QC (W4).

dHash 只能看结构相似，回答不了"这个人还是不是同一个角色"。本模块用
OpenAI 兼容的多模态 chat 接口做判官：输入【角色参考图 + 当前镜首帧】，
输出 0-100 的身份/服化一致性评分与理由。

- 开关：DRAMAMATRIX_IDENTITY_QC=1（默认 off，不产生任何 API 调用）。
- 阈值：DRAMAMATRIX_IDENTITY_THRESHOLD（默认 70，0-100）。
- 门禁：DRAMAMATRIX_IDENTITY_GATE=1 时低于阈值判不合格（走重绘）；
  默认只告警并落库指标。
- 模型：IDENTITY_MODEL_BASE_URL / IDENTITY_MODEL_API_KEY / IDENTITY_MODEL_NAME
  （回落 TEXT_MODEL_* / OPENAI_*，默认模型 gpt-4o-mini）。

设计约束：任何失败（无 key/网络/解析）都降级为"跳过该项检查"，绝不
阻断流水线；评分与理由作为 metrics 进入 shot_qc_results 证据链。
"""

from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Optional

import requests

from src.provider_errors import ProviderConfigurationError, ProviderError

_JUDGE_SYSTEM = (
    "你是短剧画面质检员。给你两张图：图1是角色参考图（标准形象），图2是当前镜头首帧。"
    "请评估图2中该角色的身份一致性（脸型/发型/服化/年龄段）。"
    '只输出 JSON：{"score": 0-100 整数, "reasons": "一句话理由"}，不要输出其它内容。'
)


def identity_qc_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_IDENTITY_QC", "0").strip().lower() in {"1", "true", "yes", "on"}


def identity_threshold() -> float:
    return float(os.getenv("DRAMAMATRIX_IDENTITY_THRESHOLD", "70"))


def identity_gate_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_IDENTITY_GATE", "0").strip().lower() in {"1", "true", "yes", "on"}


def _first_env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return default


def _image_data_uri(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > 8_000_000:  # 多模态接口常见上限约 10MB，留余量
        return None
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    return f"data:image/{mime};base64,{base64.b64encode(data).decode('ascii')}"


class VisionIdentityJudge:
    """OpenAI 兼容多模态判官（chat/completions + image_url）。"""

    def __init__(self) -> None:
        self.base_url = _first_env(
            "IDENTITY_MODEL_BASE_URL", "TEXT_MODEL_BASE_URL", "OPENAI_BASE_URL"
        ).rstrip("/")
        self.api_key = _first_env("IDENTITY_MODEL_API_KEY", "TEXT_MODEL_API_KEY", "OPENAI_API_KEY")
        self.model = _first_env("IDENTITY_MODEL_NAME", default="gpt-4o-mini")
        self.timeout = float(_first_env("IDENTITY_TIMEOUT_SECONDS", default="60"))
        if not self.api_key or not self.base_url:
            raise ProviderConfigurationError(
                "启用 DRAMAMATRIX_IDENTITY_QC 需要 IDENTITY_MODEL_API_KEY 与 "
                "IDENTITY_MODEL_BASE_URL（或 TEXT_MODEL_* / OPENAI_* 回落）。"
            )

    def compare(self, reference_frame: Path, current_frame: Path) -> dict:
        """Returns {"score": float, "reasons": str}；失败抛 ProviderError。"""
        ref_uri = _image_data_uri(reference_frame)
        cur_uri = _image_data_uri(current_frame)
        if not ref_uri or not cur_uri:
            raise ProviderError("参考图或当前帧不可读，无法做身份比对。")
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "图1=角色参考图；图2=当前镜头首帧。请评分。"},
                        {"type": "image_url", "image_url": {"url": ref_uri}},
                        {"type": "image_url", "image_url": {"url": cur_uri}},
                    ],
                },
            ],
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ProviderError(f"身份质检请求失败：{exc}") from exc
        if not response.ok:
            raise ProviderError(f"身份质检返回 HTTP {response.status_code}: {response.text[:300]}")
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"身份质检响应格式无效：{exc}") from exc
        return _parse_verdict(str(content))


def _parse_verdict(content: str) -> dict:
    """宽松解析判官输出中的 JSON（容忍 markdown 代码块/前后缀文本）。"""
    match = re.search(r"\{[^{}]*\}", content, re.S)
    if not match:
        raise ProviderError(f"判官输出中未找到 JSON：{content[:200]}")
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score"))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProviderError(f"判官输出解析失败：{exc}") from exc
    if not 0 <= score <= 100:
        raise ProviderError(f"判官评分越界（0-100）：{score}")
    return {"score": score, "reasons": str(data.get("reasons", ""))[:300]}


_judge: Optional[VisionIdentityJudge] = None
_judge_failed = False


def judge_identity(reference_frame: Path, current_frame: Path) -> Optional[dict]:
    """带缓存的入口；配置缺失/任何失败返回 None（调用方按告警处理）。"""
    global _judge, _judge_failed
    if not identity_qc_enabled():
        return None
    if _judge is None:
        if _judge_failed:
            return None
        try:
            _judge = VisionIdentityJudge()
        except ProviderConfigurationError as exc:
            _judge_failed = True
            print(f"   ⚠️ 身份质检未启用成功（本次运行不再重试）：{exc}")
            return None
    try:
        return _judge.compare(reference_frame, current_frame)
    except ProviderError as exc:
        print(f"   ⚠️ 身份质检失败（跳过该项，不阻断）：{exc}")
        return None
