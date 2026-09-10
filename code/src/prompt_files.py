"""Externalized prompt files (U5).

核心创作 prompt 从 Python 源码外置到 prompts/*.md，运营可在不改动代码、
不重启流水线的情况下调风格（对标 Toonflow 的 Skill 文件思路）。

- 目录：默认 <code>/prompts/，可用 DRAMAMATRIX_PROMPT_DIR 覆盖。
- 命名：load_prompt("agent3_head_writer") -> prompts/agent3_head_writer.md
- 占位符：文件内容中的 {format_instructions}/{prior_knowledge}/
  {error_message} 由调用方 .replace() 填充（不用 str.format，避免用户
  编辑引入的花括号导致 KeyError）。
- 回退：文件缺失/读失败时返回内置默认，测试与离线运行不受影响。
"""

from __future__ import annotations

import os
from pathlib import Path


def prompt_dir() -> Path:
    configured = os.getenv("DRAMAMATRIX_PROMPT_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str, default: str) -> str:
    """Return the content of prompts/<name>.md, or `default` when unavailable."""
    path = prompt_dir() / f"{name}.md"
    try:
        if path.is_file():
            content = path.read_text(encoding="utf-8")
            if content.strip():
                return content.rstrip("\n")
    except OSError as exc:
        print(f"   ⚠️ 读取 prompt 文件失败，使用内置默认（{name}）：{exc}")
    return default
