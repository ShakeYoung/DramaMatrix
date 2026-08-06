"""Command-line overrides for one DramaMatrix run.

Values are applied to the process environment only; `.env` is never rewritten.
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence


def parse_runtime_options(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume a DramaMatrix production project.")
    parser.add_argument("--project-id", help="项目 ID；相同 ID 才能恢复同一份 SQLite 快照。")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="恢复已有快照；使用 --no-resume 强制从头开始。",
    )
    parser.add_argument(
        "--agnes-get-retry-attempts",
        type=int,
        help="Agnes 轮询和下载的额外 GET 重试次数（默认读取 .env 或 3）。",
    )
    parser.add_argument(
        "--agnes-retry-delay-seconds",
        type=float,
        help="Agnes GET 重试间隔秒数（默认读取 .env 或 2）。",
    )
    parser.add_argument(
        "--agnes-preflight-only",
        action="store_true",
        help="只验证 Agnes HTTPS、代理和 API Key，不创建视频任务。",
    )
    return parser.parse_args(argv)


def apply_runtime_options(options: argparse.Namespace) -> None:
    overrides = {
        "DRAMAMATRIX_PROJECT_ID": options.project_id,
        "DRAMAMATRIX_RESUME": None if options.resume is None else ("1" if options.resume else "0"),
        "AGNES_GET_RETRY_ATTEMPTS": (
            None if options.agnes_get_retry_attempts is None else str(options.agnes_get_retry_attempts)
        ),
        "AGNES_RETRY_DELAY_SECONDS": (
            None if options.agnes_retry_delay_seconds is None else str(options.agnes_retry_delay_seconds)
        ),
        "DRAMAMATRIX_AGNES_PREFLIGHT_ONLY": "1" if options.agnes_preflight_only else None,
    }
    for name, value in overrides.items():
        if value is not None:
            os.environ[name] = value
