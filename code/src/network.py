"""Network environment setup shared by local and cluster entry points."""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse


_PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def configure_proxy_environment() -> None:
    """Enable the configured proxy only when its endpoint is reachable."""
    proxy_url = os.getenv("DRAMAMATRIX_PROXY_URL", "").strip()
    if not proxy_url or proxy_url.lower() in {"direct", "none", "off"}:
        clear_proxy_environment()
        return
    parsed = urlparse(proxy_url)
    if not parsed.hostname or not parsed.port:
        print(f"⚠️ DRAMAMATRIX_PROXY_URL 无效，已改为直连：{proxy_url}")
        clear_proxy_environment()
        return
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            pass
    except OSError as exc:
        print(f"⚠️ 全局代理不可用（{proxy_url}: {exc}），已禁用该代理。")
        clear_proxy_environment()
        return
    for name in _PROXY_VARIABLES:
        os.environ[name] = proxy_url
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1,::1")
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1,::1")


def clear_proxy_environment() -> None:
    for name in _PROXY_VARIABLES:
        os.environ.pop(name, None)
