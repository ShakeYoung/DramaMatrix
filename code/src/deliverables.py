"""Deliverable (成片级) integrity evidence helper (F4).

Records hash / real duration / audio presence / source-shots for final
deliverables (master, voiced, subtitled, teaser clips) so the experiment trail
extends past per-shot assets.
"""

from __future__ import annotations

import time
from pathlib import Path

from src.agnes_video import media_integrity
from src.state import DeliverableAsset, EpisodeState


def record_deliverable(ep_state: EpisodeState, kind: str, path: Path, source_shots: list[str]) -> None:
    """采集一份成片级交付资产的完整性证据并写入 ep_state.deliverables。

    同 kind 替换（保留最新版本），其余保留。
    """
    try:
        info = media_integrity(path)
        asset = DeliverableAsset(
            kind=kind,
            path=str(path),
            sha256=info.get("sha256"),
            file_size_bytes=info.get("file_size_bytes"),
            actual_duration=info.get("actual_duration"),
            has_audio=info.get("has_audio"),
            source_shots=list(source_shots),
            created_at=time.time(),
        )
        ep_state.deliverables = [d for d in ep_state.deliverables if d.kind != kind] + [asset]
    except Exception as exc:  # noqa: BLE001 - never block the pipeline on evidence
        print(f"   ⚠️ 成片证据采集失败（不阻断）：{exc}")
