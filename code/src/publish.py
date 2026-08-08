"""Publish/export package generation (E2).

Packs the teaser clips, a cover frame, and the shelf publication metadata
(title / description / tags / cover prompt) into a single deliverable directory
(optionally zipped) that an operator can upload to a short-video platform.

Design is file-based and mock-friendly: without ffmpeg the cover frame is
skipped and only metadata + clips are exported (callers degrade gracefully).
"""

from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

from src.agnes_video import episode_output_dir, extract_first_frame
from src.state import EpisodeState


def publish_export_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_PUBLISH_EXPORT", "1").strip().lower() not in {"0", "false", "no", "off"}


def export_publish_package(project_id: str, ep_key: str, ep_state: EpisodeState, output_dir=None) -> Path | None:
    """Export a platform-ready publish package directory (+ zip).

    Returns the package root dir, or None if no growth assets / disabled.
    The publish_meta.json contains title/description/tags/cover_prompt and the
    source clip list for traceability.
    """
    if not publish_export_enabled():
        return None
    if not ep_state.growth_assets:
        print(f"   ⚠️ {ep_key} 无投流切片，跳过投放包导出。")
        return None

    base = output_dir or (episode_output_dir(project_id, ep_key) / "publish")
    base.mkdir(parents=True, exist_ok=True)

    meta = ep_state.growth_meta
    publish_meta = {
        "project_id": project_id,
        "ep_key": ep_key,
        "title": meta.title if meta else "",
        "description": meta.description if meta else "",
        "tags": list(meta.tags) if meta and meta.tags else [],
        "cover_prompt": meta.cover_prompt if meta else "",
        "clips": [],
        "cover": None,
        "exported_at": time.time(),
    }

    # Copy each clip into the package.
    for asset in ep_state.growth_assets:
        src = Path(asset.path)
        if not src.is_file():
            continue
        dst = base / src.name
        try:
            shutil.copy2(src, dst)
            publish_meta["clips"].append({
                "name": asset.name,
                "file": src.name,
                "start_seconds": asset.start_seconds,
                "duration_seconds": asset.duration_seconds,
                "headline": asset.headline,
                "description": asset.description,
                "tags": asset.tags,
            })
        except OSError as exc:
            print(f"   ⚠️ 复制切片失败 {src}: {exc}")

    # Cover frame from the master (degrade gracefully without ffmpeg).
    cover = None
    if ep_state.final_video_path and Path(ep_state.final_video_path).is_file():
        cover = extract_first_frame(Path(ep_state.final_video_path), base / "cover.jpg")
        if cover:
            publish_meta["cover"] = cover.name

    (base / "publish_meta.json").write_text(
        json.dumps(publish_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Zip the package for convenient upload.
    try:
        zip_path = base.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in base.iterdir():
                if item.is_file() and item.suffix != ".zip":
                    zf.write(item, item.name)
    except OSError as exc:
        print(f"   ⚠️ 打包 ZIP 失败（保留目录形式）：{exc}")

    return base