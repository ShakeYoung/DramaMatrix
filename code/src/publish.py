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

from src.agnes_video import episode_output_dir, extract_first_frame, sha256_file
from src.state import EpisodeState


def publish_export_enabled() -> bool:
    return os.getenv("DRAMAMATRIX_PUBLISH_EXPORT", "1").strip().lower() not in {"0", "false", "no", "off"}


def export_publish_package(project_id: str, ep_key: str, ep_state: EpisodeState, output_dir=None) -> Path | None:
    """Export a platform-ready publish package directory (+ zip).

    Returns the package root dir, or None if no growth assets / disabled.
    The publish_meta.json contains title/description/tags/cover_prompt and the
    source clip list (R2: 每个切片/封面带 SHA-256) for traceability.
    """
    if not publish_export_enabled():
        return None
    if not ep_state.growth_assets:
        print(f"   ⚠️ {ep_key} 无投流切片，跳过投放包导出。")
        return None

    base = output_dir or (episode_output_dir(project_id, ep_key) / "publish")
    base.mkdir(parents=True, exist_ok=True)

    meta = ep_state.growth_meta
    # W3：投放包携带权属块——来源授权 + 生成物权利声明 + AI 内容披露。
    from src.rights import rights_block

    publish_meta = {
        "project_id": project_id,
        "ep_key": ep_key,
        "title": meta.title if meta else "",
        "description": meta.description if meta else "",
        "tags": list(meta.tags) if meta and meta.tags else [],
        "cover_prompt": meta.cover_prompt if meta else "",
        "rights": rights_block(project_id),
        "clips": [],
        "cover": None,
        "exported_at": time.time(),
    }

    # Copy each clip into the package. R2：缺失的切片不再静默跳过——记录到
    # missing 列表，导出后由 verify_publish_package 阻断 growth_ready。
    missing_clips: list[str] = []
    for asset in ep_state.growth_assets:
        src = Path(asset.path)
        if not src.is_file():
            missing_clips.append(str(src))
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
                "sha256": sha256_file(dst),
                "file_size_bytes": dst.stat().st_size if dst.is_file() else None,
            })
        except OSError as exc:
            missing_clips.append(f"{src}（复制失败：{exc}）")
    if missing_clips:
        print(f"   ⚠️ {ep_key} 有 {len(missing_clips)} 个切片缺失/复制失败：{missing_clips}")

    # Cover frame from the master (degrade gracefully without ffmpeg).
    cover = None
    if ep_state.final_video_path and Path(ep_state.final_video_path).is_file():
        cover = extract_first_frame(Path(ep_state.final_video_path), base / "cover.jpg")
        if cover:
            publish_meta["cover"] = cover.name
            publish_meta["cover_sha256"] = sha256_file(cover)

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


def verify_publish_package(base: Path, ep_state: EpisodeState) -> list[str]:
    """R2 交付门禁：验证投放包完整性，返回问题清单（空=通过）。

    检查项：publish_meta.json 存在且可解析；每个投流切片都在包内、非空且
    哈希与清单一致；封面存在（清单声明时）；ZIP 存在。任一不满足都不能
    标记 growth_ready（"就绪"必须意味着完整交付包已经存在）。
    """
    problems: list[str] = []
    meta_path = base / "publish_meta.json"
    if not meta_path.is_file():
        return [f"缺少 publish_meta.json：{meta_path}"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [f"publish_meta.json 无法解析：{exc}"]

    clips_in_meta = {c.get("file"): c for c in (meta.get("clips") or [])}
    for asset in ep_state.growth_assets:
        expected = Path(asset.path).name
        entry = clips_in_meta.get(expected)
        if not entry:
            problems.append(f"切片未进入交付包：{expected}（源文件缺失或复制失败）")
            continue
        packaged = base / expected
        if not packaged.is_file() or packaged.stat().st_size == 0:
            problems.append(f"包内切片缺失或为空：{packaged}")
            continue
        recorded_sha = entry.get("sha256")
        if recorded_sha:
            actual_sha = sha256_file(packaged)
            if actual_sha != recorded_sha:
                problems.append(f"包内切片哈希与清单不一致：{expected}")
        else:
            problems.append(f"清单缺少切片哈希：{expected}")

    if meta.get("cover"):
        cover_path = base / str(meta["cover"])
        if not cover_path.is_file() or cover_path.stat().st_size == 0:
            problems.append(f"封面缺失或为空：{cover_path}")

    zip_path = base.with_suffix(".zip")
    if not zip_path.is_file() or zip_path.stat().st_size == 0:
        problems.append(f"交付 ZIP 缺失或为空：{zip_path}")

    return problems