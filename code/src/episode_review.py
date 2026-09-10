"""R2 整集验收（episode-level review）。

逐镜通过不等于整集可用：对白缺失/被截断、字幕不对齐、剪辑衔接问题只有
看成片才能发现。Agent6 合成完成后（DRAMAMATRIX_EPISODE_REVIEW=1，默认开）
生成整集验收清单并暂停（awaiting_episode_review）；人工核片后通过 CLI
标记 approve（进入投流）或 rework（重走 Agent6 重新合成）。

文件布局（与镜头级审阅同目录）：
  outputs/{project}/{ep}/review/episode_review.json    # 证据清单（自动生成）
  outputs/{project}/{ep}/review/episode_decision.json  # 人工决定（CLI 写入）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from src.agnes_video import episode_output_dir


def episode_review_enabled() -> bool:
    return (os.getenv("DRAMAMATRIX_EPISODE_REVIEW", "1").strip().lower()
            not in {"0", "false", "no", "off"})


def _review_dir(project_id: str, ep_key: str) -> Path:
    directory = episode_output_dir(project_id, ep_key) / "review"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _decision_path(project_id: str, ep_key: str) -> Path:
    return _review_dir(project_id, ep_key) / "episode_decision.json"


def build_episode_review_manifest(ep_state) -> dict[str, Any]:
    """汇总整集验收所需的证据：成片/交付资产 + 对白时间表 + 音字幕轨。"""
    deliverables = []
    for asset in ep_state.deliverables:
        entry: dict[str, Any] = {
            "kind": asset.kind,
            "path": asset.path,
            "sha256": asset.sha256,
            "actual_duration": asset.actual_duration,
        }
        path = Path(asset.path) if asset.path else None
        entry["file_exists"] = bool(path and path.is_file() and path.stat().st_size > 0)
        deliverables.append(entry)

    dialogue = ep_state.dialogue_report or {}
    overflow_lines = [
        {"index": l.get("index"), "text": (l.get("text") or "")[:60],
         "reason": "overflow" if l.get("overflow") else "unmeasured"}
        for l in (dialogue.get("lines") or [])
        if l.get("overflow") or l.get("unmeasured")
    ]

    return {
        "ep_key": ep_state.script_data.ep_id if ep_state.script_data else "ep",
        "generated_at": time.time(),
        "final_video": ep_state.final_video_path,
        "deliverables": deliverables,
        "dialogue": {
            "native_dialogue": bool(dialogue.get("native_dialogue")),
            "tts_applied": bool(dialogue.get("tts_applied")),
            "lines_total": dialogue.get("lines_total", 0),
            "lines_synthesized": dialogue.get("lines_synthesized", 0),
            "overflow_or_unmeasured": overflow_lines,
        },
        "audio_track": ep_state.audio_track,
        "subtitle_track": ep_state.subtitle_track,
        "decision": None,
    }


def write_episode_review_manifest(project_id: str, ep_key: str, ep_state) -> Path:
    manifest = build_episode_review_manifest(ep_state)
    path = _review_dir(project_id, ep_key) / "episode_review.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_episode_decision(project_id: str, ep_key: str) -> Optional[dict]:
    path = _decision_path(project_id, ep_key)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if payload.get("decision") not in {"approve", "rework"}:
        return None
    return payload


def episode_approved(project_id: str, ep_key: str) -> bool:
    decision = read_episode_decision(project_id, ep_key)
    return bool(decision and decision["decision"] == "approve")


def episode_rework(project_id: str, ep_key: str) -> bool:
    decision = read_episode_decision(project_id, ep_key)
    return bool(decision and decision["decision"] == "rework")


def write_episode_decision(project_id: str, ep_key: str, decision: str, note: str = "") -> Path:
    if decision not in {"approve", "rework"}:
        raise ValueError(f"decision 必须是 approve 或 rework，收到：{decision!r}")
    payload = {
        "decision": decision,
        "note": note,
        "decided_at": time.time(),
    }
    path = _decision_path(project_id, ep_key)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="整集验收决定：approve 进入投流；rework 重走 Agent6 重新合成。"
    )
    parser.add_argument("project_id")
    parser.add_argument("ep_key")
    parser.add_argument("decision", choices=["approve", "rework"])
    parser.add_argument("--note", default="", help="验收备注（rework 时建议写明问题）")
    args = parser.parse_args(argv)

    manifest = _review_dir(args.project_id, args.ep_key) / "episode_review.json"
    path = write_episode_decision(args.project_id, args.ep_key, args.decision, args.note)
    print(f"✅ 已记录整集验收决定 {args.decision} -> {path}")
    if manifest.is_file():
        print(f"   验收清单：{manifest}")
    else:
        print(f"   （未找到验收清单 {manifest}，仍已记录决定）")
    if args.decision == "rework":
        print("   下次运行将重走 Agent6 重新合成；修复音频/字幕配置后 --resume 即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
