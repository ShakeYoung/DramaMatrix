"""CLI to review and approve per-shot decisions (E1).

Reads a review manifest, lets the operator mark each shot approve/redraw/delete,
and writes decisions.json. A later pipeline re-run honors those decisions.

Usage:
    python -m src.review_approver <project_id> <ep_key>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.agnes_video import episode_output_dir


def _review_dir(project_id: str, ep_key: str) -> Path:
    return episode_output_dir(project_id, ep_key) / "review"


def interactive_approve(project_id: str, ep_key: str) -> None:
    review_dir = _review_dir(project_id, ep_key)
    manifest_path = review_dir / "review.json"
    decisions_path = review_dir / "decisions.json"
    if not manifest_path.exists():
        print(f"❌ 未找到审阅清单：{manifest_path}\n（先运行流水线生成视频）")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else {}
    print(f"== 审阅清单：{project_id}/{ep_key} ==")
    for shot in manifest["shots"]:
        sid = shot["shot_id"]
        cur = decisions.get(sid, "")
        print(f"\n镜头 {sid}（时长 {shot.get('actual_duration')}s, status={shot.get('status')}）")
        th = shot.get("thumbnail")
        if th:
            print(f"  缩略图: {th}")
        for q in shot.get("qc_issues", []):
            print(f"  QC: {q}")
        if cur:
            print(f"  当前决定: {cur}")
        choice = input("  [a]pprove  [r]edraw  [d]elete  [s]kip : ").strip().lower()
        if choice in {"a", "approve"}:
            decisions[sid] = "approve"
        elif choice in {"r", "redraw"}:
            decisions[sid] = "redraw"
        elif choice in {"d", "delete"}:
            decisions[sid] = "delete"
        elif cur:
            print(f"  保留当前决定: {cur}")
    decisions_path.write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 已写回 {decisions_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="人工审阅并标记镜头决定")
    parser.add_argument("project_id")
    parser.add_argument("ep_key")
    args = parser.parse_args(argv)
    interactive_approve(args.project_id, args.ep_key)


if __name__ == "__main__":
    main()