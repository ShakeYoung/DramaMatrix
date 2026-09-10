"""Storyboard correction tool (W1).

storyboard_blocked 此前只能直接改数据库。本工具给出安全的修正路径：

    python -m src.storyboard_editor <project> <ep> export [--out file]
    python -m src.storyboard_editor <project> <ep> import --file file [--to storyboard_done]
    python -m src.storyboard_editor <project> <ep> reset [--to script_done]

- export：把该集分镜（含全部连续性字段）导出为 JSON，人工编辑。
- import：pydantic 校验后整表替换 storyboard_data；分镜版本号递增并清理旧
  版本镜头目录（复用 Agent4 recovery 的隔离机制），再落库保存。
- reset：不改编镜，仅把 blocked/失败状态拨回 script_done（Agent4 重写）
  或 storyboard_done（Agent5 直接重渲染）。

设计约束：所有状态变更都经 db_save_project_state（进入 state_history，
可回溯审计）；镜头目录清理复用 purge_shot_versions_except，避免新旧混用。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src.agnes_video import purge_shot_versions_except
from src.db import db_get_project_state_snapshot, db_save_project_state
from src.state import DramaState, EpisodeState, ShotStoryboard

ALLOWED_TARGET_STATUS = {
    "script_done": "退回 Agent4 重写分镜",
    "storyboard_done": "直接进 Agent5 渲染",
}


def load_state(project_id: str) -> DramaState | None:
    """从快照恢复状态，并把 episodes 反序列化为 EpisodeState 模型。"""
    snapshot = db_get_project_state_snapshot(project_id)
    if not snapshot:
        return None
    state: DramaState = snapshot["state"]
    episodes: dict[str, EpisodeState] = {}
    for key, raw in (state.get("episodes") or {}).items():
        episodes[key] = raw if isinstance(raw, EpisodeState) else EpisodeState.model_validate(raw)
    state["episodes"] = episodes
    return state


def export_storyboard(project_id: str, ep_key: str, out_path: Path | None = None) -> Path:
    state = load_state(project_id)
    if not state:
        raise SystemExit(f"未找到项目快照：{project_id}")
    ep_state = state["episodes"].get(ep_key)
    if not ep_state:
        raise SystemExit(f"项目 {project_id} 中不存在剧集 {ep_key}")
    payload = {
        "project_id": project_id,
        "ep_key": ep_key,
        "current_status": ep_state.status,
        "storyboard_version": ep_state.storyboard_version or 1,
        "shots": [shot.model_dump(mode="json") for shot in ep_state.storyboard_data],
    }
    out = out_path or Path(f"{project_id}_{ep_key}_storyboard.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已导出 {len(payload['shots'])} 个分镜 -> {out}")
    print("   编辑后执行：python -m src.storyboard_editor "
          f"{project_id} {ep_key} import --file {out} --to storyboard_done")
    return out


def import_storyboard(
    project_id: str,
    ep_key: str,
    file_path: Path,
    target_status: str = "storyboard_done",
) -> None:
    if target_status not in ALLOWED_TARGET_STATUS:
        raise SystemExit(f"--to 仅支持：{', '.join(ALLOWED_TARGET_STATUS)}")
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    raw_shots = payload.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise SystemExit("导入文件缺少非空 shots 列表。")
    shots = [ShotStoryboard.model_validate(item) for item in raw_shots]
    shot_ids = [shot.shot_id for shot in shots]
    if len(set(shot_ids)) != len(shot_ids):
        raise SystemExit("导入分镜存在重复 shot_id，请修正后再导入。")

    state = load_state(project_id)
    if not state:
        raise SystemExit(f"未找到项目快照：{project_id}")
    ep_state = state["episodes"].get(ep_key)
    if not ep_state:
        raise SystemExit(f"项目 {project_id} 中不存在剧集 {ep_key}")

    ep_state.storyboard_data = shots
    ep_state.status = target_status
    # 版本隔离 + 清理旧镜头目录（与 Agent4 recovery 同机制）。
    ep_state.storyboard_version = int(ep_state.storyboard_version or 1) + 1
    purge_shot_versions_except(project_id, ep_key, ep_state.storyboard_version)
    # 旧渲染资产与反馈不再对应新分镜，一并清空。
    ep_state.video_assets = []
    ep_state.rendered_shot_count = 0
    ep_state.planned_shot_count = len(shots)
    db_save_project_state(state)
    print(f"✅ 已导入 {len(shots)} 个分镜（shots/v{ep_state.storyboard_version}/），"
          f"状态 -> {target_status}（{ALLOWED_TARGET_STATUS[target_status]}）。")


def reset_status(project_id: str, ep_key: str, target_status: str = "script_done") -> None:
    if target_status not in ALLOWED_TARGET_STATUS:
        raise SystemExit(f"--to 仅支持：{', '.join(ALLOWED_TARGET_STATUS)}")
    state = load_state(project_id)
    if not state:
        raise SystemExit(f"未找到项目快照：{project_id}")
    ep_state = state["episodes"].get(ep_key)
    if not ep_state:
        raise SystemExit(f"项目 {project_id} 中不存在剧集 {ep_key}")
    previous = ep_state.status
    ep_state.status = target_status
    if target_status == "script_done":
        # 退回重写时同样隔离版本，避免旧镜头文件混入。
        ep_state.storyboard_version = int(ep_state.storyboard_version or 1) + 1
        purge_shot_versions_except(project_id, ep_key, ep_state.storyboard_version)
        ep_state.video_assets = []
        ep_state.rendered_shot_count = 0
    db_save_project_state(state)
    print(f"✅ {ep_key} 状态 {previous} -> {target_status}（{ALLOWED_TARGET_STATUS[target_status]}）。")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 3:
        print("用法：python -m src.storyboard_editor <project> <ep> "
              "{export [--out f] | import --file f [--to s] | reset [--to s]}")
        return 2
    command, project_id, ep_key = argv[0], argv[1], argv[2]
    out_path: Path | None = None
    file_path: Path | None = None
    target = "storyboard_done" if command == "import" else "script_done"
    index = 3
    while index < len(argv):
        if argv[index] == "--out" and index + 1 < len(argv):
            out_path = Path(argv[index + 1])
            index += 2
        elif argv[index] == "--file" and index + 1 < len(argv):
            file_path = Path(argv[index + 1])
            index += 2
        elif argv[index] == "--to" and index + 1 < len(argv):
            target = argv[index + 1]
            index += 2
        else:
            index += 1
    if command == "export":
        export_storyboard(project_id, ep_key, out_path)
        return 0
    if command == "import":
        if not file_path:
            print("import 需要 --file <path>")
            return 2
        import_storyboard(project_id, ep_key, file_path, target)
        return 0
    if command == "reset":
        reset_status(project_id, ep_key, target)
        return 0
    print(f"未知命令：{command}（export / import / reset）")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
