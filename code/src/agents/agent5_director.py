"""Agent 5: submit storyboard shots to Agnes Video V2.0 and download results."""

from __future__ import annotations

import logging
from pathlib import Path

from src.cost_tracker import CostTracker
from src.agnes_video import (
    AgnesConfigurationError,
    AgnesContentPolicyViolation,
    AgnesConnectionError,
    AgnesSubmissionUncertain,
    AgnesTaskFailed,
    AgnesVideoClient,
    AgnesVideoError,
    AgnesVideoSettings,
    episode_output_dir,
    frames_for_duration,
    safe_component,
)
from src.state import DramaState, EpisodeState, FeedbackLog, GeneratedVideoAsset, ShotStoryboard
from src.db import db_save_project_state


NEGATIVE_PROMPT = (
    "deformed hands, extra fingers, distorted face, identity change, duplicate person, "
    "warped objects, melting objects, flickering, frame-to-frame inconsistency, jittery "
    "motion, unnatural body movement, blurry face, low resolution, text, subtitles, watermark, logo"
)


def build_agnes_prompt(shot: ShotStoryboard, character_block: str = "") -> str:
    """Keep all narrative and production details in the documented text prompt field."""
    dialogue = shot.dialogue.strip() or "No spoken dialogue."
    consistency = ("\n\n" + character_block) if character_block else ""
    return (
        f"{shot.visual_prompt.strip()}\n"
        f"Camera direction: {shot.camera.strip()}.\n"
        f"Character dialogue or performance cue: {dialogue}\n"
        f"Audio and atmosphere direction: {shot.audio.strip()}.\n"
        "Maintain character identity, costume, lighting continuity, physically plausible motion, "
        "and a stable cinematic composition."
        f"{consistency}"
    )


def _reject_episode(ep_key: str, ep_state: EpisodeState, message: str) -> None:
    ep_state.feedback_log.append(
        FeedbackLog(
            from_agent="Agent_5_Agnes_Director",
            to_agent="Agent_4_Storyboard",
            reason_code="AGNES_RENDER_FAILED",
            message=message,
        )
    )
    ep_state.status = "director_rejected"
    print(f"❌ {ep_key} 的 Agnes 渲染任务失败，已将明确错误反馈给 Agent 4。")


def _checkpoint(state: DramaState, ep_key: str, ep_state: EpisodeState) -> None:
    """Durably record a submitted task before any further network request."""
    state["episodes"][ep_key] = ep_state
    try:
        db_save_project_state(state)
    except Exception as exc:
        # Rendering may still continue, but the operator must know that resume safety was lost.
        print(f"⚠️ {ep_key} 的 Agnes 任务状态快照保存失败：{exc}")


def _render_episode(
    state: DramaState,
    ep_key: str,
    ep_state: EpisodeState,
    client: AgnesVideoClient,
    settings: AgnesVideoSettings,
    tracker: CostTracker,
) -> None:
    if len(ep_state.feedback_log) >= settings.max_revisions:
        ep_state.status = "render_failed"
        print(f"❌ {ep_key} 已达到 {settings.max_revisions} 次分镜重写上限。")
        return

    shots = ep_state.storyboard_data
    if settings.max_shots_per_episode > 0:
        shots = shots[:settings.max_shots_per_episode]
    if not shots:
        ep_state.status = "render_failed"
        print(f"❌ {ep_key} 没有可提交给 Agnes 的分镜。")
        return

    is_resume = ep_state.status in {"rendering", "render_pending"} or bool(ep_state.video_assets)
    ep_state.status = "rendering"
    # 阶段3：角色一致性块注入每镜 Agnes 提示
    from src.characters import render_character_block
    character_block = render_character_block(state.get("characters", []))
    if not is_resume:
        ep_state.video_assets = []
    assets_by_shot = {asset.shot_id: asset for asset in ep_state.video_assets}
    shot_directory = episode_output_dir(state["project_id"], ep_key) / "shots"
    action = "恢复轮询/下载" if is_resume else "逐镜提交"
    print(f"-> {ep_key}: {action} {len(shots)} 个 Agnes 视频任务。")

    for index, shot in enumerate(shots, start=1):
        prompt = build_agnes_prompt(shot, character_block)
        asset = assets_by_shot.get(shot.shot_id)
        if asset and asset.local_path and Path(asset.local_path).is_file():
            asset.status = "completed"
            continue
        try:
            if asset and (asset.video_id or asset.task_id):
                video_id = asset.video_id
                task_id = asset.task_id or asset.video_id
                print(f"   [{ep_key}/{shot.shot_id}] 恢复 Agnes 任务: {task_id}")
            else:
                if tracker.budget_exhausted():
                    ep_state.status = "render_pending"
                    _checkpoint(state, ep_key, ep_state)
                    print(f"⚠️ {ep_key} 已达 Agnes 创建次数预算上限（{tracker.max_creates}），熔断新任务。")
                    return
                created = client.create_video(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    duration=shot.duration,
                    seed=20260801 + index,
                )
                created_video_id = created.get("video_id")
                video_id = str(created_video_id or created.get("task_id") or created["id"])
                task_id = str(created.get("task_id") or created.get("id") or video_id)
                tracker.record_create(
                    project_id=state["project_id"],
                    ep_key=ep_key,
                    shot_id=shot.shot_id,
                    task_id=task_id,
                    frames=frames_for_duration(shot.duration, settings.frame_rate),
                    width=settings.width,
                    height=settings.height,
                )
                asset = GeneratedVideoAsset(
                    shot_id=shot.shot_id,
                    video_id=video_id,
                    task_id=task_id,
                    status="submitted",
                    prompt=prompt,
                )
                ep_state.video_assets.append(asset)
                assets_by_shot[shot.shot_id] = asset
                _checkpoint(state, ep_key, ep_state)
                print(f"   [{ep_key}/{shot.shot_id}] 已创建 Agnes 任务并保存: {task_id}")

            completed = client.wait_for_video(video_id, task_id)
            remote_url = (completed.get("metadata") or {}).get("url")
            if not remote_url:
                raise AgnesVideoError("Agnes 任务已完成但响应未包含 metadata.url。")
            asset.status = "downloading"
            asset.remote_url = str(remote_url)
            _checkpoint(state, ep_key, ep_state)
            local_path = client.download_video(
                str(remote_url),
                shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}.mp4",
            )
            asset.status = "completed"
            asset.local_path = str(local_path)
            _checkpoint(state, ep_key, ep_state)
        except AgnesSubmissionUncertain as exc:
            ep_state.status = "submission_uncertain"
            _checkpoint(state, ep_key, ep_state)
            print(f"❌ {ep_key}/{shot.shot_id} 提交状态未知，已熔断后续创建：{exc}")
            return
        except AgnesContentPolicyViolation as exc:
            _reject_episode(
                ep_key,
                ep_state,
                f"镜头 {shot.shot_id} 的 Agnes 内容策略拒绝：{exc}",
            )
            _checkpoint(state, ep_key, ep_state)
            return
        except AgnesTaskFailed as exc:
            _reject_episode(ep_key, ep_state, f"镜头 {shot.shot_id} 的 Agnes 任务失败：{exc}")
            _checkpoint(state, ep_key, ep_state)
            return
        except AgnesVideoError as exc:
            # If an ID was persisted, this is recoverable: later runs poll the
            # same remote task instead of submitting a charged duplicate task.
            ep_state.status = "render_pending" if asset and (asset.video_id or asset.task_id) else "render_failed"
            _checkpoint(state, ep_key, ep_state)
            if asset and (asset.video_id or asset.task_id):
                print(f"❌ {ep_key} 的 Agnes 任务处理失败：{exc}；已保留任务，重跑将继续恢复。")
            else:
                print(f"❌ {ep_key} 的 Agnes 创建前失败：{exc}；未创建可恢复的任务。")
            return

    ep_state.status = "video_generated"
    _checkpoint(state, ep_key, ep_state)
    print(f"✅ {ep_key} 的 {len(ep_state.video_assets)} 个分镜视频已下载完成。")


def process_agent5_director(state: DramaState) -> DramaState:
    """Render every storyboard-ready episode through the real Agnes asynchronous API."""
    print("--- [Agent 5: Agnes Video Director] ---")
    if any(ep.status == "submission_uncertain" for ep in state["episodes"].values()):
        state["system_status"] = "blocked_on_agnes_submission_uncertain"
        print("❌ 检测到结果未知的历史提交，已熔断所有新任务；请先在 Agnes 控制台核对。")
        return state
    targets = [
        (key, ep)
        for key, ep in state["episodes"].items()
        if ep.status in {"storyboard_done", "rendering", "render_pending", "render_failed"}
    ]
    if not targets:
        print("没有等待 Agnes 渲染的剧集。")
        return state

    try:
        settings = AgnesVideoSettings.from_environment()
        client = AgnesVideoClient(settings)
    except AgnesConfigurationError as exc:
        for ep_key, ep_state in targets:
            ep_state.status = "render_failed"
            state["episodes"][ep_key] = ep_state
        state["system_status"] = "blocked_on_agnes_configuration"
        print(f"❌ Agnes 配置错误：{exc}")
        return state

    try:
        client.preflight()
    except (AgnesConnectionError, AgnesConfigurationError) as exc:
        for ep_key, ep_state in targets:
            if ep_state.status in {"rendering", "render_pending"}:
                ep_state.status = "render_pending"
            elif ep_state.status != "submission_uncertain":
                ep_state.status = "render_failed"
            state["episodes"][ep_key] = ep_state
        state["system_status"] = "blocked_on_agnes_connectivity"
        print(f"❌ Agnes 连通性熔断：{exc}")
        return state

    tracker = CostTracker.from_environment()
    for ep_key, ep_state in targets:
        _render_episode(state, ep_key, ep_state, client, settings, tracker)
        state["episodes"][ep_key] = ep_state
        if ep_state.status == "director_rejected":
            break
        if ep_state.status in {"render_pending", "render_failed", "submission_uncertain"}:
            # A failed or ambiguous operation must stop later submissions.
            break

    if tracker.records:
        report = tracker.write_report()
        print(f"📊 Agnes 用量已记录 {len(tracker.records)} 条 -> {report}")

    if any(ep.status == "director_rejected" for _, ep in targets):
        state["system_status"] = "blocked_on_storyboard_revision"
    elif any(ep.status == "submission_uncertain" for _, ep in targets):
        state["system_status"] = "blocked_on_agnes_submission_uncertain"
    elif any(ep.status in {"render_failed", "render_pending"} for _, ep in targets):
        state["system_status"] = "blocked_on_agnes_render"
    else:
        state["system_status"] = "video_assets_downloaded"
    return state
