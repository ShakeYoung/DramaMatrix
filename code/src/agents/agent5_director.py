"""Agent 5: submit storyboard shots to Agnes Video V2.0 and download results."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests.exceptions as requests_exceptions

from src.cost_tracker import CostTracker
from src.agnes_video import (
    AgnesConfigurationError,
    AgnesConnectionError,
    AgnesContentPolicyViolation,
    AgnesQueueFull,
    AgnesSubmissionUncertain,
    AgnesTaskFailed,
    AgnesVideoClient,
    AgnesVideoError,
    AgnesVideoSettings,
    episode_output_dir,
    frames_for_duration,
    safe_component,
    shot_output_dir,
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
    from src.continuity import (
        conditional_generation_enabled,
        prepare_shot_reference,
        scene_seed,
    )
    from src.continuity_qc import get_checker
    from src.agnes_video import extract_last_frame, frame_to_data_uri
    character_block = render_character_block(state.get("characters", []))
    conditional = conditional_generation_enabled()
    scene_references: dict[str, str] = {}  # scene_id -> reference image_url (P1-C)
    if not is_resume:
        ep_state.video_assets = []
    assets_by_shot = {asset.shot_id: asset for asset in ep_state.video_assets}
    shot_directory = shot_output_dir(state["project_id"], ep_key, ep_state.storyboard_version or 1)
    action = ("条件链式生成" if conditional else "恢复轮询/下载" if is_resume else "逐镜提交")
    print(f"-> {ep_key}: {action} {len(shots)} 个 Agnes 视频任务。")

    previous_last_frame: str | None = None  # data URI / url of previous shot's tail frame
    previous_scene_id: str | None = None
    for index, shot in enumerate(shots, start=1):
        # P0-5：场景切换时清除上一场景尾帧，避免跨场景错误继承。
        if previous_scene_id is not None and shot.scene_id != previous_scene_id:
            previous_last_frame = None
        previous_scene_id = shot.scene_id
        prompt = build_agnes_prompt(shot, character_block)
        asset = assets_by_shot.get(shot.shot_id)
        if asset and asset.local_path and Path(asset.local_path).is_file():
            asset.status = "completed"
            # When chaining, prime the next shot's first-frame from this completed shot.
            if conditional and asset.local_path:
                frame_path = shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}_tail.png"
                # P1-3：已存在且非空的尾帧不重复提取，避免恢复时 O(n²) 抽帧。
                if not frame_path.is_file() or frame_path.stat().st_size == 0:
                    extract_last_frame(Path(asset.local_path), frame_path)
                data_uri = frame_to_data_uri(frame_path)
                previous_last_frame = data_uri if data_uri else previous_last_frame
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
                # P1-C：条件输入。同场景首镜用场景参考图；后续镜用上一镜尾帧。
                create_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    duration=shot.duration,
                    seed=scene_seed(20260801, shot.scene_id, index) if conditional else 20260801 + index,
                )
                if conditional:
                    ref = prepare_shot_reference(shot, scene_references, previous_last_frame)
                    if ref.get("image_url"):
                        create_kwargs["image_url"] = ref["image_url"]
                created = client.create_video(**create_kwargs)
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
            # P1-C：链式生成——下载完成后提取尾帧，供下一镜作为首帧。
            if conditional:
                frame_path = shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}_tail.png"
                if not frame_path.is_file() or frame_path.stat().st_size == 0:
                    extract_last_frame(Path(local_path), frame_path)
                data_uri = frame_to_data_uri(frame_path)
                previous_last_frame = data_uri if data_uri else previous_last_frame
            # P2-A：逐镜质检。不合格只重绘当前镜（受 max_revisions 约束），不等全部生成。
            qc_result = get_checker().check(
                prev_video=None,
                prev_last_frame=Path(previous_last_frame) if previous_last_frame else None,
                curr_video=Path(local_path),
                shot=shot,
            )
            if not qc_result.passed:
                _reject_episode(
                    ep_key,
                    ep_state,
                    f"镜头 {shot.shot_id} 连续性质检未通过：{'；'.join(qc_result.issues)}",
                )
                _checkpoint(state, ep_key, ep_state)
                return
            if qc_result.issues:
                print(f"   [{ep_key}/{shot.shot_id}] 质检告警：{'；'.join(qc_result.issues)}")
        except AgnesSubmissionUncertain as exc:
            ep_state.status = "submission_uncertain"
            _checkpoint(state, ep_key, ep_state)
            print(f"❌ {ep_key}/{shot.shot_id} 提交状态未知，已熔断后续创建：{exc}")
            return
        except AgnesQueueFull as exc:
            # P0-1/P0-2：队列满/限流是可恢复的——只暂停当前集，等待后重试当前镜，
            # 不标记 render_failed，也不让其他集立即重入。
            ep_state.status = "waiting_for_agnes_capacity"
            _checkpoint(state, ep_key, ep_state)
            print(f"⏸️ {ep_key}/{shot.shot_id} Agnes 队列满，暂停等待容量后重试当前镜：{exc}")
            return
        except (AgnesConnectionError, requests_exceptions.ChunkedEncodingError) as exc:
            # P0-3：瞬时连接失败不应把剩余剧集批量判 render_failed。
            # 仅把当前正在处理的集标记为 waiting_for_connectivity，其余保持原状。
            ep_state.status = "waiting_for_connectivity"
            _checkpoint(state, ep_key, ep_state)
            print(f"⏸️ {ep_key} Agnes 连接瞬时失败，标记 waiting_for_connectivity，恢复后继续：{exc}")
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
    # P0-B 门禁：角色圣经为空时禁止正式渲染，避免无一致性约束的盲渲染。
    allow_no_characters = os.getenv("DRAMAMATRIX_ALLOW_NO_CHARACTERS", "0").strip().lower() in {"1", "true", "yes"}
    if not state.get("characters") and not allow_no_characters:
        state["system_status"] = "blocked_on_missing_character_bible"
        print("❌ 角色圣经为空，已禁止渲染。请确认 Agent 3 已生成角色圣经，")
        print("   或设置 DRAMAMATRIX_ALLOW_NO_CHARACTERS=1 强制放行（不推荐，将丧失角色一致性）。")
        return state
    targets = [
        (key, ep)
        for key, ep in state["episodes"].items()
        if ep.status in {
            "storyboard_done",
            "rendering",
            "render_pending",
            "render_failed",
            "waiting_for_agnes_capacity",
            "waiting_for_connectivity",
        }
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
        # P0-3：连通性瞬时失败不得把尚未提交的剧集批量判 render_failed。
        # 仅把正在处理的集转为 waiting_for_connectivity，其余保持原状以便下次恢复。
        for ep_key, ep_state in targets:
            if ep_state.status in {"rendering", "render_pending"}:
                ep_state.status = "render_pending"
            elif ep_state.status not in {"submission_uncertain"}:
                ep_state.status = "waiting_for_connectivity"
            state["episodes"][ep_key] = ep_state
        state["system_status"] = "blocked_on_agnes_connectivity"
        print(f"❌ Agnes 连通性熔断（已转为 waiting_for_connectivity，恢复后继续）：{exc}")
        return state

    tracker = CostTracker.from_environment()
    for ep_key, ep_state in targets:
        _render_episode(state, ep_key, ep_state, client, settings, tracker)
        state["episodes"][ep_key] = ep_state
        if ep_state.status == "director_rejected":
            break
        if ep_state.status in {
            "render_pending",
            "render_failed",
            "submission_uncertain",
            "waiting_for_agnes_capacity",
            "waiting_for_connectivity",
        }:
            # A failed/ambiguous/waiting operation must stop later submissions.
            break

    if tracker.records:
        report = tracker.write_report()
        print(f"📊 Agnes 用量已记录 {len(tracker.records)} 条 -> {report}")

    if any(ep.status == "director_rejected" for _, ep in targets):
        state["system_status"] = "blocked_on_storyboard_revision"
    elif any(ep.status == "submission_uncertain" for _, ep in targets):
        state["system_status"] = "blocked_on_agnes_submission_uncertain"
    elif any(ep.status == "waiting_for_agnes_capacity" for _, ep in targets):
        state["system_status"] = "waiting_for_agnes_capacity"
    elif any(ep.status == "waiting_for_connectivity" for _, ep in targets):
        state["system_status"] = "waiting_for_connectivity"
    elif any(ep.status in {"render_failed", "render_pending"} for _, ep in targets):
        state["system_status"] = "blocked_on_agnes_render"
    else:
        state["system_status"] = "video_assets_downloaded"
    return state
