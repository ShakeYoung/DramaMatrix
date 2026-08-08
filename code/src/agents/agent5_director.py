"""Agent 5: submit storyboard shots to Agnes Video V2.0 and download results."""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path

import requests.exceptions as requests_exceptions

from src.cost_tracker import CostTracker
from src.agnes_video import (
    AgnesConfigurationError,
    AgnesConnectionError,
    AgnesContentPolicyViolation,
    AgnesGatewayUncertain,
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


def _mark_shot_for_redraw(
    ep_state: EpisodeState,
    shot_id: str,
    issues,
    settings,
    *,
    shot_directory: Path,
    shot_index: int,
) -> None:
    """R4：镜头级重绘——只删当前镜 asset，保留已通过的镜头，记录重绘次数。

    与 _reject_episode（退回整集）不同：质检失败仅重做当前镜，受 max_revisions 约束。
    """
    max_redraw = getattr(settings, "max_revisions", 2)
    # 统计该镜已重绘次数（通过 feedback_log 的 QC_FAIL 记录）。
    redraw_count = sum(
        1 for fb in ep_state.feedback_log
        if fb.reason_code == "QC_REDRAW" and shot_id in (fb.message or "")
    )
    if redraw_count >= max_redraw:
        # 重绘次数耗尽 → 退回整集重写分镜。
        _reject_episode(ep_state.script_data.ep_id if ep_state.script_data else "ep",
                        ep_state,
                        f"镜头 {shot_id} 连续性质检重绘 {redraw_count} 次仍不合格：{'；'.join(issues)}")
        return
    # 将被拒绝的视频及其首尾帧归档。若只删 state 中的 asset 而保留
    # *_head.png / *_tail.png，下一轮会误用旧帧做 QC 和条件生成。
    attempt = redraw_count + 1
    rejected_dir = (
        shot_directory
        / "rejected"
        / safe_component(shot_id)
        / f"attempt_{attempt}"
    )
    rejected_dir.mkdir(parents=True, exist_ok=True)
    asset = next((a for a in ep_state.video_assets if a.shot_id == shot_id), None)
    candidates: list[Path] = []
    if asset and asset.local_path:
        candidates.append(Path(asset.local_path))
    stem = f"{shot_index:03d}_{safe_component(shot_id)}"
    candidates.extend(
        [
            shot_directory / f"{stem}_head.png",
            shot_directory / f"{stem}_tail.png",
        ]
    )
    for path in candidates:
        if path.is_file():
            destination = rejected_dir / path.name
            try:
                shutil.move(str(path), str(destination))
            except OSError as exc:
                print(f"   ⚠️ 归档质检失败素材失败 {path}: {exc}")

    # 删除当前镜的 asset（保留前面的），下一轮 Agent5 重入会重新创建该镜。
    ep_state.video_assets = [a for a in ep_state.video_assets if a.shot_id != shot_id]
    ep_state.feedback_log.append(
        FeedbackLog(
            from_agent="Agent_5_Agnes_Director",
            to_agent="Agent_5_Agnes_Director",
            reason_code="QC_REDRAW",
            message=f"镜头 {shot_id} 质检不合格，将重绘（第 {redraw_count + 1}/{max_redraw} 次）：{'；'.join(issues)}",
        )
    )
    # 保持 storyboard_done，让路由回到 Agent5 重绘该镜（而非进 Agent6）。
    ep_state.status = "storyboard_done"
    print(f"⏸️ 镜头 {shot_id} 质检不合格，标记重绘（保留已通过镜头）。")


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
    storyboard_revision_count = sum(
        1
        for fb in ep_state.feedback_log
        if fb.reason_code == "AGNES_RENDER_FAILED" and fb.to_agent == "Agent_4_Storyboard"
    )
    if storyboard_revision_count >= settings.max_revisions:
        ep_state.status = "render_failed"
        print(f"❌ {ep_key} 已达到 {settings.max_revisions} 次分镜重写上限。")
        return

    all_shots = ep_state.storyboard_data
    shots = all_shots
    if settings.max_shots_per_episode > 0:
        shots = shots[:settings.max_shots_per_episode]
    if not shots:
        ep_state.status = "render_failed"
        print(f"❌ {ep_key} 没有可提交给 Agnes 的分镜。")
        return

    is_resume = ep_state.status in {"rendering", "render_pending"} or bool(ep_state.video_assets)
    ep_state.status = "rendering"
    ep_state.planned_shot_count = len(all_shots)
    # 阶段3：角色一致性块注入每镜 Agnes 提示
    from src.characters import render_character_block
    from src.continuity import (
        conditional_generation_enabled,
        load_scene_references,
        prepare_shot_reference,
        scene_seed,
        scene_segments,
    )
    from src.continuity_qc import get_checker
    from src.agnes_video import extract_first_frame, extract_last_frame, frame_to_data_uri
    from src.render_concurrency import CapacityThrottle, max_in_flight
    from src.tts import agnes_voice_enabled
    character_block = render_character_block(state.get("characters", []))
    conditional = conditional_generation_enabled()
    # R8：加载场景参考图映射（部署时预生成角色/场景参考图并注册）。
    scene_references: dict[str, str] = load_scene_references()
    # G1：跨场景共享的容量节流器——任一 queue_full 即降级，成功创建即恢复。
    throttle = CapacityThrottle()
    in_flight = max_in_flight()
    if not is_resume:
        ep_state.video_assets = []
    assets_by_shot = {asset.shot_id: asset for asset in ep_state.video_assets}
    shot_directory = shot_output_dir(state["project_id"], ep_key, ep_state.storyboard_version or 1)
    action = ("条件链式生成" if conditional else "恢复轮询/下载" if is_resume else "逐镜提交")
    print(f"-> {ep_key}: {action} {len(shots)} 个 Agnes 视频任务。")

    # R3：分离两条用途——previous_tail_path 供本地 QC（合法本地路径），
    # previous_tail_reference 供 Agnes（data URI / HTTPS URL）。两者同步更新。
    previous_tail_path: Path | None = None
    previous_tail_reference: str | None = None
    previous_scene_id: str | None = None
    for index, shot in enumerate(shots, start=1):
        # P0-5：场景切换时清除上一场景尾帧，避免跨场景错误继承。
        if previous_scene_id is not None and shot.scene_id != previous_scene_id:
            previous_tail_path = None
            previous_tail_reference = None
        previous_scene_id = shot.scene_id
        prompt = build_agnes_prompt(shot, character_block)
        asset = assets_by_shot.get(shot.shot_id)
        if asset and asset.local_path and Path(asset.local_path).is_file():
            asset.status = "completed"
            # 无论是否启用条件生成，都为下一镜准备本地尾帧做 QC；条件生成仅额外
            # 将它编码为 data URI 发送给 Agnes。
            if asset.local_path and index < len(shots):
                frame_path = shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}_tail.png"
                # P1-3：已存在且非空的尾帧不重复提取，避免恢复时 O(n²) 抽帧。
                if not frame_path.is_file() or frame_path.stat().st_size == 0:
                    extract_last_frame(Path(asset.local_path), frame_path)
                if frame_path.is_file():
                    previous_tail_path = frame_path
                    if conditional:
                        data_uri = frame_to_data_uri(frame_path)
                        previous_tail_reference = data_uri if data_uri else None
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
                # G1：跨场景节流——若已被 queue_full 降级，等待容量恢复再提交。
                if in_flight > 1 and not throttle.acquire_create():
                    ep_state.status = "waiting_for_agnes_capacity"
                    _checkpoint(state, ep_key, ep_state)
                    print(f"⏸️ {ep_key}/{shot.shot_id} Agnes 容量降级中，等待恢复后重试。")
                    return
                # P1-C：条件输入。同场景首镜用场景参考图；后续镜用上一镜尾帧。
                create_kwargs = dict(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    duration=shot.duration,
                    seed=scene_seed(20260801, shot.scene_id, index) if conditional else 20260801 + index,
                )
                if conditional:
                    ref = prepare_shot_reference(shot, scene_references, previous_tail_reference)
                    if ref.get("image_url"):
                        create_kwargs["image_url"] = ref["image_url"]
                        print(f"   [{ep_key}/{shot.shot_id}] 携带参考图（{'上一镜尾帧' if previous_tail_reference else '场景参考'}）。")
                # G4a：Agnes 原生语音——把对白作为旁白传给 Agnes（字段名可配）。
                # 是否被接受需服务器端验证；Agent6 会检测无声并回退独立 TTS。
                if agnes_voice_enabled() and (shot.dialogue or "").strip():
                    create_kwargs["narration"] = (shot.dialogue or "").strip()
                created = client.create_video(**create_kwargs)
                # G1：成功创建说明容量已恢复，清除降级标记并释放槽位。
                if in_flight > 1:
                    throttle.report_success()
                    throttle.release_create()
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
                # 一次明确成功的创建说明容量已经恢复，后续 queue_full 从首级退避开始。
                ep_state.queue_retry_count = 0
                ep_state.next_retry_at = 0.0
                ep_state.last_queue_error = None
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
            # R3：正确的质检顺序——提取当前镜首帧，与上一镜尾帧比较；通过后再提取
            # 当前镜尾帧供下一镜使用。质检用合法本地路径，不用 data URI。
            current_first_frame: Path | None = None
            first_path = shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}_head.png"
            if not first_path.is_file() or first_path.stat().st_size == 0:
                extract_first_frame(Path(local_path), first_path)
            if first_path.is_file():
                current_first_frame = first_path
            # P2-A：逐镜质检（上一镜尾帧 vs 当前镜首帧）。不合格只重绘当前镜。
            qc_result = get_checker().check(
                prev_video=None,
                prev_last_frame=previous_tail_path,
                curr_video=Path(local_path),
                curr_first_frame=current_first_frame,
                shot=shot,
            )
            if not qc_result.passed:
                # R4：镜头级重绘而非退回整集。删除当前镜 asset，标记 redraw_pending。
                _mark_shot_for_redraw(
                    ep_state,
                    shot.shot_id,
                    qc_result.issues,
                    settings,
                    shot_directory=shot_directory,
                    shot_index=index,
                )
                _checkpoint(state, ep_key, ep_state)
                return
            if qc_result.issues:
                print(f"   [{ep_key}/{shot.shot_id}] 质检告警：{'；'.join(qc_result.issues)}")
            if qc_result.metrics:
                metric_text = ", ".join(f"{k}={v:.3f}" for k, v in qc_result.metrics.items())
                print(f"   [{ep_key}/{shot.shot_id}] QC 指标：{metric_text}")
            # 质检通过：提取当前镜尾帧，更新两条轨道供下一镜。
            if index < len(shots):
                frame_path = shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}_tail.png"
                if not frame_path.is_file() or frame_path.stat().st_size == 0:
                    extract_last_frame(Path(local_path), frame_path)
                if frame_path.is_file():
                    previous_tail_path = frame_path
                    if conditional:
                        data_uri = frame_to_data_uri(frame_path)
                        previous_tail_reference = data_uri if data_uri else None
        except (AgnesSubmissionUncertain, AgnesGatewayUncertain) as exc:
            # 提交结果未知 / 网关不确定（任务可能已创建）→ 熔断，需人工核对，不自动重试。
            ep_state.status = "submission_uncertain"
            _checkpoint(state, ep_key, ep_state)
            print(f"❌ {ep_key}/{shot.shot_id} 提交状态未知/网关不确定，已熔断后续创建：{exc}")
            return
        except AgnesQueueFull as exc:
            # P0-1/P0-2/R10：队列满/限流是可恢复的——只暂停当前集，记录 next_retry_at，
            # 等待后重试当前镜，不标记 render_failed，也不让其他集立即重入。
            # G1：通知跨场景节流器降级，让并发中的其他场景也暂停新提交。
            if in_flight > 1:
                throttle.report_queue_full()
                throttle.release_create()
            ep_state.status = "waiting_for_agnes_capacity"
            ep_state.queue_retry_count = int(ep_state.queue_retry_count or 0) + 1
            # 退避：60s × 2^(retry-1)，上限 600s。
            backoff = min(60 * (2 ** (ep_state.queue_retry_count - 1)), 600)
            ep_state.next_retry_at = time.time() + backoff
            ep_state.last_queue_error = str(exc)[:200]
            _checkpoint(state, ep_key, ep_state)
            print(f"⏸️ {ep_key}/{shot.shot_id} Agnes 队列满，第 {ep_state.queue_retry_count} 次等待，"
                  f"约 {backoff}s 后（next_retry_at 已持久化）可重试当前镜：{exc}")
            return
        except (AgnesConnectionError, requests_exceptions.ChunkedEncodingError) as exc:
            # P0-3：瞬时连接失败不应把剩余剧集批量判 render_failed。
            # 仅把当前正在处理的集标记为 waiting_for_connectivity，其余保持原状。
            ep_state.status = "waiting_for_connectivity"
            ep_state.next_retry_at = time.time() + 60
            ep_state.last_queue_error = str(exc)[:200]
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

    completed_ids = {
        asset.shot_id
        for asset in ep_state.video_assets
        if asset.status == "completed" and asset.local_path and Path(asset.local_path).is_file()
    }
    ep_state.rendered_shot_count = sum(1 for shot in all_shots if shot.shot_id in completed_ids)
    if ep_state.rendered_shot_count < len(all_shots):
        ep_state.status = "render_partial"
        _checkpoint(state, ep_key, ep_state)
        print(
            f"⏸️ {ep_key} 仅完成 {ep_state.rendered_shot_count}/{len(all_shots)} 个分镜，"
            "已标记 render_partial；不会进入 Agent6。解除镜头上限后可 --resume 续跑。"
        )
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
    now = time.time()
    # 容量/连通性属于当前 Agnes 账户的全局背压。只要任一剧集仍在退避窗口，
    # 就不能跳过它而提交后续剧集，否则会继续冲击已经饱和的队列。
    active_backoffs = [
        (key, ep)
        for key, ep in state["episodes"].items()
        if ep.status in {"waiting_for_agnes_capacity", "waiting_for_connectivity"}
        and ep.next_retry_at
        and now < ep.next_retry_at
    ]
    if active_backoffs:
        key, ep = min(active_backoffs, key=lambda item: item[1].next_retry_at)
        remaining = ep.next_retry_at - now
        state["system_status"] = (
            "waiting_for_agnes_capacity"
            if ep.status == "waiting_for_agnes_capacity"
            else "waiting_for_connectivity"
        )
        print(f"⏳ {key} 仍在全局退避窗口（剩余 {remaining:.0f}s），本次不提交任何新任务。")
        return state

    targets = []
    for key, ep in state["episodes"].items():
        if ep.status not in {
            "storyboard_done",
            "rendering",
            "render_pending",
            "render_failed",
            "render_partial",
            "waiting_for_agnes_capacity",
            "waiting_for_connectivity",
        }:
            continue
        targets.append((key, ep))
    if not targets:
        print("没有等待 Agnes 渲染的剧集。")
        return state

    try:
        settings = AgnesVideoSettings.from_environment()
        client = AgnesVideoClient(settings)
    except AgnesConfigurationError as exc:
        state["system_status"] = "blocked_on_agnes_configuration"
        print(f"❌ Agnes 配置错误：{exc}")
        return state

    try:
        client.preflight()
    except AgnesConfigurationError as exc:
        state["system_status"] = "blocked_on_agnes_configuration"
        print(f"❌ Agnes 配置错误：{exc}")
        return state
    except AgnesConnectionError as exc:
        # P0-3：连通性瞬时失败不得把尚未提交的剧集批量判 render_failed。
        # 仅把正在处理的集转为 waiting_for_connectivity，其余保持原状以便下次恢复。
        for ep_key, ep_state in targets:
            if ep_state.status in {"rendering", "render_pending"}:
                ep_state.status = "render_pending"
            elif ep_state.status not in {"submission_uncertain"}:
                ep_state.status = "waiting_for_connectivity"
                ep_state.next_retry_at = time.time() + 60
                ep_state.last_queue_error = str(exc)[:200]
            state["episodes"][ep_key] = ep_state
        state["system_status"] = "blocked_on_agnes_connectivity"
        print(f"❌ Agnes 连通性熔断（已转为 waiting_for_connectivity，恢复后继续）：{exc}")
        return state

    tracker = CostTracker.from_environment(state["project_id"])
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
    elif any(ep.status == "render_partial" for _, ep in targets):
        state["system_status"] = "waiting_for_full_render"
    else:
        state["system_status"] = "video_assets_downloaded"
    return state
