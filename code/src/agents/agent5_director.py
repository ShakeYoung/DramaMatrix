"""Agent 5: submit storyboard shots to Agnes Video V2.0 and download results."""

from __future__ import annotations

from src.agnes_video import (
    AgnesConfigurationError,
    AgnesTaskFailed,
    AgnesVideoClient,
    AgnesVideoError,
    AgnesVideoSettings,
    episode_output_dir,
    safe_component,
)
from src.state import DramaState, EpisodeState, FeedbackLog, GeneratedVideoAsset, ShotStoryboard


NEGATIVE_PROMPT = (
    "deformed hands, extra fingers, distorted face, identity change, duplicate person, "
    "warped objects, melting objects, flickering, frame-to-frame inconsistency, jittery "
    "motion, unnatural body movement, blurry face, low resolution, text, subtitles, watermark, logo"
)


def build_agnes_prompt(shot: ShotStoryboard) -> str:
    """Keep all narrative and production details in the documented text prompt field."""
    dialogue = shot.dialogue.strip() or "No spoken dialogue."
    return (
        f"{shot.visual_prompt.strip()}\n"
        f"Camera direction: {shot.camera.strip()}.\n"
        f"Character dialogue or performance cue: {dialogue}\n"
        f"Audio and atmosphere direction: {shot.audio.strip()}.\n"
        "Maintain character identity, costume, lighting continuity, physically plausible motion, "
        "and a stable cinematic composition."
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


def _render_episode(
    project_id: str,
    ep_key: str,
    ep_state: EpisodeState,
    client: AgnesVideoClient,
    settings: AgnesVideoSettings,
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

    ep_state.status = "rendering"
    ep_state.video_assets = []
    shot_directory = episode_output_dir(project_id, ep_key) / "shots"
    print(f"-> {ep_key}: 逐镜提交 {len(shots)} 个 Agnes 视频任务。")

    for index, shot in enumerate(shots, start=1):
        prompt = build_agnes_prompt(shot)
        try:
            created = client.create_video(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                duration=shot.duration,
                seed=20260801 + index,
            )
            created_video_id = created.get("video_id")
            video_id = str(created_video_id or created.get("task_id") or created["id"])
            task_id = str(created.get("task_id") or created.get("id") or video_id)
            print(f"   [{ep_key}/{shot.shot_id}] 已创建 Agnes 任务: {task_id}")
            completed = client.wait_for_video(str(created_video_id) if created_video_id else None, task_id)
            remote_url = (completed.get("metadata") or {}).get("url")
            if not remote_url:
                raise AgnesVideoError("Agnes 任务已完成但响应未包含 metadata.url。")
            local_path = client.download_video(
                str(remote_url),
                shot_directory / f"{index:03d}_{safe_component(shot.shot_id)}.mp4",
            )
            ep_state.video_assets.append(
                GeneratedVideoAsset(
                    shot_id=shot.shot_id,
                    video_id=video_id,
                    task_id=task_id,
                    status="completed",
                    prompt=prompt,
                    remote_url=str(remote_url),
                    local_path=str(local_path),
                )
            )
        except AgnesTaskFailed as exc:
            _reject_episode(ep_key, ep_state, f"镜头 {shot.shot_id} 的 Agnes 任务失败：{exc}")
            return
        except AgnesVideoError as exc:
            ep_state.status = "render_failed"
            print(f"❌ {ep_key} 的 Agnes 调用或下载失败：{exc}")
            return

    ep_state.status = "video_generated"
    print(f"✅ {ep_key} 的 {len(ep_state.video_assets)} 个分镜视频已下载完成。")


def process_agent5_director(state: DramaState) -> DramaState:
    """Render every storyboard-ready episode through the real Agnes asynchronous API."""
    print("--- [Agent 5: Agnes Video Director] ---")
    targets = [(key, ep) for key, ep in state["episodes"].items() if ep.status == "storyboard_done"]
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

    for ep_key, ep_state in targets:
        _render_episode(state["project_id"], ep_key, ep_state, client, settings)
        state["episodes"][ep_key] = ep_state

    if any(ep.status == "director_rejected" for _, ep in targets):
        state["system_status"] = "blocked_on_storyboard_revision"
    elif any(ep.status == "render_failed" for _, ep in targets):
        state["system_status"] = "blocked_on_agnes_render"
    else:
        state["system_status"] = "video_assets_downloaded"
    return state
