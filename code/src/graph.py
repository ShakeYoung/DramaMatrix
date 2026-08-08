from src.state import DramaState
from langgraph.graph import StateGraph, START, END
import os
from src.agents.agent1_scout import process_agent1_scout
from src.agents.agent2_hook_analyzer import process_agent2_hook_analyzer
from src.agents.agent3_head_writer import process_agent3_head_writer
from src.agents.agent4_storyboard import process_agent4_storyboard
from src.agents.agent5_director import process_agent5_director
from src.agents.agent6_editor import process_agent6_editor
from src.agents.agent7_growth import process_agent7_growth
from src.agents.agent8_analytics import process_agent8_analytics

def route_after_agent2(state: DramaState) -> str:
    # 评审未通过时，若换书尝试次数未达上限则退回 Agent 1 换一本；否则结束。
    report = state.get("source_material", {}).get("report")
    if not report or not report.is_approved:
        attempts = state.get("scout_attempts", 0)
        max_attempts = int(os.getenv("DRAMAMATRIX_MAX_SCOUT_ATTEMPTS", "3"))
        if attempts < max_attempts:
            print(f">> 立项被否，换书重试（第 {attempts}/{max_attempts} 次后）→ Agent 1")
            return "agent1_scout"
        return END
    return "agent3_head_writer"


def _all_episodes_finished(episodes) -> bool:
    """Whether every episode has left the production flow (completed or terminal-failed)."""
    if not episodes:
        return False
    terminal = {"edit_completed", "growth_ready", "growth_failed", "editing_failed"}
    return all(ep.status in terminal for ep in episodes.values())


def _render_failed_retryable(ep) -> bool:
    """A render_failed episode is only retryable if it still has shots to render
    and has not exhausted its storyboard re-write budget. Otherwise it is terminal
    and must not be re-entered on every process restart (阶段4/T10)."""
    max_revisions = int(os.getenv("AGNES_MAX_REVISIONS", "2"))
    if not ep.storyboard_data:
        return False
    storyboard_revision_count = sum(
        1
        for feedback in ep.feedback_log
        if feedback.reason_code == "AGNES_RENDER_FAILED"
        and feedback.to_agent == "Agent_4_Storyboard"
    )
    if storyboard_revision_count >= max_revisions:
        return False
    return True


def _has_retryable_render_failed(episodes) -> bool:
    return any(ep.status == "render_failed" and _render_failed_retryable(ep) for ep in episodes)


def route_after_cycles(state: DramaState) -> str:
    """Decide whether to loop back to Agent 1 for another market-driven cycle.

    Agent 8 increments task_cycle when it completes a cycle; here we compare the
    (now completed) cycle count against the configured upper bound. So the check
    is `<=`: when task_cycle has just reached max_cycles, we still run that final
    cycle; only when it exceeds the bound do we stop.
    """
    cycle = state.get("task_cycle", 1)
    max_cycles = int(os.getenv("DRAMAMATRIX_MAX_CYCLES", "1"))
    if _all_episodes_finished(state.get("episodes", {})) and cycle <= max_cycles:
        next_cycle = cycle + 1
        print(f">> 第 {cycle} 周期成片完成，市场回环 → Agent 1（下一周期 {next_cycle}）")
        return "agent1_scout"
    return END


def _has_review_redraw(project_id, ep) -> bool:
    """Whether an awaiting_review episode has any .redraw decision (E1)."""
    try:
        from src.review import pending_shot_ids
        return bool(pending_shot_ids(project_id, _ep_key(ep), ep, decision="redraw"))
    except Exception:
        return False


def _ep_key(ep) -> str:
    return ep.script_data.ep_id if ep.script_data and ep.script_data.ep_id else "ep"


def _review_handled(project_id, ep) -> bool:
    """True if the episode's review manifest exists and has decisions for all shots."""
    try:
        from src.review import all_decided
        return all_decided(project_id, _ep_key(ep), ep)
    except Exception:
        return False


def route_from_start(state: DramaState) -> str:
    """Resume a persisted project at its first unfinished stage."""
    episodes = list(state.get("episodes", {}).values())
    project_id = state.get("project_id")
    if episodes:
        # R2：分镜门禁拦截为最高优先级——即使存在其他 storyboard_done 集，
        # 也不能进入昂贵的视频生成，必须先解决分镜阻塞。
        if any(ep.status == "storyboard_blocked" for ep in episodes):
            print(">> Router: 检测到 storyboard_blocked，恢复中止（需人工修正分镜后重置状态）")
            return END
        # E1：人工质检——awaiting_review 若有 .redraw 决定则回 Agent5 重绘；
        # 未审阅完则暂停；全部 approve 则放行到 Agent6。
        review_eps = [ep for ep in episodes if ep.status == "awaiting_review"]
        if review_eps and any(_has_review_redraw(project_id, ep) for ep in review_eps):
            print(">> Router: 审阅清单存在 .redraw 决定，回 Agent5 重绘。")
            return "agent5_director"
        if review_eps and not all(_review_handled(project_id, ep) for ep in review_eps):
            print(">> Router: 检测到 await review，暂停等待人工标记。")
            return END
        if any(ep.status == "submission_uncertain" for ep in episodes):
            return END
        if any(ep.status in {"script_done", "director_rejected"} for ep in episodes):
            return "agent4_storyboard"
        # P0-2/P0-3：恢复运行时，等待中的集重新进入 Agent5 继续轮询/重试当前镜。
        if any(
            ep.status in {
                "storyboard_done",
                "rendering",
                "render_pending",
                "render_partial",
                "waiting_for_agnes_capacity",
                "waiting_for_connectivity",
            }
            for ep in episodes
        ) or _has_retryable_render_failed(episodes):
            return "agent5_director"
        if any(ep.status == "video_generated" for ep in episodes):
            return "agent6_editor"
        if any(ep.status == "edit_completed" for ep in episodes):
            return "agent7_growth"
        if any(ep.status == "growth_ready" for ep in episodes):
            return "agent8_analytics"
        return END

    source_material = state.get("source_material", {})
    report = source_material.get("report")
    if report and report.is_approved:
        return "agent3_head_writer"
    if source_material.get("raw_text"):
        return "agent2_hook_analyzer"
    return "agent1_scout"

def route_next_step_for_episode(state: DramaState) -> str:
    """Route according to all episode states, rather than a hard-coded ep_01."""
    episodes = list(state.get("episodes", {}).values())
    if not episodes:
        return END
    if any(ep.status == "director_rejected" for ep in episodes):
        print(">> Router: 检测到 Agnes 渲染反馈，重新路由至 Agent 4")
        return "agent4_storyboard"
    # P0-4：分镜门禁拦截（数量/LLM 失败）为终态，禁止进入昂贵的视频生成。
    if any(ep.status == "storyboard_blocked" for ep in episodes):
        return END
    # A submitted task is intentionally paused after a recoverable network error.
    # The next process start resumes polling it before any new render is submitted.
    if any(ep.status == "render_pending" for ep in episodes):
        return END
    # 受控测试只生成了分镜子集；下次解除/提高镜头上限后可从 Agent5 续跑，
    # 但本次绝不能把部分素材交给 Agent6 当作完整成片。
    if any(ep.status == "render_partial" for ep in episodes):
        return END
    if any(ep.status == "submission_uncertain" for ep in episodes):
        return END
    # P0-2：队列满/连接等待是"本次运行暂停、下次进程恢复"的状态，
    # 绝不能让其他 storyboard_done 集触发 Agent5 无等待重入。
    if any(ep.status in {"waiting_for_agnes_capacity", "waiting_for_connectivity"} for ep in episodes):
        return END
    if any(ep.status == "storyboard_done" for ep in episodes):
        return "agent5_director"
    # Failed creates are retried only after a new process start, where the
    # connectivity preflight runs again. Never loop POST attempts in one run.
    if any(ep.status == "render_failed" for ep in episodes):
        return END
    # E1：运行时若某集进入 await review，暂停本次运行（等人工标记后重跑推进）。
    if any(ep.status == "awaiting_review" for ep in episodes):
        print(">> Router: 有剧集等待人工审阅，本次运行暂停。")
        return END
    if any(ep.status == "video_generated" for ep in episodes):
        return "agent6_editor"
    if any(ep.status == "edit_completed" for ep in episodes):
        return "agent7_growth"
    if any(ep.status == "growth_ready" for ep in episodes):
        return "agent8_analytics"
    return END

def build_drama_matrix_graph():
    """
    Construct the complete agent graph for the DramaMatrix pipeline
    """
    workflow = StateGraph(DramaState)
    
    # 注册所有节点
    workflow.add_node("agent1_scout", process_agent1_scout)
    workflow.add_node("agent2_hook_analyzer", process_agent2_hook_analyzer)
    workflow.add_node("agent3_head_writer", process_agent3_head_writer)
    workflow.add_node("agent4_storyboard", process_agent4_storyboard)
    workflow.add_node("agent5_director", process_agent5_director)
    workflow.add_node("agent6_editor", process_agent6_editor)
    workflow.add_node("agent7_growth", process_agent7_growth)
    workflow.add_node("agent8_analytics", process_agent8_analytics)
    
    # 建立宏观主轴边
    workflow.add_conditional_edges(
        START,
        route_from_start,
        {
            "agent1_scout": "agent1_scout",
            "agent2_hook_analyzer": "agent2_hook_analyzer",
            "agent3_head_writer": "agent3_head_writer",
            "agent4_storyboard": "agent4_storyboard",
            "agent5_director": "agent5_director",
            "agent6_editor": "agent6_editor",
            "agent7_growth": "agent7_growth",
            "agent8_analytics": "agent8_analytics",
            END: END,
        },
    )
    workflow.add_edge("agent1_scout", "agent2_hook_analyzer")
    
    # 立项会审判定
    workflow.add_conditional_edges(
        "agent2_hook_analyzer",
        route_after_agent2,
        {
            "agent3_head_writer": "agent3_head_writer",
            "agent1_scout": "agent1_scout",
            END: END
        }
    )
    
    workflow.add_edge("agent3_head_writer", "agent4_storyboard")
    
    # 分镜场记与生成流转
    workflow.add_conditional_edges(
        "agent4_storyboard",
        route_next_step_for_episode,
        {
            "agent5_director": "agent5_director",
            "agent4_storyboard": "agent4_storyboard",
            "agent6_editor": "agent6_editor",
            "agent7_growth": "agent7_growth",
            "agent8_analytics": "agent8_analytics",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "agent5_director",
        route_next_step_for_episode,
        {
            "agent4_storyboard": "agent4_storyboard",
            "agent6_editor": "agent6_editor",
            "agent5_director": "agent5_director",
            "agent7_growth": "agent7_growth",
            "agent8_analytics": "agent8_analytics",
            END: END
        }
    )
    
    # 后期发行流转
    workflow.add_conditional_edges(
        "agent6_editor",
        route_next_step_for_episode,
        {
            "agent5_director": "agent5_director",
            "agent6_editor": "agent6_editor",
            "agent7_growth": "agent7_growth",
            "agent8_analytics": "agent8_analytics",
            "agent4_storyboard": "agent4_storyboard",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "agent7_growth",
        route_next_step_for_episode,
        {
            "agent5_director": "agent5_director",
            "agent6_editor": "agent6_editor",
            "agent7_growth": "agent7_growth",
            "agent8_analytics": "agent8_analytics",
            "agent4_storyboard": "agent4_storyboard",
            END: END,
        },
    )
    # 市场回环：数据洞察后按周期上限决定是否回到 Agent 1 进行下一轮选品
    workflow.add_conditional_edges(
        "agent8_analytics",
        route_after_cycles,
        {
            "agent1_scout": "agent1_scout",
            END: END,
        },
    )
    
    # 编译成可运行对象
    app = workflow.compile()
    return app
