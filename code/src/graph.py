from src.state import DramaState
from langgraph.graph import StateGraph, START, END
from src.agents.agent1_scout import process_agent1_scout
from src.agents.agent2_hook_analyzer import process_agent2_hook_analyzer
from src.agents.agent3_head_writer import process_agent3_head_writer
from src.agents.agent4_storyboard import process_agent4_storyboard
from src.agents.agent5_director import process_agent5_director
from src.agents.agent6_editor import process_agent6_editor
from src.agents.agent7_growth import process_agent7_growth
from src.agents.agent8_analytics import process_agent8_analytics

def route_after_agent2(state: DramaState) -> str:
    # 评审未通过则直接结束（或退回 A1，这里演示结束）
    report = state.get("source_material", {}).get("report")
    if not report or not report.is_approved:
        return END
    return "agent3_head_writer"


def route_from_start(state: DramaState) -> str:
    """Resume a persisted project at its first unfinished stage."""
    episodes = list(state.get("episodes", {}).values())
    if episodes:
        if any(ep.status == "submission_uncertain" for ep in episodes):
            return END
        if any(ep.status in {"script_done", "director_rejected"} for ep in episodes):
            return "agent4_storyboard"
        if any(ep.status in {"storyboard_done", "rendering", "render_pending", "render_failed"} for ep in episodes):
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
    # A submitted task is intentionally paused after a recoverable network error.
    # The next process start resumes polling it before any new render is submitted.
    if any(ep.status == "render_pending" for ep in episodes):
        return END
    if any(ep.status == "submission_uncertain" for ep in episodes):
        return END
    if any(ep.status == "storyboard_done" for ep in episodes):
        return "agent5_director"
    # Failed creates are retried only after a new process start, where the
    # connectivity preflight runs again. Never loop POST attempts in one run.
    if any(ep.status == "render_failed" for ep in episodes):
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
    workflow.add_edge("agent8_analytics", END)
    
    # 编译成可运行对象
    app = workflow.compile()
    return app
