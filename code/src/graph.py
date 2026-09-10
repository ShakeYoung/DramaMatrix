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

def _is_system_blocked(state: DramaState) -> bool:
    """R1：系统级阻塞/等待状态（blocked_* / waiting_*）——本次运行结束。

    各阻塞点（缺书源/文本模型失败/编剧失败/等待投放数据）修复配置后
    --resume 会由 route_from_start 自然回到对应节点重试。
    """
    status = state.get("system_status", "") or ""
    return status.startswith(("blocked_", "waiting_"))


def route_after_agent1(state: DramaState) -> str:
    # R1：agent1 缺真实书源时置 blocked_on_source——直接结束本次运行，
    # 不能带着空素材流入 agent2 空转消耗换书额度。
    if _is_system_blocked(state):
        print(f">> Router: 检测到系统阻塞状态 {state.get('system_status')}，本次运行结束。")
        return END
    return "agent2_hook_analyzer"


def route_after_agent2(state: DramaState) -> str:
    # R1：评审模型失败（blocked_on_text_model）不是"被否换书"——先于
    # 换书逻辑判断，避免失败被误当作退回重选消耗 scout_attempts。
    if _is_system_blocked(state):
        print(f">> Router: 检测到系统阻塞状态 {state.get('system_status')}，本次运行结束。")
        return END
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
    terminal = {"edit_completed", "growth_ready", "growth_failed", "editing_failed", "analytics_done"}
    return all(ep.status in terminal for ep in episodes.values())


def route_after_agent3(state: DramaState) -> str:
    # R1：编剧失败（blocked_on_script）保留失败状态，不进入分镜——
    # 否则空 episodes/半成品会流入 agent4 产出"格式完整但故事不成立"的分镜。
    if _is_system_blocked(state):
        print(f">> Router: 检测到系统阻塞状态 {state.get('system_status')}，本次运行结束。")
        return END
    return "agent4_storyboard"


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

    R1：自动换书回环默认关闭（DRAMAMATRIX_AUTO_NEXT_CYCLE=0）——回环会在
    同一项目、同集键（ep_01...）、同输出路径上开新书，旧书剧集残留/被覆盖。
    开启前必须先建立"项目—作品—周期"隔离。等待投放数据（waiting_*）时
    也绝不进入下一周期。
    """
    if _is_system_blocked(state):
        print(f">> Router: 检测到系统等待/阻塞状态 {state.get('system_status')}，不进入下一周期。")
        return END
    auto_next = os.getenv("DRAMAMATRIX_AUTO_NEXT_CYCLE", "0").strip().lower() in {"1", "true", "yes"}
    if not auto_next:
        if _all_episodes_finished(state.get("episodes", {})):
            print(">> 市场回环已关闭（DRAMAMATRIX_AUTO_NEXT_CYCLE=0），本项目生产结束。")
        return END
    cycle = state.get("task_cycle", 1)
    max_cycles = int(os.getenv("DRAMAMATRIX_MAX_CYCLES", "1"))
    if _all_episodes_finished(state.get("episodes", {})) and cycle <= max_cycles:
        next_cycle = cycle + 1
        print(f">> 第 {cycle} 周期成片完成，市场回环 → Agent 1（下一周期 {next_cycle}）")
        return "agent1_scout"
    return END


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
        # E1：后台人工质检——未审阅完则暂停；决定齐全后统一由
        # Agent5 应用 approve/redraw/delete 状态转换。
        review_eps = [ep for ep in episodes if ep.status == "awaiting_review"]
        if review_eps and not all(_review_handled(project_id, ep) for ep in review_eps):
            try:
                from src.review import interactive_review_available, review_mode
                if review_mode() == "interactive" and interactive_review_available():
                    print(">> Router: 检测到未完成审阅，交由 Agent5 在当前终端恢复交互。")
                    return "agent5_director"
            except Exception as exc:
                print(f">> Router: 无法启动交互审阅，安全暂停：{exc}")
            print(">> Router: 检测到 await review，暂停等待人工标记。")
            return END
        if review_eps:
            print(">> Router: 人工审阅决定已齐全，交由 Agent5 应用。")
            return "agent5_director"
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
        # R1：agent6/agent7 的失败状态此前是死路（无路由分支，resume 直接 END，
        # 无法重试）。修复 ffmpeg/素材/路径后 --resume 应能从失败步骤重进。
        if any(ep.status == "editing_failed" for ep in episodes):
            print(">> Router: 检测到 editing_failed，恢复进入 Agent 6 重试后期合成。")
            return "agent6_editor"
        if any(ep.status == "growth_failed" for ep in episodes):
            print(">> Router: 检测到 growth_failed，恢复进入 Agent 7 重试投流切片。")
            return "agent7_growth"
        # R2：整集验收——已 approve 进入 Agent7；rework 回 Agent6 重合成；
        # 未决定则 END 暂停（人工核片）。
        review_eps = [ep for ep in episodes if ep.status == "awaiting_episode_review"]
        if review_eps:
            for ep in review_eps:
                ep_key = _ep_key(ep)
                try:
                    from src.episode_review import episode_approved, episode_rework

                    if episode_approved(project_id, ep_key):
                        print(f">> Router: {ep_key} 整集验收已通过，进入投流。")
                        return "agent7_growth"
                    if episode_rework(project_id, ep_key):
                        print(f">> Router: {ep_key} 整集验收标记 rework，重走 Agent 6 合成。")
                        return "agent6_editor"
                except Exception as exc:
                    print(f">> Router: 读取整集验收决定失败（{exc}），暂停。")
                    return END
            print(">> Router: 有剧集等待整集验收（episode_review），暂停等待人工核片。")
            return END
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
    # R2：整集验收未决时暂停本次运行（approve/rework 由下次 resume 路由）。
    if any(ep.status == "awaiting_episode_review" for ep in episodes):
        print(">> Router: 有剧集等待整集验收，本次运行暂停。")
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
    # R1：agent1/agent3 之后改为条件边——系统级阻塞（缺书源/编剧失败）时
    # 结束本次运行，而不是带着空素材继续流向下游。
    workflow.add_conditional_edges(
        "agent1_scout",
        route_after_agent1,
        {
            "agent2_hook_analyzer": "agent2_hook_analyzer",
            END: END,
        },
    )

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

    workflow.add_conditional_edges(
        "agent3_head_writer",
        route_after_agent3,
        {
            "agent4_storyboard": "agent4_storyboard",
            END: END,
        }
    )
    
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
