"""R1 路由修复测试：死路状态可恢复重试、系统级阻塞守卫、回环门禁。

回归背景：editing_failed/growth_failed 在 route_from_start 无分支，agent6/7
失败后 --resume 直接 END 无法重试；agent1/2/3 的系统级阻塞状态会把空素材
继续流向下游或被误判为"被否换书"。
"""

import os
import unittest
from unittest.mock import patch

from langgraph.graph import END

from src.state import EpisodeState


def state_with(status):
    return {
        "project_id": "router_p",
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": EpisodeState(status=status)},
        "characters": [],
        "system_status": "starting",
    }


class DeadEndRecoveryTests(unittest.TestCase):
    def test_editing_failed_resumes_into_agent6(self):
        from src.graph import route_from_start

        self.assertEqual(route_from_start(state_with("editing_failed")), "agent6_editor")

    def test_growth_failed_resumes_into_agent7(self):
        from src.graph import route_from_start

        self.assertEqual(route_from_start(state_with("growth_failed")), "agent7_growth")

    def test_analytics_done_does_not_reenter_agent8(self):
        from src.graph import route_from_start

        self.assertEqual(route_from_start(state_with("analytics_done")), END)

    def test_analytics_done_counts_as_finished(self):
        from src.graph import _all_episodes_finished

        self.assertTrue(_all_episodes_finished({"ep_01": EpisodeState(status="analytics_done")}))


class SystemBlockGuardTests(unittest.TestCase):
    def test_route_after_agent1_ends_when_blocked_on_source(self):
        from src.graph import route_after_agent1

        state = state_with("pending_script")
        state["system_status"] = "blocked_on_source"
        self.assertEqual(route_after_agent1(state), END)

    def test_route_after_agent1_proceeds_normally(self):
        from src.graph import route_after_agent1

        state = state_with("pending_script")
        state["system_status"] = "evaluating"
        self.assertEqual(route_after_agent1(state), "agent2_hook_analyzer")

    def test_route_after_agent2_treats_model_failure_as_block_not_rejection(self):
        from src.graph import route_after_agent2

        state = state_with("pending_script")
        state["system_status"] = "blocked_on_text_model"
        state["scout_attempts"] = 0
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_SCOUT_ATTEMPTS": "3"}, clear=False):
            self.assertEqual(route_after_agent2(state), END)

    def test_route_after_agent3_ends_when_blocked_on_script(self):
        from src.graph import route_after_agent3

        state = state_with("script_done")
        state["system_status"] = "blocked_on_script"
        self.assertEqual(route_after_agent3(state), END)

    def test_route_after_cycles_waits_when_waiting_for_market_data(self):
        from src.graph import route_after_cycles

        state = state_with("growth_ready")
        state["system_status"] = "waiting_for_market_data"
        with patch.dict(
            os.environ, {"DRAMAMATRIX_AUTO_NEXT_CYCLE": "1", "DRAMAMATRIX_MAX_CYCLES": "3"}, clear=False
        ):
            self.assertEqual(route_after_cycles(state), END)

    def test_system_block_ends_midrun_routing_instead_of_agent5_loop(self):
        """回归（集群实测）：Agent5 早退且集状态仍 storyboard_done 时，
        route_next_step_for_episode 只看集状态会把 Agent5 无限重入直至
        GraphRecursionError——系统级 blocked_/waiting_ 必须结束本次运行。
        """
        from src.graph import route_next_step_for_episode

        for blocked_status in (
            "blocked_on_missing_character_bible",
            "blocked_on_agnes_configuration",
            "waiting_for_episode_review",
        ):
            state = state_with("storyboard_done")
            state["system_status"] = blocked_status
            self.assertEqual(
                route_next_step_for_episode(state), END,
                f"{blocked_status} 应结束本次运行而非重入 Agent5",
            )

    def test_storyboard_revision_loop_not_cut_by_system_guard(self):
        """director_rejected → Agent4 的重写循环是有意的，守卫不得拦截。"""
        from src.graph import route_next_step_for_episode

        state = state_with("director_rejected")
        state["system_status"] = "blocked_on_storyboard_revision"
        self.assertEqual(route_next_step_for_episode(state), "agent4_storyboard")

    def test_healthy_system_status_still_routes_storyboard_done(self):
        from src.graph import route_next_step_for_episode

        state = state_with("storyboard_done")
        state["system_status"] = "ready_for_storyboard"
        self.assertEqual(route_next_step_for_episode(state), "agent5_director")


class FailureReportSystemBlockTests(unittest.TestCase):
    def test_system_level_block_is_reported_without_episode_states(self):
        from src.failure_report import build_failure_report

        state = state_with("growth_ready")
        state["system_status"] = "waiting_for_market_data"
        report = build_failure_report(state)
        self.assertIn("system_block", report)
        self.assertEqual(report["system_block"]["status"], "waiting_for_market_data")
        self.assertTrue(report["system_block"]["disposition"])

    def test_no_report_when_healthy(self):
        from src.failure_report import build_failure_report

        state = state_with("analytics_done")
        state["system_status"] = "cycle_completed"
        report = build_failure_report(state)
        self.assertNotIn("system_block", report)


if __name__ == "__main__":
    unittest.main()
