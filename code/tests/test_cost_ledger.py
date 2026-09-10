"""R3 成本账本测试：幂等记账、预留/确认、预算门禁、分镜预估、报表。"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import src.db as db_module
from src.state import EpisodeScriptData, EpisodeState, ShotStoryboard


class CostLedgerTestBase(unittest.TestCase):
    def setUp(self):
        self._original = db_module.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        db_module.DB_PATH = os.path.join(self.tmp.name, "cost.db")
        db_module.init_db()
        self._env = patch.dict(os.environ, {
            "DRAMAMATRIX_PRICE_VIDEO_CREATE": "0.8",
            "DRAMAMATRIX_EPISODE_BUDGET": "0",
            "DRAMAMATRIX_PROJECT_BUDGET": "0",
        }, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def tearDown(self):
        db_module.DB_PATH = self._original
        self.tmp.cleanup()


class CostLedgerTests(CostLedgerTestBase):
    def test_reservation_idempotent_on_same_task(self):
        from src import cost_ledger

        self.assertTrue(cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1"))
        self.assertFalse(cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1"))
        self.assertAlmostEqual(cost_ledger.episode_spent("p", "ep_01"), 0.8)

    def test_confirm_upgrades_without_double_count(self):
        from src import cost_ledger

        cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1")
        self.assertTrue(cost_ledger.confirm_video_cost("p", "task-1"))
        self.assertFalse(cost_ledger.confirm_video_cost("p", "task-1"), "重复确认幂等")
        self.assertAlmostEqual(cost_ledger.episode_spent("p", "ep_01"), 0.8,
                               msg="confirmed 升级而非追加")

    def test_redraw_attempt_counted_for_same_shot(self):
        from src import cost_ledger

        cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1")
        cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-2")  # 重绘
        events = db_module.db_query_cost_events("p", "ep_01")
        attempts = sorted(e["attempt"] for e in events)
        self.assertEqual(attempts, [1, 2], "同镜第二次付费创建应记为 attempt=2")
        self.assertAlmostEqual(cost_ledger.episode_spent("p", "ep_01"), 1.6)

    def test_estimate_idempotent_and_not_counted_as_spend(self):
        from src import cost_ledger

        amount = cost_ledger.record_storyboard_estimate("p", "ep_01", 1, 20)
        self.assertAlmostEqual(amount, 16.0)
        cost_ledger.record_storyboard_estimate("p", "ep_01", 1, 20)
        events = [e for e in db_module.db_query_cost_events("p", "ep_01") if e["kind"] == "estimate"]
        self.assertEqual(len(events), 1, "同版本预估幂等")
        self.assertAlmostEqual(cost_ledger.episode_spent("p", "ep_01"), 0.0,
                                msg="预估不参与花费口径")

    def test_zero_amount_not_recorded(self):
        from src import cost_ledger

        with patch.dict(os.environ, {"DRAMAMATRIX_PRICE_VIDEO_CREATE": "0"}, clear=False):
            self.assertFalse(cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1"))

    def test_episode_budget_allows_blocks_overrun(self):
        from src import cost_ledger

        cost_ledger.record_video_reservation("p", "ep_01", "s1", "task-1")  # 0.8
        with patch.dict(os.environ, {"DRAMAMATRIX_EPISODE_BUDGET": "1.0"}, clear=False):
            self.assertTrue(cost_ledger.episode_budget_allows("p", "ep_01", 0.2))
            self.assertFalse(cost_ledger.episode_budget_allows("p", "ep_01", 0.5))
        with patch.dict(os.environ, {"DRAMAMATRIX_EPISODE_BUDGET": "0"}, clear=False):
            self.assertTrue(cost_ledger.episode_budget_allows("p", "ep_01", 99))


class StoryboardEstimateGateTests(CostLedgerTestBase):
    def _state(self):
        shots = [
            ShotStoryboard(shot_id=f"s{i:02d}", camera="Static", visual_prompt="v",
                           dialogue="", duration="4s", audio="a")
            for i in range(10)
        ]
        return {
            "project_id": "gate_p",
            "meta_info": {}, "market_feedback": None, "source_material": {},
            "master_script_outline": "",
            "episodes": {"ep_01": EpisodeState(
                status="storyboard_done",
                script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
                storyboard_data=shots,
            )},
            "characters": [], "task_cycle": 1, "scout_attempts": 0,
            "system_status": "storyboard_done",
        }

    def test_gate_blocks_when_estimate_exceeds_episode_budget(self):
        from src.agents.agent4_storyboard import _apply_cost_estimate_gate

        state = self._state()
        with patch.dict(os.environ, {"DRAMAMATRIX_EPISODE_BUDGET": "5.0"}, clear=False):
            _apply_cost_estimate_gate(state)  # 10 镜 x 0.8 = 8.0 > 5.0
        self.assertEqual(state["episodes"]["ep_01"].status, "storyboard_blocked")
        self.assertTrue(any(fb.reason_code == "COST_ESTIMATE_EXCEEDED"
                            for fb in state["episodes"]["ep_01"].feedback_log))
        self.assertEqual(state["system_status"], "blocked_on_storyboard_generation")

    def test_gate_passes_when_budget_sufficient(self):
        from src.agents.agent4_storyboard import _apply_cost_estimate_gate

        state = self._state()
        with patch.dict(os.environ, {"DRAMAMATRIX_EPISODE_BUDGET": "10.0"}, clear=False):
            _apply_cost_estimate_gate(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "storyboard_done")

    def test_project_budget_blocks_when_planned_total_exceeds(self):
        from src.agents.agent4_storyboard import _apply_cost_estimate_gate

        state = self._state()
        with patch.dict(os.environ, {
            "DRAMAMATRIX_PROJECT_BUDGET": "7.0",  # 10 镜 x 0.8 = 8.0 > 7.0
        }, clear=False):
            _apply_cost_estimate_gate(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "storyboard_blocked")
        self.assertTrue(any(fb.reason_code == "COST_ESTIMATE_EXCEEDED"
                            for fb in state["episodes"]["ep_01"].feedback_log))

    def test_gate_dormant_without_price(self):
        from src.agents.agent4_storyboard import _apply_cost_estimate_gate

        state = self._state()
        with patch.dict(os.environ, {
            "DRAMAMATRIX_PRICE_VIDEO_CREATE": "0",
            "DRAMAMATRIX_EPISODE_BUDGET": "0.01",
        }, clear=False):
            _apply_cost_estimate_gate(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "storyboard_done",
                         "价目未配置时金额门禁休眠（次数护栏仍生效）")


class Agent5BudgetBreakerTests(CostLedgerTestBase):
    def test_money_budget_breaks_before_paid_create(self):
        """单集金额预算不足时，新请求发出前熔断（不产生任何创建/入账）。"""
        from src.agents.agent5_director import process_agent5_director

        shot = ShotStoryboard(shot_id="s01", camera="Static", visual_prompt="v",
                              dialogue="", duration="5s", audio="a")
        state = {
            "project_id": "breaker_p", "meta_info": {},
            "market_feedback": None, "source_material": {}, "master_script_outline": "",
            "episodes": {"ep_01": EpisodeState(
                status="storyboard_done",
                script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
                storyboard_data=[shot],
            )},
            "characters": [], "task_cycle": 1, "scout_attempts": 0,
            "system_status": "starting",
        }
        with patch.dict(os.environ, {
            "DRAMAMATRIX_EPISODE_BUDGET": "0.5",  # < 单次 0.8
            "DRAMAMATRIX_ALLOW_NO_CHARACTERS": "1",
            "DRAMAMATRIX_REVIEW_MODE": "0",
            "DRAMAMATRIX_VIDEO_PROVIDER": "dummy",
            "DRAMAMATRIX_IMAGE_PROVIDER": "off",
        }, clear=False), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        ep = state["episodes"]["ep_01"]
        self.assertEqual(ep.status, "render_pending")
        self.assertTrue(any(fb.reason_code == "EPISODE_BUDGET" for fb in ep.feedback_log))
        self.assertEqual(state["system_status"], "blocked_on_agnes_render")
        self.assertAlmostEqual(db_module.db_cost_spent("breaker_p"), 0.0,
                               msg="熔断发生在请求发出前，零入账")
        self.assertEqual(ep.video_assets, [], "未产生任何视频资产")


class CostReportTests(CostLedgerTestBase):
    def test_report_metrics(self):
        from src import cost_ledger
        from src.cost_report import build_cost_report, render_report_text

        cost_ledger.record_video_reservation("rp", "ep_01", "s1", "task-1")
        cost_ledger.confirm_video_cost("rp", "task-1")
        cost_ledger.record_video_reservation("rp", "ep_01", "s2", "task-2")
        cost_ledger.record_video_reservation("rp", "ep_01", "s1", "task-3")  # 重绘
        cost_ledger.confirm_video_cost("rp", "task-3")
        # 成片时长证据：ep_01 字幕版 60s。
        db_module.db_save_project_state({
            "project_id": "rp", "system_status": "done",
            "episodes": {"ep_01": {
                "status": "growth_ready",
                "deliverables": [{"kind": "subtitled", "path": "/x.mp4", "actual_duration": 60.0}],
            }},
        })

        report = build_cost_report("rp")
        ep = report["episodes"]["ep_01"]
        self.assertAlmostEqual(ep["confirmed"], 1.6)
        self.assertAlmostEqual(ep["reserved_open"], 0.8)
        self.assertAlmostEqual(ep["spent"], 2.4)
        self.assertAlmostEqual(ep["redraw_cost"], 0.8)
        self.assertAlmostEqual(ep["redraw_share"], 0.8 / 2.4, places=3)
        # 每分钟合格成片成本 = confirmed 1.6 / 1 分钟。
        self.assertAlmostEqual(ep["per_minute_cost"], 1.6)
        text = render_report_text(report)
        self.assertIn("ep_01", text)
        self.assertIn("重绘占比", text)


if __name__ == "__main__":
    unittest.main()
