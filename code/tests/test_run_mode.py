"""R1 运行模式测试：production 默认无模拟回退，demo 显式可用且带标记。

回归背景：此前无密钥/无数据时 agent1 用 mock 书目、agent2 模拟 88 分过审、
agent3 写固定"豪车"剧情、agent8 写随机 CPA——整条链路能"成功"产出演示内容
且与真实产出无法区分。
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import src.db as db_module
from src.runtime_options import is_demo_mode, run_mode
from src.state import EpisodeState


def base_state(project_id="mode_p"):
    return {
        "project_id": project_id,
        "meta_info": {"source_title": "", "genre_tags": [], "scout_excluded": []},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {},
        "task_cycle": 1,
        "scout_attempts": 0,
        "characters": [],
        "system_status": "starting",
    }


class RunModeHelpersTests(unittest.TestCase):
    def test_default_is_production(self):
        env = {k: v for k, v in os.environ.items() if k != "DRAMAMATRIX_RUN_MODE"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(run_mode(), "production")
            self.assertFalse(is_demo_mode())

    def test_demo_mode_opt_in(self):
        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": "demo"}, clear=False):
            self.assertTrue(is_demo_mode())

    def test_invalid_mode_fails_fast(self):
        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": "staging"}, clear=False):
            with self.assertRaises(ValueError):
                run_mode()


class Agent1ModeTests(unittest.TestCase):
    def test_production_blocks_without_real_source(self):
        from src.agents.agent1_scout import process_agent1_scout

        state = base_state("mode_a1")
        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": "production"}, clear=False), patch(
            "src.agents.agent1_scout.db_get_unprocessed_novel", return_value=None
        ), patch("src.agents.agent1_scout.scrape_biquge_novel") as mock_scrape:
            process_agent1_scout(state)
        mock_scrape.assert_not_called()
        self.assertEqual(state["system_status"], "blocked_on_source")
        self.assertEqual(state["scout_attempts"], 0, "阻塞不算一次换书尝试")
        self.assertNotIn("raw_text", state["source_material"])

    def test_demo_uses_mock_scraper(self):
        from src.agents.agent1_scout import process_agent1_scout

        state = base_state("mode_a1_demo")
        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": "demo"}, clear=False), patch(
            "src.agents.agent1_scout.db_get_unprocessed_novel", return_value=None
        ), patch(
            "src.agents.agent1_scout.db_insert_novel", return_value=True
        ), patch("src.agents.agent1_scout.db_mark_novel_processed"), patch("time.sleep"):
            process_agent1_scout(state)
        self.assertEqual(state["system_status"], "evaluating")
        self.assertTrue(state["source_material"].get("raw_text"))


class Agent2ModeTests(unittest.TestCase):
    def _run_agent2(self, state, mode):
        from src.agents.agent2_hook_analyzer import process_agent2_hook_analyzer

        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": mode}, clear=False), patch(
            "src.agents.agent2_hook_analyzer.create_text_model",
            side_effect=RuntimeError("no key"),
        ):
            process_agent2_hook_analyzer(state)

    def test_production_blocks_on_model_failure(self):
        state = base_state("mode_a2")
        state["source_material"]["raw_text"] = "剧情文本" * 100
        self._run_agent2(state, "production")
        self.assertEqual(state["system_status"], "blocked_on_text_model")
        self.assertIsNone(state["source_material"].get("report"), "失败不得写入模拟过审 report")

    def test_demo_marks_simulated_verdict(self):
        state = base_state("mode_a2_demo")
        state["source_material"]["raw_text"] = "剧情文本" * 100
        self._run_agent2(state, "demo")
        report = state["source_material"]["report"]
        self.assertIsNotNone(report)
        self.assertTrue(report.simulated, "模拟评审必须带 simulated 标记")
        self.assertTrue(report.is_approved)


class Agent3ModeTests(unittest.TestCase):
    def _run_agent3(self, state, mode):
        from src.agents.agent3_head_writer import process_agent3_head_writer

        with patch.dict(os.environ, {"DRAMAMATRIX_RUN_MODE": mode}, clear=False), patch(
            "src.agents.agent3_head_writer.create_text_model",
            side_effect=RuntimeError("no key"),
        ):
            process_agent3_head_writer(state)

    def _approved(self):
        state = base_state("mode_a3")
        state["meta_info"]["source_title"] = "测试书"
        state["source_material"]["raw_text"] = "剧情文本" * 100
        state["source_material"]["report"] = type(
            "R", (), {"is_approved": True}
        )()
        return state

    def test_production_keeps_failure_without_fallback_plot(self):
        state = self._approved()
        self._run_agent3(state, "production")
        self.assertEqual(state["system_status"], "blocked_on_script")
        self.assertEqual(state["episodes"], {}, "不得写入与原作无关的固定剧情")
        self.assertEqual(state.get("master_script_outline", ""), "")

    def test_demo_fallback_is_marked_and_consistent(self):
        state = self._approved()
        self._run_agent3(state, "demo")
        self.assertEqual(state["system_status"], "ready_for_storyboard")
        self.assertTrue(state["episodes"])
        for ep in state["episodes"].values():
            self.assertEqual(ep.content_origin, "demo_fallback")
        # 演示回退的大纲不得谎报集数（旧版写"全剧共80集"实际只建 2 集）。
        self.assertNotIn("80集", state["master_script_outline"])
        self.assertEqual(
            len(state["episodes"]),
            int(state["master_script_outline"].split("共")[1].split("集")[0]),
        )


class Agent8ModeTests(unittest.TestCase):
    def setUp(self):
        self._original = db_module.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        db_module.DB_PATH = os.path.join(self.tmp.name, "mode.db")
        db_module.init_db()

    def tearDown(self):
        db_module.DB_PATH = self._original
        self.tmp.cleanup()

    def _count_rows(self):
        conn = db_module._connect()
        try:
            return conn.execute("SELECT COUNT(*) FROM analytics").fetchone()[0]
        finally:
            conn.close()

    def _growth_state(self):
        state = base_state("mode_a8")
        state["episodes"] = {"ep_01": EpisodeState(status="growth_ready")}
        return state

    def test_production_waits_without_data_and_writes_nothing(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        state = self._growth_state()
        env = {k: v for k, v in os.environ.items() if k != "DRAMAMATRIX_ANALYTICS_IMPORT"}
        with patch.dict(os.environ, env, clear=True):
            process_agent8_analytics(state)
        self.assertEqual(state["system_status"], "waiting_for_market_data")
        self.assertEqual(self._count_rows(), 0, "production 无数据时零写入")
        self.assertEqual(state["episodes"]["ep_01"].status, "growth_ready", "保持待回流，可 resume 重进")
        self.assertEqual(state["task_cycle"], 1, "周期不虚增")

    def test_production_waits_on_import_error_without_fallback(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        state = self._growth_state()
        with patch.dict(
            os.environ,
            {"DRAMAMATRIX_RUN_MODE": "production", "DRAMAMATRIX_ANALYTICS_IMPORT": "/nonexistent.csv"},
            clear=False,
        ):
            process_agent8_analytics(state)
        self.assertEqual(state["system_status"], "waiting_for_market_data")
        self.assertEqual(self._count_rows(), 0)

    def test_demo_writes_simulated_row_marked(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        state = self._growth_state()
        env = {k: v for k, v in os.environ.items() if k != "DRAMAMATRIX_ANALYTICS_IMPORT"}
        env["DRAMAMATRIX_RUN_MODE"] = "demo"
        with patch.dict(os.environ, env, clear=True):
            process_agent8_analytics(state)
        conn = db_module._connect()
        try:
            row = conn.execute(
                "SELECT source, project_id, platform FROM analytics"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row)
        self.assertEqual(row[0], "simulated")
        self.assertEqual(row[1], "mode_a8")
        self.assertEqual(row[2], "simulated")
        self.assertEqual(state["episodes"]["ep_01"].status, "analytics_done")


if __name__ == "__main__":
    unittest.main()
