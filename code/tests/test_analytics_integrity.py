"""R1 analytics 完整性测试：幂等导入、来源隔离、旧库迁移、回流生命周期。

回归背景：analytics 表无唯一约束，同一文件重复导入行数翻倍；模拟数据与
真实数据混写一张表并参与选题推荐；growth_ready 处理后不翻转，重启重复
插入且 task_cycle 虚增。
"""

import json
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import src.db as db_module
from src.state import EpisodeState


def base_state(project_id="ana_p"):
    return {
        "project_id": project_id,
        "meta_info": {"genre_tags": ["女频", "复仇"]},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": EpisodeState(status="growth_ready")},
        "task_cycle": 1,
        "scout_attempts": 0,
        "characters": [],
        "system_status": "x",
    }


class AnalyticsIntegrityTests(unittest.TestCase):
    def setUp(self):
        self._original = db_module.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        db_module.DB_PATH = os.path.join(self.tmp.name, "ana.db")
        db_module.init_db()

    def tearDown(self):
        db_module.DB_PATH = self._original
        self.tmp.cleanup()

    def _write_import_csv(self, rows):
        path = os.path.join(self.tmp.name, "import.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("ep_id,views,cpa,completion_rate,tags,platform\n")
            for row in rows:
                handle.write(row + "\n")
        return path

    def _rows(self):
        conn = db_module._connect()
        try:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM analytics").fetchall()]
        finally:
            conn.close()

    def test_import_is_idempotent_for_same_file(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        csv_path = self._write_import_csv([
            "ep_01,12000,8.5,0.32,女频;复仇,douyin",
            "ep_02,8000,12.0,0.21,女频;甜宠,douyin",
        ])
        state = base_state()
        with patch.dict(
            os.environ, {"DRAMAMATRIX_RUN_MODE": "production", "DRAMAMATRIX_ANALYTICS_IMPORT": csv_path}
        ):
            process_agent8_analytics(state)
            self.assertEqual(len(self._rows()), 2)
            # 同一文件第二次运行（如 --resume 重试）：行数不得翻倍。
            state2 = base_state()
            state2["episodes"]["ep_01"].status = "growth_ready"
            process_agent8_analytics(state2)
        rows = self._rows()
        self.assertEqual(len(rows), 2, "同一导入文件重复执行必须全部去重")
        for row in rows:
            self.assertEqual(row["source"], "imported")
            self.assertEqual(row["project_id"], "ana_p")
            self.assertEqual(row["platform"], "douyin")

    def test_new_file_content_produces_new_rows(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        first = self._write_import_csv(["ep_01,100,1.0,0.1,t1,douyin"])
        with patch.dict(
            os.environ, {"DRAMAMATRIX_RUN_MODE": "production", "DRAMAMATRIX_ANALYTICS_IMPORT": first}
        ):
            process_agent8_analytics(base_state())
        # 平台重新导出（内容变化）：新键，正常进入。
        second = self._write_import_csv(["ep_01,200,2.0,0.2,t1,douyin"])
        with patch.dict(
            os.environ, {"DRAMAMATRIX_RUN_MODE": "production", "DRAMAMATRIX_ANALYTICS_IMPORT": second}
        ):
            process_agent8_analytics(base_state())
        self.assertEqual(len(self._rows()), 2)

    def test_agent8_marks_analytics_done_and_cycle_increments_once(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        csv_path = self._write_import_csv(["ep_01,12000,8.5,0.32,女频;复仇,douyin"])
        state = base_state()
        with patch.dict(
            os.environ, {"DRAMAMATRIX_RUN_MODE": "production", "DRAMAMATRIX_ANALYTICS_IMPORT": csv_path}
        ):
            process_agent8_analytics(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "analytics_done")
        self.assertEqual(state["task_cycle"], 2)
        self.assertEqual(state["scout_attempts"], 0)

        # 重启后 growth_ready 已不存在，路由不再进 agent8。
        from src.graph import route_from_start

        self.assertNotEqual(route_from_start(state), "agent8_analytics")

    def test_legacy_rows_backfilled_and_excluded_from_trends(self):
        # 模拟旧版库：直接按旧 schema 插入无 source 的行。
        conn = db_module._connect()
        conn.execute(
            "INSERT INTO analytics (ep_id, views, cpa, completion_rate, tags) VALUES (?, ?, ?, ?, ?)",
            ("ep_legacy", 999999, 1.0, 0.99, json.dumps(["稀有标签"])),
        )
        conn.commit()
        conn.close()
        # 重新跑迁移（幂等）：旧行应被标记 legacy_unverified。
        db_module.init_db()
        rows = self._rows()
        self.assertEqual(rows[0]["source"], "legacy_unverified")
        self.assertTrue(rows[0]["dedup_key"].startswith("legacy-"))

        from src.agents.agent8_analytics import query_market_trends

        trends = query_market_trends()
        self.assertNotIn("稀有标签", trends["tags"], "未验证存量数据不得驱动选题推荐")

    def test_trends_use_only_imported_and_weight_by_views(self):
        from src.agents.agent8_analytics import query_market_trends

        conn = db_module._connect()
        # 同一标签组合内的混合样本：裸 AVG 会被小样本高分带偏，播放量加权
        # 反映真实体感完播率。
        #   "稳健大盘"：(1000×0.5 + 10×0.1)/1010 ≈ 0.496
        #   "小样本虚高"：(10×0.9 + 1000×0.25)/1010 ≈ 0.257（裸 AVG=0.575 会反超）
        conn.executemany(
            """INSERT INTO analytics
               (ep_id, views, cpa, completion_rate, tags, project_id, platform, source, dedup_key)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("ep_a1", 1000, 5.0, 0.50, json.dumps(["稳健大盘"]), "p1", "douyin", "imported", "k1"),
                ("ep_a2", 10, 5.0, 0.10, json.dumps(["稳健大盘"]), "p1", "douyin", "imported", "k2"),
                ("ep_b1", 10, 5.0, 0.90, json.dumps(["小样本虚高"]), "p1", "douyin", "imported", "k3"),
                ("ep_b2", 1000, 5.0, 0.25, json.dumps(["小样本虚高"]), "p1", "douyin", "imported", "k4"),
                # 模拟行：即使指标极高也不得参与推荐。
                ("ep_c", 999999, 1.0, 1.0, json.dumps(["模拟爆款"]), "p1", "simulated", "simulated", "k5"),
            ],
        )
        conn.commit()
        conn.close()
        trends = query_market_trends()
        self.assertEqual(trends["tags"], ["稳健大盘"])
        self.assertNotIn("模拟爆款", trends["tags"])
        self.assertNotIn("小样本虚高", trends["tags"])


if __name__ == "__main__":
    unittest.main()
