"""R1 状态隔离测试：按集限定运行不得缩减项目快照。

回归背景：main.py 曾把 episodes 整体替换为单集字典，而
db_save_project_state 按 project_id 整份覆盖——只跑一集后其他集
从最新快照消失。现在保存时按键合并（传入方优先，快照独有集保留）。
"""

import os
import tempfile
import unittest
from unittest.mock import patch

import src.db as db_module
from src.state import EpisodeScriptData, EpisodeState


def make_state(project_id="iso_p", episodes=None):
    return {
        "project_id": project_id,
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": episodes if episodes is not None else {},
        "characters": [],
        "system_status": "starting",
    }


def make_episode(status="script_done"):
    return EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep", outline="o", ending_hook="h"),
    )


class SnapshotMergeTests(unittest.TestCase):
    def setUp(self):
        self._original = db_module.DB_PATH
        self.tmp = tempfile.TemporaryDirectory()
        db_module.DB_PATH = os.path.join(self.tmp.name, "iso.db")
        db_module.init_db()

    def tearDown(self):
        db_module.DB_PATH = self._original
        self.tmp.cleanup()

    def _snapshot_episodes(self, project_id):
        snap = db_module.db_get_project_state_snapshot(project_id)
        assert snap is not None
        return snap["state"]["episodes"]

    def test_filtered_save_preserves_other_episodes(self):
        full = make_state(
            episodes={
                "ep_01": make_episode(status="analytics_done"),
                "ep_02": make_episode(status="storyboard_done"),
            }
        )
        db_module.db_save_project_state(full)

        # 模拟按集限定运行：内存中只有 ep_02，且其状态被推进。
        filtered = make_state(
            episodes={"ep_02": make_episode(status="video_generated")}
        )
        filtered["system_status"] = "video_assets_downloaded"
        db_module.db_save_project_state(filtered)

        merged = self._snapshot_episodes("iso_p")
        self.assertIn("ep_01", merged, "未运行的集必须保留在快照中")
        self.assertIn("ep_02", merged)
        self.assertEqual(merged["ep_01"]["status"], "analytics_done", "未运行的集状态不得被改动")
        self.assertEqual(merged["ep_02"]["status"], "video_generated")

    def test_incoming_wins_on_same_key(self):
        first = make_state(episodes={"ep_01": make_episode(status="script_done")})
        db_module.db_save_project_state(first)

        updated = make_state(episodes={"ep_01": make_episode(status="storyboard_done")})
        db_module.db_save_project_state(updated)

        merged = self._snapshot_episodes("iso_p")
        self.assertEqual(merged["ep_01"]["status"], "storyboard_done")

    def test_repeated_filtered_saves_do_not_resurrect_old_state(self):
        """过滤运行多次保存后，目标集最新状态生效，其余集不回退。"""
        full = make_state(
            episodes={"ep_01": make_episode(status="analytics_done"), "ep_02": make_episode()}
        )
        db_module.db_save_project_state(full)

        filtered = make_state(episodes={"ep_02": make_episode(status="rendering")})
        db_module.db_save_project_state(filtered)
        db_module.db_save_project_state(filtered)

        merged = self._snapshot_episodes("iso_p")
        self.assertEqual(merged["ep_02"]["status"], "rendering")
        self.assertEqual(merged["ep_01"]["status"], "analytics_done")

    def test_new_project_save_without_existing_snapshot(self):
        state = make_state(episodes={"ep_01": make_episode()})
        db_module.db_save_project_state(state)  # 不应因无既有快照报错
        self.assertIn("ep_01", self._snapshot_episodes("iso_p"))


if __name__ == "__main__":
    unittest.main()
