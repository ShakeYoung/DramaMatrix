"""W1–W7 路线图升级测试：分镜编辑器 / 真实数据源 / 权属 / 身份质检 / 看板。"""

import json
import os
import sqlite3
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.state import (
    CharacterSheet,
    EpisodeScriptData,
    EpisodeState,
    GrowthAsset,
    GrowthMeta,
    ShotStoryboard,
)


def make_shot(shot_id="s01", prompt="A woman looks up in the rain."):
    return ShotStoryboard(
        shot_id=shot_id, camera="Static", visual_prompt=prompt,
        dialogue="", duration="5s", audio="rain", scene_id="scene_01",
    )


def make_state(**overrides):
    episode = EpisodeState(
        status=overrides.pop("status", "storyboard_blocked"),
        script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
        storyboard_data=[make_shot(), make_shot("s02", "A man turns around.")],
    )
    state = {
        "project_id": "roadmap_project",
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": episode},
        "characters": [],
        "system_status": "blocked_on_storyboard",
    }
    state.update(overrides)
    return state


class IsolatedEnv(unittest.TestCase):
    """临时输出目录 + 临时数据库 + 环境变量还原。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._overrides = {
            "DRAMAMATRIX_OUTPUT_DIR": str(self.tmp),
            "DRAMAMATRIX_LOCAL_NOVEL_DIR": "",
            "DRAMAMATRIX_ANALYTICS_IMPORT": "",
            "DRAMAMATRIX_SOURCE_LICENSE": "",
            "DRAMAMATRIX_SOURCE_OWNER": "",
            "DRAMAMATRIX_SOURCE_NOTE": "",
            "DRAMAMATRIX_IDENTITY_QC": "",
            "DRAMAMATRIX_IDENTITY_GATE": "",
        }
        self._saved = {k: os.environ.get(k) for k in self._overrides}
        os.environ.update(self._overrides)
        self.addCleanup(self._restore)
        import src.db as db_module

        self._db_path = db_module.DB_PATH
        db_module.DB_PATH = str(self.tmp / "test.db")
        self.addCleanup(self._restore_db)
        db_module.init_db()

    def _restore(self):
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _restore_db(self):
        import src.db as db_module

        db_module.DB_PATH = self._db_path

    def db(self):
        import src.db as db_module

        return db_module.DB_PATH


# ----------------------------- W1：分镜编辑器 -----------------------------

class StoryboardEditorTests(IsolatedEnv):
    def test_export_import_and_reset(self):
        from src.db import db_save_project_state
        from src.storyboard_editor import (
            export_storyboard,
            import_storyboard,
            load_state,
            reset_status,
        )

        state = make_state()
        db_save_project_state(state)

        out = export_storyboard("roadmap_project", "ep_01")
        payload = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["shots"]), 2)
        self.assertEqual(payload["current_status"], "storyboard_blocked")

        # 人工修正：改 prompt + 删一镜。
        payload["shots"][0]["visual_prompt"] = "Corrected close-up."
        payload["shots"] = payload["shots"][:1]
        edited = self.tmp / "edited.json"
        edited.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        import_storyboard("roadmap_project", "ep_01", edited, target_status="storyboard_done")
        reloaded = load_state("roadmap_project")
        ep = reloaded["episodes"]["ep_01"]
        self.assertEqual(ep.status, "storyboard_done")
        self.assertEqual(len(ep.storyboard_data), 1)
        self.assertEqual(ep.storyboard_data[0].visual_prompt, "Corrected close-up.")
        self.assertEqual(ep.storyboard_version, 2)  # 版本递增隔离

        reset_status("roadmap_project", "ep_01", target_status="script_done")
        reloaded = load_state("roadmap_project")
        self.assertEqual(reloaded["episodes"]["ep_01"].status, "script_done")

    def test_import_rejects_duplicate_shot_ids(self):
        from src.db import db_save_project_state
        from src.storyboard_editor import import_storyboard

        db_save_project_state(make_state())
        payload = {"shots": [make_shot().model_dump(), make_shot().model_dump()]}
        path = self.tmp / "dup.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            import_storyboard("roadmap_project", "ep_01", path)
        # 校验失败不应改动已保存状态。
        from src.storyboard_editor import load_state

        self.assertEqual(load_state("roadmap_project")["episodes"]["ep_01"].status,
                         "storyboard_blocked")


# ----------------------------- W2：真实数据源 -----------------------------

class LocalSourcesTests(IsolatedEnv):
    def test_load_local_novel(self):
        from src.local_sources import load_local_novel

        library = self.tmp / "novels"
        library.mkdir()
        (library / "book_a.txt").write_text(
            "# tags: 男频,玄幻,复仇\n第一章 张若尘醒来。", encoding="utf-8")
        (library / "book_b.txt").write_text("第二章 另一本书。", encoding="utf-8")

        os.environ["DRAMAMATRIX_LOCAL_NOVEL_DIR"] = str(library)
        novel = load_local_novel()
        self.assertEqual(novel["title"], "book_a")
        self.assertEqual(novel["tags"], ["男频", "玄幻", "复仇"])
        self.assertIn("张若尘", novel["content"])

        skipped = load_local_novel(exclude_titles={"book_a"})
        self.assertEqual(skipped["title"], "book_b")
        self.assertEqual(skipped["tags"], ["未分类"])

        os.environ["DRAMAMATRIX_LOCAL_NOVEL_DIR"] = ""
        self.assertIsNone(load_local_novel())

    def test_agent1_uses_local_library_and_records_rights(self):
        from src.agents.agent1_scout import process_agent1_scout
        from src.db import db_insert_novel  # noqa: F401 - 确认表可用

        library = self.tmp / "novels"
        library.mkdir()
        (library / "我的小说.txt").write_text(
            "# tags: 女频,重生\n第一章 沈娇娇重生了。", encoding="utf-8")
        os.environ["DRAMAMATRIX_LOCAL_NOVEL_DIR"] = str(library)
        os.environ["DRAMAMATRIX_SOURCE_LICENSE"] = "licensed"
        os.environ["DRAMAMATRIX_SOURCE_OWNER"] = "某某文学 授权"

        state = make_state(status="pending_script")
        state = process_agent1_scout(state)
        self.assertEqual(state["meta_info"]["source_title"], "我的小说")
        self.assertIn("沈娇娇", state["source_material"]["raw_text"])
        self.assertIn("我的小说", state["meta_info"].get("scout_excluded", []))

        # W3：权属声明落库（licensed）
        conn = sqlite3.connect(self.db())
        rows = conn.execute(
            "SELECT scope, ref_id, license, owner FROM rights_records"
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], ("source", "我的小说", "licensed", "某某文学 授权"))

    def test_load_analytics_records_csv_and_json(self):
        from src.local_sources import load_analytics_records

        csv_path = self.tmp / "ads.csv"
        csv_path.write_text(
            "ep_id,views,cpa,completion_rate,tags\nep_01,12000,8.5,0.32,女频;重生\nep_02,8000,12.0,0.21,男频\n",
            encoding="utf-8")
        os.environ["DRAMAMATRIX_ANALYTICS_IMPORT"] = str(csv_path)
        records = load_analytics_records()
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["ep_id"], "ep_01")
        self.assertEqual(json.loads(records[0]["tags"]), ["女频", "重生"])

        json_path = self.tmp / "ads.json"
        json_path.write_text(json.dumps({"records": [
            {"ep_id": "ep_03", "views": 500, "cpa": 3.2, "completion_rate": 0.5, "tags": "系统"}
        ]}, ensure_ascii=False), encoding="utf-8")
        os.environ["DRAMAMATRIX_ANALYTICS_IMPORT"] = str(json_path)
        self.assertEqual(load_analytics_records()[0]["ep_id"], "ep_03")

        os.environ["DRAMAMATRIX_ANALYTICS_IMPORT"] = str(self.tmp / "missing.csv")
        with self.assertRaises(ValueError):
            load_analytics_records()
        os.environ["DRAMAMATRIX_ANALYTICS_IMPORT"] = ""
        self.assertIsNone(load_analytics_records())

    def test_agent8_imports_real_analytics(self):
        from src.agents.agent8_analytics import process_agent8_analytics

        csv_path = self.tmp / "ads.csv"
        csv_path.write_text(
            "ep_id,views,cpa,completion_rate,tags\nep_01,10000,6.0,0.45,女频;重生\n",
            encoding="utf-8")
        os.environ["DRAMAMATRIX_ANALYTICS_IMPORT"] = str(csv_path)

        state = make_state()
        state["episodes"]["ep_01"].status = "growth_ready"
        state["meta_info"]["genre_tags"] = ["女频"]
        state = process_agent8_analytics(state)
        self.assertIsNotNone(state["market_feedback"])
        conn = sqlite3.connect(self.db())
        rows = conn.execute(
            "SELECT ep_id, views, cpa, completion_rate, tags FROM analytics"
        ).fetchall()
        conn.close()
        self.assertEqual(rows, [("ep_01", 10000, 6.0, 0.45, '["女频", "重生"]')])


# ----------------------------- W3：版权权属 -----------------------------

class RightsTests(IsolatedEnv):
    def test_record_is_idempotent(self):
        from src.rights import record_source_rights

        os.environ["DRAMAMATRIX_SOURCE_LICENSE"] = "owned"
        record_source_rights("p1", "小说A")
        record_source_rights("p1", "小说A")
        conn = sqlite3.connect(self.db())
        count = conn.execute("SELECT COUNT(*) FROM rights_records").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)

    def test_publish_package_carries_rights(self):
        from src.publish import export_publish_package
        from src.rights import rights_block

        os.environ["DRAMAMATRIX_SOURCE_LICENSE"] = "licensed"
        os.environ["DRAMAMATRIX_SOURCE_OWNER"] = "版权方X"
        clip = self.tmp / "clip.mp4"
        clip.write_bytes(b"mp4")
        ep = EpisodeState(
            status="growth_ready",
            script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
            storyboard_data=[make_shot()],
            growth_assets=[GrowthAsset(
                name="hook", path=str(clip), start_seconds=0.0, duration_seconds=5.0
            )],
            growth_meta=GrowthMeta(title="t", description="d", tags=["x"], cover_prompt="c"),
        )
        base = export_publish_package("roadmap_project", "ep_01", ep, output_dir=self.tmp / "pub")
        self.assertIsNotNone(base)
        meta = json.loads((base / "publish_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["rights"]["source"]["license"], "licensed")
        self.assertEqual(meta["rights"]["source"]["owner"], "版权方X")
        self.assertIn("AI", meta["rights"]["generated_assets"]["disclosure"])

        # 环境未配置时 unknown 而非缺失。
        os.environ.pop("DRAMAMATRIX_SOURCE_LICENSE", None)
        self.assertEqual(rights_block("no-such-project")["source"]["license"], "unknown")


# ----------------------------- W4：身份质检 -----------------------------

class IdentityQCTests(IsolatedEnv):
    def setUp(self):
        super().setUp()
        import src.identity_qc as identity_module

        identity_module._judge = None
        identity_module._judge_failed = False
        self.identity_module = identity_module

    def test_parse_verdict_lenient(self):
        from src.identity_qc import _parse_verdict

        verdict = _parse_verdict('```json\n{"score": 82, "reasons": "发型一致"}\n```')
        self.assertEqual(verdict["score"], 82.0)
        with self.assertRaises(Exception):
            _parse_verdict("评分很高")  # 无 JSON
        with self.assertRaises(Exception):
            _parse_verdict('{"score": 150}')  # 越界

    def test_judge_requires_config(self):
        from src.identity_qc import VisionIdentityJudge
        from src.provider_errors import ProviderConfigurationError

        for key in ("IDENTITY_MODEL_API_KEY", "TEXT_MODEL_API_KEY", "OPENAI_API_KEY",
                    "IDENTITY_MODEL_BASE_URL", "TEXT_MODEL_BASE_URL", "OPENAI_BASE_URL"):
            os.environ.pop(key, None)
        with self.assertRaises(ProviderConfigurationError):
            VisionIdentityJudge()

    def test_judge_identity_request_shape(self):
        from src.identity_qc import judge_identity

        os.environ["DRAMAMATRIX_IDENTITY_QC"] = "1"
        os.environ["IDENTITY_MODEL_BASE_URL"] = "https://llm.example.com/v1"
        os.environ["IDENTITY_MODEL_API_KEY"] = "k"
        ref = self.tmp / "ref.png"
        ref.write_bytes(b"png-ref")
        cur = self.tmp / "cur.png"
        cur.write_bytes(b"png-cur")

        response = MagicMock(status_code=200, ok=True)
        response.json.return_value = {
            "choices": [{"message": {"content": '{"score": 91, "reasons": "同一人"}'}}]
        }
        with patch("src.identity_qc.requests.post", return_value=response) as post:
            verdict = judge_identity(ref, cur)
        self.assertEqual(verdict["score"], 91.0)
        payload = post.call_args.kwargs["json"]
        self.assertIn("/chat/completions", post.call_args.args[0])
        image_urls = [
            part["image_url"]["url"]
            for part in payload["messages"][1]["content"] if part["type"] == "image_url"
        ]
        self.assertEqual(len(image_urls), 2)
        self.assertTrue(image_urls[0].startswith("data:image/png;base64,"))

    def test_default_checker_identity_advisory_and_gate(self):
        from src.continuity_qc import DefaultChecker

        clip = self.tmp / "v.mp4"
        clip.write_bytes(b"video")
        ref = self.tmp / "ref.png"
        ref.write_bytes(b"r")
        cur = self.tmp / "cur.png"
        cur.write_bytes(b"c")
        shot = make_shot()
        with patch.object(self.identity_module, "identity_qc_enabled", return_value=True), patch(
            "src.continuity_qc._ffmpeg_available", return_value=False
        ), patch.object(
            self.identity_module, "judge_identity", return_value={"score": 40.0, "reasons": "发型突变"}
        ):
            os.environ["DRAMAMATRIX_IDENTITY_GATE"] = "0"
            result = DefaultChecker().check(None, None, clip, shot, cur, reference_frame=ref)
            self.assertTrue(result.passed)  # 默认告警
            self.assertTrue(any("身份一致性低于阈值" in i for i in result.issues))
            self.assertEqual(result.metrics["identity_score"], 40.0)

            os.environ["DRAMAMATRIX_IDENTITY_GATE"] = "1"
            result = DefaultChecker().check(None, None, clip, shot, cur, reference_frame=ref)
            self.assertFalse(result.passed)  # 开门禁即拦截
        os.environ.pop("DRAMAMATRIX_IDENTITY_GATE", None)


# ----------------------------- W5：运营看板 -----------------------------

class DashboardTests(IsolatedEnv):
    def _seed(self):
        from src.db import db_insert_qc_result, db_record_agnes_usage, db_save_project_state

        state = make_state()
        state["episodes"]["ep_01"].status = "awaiting_review"
        state["episodes"]["ep_01"].rendered_shot_count = 2
        state["episodes"]["ep_01"].planned_shot_count = 2
        db_save_project_state(state)
        db_record_agnes_usage(
            task_id="t1", project_id="roadmap_project", ep_key="ep_01", shot_id="s01",
            frames=121, width=720, height=1280, provider="agnes", queue_wait_seconds=2.0,
            render_seconds=30.0, redraw_count=1,
        )
        db_record_agnes_usage(
            task_id="t2", project_id="roadmap_project", ep_key="ep_01", shot_id="s02",
            frames=121, width=720, height=1280, provider="dummy",
        )
        db_insert_qc_result(
            project_id="roadmap_project", ep_key="ep_01", shot_id="s01", passed=1,
            brightness_diff=3.0, threshold=45.0, metrics={"frame_similarity": 0.9},
        )
        db_insert_qc_result(
            project_id="roadmap_project", ep_key="ep_01", shot_id="s02", passed=0,
            brightness_diff=80.0, threshold=45.0, metrics={"frame_similarity": 0.2},
        )

    def test_payloads_aggregate_evidence(self):
        from src.dashboard_server import project_payload, projects_payload

        self._seed()
        projects = projects_payload()
        self.assertEqual(projects["summary"]["projects"], 1)
        self.assertEqual(projects["summary"]["creates"], 2)
        self.assertEqual(projects["summary"]["redraws"], 1)
        self.assertEqual(projects["summary"]["qc_pass_rate"], "50%")
        p = projects["projects"][0]
        self.assertEqual(p["project_id"], "roadmap_project")
        self.assertIn("awaiting_review", p["status_brief"])

        detail = project_payload("roadmap_project")
        self.assertEqual(detail["episodes"][0]["ep_key"], "ep_01")
        providers = {row["provider"]: row for row in detail["providers"]}
        self.assertEqual(providers["agnes"]["creates"], 1)
        self.assertEqual(providers["agnes"]["redraws"], 1)
        self.assertEqual(providers["dummy"]["creates"], 1)
        qc = detail["qc"][0]
        self.assertEqual(qc["total"], 2)
        self.assertEqual(qc["pass_rate"], "50%")
        self.assertAlmostEqual(qc["avg_similarity"], 0.55, places=2)
        self.assertGreaterEqual(len(detail["state_versions"]), 1)

    def test_server_serves_page_and_api(self):
        import threading
        from http.server import ThreadingHTTPServer

        from src.dashboard_server import DashboardHandler

        self._seed()
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        with urllib.request.urlopen(f"{base}/", timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("运营看板", response.read().decode("utf-8"))
        with urllib.request.urlopen(f"{base}/api/projects", timeout=5) as response:
            payload = json.loads(response.read())
            self.assertEqual(payload["summary"]["creates"], 2)


if __name__ == "__main__":
    unittest.main()
