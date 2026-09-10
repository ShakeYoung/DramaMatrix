"""U1–U6 升级项测试：供应商接线 / 视觉 QC / 参考图链路 / prompt 外置 / 知识库 / 审阅台。"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.state import (
    CharacterSheet,
    EpisodeScriptData,
    EpisodeState,
    ShotStoryboard,
)


def make_shot(shot_id="s01", visual_prompt="A woman looks up in the rain.", dialogue=""):
    return ShotStoryboard(
        shot_id=shot_id,
        camera="Static close-up",
        visual_prompt=visual_prompt,
        dialogue=dialogue,
        duration="5s",
        audio="soft rain",
        scene_id="scene_01",
    )


def make_state(characters=None, shots=None):
    episode = EpisodeState(
        status="storyboard_done",
        script_data=EpisodeScriptData(ep_id="ep_01", outline="test", ending_hook="hook"),
        storyboard_data=shots if shots is not None else [make_shot()],
    )
    state = {
        "project_id": "upgrade_test_project",
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": episode},
        "characters": characters if characters is not None else [],
        "system_status": "starting",
    }
    return state


class IsolatedEnv(unittest.TestCase):
    """临时输出目录 + 临时数据库 + 关键环境变量清理。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self._env = {
            "DRAMAMATRIX_OUTPUT_DIR": str(self.tmp),
            "DRAMAMATRIX_REVIEW_MODE": "0",
            "DRAMAMATRIX_ALLOW_NO_CHARACTERS": "1",
            "DRAMAMATRIX_CONDITIONAL_GENERATION": "0",
            "DRAMAMATRIX_VIDEO_PROVIDER": "",
            "DRAMAMATRIX_IMAGE_PROVIDER": "",
            "DRAMAMATRIX_PROMPT_DIR": "",
        }
        self._saved = {k: os.environ.get(k) for k in self._env}
        os.environ.update(self._env)
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


# ----------------------------- U1：供应商接线 -----------------------------

class ProviderWiringTests(IsolatedEnv):
    def test_neutral_exception_hierarchy(self):
        from src.agnes_video import (
            AgnesConfigurationError,
            AgnesConnectionError,
            AgnesContentPolicyViolation,
            AgnesGatewayUncertain,
            AgnesQueueFull,
            AgnesSubmissionUncertain,
            AgnesTaskFailed,
        )
        from src.provider_errors import (
            ProviderConfigurationError,
            ProviderConnectionError,
            ProviderContentPolicyViolation,
            ProviderError,
            ProviderGatewayUncertain,
            ProviderQueueFull,
            ProviderSubmissionUncertain,
            ProviderTaskFailed,
        )

        cases = [
            (AgnesConfigurationError, ProviderConfigurationError),
            (AgnesConnectionError, ProviderConnectionError),
            (AgnesContentPolicyViolation, ProviderContentPolicyViolation),
            (AgnesGatewayUncertain, ProviderGatewayUncertain),
            (AgnesQueueFull, ProviderQueueFull),
            (AgnesSubmissionUncertain, ProviderSubmissionUncertain),
            (AgnesTaskFailed, ProviderTaskFailed),
        ]
        for agnes_cls, provider_cls in cases:
            self.assertTrue(issubclass(agnes_cls, provider_cls), agnes_cls.__name__)
            self.assertTrue(issubclass(agnes_cls, ProviderError))

    def test_validate_allows_newer_agnes_models(self):
        from src.agnes_video import AgnesVideoSettings

        AgnesVideoSettings(api_key="k", model="agnes-video-v3.0").validate()  # 不再抛错
        with self.assertRaises(Exception):
            AgnesVideoSettings(api_key="k", model="  ").validate()

    def test_unknown_provider_fails_loudly(self):
        from src.model_providers import get_video_provider
        from src.provider_errors import ProviderConfigurationError

        os.environ["DRAMAMATRIX_VIDEO_PROVIDER"] = "kling"
        with self.assertRaises(ProviderConfigurationError):
            get_video_provider()

    def test_unknown_provider_blocks_pipeline(self):
        from src.agents.agent5_director import process_agent5_director

        os.environ["DRAMAMATRIX_VIDEO_PROVIDER"] = "kling"
        state = make_state()
        with patch("src.agents.agent5_director.db_save_project_state"):
            state = process_agent5_director(state)
        self.assertEqual(state["system_status"], "blocked_on_agnes_configuration")
        self.assertEqual(state["episodes"]["ep_01"].status, "storyboard_done")

    def test_dummy_provider_end_to_end(self):
        """DRAMAMATRIX_VIDEO_PROVIDER=dummy 可端到端跑完一集（不需要 Agnes key）。"""
        from src.agents.agent5_director import process_agent5_director

        os.environ["DRAMAMATRIX_VIDEO_PROVIDER"] = "dummy"
        state = make_state(shots=[make_shot("s01"), make_shot("s02")])
        with patch("src.agents.agent5_director.db_save_project_state"):
            state = process_agent5_director(state)
        ep = state["episodes"]["ep_01"]
        self.assertEqual(ep.status, "video_generated")
        for asset in ep.video_assets:
            self.assertTrue(Path(asset.local_path).is_file())
            self.assertEqual(asset.model_version, "dummy-video-v1")

    def test_agnes_provider_profile_reflects_settings(self):
        from src.agnes_video import AgnesVideoSettings
        from src.model_providers import AgnesProvider

        settings = AgnesVideoSettings(api_key="k", width=1080, height=1920, frame_rate=30)
        provider = AgnesProvider(settings=settings, client=MagicMock())
        profile = provider.render_profile()
        self.assertEqual((profile.width, profile.height, profile.frame_rate), (1080, 1920, 30))
        self.assertEqual(profile.model, settings.model)


# ----------------------------- U3：视觉相似度 QC -----------------------------

class VisualSimilarityTests(IsolatedEnv):
    def test_perceptual_hash_identical_and_different(self):
        from src.continuity_qc import hash_bit_count, perceptual_hash

        flat = bytes([128]) * 256
        stripes = bytes([0, 255]) * 128  # 交替条纹产生非零梯度
        self.assertEqual(perceptual_hash(flat), perceptual_hash(flat))
        distance = bin(perceptual_hash(flat) ^ perceptual_hash(stripes)).count("1")
        self.assertGreater(distance, 0)
        self.assertLessEqual(distance, hash_bit_count())

    def test_similarity_score_bounds(self):
        from src.continuity_qc import hash_bit_count, perceptual_hash

        gray = bytes([100]) * 256
        self.assertEqual(perceptual_hash(gray), perceptual_hash(gray))
        self.assertEqual(
            1.0 - bin(perceptual_hash(gray) ^ perceptual_hash(gray)).count("1") / hash_bit_count(),
            1.0,
        )

    def test_default_checker_advisory_then_gating(self):
        from src.continuity_qc import DefaultChecker

        shot = make_shot()
        with tempfile.TemporaryDirectory() as d:
            curr = Path(d) / "v.mp4"
            curr.write_bytes(b"video")
            prev_frame = Path(d) / "prev.png"
            curr_frame = Path(d) / "curr.png"
            prev_frame.write_bytes(b"p")
            curr_frame.write_bytes(b"c")
            with patch("src.continuity_qc.frame_similarity", return_value=0.2), patch(
                "src.continuity_qc._frame_brightness_distance", return_value=1.0
            ), patch("src.continuity_qc._ffmpeg_available", return_value=True):
                os.environ["DRAMAMATRIX_QC_SIMILARITY_GATE"] = "0"
                result = DefaultChecker().check(None, prev_frame, curr, shot, curr_frame)
                self.assertTrue(result.passed)  # 默认只告警
                self.assertTrue(any("结构相似度过低" in i for i in result.issues))
                self.assertAlmostEqual(result.metrics["frame_similarity"], 0.2)

                os.environ["DRAMAMATRIX_QC_SIMILARITY_GATE"] = "1"
                result = DefaultChecker().check(None, prev_frame, curr, shot, curr_frame)
                self.assertFalse(result.passed)  # 开门禁即拦截
                self.assertAlmostEqual(result.metrics["similarity_threshold"], 0.55)
            os.environ.pop("DRAMAMATRIX_QC_SIMILARITY_GATE", None)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_frame_similarity_with_real_frames(self):
        import subprocess

        from src.continuity_qc import frame_similarity

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, source in [("a.png", "testsrc=s=64x64"), ("flat.png", "color=black:s=64x64")]:
                path = Path(d) / name
                subprocess.run(
                    [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error",
                     "-f", "lavfi", "-i", source, "-frames:v", "1", str(path)],
                    check=True,
                )
                paths.append(path)
            self.assertAlmostEqual(frame_similarity(paths[0], paths[0]), 1.0)
            self.assertLess(frame_similarity(paths[0], paths[1]), 1.0)


# ----------------------------- U2：参考图链路 -----------------------------

class ImageProviderTests(IsolatedEnv):
    def test_provider_factory_off_dummy_unknown(self):
        from src.image_providers import DummyImageProvider, get_image_provider
        from src.provider_errors import ProviderConfigurationError

        self.assertIsNone(get_image_provider())  # 默认 off，保持旧行为
        os.environ["DRAMAMATRIX_IMAGE_PROVIDER"] = "dummy"
        self.assertIsInstance(get_image_provider(), DummyImageProvider)
        os.environ["DRAMAMATRIX_IMAGE_PROVIDER"] = "midjourney"
        with self.assertRaises(ProviderConfigurationError):
            get_image_provider()

    def test_openai_provider_requires_config(self):
        from src.image_providers import OpenAICompatImageProvider
        from src.provider_errors import ProviderConfigurationError

        for key in ("IMAGE_MODEL_API_KEY", "TEXT_MODEL_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(key, None)
        with self.assertRaises(ProviderConfigurationError):
            OpenAICompatImageProvider()

    def test_openai_provider_writes_b64_image(self):
        import base64

        from src.image_providers import OpenAICompatImageProvider

        os.environ["IMAGE_MODEL_BASE_URL"] = "https://img.example.com/v1"
        os.environ["IMAGE_MODEL_API_KEY"] = "k"
        payload = base64.b64encode(b"png-bytes").decode("ascii")
        response = MagicMock(status_code=200, ok=True)
        response.json.return_value = {"data": [{"b64_json": payload}]}
        destination = self.tmp / "refs" / "char_01.png"
        with patch("src.image_providers.requests.post", return_value=response) as post:
            OpenAICompatImageProvider().generate("红衣少年立绘", destination)
        self.assertEqual(destination.read_bytes(), b"png-bytes")
        self.assertIn("/images/generations", post.call_args.args[0])


class CharacterReferenceTests(IsolatedEnv):
    def _characters(self):
        return [
            CharacterSheet(
                name="萧寒",
                appearance="红衣黑发青年",
                signature="",
                role="男主",
                character_id="char_01",
                canonical_name="萧寒",
                reference_image_prompt="红衣黑发青年全身立绘",
            ),
        ]

    def test_ensure_generates_records_and_backfills(self):
        from src.character_refs import ensure_character_reference_images
        from src.image_providers import DummyImageProvider

        characters = self._characters()
        provider = DummyImageProvider()
        mapping = ensure_character_reference_images("upgrade_test_project", characters, provider)

        self.assertIn("萧寒", mapping)
        path = Path(mapping["萧寒"])
        self.assertTrue(path.is_file())
        self.assertEqual(characters[0].reference_image_path, str(path))
        # 证据链落库：character 类型 + sha256
        conn = sqlite3.connect(self._db())
        rows = conn.execute(
            "SELECT asset_type, ref_id, sha256 FROM reference_assets WHERE asset_type='character'"
        ).fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], "char_01")
        self.assertTrue(rows[0][2])

    def _db(self):
        import src.db as db_module

        return db_module.DB_PATH

    def test_ensure_is_idempotent(self):
        from src.character_refs import ensure_character_reference_images
        from src.image_providers import DummyImageProvider

        characters = self._characters()
        provider = DummyImageProvider()
        ensure_character_reference_images("upgrade_test_project", characters, provider)
        with patch.object(DummyImageProvider, "generate") as generate:
            mapping = ensure_character_reference_images("upgrade_test_project", characters, provider)
            generate.assert_not_called()
        self.assertIn("萧寒", mapping)

    def test_prepare_shot_reference_priority_and_match(self):
        from src.continuity import prepare_shot_reference

        char_refs = {"萧寒": "/refs/char_01.png", "小萧寒": "/refs/char_02.png"}
        scene_refs = {"scene_01": "https://cdn/scene_01.png"}
        # 尾帧最优先
        ref = prepare_shot_reference(make_shot(visual_prompt="萧寒转身"), scene_refs, "data:image/png;base64,xxx", char_refs)
        self.assertEqual(ref["source"], "tail")
        # 场景参考次之
        ref = prepare_shot_reference(make_shot(visual_prompt="萧寒转身"), scene_refs, None, char_refs)
        self.assertEqual(ref["source"], "scene")
        # 无场景参考时命中角色（长名优先）
        shot = make_shot(visual_prompt="小萧寒站在雨中")
        ref = prepare_shot_reference(shot, {}, None, char_refs)
        self.assertEqual(ref["source"], "character")
        self.assertEqual(ref["image_url"], "/refs/char_02.png")
        self.assertEqual(ref["ref_id"], "小萧寒")
        # 未命中角色 → 无参考
        ref = prepare_shot_reference(make_shot(visual_prompt="空旷的街道"), {}, None, char_refs)
        self.assertIsNone(ref["image_url"])

    def test_agent5_sends_character_reference_as_data_uri(self):
        """条件生成开启时，场景首镜应携带角色参考图（data URI）并落证据链。"""
        from src.agents.agent5_director import process_agent5_director
        from src.agnes_video import AgnesVideoSettings
        from src.character_refs import ensure_character_reference_images
        from src.image_providers import DummyImageProvider

        os.environ["DRAMAMATRIX_CONDITIONAL_GENERATION"] = "1"
        os.environ["DRAMAMATRIX_IMAGE_PROVIDER"] = "dummy"
        characters = self._characters()
        # 预生成参考图（走真实链路：dummy 图像供应商 + DB 证据）
        ensure_character_reference_images("upgrade_test_project", characters, DummyImageProvider())

        fake_client = MagicMock()
        fake_client.preflight.return_value = None
        fake_client.create_video.return_value = {"video_id": "v1", "task_id": "t1"}
        fake_client.wait_for_video.return_value = {
            "status": "completed",
            "metadata": {"url": "http://cdn/v1.mp4"},
        }
        def _download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"mp4")
            return dest

        fake_client.download_video.side_effect = _download
        state = make_state(characters=characters, shots=[make_shot(visual_prompt="萧寒立于雨中")])
        settings = AgnesVideoSettings(api_key="test-key")
        with patch("src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=fake_client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        kwargs = fake_client.create_video.call_args.kwargs
        self.assertTrue(kwargs.get("image_url", "").startswith("data:image/"), kwargs.get("image_url"))
        conn = sqlite3.connect(self._db())
        kinds = [row[0] for row in conn.execute(
            "SELECT asset_type FROM reference_assets"
        ).fetchall()]
        conn.close()
        self.assertIn("character", kinds)


# ----------------------------- U5：prompt 外置 -----------------------------

class PromptFileTests(IsolatedEnv):
    def test_fallback_default_when_file_missing(self):
        from src.prompt_files import load_prompt

        os.environ["DRAMAMATRIX_PROMPT_DIR"] = str(self.tmp / "empty")
        self.assertEqual(load_prompt("nope", "内置默认"), "内置默认")

    def test_file_override_wins(self):
        from src.prompt_files import load_prompt, prompt_dir

        (self.tmp / "prompts").mkdir()
        (self.tmp / "prompts" / "agent9_test.md").write_text("运营自定义风格", encoding="utf-8")
        os.environ["DRAMAMATRIX_PROMPT_DIR"] = str(self.tmp / "prompts")
        self.assertEqual(load_prompt("agent9_test", "内置默认"), "运营自定义风格")

    def test_repo_prompt_files_carry_placeholders(self):
        """仓库自带 prompts/*.md 应存在且包含占位符，供 replace 填充。"""
        from src.prompt_files import prompt_dir

        repo_dir = Path(prompt_dir())
        if not repo_dir.is_dir():
            self.skipTest("prompt 目录不在预期位置")
        for name, placeholder in [
            ("agent2_forum", "{prior_knowledge}"),
            ("agent3_head_writer", "{format_instructions}"),
            ("agent4_storyboard_recovery", "{error_message}"),
            ("character_bible", "{format_instructions}"),
        ]:
            path = repo_dir / f"{name}.md"
            self.assertTrue(path.is_file(), path)
            self.assertIn(placeholder, path.read_text(encoding="utf-8"))

    def test_agent2_template_renders_like_before(self):
        from src.agents.agent2_hook_analyzer import _AGENT2_FORUM_TEMPLATE

        rendered = (
            _AGENT2_FORUM_TEMPLATE
            .replace("{prior_knowledge}", "先验文本")
            .replace("{format_instructions}", "格式说明")
        )
        self.assertIn("先验文本", rendered)
        self.assertIn("格式说明", rendered)
        self.assertNotIn("{prior_knowledge}", rendered)


# ----------------------------- U6：知识库检索 -----------------------------

class KnowledgeBaseTests(IsolatedEnv):
    def test_seed_is_idempotent(self):
        from src.knowledge_base import seed_knowledge_base

        first = seed_knowledge_base()
        self.assertGreater(first, 0)
        self.assertEqual(seed_knowledge_base(), 0)

    def test_retrieve_ranks_by_relevance(self):
        from src.knowledge_base import retrieve, seed_knowledge_base

        seed_knowledge_base()
        visual = retrieve("视觉 奇观 生视频 画面", k=3)
        self.assertTrue(visual)
        self.assertIn("视觉", visual[0]["title"])

        ad = retrieve("投放 切片 投流 标题 完播", k=3)
        self.assertTrue(ad)
        self.assertIn("投放", ad[0]["title"])

        nothing = retrieve("量子力学偏微分方程", k=3)
        self.assertEqual(nothing, [])

    def test_format_prior_knowledge(self):
        from src.knowledge_base import format_prior_knowledge, retrieve, seed_knowledge_base

        seed_knowledge_base()
        formatted = format_prior_knowledge(retrieve("节奏 情绪 转折", k=2))
        self.assertIn("检索自知识库", formatted)
        self.assertIn("1.", formatted)

    def test_agent2_prior_uses_real_retrieval(self):
        from src.agents.agent2_hook_analyzer import load_prior_knowledge

        text = load_prior_knowledge("主角隐藏身份 巨大反差 打脸")
        self.assertIn("先验知识", text)
        self.assertIn("反差", text)


# ----------------------------- U4：审阅台 -----------------------------

class ReviewServerTests(IsolatedEnv):
    def _start_server(self):
        from src.review_server import ReviewRequestHandler, serve  # noqa: F401
        from http.server import ThreadingHTTPServer

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), ReviewRequestHandler)
        import threading

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    def _get(self, url):
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read()

    def test_manifest_decisions_and_traversal_guard(self):
        from src.review import write_review_manifest
        from src.state import EpisodeState as ES

        base = self._start_server()
        ep = EpisodeState(
            status="video_generated",
            script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
            storyboard_data=[make_shot()],
        )
        write_review_manifest("upgrade_test_project", "ep_01", ep)

        status, body = self._get(f"{base}/api/manifest?project=upgrade_test_project&ep=ep_01")
        self.assertEqual(status, 200)
        manifest = json.loads(body)
        self.assertEqual(manifest["shots"][0]["shot_id"], "s01")

        # 保存决定：写回 decisions.json（与 CLI 完全同一路径）。
        request = urllib.request.Request(
            f"{base}/api/decisions?project=upgrade_test_project&ep=ep_01",
            data=json.dumps({"s01": "approve"}).encode("utf-8"),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            self.assertEqual(response.status, 200)
        status, body = self._get(f"{base}/api/decisions?project=upgrade_test_project&ep=ep_01")
        self.assertEqual(json.loads(body), {"s01": "approve"})

        # 路径穿越被拒绝。
        try:
            self._get(f"{base}/thumb?path=/etc/passwd")
            self.fail("预期 403")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 403)

        # 页面本身可渲染。
        status, body = self._get(f"{base}/?project=upgrade_test_project&ep=ep_01")
        self.assertEqual(status, 200)
        self.assertIn(b"DramaMatrix", body)

    def test_missing_manifest_returns_404_with_hint(self):
        base = self._start_server()
        try:
            self._get(f"{base}/api/manifest?project=nope&ep=ep_99")
            self.fail("预期 404")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            self.assertIn(b"review.json", exc.read())


if __name__ == "__main__":
    unittest.main()
