"""Tests for E1–E5: manual review, publish export, voice chain, scheduling,
and provider routing (MVP pipeline).

All media/Agnes/ffmpeg calls are mocked; runs green without ffmpeg.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.state import (
    EpisodeScriptData,
    EpisodeState,
    GeneratedVideoAsset,
    GrowthAsset,
    GrowthMeta,
    ShotStoryboard,
)


def make_shot(shot_id, scene_id=None, duration="4s"):
    kw = dict(shot_id=shot_id, camera="c", visual_prompt="v", dialogue="", duration=duration, audio="a")
    if scene_id:
        kw["scene_id"] = scene_id
    return ShotStoryboard(**kw)


def _ep(assets, shots=None, status="video_generated"):
    ep = EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"),
        storyboard_data=shots or [make_shot("s1")],
        video_assets=assets,
    )
    return ep


class ReviewManifestTests(unittest.TestCase):
    """E1: review manifest generation + routing decision."""

    def _pending_dir(self, ep_state):
        from src.agnes_video import episode_output_dir
        return episode_output_dir("p1", "ep_01") / "review"

    def _review_ep(self, project_id, ep_state):
        from src.review import write_review_manifest
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False):
            path = write_review_manifest(project_id, "ep_01", ep_state)
        return path

    def test_manifest_lists_shots(self):
        ep = _ep([], shots=[make_shot("s1"), make_shot("s2")])
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False):
            from src.review import build_review_manifest
            m = build_review_manifest("p1", "ep_01", ep)
        self.assertEqual({s["shot_id"] for s in m["shots"]}, {"s1", "s2"})

    def test_all_decided_requires_every_shot(self):
        from src.review import all_decided, load_decisions
        ep = _ep([], shots=[make_shot("s1"), make_shot("s2")])
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False):
            from src.review import write_review_manifest, _review_dir
            review_dir = _review_dir("p1", "ep_01")
            write_review_manifest("p1", "ep_01", ep)
            self.assertFalse(all_decided("p1", "ep_01", ep))
            # decide all
            (review_dir / "decisions.json").write_text(json.dumps({"s1": "approve", "s2": "approve"}), encoding="utf-8")
            self.assertTrue(all_decided("p1", "ep_01", ep))

    def test_review_mode_parsing(self):
        from src.review import review_mode
        for value, expected in [
            ("interactive", "interactive"),
            ("background", "background"),
            ("1", "background"),
            ("off", "off"),
            ("0", "off"),
        ]:
            with self.subTest(value=value), patch.dict(
                os.environ, {"DRAMAMATRIX_REVIEW_MODE": value}, clear=False
            ):
                self.assertEqual(review_mode(), expected)

    def test_all_approve_transitions_to_video_generated(self):
        from src.review import apply_review_decisions, write_review_manifest, _review_dir
        ep = _ep(
            [
                GeneratedVideoAsset(shot_id="s1", video_id="v1", status="completed", prompt="p"),
                GeneratedVideoAsset(shot_id="s2", video_id="v2", status="completed", prompt="p"),
            ],
            shots=[make_shot("s1"), make_shot("s2")],
            status="awaiting_review",
        )
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False
        ):
            write_review_manifest("p1", "ep_01", ep)
            (_review_dir("p1", "ep_01") / "decisions.json").write_text(
                json.dumps({"s1": "approve", "s2": "approve"}), encoding="utf-8"
            )
            self.assertEqual(apply_review_decisions("p1", "ep_01", ep), "video_generated")
        self.assertEqual(len(ep.video_assets), 2)

    def test_all_redraw_transitions_and_consumes_decisions(self):
        from src.review import apply_review_decisions, load_decisions, write_review_manifest, _review_dir
        ep = _ep(
            [
                GeneratedVideoAsset(shot_id="s1", video_id="v1", status="completed", prompt="p"),
                GeneratedVideoAsset(shot_id="s2", video_id="v2", status="completed", prompt="p"),
            ],
            shots=[make_shot("s1"), make_shot("s2")],
            status="awaiting_review",
        )
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False
        ):
            write_review_manifest("p1", "ep_01", ep)
            (_review_dir("p1", "ep_01") / "decisions.json").write_text(
                json.dumps({"s1": "redraw", "s2": "redraw"}), encoding="utf-8"
            )
            self.assertEqual(apply_review_decisions("p1", "ep_01", ep), "storyboard_done")
            self.assertEqual(load_decisions("p1", "ep_01"), {})
        self.assertEqual(ep.video_assets, [])

    def test_all_delete_transitions_to_render_failed(self):
        from src.review import apply_review_decisions, write_review_manifest, _review_dir
        ep = _ep(
            [GeneratedVideoAsset(shot_id="s1", video_id="v1", status="completed", prompt="p")],
            shots=[make_shot("s1")],
            status="awaiting_review",
        )
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False
        ):
            write_review_manifest("p1", "ep_01", ep)
            (_review_dir("p1", "ep_01") / "decisions.json").write_text(
                json.dumps({"s1": "delete"}), encoding="utf-8"
            )
            self.assertEqual(apply_review_decisions("p1", "ep_01", ep), "render_failed")
        self.assertEqual(ep.storyboard_data, [])

    def test_completed_background_review_routes_through_agent5(self):
        from src.graph import route_from_start
        from src.review import write_review_manifest, _review_dir
        ep = _ep(
            [GeneratedVideoAsset(shot_id="s1", video_id="v1", status="completed", prompt="p")],
            status="awaiting_review",
        )
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False
        ):
            write_review_manifest("p1", "ep_01", ep)
            (_review_dir("p1", "ep_01") / "decisions.json").write_text(
                json.dumps({"s1": "approve"}), encoding="utf-8"
            )
            self.assertEqual(
                route_from_start({"project_id": "p1", "episodes": {"ep_01": ep}}),
                "agent5_director",
            )

    def test_incomplete_interactive_review_routes_through_agent5(self):
        from src.graph import route_from_start
        from src.review import write_review_manifest
        ep = _ep(
            [GeneratedVideoAsset(shot_id="s1", video_id="v1", status="completed", prompt="p")],
            status="awaiting_review",
        )
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ,
            {"DRAMAMATRIX_OUTPUT_DIR": d, "DRAMAMATRIX_REVIEW_MODE": "interactive"},
            clear=False,
        ), patch("src.review.interactive_review_available", return_value=True):
            write_review_manifest("p1", "ep_01", ep)
            self.assertEqual(
                route_from_start({"project_id": "p1", "episodes": {"ep_01": ep}}),
                "agent5_director",
            )


class PublishExportTests(unittest.TestCase):
    """E2: publish package (clips + meta + cover)."""

    def test_export_produces_package_dir_and_meta(self):
        from src.publish import export_publish_package
        ep = _ep([], status="growth_ready")
        ep.growth_meta = GrowthMeta(title="t", description="d", tags=["a"], cover_prompt="c")
        with tempfile.TemporaryDirectory() as d:
            clip = Path(d) / "ep_01_hook.mp4"
            clip.write_bytes(b"clip")
            ep.growth_assets = [GrowthAsset(name="hook", path=str(clip), start_seconds=0, duration_seconds=10, headline="h")]
            ep.final_video_path = str(Path(d) / "master.mp4")
            with patch.dict(os.environ, {"DRAMAMATRIX_PUBLISH_EXPORT": "1"}, clear=False), \
                 patch("src.publish.extract_first_frame", return_value=None):  # no ffmpeg → no cover
                pkg = export_publish_package("p1", "ep_01", ep, output_dir=Path(d) / "publish")
                meta = json.loads((pkg / "publish_meta.json").read_text(encoding="utf-8"))
                self.assertEqual(meta["title"], "t")
                self.assertEqual(meta["clips"][0]["name"], "hook")
                self.assertTrue((pkg / "ep_01_hook.mp4").exists())

    def test_disabled_returns_none(self):
        from src.publish import export_publish_package
        ep = _ep([], status="growth_ready")
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"DRAMAMATRIX_PUBLISH_EXPORT": "0"}, clear=False):
            self.assertIsNone(export_publish_package("p1", "ep_01", ep, output_dir=Path(d)))


class DialogueDetectionTests(unittest.TestCase):
    """E3: has_dialogue_stream vs plain has_audio_stream."""

    def test_no_ffprobe_returns_none(self):
        from src.agnes_video import has_dialogue_stream
        with tempfile.TemporaryDirectory() as d, patch("src.agnes_video.shutil.which", return_value=None):
            self.assertIsNone(has_dialogue_stream(Path(d) / "x.mp4"))


class VoiceRoleTests(unittest.TestCase):
    """E3: per-role voice map + loudnorm/BGM function presence."""

    def test_role_voice_from_map(self):
        from src.tts import tts_voice
        with patch.dict(os.environ, {
            "DRAMAMATRIX_TTS_VOICE": "zh-CN-XiaoxiaoNeural",
            "DRAMAMATRIX_TTS_VOICE_MAP": '{"男主": "zh-CN-YunjianNeural"}',
        }, clear=False):
            self.assertEqual(tts_voice("男主"), "zh-CN-YunjianNeural")
            self.assertEqual(tts_voice("女主"), "zh-CN-XiaoxiaoNeural")  # fallback

    def test_loudnorm_in_bgm_mix(self):
        import inspect
        from src.tts import mix_with_bgm_and_normalize
        src = inspect.getsource(mix_with_bgm_and_normalize)
        self.assertIn("loudnorm", src)


class FailureReportTests(unittest.TestCase):
    """E4: failure report lists blocked/missing shots."""

    def test_blocked_episode_reported(self):
        from src.failure_report import build_failure_report
        ep = EpisodeState(status="render_partial", storyboard_data=[make_shot("s1"), make_shot("s2")])
        ep.video_assets = [GeneratedVideoAsset(shot_id="s1", video_id="v", status="completed", prompt="p", local_path="/x/s1.mp4")]
        report = build_failure_report({"project_id": "p1", "system_status": "blocked_on_agnes_render", "episodes": {"ep_01": ep}})
        self.assertEqual(len(report["blocked_episodes"]), 1)
        self.assertEqual(report["blocked_episodes"][0]["missing_shots"], ["s2"])


class ProviderFactoryTests(unittest.TestCase):
    """E5: provider factory + agnes_usage provider column."""

    def test_factory_returns_dummy(self):
        from src.model_providers import get_video_provider
        with patch.dict(os.environ, {"DRAMAMATRIX_VIDEO_PROVIDER": "dummy"}, clear=False):
            p = get_video_provider()
        self.assertEqual(p.name, "dummy")

    def test_factory_unknown_raises(self):
        from src.model_providers import get_video_provider
        with patch.dict(os.environ, {"DRAMAMATRIX_VIDEO_PROVIDER": "nope"}, clear=False):
            with self.assertRaises(Exception):
                get_video_provider()

    def test_agnes_usage_has_provider_column(self):
        import sqlite3
        import src.db
        with tempfile.TemporaryDirectory() as d:
            src.db.DB_PATH = os.path.join(d, "t.db")
            src.db.init_db()
            conn = sqlite3.connect(src.db.DB_PATH)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(agnes_usage)").fetchall()}
            conn.close()
        self.assertIn("provider", cols)


if __name__ == "__main__":
    unittest.main()
