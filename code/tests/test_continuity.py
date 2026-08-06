"""Tests for the continuity/consistency overhaul (P0–P2).

Covers: storyboard version isolation (P0-A), character bible + render gate
(P0-B), boundary-state normalization (P1-A), conditional create_video inputs
(P1-B), scene grouping + tail-frame chaining + per-scene seed (P1-C), QC hook
(P2-A), and edit-side trim/no-global-crossfade (P2-B).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.state import (
    EpisodeState,
    EpisodeScriptData,
    ShotStoryboard,
    CharacterSheet,
)


def make_shot(shot_id, scene_id=None, **kw):
    base = dict(
        shot_id=shot_id,
        camera="Static",
        visual_prompt="v",
        dialogue="",
        duration="4s",
        audio="a",
    )
    base.update(kw)
    if scene_id is not None:
        base["scene_id"] = scene_id
    return ShotStoryboard(**base)


class VersionIsolationTests(unittest.TestCase):
    """P0-A: storyboard rewrite isolates shot directories."""

    def test_shot_output_dir_is_versioned(self):
        from src.agnes_video import shot_output_dir
        with tempfile.TemporaryDirectory() as d:
            os.environ["DRAMAMATRIX_OUTPUT_DIR"] = d
            v1 = shot_output_dir("proj", "ep_01", 1)
            v2 = shot_output_dir("proj", "ep_01", 2)
            self.assertTrue(v1.name == "v1")
            self.assertTrue(v2.name == "v2")
            self.assertNotEqual(v1, v2)

    def test_purge_removes_other_versions(self):
        from src.agnes_video import shot_output_dir, purge_shot_versions_except
        with tempfile.TemporaryDirectory() as d:
            os.environ["DRAMAMATRIX_OUTPUT_DIR"] = d
            v1 = shot_output_dir("proj", "ep_01", 1)
            v2 = shot_output_dir("proj", "ep_01", 2)
            (v1 / "old.mp4").write_bytes(b"old")
            (v2 / "new.mp4").write_bytes(b"new")
            purge_shot_versions_except("proj", "ep_01", 2)
            self.assertFalse(v1.exists())
            self.assertTrue(v2.exists())


class CharacterBibleTests(unittest.TestCase):
    """P0-B: character bible generation + empty-bible render gate."""

    def test_enhanced_extractor_finds_narrative_characters(self):
        from src.characters import extract_characters_enhanced
        text = "沈娇娇穿着红衣立于城楼。萧寒身披黑甲缓缓走近。"
        chars = extract_characters_enhanced(text)
        names = [c.name for c in chars]
        self.assertIn("沈娇娇", names)
        self.assertIn("萧寒", names)

    def test_build_bible_falls_back_when_no_llm(self):
        from src.characters import build_character_bible
        chars = build_character_bible("复仇", [], "沈娇娇穿着红衣。萧寒身披黑甲。")
        self.assertTrue(any(c.name == "沈娇娇" for c in chars))

    def test_render_blocked_without_character_bible(self):
        from src.agents.agent5_director import process_agent5_director
        state = {
            "project_id": "p",
            "characters": [],  # empty bible
            "episodes": {"ep_01": EpisodeState(status="storyboard_done", storyboard_data=[make_shot("s01")])},
            "system_status": "x",
        }
        with patch.dict(os.environ, {"DRAMAMATRIX_ALLOW_NO_CHARACTERS": "0"}, clear=False):
            process_agent5_director(state)
        self.assertEqual(state["system_status"], "blocked_on_missing_character_bible")

    def test_render_allowed_with_escape_flag(self):
        from src.agents.agent5_director import process_agent5_director
        ep = EpisodeState(status="storyboard_done", storyboard_data=[make_shot("s01")])
        state = {"project_id": "p", "characters": [], "episodes": {"ep_01": ep}, "system_status": "x"}
        with patch.dict(os.environ, {"DRAMAMATRIX_ALLOW_NO_CHARACTERS": "1"}, clear=False), patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            side_effect=Exception("stop after gate"),
        ):
            try:
                process_agent5_director(state)
            except Exception:
                pass
        # Gate passed (it proceeded to settings load, which we broke on purpose).
        self.assertNotEqual(state["system_status"], "blocked_on_missing_character_bible")


class ContinuityNormalizationTests(unittest.TestCase):
    """P1-A: end_state normalization + scene backfill."""

    def test_end_state_normalized_to_next_start(self):
        from src.continuity import normalize_continuity
        s1 = make_shot("s1", scene_id="A", start_pose="站立")
        s2 = make_shot("s2", scene_id="A", start_pose="转身")
        warnings = normalize_continuity([s1, s2])
        self.assertIsNotNone(s1.end_state)
        self.assertEqual(s1.end_state.pose, "转身")
        self.assertEqual(s2.previous_shot_id, "s1")

    def test_missing_scene_id_backfilled_and_warned(self):
        from src.continuity import normalize_continuity
        s1 = make_shot("s1", scene_id="A")
        s2 = make_shot("s2")  # no scene_id
        warnings = normalize_continuity([s1, s2])
        self.assertEqual(s2.scene_id, "A")
        self.assertTrue(any(w.field == "scene_id" for w in warnings))


class ConditionalCreateVideoTests(unittest.TestCase):
    """P1-B: create_video includes image fields only when provided."""

    def setUp(self):
        from src.agnes_video import AgnesVideoSettings
        self.settings = AgnesVideoSettings(api_key="k")
        # _request_json is patched per-test below via object patching.

    def _client(self):
        from src.agnes_video import AgnesVideoClient
        client = AgnesVideoClient(self.settings)
        return client

    def test_image_included_when_provided(self):
        client = self._client()
        with patch.object(client, "_request_json", return_value={"video_id": "v1"}) as req:
            client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1, image_url="http://x/a.png")
        payload = req.call_args.args[2]
        self.assertEqual(payload.get("image_url"), "http://x/a.png")

    def test_image_omitted_when_absent(self):
        client = self._client()
        with patch.object(client, "_request_json", return_value={"video_id": "v1"}) as req:
            client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)
        payload = req.call_args.args[2]
        self.assertNotIn("image", payload)
        self.assertNotIn("image_url", payload)

    def test_extract_last_frame_without_ffmpeg_returns_none(self):
        from src.agnes_video import extract_last_frame, AgnesConfigurationError
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "v.mp4"
            src.write_bytes(b"video")
            with patch("src.agnes_video._require_binary", side_effect=AgnesConfigurationError("no ffmpeg")):
                result = extract_last_frame(src, Path(d) / "f.png")
        self.assertIsNone(result)


class SceneChainingTests(unittest.TestCase):
    """P1-C: scene grouping + reference selection + per-scene seed."""

    def test_group_by_scene(self):
        from src.continuity import group_by_scene
        shots = [make_shot("s1", "A"), make_shot("s2", "A"), make_shot("s3", "B")]
        groups = group_by_scene(shots)
        self.assertEqual([len(g) for g in groups], [2, 1])

    def test_subsequent_shot_uses_previous_last_frame(self):
        from src.continuity import prepare_shot_reference
        ref = prepare_shot_reference(make_shot("s2", "A"), {"A": "scene_a.png"}, "/tmp/prev_tail.png")
        self.assertEqual(ref["image_url"], "/tmp/prev_tail.png")

    def test_first_shot_uses_scene_reference(self):
        from src.continuity import prepare_shot_reference
        ref = prepare_shot_reference(make_shot("s1", "A"), {"A": "scene_a.png"}, None)
        self.assertEqual(ref["image_url"], "scene_a.png")

    def test_same_scene_shares_seed(self):
        from src.continuity import scene_seed
        seed_a1 = scene_seed(20260801, "A", 1)
        seed_a2 = scene_seed(20260801, "A", 2)
        seed_b = scene_seed(20260801, "B", 1)
        self.assertEqual(seed_a1, seed_a2)  # same scene => same seed
        self.assertNotEqual(seed_a1, seed_b)


class QCHookTests(unittest.TestCase):
    """P2-A: continuity QC hook degrades gracefully."""

    def test_qc_passes_when_file_present(self):
        from src.continuity_qc import DefaultChecker
        with tempfile.TemporaryDirectory() as d:
            curr = Path(d) / "v.mp4"
            curr.write_bytes(b"video")
            result = DefaultChecker().check(None, None, curr, make_shot("s1"))
        self.assertTrue(result.passed)

    def test_qc_fails_when_file_missing(self):
        from src.continuity_qc import DefaultChecker
        result = DefaultChecker().check(None, None, Path("/nope/x.mp4"), make_shot("s1"))
        self.assertFalse(result.passed)


class EditTrimTests(unittest.TestCase):
    """P2-B: trim default off; concat does not global-crossfade."""

    def test_trim_returns_input_when_disabled(self):
        from src.agnes_video import trim_unstable_frames
        with tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_TRIM_HEAD": "0", "DRAMAMATRIX_TRIM_TAIL": "0"}, clear=False
        ):
            src = Path(d) / "v.mp4"
            src.write_bytes(b"video")
            out = trim_unstable_frames(src, Path(d))
        self.assertEqual(out, src)


class ChainGenerationIntegrationTests(unittest.TestCase):
    """P1-C integration: with conditional generation on, shot 2 receives the
    previous shot's tail frame as image_url; with it off, it does not."""

    def _state(self):
        ep = EpisodeState(
            status="storyboard_done",
            storyboard_data=[make_shot("s1", "A"), make_shot("s2", "A")],
        )
        return {
            "project_id": "p",
            "characters": [CharacterSheet(name="男主", appearance="红衣", signature="")],
            "episodes": {"ep_01": ep},
            "system_status": "x",
        }

    def _run(self, conditional):
        from src.agents.agent5_director import process_agent5_director
        from src.agnes_video import AgnesVideoSettings
        env = {"DRAMAMATRIX_CONDITIONAL_GENERATION": "1" if conditional else "0"}
        state = self._state()
        client = MagicMock()
        client.preflight.return_value = None
        client.create_video.return_value = {"video_id": "v", "task_id": "t"}
        client.wait_for_video.return_value = {"status": "completed", "metadata": {"url": "http://x/v.mp4"}}

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"video")
            return dest

        client.download_video.side_effect = fake_download
        with patch.dict(os.environ, env, clear=False), tempfile.TemporaryDirectory() as d, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": d}, clear=False
        ), patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=AgnesVideoSettings(api_key="x", max_revisions=5),
        ), patch("src.agents.agent5_director.AgnesVideoClient", return_value=client), patch(
            "src.agents.agent5_director.db_save_project_state"
        ), patch(
            "src.agnes_video.extract_last_frame",
            side_effect=lambda v, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"png"), d)[2],
        ):
            process_agent5_director(state)
        return client

    def test_shot2_inherits_tail_frame_when_conditional(self):
        client = self._run(conditional=True)
        # Two create_video calls; the second should carry image_url (the tail frame).
        self.assertEqual(client.create_video.call_count, 2)
        second_kwargs = client.create_video.call_args_list[1].kwargs
        self.assertIn("image_url", second_kwargs)

    def test_shot2_has_no_image_when_conditional_off(self):
        client = self._run(conditional=False)
        self.assertEqual(client.create_video.call_count, 2)
        for call in client.create_video.call_args_list:
            self.assertNotIn("image_url", call.kwargs)


if __name__ == "__main__":
    unittest.main()