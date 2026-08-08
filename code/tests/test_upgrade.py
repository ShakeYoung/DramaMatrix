"""Tests for the G-series features: cross-scene concurrency (G1), clip export
controls (G2), Agnes native voice params (G4a), and independent TTS providers
(G4b).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agnes_video import AgnesVideoSettings, AgnesVideoClient
from src.state import EpisodeState, EpisodeScriptData, ShotStoryboard


def make_shot(shot_id, scene_id=None, dialogue="", **kw):
    base = dict(shot_id=shot_id, camera="Static", visual_prompt="v", dialogue=dialogue,
                duration="4s", audio="a")
    base.update(kw)
    if scene_id is not None:
        base["scene_id"] = scene_id
    return ShotStoryboard(**base)


class SceneSegmentsTests(unittest.TestCase):
    """G1: scene_segments splits shots at scene boundaries for concurrency."""

    def test_single_scene_one_segment(self):
        from src.continuity import scene_segments
        shots = [make_shot("s1", "A"), make_shot("s2", "A")]
        segs = scene_segments(shots)
        self.assertEqual(segs, [(0, 2, "A")])

    def test_multi_scene_splits(self):
        from src.continuity import scene_segments
        shots = [make_shot("s1", "A"), make_shot("s2", "A"), make_shot("s3", "B"), make_shot("s4", "B")]
        segs = scene_segments(shots)
        self.assertEqual(segs, [(0, 2, "A"), (2, 4, "B")])

    def test_empty(self):
        from src.continuity import scene_segments
        self.assertEqual(scene_segments([]), [])


class CapacityThrottleTests(unittest.TestCase):
    """G1: throttle degrades on queue_full, recovers on success."""

    def test_acquire_blocks_when_degraded(self):
        from src.render_concurrency import CapacityThrottle
        t = CapacityThrottle()
        self.assertTrue(t.acquire_create())
        t.report_queue_full()
        self.assertFalse(t.acquire_create())

    def test_success_clears_degraded(self):
        from src.render_concurrency import CapacityThrottle
        t = CapacityThrottle()
        t.report_queue_full()
        self.assertFalse(t.acquire_create())
        t.report_success()
        self.assertTrue(t.acquire_create())

    def test_release_decrements_in_flight(self):
        from src.render_concurrency import CapacityThrottle
        t = CapacityThrottle()
        t.acquire_create()
        self.assertEqual(t.in_flight, 1)
        t.release_create()
        self.assertEqual(t.in_flight, 0)

    def test_max_in_flight_default_one(self):
        from src.render_concurrency import max_in_flight
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(max_in_flight(), 1)
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_IN_FLIGHT": "3"}, clear=False):
            self.assertEqual(max_in_flight(), 3)


class ClipExportTests(unittest.TestCase):
    """G2: clip count and climax anchoring."""

    def test_single_clip_when_count_one(self):
        from src.agents.agent7_growth import detect_emotion_segments
        ep = EpisodeState(storyboard_data=[make_shot("s1", "A")])
        with patch.dict(os.environ, {"DRAMAMATRIX_GROWTH_CLIP_COUNT": "1"}, clear=False):
            segs = detect_emotion_segments(60.0, ep)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], "hook")

    def test_climax_anchored_at_last_shot_and_shortened(self):
        from src.agents.agent7_growth import detect_emotion_segments
        ep = EpisodeState(storyboard_data=[make_shot(f"s{i}", "A") for i in range(6)])
        with patch.dict(os.environ, {
            "DRAMAMATRIX_GROWTH_CLIP_COUNT": "2",
            "DRAMAMATRIX_GROWTH_CLIMAX_DURATION": "8",
            "DRAMAMATRIX_GROWTH_CLIP_DURATION": "15",
        }, clear=False):
            segs = detect_emotion_segments(60.0, ep)
        self.assertEqual(len(segs), 2)
        climax_name, climax_start, climax_dur = segs[1]
        self.assertEqual(climax_name, "climax")
        # Climax shortened to 8s (not 15).
        self.assertEqual(climax_dur, 8.0)
        # Anchored near last shot (mean=10s × 5 = 50), not raw tail (52).
        self.assertAlmostEqual(climax_start, 50.0, delta=2.0)
        # Must not overrun.
        self.assertLessEqual(climax_start + climax_dur, 60.0)


class AgnesVoiceParamsTests(unittest.TestCase):
    """G4a: create_video injects voice/narration when provided."""

    def setUp(self):
        self.settings = AgnesVideoSettings(api_key="k")

    def test_narration_included_when_provided(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "_request_json", return_value={"video_id": "v1"}) as req:
            client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1, narration="对白")
        payload = req.call_args.args[2]
        self.assertEqual(payload.get("narration"), "对白")

    def test_voice_fields_omitted_when_absent(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "_request_json", return_value={"video_id": "v1"}) as req:
            client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)
        payload = req.call_args.args[2]
        self.assertNotIn("narration", payload)
        self.assertNotIn("voice_prompt", payload)

    def test_voice_field_name_configurable(self):
        client = AgnesVideoClient(self.settings)
        with patch.dict(os.environ, {"AGNES_NARRATION_FIELD": "audio_prompt"}, clear=False), \
             patch.object(client, "_request_json", return_value={"video_id": "v1"}) as req:
            client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1, narration="对白")
        payload = req.call_args.args[2]
        self.assertIn("audio_prompt", payload)


class TTSProviderTests(unittest.TestCase):
    """G4b: provider config + synthesize_line dispatch."""

    def test_no_provider_returns_none(self):
        from src.tts import synthesize_line, tts_provider
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": ""}, clear=False):
            self.assertEqual(tts_provider(), "")
            with tempfile.TemporaryDirectory() as d:
                self.assertIsNone(synthesize_line("hello", Path(d) / "out.mp3"))

    def test_edge_provider_dispatched(self):
        from src.tts import synthesize_line
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge"}, clear=False), \
             patch("src.tts._synthesize_edge", return_value=Path("/fake/out.mp3")) as edge:
            with tempfile.TemporaryDirectory() as d:
                synthesize_line("hello", Path(d) / "out.mp3")
        edge.assert_called_once()

    def test_openai_provider_dispatched(self):
        from src.tts import synthesize_line
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "openai"}, clear=False), \
             patch("src.tts._synthesize_openai", return_value=Path("/fake/out.mp3")) as oai:
            with tempfile.TemporaryDirectory() as d:
                synthesize_line("hello", Path(d) / "out.mp3")
        oai.assert_called_once()

    def test_agnes_voice_flag_default_off(self):
        from src.tts import agnes_voice_enabled
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(agnes_voice_enabled())
        with patch.dict(os.environ, {"DRAMAMATRIX_AGNES_VOICE": "1"}, clear=False):
            self.assertTrue(agnes_voice_enabled())

    def test_build_voiceover_no_provider_returns_none(self):
        from src.tts import build_voiceover
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": ""}, clear=False):
            result = build_voiceover([("hi", 4.0)], Path(tempfile.mkdtemp()))
        self.assertIsNone(result.audio_path)


class TTSAlignmentTests(unittest.TestCase):
    """H1: each dialogue line is timed to its shot; no -shortest truncation."""

    def test_build_voiceover_pads_silence_per_shot(self):
        from src.tts import build_voiceover
        # 3 shots: line1(4s), empty(4s), line3(4s). Each segment must be fit to
        # its shot duration, not concatenated raw.
        segs = [("第一句", 4.0), ("", 4.0), ("第三句", 4.0)]
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge",
                                     "DRAMAMATRIX_TTS_ENABLED": "1"}, clear=False), \
             patch("src.tts.synthesize_line", side_effect=lambda text, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"mp3"), dest)[2]), \
             patch("src.tts._fit_clip_to_duration", side_effect=lambda clip, dur, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"fit"), dest)[2]) as fit, \
             patch("src.tts._make_silent_track", side_effect=lambda dur, dest: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(b"sil"), dest)[2]) as silent, \
             patch("src.tts._concat_audio", return_value=True):
            result = build_voiceover(segs, Path(tempfile.mkdtemp()) / "audio")
        self.assertTrue(result.voiceover)
        self.assertEqual(result.segments_built, 2)  # 2 non-empty lines
        # _fit_clip called for the 2 dialogue shots; _make_silent_track for the empty one.
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(silent.call_count, 1)

    def test_mix_audio_has_no_shortest(self):
        import inspect
        from src.tts import mix_audio_into_video
        src = inspect.getsource(mix_audio_into_video)
        # The ffmpeg command must not contain -shortest (which truncates video).
        # Check the command-construction line, not comments that mention it.
        command_lines = [l for l in src.splitlines() if "command" in l or '"-map"' in l or "aac" in l]
        joined = "\n".join(command_lines)
        self.assertNotIn('"-shortest"', joined)
        self.assertNotIn("'-shortest'", joined)


class ClipOverlapTests(unittest.TestCase):
    """H3: overlapping hook/climax on short videos drops the duplicate."""

    def test_short_video_drops_overlapping_climax(self):
        from src.agents.agent7_growth import detect_emotion_segments
        # 13s video, default hook=15s (capped to 13). hook covers 0-13, climax
        # would cover ~0-13 too → overlap > 50% → only hook returned.
        ep = EpisodeState(storyboard_data=[make_shot(f"s{i}", "A") for i in range(4)])
        with patch.dict(os.environ, {}, clear=False):
            segs = detect_emotion_segments(13.0, ep)
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0][0], "hook")

    def test_long_video_keeps_both(self):
        from src.agents.agent7_growth import detect_emotion_segments
        ep = EpisodeState(storyboard_data=[make_shot(f"s{i}", "A") for i in range(8)])
        with patch.dict(os.environ, {"DRAMAMATRIX_GROWTH_CLIP_DURATION": "8",
                                     "DRAMAMATRIX_GROWTH_CLIMAX_DURATION": "5"}, clear=False):
            segs = detect_emotion_segments(60.0, ep)
        self.assertEqual(len(segs), 2)


class AudioDetectionTests(unittest.TestCase):
    """H4: has_audio_stream detects audio; missing ffprobe → False (safe)."""

    def test_no_ffprobe_returns_false(self):
        from src.agnes_video import has_audio_stream
        with patch("src.agnes_video.shutil.which", return_value=None):
            with tempfile.TemporaryDirectory() as d:
                self.assertFalse(has_audio_stream(Path(d) / "x.mp4"))


if __name__ == "__main__":
    unittest.main()