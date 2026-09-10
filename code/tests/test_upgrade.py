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
    """H1/R2: dialogue lines are placed on a no-truncation timetable."""

    def _fake_clip(self, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"clip")
        return dest

    def test_build_voiceover_timetable_with_real_durations(self):
        from src.tts import build_voiceover
        # 3 shots: line1(4s), empty(4s), line3(4s)。注入实测时长：line1=3s（窗口内），
        # line3=3s（窗口内）。时间表：line1 [0,3]，line3 [8,11]；句间 5s 静音、
        # 尾部 1s 静音补齐到视频总长 12s。
        segs = [("第一句", 4.0, "女主"), ("", 4.0, None), ("第三句", 4.0, "男主")]
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge",
                                     "DRAMAMATRIX_TTS_ENABLED": "1"}, clear=False), \
             patch("src.tts.synthesize_line", side_effect=lambda text, dest, role=None: self._fake_clip(dest)), \
             patch("src.tts._prepare_clip", side_effect=lambda clip, ratio, dest: self._fake_clip(dest)) as prep, \
             patch("src.tts._make_silent_track", side_effect=lambda dur, dest: self._fake_clip(dest)) as silent, \
             patch("src.tts._concat_audio", return_value=True):
            result = build_voiceover(
                segs, Path(tempfile.mkdtemp()) / "audio", probe_duration=lambda p: 3.0
            )
        self.assertTrue(result.voiceover)
        self.assertEqual(result.segments_built, 2)
        lines = {l.index: l for l in result.lines}
        self.assertEqual(lines[0].start, 0.0)
        self.assertEqual(lines[0].end, 3.0)
        self.assertEqual(lines[2].start, 8.0)  # 空镜后从本镜头起点开始
        self.assertEqual(lines[2].end, 11.0)
        self.assertFalse(lines[0].overflow)
        # 句间静音(3→8) + 尾部静音(11→12)
        self.assertEqual(silent.call_count, 2)
        self.assertEqual(prep.call_count, 2)
        # 角色音色：synthesize_line 收到说话人
        self.assertEqual(result.lines[0].role, "女主")

    def test_long_line_speed_capped_and_overflow_recorded(self):
        from src.tts import build_voiceover
        # 第一句语音 10s、镜头窗口 4s：变速上限 1.35 → 有效 ~7.41s 仍溢出窗口，
        # 保留完整语音（不截断）并标记 overflow；第二句 3s 被顺延到第一句之后。
        segs = [("很长的一句台词", 4.0), ("第二句", 4.0)]
        probe = unittest.mock.Mock(side_effect=[10.0, 3.0])
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge",
                                     "DRAMAMATRIX_TTS_ENABLED": "1",
                                     "DRAMAMATRIX_TTS_MAX_SPEED": "1.35"}, clear=False), \
             patch("src.tts.synthesize_line", side_effect=lambda text, dest, role=None: self._fake_clip(dest)), \
             patch("src.tts._prepare_clip", side_effect=lambda clip, ratio, dest: self._fake_clip(dest)) as prep, \
             patch("src.tts._make_silent_track", side_effect=lambda dur, dest: self._fake_clip(dest)), \
             patch("src.tts._concat_audio", return_value=True):
            result = build_voiceover(
                segs, Path(tempfile.mkdtemp()) / "audio", probe_duration=probe
            )
        lines = {l.index: l for l in result.lines}
        self.assertAlmostEqual(lines[0].speed_ratio, 1.35, places=3)
        self.assertAlmostEqual(lines[0].end, 10.0 / 1.35, places=3)
        self.assertTrue(lines[0].overflow, "变速到上限仍放不下应标记溢出")
        # 第二句窗口内原速，但被顺延到第一句之后（而非其镜头起点 4s）
        self.assertAlmostEqual(lines[1].speed_ratio, 1.0, places=3)
        self.assertAlmostEqual(lines[1].start, lines[0].end, places=3)
        self.assertEqual(prep.call_count, 2)
        self.assertAlmostEqual(prep.call_args_list[0][0][1], 1.35, places=3)  # 首句变速
        self.assertAlmostEqual(prep.call_args_list[1][0][1], 1.0, places=3)   # 次句原速

    def test_unmeasured_line_degrades_to_slot_fit_with_flag(self):
        from src.tts import build_voiceover
        segs = [("无法测量时长的一句", 4.0)]
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge",
                                     "DRAMAMATRIX_TTS_ENABLED": "1"}, clear=False), \
             patch("src.tts.synthesize_line", side_effect=lambda text, dest, role=None: self._fake_clip(dest)), \
             patch("src.tts._fit_clip_to_duration", side_effect=lambda clip, dur, dest: self._fake_clip(dest)) as fit, \
             patch("src.tts._concat_audio", return_value=True):
            result = build_voiceover(
                segs, Path(tempfile.mkdtemp()) / "audio", probe_duration=lambda p: None
            )
        line = result.lines[0]
        self.assertTrue(line.synthesized)
        self.assertTrue(line.unmeasured, "无法实测时长必须显式标记，供整集验收复核")
        self.assertEqual(line.end, 4.0)
        fit.assert_called_once()

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