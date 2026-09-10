"""R2 对白链路测试：说话人字段、角色音色传递、字幕对齐、三态对白检测。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.state import EpisodeScriptData, EpisodeState, ShotStoryboard
from src.tts import TTSLine, TTSResult


def make_episode(status="video_generated", speakers=True):
    shots = [
        ShotStoryboard(
            shot_id="s01", camera="Static", visual_prompt="A woman looks up.",
            dialogue="你以为你赢定了？", speaker="女主" if speakers else None,
            duration="4s", audio="tense",
        ),
        ShotStoryboard(
            shot_id="s02", camera="Close-up", visual_prompt="A man smirks.",
            dialogue="我才是龙王", speaker="男主" if speakers else None,
            duration="5s", audio="climax",
        ),
    ]
    return EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep_01", outline="复仇", ending_hook="摔杯"),
        storyboard_data=shots,
        video_assets=[],
    )


class SpeakerFieldTests(unittest.TestCase):
    def test_speaker_field_optional_with_default(self):
        shot = ShotStoryboard(
            shot_id="s01", camera="Static", visual_prompt="v",
            dialogue="d", duration="4s", audio="a",
        )
        self.assertIsNone(shot.speaker)

    def test_storyboard_prompt_requires_speaker(self):
        from src.agents.agent4_storyboard import STORYBOARD_SYSTEM_PROMPT
        from src.prompt_files import load_prompt

        self.assertIn("speaker", STORYBOARD_SYSTEM_PROMPT)
        external = load_prompt("agent4_storyboard", STORYBOARD_SYSTEM_PROMPT)
        self.assertIn("speaker", external, "外置 prompts/agent4_storyboard.md 须同步 speaker 规则")

    def test_mock_storyboard_shot_has_speaker(self):
        import inspect
        from src.agents import agent4_storyboard

        source = inspect.getsource(agent4_storyboard)
        self.assertIn('speaker="男主"', source)


class VoiceoverRolePassingTests(unittest.TestCase):
    def test_apply_voiceover_passes_speaker_as_role(self):
        from src.agents.agent6_editor import _apply_voiceover

        ep = make_episode()
        captured = {}

        def fake_build(segments, dest, probe_duration=None):
            captured["segments"] = list(segments)
            return TTSResult(audio_path=str(dest / "voiceover.m4a"), voiceover=True,
                             segments_built=2, lines=[])

        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_PROVIDER": "edge"}, clear=False), patch(
            "src.tts.tts_provider", return_value="edge"
        ), patch("src.agents.agent6_editor.build_voiceover", side_effect=fake_build):
            audio, result = _apply_voiceover(ep, Path(tempfile.mkdtemp()))
        self.assertIsNotNone(audio)
        segments = captured["segments"]
        self.assertEqual(len(segments), 2)
        # 三元组：每句携带说话人，供 tts_voice 查 DRAMAMATRIX_TTS_VOICE_MAP。
        self.assertEqual(segments[0][2], "女主")
        self.assertEqual(segments[1][2], "男主")

    def test_dialogue_report_records_timetable(self):
        from src.agents.agent6_editor import _build_dialogue_report

        ep = make_episode()
        tts_result = TTSResult(
            audio_path="/tmp/v.m4a", voiceover=True, segments_built=2,
            lines=[
                TTSLine(index=0, role="女主", text="你以为你赢定了？", synthesized=True,
                        start=0.0, end=3.2, speed_ratio=1.0),
                TTSLine(index=1, role="男主", text="我才是龙王", synthesized=True,
                        start=4.0, end=11.4, speed_ratio=1.35, overflow=True),
            ],
        )
        report = _build_dialogue_report(ep, tts_result, native=False)
        self.assertTrue(report["tts_applied"])
        self.assertEqual(report["lines_total"], 2)
        self.assertEqual(report["lines_synthesized"], 2)
        self.assertEqual(report["overflow_count"], 1)
        self.assertEqual(report["lines"][1]["speed_ratio"], 1.35)


class SubtitleAlignmentTests(unittest.TestCase):
    def test_subtitles_use_speech_timings_when_available(self):
        from src.agents.agent6_editor import _apply_subtitles

        ep = make_episode()
        line_timings = [
            TTSLine(index=0, role=None, text="你以为你赢定了？", synthesized=True,
                    start=0.5, end=3.9),
            TTSLine(index=1, role=None, text="我才是龙王", synthesized=True,
                    start=4.2, end=10.8),
        ]
        captured = {}

        def fake_build_ass(segments, dest):
            captured["segments"] = list(segments)
            dest.write_text("stub", encoding="utf-8")
            return dest

        with patch("src.agents.agent6_editor.build_ass_track", side_effect=fake_build_ass), patch(
            "src.agents.agent6_editor.burn_subtitles",
            side_effect=lambda video, ass, out: (out.write_bytes(b"b"), out)[1],
        ):
            result = _apply_subtitles(ep, Path("v.mp4"), Path(tempfile.mkdtemp()),
                                      line_timings=line_timings)
        self.assertIsNotNone(result)
        segs = captured["segments"]
        # 字幕时间窗=语音真实起止，而非镜头窗口（0/4 起点的整数窗）。
        self.assertAlmostEqual(segs[0][1], 0.5)
        self.assertAlmostEqual(segs[0][2], 3.4, places=3)
        self.assertAlmostEqual(segs[1][1], 4.2)

    def test_subtitles_fall_back_to_shot_windows_without_timings(self):
        from src.agents.agent6_editor import _apply_subtitles

        ep = make_episode()
        captured = {}

        def fake_build_ass(segments, dest):
            captured["segments"] = list(segments)
            dest.write_text("stub", encoding="utf-8")
            return dest

        with patch("src.agents.agent6_editor.build_ass_track", side_effect=fake_build_ass), patch(
            "src.agents.agent6_editor.burn_subtitles",
            side_effect=lambda video, ass, out: (out.write_bytes(b"b"), out)[1],
        ):
            _apply_subtitles(ep, Path("v.mp4"), Path(tempfile.mkdtemp()))
        segs = captured["segments"]
        # 无配音（原生对白）：回退镜头窗口（4s + 5s 累进，无真实时长时用计划值）。
        self.assertAlmostEqual(segs[0][1], 0.0)
        self.assertAlmostEqual(segs[1][1], 4.0)


class DialogueDetectionTests(unittest.TestCase):
    """R2：单音轨不再判为对白——不确定（None）时应用 TTS 而非跳过配音。"""

    def _probe(self, streams_json):
        import src.agnes_video as av

        tmp = Path(tempfile.mkdtemp()) / "clip.mp4"
        tmp.write_bytes(b"f")

        payload = {"streams": streams_json}

        class FakeResult:
            stdout = __import__("json").dumps(payload)

        def fake_run(cmd, **kw):
            return FakeResult()

        with patch.object(av.shutil, "which", return_value="/usr/bin/ffprobe"), patch.object(
            av.subprocess, "run", side_effect=fake_run
        ):
            return av.has_dialogue_stream(tmp)

    def test_single_plain_track_is_uncertain(self):
        verdict = self._probe([{"codec_name": "aac", "profile": "LC"}])
        self.assertIsNone(verdict, "单音轨（可能是纯 BGM）不得判为对白")

    def test_no_audio_streams_is_false(self):
        verdict = self._probe([])
        self.assertFalse(verdict)

    def test_speech_title_is_true(self):
        verdict = self._probe([{"codec_name": "aac", "profile": "LC",
                                "tags": {"title": "对白 narration"}}])
        self.assertTrue(verdict)

    def test_editor_applies_tts_when_uncertain(self):
        """agent6 对 None/False 都走 TTS；仅 True 保留原音轨。"""
        from src.agents.agent6_editor import _apply_voiceover

        for verdict in (None, False):
            with patch("src.agents.agent6_editor.has_dialogue_stream", return_value=verdict), patch(
                "src.agents.agent6_editor._apply_voiceover", return_value=(None, None)
            ) as vo:
                _apply_voiceover(make_episode(), Path(tempfile.mkdtemp()))
            vo.assert_not_called()  # 直接调用 _apply_voiceover 不代表 agent6 流程

    def test_process_agent6_routes_by_tri_state(self):
        from src.agents.agent6_editor import process_agent6_editor

        def run(has_dialogue_value):
            ep = make_episode()
            state = {"project_id": "dlg_p", "episodes": {"ep_01": ep},
                     "system_status": "x", "meta_info": {}, "source_material": {},
                     "master_script_outline": "", "characters": [], "task_cycle": 1,
                     "scout_attempts": 0, "market_feedback": None}
            with patch.dict(os.environ, {"DRAMAMATRIX_EPISODE_REVIEW": "0",
                                         "DRAMAMATRIX_OUTPUT_DIR": tempfile.mkdtemp()}, clear=False), patch(
                "src.agents.agent6_editor.has_dialogue_stream", return_value=has_dialogue_value
            ), patch(
                "src.agents.agent6_editor.concat_videos",
                side_effect=lambda inputs, out: (out.write_bytes(b"m"), out)[1],
            ), patch(
                "src.agents.agent6_editor._apply_voiceover", return_value=(None, None)
            ) as vo, patch("src.agents.agent6_editor._apply_subtitles", return_value=None):
                process_agent6_editor(state)
            return vo.called

        self.assertFalse(run(True), "明确对白轨：保留原音轨，不调用 TTS")
        self.assertTrue(run(None), "不确定（单音轨）：必须应用 TTS")
        self.assertTrue(run(False), "无声：必须应用 TTS")


if __name__ == "__main__":
    unittest.main()
