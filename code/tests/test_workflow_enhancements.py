"""Tests for the workflow/feature enhancements (阶段1–4).

Covers: market cycle loop, rejected-book retry routing, Agent1换书 exclusion,
Agent7 emotion segments + 投流 metadata, TTS graceful degradation, ASS subtitle
generation, cost guardrail, render_failed terminal detection, character
extraction/prompt injection, and the expanded EpisodeState round-trip.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from langgraph.graph import END

from src.state import (
    DramaState,
    EpisodeState,
    EpisodeScriptData,
    EvaluationReport,
    FeedbackLog,
    GeneratedVideoAsset,
    GrowthAsset,
    GrowthMeta,
    ShotStoryboard,
    CharacterSheet,
)

project = "tests/"  # placeholder replaced below


def make_episode(status="edit_completed"):
    return EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep_01", outline="女主复仇", ending_hook="男主摔杯"),
        storyboard_data=[
            ShotStoryboard(
                shot_id="s01",
                camera="Static",
                visual_prompt="A woman looks up",
                dialogue="你以为你赢定了？",
                duration="4s",
                audio="tense",
            ),
            ShotStoryboard(
                shot_id="s02",
                camera="Close-up",
                visual_prompt="A man smirks",
                dialogue="我才是龙王",
                duration="5s",
                audio="climax",
            ),
        ],
    )


class RouteTests(unittest.TestCase):
    def test_rejected_book_retries_when_attempts_remaining(self):
        from src.graph import route_after_agent2

        s = {
            "scout_attempts": 1,
            "source_material": {
                "report": EvaluationReport(score=10, hook_analysis="x", is_approved=False, feedback="bad")
            },
        }
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_SCOUT_ATTEMPTS": "3"}, clear=False):
            self.assertEqual(route_after_agent2(s), "agent1_scout")

    def test_rejected_book_ends_when_attempts_exhausted(self):
        from src.graph import route_after_agent2

        s = {
            "scout_attempts": 3,
            "source_material": {
                "report": EvaluationReport(score=10, hook_analysis="x", is_approved=False, feedback="bad")
            },
        }
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_SCOUT_ATTEMPTS": "3"}, clear=False):
            self.assertEqual(route_after_agent2(s), END)

    def test_market_cycle_loops_when_under_limit(self):
        from src.graph import route_after_cycles

        s = {"task_cycle": 1, "episodes": {"ep_01": EpisodeState(status="growth_ready")}}
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_CYCLES": "2"}, clear=False):
            self.assertEqual(route_after_cycles(s), "agent1_scout")

    def test_market_cycle_ends_at_limit(self):
        from src.graph import route_after_cycles

        # Agent8 increments task_cycle before routing; once it exceeds the max
        # (here 3 > 2) the loop stops.
        s = {"task_cycle": 3, "episodes": {"ep_01": EpisodeState(status="growth_ready")}}
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_CYCLES": "2"}, clear=False):
            self.assertEqual(route_after_cycles(s), END)

    def test_terminal_render_failed_not_routed_to_agent5_on_start(self):
        from src.graph import route_from_start

        terminal = EpisodeState(
            status="render_failed",
            feedback_log=[
                FeedbackLog(from_agent="a", to_agent="b", reason_code="R", message="m"),
                FeedbackLog(from_agent="a", to_agent="b", reason_code="R", message="m"),
            ],
        )
        # storyboard_data empty → not retryable
        s = {"episodes": {"ep_01": terminal}}
        with patch.dict(os.environ, {"AGNES_MAX_REVISIONS": "2"}, clear=False):
            self.assertNotEqual(route_from_start(s), "agent5_director")

    def test_retryable_render_failed_routed_to_agent5(self):
        from src.graph import route_from_start, _render_failed_retryable

        retryable = EpisodeState(status="render_failed", storyboard_data=[make_episode().storyboard_data[0]])
        self.assertTrue(_render_failed_retryable(retryable))
        s = {"episodes": {"ep_01": retryable}}
        with patch.dict(os.environ, {"AGNES_MAX_REVISIONS": "2"}, clear=False):
            self.assertEqual(route_from_start(s), "agent5_director")

    def test_render_failed_exhausted_revisions_not_retryable(self):
        from src.graph import _render_failed_retryable

        ep = EpisodeState(
            status="render_failed",
            storyboard_data=[make_episode().storyboard_data[0]],
            feedback_log=[
                FeedbackLog(from_agent="a", to_agent="b", reason_code="R", message="m"),
                FeedbackLog(from_agent="a", to_agent="b", reason_code="R", message="m"),
            ],
        )
        with patch.dict(os.environ, {"AGNES_MAX_REVISIONS": "2"}, clear=False):
            self.assertFalse(_render_failed_retryable(ep))


class Agent1ScoutTests(unittest.TestCase):
    def test_scraper_excludes_attempted_titles_and_rotates(self):
        from src.agents.agent1_scout import scrape_biquge_novel

        with patch("time.sleep"):
            first = scrape_biquge_novel(exclude_titles=set())
            second = scrape_biquge_novel(exclude_titles={first["title"]})
            third = scrape_biquge_novel(exclude_titles={first["title"], second["title"]})
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNotNone(third)
        titles = {first["title"], second["title"], third["title"]}
        self.assertEqual(len(titles), 3, "换书应每次取不同书目")
        # exhaustion → None
        with patch("time.sleep"):
            none = scrape_biquge_novel(exclude_titles=titles)
        self.assertIsNone(none)

    def test_agent1_records_scout_excluded_in_state(self):
        from src.agents.agent1_scout import process_agent1_scout

        state = {
            "project_id": "p",
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
        with patch("src.agents.agent1_scout.db_get_unprocessed_novel", return_value=None), patch(
            "src.agents.agent1_scout.db_insert_novel", return_value=True
        ), patch("src.agents.agent1_scout.db_mark_novel_processed"), patch("time.sleep"):
            process_agent1_scout(state)
        self.assertIn(state["meta_info"]["source_title"], state["meta_info"]["scout_excluded"])
        self.assertEqual(state["scout_attempts"], 1)


class Agent7Tests(unittest.TestCase):
    def test_emotion_segments_and_meta(self):
        from src.agents.agent7_growth import detect_emotion_segments, _asset_meta, build_growth_meta

        ep = make_episode()
        segs = detect_emotion_segments(60.0, ep)
        names = [n for n, _, _ in segs]
        self.assertEqual(names, ["hook", "climax"])

        meta = build_growth_meta(ep, genre_tags=["女频", "复仇"])
        self.assertEqual(meta.tags, ["女频", "复仇"])
        hook_headline, hook_desc, hook_tags = _asset_meta("hook", meta, 60.0)
        self.assertIn("开场", hook_headline)
        self.assertIn("开场", hook_tags)

    def test_segments_honor_short_total(self):
        from src.agents.agent7_growth import detect_emotion_segments

        ep = make_episode()
        with patch.dict(os.environ, {"DRAMAMATRIX_GROWTH_CLIP_DURATION": "8"}, clear=False):
            segs = detect_emotion_segments(20.0, ep)
        duration = segs[0][2]
        self.assertEqual(duration, 8.0)


class TTSTests(unittest.TestCase):
    def test_disabled_returns_no_track(self):
        from src.tts import build_voiceover

        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_ENABLED": "0"}, clear=False):
            result = build_voiceover([("hi", 4.0)], Path(tempfile.mkdtemp()))
        self.assertIsNone(result.audio_path)

    def test_no_voiceover_when_no_tts_url(self):
        from src.agents.agent6_editor import _apply_voiceover

        ep = make_episode()
        with patch.dict(os.environ, {"DRAMAMATRIX_TTS_URL": "", "DRAMAMATRIX_TTS_ENABLED": "1"}, clear=False):
            self.assertIsNone(_apply_voiceover(ep, Path(tempfile.mkdtemp())))


class SubtitleTests(unittest.TestCase):
    def test_ass_track_built_with_style_and_blank_skipped(self):
        from src.subtitles import build_ass_track, _ass_timestamp

        segments = [("你以为你赢定了？", 0.0, 4.0), ("", 5.0, 4.0), ("我才是龙王", 10.0, 5.0)]
        destination = Path(tempfile.mkdtemp()) / "subs.ass"
        path = build_ass_track(segments, destination)
        content = path.read_text(encoding="utf-8")
        self.assertIn("Style: Default", content)
        self.assertEqual(content.count("Dialogue:"), 2)
        self.assertEqual(_ass_timestamp(65.25), "0:01:05.25")


class CostTrackerTests(unittest.TestCase):
    def test_budget_exhausted_breaks_and_report_written(self):
        from src.cost_tracker import CostTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = CostTracker(max_creates=2)
            self.assertFalse(tracker.budget_exhausted())
            tracker.record_create(project_id="p", ep_key="ep_01", shot_id="s01", task_id="t1", frames=49, width=720, height=1280)
            tracker.record_create(project_id="p", ep_key="ep_01", shot_id="s02", task_id="t2", frames=49, width=720, height=1280)
            self.assertTrue(tracker.budget_exhausted())
            report = tracker.write_report(Path(directory))
            self.assertTrue(report.is_file())
            self.assertIn("task_id", report.read_text(encoding="utf-8"))


class CharacterTests(unittest.TestCase):
    def test_extract_characters_cleans_noise(self):
        from src.characters import extract_characters, render_character_block

        chars = extract_characters("男主张若尘说：复仇。女主说：为何。反派冷冷道：完了。")
        names = [c.name for c in chars]
        self.assertIn("男主张若尘", names)
        self.assertIn("反派", names)
        self.assertNotIn("反派冷冷", names)
        block = render_character_block(chars)
        self.assertIn("角色一致性参考", block)

    def test_agnes_prompt_injects_consistency(self):
        from src.agents.agent5_director import build_agnes_prompt
        from src.characters import extract_characters, render_character_block

        chars = extract_characters("男主张若尘说：复仇。")
        shot = ShotStoryboard(shot_id="s1", camera="Static", visual_prompt="closeup", dialogue="", duration="4s", audio="a")
        prompt = build_agnes_prompt(shot, render_character_block(chars))
        self.assertIn("角色一致性参考", prompt)


class EpisodeStateRoundTripTests(unittest.TestCase):
    def test_new_fields_round_trip(self):
        ep = make_episode()
        ep.audio_track = "voice.m4a"
        ep.subtitle_track = "subs.ass"
        ep.growth_meta = GrowthMeta(title="t", description="d", tags=["a"], cover_prompt="c")
        ep.characters = [CharacterSheet(name="男主", appearance="红衣", signature="", role="")]
        ep.growth_assets = [
            GrowthAsset(name="hook", path="p.mp4", start_seconds=0, duration_seconds=15.0, headline="h", description="d", tags=["t"])
        ]
        data = ep.model_dump(mode="json")
        restored = EpisodeState.model_validate(data)
        self.assertEqual(restored.audio_track, "voice.m4a")
        self.assertEqual(restored.subtitle_track, "subs.ass")
        self.assertEqual(restored.growth_meta.tags, ["a"])
        self.assertEqual(restored.characters[0].name, "男主")
        self.assertEqual(restored.growth_assets[0].headline, "h")


if __name__ == "__main__":
    unittest.main()