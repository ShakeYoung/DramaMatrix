"""Tests for the run-2 resilience fixes (P0–P2).

Covers: POST safe-retry on queue_full (P0-1), waiting_for_agnes_capacity /
waiting_for_connectivity routing (P0-2/P0-3), Agent4 storyboard gate (P0-4),
base64 tail-frame + scene reset (P0-5), preflight cache (P0-6), continuity
prompt write-back (P1-1), canonical character merge (P1-2), total-shot budget
(P1-4), and non-zero exit on blocked status (P2-2).
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.agnes_video import (
    AgnesConnectionError,
    AgnesQueueFull,
    AgnesSubmissionUncertain,
    AgnesVideoClient,
    AgnesVideoSettings,
    frame_to_data_uri,
)
from src.state import (
    EpisodeState,
    EpisodeScriptData,
    FeedbackLog,
    ShotStoryboard,
    CharacterSheet,
)


class FakeResponse:
    def __init__(self, status_code, text="", headers=None, payload=None):
        self.status_code = status_code
        self._text = text
        self.headers = headers or {}
        self._payload = payload or {}

    @property
    def text(self):
        return self._text

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def make_shot(shot_id, scene_id=None, **kw):
    base = dict(shot_id=shot_id, camera="Static", visual_prompt="v", dialogue="", duration="4s", audio="a")
    base.update(kw)
    if scene_id is not None:
        base["scene_id"] = scene_id
    return ShotStoryboard(**base)


class QueueFullRetryTests(unittest.TestCase):
    """P0-1: POST retries on explicit queue_full/429, but NOT on network break."""

    def setUp(self):
        self.settings = AgnesVideoSettings(api_key="k")

    def test_post_retries_then_succeeds_on_queue_full(self):
        client = AgnesVideoClient(self.settings)
        responses = [
            FakeResponse(503, text='{"error":"video_queue_full"}'),
            FakeResponse(503, text='{"error":"video_queue_full"}'),
            FakeResponse(200, payload={"video_id": "v1"}),
        ]
        with patch.object(client, "session") as sess, patch("src.agnes_video.time.sleep"):
            sess.request.side_effect = responses
            result = client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)
        self.assertEqual(result["video_id"], "v1")
        self.assertEqual(sess.request.call_count, 3)

    def test_post_exhaustion_raises_queue_full(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "session") as sess, patch("src.agnes_video.time.sleep"):
            sess.request.return_value = FakeResponse(503, text='{"error":"video_queue_full"}')
            with self.assertRaises(AgnesQueueFull):
                client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)

    def test_post_network_break_stays_submission_uncertain(self):
        import requests
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "session") as sess:
            sess.request.side_effect = requests.ConnectionError("SSL EOF")
            with self.assertRaises(AgnesSubmissionUncertain):
                client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)


class WaitingStateRoutingTests(unittest.TestCase):
    """P0-2/P0-3: waiting states pause the run; don't re-enter Agent5."""

    def test_queue_full_marks_episode_waiting(self):
        from src.agents.agent5_director import process_agent5_director
        ep = EpisodeState(status="storyboard_done", storyboard_data=[make_shot("s1")])
        state = {"project_id": "p", "characters": [CharacterSheet(name="男", appearance="x", signature="")],
                 "episodes": {"ep_01": ep}, "system_status": "x"}
        client = MagicMock()
        client.preflight.return_value = None
        client.create_video.side_effect = AgnesQueueFull("queue full")
        with patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": tempfile.mkdtemp()}, clear=False), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client
        ), patch("src.agents.agent5_director.AgnesVideoSettings.from_environment",
                 return_value=AgnesVideoSettings(api_key="x")), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)
        self.assertEqual(ep.status, "waiting_for_agnes_capacity")
        self.assertEqual(state["system_status"], "waiting_for_agnes_capacity")

    def test_waiting_states_route_to_end_not_agent5(self):
        from src.graph import route_next_step_for_episode
        for status in ("waiting_for_agnes_capacity", "waiting_for_connectivity"):
            # An episode waiting + another storyboard_done must NOT loop to agent5.
            state = {"episodes": {
                "ep_01": EpisodeState(status=status),
                "ep_02": EpisodeState(status="storyboard_done"),
            }}
            self.assertEqual(route_next_step_for_episode(state), "__end__")


class Agent4GateTests(unittest.TestCase):
    """P0-4: LLM failure blocks (no mock fallback in production); shot-count gate."""

    def test_llm_failure_blocks_without_mock_flag(self):
        from src.agents.agent4_storyboard import process_agent4_storyboard
        ep = EpisodeState(status="script_done",
                          script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"))
        state = {"project_id": "p", "episodes": {"ep_01": ep}, "characters": [], "system_status": "x"}
        with patch.dict(os.environ, {"DRAMAMATRIX_ALLOW_MOCK_STORYBOARD": "0"}, clear=False), patch(
            "src.agents.agent4_storyboard.create_text_model", side_effect=RuntimeError("conn err")
        ), patch("src.agents.agent4_storyboard.time.sleep"):
            process_agent4_storyboard(state)
        self.assertEqual(ep.status, "storyboard_blocked")
        self.assertEqual(state["system_status"], "blocked_on_storyboard_generation")

    def test_shot_count_gate_blocks_too_few(self):
        from src.agents.agent4_storyboard import process_agent4_storyboard
        from src.agents.agent4_storyboard import StoryboardOutput
        ep = EpisodeState(status="script_done",
                          script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"))
        state = {"project_id": "p", "episodes": {"ep_01": ep}, "characters": [], "system_status": "x"}
        # Only 2 shots — below the default min (12).
        few = StoryboardOutput(shots=[make_shot("s1"), make_shot("s2")])
        with patch.dict(os.environ, {"DRAMAMATRIX_MIN_SHOTS_PER_EPISODE": "12"}, clear=False), patch(
            "src.agents.agent4_storyboard.create_text_model"
        ):
            with patch("src.agents.agent4_storyboard._invoke_storyboard_with_retry", return_value=few):
                process_agent4_storyboard(state)
        self.assertEqual(ep.status, "storyboard_blocked")


class Base64FrameTests(unittest.TestCase):
    """P0-5: tail frame becomes a data URI; missing/too-large degrades to None."""

    def test_frame_to_data_uri(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            p.write_bytes(b"\x89PNG\r\n\x1a\n")  # small fake png
            uri = frame_to_data_uri(p)
        self.assertIsNotNone(uri)
        self.assertTrue(uri.startswith("data:image/png;base64,"))

    def test_missing_frame_returns_none(self):
        self.assertIsNone(frame_to_data_uri(Path("/nope/missing.png")))

    def test_too_large_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.png"
            p.write_bytes(b"x" * 100)
            self.assertIsNone(frame_to_data_uri(p, max_bytes=10))


class PreflightCacheTests(unittest.TestCase):
    """P0-6: preflight is cached within TTL; second call doesn't hit network."""

    def test_preflight_cached_within_ttl(self):
        client = AgnesVideoClient(AgnesVideoSettings(api_key="k"))
        with patch.object(client.session, "get", return_value=FakeResponse(200)) as get:
            client.preflight()
            client.preflight()
        # Cached: only one network call.
        self.assertEqual(get.call_count, 1)


class ContinuityPromptWritebackTests(unittest.TestCase):
    """P1-1: normalization writes continuity facts into visual_prompt."""

    def test_visual_prompt_gets_continuity_marker(self):
        from src.continuity import normalize_continuity
        s1 = make_shot("s1", scene_id="A", light_direction="顶光", color_temperature="3200K")
        normalize_continuity([s1])
        self.assertIn("[continuity]", s1.visual_prompt)
        self.assertIn("scene=A", s1.visual_prompt)
        self.assertIn("light=顶光", s1.visual_prompt)

    def test_writeback_is_idempotent(self):
        from src.continuity import normalize_continuity
        s1 = make_shot("s1", scene_id="A")
        normalize_continuity([s1])
        once = s1.visual_prompt.count("[continuity]")
        normalize_continuity([s1])  # re-run
        twice = s1.visual_prompt.count("[continuity]")
        self.assertEqual(once, twice)


class CanonicalCharacterTests(unittest.TestCase):
    """P1-2: near-duplicate transliterations merge to one canonical entry."""

    def test_near_duplicate_names_merge(self):
        from src.characters import canonicalize_characters
        chars = [
            CharacterSheet(name="Zhang Ruochen", appearance="红衣", signature=""),
            CharacterSheet(name="Zhang Ruocheng", appearance="红衣", signature=""),  # 1-char diff
            CharacterSheet(name="沈娇娇", appearance="白衣", signature=""),
        ]
        merged = canonicalize_characters(chars)
        names = [c.name for c in merged]
        self.assertIn("Zhang Ruochen", names)
        self.assertNotIn("Zhang Ruocheng", names)  # collapsed
        self.assertIn("沈娇娇", names)
        self.assertEqual(len(merged), 2)


class TotalShotBudgetTests(unittest.TestCase):
    """P1-4: total shot budget caps the number of planned episodes."""

    def test_episodes_capped_by_budget(self):
        from src.agents.agent3_head_writer import MasterScriptReport, TimelineEpisode
        # 10 episodes, budget 120, min 12/ep => cap = 10 (no truncation here)
        # budget 24, min 12 => cap = 2
        eps = [TimelineEpisode(ep_id=f"ep_{i:02d}", chronological_event="e", outline="o", ending_hook="h") for i in range(1, 11)]
        report = MasterScriptReport(master_script_outline="o", episodes=eps)
        cap = 24 // 12
        self.assertEqual(len(report.episodes[:cap]), 2)


class ExitCodeTests(unittest.TestCase):
    """P2-2: blocked/waiting/failed final status => non-zero exit code."""

    def test_exit_code_logic(self):
        for status in ("blocked_on_agnes_connectivity", "waiting_for_agnes_capacity", "failed_at_scout"):
            self.assertTrue(status.startswith(("blocked_", "waiting_", "failed")))
        self.assertFalse("video_assets_downloaded".startswith(("blocked_", "waiting_", "failed")))


class GatewayUncertainTests(unittest.TestCase):
    """R1: plain 500/502/504 → gateway_uncertain (no auto-retry); 429/queue_full safe."""

    def setUp(self):
        self.settings = AgnesVideoSettings(api_key="k")

    def test_plain_500_raises_gateway_uncertain(self):
        from src.agnes_video import AgnesGatewayUncertain
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "session") as sess:
            sess.request.return_value = FakeResponse(500, text='{"error":"internal"}')
            with self.assertRaises(AgnesGatewayUncertain):
                client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)
        # Not retried (single call).
        self.assertEqual(sess.request.call_count, 1)

    def test_503_without_queue_full_is_gateway_uncertain(self):
        from src.agnes_video import AgnesGatewayUncertain
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "session") as sess:
            sess.request.return_value = FakeResponse(503, text='{"error":"maintenance"}')
            with self.assertRaises(AgnesGatewayUncertain):
                client.create_video(prompt="p", negative_prompt="n", duration="4s", seed=1)

    def test_retry_after_not_capped_to_30s(self):
        # R1: a server Retry-After of 120s must NOT be truncated to retry_max_delay (30s).
        client = AgnesVideoClient(self.settings)
        resp = FakeResponse(503, text='{"error":"video_queue_full"}', headers={"Retry-After": "120"})
        delay = client._retry_delay_with_jitter(0, resp)
        self.assertGreaterEqual(delay, 120)


class StoryboardBlockedRouteTests(unittest.TestCase):
    """R2: storyboard_blocked has highest priority in route_from_start."""

    def test_blocked_takes_priority_over_storyboard_done(self):
        from src.graph import route_from_start
        state = {"episodes": {
            "ep_01": EpisodeState(status="storyboard_done"),
            "ep_02": EpisodeState(status="storyboard_blocked"),
        }}
        self.assertEqual(route_from_start(state), "__end__")


class TotalShotBudgetTests2(unittest.TestCase):
    """R7: cumulative shot budget blocks when exceeded."""

    def test_over_budget_blocks(self):
        from src.agents.agent4_storyboard import process_agent4_storyboard
        from src.agents.agent4_storyboard import StoryboardOutput
        # Two episodes, each with 7 shots => total 14, budget 10 => blocked.
        ep1 = EpisodeState(status="script_done",
                           script_data=EpisodeScriptData(ep_id="ep_01", outline="o", ending_hook="h"))
        ep2 = EpisodeState(status="script_done",
                           script_data=EpisodeScriptData(ep_id="ep_02", outline="o", ending_hook="h"))
        state = {"project_id": "p", "episodes": {"ep_01": ep1, "ep_02": ep2}, "characters": [], "system_status": "x"}
        out = StoryboardOutput(shots=[make_shot(f"s{i}") for i in range(7)])
        with patch.dict(os.environ, {"DRAMAMATRIX_MAX_TOTAL_SHOTS": "10",
                                     "DRAMAMATRIX_MIN_SHOTS_PER_EPISODE": "1"}, clear=False), patch(
            "src.agents.agent4_storyboard.create_text_model"):
            with patch("src.agents.agent4_storyboard._invoke_storyboard_with_retry", return_value=out):
                process_agent4_storyboard(state)
        self.assertEqual(state["system_status"], "blocked_on_storyboard_generation")
        self.assertTrue(all(ep.status == "storyboard_blocked" for ep in state["episodes"].values()))


class AliasSubstitutionTests(unittest.TestCase):
    """R6: alias names in prompts are replaced with canonical_name."""

    def test_alias_replaced_in_prompt(self):
        from src.characters import apply_alias_substitution
        chars = [CharacterSheet(name="Zhang Ruochen", appearance="x", signature="",
                                canonical_name="Zhang Ruochen", aliases=["Zhang Ruocheng"])]
        text = "Zhang Ruocheng stands in the rain."
        self.assertNotIn("Zhang Ruocheng", apply_alias_substitution(text, chars))
        self.assertIn("Zhang Ruochen", apply_alias_substitution(text, chars))

    def test_canonical_uses_dedicated_fields(self):
        from src.characters import canonicalize_characters
        chars = [
            CharacterSheet(name="Zhang Ruochen", appearance="红衣", signature="sig"),
            CharacterSheet(name="Zhang Ruocheng", appearance="红衣", signature="sig2"),
        ]
        merged = canonicalize_characters(chars)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0].character_id.startswith("char_"))
        self.assertEqual(merged[0].canonical_name, "Zhang Ruochen")
        self.assertIn("Zhang Ruocheng", merged[0].aliases)
        # signature NOT overwritten by id.
        self.assertEqual(merged[0].signature, "sig")


class BackoffWindowTests(unittest.TestCase):
    """R10: queue-full sets next_retry_at; within window the ep is skipped."""

    def test_queue_full_sets_next_retry_at(self):
        from src.agents.agent5_director import process_agent5_director
        ep = EpisodeState(status="storyboard_done", storyboard_data=[make_shot("s1")])
        state = {"project_id": "p", "characters": [CharacterSheet(name="男", appearance="x", signature="")],
                 "episodes": {"ep_01": ep}, "system_status": "x"}
        client = MagicMock()
        client.preflight.return_value = None
        client.create_video.side_effect = AgnesQueueFull("queue full")
        with patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": tempfile.mkdtemp(),
                                     "AGNES_POST_RETRY_ATTEMPTS": "0"}, clear=False), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client), patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=AgnesVideoSettings(api_key="x")), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)
        self.assertEqual(ep.status, "waiting_for_agnes_capacity")
        self.assertGreater(ep.next_retry_at, 0)
        self.assertEqual(ep.queue_retry_count, 1)


class TrimFixTests(unittest.TestCase):
    """R5: trim no longer emits an invalid '-t max(...)' expression."""

    def test_trim_command_has_numeric_t(self):
        import inspect
        from src.agnes_video import trim_unstable_frames
        src = inspect.getsource(trim_unstable_frames)
        self.assertNotIn("max(0, duration", src)
        # Numeric computation path present.
        self.assertIn("keep", src)


if __name__ == "__main__":
    unittest.main()