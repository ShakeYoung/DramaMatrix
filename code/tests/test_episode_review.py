"""R2 整集验收测试：合成后暂停、manifest 证据、决定路由、CLI。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langgraph.graph import END

from src.state import EpisodeScriptData, EpisodeState, ShotStoryboard


def make_episode(status="video_generated"):
    return EpisodeState(
        status=status,
        script_data=EpisodeScriptData(ep_id="ep_01", outline="复仇", ending_hook="摔杯"),
        storyboard_data=[
            ShotStoryboard(shot_id="s01", camera="Static", visual_prompt="v",
                           dialogue="你以为你赢定了？", speaker="女主", duration="4s", audio="tense"),
        ],
        video_assets=[],
    )


def make_state(ep=None):
    return {
        "project_id": "epr_p",
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": ep or make_episode()},
        "characters": [],
        "task_cycle": 1,
        "scout_attempts": 0,
        "system_status": "x",
    }


def run_agent6(state):
    from src.agents.agent6_editor import process_agent6_editor

    with patch("src.agents.agent6_editor.has_dialogue_stream", return_value=None), patch(
        "src.agents.agent6_editor._apply_voiceover", return_value=(None, None)
    ), patch(
        "src.agents.agent6_editor.concat_videos",
        side_effect=lambda inputs, out: (out.write_bytes(b"m"), out)[1],
    ), patch("src.agents.agent6_editor._apply_subtitles", return_value=None):
        return process_agent6_editor(state)


class EpisodeReviewGateTests(unittest.TestCase):
    def setUp(self):
        self._out = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            os.environ,
            {"DRAMAMATRIX_OUTPUT_DIR": self._out.name, "DRAMAMATRIX_EPISODE_REVIEW": "1"},
            clear=False,
        )
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._out.cleanup)

    def test_gate_on_pauses_with_manifest(self):
        state = make_state()
        run_agent6(state)
        ep = state["episodes"]["ep_01"]
        self.assertEqual(ep.status, "awaiting_episode_review")
        self.assertEqual(state["system_status"], "waiting_for_episode_review")
        manifest = Path(self._out.name) / "epr_p" / "ep_01" / "review" / "episode_review.json"
        self.assertTrue(manifest.is_file())
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertIn("deliverables", payload)
        self.assertIn("dialogue", payload)

    def test_gate_off_completes_directly(self):
        os.environ["DRAMAMATRIX_EPISODE_REVIEW"] = "0"
        state = make_state()
        run_agent6(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "edit_completed")
        self.assertEqual(state["system_status"], "episodes_edited")


class EpisodeReviewRoutingTests(unittest.TestCase):
    def setUp(self):
        self._out = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": self._out.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._out.cleanup)

    def _decide(self, decision, note=""):
        from src.episode_review import write_episode_decision

        if decision:
            write_episode_decision("epr_p", "ep_01", decision, note)

    def test_undecided_pauses(self):
        from src.graph import route_from_start

        self._decide(None)
        state = make_state(make_episode(status="awaiting_episode_review"))
        self.assertEqual(route_from_start(state), END)

    def test_approved_routes_to_agent7(self):
        from src.graph import route_from_start

        self._decide("approve")
        state = make_state(make_episode(status="awaiting_episode_review"))
        self.assertEqual(route_from_start(state), "agent7_growth")

    def test_rework_routes_back_to_agent6(self):
        from src.graph import route_from_start

        self._decide("rework", note="配音缺失")
        state = make_state(make_episode(status="awaiting_episode_review"))
        self.assertEqual(route_from_start(state), "agent6_editor")

    def test_rework_episode_is_re_edited(self):
        """rework 决定后 resume，agent6 重新合成该集并再次进入验收。"""
        self._decide("rework", note="重合成")
        state = make_state(make_episode(status="awaiting_episode_review"))
        run_agent6(state)
        self.assertEqual(state["episodes"]["ep_01"].status, "awaiting_episode_review")

    def test_approved_episode_is_picked_up_by_agent7(self):
        from src.agents.agent7_growth import _growth_targets

        self._decide("approve")
        state = make_state(make_episode(status="awaiting_episode_review"))
        targets = _growth_targets("epr_p", state)
        self.assertEqual([k for k, _ in targets], ["ep_01"])

    def test_midrun_router_pauses(self):
        from src.graph import route_next_step_for_episode

        state = make_state(make_episode(status="awaiting_episode_review"))
        self.assertEqual(route_next_step_for_episode(state), END)


class EpisodeReviewCLITests(unittest.TestCase):
    def setUp(self):
        self._out = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"DRAMAMATRIX_OUTPUT_DIR": self._out.name}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)
        self.addCleanup(self._out.cleanup)

    def test_cli_writes_decision(self):
        from src import episode_review

        path = Path(self._out.name) / "epr_p" / "ep_01" / "review" / "episode_decision.json"
        code = episode_review.main(["epr_p", "ep_01", "approve", "--note", "对白完整"])
        self.assertEqual(code, 0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["decision"], "approve")
        self.assertEqual(payload["note"], "对白完整")
        self.assertTrue(episode_review.episode_approved("epr_p", "ep_01"))

    def test_invalid_decision_rejected(self):
        from src import episode_review

        with self.assertRaises(SystemExit):
            episode_review.main(["epr_p", "ep_01", "maybe"])


if __name__ == "__main__":
    unittest.main()
