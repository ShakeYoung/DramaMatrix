"""End-to-end smoke test: run the full graph with mocked Agnes/media.

Verifies the market-cycle loop (MAX_CYCLES=2 => agent8 loops back to agent1)
and that the whole pipeline completes without real ffmpeg/Agnes.
"""
import os
import sqlite3
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import src.db
from src.graph import build_drama_matrix_graph
from src.project_state import new_project_state


class FakeClient:
    def preflight(self):
        print("[mock] preflight ok")

    def create_video(self, **kw):
        return {"video_id": "v1", "task_id": "t1"}

    def wait_for_video(self, *a):
        return {"status": "completed", "metadata": {"url": "http://x/v.mp4"}}

    def download_video(self, url, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"video")
        return dest


def fake_concat(inputs, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"master")
    return destination


def fake_cut(source, destination, start_seconds, duration_seconds):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"clip")
    return destination


class EndToEndSmokeTests(unittest.TestCase):
    def test_full_pipeline_with_market_loop(self):
        os.environ["DRAMAMATRIX_MAX_CYCLES"] = "2"
        os.environ["DRAMAMATRIX_RESUME"] = "0"
        os.environ["DRAMAMATRIX_MAX_SCOUT_ATTEMPTS"] = "3"
        # This smoke exercises the market loop, not character enforcement.
        os.environ["DRAMAMATRIX_ALLOW_NO_CHARACTERS"] = "1"
        # Allow Agent4 mock fallback so the text-model-less smoke can produce shots.
        os.environ["DRAMAMATRIX_ALLOW_MOCK_STORYBOARD"] = "1"
        # This smoke exercises the loop, not manual review; disable the review gate.
        os.environ["DRAMAMATRIX_REVIEW_MODE"] = "0"

        tmp = tempfile.mkdtemp()
        os.environ["DRAMAMATRIX_OUTPUT_DIR"] = os.path.join(tmp, "out")
        src.db.DB_PATH = os.path.join(tmp, "smoke.db")
        src.db.init_db()  # create tables on the temp DB

        state = new_project_state("smoke_p")
        g = build_drama_matrix_graph()

        from src.agnes_video import AgnesVideoSettings

        with mock.patch("src.agents.agent5_director.AgnesVideoClient", return_value=FakeClient()), mock.patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=AgnesVideoSettings(api_key="x"),
        ), mock.patch("src.agents.agent6_editor.concat_videos", side_effect=fake_concat), mock.patch(
            "src.agents.agent7_growth.video_duration", return_value=60.0
        ), mock.patch("src.agents.agent7_growth.cut_video", side_effect=fake_cut), mock.patch(
            "src.agents.agent6_editor.mix_audio_into_video",
            side_effect=lambda v, a, d: (d.parent.mkdir(parents=True, exist_ok=True), d.write_bytes(b"x"), d)[2],
        ), mock.patch("src.agents.agent6_editor._apply_voiceover", return_value=None), mock.patch(
            "src.agents.agent6_editor._apply_subtitles", return_value=None
        ):
            nodes = []
            for s in g.stream(state, {"recursion_limit": 80}):
                for node, up in s.items():
                    state.update(up)
                    nodes.append(node)

        # With MAX_CYCLES=2 the graph should visit agent1 twice (two cycles).
        self.assertGreaterEqual(nodes.count("agent1_scout"), 2, f"nodes={nodes}")
        # Final status should indicate a completed cycle.
        self.assertIn(state["system_status"], {"cycle_completed_ready_for_next", "episodes_edited", "growth_assets_ready", "video_assets_downloaded"})


if __name__ == "__main__":
    unittest.main()