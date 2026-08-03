import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.agent5_director import process_agent5_director
from src.agents.agent6_editor import process_agent6_editor
from src.agents.agent7_growth import process_agent7_growth
from src.agnes_video import AgnesContentPolicyViolation, AgnesVideoError, AgnesVideoSettings
from src.state import EpisodeScriptData, EpisodeState, GeneratedVideoAsset, ShotStoryboard


def make_state():
    shot = ShotStoryboard(
        shot_id="s01",
        camera="Static close-up",
        visual_prompt="A woman looks up in the rain.",
        dialogue="",
        duration="5s",
        audio="soft rain",
    )
    episode = EpisodeState(
        status="storyboard_done",
        script_data=EpisodeScriptData(ep_id="ep_01", outline="test", ending_hook="hook"),
        storyboard_data=[shot],
    )
    return {
        "project_id": "test_project",
        "meta_info": {},
        "market_feedback": None,
        "source_material": {},
        "master_script_outline": "",
        "episodes": {"ep_01": episode},
        "system_status": "starting",
    }


class MediaAgentTests(unittest.TestCase):
    def test_agents_generate_download_edit_and_cut_assets(self):
        state = make_state()
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ", {"DRAMAMATRIX_OUTPUT_DIR": directory}, clear=False):
            client = MagicMock()
            client.create_video.return_value = {"video_id": "video_1", "task_id": "task_1"}
            client.wait_for_video.return_value = {"status": "completed", "metadata": {"url": "https://example.test/video.mp4"}}

            def fake_download(remote_url, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"video")
                return destination

            client.download_video.side_effect = fake_download
            with patch("src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings), patch("src.agents.agent5_director.AgnesVideoClient", return_value=client), patch("src.agents.agent5_director.db_save_project_state"):
                process_agent5_director(state)

            episode = state["episodes"]["ep_01"]
            self.assertEqual(episode.status, "video_generated")
            self.assertEqual(len(episode.video_assets), 1)
            self.assertTrue(Path(episode.video_assets[0].local_path).is_file())

            def fake_concat(inputs, destination):
                destination.write_bytes(b"master")
                return destination

            with patch("src.agents.agent6_editor.concat_videos", side_effect=fake_concat):
                process_agent6_editor(state)
            self.assertEqual(episode.status, "edit_completed")
            self.assertTrue(Path(episode.final_video_path).is_file())

            def fake_cut(source, destination, start_seconds, duration_seconds):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"clip")
                return destination

            with patch("src.agents.agent7_growth.video_duration", return_value=20.0), patch("src.agents.agent7_growth.cut_video", side_effect=fake_cut):
                process_agent7_growth(state)
            self.assertEqual(episode.status, "growth_ready")
            self.assertEqual(len(episode.growth_assets), 2)
            self.assertTrue(all(Path(asset.path).is_file() for asset in episode.growth_assets))

    def test_submitted_task_is_checkpointed_and_resumed_without_recreation(self):
        state = make_state()
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        checkpoints = []

        def record_checkpoint(saved_state):
            assets = saved_state["episodes"]["ep_01"].video_assets
            if assets:
                checkpoints.append(assets[0].task_id)

        first_client = MagicMock()
        first_client.create_video.return_value = {"video_id": "video_1", "task_id": "task_1"}
        first_client.wait_for_video.side_effect = AgnesVideoError("transient EOF")
        with patch("src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=first_client
        ), patch("src.agents.agent5_director.db_save_project_state", side_effect=record_checkpoint):
            process_agent5_director(state)

        episode = state["episodes"]["ep_01"]
        self.assertEqual(episode.status, "render_pending")
        self.assertEqual(episode.video_assets[0].task_id, "task_1")
        self.assertIn("task_1", checkpoints)

        resumed_client = MagicMock()
        resumed_client.wait_for_video.return_value = {
            "status": "completed",
            "metadata": {"url": "https://example.test/video.mp4"},
        }

        def fake_download(remote_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"video")
            return destination

        resumed_client.download_video.side_effect = fake_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            "os.environ", {"DRAMAMATRIX_OUTPUT_DIR": directory}, clear=False
        ), patch("src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=resumed_client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        self.assertFalse(resumed_client.create_video.called)
        resumed_client.wait_for_video.assert_called_once_with("video_1", "task_1")
        self.assertEqual(episode.status, "video_generated")

    def test_content_policy_violation_routes_to_storyboard_revision(self):
        state = make_state()
        episode = state["episodes"]["ep_01"]
        episode.status = "render_pending"
        episode.video_assets = [
            GeneratedVideoAsset(
                shot_id="s01",
                video_id="video_1",
                task_id="task_1",
                status="submitted",
                prompt="blocked prompt",
            )
        ]
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        client = MagicMock()
        client.wait_for_video.side_effect = AgnesContentPolicyViolation(
            "Agnes API HTTP 400: content_policy_violation"
        )

        with patch("src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        self.assertEqual(episode.status, "director_rejected")
        self.assertEqual(state["system_status"], "blocked_on_storyboard_revision")
        self.assertIn("内容策略拒绝", episode.feedback_log[-1].message)


if __name__ == "__main__":
    unittest.main()
