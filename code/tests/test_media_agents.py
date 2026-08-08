import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agents.agent4_storyboard import process_agent4_storyboard
from src.agents.agent5_director import process_agent5_director
from src.agents.agent6_editor import process_agent6_editor
from src.agents.agent7_growth import process_agent7_growth
from src.agnes_video import (
    AgnesConnectionError,
    AgnesContentPolicyViolation,
    AgnesSubmissionUncertain,
    AgnesVideoError,
    AgnesVideoSettings,
)
from src.state import EpisodeScriptData, EpisodeState, FeedbackLog, GeneratedVideoAsset, ShotStoryboard


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
    def setUp(self):
        # These tests exercise render mechanics (resume/uncertain/preflight),
        # not character enforcement; bypass the empty-bible gate (P0-B).
        os.environ["DRAMAMATRIX_ALLOW_NO_CHARACTERS"] = "1"

    def tearDown(self):
        os.environ.pop("DRAMAMATRIX_ALLOW_NO_CHARACTERS", None)

    def test_preflight_failure_blocks_all_create_requests(self):
        state = make_state()
        state["episodes"]["ep_02"] = state["episodes"]["ep_01"].model_copy(deep=True)
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        client = MagicMock()
        client.preflight.side_effect = AgnesConnectionError("timed out")

        with patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=settings,
        ), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        client.create_video.assert_not_called()
        self.assertEqual(state["system_status"], "blocked_on_agnes_connectivity")
        # P0-3: a transient connectivity failure must NOT mass-mark unstarted
        # episodes as render_failed; they become waiting_for_connectivity so the
        # next run can resume without losing them.
        self.assertTrue(
            all(ep.status == "waiting_for_connectivity" for ep in state["episodes"].values())
        )

    def test_uncertain_submission_stops_later_episodes(self):
        state = make_state()
        state["episodes"]["ep_02"] = state["episodes"]["ep_01"].model_copy(deep=True)
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        client = MagicMock()
        client.create_video.side_effect = AgnesSubmissionUncertain("timed out")

        with patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=settings,
        ), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        self.assertEqual(client.create_video.call_count, 1)
        self.assertEqual(state["episodes"]["ep_01"].status, "submission_uncertain")
        self.assertEqual(state["episodes"]["ep_02"].status, "storyboard_done")
        self.assertEqual(
            state["system_status"], "blocked_on_agnes_submission_uncertain"
        )

        client.reset_mock()
        process_agent5_director(state)
        client.create_video.assert_not_called()

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

    def test_shot_cap_marks_partial_and_never_routes_to_editor(self):
        from src.graph import route_from_start, route_next_step_for_episode

        state = make_state()
        state["episodes"]["ep_01"].storyboard_data.append(
            ShotStoryboard(
                shot_id="s02",
                camera="Static wide shot",
                visual_prompt="The woman walks away.",
                dialogue="",
                duration="5s",
                audio="soft rain",
            )
        )
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        client = MagicMock()
        client.create_video.return_value = {
            "video_id": "video_partial_1",
            "task_id": "task_partial_1",
        }
        client.wait_for_video.return_value = {
            "status": "completed",
            "metadata": {"url": "https://example.test/partial.mp4"},
        }

        def fake_download(remote_url, destination):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"video")
            return destination

        client.download_video.side_effect = fake_download
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"DRAMAMATRIX_OUTPUT_DIR": directory}, clear=False
        ), patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment",
            return_value=settings,
        ), patch(
            "src.agents.agent5_director.AgnesVideoClient", return_value=client
        ), patch("src.agents.agent5_director.db_save_project_state"):
            process_agent5_director(state)

        episode = state["episodes"]["ep_01"]
        self.assertEqual(episode.status, "render_partial")
        self.assertEqual(episode.rendered_shot_count, 1)
        self.assertEqual(episode.planned_shot_count, 2)
        self.assertEqual(state["system_status"], "waiting_for_full_render")
        self.assertEqual(route_next_step_for_episode(state), "__end__")
        self.assertEqual(route_from_start(state), "agent5_director")

    def test_active_backoff_blocks_submissions_for_other_episodes(self):
        import time

        state = make_state()
        waiting = state["episodes"]["ep_01"]
        waiting.status = "waiting_for_agnes_capacity"
        waiting.next_retry_at = time.time() + 300
        state["episodes"]["ep_02"] = waiting.model_copy(deep=True)
        state["episodes"]["ep_02"].status = "storyboard_done"
        state["episodes"]["ep_02"].next_retry_at = 0

        with patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment"
        ) as settings_loader:
            process_agent5_director(state)

        settings_loader.assert_not_called()
        self.assertEqual(state["episodes"]["ep_02"].status, "storyboard_done")
        self.assertEqual(state["system_status"], "waiting_for_agnes_capacity")

    def test_qc_redraw_archives_rejected_video_and_frames(self):
        from src.agents.agent5_director import _mark_shot_for_redraw

        with tempfile.TemporaryDirectory() as directory:
            shot_directory = Path(directory)
            video = shot_directory / "001_s01.mp4"
            head = shot_directory / "001_s01_head.png"
            tail = shot_directory / "001_s01_tail.png"
            for path in (video, head, tail):
                path.write_bytes(b"rejected")
            episode = make_state()["episodes"]["ep_01"]
            episode.video_assets = [
                GeneratedVideoAsset(
                    shot_id="s01",
                    video_id="v1",
                    task_id="t1",
                    status="completed",
                    prompt="p",
                    local_path=str(video),
                )
            ]

            _mark_shot_for_redraw(
                episode,
                "s01",
                ["brightness jump"],
                AgnesVideoSettings(api_key="test-key", max_revisions=2),
                shot_directory=shot_directory,
                shot_index=1,
            )

            archive = shot_directory / "rejected" / "s01" / "attempt_1"
            self.assertTrue((archive / video.name).is_file())
            self.assertTrue((archive / head.name).is_file())
            self.assertTrue((archive / tail.name).is_file())
            self.assertFalse(episode.video_assets)
            self.assertEqual(episode.status, "storyboard_done")

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

    def test_render_failed_with_assets_resumes_without_clearing_completed_work(self):
        state = make_state()
        episode = state["episodes"]["ep_01"]
        episode.status = "render_failed"
        episode.video_assets = [
            GeneratedVideoAsset(
                shot_id="s01",
                video_id="video_1",
                task_id="task_1",
                status="completed",
                prompt="done prompt",
                remote_url="https://example.test/video.mp4",
                local_path="/tmp/existing.mp4",
            )
        ]
        settings = AgnesVideoSettings(api_key="test-key", max_shots_per_episode=1)
        client = MagicMock()

        with patch("src.agents.agent5_director.Path.is_file", return_value=True), patch(
            "src.agents.agent5_director.AgnesVideoSettings.from_environment", return_value=settings
        ), patch("src.agents.agent5_director.AgnesVideoClient", return_value=client), patch(
            "src.agents.agent5_director.db_save_project_state"
        ):
            process_agent5_director(state)

        self.assertFalse(client.create_video.called)
        self.assertEqual(len(episode.video_assets), 1)
        self.assertEqual(episode.video_assets[0].task_id, "task_1")
        self.assertEqual(episode.status, "video_generated")

    def test_storyboard_recovery_escapes_json_error_feedback(self):
        state = make_state()
        episode = state["episodes"]["ep_01"]
        episode.status = "director_rejected"
        episode.feedback_log = [
            FeedbackLog(
                from_agent="Agent_5_Agnes_Director",
                to_agent="Agent_4_Storyboard",
                reason_code="AGNES_RENDER_FAILED",
                message='镜头 ep_04_s10 的 Agnes 内容策略拒绝：{"error":{"code":"content_policy_violation"}}',
            )
        ]

        with patch("src.agents.agent4_storyboard.create_text_model", side_effect=RuntimeError("no model")), patch.dict(
            os.environ, {"DRAMAMATRIX_ALLOW_MOCK_STORYBOARD": "1"}, clear=False
        ):
            process_agent4_storyboard(state)

        self.assertEqual(episode.status, "storyboard_done")
        self.assertEqual(episode.storyboard_data[0].shot_id, "ep_01_s01")


if __name__ == "__main__":
    unittest.main()
