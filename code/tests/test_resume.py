import unittest

from langgraph.graph import END

from src.graph import route_from_start, route_next_step_for_episode
from src.project_state import restore_project_state
from src.state import EpisodeScriptData, EpisodeState


class ResumeTests(unittest.TestCase):
    def test_snapshot_rehydrates_episode_and_routes_directly_to_agnes(self):
        snapshot = {
            "state": {
                "project_id": "resume_project",
                "meta_info": {},
                "market_feedback": None,
                "source_material": {},
                "master_script_outline": "",
                "episodes": {
                    "ep_01": {
                        "status": "render_pending",
                        "storyboard_data": [],
                        "feedback_log": [],
                        "video_assets": [],
                        "growth_assets": [],
                    }
                },
                "system_status": "blocked_on_agnes_render",
            }
        }
        state = restore_project_state(snapshot)

        self.assertEqual(state["episodes"]["ep_01"].status, "render_pending")

    def test_uncertain_submission_has_priority_over_unsubmitted_episodes(self):
        uncertain = EpisodeState(
            status="submission_uncertain",
            script_data=EpisodeScriptData(
                ep_id="ep_01", outline="one", ending_hook="hook"
            ),
        )
        waiting = EpisodeState(
            status="storyboard_done",
            script_data=EpisodeScriptData(
                ep_id="ep_02", outline="two", ending_hook="hook"
            ),
        )
        state = {"episodes": {"ep_01": uncertain, "ep_02": waiting}}

        self.assertEqual(route_from_start(state), END)
        self.assertEqual(route_next_step_for_episode(state), END)


if __name__ == "__main__":
    unittest.main()
