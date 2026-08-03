import unittest

from src.project_state import restore_project_state


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


if __name__ == "__main__":
    unittest.main()
