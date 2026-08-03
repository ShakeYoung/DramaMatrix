import json
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from src.agnes_video import (
    AgnesConfigurationError,
    AgnesTaskFailed,
    AgnesVideoClient,
    AgnesVideoSettings,
    frames_for_duration,
    safe_component,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeBinaryResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        if size == -1:
            result, self.payload = self.payload, b""
            return result
        result, self.payload = self.payload[:size], self.payload[size:]
        return result


class AgnesVideoClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = AgnesVideoSettings(
            api_key="test-key",
            poll_interval_seconds=0,
            max_poll_seconds=2,
        )

    def test_create_video_uses_documented_endpoint_and_valid_frame_count(self):
        client = AgnesVideoClient(self.settings)
        with patch("src.agnes_video.urlopen", return_value=FakeResponse({"video_id": "video_123", "status": "queued"})) as mocked_open:
            response = client.create_video("a scene", "no text", "5s", seed=42)

        self.assertEqual(response["video_id"], "video_123")
        request = mocked_open.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(request.full_url, "https://apihub.agnes-ai.com/v1/videos")
        self.assertEqual(payload["model"], "agnes-video-v2.0")
        self.assertEqual(payload["num_frames"], 121)
        self.assertEqual(payload["frame_rate"], 24)

    def test_wait_for_video_returns_completed_payload(self):
        client = AgnesVideoClient(self.settings)
        responses = [
            {"status": "queued", "progress": 0},
            {"status": "completed", "progress": 100, "metadata": {"url": "https://example.test/video.mp4"}},
        ]
        with patch.object(client, "get_video", side_effect=responses):
            result = client.wait_for_video("video_123")
        self.assertEqual(result["metadata"]["url"], "https://example.test/video.mp4")

    def test_wait_for_video_falls_back_to_task_detail_for_completed_asset_url(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "get_video", return_value={"status": "completed"}), patch.object(
            client,
            "get_task",
            return_value={"status": "completed", "metadata": {"url": "https://example.test/video.mp4"}},
        ):
            result = client.wait_for_video("video_123", "task_123")
        self.assertEqual(result["metadata"]["url"], "https://example.test/video.mp4")

    def test_wait_for_video_accepts_pending_status(self):
        client = AgnesVideoClient(self.settings)
        responses = [
            {"status": "pending"},
            {"status": "completed", "metadata": {"url": "https://example.test/video.mp4"}},
        ]
        with patch.object(client, "get_video", side_effect=responses):
            result = client.wait_for_video("video_123")
        self.assertEqual(result["status"], "completed")

    def test_get_request_retries_network_eof(self):
        client = AgnesVideoClient(self.settings)
        with patch("src.agnes_video.urlopen", side_effect=[URLError("SSL EOF"), FakeResponse({"status": "queued"})]) as mocked_open:
            result = client.get_task("task_123")
        self.assertEqual(result["status"], "queued")
        self.assertEqual(mocked_open.call_count, 2)

    def test_download_retries_network_eof(self):
        client = AgnesVideoClient(self.settings)
        with tempfile.TemporaryDirectory() as directory, patch(
            "src.agnes_video.urlopen",
            side_effect=[URLError("SSL EOF"), FakeBinaryResponse(b"video-bytes")],
        ) as mocked_open:
            destination = client.download_video("https://example.test/video.mp4", __import__("pathlib").Path(directory) / "video.mp4")
        self.assertEqual(mocked_open.call_count, 2)
        self.assertEqual(destination.name, "video.mp4")

    def test_wait_for_video_raises_for_remote_failure(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(client, "get_video", return_value={"status": "failed", "error": {"message": "bad prompt"}}):
            with self.assertRaises(AgnesTaskFailed):
                client.wait_for_video("video_123")

    def test_configuration_and_filename_guards(self):
        with self.assertRaises(AgnesConfigurationError):
            AgnesVideoClient(AgnesVideoSettings(api_key=""))
        self.assertEqual(safe_component("../../ep 01"), "ep_01")
        self.assertEqual(frames_for_duration("5s", 24), 121)
        self.assertEqual(frames_for_duration("4s", 24), 97)


if __name__ == "__main__":
    unittest.main()
