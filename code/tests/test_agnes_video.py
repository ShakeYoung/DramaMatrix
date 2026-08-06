import json
import tempfile
import unittest
from unittest.mock import patch

import requests

from src.agnes_video import (
    AgnesConfigurationError,
    AgnesConnectionError,
    AgnesContentPolicyViolation,
    AgnesSubmissionUncertain,
    AgnesTaskFailed,
    AgnesVideoClient,
    AgnesVideoSettings,
    frames_for_duration,
    safe_component,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, headers=None, chunks=None):
        self.payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.chunks = chunks

    @property
    def text(self):
        return json.dumps(self.payload)

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def json(self):
        return self.payload

    def raise_for_status(self):
        if not self.ok:
            raise requests.HTTPError(response=self)

    def iter_content(self, chunk_size):
        for chunk in self.chunks if self.chunks is not None else [b"video-bytes"]:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


class AgnesVideoClientTests(unittest.TestCase):
    def setUp(self):
        self.settings = AgnesVideoSettings(
            api_key="test-key",
            poll_interval_seconds=0,
            max_poll_seconds=2,
        )

    def test_create_video_uses_documented_endpoint_and_valid_frame_count(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(
            client.session,
            "request",
            return_value=FakeResponse({"video_id": "video_123", "status": "queued"}),
        ) as mocked_request:
            response = client.create_video("a scene", "no text", "5s", seed=42)

        self.assertEqual(response["video_id"], "video_123")
        self.assertEqual(
            mocked_request.call_args.args,
            ("POST", "https://apihub.agnes-ai.com/v1/videos"),
        )
        payload = mocked_request.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "agnes-video-v2.0")
        self.assertEqual(payload["num_frames"], 121)
        self.assertEqual(payload["frame_rate"], 24)

    def test_preflight_checks_https_without_creating_task(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(
            client.session, "get", return_value=FakeResponse({"object": "list"})
        ) as mocked_get, patch.object(client.session, "request") as mocked_request:
            client.preflight()
        self.assertEqual(mocked_get.call_args.args[0], "https://apihub.agnes-ai.com/v1/models")
        mocked_request.assert_not_called()

    def test_preflight_has_actionable_connection_error(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(
            client.session, "get", side_effect=requests.ConnectTimeout("timed out")
        ):
            with self.assertRaisesRegex(AgnesConnectionError, "AGNES_PROXY_URL"):
                client.preflight()

    def test_create_timeout_is_uncertain_and_never_retried(self):
        client = AgnesVideoClient(self.settings)
        with patch.object(
            client.session, "request", side_effect=requests.ReadTimeout("timed out")
        ) as mocked_request:
            with self.assertRaises(AgnesSubmissionUncertain):
                client.create_video("a scene", "no text", "5s", seed=42)
        self.assertEqual(mocked_request.call_count, 1)

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
        with patch.object(
            client.session,
            "request",
            side_effect=[requests.ConnectionError("SSL EOF"), FakeResponse({"status": "queued"})],
        ) as mocked_request, patch("src.agnes_video.time.sleep"):
            result = client.get_task("task_123")
        self.assertEqual(result["status"], "queued")
        self.assertEqual(mocked_request.call_count, 2)

    def test_content_policy_http_error_is_typed(self):
        client = AgnesVideoClient(self.settings)
        response = FakeResponse(
            {"error": {"code": "content_policy_violation", "message": "blocked"}},
            status_code=400,
        )
        with patch.object(client.session, "request", return_value=response):
            with self.assertRaises(AgnesContentPolicyViolation):
                client.get_task("task_123")

    def test_download_retries_network_eof(self):
        client = AgnesVideoClient(self.settings)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            client.session,
            "get",
            side_effect=[requests.ConnectionError("SSL EOF"), FakeResponse({}, chunks=[b"video-bytes"])],
        ) as mocked_get, patch("src.agnes_video.time.sleep"):
            destination = client.download_video("https://example.test/video.mp4", __import__("pathlib").Path(directory) / "video.mp4")
        self.assertEqual(mocked_get.call_count, 2)
        self.assertEqual(destination.name, "video.mp4")

    def test_download_resumes_a_partial_file_with_range(self):
        client = AgnesVideoClient(self.settings)
        interrupted = FakeResponse(
            {}, chunks=[b"partial", requests.ConnectionError("connection reset")]
        )
        resumed = FakeResponse({}, status_code=206, chunks=[b"-video"])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            client.session, "get", side_effect=[interrupted, resumed]
        ) as mocked_get, patch("src.agnes_video.time.sleep"):
            destination = client.download_video(
                "https://example.test/video.mp4",
                __import__("pathlib").Path(directory) / "video.mp4",
            )
            self.assertEqual(destination.read_bytes(), b"partial-video")
            self.assertEqual(
                mocked_get.call_args_list[1].kwargs["headers"]["Range"], "bytes=7-"
            )

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
