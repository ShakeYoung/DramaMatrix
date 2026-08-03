import os
import unittest
from unittest.mock import patch

from src.network import configure_proxy_environment
from src.runtime_options import apply_runtime_options, parse_runtime_options


class RuntimeOptionsTests(unittest.TestCase):
    def test_command_line_overrides_apply_only_to_current_environment(self):
        options = parse_runtime_options(
            [
                "--project-id", "project_002",
                "--resume",
                "--agnes-get-retry-attempts", "5",
                "--agnes-retry-delay-seconds", "1.5",
            ]
        )
        with patch.dict(os.environ, {}, clear=True):
            apply_runtime_options(options)
            self.assertEqual(os.environ["DRAMAMATRIX_PROJECT_ID"], "project_002")
            self.assertEqual(os.environ["DRAMAMATRIX_RESUME"], "1")
            self.assertEqual(os.environ["AGNES_GET_RETRY_ATTEMPTS"], "5")
            self.assertEqual(os.environ["AGNES_RETRY_DELAY_SECONDS"], "1.5")

    def test_no_resume_is_supported(self):
        options = parse_runtime_options(["--no-resume"])
        with patch.dict(os.environ, {}, clear=True):
            apply_runtime_options(options)
            self.assertEqual(os.environ["DRAMAMATRIX_RESUME"], "0")

    def test_direct_mode_clears_inherited_proxy_variables(self):
        with patch.dict(
            os.environ,
            {
                "DRAMAMATRIX_PROXY_URL": "direct",
                "HTTPS_PROXY": "http://127.0.0.1:7890",
                "https_proxy": "http://127.0.0.1:7890",
            },
            clear=True,
        ):
            configure_proxy_environment()
            self.assertNotIn("HTTPS_PROXY", os.environ)
            self.assertNotIn("https_proxy", os.environ)

    def test_unavailable_proxy_falls_back_to_direct_connection(self):
        with patch.dict(
            os.environ,
            {"DRAMAMATRIX_PROXY_URL": "http://127.0.0.1:7890"},
            clear=True,
        ), patch("src.network.socket.create_connection", side_effect=OSError("Connection refused")):
            configure_proxy_environment()
            self.assertNotIn("HTTP_PROXY", os.environ)
            self.assertNotIn("HTTPS_PROXY", os.environ)


if __name__ == "__main__":
    unittest.main()
