import os
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
