import os
import unittest
from unittest.mock import patch

from src.text_model import TextModelConfigurationError, TextModelSettings, has_text_model_credentials


class TextModelSettingsTests(unittest.TestCase):
    def test_provider_neutral_variables_take_precedence(self):
        environment = {
            "OPENAI_API_KEY": "openai-key",
            "OPENAI_MODEL": "openai-model",
            "TEXT_MODEL_API_KEY": "provider-key",
            "TEXT_MODEL_NAME": "provider-model",
            "TEXT_MODEL_BASE_URL": "https://gateway.example.com/v1/",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = TextModelSettings.from_environment()
            kwargs = settings.chat_openai_kwargs(temperature=0.2)

        self.assertEqual(kwargs["api_key"], "provider-key")
        self.assertEqual(kwargs["model"], "provider-model")
        self.assertEqual(kwargs["base_url"], "https://gateway.example.com/v1")
        self.assertEqual(kwargs["temperature"], 0.2)

    def test_legacy_openai_variables_remain_supported(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key", "OPENAI_API_BASE": "https://legacy.example/v1", "OPENAI_MODEL": "legacy-model"}, clear=True):
            settings = TextModelSettings.from_environment()

        self.assertEqual(settings.base_url, "https://legacy.example/v1")
        self.assertEqual(settings.model, "legacy-model")

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(TextModelConfigurationError):
            TextModelSettings(api_key="", model="model").validate()
        with self.assertRaises(TextModelConfigurationError):
            TextModelSettings(api_key="key", model="model", base_url="not-a-url").validate()
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(has_text_model_credentials())


if __name__ == "__main__":
    unittest.main()
