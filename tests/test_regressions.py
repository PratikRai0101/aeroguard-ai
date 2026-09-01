"""Regression tests for the local setup and backend integration seams."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import setup_and_run
from backend import config
from backend import chat
from backend import main
from database import SensorDatabase


class SetupRegressionTests(unittest.TestCase):
    def test_default_model_is_available_in_the_local_setup(self):
        self.assertEqual(config.OLLAMA_MODEL, "qwen3.5:2b")

    def test_unix_ollama_install_uses_a_real_shell_pipeline(self):
        with (
            patch.object(setup_and_run.platform, "system", return_value="Darwin"),
            patch.object(setup_and_run, "run") as run,
        ):
            self.assertTrue(setup_and_run.install_ollama())

        run.assert_called_once_with(
            ["sh", "-c", "curl -fsSL https://ollama.com/install.sh | sh"]
        )

    def test_macos_uses_the_ollama_cli_not_the_gui_app_binary(self):
        gui_binary = Path("/Applications/Ollama.app/Contents/MacOS/ollama")
        cli_binary = Path("/Applications/Ollama.app/Contents/Resources/ollama")

        def exists(path):
            return path in {gui_binary, cli_binary}

        with (
            patch.object(setup_and_run.platform, "system", return_value="Darwin"),
            patch.object(Path, "exists", exists),
            patch.dict(setup_and_run.os.environ, {"PATH": ""}, clear=False),
        ):
            executable = setup_and_run.find_ollama()

        self.assertEqual(executable, str(cli_binary))

    def test_default_database_path_is_absolute_and_project_local(self):
        expected = str(Path(__file__).resolve().parents[1] / "aeroguard.db")
        self.assertEqual(config.DB_FILE, expected)


class BackendRegressionTests(unittest.TestCase):
    def test_chat_disables_qwen_thinking_and_handles_empty_content(self):
        response = unittest.mock.Mock()
        response.json.return_value = {"message": {"content": ""}}
        response.raise_for_status.return_value = None

        with patch.object(chat.requests, "post", return_value=response) as post:
            result = chat.chat_with_slm("Is the air safe?", {}, [])

        payload = post.call_args.kwargs["json"]
        self.assertFalse(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 512)
        self.assertIn("empty response", result["answer"])

    def test_sensor_database_can_return_reading_source(self):
        with tempfile.TemporaryDirectory() as directory:
            db = SensorDatabase(os.path.join(directory, "readings.db"))
            db.add_reading(25, 55, 150, 75, "Moderate", source="mock")

            reading = db.get_recent_readings(1, include_source=True)[0]

        self.assertEqual(reading[-1], "mock")

    def test_stats_identifies_mock_mode_from_source_not_aqi_label(self):
        class FakeDatabase:
            requested_source = False

            def get_recent_readings(self, limit, include_source=False):
                self.requested_source = include_source
                return [("2026-07-31 12:00:00", 25, 55, 150, 75, "Moderate", "mock")]

            def get_statistics(self):
                return {"readings": 1, "alerts": 0}

        db = FakeDatabase()
        with (
            patch.object(main, "db", db),
            patch.object(main, "fetch_outdoor_aqi", return_value={"aqi": None, "pm25": None}),
            patch.object(main, "predictor", None),
        ):
            stats = main.get_latest_stats()

        self.assertTrue(db.requested_source)
        self.assertEqual(stats["mode"], "MOCK")


if __name__ == "__main__":
    unittest.main()
