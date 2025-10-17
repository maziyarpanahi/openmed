import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch, ANY

from openmed.cli import main as cli_main
from openmed.processing.outputs import EntityPrediction, PredictionResult


class CLITestCase(unittest.TestCase):
    def test_analyze_uses_inline_text(self):
        class DummyResult:
            def to_dict(self):
                return {"text": "demo", "entities": []}

        buffer = StringIO()
        with patch("openmed.cli.main_module.analyze_text", return_value=DummyResult()):
            with redirect_stdout(buffer):
                exit_code = cli_main(
                    ["analyze", "--text", "Sample clinical note."]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["text"], "demo")
        self.assertEqual(payload["entities"], [])

    def test_models_list_outputs(self):
        buffer = StringIO()
        with patch(
            "openmed.cli.main_module.list_models",
            return_value=["model-a", "model-b"],
        ) as mock_list_models:
            with redirect_stdout(buffer):
                exit_code = cli_main(["models", "list"])

        self.assertEqual(exit_code, 0)
        lines = buffer.getvalue().strip().splitlines()
        self.assertEqual(lines, ["model-a", "model-b"])
        mock_list_models.assert_called_once_with(
            include_registry=True,
            include_remote=False,
            config=ANY,
        )

    def test_config_show_without_file(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(
                    ["--config-path", str(config_path), "config", "show"]
                )

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["_source"], "defaults (not yet saved)")
        self.assertIn("default_org", payload)

    def test_config_set_persists_value(self):
        with TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.toml"
            exit_code = cli_main(
                [
                    "--config-path",
                    str(config_path),
                    "config",
                    "set",
                    "timeout",
                    "120",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(config_path.exists())
            content = config_path.read_text(encoding="utf-8")
            self.assertIn("timeout = 120", content)

    def test_analyze_json_handles_non_native_numbers(self):
        class FakeFloat:
            def __float__(self):
                return 0.42

        result = PredictionResult(
            text="demo",
            entities=[
                EntityPrediction(
                    text="entity",
                    label="LABEL",
                    confidence=FakeFloat(),
                    start=0,
                    end=6,
                )
            ],
            model_name="test-model",
            timestamp="2025-10-17T00:00:00",
            processing_time=FakeFloat(),
        )

        with patch(
            "openmed.cli.main_module.analyze_text",
            return_value=result,
        ):
            buffer = StringIO()
            with redirect_stdout(buffer):
                exit_code = cli_main(["analyze", "--text", "demo"])

        self.assertEqual(exit_code, 0)
        payload = json.loads(buffer.getvalue())
        self.assertEqual(payload["entities"][0]["confidence"], 0.42)

if __name__ == "__main__":
    unittest.main()
