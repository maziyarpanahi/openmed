"""CLI JSON output and shell-completion contracts for OM-184."""

from __future__ import annotations

import importlib
import json

import pytest

cli_main = importlib.import_module("openmed.cli.main")


class _SyntheticPrediction:
    def to_dict(self) -> dict[str, object]:
        return {
            "text": "synthetic clinical note",
            "model_name": "synthetic-model",
            "entities": [
                {
                    "text": "synthetic condition",
                    "label": "CONDITION",
                    "start": 0,
                    "end": 19,
                    "confidence": 0.99,
                }
            ],
        }


def test_analyze_json_output_is_one_parseable_document(monkeypatch, capsys):
    def fake_analyze(text, **kwargs):
        assert text == "synthetic clinical note"
        assert kwargs["model_name"] == "synthetic-model"
        return _SyntheticPrediction()

    monkeypatch.setattr(cli_main, "_lazy_api", lambda: (fake_analyze, None, None, None))
    monkeypatch.setattr(cli_main, "_load_and_apply_config", lambda args: None)

    rc = cli_main.main(
        [
            "analyze",
            "--text",
            "synthetic clinical note",
            "--model",
            "synthetic-model",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert set(payload) == {"ok", "command", "data"}
    assert payload["ok"] is True
    assert payload["command"] == "analyze"
    assert payload["data"]["model_name"] == "synthetic-model"
    assert payload["data"]["entities"][0]["label"] == "CONDITION"


@pytest.mark.parametrize("shell", ("bash", "zsh", "fish"))
def test_completion_emits_a_non_empty_script(shell, capsys):
    rc = cli_main.main(["completion", shell])
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.err == ""
    assert captured.out.strip()
    assert "openmed" in captured.out
    assert "analyze" in captured.out
    assert "init" in captured.out
    if shell == "bash":
        assert "complete -F _openmed_completion openmed" in captured.out
        assert "--json" in captured.out
    elif shell == "zsh":
        assert "#compdef openmed" in captured.out
        assert "compdef _openmed openmed" in captured.out
    else:
        assert "complete -c openmed" in captured.out
        assert '-l "json"' in captured.out
