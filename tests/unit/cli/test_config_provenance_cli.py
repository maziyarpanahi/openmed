from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import main_module

SECRET = "planted-token-value-never-print"


def _write_config(tmp_path: Path) -> Path:
    config_file = tmp_path / "openmed.toml"
    config_file.write_text(
        f'device = "gpu"\nhf_token = "{SECRET}"\n',
        encoding="utf-8",
    )
    return config_file


def test_config_provenance_json_reports_sources_without_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENMED_CONFIG", str(_write_config(tmp_path)))
    monkeypatch.setenv("OPENMED_DEVICE", "cpu")

    result = main_module.main(["config", "provenance", "--json"])
    captured = capsys.readouterr()

    assert result == 0
    envelope = json.loads(captured.out)
    assert envelope["ok"] is True
    assert envelope["command"] == "config provenance"

    report = envelope["data"]
    assert report["schema_version"] == 1
    assert report["precedence"] == ["default", "file", "environment", "cli"]
    assert report["keys"]["device"]["source_class"] == "environment"
    assert report["keys"]["device"]["conflict_category"] == "overridden"
    assert report["keys"]["hf_token"]["source_class"] == "file"

    assert SECRET not in captured.out
    assert SECRET not in captured.err


def test_config_provenance_human_output_is_one_line_per_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENMED_CONFIG", str(_write_config(tmp_path)))
    monkeypatch.setenv("OPENMED_DEVICE", "cpu")

    result = main_module.main(["config", "provenance"])
    captured = capsys.readouterr()

    assert result == 0
    assert "device  environment  overridden" in captured.out.splitlines()
    assert SECRET not in captured.out
    assert SECRET not in captured.err


def test_config_provenance_without_config_file_uses_defaults_and_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENMED_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("OPENMED_DEVICE", "cpu")

    result = main_module.main(["config", "provenance", "--json"])
    captured = capsys.readouterr()

    assert result == 0
    report = json.loads(captured.out)["data"]
    assert report["keys"]["device"]["source_class"] == "environment"
