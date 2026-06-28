"""Tests for the ``openmed models card`` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.core.model_card import render_model_card


def _manifest_row() -> dict[str, Any]:
    return {
        "repo_id": "OpenMed/unit-card-model",
        "family": "PII",
        "task": "token-classification",
        "languages": ["en"],
        "tier": "Small",
        "param_count": 44_000_000,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/base-unit-card-model",
        "formats": ["mlx-fp", "pytorch"],
        "canonical_labels": ["PERSON", "DATE", "ID_NUM"],
        "benchmark": {
            "dataset": "synthetic-card-fixture",
            "micro_f1": 0.9823,
            "recall": 0.991,
        },
        "arxiv": "2508.01630",
        "license": "apache-2.0",
        "reproducibility_hash": (
            "sha256:1111111111111111111111111111111111111111111111111111111111111111"
        ),
        "released": "2026-06-14",
    }


def test_models_card_prints_publish_time_markdown(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = _manifest_row()
    monkeypatch.setattr(main_module, "load_manifest_rows", lambda path: [row])

    result = main_module.main(["models", "card", row["repo_id"]])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == render_model_card(row)
    assert captured.err == ""


def test_models_card_writes_output_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    row = _manifest_row()
    output = tmp_path / "cards" / "README.md"
    monkeypatch.setattr(main_module, "load_manifest_rows", lambda path: [row])

    result = main_module.main(
        ["models", "card", row["repo_id"], "--output", str(output)]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert output.read_text(encoding="utf-8") == render_model_card(row)
    assert captured.out == ""
    assert captured.err == ""


def test_models_card_check_exits_zero_when_readme_matches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    row = _manifest_row()
    readme = tmp_path / "README.md"
    readme.write_text(render_model_card(row), encoding="utf-8")
    monkeypatch.setattr(main_module, "load_manifest_rows", lambda path: [row])

    result = main_module.main(
        ["models", "card", row["repo_id"], "--check", str(readme)]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out == ""
    assert captured.err == ""


def test_models_card_check_exits_nonzero_with_unified_diff_on_drift(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    row = _manifest_row()
    readme = tmp_path / "README.md"
    readme.write_text("# stale card\n", encoding="utf-8")
    monkeypatch.setattr(main_module, "load_manifest_rows", lambda path: [row])

    result = main_module.main(
        ["models", "card", row["repo_id"], "--check", str(readme)]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.err == ""
    assert f"--- {readme}" in captured.out
    assert "+++ rendered:OpenMed/unit-card-model" in captured.out
    assert "-# stale card" in captured.out
    assert "+# unit-card-model" in captured.out


def test_models_card_unknown_repo_id_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_module, "load_manifest_rows", lambda path: [_manifest_row()]
    )

    result = main_module.main(["models", "card", "OpenMed/missing-model"])
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "repo_id not found in model manifest: OpenMed/missing-model" in captured.err
