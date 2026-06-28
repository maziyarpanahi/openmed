"""Tests for the ``openmed models search`` command."""

from __future__ import annotations

from typing import Any

import pytest

from openmed.cli import main_module
from openmed.core.model_search import ModelSearchResult


def test_models_search_prints_table_and_passes_filters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_search_models(**kwargs: Any) -> list[ModelSearchResult]:
        calls.update(kwargs)
        return [
            ModelSearchResult(
                repo_id="OpenMed/unit-fr-mlx",
                family="PII",
                task="token-classification",
                languages=("fr",),
                tier="Small",
                param_count=44_000_000,
                formats=("mlx-fp", "pytorch"),
                license="apache-2.0",
            )
        ]

    monkeypatch.setattr(main_module, "search_models", fake_search_models)

    result = main_module.main(
        [
            "models",
            "search",
            "privacy",
            "--task",
            "token-classification",
            "--language",
            "fr",
            "--tier",
            "Small",
            "--max-params",
            "200000000",
            "--min-params",
            "1000000",
            "--format",
            "mlx",
            "--license",
            "apache-2.0",
            "--require-params",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert "repo_id" in captured.out
    assert "OpenMed/unit-fr-mlx" in captured.out
    assert "mlx-fp,pytorch" in captured.out
    assert captured.err == ""
    assert calls == {
        "task": "token-classification",
        "language": "fr",
        "tier": "Small",
        "max_params": 200_000_000,
        "min_params": 1_000_000,
        "format": "mlx",
        "license": "apache-2.0",
        "query": "privacy",
        "require_params": True,
    }


def test_models_search_no_matches_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "search_models", lambda **_kwargs: [])

    result = main_module.main(["models", "search", "no-such-model"])
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "No models matched" in captured.err
