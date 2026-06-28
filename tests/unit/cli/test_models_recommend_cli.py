"""Tests for the ``openmed models recommend`` command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmed.cli import main_module
from openmed.core.model_search import ModelSearchResult, recommend_models


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_models_recommend_prints_ranked_table_and_passes_filters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, Any] = {}

    def fake_recommend_models(**kwargs: Any) -> list[ModelSearchResult]:
        calls.update(kwargs)
        return [
            ModelSearchResult(
                repo_id="OpenMed/small-en",
                family="NER",
                task="token-classification",
                languages=("en",),
                param_count=33_000_000,
            ),
            ModelSearchResult(
                repo_id="OpenMed/large-en",
                family="NER",
                task="token-classification",
                languages=("en",),
                param_count=434_000_000,
            ),
        ]

    monkeypatch.setattr(main_module, "recommend_models", fake_recommend_models)

    result = main_module.main(
        [
            "models",
            "recommend",
            "--task",
            "token-classification",
            "--language",
            "en",
            "--tier",
            "phone",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    assert "Recommended for phone: OpenMed/small-en" in captured.out
    # The ranked order is preserved: the smallest model is listed first.
    assert captured.out.index("OpenMed/small-en") < captured.out.index(
        "OpenMed/large-en"
    )
    assert captured.err == ""
    assert calls == {
        "device_tier": "phone",
        "task": "token-classification",
        "language": "en",
    }


def test_models_recommend_no_fit_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "recommend_models", lambda **_kwargs: [])

    result = main_module.main(
        ["models", "recommend", "--task", "token-classification", "--tier", "phone"]
    )
    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == ""
    assert "No model fits the 'phone' device tier" in captured.err


def test_models_recommend_json_emits_single_document(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        main_module,
        "recommend_models",
        lambda **_kwargs: [
            ModelSearchResult(
                repo_id="OpenMed/small-en",
                family="NER",
                task="token-classification",
                languages=("en",),
                param_count=33_000_000,
            )
        ],
    )

    result = main_module.main(
        [
            "models",
            "recommend",
            "--task",
            "token-classification",
            "--language",
            "en",
            "--tier",
            "phone",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    payload = json.loads(captured.out)  # parses as a single JSON document
    assert payload["tier"] == "phone"
    assert payload["recommended"] == "OpenMed/small-en"
    assert [model["repo_id"] for model in payload["models"]] == ["OpenMed/small-en"]


def test_recommend_models_ranks_by_param_count_without_enrichment(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "models.jsonl",
        [
            {
                "repo_id": "OpenMed/large",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 434_000_000,
            },
            {
                "repo_id": "OpenMed/small",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 33_000_000,
            },
            {
                "repo_id": "OpenMed/mid",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 110_000_000,
            },
            {
                "repo_id": "OpenMed/other-language",
                "task": "token-classification",
                "languages": ["fr"],
                "param_count": 10_000_000,
            },
        ],
    )

    results = recommend_models(
        device_tier="phone",
        task="token-classification",
        language="en",
        manifest_path=manifest,
    )

    # Smallest tier-fitting model first; the French model is filtered out.
    assert [result.repo_id for result in results] == [
        "OpenMed/small",
        "OpenMed/mid",
        "OpenMed/large",
    ]


def test_recommend_models_filters_by_recommended_tier_and_ram(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(
        tmp_path / "models.jsonl",
        [
            {
                "repo_id": "OpenMed/phone-fit",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 50_000_000,
                "recommended_tier": "phone",
                "peak_ram_mb": {"phone": 2_000},
            },
            {
                "repo_id": "OpenMed/server-only",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 20_000_000,
                "recommended_tier": "server",
            },
            {
                "repo_id": "OpenMed/too-much-ram",
                "task": "token-classification",
                "languages": ["en"],
                "param_count": 10_000_000,
                "peak_ram_mb": {"phone": 999_999},
            },
        ],
    )

    results = recommend_models(
        device_tier="phone",
        task="token-classification",
        language="en",
        manifest_path=manifest,
    )

    # ``server-only`` exceeds the tier; ``too-much-ram`` blows the RAM budget.
    assert [result.repo_id for result in results] == ["OpenMed/phone-fit"]


def test_recommend_models_rejects_unknown_device_tier(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        recommend_models(
            device_tier="toaster",
            manifest_path=tmp_path / "missing.jsonl",
        )
