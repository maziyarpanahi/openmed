"""Tests for manifest-backed model search helpers."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from openmed import ModelQuery, ModelSearchResult, search_models
from openmed.core import model_search


def _row(
    repo_id: str,
    *,
    family: str = "PII",
    task: str = "token-classification",
    languages: list[str] | None = None,
    tier: str | None = "Small",
    param_count: int | None = 90_000_000,
    formats: list[str] | None = None,
    license: str | None = "apache-2.0",
    benchmark: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "repo_id": repo_id,
        "family": family,
        "task": task,
        "languages": languages if languages is not None else ["fr"],
        "tier": tier,
        "param_count": param_count,
        "architecture": "deberta-v2",
        "base_model": "OpenMed/source",
        "formats": formats if formats is not None else ["mlx-fp", "pytorch"],
        "canonical_labels": ["PERSON"],
        "benchmark": benchmark if benchmark is not None else {"dataset": "synthetic"},
        "arxiv": None,
        "license": license,
        "reproducibility_hash": "sha256:test",
        "released": "2026-01-01",
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "\n".join(json.dumps(row) for row in rows)
    path.write_text(payload + "\n", encoding="utf-8")


def test_search_models_filters_manifest_rows_and_sorts_results(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    _write_manifest(
        manifest,
        [
            _row("OpenMed/zeta-fr-mlx", param_count=90_000_000),
            _row("OpenMed/beta-fr-large", param_count=300_000_000),
            _row("OpenMed/alpha-fr-null-params", param_count=None),
            _row("OpenMed/gamma-en-mlx", languages=["en"]),
            _row("OpenMed/delta-fr-mit", license="mit"),
        ],
    )

    results = search_models(
        task="token-classification",
        language="fr",
        max_params=200_000_000,
        license="apache-2.0",
        manifest_path=manifest,
    )

    assert [result.repo_id for result in results] == [
        "OpenMed/alpha-fr-null-params",
        "OpenMed/zeta-fr-mlx",
    ]
    assert all(isinstance(result, ModelSearchResult) for result in results)
    assert all(result.task == "token-classification" for result in results)
    assert all("fr" in result.languages for result in results)
    assert all(result.license == "apache-2.0" for result in results)


def test_search_models_can_require_known_params(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    _write_manifest(
        manifest,
        [
            _row("OpenMed/known", param_count=90_000_000),
            _row("OpenMed/unknown", param_count=None),
        ],
    )

    default_results = search_models(max_params=100_000_000, manifest_path=manifest)

    assert [result.repo_id for result in default_results] == [
        "OpenMed/known",
        "OpenMed/unknown",
    ]

    results = search_models(
        max_params=100_000_000,
        require_params=True,
        manifest_path=manifest,
    )

    assert [result.repo_id for result in results] == ["OpenMed/known"]


def test_search_models_filters_by_format_and_query(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    _write_manifest(
        manifest,
        [
            _row("OpenMed/privacy-filter-fr-mlx", formats=["mlx-fp", "pytorch"]),
            _row("OpenMed/privacy-filter-fr-coreml", formats=["coreml-int8"]),
            _row("OpenMed/clinical-ner-fr-mlx", family="NER", formats=["mlx-fp"]),
        ],
    )

    results = search_models(format="mlx", query="privacy", manifest_path=manifest)

    assert [result.repo_id for result in results] == ["OpenMed/privacy-filter-fr-mlx"]


def test_search_models_preserves_benchmark_suite_lists(tmp_path: Path) -> None:
    manifest = tmp_path / "models.jsonl"
    benchmark = [
        {
            "suite": "shield",
            "dataset": "openmed-golden-pii",
            "micro_f1": 0.9823,
            "recall": 0.991,
            "leakage": 0.0,
        }
    ]
    _write_manifest(
        manifest,
        [_row("OpenMed/privacy-filter-fr-mlx", benchmark=benchmark)],
    )

    result = search_models(query="privacy", manifest_path=manifest)[0]

    assert result.benchmark == benchmark
    assert result.manifest_row["benchmark"] == benchmark


def test_search_models_uses_local_manifest_loader_without_network(
    monkeypatch,
) -> None:
    calls: list[Path] = []

    def fake_load_manifest_rows(path: Path) -> list[dict[str, Any]]:
        calls.append(path)
        return [_row("OpenMed/local-only")]

    def fail_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("search_models should not open network connections")

    monkeypatch.setattr(model_search, "load_manifest_rows", fake_load_manifest_rows)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    results = model_search.search_models(task="token-classification")

    assert calls == [model_search.MANIFEST_PATH]
    assert [result.repo_id for result in results] == ["OpenMed/local-only"]


def test_search_models_exports_top_level_types() -> None:
    assert ModelQuery().require_params is False
    assert callable(search_models)
