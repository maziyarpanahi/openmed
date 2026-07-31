"""Tests for the ``openmed models size`` command."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from openmed.cli import main_module, typer_app
from openmed.core.hf_hub import CachedModel

FIXTURE_PATH = Path(__file__).with_name("fixtures") / "model_snapshot_sizes.json"


def test_models_size_alias_is_offline_and_network_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_network(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("models size attempted network access without --remote")

    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])
    monkeypatch.setattr(main_module, "get_remote_model_size_mb", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(socket, "socket", fail_network)

    result = main_module.main(["models", "size", "disease_detection_tiny"])
    captured = capsys.readouterr()

    assert result == 0
    assert "download_mb" in captured.out
    assert "disk_mb" in captured.out
    assert "peak_ram_mb" in captured.out
    assert "OpenMed/OpenMed-NER-DiseaseDetect-TinyMed-135M" in captured.out
    assert captured.err == ""


def test_models_size_budget_filters_and_recommends_each_qualifying_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])

    report = main_module.build_models_size_report(budget_mb=100)

    assert report["models"]
    assert all(model["download_mb"] <= 100 for model in report["models"])
    qualifying_tasks = {
        model["task"] for model in report["models"] if model["snapshot_mb"] is not None
    }
    assert {item["task"] for item in report["recommendations"]} == qualifying_tasks
    assert all(
        any(
            model["repo_id"] == item["repo_id"] and model["recommended"]
            for model in report["models"]
        )
        for item in report["recommendations"]
    )


def test_models_size_marks_cached_model_and_zeroes_remaining_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_id = "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M"
    monkeypatch.setattr(
        main_module,
        "list_cached_models",
        lambda: [
            CachedModel(
                repo_id=repo_id,
                size_on_disk=67_905_891,
                last_accessed=None,
                path=tmp_path / "cached-model",
            )
        ],
    )

    report = main_module.build_models_size_report(repo_id)
    model = report["models"][0]

    assert model["cached"] is True
    assert model["download_mb"] == 0.0
    assert model["disk_mb"] == 67.906
    assert model["status"] == "cached — 0 MB to download"


def test_models_size_does_not_treat_empty_cache_entry_as_downloaded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_id = "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M"
    monkeypatch.setattr(
        main_module,
        "list_cached_models",
        lambda: [
            CachedModel(
                repo_id=repo_id,
                size_on_disk=0,
                last_accessed=None,
                path=tmp_path / "empty-cache-entry",
            )
        ],
    )

    model = main_module.build_models_size_report(repo_id)["models"][0]

    assert model["cached"] is False
    assert model["download_mb"] == 67.906
    assert model["status"] == "not cached"


@pytest.mark.parametrize("budget_mb", [-1, float("inf"), float("nan"), True])
def test_models_size_rejects_invalid_programmatic_budget(
    monkeypatch: pytest.MonkeyPatch,
    budget_mb: Any,
) -> None:
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])

    with pytest.raises(ValueError, match="finite non-negative"):
        main_module.build_models_size_report(budget_mb=budget_mb)


def test_models_size_json_is_scriptable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])

    result = main_module.main(
        [
            "models",
            "size",
            "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()

    assert result == 0
    payload = json.loads(captured.out)
    assert payload["remote"] is False
    assert payload["models"][0]["download_mb"] == 67.906
    assert payload["models"][0]["disk_mb"] == 67.906
    assert captured.err == ""


def test_typer_models_size_json_matches_shared_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])
    repo_id = "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M"

    result = CliRunner().invoke(
        typer_app.build_app(),
        ["models", "size", repo_id, "--format", "json"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == main_module.build_models_size_report(repo_id)


def test_models_size_remote_refinement_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_id = "OpenMed/OpenMed-NER-AnatomyDetect-ElectraMed-33M"
    calls: list[str] = []
    monkeypatch.delenv("OPENMED_OFFLINE", raising=False)
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])

    def fake_remote_size(requested_repo_id: str) -> float:
        calls.append(requested_repo_id)
        return 68.25

    monkeypatch.setattr(main_module, "get_remote_model_size_mb", fake_remote_size)

    offline_report = main_module.build_models_size_report(repo_id)
    remote_report = main_module.build_models_size_report(repo_id, remote=True)

    assert calls == [repo_id]
    assert offline_report["models"][0]["source"] == "manifest"
    assert remote_report["models"][0]["source"] == "remote"
    assert remote_report["models"][0]["download_mb"] == 68.25


def test_recorded_snapshot_estimates_are_within_fifteen_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(main_module, "list_cached_models", lambda: [])

    for record in fixture["models"]:
        report = main_module.build_models_size_report(record["repo_id"])
        estimate_mb = report["models"][0]["snapshot_mb"]
        actual_mb = record["actual_snapshot_bytes"] / 1_000_000
        relative_error = abs(estimate_mb - actual_mb) / actual_mb

        assert relative_error <= 0.15, record["repo_id"]
