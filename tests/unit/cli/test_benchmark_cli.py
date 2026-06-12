"""Tests for the benchmark CLI command group."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.cli import benchmark as benchmark_cli
from openmed.cli import main_module
from openmed.eval.report import BenchmarkReport


def test_benchmark_pii_golden_writes_report_from_committed_fixtures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, object] = {}

    def fake_run_benchmark(fixtures, **kwargs):
        seen["fixture_count"] = len(fixtures)
        seen["suite"] = kwargs["suite"]
        seen["model_name"] = kwargs["model_name"]
        seen["device"] = kwargs["device"]
        seen["metadata"] = kwargs["metadata"]
        return BenchmarkReport(
            suite=kwargs["suite"],
            model_name=kwargs["model_name"],
            device=kwargs["device"],
            fixture_count=len(fixtures),
            generated_at=kwargs["generated_at"],
            metrics={
                "leakage": {"overall": 0.0},
                "exact_span_f1": {"f1": 1.0},
            },
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(benchmark_cli, "run_benchmark", fake_run_benchmark)

    result = main_module.main(
        [
            "benchmark",
            "pii",
            "--suite",
            "golden",
            "--models",
            "test-model",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert seen["fixture_count"]
    assert seen["suite"] == "golden"
    assert seen["model_name"] == "test-model"
    assert seen["device"] == "cpu"

    json_path = tmp_path / "pii" / "golden" / "test-model-cpu.json"
    markdown_path = tmp_path / "pii" / "golden" / "test-model-cpu.md"
    assert json_path.is_file()
    assert markdown_path.is_file()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "golden"
    assert payload["model_name"] == "test-model"
    assert payload["metadata"]["benchmark_domain"] == "pii"
    assert str(json_path) in capsys.readouterr().out


def test_models_manifest_shortcut_resolves_canonical_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen_models: list[str] = []

    def fake_load_manifest_rows(path=benchmark_cli.MANIFEST_PATH):
        assert path == benchmark_cli.MANIFEST_PATH
        return [
            {"repo_id": "OpenMed/model-a"},
            {"repo_id": "OpenMed/model-b"},
        ]

    def fake_run_benchmark(fixtures, **kwargs):
        seen_models.append(kwargs["model_name"])
        return BenchmarkReport(
            suite=kwargs["suite"],
            model_name=kwargs["model_name"],
            device=kwargs["device"],
            fixture_count=len(fixtures),
            generated_at=kwargs["generated_at"],
            metrics={"leakage": {"overall": 0.0}},
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(benchmark_cli, "load_manifest_rows", fake_load_manifest_rows)
    monkeypatch.setattr(benchmark_cli, "run_benchmark", fake_run_benchmark)

    result = main_module.main(
        [
            "benchmark",
            "pii",
            "--suite",
            "golden",
            "--models",
            "@manifest",
            "--device",
            "cpu",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    assert seen_models == ["OpenMed/model-a", "OpenMed/model-b"]


def test_explicit_model_ids_accept_space_and_comma_separated_values() -> None:
    assert benchmark_cli.resolve_model_ids(["a", "b,c"]) == ["a", "b", "c"]


def test_manifest_shortcut_cannot_be_combined_with_explicit_ids() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        benchmark_cli.resolve_model_ids(["@manifest", "explicit-model"])


def test_clinical_command_parses_documented_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "clinical",
            "--suite",
            "drugprot",
            "--task",
            "ner",
        ]
    )

    assert result == 1
    assert "not implemented yet" in capsys.readouterr().err


def test_mobile_command_parses_documented_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "mobile",
            "--models",
            "OpenMed/mobile-model",
            "--device",
            "mlx",
            "--tier",
            "phone",
        ]
    )

    assert result == 1
    captured = capsys.readouterr()
    assert "not implemented yet" in captured.err
    assert "phone" in captured.err
    assert "OpenMed/mobile-model" in captured.err
