"""Tests for the benchmark CLI command group."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.cli import main_module
from openmed.core.offline import HF_OFFLINE_ENV_VARS, OfflineModeError
from openmed.eval import harness
from openmed.eval.report import BenchmarkReport
from openmed.onnx.inference import OnnxModel


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
            generated_at=kwargs.get("generated_at"),
            metrics={
                "leakage": {"overall": 0.0},
                "exact_span_f1": {"f1": 1.0},
            },
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(harness, "run_benchmark", fake_run_benchmark)

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

    def fake_load_manifest_rows(path=main_module.MANIFEST_PATH):
        assert path == main_module.MANIFEST_PATH
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
            generated_at=kwargs.get("generated_at"),
            metrics={"leakage": {"overall": 0.0}},
            metadata=kwargs["metadata"],
        )

    monkeypatch.setattr(main_module, "load_manifest_rows", fake_load_manifest_rows)
    monkeypatch.setattr(harness, "run_benchmark", fake_run_benchmark)

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
    assert main_module._parse_model_args(["a", "b,c"]) == ["a", "b", "c"]


def test_manifest_shortcut_cannot_be_combined_with_explicit_ids() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        main_module._parse_model_args(["@manifest", "explicit-model"])


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
            "linking",
        ]
    )

    assert result == 1
    assert "not implemented yet" in capsys.readouterr().err


def test_mobile_command_parses_documented_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "mobile",
            "--device",
            "cpu",
            "--tier",
            "base",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert "Mobile benchmark reports written" in captured.out
    assert captured.err == ""
    report_path = (
        tmp_path / "mobile" / "perf" / "synthetic-one-page-note-runner-cpu.json"
    )
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["device"] == "cpu"
    assert payload["tier"] == "base"
    assert payload["document_count"] == 2
    assert "docs_per_second" in payload


def test_mobile_command_default_workload_writes_report_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main_module.main(
        [
            "benchmark",
            "mobile",
            "--device",
            "cpu",
            "--tier",
            "base",
        ]
    )

    assert result == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["model_name"] == "synthetic-one-page-note-runner"
    assert payload["device"] == "cpu"
    assert payload["tier"] == "base"
    assert payload["canonical_tier"] == "Base"
    assert payload["slo_results"]["p95_latency_ms"]["passed"] is True


def test_latency_command_emits_offline_int8_json_and_blocks_sockets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeModel:
        variant = "int8"
        model_path = Path("model_int8.onnx")

        def predict(self, text: str) -> list[object]:
            return []

    socket_guard_seen = False

    def fake_from_pretrained(cls, model_id, **kwargs):
        nonlocal socket_guard_seen
        assert kwargs["variant"] == "int8"
        assert kwargs["local_files_only"] is True
        with pytest.raises(OfflineModeError, match="socket connection"):
            socket.create_connection(("example.invalid", 443))
        socket_guard_seen = True
        return FakeModel()

    monkeypatch.setenv("OPENMED_OFFLINE", "1")
    for name in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(name, "0")
    monkeypatch.setattr(OnnxModel, "from_pretrained", classmethod(fake_from_pretrained))
    output_path = tmp_path / "arm-latency.json"

    result = main_module.main(
        [
            "benchmark",
            "latency",
            "--model",
            str(tmp_path / "cached-model"),
            "--warmup-runs",
            "0",
            "--repeat",
            "1",
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    assert socket_guard_seen is True
    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["benchmark"] == "sms_arm_latency"
    assert payload["offline"] is True
    assert payload["model"]["artifact"] == "model_int8.onnx"
    assert payload["model"]["quantization"] == "int8"
    assert payload["latency_ms"]["p50"] >= 0.0
    assert "throughput_texts_per_second" in payload
    assert "peak_rss_mib" in payload
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
