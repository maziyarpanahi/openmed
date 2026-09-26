"""Focused offline acceptance tests for clinical SLM memory preflight."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from openmed.models.clinical_slm_memory import (
    ClinicalSLMMemoryError,
    ClinicalSLMRuntimeProfile,
    MemoryPreflightStatus,
    estimate_clinical_slm_memory,
    load_clinical_slm_artifact_memory,
    preflight_clinical_slm_memory,
)


def _profile(**overrides: int) -> ClinicalSLMRuntimeProfile:
    values: dict[str, Any] = {
        "memory_budget_bytes": 100_000,
        "headroom_bytes": 1_000,
        "context_tokens": 4,
        "batch_size": 2,
        "cache_bytes_per_token": 10,
        "context_bytes_per_token": 20,
        "batch_bytes": 30,
        "runtime_overhead_bytes": 40,
        "name": "synthetic-test",
    }
    values.update(overrides)
    return ClinicalSLMRuntimeProfile(**values)


def _manifest() -> dict[str, object]:
    return {
        "model_id": "synthetic-clinical-package",
        "prompt": "synthetic note text must not enter the report",
        "components": [
            {
                "component": "weights",
                "path": "weights/model.safetensors",
                "size_bytes": 1_000,
            },
            {
                "component": "tokenizer",
                "path": "tokenizer.json",
                "size_bytes": 200,
            },
            {
                "component": "templates",
                "path": "templates/chat.json",
                "size_bytes": 100,
            },
        ],
    }


def test_estimate_is_deterministic_and_covers_each_memory_component() -> None:
    profile = _profile()

    first = estimate_clinical_slm_memory(_manifest(), profile)
    second = estimate_clinical_slm_memory(_manifest(), profile)

    assert first == second
    assert first.weights_bytes == 1_000
    assert first.cache_bytes == 4 * 2 * 10
    assert first.context_bytes == 4 * 2 * 20
    assert first.batch_bytes == 2 * 30
    assert first.runtime_overhead_bytes == 40
    assert first.total_bytes == 1_340
    assert first.remaining_headroom_bytes == 98_660
    assert first.fits is True
    assert first.artifact_component_count == 3


def test_preflight_is_offline_value_free_and_serializes_stably(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("memory preflight attempted network access")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)

    first = preflight_clinical_slm_memory(_manifest(), _profile())
    second = preflight_clinical_slm_memory(_manifest(), _profile())

    assert first.status is MemoryPreflightStatus.ACCEPT
    assert first.ready is True
    assert bool(first) is True
    assert first.reason_codes == ()
    assert first.to_json() == second.to_json()
    rendered = first.to_json()
    assert "synthetic-clinical-package" not in rendered
    assert "model.safetensors" not in rendered
    assert "synthetic note text" not in rendered
    assert first.resource_report["artifact"]["component_count"] == 3
    assert first.resource_report == first.to_dict()


def test_headroom_shortfall_rejects_before_load_with_aggregate_report() -> None:
    profile = _profile(
        memory_budget_bytes=1_000,
        headroom_bytes=100,
        context_tokens=1,
        batch_size=1,
        cache_bytes_per_token=100,
        context_bytes_per_token=100,
        batch_bytes=50,
        runtime_overhead_bytes=0,
    )
    artifact = {
        "model_id": "private-model-id-should-not-escape",
        "path": "/private/sensitive/model",
        "weights_bytes": 700,
        "patient_note": "sensitive clinical text must stay out of reports",
    }

    report = preflight_clinical_slm_memory(artifact, profile)

    assert report.status is MemoryPreflightStatus.REJECT
    assert report.ready is False
    assert bool(report) is False
    assert report.reason_codes == ("headroom_insufficient",)
    assert report.estimate.total_bytes == 950
    assert report.estimate.available_memory_bytes == 1_000
    assert report.estimate.remaining_headroom_bytes == 50
    assert report.estimate.headroom_deficit_bytes == 50
    assert report.estimate.memory_budget_met is True
    rendered = report.to_json()
    assert "private-model-id-should-not-escape" not in rendered
    assert "/private/sensitive/model" not in rendered
    assert "sensitive clinical text" not in rendered


def test_component_manifest_is_metadata_only_and_does_not_require_weight_files(
    tmp_path: Path,
) -> None:
    package = tmp_path / "synthetic-package"
    package.mkdir()
    (package / "clinical-slm-manifest.json").write_text(
        json.dumps(_manifest()),
        encoding="utf-8",
    )

    artifact = load_clinical_slm_artifact_memory(package)

    assert artifact.weights_bytes == 1_000
    assert artifact.artifact_bytes == 1_300
    assert artifact.component_count == 3
    assert str(package) not in repr(artifact)


def test_binary_artifact_path_uses_stat_without_reading_model_bytes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "weights.bin"
    artifact_path.write_bytes(b"synthetic model bytes")

    def fail_read_bytes(_self):
        raise AssertionError("model bytes were opened")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    artifact = load_clinical_slm_artifact_memory(artifact_path)

    assert artifact.weights_bytes == len(b"synthetic model bytes")


def test_runtime_profile_aliases_are_normalized_without_network_or_model_load() -> None:
    profile = ClinicalSLMRuntimeProfile(
        available_memory_bytes=10_000,
        required_headroom_bytes=500,
        max_context_tokens=2,
        batch_size=3,
        kv_cache_bytes_per_token=7,
        activation_bytes_per_token=11,
        batch_overhead_bytes=13,
        baseline_memory_bytes=100,
    )

    estimate = estimate_clinical_slm_memory({"weights_bytes": 1_000}, profile)

    assert profile.available_memory_bytes == 9_900
    assert estimate.cache_bytes == 2 * 3 * 7
    assert estimate.context_bytes == 2 * 3 * 11
    assert estimate.batch_bytes == 3 * 13
    assert estimate.remaining_headroom_bytes == 8_753


def test_budget_and_headroom_failures_have_stable_reason_order() -> None:
    profile = _profile(
        memory_budget_bytes=500,
        headroom_bytes=100,
        context_tokens=1,
        batch_size=1,
        cache_bytes_per_token=100,
        context_bytes_per_token=100,
        batch_bytes=100,
        runtime_overhead_bytes=0,
    )

    report = preflight_clinical_slm_memory({"weights_bytes": 300}, profile)

    assert report.reason_codes == (
        "memory_budget_exceeded",
        "headroom_insufficient",
    )
    assert report.estimate.total_bytes == 600
    assert report.estimate.remaining_headroom_bytes == -100
    assert report.estimate.headroom_deficit_bytes == 200


def test_invalid_metadata_raises_value_free_errors() -> None:
    duplicate = '{"weights_bytes": 1, "weights_bytes": 2}'
    with pytest.raises(ClinicalSLMMemoryError) as duplicate_error:
        load_clinical_slm_artifact_memory(duplicate)
    assert duplicate_error.value.reason_code == "duplicate_field"
    assert "weights_bytes" not in str(duplicate_error.value)

    marker = "synthetic-sensitive-path-value"

    class FailingPath:
        def __fspath__(self) -> str:
            raise RuntimeError(marker)

    with pytest.raises(ClinicalSLMMemoryError) as path_error:
        load_clinical_slm_artifact_memory(FailingPath())  # type: ignore[arg-type]
    assert path_error.value.reason_code == "artifact_unreadable"
    assert marker not in str(path_error.value)


def test_missing_weight_metadata_fails_closed_before_any_resource_estimate() -> None:
    with pytest.raises(ClinicalSLMMemoryError) as raised:
        estimate_clinical_slm_memory(
            {"model_id": "synthetic-only-metadata", "components": []},
            _profile(),
        )

    assert raised.value.reason_code == "weights_missing"
    assert "synthetic-only-metadata" not in str(raised.value)
