"""Offline acceptance tests for clinical SLM capability probing."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

import openmed.models.clinical_slm_capabilities as capabilities
from openmed.models import probe_clinical_slm_capabilities
from openmed.models.clinical_slm_capabilities import (
    BOUNDED_SUMMARIZATION,
    NLI,
    ClinicalSLMCapabilityError,
    load_clinical_slm_capability_manifest,
)


def _manifest() -> dict[str, object]:
    return {
        "model_id": "synthetic-clinical-package",
        "model_path": "/private/synthetic/clinical-note.txt",
        "supported_tasks": ["clinical-summarization", "nli", "clinical-ner"],
        "context_limits": {
            "max_context_tokens": 4608,
            "max_input_tokens": 4096,
            "max_output_tokens": 512,
        },
        "quantization": {"scheme": "int4", "bits": 4},
        "required_runtime_features": ["tokenizers"],
        "offline": True,
        "human_review_required": True,
        "cloud_fallback": False,
        "prompt": "synthetic prompt that must not enter a report",
    }


def test_probe_is_deterministic_offline_and_value_free(monkeypatch) -> None:
    def fail_network(*_args, **_kwargs):
        raise AssertionError("capability probe attempted network access")

    def fail_optional_import(*_args, **_kwargs):
        raise AssertionError("explicit runtime profile should avoid import probing")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(socket.socket, "connect_ex", fail_network)
    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(capabilities.importlib.util, "find_spec", fail_optional_import)

    first = probe_clinical_slm_capabilities(
        _manifest(),
        available_runtime_features={"tokenizers"},
    )
    second = probe_clinical_slm_capabilities(
        _manifest(),
        available_runtime_features={"tokenizers"},
    )

    assert first.supported is True
    assert first.ready is True
    assert first.capability(BOUNDED_SUMMARIZATION).supported is True
    assert first.capability(NLI).supported is True
    assert first.network_required is False
    assert first.cloud_fallback_allowed is False
    assert first.to_json() == second.to_json()
    assert "synthetic-clinical-package" not in first.to_json()
    assert "clinical-note.txt" not in first.to_json()
    assert "synthetic prompt" not in first.to_json()
    assert "synthetic-clinical-package" not in repr(first)

    payload = first.to_dict()
    assert payload["network"] == {
        "mandatory": False,
        "cloud_fallback": "disabled",
    }
    assert payload["inspected"]["supported_tasks"] == [
        BOUNDED_SUMMARIZATION,
        NLI,
    ]
    assert payload["inspected"]["quantization"] == {
        "declared": True,
        "scheme": "int4",
        "bits": 4,
    }


def test_local_manifest_file_is_metadata_only(tmp_path: Path) -> None:
    path = tmp_path / "clinical-slm-manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    loaded = load_clinical_slm_capability_manifest(path)
    report = probe_clinical_slm_capabilities(
        path,
        available_runtime_features={"tokenizers"},
    )

    assert loaded["model_id"] == "synthetic-clinical-package"
    assert report.supported is True
    assert report.manifest_fingerprint.startswith("sha256:")
    assert str(path) not in report.to_json()


def test_machine_readable_reasons_fail_closed_without_values() -> None:
    manifest = _manifest()
    manifest.update(
        {
            "supported_tasks": ["clinical-summarization"],
            "context_limits": {"max_input_tokens": 128},
            "quantization": {"scheme": "unknown-private-scheme"},
            "required_runtime_features": ["private-runtime-feature"],
            "offline": False,
            "human_review_required": False,
            "cloud_fallback": True,
        }
    )

    report = probe_clinical_slm_capabilities(
        manifest,
        required_capabilities=[NLI],
        available_runtime_features=(),
        min_context_tokens=256,
    )

    assert report.supported is False
    assert report.reason_codes == (
        "offline_required",
        "human_review_required",
        "cloud_fallback_forbidden",
        "task_not_declared",
        "context_limit_too_small",
        "quantization_invalid",
        "runtime_feature_missing",
    )
    assert report.unsupported_reasons[0].to_dict() == {
        "capability": NLI,
        "code": "offline_required",
    }
    rendered = report.to_json()
    assert "unknown-private-scheme" not in rendered
    assert "private-runtime-feature" not in rendered
    assert "synthetic-clinical-package" not in rendered


def test_bounded_summary_requires_an_output_budget() -> None:
    manifest = _manifest()
    manifest["context_limits"] = {"max_context_tokens": 4096}

    report = probe_clinical_slm_capabilities(
        manifest,
        required_capabilities=[BOUNDED_SUMMARIZATION],
        available_runtime_features={"tokenizers"},
    )

    assert report.supported is False
    assert report.capability(BOUNDED_SUMMARIZATION).reason_codes == (
        "output_limit_missing",
    )


def test_invalid_json_and_hostile_path_values_are_value_free(tmp_path: Path) -> None:
    duplicate = '{"supported_tasks": ["nli"], "supported_tasks": ["nli"]}'
    with pytest.raises(ClinicalSLMCapabilityError) as duplicate_error:
        load_clinical_slm_capability_manifest(duplicate)
    assert duplicate_error.value.reason_code == "duplicate_manifest_field"
    assert "nli" not in str(duplicate_error.value)

    marker = "synthetic-sensitive-path-value"

    class FailingPath:
        def __fspath__(self):
            raise RuntimeError(marker)

    with pytest.raises(ClinicalSLMCapabilityError) as path_error:
        load_clinical_slm_capability_manifest(FailingPath())  # type: ignore[arg-type]
    assert path_error.value.reason_code == "manifest_unreadable"
    assert marker not in str(path_error.value)


def test_lazy_model_package_exports_probe() -> None:
    report = probe_clinical_slm_capabilities(
        _manifest(),
        required_tasks=["nli"],
        available_runtime_features={"tokenizers"},
    )
    assert set(report.capabilities) == {NLI}
    assert bool(report) is True
