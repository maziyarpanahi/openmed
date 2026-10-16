"""Offline acceptance tests for clinical SLM capability probing."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
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


def test_unknown_runtime_is_not_reported_as_supported_or_echoed():
    manifest = _manifest()
    manifest["required_runtime_features"] = ["synthetic_private_runtime"]
    report = probe_clinical_slm_capabilities(
        manifest, available_runtime_features=["synthetic_private_runtime"]
    )
    assert not report.supported
    assert "synthetic_private_runtime" not in report.to_json()


@pytest.mark.parametrize("field", ["cloud_fallback", "network_required"])
@pytest.mark.parametrize("value", [0, None, "", []])
def test_falsey_non_boolean_policy_is_not_accepted(field, value):
    manifest = _manifest()
    manifest[field] = value
    report = probe_clinical_slm_capabilities(
        manifest, available_runtime_features=["tokenizers"]
    )
    assert not report.supported


def test_conflicting_context_aliases_are_rejected():
    manifest = _manifest()
    manifest["context_limits"]["input_tokens"] = 1
    with pytest.raises(ClinicalSLMCapabilityError):
        probe_clinical_slm_capabilities(
            manifest, available_runtime_features=["tokenizers"]
        )


def test_impossible_context_budget_is_unsupported():
    manifest = _manifest()
    manifest["context_limits"]["max_context_tokens"] = 4096
    report = probe_clinical_slm_capabilities(
        manifest, available_runtime_features=["tokenizers"]
    )
    assert not report.supported


def test_conflicting_quantization_aliases_are_rejected():
    manifest = _manifest()
    manifest["quantization"]["method"] = "fp32"
    with pytest.raises(ClinicalSLMCapabilityError):
        probe_clinical_slm_capabilities(
            manifest, available_runtime_features=["tokenizers"]
        )


def test_report_revalidates_typed_records_and_fixed_vocabulary():
    report = probe_clinical_slm_capabilities(
        _manifest(), available_runtime_features=["tokenizers"]
    )
    object.__setattr__(report.runtime_features, "required", ("synthetic_private",))
    with pytest.raises(ClinicalSLMCapabilityError):
        replace(report)


def test_empty_checks_cannot_be_success():
    report = probe_clinical_slm_capabilities(
        _manifest(), available_runtime_features=["tokenizers"]
    )
    with pytest.raises(ClinicalSLMCapabilityError):
        replace(report, _checks=())


def test_invalid_json_does_not_retain_private_exception_context():
    with pytest.raises(ClinicalSLMCapabilityError) as caught:
        load_clinical_slm_capability_manifest('{"SYNTHETIC_PRIVATE":')
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "synthetic_private", "supported": True},
        {"name": NLI, "supported": True, "reason_codes": ("synthetic_private",)},
        {"name": NLI, "supported": True, "reason_codes": ("task_not_declared",)},
    ],
)
def test_capability_check_rejects_invalid_or_inconsistent_records(kwargs):
    with pytest.raises(ClinicalSLMCapabilityError):
        capabilities.ClinicalSLMCapabilityCheck(**kwargs)


def test_unbounded_requirement_stops_at_limit():
    read_count = 0

    def requested():
        nonlocal read_count
        while True:
            read_count += 1
            assert read_count <= capabilities._MAX_ITEMS + 1
            yield NLI

    with pytest.raises(ClinicalSLMCapabilityError):
        probe_clinical_slm_capabilities(_manifest(), required_capabilities=requested())
    assert read_count == capabilities._MAX_ITEMS + 1


def test_total_only_context_reserves_output_tokens():
    manifest = _manifest()
    manifest["context_limits"] = {"max_context_tokens": 4096, "max_output_tokens": 512}
    report = probe_clinical_slm_capabilities(
        manifest, available_runtime_features=["tokenizers"], min_context_tokens=4096
    )
    assert not report.supported
    assert report.context_limits.max_input_tokens == 3584


def test_nested_and_top_level_context_cannot_conflict():
    manifest = _manifest()
    manifest["max_input_tokens"] = 1
    with pytest.raises(ClinicalSLMCapabilityError):
        probe_clinical_slm_capabilities(manifest)


def test_runtime_generator_is_bounded_and_supported():
    report = probe_clinical_slm_capabilities(
        _manifest(), available_runtime_features=iter(["tokenizers"])
    )
    assert report.supported


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
        "runtime_feature_unknown",
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
