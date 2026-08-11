"""Tests for deterministic, privacy-safe model provenance diffs."""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from openmed.eval.model_provenance_diff import (
    DRIFT_FINGERPRINT_CHANGED,
    DRIFT_VERSION_CHANGED,
    MODEL_PROVENANCE_DIFF_SCHEMA_VERSION,
    ModelProvenanceInputError,
    ModelProvenancePrivacyError,
    assert_no_raw_text,
    build_model_provenance_manifest,
    diff_model_provenance,
    load_model_provenance_manifest,
    write_model_provenance_manifest,
)


def _manifest(*, model_fingerprint: str = "sha256:model-a", model_version: str = "v1"):
    return {
        "model": {"fingerprint": model_fingerprint, "version": model_version},
        "tokenizer": {"fingerprint": "sha256:tokenizer-a", "version": "v3"},
        "policy": {"fingerprint": "sha256:policy-a", "version": "policy-v2"},
        "fixtures": {"fingerprint": "sha256:fixtures-a", "version": "fixture-v4"},
        "evaluation_slices": [
            {
                "name": "baseline",
                "fingerprint": "sha256:baseline-a",
                "version": "slice-v1",
            },
            "multilingual",
        ],
    }


def test_identical_manifests_produce_a_deterministic_no_drift_report() -> None:
    first = diff_model_provenance(_manifest(), _manifest())
    second = diff_model_provenance(
        baseline={**_manifest(), "prompt": "synthetic prompt payload"},
        candidate={**_manifest(), "notes": "synthetic note payload"},
    )

    assert first.to_dict() == second.to_dict()
    assert first.changed is False
    assert first.changed_components == ()
    assert first.to_dict()["schema_version"] == MODEL_PROVENANCE_DIFF_SCHEMA_VERSION
    assert "synthetic" not in first.to_json()


def test_component_drift_is_classified_by_fingerprint_and_version() -> None:
    candidate = _manifest(
        model_fingerprint="sha256:model-b",
        model_version="v2",
    )
    report = diff_model_provenance(_manifest(), candidate)

    model_change = report.components["model"]
    assert model_change.classification == "fingerprint_and_version_changed"
    assert model_change.reasons == (
        DRIFT_FINGERPRINT_CHANGED,
        DRIFT_VERSION_CHANGED,
    )
    assert report.changed_components == ("model",)
    assert report.to_dict()["components"]["model"]["after"] == {
        "fingerprint": "sha256:model-b",
        "version": "v2",
    }


def test_slice_drift_reports_added_removed_and_changed_declarations() -> None:
    candidate = _manifest()
    candidate["evaluation_slices"] = [
        {
            "name": "baseline",
            "fingerprint": "sha256:baseline-b",
            "version": "slice-v1",
        },
        {"name": "regression", "version": "slice-v2"},
    ]

    report = diff_model_provenance(_manifest(), candidate)
    slices = report.to_dict()["evaluation_slices"]

    assert slices["added"] == ["regression"]
    assert slices["removed"] == ["multilingual"]
    assert slices["changed"][0]["name"] == "baseline"
    assert slices["changed"][0]["reasons"] == [DRIFT_FINGERPRINT_CHANGED]
    assert report.changed_components == ("evaluation_slices",)


def test_builder_and_local_json_round_trip_only_safe_provenance(tmp_path: Path) -> None:
    payload = build_model_provenance_manifest(
        model=_manifest()["model"],
        tokenizer=_manifest()["tokenizer"],
        policy=_manifest()["policy"],
        fixtures=_manifest()["fixtures"],
        evaluation_slices=_manifest()["evaluation_slices"],
    )
    path = write_model_provenance_manifest(tmp_path / "run.json", payload)
    loaded = load_model_provenance_manifest(path)

    assert path.exists()
    assert loaded.to_payload() == payload
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_diff_does_not_open_a_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("model provenance diff attempted network access")

    monkeypatch.setattr(socket, "socket", fail_socket)
    assert diff_model_provenance(_manifest(), _manifest()).changed is False


def test_known_free_form_values_fail_without_echoing_the_value() -> None:
    secret = "synthetic prompt payload"
    leaky = _manifest()
    leaky["model"] = {"fingerprint": secret, "version": "v1"}

    with pytest.raises(ModelProvenancePrivacyError) as caught:
        diff_model_provenance(_manifest(), leaky)

    assert secret not in str(caught.value)


def test_report_guard_rejects_raw_value_and_unsafe_field() -> None:
    with pytest.raises(ModelProvenancePrivacyError):
        assert_no_raw_text({"notes": "synthetic note payload"})
    with pytest.raises(ModelProvenancePrivacyError):
        assert_no_raw_text({"safe_field": "synthetic free form value"})


def test_malformed_manifest_does_not_accept_missing_provenance() -> None:
    with pytest.raises(ModelProvenanceInputError):
        diff_model_provenance(
            _manifest(),
            {"model": {"fingerprint": "sha256:model-b", "version": "v2"}},
        )
