"""Tests for counts-only privacy evidence replay."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from openmed.risk import (
    EvidenceReplaySchemaError,
    UnsafeReplayInputError,
    build_evidence_manifest,
    compute_environment_fingerprint,
    compute_policy_fingerprint,
    compute_result_fingerprint,
    replay_evidence,
)


def _policy() -> dict[str, object]:
    return {
        "id": "synthetic-privacy-policy",
        "version": "1",
        "rules": {"EMAIL": "mask", "PERSON": "mask", "PHONE": "redact"},
        "default_action": "keep",
    }


def _environment() -> dict[str, object]:
    return {"runtime": "local", "runtime_version": "1", "offline": True}


def _inputs() -> list[dict[str, object]]:
    return [
        {"category_counts": {"PERSON": 2, "PHONE": 1}},
        {"category_counts": {"EMAIL": 1, "UNKNOWN": 3}},
    ]


def _manifest() -> dict[str, object]:
    return build_evidence_manifest(
        policy=_policy(),
        environment=_environment(),
        synthetic_inputs=_inputs(),
    )


def test_matching_replay_is_deterministic_and_aggregate_only() -> None:
    manifest = _manifest()

    first = replay_evidence(manifest)
    second = replay_evidence(copy.deepcopy(manifest))

    assert first.matched is True
    assert first.mismatch_categories == ()
    assert first.actual_decision_counts == {"keep": 3, "mask": 3, "redact": 1}
    assert first.actual_result_fingerprint == second.actual_result_fingerprint
    assert first.to_dict() == second.to_dict()
    assert "synthetic_inputs" not in first.to_dict()
    assert "PERSON" not in first.to_json()
    assert "PERSON" not in first.to_markdown()


def test_replay_is_invariant_to_synthetic_input_order() -> None:
    manifest = _manifest()

    reordered = replay_evidence(manifest, synthetic_inputs=list(reversed(_inputs())))

    assert reordered.matched is True
    assert reordered.actual_decision_counts == {"keep": 3, "mask": 3, "redact": 1}
    assert (
        reordered.actual_result_fingerprint
        == replay_evidence(manifest).actual_result_fingerprint
    )


@pytest.mark.parametrize(
    ("field", "value", "category"),
    [
        ("environment", {"runtime": "other", "offline": True}, "environment"),
        (
            "policy",
            {
                **_policy(),
                "rules": {"EMAIL": "keep", "PERSON": "mask", "PHONE": "redact"},
            },
            "policy",
        ),
    ],
)
def test_replay_classifies_environment_and_policy_mismatches(
    field: str,
    value: dict[str, object],
    category: str,
) -> None:
    kwargs = {field: value}

    report = replay_evidence(_manifest(), **kwargs)

    assert category in report.mismatch_categories
    assert report.matched is False
    assert report.to_dict()["mismatches"][0]["category"] == category


def test_policy_drift_that_changes_counts_is_classified_as_result_too() -> None:
    changed_policy = {
        **_policy(),
        "rules": {"EMAIL": "keep", "PERSON": "mask", "PHONE": "redact"},
    }

    report = replay_evidence(_manifest(), policy=changed_policy)

    assert report.mismatch_categories == ("policy", "result")
    assert report.actual_decision_counts == {"keep": 4, "mask": 2, "redact": 1}


def test_schema_version_mismatch_is_reported_without_loading_payload_values() -> None:
    manifest = _manifest()
    manifest["schema_version"] = 2

    report = replay_evidence(manifest)

    assert report.matched is False
    assert report.mismatch_categories == ("schema",)
    assert report.to_dict()["mismatches"] == [
        {
            "actual": 1,
            "category": "schema",
            "expected": 2,
            "field": "schema_version",
        }
    ]


def test_result_mismatch_is_classified_when_expected_counts_are_tampered() -> None:
    manifest = _manifest()
    expected = manifest["expected"]
    assert isinstance(expected, dict)
    expected["decision_counts"] = {"keep": 999}

    report = replay_evidence(manifest)

    assert report.mismatch_categories == ("result",)
    assert report.expected_decision_counts == {"keep": 999}
    assert report.actual_decision_counts == {"keep": 3, "mask": 3, "redact": 1}


def test_path_round_trip_and_json_rendering_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "replay-manifest.json"
    path.write_text(json.dumps(_manifest(), sort_keys=True), encoding="utf-8")

    first = replay_evidence(path)
    second = replay_evidence(path)

    assert first.to_json() == second.to_json()
    assert "sha256:" in first.to_json()
    assert "Decision counts" in first.to_markdown()


def test_fingerprint_helpers_ignore_mapping_order() -> None:
    assert compute_policy_fingerprint(_policy()) == compute_policy_fingerprint(
        {
            "default_action": "keep",
            "rules": {"PHONE": "redact", "PERSON": "mask", "EMAIL": "mask"},
            "version": "1",
            "id": "synthetic-privacy-policy",
        }
    )
    assert compute_environment_fingerprint(
        _environment()
    ) == compute_environment_fingerprint(
        {"offline": True, "runtime_version": "1", "runtime": "local"}
    )
    assert compute_result_fingerprint(
        {"mask": 3, "keep": 3}
    ) == compute_result_fingerprint({"keep": 3, "mask": 3})


def test_opaque_environment_digest_is_supported() -> None:
    digest = "sha256:" + "a" * 64
    manifest = build_evidence_manifest(
        policy=_policy(),
        environment=digest,
        synthetic_inputs=[],
    )

    assert replay_evidence(manifest).matched is True
    assert manifest["environment"] == {"fingerprint": digest}


def test_payload_bearing_synthetic_input_is_rejected_without_echoing_value() -> None:
    manifest = _manifest()
    inputs = manifest["synthetic_inputs"]
    assert isinstance(inputs, list)
    inputs[0]["raw_text"] = "forbidden-payload-marker"

    with pytest.raises(UnsafeReplayInputError) as exc_info:
        replay_evidence(manifest)

    assert "forbidden-payload-marker" not in str(exc_info.value)


def test_unknown_manifest_field_is_rejected_without_echoing_payload() -> None:
    manifest = _manifest()
    manifest["protected_payload"] = "forbidden-payload-marker"

    with pytest.raises(EvidenceReplaySchemaError) as exc_info:
        replay_evidence(manifest)

    assert "forbidden-payload-marker" not in str(exc_info.value)
