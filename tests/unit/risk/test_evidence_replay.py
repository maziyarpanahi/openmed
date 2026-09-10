"""Tests for counts-only privacy evidence replay."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from openmed.risk import (
    EvidenceReplayError,
    EvidenceReplaySchemaError,
    ReplayMismatch,
    UnsafeReplayInputError,
    build_evidence_manifest,
    compute_environment_fingerprint,
    compute_policy_fingerprint,
    compute_result_fingerprint,
    replay_evidence,
)


class _ExplodingMapping(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise RuntimeError("synthetic-sensitive-container-error")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("synthetic-sensitive-container-error")

    def __len__(self) -> int:
        return 1


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


def test_unknown_action_names_are_fingerprinted_in_reports() -> None:
    raw_action = "syntheticPrivateActionCanary"
    manifest = build_evidence_manifest(
        policy={
            **_policy(),
            "default_action": raw_action,
            "rules": {},
        },
        environment=_environment(),
        synthetic_inputs=[{"category_counts": {"UNKNOWN": 1}}],
    )

    report = replay_evidence(manifest)
    serialized = report.to_json() + report.to_markdown()

    assert report.matched is True
    assert all(key.startswith("action:") for key in report.actual_decision_counts)
    assert raw_action not in serialized


def test_public_mismatch_evidence_is_closed_and_value_safe() -> None:
    raw_action = "syntheticPrivateActionCanary"
    mismatch = ReplayMismatch(
        category="result",
        field="decision_counts",
        expected={raw_action: 1},
        actual={"mask": 1},
    )

    assert raw_action not in json.dumps(mismatch.to_dict(), sort_keys=True)
    with pytest.raises(ValueError):
        ReplayMismatch(
            category="result",
            field="arbitrary_field",
            expected=1,
            actual=1,
        )
    with pytest.raises(ValueError):
        ReplayMismatch(
            category="schema",
            field="schema_version",
            expected="syntheticPrivateSchemaCanary",
            actual=1,
        )


def test_duplicate_nonfinite_and_oversized_json_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(EvidenceReplaySchemaError):
        replay_evidence(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(EvidenceReplaySchemaError):
        replay_evidence(nonfinite)

    import openmed.risk.evidence_replay as replay_module

    monkeypatch.setattr(replay_module, "_MAX_FILE_BYTES", 32)
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 33)
    with pytest.raises(EvidenceReplaySchemaError):
        replay_evidence(oversized)


def test_hostile_containers_and_excessive_counts_fail_without_leaking() -> None:
    with pytest.raises(EvidenceReplayError) as error:
        replay_evidence(_ExplodingMapping())
    assert "synthetic-sensitive-container-error" not in str(error.value)

    with pytest.raises(EvidenceReplayError) as error:
        compute_result_fingerprint(_ExplodingMapping())
    assert "synthetic-sensitive-container-error" not in str(error.value)

    with pytest.raises(UnsafeReplayInputError):
        compute_result_fingerprint({"mask": 2**63})

    policy = _policy()
    manifest = build_evidence_manifest(
        policy=policy,
        environment=_environment(),
        synthetic_inputs=[{"category_counts": {"PERSON": 2**63 - 1}}],
    )
    with pytest.raises(UnsafeReplayInputError):
        replay_evidence(
            manifest,
            synthetic_inputs=[{"category_counts": {"PERSON": 2**63 - 1, "EMAIL": 1}}],
        )


def test_file_errors_do_not_retain_sensitive_exception_context(tmp_path: Path) -> None:
    canary = "synthetic-sensitive-file-canary"
    missing = tmp_path / canary

    with pytest.raises(EvidenceReplaySchemaError) as error:
        replay_evidence(missing)

    assert canary not in str(error.value)
    assert error.value.__cause__ is None
