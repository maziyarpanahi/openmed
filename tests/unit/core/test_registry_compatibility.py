"""Focused tests for offline model-registry rollback compatibility."""

from __future__ import annotations

import json

import pytest

from openmed.core.registry_compatibility import (
    DECISION_BLOCKED,
    DECISION_COMPATIBLE,
    build_rollback_compatibility_report,
)


def _checkpoint(
    model_id: str,
    *,
    version: str,
    lineage: object,
    policy: object = "synthetic-policy-v1",
    tokenizer: object = (101, 202, 303),
    evidence: object = ("openmed.evidence.v1",),
    constraint: str = ">=1.0.0,<3.0.0",
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "family": "PII",
        "version": version,
        "semver_constraint": constraint,
        "lineage": lineage,
        "policy_fingerprint": policy,
        "tokenizer_ids": tokenizer,
        "evidence_schema_versions": evidence,
    }


def test_compatible_report_compares_all_contract_dimensions_offline() -> None:
    current = _checkpoint(
        "synthetic-model-v2",
        version="2.0.0",
        lineage=[
            {
                "relation": "supersedes",
                "from": "synthetic-model-v1",
                "to": "synthetic-model-v2",
            }
        ],
    )
    rollback = _checkpoint(
        "synthetic-model-v1",
        version="1.0.0",
        lineage=(),
    )

    report = build_rollback_compatibility_report(current, rollback)

    assert report.decision == DECISION_COMPATIBLE
    assert report.compatible is True
    assert report.blocked_reasons == ()
    assert set(report.checks) == {
        "identity",
        "family",
        "lineage",
        "semver",
        "policy_fingerprint",
        "tokenizer_contract",
        "evidence_schema",
    }
    assert report.check("lineage").code == "ROLLBACK_ANCESTOR"
    assert report.to_dict() == json.loads(report.to_json())
    assert report.fingerprint == report.to_dict()["fingerprint"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("policy_fingerprint", "synthetic-policy-v2", "POLICY_FINGERPRINT_MISMATCH"),
        ("tokenizer_ids", (101, 999), "TOKENIZER_CONTRACT_MISMATCH"),
        (
            "evidence_schema_versions",
            ("openmed.evidence.v2",),
            "EVIDENCE_SCHEMA_MISMATCH",
        ),
    ],
)
def test_contract_mismatch_blocks_without_copying_raw_values(
    field: str,
    value: object,
    reason: str,
) -> None:
    current = _checkpoint(
        "/synthetic/private/current",
        version="2.0.0",
        lineage=["/synthetic/private/rollback"],
    )
    rollback = _checkpoint(
        "/synthetic/private/rollback",
        version="1.0.0",
        lineage=(),
    )
    rollback[field] = value

    report = build_rollback_compatibility_report(current, rollback)
    serialized = report.to_json()

    assert report.decision == DECISION_BLOCKED
    assert reason in report.blocked_reasons
    assert "/synthetic/private" not in serialized
    assert "synthetic-policy-v2" not in serialized
    assert "999" not in serialized


def test_missing_contract_metadata_fails_closed_and_is_deterministic() -> None:
    current = {"model_id": "synthetic-model-v2", "family": "PII", "version": "2.0.0"}
    rollback = {"model_id": "synthetic-model-v1", "family": "PII", "version": "1.0.0"}

    first = build_rollback_compatibility_report(current, rollback)
    second = build_rollback_compatibility_report(current, rollback)

    assert first.decision == DECISION_BLOCKED
    assert {
        "LINEAGE_NOT_ANCESTOR",
        "SEMVER_CONSTRAINT_MISSING",
        "POLICY_FINGERPRINT_MISSING",
        "TOKENIZER_CONTRACT_MISSING",
        "EVIDENCE_SCHEMA_MISSING",
    }.issubset(first.blocked_reasons)
    assert first.to_dict() == second.to_dict()


def test_semver_ranges_and_state_pointers_are_local_only() -> None:
    state = {
        "schema_version": 1,
        "families": {
            "PII": {
                "pointers": {
                    "latest": "synthetic-model-v2",
                    "canary": None,
                    "last_green": "synthetic-model-v1",
                },
                "versions": {
                    "synthetic-model-v1": "1.0.0",
                    "synthetic-model-v2": "2.0.0",
                },
                "lineage": [
                    {
                        "relation": "supersedes",
                        "from": "synthetic-model-v1",
                        "to": "synthetic-model-v2",
                    }
                ],
            }
        },
        "checkpoints": {
            "synthetic-model-v1": _checkpoint(
                "synthetic-model-v1",
                version="1.0.0",
                lineage=(),
            ),
            "synthetic-model-v2": _checkpoint(
                "synthetic-model-v2",
                version="2.0.0",
                lineage=(),
            ),
        },
    }

    report = build_rollback_compatibility_report(
        registry_state=state,
        family="PII",
    )

    assert report.decision == DECISION_COMPATIBLE
    assert report.check("semver").code == "SEMVER_CONSTRAINT_SATISFIED"


def test_invalid_semver_constraint_blocks_without_echoing_input() -> None:
    current = _checkpoint(
        "synthetic-model-v2",
        version="2.0.0",
        lineage=["synthetic-model-v1"],
        constraint="not-a-version-constraint",
    )
    rollback = _checkpoint(
        "synthetic-model-v1",
        version="1.0.0",
        lineage=(),
    )

    report = build_rollback_compatibility_report(current, rollback)

    assert report.decision == DECISION_BLOCKED
    assert "SEMVER_CONSTRAINT_INVALID" in report.blocked_reasons
    assert "not-a-version-constraint" not in report.to_json()


def test_policy_schema_version_is_bound_to_policy_fingerprint() -> None:
    current = _checkpoint(
        "synthetic-model-v2",
        version="2.0.0",
        lineage=["synthetic-model-v1"],
    )
    current["policy_schema_version"] = 2
    rollback = _checkpoint(
        "synthetic-model-v1",
        version="1.0.0",
        lineage=(),
    )
    rollback["policy_schema_version"] = 1

    report = build_rollback_compatibility_report(current, rollback)

    assert report.decision == DECISION_BLOCKED
    assert "POLICY_FINGERPRINT_MISMATCH" in report.blocked_reasons


def test_lineage_direction_must_point_from_current_to_ancestor() -> None:
    current = _checkpoint(
        "synthetic-model-v2",
        version="2.0.0",
        lineage=[
            {
                "relation": "supersedes",
                "from": "synthetic-model-v2",
                "to": "synthetic-model-v3",
            }
        ],
    )
    rollback = _checkpoint(
        "synthetic-model-v3",
        version="1.0.0",
        lineage=(),
    )

    report = build_rollback_compatibility_report(current, rollback)

    assert report.decision == DECISION_BLOCKED
    assert "LINEAGE_NOT_ANCESTOR" in report.blocked_reasons
