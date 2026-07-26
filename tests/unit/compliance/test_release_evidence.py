"""Tests for release-to-expert-review evidence integration."""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest

from openmed.__about__ import __version__
from openmed.compliance import (
    ReleaseAssumptions,
    build_release_expert_review_evidence,
)
from openmed.compliance import release_evidence as release_evidence_module
from openmed.core.audit import stable_hash
from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    validate_released_output,
)


def _result():
    rows = [
        {
            "patient_id": "patient-alpha",
            "patient_name": "Alice Canary",
            "age": 31,
            "zip": "10001",
            "disease": "flu-canary",
            "site_code": 101,
            "source_batch": "batch-canary-a",
        },
        {
            "patient_id": "patient-beta",
            "patient_name": "Bob Canary",
            "age": 32,
            "zip": "10002",
            "disease": "cold-canary",
            "site_code": 101,
            "source_batch": "batch-canary-a",
        },
        {
            "patient_id": "patient-gamma",
            "patient_name": "Carol Canary",
            "age": 41,
            "zip": "20001",
            "disease": "flu-canary",
            "site_code": 202,
            "source_batch": "batch-canary-b",
        },
        {
            "patient_id": "patient-delta",
            "patient_name": "Dan Canary",
            "age": 42,
            "zip": "20002",
            "disease": "cold-canary",
            "site_code": 202,
            "source_batch": "batch-canary-b",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("disease",),
        direct_identifiers=("patient_name",),
        non_sensitive_attributes=("site_code",),
        excluded_attributes=("source_batch",),
        privacy_unit="patient_id",
        target_k=2,
        target_l=2,
    )
    return anonymize_release(rows, policy)


def _assumptions(*, privacy_unit: str = "patient") -> ReleaseAssumptions:
    return ReleaseAssumptions(
        privacy_unit=privacy_unit,
        population_scope="release_cohort",
        release_model="restricted",
        recipient_model="named_researchers",
        auxiliary_data_model="reasonably_available",
        notes_digest=stable_hash(
            {
                "kind": "unit-test-release-assumptions",
                "notes": "reviewed outside the shareable bundle",
            }
        ),
    )


def test_anonymization_builds_verifiable_phi_safe_expert_review_evidence() -> None:
    result = _result()
    validation = validate_released_output(result.records, result)

    report = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=_assumptions(),
    )

    assert report.verify() is True
    assert report.privacy_models.configured_k == 2
    assert report.privacy_models.achieved_k >= 2
    assert report.search.complete is True
    assert report.post_metrics.privacy_unit_count == result.after.privacy_unit_count
    assert report.digests.source_dataset == result.source_dataset_digest
    assert report.digests.dataset == validation.dataset_digest
    assert report.search.evaluated_candidates == result.generalization.nodes_evaluated
    assert report.search.total_candidates == result.generalization.search_space_size
    assert report.search.candidate_limit == result.generalization.max_lattice_nodes
    assert report.search.suppression_subsets_evaluated == (
        result.generalization.suppression_subsets_evaluated
    )
    assert report.search.suppression_subsets_total == (
        result.generalization.suppression_subsets_possible
    )
    assert report.search.suppression_subset_limit == (
        result.generalization.max_suppression_subsets
    )
    assert report.digests.schema == validation.schema_digest
    assert report.digests.hierarchy == result.hierarchy_digest
    assert report.pre_metrics.k_violating_class_count == (
        result.before.k_violating_class_count
    )
    assert report.pre_metrics.l_violating_class_count == (
        result.before.l_violating_class_count
    )
    assert report.pre_metrics.t_violating_class_count == (
        result.before.t_violating_class_count
    )
    roles = {item.attribute: item.roles for item in report.attribute_reviews}
    assert roles == {
        "age": ("quasi_identifier",),
        "disease": ("sensitive_attribute",),
        "patient_id": ("direct_identifier", "privacy_unit"),
        "patient_name": ("direct_identifier",),
        "site_code": ("non_sensitive",),
        "source_batch": ("excluded",),
        "zip": ("quasi_identifier",),
    }
    direct_identifier_metric = next(
        item
        for item in report.utility
        if item.metric == "direct_identifier_cells_remaining"
    )
    assert direct_identifier_metric.before > 0
    assert direct_identifier_metric.after == 0
    assert report.title.endswith("Not an Expert Determination")
    serialized = report.to_json()
    markdown = report.to_markdown()
    for canary in (
        "patient-alpha",
        "Alice Canary",
        "10001",
        "flu-canary",
        "batch-canary-a",
    ):
        assert canary not in serialized
        assert canary not in markdown
    assert '"records"' not in serialized
    assert "equivalence_classes" not in serialized
    assert "qualified expert" in markdown.lower()


def test_release_evidence_is_deterministic_and_rejects_tampering() -> None:
    result = _result()
    assumptions = _assumptions()
    validation = validate_released_output(result.records, result)

    first = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=assumptions,
    )
    second = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=assumptions,
    )

    assert first.to_json() == second.to_json()
    assert first.digests.software != stable_hash(
        {
            "kind": "openmed-software",
            "package": "openmed",
            "version": __version__,
        }
    )
    payload = first.to_dict()
    payload["privacy_models"]["k_anonymity"]["achieved_k"] = 1

    from openmed.compliance import ExpertReviewEvidenceReport

    try:
        ExpertReviewEvidenceReport.from_json(json.dumps(payload))
    except ValueError as exc:
        assert "integrity" in str(exc) or "match" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("tampered expert-review evidence was accepted")


def test_release_evidence_cannot_erase_mandatory_caveats() -> None:
    result = _result()
    validation = validate_released_output(result.records, result)

    report = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=_assumptions(),
        limitations=(),
        unsupported_modalities=(),
    )

    assert report.limitations == (
        "not_compliance_certificate",
        "population_risk_not_estimated",
        "qualified_expert_review_required",
    )
    assert report.unsupported_modalities == (
        "free_text",
        "images",
        "genomic_data",
    )

    extended = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=_assumptions(),
        limitations=("recipient_linkage_not_assessed",),
        unsupported_modalities=("audio",),
    )
    assert extended.limitations == (
        *report.limitations,
        "recipient_linkage_not_assessed",
    )
    assert extended.unsupported_modalities == (
        *report.unsupported_modalities,
        "audio",
    )


def test_software_digest_covers_transitive_modules_and_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    expected_digest = "sha256:" + ("a" * 64)

    def capture_payload(payload: dict[str, object]) -> str:
        captured.update(payload)
        return expected_digest

    monkeypatch.setattr(release_evidence_module, "stable_hash", capture_payload)

    assert release_evidence_module._software_digest() == expected_digest
    assert set(captured) == {"kind", "package", "runtime", "modules"}
    assert captured["kind"] == "openmed-software-content"
    assert captured["package"] == {
        "name": "openmed",
        "version": __version__,
    }
    assert captured["runtime"] == {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        "python_cache_tag": sys.implementation.cache_tag or "unknown",
    }
    modules = captured["modules"]
    assert isinstance(modules, dict)
    assert set(modules) == set(release_evidence_module._SOFTWARE_EVIDENCE_MODULES)
    assert "openmed.risk.reid" in modules
    assert all(
        isinstance(value, str) and value.startswith("sha256:")
        for value in modules.values()
    )


def test_release_evidence_records_reviewed_coarsening_and_nonzero_loss() -> None:
    result = anonymize_release(
        [
            {"patient_id": "a", "facility": "north"},
            {"patient_id": "b", "facility": "south"},
        ],
        AnonymityPolicy(
            quasi_identifiers=("facility",),
            privacy_unit="patient_id",
            target_k=2,
        ),
        hierarchies={
            "facility": [
                {"name": "exact", "loss": 0.0},
                {"name": "collapsed", "loss": 0.5, "default": "*"},
            ]
        },
    )

    report = build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=_assumptions(),
    )

    assert len(report.transformations) == 1
    assert report.transformations[0].attribute == "facility"
    assert report.transformations[0].method == "suppress"
    assert report.suppression.rows_suppressed == 0
    assert report.suppression.cells_suppressed == 2
    information_loss = next(
        item for item in report.utility if item.metric == "information_loss"
    )
    assert information_loss.after == pytest.approx(0.5)


def test_release_evidence_counts_row_level_qi_cell_suppression() -> None:
    result = anonymize_release(
        [{"code": value} for value in ("a", "b", "c", "d")],
        AnonymityPolicy(
            quasi_identifiers=("code",),
            target_k=2,
        ),
        hierarchies={
            "code": [
                {"name": "exact", "loss": 0.0},
                {"name": "suppressed", "loss": 1.0, "default": "*"},
            ]
        },
    )

    report = build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=_assumptions(privacy_unit="row"),
    )

    assert result.generalization.affected_qi_cells == (("code", 4),)
    assert result.generalization.suppressed_qi_cells == (("code", 4),)
    assert report.transformations[0].method == "suppress"
    assert report.suppression.privacy_units_suppressed == 0
    assert report.suppression.rows_suppressed == 0
    assert report.suppression.cells_suppressed == 4


def test_release_evidence_distinguishes_suppressed_rows_and_privacy_units() -> None:
    rows = [
        {"patient_id": "a", "age": 30, "zip": "10001"},
        {"patient_id": "a", "age": 30, "zip": "10001"},
        {"patient_id": "b", "age": 30, "zip": "10001"},
        {"patient_id": "b", "age": 30, "zip": "10001"},
        {"patient_id": "outlier", "age": 99, "zip": "99999"},
        {"patient_id": "outlier", "age": 99, "zip": "99999"},
    ]
    result = anonymize_release(
        rows,
        AnonymityPolicy(
            quasi_identifiers=("age", "zip"),
            privacy_unit="patient_id",
            target_k=2,
            suppression_limit=1,
        ),
    )

    report = build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=_assumptions(),
    )

    assert report.suppression.privacy_units_suppressed == 1
    assert report.suppression.rows_suppressed == 2


def test_release_evidence_counts_only_privacy_units_changed_by_mapping() -> None:
    result = anonymize_release(
        [
            {"patient_id": "a", "facility": "north"},
            {"patient_id": "b", "facility": "south"},
            {"patient_id": "c", "facility": "east"},
            {"patient_id": "d", "facility": "east"},
        ],
        AnonymityPolicy(
            quasi_identifiers=("facility",),
            privacy_unit="patient_id",
            target_k=2,
        ),
        hierarchies={
            "facility": [
                {"name": "exact", "loss": 0.0},
                {
                    "name": "region",
                    "loss": 0.5,
                    "values": {"north": "region", "south": "region"},
                },
                {"name": "suppressed", "loss": 1.0, "default": "*"},
            ]
        },
    )
    report = build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=_assumptions(),
    )

    assert result.utility.quasi_identifier_cells_changed == 2
    assert result.generalization.affected_privacy_units == (("facility", 2),)
    assert report.transformations[0].affected_privacy_unit_count == 2


def test_release_evidence_requires_a_passing_materialized_validation() -> None:
    result = _result()
    validation = validate_released_output(result.records, result)
    failed = replace(validation, row_count=validation.row_count + 1)

    assert failed.passed is False
    with pytest.raises(ValueError, match="validation must pass"):
        build_release_expert_review_evidence(
            result,
            validation=failed,
            assumptions=_assumptions(),
        )

    with pytest.raises(TypeError, match="ReleasedOutputValidation"):
        build_release_expert_review_evidence(
            result,
            validation=object(),  # type: ignore[arg-type]
            assumptions=_assumptions(),
        )


def test_release_evidence_rejects_validation_not_bound_to_the_result() -> None:
    result = _result()
    validation = validate_released_output(result.records, result)
    unrelated_dataset_digest = stable_hash({"kind": "unrelated-dataset"})
    forged_dataset = replace(
        validation,
        dataset_digest=unrelated_dataset_digest,
        expected_digest=unrelated_dataset_digest,
    )

    assert forged_dataset.passed is True
    with pytest.raises(ValueError, match="dataset binding"):
        build_release_expert_review_evidence(
            result,
            validation=forged_dataset,
            assumptions=_assumptions(),
        )

    unrelated_schema_digest = stable_hash({"kind": "unrelated-schema"})
    forged_schema = replace(
        validation,
        schema_digest=unrelated_schema_digest,
        expected_schema_digest=unrelated_schema_digest,
    )
    assert forged_schema.passed is True
    with pytest.raises(ValueError, match="schema binding"):
        build_release_expert_review_evidence(
            result,
            validation=forged_schema,
            assumptions=_assumptions(),
        )


def test_release_evidence_accepts_validated_delimited_scalar_encoding() -> None:
    result = _result()
    reread = [
        {field: "" if value is None else str(value) for field, value in row.items()}
        for row in result.records
    ]
    validation = validate_released_output(
        reread,
        result,
        preserve_scalar_types=False,
    )

    assert validation.passed is True
    assert validation.expected_schema_digest != result.released_schema_digest
    report = build_release_expert_review_evidence(
        result,
        validation=validation,
        assumptions=_assumptions(),
    )

    assert report.digests.dataset == validation.dataset_digest
    assert report.digests.schema == validation.schema_digest


def test_release_assumption_privacy_unit_matches_policy_semantics() -> None:
    keyed_result = _result()
    keyed_validation = validate_released_output(
        keyed_result.records,
        keyed_result,
    )
    with pytest.raises(ValueError, match="cannot use.*'row'"):
        build_release_expert_review_evidence(
            keyed_result,
            validation=keyed_validation,
            assumptions=_assumptions(privacy_unit="row"),
        )

    row_records = [
        {"name": "A Canary", "age": 31, "disease": "flu", "site_code": 101},
        {"name": "B Canary", "age": 31, "disease": "cold", "site_code": 101},
        {"name": "C Canary", "age": 41, "disease": "flu", "site_code": 202},
        {"name": "D Canary", "age": 41, "disease": "cold", "site_code": 202},
    ]
    row_result = anonymize_release(
        row_records,
        AnonymityPolicy(
            quasi_identifiers=("age",),
            sensitive_attributes=("disease",),
            direct_identifiers=("name",),
            non_sensitive_attributes=("site_code",),
            target_k=2,
        ),
    )
    row_validation = validate_released_output(row_result.records, row_result)

    report = build_release_expert_review_evidence(
        row_result,
        validation=row_validation,
        assumptions=_assumptions(privacy_unit="row"),
    )
    assert report.assumptions.privacy_unit == "row"

    with pytest.raises(ValueError, match="require.*'row'"):
        build_release_expert_review_evidence(
            row_result,
            validation=row_validation,
            assumptions=_assumptions(),
        )
