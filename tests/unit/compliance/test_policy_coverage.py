"""Focused tests for the deterministic privacy-policy coverage matrix."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    POLICY_COVERAGE_MANIFEST_FILENAME,
    POLICY_COVERAGE_MARKDOWN_FILENAME,
    PolicyCoverageBinding,
    UncoveredPolicyRuleError,
    build_policy_coverage_matrix,
    generate_policy_coverage,
    render_policy_coverage_markdown,
)
from openmed.core.labels import CANONICAL_LABELS

_FOCUSED_TEST = (
    "tests/unit/compliance/test_policy_coverage.py::"
    "test_required_rules_have_fixture_and_focused_test_coverage"
)


def test_required_rules_have_fixture_and_focused_test_coverage(monkeypatch) -> None:
    matrix = build_policy_coverage_matrix()

    assert matrix.verified is True
    assert matrix.required_rule_count == matrix.covered_required_rule_count
    assert matrix.uncovered_required_rules == ()
    assert matrix.policy_count == 19
    assert matrix.fixture_ids == (
        "synthetic-policy-clinical-concepts",
        "synthetic-policy-direct-identifiers",
        "synthetic-policy-quasi-identifiers",
        "synthetic-policy-sensitive-attributes",
    )
    assert matrix.focused_tests == (_FOCUSED_TEST,)
    assert len(matrix.structured_field_ids) == 32
    assert all(row.fixture_id and row.focused_test for row in matrix.rows)
    assert all(
        row.resource_path.startswith("openmed/core/policies/") for row in matrix.rows
    )

    # Exercise each declared evidence link through the public policy pipeline.
    # These synthetic category fixtures live here; the report contains only IDs.
    from openmed.core import pii
    from openmed.processing.outputs import EntityPrediction, PredictionResult

    fixtures = {
        "synthetic-policy-direct-identifiers": "Synthetic Example",
        "synthetic-policy-quasi-identifiers": "Synthetic Region",
        "synthetic-policy-sensitive-attributes": "Synthetic Attribute",
        "synthetic-policy-clinical-concepts": "Synthetic Finding",
    }
    for row in matrix.rows:
        surface = fixtures[row.fixture_id]
        entity = EntityPrediction(
            text=surface, label=row.label, start=0, end=len(surface), confidence=0.99
        )

        def extract(text, *args, **kwargs):
            return PredictionResult(
                text=text,
                entities=[entity],
                model_name="synthetic",
                timestamp="2026-01-01T00:00:00",
            )

        monkeypatch.setattr(pii, "extract_pii", extract)
        result = pii.deidentify(surface, policy=row.policy_name, use_safety_sweep=False)
        if row.action == "keep":
            assert result.deidentified_text == surface, row.rule_id
        else:
            assert result.pii_entities, row.rule_id
            assert (
                result.pii_entities[0].metadata["policy_action"]["action"] == row.action
            ), row.rule_id


def test_matrix_includes_structured_field_links_and_keep_rules() -> None:
    matrix = build_policy_coverage_matrix(policies=("gdpr_pseudonymization",))

    id_rows = [row for row in matrix.rows if row.label == "ID_NUM"]
    assert id_rows
    assert all("medical_record_number" in row.structured_fields for row in id_rows)

    keep_row = next(row for row in matrix.rows if row.label == "DISEASE")
    assert keep_row.action == "keep"
    assert keep_row.required is False
    assert keep_row.status == "covered"


def test_uncovered_required_rules_fail_without_raw_values_in_exception() -> None:
    binding = PolicyCoverageBinding(
        policy_name="clinical_minimal_redaction",
        label="PERSON",
        fixture_id="synthetic-policy-direct-identifiers",
        focused_test=_FOCUSED_TEST,
    )

    with pytest.raises(UncoveredPolicyRuleError) as raised:
        build_policy_coverage_matrix(
            policies=("clinical_minimal_redaction",),
            bindings=(binding,),
        )

    error = raised.value
    assert error.uncovered_rule_ids
    assert "raw" not in str(error).lower()
    assert "surface" not in str(error).lower()
    assert "value" not in str(error).lower()


def test_manifest_and_markdown_are_deterministic_and_counts_only(tmp_path) -> None:
    first = generate_policy_coverage(tmp_path / "first")
    second = generate_policy_coverage(tmp_path / "second")

    assert first.manifest == second.manifest
    assert first.markdown == second.markdown
    assert first.manifest_path.name == POLICY_COVERAGE_MANIFEST_FILENAME
    assert first.markdown_path.name == POLICY_COVERAGE_MARKDOWN_FILENAME
    assert json.loads(first.manifest_path.read_text(encoding="utf-8")) == first.manifest
    assert first.matrix.fingerprint == first.manifest["hashes"]["matrix"]

    rendered = render_policy_coverage_markdown(first.matrix)
    serialized = json.dumps(first.manifest, sort_keys=True) + rendered
    assert "raw_text" not in serialized
    assert "source_text" not in serialized
    assert "text" not in first.manifest
    assert "surface" not in first.manifest
    assert first.manifest["summary"]["uncovered_required_rule_count"] == 0


def test_matrix_covers_every_canonical_label_for_a_selected_policy() -> None:
    matrix = build_policy_coverage_matrix(policies=("hipaa_safe_harbor",))

    assert {row.label for row in matrix.rows} == CANONICAL_LABELS
    assert all(row.action != "keep" for row in matrix.rows)
    assert all(row.required for row in matrix.rows)
