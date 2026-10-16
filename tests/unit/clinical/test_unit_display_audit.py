"""Focused tests for the deterministic unit-display normalization audit."""

from __future__ import annotations

import re

import pytest

from openmed.clinical.units.display_audit import (
    UNIT_DISPLAY_AUDIT_SCHEMA_VERSION,
    audit_unit_display_labels,
)

CANONICAL_CODES = ("Cel", "g/L", "mg/dL")


def test_synthetic_multilingual_labels_pass_against_explicit_aliases() -> None:
    report = audit_unit_display_labels(
        {
            "fr-FR": {
                "Cel": "Cel",
                "g/L": "g/L",
                "mg/dL": "mg par décilitre",
            },
            "en": {
                "Cel": "Celsius",
                "g/L": "g/L",
                "mg/dL": "mg/dL",
            },
        },
        CANONICAL_CODES,
        {
            "en": {"Celsius": "Cel"},
            "fr": {"mg par décilitre": "mg/dL"},
        },
    )

    assert report.passed is True
    assert report.to_dict() == {
        "canonical_unit_count": 3,
        "display_label_count": 6,
        "issues": [],
        "locales": ["en", "fr"],
        "passed": True,
        "repro_hash": report.repro_hash,
        "schema_version": UNIT_DISPLAY_AUDIT_SCHEMA_VERSION,
        "status": "pass",
        "summary": {
            "canonical_units": 3,
            "conflict": 0,
            "display_labels": 6,
            "duplicate": 0,
            "issues": 0,
            "locales": 2,
            "missing": 0,
        },
    }
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", report.repro_hash)


def test_default_registered_aliases_are_local_and_deterministic() -> None:
    report = audit_unit_display_labels(
        {"es": {"mg/dL": "mg por decilitro"}},
        ("mg/dL",),
    )

    assert report.passed is True
    assert report.summary == {
        "canonical_units": 1,
        "conflict": 0,
        "display_labels": 1,
        "duplicate": 0,
        "issues": 0,
        "locales": 1,
        "missing": 0,
    }


def test_missing_duplicate_and_conflicting_labels_are_aggregated() -> None:
    report = audit_unit_display_labels(
        {
            "es": {"mg/dL": "mg por decilitro"},
            "zz": {
                "Cel": "wrong-display",
                "g/L": "shared-display",
                "mg/dL": "shared-display",
            },
        },
        CANONICAL_CODES,
        {
            "es": {"mg por decilitro": "mg/dL"},
            "zz": {
                "shared-display": "mg/dL",
                "wrong-display": "Cel",
            },
        },
    )

    assert report.passed is False
    assert report.summary == {
        "canonical_units": 3,
        "conflict": 1,
        "display_labels": 4,
        "duplicate": 2,
        "issues": 3,
        "locales": 2,
        "missing": 2,
    }
    assert {(issue.kind, issue.reason) for issue in report.issues} == {
        ("missing", "missing_display_label"),
        ("duplicate", "display_label_used_for_multiple_codes"),
        ("conflict", "alias_resolves_to_different_code"),
    }
    assert all(
        digest.startswith("sha256:")
        for issue in report.issues
        for digest in (
            *issue.label_hashes,
            *issue.canonical_code_hashes,
            *issue.resolved_code_hashes,
        )
    )

    serialized = report.to_json() + report.to_markdown()
    for raw_value in (
        "mg por decilitro",
        "wrong-display",
        "shared-display",
        "Cel",
        "g/L",
        "mg/dL",
    ):
        assert raw_value not in serialized


def test_alias_table_collisions_are_conflicts_without_echoing_labels() -> None:
    report = audit_unit_display_labels(
        {"xx": {"g/L": "g/L", "mg/dL": "ambiguous-display"}},
        ("mg/dL", "g/L"),
        {
            "xx": {
                "AMBIGUOUS-DISPLAY": "g/L",
                "ambiguous-display": "mg/dL",
            }
        },
    )

    assert report.summary["conflict"] == 2
    assert {issue.reason for issue in report.issues} == {"alias_table_conflict"}
    assert "ambiguous-display" not in report.to_json()


def test_input_order_does_not_change_report_or_repro_hash(caplog) -> None:
    labels = {
        "fr": {"mg/dL": "mg par décilitre", "g/L": "g/L"},
        "en": {"g/L": "g/L", "mg/dL": "mg/dL"},
    }
    aliases = {
        "fr": {"mg par décilitre": "mg/dL"},
        "en": {},
    }
    reordered_labels = {
        "en": dict(reversed(tuple(labels["en"].items()))),
        "fr": dict(reversed(tuple(labels["fr"].items()))),
    }
    reordered_aliases = {
        "en": {},
        "fr": {"mg par décilitre": "mg/dL"},
    }

    first = audit_unit_display_labels(labels, CANONICAL_CODES, aliases)
    second = audit_unit_display_labels(
        reordered_labels,
        tuple(reversed(CANONICAL_CODES)),
        reordered_aliases,
    )

    assert first.to_json() == second.to_json()
    assert first.repro_hash == second.repro_hash
    assert caplog.records == []


@pytest.mark.parametrize(
    "update",
    [
        {"reason": "SYNTHETIC-PRIVATE-LABEL"},
        {"locale": "SYNTHETIC-PRIVATE-LABEL"},
        {"label_hashes": ("SYNTHETIC-PRIVATE-LABEL",)},
    ],
)
def test_direct_issue_rejects_uncontrolled_output_fields(update):
    from openmed.clinical.units.display_audit import UnitDisplayAuditIssue

    values = dict(kind="missing", reason="missing_display_label", locale="en", count=1)
    values.update(update)
    with pytest.raises(ValueError):
        UnitDisplayAuditIssue(**values)


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": "SYNTHETIC-PRIVATE-LABEL"},
        {"repro_hash": "sha256:" + "0" * 64},
        {"locales": ("SYNTHETIC-PRIVATE-LABEL",)},
    ],
)
def test_report_rejects_forged_metadata(update):
    from openmed.clinical.units.display_audit import UnitDisplayAuditReport

    values = dict(locales=("en",), canonical_unit_count=1, display_label_count=1)
    values.update(update)
    with pytest.raises(ValueError):
        UnitDisplayAuditReport(**values)


def test_json_indent_cannot_emit_private_content():
    report = audit_unit_display_labels({"en": {"mg/dL": "mg/dL"}}, ("mg/dL",), {})
    with pytest.raises(ValueError):
        report.to_json(indent="SYNTHETIC-PRIVATE-LABEL")


def test_bad_locale_cannot_escape_into_report():
    with pytest.raises(ValueError):
        audit_unit_display_labels({"private patient text": {}}, ("mg/dL",), {})


def test_iterator_error_does_not_retain_source_context():
    def codes():
        raise RuntimeError("SYNTHETIC-PRIVATE-LABEL")
        yield

    with pytest.raises(ValueError) as caught:
        audit_unit_display_labels({"en": {}}, codes(), {})
    assert caught.value.__context__ is None
    assert "SYNTHETIC" not in str(caught.value)


def test_audit_collections_are_bounded(monkeypatch):
    from openmed.clinical.units import display_audit as module

    monkeypatch.setattr(module, "_MAX_ITEMS", 2)
    with pytest.raises(ValueError):
        audit_unit_display_labels({"en": {}}, (str(i) for i in range(3)), {})


def test_nonempty_canonical_catalog_requires_at_least_one_locale():
    with pytest.raises(ValueError):
        audit_unit_display_labels({}, ("mg/dL",), {})
