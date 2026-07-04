"""Tests for the GDPR DSAR subject-access export helper."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    DsarPackage,
    ErasurePlan,
    HashChainAuditLog,
    SubjectIdentifier,
    assemble_dsar_package,
    plan_erasure,
    render_dsar_summary,
)
from openmed.core.surrogate_vault import SurrogateVault

SECRET = "unit-test-secret"


def _seed(
    vault: SurrogateVault,
    surface: str,
    label: str,
    surrogate: str,
    *,
    lang: str = "en",
) -> None:
    key = vault.key_for(surface, label=label, lang=lang)
    vault.store.set(key, surrogate)


def _subject_vault() -> SurrogateVault:
    vault = SurrogateVault.in_memory(SECRET)
    _seed(vault, "John Smith", "first_name", "Robert Jones")
    _seed(vault, "MRN-12345", "id_num", "MRN-90001")
    # An unrelated data subject that must never appear in the package.
    _seed(vault, "Jane Doe", "first_name", "Alice Brown")
    return vault


def _subject_ids() -> list[SubjectIdentifier]:
    return [
        SubjectIdentifier("John Smith", "first_name"),
        SubjectIdentifier("MRN-12345", "id_num"),
    ]


# --------------------------------------------------------------------------
# Assembly: only the requested subject's data
# --------------------------------------------------------------------------


def test_package_gathers_only_the_requested_subject_entries():
    vault = _subject_vault()

    package = assemble_dsar_package(_subject_ids(), vault)

    assert isinstance(package, DsarPackage)
    surrogates = {entry.surrogate for entry in package.entries}
    assert surrogates == {"Robert Jones", "MRN-90001"}
    # The unrelated subject's surrogate is absent.
    assert "Alice Brown" not in surrogates


def test_package_matches_identifier_label_and_language():
    vault = SurrogateVault.in_memory(SECRET)
    _seed(vault, "12345", "id_num", "MRN-90001", lang="en")
    _seed(vault, "12345", "phone", "555-0100", lang="en")
    _seed(vault, "12345", "id_num", "MRN-FR-90001", lang="fr")

    package = assemble_dsar_package(
        [SubjectIdentifier("12345", "id_num", lang="en")], vault
    )

    assert {entry.surrogate for entry in package.entries} == {"MRN-90001"}


def test_package_reports_held_categories():
    vault = _subject_vault()

    package = assemble_dsar_package(_subject_ids(), vault)

    # Categories are the canonical labels held for the subject.
    assert set(package.categories) == {
        entry.canonical_label for entry in package.entries
    }
    assert len(package.categories) == 2


def test_unmatched_identifier_yields_empty_package():
    vault = _subject_vault()

    package = assemble_dsar_package(
        [SubjectIdentifier("Nobody Here", "first_name")], vault
    )

    assert package.entries == ()
    assert package.categories == ()


def test_subject_ref_is_deterministic_and_not_raw_phi():
    vault = _subject_vault()

    first = assemble_dsar_package(_subject_ids(), vault)
    second = assemble_dsar_package(_subject_ids(), vault)

    assert first.subject_ref == second.subject_ref
    assert "John Smith" not in first.subject_ref
    assert "MRN-12345" not in first.subject_ref


# --------------------------------------------------------------------------
# Audit logging
# --------------------------------------------------------------------------


def test_export_is_logged_to_the_audit_sink():
    vault = _subject_vault()
    log = HashChainAuditLog()

    package = assemble_dsar_package(_subject_ids(), vault, audit_sink=log)

    assert len(log.records) == 1
    record = log.records[0]
    assert record.event_type == "dsar.export"
    assert package.audit_record is record
    assert log.verify() is True


def test_audit_payload_carries_no_raw_phi():
    vault = _subject_vault()
    log = HashChainAuditLog()

    assemble_dsar_package(_subject_ids(), vault, audit_sink=log)

    blob = json.dumps(log.records[0].payload)
    assert "John Smith" not in blob
    assert "MRN-12345" not in blob
    assert "Robert Jones" not in blob


def test_audit_references_are_included():
    vault = _subject_vault()

    package = assemble_dsar_package(
        _subject_ids(), vault, audit_references=["run-2026-07-01", "run-2026-07-02"]
    )

    assert package.audit_references == ("run-2026-07-01", "run-2026-07-02")


# --------------------------------------------------------------------------
# Human-readable summary
# --------------------------------------------------------------------------


def test_summary_is_readable_and_counts_holdings():
    vault = _subject_vault()
    package = assemble_dsar_package(_subject_ids(), vault)

    summary = render_dsar_summary(package)

    assert isinstance(summary, str)
    assert "2" in summary  # two entries held
    assert package.subject_ref in summary


# --------------------------------------------------------------------------
# Article 17 deletion companion (non-destructive)
# --------------------------------------------------------------------------


def test_erasure_plan_lists_erasable_records_without_deleting():
    vault = _subject_vault()
    package = assemble_dsar_package(_subject_ids(), vault)
    before = len(vault.entries())

    plan = plan_erasure(package, vault)

    assert isinstance(plan, ErasurePlan)
    assert plan.executed is False
    assert {e.surrogate for e in plan.erasable} == {"Robert Jones", "MRN-90001"}
    # The vault is untouched: this is a companion, not a deletion.
    assert len(vault.entries()) == before


def test_erasure_preview_is_audit_logged_non_destructively():
    vault = _subject_vault()
    log = HashChainAuditLog()
    package = assemble_dsar_package(_subject_ids(), vault)

    plan_erasure(package, vault, audit_sink=log)

    assert log.records[0].event_type == "dsar.erasure_preview"
    assert log.records[0].payload.get("destructive") is False
