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


def test_confirmed_vault_withdrawal_erases_exact_dsar_selection(tmp_path):
    from openmed.compliance.dsar import VaultWithdrawalParticipant
    from openmed.compliance.withdrawal import ParticipantCode, WithdrawalCoordinator
    from openmed.core.surrogate_vault_crypto import SurrogateVaultCrypto

    path = tmp_path / "vault.json"
    vault = SurrogateVault.from_file(path, hmac_secret=SECRET)
    _seed(vault, "12345", "id_num", "synthetic-id")
    _seed(vault, "12345", "phone", "synthetic-phone")
    _seed(vault, "12345", "id_num", "synthetic-fr", lang="fr")
    ids = [SubjectIdentifier("12345", "id_num")]
    package = assemble_dsar_package(ids, vault)
    mirror = tmp_path / "encrypted.json"
    crypto = SurrogateVaultCrypto(bytes(range(32)))
    crypto.write(
        mirror,
        {"synthetic-id": "12345", "synthetic-phone": "12345", "synthetic-fr": "12345"},
    )
    participant = VaultWithdrawalParticipant(
        vault, ids, encrypted_mapping=(mirror, crypto)
    )
    coordinator = WithdrawalCoordinator({ParticipantCode.SURROGATE_VAULT: participant})
    preview = coordinator.plan(package.subject_ref)
    assert preview.participants[0][1].count == len(package.entries) == 1
    receipt = coordinator.withdraw(
        package.subject_ref, confirmation=preview.confirmation_token
    )
    assert receipt.state == "success" and receipt.participants[0].applied == 1
    assert "12345" not in str(receipt.to_dict()) and "synthetic-id" not in str(
        receipt.to_dict()
    )
    reopened = SurrogateVault.from_file(path, hmac_secret=SECRET, create=False)
    assert not assemble_dsar_package(ids, reopened).entries
    assert {e.surrogate for e in reopened.entries()} == {
        "synthetic-phone",
        "synthetic-fr",
    }
    assert crypto.read(mirror) == {"synthetic-phone": "12345", "synthetic-fr": "12345"}
    assert b"synthetic-phone" not in mirror.read_bytes()
    assert participant.verify(package.subject_ref)
    crypto.write(mirror, {"synthetic-id": "12345"})
    assert not participant.verify(package.subject_ref)


def test_vault_erasure_rejects_stale_selection_and_restores_failed_save(
    tmp_path, monkeypatch
):
    import pytest

    from openmed.compliance.dsar import VaultWithdrawalParticipant

    vault = SurrogateVault.from_file(tmp_path / "vault", hmac_secret=SECRET)
    ids = _subject_ids()
    package = assemble_dsar_package(ids, vault)
    participant = VaultWithdrawalParticipant(vault, ids)
    stale = participant.plan(package.subject_ref)
    _seed(vault, "MRN-12345", "id_num", "synthetic-id")
    with pytest.raises(ValueError, match="selection changed"):
        participant.apply(package.subject_ref, stale)

    def fail_write():
        raise OSError("fixture error")

    monkeypatch.setattr(vault.store, "save", fail_write)
    with pytest.raises(OSError):
        participant.apply(package.subject_ref, participant.plan(package.subject_ref))
    assert len(assemble_dsar_package(ids, vault).entries) == 1
