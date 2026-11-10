"""Tests for expert-authored de-identification attestation envelopes."""

from __future__ import annotations

import builtins
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from openmed.compliance import (
    EXPERT_ATTESTATION_CANONICALIZATION,
    EXPERT_ATTESTATION_DISCLAIMER,
    ExpertAttestationEnvelope,
    ReleaseAssumptions,
    build_release_expert_review_evidence,
    create_expert_attestation,
)
from openmed.core.audit import stable_hash
from openmed.risk import (
    AnonymityPolicy,
    anonymize_release,
    validate_released_output,
)

_ISSUED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
_REASSESSMENT_AT = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)


def _digest(name: str) -> str:
    return stable_hash({"kind": "expert-attestation-test", "name": name})


@pytest.fixture
def evidence():
    rows = [
        {"patient_id": "patient-a", "patient_name": "A Canary", "age": 30},
        {"patient_id": "patient-b", "patient_name": "B Canary", "age": 30},
    ]
    result = anonymize_release(
        rows,
        AnonymityPolicy(
            quasi_identifiers=("age",),
            direct_identifiers=("patient_name",),
            privacy_unit="patient_id",
            target_k=2,
        ),
    )
    return build_release_expert_review_evidence(
        result,
        validation=validate_released_output(result.records, result),
        assumptions=ReleaseAssumptions(
            privacy_unit="patient",
            population_scope="release_cohort",
            release_model="restricted",
            recipient_model="named_researchers",
            auxiliary_data_model="reasonably_available",
            notes_digest=_digest("reviewed-assumptions"),
        ),
    )


@pytest.fixture
def ed25519_keys() -> tuple[Any, Any]:
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def _attestation(
    evidence: Any,
    private_key: Any,
    *,
    expert_identity: str = "Dr. Taylor Example",
    conclusion: str = "very_small_risk",
    supporting_evidence_digests: (dict[str, str] | tuple[tuple[str, str], ...]) = (),
) -> ExpertAttestationEnvelope:
    return create_expert_attestation(
        evidence,
        expert_identity=expert_identity,
        qualifications="Independent statistical disclosure-control expert",
        scope_and_methodology=(
            "Reviewed the declared recipient, release context, population, "
            "technical transformations, residual risk, and supporting evidence."
        ),
        conclusion=conclusion,
        issued_at=_ISSUED_AT,
        reassessment_at=_REASSESSMENT_AT,
        private_key=private_key,
        key_id="expert-key-2026",
        supporting_evidence_digests=supporting_evidence_digests,
    )


def test_expert_authored_attestation_round_trips_and_verifies_separate_facts(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, public_key = ed25519_keys
    supporting = {
        "composition_review": _digest("composition"),
        "population_risk": _digest("population"),
    }
    report = _attestation(
        evidence,
        private_key,
        supporting_evidence_digests=supporting,
    )

    reparsed = ExpertAttestationEnvelope.from_json(report.to_json())
    verification = reparsed.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert reparsed == report
    assert verification.cryptographically_valid is True
    assert verification.key_id_matches is True
    assert verification.evidence_integrity_valid is True
    assert verification.bindings_match is True
    assert verification.conclusion == "very_small_risk"
    assert verification.freshness_status == "current"
    assert verification.fresh is True
    with pytest.raises(TypeError, match="no combined truth value"):
        bool(verification)
    assert "approved" not in verification.to_dict()
    assert "automated Expert Determination" in EXPERT_ATTESTATION_DISCLAIMER
    payload = report.to_dict()
    assert set(payload) == {
        "schema_version",
        "report_type",
        "disclaimer",
        "expert",
        "conclusion",
        "issued_at",
        "reassessment_at",
        "bindings",
        "supporting_evidence_digests",
        "signature",
    }
    assert payload["supporting_evidence_digests"] == dict(sorted(supporting.items()))
    assert (
        payload["signature"]["canonicalization"]
        == EXPERT_ATTESTATION_CANONICALIZATION
        == "RFC8785"
    )
    assert payload["bindings"] == {
        "evidence_integrity_hash": evidence.integrity_hash,
        "source_dataset_digest": evidence.digests.source_dataset,
        "released_dataset_digest": evidence.digests.dataset,
        "released_schema_digest": evidence.digests.schema,
        "policy_digest": evidence.digests.policy,
        "hierarchy_digest": evidence.digests.hierarchy,
        "config_digest": evidence.digests.config,
        "software_digest": evidence.digests.software,
    }
    serialized = report.to_json()
    assert "PRIVATE KEY" not in serialized
    assert "PUBLIC KEY" not in serialized
    assert "patient-a" not in serialized
    assert "A Canary" not in serialized


def test_rfc8785_canonicalization_handles_valid_unicode_and_rejects_surrogates(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, public_key = ed25519_keys
    report = _attestation(
        evidence,
        private_key,
        expert_identity="Dr. Élodie 李",
    )

    verification = ExpertAttestationEnvelope.from_json(report.to_json()).verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert verification.cryptographically_valid is True
    with pytest.raises(ValueError, match="Unicode surrogate"):
        _attestation(
            evidence,
            private_key,
            expert_identity="Dr. \ud800",
        )
    for invisible in ("\u200b", "\u2028", "\u2029", "\u202e"):
        with pytest.raises(ValueError, match="invisible"):
            _attestation(
                evidence,
                private_key,
                expert_identity=f"Dr. Taylor{invisible} Example",
            )


@pytest.mark.parametrize("conclusion", ["not_approved", "requires_changes"])
def test_valid_signature_never_converts_expert_conclusion_into_approval(
    conclusion: str,
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, public_key = ed25519_keys
    report = _attestation(evidence, private_key, conclusion=conclusion)

    verification = report.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert verification.cryptographically_valid is True
    assert verification.conclusion == conclusion
    assert "approved" not in verification.to_dict()


def test_verification_detects_tampering_key_and_binding_mismatches_independently(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )

    private_key, public_key = ed25519_keys
    supporting = {"population_risk": _digest("population")}
    report = _attestation(
        evidence,
        private_key,
        supporting_evidence_digests=supporting,
    )
    payload = report.to_dict()
    payload["conclusion"] = "requires_changes"
    tampered = ExpertAttestationEnvelope.from_dict(payload)
    binding_payload = report.to_dict()
    binding_payload["bindings"]["released_dataset_digest"] = _digest("tampered-release")
    binding_tampered = ExpertAttestationEnvelope.from_dict(binding_payload)
    support_payload = report.to_dict()
    support_payload["supporting_evidence_digests"]["population_risk"] = _digest(
        "tampered-support"
    )
    support_tampered = ExpertAttestationEnvelope.from_dict(support_payload)
    key_id_payload = report.to_dict()
    key_id_payload["signature"]["key_id"] = "attacker-key"
    key_id_tampered = ExpertAttestationEnvelope.from_dict(key_id_payload)

    tampered_result = tampered.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    wrong_key_id = report.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="different-key",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    wrong_public_key = report.verify(
        evidence=evidence,
        public_key=Ed25519PrivateKey.generate().public_key(),
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    binding_tampered_result = binding_tampered.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    support_tampered_result = support_tampered.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    key_id_tampered_result = key_id_tampered.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="attacker-key",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    missing_support = report.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    invalid_evidence = replace(
        evidence,
        integrity_hash=_digest("invalid-evidence-integrity"),
    )
    invalid_evidence_result = report.verify(
        evidence=invalid_evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        expected_supporting_evidence_digests=supporting,
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )

    assert tampered_result.cryptographically_valid is False
    assert tampered_result.conclusion == "requires_changes"
    assert tampered_result.bindings_match is True
    with pytest.raises(TypeError, match="no combined truth value"):
        bool(tampered_result)
    assert wrong_key_id.cryptographically_valid is True
    assert wrong_key_id.key_id_matches is False
    assert wrong_public_key.cryptographically_valid is False
    assert wrong_public_key.key_id_matches is True
    assert binding_tampered_result.cryptographically_valid is False
    assert binding_tampered_result.bindings_match is False
    assert support_tampered_result.cryptographically_valid is False
    assert support_tampered_result.bindings_match is False
    assert key_id_tampered_result.cryptographically_valid is False
    assert key_id_tampered_result.key_id_matches is True
    assert missing_support.cryptographically_valid is True
    assert missing_support.bindings_match is False
    assert invalid_evidence_result.evidence_integrity_valid is False
    assert invalid_evidence_result.bindings_match is False


def test_verification_reports_not_yet_valid_and_reassessment_due(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, public_key = ed25519_keys
    report = _attestation(evidence, private_key)

    early = report.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        as_of=_ISSUED_AT - timedelta(seconds=1),
    )
    due = report.verify(
        evidence=evidence,
        public_key=public_key,
        expected_key_id="expert-key-2026",
        as_of=_REASSESSMENT_AT,
    )

    assert early.cryptographically_valid is True
    assert early.freshness_status == "not_yet_valid"
    assert early.fresh is False
    assert due.cryptographically_valid is True
    assert due.freshness_status == "reassessment_due"
    assert due.fresh is False


def test_creation_requires_verified_evidence_and_explicit_expert_inputs(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, _public_key = ed25519_keys
    invalid_evidence = replace(evidence, integrity_hash=_digest("tampered"))

    with pytest.raises(ValueError, match="evidence must verify"):
        _attestation(invalid_evidence, private_key)
    with pytest.raises(ValueError, match="expert_identity"):
        create_expert_attestation(
            evidence,
            expert_identity="",
            qualifications="Qualified",
            scope_and_methodology="Reviewed scope and methodology",
            conclusion="very_small_risk",
            issued_at=_ISSUED_AT,
            reassessment_at=_REASSESSMENT_AT,
            private_key=private_key,
            key_id="expert-key-2026",
        )
    with pytest.raises(ValueError, match="allowed coded value"):
        create_expert_attestation(
            evidence,
            expert_identity="Dr. Taylor Example",
            qualifications="Qualified",
            scope_and_methodology="Reviewed scope and methodology",
            conclusion="approved",
            issued_at=_ISSUED_AT,
            reassessment_at=_REASSESSMENT_AT,
            private_key=private_key,
            key_id="expert-key-2026",
        )
    with pytest.raises(ValueError, match="issued_at must be timezone-aware"):
        create_expert_attestation(
            evidence,
            expert_identity="Dr. Taylor Example",
            qualifications="Qualified",
            scope_and_methodology="Reviewed scope and methodology",
            conclusion="not_approved",
            issued_at=_ISSUED_AT.replace(tzinfo=None),
            reassessment_at=_REASSESSMENT_AT,
            private_key=private_key,
            key_id="expert-key-2026",
        )
    with pytest.raises(ValueError, match="issued_at must use UTC"):
        create_expert_attestation(
            evidence,
            expert_identity="Dr. Taylor Example",
            qualifications="Qualified",
            scope_and_methodology="Reviewed scope and methodology",
            conclusion="not_approved",
            issued_at=datetime(
                2026,
                1,
                1,
                13,
                0,
                tzinfo=timezone(timedelta(hours=1)),
            ),
            reassessment_at=_REASSESSMENT_AT,
            private_key=private_key,
            key_id="expert-key-2026",
        )
    with pytest.raises(ValueError, match="later than"):
        create_expert_attestation(
            evidence,
            expert_identity="Dr. Taylor Example",
            qualifications="Qualified",
            scope_and_methodology="Reviewed scope and methodology",
            conclusion="requires_changes",
            issued_at=_ISSUED_AT,
            reassessment_at=_ISSUED_AT,
            private_key=private_key,
            key_id="expert-key-2026",
        )


def test_attestation_strict_json_rejects_unknown_duplicate_and_unsafe_metadata(
    evidence: Any,
    ed25519_keys: tuple[Any, Any],
) -> None:
    private_key, _public_key = ed25519_keys
    report = _attestation(evidence, private_key)

    unknown = report.to_dict()
    unknown["unexpected"] = "value"
    with pytest.raises(ValueError, match="documented fields"):
        ExpertAttestationEnvelope.from_dict(unknown)

    nested_unknown = report.to_dict()
    nested_unknown["expert"]["unexpected"] = "value"
    with pytest.raises(ValueError, match="documented fields"):
        ExpertAttestationEnvelope.from_dict(nested_unknown)

    compact = report.to_json(indent=None)
    duplicate = '{"schema_version":1,' + compact[1:]
    with pytest.raises(ValueError, match="malformed"):
        ExpertAttestationEnvelope.from_json(duplicate)
    non_finite = compact.replace('"schema_version":1', '"schema_version":NaN', 1)
    with pytest.raises(ValueError, match="malformed"):
        ExpertAttestationEnvelope.from_json(non_finite)

    with pytest.raises(ValueError, match="safe metadata identifier"):
        _attestation(
            evidence,
            private_key,
            supporting_evidence_digests={"unsafe name": _digest("support")},
        )
    with pytest.raises(ValueError, match="canonical sha256"):
        _attestation(
            evidence,
            private_key,
            supporting_evidence_digests={"population_risk": "not-a-digest"},
        )
    with pytest.raises(ValueError, match="unique"):
        _attestation(
            evidence,
            private_key,
            supporting_evidence_digests=(
                ("population_risk", _digest("one")),
                ("population_risk", _digest("two")),
            ),
        )


def test_cryptography_dependency_error_is_lazy_and_actionable(
    monkeypatch: pytest.MonkeyPatch,
    evidence: Any,
) -> None:
    original_import = builtins.__import__

    def reject_cryptography(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("cryptography"):
            raise ImportError("simulated missing dependency")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_cryptography)

    with pytest.raises(ImportError, match=r"openmed\[integrity\]"):
        _attestation(evidence, b"\x00" * 32)
