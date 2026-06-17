"""Tests for the deterministic post-ML PII safety sweep."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from openmed.core.anonymizer.providers import clinical_ids
from openmed.core.labels import ID_SUBTYPE_NPI
from openmed.core.pii_entity_merger import PIIPattern
from openmed.core.pii import deidentify
from openmed.core.quality_gates import detect_overlapping_entities
from openmed.core.safety_sweep import (
    SAFETY_SWEEP_PATTERNS_VERSION,
    SAFETY_SWEEP_SOURCE,
    safety_sweep,
)
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _entity_for(text: str, value: str, label: str = "MODEL") -> EntityPrediction:
    start = text.index(value)
    return EntityPrediction(
        text=value,
        label=label,
        start=start,
        end=start + len(value),
        confidence=0.99,
        metadata={"source": "model"},
    )


def _swept_by_label(entities):
    return {
        entity.label: entity
        for entity in entities
        if (entity.metadata or {}).get("source") == SAFETY_SWEEP_SOURCE
    }


def test_safety_sweep_recovers_ml_missed_deterministic_identifiers():
    text = (
        "Card 4111 1111 1111 1111. "
        "IBAN GB82 WEST 1234 5698 7654 32. "
        "SSN 123-45-6789. "
        "Email jane.patient@example.com. "
        "Phone 415-555-2671."
    )

    entities = safety_sweep(text, [])
    swept = _swept_by_label(entities)

    assert "credit_debit_card" in swept
    assert "iban" in swept
    assert "ssn" in swept
    assert "email" in swept
    assert "phone_number" in swept
    assert swept["iban"].metadata["patterns_version"] == SAFETY_SWEEP_PATTERNS_VERSION
    assert swept["email"].metadata["source"] == SAFETY_SWEEP_SOURCE


def test_safety_sweep_rejects_invalid_checksum_identifiers():
    text = (
        "Card 4111 1111 1111 1112. "
        "IBAN GB83 WEST 1234 5698 7654 32."
    )

    swept = _swept_by_label(safety_sweep(text, []))

    assert "credit_debit_card" not in swept
    assert "iban" not in swept


def test_safety_sweep_delegates_checksum_validation_to_clinical_ids(monkeypatch):
    luhn_calls = []
    iban_calls = []

    def fake_luhn(value: str) -> bool:
        luhn_calls.append(value)
        return True

    def fake_iban(value: str) -> bool:
        iban_calls.append(value)
        return False

    monkeypatch.setattr(clinical_ids, "validate_luhn", fake_luhn)
    monkeypatch.setattr(clinical_ids, "validate_iban", fake_iban)

    text = "Card 1234 5678 9012 3456. IBAN GB82 WEST 1234 5698 7654 32."
    swept = _swept_by_label(safety_sweep(text, []))

    assert luhn_calls == ["1234 5678 9012 3456"]
    assert iban_calls == ["GB82 WEST 1234 5698 7654 32"]
    assert "credit_debit_card" in swept
    assert "iban" not in swept


def test_safety_sweep_includes_id_subtype_metadata_for_id_matches():
    text = "Provider NPI TEST-12345"
    pattern = PIIPattern(
        r"\bTEST-\d{5}\b",
        "npi",
        priority=1,
        base_score=0.8,
        context_words=["npi"],
        context_boost=0.1,
    )

    swept = _swept_by_label(safety_sweep(text, [], patterns=[pattern]))

    assert swept["npi"].metadata["id_subtype"] == ID_SUBTYPE_NPI
    assert swept["npi"].metadata["safety_sweep"]["id_subtype"] == ID_SUBTYPE_NPI


def test_safety_sweep_never_adds_spans_overlapping_model_spans():
    text = "Card 4111 1111 1111 1111. Email jane.patient@example.com."
    model_card = _entity_for(text, "4111 1111 1111 1111", label="ID_NUM")

    entities = safety_sweep(text, [model_card])
    swept = _swept_by_label(entities)

    assert "credit_debit_card" not in swept
    assert "email" in swept
    for entity in swept.values():
        assert entity.start >= model_card.end or entity.end <= model_card.start


def test_safety_sweep_resolves_overlapping_existing_spans():
    text = "Patient SSN 123-45-6789 was verified."
    broad = EntityPrediction(
        text="Patient SSN 123-45-6789",
        label="OTHER",
        start=0,
        end=24,
        confidence=0.99,
    )
    sensitive = EntityPrediction(
        text="123-45-6789",
        label="SSN",
        start=12,
        end=23,
        confidence=0.50,
    )

    entities = safety_sweep(text, [broad, sensitive])

    assert detect_overlapping_entities(entities) == []
    assert [entity.label for entity in entities] == ["SSN"]


@patch("openmed.core.pii.extract_pii")
def test_deidentify_runs_safety_sweep_by_default_for_ml_misses(mock_extract):
    text = "Email: jane.patient@example.com"
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask")

    assert result.deidentified_text == "Email: [email]"
    assert result.metadata["safety_sweep"]["spans_added"] == 1
    assert result.pii_entities[0].metadata["source"] == SAFETY_SWEEP_SOURCE
    assert result.to_dict()["pii_entities"][0]["metadata"]["patterns_version"] == (
        SAFETY_SWEEP_PATTERNS_VERSION
    )


@patch("openmed.core.pii.extract_pii")
def test_deidentify_can_disable_safety_sweep(mock_extract):
    text = "Email: jane.patient@example.com"
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert result.deidentified_text == text
    assert result.pii_entities == []


@patch("openmed.core.pii.extract_pii")
def test_deidentify_resolves_overlaps_before_redaction(mock_extract):
    text = "Patient SSN 123-45-6789 was verified."
    mock_extract.return_value = PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text="Patient SSN 123-45-6789",
                label="OTHER",
                start=0,
                end=24,
                confidence=0.99,
            ),
            EntityPrediction(
                text="123-45-6789",
                label="SSN",
                start=12,
                end=23,
                confidence=0.50,
            ),
        ],
        model_name="stub",
        timestamp=datetime.now().isoformat(),
    )

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert detect_overlapping_entities(result.pii_entities) == []
    assert [entity.label for entity in result.pii_entities] == ["SSN"]
    assert result.deidentified_text == "Patient SSN [SSN] was verified."
