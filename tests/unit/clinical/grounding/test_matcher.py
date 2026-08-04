"""Tests for the shared offline lexical matcher."""

from __future__ import annotations

import json
import socket
from dataclasses import asdict

import pytest

from openmed.clinical.grounding.matcher import (
    ConceptMatch,
    LexicalConcept,
    LexicalMatcher,
    normalize_term,
)

SYSTEM_URI = "https://example.org/fhir/CodeSystem/synthetic-free"


def _matcher() -> LexicalMatcher:
    aspirin = LexicalConcept(
        system_uri=SYSTEM_URI,
        code="MED-001",
        display="Aspirin",
        metadata={"kind": "ingredient"},
    )
    return LexicalMatcher(
        {
            "Aspirin": aspirin,
            "acetylsalicylic acid": aspirin,
            "Complete blood count": {
                "code": "LAB-001",
                "display": "Complete blood count",
                "kind": "laboratory-test",
            },
            "type-2 diabetes": {
                "code": "DX-001",
                "display": "Type 2 diabetes mellitus",
            },
        },
        system_uri=SYSTEM_URI,
        abbreviations={"ASA": "acetylsalicylic acid"},
    )


def test_exact_and_normalized_queries_return_ranked_concept_matches():
    matcher = _matcher()

    exact = matcher.lookup("Aspirin")
    normalized = matcher.lookup("  ASPIRIN  ")
    punctuation_normalized = matcher.lookup("Type 2 diabetes")

    assert exact == (
        ConceptMatch(
            system_uri=SYSTEM_URI,
            code="MED-001",
            display="Aspirin",
            score=1.0,
            match_type="exact",
            matched_term="Aspirin",
            metadata={"kind": "ingredient"},
        ),
    )
    assert normalized[0].code == "MED-001"
    assert normalized[0].match_type == "normalized"
    assert normalized[0].score < exact[0].score
    assert punctuation_normalized[0].code == "DX-001"
    assert punctuation_normalized[0].match_type == "normalized"


def test_configured_and_automatic_abbreviations_resolve_offline():
    matcher = _matcher()

    configured = matcher.lookup("asa")
    automatic = matcher.lookup("CBC")

    assert configured[0].code == "MED-001"
    assert configured[0].matched_term == "acetylsalicylic acid"
    assert configured[0].match_type == "abbreviation"
    assert automatic[0].code == "LAB-001"
    assert automatic[0].match_type == "abbreviation"
    assert automatic[0].metadata["kind"] == "laboratory-test"
    assert json.loads(json.dumps(asdict(automatic[0])))["code"] == "LAB-001"


def test_matcher_ranks_match_kinds_and_deduplicates_concepts():
    matcher = LexicalMatcher(
        {
            "ASA": {"code": "OTHER", "display": "A synthetic exact term"},
            "acetylsalicylic acid": [
                {"code": "ASPIRIN", "display": "Aspirin"},
                {"code": "SECOND", "display": "Second candidate"},
            ],
            "Acetylsalicylic-acid": {
                "code": "ASPIRIN",
                "display": "Aspirin",
            },
        },
        system_uri=SYSTEM_URI,
        abbreviations={"ASA": "acetylsalicylic acid"},
    )

    matches = matcher.lookup("ASA")

    assert [match.code for match in matches] == ["OTHER", "ASPIRIN", "SECOND"]
    assert [match.score for match in matches] == [1.0, 0.9, 0.9]
    assert sum(match.code == "ASPIRIN" for match in matches) == 1
    assert matcher.lookup("ASA", limit=2) == matches[:2]


def test_bare_code_dictionary_uses_matcher_system_uri():
    matcher = LexicalMatcher({"synthetic term": "CODE-1"}, system_uri=SYSTEM_URI)

    assert matcher.lookup("synthetic term")[0].key == (SYSTEM_URI, "CODE-1")
    assert matcher.term_count == 1
    assert matcher.concept_count == 1


def test_lookup_path_never_opens_a_network_connection(monkeypatch):
    matcher = _matcher()

    def fail_connection(*args, **kwargs):
        raise AssertionError("lexical lookup attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", fail_connection)

    assert matcher.lookup("CBC")[0].code == "LAB-001"


def test_normalization_is_unicode_aware_and_input_validation_is_clear():
    assert normalize_term("  β-Blocker\u2014Dose  ") == "β blocker dose"
    assert normalize_term("తెలుగు") == "తెలుగు"

    with pytest.raises(ValueError, match="positive integer"):
        _matcher().lookup("aspirin", limit=0)
    with pytest.raises(ValueError, match="does not match an indexed"):
        LexicalMatcher(
            {"known term": "CODE-1"},
            system_uri=SYSTEM_URI,
            abbreviations={"KT": "missing expansion"},
        )
    with pytest.raises(ValueError, match="must not be empty"):
        LexicalMatcher({"known term": []}, system_uri=SYSTEM_URI)
