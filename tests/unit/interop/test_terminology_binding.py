from __future__ import annotations

from pathlib import Path

import pytest

from openmed.clinical.exporters.codeable_concept import SYSTEM_URI
from openmed.clinical.grounding import Candidate, GroundedSpan
from openmed.interop.athena import load_athena_vocab
from openmed.interop.terminology_binding import (
    RestrictedVocabularyBindingError,
    bind_codeable_concept,
)

FIXTURES = Path(__file__).with_name("fixtures")


def _athena_index():
    return load_athena_vocab(FIXTURES)


def test_present_code_binds_canonical_system_and_concept_display() -> None:
    span = GroundedSpan(
        text="synthetic medicine beta",
        start=0,
        end=24,
        candidates=(Candidate("RXTEST", "RX-200", "source display", 1.0),),
    )

    result = bind_codeable_concept(
        span,
        _athena_index(),
        vocabularies={"RXTEST": "RXNORM"},
    )

    assert result == {
        "coding": [
            {
                "system": SYSTEM_URI["RXNORM"],
                "code": "RX-200",
                "display": "Example medicine beta",
            }
        ],
        "text": "synthetic medicine beta",
    }


def test_absent_code_is_text_only_and_recorded_as_a_phi_free_miss() -> None:
    index = _athena_index()
    span = GroundedSpan(
        text="unlisted synthetic medicine",
        start=0,
        end=28,
        candidates=(Candidate("RXTEST", "RX-404", "ignored", 1.0),),
    )

    result = bind_codeable_concept(
        span,
        index,
        vocabularies={"RXTEST": "RXNORM"},
    )

    assert result == {"text": "unlisted synthetic medicine"}
    assert index["_meta"]["binding_misses"] == [
        {"vocabulary_id": "RXTEST", "code": "RX-404", "reason": "not_found"}
    ]


def test_restricted_vocabulary_requires_explicit_user_opt_in() -> None:
    index = _athena_index()
    index["CPT4"] = {
        "99213": {
            "concept_name": "Synthetic office visit",
            "vocabulary_id": "CPT4",
            "concept_code": "99213",
            "system_uri": "http://www.ama-assn.org/go/cpt",
        }
    }
    span = {"text": "synthetic office visit", "code": "99213", "system": "CPT4"}

    with pytest.raises(RestrictedVocabularyBindingError, match="CPT4"):
        bind_codeable_concept(span, index)

    result = bind_codeable_concept(span, index, vocabularies={"CPT4"})
    assert result["coding"][0]["system"] == "http://www.ama-assn.org/go/cpt"


def test_bundled_athena_provenance_is_rejected() -> None:
    index = _athena_index()
    index["_meta"]["provenance"]["bundled"] = True

    with pytest.raises(ValueError, match="user-supplied"):
        bind_codeable_concept(
            {"text": "synthetic medicine beta", "code": "RX-200", "system": "RXTEST"},
            index,
            vocabularies={"RXTEST": "RXNORM"},
        )
