"""Tests for guarded procedure-to-indication candidates."""

import json

from openmed.clinical.relations.procedure_indications import (
    generate_procedure_indication_candidates,
)


def _span(text: str, value: str, label: str, **extra):
    start = text.index(value)
    return {"label": label, "start": start, "end": start + len(value), **extra}


def test_generates_section_scoped_evidence_with_assertion() -> None:
    text = "Plan: Biopsy performed for possible lung mass."
    spans = [
        _span(text, "Biopsy", "PROCEDURE"),
        _span(text, "lung mass", "CONDITION", certainty="uncertain"),
    ]

    (candidate,) = generate_procedure_indication_candidates(text, spans)

    assert candidate.indication_assertion.certainty == "uncertain"
    assert candidate.confirmation_required is True
    assert candidate.appropriateness_assessed is False
    payload = json.dumps(candidate.to_dict(), sort_keys=True)
    assert "Biopsy" not in payload
    assert "lung mass" not in payload


def test_rejects_cross_section_and_cueless_pairs() -> None:
    text = "Biopsy for mass."
    procedure = _span(text, "Biopsy", "PROCEDURE", section="plan")
    indication = _span(text, "mass", "CONDITION", section="history")
    assert generate_procedure_indication_candidates(text, [procedure, indication]) == ()

    cueless = "Biopsy and mass."
    assert (
        generate_procedure_indication_candidates(
            cueless,
            [
                _span(cueless, "Biopsy", "PROCEDURE"),
                _span(cueless, "mass", "CONDITION"),
            ],
        )
        == ()
    )
