"""Unit tests for the local-first clinical NLI verifier."""

from __future__ import annotations

import pytest

from openmed.clinical.nli import (
    MEDNLI_DATA_POLICY,
    NLI_LABELS,
    HeuristicNLIBackend,
    nli,
    verify,
)


def test_nli_returns_the_documented_two_field_shape() -> None:
    result = nli(
        "Synthetic patient has pneumonia.",
        "The patient has pneumonia.",
    )

    assert set(result) == {"label", "score"}
    assert result["label"] in NLI_LABELS
    assert 0.0 <= result["score"] <= 1.0
    assert result["label"] == "entailment"


def test_verify_flags_contradiction_and_retains_entailed_claim() -> None:
    results = verify(
        [
            "The patient has no pneumonia.",
            "The patient has pneumonia.",
        ],
        "Synthetic patient has pneumonia.",
    )

    assert [result["label"] for result in results] == [
        "contradiction",
        "entailment",
    ]
    assert results[0]["contradicted"] is True
    assert results[1]["contradicted"] is False
    assert [result["claim"] for result in results] == [
        "The patient has no pneumonia.",
        "The patient has pneumonia.",
    ]


def test_verify_pairs_aligned_source_spans() -> None:
    results = verify(
        ["No fever is present.", "Pneumonia is present."],
        ["Synthetic fever is present.", "Synthetic pneumonia is present."],
    )

    assert [result["label"] for result in results] == [
        "contradiction",
        "entailment",
    ]


def test_nli_accepts_a_swappable_backend_without_api_changes() -> None:
    calls: list[tuple[str, str]] = []

    class StubBackend:
        def predict(self, premise: str, hypothesis: str) -> dict[str, object]:
            calls.append((premise, hypothesis))
            return {"label": "neutral", "score": 0.25}

    result = nli("synthetic source", "synthetic claim", backend=StubBackend())

    assert result == {"label": "neutral", "score": 0.25}
    assert calls == [("synthetic source", "synthetic claim")]


def test_heuristic_backend_is_explicitly_dependency_free() -> None:
    assert isinstance(HeuristicNLIBackend(), HeuristicNLIBackend)
    assert "DUA-gated" in MEDNLI_DATA_POLICY
    assert "eval-only" in MEDNLI_DATA_POLICY
    assert "BigBio" in MEDNLI_DATA_POLICY


@pytest.mark.parametrize(
    ("premise", "hypothesis"),
    [("", "claim"), ("source", "")],
)
def test_nli_rejects_empty_text(premise: str, hypothesis: str) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        nli(premise, hypothesis)
