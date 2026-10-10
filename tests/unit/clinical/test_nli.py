"""Unit tests for the local-first clinical NLI verifier."""

from __future__ import annotations

import pytest

from openmed.clinical.nli import (
    MEDNLI_DATA_POLICY,
    NLI_LABELS,
    ClaimVerification,
    HeuristicNLIBackend,
    nli,
    verify,
)


def test_hook_verification_record_is_closed_and_value_free():
    record = ClaimVerification(
        0,
        "contradiction",
        0.9,
        "heuristic",
        (0, 10),
        (0, 8),
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        True,
    )
    assert ClaimVerification.from_dict(record.to_dict()) == record
    payload = record.to_dict()
    payload["source"] = "Synthetic private source"
    with pytest.raises(
        ValueError, match="invalid claim verification metadata"
    ) as raised:
        ClaimVerification.from_dict(payload)
    assert "private source" not in str(raised.value)


@pytest.mark.parametrize(
    "override",
    [
        {"score": True},
        {"score": float("nan")},
        {"score": 10**500},
        {"claim_index": False},
        {"backend_id": "/private/synthetic"},
        {"source_offset": [0, True]},
        {"source_digest": None},
        {"contradicted": True},
        {"review_required": False, "label": "abstention"},
    ],
)
def test_hook_record_rejects_malformed_or_forged_metadata(override):
    record = ClaimVerification(
        0,
        "entailment",
        0.9,
        "heuristic",
        (0, 10),
        None,
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
        False,
    )
    payload = record.to_dict() | override
    with pytest.raises(ValueError, match="invalid claim verification metadata"):
        ClaimVerification.from_dict(payload)


def test_nli_returns_the_documented_value_free_shape() -> None:
    result = nli(
        "Synthetic patient has pneumonia.",
        "The patient has pneumonia.",
        backend="heuristic",
    )

    assert set(result) == {"label", "score", "backend_id"}
    assert result["label"] in NLI_LABELS
    assert 0.0 <= result["score"] <= 1.0
    assert result["label"] == "entailment"
    assert result["backend_id"] == "heuristic"


def test_verify_flags_contradiction_without_echoing_claims() -> None:
    results = verify(
        [
            "The patient has no pneumonia.",
            "The patient has pneumonia.",
        ],
        "Synthetic patient has pneumonia.",
        backend="heuristic",
    )

    assert [result["label"] for result in results] == [
        "contradiction",
        "entailment",
    ]
    assert results[0]["contradicted"] is True
    assert results[1]["contradicted"] is False
    assert [result["claim_index"] for result in results] == [0, 1]
    assert "pneumonia" not in str(results)


def test_verify_pairs_aligned_source_spans() -> None:
    results = verify(
        ["No fever is present.", "Pneumonia is present."],
        ["Synthetic fever is present.", "Synthetic pneumonia is present."],
        backend="heuristic",
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

    assert result == {"label": "neutral", "score": 0.25, "backend_id": "custom-local"}
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


def test_invalid_backend_label_does_not_echo_sensitive_value() -> None:
    sensitive = "Synthetic identifier 555-0199"

    with pytest.raises(ValueError) as excinfo:
        nli(
            "synthetic source",
            "synthetic claim",
            backend=lambda premise, hypothesis: {"label": sensitive, "score": 0.5},
        )

    assert sensitive not in str(excinfo.value)
    assert "invalid label" in str(excinfo.value)


@pytest.mark.parametrize("score", [10**400, -(10**400), float("nan"), float("inf")])
def test_malformed_backend_scores_fail_with_a_stable_error(score) -> None:
    with pytest.raises(ValueError, match=r"must be finite and in \[0, 1\]"):
        nli(
            "synthetic source",
            "synthetic claim",
            backend=lambda premise, hypothesis: {"label": "neutral", "score": score},
        )


@pytest.mark.parametrize(
    ("premise", "hypothesis"),
    [
        ("Temperature is high.", "Glucose is low."),
        ("Pain improved.", "Vision declined."),
    ],
)
def test_opposites_in_unrelated_claims_remain_neutral(premise, hypothesis) -> None:
    assert nli(premise, hypothesis, backend="heuristic")["label"] == "neutral"
