from __future__ import annotations

import socket
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.nli_cascade import (
    LocalNliModelOutput,
    NliCascadeError,
    NliCascadePair,
    NliCascadeResult,
    NliCascadeStage,
    NliRuleStatus,
    evaluate_nli_pair,
    run_nli_cascade,
)


class RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, premise: str, hypothesis: str) -> LocalNliModelOutput:
        self.calls.append((premise, hypothesis))
        return LocalNliModelOutput(
            label="entailment", score=0.9, backend_id="local-test"
        )


@pytest.mark.parametrize(
    ("field_name", "expected_stage"),
    [
        ("assertion", NliCascadeStage.ASSERTION),
        ("temporality", NliCascadeStage.TEMPORALITY),
        ("experiencer", NliCascadeStage.EXPERIENCER),
        ("numeric", NliCascadeStage.NUMERIC),
    ],
)
def test_exact_structured_conflict_bypasses_model(
    field_name: str, expected_stage: NliCascadeStage
) -> None:
    model = RecordingModel()
    pair = NliCascadePair(
        pair_id="pair-a",
        premise="synthetic premise",
        hypothesis="synthetic hypothesis",
        **{field_name: NliRuleStatus.CONFLICT},
    )

    result = evaluate_nli_pair(pair, model)

    assert result.label == "contradiction"
    assert result.deciding_stage is expected_stage
    assert result.reason_code == f"{expected_stage.value}_conflict"
    assert result.model_invoked is False
    assert model.calls == []


def test_first_conflict_in_fixed_stage_order_decides() -> None:
    result = evaluate_nli_pair(
        NliCascadePair(
            pair_id="pair-a",
            premise="synthetic premise",
            hypothesis="synthetic hypothesis",
            temporality=NliRuleStatus.CONFLICT,
            numeric=NliRuleStatus.CONFLICT,
        ),
        RecordingModel(),
    )

    assert result.deciding_stage is NliCascadeStage.TEMPORALITY


def test_model_runs_only_when_rules_do_not_resolve_pair() -> None:
    model = RecordingModel()
    pair = NliCascadePair(
        pair_id="pair-a",
        premise="synthetic premise",
        hypothesis="synthetic hypothesis",
        assertion=NliRuleStatus.COMPATIBLE,
        temporality=NliRuleStatus.UNRESOLVED,
        experiencer=NliRuleStatus.NOT_APPLICABLE,
        numeric=NliRuleStatus.COMPATIBLE,
    )

    result = evaluate_nli_pair(pair, model)

    assert model.calls == [("synthetic premise", "synthetic hypothesis")]
    assert result.label == "entailment"
    assert result.deciding_stage is NliCascadeStage.MODEL
    assert result.model_invoked is True
    assert result.reason_code == "model_decision"
    assert result.score == 0.9
    assert result.backend_id == "local-test"


def test_batch_invokes_model_only_for_unresolved_pairs() -> None:
    model = RecordingModel()
    results = run_nli_cascade(
        [
            NliCascadePair(
                "pair-a",
                "synthetic premise a",
                "synthetic hypothesis a",
                assertion=NliRuleStatus.CONFLICT,
            ),
            NliCascadePair("pair-b", "synthetic premise b", "synthetic hypothesis b"),
            NliCascadePair(
                "pair-c",
                "synthetic premise c",
                "synthetic hypothesis c",
                numeric=NliRuleStatus.CONFLICT,
            ),
        ],
        model,
    )

    assert [result.deciding_stage for result in results] == [
        NliCascadeStage.ASSERTION,
        NliCascadeStage.MODEL,
        NliCascadeStage.NUMERIC,
    ]
    assert model.calls == [("synthetic premise b", "synthetic hypothesis b")]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("contradiction", NliRuleStatus.CONFLICT),
        ("incompatible", NliRuleStatus.CONFLICT),
        ({"status": "review_required"}, NliRuleStatus.UNRESOLVED),
        ({"status": "compatible"}, NliRuleStatus.COMPATIBLE),
        ({"status": "not_applicable"}, NliRuleStatus.NOT_APPLICABLE),
    ],
)
def test_precheck_status_adapter_uses_controlled_vocabularies(
    source: object, expected: NliRuleStatus
) -> None:
    assert NliRuleStatus.from_precheck(source) is expected


def test_cascade_is_deterministic_for_same_inputs_and_model() -> None:
    pair = NliCascadePair("pair-a", "synthetic premise", "synthetic hypothesis")

    first = evaluate_nli_pair(pair, RecordingModel())
    second = evaluate_nli_pair(pair, RecordingModel())

    assert first == second
    assert first.to_audit_dict() == second.to_audit_dict()


def test_cascade_performs_no_mandatory_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", fail_socket)
    result = evaluate_nli_pair(
        NliCascadePair(
            "pair-a",
            "synthetic premise",
            "synthetic hypothesis",
            assertion=NliRuleStatus.CONFLICT,
        ),
        RecordingModel(),
    )

    assert result.model_invoked is False


def test_representations_and_reports_hide_source_values() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    pair = NliCascadePair(sentinel, sentinel + " premise", sentinel + " hypothesis")
    result = evaluate_nli_pair(pair, RecordingModel())

    rendered = repr(pair) + repr(result) + str(result.to_audit_dict())
    assert sentinel not in rendered
    assert "synthetic" not in rendered


def test_model_exception_is_replaced_with_value_free_error() -> None:
    sentinel = "SENSITIVE_SENTINEL"

    def fail_model(premise: str, hypothesis: str) -> str:
        raise RuntimeError(sentinel + premise + hypothesis)

    with pytest.raises(NliCascadeError, match="local model inference failed") as exc:
        evaluate_nli_pair(
            NliCascadePair("pair-a", sentinel, sentinel),
            fail_model,
        )

    assert sentinel not in str(exc.value)


@pytest.mark.parametrize("label", ["unknown", "", "model-label-2"])
def test_noncanonical_model_label_fails_without_echoing_value(label: str) -> None:
    with pytest.raises(NliCascadeError, match="NLI label must be canonical") as exc:
        evaluate_nli_pair(
            NliCascadePair("pair-a", "synthetic premise", "synthetic hypothesis"),
            lambda premise, hypothesis: label,
        )

    if label:
        assert label not in str(exc.value)


def test_duplicate_pair_identifiers_fail_before_any_model_call() -> None:
    model = RecordingModel()
    pair = NliCascadePair("pair-a", "synthetic premise", "synthetic hypothesis")

    with pytest.raises(NliCascadeError, match="identifiers must be unique"):
        run_nli_cascade([pair, pair], model)

    assert model.calls == []


def test_pair_result_and_model_output_are_immutable() -> None:
    pair = NliCascadePair("pair-a", "synthetic premise", "synthetic hypothesis")
    output = LocalNliModelOutput("neutral", score=0.5)
    result = evaluate_nli_pair(pair, lambda premise, hypothesis: output)

    with pytest.raises(FrozenInstanceError):
        pair.premise = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        output.label = "entailment"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.label = "entailment"  # type: ignore[misc]


@pytest.mark.parametrize("score", [10**400, -(10**400)])
def test_oversized_model_scores_fail_with_controlled_error(score: int) -> None:
    with pytest.raises(NliCascadeError, match="model score"):
        LocalNliModelOutput("entailment", score=score)
    with pytest.raises(NliCascadeError, match="model score"):
        NliCascadeResult(
            "pair-a",
            "entailment",
            NliCascadeStage.MODEL,
            True,
            "model_decision",
            score=score,
        )


def test_invalid_reason_code_has_controlled_error() -> None:
    with pytest.raises(NliCascadeError, match="cascade reason code"):
        NliCascadeResult(
            "pair-a", "contradiction", NliCascadeStage.NUMERIC, False, None
        )


def test_unencodable_pair_id_fails_before_model_dispatch() -> None:
    with pytest.raises(NliCascadeError, match="pair identifier must be valid Unicode"):
        NliCascadePair("pair-\ud800", "synthetic premise", "synthetic hypothesis")
