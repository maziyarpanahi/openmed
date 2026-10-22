"""Per-format quantized recall-delta evidence for private model candidates."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from openmed.eval.private_candidate_quant_gate import (
    REASON_BASE_PRECISION_INCOMPATIBLE,
    REASON_MISSING_RECALL_EVIDENCE,
    REASON_PREDICTION_FAILED,
    REASON_RECALL_DELTA_EXCEEDS_LIMIT,
    PrecisionCompatibilityDeclaration,
    PrivateCandidateQuantGateError,
    evaluate_private_candidate_quant_evidence,
)
from openmed.eval.quant_delta import (
    INT4_RECALL_DELTA_LIMIT,
    INT8_RECALL_DELTA_LIMIT,
    evaluate_quant_recall_delta,
)

_ROOT = Path(__file__).resolve().parents[3]
_SENTINEL = "Zqxsentinelname"
_SENTINEL_MAIL = "zqx.sentinel@example.invalid"
_CANDIDATE = "sha256:" + "a" * 64
_BASE = "sha256:" + "b" * 64
_HARNESS = "sha256:" + "c" * 64
_DECLARATION_EVIDENCE = "sha256:" + "d" * 64
_OTHER = "sha256:" + "e" * 64

Predictor = Callable[[str], Sequence[Mapping[str, Any]]]


def _holdout(size: int = 100) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(size):
        name = f"{_SENTINEL}{index}"
        prefix = "Patient "
        text = f"{prefix}{name} wrote from {_SENTINEL_MAIL} today."
        name_start = len(prefix)
        mail_start = text.index(_SENTINEL_MAIL)
        cases.append(
            {
                "text": text,
                "spans": [
                    {
                        "label": "PERSON",
                        "start": name_start,
                        "end": name_start + len(name),
                        "text": name,
                    },
                    {
                        "label": "EMAIL",
                        "start": mail_start,
                        "end": mail_start + len(_SENTINEL_MAIL),
                        "text": _SENTINEL_MAIL,
                    },
                ],
            }
        )
    return cases


def _gold_by_text(holdout: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    return {case["text"]: [dict(span) for span in case["spans"]] for case in holdout}


def _predictor(
    holdout: Sequence[Mapping[str, Any]],
    *,
    missed_person: int = 0,
) -> Predictor:
    gold = _gold_by_text(holdout)
    order = [case["text"] for case in holdout]
    dropped = set(order[:missed_person])

    def predict(text: str) -> list[dict[str, Any]]:
        spans = gold[text]
        if text in dropped:
            return [span for span in spans if span["label"] != "PERSON"]
        return spans

    return predict


def _evaluate(
    *,
    formats: Sequence[str] = ("fp16", "int8", "int4"),
    training_base_precision: str = "fp16",
    holdout: Sequence[Mapping[str, Any]] | None = None,
    format_predictors: Mapping[str, Predictor] | None = None,
    reference_predictor: Predictor | None = None,
    declarations: Sequence[PrecisionCompatibilityDeclaration] = (),
    candidate_digest: str = _CANDIDATE,
    base_digest: str = _BASE,
    harness_digest: str = _HARNESS,
):
    cases = _holdout() if holdout is None else holdout
    if format_predictors is None:
        format_predictors = {
            fmt: _predictor(cases) for fmt in formats if fmt.endswith(("int8", "int4"))
        }
    return evaluate_private_candidate_quant_evidence(
        candidate_digest=candidate_digest,
        base_digest=base_digest,
        harness_digest=harness_digest,
        training_base_precision=training_base_precision,
        deployment_formats=formats,
        holdout=cases,
        reference_predictor=reference_predictor or _predictor(cases),
        format_predictors=format_predictors,
        compatibility_declarations=declarations,
    )


def _by_format(evidence) -> dict[str, Any]:
    return {result.format: result for result in evidence.format_results}


# ---------------------------------------------------------------------------
# Acceptance criterion 1: per-format, non-compensable recall-delta results
# ---------------------------------------------------------------------------


def test_int8_delta_over_existing_limit_fails_int8_only() -> None:
    cases = _holdout()
    evidence = _evaluate(
        holdout=cases,
        format_predictors={
            "int8": _predictor(cases, missed_person=1),
            "int4": _predictor(cases),
        },
    )
    results = _by_format(evidence)

    assert evidence.passed is False
    assert evidence.failed_formats == ("int8",)
    assert results["int8"].passed is False
    assert results["int8"].reason_codes == (REASON_RECALL_DELTA_EXCEEDS_LIMIT,)
    assert results["int8"].limit == INT8_RECALL_DELTA_LIMIT
    assert results["int8"].offending_labels == ("PERSON",)
    assert results["int8"].per_label_delta["PERSON"] == pytest.approx(0.01)
    assert results["int4"].passed is True
    assert results["int4"].reason_codes == ()
    assert results["int4"].limit == INT4_RECALL_DELTA_LIMIT
    assert results["fp16"].passed is True
    assert results["fp16"].quantized is False


def test_int4_delta_over_existing_limit_fails_int4_only() -> None:
    cases = _holdout()
    evidence = _evaluate(
        holdout=cases,
        format_predictors={
            "int8": _predictor(cases),
            "int4": _predictor(cases, missed_person=2),
        },
    )
    results = _by_format(evidence)

    assert evidence.failed_formats == ("int4",)
    assert results["int4"].reason_codes == (REASON_RECALL_DELTA_EXCEEDS_LIMIT,)
    assert results["int4"].max_delta == pytest.approx(0.02)
    assert results["int8"].passed is True
    assert results["fp16"].passed is True


def test_identical_loss_is_judged_by_each_formats_existing_limit() -> None:
    cases = _holdout(125)
    evidence = _evaluate(
        holdout=cases,
        format_predictors={
            "int8": _predictor(cases, missed_person=1),
            "int4": _predictor(cases, missed_person=1),
        },
    )
    results = _by_format(evidence)

    assert results["int8"].max_delta == pytest.approx(0.008)
    assert results["int4"].max_delta == pytest.approx(0.008)
    assert results["int8"].passed is False
    assert results["int4"].passed is True
    assert evidence.failed_formats == ("int8",)


def test_format_results_are_separate_with_slice_sizes() -> None:
    cases = _holdout(40)
    evidence = _evaluate(holdout=cases)

    assert evidence.passed is True
    assert [result.format for result in evidence.format_results] == [
        "fp16",
        "int8",
        "int4",
    ]
    for result in evidence.format_results:
        assert result.holdout_items == 40
        assert dict(result.label_support) == {"EMAIL": 40, "PERSON": 40}
    payload = evidence.to_dict()
    assert len(payload["format_results"]) == 3
    assert "mean_delta" not in json.dumps(payload)


def test_passing_formats_never_offset_a_failing_format() -> None:
    cases = _holdout()
    evidence = _evaluate(
        formats=("onnx-int8", "coreml-int8", "mlx-int4"),
        holdout=cases,
        format_predictors={
            "onnx-int8": _predictor(cases),
            "coreml-int8": _predictor(cases, missed_person=1),
            "mlx-int4": _predictor(cases),
        },
    )

    assert evidence.passed is False
    assert evidence.failed_formats == ("coreml-int8",)
    assert evidence.reason_codes == (REASON_RECALL_DELTA_EXCEEDS_LIMIT,)
    assert _by_format(evidence)["onnx-int8"].passed is True
    assert _by_format(evidence)["mlx-int4"].passed is True


def test_non_direct_identifier_gold_labels_are_not_evaluated() -> None:
    cases = _holdout(20)
    for case in cases:
        case["spans"].append({"label": "DATE", "start": 0, "end": 7})
    evidence = _evaluate(holdout=cases)

    for result in evidence.format_results:
        assert "DATE" not in result.label_support
        assert "DATE" not in result.per_label_delta


def test_missing_format_predictor_fails_closed() -> None:
    cases = _holdout(10)
    evidence = _evaluate(
        holdout=cases,
        format_predictors={"int4": _predictor(cases)},
    )
    results = _by_format(evidence)

    assert results["int8"].reason_codes == (REASON_MISSING_RECALL_EVIDENCE,)
    assert results["int8"].max_delta is None
    assert results["int4"].passed is True


def test_format_predictor_failure_fails_that_format_without_detail() -> None:
    cases = _holdout(10)

    def broken(text: str) -> list[dict[str, Any]]:
        raise RuntimeError(f"model crashed on {text}")

    evidence = _evaluate(
        holdout=cases,
        format_predictors={"int8": broken, "int4": _predictor(cases)},
    )
    results = _by_format(evidence)

    assert results["int8"].reason_codes == (REASON_PREDICTION_FAILED,)
    assert results["int4"].passed is True
    assert _SENTINEL not in json.dumps(evidence.to_dict())
    assert _SENTINEL not in repr(evidence)


@pytest.mark.parametrize(
    "bad_span",
    [
        {"label": "PERSON", "start": 5, "end": 2},
        {"label": "PERSON", "start": -1, "end": 2},
        {"label": "PERSON", "start": 0, "end": 10_000},
        {"label": "PERSON", "start": True, "end": 2},
        {"label": 7, "start": 0, "end": 2},
        {"start": 0, "end": 2},
        "PERSON 0 2",
    ],
)
def test_malformed_predicted_span_fails_that_format(bad_span: Any) -> None:
    cases = _holdout(5)
    good = _predictor(cases)
    evidence = _evaluate(
        holdout=cases,
        format_predictors={
            "int8": lambda text: [*good(text), bad_span],
            "int4": good,
        },
    )
    results = _by_format(evidence)

    assert results["int8"].reason_codes == (REASON_PREDICTION_FAILED,)
    assert results["int4"].passed is True


# ---------------------------------------------------------------------------
# Acceptance criterion 2: base training precision compatibility
# ---------------------------------------------------------------------------


def test_four_bit_base_full_precision_deployment_without_declaration_fails() -> None:
    evidence = _evaluate(formats=("fp32",), training_base_precision="int4")
    (result,) = evidence.format_results

    assert REASON_BASE_PRECISION_INCOMPATIBLE == "base_precision_incompatible"
    assert result.passed is False
    assert result.reason_codes == (REASON_BASE_PRECISION_INCOMPATIBLE,)
    assert result.training_base_precision == "int4"
    assert result.deployment_precision == "fp32"
    assert result.precision_compatibility == "undeclared"
    assert evidence.passed is False
    assert evidence.reason_codes == (REASON_BASE_PRECISION_INCOMPATIBLE,)
    assert evidence.training_base_precision == "int4"


def test_four_bit_base_with_matching_declaration_proceeds() -> None:
    declaration = PrecisionCompatibilityDeclaration(
        candidate_digest=_CANDIDATE,
        training_precision="int4",
        deployment_precision="fp32",
        evidence_digest=_DECLARATION_EVIDENCE,
    )
    evidence = _evaluate(
        formats=("fp32",),
        training_base_precision="int4",
        declarations=(declaration,),
    )
    (result,) = evidence.format_results

    assert result.passed is True
    assert result.reason_codes == ()
    assert result.precision_compatibility == "declared"
    assert result.compatibility_digest == _DECLARATION_EVIDENCE
    assert evidence.passed is True


@pytest.mark.parametrize(
    "declaration",
    [
        PrecisionCompatibilityDeclaration(
            candidate_digest=_OTHER,
            training_precision="int4",
            deployment_precision="fp32",
            evidence_digest=_DECLARATION_EVIDENCE,
        ),
        PrecisionCompatibilityDeclaration(
            candidate_digest=_CANDIDATE,
            training_precision="int4",
            deployment_precision="int8",
            evidence_digest=_DECLARATION_EVIDENCE,
        ),
        PrecisionCompatibilityDeclaration(
            candidate_digest=_CANDIDATE,
            training_precision="int8",
            deployment_precision="fp32",
            evidence_digest=_DECLARATION_EVIDENCE,
        ),
    ],
)
def test_declaration_must_match_candidate_and_precision_pair(
    declaration: PrecisionCompatibilityDeclaration,
) -> None:
    evidence = _evaluate(
        formats=("fp32",),
        training_base_precision="int4",
        declarations=(declaration,),
    )

    assert evidence.format_results[0].reason_codes == (
        REASON_BASE_PRECISION_INCOMPATIBLE,
    )
    assert evidence.format_results[0].compatibility_digest is None


def test_precision_mismatch_fails_only_the_mismatched_format() -> None:
    evidence = _evaluate(
        formats=("fp32", "int8", "int4"),
        training_base_precision="int4",
    )
    results = _by_format(evidence)

    assert evidence.failed_formats == ("fp32", "int8")
    assert results["fp32"].reason_codes == (REASON_BASE_PRECISION_INCOMPATIBLE,)
    assert results["int8"].reason_codes == (REASON_BASE_PRECISION_INCOMPATIBLE,)
    assert results["int4"].passed is True
    assert results["int4"].precision_compatibility == "same_precision"


@pytest.mark.parametrize("training", ["fp32", "bf16", "fp16"])
def test_full_precision_base_is_the_reference_for_every_deployment(
    training: str,
) -> None:
    evidence = _evaluate(
        formats=("fp32", "bf16", "fp16", "int8", "int4"),
        training_base_precision=training,
    )

    assert evidence.passed is True
    for result in evidence.format_results:
        assert result.precision_compatibility in {
            "same_precision",
            "full_precision_reference",
        }


def test_precision_and_recall_failures_are_both_reported() -> None:
    cases = _holdout()
    evidence = _evaluate(
        formats=("int8",),
        training_base_precision="int4",
        holdout=cases,
        format_predictors={"int8": _predictor(cases, missed_person=3)},
    )

    assert evidence.format_results[0].reason_codes == (
        REASON_BASE_PRECISION_INCOMPATIBLE,
        REASON_RECALL_DELTA_EXCEEDS_LIMIT,
    )


# ---------------------------------------------------------------------------
# Acceptance criterion 3: digest binding and content-free evidence
# ---------------------------------------------------------------------------


def test_results_bind_candidate_base_format_and_harness_digests() -> None:
    evidence = _evaluate()
    payload = evidence.to_dict()

    assert payload["candidate_digest"] == _CANDIDATE
    assert payload["base_digest"] == _BASE
    assert payload["harness_digest"] == _HARNESS
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", payload["evidence_digest"])
    for result in payload["format_results"]:
        assert result["candidate_digest"] == _CANDIDATE
        assert result["base_digest"] == _BASE
        assert result["harness_digest"] == _HARNESS
        assert result["format"] in {"fp16", "int8", "int4"}
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", result["evidence_digest"])
    digests = {result["evidence_digest"] for result in payload["format_results"]}
    assert len(digests) == 3


def test_evidence_digest_is_deterministic() -> None:
    assert _evaluate().to_dict() == _evaluate().to_dict()


@pytest.mark.parametrize(
    "override",
    [
        {"candidate_digest": _OTHER},
        {"base_digest": _OTHER},
        {"harness_digest": _OTHER},
    ],
)
def test_evidence_digest_changes_with_each_bound_digest(
    override: dict[str, str],
) -> None:
    baseline = _evaluate()
    changed = _evaluate(**override)

    assert changed.evidence_digest != baseline.evidence_digest
    for left, right in zip(baseline.format_results, changed.format_results):
        assert left.evidence_digest != right.evidence_digest


def test_format_evidence_digest_covers_the_verdict() -> None:
    (result,) = _evaluate(formats=("int8",)).format_results
    tampered = replace(result, passed=False)

    assert tampered.computed_digest() != result.evidence_digest
    assert result.computed_digest() == result.evidence_digest


def test_evidence_contains_no_spans_or_text() -> None:
    cases = _holdout(30)
    evidence = _evaluate(
        formats=("fp32", "int8", "int4"),
        training_base_precision="int4",
        holdout=cases,
        format_predictors={
            "int8": _predictor(cases, missed_person=2),
            "int4": _predictor(cases),
        },
    )
    serialized = json.dumps(evidence.to_dict(), sort_keys=True)
    rendered = repr(evidence) + str(evidence)

    for forbidden in (_SENTINEL, _SENTINEL_MAIL, "Patient", "wrote from"):
        assert forbidden not in serialized
        assert forbidden not in rendered

    def keys(node: Any) -> set[str]:
        if isinstance(node, dict):
            return set(node) | {key for value in node.values() for key in keys(value)}
        if isinstance(node, list):
            return {key for value in node for key in keys(value)}
        return set()

    assert not keys(evidence.to_dict()) & {"start", "end", "text", "spans", "span"}


# ---------------------------------------------------------------------------
# Fail-closed input validation with stable, content-free errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"candidate_digest": f"sha256:{_SENTINEL}"}, "invalid_digest"),
        ({"base_digest": "sha256:" + "A" * 64}, "invalid_digest"),
        ({"harness_digest": ""}, "invalid_digest"),
        ({"training_base_precision": _SENTINEL}, "invalid_precision"),
        ({"training_base_precision": "nf4"}, "invalid_precision"),
        ({"formats": ()}, "missing_deployment_formats"),
        ({"formats": ("int8", "INT8")}, "invalid_deployment_format"),
        ({"formats": ("int8", "int8")}, "duplicate_deployment_format"),
        ({"formats": (f"{_SENTINEL}-int8",)}, "invalid_deployment_format"),
        ({"formats": ("4bit",)}, "invalid_deployment_format"),
    ],
)
def test_invalid_configuration_raises_stable_code(
    overrides: dict[str, Any],
    code: str,
) -> None:
    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        _evaluate(format_predictors={}, **overrides)

    assert excinfo.value.code == code
    assert str(excinfo.value).startswith(f"{code}: ")
    assert _SENTINEL not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_predictor_for_undeclared_format_is_rejected() -> None:
    cases = _holdout(3)
    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        _evaluate(
            formats=("int8",),
            holdout=cases,
            format_predictors={"int8": _predictor(cases), "int-8": _predictor(cases)},
        )

    assert excinfo.value.code == "undeclared_format_predictor"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cases: cases.clear(),
        lambda cases: cases[0].update(text=""),
        lambda cases: cases[0].update(text=7),
        lambda cases: cases[0].update(spans="PERSON"),
        lambda cases: cases[0]["spans"].append({"label": "PERSON", "start": 0}),
        lambda cases: cases[0]["spans"].append(
            {"label": "PERSON", "start": 0, "end": 10_000}
        ),
        lambda cases: [case.update(spans=[]) for case in cases],
        lambda cases: [
            case.update(spans=[{"label": "DATE", "start": 0, "end": 3}])
            for case in cases
        ],
    ],
)
def test_invalid_holdout_raises_without_echoing_content(mutate: Any) -> None:
    cases = _holdout(3)
    mutate(cases)
    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        _evaluate(
            holdout=cases,
            format_predictors={},
            reference_predictor=lambda text: [],
        )

    assert excinfo.value.code == "invalid_holdout"
    assert _SENTINEL not in str(excinfo.value)


def test_reference_predictor_failure_raises_without_detail() -> None:
    def broken(text: str) -> list[dict[str, Any]]:
        raise RuntimeError(f"reference crashed on {text}")

    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        _evaluate(reference_predictor=broken)

    assert excinfo.value.code == "reference_prediction_failed"
    assert _SENTINEL not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidate_digest": "sha256:" + "z" * 64},
        {"training_precision": "nf4"},
        {"deployment_precision": "int2"},
        {"evidence_digest": _SENTINEL},
    ],
)
def test_invalid_declaration_is_rejected(kwargs: dict[str, str]) -> None:
    fields = {
        "candidate_digest": _CANDIDATE,
        "training_precision": "int4",
        "deployment_precision": "fp32",
        "evidence_digest": _DECLARATION_EVIDENCE,
        **kwargs,
    }
    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        PrecisionCompatibilityDeclaration(**fields)

    assert _SENTINEL not in str(excinfo.value)


def test_declarations_must_be_declaration_instances() -> None:
    with pytest.raises(PrivateCandidateQuantGateError) as excinfo:
        _evaluate(
            formats=("fp32",),
            training_base_precision="int4",
            declarations=({"training_precision": "int4"},),  # type: ignore[arg-type]
        )

    assert excinfo.value.code == "invalid_compatibility_declaration"


# ---------------------------------------------------------------------------
# Cross-module consistency and documentation
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_format_verdicts_match_existing_quant_delta_gate() -> None:
    cases = _holdout()
    evidence = _evaluate(
        holdout=cases,
        format_predictors={
            "int8": _predictor(cases, missed_person=1),
            "int4": _predictor(cases, missed_person=1),
        },
    )
    results = _by_format(evidence)
    reference = {"PERSON": 1.0, "EMAIL": 1.0}
    candidate = {"PERSON": 0.99, "EMAIL": 1.0}

    for fmt in ("int8", "int4"):
        direct = evaluate_quant_recall_delta(
            format_name=fmt,
            candidate_recall=candidate,
            parent_recall=reference,
            labels=["EMAIL", "PERSON"],
        )
        assert results[fmt].passed is direct.passed
        assert results[fmt].limit == direct.limit
        assert results[fmt].max_delta == pytest.approx(direct.max_delta)


def test_documentation_example_runs_offline() -> None:
    page = _ROOT / "docs" / "evaluation" / "private-candidate-quant-gate.md"
    blocks = re.findall(r"(?ms)^```python\n(.*?)^```\s*$", page.read_text())

    assert blocks
    namespace: dict[str, Any] = {}
    exec(compile(blocks[0], str(page), "exec"), namespace)  # noqa: S102
    assert namespace["evidence"].failed_formats == ("int8",)
