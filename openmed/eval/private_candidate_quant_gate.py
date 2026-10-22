"""Quantized-deployment recall-delta evidence for private model candidates.

A privately improved adapter is merged into a full-precision base and then
shipped in one or more deployment formats. This module measures
direct-identifier recall of each quantized deployment against the
full-precision merge on a synthetic or sealed holdout and judges the loss with
:func:`openmed.eval.quant_delta.evaluate_quant_recall_delta`, so the existing
per-format limits apply unchanged. It also records the base precision used
during training and fails a deployment whose precision differs from it unless
the candidate carries an explicit compatibility declaration.

Model inference is injected through prediction callables. Results are
separate per format, never averaged, and contain only controlled reason codes,
precision tokens, rates, slice sizes and digests: no text, spans or offsets.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Sequence

from openmed.core.labels import DIRECT_IDENTIFIER, normalize_label, policy_label_for
from openmed.eval.quant_delta import (
    QuantRecallDeltaResult,
    evaluate_quant_recall_delta,
)

PRIVATE_CANDIDATE_QUANT_SCHEMA_VERSION: Final = (
    "openmed.eval.private_candidate_quant_gate.v1"
)

REASON_BASE_PRECISION_INCOMPATIBLE: Final = "base_precision_incompatible"
REASON_MISSING_RECALL_EVIDENCE: Final = "missing_recall_evidence"
REASON_PREDICTION_FAILED: Final = "prediction_failed"
REASON_RECALL_DELTA_EXCEEDS_LIMIT: Final = "recall_delta_exceeds_limit"

FULL_PRECISIONS: Final = ("fp32", "bf16", "fp16")
QUANTIZED_PRECISIONS: Final = ("int8", "int4")
PRECISIONS: Final = FULL_PRECISIONS + QUANTIZED_PRECISIONS

COMPATIBILITY_SAME_PRECISION: Final = "same_precision"
COMPATIBILITY_FULL_PRECISION_REFERENCE: Final = "full_precision_reference"
COMPATIBILITY_DECLARED: Final = "declared"
COMPATIBILITY_UNDECLARED: Final = "undeclared"

Predictor = Callable[[str], Sequence[Mapping[str, Any]]]

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_FORMAT: Final = re.compile(r"(?:[a-z]{1,32}-)?(fp32|bf16|fp16|int8|int4)\Z")
_MAX_FORMATS: Final = 32
_MAX_HOLDOUT_ITEMS: Final = 1_000_000


class PrivateCandidateQuantGateError(ValueError):
    """Raised for invalid gate inputs without echoing submitted values.

    Attributes:
        code: Stable machine-readable error code.
        field: Name of the offending argument or field.
    """

    def __init__(self, code: str, field: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code}: {field}")


class _PredictionFailure(Exception):
    """Internal marker for a failed or malformed prediction provider call."""


@dataclass(frozen=True)
class PrecisionCompatibilityDeclaration:
    """Explicit statement that a training/deployment precision pair is valid.

    A declaration applies only to the candidate it names and only to the exact
    training and deployment precision pair. The referenced evidence, such as a
    reviewed parity study, stays with the caller; only its digest is bound.

    Args:
        candidate_digest: ``sha256:`` digest of the private candidate.
        training_precision: Precision of the base used during training.
        deployment_precision: Precision of the deployment being declared.
        evidence_digest: ``sha256:`` digest of the supporting evidence.

    Raises:
        PrivateCandidateQuantGateError: If a digest or precision is invalid.
    """

    candidate_digest: str
    training_precision: str
    deployment_precision: str
    evidence_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.candidate_digest, "candidate_digest")
        _require_precision(self.training_precision, "training_precision")
        _require_precision(self.deployment_precision, "deployment_precision")
        _require_digest(self.evidence_digest, "evidence_digest")


@dataclass(frozen=True)
class PrivateCandidateFormatResult:
    """Content-free recall-delta and precision evidence for one format.

    Attributes:
        format: Declared deployment format, such as ``int8`` or ``onnx-int4``.
        deployment_precision: Precision token parsed from ``format``.
        training_base_precision: Precision of the base used during training.
        quantized: Whether the existing quant-delta gate treats the format as
            quantized and therefore applies a recall-delta limit.
        passed: Whether this format has no failing reason code.
        reason_codes: Sorted stable reason codes; empty when passed.
        precision_compatibility: ``same_precision``,
            ``full_precision_reference``, ``declared`` or ``undeclared``.
        limit: Existing per-format recall-delta limit, if quantized.
        max_delta: Largest per-label recall loss, if measured.
        per_label_delta: Recall loss per direct-identifier label.
        offending_labels: Labels whose loss meets or exceeds ``limit``.
        label_support: Gold span count per direct-identifier label.
        holdout_items: Number of holdout items evaluated.
        candidate_digest: Bound private candidate digest.
        base_digest: Bound deployment base digest.
        harness_digest: Bound evaluation harness digest.
        compatibility_digest: Evidence digest of the applied declaration.
        evidence_digest: Digest over every other field of this result.
    """

    format: str
    deployment_precision: str
    training_base_precision: str
    quantized: bool
    passed: bool
    reason_codes: tuple[str, ...]
    precision_compatibility: str
    limit: float | None
    max_delta: float | None
    per_label_delta: Mapping[str, float]
    offending_labels: tuple[str, ...]
    label_support: Mapping[str, int]
    holdout_items: int
    candidate_digest: str
    base_digest: str
    harness_digest: str
    compatibility_digest: str | None = None
    evidence_digest: str = ""
    schema_version: str = field(default=PRIVATE_CANDIDATE_QUANT_SCHEMA_VERSION)

    def computed_digest(self) -> str:
        """Return the digest of this result's fields, excluding the digest.

        Returns:
            A ``sha256:`` digest that changes when any bound field changes.
        """
        payload = self.to_dict()
        payload.pop("evidence_digest")
        return _digest(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of this format result.

        Returns:
            A fresh mapping containing codes, rates, counts and digests only.
        """
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "deployment_precision": self.deployment_precision,
            "training_base_precision": self.training_base_precision,
            "quantized": self.quantized,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "precision_compatibility": self.precision_compatibility,
            "limit": self.limit,
            "max_delta": self.max_delta,
            "per_label_delta": dict(self.per_label_delta),
            "offending_labels": list(self.offending_labels),
            "label_support": dict(self.label_support),
            "holdout_items": self.holdout_items,
            "candidate_digest": self.candidate_digest,
            "base_digest": self.base_digest,
            "harness_digest": self.harness_digest,
            "compatibility_digest": self.compatibility_digest,
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class PrivateCandidateQuantEvidence:
    """Per-format quantized-deployment evidence for one private candidate.

    ``passed`` is true only when every format passed; a passing format never
    offsets a failing one. A promotion gate should consume ``format_results``
    individually.

    Attributes:
        passed: Whether every declared deployment format passed.
        candidate_digest: Bound private candidate digest.
        base_digest: Bound deployment base digest.
        harness_digest: Bound evaluation harness digest.
        training_base_precision: Precision of the base used during training.
        format_results: One result per declared format, in declaration order.
        reason_codes: Sorted union of the failing formats' reason codes.
        failed_formats: Failing formats in declaration order.
        evidence_digest: Digest over the bound fields and format digests.
    """

    passed: bool
    candidate_digest: str
    base_digest: str
    harness_digest: str
    training_base_precision: str
    format_results: tuple[PrivateCandidateFormatResult, ...]
    reason_codes: tuple[str, ...]
    failed_formats: tuple[str, ...]
    evidence_digest: str = ""
    schema_version: str = field(default=PRIVATE_CANDIDATE_QUANT_SCHEMA_VERSION)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping of the candidate evidence.

        Returns:
            A fresh mapping containing codes, rates, counts and digests only.
        """
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "candidate_digest": self.candidate_digest,
            "base_digest": self.base_digest,
            "harness_digest": self.harness_digest,
            "training_base_precision": self.training_base_precision,
            "reason_codes": list(self.reason_codes),
            "failed_formats": list(self.failed_formats),
            "format_results": [result.to_dict() for result in self.format_results],
            "evidence_digest": self.evidence_digest,
        }


def evaluate_private_candidate_quant_evidence(
    *,
    candidate_digest: str,
    base_digest: str,
    harness_digest: str,
    training_base_precision: str,
    deployment_formats: Sequence[str],
    holdout: Sequence[Mapping[str, Any]],
    reference_predictor: Predictor,
    format_predictors: Mapping[str, Predictor],
    compatibility_declarations: Sequence[PrecisionCompatibilityDeclaration] = (),
) -> PrivateCandidateQuantEvidence:
    """Produce per-format recall-delta evidence for a merged private candidate.

    Direct-identifier recall of each quantized deployment is compared with the
    full-precision merge using
    :func:`openmed.eval.quant_delta.evaluate_quant_recall_delta`, which applies
    the existing per-format limits. Formats that the existing gate does not
    treat as quantized need no prediction provider. Every format is also
    checked for compatibility with the training base precision: a quantized
    training base requires the same precision at deployment or a matching
    :class:`PrecisionCompatibilityDeclaration`.

    Args:
        candidate_digest: ``sha256:`` digest of the merged private candidate.
        base_digest: ``sha256:`` digest of the deployment base.
        harness_digest: ``sha256:`` digest of the evaluation harness, such as
            a sealed holdout commitment.
        training_base_precision: One of ``fp32``, ``bf16``, ``fp16``,
            ``int8`` or ``int4``.
        deployment_formats: Declared formats, each a precision token with an
            optional lowercase runtime prefix such as ``onnx-int8``.
        holdout: Synthetic or sealed items, each a mapping with ``text`` and
            ``spans`` (``label``, ``start``, ``end``). Text is passed only to
            the prediction providers and never stored.
        reference_predictor: Returns spans from the full-precision merge.
        format_predictors: Returns spans from each quantized format, keyed by
            declared format. A missing provider fails that format closed.
        compatibility_declarations: Explicit training/deployment precision
            compatibility declarations.

    Returns:
        Separate, non-compensable per-format evidence bound to the candidate,
        base, format and harness digests.

    Raises:
        PrivateCandidateQuantGateError: If an input is invalid or the
            reference predictor fails. Messages hold a code and field only.
    """
    _require_digest(candidate_digest, "candidate_digest")
    _require_digest(base_digest, "base_digest")
    _require_digest(harness_digest, "harness_digest")
    _require_precision(training_base_precision, "training_base_precision")
    formats = _deployment_formats(deployment_formats)
    predictors = _format_predictors(format_predictors, formats)
    if not callable(reference_predictor):
        raise PrivateCandidateQuantGateError("invalid_predictor", "reference_predictor")
    cases, support = _holdout_cases(holdout)
    declarations = _declarations(compatibility_declarations)

    try:
        reference_recall = _recall(reference_predictor, cases, support)
    except _PredictionFailure:
        raise PrivateCandidateQuantGateError(
            "reference_prediction_failed", "reference_predictor"
        ) from None

    results = tuple(
        _format_result(
            format_name=format_name,
            training_precision=training_base_precision,
            predictor=predictors.get(format_name),
            cases=cases,
            support=support,
            reference_recall=reference_recall,
            declarations=declarations,
            candidate_digest=candidate_digest,
            base_digest=base_digest,
            harness_digest=harness_digest,
        )
        for format_name in formats
    )
    failed = tuple(result.format for result in results if not result.passed)
    reasons = tuple(
        sorted({code for result in results for code in result.reason_codes})
    )
    evidence = PrivateCandidateQuantEvidence(
        passed=not failed,
        candidate_digest=candidate_digest,
        base_digest=base_digest,
        harness_digest=harness_digest,
        training_base_precision=training_base_precision,
        format_results=results,
        reason_codes=reasons,
        failed_formats=failed,
    )
    payload = evidence.to_dict()
    payload.pop("evidence_digest")
    payload["format_results"] = [result.evidence_digest for result in results]
    return replace(evidence, evidence_digest=_digest(payload))


def _format_result(
    *,
    format_name: str,
    training_precision: str,
    predictor: Predictor | None,
    cases: Sequence[tuple[str, frozenset[tuple[str, int, int]]]],
    support: Mapping[str, int],
    reference_recall: Mapping[str, float],
    declarations: Sequence[PrecisionCompatibilityDeclaration],
    candidate_digest: str,
    base_digest: str,
    harness_digest: str,
) -> PrivateCandidateFormatResult:
    precision = _format_precision(format_name)
    reasons: set[str] = set()
    compatibility, compatibility_digest = _compatibility(
        training_precision=training_precision,
        deployment_precision=precision,
        candidate_digest=candidate_digest,
        declarations=declarations,
    )
    if compatibility == COMPATIBILITY_UNDECLARED:
        reasons.add(REASON_BASE_PRECISION_INCOMPATIBLE)

    quant = _quant_result(
        format_name=format_name,
        predictor=predictor,
        cases=cases,
        support=support,
        reference_recall=reference_recall,
        reasons=reasons,
    )
    if quant.quantized != (precision in QUANTIZED_PRECISIONS):
        raise PrivateCandidateQuantGateError("invalid_deployment_format", "format")

    result = PrivateCandidateFormatResult(
        format=format_name,
        deployment_precision=precision,
        training_base_precision=training_precision,
        quantized=quant.quantized,
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
        precision_compatibility=compatibility,
        limit=quant.limit,
        max_delta=quant.max_delta,
        per_label_delta=MappingProxyType(dict(sorted(quant.per_label_delta.items()))),
        offending_labels=tuple(sorted(quant.offending_labels)),
        label_support=MappingProxyType(dict(sorted(support.items()))),
        holdout_items=len(cases),
        candidate_digest=candidate_digest,
        base_digest=base_digest,
        harness_digest=harness_digest,
        compatibility_digest=compatibility_digest,
    )
    return replace(result, evidence_digest=result.computed_digest())


def _quant_result(
    *,
    format_name: str,
    predictor: Predictor | None,
    cases: Sequence[tuple[str, frozenset[tuple[str, int, int]]]],
    support: Mapping[str, int],
    reference_recall: Mapping[str, float],
    reasons: set[str],
) -> QuantRecallDeltaResult:
    labels = sorted(support)
    if _format_precision(format_name) not in QUANTIZED_PRECISIONS:
        return evaluate_quant_recall_delta(
            format_name=format_name,
            candidate_recall=reference_recall,
            parent_recall=reference_recall,
            labels=labels,
        )
    if predictor is None:
        reasons.add(REASON_MISSING_RECALL_EVIDENCE)
        return evaluate_quant_recall_delta(format_name=format_name, candidate_recall={})
    try:
        candidate_recall = _recall(predictor, cases, support)
    except _PredictionFailure:
        reasons.add(REASON_PREDICTION_FAILED)
        return evaluate_quant_recall_delta(format_name=format_name, candidate_recall={})
    quant = evaluate_quant_recall_delta(
        format_name=format_name,
        candidate_recall=candidate_recall,
        parent_recall=reference_recall,
        labels=labels,
    )
    if not quant.passed:
        if quant.max_delta is None:
            reasons.add(REASON_MISSING_RECALL_EVIDENCE)
        else:
            reasons.add(REASON_RECALL_DELTA_EXCEEDS_LIMIT)
    return quant


def _compatibility(
    *,
    training_precision: str,
    deployment_precision: str,
    candidate_digest: str,
    declarations: Sequence[PrecisionCompatibilityDeclaration],
) -> tuple[str, str | None]:
    training_class = _precision_class(training_precision)
    deployment_class = _precision_class(deployment_precision)
    if training_class == deployment_class:
        return COMPATIBILITY_SAME_PRECISION, None
    if training_class == "full":
        return COMPATIBILITY_FULL_PRECISION_REFERENCE, None
    for declaration in declarations:
        if (
            declaration.candidate_digest == candidate_digest
            and declaration.training_precision == training_precision
            and declaration.deployment_precision == deployment_precision
        ):
            return COMPATIBILITY_DECLARED, declaration.evidence_digest
    return COMPATIBILITY_UNDECLARED, None


def _precision_class(precision: str) -> str:
    return "full" if precision in FULL_PRECISIONS else precision


def _recall(
    predictor: Predictor,
    cases: Sequence[tuple[str, frozenset[tuple[str, int, int]]]],
    support: Mapping[str, int],
) -> dict[str, float]:
    hits: Counter[str] = Counter()
    for text, gold in cases:
        try:
            raw = predictor(text)
        except Exception:
            raise _PredictionFailure from None
        predicted = _predicted_keys(raw, len(text))
        for label, _start, _end in gold & predicted:
            hits[label] += 1
    return {label: hits[label] / count for label, count in sorted(support.items())}


def _predicted_keys(raw: Any, text_length: int) -> frozenset[tuple[str, int, int]]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise _PredictionFailure
    keys: set[tuple[str, int, int]] = set()
    for span in raw:
        key = _span_key(span, text_length)
        if key is None:
            raise _PredictionFailure
        keys.add(key)
    return frozenset(keys)


def _span_key(span: Any, text_length: int) -> tuple[str, int, int] | None:
    if not isinstance(span, Mapping):
        return None
    label = span.get("label")
    start = span.get("start")
    end = span.get("end")
    if type(label) is not str or type(start) is not int or type(end) is not int:
        return None
    if not 0 <= start < end <= text_length:
        return None
    return normalize_label(label), start, end


def _holdout_cases(
    holdout: Any,
) -> tuple[list[tuple[str, frozenset[tuple[str, int, int]]]], dict[str, int]]:
    if (
        isinstance(holdout, (str, bytes, Mapping))
        or not isinstance(holdout, Sequence)
        or not 0 < len(holdout) <= _MAX_HOLDOUT_ITEMS
    ):
        raise PrivateCandidateQuantGateError("invalid_holdout", "holdout")
    cases: list[tuple[str, frozenset[tuple[str, int, int]]]] = []
    support: Counter[str] = Counter()
    for case in holdout:
        if not isinstance(case, Mapping):
            raise PrivateCandidateQuantGateError("invalid_holdout", "holdout")
        text = case.get("text")
        spans = case.get("spans")
        if type(text) is not str or not text:
            raise PrivateCandidateQuantGateError("invalid_holdout", "text")
        if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
            raise PrivateCandidateQuantGateError("invalid_holdout", "spans")
        gold: set[tuple[str, int, int]] = set()
        for span in spans:
            key = _span_key(span, len(text))
            if key is None:
                raise PrivateCandidateQuantGateError("invalid_holdout", "spans")
            if policy_label_for(key[0]) == DIRECT_IDENTIFIER:
                gold.add(key)
        support.update(label for label, _start, _end in gold)
        cases.append((text, frozenset(gold)))
    if not support:
        raise PrivateCandidateQuantGateError("invalid_holdout", "spans")
    return cases, dict(support)


def _deployment_formats(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PrivateCandidateQuantGateError(
            "invalid_deployment_format", "deployment_formats"
        )
    if not values:
        raise PrivateCandidateQuantGateError(
            "missing_deployment_formats", "deployment_formats"
        )
    if len(values) > _MAX_FORMATS:
        raise PrivateCandidateQuantGateError(
            "invalid_deployment_format", "deployment_formats"
        )
    formats: list[str] = []
    for value in values:
        if type(value) is not str or _FORMAT.fullmatch(value) is None:
            raise PrivateCandidateQuantGateError(
                "invalid_deployment_format", "deployment_formats"
            )
        if value in formats:
            raise PrivateCandidateQuantGateError(
                "duplicate_deployment_format", "deployment_formats"
            )
        formats.append(value)
    return tuple(formats)


def _format_predictors(
    values: Any,
    formats: Sequence[str],
) -> dict[str, Predictor]:
    if not isinstance(values, Mapping):
        raise PrivateCandidateQuantGateError("invalid_predictor", "format_predictors")
    predictors: dict[str, Predictor] = {}
    for key, predictor in values.items():
        if key not in formats:
            raise PrivateCandidateQuantGateError(
                "undeclared_format_predictor", "format_predictors"
            )
        if not callable(predictor):
            raise PrivateCandidateQuantGateError(
                "invalid_predictor", "format_predictors"
            )
        predictors[key] = predictor
    return predictors


def _declarations(values: Any) -> tuple[PrecisionCompatibilityDeclaration, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PrivateCandidateQuantGateError(
            "invalid_compatibility_declaration", "compatibility_declarations"
        )
    for value in values:
        if type(value) is not PrecisionCompatibilityDeclaration:
            raise PrivateCandidateQuantGateError(
                "invalid_compatibility_declaration", "compatibility_declarations"
            )
    return tuple(values)


def _format_precision(format_name: str) -> str:
    match = _FORMAT.fullmatch(format_name)
    if match is None:
        raise PrivateCandidateQuantGateError("invalid_deployment_format", "format")
    return match.group(1)


def _require_digest(value: Any, field_name: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise PrivateCandidateQuantGateError("invalid_digest", field_name)


def _require_precision(value: Any, field_name: str) -> None:
    if type(value) is not str or value not in PRECISIONS:
        raise PrivateCandidateQuantGateError("invalid_precision", field_name)


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "COMPATIBILITY_DECLARED",
    "COMPATIBILITY_FULL_PRECISION_REFERENCE",
    "COMPATIBILITY_SAME_PRECISION",
    "COMPATIBILITY_UNDECLARED",
    "FULL_PRECISIONS",
    "PRECISIONS",
    "PRIVATE_CANDIDATE_QUANT_SCHEMA_VERSION",
    "PrecisionCompatibilityDeclaration",
    "Predictor",
    "PrivateCandidateFormatResult",
    "PrivateCandidateQuantEvidence",
    "PrivateCandidateQuantGateError",
    "QUANTIZED_PRECISIONS",
    "REASON_BASE_PRECISION_INCOMPATIBLE",
    "REASON_MISSING_RECALL_EVIDENCE",
    "REASON_PREDICTION_FAILED",
    "REASON_RECALL_DELTA_EXCEEDS_LIMIT",
    "evaluate_private_candidate_quant_evidence",
]
