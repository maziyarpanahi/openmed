"""Content-free preparation report for a clinical NLI checkpoint candidate.

This composes existing NLI evidence into a BenchmarkReport. It never promotes a
checkpoint: publication, independent evaluation, and registry activation are
separate release actions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from openmed.eval.nli_calibration import NLICalibrationReport
from openmed.eval.nli_error_slices import NLIErrorSliceReport
from openmed.eval.nli_negation_challenge import NliNegationReport
from openmed.eval.release_gates import GateCheck
from openmed.eval.report import BenchmarkReport

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODEL_ID = re.compile(r"OpenMed/[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\Z")


@dataclass(frozen=True)
class NLIEvaluationCounts:
    """Aggregate counts and a content digest for one local evaluation split."""

    count: int
    correct: int
    fixture_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.count) is not int
            or self.count < 1
            or type(self.correct) is not int
            or not 0 <= self.correct <= self.count
            or not isinstance(self.fixture_digest, str)
            or _SHA256.fullmatch(self.fixture_digest) is None
        ):
            raise ValueError("NLI evaluation counts require bounded aggregates")

    @property
    def accuracy(self) -> float:
        """Return accuracy derived from validated counts."""

        return self.correct / self.count


def build_nli_preparation_report(
    *,
    model_id: str,
    model_revision: str,
    public: NLIEvaluationCounts,
    synthetic: NLIEvaluationCounts,
    error_slices: NLIErrorSliceReport,
    calibration: NLICalibrationReport,
    negation: NliNegationReport,
) -> tuple[BenchmarkReport, tuple[GateCheck, ...]]:
    """Compose sanitized local evidence without granting release eligibility.

    All supplied reports must come from a real candidate run before they can
    serve as evidence. This function accepts aggregates and digests only and
    deliberately emits a failing publication gate for every candidate.
    """

    if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("NLI model ID must be an opaque identifier")
    if (
        not isinstance(model_revision, str)
        or _REVISION.fullmatch(model_revision) is None
    ):
        raise ValueError("NLI model revision must be immutable")
    if not isinstance(public, NLIEvaluationCounts) or not isinstance(
        synthetic, NLIEvaluationCounts
    ):
        raise TypeError("NLI evaluation splits require aggregate counts")
    if not isinstance(error_slices, NLIErrorSliceReport):
        raise TypeError("NLI error-slice evidence is required")
    if not isinstance(calibration, NLICalibrationReport):
        raise TypeError("NLI calibration evidence is required")
    if not isinstance(negation, NliNegationReport):
        raise TypeError("NLI negation evidence is required")
    if calibration.model_id != model_id or calibration.model_revision != model_revision:
        raise ValueError("NLI calibration model provenance differs")

    checks = (
        GateCheck(
            "nli_error_slices",
            error_slices.fixture_count > 0,
            reason=(
                "ok" if error_slices.fixture_count > 0 else "empty error-slice evidence"
            ),
            details={"fixture_count": error_slices.fixture_count},
        ),
        GateCheck(
            "nli_calibration",
            calibration.fixture_count > 0,
            reason=(
                "ok" if calibration.fixture_count > 0 else "empty calibration evidence"
            ),
            details={"fixture_count": calibration.fixture_count},
        ),
        GateCheck(
            "nli_negation",
            negation.gate_passed,
            reason="ok" if negation.gate_passed else "negation challenge failed",
            details={
                "fixture_count": negation.fixture_count,
                "false_entailment_count": negation.false_entailment_count,
            },
        ),
        GateCheck(
            "nli_publication",
            False,
            reason="candidate publication and independent release evidence pending",
        ),
    )
    report = BenchmarkReport(
        suite="clinical-nli-preparation",
        model_name=model_id,
        device="local",
        fixture_count=public.count + synthetic.count,
        metrics={
            "public": {
                "count": public.count,
                "correct": public.correct,
                "accuracy": public.accuracy,
            },
            "synthetic": {
                "count": synthetic.count,
                "correct": synthetic.correct,
                "accuracy": synthetic.accuracy,
            },
            "negation": {
                "count": negation.fixture_count,
                "accuracy": negation.aggregate_accuracy,
                "false_entailment_rate": negation.false_entailment_rate,
            },
            "calibration": {
                "count": calibration.fixture_count,
                "recommended_threshold": calibration.recommended_threshold,
            },
            "error_slices": {
                "count": error_slices.fixture_count,
                "abstention_rate": error_slices.summary["abstention_rate"],
            },
        },
        metadata={
            "stage": "preparation_only",
            "model_revision": model_revision,
            "public_fixture_digest": public.fixture_digest,
            "synthetic_fixture_digest": synthetic.fixture_digest,
            "calibration_fixture_fingerprint": calibration.fixture_fingerprint,
            "error_slice_fixture_digest": error_slices.provenance.fixture_set_digest,
            "negation_fixture_digest": negation.fixture_set_hash,
            "checks": [check.to_dict() for check in checks],
        },
    )
    return report, checks
