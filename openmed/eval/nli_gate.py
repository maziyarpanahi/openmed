"""Content-free preparation report for a clinical NLI checkpoint candidate.

This composes existing NLI evidence into a BenchmarkReport. It never promotes a
checkpoint: publication, independent evaluation, and registry activation are
separate release actions.
"""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class NLIFormatParity:
    """Measured agreement of an export with the same source weight digest."""

    runtime: str
    count: int
    agreement_count: int
    maximum_probability_delta: float
    artifact_digest: str

    def __post_init__(self) -> None:
        if (
            self.runtime not in {"onnx_int8", "mlx"}
            or type(self.count) is not int
            or self.count < 1
            or type(self.agreement_count) is not int
            or not 0 <= self.agreement_count <= self.count
            or isinstance(self.maximum_probability_delta, bool)
            or not isinstance(self.maximum_probability_delta, (int, float))
            or not math.isfinite(self.maximum_probability_delta)
            or not 0 <= self.maximum_probability_delta <= 1
            or not isinstance(self.artifact_digest, str)
            or _SHA256.fullmatch(self.artifact_digest) is None
        ):
            raise ValueError("NLI format parity requires bounded measured aggregates")

    @property
    def passed(self) -> bool:
        """Apply fixed parity floors; too few fixtures cannot qualify an export."""

        delta = 0.001 if self.runtime == "mlx" else 0.05
        agreement = 1.0 if self.runtime == "mlx" else 0.98
        return (
            self.count >= 96
            and self.maximum_probability_delta <= delta
            and self.agreement_count / self.count >= agreement
        )


def build_nli_candidate_report(
    *,
    model_id: str,
    artifact_digest: str,
    public: NLIEvaluationCounts,
    biomedical: NLIEvaluationCounts,
    synthetic: NLIEvaluationCounts,
    error_slices: NLIErrorSliceReport,
    calibration: NLICalibrationReport,
    contradiction_calibration: NLICalibrationReport,
    negation: NliNegationReport,
    synthetic_entailment_support: int,
    synthetic_entailment_accepted: int,
    parity: tuple[NLIFormatParity, ...],
) -> tuple[BenchmarkReport, tuple[GateCheck, ...]]:
    """Evaluate real candidate evidence without asserting Hub publication.

    The serving ONNX export supplies accuracy, selective decisions, error
    slices and the negation run. Torch/MLX are independently compared against
    its pinned source weights. Failures remain failures, including calibration
    fallback when no point met the declared constraints. An all-abstention
    candidate cannot qualify. Text, raw logits and patient data are excluded.
    """

    if not isinstance(model_id, str) or _MODEL_ID.fullmatch(model_id) is None:
        raise ValueError("NLI model ID must be an opaque identifier")
    if (
        not isinstance(artifact_digest, str)
        or _SHA256.fullmatch(artifact_digest) is None
    ):
        raise ValueError("NLI candidate requires an exact weight digest")
    if any(
        not isinstance(value, NLIEvaluationCounts)
        for value in (public, biomedical, synthetic)
    ):
        raise TypeError("NLI candidate requires three aggregate evaluation splits")
    for value in (calibration, contradiction_calibration):
        if not isinstance(value, NLICalibrationReport) or (
            value.model_id != model_id or value.model_revision != artifact_digest
        ):
            raise ValueError("candidate calibration provenance differs")
    if not isinstance(error_slices, NLIErrorSliceReport) or not isinstance(
        negation, NliNegationReport
    ):
        raise TypeError("NLI candidate requires clinical safety reports")
    if (
        type(synthetic_entailment_support) is not int
        or synthetic_entailment_support < 1
        or type(synthetic_entailment_accepted) is not int
        or not 0 <= synthetic_entailment_accepted <= synthetic_entailment_support
    ):
        raise ValueError("NLI selective coverage requires bounded counts")
    if (
        len(parity) != 2
        or any(not isinstance(value, NLIFormatParity) for value in parity)
        or {value.runtime for value in parity} != {"onnx_int8", "mlx"}
        or any(value.artifact_digest != artifact_digest for value in parity)
    ):
        raise ValueError("both exports must match the candidate source digest")
    point = calibration.recommended_point
    coverage = synthetic_entailment_accepted / synthetic_entailment_support
    calibration_passed = all(
        not value.selection.endswith("no_point_met_constraints")
        and value.selection_constraints.get("precision_floor", 0) >= 0.99
        and value.selection_constraints.get("recall_floor", 0) >= 0.25
        and value.selection_constraints.get("false_positive_rate_ceiling", 1) <= 0.01
        and value.recommended_point.precision >= 0.99
        and value.recommended_point.recall >= 0.25
        and value.recommended_point.false_positive_rate <= 0.01
        for value in (calibration, contradiction_calibration)
    )
    checks = tuple(
        GateCheck(name, passed, reason="ok" if passed else reason)
        for name, passed, reason in (
            (
                "nli_public_accuracy",
                public.accuracy >= 0.65,
                "public three-way accuracy below 0.65",
            ),
            (
                "nli_biomedical_accuracy",
                biomedical.accuracy >= 0.75,
                "biomedical binary accuracy below 0.75",
            ),
            (
                "nli_synthetic_accuracy",
                synthetic.accuracy >= 0.90,
                "synthetic three-way accuracy below 0.90",
            ),
            ("nli_calibration", calibration_passed, "calibration constraints not met"),
            (
                "nli_selective_coverage",
                coverage >= 0.25,
                "held-out entailment recall below 0.25",
            ),
            (
                "nli_error_slices",
                all(value.fixture_count > 0 for value in error_slices.slices.values()),
                "a required clinical slice has no evidence",
            ),
            (
                "nli_negation",
                negation.gate_passed and negation.false_entailment_count == 0,
                "critical false entailment detected",
            ),
            (
                "nli_format_parity",
                all(value.passed for value in parity),
                "export agreement or probability delta failed",
            ),
            (
                "nli_publication",
                False,
                "candidate has not been independently resolved at a public immutable Hub revision",
            ),
        )
    )
    report = BenchmarkReport(
        suite="clinical-nli-candidate",
        model_name=model_id,
        device="local-onnx-cpu",
        fixture_count=public.count + biomedical.count + synthetic.count,
        metrics={
            name: {
                "count": value.count,
                "correct": value.correct,
                "accuracy": value.accuracy,
            }
            for name, value in (
                ("public_three_way", public),
                ("biomedical_binary", biomedical),
                ("synthetic_three_way", synthetic),
            )
        }
        | {
            "calibration": {
                "precision": point.precision,
                "recall": point.recall,
                "false_positive_rate": point.false_positive_rate,
                "threshold": point.threshold,
            },
            "held_out_entailment": {
                "support": synthetic_entailment_support,
                "accepted": synthetic_entailment_accepted,
                "recall": coverage,
            },
            "negation": {
                "count": negation.fixture_count,
                "accuracy": negation.aggregate_accuracy,
                "false_entailment_count": negation.false_entailment_count,
            },
            "formats": {
                value.runtime: {
                    "count": value.count,
                    "agreement_count": value.agreement_count,
                    "maximum_probability_delta": value.maximum_probability_delta,
                }
                for value in parity
            },
        },
        metadata={
            "stage": "qualified_unpublished_candidate"
            if all(check.passed for check in checks[:-1])
            else "failed_candidate",
            "artifact_digest": artifact_digest,
            "public_fixture_digest": public.fixture_digest,
            "biomedical_fixture_digest": biomedical.fixture_digest,
            "synthetic_fixture_digest": synthetic.fixture_digest,
            "calibration_fixture_fingerprint": calibration.fixture_fingerprint,
            "contradiction_calibration_fixture_fingerprint": contradiction_calibration.fixture_fingerprint,
            "error_slice_fixture_digest": error_slices.provenance.fixture_set_digest,
            "negation_fixture_digest": negation.fixture_set_hash,
            "checks": [check.to_dict() for check in checks],
        },
    )
    return report, checks
