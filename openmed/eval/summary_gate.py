"""Fail-closed summary release checks over explicit, local evaluation evidence.

No source text, extracted facts, or generated text is serialized by this module.
Citation matching is not clinician adjudication; missing adjudication fails the
semantic support gate instead of manufacturing a passing score.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from openmed.clinical.summarize import _build_leakage_check
from openmed.core.pii import DeidentificationResult
from openmed.eval.citation_support_metrics import compute_citation_support_metrics
from openmed.eval.metrics import fact_recall
from openmed.eval.release_gates import GateCheck
from openmed.eval.summary_coverage import compute_summary_fact_coverage
from openmed.eval.summary_review import ImportedSummaryReview
from openmed.eval.summary_unsupported_claims import score_summary_claims

THRESHOLD_KEYS = frozenset(
    {
        "clinical_fact_recall_min",
        "fact_coverage_min",
        "unsupported_claim_rate_max",
        "citation_support_min",
        "leaked_identifier_count_max",
    }
)


def evaluate_summary_gate(
    *,
    deidentified: DeidentificationResult,
    summary: str,
    source_facts: Sequence[Any],
    summary_facts: Sequence[Any],
    source_evidence: Sequence[Mapping[str, Any]],
    claims: Sequence[Mapping[str, Any]],
    support_evidence: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
    adjudications: Sequence[Mapping[str, Any]] | None = None,
    review_import: ImportedSummaryReview | None = None,
) -> tuple[GateCheck, ...]:
    """Compose existing metrics into value-free, blocking release checks.

    Evidence and structured facts must come from the evaluation protocol, not
    be copied from a model's self-assessment. A missing source set, empty output,
    invalid threshold, malformed metric input or absent adjudication fails
    closed. Source-identifier leakage is always required to be zero.
    ``review_import`` adds separate machine and review checks. Synthetic or
    incomplete imports cannot pass adjudication; current artifacts must match.
    """
    failed = False
    try:
        if (
            set(thresholds) != THRESHOLD_KEYS
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
                for v in thresholds.values()
            )
            or thresholds["leaked_identifier_count_max"] != 0
        ):
            raise ValueError("thresholds")
        if type(deidentified) is not DeidentificationResult or not isinstance(
            summary, str
        ):
            raise ValueError("input")
        if not source_facts or not source_evidence or not claims or not summary.strip():
            return (GateCheck("summary_evidence", False, "missing_evidence"),)
        recall = fact_recall(source_facts, summary_facts)
        coverage = compute_summary_fact_coverage(
            [{**row, "fact_id": row.get("evidence_id")} for row in source_evidence],
            claims,
        )
        unsupported = score_summary_claims(
            [{k: v for k, v in claim.items() if k != "citations"} for claim in claims],
            support_evidence,
        )
        citation_claims = [
            {k: v for k, v in claim.items() if k != "evidence_ids"} for claim in claims
        ]
        if review_import is not None:
            if (
                type(review_import) is not ImportedSummaryReview
                or adjudications is not None
                or not review_import.matches(
                    citation_claims,
                    source_evidence,
                    summary=summary,
                    source=deidentified.deidentified_text,
                )
            ):
                raise ValueError("review_import")
        support = compute_citation_support_metrics(
            citation_claims,
            source_evidence,
            adjudications=(
                review_import.adjudications if review_import else adjudications
            ),
        )
        leakage = _build_leakage_check(deidentified, summary)
        values = (
            (
                "clinical_fact_recall",
                recall.recall,
                thresholds["clinical_fact_recall_min"],
                True,
            ),
            (
                "fact_coverage",
                coverage.recall if not coverage.fail_closed else None,
                thresholds["fact_coverage_min"],
                True,
            ),
            (
                "unsupported_claim_rate",
                unsupported.unsupported_rate,
                thresholds["unsupported_claim_rate_max"],
                False,
            ),
            (
                "citation_support",
                support.adjudication.support_recall
                if review_import is None or review_import.reviewer_evidence_available
                else None,
                thresholds["citation_support_min"],
                True,
            ),
        )
        review_reason = (
            (
                "synthetic_review"
                if review_import.evidence_kind == "synthetic"
                else "unevaluable_review"
            )
            if review_import is not None
            else "missing_adjudication"
        )
        checks = [
            GateCheck(
                "summary_" + name,
                value is not None
                and (value >= threshold if floor else value <= threshold),
                review_reason
                if value is None and name == "citation_support"
                else "measured"
                if value is not None
                else "missing_evidence",
                {"value": value, "threshold": threshold},
            )
            for name, value, threshold, floor in values
        ]
        checks.append(
            GateCheck("summary_leakage", leakage.passed, "measured", leakage.to_dict())
        )
        if review_import is not None:
            checks.extend(
                (
                    GateCheck(
                        "summary_machine_spans",
                        support.deterministic.passed,
                        "machine_span_checks",
                        support.deterministic.to_dict(),
                    ),
                    GateCheck(
                        "summary_review",
                        review_import.reviewer_evidence_available,
                        "reviewer_evidence"
                        if review_import.reviewer_evidence_available
                        else "synthetic_review"
                        if review_import.evidence_kind == "synthetic"
                        else "unevaluable_review",
                        review_import.to_dict(),
                    ),
                )
            )
        return tuple(checks)
    except Exception:
        failed = True
    if failed:
        return (GateCheck("summary_input", False, "invalid_evaluation_input"),)
    raise AssertionError("unreachable")


def load_summary_eval_dataset(name: str, *, path=None):
    """Resolve the optional credentialed dataset without fetching or caching it."""
    if name != "mimic-iv-bhc":
        raise ValueError("unsupported summary evaluation dataset")
    from openmed.eval.datasets.mimic_iv_bhc import load_mimic_iv_bhc

    return load_mimic_iv_bhc(path)
