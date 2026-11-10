"""India DPDP policy-coverage evaluation with per-label reporting.

This suite loads the synthetic OM-677 India clinical de-identification corpus
and applies the ``india_dpdp_act`` policy profile to every annotated
identifier. It produces deterministic per-label totals, covered counts,
coverage rates, and action counts, and enforces a policy gate requiring every
direct identifier to resolve to ``replace`` with zero ``keep`` actions.

Reports are privacy-safe: they emit counts, canonical labels, offsets, surface
hashes, and verdicts rather than any raw fixture identifier value.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from openmed.core.labels import normalize_label, policy_label_for
from openmed.core.policy import PolicyName, PolicyProfile, load_policy
from openmed.core.safety_sweep import hashed_span_surface
from openmed.core.schemas.span import ACTION_KEEP
from openmed.eval.datasets.clinical_phi import (
    IndiaClinicalPHICorpus,
    load_india_clinical_phi_corpus,
)

INDIA_DPDP_COVERAGE = "india_dpdp_coverage"
INDIA_DPDP_POLICY = PolicyName.INDIA_DPDP_ACT.value
REPLACE_ACTION = "replace"


@dataclass(frozen=True)
class DpdpLabelCoverage:
    """Per-label policy-coverage aggregates for the India DPDP suite."""

    label: str
    policy_label: str
    direct_identifier: bool
    total: int
    covered: int
    coverage_rate: float
    action_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free mapping for this label."""

        return {
            "label": self.label,
            "policy_label": self.policy_label,
            "direct_identifier": self.direct_identifier,
            "total": self.total,
            "covered": self.covered,
            "coverage_rate": self.coverage_rate,
            "action_counts": {
                action: int(self.action_counts[action])
                for action in sorted(self.action_counts)
            },
        }


@dataclass(frozen=True)
class DpdpCoverageGateFailure:
    """Raw-text-free failure emitted by the India DPDP coverage gate."""

    document_id: str
    label: str
    identifier_type: str
    action: str
    reason: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "label": self.label,
            "identifier_type": self.identifier_type,
            "action": self.action,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class DpdpCoverageReport:
    """Deterministic India DPDP policy-coverage report."""

    policy: str
    policy_schema_version: int
    document_count: int
    span_count: int
    direct_identifier_span_count: int
    covered_direct_identifier_span_count: int
    direct_identifier_coverage_rate: float
    keep_action_count: int
    per_label: tuple[DpdpLabelCoverage, ...]
    passed: bool
    failures: tuple[DpdpCoverageGateFailure, ...] = field(default_factory=tuple)

    def direct_identifier_labels(self) -> tuple[str, ...]:
        """Return the canonical direct-identifier labels present in the corpus."""

        return tuple(
            coverage.label for coverage in self.per_label if coverage.direct_identifier
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free report mapping."""

        return {
            "suite": INDIA_DPDP_COVERAGE,
            "policy": self.policy,
            "policy_schema_version": self.policy_schema_version,
            "document_count": self.document_count,
            "span_count": self.span_count,
            "direct_identifier_span_count": self.direct_identifier_span_count,
            "covered_direct_identifier_span_count": (
                self.covered_direct_identifier_span_count
            ),
            "direct_identifier_coverage_rate": self.direct_identifier_coverage_rate,
            "keep_action_count": self.keep_action_count,
            "direct_identifier_labels": list(self.direct_identifier_labels()),
            "per_label": [coverage.to_dict() for coverage in self.per_label],
            "passed": self.passed,
            "failures": [failure.to_dict() for failure in self.failures],
        }


def run_india_dpdp_coverage(
    *,
    policy: PolicyProfile | None = None,
    corpus: IndiaClinicalPHICorpus | None = None,
) -> DpdpCoverageReport:
    """Evaluate India DPDP policy coverage over the synthetic India corpus.

    Args:
        policy: Optional loaded profile. Defaults to the ``india_dpdp_act``
            profile. Callers may inject a derived profile to exercise the gate.
        corpus: Optional preloaded India clinical corpus. Defaults to the
            committed synthetic OM-677 corpus.

    Returns:
        A deterministic, raw-text-free per-label coverage report plus gate
        verdict and failures.
    """

    active_policy = policy if policy is not None else load_policy(INDIA_DPDP_POLICY)
    active_corpus = corpus if corpus is not None else load_india_clinical_phi_corpus()

    per_label_total: Counter[str] = Counter()
    per_label_covered: Counter[str] = Counter()
    per_label_actions: dict[str, Counter[str]] = {}
    per_label_direct: dict[str, bool] = {}

    span_count = 0
    direct_span_count = 0
    covered_direct_span_count = 0
    keep_action_count = 0
    failures: list[DpdpCoverageGateFailure] = []

    for record in active_corpus.records:
        for span in record.gold_spans:
            span_count += 1
            label = normalize_label(span.label, lang=record.language)
            action = active_policy.action_for(label, lang=record.language)
            is_direct = bool(span.metadata.get("direct_identifier"))
            identifier_type = str(span.metadata.get("identifier_type") or "")

            per_label_total[label] += 1
            per_label_actions.setdefault(label, Counter())[action] += 1
            per_label_direct[label] = per_label_direct.get(label, False) or is_direct
            if action != ACTION_KEEP:
                per_label_covered[label] += 1
            if action == ACTION_KEEP:
                keep_action_count += 1

            if is_direct:
                direct_span_count += 1
                if action == REPLACE_ACTION:
                    covered_direct_span_count += 1
                else:
                    reason = (
                        "direct_identifier_keep_action"
                        if action == ACTION_KEEP
                        else "direct_identifier_not_replaced"
                    )
                    failures.append(
                        DpdpCoverageGateFailure(
                            document_id=record.document_id,
                            label=label,
                            identifier_type=identifier_type,
                            action=action,
                            reason=reason,
                            evidence=hashed_span_surface(
                                record.text,
                                span.start,
                                span.end,
                                label=label,
                            ),
                        )
                    )

    per_label = tuple(
        DpdpLabelCoverage(
            label=label,
            policy_label=policy_label_for(label),
            direct_identifier=per_label_direct[label],
            total=per_label_total[label],
            covered=per_label_covered[label],
            coverage_rate=round(
                per_label_covered[label] / per_label_total[label]
                if per_label_total[label]
                else 0.0,
                6,
            ),
            action_counts=dict(per_label_actions[label]),
        )
        for label in sorted(per_label_total)
    )

    direct_coverage_rate = round(
        covered_direct_span_count / direct_span_count if direct_span_count else 0.0,
        6,
    )

    return DpdpCoverageReport(
        policy=active_policy.name,
        policy_schema_version=active_policy.schema_version,
        document_count=len(active_corpus.records),
        span_count=span_count,
        direct_identifier_span_count=direct_span_count,
        covered_direct_identifier_span_count=covered_direct_span_count,
        direct_identifier_coverage_rate=direct_coverage_rate,
        keep_action_count=keep_action_count,
        per_label=per_label,
        passed=not failures,
        failures=tuple(failures),
    )


def assert_india_dpdp_coverage_gate(
    *,
    policy: PolicyProfile | None = None,
    corpus: IndiaClinicalPHICorpus | None = None,
) -> DpdpCoverageReport:
    """Return a passing report or raise with raw-text-free diagnostics."""

    report = run_india_dpdp_coverage(policy=policy, corpus=corpus)
    if not report.passed:
        offenders = ", ".join(
            sorted(
                {
                    f"{failure.document_id}:{failure.label}"
                    for failure in report.failures
                }
            )
        )
        raise AssertionError(
            f"India DPDP policy-coverage gate failed for direct identifiers: {offenders}"
        )
    return report


def india_dpdp_coverage_metadata() -> dict[str, Any]:
    """Return raw-text-free metadata describing the India DPDP coverage suite."""

    return {
        "suite": INDIA_DPDP_COVERAGE,
        "policy": INDIA_DPDP_POLICY,
        "synthetic": True,
        "required_direct_identifier_coverage": 1.0,
        "required_keep_actions": 0,
    }


__all__ = [
    "INDIA_DPDP_COVERAGE",
    "INDIA_DPDP_POLICY",
    "DpdpCoverageGateFailure",
    "DpdpCoverageReport",
    "DpdpLabelCoverage",
    "assert_india_dpdp_coverage_gate",
    "india_dpdp_coverage_metadata",
    "run_india_dpdp_coverage",
]
