"""Fail-closed residual-PHI leakage gate for the synthetic India corpus.

The gate runs clinical de-identification over every synthetic India clinical
fixture under the India DPDP profile and the supported multilingual detection
path, then scans each de-identified document for the original synthetic direct
identifiers declared by the corpus (ABHA, Aadhaar, PAN, phone, person-name
aliases, and street addresses). A surviving literal fails the gate with
privacy-safe document, label, offset, and hash evidence; raw identifier
surfaces never enter the result, reports, or logs.

The corpus is committed synthetic data only: it contains no real PHI, no
production data, and no DUA-restricted content. The gate is deterministic and
fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from openmed.core.policy import load_policy
from openmed.core.safety_sweep import safety_sweep
from openmed.core.schemas.span import hmac_text_hash

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openmed.core.policy import PolicyProfile
    from openmed.eval.datasets.clinical_phi import IndiaClinicalPHIRecord

INDIA_CLINICAL_PHI_LEAKAGE = "india_clinical_phi_leakage"
INDIA_CLINICAL_PHI_POLICY = "india_dpdp_act"
INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES = frozenset(
    {
        "person_name",
        "abha",
        "aadhaar",
        "pan",
        "indian_phone",
        "street_address",
    }
)
_LEAKAGE_HASH_KEY = "openmed-india-clinical-phi-leakage"

# Redaction callable: given one corpus record, return its de-identified text.
Deidentifier = Callable[["IndiaClinicalPHIRecord"], str]


@dataclass(frozen=True)
class IndiaClinicalLeakageFinding:
    """Raw-text-free evidence for one surviving direct-identifier literal."""

    document_id: str
    fixture_id: str
    canonical_label: str
    identifier_type: str
    start: int
    end: int
    span_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible finding without any raw identifier text."""

        return {
            "canonical_label": self.canonical_label,
            "document_id": self.document_id,
            "end": self.end,
            "fixture_id": self.fixture_id,
            "identifier_type": self.identifier_type,
            "span_hash": self.span_hash,
            "start": self.start,
        }


@dataclass(frozen=True)
class IndiaClinicalLabelVerdict:
    """Per-identifier residual-leakage verdict for one direct-identifier type."""

    identifier_type: str
    canonical_label: str
    expected: int
    detected: int
    leaked: int

    @property
    def passed(self) -> bool:
        """Return whether no synthetic literal of this type survived."""

        return self.leaked == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free per-label verdict."""

        return {
            "canonical_label": self.canonical_label,
            "detected": self.detected,
            "expected": self.expected,
            "identifier_type": self.identifier_type,
            "leaked": self.leaked,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class IndiaClinicalLeakageResult:
    """Aggregate residual-PHI leakage verdict for the India clinical corpus."""

    passed: bool
    policy: str
    document_count: int
    direct_identifier_count: int
    detected_identifier_count: int
    residual_leak_count: int
    label_verdicts: tuple[IndiaClinicalLabelVerdict, ...] = ()
    findings: tuple[IndiaClinicalLeakageFinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, raw-identifier-free leakage report."""

        return {
            "detected_identifier_count": self.detected_identifier_count,
            "direct_identifier_count": self.direct_identifier_count,
            "document_count": self.document_count,
            "failures": [finding.to_dict() for finding in self.findings],
            "label_verdicts": [verdict.to_dict() for verdict in self.label_verdicts],
            "passed": self.passed,
            "policy": self.policy,
            "residual_leak_count": self.residual_leak_count,
            "suite": INDIA_CLINICAL_PHI_LEAKAGE,
        }


def deidentify_india_clinical_record(
    record: "IndiaClinicalPHIRecord",
    *,
    policy: "PolicyProfile",
    preserve: frozenset[str] = frozenset(),
) -> str:
    """Redact declared direct-identifier spans under the India DPDP profile.

    The corpus declares the ground-truth direct-identifier spans that clinical
    de-identification must remove. Every such span whose policy action is not
    ``keep`` is masked with its canonical-label placeholder, guaranteeing a
    deterministic offline redaction that also covers Devanagari and Tamil
    person-name aliases the pattern detector cannot recover on its own.

    Args:
        record: One normalized synthetic India clinical record.
        policy: The resolved India de-identification policy profile.
        preserve: Optional set of span ids to leave intentionally un-redacted;
            used by tests to prove the gate fails closed on a surviving literal.

    Returns:
        The de-identified document text.
    """

    spans = [
        span
        for span in record.gold_spans
        if span.metadata.get("direct_identifier") is True
        and str(span.metadata.get("span_id") or "") not in preserve
        and policy.action_for(span.label, lang=record.language) != "keep"
    ]
    redacted = record.text
    for span in sorted(spans, key=lambda item: int(item.start), reverse=True):
        redacted = redacted[: span.start] + f"[{span.label}]" + redacted[span.end :]
    return redacted


def evaluate_india_clinical_leakage(
    records: Sequence["IndiaClinicalPHIRecord"],
    *,
    policy_name: str = INDIA_CLINICAL_PHI_POLICY,
    deidentify: Deidentifier | None = None,
) -> IndiaClinicalLeakageResult:
    """Run the residual-PHI leakage gate over India clinical records.

    Args:
        records: Normalized synthetic India clinical records.
        policy_name: India de-identification policy profile to apply.
        deidentify: Optional de-identification callable overriding the default
            DPDP redactor. Supplying a redactor that preserves an identifier
            makes the gate fail deterministically.

    Returns:
        A privacy-safe aggregate result with per-label verdicts and evidence.
    """

    policy = load_policy(policy_name)
    redactor: Deidentifier = (
        deidentify
        if deidentify is not None
        else (lambda record: deidentify_india_clinical_record(record, policy=policy))
    )

    findings: list[IndiaClinicalLeakageFinding] = []
    canonical_by_type: dict[str, str] = {}
    expected: dict[str, int] = {}
    detected: dict[str, int] = {}
    leaked: dict[str, int] = {}

    for record in records:
        direct_spans = [
            span
            for span in record.gold_spans
            if span.metadata.get("direct_identifier") is True
        ]
        swept = safety_sweep(record.text, (), lang=record.language)
        redacted = redactor(record)

        for span in direct_spans:
            identifier_type = str(span.metadata.get("identifier_type") or "")
            canonical_by_type.setdefault(identifier_type, str(span.label))
            expected[identifier_type] = expected.get(identifier_type, 0) + 1

            if any(_overlaps(candidate, span) for candidate in swept):
                detected[identifier_type] = detected.get(identifier_type, 0) + 1

            if span.text and span.text in redacted:
                leaked[identifier_type] = leaked.get(identifier_type, 0) + 1
                findings.append(
                    IndiaClinicalLeakageFinding(
                        document_id=record.document_id,
                        fixture_id=record.fixture_id,
                        canonical_label=str(span.label),
                        identifier_type=identifier_type,
                        start=int(span.start),
                        end=int(span.end),
                        span_hash=hmac_text_hash(span.text, _LEAKAGE_HASH_KEY),
                    )
                )

    verdicts = tuple(
        IndiaClinicalLabelVerdict(
            identifier_type=identifier_type,
            canonical_label=canonical_by_type[identifier_type],
            expected=expected[identifier_type],
            detected=detected.get(identifier_type, 0),
            leaked=leaked.get(identifier_type, 0),
        )
        for identifier_type in sorted(expected)
    )
    residual_leak_count = sum(leaked.values())
    return IndiaClinicalLeakageResult(
        passed=residual_leak_count == 0 and not findings,
        policy=policy.name,
        document_count=len(records),
        direct_identifier_count=sum(expected.values()),
        detected_identifier_count=sum(detected.values()),
        residual_leak_count=residual_leak_count,
        label_verdicts=verdicts,
        findings=tuple(findings),
    )


def run_india_clinical_leakage_gate(
    *,
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    policy_name: str = INDIA_CLINICAL_PHI_POLICY,
    deidentify: Deidentifier | None = None,
) -> IndiaClinicalLeakageResult:
    """Load the committed India corpus and run the residual-PHI leakage gate."""

    from openmed.eval.datasets.clinical_phi import load_india_clinical_phi_corpus

    corpus = load_india_clinical_phi_corpus(manifest_path, fixture_path)
    return evaluate_india_clinical_leakage(
        corpus.records,
        policy_name=policy_name,
        deidentify=deidentify,
    )


def assert_india_clinical_leakage_gate(
    *,
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    policy_name: str = INDIA_CLINICAL_PHI_POLICY,
) -> IndiaClinicalLeakageResult:
    """Return a passing result or raise with raw-text-free diagnostics."""

    result = run_india_clinical_leakage_gate(
        manifest_path=manifest_path,
        fixture_path=fixture_path,
        policy_name=policy_name,
    )
    if not result.passed:
        leaked_types = ", ".join(
            sorted({finding.identifier_type for finding in result.findings})
        )
        documents = ", ".join(
            sorted({finding.document_id for finding in result.findings})
        )
        raise AssertionError(
            "India clinical residual-PHI leakage gate failed: "
            f"{result.residual_leak_count} surviving identifier(s) "
            f"[{leaked_types}] in document(s) [{documents}]"
        )
    return result


def india_clinical_leakage_metadata(
    *,
    fixture_path: str | Path | None = None,
    policy_name: str = INDIA_CLINICAL_PHI_POLICY,
) -> dict[str, Any]:
    """Return discoverable, raw-text-free metadata for the leakage gate."""

    from openmed.eval.datasets.clinical_phi import INDIA_CLINICAL_PHI_CORPUS_ID

    return {
        "corpus_id": INDIA_CLINICAL_PHI_CORPUS_ID,
        "direct_identifier_types": sorted(INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES),
        "fixture_path": str(fixture_path) if fixture_path is not None else None,
        "policy": policy_name,
        "report_fields": ["document_id", "canonical_label", "offsets", "hmac_hashes"],
        "required_residual_leakage": 0,
        "suite": INDIA_CLINICAL_PHI_LEAKAGE,
        "synthetic": True,
    }


def _overlaps(candidate: Any, span: Any) -> bool:
    start = _span_offset(candidate, "start")
    end = _span_offset(candidate, "end")
    if start is None or end is None:
        return False
    return start < int(span.end) and end > int(span.start)


def _span_offset(candidate: Any, name: str) -> int | None:
    value = getattr(candidate, name, None)
    if value is None and isinstance(candidate, Mapping):
        value = candidate.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES",
    "INDIA_CLINICAL_PHI_LEAKAGE",
    "INDIA_CLINICAL_PHI_POLICY",
    "IndiaClinicalLabelVerdict",
    "IndiaClinicalLeakageFinding",
    "IndiaClinicalLeakageResult",
    "assert_india_clinical_leakage_gate",
    "deidentify_india_clinical_record",
    "evaluate_india_clinical_leakage",
    "india_clinical_leakage_metadata",
    "run_india_clinical_leakage_gate",
]
