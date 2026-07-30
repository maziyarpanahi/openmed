"""Fail-closed residual-PHI leakage gate for the synthetic India corpus.

The gate runs clinical de-identification over every synthetic India clinical
fixture under the India DPDP profile and the supported multilingual detection
path, then scans each de-identified document for the original synthetic direct
identifiers declared by the corpus (ABHA, Aadhaar, PAN, phone, person-name
aliases across Latin/Devanagari/Tamil, and street addresses). A surviving
literal fails the gate with privacy-safe document, label, offset, and hash
evidence; raw identifier surfaces never enter the result, reports, or logs.

The gate verdict is the conjunction of two independent checks:

* **Zero residual after redaction.** Redaction is driven by the corpus
  ground-truth direct-identifier spans under the DPDP profile, so it
  deterministically removes even the Devanagari and Tamil person-name aliases
  that the offline pattern detector cannot recover on its own.
* **Must-detect recall floor.** Because gold-driven redaction would otherwise
  let a detector regression pass silently, the gate also asserts that the
  offline detector (:func:`openmed.core.safety_sweep.safety_sweep`) actually
  finds every gold span whose identifier type is in the must-detect set
  (:data:`INDIA_CLINICAL_MUST_DETECT_TYPES`). These are the structured direct
  identifiers the detector handles at 100% recall on the shipped corpus:
  Aadhaar, ABHA, and PAN. A miss on any of them fails the gate.

Detector-capability gap (recorded, not gated): the offline detector cannot
reliably recover ``person_name``, ``indian_phone``, or ``street_address`` on
this corpus (person-name recall is 0/3). Their detection coverage is recorded
in every result for transparency but is deliberately NOT gated, so a PASS must
never be read as evidence that the production pipeline detects and redacts
Indic-script names, India phone numbers, or street addresses on its own. The
zero-residual guarantee for those types comes from the corpus gold spans, not
from detection.

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
# Structured direct identifiers the offline detector recovers at 100% recall on
# the shipped corpus; a miss on any of these is a gated detector regression.
INDIA_CLINICAL_MUST_DETECT_TYPES = frozenset({"aadhaar", "abha", "pan"})

RESIDUAL_IDENTIFIER_SURVIVED = "residual_identifier_survived"
MUST_DETECT_IDENTIFIER_MISSED = "must_detect_identifier_missed"

_LEAKAGE_HASH_KEY = "openmed-india-clinical-phi-leakage"

# Redaction callable: given one corpus record, return its de-identified text.
Deidentifier = Callable[["IndiaClinicalPHIRecord"], str]


@dataclass(frozen=True)
class IndiaClinicalLeakageFinding:
    """Raw-text-free evidence for one gate failure on a direct identifier."""

    document_id: str
    fixture_id: str
    canonical_label: str
    identifier_type: str
    reason: str
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
            "reason": self.reason,
            "span_hash": self.span_hash,
            "start": self.start,
        }


@dataclass(frozen=True)
class IndiaClinicalLabelVerdict:
    """Per-identifier verdict combining residual leakage and detection recall."""

    identifier_type: str
    canonical_label: str
    expected: int
    detected: int
    leaked: int
    must_detect: bool

    @property
    def detection_complete(self) -> bool:
        """Return whether every span of this type was detected offline."""

        return self.detected >= self.expected

    @property
    def passed(self) -> bool:
        """Return whether this type leaked nothing and cleared any recall floor.

        Recall is only gated for must-detect types; partial-recall types are
        recorded but never fail the gate on detection alone.
        """

        if self.leaked != 0:
            return False
        return self.detection_complete if self.must_detect else True

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free per-label verdict."""

        return {
            "canonical_label": self.canonical_label,
            "detected": self.detected,
            "detection_complete": self.detection_complete,
            "expected": self.expected,
            "identifier_type": self.identifier_type,
            "leaked": self.leaked,
            "must_detect": self.must_detect,
            "passed": self.passed,
            "role": "must_detect" if self.must_detect else "recorded_not_gated",
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
    must_detect_miss_count: int
    must_detect_types: tuple[str, ...] = ()
    recorded_not_gated_types: tuple[str, ...] = ()
    label_verdicts: tuple[IndiaClinicalLabelVerdict, ...] = ()
    findings: tuple[IndiaClinicalLeakageFinding, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, raw-identifier-free leakage report."""

        return {
            "detected_identifier_count": self.detected_identifier_count,
            "detector_capability_gap": (
                "Recall is gated only for must_detect_types; "
                "recorded_not_gated_types (Indic person names, India phones, "
                "and street addresses) are recorded for transparency and are "
                "redacted via corpus gold spans, not offline detection."
            ),
            "direct_identifier_count": self.direct_identifier_count,
            "document_count": self.document_count,
            "failures": [finding.to_dict() for finding in self.findings],
            "label_verdicts": [verdict.to_dict() for verdict in self.label_verdicts],
            "must_detect_miss_count": self.must_detect_miss_count,
            "must_detect_types": list(self.must_detect_types),
            "passed": self.passed,
            "policy": self.policy,
            "recorded_not_gated_types": list(self.recorded_not_gated_types),
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
    """Run the residual-PHI leakage and must-detect recall gate.

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
    must_detect_miss_count = 0

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

            was_detected = any(_overlaps(candidate, span) for candidate in swept)
            if was_detected:
                detected[identifier_type] = detected.get(identifier_type, 0) + 1
            elif identifier_type in INDIA_CLINICAL_MUST_DETECT_TYPES:
                must_detect_miss_count += 1
                findings.append(_finding(record, span, MUST_DETECT_IDENTIFIER_MISSED))

            if span.text and span.text in redacted:
                leaked[identifier_type] = leaked.get(identifier_type, 0) + 1
                findings.append(_finding(record, span, RESIDUAL_IDENTIFIER_SURVIVED))

    verdicts = tuple(
        IndiaClinicalLabelVerdict(
            identifier_type=identifier_type,
            canonical_label=canonical_by_type[identifier_type],
            expected=expected[identifier_type],
            detected=detected.get(identifier_type, 0),
            leaked=leaked.get(identifier_type, 0),
            must_detect=identifier_type in INDIA_CLINICAL_MUST_DETECT_TYPES,
        )
        for identifier_type in sorted(expected)
    )
    residual_leak_count = sum(leaked.values())
    recorded_not_gated = tuple(
        sorted(
            identifier_type
            for identifier_type in expected
            if identifier_type not in INDIA_CLINICAL_MUST_DETECT_TYPES
        )
    )
    gated_types = tuple(
        sorted(
            identifier_type
            for identifier_type in expected
            if identifier_type in INDIA_CLINICAL_MUST_DETECT_TYPES
        )
    )
    return IndiaClinicalLeakageResult(
        passed=(
            residual_leak_count == 0 and must_detect_miss_count == 0 and not findings
        ),
        policy=policy.name,
        document_count=len(records),
        direct_identifier_count=sum(expected.values()),
        detected_identifier_count=sum(detected.values()),
        residual_leak_count=residual_leak_count,
        must_detect_miss_count=must_detect_miss_count,
        must_detect_types=gated_types,
        recorded_not_gated_types=recorded_not_gated,
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
        reasons = ", ".join(sorted({finding.reason for finding in result.findings}))
        leaked_types = ", ".join(
            sorted({finding.identifier_type for finding in result.findings})
        )
        documents = ", ".join(
            sorted({finding.document_id for finding in result.findings})
        )
        raise AssertionError(
            "India clinical residual-PHI leakage gate failed: "
            f"{len(result.findings)} finding(s) "
            f"(residual={result.residual_leak_count}, "
            f"must_detect_miss={result.must_detect_miss_count}); "
            f"reasons [{reasons}]; identifier(s) [{leaked_types}]; "
            f"document(s) [{documents}]"
        )
    return result


def india_clinical_leakage_metadata(
    *,
    fixture_path: str | Path | None = None,
    policy_name: str = INDIA_CLINICAL_PHI_POLICY,
) -> dict[str, Any]:
    """Return discoverable, raw-text-free metadata for the leakage gate."""

    from openmed.eval.datasets.clinical_phi import INDIA_CLINICAL_PHI_CORPUS_ID

    recorded_not_gated = sorted(
        INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES - INDIA_CLINICAL_MUST_DETECT_TYPES
    )
    return {
        "corpus_id": INDIA_CLINICAL_PHI_CORPUS_ID,
        "direct_identifier_types": sorted(INDIA_CLINICAL_DIRECT_IDENTIFIER_TYPES),
        "fixture_path": str(fixture_path) if fixture_path is not None else None,
        "must_detect_types": sorted(INDIA_CLINICAL_MUST_DETECT_TYPES),
        "policy": policy_name,
        "recorded_not_gated_types": recorded_not_gated,
        "report_fields": ["document_id", "canonical_label", "offsets", "hmac_hashes"],
        "required_residual_leakage": 0,
        "suite": INDIA_CLINICAL_PHI_LEAKAGE,
        "synthetic": True,
    }


def _finding(
    record: "IndiaClinicalPHIRecord",
    span: Any,
    reason: str,
) -> IndiaClinicalLeakageFinding:
    return IndiaClinicalLeakageFinding(
        document_id=record.document_id,
        fixture_id=record.fixture_id,
        canonical_label=str(span.label),
        identifier_type=str(span.metadata.get("identifier_type") or ""),
        reason=reason,
        start=int(span.start),
        end=int(span.end),
        span_hash=hmac_text_hash(span.text, _LEAKAGE_HASH_KEY),
    )


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
    "INDIA_CLINICAL_MUST_DETECT_TYPES",
    "INDIA_CLINICAL_PHI_LEAKAGE",
    "INDIA_CLINICAL_PHI_POLICY",
    "MUST_DETECT_IDENTIFIER_MISSED",
    "RESIDUAL_IDENTIFIER_SURVIVED",
    "IndiaClinicalLabelVerdict",
    "IndiaClinicalLeakageFinding",
    "IndiaClinicalLeakageResult",
    "assert_india_clinical_leakage_gate",
    "deidentify_india_clinical_record",
    "evaluate_india_clinical_leakage",
    "india_clinical_leakage_metadata",
    "run_india_clinical_leakage_gate",
]
