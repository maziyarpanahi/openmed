"""Policy-profile compliance suite for bundled de-identification profiles."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core import pii as pii_module
from openmed.core.labels import (
    CANONICAL_LABELS,
    DIRECT_IDENTIFIER,
    HIPAA_SAFE_HARBOR_CLASSES,
    LABEL_TO_HIPAA,
    policy_label_for,
)
from openmed.core.policy import PolicyName, PolicyProfile, load_policy
from openmed.core.schemas.span import ACTION_KEEP, ACTION_VALUES
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport
from openmed.processing.outputs import EntityPrediction, PredictionResult

POLICY_COMPLIANCE = "policy_compliance"
POLICY_COMPLIANCE_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "policy_compliance.jsonl"
)
BUNDLED_DEIDENTIFICATION_POLICIES: tuple[str, ...] = (
    PolicyName.HIPAA_SAFE_HARBOR.value,
    PolicyName.HIPAA_EXPERT_REVIEW_ASSIST.value,
    PolicyName.GDPR_PSEUDONYMIZATION.value,
    PolicyName.RESEARCH_LIMITED_DATASET.value,
    PolicyName.STRICT_NO_LEAK.value,
    PolicyName.CLINICAL_MINIMAL_REDACTION.value,
)
EXPECTATION_SOURCE = "openmed.core.policy.PolicyProfile.action_for"


@dataclass(frozen=True)
class PolicyActionExpectation:
    """One policy-derived expected action for a canonical label."""

    label: str
    policy_label: str
    hipaa_safe_harbor_class: str
    action: str
    source: str = EXPECTATION_SOURCE

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "action": self.action,
            "hipaa_safe_harbor_class": self.hipaa_safe_harbor_class,
            "label": self.label,
            "policy_label": self.policy_label,
            "source": self.source,
        }


@dataclass(frozen=True)
class PolicyComplianceFailure:
    """A policy-profile compliance failure without raw identifier text."""

    fixture_id: str
    label: str
    policy_label: str
    hipaa_safe_harbor_class: str | None
    start: int
    end: int
    span_hash: str
    expected_action: str
    observed_action: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "end": self.end,
            "expected_action": self.expected_action,
            "fixture_id": self.fixture_id,
            "hipaa_safe_harbor_class": self.hipaa_safe_harbor_class,
            "label": self.label,
            "observed_action": self.observed_action,
            "policy_label": self.policy_label,
            "reason": self.reason,
            "span_hash": self.span_hash,
            "start": self.start,
        }


@dataclass(frozen=True)
class PolicyProfileComplianceResult:
    """Per-profile compliance result for policy profile action behavior."""

    profile: str
    passed: bool
    fixture_count: int
    span_count: int
    residual_direct_identifier_count: int
    action_counts: Mapping[str, int]
    expected_action_counts: Mapping[str, int]
    covered_safe_harbor_classes: tuple[str, ...] = ()
    missing_safe_harbor_classes: tuple[str, ...] = ()
    failures: tuple[PolicyComplianceFailure, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation."""
        return {
            "action_counts": dict(self.action_counts),
            "covered_safe_harbor_classes": list(self.covered_safe_harbor_classes),
            "expected_action_counts": dict(self.expected_action_counts),
            "failure_count": len(self.failures),
            "failures": [failure.to_dict() for failure in self.failures],
            "fixture_count": self.fixture_count,
            "missing_safe_harbor_classes": list(self.missing_safe_harbor_classes),
            "passed": self.passed,
            "profile": self.profile,
            "residual_direct_identifier_count": (self.residual_direct_identifier_count),
            "span_count": self.span_count,
        }


def load_policy_compliance_fixtures(
    path: str | Path | None = None,
) -> list[BenchmarkFixture]:
    """Load synthetic policy compliance fixtures from JSONL."""
    fixture_path = Path(path) if path is not None else POLICY_COMPLIANCE_FIXTURE_PATH
    fixtures: list[BenchmarkFixture] = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{fixture_path}:{line_number} must contain a JSON object"
                )
            fixture = BenchmarkFixture.from_mapping(payload)
            _validate_fixture(
                fixture,
                fixture_path=fixture_path,
                line_number=line_number,
            )
            fixtures.append(fixture)
    _validate_unique_fixture_ids(fixtures, fixture_path=fixture_path)
    if not fixtures:
        raise ValueError(f"{fixture_path} does not contain policy fixtures")
    return fixtures


def policy_compliance_metadata(
    *,
    profiles: Sequence[str] = BUNDLED_DEIDENTIFICATION_POLICIES,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return suite metadata for policy compliance benchmark reports."""
    resolved_path = (
        Path(fixture_path) if fixture_path else POLICY_COMPLIANCE_FIXTURE_PATH
    )
    return {
        "expectation_source": EXPECTATION_SOURCE,
        "fixture_path": str(resolved_path),
        "profiles": list(profiles),
        "safe_harbor_classes": sorted(HIPAA_SAFE_HARBOR_CLASSES),
        "suite": POLICY_COMPLIANCE,
        "synthetic": True,
    }


def derive_profile_expectations(
    profile: str | PolicyProfile,
) -> dict[str, PolicyActionExpectation]:
    """Derive expected label actions from a loaded policy profile."""
    resolved = _resolve_profile(profile)
    expectations: dict[str, PolicyActionExpectation] = {}
    for label in sorted(CANONICAL_LABELS):
        action = str(resolved.action_for(label))
        if action not in ACTION_VALUES:
            raise ValueError(
                f"unsupported policy action {action!r} for "
                f"{resolved.name}.{label}; expected one of {ACTION_VALUES!r}"
            )
        expectations[label] = PolicyActionExpectation(
            label=label,
            policy_label=policy_label_for(label),
            hipaa_safe_harbor_class=str(LABEL_TO_HIPAA[label]),
            action=action,
        )
    return expectations


def evaluate_profile_compliance(
    profile: str | PolicyProfile,
    fixtures: Sequence[BenchmarkFixture],
) -> PolicyProfileComplianceResult:
    """Evaluate one profile against policy compliance fixtures."""
    resolved = _resolve_profile(profile)
    expectations = derive_profile_expectations(resolved)
    failures: list[PolicyComplianceFailure] = []
    action_counts: Counter[str] = Counter()
    expected_action_counts: Counter[str] = Counter()
    residual_direct_identifier_count = 0
    covered_safe_harbor_classes = _covered_safe_harbor_classes(fixtures)
    missing_safe_harbor_classes = _missing_safe_harbor_classes(
        resolved,
        covered_safe_harbor_classes,
    )

    for safe_harbor_class in missing_safe_harbor_classes:
        failures.append(
            PolicyComplianceFailure(
                fixture_id="__suite__",
                label="",
                policy_label="",
                hipaa_safe_harbor_class=safe_harbor_class,
                start=0,
                end=0,
                span_hash="",
                expected_action="non_keep",
                observed_action=ACTION_KEEP,
                reason="safe_harbor_class_missing",
            )
        )

    span_count = 0
    for fixture in fixtures:
        deidentified_text, observed_actions = _deidentify_fixture_with_policy(
            fixture,
            profile=resolved,
            expectations=expectations,
        )
        for span in fixture.gold_spans:
            span_count += 1
            expectation = expectations[span.label]
            expected_action_counts[expectation.action] += 1
            observed_action = observed_actions.get(_span_key(span), ACTION_KEEP)
            action_counts[observed_action] += 1

            if observed_action != expectation.action:
                failures.append(
                    _failure(
                        fixture,
                        span,
                        expectation=expectation,
                        observed_action=observed_action,
                        reason="policy_action_mismatch",
                    )
                )

            surface_survived = _surface_survived(span, deidentified_text)
            if expectation.policy_label == DIRECT_IDENTIFIER and surface_survived:
                residual_direct_identifier_count += 1
                failures.append(
                    _failure(
                        fixture,
                        span,
                        expectation=expectation,
                        observed_action=observed_action,
                        reason="direct_identifier_residual",
                    )
                )

            if (
                resolved.name == PolicyName.HIPAA_SAFE_HARBOR.value
                and _fixture_safe_harbor_class(span) is not None
                and expectation.action == ACTION_KEEP
            ):
                failures.append(
                    _failure(
                        fixture,
                        span,
                        expectation=expectation,
                        observed_action=observed_action,
                        reason="safe_harbor_action_keeps_identifier",
                    )
                )

    return PolicyProfileComplianceResult(
        profile=resolved.name,
        passed=not failures,
        fixture_count=len(fixtures),
        span_count=span_count,
        residual_direct_identifier_count=residual_direct_identifier_count,
        action_counts=_action_counts(action_counts),
        expected_action_counts=_action_counts(expected_action_counts),
        covered_safe_harbor_classes=tuple(sorted(covered_safe_harbor_classes)),
        missing_safe_harbor_classes=tuple(sorted(missing_safe_harbor_classes)),
        failures=tuple(failures),
    )


def run_policy_compliance(
    *,
    fixture_path: str | Path | None = None,
    fixtures: Sequence[BenchmarkFixture] | None = None,
    profiles: Sequence[str | PolicyProfile] = BUNDLED_DEIDENTIFICATION_POLICIES,
    model_name: str = "policy-profile-compliance",
    device: str = "cpu",
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run all bundled policy profiles and return a BenchmarkReport."""
    loaded_fixtures = (
        tuple(fixtures)
        if fixtures is not None
        else tuple(load_policy_compliance_fixtures(fixture_path))
    )
    results = [
        evaluate_profile_compliance(profile, loaded_fixtures) for profile in profiles
    ]
    metrics = {
        "overall_passed": all(result.passed for result in results),
        "profile_count": len(results),
        "profiles": {result.profile: result.to_dict() for result in results},
        "profiles_failed": sum(1 for result in results if not result.passed),
        "profiles_passed": sum(1 for result in results if result.passed),
        "residual_direct_identifier_count": sum(
            result.residual_direct_identifier_count for result in results
        ),
    }
    metadata = policy_compliance_metadata(
        profiles=tuple(_resolve_profile(profile).name for profile in profiles),
        fixture_path=fixture_path,
    )
    metadata["fixture_ids"] = [fixture.fixture_id for fixture in loaded_fixtures]

    return BenchmarkReport(
        suite=POLICY_COMPLIANCE,
        model_name=model_name,
        device=device,
        fixture_count=len(loaded_fixtures),
        metrics=metrics,
        generated_at=generated_at,
        metadata=metadata,
    )


def _resolve_profile(profile: str | PolicyProfile) -> PolicyProfile:
    return load_policy(profile)


def _deidentify_fixture_with_policy(
    fixture: BenchmarkFixture,
    *,
    profile: PolicyProfile,
    expectations: Mapping[str, PolicyActionExpectation],
) -> tuple[str, dict[tuple[int, int, str], str]]:
    entities: list[EntityPrediction] = []
    for span in fixture.gold_spans:
        expectation = expectations[span.label]
        if expectation.action == ACTION_KEEP:
            continue
        entities.append(
            EntityPrediction(
                text=span.text or fixture.text[span.start : span.end],
                label=span.label,
                confidence=1.0,
                start=span.start,
                end=span.end,
                metadata={
                    **dict(span.metadata),
                    "canonical_label": span.label,
                    "policy_action": {
                        "action": expectation.action,
                        "policy": profile.name,
                        "schema_version": profile.schema_version,
                        "source": "policy_profile",
                    },
                    "policy_label": expectation.policy_label,
                },
            )
        )

    pii_result = PredictionResult(
        text=fixture.text,
        entities=entities,
        model_name=f"policy:{profile.name}",
        timestamp="policy-compliance",
        metadata={"policy": profile.to_dict(), "suite": POLICY_COMPLIANCE},
    )
    deidentified = pii_module._build_deidentification_result(
        fixture.text,
        pii_result,
        effective_method="mask",
        keep_year=False,
        date_shift_days=None,
        keep_mapping=profile.keep_mapping,
        lang=fixture.language,
        consistent=True,
        seed=0,
        locale=None,
        model_name=f"policy:{profile.name}",
        confidence_threshold=1.0,
        normalize_accents=None,
        use_smart_merging=False,
        use_safety_sweep=False,
        reversible_ids=profile.reversible_id,
        policy_name=profile.name,
        policy=profile.name,
        audit=False,
    )
    observed_actions: dict[tuple[int, int, str], str] = {}
    for entity in deidentified.pii_entities:
        policy_action = (entity.metadata or {}).get("policy_action")
        observed_action = ""
        if isinstance(policy_action, Mapping):
            observed_action = str(policy_action.get("action") or "")
        observed_actions[
            (int(entity.start or 0), int(entity.end or 0), entity.canonical_label or "")
        ] = observed_action
    return deidentified.deidentified_text, observed_actions


def _validate_fixture(
    fixture: BenchmarkFixture,
    *,
    fixture_path: Path,
    line_number: int,
) -> None:
    if fixture.metadata.get("synthetic") is not True:
        raise ValueError(
            f"{fixture_path}:{line_number} must be marked metadata.synthetic=true"
        )
    if fixture.metadata.get("suite") != POLICY_COMPLIANCE:
        raise ValueError(
            f"{fixture_path}:{line_number} must be marked suite={POLICY_COMPLIANCE!r}"
        )
    if not fixture.gold_spans:
        raise ValueError(f"{fixture_path}:{line_number} must include gold spans")

    for span in fixture.gold_spans:
        if span.start < 0 or span.end <= span.start or span.end > len(fixture.text):
            raise ValueError(
                f"{fixture_path}:{line_number} has invalid span offsets "
                f"{span.start}:{span.end}"
            )
        if fixture.text[span.start : span.end] != span.text:
            raise ValueError(
                f"{fixture_path}:{line_number} has span text mismatch for "
                f"{span.label} at {span.start}:{span.end}"
            )
        safe_harbor_class = _fixture_safe_harbor_class(span)
        if (
            safe_harbor_class is not None
            and safe_harbor_class not in HIPAA_SAFE_HARBOR_CLASSES
        ):
            raise ValueError(
                f"{fixture_path}:{line_number} has unknown Safe Harbor class "
                f"{safe_harbor_class!r}"
            )


def _validate_unique_fixture_ids(
    fixtures: Sequence[BenchmarkFixture],
    *,
    fixture_path: Path,
) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen and fixture.fixture_id not in duplicates:
            duplicates.append(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        quoted = ", ".join(repr(value) for value in duplicates)
        raise ValueError(f"{fixture_path} has duplicate fixture id(s): {quoted}")


def _missing_safe_harbor_classes(
    profile: PolicyProfile,
    covered_classes: set[str],
) -> set[str]:
    if profile.name != PolicyName.HIPAA_SAFE_HARBOR.value:
        return set()
    return set(HIPAA_SAFE_HARBOR_CLASSES) - covered_classes


def _covered_safe_harbor_classes(fixtures: Sequence[BenchmarkFixture]) -> set[str]:
    return {
        safe_harbor_class
        for fixture in fixtures
        for span in fixture.gold_spans
        for safe_harbor_class in [_fixture_safe_harbor_class(span)]
        if safe_harbor_class is not None
    }


def _fixture_safe_harbor_class(span: EvalSpan) -> str | None:
    value = span.metadata.get("hipaa_safe_harbor_class")
    if value is None:
        return None
    return str(value)


def _surface_survived(span: EvalSpan, deidentified_text: str) -> bool:
    surface = span.text.strip()
    return bool(surface and surface in deidentified_text)


def _span_key(span: EvalSpan) -> tuple[int, int, str]:
    return (span.start, span.end, span.label)


def _failure(
    fixture: BenchmarkFixture,
    span: EvalSpan,
    *,
    expectation: PolicyActionExpectation,
    observed_action: str,
    reason: str,
) -> PolicyComplianceFailure:
    return PolicyComplianceFailure(
        fixture_id=fixture.fixture_id,
        label=span.label,
        policy_label=expectation.policy_label,
        hipaa_safe_harbor_class=_fixture_safe_harbor_class(span),
        start=span.start,
        end=span.end,
        span_hash=_span_hash(span),
        expected_action=expectation.action,
        observed_action=observed_action,
        reason=reason,
    )


def _span_hash(span: EvalSpan) -> str:
    return hashlib.sha256(span.text.encode("utf-8")).hexdigest()[:16]


def _action_counts(counter: Counter[str]) -> dict[str, int]:
    return {action: int(counter.get(action, 0)) for action in ACTION_VALUES}


__all__ = [
    "BUNDLED_DEIDENTIFICATION_POLICIES",
    "EXPECTATION_SOURCE",
    "POLICY_COMPLIANCE",
    "POLICY_COMPLIANCE_FIXTURE_PATH",
    "PolicyActionExpectation",
    "PolicyComplianceFailure",
    "PolicyProfileComplianceResult",
    "derive_profile_expectations",
    "evaluate_profile_compliance",
    "load_policy_compliance_fixtures",
    "policy_compliance_metadata",
    "run_policy_compliance",
]
