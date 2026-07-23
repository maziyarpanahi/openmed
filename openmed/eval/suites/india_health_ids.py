"""Synthetic zero-leakage gate for Indian health identifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core import pii as pii_module
from openmed.core.labels import normalize_label
from openmed.core.policy import load_policy
from openmed.core.safety_sweep import hashed_span_surface, safety_sweep
from openmed.eval.golden import GoldenFixture, load_golden_fixtures
from openmed.processing.outputs import PredictionResult

INDIA_HEALTH_ID_LEAKAGE = "india_health_id_leakage"
INDIA_HEALTH_ID_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "india_health_ids.json"
)
INDIA_HEALTH_ID_ENTITY_TYPES = frozenset(
    {"abha_number", "abha_address", "upi_id", "ration_card"}
)


@dataclass(frozen=True)
class IndiaHealthIdGateFailure:
    """Raw-text-free failure emitted by the India health-ID gate."""

    fixture_id: str
    reason: str
    identifier_type: str
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "reason": self.reason,
            "identifier_type": self.identifier_type,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class IndiaHealthIdGateResult:
    """Result of the synthetic Indian health-identifier leakage gate."""

    passed: bool
    fixture_count: int
    expected_entity_count: int
    detected_entity_count: int
    leaked_entity_count: int
    false_accept_count: int
    entity_leakage: float
    failures: tuple[IndiaHealthIdGateFailure, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": INDIA_HEALTH_ID_LEAKAGE,
            "passed": self.passed,
            "fixture_count": self.fixture_count,
            "expected_entity_count": self.expected_entity_count,
            "detected_entity_count": self.detected_entity_count,
            "leaked_entity_count": self.leaked_entity_count,
            "false_accept_count": self.false_accept_count,
            "entity_leakage": self.entity_leakage,
            "failures": [failure.to_dict() for failure in self.failures],
        }


def load_india_health_id_fixtures(
    path: str | Path | None = None,
) -> list[GoldenFixture]:
    """Load the synthetic valid/invalid India health-ID fixtures."""

    fixture_path = Path(path) if path is not None else INDIA_HEALTH_ID_FIXTURE_PATH
    fixtures = load_golden_fixtures(fixture_path)
    if not fixtures:
        raise ValueError(f"{fixture_path} does not contain India health-ID fixtures")
    return fixtures


def india_health_id_metadata(
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return raw-text-free metadata for the India health-ID suite."""

    resolved = Path(fixture_path) if fixture_path else INDIA_HEALTH_ID_FIXTURE_PATH
    return {
        "suite": INDIA_HEALTH_ID_LEAKAGE,
        "fixture_path": str(resolved),
        "policy": "india_health_id",
        "entity_types": sorted(INDIA_HEALTH_ID_ENTITY_TYPES),
        "synthetic": True,
        "required_entity_leakage": 0.0,
        "required_false_accepts": 0,
    }


def run_india_health_id_leakage_gate(
    *,
    fixture_path: str | Path | None = None,
    fixtures: Sequence[GoldenFixture] | None = None,
) -> IndiaHealthIdGateResult:
    """Run the offline India health-ID detection, policy, and leakage gate."""

    loaded = tuple(
        fixtures
        if fixtures is not None
        else load_india_health_id_fixtures(fixture_path)
    )
    policy = load_policy("india_health_id")
    failures: list[IndiaHealthIdGateFailure] = []
    expected_count = 0
    detected_count = 0
    leaked_count = 0
    false_accept_count = 0

    for fixture in loaded:
        detected = safety_sweep(fixture.text, (), lang=fixture.language)
        india_entities = [
            entity
            for entity in detected
            if _identifier_type(entity) in INDIA_HEALTH_ID_ENTITY_TYPES
        ]
        detected_by_span = {
            (int(entity.start or 0), int(entity.end or 0), _identifier_type(entity))
            for entity in india_entities
        }

        for span in fixture.gold_spans:
            expected_count += 1
            identifier_type = str(span.metadata.get("identifier_type") or "")
            key = (span.start, span.end, identifier_type)
            if key in detected_by_span:
                detected_count += 1
            else:
                failures.append(
                    _failure(
                        fixture,
                        reason="expected_identifier_not_detected",
                        identifier_type=identifier_type,
                        start=span.start,
                        end=span.end,
                    )
                )

        for negative in fixture.metadata.get("hard_negatives", ()):
            if not isinstance(negative, Mapping):
                continue
            start = int(negative["start"])
            end = int(negative["end"])
            overlaps = [
                entity
                for entity in india_entities
                if int(entity.start or 0) < end and int(entity.end or 0) > start
            ]
            for entity in overlaps:
                false_accept_count += 1
                failures.append(
                    _failure(
                        fixture,
                        reason=str(negative.get("reason") or "hard_negative_accepted"),
                        identifier_type=_identifier_type(entity),
                        start=start,
                        end=end,
                    )
                )

        for entity in india_entities:
            metadata = dict(entity.metadata or {})
            metadata["policy_action"] = {
                "policy": policy.name,
                "schema_version": policy.schema_version,
                "action": policy.action_for(
                    normalize_label(entity.label, lang=fixture.language),
                    lang=fixture.language,
                ),
                "source": "policy_profile",
            }
            entity.metadata = metadata

        prediction = PredictionResult(
            text=fixture.text,
            entities=india_entities,
            model_name="india-health-id-offline-gate",
            timestamp="synthetic-gate",
            metadata={"policy": policy.to_dict(), "synthetic": True},
        )
        result = pii_module._build_deidentification_result(
            fixture.text,
            prediction,
            effective_method="replace",
            keep_year=False,
            date_shift_days=None,
            keep_mapping=False,
            lang=fixture.language,
            consistent=True,
            seed=0,
            locale=None,
            model_name="india-health-id-offline-gate",
            confidence_threshold=1.0,
            normalize_accents=None,
            use_smart_merging=False,
            use_safety_sweep=True,
            policy_name=policy.name,
            policy=policy.name,
            audit=False,
        )
        for span in fixture.gold_spans:
            if span.text and span.text in result.deidentified_text:
                leaked_count += 1
                failures.append(
                    _failure(
                        fixture,
                        reason="identifier_surface_survived",
                        identifier_type=str(span.metadata.get("identifier_type") or ""),
                        start=span.start,
                        end=span.end,
                    )
                )

    leakage = leaked_count / expected_count if expected_count else 0.0
    passed = not failures and leakage == 0.0 and false_accept_count == 0
    return IndiaHealthIdGateResult(
        passed=passed,
        fixture_count=len(loaded),
        expected_entity_count=expected_count,
        detected_entity_count=detected_count,
        leaked_entity_count=leaked_count,
        false_accept_count=false_accept_count,
        entity_leakage=leakage,
        failures=tuple(failures),
    )


def assert_india_health_id_leakage_gate(
    *,
    fixture_path: str | Path | None = None,
) -> IndiaHealthIdGateResult:
    """Return a passing gate result or raise with raw-text-free diagnostics."""

    result = run_india_health_id_leakage_gate(fixture_path=fixture_path)
    if not result.passed:
        reasons = ", ".join(sorted({failure.reason for failure in result.failures}))
        raise AssertionError(f"India health-ID leakage gate failed: {reasons}")
    return result


def _identifier_type(entity: Any) -> str:
    metadata = getattr(entity, "metadata", None) or {}
    sweep = metadata.get("safety_sweep")
    if isinstance(sweep, Mapping):
        return str(sweep.get("entity_type") or entity.label)
    return str(getattr(entity, "label", ""))


def _failure(
    fixture: GoldenFixture,
    *,
    reason: str,
    identifier_type: str,
    start: int,
    end: int,
) -> IndiaHealthIdGateFailure:
    return IndiaHealthIdGateFailure(
        fixture_id=fixture.fixture_id,
        reason=reason,
        identifier_type=identifier_type,
        evidence=hashed_span_surface(
            fixture.text,
            start,
            end,
            label=identifier_type,
        ),
    )


__all__ = [
    "INDIA_HEALTH_ID_ENTITY_TYPES",
    "INDIA_HEALTH_ID_FIXTURE_PATH",
    "INDIA_HEALTH_ID_LEAKAGE",
    "IndiaHealthIdGateFailure",
    "IndiaHealthIdGateResult",
    "assert_india_health_id_leakage_gate",
    "india_health_id_metadata",
    "load_india_health_id_fixtures",
    "run_india_health_id_leakage_gate",
]
