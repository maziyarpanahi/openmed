"""Typed GraphQL schema for the OpenMed service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

import strawberry
from graphql import GraphQLError
from strawberry.fastapi import BaseContext
from strawberry.scalars import JSON
from strawberry.types import Info

from openmed.core.labels import CANONICAL_LABELS, policy_label_for
from openmed.core.policy import PolicyProfile, load_policy
from openmed.risk import risk_report

from .runtime import ServiceRuntime
from .schemas import AnalyzeRequest, PIIDeidentifyRequest

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
SAFE_RESOLVER_ERROR = "The OpenMed GraphQL operation could not be completed."


class OpenMedGraphQLContext(BaseContext):
    """Request context carrying the service's shared runtime."""

    def __init__(self, runtime: ServiceRuntime) -> None:
        self.runtime = runtime


@strawberry.input
class AnalyzeInput:
    """Inputs accepted by the ``analyze`` query."""

    text: str
    model_name: str = "disease_detection_superclinical"
    confidence_threshold: Optional[float] = 0.0
    group_entities: bool = False
    aggregation_strategy: Optional[str] = "simple"
    sentence_detection: bool = True
    sentence_language: str = "en"
    sentence_clean: bool = False
    use_fast_tokenizer: bool = True
    keep_alive: Optional[JSON] = None

    def to_request(self) -> AnalyzeRequest:
        """Validate and convert this input to the REST pipeline contract."""
        return AnalyzeRequest(
            text=self.text,
            model_name=self.model_name,
            confidence_threshold=self.confidence_threshold,
            group_entities=self.group_entities,
            aggregation_strategy=self.aggregation_strategy,
            sentence_detection=self.sentence_detection,
            sentence_language=self.sentence_language,
            sentence_clean=self.sentence_clean,
            use_fast_tokenizer=self.use_fast_tokenizer,
            keep_alive=self.keep_alive,
        )


@strawberry.input
class DeidentifyInput:
    """Inputs accepted by the ``deidentify`` query."""

    text: str
    method: str = "mask"
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.7
    keep_year: bool = False
    shift_dates: Optional[bool] = None
    date_shift_days: Optional[int] = None
    keep_mapping: bool = False
    policy: Optional[str] = None
    use_smart_merging: bool = True
    use_safety_sweep: bool = True
    lang: str = "en"
    normalize_accents: Optional[bool] = None
    keep_alive: Optional[JSON] = None

    def to_request(self) -> PIIDeidentifyRequest:
        """Validate and convert this input to the REST pipeline contract."""
        return PIIDeidentifyRequest(
            text=self.text,
            method=self.method,
            model_name=self.model_name,
            confidence_threshold=self.confidence_threshold,
            keep_year=self.keep_year,
            shift_dates=self.shift_dates,
            date_shift_days=self.date_shift_days,
            keep_mapping=self.keep_mapping,
            policy=self.policy,
            use_smart_merging=self.use_smart_merging,
            use_safety_sweep=self.use_safety_sweep,
            lang=self.lang,
            normalize_accents=self.normalize_accents,
            keep_alive=self.keep_alive,
        )


@strawberry.type
class Entity:
    """A model-produced entity with source offsets."""

    text: str
    label: str
    confidence: float
    start: Optional[int]
    end: Optional[int]
    metadata: JSON

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> Entity:
        """Build a typed entity from a service payload."""
        return cls(
            text=str(item.get("text") or ""),
            label=str(item.get("label") or item.get("entity_type") or "UNKNOWN"),
            confidence=_float(item.get("confidence")),
            start=_optional_int(item.get("start")),
            end=_optional_int(item.get("end")),
            metadata=_json_mapping(item.get("metadata")),
        )


@strawberry.type(name="OpenMedSpan")
class OpenMedSpanType:
    """Canonical OpenMed span fields exposed for selective retrieval."""

    label: str
    start: Optional[int]
    end: Optional[int]
    text: Optional[str]
    schema_version: Optional[int]
    doc_id: Optional[str]
    text_hash: Optional[str]
    entity_type: Optional[str]
    canonical_label: Optional[str]
    policy_label: Optional[str]
    regulatory_tags: list[str]
    score: Optional[float]
    detector: Optional[str]
    evidence: JSON
    action: Optional[str]
    replacement: Optional[str]
    reversible_id: Optional[str]
    section: Optional[str]
    metadata: JSON

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> OpenMedSpanType:
        """Build a canonical span view from a span or entity mapping."""
        entity_type = _optional_str(item.get("entity_type"))
        canonical_label = _optional_str(item.get("canonical_label"))
        label = str(item.get("label") or canonical_label or entity_type or "UNKNOWN")
        score_value = item.get("score", item.get("confidence"))
        return cls(
            label=label,
            start=_optional_int(item.get("start")),
            end=_optional_int(item.get("end")),
            text=_optional_str(item.get("text")),
            schema_version=_optional_int(item.get("schema_version")),
            doc_id=_optional_str(item.get("doc_id")),
            text_hash=_optional_str(item.get("text_hash")),
            entity_type=entity_type,
            canonical_label=canonical_label,
            policy_label=_optional_str(item.get("policy_label")),
            regulatory_tags=[str(tag) for tag in item.get("regulatory_tags") or ()],
            score=_optional_float(score_value),
            detector=_optional_str(item.get("detector")),
            evidence=_json_mapping(item.get("evidence")),
            action=_optional_str(item.get("action")),
            replacement=_optional_str(
                item.get("replacement", item.get("redacted_text"))
            ),
            reversible_id=_optional_str(item.get("reversible_id")),
            section=_optional_str(item.get("section")),
            metadata=_json_mapping(item.get("metadata")),
        )


@strawberry.type
class PolicyAction:
    """One label-to-action entry from a policy profile."""

    label: str
    action: str


@strawberry.type(name="PolicyProfile")
class PolicyProfileType:
    """Selected de-identification policy details."""

    name: str
    schema_version: int
    posture: str
    threshold_profile: str
    default_action: str
    default_action_bias: str
    arbitration_mode: str
    strict_no_leak: bool
    safety_sweep_mandatory: bool
    keep_mapping: bool
    reversible_id: bool
    forced_cascade_tiers: list[str]
    actions: list[PolicyAction]
    policy_label_actions: list[PolicyAction]
    metadata: JSON

    @classmethod
    def from_profile(cls, profile: PolicyProfile) -> PolicyProfileType:
        """Build a field-selectable policy profile."""
        return cls(
            name=profile.name,
            schema_version=profile.schema_version,
            posture=profile.posture,
            threshold_profile=profile.threshold_profile,
            default_action=profile.default_action,
            default_action_bias=profile.default_action_bias,
            arbitration_mode=profile.arbitration_mode,
            strict_no_leak=profile.strict_no_leak,
            safety_sweep_mandatory=profile.safety_sweep_mandatory,
            keep_mapping=profile.keep_mapping,
            reversible_id=profile.reversible_id,
            forced_cascade_tiers=list(profile.forced_cascade_tiers),
            actions=[
                PolicyAction(label=label, action=action)
                for label, action in sorted(profile.actions.items())
            ],
            policy_label_actions=[
                PolicyAction(label=label, action=action)
                for label, action in sorted(profile.policy_label_actions.items())
            ],
            metadata=dict(profile.metadata),
        )


@strawberry.type
class RiskFacets:
    """Aggregate residual-risk facets without source identifier values."""

    leakage_rate: float
    reidentification_rate: float
    minimum_k: int
    singleton_record_count: int
    quasi_identifier_count: int

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> RiskFacets:
        """Compute safe aggregate risk facets for a de-identification result."""
        report = risk_report(
            str(result.get("deidentified_text") or ""),
            original=str(result.get("original_text") or ""),
        )
        return cls(
            leakage_rate=_float(report.get("leakage_rate")),
            reidentification_rate=_float(report.get("reid_rate")),
            minimum_k=int(report.get("k_min") or 0),
            singleton_record_count=len(report.get("singleton_records") or ()),
            quasi_identifier_count=len(report.get("quasi_identifiers") or ()),
        )


@strawberry.type
class EntityType:
    """A canonical entity label and its policy category."""

    label: str
    policy_label: str


@strawberry.type
class AnalyzePayload:
    """Typed result returned by the ``analyze`` query."""

    text: str
    entities: list[Entity]
    spans: list[OpenMedSpanType]
    model_name: str
    timestamp: str
    processing_time: Optional[float]
    metadata: JSON

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> AnalyzePayload:
        """Build a typed analysis response from the service payload."""
        entity_items = _mapping_items(result.get("entities"))
        span_items = _mapping_items(result.get("spans")) or entity_items
        return cls(
            text=str(result.get("text") or ""),
            entities=[Entity.from_mapping(item) for item in entity_items],
            spans=[OpenMedSpanType.from_mapping(item) for item in span_items],
            model_name=str(result.get("model_name") or ""),
            timestamp=str(result.get("timestamp") or ""),
            processing_time=_optional_float(result.get("processing_time")),
            metadata=_json_mapping(result.get("metadata")),
        )


@strawberry.type
class DeidentifyPayload:
    """Typed result returned by the ``deidentify`` query."""

    original_text: str
    deidentified_text: str
    entities: list[Entity]
    spans: list[OpenMedSpanType]
    method: str
    timestamp: str
    num_entities_redacted: int
    metadata: JSON
    mapping: Optional[JSON]
    policy: Optional[PolicyProfileType]
    risk: RiskFacets

    @classmethod
    def from_mapping(
        cls,
        result: Mapping[str, Any],
        *,
        policy_name: Optional[str],
    ) -> DeidentifyPayload:
        """Build a typed de-identification response from the service payload."""
        entity_items = _mapping_items(result.get("pii_entities"))
        span_items = _mapping_items(result.get("spans")) or entity_items
        mapping = result.get("mapping")
        return cls(
            original_text=str(result.get("original_text") or ""),
            deidentified_text=str(result.get("deidentified_text") or ""),
            entities=[Entity.from_mapping(item) for item in entity_items],
            spans=[OpenMedSpanType.from_mapping(item) for item in span_items],
            method=str(result.get("method") or ""),
            timestamp=str(result.get("timestamp") or ""),
            num_entities_redacted=int(result.get("num_entities_redacted") or 0),
            metadata=_json_mapping(result.get("metadata")),
            mapping=dict(mapping) if isinstance(mapping, Mapping) else None,
            policy=(
                PolicyProfileType.from_profile(load_policy(policy_name))
                if policy_name is not None
                else None
            ),
            risk=RiskFacets.from_result(result),
        )


@strawberry.type
class Query:
    """Read-only OpenMed GraphQL operations."""

    @strawberry.field
    async def analyze(
        self,
        info: Info[OpenMedGraphQLContext, None],
        input: AnalyzeInput,
    ) -> AnalyzePayload:
        """Analyze text with the shared service runtime and warm pool."""
        try:
            payload = input.to_request()
            result = await _run_analyze(info.context.runtime, payload)
            return AnalyzePayload.from_mapping(result)
        except Exception:
            raise GraphQLError(
                SAFE_RESOLVER_ERROR,
                extensions={"code": "OPENMED_RESOLVER_ERROR"},
            ) from None

    @strawberry.field
    async def deidentify(
        self,
        info: Info[OpenMedGraphQLContext, None],
        input: DeidentifyInput,
    ) -> DeidentifyPayload:
        """De-identify text with the shared service runtime and warm pool."""
        try:
            payload = input.to_request()
            result = await _run_deidentify(info.context.runtime, payload)
            return DeidentifyPayload.from_mapping(result, policy_name=payload.policy)
        except Exception:
            raise GraphQLError(
                SAFE_RESOLVER_ERROR,
                extensions={"code": "OPENMED_RESOLVER_ERROR"},
            ) from None

    @strawberry.field
    async def entity_types(self) -> list[EntityType]:
        """Return the canonical entity catalog without loading a model."""
        return [
            EntityType(label=label, policy_label=policy_label_for(label))
            for label in sorted(CANONICAL_LABELS)
        ]


class PrivacySafeSchema(strawberry.Schema):
    """Schema that never sends execution exceptions to a logger."""

    def process_errors(
        self,
        errors: list[GraphQLError],
        execution_context: Any = None,
    ) -> None:
        """Suppress Strawberry's default exception logger to protect PHI."""


schema = PrivacySafeSchema(query=Query)


async def _run_analyze(
    runtime: ServiceRuntime,
    payload: AnalyzeRequest,
) -> Mapping[str, Any]:
    from .app import _analyze_payload, _run_with_timeout

    return await _run_with_timeout(
        runtime,
        lambda: runtime.run_model_request(
            payload.model_name,
            payload.keep_alive,
            lambda: _analyze_payload(payload, runtime),
        ),
    )


async def _run_deidentify(
    runtime: ServiceRuntime,
    payload: PIIDeidentifyRequest,
) -> Mapping[str, Any]:
    from .app import _pii_deidentify_payload, _run_with_timeout

    return await _run_with_timeout(
        runtime,
        lambda: runtime.run_model_request(
            payload.model_name,
            payload.keep_alive,
            lambda: _pii_deidentify_payload(payload, runtime),
        ),
    )


def _mapping_items(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _json_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)
