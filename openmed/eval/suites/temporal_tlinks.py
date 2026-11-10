"""Synthetic temporal TLINK gold loading and merge-gate evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

from openmed.clinical.relations.temporal import (
    TEMPORAL_RELATION_TYPES,
    TemporalCueReference,
    TemporalRelationCandidate,
    TemporalRelationType,
    TemporalSpanReference,
    TemporalSpanRole,
    decode_tlink_candidates,
)
from openmed.core.audit import hash_text
from openmed.core.decoding import SpanGraph
from openmed.eval.metrics import (
    TemporalConsistencyGateResult,
    compute_temporal_awareness_f1,
    compute_temporal_closure_consistency,
    evaluate_temporal_consistency_gate,
    normalize_temporal_edges,
)

TEMPORAL_TLINK_FIXTURE_SCHEMA_VERSION = 1
TEMPORAL_TLINK_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "temporal_tlinks.jsonl"
)


@dataclass(frozen=True)
class TemporalFixtureSpan:
    """One EVENT or TIMEX span in a synthetic temporal fixture."""

    span_id: str
    label: str
    role: TemporalSpanRole
    start: int
    end: int
    normalized_value: str = ""
    is_dct: bool = False


@dataclass(frozen=True)
class TemporalFixtureCandidate:
    """One scored TLINK candidate supplied to the temporal graph decoder."""

    candidate_id: str
    relation_type: TemporalRelationType
    source_id: str
    target_id: str
    confidence: float
    cue_start: int
    cue_end: int


@dataclass(frozen=True)
class TemporalTLinkFixture:
    """Validated synthetic discharge-summary TLINK gold and decoder inputs."""

    fixture_id: str
    language: str
    text: str
    spans: tuple[TemporalFixtureSpan, ...]
    gold_tlinks: tuple[tuple[str, str, str], ...]
    reduced_graph: tuple[tuple[str, str, str], ...]
    candidates: tuple[TemporalFixtureCandidate, ...]
    contradictory_candidate_traps: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class TemporalTLinkEvaluationResult:
    """Aggregate PHI-safe outcome for the committed temporal merge gate."""

    passed: bool
    fixture_count: int
    gold_tlink_count: int
    predicted_reduced_relation_count: int
    contradictory_trap_count: int
    pruned_contradictory_trap_count: int
    gate: TemporalConsistencyGateResult
    failure_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return counts and gate metrics without note text or node ids."""

        return {
            "contradictory_trap_count": self.contradictory_trap_count,
            "failure_reasons": list(self.failure_reasons),
            "fixture_count": self.fixture_count,
            "gate": self.gate.to_dict(),
            "gold_tlink_count": self.gold_tlink_count,
            "passed": self.passed,
            "predicted_reduced_relation_count": (self.predicted_reduced_relation_count),
            "pruned_contradictory_trap_count": (self.pruned_contradictory_trap_count),
        }


def load_temporal_tlink_fixtures(
    path: str | Path | None = None,
) -> tuple[TemporalTLinkFixture, ...]:
    """Load and validate synthetic discharge-summary temporal gold.

    Args:
        path: Optional JSONL fixture path; defaults to the bundled gold.

    Returns:
        Validated temporal fixtures in file order.

    Raises:
        ValueError: If the fixture schema, safety metadata, or graph is invalid.
    """

    fixture_path = Path(path) if path is not None else TEMPORAL_TLINK_FIXTURE_PATH
    rows = [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixtures = tuple(_temporal_fixture_from_mapping(row) for row in rows)
    if not fixtures:
        raise ValueError("temporal TLINK fixture file must contain at least one row")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("temporal TLINK fixture ids must be unique")
    return fixtures


def decode_temporal_tlink_fixture(fixture: TemporalTLinkFixture) -> SpanGraph:
    """Decode one fixture's scored candidates into a reduced temporal graph.

    Args:
        fixture: Validated synthetic temporal fixture.

    Returns:
        The deterministic reduced graph with kept/pruned decision provenance.
    """

    spans_by_id = {span.span_id: span for span in fixture.spans}
    candidates: list[TemporalRelationCandidate] = []
    for candidate in fixture.candidates:
        source = spans_by_id[candidate.source_id]
        target = spans_by_id[candidate.target_id]
        candidates.append(
            TemporalRelationCandidate(
                relation_type=candidate.relation_type,
                source=_runtime_span(fixture.text, source),
                target=_runtime_span(fixture.text, target),
                confidence=candidate.confidence,
                cue=TemporalCueReference(
                    category=candidate.relation_type,
                    start=candidate.cue_start,
                    end=candidate.cue_end,
                    text_hash=hash_text(
                        fixture.text[candidate.cue_start : candidate.cue_end]
                    ),
                ),
                provenance={
                    "candidate_id": candidate.candidate_id,
                    "fixture_id": fixture.fixture_id,
                    "synthetic": True,
                },
            )
        )
    return decode_tlink_candidates(candidates)


def evaluate_temporal_tlink_fixtures(
    fixtures: tuple[TemporalTLinkFixture, ...] | None = None,
) -> TemporalTLinkEvaluationResult:
    """Decode committed candidates and evaluate the blocking temporal gate.

    Args:
        fixtures: Optional validated fixtures; defaults to the bundled gold.

    Returns:
        Aggregate awareness, consistency, and contradiction-trap evidence.
    """

    resolved = fixtures if fixtures is not None else load_temporal_tlink_fixtures()
    gold_edges: list[tuple[str, str, str]] = []
    predicted_edges: list[tuple[str, str, str]] = []
    trap_count = 0
    pruned_trap_count = 0

    for fixture in resolved:
        graph = decode_temporal_tlink_fixture(fixture)
        gold_edges.extend(_fixture_scoped_edges(fixture, fixture.reduced_graph))
        predicted_edges.extend(_fixture_scoped_edges(fixture, graph.edge_keys()))
        trap_ids = set(fixture.contradictory_candidate_traps)
        trap_count += len(trap_ids)
        pruned_trap_count += sum(
            1
            for decision in graph.decisions
            if decision.status == "pruned"
            and str(decision.constraint or "").startswith("acyclicity:")
            and decision.edge.metadata.get("provenance", {}).get("candidate_id")
            in trap_ids
        )

    gate = evaluate_temporal_consistency_gate(gold_edges, predicted_edges)
    failure_reasons = list(gate.failure_reasons)
    if pruned_trap_count != trap_count:
        failure_reasons.append("contradictory_candidate_not_pruned")
    return TemporalTLinkEvaluationResult(
        passed=not failure_reasons,
        fixture_count=len(resolved),
        gold_tlink_count=sum(len(fixture.gold_tlinks) for fixture in resolved),
        predicted_reduced_relation_count=len(predicted_edges),
        contradictory_trap_count=trap_count,
        pruned_contradictory_trap_count=pruned_trap_count,
        gate=gate,
        failure_reasons=tuple(failure_reasons),
    )


def assert_temporal_tlink_gate(
    fixtures: tuple[TemporalTLinkFixture, ...] | None = None,
) -> TemporalTLinkEvaluationResult:
    """Return the temporal result or raise a PHI-safe blocking assertion.

    Args:
        fixtures: Optional validated fixtures; defaults to the bundled gold.

    Returns:
        The passing aggregate temporal evaluation result.

    Raises:
        AssertionError: If quality, closure consistency, or a trap fails.
    """

    result = evaluate_temporal_tlink_fixtures(fixtures)
    if not result.passed:
        reasons = ", ".join(result.failure_reasons)
        raise AssertionError(f"merge-blocking temporal TLINK gate failed: {reasons}")
    return result


def _temporal_fixture_from_mapping(data: Mapping[str, Any]) -> TemporalTLinkFixture:
    if not isinstance(data, Mapping):
        raise ValueError("temporal TLINK fixture rows must be objects")
    if data.get("schema_version") != TEMPORAL_TLINK_FIXTURE_SCHEMA_VERSION:
        raise ValueError("temporal TLINK fixture schema_version must be 1")
    metadata = data.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("synthetic") is not True:
        raise ValueError("temporal TLINK fixtures must be explicitly synthetic")
    if metadata.get("contains_real_phi") is not False:
        raise ValueError("temporal TLINK fixtures must declare contains_real_phi=false")
    fixture_id = str(data.get("id") or "").strip()
    text = str(data.get("text") or "")
    if not fixture_id or not text:
        raise ValueError("temporal TLINK fixture id and text are required")

    raw_spans = data.get("spans")
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("temporal TLINK fixtures require spans")
    spans = tuple(_fixture_span(item, text) for item in raw_spans)
    spans_by_id = {span.span_id: span for span in spans}
    if len(spans_by_id) != len(spans):
        raise ValueError("temporal TLINK span ids must be unique")
    dct_ids = {span.span_id for span in spans if span.is_dct}
    if len(dct_ids) != 1 or any(
        spans_by_id[span_id].role != "TIMEX" for span_id in dct_ids
    ):
        raise ValueError("temporal TLINK fixtures require exactly one TIMEX DCT span")

    gold_tlinks = _fixture_edges(data.get("tlinks"), spans_by_id, "tlinks")
    reduced_graph = _fixture_edges(
        data.get("reduced_graph"), spans_by_id, "reduced_graph"
    )
    if compute_temporal_closure_consistency(reduced_graph).violations:
        raise ValueError("temporal TLINK reduced gold graph must be consistent")
    if compute_temporal_awareness_f1(reduced_graph, gold_tlinks).f1 != 1.0:
        raise ValueError("temporal TLINK strict gold and reduced graph disagree")

    raw_candidates = data.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("temporal TLINK fixtures require candidates")
    candidates = tuple(
        _fixture_candidate(item, text, spans_by_id) for item in raw_candidates
    )
    candidate_ids = {candidate.candidate_id for candidate in candidates}
    if len(candidate_ids) != len(candidates):
        raise ValueError("temporal TLINK candidate ids must be unique")
    traps_value = data.get("contradictory_candidate_traps")
    if not isinstance(traps_value, list) or not traps_value:
        raise ValueError("temporal TLINK fixtures require contradiction traps")
    traps = tuple(str(value) for value in traps_value)
    if len(traps) != len(set(traps)) or any(not value for value in traps):
        raise ValueError("temporal TLINK contradiction trap ids must be unique")
    if not set(traps) <= candidate_ids:
        raise ValueError(
            "temporal TLINK contradiction traps reference unknown candidates"
        )
    _validate_required_temporal_coverage(
        gold_tlinks,
        spans_by_id=spans_by_id,
        dct_ids=dct_ids,
    )
    return TemporalTLinkFixture(
        fixture_id=fixture_id,
        language=str(data.get("language") or "en"),
        text=text,
        spans=spans,
        gold_tlinks=gold_tlinks,
        reduced_graph=reduced_graph,
        candidates=candidates,
        contradictory_candidate_traps=traps,
        metadata=dict(metadata),
    )


def _fixture_span(data: Mapping[str, Any], text: str) -> TemporalFixtureSpan:
    if not isinstance(data, Mapping):
        raise ValueError("temporal TLINK spans must be objects")
    span_id = str(data.get("id") or "").strip()
    label = str(data.get("label") or "").strip().upper()
    role = str(data.get("role") or "").strip().upper()
    start = _fixture_offset(data.get("start"), "span start")
    end = _fixture_offset(data.get("end"), "span end")
    if not span_id or not label or role not in {"EVENT", "TIMEX"}:
        raise ValueError("temporal TLINK spans require id, label, and EVENT/TIMEX role")
    if not 0 <= start < end <= len(text):
        raise ValueError("temporal TLINK span offsets are outside fixture text")
    normalized_value = str(data.get("normalized_value") or "")
    if role == "TIMEX" and not normalized_value:
        raise ValueError("temporal TLINK TIMEX spans require normalized_value")
    return TemporalFixtureSpan(
        span_id=span_id,
        label=label,
        role=cast(TemporalSpanRole, role),
        start=start,
        end=end,
        normalized_value=normalized_value,
        is_dct=data.get("is_dct") is True,
    )


def _fixture_candidate(
    data: Mapping[str, Any],
    text: str,
    spans_by_id: Mapping[str, TemporalFixtureSpan],
) -> TemporalFixtureCandidate:
    if not isinstance(data, Mapping):
        raise ValueError("temporal TLINK candidates must be objects")
    candidate_id = str(data.get("id") or "").strip()
    relation_type = str(data.get("type") or "").strip().upper()
    source_id = str(data.get("source") or "").strip()
    target_id = str(data.get("target") or "").strip()
    confidence_value = data.get("confidence")
    if isinstance(confidence_value, bool) or not isinstance(
        confidence_value, int | float
    ):
        raise ValueError("temporal TLINK candidate confidence must be numeric")
    confidence = float(confidence_value)
    cue_start = _fixture_offset(data.get("cue_start"), "candidate cue_start")
    cue_end = _fixture_offset(data.get("cue_end"), "candidate cue_end")
    if not candidate_id or relation_type not in TEMPORAL_RELATION_TYPES:
        raise ValueError("temporal TLINK candidates require id and supported type")
    if source_id not in spans_by_id or target_id not in spans_by_id:
        raise ValueError("temporal TLINK candidate references an unknown span")
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("temporal TLINK candidate confidence must be between 0 and 1")
    if not 0 <= cue_start < cue_end <= len(text):
        raise ValueError("temporal TLINK candidate cue offsets are invalid")
    return TemporalFixtureCandidate(
        candidate_id=candidate_id,
        relation_type=cast(TemporalRelationType, relation_type),
        source_id=source_id,
        target_id=target_id,
        confidence=confidence,
        cue_start=cue_start,
        cue_end=cue_end,
    )


def _fixture_edges(
    value: Any,
    spans_by_id: Mapping[str, TemporalFixtureSpan],
    field_name: str,
) -> tuple[tuple[str, str, str], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"temporal TLINK fixtures require {field_name}")
    edges = normalize_temporal_edges(value)
    if len(edges) != len(value) or any(head == tail for _, head, tail in edges):
        raise ValueError(f"temporal TLINK {field_name} contains duplicate/self edges")
    if any(
        head not in spans_by_id or tail not in spans_by_id for _, head, tail in edges
    ):
        raise ValueError(f"temporal TLINK {field_name} references an unknown span")
    return edges


def _validate_required_temporal_coverage(
    edges: tuple[tuple[str, str, str], ...],
    *,
    spans_by_id: Mapping[str, TemporalFixtureSpan],
    dct_ids: set[str],
) -> None:
    role_pairs = [
        {spans_by_id[head].role, spans_by_id[tail].role} for _, head, tail in edges
    ]
    if not any(head in dct_ids or tail in dct_ids for _, head, tail in edges):
        raise ValueError("temporal TLINK gold must cover DCT anchoring")
    if {"EVENT", "TIMEX"} not in role_pairs:
        raise ValueError("temporal TLINK gold must cover EVENT-TIMEX links")
    if {"EVENT"} not in role_pairs:
        raise ValueError("temporal TLINK gold must cover EVENT-EVENT ordering")


def _runtime_span(text: str, span: TemporalFixtureSpan) -> TemporalSpanReference:
    return TemporalSpanReference(
        span_id=span.span_id,
        label=span.label,
        role=span.role,
        start=span.start,
        end=span.end,
        score=1.0,
        text_hash=hash_text(text[span.start : span.end]),
    )


def _fixture_scoped_edges(
    fixture: TemporalTLinkFixture,
    edges: Any,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (label, f"{fixture.fixture_id}:{head}", f"{fixture.fixture_id}:{tail}")
        for label, head, tail in normalize_temporal_edges(edges)
    )


def _fixture_offset(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"temporal TLINK {field_name} must be an integer")
    return value


__all__ = [
    "TEMPORAL_TLINK_FIXTURE_PATH",
    "TEMPORAL_TLINK_FIXTURE_SCHEMA_VERSION",
    "TemporalFixtureCandidate",
    "TemporalFixtureSpan",
    "TemporalTLinkEvaluationResult",
    "TemporalTLinkFixture",
    "assert_temporal_tlink_gate",
    "decode_temporal_tlink_fixture",
    "evaluate_temporal_tlink_fixtures",
    "load_temporal_tlink_fixtures",
]
