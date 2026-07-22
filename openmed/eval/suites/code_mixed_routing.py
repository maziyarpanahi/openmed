"""Synthetic evaluation gate for code-mixed Hinglish de-identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.custom_recognizer import build_transliterated_name_recognizer
from openmed.core.labels import normalize_label
from openmed.core.pii_entity_merger import merge_entities_with_semantic_units
from openmed.core.pii_i18n import (
    CodeMixedTokenTag,
    code_mixed_route_active,
    get_patterns_for_code_mixed_tags,
    normalize_code_mixed_token_tags,
)
from openmed.core.quality_gates import resolve_overlapping_entities
from openmed.core.safety_sweep import hashed_span_surface, safety_sweep_code_mixed
from openmed.processing.outputs import EntityPrediction

CODE_MIXED_ROUTING = "code_mixed_routing"
CODE_MIXED_PHI_RECALL_GATE = 1.0
CODE_MIXED_ENTITY_LEAKAGE_GATE = 0
CODE_MIXED_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "golden"
    / "fixtures"
    / "code_mixed_deidentification.jsonl"
)


@dataclass(frozen=True)
class CodeMixedGoldSpan:
    """One synthetic gold PHI span."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class CodeMixedFixture:
    """One synthetic Hinglish note and its offset-only routing annotations."""

    fixture_id: str
    text: str
    token_language_tags: tuple[CodeMixedTokenTag, ...]
    gold_spans: tuple[CodeMixedGoldSpan, ...]
    synthetic: bool = True


@dataclass(frozen=True)
class CodeMixedRoutingReport:
    """Aggregate-only code-mixed recall and source-entity leakage report."""

    fixture_count: int
    gold_span_count: int
    matched_span_count: int
    recall: float
    entity_leakage_count: int
    english_false_positive_count: int
    recall_gate: float = CODE_MIXED_PHI_RECALL_GATE
    leakage_gate: int = CODE_MIXED_ENTITY_LEAKAGE_GATE
    failures: tuple[Mapping[str, Any], ...] = ()

    @property
    def passed(self) -> bool:
        """Return whether all documented code-mixed gates pass."""
        return (
            self.recall >= self.recall_gate
            and self.entity_leakage_count <= self.leakage_gate
            and self.english_false_positive_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic aggregate metrics without raw note text."""
        return {
            "english_false_positive_count": self.english_false_positive_count,
            "entity_leakage_count": self.entity_leakage_count,
            "failures": [dict(failure) for failure in self.failures],
            "fixture_count": self.fixture_count,
            "gates": {
                "entity_leakage_max": self.leakage_gate,
                "phi_recall_min": self.recall_gate,
            },
            "gold_span_count": self.gold_span_count,
            "matched_span_count": self.matched_span_count,
            "passed": self.passed,
            "phi_recall": self.recall,
            "suite": CODE_MIXED_ROUTING,
            "synthetic": True,
        }


def load_code_mixed_fixtures(
    path: str | Path | None = None,
) -> list[CodeMixedFixture]:
    """Load and validate the committed synthetic Hinglish fixture set."""
    fixture_path = Path(path) if path is not None else CODE_MIXED_FIXTURE_PATH
    fixtures: list[CodeMixedFixture] = []
    seen_ids: set[str] = set()
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{fixture_path}:{line_number} must be an object")
            fixture = _fixture_from_mapping(payload)
            if fixture.fixture_id in seen_ids:
                raise ValueError(f"duplicate fixture id: {fixture.fixture_id}")
            seen_ids.add(fixture.fixture_id)
            fixtures.append(fixture)
    if not fixtures:
        raise ValueError(f"{fixture_path} does not contain code-mixed fixtures")
    return fixtures


def evaluate_code_mixed_routing(
    fixtures: Sequence[CodeMixedFixture] | None = None,
) -> CodeMixedRoutingReport:
    """Evaluate deterministic Hinglish PHI recall and post-redaction leakage."""
    fixture_rows = (
        list(fixtures) if fixtures is not None else load_code_mixed_fixtures()
    )
    matched = 0
    gold_count = 0
    leakage_count = 0
    english_false_positive_count = 0
    failures: list[Mapping[str, Any]] = []

    for fixture in fixture_rows:
        entities = detect_code_mixed_fixture_entities(fixture)
        redacted = _redact_with_masks(fixture.text, entities)
        active = code_mixed_route_active(
            fixture.text,
            fixture.token_language_tags,
        )
        if not active:
            english_false_positive_count += len(entities)

        for gold in fixture.gold_spans:
            gold_count += 1
            matched_entity = next(
                (
                    entity
                    for entity in entities
                    if int(entity.start or 0) == gold.start
                    and int(entity.end or 0) == gold.end
                    and normalize_label(entity.label, "en")
                    == normalize_label(gold.label, "en")
                ),
                None,
            )
            if matched_entity is not None:
                matched += 1
            else:
                failures.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "reason": "gold_span_missed",
                        **hashed_span_surface(
                            fixture.text,
                            gold.start,
                            gold.end,
                            label=gold.label,
                        ),
                    }
                )

            surface = fixture.text[gold.start : gold.end]
            if surface and surface.casefold() in redacted.casefold():
                leakage_count += 1
                failures.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "reason": "source_entity_leaked",
                        **hashed_span_surface(
                            fixture.text,
                            gold.start,
                            gold.end,
                            label=gold.label,
                        ),
                    }
                )

    recall = matched / gold_count if gold_count else 1.0
    return CodeMixedRoutingReport(
        fixture_count=len(fixture_rows),
        gold_span_count=gold_count,
        matched_span_count=matched,
        recall=recall,
        entity_leakage_count=leakage_count,
        english_false_positive_count=english_false_positive_count,
        failures=tuple(failures),
    )


def detect_code_mixed_fixture_entities(
    fixture: CodeMixedFixture,
) -> list[EntityPrediction]:
    """Run the model-free rule/name bridge used by the synthetic eval gate."""
    patterns = get_patterns_for_code_mixed_tags(
        fixture.text,
        fixture.token_language_tags,
        base_lang="en",
    )
    merged = merge_entities_with_semantic_units(
        [],
        fixture.text,
        patterns=patterns,
        use_semantic_patterns=True,
        allow_semantic_only_matches=True,
    )
    entities = [
        EntityPrediction(
            text=fixture.text[item["start"] : item["end"]],
            label=str(item["entity_type"]),
            start=int(item["start"]),
            end=int(item["end"]),
            confidence=float(item["score"]),
        )
        for item in merged
    ]
    if code_mixed_route_active(fixture.text, fixture.token_language_tags):
        entities.extend(
            build_transliterated_name_recognizer().detect_entities(fixture.text)
        )

    swept = safety_sweep_code_mixed(
        fixture.text,
        resolve_overlapping_entities(entities),
        token_language_tags=fixture.token_language_tags,
        base_lang="en",
    )
    return list(resolve_overlapping_entities(swept))


def _fixture_from_mapping(payload: Mapping[str, Any]) -> CodeMixedFixture:
    fixture_id = str(payload.get("id") or "")
    text = str(payload.get("text") or "")
    metadata = payload.get("metadata") or {}
    if not fixture_id or not text:
        raise ValueError("code-mixed fixtures require id and text")
    if not isinstance(metadata, Mapping) or metadata.get("synthetic") is not True:
        raise ValueError("code-mixed fixtures must set metadata.synthetic=true")
    raw_tags = payload.get("token_language_tags")
    if not isinstance(raw_tags, list):
        raise ValueError("code-mixed fixtures require token_language_tags")
    tags = normalize_code_mixed_token_tags(text, raw_tags)

    raw_spans = payload.get("gold_spans") or []
    if not isinstance(raw_spans, list):
        raise ValueError("gold_spans must be a list")
    spans = []
    for item in raw_spans:
        if not isinstance(item, Mapping):
            raise ValueError("gold spans must be objects")
        span = CodeMixedGoldSpan(
            start=int(item["start"]),
            end=int(item["end"]),
            label=str(item["label"]),
        )
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError("gold span offsets are invalid")
        spans.append(span)
    return CodeMixedFixture(
        fixture_id=fixture_id,
        text=text,
        token_language_tags=tags,
        gold_spans=tuple(spans),
    )


def _redact_with_masks(text: str, entities: Sequence[EntityPrediction]) -> str:
    redacted = text
    for entity in sorted(entities, key=lambda item: int(item.start or 0), reverse=True):
        start = int(entity.start or 0)
        end = int(entity.end or start)
        redacted = redacted[:start] + f"[{entity.label}]" + redacted[end:]
    return redacted


__all__ = [
    "CODE_MIXED_ENTITY_LEAKAGE_GATE",
    "CODE_MIXED_FIXTURE_PATH",
    "CODE_MIXED_PHI_RECALL_GATE",
    "CODE_MIXED_ROUTING",
    "CodeMixedFixture",
    "CodeMixedGoldSpan",
    "CodeMixedRoutingReport",
    "detect_code_mixed_fixture_entities",
    "evaluate_code_mixed_routing",
    "load_code_mixed_fixtures",
]
