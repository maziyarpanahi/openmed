"""Synthetic evaluation gate for code-mixed Hinglish de-identification."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from openmed.core.custom_recognizer import build_transliterated_name_recognizer
from openmed.core.labels import normalize_label
from openmed.core.lang_id_codemix import (
    TokenLanguage,
    TokenLIDHook,
    identify_token_languages,
)
from openmed.core.pii_entity_merger import (
    PIIPattern,
    find_semantic_units,
    merge_entities_with_semantic_units,
)
from openmed.core.pii_i18n import (
    CodeMixedTokenTag,
    code_mixed_route_active,
    get_code_mixed_pattern_runs,
    get_patterns_for_code_mixed_tags,
    get_patterns_for_code_mixed_text,
    get_patterns_for_language,
    normalize_code_mixed_token_tags,
)
from openmed.core.quality_gates import resolve_overlapping_entities
from openmed.core.safety_sweep import hashed_span_surface, safety_sweep_code_mixed
from openmed.processing.outputs import EntityPrediction

CODE_MIXED_ROUTING = "code_mixed_routing"
CODE_MIXED_PHI_RECALL_GATE = 1.0
CODE_MIXED_ENTITY_LEAKAGE_GATE = 0
MIN_TOKEN_LID_ACCURACY = 0.80
CODE_MIXED_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "golden"
    / "fixtures"
    / "code_mixed_deidentification.jsonl"
)
CODE_MIXED_ROUTING_FIXTURE_PATH = (
    Path(__file__).parents[1] / "golden" / "fixtures" / "code_mixed_hinglish.jsonl"
)

_GOLD_TO_PATTERN_LABEL = {
    "DATE": "date",
    "ID_NUM": "national_id",
    "PHONE": "phone_number",
    "ZIPCODE": "postcode",
}
_RAW_TOKEN_FIELDS = frozenset({"surface", "text", "token", "value"})


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


@dataclass(frozen=True)
class GoldTokenLanguage:
    """Offset-only token language expectation."""

    start: int
    end: int
    label: str


@dataclass(frozen=True)
class CodeMixedRoutingFixture:
    """One synthetic Hinglish note with token-LID gold annotations."""

    fixture_id: str
    text: str
    gold_tokens: tuple[GoldTokenLanguage, ...]
    gold_spans: tuple[CodeMixedGoldSpan, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CodeMixedRoutingResult:
    """Aggregate token-LID and downstream routing metrics."""

    fixture_count: int
    token_count: int
    token_lid_accuracy: float
    minimum_token_lid_accuracy: float
    baseline_deid_recall: float
    code_mixed_deid_recall: float
    recall_improvement: float
    entity_leakage_count: int
    named_entity_token_count: int
    named_entity_tokens_retained: int
    deterministic: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready metrics without fixture or token surfaces."""
        return {
            "baseline_deid_recall": self.baseline_deid_recall,
            "code_mixed_deid_recall": self.code_mixed_deid_recall,
            "deterministic": self.deterministic,
            "entity_leakage_count": self.entity_leakage_count,
            "fixture_count": self.fixture_count,
            "minimum_token_lid_accuracy": self.minimum_token_lid_accuracy,
            "named_entity_token_count": self.named_entity_token_count,
            "named_entity_tokens_retained": self.named_entity_tokens_retained,
            "passed": self.passed,
            "recall_improvement": self.recall_improvement,
            "token_count": self.token_count,
            "token_lid_accuracy": self.token_lid_accuracy,
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
    fixtures: Sequence[CodeMixedFixture]
    | Sequence[CodeMixedRoutingFixture]
    | None = None,
    *,
    lid_model: TokenLIDHook | None = None,
) -> CodeMixedRoutingReport | CodeMixedRoutingResult:
    """Evaluate the explicit de-id gate or token-LID routing fixtures."""
    if lid_model is not None or (
        fixtures and isinstance(fixtures[0], CodeMixedRoutingFixture)
    ):
        routing_fixtures: Sequence[CodeMixedRoutingFixture] = (
            cast(Sequence[CodeMixedRoutingFixture], fixtures)
            if fixtures is not None
            else load_code_mixed_routing_fixtures()
        )
        return _evaluate_token_lid_routing(
            routing_fixtures,
            lid_model=lid_model,
        )

    fixture_rows: Sequence[CodeMixedFixture] = (
        cast(Sequence[CodeMixedFixture], fixtures)
        if fixtures is not None
        else load_code_mixed_fixtures()
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


def load_code_mixed_routing_fixtures(
    path: str | Path | None = None,
) -> list[CodeMixedRoutingFixture]:
    """Load and validate the token-LID-focused synthetic Hinglish fixtures."""
    fixture_path = Path(path) if path is not None else CODE_MIXED_ROUTING_FIXTURE_PATH
    fixtures: list[CodeMixedRoutingFixture] = []
    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"fixture line {line_number} must be an object")
            fixtures.append(
                _routing_fixture_from_mapping(payload, line_number=line_number)
            )
    if not fixtures:
        raise ValueError("code-mixed routing fixture set is empty")
    fixture_ids = [fixture.fixture_id for fixture in fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ValueError("code-mixed routing fixture IDs must be unique")
    return fixtures


def code_mixed_routing_metadata(
    *,
    fixture_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return the combined explicit-routing and token-LID evaluation gates."""
    return {
        "fixture_path": str(fixture_path or CODE_MIXED_ROUTING_FIXTURE_PATH),
        "gates": {
            "entity_leakage_max": CODE_MIXED_ENTITY_LEAKAGE_GATE,
            "phi_recall_min": CODE_MIXED_PHI_RECALL_GATE,
        },
        "minimum_token_lid_accuracy": MIN_TOKEN_LID_ACCURACY,
        "requires_measurable_recall_improvement": True,
        "requires_zero_entity_leakage": True,
        "suite": CODE_MIXED_ROUTING,
        "synthetic": True,
    }


def run_code_mixed_routing(
    *,
    fixture_path: str | Path | None = None,
    fixtures: Sequence[CodeMixedRoutingFixture] | None = None,
    lid_model: TokenLIDHook | None = None,
) -> CodeMixedRoutingResult:
    """Load the token-LID fixtures and run their routing gates."""
    loaded = (
        tuple(fixtures)
        if fixtures is not None
        else tuple(load_code_mixed_routing_fixtures(fixture_path))
    )
    return _evaluate_token_lid_routing(loaded, lid_model=lid_model)


def _evaluate_token_lid_routing(
    fixtures: Sequence[CodeMixedRoutingFixture],
    *,
    lid_model: TokenLIDHook | None = None,
) -> CodeMixedRoutingResult:
    token_total = 0
    token_correct = 0
    gold_span_total = 0
    baseline_hits = 0
    routed_hits = 0
    entity_leakage_count = 0
    named_entity_token_count = 0
    named_entity_tokens_retained = 0
    first_pass: list[tuple[TokenLanguage, ...]] = []
    second_pass: list[tuple[TokenLanguage, ...]] = []

    for fixture in fixtures:
        observed_tokens = identify_token_languages(fixture.text, model=lid_model)
        repeated_tokens = identify_token_languages(fixture.text, model=lid_model)
        first_pass.append(observed_tokens)
        second_pass.append(repeated_tokens)
        token_total += len(fixture.gold_tokens)
        token_correct += _correct_token_count(fixture.gold_tokens, observed_tokens)

        routes = get_code_mixed_pattern_runs(
            fixture.text,
            lid_model=lid_model,
        )
        for expected in fixture.gold_tokens:
            if expected.label != "ne":
                continue
            named_entity_token_count += 1
            if any(
                route.start <= expected.start
                and route.end >= expected.end
                and set(route.languages) == {"hi", "en"}
                and route.patterns
                for route in routes
            ):
                named_entity_tokens_retained += 1

        baseline = _observed_spans(
            fixture.text,
            get_patterns_for_language("en"),
        )
        routed = _observed_spans(
            fixture.text,
            get_patterns_for_code_mixed_text(
                fixture.text,
                lid_model=lid_model,
            ),
        )
        for gold_span in fixture.gold_spans:
            gold_span_total += 1
            expected_span = (
                gold_span.start,
                gold_span.end,
                _GOLD_TO_PATTERN_LABEL[gold_span.label],
            )
            if expected_span in baseline:
                baseline_hits += 1
            if expected_span in routed:
                routed_hits += 1
            else:
                entity_leakage_count += 1

    token_accuracy = _ratio(token_correct, token_total)
    baseline_recall = _ratio(baseline_hits, gold_span_total)
    routed_recall = _ratio(routed_hits, gold_span_total)
    recall_improvement = routed_recall - baseline_recall
    deterministic = first_pass == second_pass
    passed = (
        token_accuracy >= MIN_TOKEN_LID_ACCURACY
        and recall_improvement > 0.0
        and entity_leakage_count == 0
        and named_entity_tokens_retained == named_entity_token_count
        and deterministic
    )
    return CodeMixedRoutingResult(
        fixture_count=len(fixtures),
        token_count=token_total,
        token_lid_accuracy=token_accuracy,
        minimum_token_lid_accuracy=MIN_TOKEN_LID_ACCURACY,
        baseline_deid_recall=baseline_recall,
        code_mixed_deid_recall=routed_recall,
        recall_improvement=recall_improvement,
        entity_leakage_count=entity_leakage_count,
        named_entity_token_count=named_entity_token_count,
        named_entity_tokens_retained=named_entity_tokens_retained,
        deterministic=deterministic,
        passed=passed,
    )


def _routing_fixture_from_mapping(
    payload: Mapping[str, Any],
    *,
    line_number: int,
) -> CodeMixedRoutingFixture:
    fixture_id = str(payload.get("id", "")).strip()
    text = payload.get("text")
    metadata = payload.get("metadata")
    if not fixture_id or not isinstance(text, str) or not text:
        raise ValueError(f"fixture line {line_number} requires id and text")
    if not isinstance(metadata, Mapping) or metadata.get("synthetic") is not True:
        raise ValueError(f"fixture {fixture_id} must be marked synthetic")

    gold_tokens = tuple(
        _gold_token_from_mapping(item, fixture_id=fixture_id)
        for item in _mapping_sequence(payload.get("gold_tokens"), "gold_tokens")
    )
    gold_spans = tuple(
        _routing_gold_span_from_mapping(item, fixture_id=fixture_id)
        for item in _mapping_sequence(payload.get("gold_spans"), "gold_spans")
    )
    fixture = CodeMixedRoutingFixture(
        fixture_id=fixture_id,
        text=text,
        gold_tokens=gold_tokens,
        gold_spans=gold_spans,
        metadata=dict(metadata),
    )
    _validate_routing_fixture_offsets(fixture)
    return fixture


def _mapping_sequence(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    if not all(isinstance(item, Mapping) for item in value):
        raise ValueError(f"{field_name} entries must be objects")
    return tuple(value)


def _gold_token_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_id: str,
) -> GoldTokenLanguage:
    if _RAW_TOKEN_FIELDS & set(payload):
        raise ValueError(f"fixture {fixture_id} token gold must be offset-only")
    return GoldTokenLanguage(
        start=int(payload["start"]),
        end=int(payload["end"]),
        label=str(payload["label"]),
    )


def _routing_gold_span_from_mapping(
    payload: Mapping[str, Any],
    *,
    fixture_id: str,
) -> CodeMixedGoldSpan:
    label = str(payload["label"])
    if label not in _GOLD_TO_PATTERN_LABEL:
        raise ValueError(f"fixture {fixture_id} has unsupported gold span label")
    return CodeMixedGoldSpan(
        start=int(payload["start"]),
        end=int(payload["end"]),
        label=label,
    )


def _validate_routing_fixture_offsets(fixture: CodeMixedRoutingFixture) -> None:
    observed = identify_token_languages(fixture.text)
    expected_bounds = [(token.start, token.end) for token in fixture.gold_tokens]
    observed_bounds = [(token.start, token.end) for token in observed]
    if expected_bounds != observed_bounds:
        raise ValueError(f"fixture {fixture.fixture_id} token offsets do not align")
    for token in fixture.gold_tokens:
        if not (0 <= token.start < token.end <= len(fixture.text)):
            raise ValueError(f"fixture {fixture.fixture_id} contains invalid offsets")
    for span in fixture.gold_spans:
        if not (0 <= span.start < span.end <= len(fixture.text)):
            raise ValueError(f"fixture {fixture.fixture_id} contains invalid offsets")


def _correct_token_count(
    expected: Sequence[GoldTokenLanguage],
    observed: Sequence[TokenLanguage],
) -> int:
    observed_map = {(token.start, token.end): token.label for token in observed}
    return sum(
        observed_map.get((token.start, token.end)) == token.label for token in expected
    )


def _observed_spans(
    text: str,
    patterns: Sequence[PIIPattern],
) -> set[tuple[int, int, str]]:
    observed: set[tuple[int, int, str]] = set()
    for unit in find_semantic_units(text, list(patterns)):
        start, end, label = int(unit[0]), int(unit[1]), str(unit[2])
        validated = bool(unit[5]) if len(unit) >= 6 else True
        if validated:
            observed.add((start, end, label))
    return observed


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


__all__ = [
    "CODE_MIXED_ENTITY_LEAKAGE_GATE",
    "CODE_MIXED_FIXTURE_PATH",
    "CODE_MIXED_PHI_RECALL_GATE",
    "CODE_MIXED_ROUTING",
    "CODE_MIXED_ROUTING_FIXTURE_PATH",
    "MIN_TOKEN_LID_ACCURACY",
    "CodeMixedFixture",
    "CodeMixedGoldSpan",
    "CodeMixedRoutingReport",
    "CodeMixedRoutingFixture",
    "CodeMixedRoutingResult",
    "GoldTokenLanguage",
    "code_mixed_routing_metadata",
    "detect_code_mixed_fixture_entities",
    "evaluate_code_mixed_routing",
    "load_code_mixed_fixtures",
    "load_code_mixed_routing_fixtures",
    "run_code_mixed_routing",
]
