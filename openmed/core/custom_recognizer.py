"""Config-driven custom deny-list and allow-list PII recognizer."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..processing.outputs import EntityPrediction
from .labels import hipaa_class_for, normalize_label, policy_label_for
from .schemas.span import OpenMedSpan, hmac_text_hash

CUSTOM_DENY_DETECTOR = "custom:deny"
DEFAULT_CUSTOM_HASH_SECRET = b"openmed-custom-recognizer-v1"


@dataclass(frozen=True)
class CustomMatch:
    """A raw custom recognizer match with safe rule provenance."""

    start: int
    end: int
    label: str
    kind: str
    rule_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class _CompiledRule:
    kind: str
    label: str
    rule_id: str
    pattern: re.Pattern[str]
    confidence: float = 1.0

    def iter_matches(self, text: str) -> Iterable[CustomMatch]:
        for match in self.pattern.finditer(text):
            if match.start() == match.end():
                continue
            yield CustomMatch(
                start=match.start(),
                end=match.end(),
                label=self.label,
                kind=self.kind,
                rule_id=self.rule_id,
                confidence=self.confidence,
            )


class CustomRecognizer:
    """Recognize user-supplied deny-list spans and suppress allow-list spans.

    Config shape supports nested or flat lists:

    .. code-block:: python

        {
            "case_sensitive": False,
            "deny": {
                "terms": [{"term": "Ward Phoenix", "label": "LOCATION"}],
                "patterns": [{"pattern": r"\\bSTUDY-\\d+\\b", "label": "ID_NUM"}],
            },
            "allow": {
                "terms": ["Mercy Trial"],
                "patterns": [r"\\bCONTROL-\\d+\\b"],
            },
        }

    Deny entries require a label. Allow entries ignore labels because any
    overlapping detector span is suppressed.
    """

    def __init__(
        self,
        *,
        deny_terms: Sequence[Any] | Mapping[str, Any] | None = None,
        deny_patterns: Sequence[Any] | Mapping[str, Any] | None = None,
        allow_terms: Sequence[Any] | Mapping[str, Any] | None = None,
        allow_patterns: Sequence[Any] | Mapping[str, Any] | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.case_sensitive = bool(case_sensitive)
        self._deny_rules = (
            *_parse_rules(
                deny_terms,
                kind="deny_term",
                case_sensitive=self.case_sensitive,
                require_label=True,
            ),
            *_parse_rules(
                deny_patterns,
                kind="deny_pattern",
                case_sensitive=self.case_sensitive,
                require_label=True,
            ),
        )
        self._allow_rules = (
            *_parse_rules(
                allow_terms,
                kind="allow_term",
                case_sensitive=self.case_sensitive,
                require_label=False,
            ),
            *_parse_rules(
                allow_patterns,
                kind="allow_pattern",
                case_sensitive=self.case_sensitive,
                require_label=False,
            ),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | str | Path) -> "CustomRecognizer":
        """Build a recognizer from an in-memory mapping or JSON/YAML file."""

        payload = _load_config(config)
        deny_block = _mapping_or_empty(payload.get("deny"))
        allow_block = _mapping_or_empty(payload.get("allow"))
        return cls(
            deny_terms=_first_present(payload, deny_block, "deny_terms", "terms"),
            deny_patterns=_first_present(
                payload,
                deny_block,
                "deny_patterns",
                "patterns",
                "regex",
                "regexes",
            ),
            allow_terms=_first_present(payload, allow_block, "allow_terms", "terms"),
            allow_patterns=_first_present(
                payload,
                allow_block,
                "allow_patterns",
                "patterns",
                "regex",
                "regexes",
            ),
            case_sensitive=_case_sensitive(payload),
        )

    @property
    def has_allow_rules(self) -> bool:
        """Return whether this recognizer has any allow-list rules."""

        return bool(self._allow_rules)

    @property
    def has_deny_rules(self) -> bool:
        """Return whether this recognizer has any deny-list rules."""

        return bool(self._deny_rules)

    def allow_matches(self, text: str) -> tuple[CustomMatch, ...]:
        """Return allow-list matches for *text*."""

        return _dedupe_matches(
            match for rule in self._allow_rules for match in rule.iter_matches(text)
        )

    def deny_matches(self, text: str) -> tuple[CustomMatch, ...]:
        """Return deny-list matches not suppressed by allow-list rules."""

        allow_intervals = _intervals(self.allow_matches(text))
        return _dedupe_matches(
            match
            for rule in self._deny_rules
            for match in rule.iter_matches(text)
            if not _overlaps_any(match.start, match.end, allow_intervals)
        )

    def detect_entities(
        self,
        text: str,
        *,
        hmac_secret: str | bytes = DEFAULT_CUSTOM_HASH_SECRET,
    ) -> list[EntityPrediction]:
        """Return deny-list matches as ``EntityPrediction`` records."""

        return [
            EntityPrediction(
                text=text[match.start : match.end],
                label=match.label,
                start=match.start,
                end=match.end,
                confidence=match.confidence,
                metadata=_match_metadata(
                    match,
                    text=text,
                    hmac_secret=hmac_secret,
                ),
            )
            for match in self.deny_matches(text)
        ]

    def detect_spans(
        self,
        text: str,
        *,
        doc_id: str,
        hmac_secret: str | bytes,
        lang: str = "en",
    ) -> tuple[OpenMedSpan, ...]:
        """Return deny-list matches as canonical ``OpenMedSpan`` records."""

        spans: list[OpenMedSpan] = []
        for match in self.deny_matches(text):
            surface = text[match.start : match.end]
            canonical = normalize_label(match.label, lang=lang)
            spans.append(
                OpenMedSpan(
                    doc_id=doc_id,
                    start=match.start,
                    end=match.end,
                    text_hash=hmac_text_hash(surface, hmac_secret),
                    entity_type=match.label,
                    canonical_label=canonical,
                    policy_label=policy_label_for(canonical, lang=lang),
                    regulatory_tags=(hipaa_class_for(canonical, lang=lang),),
                    score=match.confidence,
                    detector=CUSTOM_DENY_DETECTOR,
                    evidence={},
                    metadata=_match_metadata(
                        match,
                        text=text,
                        hmac_secret=hmac_secret,
                    ),
                )
            )
        return tuple(spans)

    def suppress_entities(
        self,
        text: str,
        entities: Sequence[Any],
    ) -> tuple[list[Any], int]:
        """Remove entities whose spans overlap the allow-list."""

        if not self._allow_rules:
            return list(entities), 0

        allow_intervals = _intervals(self.allow_matches(text))
        retained = []
        suppressed = 0
        for entity in entities:
            bounds = _entity_bounds(entity, text)
            if bounds is not None and _overlaps_any(*bounds, allow_intervals):
                suppressed += 1
                continue
            retained.append(entity)
        return retained, suppressed

    def suppress_spans(
        self,
        text: str,
        spans: Sequence[OpenMedSpan],
    ) -> tuple[tuple[OpenMedSpan, ...], int]:
        """Remove spans whose offsets overlap the allow-list."""

        if not self._allow_rules:
            return tuple(spans), 0

        allow_intervals = _intervals(self.allow_matches(text))
        retained = []
        suppressed = 0
        for span in spans:
            if _overlaps_any(span.start, span.end, allow_intervals):
                suppressed += 1
                continue
            retained.append(span)
        return tuple(retained), suppressed

    def apply_to_prediction_result(
        self,
        result: Any,
        *,
        text: str | None = None,
        hmac_secret: str | bytes = DEFAULT_CUSTOM_HASH_SECRET,
    ) -> Any:
        """Add deny-list entities and suppress allow-listed entities in-place."""

        source_text = text if text is not None else str(getattr(result, "text", ""))
        existing = list(getattr(result, "entities", ()) or ())
        custom_entities = self.detect_entities(source_text, hmac_secret=hmac_secret)
        combined = [*existing, *custom_entities]
        retained, suppressed = self.suppress_entities(source_text, combined)
        result.entities = retained
        if hasattr(result, "num_entities"):
            result.num_entities = len(retained)

        metadata = dict(getattr(result, "metadata", None) or {})
        metadata["custom_recognizer"] = {
            "detector": CUSTOM_DENY_DETECTOR,
            "deny_spans_added": len(custom_entities),
            "allow_spans": len(self.allow_matches(source_text)),
            "spans_suppressed_by_allow": suppressed,
        }
        result.metadata = metadata
        return result


def coerce_custom_recognizer(config: Any) -> CustomRecognizer | None:
    """Return a ``CustomRecognizer`` from supported inputs."""

    if config is None:
        return None
    if isinstance(config, CustomRecognizer):
        return config
    return CustomRecognizer.from_config(config)


def _load_config(config: Mapping[str, Any] | str | Path) -> Mapping[str, Any]:
    if isinstance(config, Mapping):
        return config

    path = Path(config).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("custom recognizer config path must be JSON or YAML")

    if not isinstance(payload, Mapping):
        raise ValueError("custom recognizer config must be a mapping")
    return payload


def _case_sensitive(config: Mapping[str, Any]) -> bool:
    if "case_sensitive" in config:
        return bool(config["case_sensitive"])
    if "case_fold" in config:
        return not bool(config["case_fold"])
    if "case_folding" in config:
        return not bool(config["case_folding"])
    return False


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(
    top_level: Mapping[str, Any],
    nested: Mapping[str, Any],
    *keys: str,
) -> Any:
    for key in keys:
        if key in top_level:
            return top_level[key]
        if key in nested:
            return nested[key]
    return ()


def _parse_rules(
    values: Sequence[Any] | Mapping[str, Any] | None,
    *,
    kind: str,
    case_sensitive: bool,
    require_label: bool,
) -> tuple[_CompiledRule, ...]:
    entries = _expand_entries(values, require_label=require_label)
    rules = []
    for index, (entry, default_label) in enumerate(entries):
        raw_value, label, entry_case_sensitive, confidence, rule_id = _parse_entry(
            entry,
            default_label=default_label,
            kind=kind,
            index=index,
            require_label=require_label,
            case_sensitive=case_sensitive,
        )
        pattern = re.escape(raw_value) if kind.endswith("_term") else raw_value
        flags = 0 if entry_case_sensitive else re.IGNORECASE
        rules.append(
            _CompiledRule(
                kind=kind,
                label=label,
                rule_id=rule_id,
                pattern=re.compile(pattern, flags),
                confidence=confidence,
            )
        )
    return tuple(rules)


def _expand_entries(
    values: Sequence[Any] | Mapping[str, Any] | None,
    *,
    require_label: bool,
) -> list[tuple[Any, str | None]]:
    if values is None:
        return []
    if isinstance(values, Mapping) and not _looks_like_rule(values):
        expanded: list[tuple[Any, str | None]] = []
        for label, grouped_values in values.items():
            if isinstance(grouped_values, str) or _looks_like_rule(grouped_values):
                expanded.append((grouped_values, str(label)))
            else:
                for item in grouped_values or ():
                    expanded.append((item, str(label)))
        return expanded
    if isinstance(values, str) or _looks_like_rule(values):
        return [(values, None)]
    return [(item, None) for item in values or ()]


def _looks_like_rule(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(
        {
            "term",
            "value",
            "literal",
            "text",
            "pattern",
            "regex",
            "label",
            "entity_type",
            "target_label",
        }
        & set(value)
    )


def _parse_entry(
    entry: Any,
    *,
    default_label: str | None,
    kind: str,
    index: int,
    require_label: bool,
    case_sensitive: bool,
) -> tuple[str, str, bool, float, str]:
    value: Any = entry
    label = default_label
    entry_case_sensitive = case_sensitive
    confidence = 1.0
    rule_id = f"{kind}_{index}"

    if isinstance(entry, Mapping):
        value = (
            entry.get("term")
            or entry.get("value")
            or entry.get("literal")
            or entry.get("text")
            or entry.get("pattern")
            or entry.get("regex")
        )
        label = (
            entry.get("label")
            or entry.get("entity_type")
            or entry.get("target_label")
            or default_label
        )
        if "case_sensitive" in entry:
            entry_case_sensitive = bool(entry["case_sensitive"])
        if "confidence" in entry:
            confidence = float(entry["confidence"])
        if entry.get("id") is not None:
            rule_id = str(entry["id"])
    elif isinstance(entry, Sequence) and not isinstance(entry, str) and len(entry) >= 2:
        value, label = entry[0], str(entry[1])

    if not isinstance(value, str) or not value:
        raise ValueError(f"{kind} entries require a non-empty string")
    if require_label and not label:
        raise ValueError(f"{kind} entries require a label")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{kind} confidence must be between 0.0 and 1.0")
    return value, str(label or ""), entry_case_sensitive, confidence, rule_id


def _dedupe_matches(matches: Iterable[CustomMatch]) -> tuple[CustomMatch, ...]:
    seen: set[tuple[int, int, str, str, str]] = set()
    deduped = []
    for match in sorted(matches, key=lambda item: (item.start, item.end, item.kind)):
        key = (match.start, match.end, match.label, match.kind, match.rule_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return tuple(deduped)


def _intervals(matches: Sequence[CustomMatch]) -> tuple[tuple[int, int], ...]:
    return tuple((match.start, match.end) for match in matches)


def _overlaps_any(
    start: int,
    end: int,
    intervals: Sequence[tuple[int, int]],
) -> bool:
    return any(
        start < allow_end and end > allow_start for allow_start, allow_end in intervals
    )


def _entity_bounds(entity: Any, text: str) -> tuple[int, int] | None:
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        return start, end

    surface = str(getattr(entity, "text", "") or "")
    if not surface:
        return None
    found = text.find(surface)
    if found < 0:
        return None
    return found, found + len(surface)


def _match_metadata(
    match: CustomMatch,
    *,
    text: str,
    hmac_secret: str | bytes,
) -> dict[str, Any]:
    surface = text[match.start : match.end]
    return {
        "source": CUSTOM_DENY_DETECTOR,
        "detector": CUSTOM_DENY_DETECTOR,
        "custom_recognizer": {
            "kind": match.kind,
            "rule_id": match.rule_id,
            "text_hash": hmac_text_hash(surface, hmac_secret),
        },
    }


__all__ = [
    "CUSTOM_DENY_DETECTOR",
    "CustomMatch",
    "CustomRecognizer",
    "coerce_custom_recognizer",
]
