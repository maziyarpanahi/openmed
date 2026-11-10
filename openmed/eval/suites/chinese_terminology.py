"""Privacy gate for user-supplied Chinese terminology grounding."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from openmed.clinical.normalization.chinese import ChineseTerminologyGrounder
from openmed.core.labels import CONDITION, MEDICATION, normalize_label
from openmed.eval.metrics import compute_extraction_reemission_leakage
from openmed.processing.outputs import EntityPrediction

_GROUNDING_KEYS = frozenset({"system", "code", "display"})
_TARGET_LABELS = frozenset({CONDITION, MEDICATION})


@dataclass(frozen=True)
class ChineseTerminologyLeakageReport:
    """Leakage and span-integrity measurements without raw source text."""

    groundable_span_count: int
    grounded_span_count: int
    leaked_phi_characters: int
    phi_character_count: int
    phi_leakage_rate: float
    offsets_unchanged: bool
    actions_unchanged: bool
    metadata_keys_valid: bool
    passed: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-compatible report containing no raw PHI."""

        return {
            "groundable_span_count": self.groundable_span_count,
            "grounded_span_count": self.grounded_span_count,
            "leaked_phi_characters": self.leaked_phi_characters,
            "phi_character_count": self.phi_character_count,
            "phi_leakage_rate": self.phi_leakage_rate,
            "offsets_unchanged": self.offsets_unchanged,
            "actions_unchanged": self.actions_unchanged,
            "metadata_keys_valid": self.metadata_keys_valid,
            "passed": self.passed,
        }


def evaluate_chinese_terminology_leakage(
    *,
    source_text: str,
    entities: Iterable[EntityPrediction],
    phi_spans: Iterable[Any],
    grounder: ChineseTerminologyGrounder,
) -> ChineseTerminologyLeakageReport:
    """Ground spans and verify zero PHI re-emission plus structural integrity.

    The leakage scan receives only the added ``system``, ``code``, and
    ``display`` values. Offsets and action fields are measured separately and
    never treated as extraction payloads.
    """

    before = list(entities)
    after = grounder.ground_entities(before)

    offsets_unchanged = all(
        (left.start, left.end) == (right.start, right.end)
        for left, right in zip(before, after)
    )
    actions_unchanged = all(
        getattr(left, "action", None) == getattr(right, "action", None)
        for left, right in zip(before, after)
    )
    metadata_keys_valid = all(
        _metadata_change_is_grounding_only(left, right)
        for left, right in zip(before, after)
    )

    grounding_payload = [
        {key: metadata[key] for key in sorted(_GROUNDING_KEYS)}
        for entity in after
        if _GROUNDING_KEYS.issubset(metadata := dict(entity.metadata or {}))
    ]
    leakage = compute_extraction_reemission_leakage(
        phi_spans,
        grounding_payload,
        source_text=source_text,
        default_language="zh",
    )
    groundable_count = sum(
        normalize_label(entity.label, lang="zh") in _TARGET_LABELS for entity in before
    )
    grounded_count = len(grounding_payload)
    passed = (
        leakage.overall == 0.0
        and offsets_unchanged
        and actions_unchanged
        and metadata_keys_valid
    )
    return ChineseTerminologyLeakageReport(
        groundable_span_count=groundable_count,
        grounded_span_count=grounded_count,
        leaked_phi_characters=leakage.leaked_chars,
        phi_character_count=leakage.total_chars,
        phi_leakage_rate=leakage.overall,
        offsets_unchanged=offsets_unchanged,
        actions_unchanged=actions_unchanged,
        metadata_keys_valid=metadata_keys_valid,
        passed=passed,
    )


def _metadata_change_is_grounding_only(
    before: EntityPrediction,
    after: EntityPrediction,
) -> bool:
    before_metadata = dict(before.metadata or {})
    after_metadata = dict(after.metadata or {})
    if any(after_metadata.get(key) != value for key, value in before_metadata.items()):
        return False
    added = set(after_metadata).difference(before_metadata)
    return added in (set(), set(_GROUNDING_KEYS))


__all__ = [
    "ChineseTerminologyLeakageReport",
    "evaluate_chinese_terminology_leakage",
]
