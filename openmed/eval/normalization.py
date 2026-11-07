"""Evaluation helpers for synthetic multilingual clinical normalization gold."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.clinical import (
    derive_abnormal_flag,
    normalize_frequency,
    parse_measurement,
    structure_vital_sign,
)

DEFAULT_MULTILINGUAL_NORM_FIXTURE = (
    Path(__file__).resolve().parent / "golden" / "fixtures" / "norm_multilingual.jsonl"
)


@dataclass(frozen=True)
class MultilingualNormScore:
    """Accuracy summary for multilingual normalization fixture records."""

    accuracy: float
    per_language: Mapping[str, float]
    correct_by_language: Mapping[str, int]
    total_by_language: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready metric payload."""

        return {
            "accuracy": self.accuracy,
            "per_language": dict(sorted(self.per_language.items())),
            "correct_by_language": dict(sorted(self.correct_by_language.items())),
            "total_by_language": dict(sorted(self.total_by_language.items())),
        }


def load_multilingual_norm_fixture(
    path: str | Path = DEFAULT_MULTILINGUAL_NORM_FIXTURE,
) -> tuple[dict[str, Any], ...]:
    """Load synthetic multilingual normalization fixture records."""

    fixture_path = Path(path)
    return tuple(
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def score_multilingual_norm_records(
    records: Iterable[Mapping[str, Any]],
) -> MultilingualNormScore:
    """Score exact normalization accuracy by source language."""

    total = 0
    correct = 0
    total_by_language: defaultdict[str, int] = defaultdict(int)
    correct_by_language: defaultdict[str, int] = defaultdict(int)

    for record in records:
        language = str(record.get("language") or "en")
        hit = _record_matches(record, language=language)
        total += 1
        total_by_language[language] += 1
        if hit:
            correct += 1
            correct_by_language[language] += 1

    per_language = {
        language: correct_by_language[language] / count
        for language, count in total_by_language.items()
    }
    accuracy = correct / total if total else 0.0
    return MultilingualNormScore(
        accuracy=accuracy,
        per_language=per_language,
        correct_by_language=dict(correct_by_language),
        total_by_language=dict(total_by_language),
    )


def score_multilingual_norm_fixture(
    path: str | Path = DEFAULT_MULTILINGUAL_NORM_FIXTURE,
) -> MultilingualNormScore:
    """Score the bundled multilingual normalization fixture."""

    return score_multilingual_norm_records(load_multilingual_norm_fixture(path))


def _record_matches(record: Mapping[str, Any], *, language: str) -> bool:
    task = str(record.get("task") or "measurement")
    expected = record.get("expected")
    if not isinstance(expected, Mapping):
        return False
    if task == "measurement":
        parsed = parse_measurement(record.get("text"), language=language)
        return (
            parsed["status"] == "ok"
            and _float_matches(
                parsed.get("canonical_magnitude"),
                expected.get("canonical_magnitude"),
            )
            and parsed.get("canonical_unit") == expected.get("canonical_unit")
        )
    if task == "frequency":
        parsed = normalize_frequency(record.get("text"), language=language)
        return (
            parsed["recognized"] is True
            and _float_matches(
                parsed.get("frequency_per_day"),
                expected.get("frequency_per_day"),
            )
            and parsed.get("as_needed") == expected.get("as_needed", False)
        )
    if task == "vital":
        parsed = structure_vital_sign(record.get("text"), language=language)
        return _plain(parsed) == _plain(expected)
    if task == "lab_flag":
        parsed = derive_abnormal_flag(
            record.get("value"),
            record.get("reference_range"),
            value_unit=record.get("value_unit"),
            reference_unit=record.get("reference_unit"),
            explicit_flag=record.get("explicit_flag"),
            language=language,
        )
        return parsed == expected.get("flag")
    return False


def _float_matches(actual: object, expected: object) -> bool:
    if actual is None or expected is None:
        return actual is expected
    try:
        actual_float = float(actual)
        expected_float = float(expected)
    except (TypeError, ValueError):
        return False
    return math.isclose(actual_float, expected_float, rel_tol=1e-9, abs_tol=1e-12)


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DEFAULT_MULTILINGUAL_NORM_FIXTURE",
    "MultilingualNormScore",
    "load_multilingual_norm_fixture",
    "score_multilingual_norm_fixture",
    "score_multilingual_norm_records",
]
