"""Deterministic status vocabulary helpers for SDOH cue normalization.

The helpers in this module normalize explicit status cues into compact
canonical values for downstream SDOH extractors. Outputs are advisory labels for
review and downstream processing, not clinical decisions. They deliberately do
not infer a status from absence of mention; text with no configured cue returns
``"unknown"`` unless an explicit context axis is supplied by the caller.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from .context import HISTORICAL, NEGATED

StatusValue = Literal["current", "former", "never", "unknown"]

CURRENT: StatusValue = "current"
FORMER: StatusValue = "former"
NEVER: StatusValue = "never"
UNKNOWN: StatusValue = "unknown"

STATUS_VALUES = (CURRENT, FORMER, NEVER, UNKNOWN)
STATUS_VOCAB_RESOURCE = "data/status_vocab.yaml"
STATUS_NORMALIZATION_ADVISORY = (
    "Status normalization is an advisory deterministic cue mapping for review "
    "and downstream processing; it is not a clinical decision rule and does "
    "not infer status from absence of mention."
)

_VOCABULARY_PACKAGE = "openmed.clinical"
_TRUE_VALUES = {"1", "true", "yes", "y", "negated", NEGATED}
_FALSE_VALUES = {"0", "false", "no", "n", "affirmed"}


def normalize_substance_status(
    phrase: object,
    negated: bool | str | None = None,
    temporality: str | None = None,
) -> StatusValue:
    """Normalize a substance-use status phrase to a canonical status.

    Explicit negation folds to ``"never"``. A supplied historical temporality
    folds otherwise-current cues such as ``"smoker"`` to ``"former"``.
    Unrecognized phrases return ``"unknown"``.
    """

    return _normalize_status(
        "substance",
        phrase,
        negated=negated,
        temporality=temporality,
    )


def normalize_employment_status(
    phrase: object,
    negated: bool | str | None = None,
    temporality: str | None = None,
) -> str:
    """Normalize an employment status phrase using the committed cue table."""

    return _normalize_status(
        "employment",
        phrase,
        negated=negated,
        temporality=temporality,
    )


def normalize_living_status(
    phrase: object,
    negated: bool | str | None = None,
    temporality: str | None = None,
) -> str:
    """Normalize a living-situation status phrase using the cue table."""

    return _normalize_status(
        "living",
        phrase,
        negated=negated,
        temporality=temporality,
    )


def load_status_vocab(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the committed status vocabulary.

    Args:
        path: Optional filesystem path for tests or downstream validation. When
            omitted, the packaged ``status_vocab.yaml`` resource is loaded.

    Returns:
        The parsed vocabulary payload as a plain dictionary.
    """

    if path is None:
        payload = copy.deepcopy(_load_default_status_vocab())
    else:
        payload = _parse_yaml_subset(Path(path).read_text(encoding="utf-8"))
    _validate_status_vocab(payload)
    return payload


@lru_cache(maxsize=1)
def _load_default_status_vocab() -> dict[str, Any]:
    resource = resources.files(_VOCABULARY_PACKAGE).joinpath(STATUS_VOCAB_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        payload = _parse_yaml_subset(handle.read())
    _validate_status_vocab(payload)
    return payload


def _normalize_status(
    vocabulary_name: str,
    phrase: object,
    *,
    negated: bool | str | None,
    temporality: str | None,
) -> str:
    payload = _load_default_status_vocab()
    vocabulary = payload["vocabularies"][vocabulary_name]
    unknown = str(payload.get("defaults", {}).get("unknown_status") or UNKNOWN)
    overrides = vocabulary.get("axis_overrides") or {}
    text = _normalize_phrase(phrase)

    if _is_negated(negated):
        return str(overrides.get("negated") or NEVER)

    status = _match_vocab_status(text, vocabulary, unknown)
    if _is_historical(temporality) and status in vocabulary.get("current_statuses", ()):
        return str(overrides.get("historical_current") or FORMER)
    return status


def _match_vocab_status(text: str, vocabulary: Mapping[str, Any], unknown: str) -> str:
    if not text:
        return unknown

    statuses = vocabulary["statuses"]
    for status in vocabulary["priority"]:
        entry = statuses[status]
        if any(_cue_matches(text, cue) for cue in entry["cues"]):
            return str(status)
    return unknown


def _is_negated(value: bool | str | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    normalized = str(value).strip().casefold()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def _is_historical(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().casefold() == HISTORICAL


def _normalize_phrase(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _cue_matches(text: str, cue: object) -> bool:
    normalized_cue = _normalize_phrase(cue)
    if not normalized_cue:
        return False
    return _cue_pattern(normalized_cue).search(text) is not None


@lru_cache(maxsize=512)
def _cue_pattern(cue: str) -> re.Pattern[str]:
    escaped = re.escape(cue).replace(r"\ ", r"\s+")
    prefix = r"(?<!\w)"
    suffix = "" if cue.endswith("-") else r"(?!\w)"
    return re.compile(prefix + escaped + suffix)


def _validate_status_vocab(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or schema_version < 1:
        raise ValueError("status vocabulary requires positive integer schema_version")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance.get("source"):
        raise ValueError("status vocabulary requires provenance.source")
    disclaimer = provenance.get("disclaimer")
    if not isinstance(disclaimer, str) or "clinical decision" not in disclaimer:
        raise ValueError("status vocabulary requires an advisory disclaimer")

    defaults = payload.get("defaults")
    if not isinstance(defaults, Mapping) or defaults.get("unknown_status") != UNKNOWN:
        raise ValueError("status vocabulary requires defaults.unknown_status")

    vocabularies = payload.get("vocabularies")
    if not isinstance(vocabularies, Mapping) or not vocabularies:
        raise ValueError("status vocabulary requires vocabularies")
    for vocabulary_name, vocabulary in vocabularies.items():
        _validate_vocabulary(str(vocabulary_name), vocabulary)


def _validate_vocabulary(vocabulary_name: str, vocabulary: object) -> None:
    if not isinstance(vocabulary, Mapping):
        raise ValueError(f"{vocabulary_name} vocabulary must be a mapping")

    priority = vocabulary.get("priority")
    statuses = vocabulary.get("statuses")
    if not _is_str_sequence(priority):
        raise ValueError(f"{vocabulary_name} vocabulary requires priority")
    if not isinstance(statuses, Mapping) or not statuses:
        raise ValueError(f"{vocabulary_name} vocabulary requires statuses")

    for status in priority:
        entry = statuses.get(status)
        if not isinstance(entry, Mapping):
            raise ValueError(f"{vocabulary_name}.{status} must be a mapping")
        cues = entry.get("cues")
        if not _is_str_sequence(cues):
            raise ValueError(f"{vocabulary_name}.{status} requires string cues")

    missing = set(statuses) - set(priority)
    if missing:
        missing_list = ", ".join(sorted(str(item) for item in missing))
        raise ValueError(f"{vocabulary_name} priority omits statuses: {missing_list}")

    current_statuses = vocabulary.get("current_statuses")
    if not _is_str_sequence(current_statuses):
        raise ValueError(f"{vocabulary_name} vocabulary requires current_statuses")
    unknown_current = set(current_statuses) - set(statuses)
    if unknown_current:
        unknown_list = ", ".join(sorted(str(item) for item in unknown_current))
        raise ValueError(
            f"{vocabulary_name} current_statuses references unknown statuses: "
            f"{unknown_list}"
        )

    overrides = vocabulary.get("axis_overrides")
    if not isinstance(overrides, Mapping):
        raise ValueError(f"{vocabulary_name} vocabulary requires axis_overrides")
    for key in ("negated", "historical_current"):
        if not isinstance(overrides.get(key), str) or not overrides[key]:
            raise ValueError(f"{vocabulary_name} axis_overrides.{key} is required")


def _is_str_sequence(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and all(isinstance(item, str) and item for item in value)
    )


def _parse_yaml_subset(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"invalid indentation at line {line_number}")
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"expected key/value pair at line {line_number}")
        key = key.strip()
        if not key:
            raise ValueError(f"empty key at line {line_number}")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"invalid indentation at line {line_number}")
        parent = stack[-1][1]

        scalar = value.strip()
        if not scalar:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(scalar)

    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


__all__ = [
    "CURRENT",
    "FORMER",
    "NEVER",
    "UNKNOWN",
    "STATUS_NORMALIZATION_ADVISORY",
    "STATUS_VALUES",
    "STATUS_VOCAB_RESOURCE",
    "StatusValue",
    "load_status_vocab",
    "normalize_employment_status",
    "normalize_living_status",
    "normalize_substance_status",
]
