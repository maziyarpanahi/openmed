"""Local grounding against user-supplied Chinese clinical dictionaries.

No terminology data is bundled or downloaded by this module. Callers must
provide a local CSV, TSV, or JSONL dictionary and explicitly acknowledge that
they have the right to use its contents.
"""

from __future__ import annotations

import csv
import json
import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    CHINESE_DRUG,
    CHINESE_ICD_10,
    CONDITION,
    MEDICATION,
    normalize_label,
)
from openmed.processing.outputs import EntityPrediction

CHINESE_ICD_10_SYSTEM = CHINESE_ICD_10
CHINESE_DRUG_SYSTEM = CHINESE_DRUG

CHINESE_TERMINOLOGY_DISCLAIMER = (
    "OpenMed Chinese terminology grounding is assistive software, not a medical "
    "device, and must not be used to diagnose, prescribe, or automate care."
)
CHINESE_TERMINOLOGY_VOCABULARY_NOTE = (
    "Codes are user-vocabulary-dependent and are not for clinical decisions; "
    "verify every match against the supplied dictionary and its license."
)

_SUPPORTED_SUFFIXES = frozenset({".csv", ".jsonl", ".tsv"})
_GROUNDING_METADATA_KEYS = frozenset({"system", "code", "display"})
_TARGET_LABELS = frozenset({CONDITION, MEDICATION})
_DEFAULT_SYSTEMS = {
    CONDITION: CHINESE_ICD_10_SYSTEM,
    MEDICATION: CHINESE_DRUG_SYSTEM,
}


class ChineseTerminologyError(ValueError):
    """Base error for Chinese terminology dictionary validation."""


class ChineseTerminologyPathError(ChineseTerminologyError):
    """Raised when a caller does not provide a usable local dictionary path."""


class ChineseTerminologyLicenseError(ChineseTerminologyError):
    """Raised until the caller explicitly acknowledges dictionary licensing."""


@dataclass(frozen=True)
class ChineseTerminologyEntry:
    """One diagnosis or medication entry from a user-supplied dictionary."""

    label: str
    system: str
    code: str
    display: str
    aliases: tuple[str, ...]

    @property
    def match_terms(self) -> tuple[str, ...]:
        """Return the preferred display followed by unique aliases."""

        return _unique_text((self.display, *self.aliases))


@dataclass(frozen=True)
class ChineseTerminologyMatch:
    """A deterministic exact or fuzzy terminology match."""

    entry: ChineseTerminologyEntry
    score: float
    exact: bool

    @property
    def metadata(self) -> dict[str, str]:
        """Return the only span metadata fields added by grounding."""

        return {
            "system": self.entry.system,
            "code": self.entry.code,
            "display": self.entry.display,
        }


class ChineseTerminologyDictionary:
    """An in-memory, local-only index of Chinese clinical terminology."""

    def __init__(self, entries: Sequence[ChineseTerminologyEntry]) -> None:
        if not entries:
            raise ChineseTerminologyError(
                "The Chinese terminology dictionary is empty; provide at least one "
                "CONDITION or MEDICATION row."
            )
        self.entries = tuple(entries)
        self._exact: dict[tuple[str, str], ChineseTerminologyEntry] = {}
        self._terms: dict[str, list[tuple[str, ChineseTerminologyEntry]]] = {
            CONDITION: [],
            MEDICATION: [],
        }
        self._build_indexes()

    def _build_indexes(self) -> None:
        seen_codes: dict[tuple[str, str], ChineseTerminologyEntry] = {}
        for entry in self.entries:
            code_key = (entry.label, entry.code.casefold())
            existing_code = seen_codes.get(code_key)
            if existing_code is not None and existing_code != entry:
                raise ChineseTerminologyError(
                    f"Duplicate code {entry.code!r} for {entry.label} has "
                    "conflicting dictionary rows."
                )
            seen_codes[code_key] = entry

            normalized_terms: set[str] = set()
            for term in entry.match_terms:
                normalized = normalize_chinese_clinical_surface(term)
                if not normalized or normalized in normalized_terms:
                    continue
                normalized_terms.add(normalized)
                alias_key = (entry.label, normalized)
                existing_alias = self._exact.get(alias_key)
                if existing_alias is not None and existing_alias != entry:
                    raise ChineseTerminologyError(
                        f"Ambiguous {entry.label} term {term!r} maps to both "
                        f"{existing_alias.code!r} and {entry.code!r}."
                    )
                self._exact[alias_key] = entry
                self._terms[entry.label].append((normalized, entry))

    def match(
        self,
        surface: str,
        label: str,
        *,
        fuzzy: bool = True,
        min_score: float = 0.8,
    ) -> ChineseTerminologyMatch | None:
        """Return the best label-aware exact or fuzzy local match.

        Args:
            surface: CONDITION or MEDICATION surface text from the source note.
            label: Entity label; only CONDITION and MEDICATION are eligible.
            fuzzy: Whether to run deterministic ``SequenceMatcher`` fallback.
            min_score: Inclusive fuzzy similarity threshold between 0 and 1.
        """

        canonical_label = normalize_label(label, lang="zh")
        if canonical_label not in _TARGET_LABELS:
            return None
        threshold = _validate_min_score(min_score)
        normalized = normalize_chinese_clinical_surface(surface)
        if not normalized:
            return None

        exact_entry = self._exact.get((canonical_label, normalized))
        if exact_entry is not None:
            return ChineseTerminologyMatch(exact_entry, 1.0, True)
        if not fuzzy or len(normalized) < 3:
            return None

        best: tuple[float, str, str, ChineseTerminologyEntry] | None = None
        for term, entry in self._terms[canonical_label]:
            if len(term) < 3:
                continue
            score = SequenceMatcher(None, normalized, term, autojunk=False).ratio()
            candidate = (score, entry.code.casefold(), term, entry)
            if best is None or _match_sort_key(candidate) < _match_sort_key(best):
                best = candidate

        if best is None or best[0] < threshold:
            return None
        return ChineseTerminologyMatch(best[3], best[0], False)


class ChineseTerminologyGrounder:
    """Ground Chinese clinical spans without changing PHI behavior or offsets."""

    def __init__(
        self,
        dictionary: ChineseTerminologyDictionary,
        *,
        fuzzy: bool = True,
        min_score: float = 0.8,
    ) -> None:
        self.dictionary = dictionary
        self.fuzzy = bool(fuzzy)
        self.min_score = _validate_min_score(min_score)

    @classmethod
    def from_path(
        cls,
        dictionary_path: str | Path | None,
        *,
        license_acknowledged: bool = False,
        fuzzy: bool = True,
        min_score: float = 0.8,
    ) -> ChineseTerminologyGrounder:
        """Load a local user dictionary behind the required license gate."""

        dictionary = load_chinese_terminology_dictionary(
            dictionary_path,
            license_acknowledged=license_acknowledged,
        )
        return cls(dictionary, fuzzy=fuzzy, min_score=min_score)

    @property
    def result_metadata(self) -> dict[str, Any]:
        """Return result-level provenance and the required safety notices."""

        return {
            "local_only": True,
            "dictionary_entry_count": len(self.dictionary.entries),
            "medical_device_disclaimer": CHINESE_TERMINOLOGY_DISCLAIMER,
            "vocabulary_note": CHINESE_TERMINOLOGY_VOCABULARY_NOTE,
        }

    def match(self, surface: str, label: str) -> ChineseTerminologyMatch | None:
        """Return a configured match for one clinical surface."""

        return self.dictionary.match(
            surface,
            label,
            fuzzy=self.fuzzy,
            min_score=self.min_score,
        )

    def ground_entity(self, entity: EntityPrediction) -> EntityPrediction:
        """Return a span with code metadata, preserving all non-metadata fields."""

        match = self.match(entity.text, entity.label)
        if match is None:
            return entity

        original_metadata = dict(entity.metadata or {})
        existing_grounding = _GROUNDING_METADATA_KEYS.intersection(original_metadata)
        if existing_grounding:
            return entity
        original_metadata.update(match.metadata)
        return replace(entity, metadata=original_metadata)

    def ground_entities(
        self, entities: Iterable[EntityPrediction]
    ) -> list[EntityPrediction]:
        """Ground eligible spans in source order."""

        return [self.ground_entity(entity) for entity in entities]

    def ground_result(self, result: Any) -> Any:
        """Ground a dataclass result carrying ``entities`` and ``metadata``.

        ``AnalyzeResult`` and ``PredictionResult`` are both supported. The
        returned object has the same type, span offsets, confidence values, and
        de-identification actions as the input. Safety notices are emitted in
        result-level metadata; matched spans gain only ``system``, ``code``,
        and ``display``.
        """

        if not is_dataclass(result) or not hasattr(result, "entities"):
            raise TypeError(
                "ground_result expects an AnalyzeResult or PredictionResult "
                "dataclass with an entities field."
            )
        entities = self.ground_entities(result.entities)
        metadata = dict(getattr(result, "metadata", None) or {})
        metadata["chinese_terminology_grounding"] = self.result_metadata
        return replace(result, entities=entities, metadata=metadata)


def load_chinese_terminology_dictionary(
    dictionary_path: str | Path | None,
    *,
    license_acknowledged: bool = False,
) -> ChineseTerminologyDictionary:
    """Load a user-supplied Chinese dictionary from a local file.

    The file must contain ``label``, ``code``, and ``display`` columns/keys.
    ``aliases`` is optional and uses ``|`` separators in CSV/TSV or a string
    array in JSONL. ``system`` is optional but, when supplied, must match the
    label-specific ``ICD-10-CN`` or ``CN-DRUG`` tag.
    """

    if dictionary_path is None or not str(dictionary_path).strip():
        raise ChineseTerminologyPathError(
            "A user-supplied Chinese terminology dictionary path is required. "
            "Pass dictionary_path=...; OpenMed never bundles or downloads a "
            "Chinese ICD-10 or drug vocabulary."
        )
    if license_acknowledged is not True:
        raise ChineseTerminologyLicenseError(
            "Dictionary loading requires explicit license acknowledgement. "
            "Confirm that you have the right to use the file, then pass "
            "license_acknowledged=True."
        )

    path = Path(dictionary_path).expanduser()
    if not path.is_file():
        raise ChineseTerminologyPathError(
            f"Chinese terminology dictionary not found at {path}. Provide an "
            "existing local CSV, TSV, or JSONL file."
        )
    suffix = path.suffix.casefold()
    if suffix not in _SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise ChineseTerminologyPathError(
            f"Unsupported Chinese terminology dictionary format {suffix!r}; "
            f"expected one of: {supported}."
        )

    rows = _load_rows(path, suffix)
    entries = tuple(
        _entry_from_row(row, row_number=index, source=path)
        for index, row in enumerate(rows, start=2 if suffix != ".jsonl" else 1)
    )
    return ChineseTerminologyDictionary(entries)


def normalize_chinese_clinical_surface(value: object) -> str:
    """Normalize a Chinese clinical surface for exact and fuzzy matching."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[:1] in {"L", "M", "N"}
    )


def _load_rows(path: Path, suffix: str) -> list[Mapping[str, Any]]:
    if suffix == ".jsonl":
        rows: list[Mapping[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ChineseTerminologyError(
                        f"Invalid JSON on line {line_number} of {path}: {exc.msg}."
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ChineseTerminologyError(
                        f"Line {line_number} of {path} must be a JSON object."
                    )
                rows.append(value)
        return rows

    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ChineseTerminologyError(f"Dictionary {path} has no header row.")
        return [dict(row) for row in reader]


def _entry_from_row(
    row: Mapping[str, Any],
    *,
    row_number: int,
    source: Path,
) -> ChineseTerminologyEntry:
    label = normalize_label(_required_text(row, "label", row_number, source), lang="zh")
    if label not in _TARGET_LABELS:
        raise ChineseTerminologyError(
            f"Row {row_number} of {source} has unsupported label {row.get('label')!r}; "
            "expected CONDITION or MEDICATION."
        )

    code = _required_text(row, "code", row_number, source)
    display = _required_text(row, "display", row_number, source)
    expected_system = _DEFAULT_SYSTEMS[label]
    system = _optional_text(row.get("system")) or expected_system
    if system != expected_system:
        raise ChineseTerminologyError(
            f"Row {row_number} of {source} uses system {system!r} for {label}; "
            f"expected {expected_system!r}."
        )

    aliases = _parse_aliases(row.get("aliases"), row_number=row_number, source=source)
    return ChineseTerminologyEntry(
        label=label,
        system=system,
        code=code,
        display=display,
        aliases=aliases,
    )


def _required_text(
    row: Mapping[str, Any], key: str, row_number: int, source: Path
) -> str:
    value = _optional_text(row.get(key))
    if value is None:
        raise ChineseTerminologyError(
            f"Row {row_number} of {source} is missing required field {key!r}."
        )
    return value


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_aliases(
    value: Any,
    *,
    row_number: int,
    source: Path,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return _unique_text(part.strip() for part in value.split("|"))
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return _unique_text(str(part).strip() for part in value)
    raise ChineseTerminologyError(
        f"Row {row_number} of {source} has invalid aliases; use a | separated "
        "string or a JSON string array."
    )


def _unique_text(values: Iterable[str]) -> tuple[str, ...]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            unique.append(text)
            seen.add(text)
    return tuple(unique)


def _validate_min_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ChineseTerminologyError(
            "min_score must be a number between 0 and 1."
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ChineseTerminologyError("min_score must be a number between 0 and 1.")
    return score


def _match_sort_key(
    candidate: tuple[float, str, str, ChineseTerminologyEntry],
) -> tuple[float, str, str]:
    return (-candidate[0], candidate[1], candidate[2])


__all__ = [
    "CHINESE_DRUG_SYSTEM",
    "CHINESE_ICD_10_SYSTEM",
    "CHINESE_TERMINOLOGY_DISCLAIMER",
    "CHINESE_TERMINOLOGY_VOCABULARY_NOTE",
    "ChineseTerminologyDictionary",
    "ChineseTerminologyEntry",
    "ChineseTerminologyError",
    "ChineseTerminologyGrounder",
    "ChineseTerminologyLicenseError",
    "ChineseTerminologyMatch",
    "ChineseTerminologyPathError",
    "load_chinese_terminology_dictionary",
    "normalize_chinese_clinical_surface",
]
