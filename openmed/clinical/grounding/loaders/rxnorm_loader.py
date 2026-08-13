"""Local RxNorm vocabulary loading with TTY-aware ingredient roll-up.

The loader consumes a caller-supplied RxNorm release.  It never downloads or
ships RxNorm data.  Standard ``RXNCONSO.RRF`` and ``RXNREL.RRF`` files are
supported, as are small JSONL/TSV projections that are useful for offline
fixtures and pre-staged deployments.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..matcher import (
    ConceptMatch,
    LexicalConcept,
    VocabularyTerms,
    normalize_term,
)
from ..vocab import VocabLoaderError

__all__ = [
    "DEFAULT_TTY_PRIORITY",
    "RXNORM_SYSTEM_URI",
    "RxNormLoader",
    "RxNormLoaderError",
    "RxNormVocabularyLoader",
]

RXNORM_SYSTEM_URI = "http://www.nlm.nih.gov/research/umls/rxnorm"

_INGREDIENT_TTYS = frozenset({"IN", "PIN", "MIN"})
_PRODUCT_TTYS = frozenset(
    {
        "BN",
        "BPCK",
        "GPCK",
        "SBD",
        "SCD",
        "SBDC",
        "SCDC",
        "SBDF",
        "SCDG",
        "SBDG",
    }
)
_DOSE_FORM_TTYS = frozenset({"DF", "DFG"})

# Product and brand terms are preferred for a medication mention because they
# carry the relationship edges needed to roll up to an ingredient.  The
# remaining order is also the deterministic fallback order for a requested
# TTY that is absent from a local release.
DEFAULT_TTY_PRIORITY = (
    "SBD",
    "SCD",
    "BN",
    "IN",
    "PIN",
    "MIN",
    "BPCK",
    "GPCK",
    "SBDC",
    "SCDC",
    "SBDF",
    "SCDG",
    "SBDG",
    "DF",
    "DFG",
)

_TTY_FALLBACKS = {
    "IN": ("IN", "PIN", "MIN", "SCD", "SBD", "BN"),
    "PIN": ("PIN", "IN", "MIN", "SCD", "SBD", "BN"),
    "MIN": ("MIN", "IN", "PIN", "SCD", "SBD", "BN"),
    "SCD": ("SCD", "SBD", "BN", "IN", "PIN", "MIN"),
    "SBD": ("SBD", "SCD", "BN", "IN", "PIN", "MIN"),
    "BN": ("BN", "SBD", "SCD", "IN", "PIN", "MIN"),
}

_ENGLISH_LANGUAGES = frozenset({"", "EN", "ENG", "ENGLISH"})


class RxNormLoaderError(VocabLoaderError):
    """Raised when a caller-supplied RxNorm release is invalid or unreadable."""


@dataclass
class _RxNormRecord:
    """Terms and relationships associated with one RxNorm concept/TTY pair."""

    rxcui: str
    tty: str
    order: int
    terms: list[str] = field(default_factory=list)
    preferred_terms: list[str] = field(default_factory=list)
    ingredient_rxcuis: set[str] = field(default_factory=set)
    ingredient_names: list[str] = field(default_factory=list)
    dose_form_rxcuis: set[str] = field(default_factory=set)
    dose_form_names: list[str] = field(default_factory=list)

    @property
    def preferred_term(self) -> str:
        """Return the preferred term, falling back to source order."""

        return (self.preferred_terms or self.terms or [self.rxcui])[0]

    def add_term(self, term: str, *, preferred: bool = False) -> None:
        if term and term not in self.terms:
            self.terms.append(term)
        if preferred and term and term not in self.preferred_terms:
            self.preferred_terms.append(term)

    def add_ingredient_name(self, name: str) -> None:
        if name and name not in self.ingredient_names:
            self.ingredient_names.append(name)

    def add_dose_form_name(self, name: str) -> None:
        if name and name not in self.dose_form_names:
            self.dose_form_names.append(name)


@dataclass(frozen=True)
class _RxNormRelation:
    source_rxcui: str
    target_rxcui: str
    relation: str


@dataclass
class _ParsedRelease:
    records: dict[tuple[str, str], _RxNormRecord]
    relations: list[_RxNormRelation]
    by_rxcui: dict[str, tuple[_RxNormRecord, ...]]
    ingredient_ids: set[str]
    dose_form_ids: set[str]
    ingredient_edges: dict[str, set[str]]
    bridge_edges: dict[str, set[str]]
    dose_form_edges: dict[str, set[str]]


class RxNormLoader:
    """Load a user-supplied RxNorm release for deterministic local matching.

    Args:
        path: Directory, archive, or projection containing ``RXNCONSO.RRF``.
            ``RXNREL.RRF`` and ``RXNSAT.RRF`` are read when present.
        release_path: Keyword alias for ``path``.
        source_path: Keyword alias for ``path``.

    The loader implements :class:`VocabularyLoader`: ``system_uri`` and
    ``redistributable`` are explicit, and :meth:`load` returns matcher terms.
    Use :meth:`resolve` when TTY filtering, ingredient roll-up, or dose-form
    metadata is needed.
    """

    system_uri = RXNORM_SYSTEM_URI
    redistributable = True
    restricted_license = False

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        release_path: str | Path | None = None,
        source_path: str | Path | None = None,
    ) -> None:
        supplied = [value for value in (path, release_path, source_path) if value]
        if len(supplied) > 1:
            raise TypeError("provide only one of path, release_path, or source_path")
        if not supplied:
            raise TypeError(
                "RxNormLoader requires a user-supplied release path; RxNorm data "
                "is never downloaded or bundled"
            )
        self.path = Path(supplied[0]).expanduser()
        self._release: _ParsedRelease | None = None

    @property
    def available_ttys(self) -> tuple[str, ...]:
        """Return TTY values present in the local release."""

        release = self._parsed_release()
        return tuple(
            sorted(
                {record.tty for record in release.records.values()},
                key=self._tty_sort_key,
            )
        )

    @property
    def ttys(self) -> tuple[str, ...]:
        """Alias for :attr:`available_ttys`."""

        return self.available_ttys

    def load(self, *, tty: str | None = None) -> VocabularyTerms:
        """Return aliases mapped to RxNorm-aware lexical concepts.

        ``tty`` is optional for callers that want a terms mapping restricted to
        one term type.  When the requested type is absent, the same
        deterministic fallback order used by :meth:`resolve` is applied.
        """

        release = self._parsed_release()
        records = self._records_for_load(release, tty)
        terms: dict[str, list[LexicalConcept]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for record in records:
            concept = self._lexical_concept(release, record)
            for term in record.terms:
                if concept.key in seen[term]:
                    continue
                terms[term].append(concept)
                seen[term].add(concept.key)

        return {
            term: concepts[0] if len(concepts) == 1 else tuple(concepts)
            for term, concepts in terms.items()
        }

    def resolve(
        self,
        query: str,
        *,
        tty: str | None = None,
        limit: int | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Resolve a medication term to deterministic RxNorm concept matches.

        Exact source-term matches score ``1.0`` and Unicode/case/punctuation
        normalized matches score ``0.95``.  A product or brand record is rolled
        up to its ingredient RXCUI when the release provides a relationship;
        the original matched RXCUI remains in ``metadata["matched_rxcui"]``.
        If ``tty`` is absent from the query's candidates, a stable fallback
        order selects the next available term type.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        normalized_query = normalize_term(query)
        if not normalized_query:
            return ()

        release = self._parsed_release()
        candidates: list[tuple[_RxNormRecord, str, float, str]] = []
        for record in release.records.values():
            best: tuple[str, float, str] | None = None
            for term in record.terms:
                if term == query:
                    candidate = (term, 1.0, "exact")
                elif normalize_term(term) == normalized_query:
                    candidate = (term, 0.95, "normalized")
                else:
                    continue
                if best is None or candidate[1] > best[1]:
                    best = candidate
            if best is not None:
                candidates.append((record, best[0], best[1], best[2]))

        if not candidates:
            return ()

        selected = self._filter_tty(candidates, tty)
        matches: dict[str, tuple[ConceptMatch, tuple[Any, ...]]] = {}
        for record, term, score, match_type in selected:
            match = self._concept_match(
                release,
                record,
                term=term,
                score=score,
                match_type=match_type,
            )
            sort_key = (
                -match.score,
                self._tty_sort_key(record.tty),
                match.code,
                record.rxcui,
                term,
            )
            current = matches.get(match.code)
            if current is None or sort_key < current[1]:
                matches[match.code] = (match, sort_key)

        ordered = [item[0] for item in sorted(matches.values(), key=lambda x: x[1])]
        return tuple(ordered[:limit] if limit is not None else ordered)

    def lookup(
        self,
        query: str,
        *,
        tty: str | None = None,
        limit: int | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`, matching the shared matcher vocabulary."""

        return self.resolve(query, tty=tty, limit=limit)

    def match(
        self,
        query: str,
        *,
        tty: str | None = None,
        limit: int | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`."""

        return self.resolve(query, tty=tty, limit=limit)

    def resolve_one(self, query: str, *, tty: str | None = None) -> ConceptMatch | None:
        """Return the best match for ``query`` or ``None`` when unresolved."""

        matches = self.resolve(query, tty=tty, limit=1)
        return matches[0] if matches else None

    def _parsed_release(self) -> _ParsedRelease:
        if self._release is None:
            self._release = _parse_release(self.path)
        return self._release

    def _records_for_load(
        self, release: _ParsedRelease, tty: str | None
    ) -> tuple[_RxNormRecord, ...]:
        normalized_tty = _normalize_tty(tty)
        records = tuple(release.records.values())
        if normalized_tty is not None:
            candidates = [record for record in records if record.tty == normalized_tty]
            if candidates:
                records = tuple(candidates)
            else:
                fallback_ttys = _TTY_FALLBACKS.get(
                    normalized_tty,
                    (normalized_tty, *DEFAULT_TTY_PRIORITY),
                )
                for fallback_tty in fallback_ttys:
                    candidates = [
                        record for record in records if record.tty == fallback_tty
                    ]
                    if candidates:
                        records = tuple(candidates)
                        break
        return tuple(sorted(records, key=self._record_sort_key))

    def _filter_tty(
        self,
        candidates: Sequence[tuple[_RxNormRecord, str, float, str]],
        tty: str | None,
    ) -> tuple[tuple[_RxNormRecord, str, float, str], ...]:
        normalized_tty = _normalize_tty(tty)
        by_tty: dict[str, list[tuple[_RxNormRecord, str, float, str]]] = defaultdict(
            list
        )
        for candidate in candidates:
            by_tty[candidate[0].tty].append(candidate)

        if normalized_tty is None:
            fallback_ttys = DEFAULT_TTY_PRIORITY
        else:
            fallback_ttys = _TTY_FALLBACKS.get(
                normalized_tty,
                (normalized_tty, *DEFAULT_TTY_PRIORITY),
            )
        for fallback_tty in fallback_ttys:
            selected = by_tty.get(fallback_tty)
            if selected:
                return tuple(sorted(selected, key=self._candidate_sort_key))
        return tuple(sorted(candidates, key=self._candidate_sort_key))

    def _lexical_concept(
        self, release: _ParsedRelease, record: _RxNormRecord
    ) -> LexicalConcept:
        match = self._concept_match(
            release,
            record,
            term=record.preferred_term,
            score=1.0,
            match_type="exact",
        )
        return LexicalConcept(
            system_uri=self.system_uri,
            code=match.code,
            display=match.display,
            metadata=match.metadata,
        )

    def _concept_match(
        self,
        release: _ParsedRelease,
        record: _RxNormRecord,
        *,
        term: str,
        score: float,
        match_type: str,
    ) -> ConceptMatch:
        ingredient_rxcuis = _ingredient_rxcuis(release, record.rxcui)
        ingredient = _ingredient_label(release, record, ingredient_rxcuis)
        resolved_code = (
            ingredient_rxcuis[0] if len(ingredient_rxcuis) == 1 else record.rxcui
        )
        dose_form_rxcuis = _dose_form_rxcuis(release, record.rxcui)
        dose_form = _dose_form_label(release, record, dose_form_rxcuis)

        metadata: dict[str, object] = {
            "rxcui": resolved_code,
            "matched_rxcui": record.rxcui,
            "tty": record.tty,
            "ingredient_rxcui": (
                ingredient_rxcuis[0] if len(ingredient_rxcuis) == 1 else None
            ),
            "ingredient_rxcuis": ingredient_rxcuis,
            "ingredient": ingredient,
            "normalized_ingredient": ingredient,
        }
        if dose_form:
            metadata["dose_form"] = dose_form
        if dose_form_rxcuis:
            metadata["dose_form_rxcui"] = (
                dose_form_rxcuis[0] if len(dose_form_rxcuis) == 1 else dose_form_rxcuis
            )

        return ConceptMatch(
            system_uri=self.system_uri,
            code=resolved_code,
            display=ingredient,
            score=score,
            match_type=match_type,  # type: ignore[arg-type]
            matched_term=term,
            metadata=metadata,
        )

    @staticmethod
    def _tty_sort_key(tty: str) -> tuple[int, str]:
        try:
            return DEFAULT_TTY_PRIORITY.index(tty), tty
        except ValueError:
            return len(DEFAULT_TTY_PRIORITY), tty

    def _record_sort_key(self, record: _RxNormRecord) -> tuple[Any, ...]:
        return (*self._tty_sort_key(record.tty), record.rxcui, record.order)

    def _candidate_sort_key(
        self, candidate: tuple[_RxNormRecord, str, float, str]
    ) -> tuple[Any, ...]:
        record, term, score, _ = candidate
        return (-score, *self._tty_sort_key(record.tty), record.rxcui, term)


# The longer name reads better at integration boundaries while the shorter
# name is convenient for callers and mirrors the existing RxNorm linker.
RxNormVocabularyLoader = RxNormLoader


def _parse_release(path: Path) -> _ParsedRelease:
    if not path.exists():
        raise RxNormLoaderError(f"RxNorm release path does not exist: {path}")

    records: dict[tuple[str, str], _RxNormRecord] = {}
    relations: list[_RxNormRelation] = []
    order = 0

    def add_record(
        rxcui: str,
        tty: str,
        term: str,
        *,
        preferred: bool = False,
        ingredient_rxcuis: Iterable[str] = (),
        ingredient_names: Iterable[str] = (),
        dose_form_rxcuis: Iterable[str] = (),
        dose_form_names: Iterable[str] = (),
    ) -> None:
        nonlocal order
        normalized_rxcui = str(rxcui).strip()
        normalized_tty = str(tty).strip().upper()
        normalized_term = str(term).strip()
        if not normalized_rxcui or not normalized_tty or not normalized_term:
            return
        key = (normalized_rxcui, normalized_tty)
        record = records.get(key)
        if record is None:
            record = _RxNormRecord(
                rxcui=normalized_rxcui,
                tty=normalized_tty,
                order=order,
            )
            records[key] = record
            order += 1
        record.add_term(normalized_term, preferred=preferred)
        record.ingredient_rxcuis.update(
            str(value).strip() for value in ingredient_rxcuis if str(value).strip()
        )
        for name in ingredient_names:
            record.add_ingredient_name(str(name).strip())
        record.dose_form_rxcuis.update(
            str(value).strip() for value in dose_form_rxcuis if str(value).strip()
        )
        for name in dose_form_names:
            record.add_dose_form_name(str(name).strip())

    def add_relation(source: object, target: object, relation: object) -> None:
        if source in (None, "") or target in (None, ""):
            return
        source_text = str(source).strip()
        target_text = str(target).strip()
        relation_text = _normalize_relation(relation)
        if source_text and target_text:
            relations.append(_RxNormRelation(source_text, target_text, relation_text))

    if path.is_dir():
        conso_path = _find_named_file(path, "RXNCONSO.RRF")
        if conso_path is None:
            conso_path = _find_projection(path)
        if conso_path is None:
            raise RxNormLoaderError(
                f"RxNorm release directory does not contain RXNCONSO.RRF: {path}"
            )
        _parse_conso_path(conso_path, add_record, add_relation)
        rel_path = _find_named_file(path, "RXNREL.RRF")
        if rel_path is not None:
            _parse_relation_path(rel_path, add_relation)
        sat_path = _find_named_file(path, "RXNSAT.RRF")
        if sat_path is not None:
            _parse_sat_path(sat_path, records)
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            conso_name = _find_archive_name(names, "RXNCONSO.RRF")
            if conso_name is None:
                conso_name = _find_archive_projection(names)
            if conso_name is None:
                raise RxNormLoaderError(
                    f"RxNorm archive does not contain RXNCONSO.RRF: {path}"
                )
            _parse_conso_text(
                archive.read(conso_name).decode("utf-8", errors="replace"),
                conso_name,
                add_record,
                add_relation,
            )
            rel_name = _find_archive_name(names, "RXNREL.RRF")
            if rel_name is not None:
                _parse_relation_text(
                    archive.read(rel_name).decode("utf-8", errors="replace"),
                    rel_name,
                    add_relation,
                )
            sat_name = _find_archive_name(names, "RXNSAT.RRF")
            if sat_name is not None:
                _parse_sat_text(
                    archive.read(sat_name).decode("utf-8", errors="replace"),
                    sat_name,
                    records,
                )
    else:
        if path.name.casefold() == "rxnrel.rrf":
            sibling = _find_named_file(path.parent, "RXNCONSO.RRF")
            if sibling is None:
                raise RxNormLoaderError(
                    f"RXNREL.RRF requires a neighboring RXNCONSO.RRF: {path}"
                )
            _parse_conso_path(sibling, add_record, add_relation)
            _parse_relation_path(path, add_relation)
        else:
            _parse_conso_path(path, add_record, add_relation)
            sibling = _find_named_file(path.parent, "RXNREL.RRF")
            if sibling is not None:
                _parse_relation_path(sibling, add_relation)
            sat_path = _find_named_file(path.parent, "RXNSAT.RRF")
            if sat_path is not None:
                _parse_sat_path(sat_path, records)

    if not records:
        raise RxNormLoaderError(f"No RxNorm concepts could be read from {path}")
    return _build_release(records, relations)


def _parse_conso_path(path: Path, add_record: Any, add_relation: Any) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RxNormLoaderError(
            f"Unable to read RxNorm concept file {path}: {exc}"
        ) from exc
    _parse_conso_text(text, str(path), add_record, add_relation)


def _parse_conso_text(
    text: str, source: str, add_record: Any, add_relation: Any
) -> None:
    suffix = Path(source).suffix.casefold()
    name = Path(source).name.casefold()
    if suffix in {".json", ".jsonl"} or name.endswith(".jsonl"):
        for row in _json_rows(text, source):
            _parse_projection_row(row, add_record, add_relation)
        return
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        for row in reader:
            _parse_projection_row(row, add_record, add_relation)
        return

    for line in text.splitlines():
        fields = line.rstrip("\r").split("|")
        if not fields or fields[0].strip().upper() == "RXCUI":
            continue
        if len(fields) >= 15:
            language = fields[1].strip().upper()
            if language not in _ENGLISH_LANGUAGES:
                continue
            suppress = fields[16].strip().upper() if len(fields) > 16 else ""
            if suppress == "O":
                continue
            add_record(
                fields[0],
                fields[12],
                fields[14],
                preferred=fields[6].strip().upper() == "Y",
            )
        elif len(fields) >= 3:
            add_record(fields[0], fields[1], fields[2])


def _parse_projection_row(
    row: Mapping[str, object], add_record: Any, add_relation: Any
) -> None:
    normalized = {str(key).strip().casefold(): value for key, value in row.items()}
    language = _first_value(normalized, "lat", "language", "lang")
    if language and str(language).strip().upper() not in _ENGLISH_LANGUAGES:
        return
    suppress = _first_value(normalized, "suppress", "suppressed")
    if suppress and str(suppress).strip().upper() == "O":
        return
    rxcui = _first_value(normalized, "rxcui", "code", "concept_id", "id")
    tty = _first_value(normalized, "tty", "term_type", "termtype")
    term = _first_value(
        normalized,
        "str",
        "term",
        "name",
        "display",
        "preferred_term",
    )
    if rxcui is None or tty is None or term is None:
        return
    preferred = _truthy(_first_value(normalized, "ispref", "preferred", "is_preferred"))
    ingredient_rxcuis = _split_values(
        _first_value(
            normalized,
            "ingredient_rxcui",
            "ingredient_rxcuis",
            "ingredient_ids",
        )
    )
    ingredient_names = _split_values(
        _first_value(normalized, "ingredient", "ingredients", "ingredient_name")
    )
    dose_form_rxcuis = _split_values(
        _first_value(
            normalized,
            "dose_form_rxcui",
            "dose_form_rxcuis",
            "dose_form_ids",
        )
    )
    dose_form_names = _split_values(
        _first_value(normalized, "dose_form", "dose_form_name", "form")
    )
    aliases = _split_values(_first_value(normalized, "synonyms", "aliases", "alias"))
    add_record(
        rxcui,
        tty,
        term,
        preferred=preferred,
        ingredient_rxcuis=ingredient_rxcuis,
        ingredient_names=ingredient_names,
        dose_form_rxcuis=dose_form_rxcuis,
        dose_form_names=dose_form_names,
    )
    for alias in aliases:
        add_record(
            rxcui,
            tty,
            alias,
            ingredient_rxcuis=ingredient_rxcuis,
            ingredient_names=ingredient_names,
            dose_form_rxcuis=dose_form_rxcuis,
            dose_form_names=dose_form_names,
        )
    relationships = normalized.get("relationships")
    if isinstance(relationships, str):
        relationships = _json_value(relationships)
    if isinstance(relationships, Sequence) and not isinstance(
        relationships, (str, bytes, bytearray)
    ):
        for relationship in relationships:
            if not isinstance(relationship, Mapping):
                continue
            add_relation(
                _first_value(relationship, "rxcui1", "source", "source_rxcui"),
                _first_value(relationship, "rxcui2", "target", "target_rxcui"),
                _first_value(relationship, "rela", "relation", "rel") or "",
            )


def _parse_relation_path(path: Path, add_relation: Any) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RxNormLoaderError(
            f"Unable to read RxNorm relation file {path}: {exc}"
        ) from exc
    _parse_relation_text(text, str(path), add_relation)


def _parse_relation_text(text: str, source: str, add_relation: Any) -> None:
    suffix = Path(source).suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        for row in _json_rows(text, source):
            if isinstance(row, Mapping):
                add_relation(
                    _first_value(row, "rxcui1", "source", "source_rxcui"),
                    _first_value(row, "rxcui2", "target", "target_rxcui"),
                    _first_value(row, "rela", "relation", "rel") or "",
                )
        return
    for line in text.splitlines():
        fields = line.rstrip("\r").split("|")
        if not fields or fields[0].strip().upper() in {"RXCUI1", "SOURCE"}:
            continue
        if len(fields) >= 8:
            suppress = fields[13].strip().upper() if len(fields) > 13 else ""
            if suppress == "O":
                continue
            add_relation(fields[0], fields[4], fields[7] or fields[3])
        elif len(fields) >= 3:
            add_relation(fields[0], fields[1], fields[2])


def _parse_sat_path(
    path: Path, records: Mapping[tuple[str, str], _RxNormRecord]
) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise RxNormLoaderError(
            f"Unable to read RxNorm attribute file {path}: {exc}"
        ) from exc
    _parse_sat_text(text, str(path), records)


def _parse_sat_text(
    text: str,
    source: str,
    records: Mapping[tuple[str, str], _RxNormRecord],
) -> None:
    suffix = Path(source).suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        rows = _json_rows(text, source)
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            rxcui = _first_value(row, "rxcui", "code", "concept_id")
            attribute = _first_value(row, "atn", "attribute", "name")
            value = _first_value(row, "atv", "value")
            if rxcui and attribute and value and _is_dose_form_attribute(attribute):
                _add_dose_form_name(records, str(rxcui), str(value))
        return
    for line in text.splitlines():
        fields = line.rstrip("\r").split("|")
        if len(fields) < 5:
            continue
        if _is_dose_form_attribute(fields[3]):
            _add_dose_form_name(records, fields[0], fields[4])


def _add_dose_form_name(
    records: Mapping[tuple[str, str], _RxNormRecord], rxcui: str, name: str
) -> None:
    normalized_rxcui = str(rxcui).strip()
    normalized_name = str(name).strip()
    if not normalized_rxcui or not normalized_name:
        return
    for record in records.values():
        if record.rxcui == normalized_rxcui:
            record.add_dose_form_name(normalized_name)


def _build_release(
    records: dict[tuple[str, str], _RxNormRecord],
    relations: list[_RxNormRelation],
) -> _ParsedRelease:
    grouped: dict[str, list[_RxNormRecord]] = defaultdict(list)
    for record in records.values():
        grouped[record.rxcui].append(record)
    by_rxcui = {
        rxcui: tuple(sorted(values, key=lambda record: (record.order, record.tty)))
        for rxcui, values in grouped.items()
    }

    ingredient_ids = {
        record.rxcui for record in records.values() if record.tty in _INGREDIENT_TTYS
    }
    dose_form_ids = {
        record.rxcui for record in records.values() if record.tty in _DOSE_FORM_TTYS
    }
    ingredient_edges: dict[str, set[str]] = defaultdict(set)
    bridge_edges: dict[str, set[str]] = defaultdict(set)
    dose_form_edges: dict[str, set[str]] = defaultdict(set)

    for record in records.values():
        ingredient_edges[record.rxcui].update(record.ingredient_rxcuis)
        ingredient_ids.update(record.ingredient_rxcuis)
        dose_form_edges[record.rxcui].update(record.dose_form_rxcuis)
        dose_form_ids.update(record.dose_form_rxcuis)

    for relation in relations:
        source = relation.source_rxcui
        target = relation.target_rxcui
        relation_text = relation.relation
        source_ttys = {record.tty for record in by_rxcui.get(source, ())}
        target_ttys = {record.tty for record in by_rxcui.get(target, ())}
        source_is_ingredient = (
            bool(source_ttys & _INGREDIENT_TTYS) or source in ingredient_ids
        )
        target_is_ingredient = (
            bool(target_ttys & _INGREDIENT_TTYS) or target in ingredient_ids
        )
        source_is_dose = bool(source_ttys & _DOSE_FORM_TTYS) or source in dose_form_ids
        target_is_dose = bool(target_ttys & _DOSE_FORM_TTYS) or target in dose_form_ids

        if source_is_ingredient != target_is_ingredient:
            ingredient = source if source_is_ingredient else target
            product = target if source_is_ingredient else source
            ingredient_edges[product].add(ingredient)
            ingredient_ids.add(ingredient)
            continue
        if source_is_dose != target_is_dose:
            dose_form = source if source_is_dose else target
            product = target if source_is_dose else source
            dose_form_edges[product].add(dose_form)
            dose_form_ids.add(dose_form)
            continue

        if _is_bridge_relation(relation_text) or (
            (source_ttys & _PRODUCT_TTYS or source_ttys == {"BN"})
            and (target_ttys & _PRODUCT_TTYS or target_ttys == {"BN"})
        ):
            bridge_edges[source].add(target)
            bridge_edges[target].add(source)
        elif "ingredient" in relation_text:
            if _is_reverse_relation(relation_text):
                ingredient_edges[target].add(source)
                ingredient_ids.add(source)
            else:
                ingredient_edges[source].add(target)
                ingredient_ids.add(target)
        elif "dose" in relation_text or "form" in relation_text:
            if _is_reverse_relation(relation_text):
                dose_form_edges[target].add(source)
                dose_form_ids.add(source)
            else:
                dose_form_edges[source].add(target)
                dose_form_ids.add(target)

    return _ParsedRelease(
        records=records,
        relations=relations,
        by_rxcui=by_rxcui,
        ingredient_ids=ingredient_ids,
        dose_form_ids=dose_form_ids,
        ingredient_edges=ingredient_edges,
        bridge_edges=bridge_edges,
        dose_form_edges=dose_form_edges,
    )


def _ingredient_rxcuis(release: _ParsedRelease, rxcui: str) -> tuple[str, ...]:
    found = _walk_to_targets(
        rxcui,
        release.ingredient_edges,
        release.bridge_edges,
        release.ingredient_ids,
    )
    if not found and rxcui in release.ingredient_ids:
        found.add(rxcui)
    return tuple(sorted(found))


def _dose_form_rxcuis(release: _ParsedRelease, rxcui: str) -> tuple[str, ...]:
    found = _walk_to_targets(
        rxcui,
        release.dose_form_edges,
        release.bridge_edges,
        release.dose_form_ids,
    )
    return tuple(sorted(found))


def _walk_to_targets(
    start: str,
    direct_edges: Mapping[str, set[str]],
    bridge_edges: Mapping[str, set[str]],
    target_ids: set[str],
) -> set[str]:
    found: set[str] = set()
    pending = [start]
    visited: set[str] = set()
    while pending:
        current = pending.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for target in sorted(direct_edges.get(current, ())):
            if target in target_ids:
                found.add(target)
            elif target not in visited:
                pending.append(target)
        for target in sorted(bridge_edges.get(current, ())):
            if target not in visited:
                pending.append(target)
    return found


def _ingredient_label(
    release: _ParsedRelease,
    record: _RxNormRecord,
    ingredient_rxcuis: Sequence[str],
) -> str:
    labels: list[str] = []
    for rxcui in ingredient_rxcuis:
        ingredient_records = release.by_rxcui.get(rxcui, ())
        if ingredient_records:
            candidate = sorted(
                ingredient_records,
                key=lambda item: (
                    0 if item.tty in _INGREDIENT_TTYS else 1,
                    item.order,
                    item.tty,
                ),
            )[0].preferred_term
        else:
            candidate = rxcui
        if candidate not in labels:
            labels.append(candidate)
    if not labels:
        labels.extend(record.ingredient_names)
    if not labels and record.tty in _INGREDIENT_TTYS:
        labels.append(record.preferred_term)
    return " + ".join(labels) or record.preferred_term


def _dose_form_label(
    release: _ParsedRelease,
    record: _RxNormRecord,
    dose_form_rxcuis: Sequence[str],
) -> str | None:
    labels: list[str] = []
    for rxcui in dose_form_rxcuis:
        dose_records = release.by_rxcui.get(rxcui, ())
        if dose_records:
            candidate = sorted(dose_records, key=lambda item: (item.order, item.tty))[0]
            label = candidate.preferred_term
        else:
            label = rxcui
        if label not in labels:
            labels.append(label)
    if not labels:
        labels.extend(record.dose_form_names)
    return " + ".join(labels) or None


def _find_named_file(root: Path, name: str) -> Path | None:
    wanted = name.casefold()
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.name.casefold() == wanted:
            return candidate
    return None


def _find_projection(root: Path) -> Path | None:
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and candidate.suffix.casefold() in {
            ".csv",
            ".json",
            ".jsonl",
            ".tsv",
        }:
            return candidate
    return None


def _find_archive_name(names: Iterable[str], name: str) -> str | None:
    wanted = name.casefold()
    for candidate in sorted(names):
        if candidate.rsplit("/", 1)[-1].casefold() == wanted:
            return candidate
    return None


def _find_archive_projection(names: Iterable[str]) -> str | None:
    for candidate in sorted(names):
        if candidate.casefold().endswith((".csv", ".json", ".jsonl", ".tsv")):
            return candidate
    return None


def _json_rows(text: str, source: str) -> Iterable[Mapping[str, object]]:
    if Path(source).suffix.casefold() == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                row = _json_value(line)
                if isinstance(row, Mapping):
                    yield row
        return
    parsed = _json_value(text)
    if isinstance(parsed, Mapping):
        rows = parsed.get("rows", parsed.get("concepts", parsed.get("terms")))
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes, bytearray)):
            for row in rows:
                if isinstance(row, Mapping):
                    yield row
        else:
            yield parsed
    elif isinstance(parsed, Sequence) and not isinstance(
        parsed, (str, bytes, bytearray)
    ):
        for row in parsed:
            if isinstance(row, Mapping):
                yield row


def _json_value(value: str) -> object | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise RxNormLoaderError(f"Invalid JSON in RxNorm projection: {exc}") from exc


def _first_value(row: Mapping[str, object], *keys: str) -> object | None:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _split_values(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        values = str(value).replace(";", "|").split("|")
    return tuple(str(item).strip() for item in values if str(item).strip())


def _truthy(value: object | None) -> bool:
    return str(value).strip().casefold() in {"1", "true", "y", "yes"}


def _normalize_tty(tty: str | None) -> str | None:
    if tty is None:
        return None
    if not isinstance(tty, str) or not tty.strip():
        raise ValueError("tty must be a non-empty string or None")
    return tty.strip().upper()


def _normalize_relation(value: object) -> str:
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def _is_reverse_relation(relation: str) -> bool:
    return (
        relation.endswith(" of")
        or relation.startswith("ingredient of")
        or relation.startswith("dose form of")
    )


def _is_bridge_relation(relation: str) -> bool:
    return any(
        token in relation
        for token in ("tradename", "trade name", "brand name", "has brand")
    )


def _is_dose_form_attribute(attribute: object) -> bool:
    normalized = str(attribute).strip().casefold().replace("_", " ")
    return "dose form" in normalized or normalized in {"form", "rxn dose form"}
