"""Load caller-supplied ICD-10-CM releases for local clinical grounding.

ICD-10-CM is a public-domain classification published by the Centers for
Medicare & Medicaid Services (CMS).  This module reads a release supplied by
the caller and never downloads, embeds, or redistributes ICD-10-CM content.
CMS tabular-order and alphabetic-index text files are supported, as are small
CSV, TSV, JSON, and JSONL projections useful for offline deployments and
synthetic tests.

The loader keeps hierarchy and billing status alongside lexical terms.  A
``ConceptMatch`` therefore carries the resolved code, its billable/header
status, and a root-to-code category path without requiring downstream callers
to parse the source files again.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ..matcher import (
    ConceptMatch,
    LexicalConcept,
    LexicalMatcher,
    VocabularyTerms,
    normalize_term,
)
from ..vocab import VocabLoaderError

__all__ = [
    "ICD10CM_CODE_PATTERN",
    "ICD10CM_LICENSE_NOTE",
    "ICD10CM_SYSTEM_URI",
    "ICD10CMLoader",
    "ICD10CMVocabularyLoader",
    "Icd10cmCode",
    "Icd10cmLoader",
    "Icd10cmLoaderError",
    "Icd10cmVocabularyLoader",
]

ICD10CM_SYSTEM_URI = "http://hl7.org/fhir/sid/icd-10-cm"
ICD10CM_LICENSE_NOTE = (
    "ICD-10-CM is a public domain classification published by CMS. The caller "
    "must supply release files; OpenMed does not bundle or redistribute ICD-10-CM "
    "data."
)

# The first three characters are the ICD-10-CM category.  Extensions after the
# dot may contain letters as well as digits (for example, a seventh-character
# encounter extension), so the pattern intentionally does not restrict them to
# numeric values.
ICD10CM_CODE_PATTERN = r"[A-Z][0-9]{2}(?:\.[A-Z0-9]{1,4})?"
_CODE_RE = re.compile(rf"(?<![A-Z0-9])({ICD10CM_CODE_PATTERN})(?![A-Z0-9])")
_CODE_LINE_RE = re.compile(
    rf"^\s*[#*]?\s*(?P<code>{ICD10CM_CODE_PATTERN})"
    r"(?:\s+|\t+)(?P<display>.+?)\s*$",
    re.IGNORECASE,
)
_SUPPORTED_SUFFIXES = {".csv", ".json", ".jsonl", ".tsv", ".txt"}
_TRUE_VALUES = frozenset({"1", "true", "t", "yes", "y", "billable", "leaf", "valid"})
_FALSE_VALUES = frozenset(
    {
        "0",
        "false",
        "f",
        "no",
        "n",
        "header",
        "nonbillable",
        "non-billable",
        "non_billable",
        "invalid",
    }
)


class Icd10cmLoaderError(VocabLoaderError):
    """Raised when a caller-supplied ICD-10-CM release is invalid or unreadable."""


@dataclass(frozen=True)
class Icd10cmCode:
    """One parsed ICD-10-CM code and its derived hierarchy metadata.

    ``category_path`` contains canonical codes from the three-character
    category through this code.  A category such as ``A00`` therefore has a
    one-element path, while ``A00.1`` has ``("A00", "A00.1")``.
    """

    code: str
    display: str
    billable: bool
    parent: str | None = None
    category_path: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _normalize_code(self.code) or self.code)
        object.__setattr__(self, "display", str(self.display).strip() or self.code)
        object.__setattr__(self, "billable", bool(self.billable))
        object.__setattr__(self, "parent", _normalize_code(self.parent))
        object.__setattr__(
            self,
            "category_path",
            tuple(
                normalized
                for value in self.category_path
                if (normalized := _normalize_code(value)) is not None
            ),
        )
        object.__setattr__(
            self,
            "aliases",
            _unique_text_values(self.aliases),
        )


@dataclass
class _CodeBuilder:
    """Mutable source-order record used while combining release files."""

    code: str
    display: str
    order: int
    billable: bool | None = None
    explicit_billable: bool = False
    parent: str | None = None
    declared_path: tuple[str, ...] = ()
    aliases: list[str] = field(default_factory=list)

    def add_alias(self, value: object) -> None:
        text = _text(value)
        if text and text not in self.aliases and text != self.display:
            self.aliases.append(text)


@dataclass(frozen=True)
class _ParsedRelease:
    records: tuple[Icd10cmCode, ...]
    by_code: Mapping[str, Icd10cmCode]


class Icd10cmLoader:
    """Load a user-supplied ICD-10-CM release for offline lexical grounding.

    Args:
        path: Directory, archive, or projection containing a tabular release.
        release_path: Keyword alias for ``path``.
        source_path: Keyword alias for ``path``.
        tabular_path: Explicit CMS tabular-order or projection path.
        index_path: Optional CMS alphabetic-index or projection path.

    The loader implements the redistributable vocabulary-loader contract.  Its
    ``load`` method returns terms for :class:`LexicalMatcher`; ``resolve`` adds
    ICD-10-CM billing and hierarchy metadata.  No source file is read during
    construction, and no network operation is performed by this class.
    """

    system_uri = ICD10CM_SYSTEM_URI
    redistributable = True
    restricted_license = False
    license_note = ICD10CM_LICENSE_NOTE

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        release_path: str | Path | None = None,
        source_path: str | Path | None = None,
        tabular_path: str | Path | None = None,
        tabular_order_path: str | Path | None = None,
        index_path: str | Path | None = None,
        alphabetic_index_path: str | Path | None = None,
    ) -> None:
        supplied = [value for value in (path, release_path, source_path) if value]
        if len(supplied) > 1:
            raise TypeError("provide only one of path, release_path, or source_path")

        explicit_tabular = [
            value for value in (tabular_path, tabular_order_path) if value
        ]
        if len(explicit_tabular) > 1:
            raise TypeError("provide only one of tabular_path or tabular_order_path")
        explicit_index = [
            value for value in (index_path, alphabetic_index_path) if value
        ]
        if len(explicit_index) > 1:
            raise TypeError("provide only one of index_path or alphabetic_index_path")
        if not supplied and not explicit_tabular and not explicit_index:
            raise TypeError(
                "Icd10cmLoader requires a user-supplied release or tabular path; "
                "ICD-10-CM data is never downloaded or bundled"
            )

        self.path = Path(
            supplied[0]
            if supplied
            else explicit_tabular[0]
            if explicit_tabular
            else cast(Path, explicit_index[0])
        ).expanduser()
        self.tabular_path = _optional_path(
            explicit_tabular[0] if explicit_tabular else None
        )
        self.index_path = _optional_path(explicit_index[0] if explicit_index else None)
        self._release: _ParsedRelease | None = None
        self._terms: VocabularyTerms | None = None
        self._matcher: LexicalMatcher | None = None

    @property
    def records(self) -> tuple[Icd10cmCode, ...]:
        """Return parsed codes in source order."""

        return self._parsed_release().records

    @property
    def codes(self) -> tuple[str, ...]:
        """Return canonical codes in source order."""

        return tuple(record.code for record in self.records)

    def load(self) -> VocabularyTerms:
        """Return ICD-10-CM aliases mapped to matcher concepts.

        Both tabular descriptions and alphabetic-index terms are indexed.  The
        returned mapping contains only data parsed from the caller's path.
        """

        if self._terms is not None:
            return self._terms

        terms: dict[str, list[LexicalConcept]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for record in self.records:
            concept = self._lexical_concept(record)
            for alias in (record.display, *record.aliases):
                if not alias or concept.key in seen[alias]:
                    continue
                terms[alias].append(concept)
                seen[alias].add(concept.key)

        self._terms = {
            term: concepts[0] if len(concepts) == 1 else tuple(concepts)
            for term, concepts in terms.items()
        }
        return self._terms

    def resolve(
        self,
        query: str,
        *,
        billable_only: bool = False,
        include_headers: bool | None = None,
        limit: int | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Resolve a condition term to ranked ICD-10-CM concept matches.

        Exact and normalized matches use the shared deterministic lexical
        matcher.  When scores tie, billable leaf codes rank ahead of category
        headers, followed by more-specific and then lexicographically smaller
        codes.  Headers remain available by default with ``billable=False`` in
        their metadata; pass ``billable_only=True`` or
        ``include_headers=False`` to exclude them.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not isinstance(billable_only, bool):
            raise TypeError("billable_only must be a boolean")
        if include_headers is not None and not isinstance(include_headers, bool):
            raise TypeError("include_headers must be a boolean or None")
        if include_headers is False:
            billable_only = True
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        if not normalize_term(query):
            return ()

        release = self._parsed_release()
        matches: list[ConceptMatch] = []
        for lexical_match in self._lexical_matcher().lookup(query):
            record = release.by_code.get(lexical_match.code)
            if record is None or (billable_only and not record.billable):
                continue
            matches.append(
                self._concept_match(
                    record,
                    matched_term=lexical_match.matched_term,
                    score=lexical_match.score,
                    match_type=lexical_match.match_type,
                )
            )

        matches.sort(
            key=lambda match: (
                -match.score,
                not bool(match.metadata["billable"]),
                -len(match.code.replace(".", "")),
                match.code,
                match.matched_term,
            )
        )
        return tuple(matches[:limit] if limit is not None else matches)

    def lookup(self, query: str, **kwargs: object) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`, matching the shared matcher vocabulary."""

        return self.resolve(query, **kwargs)

    def match(self, query: str, **kwargs: object) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`."""

        return self.resolve(query, **kwargs)

    def resolve_one(
        self,
        query: str,
        **kwargs: object,
    ) -> ConceptMatch | None:
        """Return the best match for ``query`` or ``None`` when unresolved."""

        kwargs["limit"] = 1
        matches = self.resolve(query, **kwargs)
        return matches[0] if matches else None

    def is_valid_code(self, code: object, *, billable_only: bool = False) -> bool:
        """Return whether ``code`` occurs in the supplied release.

        A recognized category header is a valid ICD-10-CM code but is not
        billable.  Use ``billable_only=True`` when validity means a billable
        leaf for export, or call :meth:`is_billable_code` directly.
        """

        if not isinstance(billable_only, bool):
            raise TypeError("billable_only must be a boolean")
        normalized = _normalize_code(code)
        if normalized is None:
            return False
        record = self._parsed_release().by_code.get(normalized)
        return record is not None and (not billable_only or record.billable)

    def is_billable_code(self, code: object) -> bool:
        """Return whether ``code`` is present and marked billable."""

        return self.is_valid_code(code, billable_only=True)

    def code_info(self, code: object) -> Icd10cmCode | None:
        """Return parsed metadata for ``code`` or ``None`` when unknown."""

        normalized = _normalize_code(code)
        if normalized is None:
            return None
        return self._parsed_release().by_code.get(normalized)

    def ancestors(
        self,
        code: object,
        *,
        include_self: bool = False,
    ) -> tuple[str, ...]:
        """Return loaded ancestors from the category toward ``code``.

        The default excludes ``code`` itself.  Set ``include_self=True`` to
        obtain the same inclusive root-to-code sequence as
        :meth:`category_path`.
        """

        if not isinstance(include_self, bool):
            raise TypeError("include_self must be a boolean")
        record = self.code_info(code)
        if record is None:
            return ()
        return record.category_path if include_self else record.category_path[:-1]

    def category_path(self, code: object) -> tuple[str, ...]:
        """Return the inclusive root-to-code category path."""

        return self.ancestors(code, include_self=True)

    def children(self, code: object) -> tuple[str, ...]:
        """Return direct child codes in source order."""

        normalized = _normalize_code(code)
        if normalized is None or not self.is_valid_code(normalized):
            return ()
        return tuple(
            record.code for record in self.records if record.parent == normalized
        )

    def descendants(self, code: object) -> tuple[str, ...]:
        """Return all loaded descendants of ``code`` in source order."""

        normalized = _normalize_code(code)
        if normalized is None or not self.is_valid_code(normalized):
            return ()
        return tuple(
            record.code
            for record in self.records
            if record.code != normalized and normalized in record.category_path[:-1]
        )

    def _parsed_release(self) -> _ParsedRelease:
        if self._release is None:
            self._release = _parse_release(
                self.path,
                tabular_path=self.tabular_path,
                index_path=self.index_path,
            )
        return self._release

    def _lexical_matcher(self) -> LexicalMatcher:
        if self._matcher is None:
            self._matcher = LexicalMatcher(self.load(), system_uri=self.system_uri)
        return self._matcher

    def _lexical_concept(self, record: Icd10cmCode) -> LexicalConcept:
        return LexicalConcept(
            system_uri=self.system_uri,
            code=record.code,
            display=record.display,
            metadata=_record_metadata(record),
        )

    def _concept_match(
        self,
        record: Icd10cmCode,
        *,
        matched_term: str,
        score: float,
        match_type: str,
    ) -> ConceptMatch:
        return ConceptMatch(
            system_uri=self.system_uri,
            code=record.code,
            display=record.display,
            score=score,
            match_type=cast(Any, match_type),
            matched_term=matched_term,
            metadata=_record_metadata(record),
        )


# Keep both acronym and mixed-case spellings available at integration
# boundaries.  The shorter name follows the existing loader API while the
# acronym spelling is convenient for terminology registry configuration.
Icd10cmVocabularyLoader = Icd10cmLoader
ICD10CMLoader = Icd10cmLoader
ICD10CMVocabularyLoader = Icd10cmLoader


def _parse_release(
    path: Path,
    *,
    tabular_path: Path | None,
    index_path: Path | None,
) -> _ParsedRelease:
    if not path.exists():
        raise Icd10cmLoaderError(f"ICD-10-CM release path does not exist: {path}")

    if path.is_dir():
        table_source = tabular_path or _find_local_file(path, kind="tabular")
        index_source = index_path or _find_local_file(path, kind="index")
        tabular_rows = (
            _read_rows_from_path(table_source, kind="tabular") if table_source else ()
        )
        index_rows = (
            _read_rows_from_path(index_source, kind="index") if index_source else ()
        )
        return _build_release(tabular_rows, index_rows)

    if zipfile.is_zipfile(path):
        return _parse_archive(path)

    table_source = tabular_path
    index_source = index_path
    if table_source is None:
        if path.is_file() and not _looks_like_index_filename(path.name):
            table_source = path
        else:
            table_source = _find_local_file(path.parent, kind="tabular")
    if index_source is None:
        if path.is_file() and _looks_like_index_filename(path.name):
            index_source = path
        else:
            index_source = _find_local_file(path.parent, kind="index")

    tabular_rows = (
        _read_rows_from_path(table_source, kind="tabular") if table_source else ()
    )
    index_rows = (
        _read_rows_from_path(index_source, kind="index") if index_source else ()
    )
    return _build_release(tabular_rows, index_rows)


def _parse_archive(path: Path) -> _ParsedRelease:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise Icd10cmLoaderError(
            f"Unable to read ICD-10-CM archive {path}: {exc}"
        ) from exc

    with archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        table_name = _find_archive_member(names, kind="tabular")
        index_name = _find_archive_member(names, kind="index")
        tabular_rows = (
            _read_rows_from_bytes(archive.read(table_name), table_name, kind="tabular")
            if table_name
            else ()
        )
        index_rows = (
            _read_rows_from_bytes(archive.read(index_name), index_name, kind="index")
            if index_name
            else ()
        )
    return _build_release(tabular_rows, index_rows)


def _build_release(
    tabular_rows: Iterable[Mapping[str, object]],
    index_rows: Iterable[Mapping[str, object]],
) -> _ParsedRelease:
    builders: dict[str, _CodeBuilder] = {}
    order = 0

    def add_code(
        code_value: object,
        display_value: object,
        *,
        aliases: Iterable[object] = (),
        billable: bool | None = None,
        parent: object | None = None,
        category_path: Iterable[object] = (),
    ) -> None:
        nonlocal order
        code = _normalize_code(code_value)
        if code is None:
            return
        display = _text(display_value) or code
        record = builders.get(code)
        if record is None:
            record = _CodeBuilder(code=code, display=display, order=order)
            builders[code] = record
            order += 1
        elif record.display == record.code and display != record.code:
            record.display = display

        for alias in aliases:
            record.add_alias(alias)
        if billable is not None and not record.explicit_billable:
            record.billable = billable
            record.explicit_billable = True
        normalized_parent = _normalize_code(parent)
        if normalized_parent and normalized_parent != code and record.parent is None:
            record.parent = normalized_parent
        normalized_path = tuple(
            normalized
            for value in category_path
            if (normalized := _normalize_code(value)) is not None
        )
        if normalized_path and not record.declared_path:
            record.declared_path = normalized_path

    for row in tabular_rows:
        normalized_row = _normalize_row(row)
        codes = _row_codes(normalized_row)
        if not codes:
            continue
        display = _first_value(
            normalized_row,
            "description",
            "display",
            "preferred_term",
            "preferred",
            "canonical_term",
            "long_description",
            "name",
            "term",
            "label",
        )
        aliases = _split_values(
            _first_value(
                normalized_row,
                "aliases",
                "alias",
                "synonyms",
                "synonym",
                "terms",
                "index_terms",
            )
        )
        billable = _billable_value(normalized_row)
        parent = _first_value(
            normalized_row,
            "parent",
            "parent_code",
            "parent_id",
            "category",
            "category_code",
        )
        path_value = _first_value(
            normalized_row,
            "category_path",
            "path",
            "hierarchy",
        )
        path_values = _split_values(path_value)
        for code in codes:
            add_code(
                code,
                display,
                aliases=aliases,
                billable=billable,
                parent=parent,
                category_path=path_values,
            )

        # A projection may put child codes in a parent's ``children`` field.
        for child in _split_values(_first_value(normalized_row, "children")):
            add_code(child, child, parent=codes[0])

    for row in index_rows:
        normalized_row = _normalize_row(row)
        codes = _row_codes(normalized_row)
        if not codes:
            continue
        term = _first_value(
            normalized_row,
            "term",
            "display",
            "description",
            "preferred_term",
            "name",
            "label",
            "alias",
        )
        aliases = _split_values(
            _first_value(normalized_row, "aliases", "synonyms", "terms")
        )
        for code in codes:
            add_code(code, term or code, aliases=(term, *aliases))

    if not builders:
        raise Icd10cmLoaderError(
            "No ICD-10-CM concepts could be read from the supplied release"
        )

    known_codes = set(builders)
    parents: dict[str, str | None] = {}
    children: dict[str, set[str]] = defaultdict(set)
    for record in builders.values():
        parent = record.parent if record.parent in known_codes else None
        if parent is None and record.declared_path:
            path = tuple(
                value for value in record.declared_path if value in known_codes
            )
            if record.code in path:
                parent = path[-2] if len(path) > 1 else None
        if parent is None:
            parent = _derived_parent(record.code, known_codes)
        if parent == record.code:
            parent = None
        parents[record.code] = parent
        if parent is not None:
            children[parent].add(record.code)

    built_records: list[Icd10cmCode] = []
    for record in sorted(builders.values(), key=lambda item: item.order):
        billable = (
            record.billable
            if record.explicit_billable and record.billable is not None
            else record.code not in children
        )
        parent = parents[record.code]
        path_values: list[str] = []
        cursor: str | None = record.code
        visited: set[str] = set()
        while cursor is not None and cursor not in visited:
            visited.add(cursor)
            path_values.append(cursor)
            cursor = parents.get(cursor)
        path_values.reverse()
        built_records.append(
            Icd10cmCode(
                code=record.code,
                display=record.display,
                billable=bool(billable),
                parent=parent,
                category_path=tuple(path_values),
                aliases=tuple(record.aliases),
            )
        )

    ordered = tuple(built_records)
    return _ParsedRelease(ordered, {record.code: record for record in ordered})


def _record_metadata(record: Icd10cmCode) -> dict[str, object]:
    """Return JSON-ready, non-source-text metadata for a concept."""

    ancestors = record.category_path[:-1]
    return {
        "billable": record.billable,
        "is_billable": record.billable,
        "header": not record.billable,
        "parent": record.parent,
        "ancestors": ancestors,
        "category_path": record.category_path,
        "category_path_codes": record.category_path,
        "license_note": ICD10CM_LICENSE_NOTE,
    }


def _derived_parent(code: str, known_codes: set[str]) -> str | None:
    compact = code.replace(".", "")
    if len(compact) <= 3:
        return None
    for length in range(len(compact) - 1, 2, -1):
        candidate = _format_compact_code(compact[:length])
        if candidate in known_codes:
            return candidate
    category = compact[:3]
    return category if category in known_codes else None


def _format_compact_code(value: str) -> str:
    return value[:3] if len(value) <= 3 else f"{value[:3]}.{value[3:]}"


def _normalize_code(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    if not text:
        return None
    if "." not in text and len(text) > 3:
        text = f"{text[:3]}.{text[3:]}"
    if not re.fullmatch(ICD10CM_CODE_PATTERN, text):
        return None
    return text


def _row_codes(row: Mapping[str, object]) -> tuple[str, ...]:
    value = _first_value(
        row,
        "code",
        "icd_code",
        "icd10cm",
        "icd10cm_code",
        "code_value",
        "concept_id",
        "id",
        "codes",
    )
    values = _split_values(value, split_commas=True)
    result: list[str] = []
    for candidate in values:
        normalized = _normalize_code(candidate)
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)


def _billable_value(row: Mapping[str, object]) -> bool | None:
    for key in ("billable", "is_billable", "billable_code", "leaf"):
        value = _first_value(row, key)
        if value is not None:
            parsed = _parse_bool(value)
            if parsed is not None:
                return parsed
    for key in ("header", "is_header", "non_billable", "nonbillable"):
        value = _first_value(row, key)
        if value is not None:
            parsed = _parse_bool(value)
            if parsed is not None:
                return not parsed
    status = _text(_first_value(row, "status", "code_status"))
    if status:
        parsed = _parse_bool(status)
        if parsed is not None:
            return parsed
    return None


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().casefold().replace(" ", "")
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return None


def _read_rows_from_path(
    path: Path | None,
    *,
    kind: str,
) -> tuple[Mapping[str, object], ...]:
    if path is None:
        return ()
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise Icd10cmLoaderError(
            f"Unable to read ICD-10-CM file {path}: {exc}"
        ) from exc
    return _read_rows_from_bytes(data, str(path), kind=kind)


def _read_rows_from_bytes(
    data: bytes,
    source: str,
    *,
    kind: str,
) -> tuple[Mapping[str, object], ...]:
    text = data.decode("utf-8-sig", errors="replace")
    suffix = Path(source).suffix.casefold()
    try:
        if suffix in {".json", ".jsonl"}:
            rows = _parse_json_rows(text, source)
        elif suffix in {".csv", ".tsv"}:
            rows = _parse_delimited_rows(text, suffix)
        elif kind == "index":
            rows = _parse_index_text(text)
        else:
            rows = _parse_tabular_text(text)
    except (csv.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise Icd10cmLoaderError(
            f"Unable to parse ICD-10-CM file {source}: {exc}"
        ) from exc
    return tuple(rows)


def _parse_json_rows(text: str, source: str) -> tuple[Mapping[str, object], ...]:
    if Path(source).suffix.casefold() == ".jsonl":
        rows: list[Mapping[str, object]] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                rows.append(value)
        return tuple(rows)

    value = json.loads(text)
    if isinstance(value, Mapping):
        nested = _first_value(value, "rows", "concepts", "codes", "terms")
        if isinstance(nested, Sequence) and not isinstance(
            nested, (str, bytes, bytearray)
        ):
            return tuple(row for row in nested if isinstance(row, Mapping))
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(row for row in value if isinstance(row, Mapping))
    return ()


def _parse_delimited_rows(
    text: str,
    suffix: str,
) -> tuple[Mapping[str, object], ...]:
    delimiter = "\t" if suffix == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return tuple(
        {str(key): value for key, value in row.items() if key is not None}
        for row in reader
    )


def _parse_tabular_text(text: str) -> tuple[Mapping[str, object], ...]:
    lines = [line.rstrip("\r") for line in text.splitlines() if line.strip()]
    if not lines:
        return ()

    first = lines[0].casefold()
    if any(marker in first for marker in ("code", "description", "billable")) and (
        "\t" in lines[0] or "," in lines[0] or "|" in lines[0]
    ):
        delimiter = "\t" if "\t" in lines[0] else "," if "," in lines[0] else "|"
        reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter=delimiter)
        return tuple(
            {str(key): value for key, value in row.items() if key is not None}
            for row in reader
        )

    rows: list[Mapping[str, object]] = []
    for line in lines:
        match = _CODE_LINE_RE.match(line)
        if match is None:
            continue
        rows.append(
            {"code": match.group("code"), "description": match.group("display")}
        )
    return tuple(rows)


def _parse_index_text(text: str) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        matches = tuple(_CODE_RE.finditer(line))
        if not matches:
            continue
        term = _CODE_RE.sub(" ", line)
        term = re.sub(r"[|,;]+", " ", term)
        term = " ".join(term.split()).strip(" .:-")
        if not term:
            term = line.strip()
        for match in matches:
            rows.append({"code": match.group(1), "term": term})
    return tuple(rows)


def _find_local_file(root: Path, *, kind: str) -> Path | None:
    candidates = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in _SUPPORTED_SUFFIXES
    )
    named = [
        candidate for candidate in candidates if _filename_kind(candidate.name) == kind
    ]
    if named:
        return named[0]
    if kind == "tabular":
        fallback = [
            candidate
            for candidate in candidates
            if not _looks_like_index_filename(candidate.name)
        ]
        return fallback[0] if fallback else None
    return None


def _find_archive_member(names: Iterable[str], *, kind: str) -> str | None:
    candidates = sorted(
        name for name in names if Path(name).suffix.casefold() in _SUPPORTED_SUFFIXES
    )
    named = [name for name in candidates if _filename_kind(Path(name).name) == kind]
    if named:
        return named[0]
    if kind == "tabular":
        fallback = [name for name in candidates if not _looks_like_index_filename(name)]
        return fallback[0] if fallback else None
    return None


def _filename_kind(name: str) -> str | None:
    normalized = name.casefold().replace("-", "_")
    if _looks_like_index_filename(name):
        return "index"
    if any(
        marker in normalized
        for marker in (
            "tabular",
            "order",
            "code_description",
            "codes",
            "icd10cm",
        )
    ):
        return "tabular"
    return None


def _looks_like_index_filename(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(marker in normalized for marker in ("index", "alphabetic"))


def _normalize_row(row: Mapping[object, object]) -> dict[str, object]:
    return {_normalize_key(key): value for key, value in row.items() if key is not None}


def _normalize_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _first_value(row: Mapping[object, object], *names: str) -> object | None:
    normalized = {_normalize_key(key): value for key, value in row.items()}
    for name in names:
        key = _normalize_key(name)
        if key in normalized and normalized[key] not in (None, ""):
            return normalized[key]
    return None


def _split_values(value: object, *, split_commas: bool = False) -> tuple[object, ...]:
    if value is None or value == "":
        return ()
    if isinstance(value, Mapping):
        return tuple(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith(("[", "{")):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Sequence) and not isinstance(
            parsed, (str, bytes, bytearray)
        ):
            return tuple(parsed)
    separators = r"[|;>\n]" + (r"|," if split_commas else "")
    return tuple(part.strip() for part in re.split(separators, text) if part.strip())


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _unique_text_values(values: Iterable[object]) -> tuple[str, ...]:
    result: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in result:
            result.append(text)
    return tuple(result)


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser() if value is not None else None
