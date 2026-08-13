"""Load a caller-supplied LOINC release for deterministic local matching.

LOINC is freely available under the LOINC terms of use.  This adapter reads a
release supplied by the caller and never downloads, embeds, or redistributes
LOINC rows.  It understands the standard ``Loinc.csv``, ``Part.csv``,
``AnswerList.csv``, and ``LoincAnswerListLink.csv`` files, along with small
CSV/TSV/JSONL projections suitable for synthetic offline fixtures.

The six LOINC axes are retained on every :class:`ConceptMatch` in metadata:
component, property, time, system, scale, and method.  Part filters are
available for callers that need to distinguish otherwise similar lab tests.
"""

from __future__ import annotations

import builtins
import csv
import io
import json
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
    "LOINC_LICENSE_NOTE",
    "LOINC_PART_FIELDS",
    "LOINC_SYSTEM_URI",
    "LOINCLoader",
    "LOINCVocabularyLoader",
    "LoincAnswer",
    "LoincAnswerList",
    "LoincLoader",
    "LoincLoaderError",
    "LoincVocabularyLoader",
    "LoincParts",
]

LOINC_SYSTEM_URI = "http://loinc.org"
LOINC_PART_FIELDS = ("component", "property", "time", "system", "scale", "method")
LOINC_LICENSE_NOTE = (
    "LOINC is freely available under the LOINC terms of use. The caller must "
    "supply and use release files under those terms; OpenMed does not bundle "
    "or redistribute LOINC data."
)

_TABLE_NAMES = {
    "loinc.csv",
    "loinctable.csv",
    "loinctablecore.csv",
    "loinc.tsv",
    "loinctable.tsv",
    "loinctablecore.tsv",
    "loinc.json",
    "loinc.jsonl",
}
_PART_NAMES = {
    "part.csv",
    "parts.csv",
    "loincpart.csv",
    "loincparts.csv",
    "part.tsv",
    "parts.tsv",
    "part.json",
    "part.jsonl",
}
_ANSWER_NAMES = {
    "answerlist.csv",
    "answerlists.csv",
    "answerlist.tsv",
    "answerlists.tsv",
    "answerlist.json",
    "answerlist.jsonl",
}
_LINK_NAMES = {
    "loincanswerlistlink.csv",
    "loincanswerlistlinks.csv",
    "answerlistlink.csv",
    "answerlistlinks.csv",
    "loincanswerlistlink.tsv",
    "answerlistlink.tsv",
    "loincanswerlistlink.json",
    "loincanswerlistlink.jsonl",
}

_FIELD_ALIASES = {
    "code": (
        "loincnum",
        "loincnumber",
        "loinccode",
        "loincid",
        "code",
        "id",
    ),
    "long_name": (
        "longcommonname",
        "longname",
        "preferredterm",
        "display",
        "displayname",
        "name",
        "term",
    ),
    "short_name": ("shortname", "shortcommonname", "shortterm"),
    "component": ("component", "componentname", "componentid"),
    "property": ("property", "propertyname", "propertyid"),
    "time": ("timeaspct", "timeaspect", "time", "timeid"),
    "time_aspect": ("timeaspct", "timeaspect", "time", "timeid"),
    "system": ("system", "systemname", "systemid", "specimensystem"),
    "scale": ("scaletyp", "scaletype", "scale", "scaleid"),
    "method": ("methodtyp", "methodtype", "method", "methodid"),
    "part_number": ("partnumber", "partid", "partcode", "loincpartnumber"),
    "part_type": ("parttypename", "parttype", "axistype"),
    "part_name": ("partname", "partdisplayname", "displayname", "name"),
    "answer_list_id": (
        "answerlistid",
        "answerlistids",
        "answerlist",
        "answerlistnumber",
        "answerlistcode",
        "id",
    ),
    "answer_list_name": ("answerlistname", "listname"),
    "answer_list_type": ("answerlisttype", "listtype"),
    "answer_code": (
        "answercode",
        "answerid",
        "answerlistitemid",
        "answeritemcode",
        "code",
    ),
    "answer_display": (
        "answerdisplay",
        "answerdisplayname",
        "answerlistitem",
        "answerlistitemdisplay",
        "answername",
        "answertext",
        "display",
        "name",
    ),
}

_PART_KEY_ALIASES = {
    "component": "component",
    "property": "property",
    "time": "time",
    "timeaspect": "time",
    "timeaspct": "time",
    "system": "system",
    "scale": "scale",
    "scaletype": "scale",
    "scaletyp": "scale",
    "method": "method",
    "methodtype": "method",
    "methodtyp": "method",
}


class LoincLoaderError(VocabLoaderError):
    """Raised when a caller-supplied LOINC release is invalid or unreadable."""


@dataclass(frozen=True)
class LoincParts:
    """The six axes that define a LOINC observation concept.

    ``time`` is the LOINC time aspect axis (the source column is usually
    ``TIME_ASPCT``).  :attr:`time_aspect` is provided as a descriptive alias.
    """

    component: str = ""
    property: str = ""
    time: str = ""
    system: str = ""
    scale: str = ""
    method: str = ""

    @builtins.property
    def time_aspect(self) -> str:
        """Return the time axis using the source column's descriptive name."""

        return self.time

    def as_dict(self) -> dict[str, str]:
        """Return the canonical six-axis mapping."""

        return {
            field_name: getattr(self, field_name) for field_name in LOINC_PART_FIELDS
        }


@dataclass(frozen=True)
class LoincAnswer:
    """One coded answer associated with an ordinal or nominal answer list."""

    code: str
    display: str

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-ready answer mapping."""

        return {"code": self.code, "display": self.display}


@dataclass(frozen=True)
class LoincAnswerList:
    """An answer list and its optional coded answer values."""

    identifier: str
    name: str = ""
    list_type: str = ""
    answers: tuple[LoincAnswer, ...] = ()

    @property
    def answer_list_id(self) -> str:
        """Return the stable answer-list identifier."""

        return self.identifier

    @property
    def type(self) -> str:
        """Return the answer-list type."""

        return self.list_type

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-ready answer-list mapping."""

        return {
            "id": self.identifier,
            "name": self.name,
            "type": self.list_type,
            "answers": tuple(answer.as_dict() for answer in self.answers),
        }


@dataclass
class _LoincRecord:
    code: str
    long_common_name: str
    parts: LoincParts
    order: int
    aliases: list[str] = field(default_factory=list)
    answer_list_ids: set[str] = field(default_factory=set)

    def add_alias(self, alias: object) -> None:
        value = _text(alias)
        if value and value not in self.aliases:
            self.aliases.append(value)


@dataclass
class _AnswerListBuilder:
    identifier: str
    name: str = ""
    list_type: str = ""
    answers: list[LoincAnswer] = field(default_factory=list)

    def add_answer(self, code: object, display: object) -> None:
        answer_code = _text(code)
        answer_display = _text(display) or answer_code
        if not answer_code:
            return
        answer = LoincAnswer(answer_code, answer_display)
        if answer not in self.answers:
            self.answers.append(answer)


@dataclass(frozen=True)
class _ParsedRelease:
    records: tuple[_LoincRecord, ...]
    answer_lists: Mapping[str, LoincAnswerList]
    links: Mapping[str, tuple[str, ...]]


class LoincLoader:
    """Load a user-supplied LOINC release for offline lexical grounding.

    Args:
        path: Directory, archive, or projection containing the LOINC table.
        release_path: Keyword alias for ``path``.
        source_path: Keyword alias for ``path``.
        table_path: Explicit LOINC table path when the release is split across
            caller-owned files.
        parts_path: Optional ``Part.csv`` path used to resolve part identifiers.
        answer_list_path: Optional ``AnswerList.csv`` path.
        answer_list_link_path: Optional LOINC-to-answer-list link path.

    The loader implements :class:`VocabularyLoader`: its source is explicitly
    redistributable, but no LOINC content is bundled.  ``load`` returns terms
    for the shared matcher; ``resolve`` adds optional part filtering.
    """

    system_uri = LOINC_SYSTEM_URI
    redistributable = True
    restricted_license = False
    license_note = LOINC_LICENSE_NOTE

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        release_path: str | Path | None = None,
        source_path: str | Path | None = None,
        table_path: str | Path | None = None,
        parts_path: str | Path | None = None,
        answer_list_path: str | Path | None = None,
        answer_lists_path: str | Path | None = None,
        answer_list_link_path: str | Path | None = None,
        answer_list_links_path: str | Path | None = None,
    ) -> None:
        supplied = [value for value in (path, release_path, source_path) if value]
        if len(supplied) > 1:
            raise TypeError("provide only one of path, release_path, or source_path")
        explicit_answer_paths = [
            value for value in (answer_list_path, answer_lists_path) if value
        ]
        if len(explicit_answer_paths) > 1:
            raise TypeError("provide only one of answer_list_path or answer_lists_path")
        explicit_link_paths = [
            value for value in (answer_list_link_path, answer_list_links_path) if value
        ]
        if len(explicit_link_paths) > 1:
            raise TypeError(
                "provide only one of answer_list_link_path or answer_list_links_path"
            )
        if not supplied and table_path is None:
            raise TypeError(
                "LoincLoader requires a user-supplied release or table path; "
                "LOINC data is never downloaded or bundled"
            )

        self.path = Path(
            supplied[0] if supplied else cast(Any, table_path)
        ).expanduser()
        self.table_path = _optional_path(table_path)
        self.parts_path = _optional_path(parts_path)
        self.answer_list_path = _optional_path(
            explicit_answer_paths[0] if explicit_answer_paths else None
        )
        self.answer_list_link_path = _optional_path(
            explicit_link_paths[0] if explicit_link_paths else None
        )
        self._release: _ParsedRelease | None = None
        self._terms: VocabularyTerms | None = None
        self._matcher: LexicalMatcher | None = None

    @property
    def records(self) -> tuple[_LoincRecord, ...]:
        """Return parsed concepts in deterministic source order."""

        return self._parsed_release().records

    @property
    def answer_list_ids(self) -> tuple[str, ...]:
        """Return answer-list identifiers available in the supplied release."""

        return tuple(sorted(self._parsed_release().answer_lists))

    def load(self) -> VocabularyTerms:
        """Return long-common-name aliases mapped to LOINC lexical concepts."""

        if self._terms is not None:
            return self._terms

        terms: dict[str, list[LexicalConcept]] = defaultdict(list)
        seen: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for record in self.records:
            concept = self._lexical_concept(record)
            for alias in record.aliases or [record.long_common_name]:
                if concept.key in seen[alias]:
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
        query: str | None = None,
        *,
        parts: Mapping[str, object] | LoincParts | None = None,
        component: object | None = None,
        property: object | None = None,
        time: object | None = None,
        time_aspect: object | None = None,
        system: object | None = None,
        scale: object | None = None,
        method: object | None = None,
        limit: int | None = None,
    ) -> tuple[ConceptMatch, ...]:
        """Resolve a lab-test name, optionally constrained by LOINC axes.

        A ``None`` query performs a part-only lookup and returns all concepts
        satisfying the requested axes.  Axis values are compared with the same
        Unicode/case/punctuation normalization used by the lexical matcher.
        ``time`` and ``time_aspect`` are interchangeable names for the LOINC
        time axis.
        """

        if query is not None and not isinstance(query, str):
            raise TypeError("query must be a string or None")
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")
        filters = _combine_part_filters(
            parts,
            component=component,
            property=property,
            time=time,
            time_aspect=time_aspect,
            system=system,
            scale=scale,
            method=method,
        )

        if query is None:
            matches = [
                self._concept_match(
                    record,
                    matched_term=record.long_common_name,
                    score=1.0,
                    match_type="exact",
                )
                for record in self.records
                if _matches_parts(record.parts, filters)
            ]
            return tuple(_deduplicate_matches(matches)[:limit])

        if not normalize_term(query):
            return ()
        matcher = self._lexical_matcher()
        matches = matcher.lookup(query)
        filtered = [
            match
            for match in matches
            if _matches_parts_metadata(match.metadata, filters)
        ]
        return tuple(filtered[:limit] if limit is not None else filtered)

    def lookup(
        self,
        query: str | None = None,
        **kwargs: object,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`, matching the shared matcher vocabulary."""

        return self.resolve(query, **kwargs)

    def match(
        self,
        query: str | None = None,
        **kwargs: object,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`resolve`."""

        return self.resolve(query, **kwargs)

    def lookup_by_parts(
        self,
        parts: Mapping[str, object] | LoincParts | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        **axes: object,
    ) -> tuple[ConceptMatch, ...]:
        """Return concepts constrained by one or more LOINC axes."""

        if parts is None:
            return self.resolve(query, parts=axes, limit=limit)
        if isinstance(parts, LoincParts):
            if axes:
                raise TypeError("axis keyword filters cannot accompany LoincParts")
            return self.resolve(query, parts=parts, limit=limit)
        if not isinstance(parts, Mapping):
            raise TypeError("parts must be a mapping, LoincParts, or None")
        combined = dict(parts)
        combined.update(axes)
        return self.resolve(query, parts=combined, limit=limit)

    def lookup_parts(
        self,
        parts: Mapping[str, object] | LoincParts | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        **axes: object,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup_by_parts`."""

        return self.lookup_by_parts(parts, query=query, limit=limit, **axes)

    def resolve_by_parts(
        self,
        parts: Mapping[str, object] | LoincParts | None = None,
        *,
        query: str | None = None,
        limit: int | None = None,
        **axes: object,
    ) -> tuple[ConceptMatch, ...]:
        """Alias for :meth:`lookup_by_parts`."""

        return self.lookup_by_parts(parts, query=query, limit=limit, **axes)

    def resolve_one(self, query: str, **kwargs: object) -> ConceptMatch | None:
        """Return the best match for ``query`` or ``None`` when unresolved."""

        matches = self.resolve(query, limit=1, **kwargs)
        return matches[0] if matches else None

    def answer_lists_for(self, loinc_code: str) -> tuple[LoincAnswerList, ...]:
        """Return answer lists linked to a LOINC code, if supplied."""

        if not isinstance(loinc_code, str) or not loinc_code.strip():
            raise ValueError("loinc_code must be a non-empty string")
        code = loinc_code.strip()
        release = self._parsed_release()
        identifiers = _linked_answer_ids(release, code)
        return tuple(
            release.answer_lists[identifier]
            for identifier in identifiers
            if identifier in release.answer_lists
        )

    def answer_list_for(self, loinc_code: str) -> tuple[LoincAnswerList, ...]:
        """Alias for :meth:`answer_lists_for`."""

        return self.answer_lists_for(loinc_code)

    def answers_for(self, loinc_code: str) -> tuple[LoincAnswer, ...]:
        """Return flattened coded answers linked to a LOINC code."""

        answers: list[LoincAnswer] = []
        for answer_list in self.answer_lists_for(loinc_code):
            for answer in answer_list.answers:
                if answer not in answers:
                    answers.append(answer)
        return tuple(answers)

    def _parsed_release(self) -> _ParsedRelease:
        if self._release is None:
            self._release = _parse_release(
                self.path,
                table_path=self.table_path,
                parts_path=self.parts_path,
                answer_list_path=self.answer_list_path,
                answer_list_link_path=self.answer_list_link_path,
            )
        return self._release

    def _lexical_matcher(self) -> LexicalMatcher:
        if self._matcher is None:
            self._matcher = LexicalMatcher(self.load(), system_uri=self.system_uri)
        return self._matcher

    def _lexical_concept(self, record: _LoincRecord) -> LexicalConcept:
        return LexicalConcept(
            system_uri=self.system_uri,
            code=record.code,
            display=record.long_common_name,
            metadata=_record_metadata(record, self._parsed_release()),
        )

    def _concept_match(
        self,
        record: _LoincRecord,
        *,
        matched_term: str,
        score: float,
        match_type: str,
    ) -> ConceptMatch:
        return ConceptMatch(
            system_uri=self.system_uri,
            code=record.code,
            display=record.long_common_name,
            score=score,
            match_type=cast(Any, match_type),
            matched_term=matched_term,
            metadata=_record_metadata(record, self._parsed_release()),
        )


# Keep both spellings available: the shorter one follows the existing loader
# API and the acronym spelling is convenient at terminology integration points.
LOINCLoader = LoincLoader
LOINCVocabularyLoader = LoincLoader
LoincVocabularyLoader = LoincLoader


def _parse_release(
    path: Path,
    *,
    table_path: Path | None,
    parts_path: Path | None,
    answer_list_path: Path | None,
    answer_list_link_path: Path | None,
) -> _ParsedRelease:
    if not path.exists():
        raise LoincLoaderError(f"LOINC release path does not exist: {path}")

    if path.is_dir():
        table_source = table_path or _find_local_file(
            path, _TABLE_NAMES, _looks_like_table
        )
        if table_source is None:
            raise LoincLoaderError(f"LOINC release directory has no table file: {path}")
        table_rows = _read_rows_from_path(table_source)
        part_rows = _read_optional_local_rows(
            parts_path or _find_local_file(path, _PART_NAMES, _looks_like_parts)
        )
        answer_rows = _read_optional_local_rows(
            answer_list_path
            or _find_local_file(path, _ANSWER_NAMES, _looks_like_answers)
        )
        link_rows = _read_optional_local_rows(
            answer_list_link_path
            or _find_local_file(path, _LINK_NAMES, _looks_like_links)
        )
        return _build_release(table_rows, part_rows, answer_rows, link_rows)

    if zipfile.is_zipfile(path):
        return _parse_archive(path)

    table_rows = _read_rows_from_path(table_path or path)
    sibling_root = path.parent
    part_rows = _read_optional_local_rows(
        parts_path or _find_local_file(sibling_root, _PART_NAMES, _looks_like_parts)
    )
    answer_rows = _read_optional_local_rows(
        answer_list_path
        or _find_local_file(sibling_root, _ANSWER_NAMES, _looks_like_answers)
    )
    link_rows = _read_optional_local_rows(
        answer_list_link_path
        or _find_local_file(sibling_root, _LINK_NAMES, _looks_like_links)
    )
    return _build_release(table_rows, part_rows, answer_rows, link_rows)


def _parse_archive(path: Path) -> _ParsedRelease:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise LoincLoaderError(f"Unable to read LOINC archive {path}: {exc}") from exc

    with archive:
        names = archive.namelist()
        table_name = _find_archive_member(
            names, _TABLE_NAMES, _looks_like_table_bytes, archive
        )
        if table_name is None:
            raise LoincLoaderError(f"LOINC archive has no table file: {path}")
        table_rows = _read_rows_from_bytes(archive.read(table_name), table_name)
        part_name = _find_archive_member(
            names, _PART_NAMES, _looks_like_parts_bytes, archive
        )
        answer_name = _find_archive_member(
            names, _ANSWER_NAMES, _looks_like_answers_bytes, archive
        )
        link_name = _find_archive_member(
            names, _LINK_NAMES, _looks_like_links_bytes, archive
        )
        part_rows = (
            _read_rows_from_bytes(archive.read(part_name), part_name)
            if part_name
            else ()
        )
        answer_rows = (
            _read_rows_from_bytes(archive.read(answer_name), answer_name)
            if answer_name
            else ()
        )
        link_rows = (
            _read_rows_from_bytes(archive.read(link_name), link_name)
            if link_name
            else ()
        )
        return _build_release(table_rows, part_rows, answer_rows, link_rows)


def _build_release(
    table_rows: Iterable[Mapping[str, object]],
    part_rows: Iterable[Mapping[str, object]],
    answer_rows: Iterable[Mapping[str, object]],
    link_rows: Iterable[Mapping[str, object]],
) -> _ParsedRelease:
    part_labels = _parse_part_labels(part_rows)
    records: list[_LoincRecord] = []
    by_code: dict[str, _LoincRecord] = {}
    inline_links: dict[str, set[str]] = defaultdict(set)

    for row in table_rows:
        code = _first_value(row, "code")
        long_name = _first_value(row, "long_name")
        if not code or not long_name:
            continue
        normalized_code = _text(code)
        normalized_name = _text(long_name)
        if not normalized_code or not normalized_name:
            continue
        parts = _parts_from_row(row, part_labels)
        record = by_code.get(normalized_code)
        if record is None:
            record = _LoincRecord(
                code=normalized_code,
                long_common_name=normalized_name,
                parts=parts,
                order=len(records),
            )
            by_code[normalized_code] = record
            records.append(record)
        elif record.long_common_name != normalized_name:
            # A projection can contain duplicate aliases for one code. Keep
            # the first display/axis row, but retain the extra name below.
            record.add_alias(normalized_name)
        record.add_alias(normalized_name)
        record.add_alias(_first_value(row, "short_name"))
        for identifier in _split_values(_first_value(row, "answer_list_id")):
            record.answer_list_ids.add(identifier)
            inline_links[normalized_code].add(identifier)

        for identifier in _answer_ids_from_nested_row(row):
            record.answer_list_ids.add(identifier)
            inline_links[normalized_code].add(identifier)

    if not records:
        raise LoincLoaderError(
            "No LOINC concepts could be read from the supplied release"
        )

    answer_builders = _parse_answer_rows(answer_rows)
    file_links = _parse_link_rows(link_rows)
    for code, identifiers in file_links.items():
        inline_links[code].update(identifiers)
    for record in records:
        record.answer_list_ids.update(inline_links.get(record.code, ()))

    answer_lists = {
        identifier: LoincAnswerList(
            identifier=builder.identifier,
            name=builder.name,
            list_type=builder.list_type,
            answers=tuple(builder.answers),
        )
        for identifier, builder in answer_builders.items()
    }
    links = {
        record.code: tuple(sorted(record.answer_list_ids))
        for record in records
        if record.answer_list_ids
    }
    return _ParsedRelease(tuple(records), answer_lists, links)


def _parse_part_labels(rows: Iterable[Mapping[str, object]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in rows:
        identifier = _text(_first_value(row, "part_number"))
        name = _text(_first_value(row, "part_name"))
        if identifier and name:
            labels[identifier.casefold()] = name
    return labels


def _parse_answer_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, _AnswerListBuilder]:
    builders: dict[str, _AnswerListBuilder] = {}
    for row in rows:
        identifier = _text(_first_value(row, "answer_list_id"))
        if not identifier:
            continue
        builder = builders.setdefault(identifier, _AnswerListBuilder(identifier))
        builder.name = builder.name or _text(_first_value(row, "answer_list_name"))
        builder.list_type = builder.list_type or _text(
            _first_value(row, "answer_list_type")
        )
        answer_code = _first_value(row, "answer_code")
        answer_display = _first_value(row, "answer_display")
        if answer_code:
            builder.add_answer(answer_code, answer_display or answer_code)
        nested = _row_value(row, "answers")
        for answer in _nested_answers(nested):
            builder.add_answer(answer.get("code"), answer.get("display"))
    return builders


def _parse_link_rows(
    rows: Iterable[Mapping[str, object]],
) -> dict[str, set[str]]:
    links: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        code = _text(_first_value(row, "code"))
        identifier = _text(_first_value(row, "answer_list_id"))
        if code and identifier:
            links[code].add(identifier)
    return links


def _parts_from_row(row: Mapping[str, object], labels: Mapping[str, str]) -> LoincParts:
    nested = _row_value(row, "parts")
    nested_parts = _nested_mapping(nested)
    values: dict[str, str] = {}
    for field_name in LOINC_PART_FIELDS:
        value = _first_value(row, field_name)
        if not value:
            value = _first_value(nested_parts, field_name)
        if field_name == "time" and not value:
            value = _first_value(nested_parts, "time_aspect")
        values[field_name] = _part_label(value, labels)
    return LoincParts(**values)


def _record_metadata(
    record: _LoincRecord, release: _ParsedRelease
) -> dict[str, object]:
    parts = record.parts.as_dict()
    metadata: dict[str, object] = {
        "loinc_number": record.code,
        "parts": parts,
        "loinc_parts": parts,
        **parts,
        "time_aspect": record.parts.time_aspect,
    }
    answer_lists = tuple(
        release.answer_lists[identifier].as_dict()
        for identifier in sorted(record.answer_list_ids)
        if identifier in release.answer_lists
    )
    if record.answer_list_ids:
        identifiers = tuple(sorted(record.answer_list_ids))
        metadata["answer_list_ids"] = identifiers
        metadata["answer_lists"] = answer_lists
        metadata["answers"] = tuple(
            answer for answer_list in answer_lists for answer in answer_list["answers"]
        )
        if len(answer_lists) == 1:
            metadata["answer_list"] = answer_lists[0]
    return metadata


def _linked_answer_ids(release: _ParsedRelease, code: str) -> tuple[str, ...]:
    return release.links.get(code, ())


def _matches_parts(record: LoincParts, filters: Mapping[str, str]) -> bool:
    return all(
        normalize_term(getattr(record, field_name)) == expected
        for field_name, expected in filters.items()
    )


def _matches_parts_metadata(
    metadata: Mapping[str, object], filters: Mapping[str, str]
) -> bool:
    parts = metadata.get("parts", {})
    if not isinstance(parts, Mapping):
        return False
    return all(
        normalize_term(parts.get(field_name, "")) == expected
        for field_name, expected in filters.items()
    )


def _deduplicate_matches(matches: Iterable[ConceptMatch]) -> list[ConceptMatch]:
    found: dict[tuple[str, str], ConceptMatch] = {}
    for match in matches:
        found.setdefault(match.key, match)
    return list(found.values())


def _combine_part_filters(
    parts: Mapping[str, object] | LoincParts | None,
    **explicit: object | None,
) -> dict[str, str]:
    combined: dict[str, str] = {}
    if parts is not None:
        if isinstance(parts, LoincParts):
            source: Mapping[str, object] = parts.as_dict()
        elif isinstance(parts, Mapping):
            source = parts
        else:
            raise TypeError("parts must be a mapping, LoincParts, or None")
        for raw_key, value in source.items():
            field_name = _PART_KEY_ALIASES.get(_field_name(raw_key))
            if field_name is None:
                raise ValueError(f"unknown LOINC part axis: {raw_key!r}")
            if value is not None:
                combined[field_name] = normalize_term(str(value))

    if explicit.get("time_aspect") is not None:
        if explicit.get("time") is not None and normalize_term(
            str(explicit["time"])
        ) != normalize_term(str(explicit["time_aspect"])):
            raise ValueError("time and time_aspect filters disagree")
        explicit["time"] = explicit["time_aspect"]
    for field_name in LOINC_PART_FIELDS:
        value = explicit.get(field_name)
        if value is None:
            continue
        normalized = normalize_term(str(value))
        previous = combined.get(field_name)
        if previous is not None and previous != normalized:
            raise ValueError(f"conflicting filter for LOINC {field_name}")
        combined[field_name] = normalized
    return combined


def _parse_rows(text: str, source: str) -> tuple[Mapping[str, object], ...]:
    suffix = Path(source).suffix.casefold()
    if suffix in {".json", ".jsonl"}:
        if suffix == ".jsonl":
            rows: list[Mapping[str, object]] = []
            for line in text.splitlines():
                if line.strip():
                    parsed = _json_value(line, source)
                    if isinstance(parsed, Mapping):
                        rows.append(parsed)
            return tuple(rows)
        parsed = _json_value(text, source)
        if isinstance(parsed, Mapping):
            nested = parsed.get("rows", parsed.get("concepts", parsed.get("terms")))
            if isinstance(nested, Sequence) and not isinstance(
                nested, (str, bytes, bytearray)
            ):
                return tuple(row for row in nested if isinstance(row, Mapping))
            return (parsed,)
        if isinstance(parsed, Sequence) and not isinstance(
            parsed, (str, bytes, bytearray)
        ):
            return tuple(row for row in parsed if isinstance(row, Mapping))
        return ()

    delimiter = "\t" if suffix == ".tsv" else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    return tuple(
        {str(key): value for key, value in row.items() if key is not None}
        for row in reader
    )


def _read_rows_from_path(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        return _read_rows_from_bytes(path.read_bytes(), str(path))
    except OSError as exc:
        raise LoincLoaderError(f"Unable to read LOINC file {path}: {exc}") from exc


def _read_rows_from_bytes(data: bytes, source: str) -> tuple[Mapping[str, object], ...]:
    text = data.decode("utf-8-sig", errors="replace")
    try:
        return _parse_rows(text, source)
    except (csv.Error, TypeError, ValueError) as exc:
        raise LoincLoaderError(f"Unable to parse LOINC file {source}: {exc}") from exc


def _read_optional_local_rows(path: Path | None) -> tuple[Mapping[str, object], ...]:
    return _read_rows_from_path(path) if path is not None else ()


def _find_local_file(
    root: Path,
    names: set[str],
    fallback: Any,
) -> Path | None:
    candidates = sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate.name.casefold() in names
    )
    if candidates:
        return candidates[0]
    for candidate in sorted(root.rglob("*")):
        if candidate.is_file() and fallback(candidate):
            return candidate
    return None


def _find_archive_member(
    names: Iterable[str],
    expected_names: set[str],
    fallback: Any,
    archive: zipfile.ZipFile,
) -> str | None:
    candidates = sorted(
        name for name in names if name.rsplit("/", 1)[-1].casefold() in expected_names
    )
    if candidates:
        return candidates[0]
    for name in sorted(names):
        if not name.endswith("/"):
            try:
                if fallback(archive.read(name), name):
                    return name
            except (KeyError, OSError, UnicodeError):
                continue
    return None


def _looks_like_table(path: Path) -> bool:
    try:
        return _looks_like_table_bytes(path.read_bytes()[:8192], str(path))
    except OSError:
        return False


def _looks_like_table_bytes(data: bytes, source: str) -> bool:
    rows = _read_header_rows(data, source)
    if not rows:
        return False
    fields = {_field_name(key) for key in rows[0]}
    return bool(fields & set(_FIELD_ALIASES["code"])) and bool(
        fields & set(_FIELD_ALIASES["long_name"])
    )


def _looks_like_parts(path: Path) -> bool:
    try:
        return _looks_like_parts_bytes(path.read_bytes()[:8192], str(path))
    except OSError:
        return False


def _looks_like_parts_bytes(data: bytes, source: str) -> bool:
    rows = _read_header_rows(data, source)
    if not rows:
        return False
    fields = {_field_name(key) for key in rows[0]}
    return bool(fields & set(_FIELD_ALIASES["part_number"])) and bool(
        fields & set(_FIELD_ALIASES["part_name"])
    )


def _looks_like_answers(path: Path) -> bool:
    try:
        return _looks_like_answers_bytes(path.read_bytes()[:8192], str(path))
    except OSError:
        return False


def _looks_like_answers_bytes(data: bytes, source: str) -> bool:
    rows = _read_header_rows(data, source)
    if not rows:
        return False
    fields = {_field_name(key) for key in rows[0]}
    return bool(fields & set(_FIELD_ALIASES["answer_list_id"])) and bool(
        fields & set(_FIELD_ALIASES["answer_code"])
    )


def _looks_like_links(path: Path) -> bool:
    try:
        return _looks_like_links_bytes(path.read_bytes()[:8192], str(path))
    except OSError:
        return False


def _looks_like_links_bytes(data: bytes, source: str) -> bool:
    rows = _read_header_rows(data, source)
    if not rows:
        return False
    fields = {_field_name(key) for key in rows[0]}
    return bool(fields & set(_FIELD_ALIASES["code"])) and bool(
        fields & set(_FIELD_ALIASES["answer_list_id"])
    )


def _read_header_rows(data: bytes, source: str) -> tuple[Mapping[str, object], ...]:
    try:
        return _parse_rows(data.decode("utf-8-sig", errors="replace"), source)
    except (csv.Error, TypeError, ValueError, json.JSONDecodeError):
        return ()


def _first_value(row: Mapping[str, object], logical_name: str) -> object | None:
    normalized = {_field_name(key): value for key, value in row.items()}
    for key in _FIELD_ALIASES[logical_name]:
        value = normalized.get(key)
        if value not in (None, ""):
            return value
    return None


def _row_value(row: Mapping[str, object], logical_name: str) -> object | None:
    normalized = {_field_name(key): value for key, value in row.items()}
    return normalized.get(_field_name(logical_name))


def _field_name(value: object) -> str:
    return "".join(
        character for character in str(value).casefold() if character.isalnum()
    )


def _text(value: object | None) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _optional_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser() if value is not None else None


def _part_label(value: object | None, labels: Mapping[str, str]) -> str:
    text = _text(value)
    return labels.get(text.casefold(), text)


def _split_values(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        value = _first_value(value, "answer_list_id")
        return _split_values(value)
    if isinstance(value, str) and value.strip().startswith(("[", "{")):
        parsed = _json_value(value, "nested LOINC answer-list identifiers")
        if isinstance(parsed, Mapping):
            parsed = _first_value(parsed, "answer_list_id")
        if parsed is not None:
            return _split_values(parsed)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        values = str(value).replace(";", "|").split("|")
    return tuple(_text(item) for item in values if _text(item))


def _nested_mapping(value: object | None) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        parsed = _json_value(value, "nested LOINC parts")
        return parsed if isinstance(parsed, Mapping) else {}
    return {}


def _nested_answers(value: object | None) -> tuple[Mapping[str, object], ...]:
    if isinstance(value, str) and value.strip():
        value = _json_value(value, "nested LOINC answers")
    if isinstance(value, Mapping):
        value = value.get("answers", value.get("items", ()))
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _answer_ids_from_nested_row(row: Mapping[str, object]) -> tuple[str, ...]:
    nested = _row_value(row, "answer_lists")
    if nested is None:
        nested = _row_value(row, "answers")
    identifiers: list[str] = []
    if isinstance(nested, Mapping):
        nested = nested.get("ids", nested.get("answer_list_ids", ()))
    if isinstance(nested, str) and nested.strip():
        parsed = _json_value(nested, "nested LOINC answer lists")
        nested = parsed if parsed is not None else _split_values(nested)
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes, bytearray)):
        for item in nested:
            if isinstance(item, Mapping):
                value = _first_value(item, "answer_list_id")
            else:
                value = item
            identifier = _text(value)
            if identifier and identifier not in identifiers:
                identifiers.append(identifier)
    return tuple(identifiers)


def _json_value(value: str, source: str) -> object | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise LoincLoaderError(
            f"Invalid JSON in LOINC projection {source}: {exc}"
        ) from exc
