"""Eval-only loader for the credentialed i2b2/VA 2010 concept corpus.

The 2010 release is plain text, one sentence per line, with ``.con``,
``.ast`` and ``.rel`` standoff files that address tokens as ``line:token``
(1-based line, 0-based whitespace token).  Nothing is bundled, downloaded or
cached; source text stays in memory for evaluation only.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from openmed.core.labels import PROBLEM, normalize_label
from openmed.eval.metrics import EvalSpan
from openmed.eval.relation_metrics import EvalRelation, normalize_relation_type

from ._dua import (
    fixture_id,
    require_credentialed_path,
    source_files,
    source_path_hash,
)
from ._task_fixtures import RelationTaskFixture
from .dua_stubs import DUACredentialRequired
from .licenses import license_for

I2B2_2010 = "i2b2-2010"
I2B2_2010_PATH_ENV = "OPENMED_I2B2_2010_PATH"
I2B2_2010_AUTHORITY = "n2c2/DBMI"
I2B2_2010_DUA = "n2c2/DBMI i2b2 2010 data-use agreement"
I2B2_2010_TASK = "concept_assertion_relation"
I2B2_2010_CONCEPT_TYPES: tuple[str, ...] = ("problem", "test", "treatment")
I2B2_2010_ASSERTION_CLASSES: tuple[str, ...] = (
    "present",
    "absent",
    "possible",
    "conditional",
    "hypothetical",
    "associated_with_someone_else",
)
I2B2_2010_RELATION_TYPES: tuple[str, ...] = (
    "TrIP",
    "TrWP",
    "TrCP",
    "TrAP",
    "TrNAP",
    "TeRP",
    "TeCP",
    "PIP",
)

_TOKEN_RE = re.compile(r"\S+")
# A line with its LF, CRLF or CR ending, or a last line without one.
_LINE_RE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$")
_CONCEPT_RE = re.compile(r'^c="(.*)" (\d+):(\d+) (\d+):(\d+)$')
_FIELD_RE = re.compile(r'^([tar])="(.*)"$')
_RELATION_BY_KEY = {name.casefold(): name for name in I2B2_2010_RELATION_TYPES}

_Offsets = tuple[int, int, int, int]


class I2B22010FormatError(ValueError):
    """A malformed i2b2 2010 annotation, located without echoing note text.

    Attributes:
        problem: Short description of what is wrong.
        source_path_hash: Hash of the annotation file's path below the root.
        line_number: 1-based line in that annotation file.
    """

    def __init__(self, problem: str, *, source_path_hash: str, line_number: int):
        self.problem = problem
        self.source_path_hash = source_path_hash
        self.line_number = line_number
        super().__init__(
            f"i2b2 2010 {problem} at line {line_number} of source "
            f"{source_path_hash[:12]}; no corpus rows were loaded"
        )


@dataclass(frozen=True)
class _Line:
    """Where an annotation line sits, for errors that carry no note text."""

    source_path_hash: str
    line_number: int

    def error(self, problem: str) -> I2B22010FormatError:
        return I2B22010FormatError(
            problem,
            source_path_hash=self.source_path_hash,
            line_number=self.line_number,
        )


def load_i2b2_2010(path: str | Path | None = None) -> list[RelationTaskFixture]:
    """Load i2b2/VA 2010 concepts, assertions and relations from a local copy.

    Each ``<stem>.txt`` is paired with ``<stem>.con`` and, when present,
    ``<stem>.ast`` and ``<stem>.rel`` anywhere below the root.  Concepts
    become entities (``problem`` as ``PROBLEM``, ``test`` as ``LAB_TEST``,
    ``treatment`` as ``OTHER``, each keeping ``source_label``), assertions
    land on problem concepts as ``metadata["assertion"]``, and relations
    become :class:`EvalRelation` values with upper-cased types (``TRAP``), as
    the relation scorer expects; ``metadata["source_relation_type"]`` keeps the
    file's spelling.

    Args:
        path: Credentialed directory; defaults to ``OPENMED_I2B2_2010_PATH``.

    Returns:
        One fixture per document, with line bounds as ``sentence_offsets``.

    Raises:
        DUACredentialRequired: The path is missing, inside the repository, or
            holds no annotated documents.
        I2B22010FormatError: An annotation line is malformed or disagrees with
            the text.
    """

    root = require_credentialed_path(
        path,
        dataset=I2B2_2010,
        authority=I2B2_2010_AUTHORITY,
        env_var=I2B2_2010_PATH_ENV,
    )
    documents = _documents(root)
    if not documents:
        raise DUACredentialRequired(
            f"{I2B2_2010_AUTHORITY} credentialed {I2B2_2010} path contains no "
            ".con files with a paired .txt; no corpus rows were loaded"
        )
    return [
        _fixture(stem, files, root=root) for stem, files in sorted(documents.items())
    ]


load_i2b2_2010_fixtures = load_i2b2_2010


def i2b2_2010_suite_metadata() -> dict[str, Any]:
    """Return row-free metadata for the i2b2 2010 concept view."""

    return {
        "access": (
            f"credentialed local path only; pass path=... or set {I2B2_2010_PATH_ENV}"
        ),
        "annotation_format": "line:token standoff (.txt + .con, .ast, .rel)",
        "assertion_classes": list(I2B2_2010_ASSERTION_CLASSES),
        "cache_corpus_rows": False,
        "dataset": I2B2_2010,
        "dua": I2B2_2010_DUA,
        "eval_only": True,
        "license": license_for(I2B2_2010).to_dict(),
        "network_fetch": False,
        "redistribution": "never; read-only from user-supplied credentialed path",
        "relation_types": list(I2B2_2010_RELATION_TYPES),
        "suite": I2B2_2010,
        "task": "relation",
        "task_view": I2B2_2010_TASK,
    }


def _documents(root: Path) -> dict[str, dict[str, Path]]:
    by_suffix: dict[str, dict[str, list[Path]]] = {}
    for suffix in (".txt", ".con", ".ast", ".rel"):
        stems: dict[str, list[Path]] = {}
        for source in source_files(
            root,
            {suffix},
            dataset=I2B2_2010,
            authority=I2B2_2010_AUTHORITY,
            required=False,
        ):
            stems.setdefault(source.stem, []).append(source)
        by_suffix[suffix] = stems
    for suffix in (".con", ".ast", ".rel"):
        for stem, sources in by_suffix[suffix].items():
            if len(sources) > 1:
                raise _duplicate_stem(suffix, sources[1], root)
            if suffix != ".con" and stem not in by_suffix[".con"]:
                raise ValueError(
                    f"i2b2 2010 {suffix} file in source "
                    f"{source_path_hash(sources[0], root)[:12]} has no paired .con "
                    "file; no corpus rows were loaded"
                )
    documents: dict[str, dict[str, Path]] = {}
    for stem, (con_path,) in by_suffix[".con"].items():
        # Only an annotated note must be unique; other .txt files are ignored.
        text_paths = by_suffix[".txt"].get(stem, [])
        if not text_paths:
            raise ValueError(
                "i2b2 2010 .con file in source "
                f"{source_path_hash(con_path, root)[:12]} has no paired .txt "
                "file; no corpus rows were loaded"
            )
        if len(text_paths) > 1:
            raise _duplicate_stem(".txt", text_paths[1], root)
        documents[stem] = {".txt": text_paths[0], ".con": con_path}
        for suffix in (".ast", ".rel"):
            if stem in by_suffix[suffix]:
                documents[stem][suffix] = by_suffix[suffix][stem][0]
    return documents


def _duplicate_stem(suffix: str, source: Path, root: Path) -> ValueError:
    return ValueError(
        f"duplicate i2b2 2010 {suffix} document stem in source "
        f"{source_path_hash(source, root)[:12]}; no corpus rows were loaded"
    )


class _Document:
    """Token positions of one note, for line:token offset conversion."""

    def __init__(self, text: str):
        self.text = text
        self.lines: list[list[tuple[int, int]]] = []
        self.sentence_offsets: list[tuple[int, int]] = []
        for match in _LINE_RE.finditer(text):
            start = match.start()
            body = match.group().rstrip("\r\n")
            self.lines.append(
                [(start + m.start(), start + m.end()) for m in _TOKEN_RE.finditer(body)]
            )
            self.sentence_offsets.append((start, start + len(body)))

    def span(self, offsets: _Offsets, supplied: str, where: _Line) -> tuple[int, int]:
        """Return the character span of a line:token range, checking ``c=``."""

        start_line, start_token, end_line, end_token = offsets
        line_count = len(self.lines)
        if not (1 <= start_line <= line_count and 1 <= end_line <= line_count):
            raise where.error("line index out of range")
        if (end_line, end_token) < (start_line, start_token):
            raise where.error("concept ends before it starts")
        first = self.lines[start_line - 1]
        last = self.lines[end_line - 1]
        if not (0 <= start_token < len(first) and 0 <= end_token < len(last)):
            raise where.error("token index out of range")
        covered = [
            token
            for line_index in range(start_line - 1, end_line)
            for token_index, token in enumerate(self.lines[line_index])
            if (line_index > start_line - 1 or token_index >= start_token)
            and (line_index < end_line - 1 or token_index <= end_token)
        ]
        covered_text = " ".join(self.text[start:end] for start, end in covered)
        if _normalize(covered_text) != _normalize(supplied):
            raise where.error("concept text does not match its tokens")
        return first[start_token][0], last[end_token][1]


def _fixture(stem: str, files: dict[str, Path], *, root: Path) -> RelationTaskFixture:
    document = _Document(_read_text(files[".txt"], root))
    resolved_id = fixture_id(I2B2_2010, files[".txt"], root, stem)

    entities: dict[str, EvalSpan] = {}
    concept_ids: dict[_Offsets, str] = {}
    for where, fields in _records(files[".con"], root, field_count=2):
        offsets, supplied = _concept(fields[0], where)
        concept_type = _field(fields[1], "t", where).casefold()
        if concept_type not in I2B2_2010_CONCEPT_TYPES:
            raise where.error("unknown concept type")
        start, end = document.span(offsets, supplied, where)
        if offsets in concept_ids:
            continue
        concept_ids[offsets] = f"T{len(entities) + 1}"
        entities[concept_ids[offsets]] = EvalSpan(
            start=start,
            end=end,
            label=normalize_label(concept_type),
            text=document.text[start:end],
            metadata={"source_label": concept_type},
        )

    if ".ast" in files:
        for where, fields in _records(files[".ast"], root, field_count=3):
            offsets, supplied = _concept(fields[0], where)
            document.span(offsets, supplied, where)
            if _field(fields[1], "t", where).casefold() != "problem":
                raise where.error("assertion on a non-problem concept")
            assertion = _field(fields[2], "a", where).casefold()
            if assertion not in I2B2_2010_ASSERTION_CLASSES:
                raise where.error("unknown assertion class")
            entity_id = concept_ids.get(offsets)
            if entity_id is None or entities[entity_id].label != PROBLEM:
                raise where.error("assertion on a concept missing from the .con file")
            span = entities[entity_id]
            entities[entity_id] = replace(
                span, metadata={**span.metadata, "assertion": assertion}
            )

    relations: list[EvalRelation] = []
    if ".rel" in files:
        for where, fields in _records(files[".rel"], root, field_count=3):
            head_offsets, head_text = _concept(fields[0], where)
            source_type = _field(fields[1], "r", where)
            tail_offsets, tail_text = _concept(fields[2], where)
            relation_type = _RELATION_BY_KEY.get(source_type.casefold())
            if relation_type is None:
                raise where.error("unknown relation type")
            document.span(head_offsets, head_text, where)
            document.span(tail_offsets, tail_text, where)
            head_id = concept_ids.get(head_offsets)
            tail_id = concept_ids.get(tail_offsets)
            if head_id is None or tail_id is None:
                raise where.error("relation on a concept missing from the .con file")
            relations.append(
                EvalRelation(
                    relation_type=normalize_relation_type(relation_type),
                    head=entities[head_id],
                    tail=entities[tail_id],
                    relation_id=f"R{len(relations) + 1}",
                    fixture_id=resolved_id,
                    metadata={"source_relation_type": source_type},
                )
            )

    return RelationTaskFixture(
        fixture_id=resolved_id,
        text=document.text,
        entities=entities,
        gold_relations=tuple(relations),
        metadata={
            **i2b2_2010_suite_metadata(),
            "sentence_offsets": [list(bounds) for bounds in document.sentence_offsets],
            "source_path_hash": source_path_hash(files[".txt"], root),
        },
    )


def _records(
    path: Path, root: Path, *, field_count: int
) -> Iterator[tuple[_Line, list[str]]]:
    path_hash = source_path_hash(path, root)
    for line_number, line in enumerate(_read_text(path, root).splitlines(), start=1):
        if not line.strip():
            continue
        where = _Line(path_hash, line_number)
        fields = line.strip().split("||")
        if len(fields) != field_count:
            raise where.error(f"malformed {path.suffix} line")
        yield where, fields


def _read_text(path: Path, root: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        raise ValueError(
            f"i2b2 2010 source {source_path_hash(path, root)[:12]} could not be "
            "read as UTF-8 text; no corpus rows were loaded"
        ) from None


def _concept(value: str, where: _Line) -> tuple[_Offsets, str]:
    match = _CONCEPT_RE.match(value)
    if match is None:
        raise where.error("malformed concept reference")
    supplied, *numbers = match.groups()
    start_line, start_token, end_line, end_token = (int(number) for number in numbers)
    return (start_line, start_token, end_line, end_token), supplied


def _field(value: str, name: str, where: _Line) -> str:
    match = _FIELD_RE.match(value)
    if match is None or match.group(1) != name:
        raise where.error(f"malformed {name}= field")
    return match.group(2)


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


__all__ = [
    "I2B2_2010",
    "I2B2_2010_ASSERTION_CLASSES",
    "I2B2_2010_AUTHORITY",
    "I2B2_2010_CONCEPT_TYPES",
    "I2B2_2010_DUA",
    "I2B2_2010_PATH_ENV",
    "I2B2_2010_RELATION_TYPES",
    "I2B2_2010_TASK",
    "I2B22010FormatError",
    "i2b2_2010_suite_metadata",
    "load_i2b2_2010",
    "load_i2b2_2010_fixtures",
]
