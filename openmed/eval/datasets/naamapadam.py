"""License-aware loaders for user-supplied Naamapadam/IndicGLUE NER packs.

Naamapadam follows the CoNLL PER/LOC/ORG scheme. This loader accepts the
official token/tag JSON shape (``words`` + ``ner`` or ``tokens`` +
``ner_tags``) and CoNLL-style token/tag files. It reconstructs source text with
code-point offsets and never downloads or caches benchmark records.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import (
    CANONICAL_LABELS,
    LOCATION,
    ORGANIZATION,
    OTHER,
    PERSON,
    normalize_label,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.datasets.licenses import license_for
from openmed.eval.datasets.multilingual_ner import (
    NAAMAPADAM,
    LabelMappingResult,
    MultilingualNerLoadResult,
    MultilingualNerRecord,
    MultilingualNerSpan,
)
from openmed.eval.harness import BenchmarkFixture

NAAMAPADAM_PATH_ENV = "OPENMED_NAAMAPADAM_PATH"
NAAMAPADAM_LANGUAGES: tuple[str, ...] = (
    "as",
    "bn",
    "gu",
    "hi",
    "kn",
    "ml",
    "mr",
    "or",
    "pa",
    "ta",
    "te",
)
NAAMAPADAM_LABELS: Mapping[str, str] = {
    "LOC": LOCATION,
    "ORG": ORGANIZATION,
    "PER": PERSON,
}
NAAMAPADAM_TAG_IDS: Mapping[int, str] = {
    0: "O",
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
}
NAAMAPADAM_SCRIPTS: Mapping[str, str] = {
    "as": "Bengali-Assamese",
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Devanagari",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Devanagari",
    "or": "Odia",
    "pa": "Gurmukhi",
    "ta": "Tamil",
    "te": "Telugu",
}

_REPO_ROOT = Path(__file__).resolve().parents[3]
_JSON_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
_CONLL_EXTENSIONS = {".bio", ".conll", ".iob", ".tsv", ".txt"}
_SUPPORTED_EXTENSIONS = _JSON_EXTENSIONS | _CONLL_EXTENSIONS


class NaamapadamCorpusRequired(DUACredentialRequired):
    """Raised when a Naamapadam suite lacks a usable local corpus path."""


def configured_naamapadam_path(path: str | Path | None = None) -> Path | None:
    """Return the explicit or environment-configured Naamapadam path."""

    raw_path = path if path is not None else os.environ.get(NAAMAPADAM_PATH_ENV)
    if raw_path is None or not str(raw_path).strip():
        return None
    return Path(raw_path).expanduser()


def map_naamapadam_label(label: str, *, language: str = "hi") -> LabelMappingResult:
    """Map a PER/LOC/ORG source label through ``normalize_label``."""

    source_label = str(label or OTHER).strip()
    entity_label = _entity_label(source_label)
    target = NAAMAPADAM_LABELS.get(entity_label)
    canonical = normalize_label(target or source_label, lang=language)
    mapped = target is not None and canonical in CANONICAL_LABELS
    if not mapped:
        canonical = OTHER
    return LabelMappingResult(
        source_label=source_label,
        canonical_label=canonical,
        mapped=mapped,
    )


def load_naamapadam(
    path: str | Path | None = None,
    *,
    languages: str | Sequence[str] | None = None,
    split: str = "test",
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load local Naamapadam/IndicGLUE records as benchmark fixtures.

    Args:
        path: A dataset file or directory. If omitted,
            ``OPENMED_NAAMAPADAM_PATH`` is used.
        languages: Optional ISO language code or codes. Hindi (``hi``), Telugu
            (``te``), and all other published Naamapadam packs are supported.
        split: Requested split. Directory discovery prefers files containing
            this split name and row-level split metadata is respected.
        allow_repo_path: Test-only opt-in for committed synthetic stand-ins.
    """

    root = _resolve_local_source(path, allow_repo_path=allow_repo_path)
    language_filter = _normalize_languages(languages)
    records: list[MultilingualNerRecord] = []
    unmapped: set[str] = set()
    for source_file in _source_files(root, split=split):
        file_language = _language_from_path(source_file)
        for row in _load_rows(source_file):
            row_split = str(row.get("split") or _split_from_path(source_file) or split)
            if row.get("split") is not None and row_split.lower() != split.lower():
                continue
            language = _row_language(row, file_language, language_filter)
            if language_filter and language not in language_filter:
                continue
            record = _record_from_row(
                row,
                index=len(records),
                language=language,
                split=row_split,
                source_path=source_file,
            )
            records.append(record)
            unmapped.update(
                span.source_label for span in record.spans if not span.mapped
            )

    _validate_unique_ids(records)
    return MultilingualNerLoadResult(
        benchmark=NAAMAPADAM,
        records=tuple(records),
        split=split,
        source_path=str(root),
        unmapped_labels=tuple(sorted(unmapped)),
    )


def load_naamapadam_fixtures(
    path: str | Path | None = None,
    *,
    languages: str | Sequence[str] | None = None,
    split: str = "test",
    allow_repo_path: bool = False,
) -> list[BenchmarkFixture]:
    """Load Naamapadam fixtures and reject configured-but-empty sources."""

    result = load_naamapadam(
        path,
        languages=languages,
        split=split,
        allow_repo_path=allow_repo_path,
    )
    fixtures = result.to_benchmark_fixtures()
    if not fixtures:
        raise ValueError(
            f"{NAAMAPADAM_PATH_ENV} is configured but the Naamapadam {split!r} "
            "source contains no benchmark records"
        )
    if not any(fixture.gold_spans for fixture in fixtures):
        raise ValueError(
            f"{NAAMAPADAM_PATH_ENV} is configured but the Naamapadam {split!r} "
            "source contains no annotated entity spans"
        )
    return fixtures


def naamapadam_suite_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Return Naamapadam suite metadata without reading benchmark content."""

    configured = configured_naamapadam_path(path) is not None
    reason = "" if configured else f"{NAAMAPADAM_PATH_ENV} is not set"
    return {
        "availability": {
            "configured": configured,
            "path_env": NAAMAPADAM_PATH_ENV,
            "reason": reason,
            "status": "configured" if configured else "skipped",
        },
        "entity_types": dict(NAAMAPADAM_LABELS),
        "languages": list(NAAMAPADAM_LANGUAGES),
        "license": license_for(NAAMAPADAM).to_dict(),
        "scripts": dict(NAAMAPADAM_SCRIPTS),
        "suite": NAAMAPADAM,
        "task": "general_ner",
    }


def _resolve_local_source(
    path: str | Path | None,
    *,
    allow_repo_path: bool,
) -> Path:
    configured = configured_naamapadam_path(path)
    if configured is None:
        raise NaamapadamCorpusRequired(
            f"Naamapadam requires an explicit local corpus path or "
            f"{NAAMAPADAM_PATH_ENV}; OpenMed bundles no benchmark records"
        )
    if not configured.exists():
        raise NaamapadamCorpusRequired(
            f"Naamapadam local corpus path does not exist: {configured}"
        )
    resolved = configured.resolve()
    if not allow_repo_path and _is_relative_to(resolved, _REPO_ROOT):
        raise NaamapadamCorpusRequired(
            "Naamapadam local corpus path points inside the repository tree; "
            "pass an external path for real benchmark data"
        )
    return resolved


def _normalize_languages(languages: str | Sequence[str] | None) -> tuple[str, ...]:
    if languages is None:
        return ()
    values = (languages,) if isinstance(languages, str) else tuple(languages)
    normalized = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    unsupported = sorted(set(normalized) - set(NAAMAPADAM_LANGUAGES))
    if unsupported:
        raise ValueError(f"unsupported Naamapadam language code(s): {unsupported}")
    return normalized


def _source_files(root: Path, *, split: str) -> tuple[Path, ...]:
    if root.is_file():
        if root.suffix.lower() not in _SUPPORTED_EXTENSIONS:
            raise ValueError(f"unsupported Naamapadam source file: {root}")
        return (root,)
    files = tuple(
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUPPORTED_EXTENSIONS
    )
    split_files = tuple(
        path for path in files if _split_from_path(path) == split.lower()
    )
    return split_files or files


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() in _CONLL_EXTENSIONS:
        return _rows_from_conll(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return _json_lines(text, path)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _json_lines(text, path)
    if isinstance(payload, Mapping):
        rows = (
            payload.get("records")
            or payload.get("documents")
            or payload.get("examples")
            or payload.get("data")
            or payload
        )
    else:
        rows = payload
    if isinstance(rows, Mapping):
        rows = [rows]
    if not isinstance(rows, list):
        raise ValueError(f"Naamapadam JSON must contain records: {path}")
    return [row for row in rows if isinstance(row, Mapping)]


def _json_lines(text: str, path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(
                f"Naamapadam JSON row must be an object: {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _rows_from_conll(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if tokens:
                rows.append({"tokens": tokens, "ner_tags": tags})
                tokens, tags = [], []
            continue
        if stripped.startswith("#"):
            continue
        delimiter = "\t" if "\t" in stripped else None
        fields = [
            field
            for field in next(csv.reader([stripped], delimiter=delimiter or " "))
            if field
        ]
        if len(fields) < 2:
            continue
        tokens.append(fields[0])
        tags.append(fields[-1])
    if tokens:
        rows.append({"tokens": tokens, "ner_tags": tags})
    return rows


def _row_language(
    row: Mapping[str, Any],
    file_language: str | None,
    language_filter: Sequence[str],
) -> str:
    language = str(
        row.get("language") or row.get("lang") or file_language or ""
    ).lower()
    if not language and len(language_filter) == 1:
        language = language_filter[0]
    if language not in NAAMAPADAM_LANGUAGES:
        raise ValueError(
            "Naamapadam record language is missing or unsupported; include a language "
            "field, use a language-coded path such as hi_test.json, or pass languages=..."
        )
    return language


def _record_from_row(
    row: Mapping[str, Any],
    *,
    index: int,
    language: str,
    split: str,
    source_path: Path,
) -> MultilingualNerRecord:
    tokens = row.get("tokens") or row.get("words")
    tags = row.get("ner_tags") or row.get("ner") or row.get("labels")
    if not _is_sequence(tokens) or not _is_sequence(tags):
        raise ValueError("Naamapadam records require token and NER tag sequences")
    token_values = tuple(str(token) for token in tokens)
    tag_values = tuple(_tag_name(tag) for tag in tags)
    if len(token_values) != len(tag_values):
        raise ValueError(
            "Naamapadam token and NER tag sequences must have equal length"
        )
    text, offsets = _tokens_to_text(token_values)
    spans = _spans_from_tags(
        text,
        token_values,
        tag_values,
        offsets,
        language=language,
    )
    source_record_id = str(
        row.get("id") or row.get("record_id") or row.get("sentence_id") or index + 1
    )
    record_id = f"{NAAMAPADAM}-{language}-{source_record_id}"
    return MultilingualNerRecord(
        record_id=record_id,
        benchmark=NAAMAPADAM,
        text=text,
        spans=spans,
        split=split,
        language=language,
        metadata={
            **_clean_metadata(row.get("metadata") or {}),
            "license": license_for(NAAMAPADAM).to_dict(),
            "script": NAAMAPADAM_SCRIPTS[language],
            "source_path_hash": _path_hash(source_path),
            "source_record_id": source_record_id,
            "suite": NAAMAPADAM,
        },
    )


def _spans_from_tags(
    text: str,
    tokens: Sequence[str],
    tags: Sequence[str],
    offsets: Sequence[tuple[int, int]],
    *,
    language: str,
) -> tuple[MultilingualNerSpan, ...]:
    spans: list[MultilingualNerSpan] = []
    active_label = ""
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label and active_start is not None and active_end is not None:
            mapping = map_naamapadam_label(active_label, language=language)
            spans.append(
                MultilingualNerSpan(
                    start=active_start,
                    end=active_end,
                    source_label=mapping.source_label,
                    canonical_label=mapping.canonical_label,
                    text=text[active_start:active_end],
                    mapped=mapping.mapped,
                    metadata={"tag_scheme": "BIO"},
                )
            )
        active_label, active_start, active_end = "", None, None

    for _token, tag, (start, end) in zip(tokens, tags, offsets, strict=True):
        prefix, entity_label = _bio_parts(tag)
        if not entity_label:
            close_active()
            continue
        if prefix in {"B", "S"} or entity_label != active_label:
            close_active()
            active_label, active_start, active_end = entity_label, start, end
            if prefix == "S":
                close_active()
            continue
        active_end = end
        if prefix == "E":
            close_active()
    close_active()
    return tuple(spans)


def _tokens_to_text(tokens: Sequence[str]) -> tuple[str, tuple[tuple[int, int], ...]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if parts:
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return " ".join(parts), tuple(offsets)


def _tag_name(value: Any) -> str:
    if isinstance(value, int) or (isinstance(value, str) and value.strip().isdigit()):
        tag_id = int(value)
        try:
            return NAAMAPADAM_TAG_IDS[tag_id]
        except KeyError as exc:
            raise ValueError(f"unknown Naamapadam numeric NER tag: {tag_id}") from exc
    return str(value).strip()


def _bio_parts(tag: str) -> tuple[str, str]:
    value = str(tag or "O").strip().upper()
    if value == "O":
        return "O", ""
    match = re.fullmatch(r"([BIES])-?(PER|LOC|ORG)", value)
    if match:
        return match.group(1), match.group(2)
    if value in NAAMAPADAM_LABELS:
        return "B", value
    return "B", value


def _entity_label(label: str) -> str:
    value = str(label).strip().upper()
    match = re.fullmatch(r"[BIES]-?(PER|LOC|ORG)", value)
    return match.group(1) if match else value


def _language_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        tokens = re.split(r"[^a-z]+", part.lower())
        for token in tokens:
            if token in NAAMAPADAM_LANGUAGES:
                return token
    return None


def _split_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        tokens = re.split(r"[^a-z]+", part.lower())
        for split in ("test", "validation", "val", "dev", "train"):
            if split in tokens:
                return "validation" if split in {"val", "dev"} else split
    return None


def _clean_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): item
        for key, item in value.items()
        if str(key).lower()
        not in {"text", "note", "note_text", "raw_text", "document_text"}
    }


def _path_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(str(path).encode('utf-8')).hexdigest()}"


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_unique_ids(records: Iterable[MultilingualNerRecord]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.record_id in seen:
            duplicates.add(record.record_id)
        seen.add(record.record_id)
    if duplicates:
        raise ValueError(f"duplicate Naamapadam record id(s): {sorted(duplicates)}")


__all__ = [
    "NAAMAPADAM",
    "NAAMAPADAM_LABELS",
    "NAAMAPADAM_LANGUAGES",
    "NAAMAPADAM_PATH_ENV",
    "NAAMAPADAM_SCRIPTS",
    "NAAMAPADAM_TAG_IDS",
    "NaamapadamCorpusRequired",
    "configured_naamapadam_path",
    "load_naamapadam",
    "load_naamapadam_fixtures",
    "map_naamapadam_label",
    "naamapadam_suite_metadata",
]
