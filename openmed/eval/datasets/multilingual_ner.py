"""DUA-respecting multilingual clinical NER benchmark loaders.

PharmaCoNER, CANTEMIST, DEFT, and CMeEE are external benchmark corpora.
This module never downloads, vendors, or discovers those records
automatically. Callers must pass an explicit local path that they are allowed
to use. The small fixtures committed with OpenMed are synthetic smoke inputs
only and can be loaded by tests with ``allow_repo_path=True``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.labels import (
    BODY_SITE,
    CANONICAL_LABELS,
    CONDITION,
    LAB_TEST,
    MEDICATION,
    MICROORGANISM,
    ORGANIZATION,
    OTHER,
    PROCEDURE,
    normalize_label,
)
from openmed.eval.datasets.dua_stubs import DUACredentialRequired
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan

MULTILINGUAL_NER = "multilingual-clinical-ner"
PHARMACONER = "pharmaconer"
CANTEMIST = "cantemist"
DEFT = "deft"
CMEEE = "cmeee"
DEFAULT_SPLIT = "test"

MULTILINGUAL_NER_BENCHMARKS: tuple[str, ...] = (
    PHARMACONER,
    CANTEMIST,
    DEFT,
    CMEEE,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ROW_EXTENSIONS = {".json", ".jsonl", ".ndjson"}
_CONLL_EXTENSIONS = {".bio", ".conll", ".iob", ".tsv"}


class MultilingualNerCorpusRequired(DUACredentialRequired):
    """Raised when a multilingual benchmark is requested without local data."""


@dataclass(frozen=True)
class LabelMappingResult:
    """Canonical label mapping for one benchmark source label."""

    source_label: str
    canonical_label: str
    mapped: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "canonical_label": self.canonical_label,
            "mapped": self.mapped,
            "source_label": self.source_label,
        }


@dataclass(frozen=True)
class MultilingualNerSource:
    """Configuration for one user-supplied multilingual NER benchmark."""

    benchmark: str
    display_name: str
    language: str
    label_mapping: Mapping[str, str]
    source_url: str = ""
    access_note: str = "requires explicit local corpus path"

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_note": self.access_note,
            "benchmark": self.benchmark,
            "display_name": self.display_name,
            "label_mapping": dict(sorted(self.label_mapping.items())),
            "language": self.language,
            "source_url": self.source_url,
        }


@dataclass(frozen=True)
class MultilingualNerSpan:
    """One source mention mapped into the OpenMed eval span schema."""

    start: int
    end: int
    source_label: str
    canonical_label: str
    text: str
    mapped: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_eval_span(self, *, benchmark: str, language: str) -> EvalSpan:
        return EvalSpan(
            start=self.start,
            end=self.end,
            label=self.canonical_label,
            text=self.text,
            language=language,
            metadata={
                **dict(self.metadata),
                "benchmark": benchmark,
                "canonical_label": self.canonical_label,
                "source_label": self.source_label,
                "unmapped_label": not self.mapped,
            },
        )


@dataclass(frozen=True)
class MultilingualNerRecord:
    """One benchmark document or sentence after normalization."""

    record_id: str
    benchmark: str
    text: str
    spans: tuple[MultilingualNerSpan, ...]
    split: str = DEFAULT_SPLIT
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        source = source_for(self.benchmark)
        unmapped = tuple(
            sorted({span.source_label for span in self.spans if not span.mapped})
        )
        return BenchmarkFixture(
            fixture_id=self.record_id,
            text=self.text,
            gold_spans=tuple(
                span.to_eval_span(
                    benchmark=self.benchmark,
                    language=self.language,
                )
                for span in self.spans
            ),
            language=self.language,
            metadata={
                **dict(self.metadata),
                "benchmark": self.benchmark,
                "dataset": self.benchmark,
                "display_name": source.display_name,
                "language": self.language,
                "split": self.split,
                "suite": MULTILINGUAL_NER,
                "task": "clinical_ner",
                "text_hash": _text_hash(self.text),
                "unmapped_labels": unmapped,
            },
        )


@dataclass(frozen=True)
class MultilingualNerLoadResult:
    """Loaded split for one multilingual benchmark."""

    benchmark: str
    records: tuple[MultilingualNerRecord, ...]
    split: str
    source_path: str
    unmapped_labels: tuple[str, ...] = ()

    def to_benchmark_fixtures(self) -> list[BenchmarkFixture]:
        return [record.to_benchmark_fixture() for record in self.records]

    @property
    def fixture_count(self) -> int:
        return len(self.records)


MULTILINGUAL_NER_SOURCES: Mapping[str, MultilingualNerSource] = {
    PHARMACONER: MultilingualNerSource(
        benchmark=PHARMACONER,
        display_name="PharmaCoNER",
        language="es",
        source_url="https://temu.bsc.es/pharmaconer/",
        label_mapping={
            "normalizables": MEDICATION,
            "no_normalizables": MEDICATION,
            "non_normalizables": MEDICATION,
            "proteinas": OTHER,
            "protein": OTHER,
            "unclear": OTHER,
            "medication": MEDICATION,
            "drug": MEDICATION,
            "chemical": MEDICATION,
        },
    ),
    CANTEMIST: MultilingualNerSource(
        benchmark=CANTEMIST,
        display_name="CANTEMIST",
        language="es",
        source_url="https://temu.bsc.es/cantemist/",
        label_mapping={
            "morphology_neoplasm": CONDITION,
            "neoplasm": CONDITION,
            "tumor": CONDITION,
            "tumour": CONDITION,
            "condition": CONDITION,
            "cancer": CONDITION,
            "morfologia_neoplasia": CONDITION,
        },
    ),
    DEFT: MultilingualNerSource(
        benchmark=DEFT,
        display_name="DEFT",
        language="fr",
        source_url="https://deft.lisn.upsaclay.fr/",
        label_mapping={
            "anatomie": BODY_SITE,
            "anatomy": BODY_SITE,
            "body_site": BODY_SITE,
            "dose": OTHER,
            "examen": LAB_TEST,
            "exam": LAB_TEST,
            "laboratory": LAB_TEST,
            "lab_test": LAB_TEST,
            "medicament": MEDICATION,
            "medication": MEDICATION,
            "pathologie": CONDITION,
            "problem": CONDITION,
            "procedure": PROCEDURE,
            "signe": CONDITION,
            "sign": CONDITION,
            "symptom": CONDITION,
            "traitement": PROCEDURE,
            "treatment": PROCEDURE,
        },
    ),
    CMEEE: MultilingualNerSource(
        benchmark=CMEEE,
        display_name="CMeEE",
        language="zh",
        source_url="https://tianchi.aliyun.com/dataset/95414",
        label_mapping={
            "bod": BODY_SITE,
            "body": BODY_SITE,
            "body_site": BODY_SITE,
            "dep": ORGANIZATION,
            "department": ORGANIZATION,
            "dis": CONDITION,
            "disease": CONDITION,
            "dru": MEDICATION,
            "drug": MEDICATION,
            "equ": OTHER,
            "equipment": OTHER,
            "ite": LAB_TEST,
            "item": LAB_TEST,
            "lab_test": LAB_TEST,
            "mic": MICROORGANISM,
            "microorganism": MICROORGANISM,
            "pro": PROCEDURE,
            "procedure": PROCEDURE,
            "sym": CONDITION,
            "symptom": CONDITION,
        },
    ),
}


def source_for(benchmark: str) -> MultilingualNerSource:
    """Return the source descriptor for *benchmark*."""

    key = _benchmark_key(benchmark)
    try:
        return MULTILINGUAL_NER_SOURCES[key]
    except KeyError as exc:
        allowed = ", ".join(MULTILINGUAL_NER_BENCHMARKS)
        raise ValueError(
            f"unknown multilingual NER benchmark {benchmark!r}: {allowed}"
        ) from exc


def multilingual_ner_suite_metadata(
    *,
    benchmarks: Sequence[str] = MULTILINGUAL_NER_BENCHMARKS,
    split: str = DEFAULT_SPLIT,
) -> dict[str, Any]:
    """Return PHI-free metadata for the multilingual clinical NER suite."""

    sources = [source_for(benchmark) for benchmark in benchmarks]
    return {
        "benchmarks": [source.benchmark for source in sources],
        "dua_boundary": (
            "OpenMed ships only loaders and synthetic smoke fixtures. Real "
            "benchmark records must be supplied through explicit local paths."
        ),
        "label_mapping": {
            source.benchmark: dict(sorted(source.label_mapping.items()))
            for source in sources
        },
        "languages": {source.benchmark: source.language for source in sources},
        "redistribution": "no licensed benchmark corpus text is bundled",
        "sources": {source.benchmark: source.to_dict() for source in sources},
        "split": split,
        "suite": MULTILINGUAL_NER,
        "task": "clinical_ner",
    }


def map_multilingual_ner_label(
    benchmark: str,
    label: str,
) -> LabelMappingResult:
    """Map a benchmark label to OpenMed's canonical clinical label set.

    Unknown labels are surfaced as unmapped and retained as ``OTHER`` rather
    than being silently dropped.
    """

    source = source_for(benchmark)
    source_label = str(label or OTHER)
    key = _label_key(source_label)
    canonical = source.label_mapping.get(key)
    mapped = canonical is not None
    if canonical is None:
        normalized = normalize_label(source_label, lang=source.language)
        if normalized in CANONICAL_LABELS and normalized != OTHER:
            canonical = normalized
            mapped = True
        else:
            canonical = OTHER
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(
            f"{source.benchmark} label {source_label!r} maps to "
            f"non-canonical label {canonical!r}"
        )
    return LabelMappingResult(
        source_label=source_label,
        canonical_label=canonical,
        mapped=mapped,
    )


def load_multilingual_ner_benchmark(
    benchmark: str,
    path: str | Path | None = None,
    *,
    split: str = DEFAULT_SPLIT,
    allow_repo_path: bool = False,
) -> MultilingualNerLoadResult:
    """Load one user-supplied multilingual benchmark split.

    Args:
        benchmark: One of ``pharmaconer``, ``cantemist``, ``deft``, or
            ``cmeee``.
        path: Explicit local file or directory containing records in a common
            JSON/JSONL, BRAT standoff, or BIO/CoNLL token-label shape.
        split: Split label to attach to fixture metadata.
        allow_repo_path: Internal test escape hatch for committed synthetic
            smoke fixtures. Keep this false for real benchmark paths.

    Raises:
        MultilingualNerCorpusRequired: If no path is supplied, the path is
            absent, or a repository-internal path is passed without the
            explicit synthetic-fixture override.
    """

    source = source_for(benchmark)
    root = _resolve_local_source(
        source.benchmark,
        path,
        allow_repo_path=allow_repo_path,
    )
    records = tuple(
        _record_from_row(source, row, index, split=split, source_path=root)
        for index, row in enumerate(_load_rows(root))
    )
    unmapped = tuple(
        sorted(
            {
                span.source_label
                for record in records
                for span in record.spans
                if not span.mapped
            }
        )
    )
    return MultilingualNerLoadResult(
        benchmark=source.benchmark,
        records=records,
        split=split,
        source_path=str(root),
        unmapped_labels=unmapped,
    )


def load_multilingual_ner_fixtures(
    paths: Mapping[str, str | Path] | str | Path | None = None,
    *,
    benchmarks: Sequence[str] = MULTILINGUAL_NER_BENCHMARKS,
    split: str = DEFAULT_SPLIT,
    allow_repo_path: bool = False,
) -> list[BenchmarkFixture]:
    """Load configured multilingual benchmarks as harness fixtures."""

    fixtures: list[BenchmarkFixture] = []
    for benchmark in benchmarks:
        benchmark_path = _path_for_benchmark(paths, benchmark)
        result = load_multilingual_ner_benchmark(
            benchmark,
            benchmark_path,
            split=split,
            allow_repo_path=allow_repo_path,
        )
        fixtures.extend(result.to_benchmark_fixtures())
    _validate_unique_ids(fixtures)
    return fixtures


def _resolve_local_source(
    benchmark: str,
    path: str | Path | None,
    *,
    allow_repo_path: bool,
) -> Path:
    if path is None:
        raise MultilingualNerCorpusRequired(
            f"{benchmark} requires an explicit local corpus path; OpenMed "
            "does not bundle licensed benchmark records"
        )
    root = Path(path).expanduser()
    if not root.exists():
        raise MultilingualNerCorpusRequired(
            f"{benchmark} local corpus path does not exist: {root}"
        )
    resolved = root.resolve()
    if not allow_repo_path and _is_relative_to(resolved, _REPO_ROOT):
        raise MultilingualNerCorpusRequired(
            f"{benchmark} local corpus path points inside the repository tree; "
            "pass an external credentialed path for real benchmark data"
        )
    return resolved


def _path_for_benchmark(
    paths: Mapping[str, str | Path] | str | Path | None,
    benchmark: str,
) -> str | Path | None:
    if isinstance(paths, Mapping):
        return paths.get(benchmark)
    if paths is None:
        return None
    root = Path(paths)
    if root.is_dir():
        candidate = root / f"{benchmark}.jsonl"
        if candidate.exists():
            return candidate
        candidate = root / benchmark
        if candidate.exists():
            return candidate
    return root


def _load_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.is_dir():
        rows: list[Mapping[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_dir():
                rows.extend(_load_rows(child))
            elif child.suffix.lower() in _ROW_EXTENSIONS | _CONLL_EXTENSIONS:
                rows.extend(_load_rows(child))
            elif child.suffix.lower() == ".ann":
                rows.extend(_rows_from_brat(child))
        return rows

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        return [
            row
            for row in (
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if isinstance(row, Mapping)
        ]
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = (
            raw.get("records")
            or raw.get("documents")
            or raw.get("fixtures")
            or raw.get("examples")
            or raw
            if isinstance(raw, Mapping)
            else raw
        )
        if isinstance(rows, Mapping):
            rows = [rows]
        if not isinstance(rows, list):
            raise ValueError(f"benchmark JSON must contain records: {path}")
        return [row for row in rows if isinstance(row, Mapping)]
    if suffix == ".ann":
        return _rows_from_brat(path)
    if suffix in _CONLL_EXTENSIONS:
        return _rows_from_conll(path)
    raise ValueError(f"unsupported multilingual NER source file: {path}")


def _record_from_row(
    source: MultilingualNerSource,
    row: Mapping[str, Any],
    index: int,
    *,
    split: str,
    source_path: Path,
) -> MultilingualNerRecord:
    text = _record_text(row)
    language = str(row.get("language") or row.get("lang") or source.language)
    record_id = str(
        row.get("id")
        or row.get("record_id")
        or row.get("document_id")
        or f"{source.benchmark}-{index + 1}"
    )
    raw_spans = (
        row.get("spans")
        or row.get("entities")
        or row.get("annotations")
        or row.get("mentions")
        or []
    )
    spans = tuple(
        sorted(
            (
                _span_from_mapping(source, span, text=text)
                for span in raw_spans
                if isinstance(span, Mapping)
            ),
            key=lambda span: (span.start, span.end, span.source_label),
        )
    )
    return MultilingualNerRecord(
        record_id=record_id,
        benchmark=source.benchmark,
        text=text,
        spans=spans,
        split=str(row.get("split") or split),
        language=language,
        metadata={
            **_clean_metadata(row.get("metadata") or {}),
            "source_path_hash": _path_hash(source_path),
            "source_record_id": str(row.get("source_record_id") or record_id),
        },
    )


def _span_from_mapping(
    source: MultilingualNerSource,
    span: Mapping[str, Any],
    *,
    text: str,
) -> MultilingualNerSpan:
    start, end = _span_offsets(span)
    source_label = str(
        span.get("source_label")
        or span.get("label")
        or span.get("entity_type")
        or span.get("type")
        or span.get("category")
        or OTHER
    )
    mapping = map_multilingual_ner_label(source.benchmark, source_label)
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"{source.benchmark} span offsets {start}:{end} exceed text length "
            f"{len(text)}"
        )
    span_text = str(span.get("text") or span.get("mention") or text[start:end])
    if span_text and text[start:end] != span_text:
        found = text.find(span_text)
        if found >= 0:
            start, end = found, found + len(span_text)
    return MultilingualNerSpan(
        start=start,
        end=end,
        source_label=mapping.source_label,
        canonical_label=mapping.canonical_label,
        text=text[start:end] or span_text,
        mapped=mapping.mapped,
        metadata=_clean_metadata(span.get("metadata") or {}),
    )


def _span_offsets(span: Mapping[str, Any]) -> tuple[int, int]:
    for start_key, end_key in (
        ("start", "end"),
        ("span_start", "span_end"),
        ("offset_start", "offset_end"),
        ("begin", "end"),
    ):
        start = _int_value(span.get(start_key))
        end = _int_value(span.get(end_key))
        if start is not None and end is not None:
            return start, end

    offsets = span.get("offsets") or span.get("char_offsets")
    if isinstance(offsets, Sequence) and not isinstance(offsets, (str, bytes)):
        first = offsets[0] if offsets else None
        if isinstance(first, Sequence) and len(first) >= 2:
            start = _int_value(first[0])
            end = _int_value(first[1])
            if start is not None and end is not None:
                return start, end
    raise ValueError(f"span missing integer offsets: {span!r}")


def _rows_from_brat(path: Path) -> list[Mapping[str, Any]]:
    text_path = path.with_suffix(".txt")
    if not text_path.exists():
        raise ValueError(f"BRAT annotation is missing paired text file: {text_path}")
    text = text_path.read_text(encoding="utf-8")
    spans: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        pieces = line.split("\t")
        if len(pieces) < 2:
            continue
        header = pieces[1].replace(";", " ").split()
        if len(header) < 3:
            continue
        label = header[0]
        start = int(header[1])
        end = int(header[2])
        spans.append(
            {
                "end": end,
                "label": label,
                "start": start,
                "text": pieces[2] if len(pieces) > 2 else text[start:end],
            }
        )
    return [
        {
            "id": path.stem,
            "metadata": {"loader_kind": "brat"},
            "spans": spans,
            "text": text,
        }
    ]


def _rows_from_conll(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    tokens: list[str] = []
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            if tokens:
                rows.append(_row_from_tokens(path, len(rows), tokens, labels))
                tokens = []
                labels = []
            continue
        if stripped.startswith("#"):
            continue
        delimiter = "\t" if "\t" in stripped else None
        parts = next(csv.reader([stripped], delimiter=delimiter or " "))
        parts = [part for part in parts if part]
        if len(parts) < 2:
            continue
        tokens.append(parts[0])
        labels.append(parts[-1])
    if tokens:
        rows.append(_row_from_tokens(path, len(rows), tokens, labels))
    return rows


def _row_from_tokens(
    path: Path,
    index: int,
    tokens: Sequence[str],
    labels: Sequence[str],
) -> Mapping[str, Any]:
    text, offsets = _tokens_to_text(tokens)
    spans: list[dict[str, Any]] = []
    active_label = ""
    active_start: int | None = None
    active_end: int | None = None
    for token, label, (start, end) in zip(tokens, labels, offsets, strict=True):
        prefix, source_label = _bio_parts(label)
        if prefix in {"B", "S"} or source_label != active_label:
            if active_label and active_start is not None and active_end is not None:
                spans.append(
                    {
                        "end": active_end,
                        "label": active_label,
                        "start": active_start,
                        "text": text[active_start:active_end],
                    }
                )
            active_label = source_label if source_label else ""
            active_start = start if active_label else None
            active_end = end if active_label else None
            if prefix == "S" and active_label:
                spans.append(
                    {
                        "end": end,
                        "label": active_label,
                        "start": start,
                        "text": token,
                    }
                )
                active_label = ""
                active_start = None
                active_end = None
            continue
        if source_label and active_label:
            active_end = end
        elif active_label and active_start is not None and active_end is not None:
            spans.append(
                {
                    "end": active_end,
                    "label": active_label,
                    "start": active_start,
                    "text": text[active_start:active_end],
                }
            )
            active_label = ""
            active_start = None
            active_end = None
    if active_label and active_start is not None and active_end is not None:
        spans.append(
            {
                "end": active_end,
                "label": active_label,
                "start": active_start,
                "text": text[active_start:active_end],
            }
        )
    return {
        "id": f"{path.stem}-{index + 1}",
        "metadata": {"loader_kind": "conll"},
        "spans": spans,
        "text": text,
    }


def _tokens_to_text(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    offsets: list[tuple[int, int]] = []
    for token in tokens:
        if text:
            text += " "
        start = len(text)
        text += token
        offsets.append((start, len(text)))
    return text, offsets


def _bio_parts(label: str) -> tuple[str, str]:
    normalized = str(label or "O").strip()
    if normalized.upper() == "O":
        return "O", ""
    if "-" in normalized and normalized[0].upper() in {"B", "I", "E", "S"}:
        return normalized[0].upper(), normalized[2:]
    return "B", normalized


def _record_text(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return str(row["text"])
    tokens = row.get("tokens")
    if isinstance(tokens, Sequence) and not isinstance(tokens, (str, bytes)):
        return " ".join(str(token) for token in tokens)
    parts = [
        str(row[key])
        for key in ("title", "abstract", "note_text", "sentence")
        if isinstance(row.get(key), str)
    ]
    return " ".join(part for part in parts if part).strip()


def _benchmark_key(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def _label_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _int_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _text_hash(text: str) -> str:
    return f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def _path_hash(path: Path) -> str:
    return f"sha256:{hashlib.sha256(str(path).encode('utf-8')).hexdigest()}"


def _clean_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _plain_metadata(item)
        for key, item in value.items()
        if str(key).lower()
        not in {"text", "note", "note_text", "raw_text", "document_text"}
    }


def _plain_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _clean_metadata(value)
    if isinstance(value, (list, tuple)):
        return [_plain_metadata(item) for item in value]
    return value


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_unique_ids(fixtures: Sequence[BenchmarkFixture]) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for fixture in fixtures:
        if fixture.fixture_id in seen and fixture.fixture_id not in duplicates:
            duplicates.append(fixture.fixture_id)
        seen.add(fixture.fixture_id)
    if duplicates:
        quoted = ", ".join(repr(item) for item in duplicates)
        raise ValueError(f"duplicate benchmark fixture id(s): {quoted}")


def collect_unmapped_labels(
    results: Iterable[MultilingualNerLoadResult],
) -> dict[str, tuple[str, ...]]:
    """Return unmapped source labels by benchmark."""

    labels: defaultdict[str, set[str]] = defaultdict(set)
    for result in results:
        labels[result.benchmark].update(result.unmapped_labels)
    return {
        benchmark: tuple(sorted(values))
        for benchmark, values in sorted(labels.items())
        if values
    }


__all__ = [
    "CANTEMIST",
    "CMEEE",
    "DEFT",
    "DEFAULT_SPLIT",
    "MULTILINGUAL_NER",
    "MULTILINGUAL_NER_BENCHMARKS",
    "MULTILINGUAL_NER_SOURCES",
    "PHARMACONER",
    "LabelMappingResult",
    "MultilingualNerCorpusRequired",
    "MultilingualNerLoadResult",
    "MultilingualNerRecord",
    "MultilingualNerSource",
    "MultilingualNerSpan",
    "collect_unmapped_labels",
    "load_multilingual_ner_benchmark",
    "load_multilingual_ner_fixtures",
    "map_multilingual_ner_label",
    "multilingual_ner_suite_metadata",
    "source_for",
]
