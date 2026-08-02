"""Public biomedical named-entity benchmark suite loaders.

The suite loads corpus rows from public Hugging Face/BigBIO sources on demand,
or from caller-supplied synthetic fixtures in tests. No benchmark corpus rows
are stored in the repository.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.core.labels import (
    CANONICAL_LABELS,
    CONDITION,
    MEDICATION,
    MICROORGANISM,
    OTHER,
    normalize_label,
)
from openmed.eval.datasets.licenses import DatasetLicense, license_for
from openmed.eval.harness import BenchmarkFixture, ModelRunner, run_benchmark
from openmed.eval.metrics import EvalSpan
from openmed.eval.report import BenchmarkReport

BIOMEDICAL_NER = "biomedical-ner"
BC5CDR = "bc5cdr"
NCBI_DISEASE = "ncbi_disease"
JNLPBA = "jnlpba"
SPECIES_800 = "species_800"
BC2GM = "bc2gm"
DEFAULT_SPLIT = "test"

BIOMEDICAL_NER_CORPORA: tuple[str, ...] = (
    BC5CDR,
    NCBI_DISEASE,
    JNLPBA,
    SPECIES_800,
    BC2GM,
)

RowsLoader = Callable[
    ["BiomedicalNerSource", str, str | Path | None],
    Iterable[Mapping[str, Any]],
]


@dataclass(frozen=True)
class BiomedicalNerSource:
    """Dataset coordinates for one public biomedical NER corpus."""

    corpus: str
    display_name: str
    repository: str
    config: str | None
    label_mapping: Mapping[str, str]
    loader_kind: str = "bigbio"
    token_label: str = ""
    trust_remote_code: bool = True

    @property
    def source_url(self) -> str:
        """Return the public source URL for this corpus."""
        return f"https://huggingface.co/datasets/{self.repository}"

    @property
    def license(self) -> DatasetLicense:
        """Return the registered corpus license metadata."""
        return license_for(self.corpus)


@dataclass(frozen=True)
class BiomedicalNerSpan:
    """One source biomedical NER mention mapped to a canonical label."""

    start: int
    end: int
    source_label: str
    canonical_label: str
    text: str
    entity_id: str = ""
    metadata: Mapping[str, Any] | None = None

    def to_eval_span(self, *, language: str = "en") -> EvalSpan:
        """Convert the mention to the eval harness span schema."""
        return EvalSpan(
            start=self.start,
            end=self.end,
            label=self.canonical_label,
            text=self.text,
            language=language,
            metadata={
                **dict(self.metadata or {}),
                "canonical_label": self.canonical_label,
                "entity_id": self.entity_id,
                "source_label": self.source_label,
            },
        )


@dataclass(frozen=True)
class BiomedicalNerRecord:
    """One biomedical NER document or sentence fixture."""

    record_id: str
    corpus: str
    text: str
    spans: tuple[BiomedicalNerSpan, ...]
    split: str = DEFAULT_SPLIT
    language: str = "en"
    metadata: Mapping[str, Any] | None = None

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Expose the record as an eval harness benchmark fixture."""
        source = source_for(self.corpus)
        return BenchmarkFixture(
            fixture_id=self.record_id,
            text=self.text,
            gold_spans=tuple(
                span.to_eval_span(language=self.language) for span in self.spans
            ),
            language=self.language,
            metadata={
                **dict(self.metadata or {}),
                "dataset": self.corpus,
                "display_name": source.display_name,
                "license": source.license.to_dict(),
                "source_url": source.source_url,
                "split": self.split,
                "suite": BIOMEDICAL_NER,
                "task": "ner",
            },
        )


@dataclass(frozen=True)
class BiomedicalNerCorpus:
    """Loaded split for one biomedical NER corpus."""

    corpus: str
    records: tuple[BiomedicalNerRecord, ...]
    split: str
    source_path: str

    def to_benchmark_fixtures(self) -> list[BenchmarkFixture]:
        """Return harness fixtures for this corpus."""
        return [record.to_benchmark_fixture() for record in self.records]


BIOMEDICAL_NER_SOURCES: Mapping[str, BiomedicalNerSource] = {
    BC5CDR: BiomedicalNerSource(
        corpus=BC5CDR,
        display_name="BC5CDR",
        repository="bigbio/bc5cdr",
        config="bc5cdr_bigbio_kb",
        label_mapping={
            "chemical": MEDICATION,
            "disease": CONDITION,
        },
    ),
    NCBI_DISEASE: BiomedicalNerSource(
        corpus=NCBI_DISEASE,
        display_name="NCBI Disease",
        repository="bigbio/ncbi_disease",
        config="ncbi_disease_bigbio_kb",
        label_mapping={
            "compositemention": CONDITION,
            "disease": CONDITION,
            "modifier": CONDITION,
            "specificdisease": CONDITION,
        },
    ),
    JNLPBA: BiomedicalNerSource(
        corpus=JNLPBA,
        display_name="JNLPBA",
        repository="bigbio/jnlpba",
        config="jnlpba_bigbio_kb",
        label_mapping={
            "cellline": OTHER,
            "celltype": OTHER,
            "dna": OTHER,
            "protein": OTHER,
            "rna": OTHER,
        },
    ),
    SPECIES_800: BiomedicalNerSource(
        corpus=SPECIES_800,
        display_name="Species-800",
        repository="spyysalo/species_800",
        config=None,
        label_mapping={
            "organism": MICROORGANISM,
            "species": MICROORGANISM,
            "taxon": MICROORGANISM,
        },
        loader_kind="token_tags",
        token_label="Species",
    ),
    BC2GM: BiomedicalNerSource(
        corpus=BC2GM,
        display_name="BC2GM",
        repository="bigbio/blurb",
        config="bc2gm",
        label_mapping={
            "gene": OTHER,
            "genemention": OTHER,
            "protein": OTHER,
        },
        loader_kind="token_tags",
        token_label="GENE",
        trust_remote_code=False,
    ),
}


def source_for(corpus: str) -> BiomedicalNerSource:
    """Return the source descriptor for *corpus*."""
    try:
        return BIOMEDICAL_NER_SOURCES[corpus]
    except KeyError as exc:
        allowed = ", ".join(BIOMEDICAL_NER_CORPORA)
        raise ValueError(
            f"unknown biomedical NER corpus {corpus!r}: {allowed}"
        ) from exc


def map_biomedical_ner_label(corpus: str, label: str) -> str:
    """Map a source biomedical NER label onto OpenMed's canonical taxonomy."""
    source = source_for(corpus)
    canonical = source.label_mapping.get(_label_key(label))
    if canonical is None:
        canonical = normalize_label(label)
    if canonical not in CANONICAL_LABELS:
        raise RuntimeError(
            f"{corpus} label {label!r} maps to non-canonical label {canonical!r}"
        )
    return canonical


def biomedical_ner_suite_metadata(
    *,
    split: str = DEFAULT_SPLIT,
    corpora: Sequence[str] = BIOMEDICAL_NER_CORPORA,
) -> dict[str, Any]:
    """Return source, license, and label metadata for the suite."""
    sources = [source_for(corpus) for corpus in corpora]
    return {
        "corpora": [source.corpus for source in sources],
        "label_mapping": {
            source.corpus: dict(sorted(source.label_mapping.items()))
            for source in sources
        },
        "licenses": {source.corpus: source.license.to_dict() for source in sources},
        "redistribution": (
            "not vendored; public corpora are loaded by reference and cached "
            "locally on demand"
        ),
        "sources": {
            source.corpus: {
                "config": source.config,
                "display_name": source.display_name,
                "loader_kind": source.loader_kind,
                "repository": source.repository,
                "source_url": source.source_url,
            }
            for source in sources
        },
        "split": split,
        "suite": BIOMEDICAL_NER,
        "task": "ner",
    }


def load_biomedical_ner_corpus(
    corpus: str,
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    rows_loader: RowsLoader | None = None,
    split: str = DEFAULT_SPLIT,
) -> BiomedicalNerCorpus:
    """Load one public biomedical NER corpus into normalized records."""
    source = source_for(corpus)
    if path is not None:
        source_path = _resolve_local_source(path, corpus)
        records = _records_from_local_source(source, source_path, split=split)
        return BiomedicalNerCorpus(
            corpus=corpus,
            records=tuple(records),
            split=split,
            source_path=str(source_path),
        )

    loader = rows_loader or _load_hf_rows
    rows = list(loader(source, split, cache_dir))
    records = records_from_rows(
        corpus,
        rows,
        split=split,
        source_path=source.source_url,
    )
    return BiomedicalNerCorpus(
        corpus=corpus,
        records=tuple(records),
        split=split,
        source_path=source.source_url,
    )


def load_biomedical_ner_fixtures(
    path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    corpora: Sequence[str] = BIOMEDICAL_NER_CORPORA,
    rows_loader: RowsLoader | None = None,
    split: str = DEFAULT_SPLIT,
    task: str = "ner",
) -> list[BenchmarkFixture]:
    """Load all configured biomedical NER corpora as benchmark fixtures."""
    normalized_task = task.strip().lower()
    if normalized_task != "ner":
        raise ValueError("biomedical-ner only supports task='ner'")
    fixtures: list[BenchmarkFixture] = []
    for corpus in corpora:
        fixtures.extend(
            load_biomedical_ner_corpus(
                corpus,
                path=path,
                cache_dir=cache_dir,
                rows_loader=rows_loader,
                split=split,
            ).to_benchmark_fixtures()
        )
    return fixtures


def records_from_rows(
    corpus: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str = DEFAULT_SPLIT,
    source_path: str = "<memory>",
) -> tuple[BiomedicalNerRecord, ...]:
    """Build normalized records from BigBIO or token-classification rows."""
    source = source_for(corpus)
    records: list[BiomedicalNerRecord] = []
    for index, row in enumerate(rows):
        if _is_token_tag_row(row):
            records.append(_record_from_token_tag_row(source, row, index, split=split))
        else:
            records.append(
                _record_from_bigbio_row(
                    source,
                    row,
                    index,
                    split=split,
                    source_path=source_path,
                )
            )
    return tuple(records)


def run_biomedical_ner_benchmark(
    fixtures: Sequence[BenchmarkFixture] | None = None,
    *,
    model_name: str,
    device: str = "cpu",
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    rows_loader: RowsLoader | None = None,
    split: str = DEFAULT_SPLIT,
) -> BenchmarkReport:
    """Run the suite and attach exact/relaxed F1 slices for each corpus."""
    suite_fixtures = list(
        fixtures
        if fixtures is not None
        else load_biomedical_ner_fixtures(
            path=path,
            cache_dir=cache_dir,
            rows_loader=rows_loader,
            split=split,
        )
    )
    report_metadata = biomedical_ner_suite_metadata(split=split)
    report_metadata.update(dict(metadata or {}))
    report = run_benchmark(
        suite_fixtures,
        suite=BIOMEDICAL_NER,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
        metadata=report_metadata,
    )
    metrics = dict(report.metrics)
    metrics["per_corpus"] = _per_corpus_metrics(
        suite_fixtures,
        model_name=model_name,
        device=device,
        runner=runner,
        generated_at=generated_at,
    )
    return BenchmarkReport(
        suite=report.suite,
        model_name=report.model_name,
        device=report.device,
        fixture_count=report.fixture_count,
        metrics=metrics,
        generated_at=report.generated_at,
        metadata=report.metadata,
    )


def _per_corpus_metrics(
    fixtures: Sequence[BenchmarkFixture],
    *,
    model_name: str,
    device: str,
    runner: ModelRunner | None,
    generated_at: str | None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[BenchmarkFixture]] = defaultdict(list)
    for fixture in fixtures:
        corpus = str(fixture.metadata.get("dataset") or "unknown")
        grouped[corpus].append(fixture)

    ordered = [corpus for corpus in BIOMEDICAL_NER_CORPORA if corpus in grouped]
    ordered.extend(sorted(corpus for corpus in grouped if corpus not in ordered))
    per_corpus: dict[str, dict[str, Any]] = {}
    for corpus in ordered:
        corpus_fixtures = grouped[corpus]
        corpus_report = run_benchmark(
            corpus_fixtures,
            suite=f"{BIOMEDICAL_NER}:{corpus}",
            model_name=model_name,
            device=device,
            runner=runner,
            generated_at=generated_at,
            metadata={"corpus": corpus, "suite": BIOMEDICAL_NER},
        )
        per_corpus[corpus] = {
            "exact_span_f1": corpus_report.metrics["exact_span_f1"],
            "fixture_count": corpus_report.fixture_count,
            "relaxed_span_f1": corpus_report.metrics["relaxed_span_f1"],
            "span_count": sum(len(fixture.gold_spans) for fixture in corpus_fixtures),
        }
    return per_corpus


def _records_from_local_source(
    source: BiomedicalNerSource,
    path: Path,
    *,
    split: str,
) -> tuple[BiomedicalNerRecord, ...]:
    suffix = path.suffix.lower()
    if suffix in {".conll", ".iob", ".bio", ".txt"}:
        return tuple(_records_from_conll(path, source, split=split))
    rows = _load_mapping_rows(path)
    return records_from_rows(source.corpus, rows, split=split, source_path=str(path))


def _load_hf_rows(
    source: BiomedicalNerSource,
    split: str,
    cache_dir: str | Path | None,
) -> list[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "loading biomedical-ner public corpora requires the optional "
            "'datasets' package; install the Hugging Face extras or pass a "
            "local synthetic fixture path"
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if source.config is not None:
        kwargs["name"] = source.config
    if cache_dir is not None:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    if source.trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        dataset = load_dataset(source.repository, **kwargs)
    except TypeError as exc:
        if "trust_remote_code" not in str(exc):
            raise
        kwargs.pop("trust_remote_code", None)
        dataset = load_dataset(source.repository, **kwargs)

    tag_names = _tag_names_from_dataset(dataset)
    rows: list[Mapping[str, Any]] = []
    for row in dataset:
        payload = dict(row)
        if tag_names:
            payload["_ner_tag_names"] = tag_names
        rows.append(payload)
    return rows


def _record_from_bigbio_row(
    source: BiomedicalNerSource,
    row: Mapping[str, Any],
    index: int,
    *,
    split: str,
    source_path: str,
) -> BiomedicalNerRecord:
    text, passage_offsets = _document_text_and_passages(row)
    record_id = _record_id(source.corpus, row, index)
    spans = tuple(
        _span_from_entity(source, entity, text=text, passage_offsets=passage_offsets)
        for entity in _entity_rows(row)
    )
    return BiomedicalNerRecord(
        record_id=record_id,
        corpus=source.corpus,
        text=text,
        spans=tuple(sorted(spans, key=lambda span: (span.start, span.end))),
        split=split,
        metadata={
            "loader_kind": "bigbio",
            "source_path": source_path,
            "source_record_id": str(
                row.get("document_id") or row.get("pmid") or row.get("id") or ""
            ),
        },
    )


def _span_from_entity(
    source: BiomedicalNerSource,
    entity: Mapping[str, Any],
    *,
    text: str,
    passage_offsets: Sequence[tuple[int, int, int]],
) -> BiomedicalNerSpan:
    source_label = _source_label(entity)
    start, end = _entity_offsets(entity)
    start, end = _translate_offsets(start, end, passage_offsets)
    entity_text = _text_value(entity.get("text") or entity.get("mention"))
    if entity_text and text[start:end] != entity_text:
        found = text.find(entity_text)
        if found >= 0:
            start, end = found, found + len(entity_text)
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            f"invalid {source.corpus} span offsets {start}:{end} "
            f"for text length {len(text)}"
        )
    span_text = entity_text or text[start:end]
    return BiomedicalNerSpan(
        start=start,
        end=end,
        source_label=source_label,
        canonical_label=map_biomedical_ner_label(source.corpus, source_label),
        text=text[start:end] or span_text,
        entity_id=str(entity.get("id") or entity.get("entity_id") or ""),
        metadata={"source_offsets": _plain_offsets(entity.get("offsets"))},
    )


def _record_from_token_tag_row(
    source: BiomedicalNerSource,
    row: Mapping[str, Any],
    index: int,
    *,
    split: str,
) -> BiomedicalNerRecord:
    tokens = [str(token) for token in row.get("tokens") or []]
    tags = list(row.get("ner_tags") or row.get("tags") or row.get("labels") or [])
    if len(tokens) != len(tags):
        raise ValueError(
            f"{source.corpus} token/tag row has {len(tokens)} tokens and "
            f"{len(tags)} tags"
        )
    text, token_offsets = _tokens_to_text(tokens)
    tag_names = tuple(str(name) for name in row.get("_ner_tag_names") or ())
    spans = _spans_from_token_tags(
        source,
        text=text,
        token_offsets=token_offsets,
        tags=tags,
        tag_names=tag_names,
    )
    return BiomedicalNerRecord(
        record_id=_record_id(source.corpus, row, index),
        corpus=source.corpus,
        text=text,
        spans=tuple(spans),
        split=split,
        metadata={
            "loader_kind": "token_tags",
            "source_record_id": str(row.get("id") or ""),
        },
    )


def _spans_from_token_tags(
    source: BiomedicalNerSource,
    *,
    text: str,
    token_offsets: Sequence[tuple[int, int]],
    tags: Sequence[Any],
    tag_names: Sequence[str],
) -> list[BiomedicalNerSpan]:
    spans: list[BiomedicalNerSpan] = []
    active_label = ""
    active_start: int | None = None
    active_end: int | None = None

    def close_active() -> None:
        nonlocal active_label, active_start, active_end
        if active_label and active_start is not None and active_end is not None:
            spans.append(
                BiomedicalNerSpan(
                    start=active_start,
                    end=active_end,
                    source_label=active_label,
                    canonical_label=map_biomedical_ner_label(
                        source.corpus,
                        active_label,
                    ),
                    text=text[active_start:active_end],
                )
            )
        active_label = ""
        active_start = None
        active_end = None

    for tag, (start, end) in zip(tags, token_offsets):
        prefix, label = _tag_parts(tag, tag_names, default_label=source.token_label)
        if prefix == "O" or not label:
            close_active()
            continue
        if prefix in {"B", "S"} or label != active_label or active_start is None:
            close_active()
            active_label = label
            active_start = start
        active_end = end
        if prefix in {"S", "E"}:
            close_active()
    close_active()
    return spans


def _records_from_conll(
    path: Path,
    source: BiomedicalNerSource,
    *,
    split: str,
) -> list[BiomedicalNerRecord]:
    rows: list[dict[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        if tokens:
            rows.append(
                {
                    "id": f"{path.stem}-{len(rows) + 1}",
                    "ner_tags": list(tags),
                    "tokens": list(tokens),
                }
            )
            tokens.clear()
            tags.clear()

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped.startswith("#") or stripped.startswith("-DOCSTART-"):
            continue
        columns = stripped.split()
        if len(columns) < 2:
            raise ValueError(f"CoNLL row must contain token and label: {line!r}")
        tokens.append(columns[0])
        tags.append(columns[-1])
    flush()

    return [
        _record_from_token_tag_row(source, row, index, split=split)
        for index, row in enumerate(rows)
    ]


def _document_text_and_passages(
    row: Mapping[str, Any],
) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    passages = row.get("passages")
    if isinstance(passages, Sequence) and not isinstance(passages, (str, bytes)):
        parts: list[str] = []
        offsets: list[tuple[int, int, int]] = []
        cursor = 0
        for passage in passages:
            if not isinstance(passage, Mapping):
                continue
            passage_text = _text_value(passage.get("text"))
            if not passage_text:
                continue
            if parts:
                cursor += 1
            document_start = cursor
            source_start, source_end = _passage_offsets(
                passage,
                fallback_start=document_start,
                text_length=len(passage_text),
            )
            parts.append(passage_text)
            offsets.append((source_start, source_end, document_start))
            cursor += len(passage_text)
        if parts:
            return "\n".join(parts), tuple(offsets)

    return _record_text(row), ()


def _record_text(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("text"), str):
        return str(row["text"])
    parts = [
        _text_value(row.get(key))
        for key in ("title", "abstract", "passage", "sentence")
        if row.get(key) is not None
    ]
    return " ".join(part for part in parts if part).strip()


def _entity_rows(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates = row.get("entities") or row.get("annotations") or row.get("spans") or []
    if isinstance(candidates, Mapping):
        return [candidates]
    return [item for item in candidates if isinstance(item, Mapping)]


def _source_label(entity: Mapping[str, Any]) -> str:
    return str(
        entity.get("type")
        or entity.get("label")
        or entity.get("source_label")
        or entity.get("entity_type")
        or "OTHER"
    )


def _entity_offsets(entity: Mapping[str, Any]) -> tuple[int, int]:
    for start_key, end_key in (
        ("start", "end"),
        ("span_start", "span_end"),
        ("begin", "end"),
        ("offset_start", "offset_end"),
    ):
        if start_key in entity and end_key in entity:
            return _int(entity[start_key]), _int(entity[end_key])
    return _first_offset(entity.get("offsets"))


def _passage_offsets(
    passage: Mapping[str, Any],
    *,
    fallback_start: int,
    text_length: int,
) -> tuple[int, int]:
    if passage.get("offsets") is None:
        return fallback_start, fallback_start + text_length
    try:
        return _first_offset(passage.get("offsets"))
    except ValueError:
        return fallback_start, fallback_start + text_length


def _first_offset(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        return _int(value.get("start")), _int(value.get("end"))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) == 2 and all(not isinstance(item, Sequence) for item in value):
            return _int(value[0]), _int(value[1])
        for item in value:
            if isinstance(item, Mapping):
                return _int(item.get("start")), _int(item.get("end"))
            if isinstance(item, Sequence) and len(item) >= 2:
                return _int(item[0]), _int(item[1])
    raise ValueError(f"missing entity offsets: {value!r}")


def _translate_offsets(
    start: int,
    end: int,
    passage_offsets: Sequence[tuple[int, int, int]],
) -> tuple[int, int]:
    for source_start, source_end, document_start in passage_offsets:
        if source_start <= start and end <= source_end:
            return (
                document_start + (start - source_start),
                document_start + (end - source_start),
            )
    return start, end


def _tokens_to_text(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        if pieces:
            pieces.append(" ")
            cursor += 1
        start = cursor
        pieces.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def _tag_parts(
    tag: Any,
    tag_names: Sequence[str],
    *,
    default_label: str,
) -> tuple[str, str]:
    if isinstance(tag, int):
        if tag_names and 0 <= tag < len(tag_names):
            value = tag_names[tag]
        elif tag == 0:
            return "O", ""
        elif tag == 1:
            return "B", default_label
        elif tag == 2:
            return "I", default_label
        else:
            return "O", ""
    else:
        value = str(tag)
        if value.isdigit():
            return _tag_parts(int(value), tag_names, default_label=default_label)

    normalized = value.strip()
    if not normalized or normalized.upper() == "O":
        return "O", ""
    prefix, separator, label = normalized.partition("-")
    if separator and prefix.upper() in {"B", "I", "E", "S"}:
        return prefix.upper(), label
    return "B", normalized


def _is_token_tag_row(row: Mapping[str, Any]) -> bool:
    return "tokens" in row and ("ner_tags" in row or "tags" in row or "labels" in row)


def _tag_names_from_dataset(dataset: Any) -> tuple[str, ...]:
    features = getattr(dataset, "features", None)
    if not isinstance(features, Mapping) or "ner_tags" not in features:
        return ()
    tag_feature = features["ner_tags"]
    if hasattr(tag_feature, "feature"):
        tag_feature = tag_feature.feature
    names = getattr(tag_feature, "names", None)
    if names is None:
        return ()
    return tuple(str(name) for name in names)


def _load_mapping_rows(path: Path) -> list[Mapping[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        rows = (
            raw.get("records")
            or raw.get("documents")
            or raw.get("fixtures")
            or raw.get("examples")
            or []
        )
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of records")
    return [row for row in rows if isinstance(row, Mapping)]


def _resolve_local_source(path: str | Path, corpus: str) -> Path:
    source_path = Path(path).expanduser()
    if source_path.is_file():
        return source_path
    if not source_path.exists():
        raise FileNotFoundError(f"biomedical NER source path not found: {source_path}")
    candidates = _local_candidate_names(corpus)
    for name in candidates:
        candidate = source_path / name
        if candidate.exists():
            return candidate
    wanted = ", ".join(candidates)
    raise FileNotFoundError(
        f"biomedical NER fixture for {corpus!r} not found under {source_path}; "
        f"expected one of: {wanted}"
    )


def _local_candidate_names(corpus: str) -> tuple[str, ...]:
    aliases = {corpus, corpus.replace("_", "-")}
    if corpus == SPECIES_800:
        aliases.add("species-800")
    names: list[str] = []
    for alias in sorted(aliases):
        for suffix in (".json", ".jsonl", ".ndjson", ".conll", ".iob", ".bio", ".txt"):
            names.append(f"{alias}{suffix}")
    return tuple(names)


def _record_id(corpus: str, row: Mapping[str, Any], index: int) -> str:
    raw = row.get("document_id") or row.get("pmid") or row.get("id") or index + 1
    value = str(raw)
    if value.startswith(f"{corpus}:"):
        return value
    return f"{corpus}:{value}"


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(str(item) for item in value)
    return str(value)


def _plain_offsets(value: Any) -> list[list[int]]:
    if value is None:
        return []
    offsets: list[list[int]] = []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = value
    else:
        items = [value]
    for item in items:
        try:
            start, end = _first_offset(item)
        except ValueError:
            continue
        offsets.append([start, end])
    return offsets


def _label_key(label: str) -> str:
    stripped = re.sub(r"^[BIES]-", "", label.strip(), count=1, flags=re.IGNORECASE)
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"expected integer offset, got {value!r}") from None


__all__ = [
    "BC2GM",
    "BC5CDR",
    "BIOMEDICAL_NER",
    "BIOMEDICAL_NER_CORPORA",
    "BIOMEDICAL_NER_SOURCES",
    "DEFAULT_SPLIT",
    "JNLPBA",
    "NCBI_DISEASE",
    "SPECIES_800",
    "BiomedicalNerCorpus",
    "BiomedicalNerRecord",
    "BiomedicalNerSource",
    "BiomedicalNerSpan",
    "biomedical_ner_suite_metadata",
    "load_biomedical_ner_corpus",
    "load_biomedical_ner_fixtures",
    "map_biomedical_ner_label",
    "records_from_rows",
    "run_biomedical_ner_benchmark",
    "source_for",
]
