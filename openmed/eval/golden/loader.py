"""Loader for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import NATIONAL_ID_ONLY_LANGUAGES, SUPPORTED_LANGUAGES
from openmed.eval.golden.hard_negatives import HARD_NEGATIVE_CATEGORY
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import EvalSpan, normalize_eval_spans

GOLDEN_CATEGORIES: tuple[str, ...] = (
    "nested_overlapping",
    "chunk_boundary",
    "multilingual",
    "checksum_ids",
    "financial_ids",
    "date_arithmetic",
    "policy_profile_actions",
    HARD_NEGATIVE_CATEGORY,
)

_FIXTURE_VERSION = 1
_GOLDEN_DIR = Path(__file__).resolve().parent
_FIXTURE_DIR = _GOLDEN_DIR / "fixtures"
_TOP_LEVEL_FIXTURES: tuple[Path, ...] = (_GOLDEN_DIR / "financial_ids.jsonl",)
_NON_DEID_FIXTURE_NAMES = frozenset(
    {
        "context_multilingual.jsonl",
        "grounding_crosslingual.jsonl",
        "relation_assertion.jsonl",
    }
)


@dataclass(frozen=True)
class GoldenFixture:
    """One validated golden fixture with expected post-action output."""

    fixture_id: str
    category: str
    language: str
    text: str
    gold_spans: tuple[EvalSpan, ...]
    expected_output: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GoldenFixture":
        """Build and validate a golden fixture from a JSON-ready mapping."""
        if not isinstance(data, Mapping):
            raise ValueError("golden fixture must be a mapping")

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("golden fixture metadata must be a mapping")
        metadata = dict(metadata)

        if metadata.get("synthetic") is not True:
            raise ValueError("golden fixture metadata.synthetic must be true")

        category = str(metadata.get("category", ""))
        if category not in GOLDEN_CATEGORIES:
            raise ValueError(f"unknown golden fixture category: {category!r}")

        expected_output = metadata.get("expected_output")
        if not isinstance(expected_output, Mapping):
            raise ValueError(
                "golden fixture metadata.expected_output must be a mapping"
            )
        if not str(expected_output.get("method", "")):
            raise ValueError("golden fixture expected_output.method is required")
        if not isinstance(expected_output.get("text"), str):
            raise ValueError("golden fixture expected_output.text is required")

        language = str(data.get("language") or data.get("lang") or "en")
        fixture_languages = SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES
        if language not in fixture_languages:
            raise ValueError(f"unsupported golden fixture language: {language!r}")

        text = str(data.get("text", ""))
        if not text:
            raise ValueError("golden fixture text is required")

        raw_spans = data.get("gold_spans") or []
        if not isinstance(raw_spans, list):
            raise ValueError("golden fixture gold_spans must be a list")
        if not raw_spans and category != HARD_NEGATIVE_CATEGORY:
            raise ValueError("golden fixture must include at least one gold span")
        _validate_raw_span_labels(raw_spans, language)

        gold_spans = tuple(
            normalize_eval_spans(raw_spans, default_language=language, source_text=text)
        )
        _validate_offsets(text, gold_spans)
        if category == HARD_NEGATIVE_CATEGORY:
            _validate_hard_negative_fixture(text, metadata, language)

        fixture_id = str(data.get("id") or data.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("golden fixture id is required")

        return cls(
            fixture_id=fixture_id,
            category=category,
            language=language,
            text=text,
            gold_spans=gold_spans,
            expected_output=dict(expected_output),
            metadata=metadata,
        )

    def to_benchmark_fixture(self) -> BenchmarkFixture:
        """Return the harness-compatible fixture view."""
        return BenchmarkFixture(
            fixture_id=self.fixture_id,
            text=self.text,
            gold_spans=self.gold_spans,
            language=self.language,
            metadata=dict(self.metadata),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable JSON-ready mapping."""
        return {
            "id": self.fixture_id,
            "language": self.language,
            "text": self.text,
            "gold_spans": [_span_to_mapping(span) for span in self.gold_spans],
            "metadata": _plain_mapping(self.metadata),
        }


@dataclass(frozen=True)
class AnnotationRelation:
    """One typed relation between two imported annotation spans."""

    document_id: str
    annotator_id: str
    relation_id: str
    relation_type: str
    source_id: str
    target_id: str
    source_span: EvalSpan
    target_span: EvalSpan
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple[str, str, str]:
        """Return the compact relation triple used by eval adapters."""
        return (self.relation_type, self.source_id, self.target_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready relation payload."""
        return {
            "annotator_id": self.annotator_id,
            "document_id": self.document_id,
            "metadata": _plain_mapping(self.metadata),
            "relation_id": self.relation_id,
            "relation_type": self.relation_type,
            "source_id": self.source_id,
            "source_span": _span_to_mapping(self.source_span),
            "target_id": self.target_id,
            "target_span": _span_to_mapping(self.target_span),
        }


@dataclass(frozen=True)
class MultiAnnotatorGoldDocument:
    """A synthetic document annotated by one or more reviewers."""

    document_id: str
    text: str
    spans: tuple[EvalSpan, ...]
    relations: tuple[AnnotationRelation, ...] = ()
    annotators: tuple[str, ...] = ()
    language: str = "en"
    source_format: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def spans_for(self, annotator_id: str) -> tuple[EvalSpan, ...]:
        """Return spans for one annotator."""
        return tuple(
            span
            for span in self.spans
            if span.metadata.get("annotator_id") == annotator_id
        )

    def relations_for(self, annotator_id: str) -> tuple[AnnotationRelation, ...]:
        """Return relations for one annotator."""
        return tuple(
            relation
            for relation in self.relations
            if relation.annotator_id == annotator_id
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready document payload."""
        return {
            "annotators": list(self.annotators),
            "document_id": self.document_id,
            "language": self.language,
            "metadata": _plain_mapping(self.metadata),
            "relations": [relation.to_dict() for relation in self.relations],
            "source_format": self.source_format,
            "spans": [_span_to_mapping(span) for span in self.spans],
            "text": self.text,
        }


def parse_brat_multi_annotator(
    text: str,
    annotations: Mapping[str, str],
    *,
    document_id: str,
    language: str = "en",
) -> MultiAnnotatorGoldDocument:
    """Parse BRAT standoff annotations for one document and many annotators."""
    if not document_id:
        raise ValueError("document_id is required")
    if not text:
        raise ValueError("BRAT document text is required")
    if not annotations:
        raise ValueError("BRAT annotations must include at least one annotator")

    all_spans: list[EvalSpan] = []
    all_relations: list[AnnotationRelation] = []
    annotation_items = tuple(sorted(annotations.items(), key=lambda item: str(item[0])))
    annotators = tuple(str(annotator_id) for annotator_id, _ in annotation_items)

    for raw_annotator_id, ann_text in annotation_items:
        annotator_id = str(raw_annotator_id)
        span_by_id: dict[str, EvalSpan] = {}
        relation_lines: list[tuple[str, str, int]] = []
        for line_number, line in enumerate(ann_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("T"):
                source_id, span = _parse_brat_span_line(
                    stripped,
                    text,
                    document_id=document_id,
                    annotator_id=annotator_id,
                    language=language,
                    line_number=line_number,
                )
                if source_id in span_by_id:
                    raise ValueError(
                        f"duplicate BRAT span id {source_id!r} for {annotator_id}"
                    )
                span_by_id[source_id] = span
                continue
            if stripped.startswith("R"):
                relation_id, body = _split_brat_line(stripped, line_number)
                relation_lines.append((relation_id, body, line_number))
                continue
            if stripped[0] in {"#", "A", "M", "N"}:
                continue
            raise ValueError(
                "unsupported BRAT annotation line "
                f"{line_number} for {annotator_id}: {stripped!r}"
            )

        all_spans.extend(span_by_id.values())
        for relation_id, body, line_number in relation_lines:
            all_relations.append(
                _parse_brat_relation_line(
                    relation_id,
                    body,
                    span_by_id,
                    document_id=document_id,
                    annotator_id=annotator_id,
                    line_number=line_number,
                )
            )

    all_spans.sort(
        key=lambda span: (
            str(span.metadata.get("annotator_id", "")),
            span.start,
            span.end,
            span.label,
            str(span.metadata.get("source_annotation_id", "")),
        )
    )
    all_relations.sort(
        key=lambda relation: (
            relation.annotator_id,
            relation.relation_id,
            relation.relation_type,
        )
    )
    return MultiAnnotatorGoldDocument(
        document_id=document_id,
        text=text,
        spans=tuple(all_spans),
        relations=tuple(all_relations),
        annotators=annotators,
        language=language,
        source_format="brat",
    )


def load_brat_multi_annotator_document(
    text_path: str | Path,
    ann_paths: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    document_id: str | None = None,
    language: str = "en",
) -> MultiAnnotatorGoldDocument:
    """Load one BRAT text document with per-annotator ``.ann`` files."""
    source_text_path = Path(text_path)
    text = source_text_path.read_text(encoding="utf-8")
    if isinstance(ann_paths, Mapping):
        annotations = {
            str(annotator_id): Path(path).read_text(encoding="utf-8")
            for annotator_id, path in ann_paths.items()
        }
    else:
        annotations = {
            Path(path).stem: Path(path).read_text(encoding="utf-8")
            for path in ann_paths
        }
    return parse_brat_multi_annotator(
        text,
        annotations,
        document_id=document_id or source_text_path.stem,
        language=language,
    )


def parse_label_studio_multi_annotator_export(
    payload: Any,
    *,
    document_id: str | None = None,
    default_language: str = "en",
) -> tuple[MultiAnnotatorGoldDocument, ...]:
    """Parse Label Studio JSON exports into multi-annotator documents."""
    tasks = _label_studio_tasks(payload)
    return tuple(
        _parse_label_studio_task(
            task,
            document_id=document_id,
            default_language=default_language,
        )
        for task in tasks
    )


def load_label_studio_multi_annotator_export(
    path: str | Path,
    *,
    document_id: str | None = None,
    default_language: str = "en",
) -> tuple[MultiAnnotatorGoldDocument, ...]:
    """Load a Label Studio JSON export from disk."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_label_studio_multi_annotator_export(
        payload,
        document_id=document_id,
        default_language=default_language,
    )


def list_fixture_paths(path: str | Path | None = None) -> tuple[Path, ...]:
    """Return fixture paths in deterministic order."""
    fixture_path = Path(path) if path is not None else _FIXTURE_DIR
    if fixture_path.is_file():
        return (fixture_path,)
    paths = [
        *fixture_path.glob("*.json"),
        *(
            path
            for path in fixture_path.glob("**/*.jsonl")
            if path.name not in _NON_DEID_FIXTURE_NAMES
        ),
    ]
    if path is None:
        paths.extend(fixture for fixture in _TOP_LEVEL_FIXTURES if fixture.exists())
    return tuple(sorted(paths))


def load_golden_fixtures(path: str | Path | None = None) -> list[GoldenFixture]:
    """Load and validate all golden fixtures under *path*."""
    fixtures: list[GoldenFixture] = []
    for fixture_path in list_fixture_paths(path):
        if fixture_path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in fixture_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            fixtures.extend(GoldenFixture.from_mapping(row) for row in rows)
            continue

        raw = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError(f"{fixture_path} must contain a mapping")
        if raw.get("version") != _FIXTURE_VERSION:
            raise ValueError(f"{fixture_path} has unsupported fixture version")
        if raw.get("synthetic") is not True:
            raise ValueError(f"{fixture_path} must be marked synthetic")
        rows = raw.get("fixtures")
        if not isinstance(rows, list):
            raise ValueError(f"{fixture_path} must contain a fixtures list")
        fixtures.extend(GoldenFixture.from_mapping(row) for row in rows)
    return fixtures


def load_benchmark_fixtures(path: str | Path | None = None) -> list[BenchmarkFixture]:
    """Load golden fixtures as eval harness benchmark fixtures."""
    return [fixture.to_benchmark_fixture() for fixture in load_golden_fixtures(path)]


def benchmark_fixtures_by_language(
    fixtures: list[BenchmarkFixture] | None = None,
    *,
    category: str | None = None,
) -> dict[str, list[BenchmarkFixture]]:
    """Group benchmark fixtures by language in deterministic order."""
    source = fixtures if fixtures is not None else load_benchmark_fixtures()
    grouped: defaultdict[str, list[BenchmarkFixture]] = defaultdict(list)
    for fixture in source:
        if category is None or fixture.metadata.get("category") == category:
            grouped[fixture.language].append(fixture)
    return {
        language: sorted(rows, key=lambda fixture: fixture.fixture_id)
        for language, rows in sorted(grouped.items())
    }


def benchmark_fixture_languages(
    fixtures: list[BenchmarkFixture] | None = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Return languages covered by benchmark fixtures."""
    return set(benchmark_fixtures_by_language(fixtures, category=category))


def fixtures_by_category(
    fixtures: list[GoldenFixture] | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by golden category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        grouped[fixture.category].append(fixture)
    return dict(grouped)


def fixtures_by_language(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> dict[str, list[GoldenFixture]]:
    """Group fixtures by language, optionally restricted to one category."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    grouped: defaultdict[str, list[GoldenFixture]] = defaultdict(list)
    for fixture in source:
        if category is None or fixture.category == category:
            grouped[fixture.language].append(fixture)
    return dict(grouped)


def fixture_languages(
    fixtures: list[GoldenFixture] | None = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Return languages covered by loaded fixtures."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    return {
        fixture.language
        for fixture in source
        if category is None or fixture.category == category
    }


def _parse_brat_span_line(
    line: str,
    document_text: str,
    *,
    document_id: str,
    annotator_id: str,
    language: str,
    line_number: int,
) -> tuple[str, EvalSpan]:
    source_id, body = _split_brat_line(line, line_number)
    try:
        label_and_offsets, surface = body.split("\t", 1)
    except ValueError as exc:
        raise ValueError(
            f"BRAT span line {line_number} must include label, offsets, and text"
        ) from exc
    if ";" in label_and_offsets:
        raise ValueError(
            f"BRAT span line {line_number} uses discontinuous offsets; "
            "single-span EvalSpan imports require contiguous offsets"
        )
    parts = label_and_offsets.split()
    if len(parts) != 3:
        raise ValueError(
            f"BRAT span line {line_number} must have label start end fields"
        )
    raw_label, raw_start, raw_end = parts
    start = _parse_int(raw_start, f"BRAT span {source_id} start")
    end = _parse_int(raw_end, f"BRAT span {source_id} end")
    _validate_import_offsets(
        document_text, start, end, surface, f"BRAT span {source_id}"
    )
    label = normalize_label(raw_label, language)
    return (
        source_id,
        EvalSpan(
            start=start,
            end=end,
            label=label,
            text=surface,
            language=language,
            metadata={
                "annotator_id": annotator_id,
                "document_id": document_id,
                "source_annotation_id": source_id,
                "source_format": "brat",
                "source_label": raw_label,
            },
        ),
    )


def _parse_brat_relation_line(
    relation_id: str,
    body: str,
    span_by_id: Mapping[str, EvalSpan],
    *,
    document_id: str,
    annotator_id: str,
    line_number: int,
) -> AnnotationRelation:
    parts = body.split()
    if len(parts) < 3:
        raise ValueError(
            f"BRAT relation line {line_number} must include type, Arg1, and Arg2"
        )
    relation_type = parts[0]
    arguments: dict[str, str] = {}
    for item in parts[1:]:
        if ":" not in item:
            raise ValueError(
                f"BRAT relation line {line_number} has malformed argument {item!r}"
            )
        role, span_id = item.split(":", 1)
        arguments[role] = span_id
    source_id = arguments.get("Arg1")
    target_id = arguments.get("Arg2")
    if not source_id or not target_id:
        raise ValueError(
            f"BRAT relation line {line_number} must include Arg1 and Arg2 endpoints"
        )
    return _build_annotation_relation(
        document_id=document_id,
        annotator_id=annotator_id,
        relation_id=relation_id,
        relation_type=relation_type,
        source_id=source_id,
        target_id=target_id,
        span_by_id=span_by_id,
        source_format="brat",
        metadata={"argument_roles": arguments},
    )


def _split_brat_line(line: str, line_number: int) -> tuple[str, str]:
    try:
        source_id, body = line.split("\t", 1)
    except ValueError as exc:
        raise ValueError(
            f"BRAT annotation line {line_number} must be tab-delimited"
        ) from exc
    return source_id, body


def _label_studio_tasks(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("tasks"), list):
            payload = payload["tasks"]
        else:
            payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Label Studio export must be a task object or list")
    tasks: list[Mapping[str, Any]] = []
    for index, task in enumerate(payload, start=1):
        if not isinstance(task, Mapping):
            raise ValueError(f"Label Studio task {index} must be a mapping")
        tasks.append(task)
    return tasks


def _parse_label_studio_task(
    task: Mapping[str, Any],
    *,
    document_id: str | None,
    default_language: str,
) -> MultiAnnotatorGoldDocument:
    data = task.get("data") or {}
    if not isinstance(data, Mapping):
        raise ValueError("Label Studio task data must be a mapping")
    text = _label_studio_text(data)
    task_document_id = document_id or str(
        task.get("id") or data.get("document_id") or data.get("id") or ""
    )
    if not task_document_id:
        raise ValueError("Label Studio task document id is required")
    language = str(data.get("language") or data.get("lang") or default_language)
    raw_annotations = task.get("annotations") or task.get("completions") or []
    if not isinstance(raw_annotations, list) or not raw_annotations:
        raise ValueError("Label Studio task must include annotations")

    spans: list[EvalSpan] = []
    relations: list[AnnotationRelation] = []
    annotators: list[str] = []
    for annotation_index, annotation in enumerate(raw_annotations, start=1):
        if not isinstance(annotation, Mapping):
            raise ValueError(
                f"Label Studio annotation {annotation_index} must be a mapping"
            )
        annotator_id = _label_studio_annotator_id(annotation, annotation_index)
        annotators.append(annotator_id)
        span_by_id: dict[str, EvalSpan] = {}
        relation_results: list[Mapping[str, Any]] = []
        results = annotation.get("result") or []
        if not isinstance(results, list):
            raise ValueError("Label Studio annotation result must be a list")
        for result_index, result in enumerate(results, start=1):
            if not isinstance(result, Mapping):
                raise ValueError(
                    f"Label Studio result {result_index} must be a mapping"
                )
            result_type = str(result.get("type") or "")
            if result_type == "relation":
                relation_results.append(result)
                continue
            if result_type in {"labels", "hypertextlabels"}:
                source_id, span = _parse_label_studio_span_result(
                    result,
                    text,
                    document_id=task_document_id,
                    annotator_id=annotator_id,
                    language=language,
                )
                if source_id in span_by_id:
                    raise ValueError(
                        "duplicate Label Studio span id "
                        f"{source_id!r} for {annotator_id}"
                    )
                span_by_id[source_id] = span

        spans.extend(span_by_id.values())
        for result in relation_results:
            relations.append(
                _parse_label_studio_relation_result(
                    result,
                    span_by_id,
                    document_id=task_document_id,
                    annotator_id=annotator_id,
                )
            )

    spans.sort(
        key=lambda span: (
            str(span.metadata.get("annotator_id", "")),
            span.start,
            span.end,
            span.label,
            str(span.metadata.get("source_annotation_id", "")),
        )
    )
    relations.sort(
        key=lambda relation: (
            relation.annotator_id,
            relation.relation_id,
            relation.relation_type,
        )
    )
    return MultiAnnotatorGoldDocument(
        document_id=task_document_id,
        text=text,
        spans=tuple(spans),
        relations=tuple(relations),
        annotators=tuple(sorted(set(annotators))),
        language=language,
        source_format="label_studio",
        metadata=_label_studio_task_metadata(data),
    )


def _label_studio_text(data: Mapping[str, Any]) -> str:
    text = data.get("text")
    if isinstance(text, str) and text:
        return text
    string_values = [
        value for value in data.values() if isinstance(value, str) and value
    ]
    if len(string_values) == 1:
        return string_values[0]
    raise ValueError("Label Studio task data must include source text")


def _label_studio_task_metadata(data: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("synthetic", "source", "source_dataset"):
        if key in data:
            metadata[key] = data[key]
    return metadata


def _label_studio_annotator_id(
    annotation: Mapping[str, Any],
    fallback_index: int,
) -> str:
    completed_by = annotation.get("completed_by") or annotation.get("created_by")
    if isinstance(completed_by, Mapping):
        for key in ("username", "email", "id"):
            value = completed_by.get(key)
            if value is not None and str(value):
                return str(value)
    if completed_by is not None and str(completed_by):
        return str(completed_by)
    annotation_id = annotation.get("id")
    if annotation_id is not None and str(annotation_id):
        return str(annotation_id)
    return f"annotator-{fallback_index}"


def _parse_label_studio_span_result(
    result: Mapping[str, Any],
    document_text: str,
    *,
    document_id: str,
    annotator_id: str,
    language: str,
) -> tuple[str, EvalSpan]:
    source_id = str(result.get("id") or "")
    if not source_id:
        raise ValueError("Label Studio span result id is required")
    value = result.get("value") or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Label Studio span {source_id} value must be a mapping")
    start = _parse_int(value.get("start"), f"Label Studio span {source_id} start")
    end = _parse_int(value.get("end"), f"Label Studio span {source_id} end")
    surface = str(value.get("text") or document_text[start:end])
    raw_label = _first_label(value.get("labels") or result.get("labels"))
    _validate_import_offsets(
        document_text,
        start,
        end,
        surface,
        f"Label Studio span {source_id}",
    )
    return (
        source_id,
        EvalSpan(
            start=start,
            end=end,
            label=normalize_label(raw_label, language),
            text=surface,
            language=language,
            metadata={
                "annotator_id": annotator_id,
                "document_id": document_id,
                "from_name": result.get("from_name"),
                "source_annotation_id": source_id,
                "source_format": "label_studio",
                "source_label": raw_label,
                "to_name": result.get("to_name"),
            },
        ),
    )


def _parse_label_studio_relation_result(
    result: Mapping[str, Any],
    span_by_id: Mapping[str, EvalSpan],
    *,
    document_id: str,
    annotator_id: str,
) -> AnnotationRelation:
    relation_id = str(result.get("id") or "")
    if not relation_id:
        raise ValueError("Label Studio relation result id is required")
    source_id = str(result.get("from_id") or result.get("source") or "")
    target_id = str(result.get("to_id") or result.get("target") or "")
    value = result.get("value") or {}
    if isinstance(value, Mapping):
        source_id = str(value.get("from_id") or value.get("source") or source_id)
        target_id = str(value.get("to_id") or value.get("target") or target_id)
        raw_labels = value.get("labels") or result.get("labels")
    else:
        raw_labels = result.get("labels")
    relation_type = _first_label(raw_labels, default="RELATED_TO")
    return _build_annotation_relation(
        document_id=document_id,
        annotator_id=annotator_id,
        relation_id=relation_id,
        relation_type=relation_type,
        source_id=source_id,
        target_id=target_id,
        span_by_id=span_by_id,
        source_format="label_studio",
        metadata={
            "direction": result.get("direction"),
            "from_name": result.get("from_name"),
            "to_name": result.get("to_name"),
        },
    )


def _build_annotation_relation(
    *,
    document_id: str,
    annotator_id: str,
    relation_id: str,
    relation_type: str,
    source_id: str,
    target_id: str,
    span_by_id: Mapping[str, EvalSpan],
    source_format: str,
    metadata: Mapping[str, Any] | None = None,
) -> AnnotationRelation:
    if source_id not in span_by_id or target_id not in span_by_id:
        missing = [
            span_id
            for span_id in (source_id, target_id)
            if span_id and span_id not in span_by_id
        ]
        if not missing and (not source_id or not target_id):
            missing = ["source_id", "target_id"]
        raise ValueError(
            f"missing {source_format} relation endpoint(s): {', '.join(missing)}"
        )
    return AnnotationRelation(
        document_id=document_id,
        annotator_id=annotator_id,
        relation_id=relation_id,
        relation_type=relation_type,
        source_id=source_id,
        target_id=target_id,
        source_span=span_by_id[source_id],
        target_span=span_by_id[target_id],
        metadata={
            "source_format": source_format,
            "source_annotation_id": relation_id,
            **_plain_mapping(metadata or {}),
        },
    )


def _first_label(value: Any, *, default: str | None = None) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, str) and item:
                return item
    if default is not None:
        return default
    raise ValueError("annotation label is required")


def _validate_import_offsets(
    document_text: str,
    start: int,
    end: int,
    expected_text: str,
    context: str,
) -> None:
    if start < 0 or end <= start or end > len(document_text):
        raise ValueError(f"{context} has invalid offsets")
    actual_text = document_text[start:end]
    if actual_text != expected_text:
        raise ValueError(
            f"{context} span text mismatch: {expected_text!r} != {actual_text!r}"
        )


def _parse_int(value: Any, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _validate_raw_span_labels(raw_spans: list[Any], language: str) -> None:
    for raw_span in raw_spans:
        if not isinstance(raw_span, Mapping):
            raise ValueError("gold span must be a mapping")
        raw_label = raw_span.get("label") or raw_span.get("canonical_label")
        if not isinstance(raw_label, str):
            raise ValueError("gold span label is required")
        canonical = normalize_label(raw_label, language)
        if canonical != raw_label or canonical not in CANONICAL_LABELS:
            raise ValueError(f"gold span label must be canonical: {raw_label!r}")


def _validate_hard_negative_fixture(
    text: str,
    metadata: Mapping[str, Any],
    language: str,
) -> None:
    candidates = metadata.get("hard_negative_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "hard negative fixture metadata.hard_negative_candidates is required"
        )
    source = str(metadata.get("source") or metadata.get("source_dataset") or "")
    if _is_dua_source_marker(source):
        raise ValueError("hard negative fixtures must not reference DUA sources")

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("hard negative candidate must be a mapping")
        start = _int_field(candidate, "start")
        end = _int_field(candidate, "end")
        if start < 0 or end <= start or end > len(text):
            raise ValueError("hard negative candidate has invalid offsets")
        candidate_text = str(candidate.get("text", ""))
        if text[start:end] != candidate_text:
            raise ValueError("hard negative candidate text must match offsets")
        raw_label = candidate.get("label")
        if not isinstance(raw_label, str):
            raise ValueError("hard negative candidate label is required")
        canonical = normalize_label(raw_label, language)
        if canonical != raw_label or canonical not in CANONICAL_LABELS:
            raise ValueError(
                f"hard negative candidate label must be canonical: {raw_label!r}"
            )
        if candidate.get("synthetic") is not True:
            raise ValueError("hard negative candidate synthetic must be true")
        candidate_source = str(
            candidate.get("source_dataset")
            or candidate.get("source")
            or candidate.get("source_shard_id")
            or ""
        )
        if _is_dua_source_marker(candidate_source):
            raise ValueError("hard negative candidates must not reference DUA sources")
        difficulty = candidate.get("difficulty_score")
        if difficulty is not None:
            try:
                difficulty_value = float(difficulty)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "hard negative candidate difficulty_score must be numeric"
                ) from exc
            if not 0.0 <= difficulty_value <= 1.0:
                raise ValueError(
                    "hard negative candidate difficulty_score must be in [0, 1]"
                )


def _validate_offsets(text: str, spans: tuple[EvalSpan, ...]) -> None:
    for span in spans:
        if span.start < 0 or span.end <= span.start or span.end > len(text):
            raise ValueError(f"gold span has invalid offsets: {span!r}")
        actual_text = text[span.start : span.end]
        if span.text and actual_text != span.text:
            raise ValueError(
                f"gold span text mismatch for {span.label}: "
                f"{span.text!r} != {actual_text!r}"
            )


def _span_to_mapping(span: EvalSpan) -> dict[str, Any]:
    row: dict[str, Any] = {
        "start": span.start,
        "end": span.end,
        "label": span.label,
        "text": span.text,
    }
    metadata = dict(span.metadata)
    group = metadata.pop("group", None)
    if group is not None and str(group).strip():
        row["group"] = str(group).strip()
    if metadata:
        row["metadata"] = _plain_mapping(metadata)
    return row


def _int_field(payload: Mapping[str, Any], field: str) -> int:
    try:
        return int(payload[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _is_dua_source_marker(value: str) -> bool:
    markers = {"dua", "i2b2", "n2c2", "mimic"}
    parts = {
        part.strip().lower()
        for part in value.replace("_", "-").replace(".", "-").split("-")
    }
    return bool(parts & markers)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "AnnotationRelation",
    "GOLDEN_CATEGORIES",
    "HARD_NEGATIVE_CATEGORY",
    "GoldenFixture",
    "MultiAnnotatorGoldDocument",
    "benchmark_fixture_languages",
    "benchmark_fixtures_by_language",
    "fixture_languages",
    "fixtures_by_category",
    "fixtures_by_language",
    "list_fixture_paths",
    "load_brat_multi_annotator_document",
    "load_benchmark_fixtures",
    "load_golden_fixtures",
    "load_label_studio_multi_annotator_export",
    "parse_brat_multi_annotator",
    "parse_label_studio_multi_annotator_export",
]
