"""Loader for synthetic golden de-identification fixtures."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.pii_i18n import (
    INDIC_NER_LANGUAGES,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
)
from openmed.eval.golden.hard_negatives import HARD_NEGATIVE_CATEGORY
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import (
    CRITICAL_FINDING_CATEGORIES,
    EvalSpan,
    critical_finding_category,
    normalize_critical_finding_category,
    normalize_eval_spans,
)

CRITICAL_FINDINGS_CATEGORY = "critical_findings"
GOLDEN_CATEGORIES: tuple[str, ...] = (
    "nested_overlapping",
    "chunk_boundary",
    "multilingual",
    "checksum_ids",
    "financial_ids",
    "india_health_ids",
    "date_arithmetic",
    "policy_profile_actions",
    HARD_NEGATIVE_CATEGORY,
    CRITICAL_FINDINGS_CATEGORY,
)

_FIXTURE_VERSION = 1
_GOLDEN_DIR = Path(__file__).resolve().parent
_FIXTURE_DIR = _GOLDEN_DIR / "fixtures"
_TOP_LEVEL_FIXTURES: tuple[Path, ...] = (_GOLDEN_DIR / "financial_ids.jsonl",)
_SPECIALIZED_FIXTURE_NAMES = frozenset(
    {
        "code_mixed_hinglish.jsonl",
        "context_multilingual.jsonl",
        "code_mixed_deidentification.jsonl",
        "grounding_crosslingual.jsonl",
        "india_clinical.jsonl",
        "indic_name_variants.json",
        "relation_assertion.jsonl",
        "relation_gold.jsonl",
        "surrogate_multilingual.jsonl",
        "consensus_corpus.jsonl",
        # Domain eval fixtures that are not PII de-identification gold spans and
        # must not be loaded as such by load_golden_fixtures().
        "radiology_finding.jsonl",
        "radiology_report.jsonl",
        "cancer_staging.jsonl",
    }
)

#: Committed synthetic multi-annotator consensus corpus.
_CONSENSUS_CORPUS = _FIXTURE_DIR / "consensus_corpus.jsonl"


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
        fixture_languages = (
            SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES | INDIC_NER_LANGUAGES
        )
        if language not in fixture_languages:
            raise ValueError(f"unsupported golden fixture language: {language!r}")

        text = str(data.get("text", ""))
        if not text:
            raise ValueError("golden fixture text is required")

        fixture_id = str(data.get("id") or data.get("fixture_id") or "")
        if not fixture_id:
            raise ValueError("golden fixture id is required")

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
        if category == CRITICAL_FINDINGS_CATEGORY:
            gold_spans = _validate_critical_finding_fixture(
                fixture_id,
                text,
                metadata,
                gold_spans,
            )

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


def list_fixture_paths(path: str | Path | None = None) -> tuple[Path, ...]:
    """Return fixture paths in deterministic order."""
    fixture_path = Path(path) if path is not None else _FIXTURE_DIR
    if fixture_path.is_file():
        return (fixture_path,)
    paths = [
        *(
            path
            for path in fixture_path.glob("*.json")
            if path.name not in _SPECIALIZED_FIXTURE_NAMES
        ),
        *(
            path
            for path in fixture_path.glob("**/*.jsonl")
            if path.name not in _SPECIALIZED_FIXTURE_NAMES
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


def non_latin_golden_fixtures(
    fixtures: list[GoldenFixture] | None = None,
) -> list[GoldenFixture]:
    """Return synthetic golden fixtures containing non-Latin PHI spans."""
    source = fixtures if fixtures is not None else load_golden_fixtures()
    return sorted(
        (
            fixture
            for fixture in source
            if any(_has_non_latin_alpha(span.text) for span in fixture.gold_spans)
        ),
        key=lambda fixture: fixture.fixture_id,
    )


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


def _validate_critical_finding_fixture(
    fixture_id: str,
    text: str,
    metadata: Mapping[str, Any],
    spans: tuple[EvalSpan, ...],
) -> tuple[EvalSpan, ...]:
    disclaimer = str(metadata.get("medical_device_disclaimer") or "")
    normalized_disclaimer = disclaimer.lower()
    if (
        "assistive safety probe" not in normalized_disclaimer
        or "not clinical ground truth" not in normalized_disclaimer
    ):
        raise ValueError(
            "critical finding fixtures require a medical_device_disclaimer "
            "noting the set is an assistive safety probe, not clinical ground truth"
        )

    source = str(metadata.get("source") or metadata.get("source_dataset") or "")
    if _is_dua_source_marker(source):
        raise ValueError("critical finding fixtures must not reference DUA sources")

    validated: list[EvalSpan] = []
    for span in spans:
        category = critical_finding_category(span)
        if category is None:
            raise ValueError(
                "critical finding gold spans require critical_finding_category"
            )
        category = normalize_critical_finding_category(category)
        if category not in CRITICAL_FINDING_CATEGORIES:
            raise ValueError(f"unknown critical finding category: {category!r}")
        span_fixture_id = span.metadata.get("fixture_id")
        if span_fixture_id is not None and str(span_fixture_id) != fixture_id:
            raise ValueError("critical finding span fixture_id must match fixture id")
        span_metadata = dict(span.metadata)
        span_metadata["critical_finding"] = True
        span_metadata["critical_finding_category"] = category
        span_metadata["fixture_id"] = fixture_id
        validated.append(replace(span, metadata=span_metadata))

    if not validated:
        raise ValueError("critical finding fixture must include critical gold spans")
    _validate_offsets(text, tuple(validated))
    return tuple(validated)


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


# ---------------------------------------------------------------------------
# Multi-annotator consensus corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsensusRelation:
    """An adjudicated relation between two consensus spans."""

    relation_type: str
    head: EvalSpan
    tail: EvalSpan
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "label": self.label,
            "head": _span_to_mapping(self.head),
            "tail": _span_to_mapping(self.tail),
        }


@dataclass(frozen=True)
class ConsensusDocument:
    """A synthetic document with per-annotator exports and a consensus view."""

    doc_id: str
    text: str
    annotators: Mapping[str, tuple[EvalSpan, ...]]
    annotator_relations: Mapping[str, tuple[ConsensusRelation, ...]]
    consensus_spans: tuple[EvalSpan, ...]
    consensus_relations: tuple[ConsensusRelation, ...]


def _require_synthetic(payload: Mapping[str, Any], where: str) -> None:
    if payload.get("synthetic") is not True:
        raise ValueError(f"{where} must be explicitly marked synthetic")


def _consensus_spans(raw_spans: Any, text: str) -> tuple[EvalSpan, ...]:
    spans = tuple(normalize_eval_spans(raw_spans or [], source_text=text))
    _validate_offsets(text, spans)
    return spans


def _consensus_relation(raw: Mapping[str, Any], text: str) -> ConsensusRelation:
    if not isinstance(raw, Mapping):
        raise ValueError("consensus relation must be a mapping")
    head = _consensus_spans([raw["head"]], text)[0]
    tail = _consensus_spans([raw["tail"]], text)[0]
    relation_type = str(raw.get("relation_type", "")).strip()
    if not relation_type:
        raise ValueError("consensus relation type is required")
    return ConsensusRelation(
        relation_type=relation_type,
        head=head,
        tail=tail,
        label=str(raw.get("label", "")),
    )


def _span_key(span: EvalSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def _consensus_relations(
    raw_relations: Any,
    text: str,
    spans: tuple[EvalSpan, ...],
    where: str,
) -> tuple[ConsensusRelation, ...]:
    if raw_relations is None:
        return ()
    if not isinstance(raw_relations, list):
        raise ValueError(f"{where} relations must be a list")

    relations = tuple(_consensus_relation(relation, text) for relation in raw_relations)
    span_keys = {_span_key(span) for span in spans}
    for relation in relations:
        if _span_key(relation.head) not in span_keys:
            raise ValueError(f"{where} relation head must reference one of its spans")
        if _span_key(relation.tail) not in span_keys:
            raise ValueError(f"{where} relation tail must reference one of its spans")
    return relations


def _consensus_document(data: Mapping[str, Any]) -> ConsensusDocument:
    if not isinstance(data, Mapping):
        raise ValueError("consensus record must be a mapping")
    _require_synthetic(data, "consensus document")

    text = str(data.get("text", ""))
    if not text:
        raise ValueError("consensus document text is required")

    raw_annotators = data.get("annotators") or {}
    if not isinstance(raw_annotators, Mapping) or len(raw_annotators) < 2:
        raise ValueError("consensus document requires at least two annotators")

    annotators: dict[str, tuple[EvalSpan, ...]] = {}
    annotator_relations: dict[str, tuple[ConsensusRelation, ...]] = {}
    for name, export in raw_annotators.items():
        if not isinstance(export, Mapping):
            raise ValueError("annotator export must be a mapping")
        _require_synthetic(export, f"annotator {name!r} export")
        annotator_name = str(name)
        annotator_spans = _consensus_spans(export.get("spans"), text)
        annotators[annotator_name] = annotator_spans
        annotator_relations[annotator_name] = _consensus_relations(
            export.get("relations"),
            text,
            annotator_spans,
            f"annotator {name!r}",
        )

    consensus = data.get("consensus") or {}
    if not isinstance(consensus, Mapping):
        raise ValueError("consensus view must be a mapping")
    consensus_spans = _consensus_spans(consensus.get("spans"), text)
    consensus_relations = _consensus_relations(
        consensus.get("relations"),
        text,
        consensus_spans,
        "consensus",
    )

    doc_id = str(data.get("id") or data.get("doc_id") or "")
    if not doc_id:
        raise ValueError("consensus document id is required")

    return ConsensusDocument(
        doc_id=doc_id,
        text=text,
        annotators=annotators,
        annotator_relations=annotator_relations,
        consensus_spans=consensus_spans,
        consensus_relations=consensus_relations,
    )


def load_consensus_corpus(
    path: str | Path | None = None,
) -> list[ConsensusDocument]:
    """Load the synthetic multi-annotator consensus corpus.

    Each record carries source text, at least two synthetic annotator exports,
    and an adjudicated consensus view of spans and relations. Every document and
    annotation must be explicitly marked synthetic, and all span offsets are
    validated against the document text.
    """

    corpus_path = Path(path) if path is not None else _CONSENSUS_CORPUS
    documents: list[ConsensusDocument] = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            documents.append(_consensus_document(json.loads(line)))
    return documents


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


def _has_non_latin_alpha(value: str) -> bool:
    return any(ord(char) > 127 and char.isalpha() for char in value)


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "CRITICAL_FINDINGS_CATEGORY",
    "GOLDEN_CATEGORIES",
    "HARD_NEGATIVE_CATEGORY",
    "ConsensusDocument",
    "ConsensusRelation",
    "GoldenFixture",
    "benchmark_fixture_languages",
    "benchmark_fixtures_by_language",
    "fixture_languages",
    "fixtures_by_category",
    "fixtures_by_language",
    "list_fixture_paths",
    "load_benchmark_fixtures",
    "load_consensus_corpus",
    "load_golden_fixtures",
    "non_latin_golden_fixtures",
]
