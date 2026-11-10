"""Streaming hard-negative mining for synthetic golden fixtures."""

from __future__ import annotations

import hashlib
import json
import re
import tracemalloc
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openmed.core.labels import (
    ACCOUNT_NUMBER,
    AGE,
    CREDIT_CARD,
    DATE,
    EMAIL,
    ID_NUM,
    LOCATION,
    ORGANIZATION,
    PERSON,
    PHONE,
    SSN,
    STREET_ADDRESS,
    URL,
    normalize_label,
)
from openmed.eval.harness import BenchmarkFixture, ModelRunner, default_model_runner
from openmed.eval.metrics import EvalSpan, normalize_eval_spans

HARD_NEGATIVE_CATEGORY = "hard_negatives"
HARD_NEGATIVE_PACK_VERSION = 1
DEFAULT_PER_LABEL_LANGUAGE_LIMIT = 3
DEFAULT_MAX_REFERENCE_SURFACES_PER_LABEL = 128
DEFAULT_MAX_CANDIDATES = 2048
DEFAULT_MIN_DIFFICULTY_SCORE = 0.25
DEFAULT_NEAR_DUPLICATE_SIMILARITY = 0.86
DEFAULT_NEAR_DUPLICATE_RETENTION_CEILING = 0.02
DEFAULT_LSH_BANDS = 6
DEFAULT_LSH_ROWS = 4

_DUA_SOURCE_MARKERS = frozenset({"dua", "i2b2", "n2c2", "mimic"})
_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"https?://[^\s,;]+", re.IGNORECASE), URL, "url_like"),
    (
        re.compile(
            r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])",
            re.IGNORECASE,
        ),
        EMAIL,
        "email_like",
    ),
    (
        re.compile(r"\b\d{3}[- ]\d{2}[- ]\d{4}\b"),
        SSN,
        "ssn_like",
    ),
    (
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        CREDIT_CARD,
        "card_like",
    ),
    (
        re.compile(r"\b[A-Z]{2,5}[- ]?\d{3,8}[A-Z]?\b"),
        ID_NUM,
        "identifier_like",
    ),
    (
        re.compile(r"\b[A-Za-z]\d[-/]\d{2,4}\b"),
        DATE,
        "date_like_code",
    ),
    (
        re.compile(r"\b\d{1,4}[-/]\d{1,4}(?:[-/]\d{1,4})?\b"),
        DATE,
        "numeric_date_like",
    ),
    (
        re.compile(r"\b\d{2,3}\s?(?:years?|yrs?|yo|y/o)\b", re.IGNORECASE),
        AGE,
        "age_like",
    ),
    (
        re.compile(r"\b(?:\+?\d[\d(). -]{6,}\d)\b"),
        PHONE,
        "phone_like",
    ),
    (
        re.compile(r"\b(?:Room|Suite|Unit|Ward)\s+[A-Z0-9/-]{2,12}\b"),
        ACCOUNT_NUMBER,
        "account_like_location",
    ),
    (
        re.compile(r"\b\d{1,5}\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}\b"),
        STREET_ADDRESS,
        "address_like",
    ),
    (
        re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"),
        PERSON,
        "name_like",
    ),
)


class MemoryBudgetExceeded(RuntimeError):
    """Raised when hard-negative mining exceeds its configured memory budget."""


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    """Configuration for deterministic synthetic hard-negative mining."""

    seed: int = 0
    per_label_language_limit: int = DEFAULT_PER_LABEL_LANGUAGE_LIMIT
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    max_reference_surfaces_per_label: int = DEFAULT_MAX_REFERENCE_SURFACES_PER_LABEL
    min_difficulty_score: float = DEFAULT_MIN_DIFFICULTY_SCORE
    model_false_positive_weight: float = 0.65
    lexical_proximity_weight: float = 0.35
    near_duplicate_similarity: float = DEFAULT_NEAR_DUPLICATE_SIMILARITY
    near_duplicate_retention_ceiling: float = DEFAULT_NEAR_DUPLICATE_RETENTION_CEILING
    lsh_bands: int = DEFAULT_LSH_BANDS
    lsh_rows: int = DEFAULT_LSH_ROWS
    peak_memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.per_label_language_limit <= 0:
            raise ValueError("per_label_language_limit must be positive")
        if self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if self.max_reference_surfaces_per_label <= 0:
            raise ValueError("max_reference_surfaces_per_label must be positive")
        if not 0.0 <= self.min_difficulty_score <= 1.0:
            raise ValueError("min_difficulty_score must be between 0 and 1")
        if self.model_false_positive_weight < 0:
            raise ValueError("model_false_positive_weight must be non-negative")
        if self.lexical_proximity_weight < 0:
            raise ValueError("lexical_proximity_weight must be non-negative")
        if self.model_false_positive_weight + self.lexical_proximity_weight <= 0:
            raise ValueError("at least one difficulty weight must be positive")
        if not 0.0 <= self.near_duplicate_similarity <= 1.0:
            raise ValueError("near_duplicate_similarity must be between 0 and 1")
        if not 0.0 <= self.near_duplicate_retention_ceiling <= 1.0:
            raise ValueError("near_duplicate_retention_ceiling must be between 0 and 1")
        if self.lsh_bands <= 0 or self.lsh_rows <= 0:
            raise ValueError("lsh_bands and lsh_rows must be positive")
        if (
            self.peak_memory_budget_bytes is not None
            and self.peak_memory_budget_bytes <= 0
        ):
            raise ValueError("peak_memory_budget_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready configuration payload."""
        return {
            "lsh_bands": self.lsh_bands,
            "lsh_rows": self.lsh_rows,
            "max_candidates": self.max_candidates,
            "max_reference_surfaces_per_label": (self.max_reference_surfaces_per_label),
            "min_difficulty_score": _round_score(self.min_difficulty_score),
            "model_false_positive_weight": _round_score(
                self.model_false_positive_weight
            ),
            "near_duplicate_retention_ceiling": _round_score(
                self.near_duplicate_retention_ceiling
            ),
            "near_duplicate_similarity": _round_score(self.near_duplicate_similarity),
            "peak_memory_budget_bytes": self.peak_memory_budget_bytes,
            "per_label_language_limit": self.per_label_language_limit,
            "lexical_proximity_weight": _round_score(self.lexical_proximity_weight),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class HardNegativeCandidate:
    """One synthetic non-PHI token that a model is likely to over-redact."""

    candidate_id: str
    surface: str
    label: str
    language: str
    difficulty_score: float
    model_false_positive_score: float
    lexical_proximity_score: float
    normalized_surface: str
    source_fixture_id: str
    source_shard_id: str
    start: int
    end: int
    reason: str
    synthetic: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_surface: bool = True) -> dict[str, Any]:
        """Return a deterministic JSON-ready candidate mapping."""
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "difficulty_score": _round_score(self.difficulty_score),
            "end": self.end,
            "label": self.label,
            "language": self.language,
            "lexical_proximity_score": _round_score(self.lexical_proximity_score),
            "model_false_positive_score": _round_score(self.model_false_positive_score),
            "normalized_surface": self.normalized_surface,
            "reason": self.reason,
            "source_fixture_id": self.source_fixture_id,
            "source_shard_id": self.source_shard_id,
            "start": self.start,
            "surface_sha256": _sha256(self.surface),
            "synthetic": self.synthetic,
        }
        if include_surface:
            payload["text"] = self.surface
        if self.metadata:
            payload["metadata"] = _plain_mapping(self.metadata)
        return payload


@dataclass(frozen=True)
class HardNegativeDifficultyReport:
    """Aggregate difficulty distribution for mined hard negatives."""

    candidate_count: int
    selected_count: int
    min_score: float
    max_score: float
    mean_score: float
    score_buckets: Mapping[str, int]
    label_counts: Mapping[str, int]
    language_counts: Mapping[str, int]
    near_duplicate_retention_rate: float
    duplicate_candidates_seen: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report payload."""
        return {
            "candidate_count": self.candidate_count,
            "duplicate_candidates_seen": self.duplicate_candidates_seen,
            "label_counts": {
                key: int(self.label_counts.get(key, 0))
                for key in sorted(self.label_counts)
            },
            "language_counts": {
                key: int(self.language_counts.get(key, 0))
                for key in sorted(self.language_counts)
            },
            "max_score": _round_score(self.max_score),
            "mean_score": _round_score(self.mean_score),
            "min_score": _round_score(self.min_score),
            "near_duplicate_retention_rate": _round_score(
                self.near_duplicate_retention_rate
            ),
            "score_buckets": {
                key: int(self.score_buckets.get(key, 0))
                for key in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")
            },
            "selected_count": self.selected_count,
        }

    def to_markdown(self) -> str:
        """Render a deterministic aggregate-only Markdown report."""
        lines = [
            "# Hard Negative Difficulty Distribution",
            "",
            "## Summary",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Candidates | {self.candidate_count} |",
            f"| Selected | {self.selected_count} |",
            f"| Min Score | {_round_score(self.min_score):.6f} |",
            f"| Mean Score | {_round_score(self.mean_score):.6f} |",
            f"| Max Score | {_round_score(self.max_score):.6f} |",
            (
                "| Near-Duplicate Retention Rate | "
                f"{_round_score(self.near_duplicate_retention_rate):.6f} |"
            ),
            f"| Duplicate Candidates Seen | {self.duplicate_candidates_seen} |",
            "",
            "## Score Buckets",
            "",
            "| Bucket | Count |",
            "|---|---:|",
        ]
        for bucket, count in self.to_dict()["score_buckets"].items():
            lines.append(f"| `{bucket}` | {count} |")
        lines.extend(["", "## Labels", "", "| Label | Count |", "|---|---:|"])
        for label, count in self.to_dict()["label_counts"].items():
            lines.append(f"| `{label}` | {count} |")
        lines.extend(["", "## Languages", "", "| Language | Count |", "|---|---:|"])
        for language, count in self.to_dict()["language_counts"].items():
            lines.append(f"| `{language}` | {count} |")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class HardNegativeMiningResult:
    """Result bundle returned by the hard-negative miner."""

    candidates: tuple[HardNegativeCandidate, ...]
    selected_candidates: tuple[HardNegativeCandidate, ...]
    difficulty_report: HardNegativeDifficultyReport
    scanned_records: int
    scanned_candidates: int
    peak_memory_bytes: int
    config: HardNegativeMiningConfig

    def to_dict(self, *, include_surfaces: bool = True) -> dict[str, Any]:
        """Return a deterministic JSON-ready mining result."""
        return {
            "candidates": [
                candidate.to_dict(include_surface=include_surfaces)
                for candidate in self.candidates
            ],
            "config": self.config.to_dict(),
            "difficulty_report": self.difficulty_report.to_dict(),
            "peak_memory_bytes": self.peak_memory_bytes,
            "scanned_candidates": self.scanned_candidates,
            "scanned_records": self.scanned_records,
            "selected_candidates": [
                candidate.to_dict(include_surface=include_surfaces)
                for candidate in self.selected_candidates
            ],
        }

    def to_json(self, *, include_surfaces: bool = True, indent: int = 2) -> str:
        """Serialize the result to deterministic JSON."""
        return json.dumps(
            self.to_dict(include_surfaces=include_surfaces),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def ranking_to_dict(self, *, include_surfaces: bool = True) -> dict[str, Any]:
        """Return only byte-stable difficulty rankings and distribution data."""
        return {
            "candidates": [
                candidate.to_dict(include_surface=include_surfaces)
                for candidate in self.candidates
            ],
            "difficulty_report": self.difficulty_report.to_dict(),
            "selected_candidates": [
                candidate.to_dict(include_surface=include_surfaces)
                for candidate in self.selected_candidates
            ],
        }

    def ranking_to_json(
        self,
        *,
        include_surfaces: bool = True,
        indent: int = 2,
    ) -> str:
        """Serialize byte-stable difficulty rankings to JSON."""
        return json.dumps(
            self.ranking_to_dict(include_surfaces=include_surfaces),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(
        self,
        path: str | Path,
        *,
        include_surfaces: bool = True,
        indent: int = 2,
    ) -> Path:
        """Write deterministic JSON to *path*."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.to_json(include_surfaces=include_surfaces, indent=indent) + "\n",
            encoding="utf-8",
        )
        return output_path


def iter_synthetic_corpus_shard(
    shard: str | Path | Iterable[Mapping[str, Any]],
    *,
    shard_id: str | None = None,
) -> Iterator[tuple[str, int, Mapping[str, Any]]]:
    """Yield synthetic records from one path or in-memory shard."""
    if isinstance(shard, (str, Path)):
        path = Path(shard)
        active_shard_id = shard_id or path.name
        if path.suffix.lower() == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, Mapping):
                        raise ValueError(f"{path}:{line_number} must be a mapping")
                    _validate_synthetic_record(row, active_shard_id)
                    yield active_shard_id, line_number, row
            return

        raw = json.loads(path.read_text(encoding="utf-8"))
        rows = raw.get("fixtures") if isinstance(raw, Mapping) else raw
        if not isinstance(rows, list):
            raise ValueError(f"{path} must contain a fixtures list")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{path} fixture {index} must be a mapping")
            _validate_synthetic_record(row, active_shard_id)
            yield active_shard_id, index, row
        return

    active_shard_id = shard_id or "memory"
    for index, row in enumerate(shard):
        if not isinstance(row, Mapping):
            raise ValueError(f"{active_shard_id} record {index} must be a mapping")
        _validate_synthetic_record(row, active_shard_id)
        yield active_shard_id, index, row


def mine_hard_negative_candidates(
    shards: Sequence[str | Path | Iterable[Mapping[str, Any]]],
    *,
    config: HardNegativeMiningConfig | None = None,
    model: str = "hard-negative-miner",
    device: str = "cpu",
    runner: ModelRunner | None = None,
    reference_surfaces: Mapping[str, Sequence[str]] | None = None,
) -> HardNegativeMiningResult:
    """Mine synthetic hard negatives without loading all shards into memory."""
    active_config = config or HardNegativeMiningConfig()
    active_runner = runner or default_model_runner
    references = _reference_store(reference_surfaces)
    dedup = _CandidateDeduplicator(active_config)
    scanned_records = 0
    scanned_candidates = 0

    tracemalloc.start()
    try:
        for shard_index, shard in enumerate(shards):
            for shard_id, record_index, row in iter_synthetic_corpus_shard(
                shard,
                shard_id=f"shard-{shard_index}",
            ):
                scanned_records += 1
                fixture = BenchmarkFixture.from_mapping(row)
                gold_spans = tuple(fixture.gold_spans)
                _record_reference_surfaces(
                    references,
                    gold_spans,
                    limit=active_config.max_reference_surfaces_per_label,
                )
                predictions = tuple(
                    normalize_eval_spans(
                        list(active_runner(fixture, model, device)),
                        default_language=fixture.language,
                        default_device=device,
                        source_text=fixture.text,
                    )
                )

                for raw_candidate in _candidate_spans(fixture, gold_spans):
                    scanned_candidates += 1
                    candidate = _score_candidate(
                        raw_candidate=raw_candidate,
                        fixture=fixture,
                        shard_id=shard_id,
                        record_index=record_index,
                        predictions=predictions,
                        references=references,
                        config=active_config,
                    )
                    if candidate.difficulty_score < active_config.min_difficulty_score:
                        continue
                    dedup.add(candidate)
                _assert_memory_budget(active_config)
        candidates = tuple(sorted(dedup.candidates(), key=_candidate_sort_key))
        selected = stratified_hard_negative_sample(
            candidates,
            per_label_language_limit=active_config.per_label_language_limit,
            seed=active_config.seed,
        )
        retention_rate = near_duplicate_retention_rate(
            selected,
            similarity_threshold=active_config.near_duplicate_similarity,
        )
        if retention_rate > active_config.near_duplicate_retention_ceiling:
            raise ValueError(
                "near-duplicate retention rate exceeds configured ceiling: "
                f"{retention_rate:.6f} > "
                f"{active_config.near_duplicate_retention_ceiling:.6f}"
            )
        _assert_memory_budget(active_config)
        _, peak_memory_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    report = hard_negative_difficulty_report(
        candidates,
        selected,
        near_duplicate_retention_rate=retention_rate,
        duplicate_candidates_seen=dedup.duplicate_candidates_seen,
    )
    return HardNegativeMiningResult(
        candidates=candidates,
        selected_candidates=selected,
        difficulty_report=report,
        scanned_records=scanned_records,
        scanned_candidates=scanned_candidates,
        peak_memory_bytes=int(peak_memory_bytes),
        config=active_config,
    )


def stratified_hard_negative_sample(
    candidates: Sequence[HardNegativeCandidate],
    *,
    per_label_language_limit: int = DEFAULT_PER_LABEL_LANGUAGE_LIMIT,
    seed: int = 0,
) -> tuple[HardNegativeCandidate, ...]:
    """Return the hardest distinct candidates per label and language."""
    if per_label_language_limit <= 0:
        raise ValueError("per_label_language_limit must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    grouped: dict[tuple[str, str], list[HardNegativeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.label, candidate.language)].append(candidate)

    selected: list[HardNegativeCandidate] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda candidate: _candidate_sample_key(candidate, seed),
        )
        selected.extend(ranked[:per_label_language_limit])
    return tuple(sorted(selected, key=_candidate_sort_key))


def hard_negative_difficulty_report(
    candidates: Sequence[HardNegativeCandidate],
    selected_candidates: Sequence[HardNegativeCandidate] | None = None,
    *,
    near_duplicate_retention_rate: float | None = None,
    duplicate_candidates_seen: int = 0,
) -> HardNegativeDifficultyReport:
    """Build an aggregate difficulty report for candidate rankings."""
    selected = tuple(selected_candidates or ())
    scores = [candidate.difficulty_score for candidate in candidates]
    label_counts = Counter(candidate.label for candidate in selected)
    language_counts = Counter(candidate.language for candidate in selected)
    if near_duplicate_retention_rate is None:
        near_duplicate_retention_rate = globals()["near_duplicate_retention_rate"](
            selected
        )
    return HardNegativeDifficultyReport(
        candidate_count=len(candidates),
        selected_count=len(selected),
        min_score=min(scores) if scores else 0.0,
        max_score=max(scores) if scores else 0.0,
        mean_score=(sum(scores) / len(scores)) if scores else 0.0,
        score_buckets=_score_buckets(scores),
        label_counts=label_counts,
        language_counts=language_counts,
        near_duplicate_retention_rate=near_duplicate_retention_rate,
        duplicate_candidates_seen=duplicate_candidates_seen,
    )


def build_hard_negative_fixture_pack(
    candidates: Sequence[HardNegativeCandidate],
    *,
    pack_id: str = "om-627-hard-negatives-v1",
) -> dict[str, Any]:
    """Build a synthetic golden fixture pack from selected candidates."""
    fixtures = []
    for index, candidate in enumerate(sorted(candidates, key=_candidate_sort_key)):
        text = f"{candidate.surface} remained visible in the synthetic note."
        fixture_id = f"golden-hard-negative-{candidate.label.lower()}-{index:03d}"
        fixtures.append(
            {
                "gold_spans": [],
                "id": fixture_id,
                "language": candidate.language,
                "metadata": {
                    "category": HARD_NEGATIVE_CATEGORY,
                    "expected_output": {
                        "method": "none",
                        "text": text,
                    },
                    "hard_negative_candidates": [
                        {
                            **candidate.to_dict(include_surface=True),
                            "start": 0,
                            "end": len(candidate.surface),
                        }
                    ],
                    "hard_negative_pack": pack_id,
                    "synthetic": True,
                },
                "text": text,
            }
        )
    return {
        "fixtures": fixtures,
        "notice": (
            "Synthetic-only hard-negative golden fixtures; no DUA data and no real PHI."
        ),
        "suite": "golden",
        "synthetic": True,
        "version": HARD_NEGATIVE_PACK_VERSION,
    }


def write_hard_negative_fixture_pack(
    candidates: Sequence[HardNegativeCandidate],
    path: str | Path,
    *,
    pack_id: str = "om-627-hard-negatives-v1",
    indent: int = 2,
) -> Path:
    """Write selected hard negatives as a committed fixture pack."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_hard_negative_fixture_pack(candidates, pack_id=pack_id)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def near_duplicate_retention_rate(
    candidates: Sequence[HardNegativeCandidate],
    *,
    similarity_threshold: float = DEFAULT_NEAR_DUPLICATE_SIMILARITY,
) -> float:
    """Return the retained near-duplicate pair rate among selected candidates."""
    if not candidates:
        return 0.0
    retained_pairs = 0
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if left.label != right.label or left.language != right.language:
                continue
            if (
                _surface_similarity(left.normalized_surface, right.normalized_surface)
                >= similarity_threshold
            ):
                retained_pairs += 1
    return retained_pairs / len(candidates)


def _score_candidate(
    *,
    raw_candidate: Mapping[str, Any],
    fixture: BenchmarkFixture,
    shard_id: str,
    record_index: int,
    predictions: Sequence[EvalSpan],
    references: Mapping[str, Sequence[str]],
    config: HardNegativeMiningConfig,
) -> HardNegativeCandidate:
    surface = str(raw_candidate["text"])
    label = normalize_label(str(raw_candidate["label"]), fixture.language)
    normalized_surface = _normalize_surface(surface)
    model_score = _model_false_positive_score(raw_candidate, predictions)
    lexical_score = _lexical_proximity_score(
        normalized_surface,
        label=label,
        references=references,
    )
    weight_total = config.model_false_positive_weight + config.lexical_proximity_weight
    difficulty_score = _round_score(
        (
            (model_score * config.model_false_positive_weight)
            + (lexical_score * config.lexical_proximity_weight)
        )
        / weight_total
    )
    start = int(raw_candidate["start"])
    end = int(raw_candidate["end"])
    reason = str(raw_candidate.get("reason") or "confusable_non_phi")
    candidate_id = _candidate_id(
        seed=config.seed,
        fixture_id=fixture.fixture_id,
        shard_id=shard_id,
        record_index=record_index,
        label=label,
        language=fixture.language,
        start=start,
        end=end,
        surface=surface,
    )
    return HardNegativeCandidate(
        candidate_id=candidate_id,
        surface=surface,
        label=label,
        language=fixture.language,
        difficulty_score=difficulty_score,
        model_false_positive_score=model_score,
        lexical_proximity_score=lexical_score,
        normalized_surface=normalized_surface,
        source_fixture_id=fixture.fixture_id,
        source_shard_id=shard_id,
        start=start,
        end=end,
        reason=reason,
        metadata={
            "record_index": record_index,
            "source": "synthetic_corpus",
        },
    )


def _candidate_spans(
    fixture: BenchmarkFixture,
    gold_spans: Sequence[EvalSpan],
) -> Iterator[Mapping[str, Any]]:
    yielded: set[tuple[int, int, str]] = set()
    for annotated in _annotated_negative_candidates(fixture):
        start = int(annotated["start"])
        end = int(annotated["end"])
        label = normalize_label(str(annotated["label"]), fixture.language)
        if _overlaps_any(start, end, gold_spans):
            continue
        key = (start, end, label)
        if key in yielded:
            continue
        yielded.add(key)
        yield {
            "end": end,
            "label": label,
            "reason": str(annotated.get("reason") or "annotated_non_phi"),
            "start": start,
            "text": fixture.text[start:end],
        }

    for pattern, label, reason in _TOKEN_PATTERNS:
        for match in pattern.finditer(fixture.text):
            start, end = match.span()
            surface = match.group(0).strip()
            if not surface or _overlaps_any(start, end, gold_spans):
                continue
            normalized_label = _refine_label(label, surface)
            key = (start, end, normalized_label)
            if key in yielded:
                continue
            yielded.add(key)
            yield {
                "end": end,
                "label": normalized_label,
                "reason": reason,
                "start": start,
                "text": surface,
            }


def _annotated_negative_candidates(
    fixture: BenchmarkFixture,
) -> Iterator[Mapping[str, Any]]:
    for key in ("hard_negative_candidates", "negative_candidates"):
        rows = fixture.metadata.get(key)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if "start" in row and "end" in row:
                start = int(row["start"])
                end = int(row["end"])
                text = fixture.text[start:end]
            else:
                text = str(row.get("text", ""))
                start = fixture.text.find(text)
                end = start + len(text)
            if start < 0 or end <= start or end > len(fixture.text):
                continue
            if not isinstance(row.get("label"), str):
                continue
            yield {
                "end": end,
                "label": row["label"],
                "reason": row.get("reason") or "annotated_non_phi",
                "start": start,
                "text": text,
            }


def _validate_synthetic_record(row: Mapping[str, Any], shard_id: str) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{shard_id} record metadata must be a mapping")
    if metadata.get("synthetic") is not True:
        raise ValueError(f"{shard_id} record must be marked synthetic")
    source = str(metadata.get("source_dataset") or metadata.get("source") or "")
    source_parts = {part.strip().lower() for part in re.split(r"[^a-zA-Z0-9]+", source)}
    if source_parts & _DUA_SOURCE_MARKERS:
        raise ValueError(f"{shard_id} record must not reference DUA-sourced text")


def _reference_store(
    values: Mapping[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    references: dict[str, list[str]] = defaultdict(list)
    for label, surfaces in (values or {}).items():
        canonical = normalize_label(label, "en")
        for surface in surfaces:
            normalized = _normalize_surface(str(surface))
            if normalized:
                references[canonical].append(normalized)
    return references


def _record_reference_surfaces(
    references: dict[str, list[str]],
    gold_spans: Sequence[EvalSpan],
    *,
    limit: int,
) -> None:
    for span in gold_spans:
        normalized = _normalize_surface(span.text)
        if not normalized:
            continue
        values = references[span.label]
        if len(values) < limit and normalized not in values:
            values.append(normalized)


def _model_false_positive_score(
    candidate: Mapping[str, Any],
    predictions: Sequence[EvalSpan],
) -> float:
    start = int(candidate["start"])
    end = int(candidate["end"])
    label = str(candidate["label"])
    score = 0.0
    for prediction in predictions:
        if not _spans_overlap(start, end, prediction.start, prediction.end):
            continue
        confidence = _coerce_probability(prediction.metadata.get("confidence", 1.0))
        label_factor = 1.0 if prediction.label == label else 0.75
        overlap = max(min(end, prediction.end) - max(start, prediction.start), 0)
        coverage = overlap / max(end - start, 1)
        score = max(score, confidence * label_factor * coverage)
    return _round_score(score)


def _lexical_proximity_score(
    normalized_surface: str,
    *,
    label: str,
    references: Mapping[str, Sequence[str]],
) -> float:
    reference_score = 0.0
    for reference in references.get(label, ()):
        reference_score = max(
            reference_score,
            _surface_similarity(normalized_surface, reference),
        )
    return _round_score(max(reference_score, _shape_confusability(normalized_surface)))


def _shape_confusability(normalized_surface: str) -> float:
    if not normalized_surface:
        return 0.0
    has_digit = any(char.isdigit() for char in normalized_surface)
    has_alpha = any(char.isalpha() for char in normalized_surface)
    separators = sum(1 for char in normalized_surface if not char.isalnum())
    digit_ratio = sum(char.isdigit() for char in normalized_surface) / len(
        normalized_surface
    )
    score = 0.15
    if has_digit and has_alpha:
        score += 0.25
    if has_digit and separators:
        score += 0.25
    if digit_ratio >= 0.45:
        score += 0.2
    if 6 <= len(normalized_surface) <= 32:
        score += 0.15
    return min(score, 1.0)


def _surface_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    left_shingles = _char_shingles(left)
    right_shingles = _char_shingles(right)
    if left_shingles or right_shingles:
        intersection = len(left_shingles & right_shingles)
        union = len(left_shingles | right_shingles)
        jaccard = intersection / union if union else 0.0
    else:
        jaccard = 0.0
    ratio = SequenceMatcher(a=left, b=right).ratio()
    return max(jaccard, ratio)


class _CandidateDeduplicator:
    def __init__(self, config: HardNegativeMiningConfig) -> None:
        self._config = config
        self._candidates: dict[str, HardNegativeCandidate] = {}
        self._index: dict[tuple[Any, ...], set[str]] = defaultdict(set)
        self.duplicate_candidates_seen = 0

    def add(self, candidate: HardNegativeCandidate) -> None:
        near_ids = self._near_candidate_ids(candidate)
        for existing_id in near_ids:
            existing = self._candidates.get(existing_id)
            if existing is None:
                continue
            if (
                _surface_similarity(
                    candidate.normalized_surface,
                    existing.normalized_surface,
                )
                < self._config.near_duplicate_similarity
            ):
                continue
            self.duplicate_candidates_seen += 1
            if _candidate_quality_key(candidate) > _candidate_quality_key(existing):
                self._remove(existing)
                self._insert(candidate)
            return
        self._insert(candidate)
        if len(self._candidates) > self._config.max_candidates:
            worst = max(self._candidates.values(), key=_candidate_sort_key)
            self._remove(worst)

    def candidates(self) -> tuple[HardNegativeCandidate, ...]:
        return tuple(self._candidates.values())

    def _near_candidate_ids(self, candidate: HardNegativeCandidate) -> tuple[str, ...]:
        ids: set[str] = set()
        for key in _lsh_keys(candidate.normalized_surface, self._config):
            ids.update(self._index.get((candidate.label, candidate.language, *key), ()))
        return tuple(sorted(ids))

    def _insert(self, candidate: HardNegativeCandidate) -> None:
        self._candidates[candidate.candidate_id] = candidate
        for key in _lsh_keys(candidate.normalized_surface, self._config):
            self._index[(candidate.label, candidate.language, *key)].add(
                candidate.candidate_id
            )

    def _remove(self, candidate: HardNegativeCandidate) -> None:
        self._candidates.pop(candidate.candidate_id, None)
        for key in _lsh_keys(candidate.normalized_surface, self._config):
            index_key = (candidate.label, candidate.language, *key)
            ids = self._index.get(index_key)
            if not ids:
                continue
            ids.discard(candidate.candidate_id)
            if not ids:
                self._index.pop(index_key, None)


def _lsh_keys(
    normalized_surface: str,
    config: HardNegativeMiningConfig,
) -> tuple[tuple[Any, ...], ...]:
    signature = _minhash_signature(
        normalized_surface,
        size=config.lsh_bands * config.lsh_rows,
        seed=config.seed,
    )
    keys = []
    for band in range(config.lsh_bands):
        start = band * config.lsh_rows
        end = start + config.lsh_rows
        keys.append((band, *signature[start:end]))
    normalized_digits = re.sub(r"\D+", "", normalized_surface)
    if normalized_digits:
        keys.append(("digits", normalized_digits[:8]))
    compact = re.sub(r"[^a-z0-9]+", "", normalized_surface)
    if compact:
        keys.append(("compact", compact[:10]))
    return tuple(keys)


def _minhash_signature(
    normalized_surface: str,
    *,
    size: int,
    seed: int,
) -> tuple[int, ...]:
    shingles = _char_shingles(normalized_surface) or {normalized_surface}
    signature: list[int] = []
    for index in range(size):
        min_hash = min(
            int.from_bytes(
                hashlib.blake2b(
                    f"{seed}:{index}:{shingle}".encode("utf-8"),
                    digest_size=8,
                ).digest(),
                "big",
            )
            for shingle in shingles
        )
        signature.append(min_hash)
    return tuple(signature)


def _score_buckets(scores: Sequence[float]) -> dict[str, int]:
    buckets = {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 0,
        "0.75-1.00": 0,
    }
    for score in scores:
        if score < 0.25:
            buckets["0.00-0.25"] += 1
        elif score < 0.50:
            buckets["0.25-0.50"] += 1
        elif score < 0.75:
            buckets["0.50-0.75"] += 1
        else:
            buckets["0.75-1.00"] += 1
    return buckets


def _refine_label(label: str, surface: str) -> str:
    if label == PERSON and any(
        marker in surface.lower() for marker in ("clinic", "ward", "hospital")
    ):
        return ORGANIZATION
    if label == STREET_ADDRESS and any(
        marker in surface.lower() for marker in ("room", "suite", "ward", "unit")
    ):
        return LOCATION
    return label


def _candidate_sort_key(candidate: HardNegativeCandidate) -> tuple[Any, ...]:
    return (
        -candidate.difficulty_score,
        -candidate.model_false_positive_score,
        -candidate.lexical_proximity_score,
        candidate.label,
        candidate.language,
        candidate.normalized_surface,
        candidate.candidate_id,
    )


def _candidate_sample_key(
    candidate: HardNegativeCandidate,
    seed: int,
) -> tuple[Any, ...]:
    return (
        -candidate.difficulty_score,
        -candidate.model_false_positive_score,
        -candidate.lexical_proximity_score,
        _sha256(f"{seed}:{candidate.candidate_id}"),
    )


def _candidate_quality_key(candidate: HardNegativeCandidate) -> tuple[Any, ...]:
    return (
        candidate.difficulty_score,
        candidate.model_false_positive_score,
        candidate.lexical_proximity_score,
        candidate.candidate_id,
    )


def _candidate_id(
    *,
    seed: int,
    fixture_id: str,
    shard_id: str,
    record_index: int,
    label: str,
    language: str,
    start: int,
    end: int,
    surface: str,
) -> str:
    payload = {
        "end": end,
        "fixture_id": fixture_id,
        "label": label,
        "language": language,
        "record_index": record_index,
        "seed": seed,
        "shard_id": shard_id,
        "start": start,
        "surface": surface,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "hn-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _normalize_surface(value: str) -> str:
    normalized = value.casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[\u2010-\u2015]", "-", normalized)
    return normalized


def _char_shingles(value: str, *, size: int = 3) -> set[str]:
    compact = re.sub(r"\s+", " ", value)
    if len(compact) <= size:
        return {compact} if compact else set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _overlaps_any(start: int, end: int, spans: Sequence[EvalSpan]) -> bool:
    return any(_spans_overlap(start, end, span.start, span.end) for span in spans)


def _spans_overlap(
    left_start: int,
    left_end: int,
    right_start: int,
    right_end: int,
) -> bool:
    return left_start < right_end and right_start < left_end


def _coerce_probability(value: Any) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return 1.0
    if probability < 0.0:
        return 0.0
    if probability > 1.0:
        return 1.0
    return probability


def _assert_memory_budget(config: HardNegativeMiningConfig) -> None:
    if config.peak_memory_budget_bytes is None or not tracemalloc.is_tracing():
        return
    _, peak = tracemalloc.get_traced_memory()
    if peak > config.peak_memory_budget_bytes:
        raise MemoryBudgetExceeded(
            "hard-negative mining exceeded peak memory budget: "
            f"{peak} > {config.peak_memory_budget_bytes} bytes"
        )


def _round_score(value: float) -> float:
    return round(float(value), 6)


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _plain_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain(value[key]) for key in sorted(value, key=str)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _plain_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "DEFAULT_LSH_BANDS",
    "DEFAULT_LSH_ROWS",
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_REFERENCE_SURFACES_PER_LABEL",
    "DEFAULT_MIN_DIFFICULTY_SCORE",
    "DEFAULT_NEAR_DUPLICATE_RETENTION_CEILING",
    "DEFAULT_NEAR_DUPLICATE_SIMILARITY",
    "DEFAULT_PER_LABEL_LANGUAGE_LIMIT",
    "HARD_NEGATIVE_CATEGORY",
    "HARD_NEGATIVE_PACK_VERSION",
    "HardNegativeCandidate",
    "HardNegativeDifficultyReport",
    "HardNegativeMiningConfig",
    "HardNegativeMiningResult",
    "MemoryBudgetExceeded",
    "build_hard_negative_fixture_pack",
    "hard_negative_difficulty_report",
    "iter_synthetic_corpus_shard",
    "mine_hard_negative_candidates",
    "near_duplicate_retention_rate",
    "stratified_hard_negative_sample",
    "write_hard_negative_fixture_pack",
]
