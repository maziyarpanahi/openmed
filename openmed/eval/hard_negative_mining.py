"""Mine synthetic benchmark errors into a training-ready hard-negative manifest.

Error-analysis reports intentionally persist hashes and coordinates instead of
raw source text.  This module joins those reports with explicitly synthetic
benchmark fixtures, extracts bounded context windows, and emits a deterministic
manifest.  The manifest keeps the source label and error coordinates for eval
triage while :meth:`HardNegativeManifestEntry.to_training_item` exposes the
empty-label hard-negative shape consumed by the training harness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.core.labels import normalize_label
from openmed.eval.error_analysis import (
    MISSED,
    SPURIOUS,
    ErrorAnalysisReport,
    error_report,
)
from openmed.eval.golden import HARD_NEGATIVE_CATEGORY
from openmed.eval.harness import BenchmarkFixture, ModelRunner, load_fixtures

HARD_NEGATIVE_MANIFEST_SCHEMA_VERSION = "openmed.eval.hard_negative_manifest.v1"
HARD_NEGATIVE_MANIFEST_VERSION = 1
HARD_NEGATIVE_MANIFEST_ID = "openmed-benchmark-error-hard-negatives-v1"
HARD_NEGATIVE_HARNESS_CONTRACT_REF = (
    "openmed.training.hard_negatives:HardNegativeExample.to_training_item"
)
BENCHMARK_ERROR_SUBTYPE = "benchmark_error"
FALSE_NEGATIVE = "false_negative"
FALSE_POSITIVE = "false_positive"
ERROR_TYPES = (FALSE_NEGATIVE, FALSE_POSITIVE)
DEFAULT_CONTEXT_WINDOW = 24
DEFAULT_MAX_ENTRIES = 2048

_ERROR_TYPE_ALIASES = {
    FALSE_NEGATIVE: FALSE_NEGATIVE,
    "fn": FALSE_NEGATIVE,
    "missed": FALSE_NEGATIVE,
    MISSED: FALSE_NEGATIVE,
    "recall_error": FALSE_NEGATIVE,
    FALSE_POSITIVE: FALSE_POSITIVE,
    "fp": FALSE_POSITIVE,
    "over_redaction": FALSE_POSITIVE,
    "spurious": FALSE_POSITIVE,
    SPURIOUS: FALSE_POSITIVE,
    "precision_error": FALSE_POSITIVE,
}
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"\d+")
_RAW_TEXT_KEYS = frozenset(
    {
        "context",
        "raw_context",
        "raw_source",
        "raw_text",
        "source_text",
        "span_text",
        "surface",
        "text",
    }
)
_RESTRICTED_SOURCE_MARKERS = frozenset({"dua", "i2b2", "mimic", "n2c2"})


class HardNegativeManifestError(ValueError):
    """Raised when an error source or manifest violates the contract."""


@dataclass(frozen=True)
class HardNegativeManifestEntry:
    """One deduplicated synthetic benchmark-error pattern.

    ``start`` and ``end`` are offsets in the source fixture.  ``context`` is a
    bounded source slice whose offsets are represented by ``context_start`` and
    ``context_end``.  The manifest preserves the source label in ``label`` and
    ``source_label``; training items intentionally use an empty ``labels`` list
    because these are hard negatives for the existing harness.
    """

    surface: str
    label: str
    start: int
    end: int
    context_start: int
    context_end: int
    context: str
    frequency: int
    gate_impact: float
    priority: float
    error_types: tuple[str, ...]
    source_fixture_ids: tuple[str, ...]
    surface_pattern: str = ""
    language: str = "en"
    span_hashes: tuple[str, ...] = ()
    rank: int = 0
    synthetic: bool = True

    def __post_init__(self) -> None:
        surface = str(self.surface)
        label = normalize_label(str(self.label))
        language = str(self.language or "en")
        if not surface:
            raise HardNegativeManifestError("manifest entries require a surface")
        if not label:
            raise HardNegativeManifestError("manifest entries require a label")
        if not language:
            raise HardNegativeManifestError("manifest entries require a language")
        if not self.synthetic:
            raise HardNegativeManifestError(
                "hard-negative manifest entries must be synthetic"
            )
        if not (0 <= self.context_start <= self.start < self.end <= self.context_end):
            raise HardNegativeManifestError("manifest entry offsets are inconsistent")
        if self.context_end - self.context_start != len(self.context):
            raise HardNegativeManifestError(
                "manifest context offsets do not match context length"
            )
        relative_start = self.start - self.context_start
        relative_end = self.end - self.context_start
        if self.context[relative_start:relative_end] != surface:
            raise HardNegativeManifestError(
                "manifest context does not contain the entry surface at its offsets"
            )
        if (
            isinstance(self.frequency, bool)
            or not isinstance(self.frequency, int)
            or self.frequency <= 0
        ):
            raise HardNegativeManifestError("frequency must be a positive integer")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 0
        ):
            raise HardNegativeManifestError("rank must be a non-negative integer")
        gate_impact = _finite_non_negative(self.gate_impact, "gate_impact")
        priority = _finite_non_negative(self.priority, "priority")
        error_types = tuple(sorted(set(str(value) for value in self.error_types)))
        if not error_types or any(value not in ERROR_TYPES for value in error_types):
            raise HardNegativeManifestError(
                "error_types must contain false_positive and/or false_negative"
            )
        fixture_ids = tuple(sorted({str(value) for value in self.source_fixture_ids}))
        if not fixture_ids or any(not value for value in fixture_ids):
            raise HardNegativeManifestError(
                "manifest entries require source fixture identifiers"
            )
        span_hashes = tuple(sorted({str(value) for value in self.span_hashes if value}))
        pattern = self.surface_pattern or surface_pattern(surface)

        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "gate_impact", gate_impact)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "error_types", error_types)
        object.__setattr__(self, "source_fixture_ids", fixture_ids)
        object.__setattr__(self, "span_hashes", span_hashes)
        object.__setattr__(self, "surface_pattern", pattern)

    @property
    def relative_start(self) -> int:
        """Return the surface start offset within the context window."""

        return self.start - self.context_start

    @property
    def relative_end(self) -> int:
        """Return the surface end offset within the context window."""

        return self.end - self.context_start

    @property
    def error_type(self) -> str:
        """Return the single type or a stable mixed marker."""

        return self.error_types[0] if len(self.error_types) == 1 else "mixed"

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest row with source coordinates and context."""

        return {
            "context": self.context,
            "context_end": self.context_end,
            "context_start": self.context_start,
            "end": self.end,
            "error_type": self.error_type,
            "error_types": list(self.error_types),
            "frequency": self.frequency,
            "gate_impact": _round_score(self.gate_impact),
            "hard_negative_category": HARD_NEGATIVE_CATEGORY,
            "hard_negative_subtype": BENCHMARK_ERROR_SUBTYPE,
            "is_hard_negative": True,
            "label": self.label,
            "labels": [],
            "source_label": self.label,
            "language": self.language,
            "priority": _round_score(self.priority),
            "rank": self.rank,
            "source_fixture_ids": list(self.source_fixture_ids),
            "span_hashes": list(self.span_hashes),
            "start": self.start,
            "surface": self.surface,
            "surface_pattern": self.surface_pattern,
            "synthetic": True,
            "text": self.context,
        }

    def to_training_item(self) -> dict[str, Any]:
        """Return one item in the OM-038b hard-negative harness shape."""

        metadata = {
            "context_end": self.context_end,
            "context_start": self.context_start,
            "error_type": self.error_type,
            "error_types": list(self.error_types),
            "frequency": self.frequency,
            "gate_impact": _round_score(self.gate_impact),
            "label": self.label,
            "priority": _round_score(self.priority),
            "rank": self.rank,
            "source_fixture_ids": list(self.source_fixture_ids),
            "span_hashes": list(self.span_hashes),
            "start": self.relative_start,
            "end": self.relative_end,
            "surface": self.surface,
            "surface_pattern": self.surface_pattern,
            "synthetic": True,
        }
        return {
            "text": self.context,
            "labels": [],
            "is_hard_negative": True,
            "hard_negative_category": HARD_NEGATIVE_CATEGORY,
            "hard_negative_subtype": BENCHMARK_ERROR_SUBTYPE,
            "metadata": metadata,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HardNegativeManifestEntry":
        """Build an entry from a serialized manifest row."""

        labels = payload.get("labels")
        if isinstance(labels, str):
            labels = [labels]
        label = payload.get("label") or (
            labels[0] if isinstance(labels, Sequence) and labels else None
        )
        if label is None:
            raise HardNegativeManifestError("manifest entry label is required")

        context = payload.get("context") or payload.get("text")
        if not isinstance(context, str):
            raise HardNegativeManifestError("manifest entry context must be text")
        source_fixture_ids = payload.get("source_fixture_ids") or ()
        if isinstance(source_fixture_ids, str):
            source_fixture_ids = [source_fixture_ids]
        error_types = payload.get("error_types") or payload.get("error_type")
        if isinstance(error_types, str):
            error_types = [error_types]
        if not isinstance(error_types, Sequence):
            raise HardNegativeManifestError("manifest entry error_types are required")
        span_hashes = payload.get("span_hashes") or ()
        if isinstance(span_hashes, str):
            span_hashes = [span_hashes]
        try:
            start = int(payload["start"])
            end = int(payload["end"])
            context_start = int(payload["context_start"])
            context_end = int(payload["context_end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HardNegativeManifestError(
                "manifest entry offsets are required"
            ) from exc
        return cls(
            surface=str(
                payload.get("surface")
                or context[start - context_start : end - context_start]
            ),
            label=str(label),
            start=start,
            end=end,
            context_start=context_start,
            context_end=context_end,
            context=context,
            frequency=int(payload.get("frequency", 1)),
            gate_impact=float(payload.get("gate_impact", 1.0)),
            priority=float(payload.get("priority", 1.0)),
            error_types=tuple(str(value) for value in error_types),
            source_fixture_ids=tuple(str(value) for value in source_fixture_ids),
            surface_pattern=str(payload.get("surface_pattern") or ""),
            language=str(payload.get("language") or "en"),
            span_hashes=tuple(str(value) for value in span_hashes),
            rank=int(payload.get("rank", 0)),
            synthetic=payload.get("synthetic") is True,
        )


@dataclass(frozen=True)
class HardNegativeManifest:
    """Versioned, deterministic output from benchmark-error mining."""

    entries: tuple[HardNegativeManifestEntry, ...]
    source_report_hash: str
    source_fixture_count: int
    scanned_error_count: int
    duplicate_count: int = 0
    truncated_count: int = 0
    label_gate_impacts: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = HARD_NEGATIVE_MANIFEST_SCHEMA_VERSION
    manifest_id: str = HARD_NEGATIVE_MANIFEST_ID
    synthetic: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != HARD_NEGATIVE_MANIFEST_SCHEMA_VERSION:
            raise HardNegativeManifestError(
                f"unsupported manifest schema: {self.schema_version}"
            )
        if self.manifest_id != HARD_NEGATIVE_MANIFEST_ID:
            raise HardNegativeManifestError(
                f"unsupported manifest id: {self.manifest_id}"
            )
        if not self.synthetic:
            raise HardNegativeManifestError("hard-negative manifests must be synthetic")
        if not isinstance(self.source_report_hash, str) or not self.source_report_hash:
            raise HardNegativeManifestError("source_report_hash is required")
        for name in (
            "source_fixture_count",
            "scanned_error_count",
            "duplicate_count",
            "truncated_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HardNegativeManifestError(f"{name} must be non-negative")
        entries = tuple(self.entries)
        if any(not isinstance(entry, HardNegativeManifestEntry) for entry in entries):
            raise HardNegativeManifestError("entries must be manifest entries")
        impacts = {
            normalize_label(str(label)): _finite_non_negative(value, "gate impact")
            for label, value in self.label_gate_impacts.items()
        }
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "label_gate_impacts", dict(sorted(impacts.items())))

    @property
    def entry_count(self) -> int:
        """Return the number of retained manifest entries."""

        return len(self.entries)

    @property
    def candidate_count(self) -> int:
        """Return the number of error examples scanned before deduplication."""

        return self.scanned_error_count

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready manifest payload."""

        label_counts = Counter(entry.label for entry in self.entries)
        error_type_counts = Counter(
            error_type for entry in self.entries for error_type in entry.error_types
        )
        return {
            "contract_ref": HARD_NEGATIVE_HARNESS_CONTRACT_REF,
            "entries": [entry.to_dict() for entry in self.entries],
            "manifest_id": self.manifest_id,
            "metadata": _plain(self.metadata),
            "schema_version": self.schema_version,
            "source_fixture_count": self.source_fixture_count,
            "source_report_hash": self.source_report_hash,
            "stats": {
                "candidate_count": self.candidate_count,
                "duplicate_count": self.duplicate_count,
                "entry_count": self.entry_count,
                "error_type_counts": {
                    key: error_type_counts[key] for key in ERROR_TYPES
                },
                "label_counts": {
                    key: label_counts[key] for key in sorted(label_counts)
                },
                "truncated_count": self.truncated_count,
            },
            "label_gate_impacts": {
                key: _round_score(value)
                for key, value in self.label_gate_impacts.items()
            },
            "synthetic": True,
            "version": HARD_NEGATIVE_MANIFEST_VERSION,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the manifest deterministically."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the manifest to a JSON file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_training_items(self) -> tuple[dict[str, Any], ...]:
        """Return entries in the OM-038b training-item contract."""

        return tuple(entry.to_training_item() for entry in self.entries)

    def to_fixture_pack(self) -> dict[str, Any]:
        """Return an eval-harness fixture pack for over-redaction checks."""

        fixtures = []
        for entry in self.entries:
            digest = _sha256_json(
                {
                    "label": entry.label,
                    "pattern": entry.surface_pattern,
                    "rank": entry.rank,
                }
            )[:12]
            fixtures.append(
                {
                    "gold_spans": [],
                    "id": f"{self.manifest_id}-{entry.rank:04d}-{digest}",
                    "language": entry.language,
                    "metadata": {
                        "category": HARD_NEGATIVE_CATEGORY,
                        "hard_negative_candidates": [
                            {
                                "end": entry.relative_end,
                                "error_type": entry.error_type,
                                "frequency": entry.frequency,
                                "gate_impact": _round_score(entry.gate_impact),
                                "label": entry.label,
                                "priority": _round_score(entry.priority),
                                "start": entry.relative_start,
                                "synthetic": True,
                                "text": entry.surface,
                            }
                        ],
                        "hard_negative_manifest": self.manifest_id,
                        "source_fixture_ids": list(entry.source_fixture_ids),
                        "synthetic": True,
                    },
                    "text": entry.context,
                }
            )
        return {
            "fixtures": fixtures,
            "manifest_id": self.manifest_id,
            "schema_version": self.schema_version,
            "suite": "hard-negative-benchmark-errors",
            "synthetic": True,
            "version": HARD_NEGATIVE_MANIFEST_VERSION,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HardNegativeManifest":
        """Rebuild and validate a manifest from JSON-ready data."""

        if payload.get("schema_version") != HARD_NEGATIVE_MANIFEST_SCHEMA_VERSION:
            raise HardNegativeManifestError("unsupported hard-negative manifest schema")
        if payload.get("manifest_id") != HARD_NEGATIVE_MANIFEST_ID:
            raise HardNegativeManifestError("unsupported hard-negative manifest id")
        if payload.get("synthetic") is not True:
            raise HardNegativeManifestError("manifest must be marked synthetic")
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(
            raw_entries, (str, bytes)
        ):
            raise HardNegativeManifestError("manifest entries must be a list")
        stats = payload.get("stats") or {}
        if not isinstance(stats, Mapping):
            raise HardNegativeManifestError("manifest stats must be an object")
        if payload.get("contract_ref") != HARD_NEGATIVE_HARNESS_CONTRACT_REF:
            raise HardNegativeManifestError("manifest has an unsupported contract")
        if payload.get("version") != HARD_NEGATIVE_MANIFEST_VERSION:
            raise HardNegativeManifestError("manifest has an unsupported version")
        entries = tuple(
            HardNegativeManifestEntry.from_mapping(row)
            for row in raw_entries
            if isinstance(row, Mapping)
        )
        if len(entries) != len(raw_entries):
            raise HardNegativeManifestError("manifest entries must be objects")
        return cls(
            entries=entries,
            source_report_hash=str(payload.get("source_report_hash") or ""),
            source_fixture_count=int(payload.get("source_fixture_count", 0)),
            scanned_error_count=int(
                stats.get("candidate_count", payload.get("scanned_error_count", 0))
            ),
            duplicate_count=int(stats.get("duplicate_count", 0)),
            truncated_count=int(stats.get("truncated_count", 0)),
            label_gate_impacts=(
                dict(payload.get("label_gate_impacts") or {})
                if isinstance(payload.get("label_gate_impacts") or {}, Mapping)
                else {}
            ),
            metadata=(
                dict(payload.get("metadata") or {})
                if isinstance(payload.get("metadata") or {}, Mapping)
                else {}
            ),
            schema_version=str(payload["schema_version"]),
            manifest_id=str(payload["manifest_id"]),
            synthetic=True,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "HardNegativeManifest":
        """Read and validate a JSON manifest."""

        manifest_path = Path(path)
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise HardNegativeManifestError(
                f"invalid hard-negative manifest JSON: {manifest_path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise HardNegativeManifestError("hard-negative manifest must be an object")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class HardNegativeMiner:
    """Reusable configuration wrapper around :func:`mine_hard_negative_manifest`."""

    context_window: int = DEFAULT_CONTEXT_WINDOW
    label_gate_impacts: Mapping[str, float] = field(default_factory=dict)
    max_entries: int = DEFAULT_MAX_ENTRIES

    def __post_init__(self) -> None:
        if self.context_window < 0:
            raise ValueError("context_window must be non-negative")
        if self.max_entries <= 0:
            raise ValueError("max_entries must be positive")

    def mine(
        self,
        errors: Any,
        *,
        fixtures: Any = None,
        model: str | ModelRunner | None = None,
        runner: ModelRunner | None = None,
        suite_name: str | None = None,
    ) -> HardNegativeManifest:
        """Mine one report or synthetic benchmark suite."""

        return mine_hard_negative_manifest(
            errors,
            fixtures=fixtures,
            model=model,
            runner=runner,
            suite_name=suite_name,
            context_window=self.context_window,
            label_gate_impacts=self.label_gate_impacts,
            max_entries=self.max_entries,
        )


def mine_hard_negative_manifest(
    errors: Any,
    *,
    fixtures: Any = None,
    model: str | ModelRunner | None = None,
    runner: ModelRunner | None = None,
    suite_name: str | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    label_gate_impacts: Mapping[str, float] | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    example_cap: int = 100_000,
) -> HardNegativeManifest:
    """Mine false-positive and false-negative examples into a manifest.

    ``errors`` may be an :class:`ErrorAnalysisReport`, a serialized error
    report, or a sequence of error mappings.  Report inputs must be joined with
    ``fixtures`` containing explicitly synthetic :class:`BenchmarkFixture`
    records.  Passing ``model`` treats ``errors`` as a benchmark suite and
    first runs the existing offline error-analysis harness.
    """

    _validate_mining_options(
        context_window=context_window,
        max_entries=max_entries,
        example_cap=example_cap,
    )
    impacts_override = _normalise_gate_impacts(label_gate_impacts)

    report_payload: Mapping[str, Any] | None
    raw_errors: list[Mapping[str, Any]]
    fixture_source = fixtures
    if model is not None:
        if fixture_source is None:
            fixture_source = errors
        report = error_report(
            model,
            errors,
            suite_name=suite_name,
            runner=runner,
            example_cap=example_cap,
            context_window=context_window,
            metadata={"synthetic": True},
        )
        report_payload = report.to_dict()
        raw_errors = []
    else:
        report_payload, raw_errors = _coerce_error_source(errors)

    fixture_map = _fixture_map(fixture_source) if fixture_source is not None else {}
    if report_payload is not None:
        occurrences = list(
            _iter_report_occurrences(
                report_payload,
                fixtures=fixture_map,
                context_window=context_window,
                label_gate_impacts=impacts_override,
            )
        )
    else:
        occurrences = list(
            _iter_raw_occurrences(
                raw_errors,
                fixtures=fixture_map,
                context_window=context_window,
                label_gate_impacts=impacts_override,
            )
        )

    impacts = _merge_gate_impacts(
        _report_gate_impacts(report_payload),
        impacts_override,
    )
    groups: dict[tuple[str, str], list[_ErrorOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        groups[(occurrence.label, occurrence.surface_pattern)].append(occurrence)

    ranked_entries = []
    for (label, pattern), group in groups.items():
        representative = min(
            group,
            key=lambda item: (
                len(item.context),
                item.fixture_id,
                item.start,
                item.end,
                item.surface,
            ),
        )
        gate_impact = max(
            [item.gate_impact for item in group] or [impacts.get(label, 1.0)]
        )
        frequency = len(group)
        ranked_entries.append(
            HardNegativeManifestEntry(
                surface=representative.surface,
                label=label,
                start=representative.start,
                end=representative.end,
                context_start=representative.context_start,
                context_end=representative.context_end,
                context=representative.context,
                frequency=frequency,
                gate_impact=gate_impact,
                priority=frequency * gate_impact,
                error_types=tuple(item.error_type for item in group),
                source_fixture_ids=tuple(item.fixture_id for item in group),
                surface_pattern=pattern,
                language=representative.language,
                span_hashes=tuple(item.span_hash for item in group),
                synthetic=True,
            )
        )

    ranked_entries.sort(
        key=lambda entry: (
            -entry.priority,
            -entry.frequency,
            -entry.gate_impact,
            entry.label,
            entry.surface_pattern,
            entry.source_fixture_ids,
            entry.start,
        )
    )
    duplicate_count = len(occurrences) - len(ranked_entries)
    truncated_count = max(0, len(ranked_entries) - max_entries)
    retained = tuple(
        _with_rank(entry, index)
        for index, entry in enumerate(ranked_entries[:max_entries], start=1)
    )

    source_fixture_count = len(fixture_map)
    if not source_fixture_count:
        source_fixture_count = len({item.fixture_id for item in occurrences})
    report_hash = _source_report_hash(report_payload, occurrences)
    metadata = _report_metadata(report_payload)
    metadata.update(
        {
            "context_window": context_window,
            "deduplication": "normalized_surface_and_label",
            "synthetic": True,
        }
    )
    return HardNegativeManifest(
        entries=retained,
        source_report_hash=report_hash,
        source_fixture_count=source_fixture_count,
        scanned_error_count=len(occurrences),
        duplicate_count=duplicate_count,
        truncated_count=truncated_count,
        label_gate_impacts=impacts,
        metadata=metadata,
    )


def mine_benchmark_error_manifest(
    suite: Any,
    *,
    model: str | ModelRunner,
    fixtures: Any = None,
    runner: ModelRunner | None = None,
    suite_name: str | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    label_gate_impacts: Mapping[str, float] | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    example_cap: int = 100_000,
) -> HardNegativeManifest:
    """Run the local benchmark harness and mine its error examples."""

    return mine_hard_negative_manifest(
        suite,
        fixtures=fixtures,
        model=model,
        runner=runner,
        suite_name=suite_name,
        context_window=context_window,
        label_gate_impacts=label_gate_impacts,
        max_entries=max_entries,
        example_cap=example_cap,
    )


def write_hard_negative_manifest(
    manifest: HardNegativeManifest | Mapping[str, Any],
    path: str | Path,
    *,
    indent: int = 2,
) -> Path:
    """Validate and write a mined manifest."""

    active = (
        manifest
        if isinstance(manifest, HardNegativeManifest)
        else HardNegativeManifest.from_mapping(manifest)
    )
    return active.write_json(path, indent=indent)


def load_hard_negative_manifest(path: str | Path) -> HardNegativeManifest:
    """Load and validate a mined manifest from JSON."""

    return HardNegativeManifest.read_json(path)


def validate_hard_negative_manifest(
    manifest: HardNegativeManifest | Mapping[str, Any],
) -> HardNegativeManifest:
    """Return a validated manifest or raise ``HardNegativeManifestError``."""

    if isinstance(manifest, HardNegativeManifest):
        return manifest
    if not isinstance(manifest, Mapping):
        raise HardNegativeManifestError("manifest must be an object")
    return HardNegativeManifest.from_mapping(manifest)


def manifest_training_items(
    manifest: HardNegativeManifest | Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Return validated manifest entries in the training harness shape."""

    return validate_hard_negative_manifest(manifest).to_training_items()


def surface_pattern(surface: str) -> str:
    """Normalize a surface for deterministic label-aware deduplication."""

    normalized = unicodedata.normalize("NFKC", str(surface)).casefold()
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    return _NUMBER_RE.sub("<number>", normalized)


@dataclass(frozen=True)
class _ErrorOccurrence:
    error_type: str
    label: str
    fixture_id: str
    language: str
    start: int
    end: int
    context_start: int
    context_end: int
    surface: str
    context: str
    span_hash: str
    gate_impact: float

    @property
    def surface_pattern(self) -> str:
        return surface_pattern(self.surface)


def _iter_report_occurrences(
    report: Mapping[str, Any],
    *,
    fixtures: Mapping[str, BenchmarkFixture],
    context_window: int,
    label_gate_impacts: Mapping[str, float],
) -> Iterable[_ErrorOccurrence]:
    report_impacts = _report_gate_impacts(report)
    impacts = _merge_gate_impacts(report_impacts, label_gate_impacts)
    for bucket, error_type in (
        ("false_negatives", FALSE_NEGATIVE),
        ("false_positives", FALSE_POSITIVE),
    ):
        container = report.get(bucket) or {}
        if isinstance(container, Mapping):
            values = (
                (str(label), example)
                for label in sorted(container, key=str)
                for example in _sequence_or_single(container[label])
            )
        elif isinstance(container, Sequence) and not isinstance(
            container, (str, bytes)
        ):
            values = (("", example) for example in container)
        else:
            raise HardNegativeManifestError(f"{bucket} must be a mapping or list")
        for fallback_label, example in values:
            if not isinstance(example, Mapping):
                raise HardNegativeManifestError(f"{bucket} examples must be objects")
            yield _occurrence_from_mapping(
                example,
                default_error_type=error_type,
                fallback_label=fallback_label,
                fixtures=fixtures,
                context_window=context_window,
                label_gate_impacts=impacts,
            )


def _iter_raw_occurrences(
    rows: Sequence[Mapping[str, Any]],
    *,
    fixtures: Mapping[str, BenchmarkFixture],
    context_window: int,
    label_gate_impacts: Mapping[str, float],
) -> Iterable[_ErrorOccurrence]:
    for row in rows:
        yield _occurrence_from_mapping(
            row,
            default_error_type=None,
            fallback_label="",
            fixtures=fixtures,
            context_window=context_window,
            label_gate_impacts=label_gate_impacts,
        )


def _occurrence_from_mapping(
    data: Mapping[str, Any],
    *,
    default_error_type: str | None,
    fallback_label: str,
    fixtures: Mapping[str, BenchmarkFixture],
    context_window: int,
    label_gate_impacts: Mapping[str, float],
) -> _ErrorOccurrence:
    fixture_id = str(
        data.get("fixture_id") or data.get("record_id") or data.get("id") or ""
    )
    fixture = fixtures.get(fixture_id)
    if fixture is not None:
        _assert_synthetic_fixture(fixture)
        source_text = fixture.text
        language = fixture.language
        synthetic = True
    else:
        source_text = _direct_source_text(data)
        language = str(data.get("language") or data.get("lang") or "en")
        synthetic = _mapping_is_synthetic(data)
    if not fixture_id:
        raise HardNegativeManifestError("error examples require fixture_id")
    if not synthetic:
        raise HardNegativeManifestError(
            f"error example {fixture_id!r} must be explicitly synthetic"
        )
    if source_text is None:
        raise HardNegativeManifestError(
            f"missing synthetic fixture text for {fixture_id!r}"
        )

    try:
        start = int(data["start"])
        end = int(data["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HardNegativeManifestError(
            f"error example {fixture_id!r} requires integer start/end offsets"
        ) from exc
    if not (0 <= start < end <= len(source_text)):
        raise HardNegativeManifestError(
            f"error example {fixture_id!r} has invalid span offsets"
        )

    label = data.get("label") or data.get("canonical_label") or fallback_label
    if not label:
        raise HardNegativeManifestError(
            f"error example {fixture_id!r} requires a label"
        )
    normalized_label = normalize_label(str(label))
    error_type = _normalise_error_type(
        data.get("error_type") or data.get("failure_kind") or data.get("kind"),
        default=default_error_type,
    )
    context_start = max(0, start - context_window)
    context_end = min(len(source_text), end + context_window)
    context = source_text[context_start:context_end]
    span_hash = str(data.get("text_hash") or data.get("span_hash") or "")
    if not span_hash:
        span_hash = _sha256_text(source_text[start:end])
    gate_impact = data.get("gate_impact")
    if gate_impact is None:
        gate_impact = label_gate_impacts.get(normalized_label, 1.0)
    gate_impact = _finite_non_negative(gate_impact, "gate_impact")
    return _ErrorOccurrence(
        error_type=error_type,
        label=normalized_label,
        fixture_id=fixture_id,
        language=language,
        start=start,
        end=end,
        context_start=context_start,
        context_end=context_end,
        surface=source_text[start:end],
        context=context,
        span_hash=span_hash,
        gate_impact=gate_impact,
    )


def _coerce_error_source(
    source: Any,
) -> tuple[Mapping[str, Any] | None, list[Mapping[str, Any]]]:
    if isinstance(source, ErrorAnalysisReport):
        return source.to_dict(), []
    if isinstance(source, Mapping):
        payload = source.to_dict() if hasattr(source, "to_dict") else source
        if _looks_like_error_report(payload):
            return payload, []
        rows = payload.get("errors") or payload.get("candidates")
        if rows is not None:
            return None, _coerce_error_rows(rows)
        return None, [payload]
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return None, _coerce_error_rows(source)
    if hasattr(source, "to_dict"):
        payload = source.to_dict()
        if isinstance(payload, Mapping) and _looks_like_error_report(payload):
            return payload, []
    raise TypeError(
        "errors must be an ErrorAnalysisReport, report mapping, or error rows"
    )


def _coerce_error_rows(rows: Any) -> list[Mapping[str, Any]]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise HardNegativeManifestError("error rows must be a list")
    result = []
    for row in rows:
        if hasattr(row, "to_dict"):
            row = row.to_dict()
        if not isinstance(row, Mapping):
            raise HardNegativeManifestError("error rows must contain objects")
        result.append(row)
    return result


def _looks_like_error_report(payload: Mapping[str, Any]) -> bool:
    return "false_negatives" in payload or "false_positives" in payload


def _fixture_map(source: Any) -> dict[str, BenchmarkFixture]:
    fixtures: list[BenchmarkFixture] = []
    if isinstance(source, (str, Path)):
        fixtures.extend(load_fixtures(source))
    elif isinstance(source, BenchmarkFixture):
        fixtures.append(source)
    elif isinstance(source, Mapping):
        if "fixtures" in source:
            fixtures.extend(_fixture_values(source["fixtures"]))
        elif "text" in source:
            fixtures.append(_coerce_fixture(source))
        else:
            fixtures.extend(_fixture_values(source.values()))
    elif isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        fixtures.extend(_fixture_values(source))
    else:
        raise TypeError("fixtures must be a path, fixture, mapping, or sequence")

    result: dict[str, BenchmarkFixture] = {}
    for fixture in fixtures:
        _assert_synthetic_fixture(fixture)
        previous = result.get(fixture.fixture_id)
        if previous is not None and previous.text != fixture.text:
            raise HardNegativeManifestError(
                f"conflicting synthetic fixture text for {fixture.fixture_id!r}"
            )
        result[fixture.fixture_id] = fixture
    return result


def _fixture_values(values: Any) -> list[BenchmarkFixture]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise HardNegativeManifestError("fixtures must contain a list of objects")
    result = []
    for value in values:
        if isinstance(value, BenchmarkFixture):
            result.append(value)
        elif isinstance(value, Mapping):
            result.append(_coerce_fixture(value))
        else:
            raise HardNegativeManifestError("fixtures must contain objects")
    return result


def _coerce_fixture(payload: Mapping[str, Any]) -> BenchmarkFixture:
    data = dict(payload)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        metadata = {"value": metadata}
    metadata = dict(metadata)
    if data.get("synthetic") is True:
        metadata["synthetic"] = True
    data["metadata"] = metadata
    return BenchmarkFixture.from_mapping(data)


def _assert_synthetic_fixture(fixture: BenchmarkFixture) -> None:
    if fixture.metadata.get("synthetic") is not True:
        raise HardNegativeManifestError(
            f"fixture {fixture.fixture_id!r} must be explicitly marked synthetic"
        )
    if fixture.metadata.get("contains_real_phi") is True:
        raise HardNegativeManifestError(
            f"fixture {fixture.fixture_id!r} declares real PHI"
        )
    if _has_restricted_source(fixture.metadata):
        raise HardNegativeManifestError(
            f"fixture {fixture.fixture_id!r} references a restricted source"
        )


def _mapping_is_synthetic(data: Mapping[str, Any]) -> bool:
    metadata = data.get("metadata")
    return bool(
        data.get("synthetic") is True
        or (isinstance(metadata, Mapping) and metadata.get("synthetic") is True)
    ) and not (
        data.get("contains_real_phi") is True
        or isinstance(metadata, Mapping)
        and metadata.get("contains_real_phi") is True
        or _has_restricted_source(data)
    )


def _has_restricted_source(data: Mapping[str, Any]) -> bool:
    metadata = data.get("metadata") if isinstance(data, Mapping) else None
    values = [
        data.get("source_dataset"),
        data.get("source"),
        metadata.get("source_dataset") if isinstance(metadata, Mapping) else None,
        metadata.get("source") if isinstance(metadata, Mapping) else None,
    ]
    for value in values:
        if value is None:
            continue
        parts = {
            part.strip().lower()
            for part in re.split(r"[^a-zA-Z0-9]+", str(value))
            if part.strip()
        }
        if parts & _RESTRICTED_SOURCE_MARKERS:
            return True
    return False


def _direct_source_text(data: Mapping[str, Any]) -> str | None:
    for key in ("source_text", "document_text", "raw_text", "text"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    context = data.get("context")
    return context if isinstance(context, str) else None


def _normalise_error_type(value: Any, *, default: str | None) -> str:
    if value is None:
        if default is not None:
            return default
        raise HardNegativeManifestError("error examples require an error type")
    normalized = _ERROR_TYPE_ALIASES.get(str(value).strip().lower())
    if normalized is None:
        raise HardNegativeManifestError(f"unsupported benchmark error type: {value}")
    return normalized


def _report_gate_impacts(report: Mapping[str, Any] | None) -> dict[str, float]:
    if report is None:
        return {}
    impacts: dict[str, float] = {}
    matrix = report.get("confusion_matrix") or {}
    if isinstance(matrix, Mapping):
        for raw_label, raw_row in matrix.items():
            if raw_label in (MISSED, SPURIOUS) or not isinstance(raw_row, Mapping):
                continue
            label = normalize_label(str(raw_label))
            count = 0
            for predicted, value in raw_row.items():
                if str(predicted) != label:
                    count += _non_negative_count(value)
            if count:
                impacts[label] = float(count)
        spurious = matrix.get(SPURIOUS) or {}
        if isinstance(spurious, Mapping):
            for raw_label, value in spurious.items():
                label = normalize_label(str(raw_label))
                impacts[label] = impacts.get(label, 0.0) + _non_negative_count(value)
    metadata = report.get("metadata") or {}
    if isinstance(metadata, Mapping):
        declared = metadata.get("label_gate_impacts")
        if isinstance(declared, Mapping):
            impacts.update(_normalise_gate_impacts(declared))
    return impacts


def _normalise_gate_impacts(
    impacts: Mapping[str, float] | None,
) -> dict[str, float]:
    if impacts is None:
        return {}
    if not isinstance(impacts, Mapping):
        raise TypeError("label_gate_impacts must be a mapping")
    return {
        normalize_label(str(label)): _finite_non_negative(value, "gate impact")
        for label, value in impacts.items()
    }


def _merge_gate_impacts(
    base: Mapping[str, float], override: Mapping[str, float]
) -> dict[str, float]:
    merged = {str(label): float(value) for label, value in base.items()}
    merged.update({str(label): float(value) for label, value in override.items()})
    return dict(sorted(merged.items()))


def _report_metadata(report: Mapping[str, Any] | None) -> dict[str, Any]:
    if report is None:
        return {}
    metadata: dict[str, Any] = {}
    for key in ("suite", "model_name", "device", "generated_at"):
        value = report.get(key)
        if value is not None:
            metadata[key] = str(value)
    return metadata


def _source_report_hash(
    report: Mapping[str, Any] | None,
    occurrences: Sequence[_ErrorOccurrence],
) -> str:
    if report is not None:
        payload = _without_raw_text(report)
    else:
        payload = [
            {
                "end": item.end,
                "error_type": item.error_type,
                "fixture_id": item.fixture_id,
                "label": item.label,
                "span_hash": item.span_hash,
                "start": item.start,
            }
            for item in occurrences
        ]
    return _sha256_json(payload)


def _without_raw_text(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {}
        for raw_key, raw_value in value.items():
            name = str(raw_key)
            if name.lower() in _RAW_TEXT_KEYS:
                result[f"{name}_hash"] = _sha256_text(str(raw_value))
            else:
                result[name] = _without_raw_text(raw_value)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_without_raw_text(item) for item in value]
    return value


def _with_rank(
    entry: HardNegativeManifestEntry, rank: int
) -> HardNegativeManifestEntry:
    return HardNegativeManifestEntry(
        surface=entry.surface,
        label=entry.label,
        start=entry.start,
        end=entry.end,
        context_start=entry.context_start,
        context_end=entry.context_end,
        context=entry.context,
        frequency=entry.frequency,
        gate_impact=entry.gate_impact,
        priority=entry.priority,
        error_types=entry.error_types,
        source_fixture_ids=entry.source_fixture_ids,
        surface_pattern=entry.surface_pattern,
        language=entry.language,
        span_hashes=entry.span_hashes,
        rank=rank,
        synthetic=entry.synthetic,
    )


def _sequence_or_single(value: Any) -> Sequence[Any]:
    if isinstance(value, Mapping) or hasattr(value, "to_dict"):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise HardNegativeManifestError("error examples must be objects or lists")


def _validate_mining_options(
    *, context_window: int, max_entries: int, example_cap: int
) -> None:
    for name, value in (
        ("context_window", context_window),
        ("max_entries", max_entries),
        ("example_cap", example_cap),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < 0 or (name == "max_entries" and value == 0):
            raise ValueError(
                f"{name} must be {'positive' if name == 'max_entries' else 'non-negative'}"
            )


def _non_negative_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(parsed, 0)


def _finite_non_negative(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HardNegativeManifestError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise HardNegativeManifestError(f"{name} must be finite and non-negative")
    return parsed


def _round_score(value: float) -> float:
    return round(float(value), 6)


def _sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_text(payload.decode("utf-8"))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "BENCHMARK_ERROR_SUBTYPE",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_MAX_ENTRIES",
    "ERROR_TYPES",
    "FALSE_NEGATIVE",
    "FALSE_POSITIVE",
    "HARD_NEGATIVE_HARNESS_CONTRACT_REF",
    "HARD_NEGATIVE_MANIFEST_ID",
    "HARD_NEGATIVE_MANIFEST_SCHEMA_VERSION",
    "HARD_NEGATIVE_MANIFEST_VERSION",
    "HardNegativeManifest",
    "HardNegativeManifestEntry",
    "HardNegativeManifestError",
    "HardNegativeMiner",
    "load_hard_negative_manifest",
    "manifest_training_items",
    "mine_benchmark_error_manifest",
    "mine_hard_negative_manifest",
    "surface_pattern",
    "validate_hard_negative_manifest",
    "write_hard_negative_manifest",
]
