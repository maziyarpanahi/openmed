"""Deterministic, metadata-only completion summaries for multimodal runs.

Aggregates per-asset processing outcomes (each composed strictly from the
privacy-safe primitives in :mod:`asset_manifest`, :mod:`abstention`, and
:mod:`digest`) into a single :class:`ProcessingSummary`. The summary never
carries OCR text, file paths, DICOM tag values, transcript text, or any other
free-text/source-content field -- only counts, byte/page/frame totals, and the
opaque ``asset_id``/``sha256`` values that upstream primitives already treat
as safe to expose.

Outcome codes are a fixed, closed set (see :class:`ProcessingOutcome`).
``summarize_processing_run`` fails closed: any result carrying an outcome
code outside that set raises :class:`ProcessingSummaryError` before any
aggregation happens, so no partial summary is ever produced or leaked
through the exception.

Schema version 1 (``PROCESSING_SUMMARY_SCHEMA_VERSION``) covers exactly the
fields declared below; adding, removing, or reinterpreting a field requires a
version bump.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from .abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from .asset_manifest import AssetManifest
from .digest import AssetDigest

__all__ = [
    "PROCESSING_SUMMARY_SCHEMA_VERSION",
    "AbstentionCount",
    "AssetDigestEntry",
    "AssetProcessingResult",
    "MediaTypeTotals",
    "OutcomeCount",
    "ProcessingOutcome",
    "ProcessingSummary",
    "ProcessingSummaryError",
    "render_processing_summary_markdown",
    "summarize_processing_run",
]

PROCESSING_SUMMARY_SCHEMA_VERSION: Final = 1


class ProcessingSummaryError(ValueError):
    """Raised when a processing summary cannot be built or validated safely."""


class ProcessingOutcome(str, Enum):
    """Closed set of terminal outcomes for a single asset in a run."""

    SUCCESS = "success"
    ABSTAINED = "abstained"
    ERROR = "error"


def _outcome(value: object) -> ProcessingOutcome:
    try:
        return ProcessingOutcome(value)
    except Exception:
        raise ProcessingSummaryError("outcome_code is unsupported") from None


@dataclass(frozen=True, slots=True)
class AssetProcessingResult:
    """One asset's terminal processing outcome, metadata-only.

    No field on this type accepts free text, a path, a URL, or raw source
    content -- it is composed strictly from :class:`AssetManifest`,
    :class:`AbstentionRecord`, and :class:`AssetDigest`, each of which already
    enforces that boundary independently.
    """

    manifest: AssetManifest
    outcome_code: ProcessingOutcome
    duration_seconds: float
    input_digest: AssetDigest
    abstention: AbstentionRecord | None = None
    output_digest: AssetDigest | None = None

    def __post_init__(self) -> None:
        outcome = _outcome(self.outcome_code)
        if not isinstance(self.manifest, AssetManifest):
            raise ProcessingSummaryError("manifest must be an AssetManifest")
        if not isinstance(self.input_digest, AssetDigest):
            raise ProcessingSummaryError("input_digest must be an AssetDigest")
        if (
            self.input_digest.sha256 != self.manifest.sha256
            or self.input_digest.byte_count != self.manifest.byte_size
        ):
            raise ProcessingSummaryError("input_digest does not match manifest")
        if self.output_digest is not None and not isinstance(
            self.output_digest, AssetDigest
        ):
            raise ProcessingSummaryError("output_digest must be an AssetDigest")
        if type(self.duration_seconds) not in (int, float) or not math.isfinite(
            self.duration_seconds
        ):
            raise ProcessingSummaryError(
                "duration_seconds must be a finite non-negative number"
            )
        if self.duration_seconds < 0:
            raise ProcessingSummaryError(
                "duration_seconds must be a finite non-negative number"
            )
        is_abstained = outcome is ProcessingOutcome.ABSTAINED
        if is_abstained:
            if not isinstance(self.abstention, AbstentionRecord):
                raise ProcessingSummaryError(
                    "abstention is required when outcome_code is abstained"
                )
            if self.output_digest is not None:
                raise ProcessingSummaryError(
                    "output_digest must be absent when outcome_code is abstained"
                )
        elif self.abstention is not None:
            raise ProcessingSummaryError(
                "abstention must be absent unless outcome_code is abstained"
            )
        object.__setattr__(self, "outcome_code", outcome)


@dataclass(frozen=True, slots=True)
class MediaTypeTotals:
    """Per-``media_type`` rollup of counts and size/shape totals."""

    media_type: str
    count: int
    total_bytes: int
    total_pages: int
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "media_type": self.media_type,
            "count": self.count,
            "total_bytes": self.total_bytes,
            "total_pages": self.total_pages,
            "total_frames": self.total_frames,
        }


@dataclass(frozen=True, slots=True)
class OutcomeCount:
    """Count of assets that ended a run with a given outcome code."""

    outcome: ProcessingOutcome
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"outcome": self.outcome.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class AbstentionCount:
    """Count of abstentions bucketed by ``(stage, reason)``."""

    stage: AbstentionStage
    reason: AbstentionReason
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "reason": self.reason.value,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class AssetDigestEntry:
    """Opaque per-asset digest linkage, sorted by ``asset_id``.

    ``asset_id`` and the SHA-256 hex digests are already validated as
    opaque, non-identifying values by :class:`AssetManifest` and
    :class:`AssetDigest`; nothing else about the asset is carried here.
    """

    asset_id: str
    input_sha256: str
    output_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "asset_id": self.asset_id,
            "input_sha256": self.input_sha256,
        }
        if self.output_sha256 is not None:
            data["output_sha256"] = self.output_sha256
        return data


_SUMMARY_ORDERED_FIELDS: Final = (
    "schema_version",
    "total_assets",
    "total_bytes",
    "total_duration_seconds",
    "by_media_type",
    "outcome_counts",
    "abstention_counts",
    "asset_digests",
    "asset_count_with_output_digest",
)


@dataclass(frozen=True, slots=True)
class ProcessingSummary:
    """Deterministic, metadata-only completion artifact for a run.

    Every tuple field is sorted so ``to_dict()``/``to_json()`` are identical
    regardless of the input order given to :func:`summarize_processing_run`.
    """

    schema_version: int
    total_assets: int
    total_bytes: int
    total_duration_seconds: float
    by_media_type: tuple[MediaTypeTotals, ...]
    outcome_counts: tuple[OutcomeCount, ...]
    abstention_counts: tuple[AbstentionCount, ...]
    asset_digests: tuple[AssetDigestEntry, ...]
    asset_count_with_output_digest: int

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "total_assets": self.total_assets,
            "total_bytes": self.total_bytes,
            "total_duration_seconds": self.total_duration_seconds,
            "by_media_type": [entry.to_dict() for entry in self.by_media_type],
            "outcome_counts": [entry.to_dict() for entry in self.outcome_counts],
            "abstention_counts": [entry.to_dict() for entry in self.abstention_counts],
            "asset_digests": [entry.to_dict() for entry in self.asset_digests],
            "asset_count_with_output_digest": self.asset_count_with_output_digest,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def summarize_processing_run(
    results: Iterable[AssetProcessingResult],
) -> ProcessingSummary:
    """Aggregate per-asset results into a deterministic :class:`ProcessingSummary`.

    Every ``outcome_code`` is validated against :class:`ProcessingOutcome`
    before anything is aggregated: an unrecognized outcome raises
    :class:`ProcessingSummaryError` and no summary -- partial or otherwise --
    is returned.
    """
    materialized = list(results)
    for result in materialized:
        if not isinstance(result, AssetProcessingResult):
            raise ProcessingSummaryError("each result must be an AssetProcessingResult")
        _outcome(result.outcome_code)

    media_totals: dict[str, MediaTypeTotals] = {}
    outcome_totals: dict[ProcessingOutcome, int] = {}
    abstention_totals: dict[tuple[AbstentionStage, AbstentionReason], int] = {}
    digest_entries: list[AssetDigestEntry] = []
    total_bytes = 0
    total_duration_seconds = math.fsum(
        sorted(float(result.duration_seconds) for result in materialized)
    )
    asset_count_with_output_digest = 0

    for result in materialized:
        manifest = result.manifest
        existing = media_totals.get(manifest.media_type)
        pages = manifest.pages or 0
        frames = manifest.frames or 0
        if existing is None:
            media_totals[manifest.media_type] = MediaTypeTotals(
                media_type=manifest.media_type,
                count=1,
                total_bytes=manifest.byte_size,
                total_pages=pages,
                total_frames=frames,
            )
        else:
            media_totals[manifest.media_type] = MediaTypeTotals(
                media_type=manifest.media_type,
                count=existing.count + 1,
                total_bytes=existing.total_bytes + manifest.byte_size,
                total_pages=existing.total_pages + pages,
                total_frames=existing.total_frames + frames,
            )

        outcome_totals[result.outcome_code] = (
            outcome_totals.get(result.outcome_code, 0) + 1
        )

        if result.abstention is not None:
            key = (result.abstention.stage, result.abstention.reason)
            abstention_totals[key] = abstention_totals.get(key, 0) + 1

        total_bytes += manifest.byte_size
        output_sha256 = None
        if result.output_digest is not None:
            output_sha256 = result.output_digest.sha256
            asset_count_with_output_digest += 1
        digest_entries.append(
            AssetDigestEntry(
                asset_id=manifest.asset_id,
                input_sha256=result.input_digest.sha256,
                output_sha256=output_sha256,
            )
        )

    return ProcessingSummary(
        schema_version=PROCESSING_SUMMARY_SCHEMA_VERSION,
        total_assets=len(materialized),
        total_bytes=total_bytes,
        total_duration_seconds=total_duration_seconds,
        by_media_type=tuple(
            media_totals[media_type] for media_type in sorted(media_totals)
        ),
        outcome_counts=tuple(
            OutcomeCount(outcome=outcome, count=outcome_totals[outcome])
            for outcome in ProcessingOutcome
            if outcome in outcome_totals
        ),
        abstention_counts=tuple(
            AbstentionCount(stage=stage, reason=reason, count=count)
            for (stage, reason), count in sorted(
                abstention_totals.items(),
                key=lambda item: (item[0][0].value, item[0][1].value),
            )
        ),
        asset_digests=tuple(sorted(digest_entries, key=lambda entry: entry.asset_id)),
        asset_count_with_output_digest=asset_count_with_output_digest,
    )


def render_processing_summary_markdown(summary: ProcessingSummary) -> str:
    """Render a deterministic Markdown report from a :class:`ProcessingSummary`.

    Pure function over ``summary`` alone -- it never touches raw assets, so
    the metadata-only boundary holds at the type level, not just by
    convention.
    """
    lines: list[str] = []
    lines.append("# Processing Summary")
    lines.append("")
    lines.append(f"- Schema version: {summary.schema_version}")
    lines.append(f"- Total assets: {summary.total_assets}")
    lines.append(f"- Total bytes: {summary.total_bytes}")
    lines.append(f"- Total duration (seconds): {summary.total_duration_seconds}")
    lines.append(
        f"- Assets with an output digest: {summary.asset_count_with_output_digest}"
    )
    lines.append("")

    lines.append("## By Media Type")
    lines.append("")
    lines.append("| Media Type | Count | Total Bytes | Total Pages | Total Frames |")
    lines.append("| --- | --- | --- | --- | --- |")
    for media_total in summary.by_media_type:
        lines.append(
            f"| {media_total.media_type} | {media_total.count} | "
            f"{media_total.total_bytes} | {media_total.total_pages} | "
            f"{media_total.total_frames} |"
        )
    lines.append("")

    lines.append("## Outcomes")
    lines.append("")
    lines.append("| Outcome | Count |")
    lines.append("| --- | --- |")
    for outcome_count in summary.outcome_counts:
        lines.append(f"| {outcome_count.outcome.value} | {outcome_count.count} |")
    lines.append("")

    lines.append("## Abstentions")
    lines.append("")
    lines.append("| Stage | Reason | Count |")
    lines.append("| --- | --- | --- |")
    for abstention_count in summary.abstention_counts:
        lines.append(
            f"| {abstention_count.stage.value} | "
            f"{abstention_count.reason.value} | {abstention_count.count} |"
        )
    lines.append("")

    lines.append("## Asset Digests")
    lines.append("")
    lines.append("| Asset ID | Input SHA-256 | Output SHA-256 |")
    lines.append("| --- | --- | --- |")
    for digest_entry in summary.asset_digests:
        lines.append(
            f"| {digest_entry.asset_id} | {digest_entry.input_sha256} | "
            f"{digest_entry.output_sha256 or ''} |"
        )

    return "\n".join(lines) + "\n"
