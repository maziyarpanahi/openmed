"""Deterministic, metadata-only differences between processing summaries."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

from .processing_summary import (
    PROCESSING_SUMMARY_SCHEMA_VERSION,
    ProcessingOutcome,
    ProcessingSummary,
)

__all__ = [
    "PROCESSING_DIFF_SCHEMA_VERSION",
    "AbstentionDelta",
    "DigestChange",
    "MediaTypeDelta",
    "OutcomeDelta",
    "ProcessingDiff",
    "ProcessingDiffError",
    "diff_processing_summaries",
    "render_processing_diff_markdown",
]

PROCESSING_DIFF_SCHEMA_VERSION: Final = 1


class ProcessingDiffError(ValueError):
    """Raised when summaries cannot be compared under the current schema."""


@dataclass(frozen=True, slots=True)
class MediaTypeDelta:
    """Signed change in one media type's aggregate totals."""

    media_type: str
    count: int
    total_bytes: int
    total_pages: int
    total_frames: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable, ordered fields."""
        return {
            "media_type": self.media_type,
            "count": self.count,
            "total_bytes": self.total_bytes,
            "total_pages": self.total_pages,
            "total_frames": self.total_frames,
        }


@dataclass(frozen=True, slots=True)
class OutcomeDelta:
    """Signed change in the number of assets with one terminal outcome."""

    outcome: ProcessingOutcome
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable, ordered fields."""
        return {"outcome": self.outcome.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class AbstentionDelta:
    """Signed change in an abstention stage/reason bucket."""

    stage: str
    reason: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable, ordered fields."""
        return {"stage": self.stage, "reason": self.reason, "count": self.count}


@dataclass(frozen=True, slots=True)
class DigestChange:
    """Multiplicity of an added or removed input/output digest pair."""

    input_sha256: str
    output_sha256: str | None
    count: int

    def to_dict(self) -> dict[str, Any]:
        """Return digests and multiplicity without asset identifiers."""
        data: dict[str, Any] = {"input_sha256": self.input_sha256}
        if self.output_sha256 is not None:
            data["output_sha256"] = self.output_sha256
        data["count"] = self.count
        return data


@dataclass(frozen=True, slots=True)
class ProcessingDiff:
    """A stable after-minus-before comparison of two processing summaries."""

    schema_version: int
    total_assets_delta: int
    total_bytes_delta: int
    total_duration_seconds_delta: float
    asset_count_with_output_digest_delta: int
    by_media_type: tuple[MediaTypeDelta, ...]
    outcome_counts: tuple[OutcomeDelta, ...]
    abstention_counts: tuple[AbstentionDelta, ...]
    added_digests: tuple[DigestChange, ...]
    removed_digests: tuple[DigestChange, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, content-free fields in schema order."""
        return {
            "schema_version": self.schema_version,
            "total_assets_delta": self.total_assets_delta,
            "total_bytes_delta": self.total_bytes_delta,
            "total_duration_seconds_delta": self.total_duration_seconds_delta,
            "asset_count_with_output_digest_delta": (
                self.asset_count_with_output_digest_delta
            ),
            "by_media_type": [entry.to_dict() for entry in self.by_media_type],
            "outcome_counts": [entry.to_dict() for entry in self.outcome_counts],
            "abstention_counts": [entry.to_dict() for entry in self.abstention_counts],
            "added_digests": [entry.to_dict() for entry in self.added_digests],
            "removed_digests": [entry.to_dict() for entry in self.removed_digests],
        }

    def to_json(self) -> str:
        """Serialize to compact, deterministic JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))


def _digest_changes(
    before: Counter[tuple[str, str | None]],
    after: Counter[tuple[str, str | None]],
) -> tuple[DigestChange, ...]:
    return tuple(
        DigestChange(
            input_sha256=input_sha256, output_sha256=output_sha256, count=count
        )
        for (input_sha256, output_sha256), count in sorted(
            (after - before).items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
    )


def diff_processing_summaries(
    before: ProcessingSummary, after: ProcessingSummary
) -> ProcessingDiff:
    """Compare two summaries without accessing any source asset or content.

    Digest changes compare the multiset of input/output SHA-256 pairs. Opaque
    asset identifiers are deliberately excluded from the resulting artifact.
    """
    if not isinstance(before, ProcessingSummary) or not isinstance(
        after, ProcessingSummary
    ):
        raise ProcessingDiffError("both inputs must be ProcessingSummary values")
    if (
        before.schema_version != PROCESSING_SUMMARY_SCHEMA_VERSION
        or after.schema_version != PROCESSING_SUMMARY_SCHEMA_VERSION
    ):
        raise ProcessingDiffError("processing summary schema version is unsupported")

    before_media = {entry.media_type: entry for entry in before.by_media_type}
    after_media = {entry.media_type: entry for entry in after.by_media_type}
    by_media_type = tuple(
        MediaTypeDelta(
            media_type=media_type,
            count=(right.count if right else 0) - (left.count if left else 0),
            total_bytes=(right.total_bytes if right else 0)
            - (left.total_bytes if left else 0),
            total_pages=(right.total_pages if right else 0)
            - (left.total_pages if left else 0),
            total_frames=(right.total_frames if right else 0)
            - (left.total_frames if left else 0),
        )
        for media_type in sorted(before_media.keys() | after_media.keys())
        for left, right in [(before_media.get(media_type), after_media.get(media_type))]
    )

    before_outcomes = {entry.outcome: entry.count for entry in before.outcome_counts}
    after_outcomes = {entry.outcome: entry.count for entry in after.outcome_counts}
    outcome_counts = tuple(
        OutcomeDelta(
            outcome=outcome,
            count=after_outcomes.get(outcome, 0) - before_outcomes.get(outcome, 0),
        )
        for outcome in ProcessingOutcome
        if outcome in before_outcomes or outcome in after_outcomes
    )

    before_abstentions = {
        (entry.stage.value, entry.reason.value): entry.count
        for entry in before.abstention_counts
    }
    after_abstentions = {
        (entry.stage.value, entry.reason.value): entry.count
        for entry in after.abstention_counts
    }
    abstention_counts = tuple(
        AbstentionDelta(
            stage=stage,
            reason=reason,
            count=after_abstentions.get((stage, reason), 0)
            - before_abstentions.get((stage, reason), 0),
        )
        for stage, reason in sorted(
            before_abstentions.keys() | after_abstentions.keys()
        )
    )

    before_digests = Counter(
        (entry.input_sha256, entry.output_sha256) for entry in before.asset_digests
    )
    after_digests = Counter(
        (entry.input_sha256, entry.output_sha256) for entry in after.asset_digests
    )

    return ProcessingDiff(
        schema_version=PROCESSING_DIFF_SCHEMA_VERSION,
        total_assets_delta=after.total_assets - before.total_assets,
        total_bytes_delta=after.total_bytes - before.total_bytes,
        total_duration_seconds_delta=(
            after.total_duration_seconds - before.total_duration_seconds
        ),
        asset_count_with_output_digest_delta=(
            after.asset_count_with_output_digest - before.asset_count_with_output_digest
        ),
        by_media_type=by_media_type,
        outcome_counts=outcome_counts,
        abstention_counts=abstention_counts,
        added_digests=_digest_changes(before_digests, after_digests),
        removed_digests=_digest_changes(after_digests, before_digests),
    )


def render_processing_diff_markdown(difference: ProcessingDiff) -> str:
    """Render a deterministic Markdown table report from a summary diff."""
    lines = [
        "# Processing Summary Diff",
        "",
        f"- Diff schema version: {difference.schema_version}",
        f"- Assets: {difference.total_assets_delta:+d}",
        f"- Bytes: {difference.total_bytes_delta:+d}",
        f"- Duration (seconds): {difference.total_duration_seconds_delta:+g}",
        f"- Assets with output digest: "
        f"{difference.asset_count_with_output_digest_delta:+d}",
        "",
        "## Media Types",
        "",
        "| Media Type | Assets | Bytes | Pages | Frames |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for media_entry in difference.by_media_type:
        lines.append(
            f"| {media_entry.media_type} | {media_entry.count:+d} | "
            f"{media_entry.total_bytes:+d} | {media_entry.total_pages:+d} | "
            f"{media_entry.total_frames:+d} |"
        )
    lines.extend(["", "## Outcomes", "", "| Outcome | Assets |", "| --- | ---: |"])
    for outcome_entry in difference.outcome_counts:
        lines.append(f"| {outcome_entry.outcome.value} | {outcome_entry.count:+d} |")
    lines.extend(
        [
            "",
            "## Abstentions",
            "",
            "| Stage | Reason | Assets |",
            "| --- | --- | ---: |",
        ]
    )
    for abstention_entry in difference.abstention_counts:
        lines.append(
            f"| {abstention_entry.stage} | {abstention_entry.reason} | "
            f"{abstention_entry.count:+d} |"
        )
    for title, entries in (
        ("Added Digests", difference.added_digests),
        ("Removed Digests", difference.removed_digests),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Input SHA-256 | Output SHA-256 | Count |",
                "| --- | --- | ---: |",
            ]
        )
        for digest_entry in entries:
            lines.append(
                f"| {digest_entry.input_sha256} | "
                f"{digest_entry.output_sha256 or ''} | {digest_entry.count} |"
            )
    return "\n".join(lines) + "\n"
