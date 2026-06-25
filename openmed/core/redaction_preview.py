"""Safe, single-document previews for de-identification changes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .audit import hash_text

if TYPE_CHECKING:
    from .pii import DeidentificationResult, PIIEntity


@dataclass(frozen=True)
class RedactionChange:
    """One original-text span and its de-identification replacement."""

    start: int
    end: int
    label: str
    action: str
    original_hash: str
    replacement_hash: str
    original: str | None = None
    replacement: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of this change."""
        payload: dict[str, object] = {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "action": self.action,
            "original_hash": self.original_hash,
            "replacement_hash": self.replacement_hash,
        }

        if self.original is not None:
            payload["original"] = self.original

        if self.replacement is not None:
            payload["replacement"] = self.replacement

        return payload


@dataclass(frozen=True)
class RedactionPreview:
    """Ordered preview records for one de-identification result."""

    changes: tuple[RedactionChange, ...]
    offsets_only: bool = False

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable preview structure."""
        return {
            "offsets_only": self.offsets_only,
            "changes": [change.to_dict() for change in self.changes],
        }

    def to_json(self) -> str:
        """Return a deterministic JSON representation."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    def to_text(self) -> str:
        """Render a compact human-readable preview."""
        heading = f"{len(self.changes)} redaction change(s)"
        if self.offsets_only:
            heading += " (offsets-only)"

        lines = [heading]

        for change in self.changes:
            prefix = f"[{change.start}:{change.end}] {change.label} ({change.action})"

            if self.offsets_only:
                lines.append(
                    f"{prefix}: {change.original_hash} -> {change.replacement_hash}"
                )
            else:
                lines.append(f"{prefix}: {change.original!r} -> {change.replacement!r}")

        return "\n".join(lines)


def _entity_sort_key(entity: PIIEntity) -> tuple[int, int, str, str, str]:
    """Return a deterministic ordering key for preview records."""
    replacement = entity.redacted_text or ""

    return (
        int(entity.start),
        int(entity.end),
        str(entity.label),
        str(entity.action or ""),
        hash_text(replacement),
    )


def _validate_offsets(text: str, start: int, end: int) -> None:
    """Reject offsets that do not point to a valid span in the supplied text."""
    if start < 0 or end < start or end > len(text):
        raise ValueError(
            "PII entity offsets must satisfy "
            f"0 <= start <= end <= len(text), got start={start}, end={end}"
        )


def redaction_preview(
    text: str,
    result: DeidentificationResult,
    *,
    offsets_only: bool = False,
) -> RedactionPreview:
    """Build a deterministic per-span preview of a de-identification result.

    The entity offsets refer to ``text`` before redaction. In offsets-only mode,
    plaintext surfaces are omitted while stable hashes remain available for
    review and comparison.
    """
    changes: list[RedactionChange] = []

    for entity in sorted(result.pii_entities, key=_entity_sort_key):
        start = int(entity.start)
        end = int(entity.end)
        _validate_offsets(text, start, end)

        original = text[start:end]
        replacement = entity.redacted_text or ""

        changes.append(
            RedactionChange(
                start=start,
                end=end,
                label=str(entity.label),
                action=str(entity.action or ""),
                original_hash=hash_text(original),
                replacement_hash=hash_text(replacement),
                original=None if offsets_only else original,
                replacement=None if offsets_only else replacement,
            )
        )

    return RedactionPreview(
        changes=tuple(changes),
        offsets_only=offsets_only,
    )
