"""Offsets-first preview records for one de-identification result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .audit import hash_text

_MISSING = object()


def redaction_preview(
    text: str,
    result: Any,
    *,
    offsets_only: bool = True,
) -> dict[str, Any]:
    """Build a per-span preview of the changes made by de-identification.

    Args:
        text: Original document text for the de-identification run.
        result: A :class:`~openmed.core.pii.DeidentificationResult`, its
            ``to_dict()`` payload, or a compatible object/mapping exposing
            ``pii_entities`` and ``deidentified_text``.
        offsets_only: When ``True`` (default), omit original and replacement
            plaintext and include only offsets, labels, actions, and hashes.
            Set to ``False`` for local interactive review with span surfaces.

    Returns:
        A JSON-serializable preview dictionary with deterministic change order.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    deidentified_text = str(_get_value(result, "deidentified_text", default=""))
    entities = _result_entities(result)
    replacements = _replacement_fallbacks(text, deidentified_text, entities)

    changes: list[dict[str, Any]] = []
    for index, entity in enumerate(entities):
        start, end = _entity_offsets(entity)
        if start < 0 or end < start:
            raise ValueError("entity offsets must be non-negative with end >= start")
        if end > len(text):
            raise ValueError("entity offsets must fall within text")

        original = text[start:end]
        replacement = _entity_replacement(entity)
        if replacement is None:
            replacement = _mapping_replacement(result, original)
        if replacement is None:
            replacement = replacements.get(index, "")

        action = _entity_action(entity, result)
        if action == "keep" and replacement == original:
            continue

        change = {
            "start": start,
            "end": end,
            "offsets": {"start": start, "end": end},
            "label": str(_get_value(entity, "label", default="")),
            "action": action,
            "surface_hash": hash_text(original),
            "replacement_hash": hash_text(replacement),
        }

        canonical_label = _get_value(entity, "canonical_label", default=None)
        if canonical_label is not None:
            change["canonical_label"] = str(canonical_label)

        if not offsets_only:
            change["original_text"] = original
            change["replacement_text"] = replacement

        changes.append(change)

    changes.sort(
        key=lambda change: (
            change["start"],
            change["end"],
            change["label"],
            change["action"],
        )
    )
    preview = {
        "mode": "offsets_only" if offsets_only else "full",
        "change_count": len(changes),
        "document_hash": hash_text(text),
        "deidentified_text_hash": hash_text(deidentified_text),
        "changes": changes,
    }
    preview["summary"] = render_redaction_preview(preview)
    return preview


def render_redaction_preview(preview: Mapping[str, Any]) -> str:
    """Render a compact line-oriented summary for a preview dictionary."""

    mode = str(preview.get("mode") or "offsets_only")
    changes = list(preview.get("changes") or [])
    plural = "" if len(changes) == 1 else "s"
    lines = [f"Redaction preview: {len(changes)} change{plural} ({mode})"]

    for change in changes:
        start = int(change["start"])
        end = int(change["end"])
        label = str(change.get("label") or "")
        action = str(change.get("action") or "")
        prefix = f"- {start}:{end} {label} {action}".rstrip()

        if "original_text" in change or "replacement_text" in change:
            original = str(change.get("original_text") or "")
            replacement = str(change.get("replacement_text") or "")
            lines.append(f'{prefix} "{original}" -> "{replacement}"')
        else:
            surface_hash = _short_hash(str(change.get("surface_hash") or ""))
            replacement_hash = _short_hash(str(change.get("replacement_hash") or ""))
            lines.append(
                f"{prefix} surface={surface_hash} replacement={replacement_hash}"
            )

    return "\n".join(lines)


def _short_hash(value: str) -> str:
    if ":" not in value:
        return value
    algorithm, digest = value.split(":", 1)
    return f"{algorithm}:{digest[:12]}"


def _result_entities(result: Any) -> list[Any]:
    entities = _get_value(result, "pii_entities", default=None)
    if entities is None:
        entities = _get_value(result, "entities", default=())
    if isinstance(entities, Sequence) and not isinstance(entities, (str, bytes)):
        return list(entities)
    raise TypeError("result pii_entities must be a sequence")


def _entity_offsets(entity: Any) -> tuple[int, int]:
    start = _get_value(entity, "start", default=None)
    end = _get_value(entity, "end", default=None)
    if start is None or end is None:
        raise ValueError("entity offsets are required")
    return int(start), int(end)


def _entity_replacement(entity: Any) -> str | None:
    for key in ("redacted_text", "replacement", "surrogate"):
        value = _get_value(entity, key, default=None)
        if value is not None:
            return str(value)
    return None


def _entity_action(entity: Any, result: Any) -> str:
    action = _get_value(entity, "action", default=None)
    if action is None:
        action = _get_value(result, "method", default="redact")
    return str(action or "redact")


def _mapping_replacement(result: Any, original: str) -> str | None:
    mapping = _get_value(result, "mapping", default=None)
    if not isinstance(mapping, Mapping):
        return None

    for replacement, mapped_original in mapping.items():
        if mapped_original == original:
            return str(replacement)
    return None


def _get_value(obj: Any, key: str, *, default: Any = _MISSING) -> Any:
    if isinstance(obj, Mapping):
        if key in obj:
            return obj[key]
    elif hasattr(obj, key):
        return getattr(obj, key)

    if default is _MISSING:
        raise KeyError(key)
    return default


def _replacement_fallbacks(
    text: str,
    deidentified_text: str,
    entities: Sequence[Any],
) -> dict[int, str]:
    spans: list[tuple[int, int, int]] = []
    for index, entity in enumerate(entities):
        try:
            start, end = _entity_offsets(entity)
        except (TypeError, ValueError):
            continue
        spans.append((start, end, index))
    spans.sort(key=lambda item: (item[0], item[1], item[2]))

    replacements: dict[int, str] = {}
    original_cursor = 0
    redacted_cursor = 0
    for position, (start, end, index) in enumerate(spans):
        unchanged_prefix = text[original_cursor:start]
        prefix_index = deidentified_text.find(unchanged_prefix, redacted_cursor)
        if prefix_index < 0:
            original_cursor = end
            continue

        replacement_start = prefix_index + len(unchanged_prefix)
        next_start = spans[position + 1][0] if position + 1 < len(spans) else len(text)
        unchanged_suffix = text[end:next_start]
        if unchanged_suffix:
            replacement_end = deidentified_text.find(
                unchanged_suffix,
                replacement_start,
            )
            if replacement_end < 0:
                original_cursor = end
                redacted_cursor = replacement_start
                continue
        else:
            replacement_end = len(deidentified_text)

        replacements[index] = deidentified_text[replacement_start:replacement_end]
        original_cursor = end
        redacted_cursor = replacement_end

    return replacements
