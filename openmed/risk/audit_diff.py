"""Structured diffs for de-identification audit reports."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

AuditReportInput = Mapping[str, Any] | str | Path
_MISSING = object()


@dataclass(frozen=True)
class SpanSnapshot:
    """Privacy-safe span summary used in audit diffs."""

    start: int
    end: int
    text_hash: str
    label: str
    canonical_label: str
    action: str
    confidence: float | None
    threshold: float | None
    sources: tuple[str, ...]

    @property
    def identity(self) -> tuple[int, int, str]:
        return (self.start, self.end, self.text_hash)

    @property
    def display_label(self) -> str:
        return self.canonical_label or self.label

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.start,
            self.end,
            self.text_hash,
            self.canonical_label,
            self.label,
            self.action,
            _sort_value(self.threshold),
            _sort_value(self.confidence),
            self.sources,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "action": self.action,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class FieldChange:
    """A single before/after value change for a matched span."""

    field: str
    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "before": _json_safe(self.before),
            "after": _json_safe(self.after),
        }


@dataclass(frozen=True)
class SpanChange:
    """Diff record for a span matched across reports."""

    before: SpanSnapshot
    after: SpanSnapshot
    changes: tuple[FieldChange, ...]

    @property
    def label_changed(self) -> bool:
        return any(
            change.field in {"label", "canonical_label"} for change in self.changes
        )

    @property
    def policy_action_changed(self) -> bool:
        return any(change.field == "action" for change in self.changes)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            *self.before.identity,
            tuple(change.field for change in self.changes),
            self.before.sort_key(),
            self.after.sort_key(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.before.start,
            "end": self.before.end,
            "text_hash": self.before.text_hash,
            "before_label": self.before.display_label,
            "after_label": self.after.display_label,
            "before_action": self.before.action,
            "after_action": self.after.action,
            "changes": [change.to_dict() for change in self.changes],
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class ThresholdChange:
    """Report-level threshold change for a flattened threshold path."""

    path: tuple[str, ...]
    before: Any
    after: Any
    before_present: bool
    after_present: bool
    label: str | None = None
    language: str | None = None
    model_id: str | None = None
    profile: str | None = None
    metric: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "key": _path_key(self.path),
            "label": self.label,
            "language": self.language,
            "model_id": self.model_id,
            "profile": self.profile,
            "metric": self.metric,
            "before": _json_safe(self.before),
            "after": _json_safe(self.after),
            "before_present": self.before_present,
            "after_present": self.after_present,
        }


@dataclass(frozen=True)
class ResidualRiskDelta:
    """Numeric residual-risk delta for a flattened residual-risk path."""

    path: tuple[str, ...]
    before: float | None
    after: float | None
    delta: float | None
    before_present: bool
    after_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": list(self.path),
            "key": _path_key(self.path),
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "before_present": self.before_present,
            "after_present": self.after_present,
        }


@dataclass(frozen=True)
class AuditDiff:
    """Structured difference between two de-identification audit reports."""

    added_spans: tuple[SpanSnapshot, ...]
    removed_spans: tuple[SpanSnapshot, ...]
    changed_spans: tuple[SpanChange, ...]
    threshold_changes: tuple[ThresholdChange, ...]
    residual_risk_delta: tuple[ResidualRiskDelta, ...]

    @property
    def label_changed_spans(self) -> tuple[SpanChange, ...]:
        return tuple(change for change in self.changed_spans if change.label_changed)

    @property
    def policy_action_changed_spans(self) -> tuple[SpanChange, ...]:
        return tuple(
            change for change in self.changed_spans if change.policy_action_changed
        )

    @property
    def summary(self) -> dict[str, int]:
        return {
            "added_spans": len(self.added_spans),
            "removed_spans": len(self.removed_spans),
            "changed_spans": len(self.changed_spans),
            "label_changed_spans": len(self.label_changed_spans),
            "policy_action_changed_spans": len(self.policy_action_changed_spans),
            "threshold_changes": len(self.threshold_changes),
            "residual_risk_changes": len(self.residual_risk_delta),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable diff payload."""

        return {
            "summary": self.summary,
            "added_spans": [span.to_dict() for span in self.added_spans],
            "removed_spans": [span.to_dict() for span in self.removed_spans],
            "changed_spans": [change.to_dict() for change in self.changed_spans],
            "label_changed_spans": [
                change.to_dict() for change in self.label_changed_spans
            ],
            "policy_action_changed_spans": [
                change.to_dict() for change in self.policy_action_changed_spans
            ],
            "threshold_changes": [
                change.to_dict() for change in self.threshold_changes
            ],
            "residual_risk_delta": [
                delta.to_dict() for delta in self.residual_risk_delta
            ],
        }

    def to_markdown(self) -> str:
        """Return a compact Markdown summary suitable for review comments."""

        lines = [
            "## Audit Report Diff",
            "",
            "| Span Summary | Count |",
            "|---|---:|",
            f"| Added spans | {len(self.added_spans)} |",
            f"| Removed spans | {len(self.removed_spans)} |",
            f"| Changed spans | {len(self.changed_spans)} |",
            f"| Label-changed spans | {len(self.label_changed_spans)} |",
            f"| Policy-action-changed spans | {len(self.policy_action_changed_spans)} |",
        ]
        lines.extend(self._span_changes_markdown())
        lines.extend(self._threshold_changes_markdown())
        lines.extend(self._residual_risk_markdown())
        return "\n".join(lines)

    def _span_changes_markdown(self) -> list[str]:
        lines = ["", "### Span Changes"]
        if not self.changed_spans and not self.added_spans and not self.removed_spans:
            return [*lines, "No span changes."]

        lines.extend(
            [
                "| Type | Start | End | Text hash | Details |",
                "|---|---:|---:|---|---|",
            ]
        )
        for span in self.added_spans:
            lines.append(_span_markdown_row("Added", span, ""))
        for span in self.removed_spans:
            lines.append(_span_markdown_row("Removed", span, ""))
        for change in self.changed_spans:
            details = "; ".join(
                f"{field.field}: {_format_value(field.before)} -> "
                f"{_format_value(field.after)}"
                for field in change.changes
            )
            lines.append(_span_markdown_row("Changed", change.before, details))
        return lines

    def _threshold_changes_markdown(self) -> list[str]:
        lines = ["", "### Threshold Changes"]
        if not self.threshold_changes:
            return [*lines, "No threshold changes."]

        lines.extend(["| Path | Before | After |", "|---|---:|---:|"])
        for change in self.threshold_changes:
            lines.append(
                "| "
                f"{_markdown_cell(_path_key(change.path))} | "
                f"{_markdown_cell(_format_value(change.before))} | "
                f"{_markdown_cell(_format_value(change.after))} |"
            )
        return lines

    def _residual_risk_markdown(self) -> list[str]:
        lines = ["", "### Residual Risk Delta"]
        if not self.residual_risk_delta:
            return [*lines, "No residual-risk changes."]

        lines.extend(["| Path | Before | After | Delta |", "|---|---:|---:|---:|"])
        for item in self.residual_risk_delta:
            lines.append(
                "| "
                f"{_markdown_cell(_path_key(item.path))} | "
                f"{_markdown_cell(_format_value(item.before))} | "
                f"{_markdown_cell(_format_value(item.after))} | "
                f"{_markdown_cell(_format_value(item.delta))} |"
            )
        return lines


def diff_audit_reports(
    before: AuditReportInput,
    after: AuditReportInput,
) -> AuditDiff:
    """Compare two audit reports without requiring HMAC verification keys.

    Args:
        before: Audit-report mapping or path to an audit-report JSON file.
        after: Audit-report mapping or path to an audit-report JSON file.

    Returns:
        A structured, deterministic diff of span, threshold, and residual-risk
        changes.
    """

    before_report = _coerce_report(before)
    after_report = _coerce_report(after)
    added_spans, removed_spans, changed_spans = _diff_spans(
        _report_spans(before_report),
        _report_spans(after_report),
    )
    return AuditDiff(
        added_spans=tuple(added_spans),
        removed_spans=tuple(removed_spans),
        changed_spans=tuple(changed_spans),
        threshold_changes=tuple(
            _diff_thresholds(
                before_report.get("thresholds"),
                after_report.get("thresholds"),
            )
        ),
        residual_risk_delta=tuple(
            _diff_residual_risk(
                before_report.get("residual_risk"),
                after_report.get("residual_risk"),
            )
        ),
    )


def _coerce_report(report: AuditReportInput) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return copy.deepcopy(dict(report))
    if isinstance(report, (str, Path)):
        try:
            payload = json.loads(Path(report).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in audit report {report}: {exc}") from exc
    elif hasattr(report, "to_dict") and callable(report.to_dict):
        payload = report.to_dict()
    else:
        raise TypeError("audit report must be a mapping, path, or to_dict() object")
    if not isinstance(payload, Mapping):
        raise ValueError("audit report JSON must contain an object")
    return copy.deepcopy(dict(payload))


def _report_spans(report: Mapping[str, Any]) -> list[SpanSnapshot]:
    spans = report.get("spans") or []
    if not isinstance(spans, Sequence) or isinstance(spans, (str, bytes)):
        return []
    return [_span_snapshot(span) for span in spans if isinstance(span, Mapping)]


def _span_snapshot(span: Mapping[str, Any]) -> SpanSnapshot:
    return SpanSnapshot(
        start=_int(span.get("start")),
        end=_int(span.get("end")),
        text_hash=str(
            span.get("text_hash")
            or span.get("original_text_hash")
            or span.get("hash")
            or ""
        ),
        label=str(span.get("label") or span.get("entity_type") or ""),
        canonical_label=str(
            span.get("canonical_label")
            or span.get("normalized_label")
            or span.get("entity_type")
            or ""
        ),
        action=str(span.get("action") or span.get("policy_action") or ""),
        confidence=_optional_float(span.get("confidence")),
        threshold=_optional_float(span.get("threshold")),
        sources=_sources(span.get("sources", span.get("source", ()))),
    )


def _diff_spans(
    before_spans: Sequence[SpanSnapshot],
    after_spans: Sequence[SpanSnapshot],
) -> tuple[list[SpanSnapshot], list[SpanSnapshot], list[SpanChange]]:
    before_groups = _span_groups(before_spans)
    after_groups = _span_groups(after_spans)
    added: list[SpanSnapshot] = []
    removed: list[SpanSnapshot] = []
    changed: list[SpanChange] = []

    identities = sorted(
        set(before_groups) | set(after_groups),
        key=lambda item: (item[0], item[1], item[2]),
    )
    for identity in identities:
        before_group = before_groups.get(identity, [])
        after_group = after_groups.get(identity, [])
        matched = min(len(before_group), len(after_group))
        for index in range(matched):
            changes = _span_field_changes(before_group[index], after_group[index])
            if changes:
                changed.append(
                    SpanChange(
                        before=before_group[index],
                        after=after_group[index],
                        changes=tuple(changes),
                    )
                )
        removed.extend(before_group[matched:])
        added.extend(after_group[matched:])

    return (
        sorted(added, key=lambda span: span.sort_key()),
        sorted(removed, key=lambda span: span.sort_key()),
        sorted(changed, key=lambda change: change.sort_key()),
    )


def _span_groups(
    spans: Sequence[SpanSnapshot],
) -> dict[tuple[int, int, str], list[SpanSnapshot]]:
    groups: dict[tuple[int, int, str], list[SpanSnapshot]] = {}
    for span in sorted(spans, key=lambda item: item.sort_key()):
        groups.setdefault(span.identity, []).append(span)
    return groups


def _span_field_changes(
    before: SpanSnapshot,
    after: SpanSnapshot,
) -> list[FieldChange]:
    fields = (
        ("label", before.label, after.label),
        ("canonical_label", before.canonical_label, after.canonical_label),
        ("action", before.action, after.action),
        ("confidence", before.confidence, after.confidence),
        ("threshold", before.threshold, after.threshold),
        ("sources", list(before.sources), list(after.sources)),
    )
    return [
        FieldChange(field=field, before=before_value, after=after_value)
        for field, before_value, after_value in fields
        if before_value != after_value
    ]


def _diff_thresholds(before: Any, after: Any) -> list[ThresholdChange]:
    before_flat = _flatten_scalar_leaves(before)
    after_flat = _flatten_scalar_leaves(after)
    changes: list[ThresholdChange] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        before_value = before_flat.get(path, _MISSING)
        after_value = after_flat.get(path, _MISSING)
        if before_value == after_value:
            continue
        metadata = _threshold_path_metadata(path)
        changes.append(
            ThresholdChange(
                path=path,
                before=None if before_value is _MISSING else before_value,
                after=None if after_value is _MISSING else after_value,
                before_present=before_value is not _MISSING,
                after_present=after_value is not _MISSING,
                label=metadata.get("label"),
                language=metadata.get("language"),
                model_id=metadata.get("model_id"),
                profile=metadata.get("profile"),
                metric=metadata.get("metric"),
            )
        )
    return changes


def _diff_residual_risk(before: Any, after: Any) -> list[ResidualRiskDelta]:
    before_flat = _flatten_numeric_leaves(before)
    after_flat = _flatten_numeric_leaves(after)
    changes: list[ResidualRiskDelta] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        before_value = before_flat.get(path, _MISSING)
        after_value = after_flat.get(path, _MISSING)
        if before_value == after_value:
            continue
        delta = None
        if before_value is not _MISSING and after_value is not _MISSING:
            delta = round(float(after_value) - float(before_value), 12)
        changes.append(
            ResidualRiskDelta(
                path=path,
                before=None if before_value is _MISSING else float(before_value),
                after=None if after_value is _MISSING else float(after_value),
                delta=delta,
                before_present=before_value is not _MISSING,
                after_present=after_value is not _MISSING,
            )
        )
    return changes


def _flatten_scalar_leaves(
    value: Any, path: tuple[str, ...] = ()
) -> dict[tuple[str, ...], Any]:
    if isinstance(value, Mapping):
        leaves: dict[tuple[str, ...], Any] = {}
        for key in sorted(value, key=lambda item: str(item)):
            leaves.update(_flatten_scalar_leaves(value[key], (*path, str(key))))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_flatten_scalar_leaves(item, (*path, str(index))))
        return leaves
    if path and _is_json_scalar(value):
        return {path: _json_safe(value)}
    return {}


def _flatten_numeric_leaves(
    value: Any,
    path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], float]:
    if isinstance(value, Mapping):
        leaves: dict[tuple[str, ...], float] = {}
        for key in sorted(value, key=lambda item: str(item)):
            leaves.update(_flatten_numeric_leaves(value[key], (*path, str(key))))
        return leaves
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(_flatten_numeric_leaves(item, (*path, str(index))))
        return leaves
    if path and _is_number(value):
        return {path: float(value)}
    return {}


def _threshold_path_metadata(path: tuple[str, ...]) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {
        "label": None,
        "language": None,
        "model_id": None,
        "profile": None,
        "metric": None,
    }
    parts = list(path)
    if "profiles" in parts:
        index = parts.index("profiles")
        if len(parts) > index + 1:
            metadata["profile"] = parts[index + 1]
    if "labels" in parts:
        index = parts.index("labels")
        if len(parts) > index + 1:
            metadata["label"] = parts[index + 1]
        if len(parts) > index + 2:
            metadata["language"] = parts[index + 2]
        if len(parts) > index + 3:
            metadata["metric"] = ".".join(parts[index + 3 :])
        return metadata

    if len(parts) == 1:
        metadata["label"] = parts[0]
    elif len(parts) == 2:
        metadata["label"], metadata["language"] = parts
    elif len(parts) == 3:
        metadata["model_id"], metadata["label"], metadata["language"] = parts
    elif len(parts) >= 4:
        metadata["model_id"] = parts[-4]
        metadata["label"] = parts[-3]
        metadata["language"] = parts[-2]
        metadata["metric"] = parts[-1]
    return metadata


def _sources(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, Sequence):
        candidates = [str(item) for item in value]
    else:
        candidates = []
    return tuple(sorted({candidate for candidate in candidates if candidate}))


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _optional_float(value: Any) -> float | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _is_number(value: Any) -> bool:
    if isinstance(value, bool) or value in (None, ""):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if _is_json_scalar(value):
        return value
    return str(value)


def _sort_value(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _format_value(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
    )


def _path_key(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<root>"


def _hash_preview(text_hash: str) -> str:
    if len(text_hash) <= 24:
        return text_hash
    return f"{text_hash[:21]}..."


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _span_markdown_row(kind: str, span: SpanSnapshot, details: str) -> str:
    summary = details or (
        f"label={_format_value(span.display_label)}, "
        f"action={_format_value(span.action)}"
    )
    return (
        f"| {kind} | {span.start} | {span.end} | "
        f"{_markdown_cell(_hash_preview(span.text_hash))} | "
        f"{_markdown_cell(summary)} |"
    )


__all__ = [
    "AuditDiff",
    "FieldChange",
    "ResidualRiskDelta",
    "SpanChange",
    "SpanSnapshot",
    "ThresholdChange",
    "diff_audit_reports",
]
