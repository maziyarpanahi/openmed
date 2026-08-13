"""Value-free diffs for aggregate redaction result summaries.

The diff surface in this module intentionally accepts summaries rather than
documents or individual spans.  It compares only aggregate action, category,
and count data and records policy fingerprints so reviewers can understand
what changed without receiving source or replacement values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.policy import PolicyName, PolicyProfile, load_policy
from openmed.core.schemas.span import ACTION_VALUES

REDACTION_DIFF_SCHEMA_VERSION = 1

ChangeClassification = Literal["added", "removed", "increased", "decreased"]
RedactionSummaryInput = Mapping[str, Any] | str | Path

_ACTIONS = frozenset(ACTION_VALUES)
_FINGERPRINT_RE = re.compile(r"^(?:sha256|hmac-sha256):[0-9a-f]{64}$")
_SAFE_COUNT_KEYS = frozenset(
    {
        "added",
        "changed",
        "count",
        "affected_cells",
        "batch_count",
        "detection_count",
        "detections",
        "document_count",
        "documents",
        "errors",
        "hashed",
        "input_count",
        "kept",
        "masked",
        "output_count",
        "processed",
        "processed_cells",
        "records",
        "redacted",
        "redacted_cells",
        "redaction_count",
        "removed",
        "replaced",
        "residual_span_count",
        "rows",
        "span_count",
        "spans",
        "total",
        "total_rows",
        "total_spans",
        "unchanged",
        "warnings",
    }
)
_COUNT_ALIASES = (
    "count",
    "total_count",
    "total",
    "total_rows",
    "total_spans",
    "processed_cells",
    "redacted_cells",
    "detection_count",
    "redaction_count",
    "span_count",
    "residual_span_count",
    "document_count",
)
_ACTION_ALIASES = (
    "action_counts",
    "applied_action_counts",
    "action_summary",
    "actions",
    "by_action",
)
_CATEGORY_ALIASES = (
    "category_counts",
    "categories",
    "label_counts",
    "per_label_counts",
    "per_category_counts",
    "by_category",
    "redaction_counts_by_category",
    "redaction_counts_by_label",
    "span_counts",
)
_NESTED_SUMMARY_KEYS = ("summary", "redaction_summary", "result_summary")
_COUNT_SECTION_KEYS = ("counts", "count_summary", "totals")
_CATEGORY_COUNT_FIELDS = (
    "count",
    "detection_count",
    "redaction_count",
    "span_count",
    "total",
)


@dataclass(frozen=True)
class CountChange:
    """A value-free change for one aggregate key."""

    key: str
    before: int
    after: int
    delta: int
    classification: ChangeClassification

    @property
    def change_type(self) -> ChangeClassification:
        """Return the classification under its descriptive alias."""

        return self.classification

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible change record."""

        return {
            "key": self.key,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "classification": self.classification,
        }


# These aliases make the dimension represented by a change explicit to callers
# without creating three subtly different record formats.
RedactionCountChange = CountChange
ActionCountChange = CountChange
CategoryCountChange = CountChange


@dataclass(frozen=True)
class RedactionDiff:
    """Structured, aggregate-only difference between two redaction summaries."""

    before_policy_fingerprint: str | None
    after_policy_fingerprint: str | None
    action_changes: tuple[CountChange, ...]
    category_changes: tuple[CountChange, ...]
    count_changes: tuple[CountChange, ...]

    @property
    def base_policy_fingerprint(self) -> str | None:
        """Return the baseline policy fingerprint."""

        return self.before_policy_fingerprint

    @property
    def candidate_policy_fingerprint(self) -> str | None:
        """Return the candidate policy fingerprint."""

        return self.after_policy_fingerprint

    @property
    def policy_changed(self) -> bool:
        """Whether the summaries identify different policies."""

        return self.before_policy_fingerprint != self.after_policy_fingerprint and (
            self.before_policy_fingerprint is not None
            or self.after_policy_fingerprint is not None
        )

    @property
    def is_empty(self) -> bool:
        """Whether counts and policy identity are unchanged."""

        return not (
            self.action_changes
            or self.category_changes
            or self.count_changes
            or self.policy_changed
        )

    @property
    def summary(self) -> dict[str, Any]:
        """Return deterministic aggregate counts for the diff itself."""

        return {
            "action_changes": len(self.action_changes),
            "category_changes": len(self.category_changes),
            "count_changes": len(self.count_changes),
            "policy_changed": self.policy_changed,
            "total_changes": (
                len(self.action_changes)
                + len(self.category_changes)
                + len(self.count_changes)
                + int(self.policy_changed)
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free JSON-compatible diff."""

        return {
            "schema_version": REDACTION_DIFF_SCHEMA_VERSION,
            "policy_fingerprints": {
                "before": self.before_policy_fingerprint,
                "after": self.after_policy_fingerprint,
            },
            "policy_changed": self.policy_changed,
            "summary": self.summary,
            "action_changes": [change.to_dict() for change in self.action_changes],
            "category_changes": [change.to_dict() for change in self.category_changes],
            "count_changes": [change.to_dict() for change in self.count_changes],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the diff using deterministic JSON settings."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact Markdown diff containing no source values."""

        fingerprints = {
            "before": self.before_policy_fingerprint or "unavailable",
            "after": self.after_policy_fingerprint or "unavailable",
        }
        lines = [
            "## Redaction Summary Diff",
            "",
            f"Policy fingerprints: before `{fingerprints['before']}`, "
            f"after `{fingerprints['after']}`",
            "",
            "| Change type | Changes |",
            "|---|---:|",
            f"| Action | {len(self.action_changes)} |",
            f"| Category | {len(self.category_changes)} |",
            f"| Count | {len(self.count_changes)} |",
            f"| Policy | {int(self.policy_changed)} |",
        ]
        lines.extend(_changes_markdown("Action changes", self.action_changes))
        lines.extend(_changes_markdown("Category changes", self.category_changes))
        lines.extend(_changes_markdown("Count changes", self.count_changes))
        return "\n".join(lines)


def diff_redaction_summaries(
    before: RedactionSummaryInput | Any,
    after: RedactionSummaryInput | Any,
) -> RedactionDiff:
    """Compare two aggregate redaction summaries.

    ``before`` and ``after`` may be mappings, paths to local JSON objects, or
    objects exposing ``to_dict()``.  Only numeric aggregate fields and safe
    metadata labels are retained.  Unknown category and metric keys are
    represented by stable fingerprints so accidental source values are not
    copied into the returned report.

    The function performs no network access.  Named bundled policies are
    resolved from the local package when their fingerprints are not supplied
    explicitly.
    """

    before_payload = _coerce_summary(before)
    after_payload = _coerce_summary(after)
    before_counts = _extract_counts(before_payload)
    after_counts = _extract_counts(after_payload)

    return RedactionDiff(
        before_policy_fingerprint=_summary_policy_fingerprint(before_payload),
        after_policy_fingerprint=_summary_policy_fingerprint(after_payload),
        action_changes=_count_changes(
            before_counts.actions,
            after_counts.actions,
        ),
        category_changes=_count_changes(
            before_counts.categories,
            after_counts.categories,
        ),
        count_changes=_count_changes(
            before_counts.counts,
            after_counts.counts,
        ),
    )


def diff_redaction_results(
    before: RedactionSummaryInput | Any,
    after: RedactionSummaryInput | Any,
) -> RedactionDiff:
    """Alias for :func:`diff_redaction_summaries` for result-oriented callers."""

    return diff_redaction_summaries(before, after)


def diff_redaction_reports(
    before: RedactionSummaryInput | Any,
    after: RedactionSummaryInput | Any,
) -> RedactionDiff:
    """Alias for :func:`diff_redaction_summaries` for report-oriented callers."""

    return diff_redaction_summaries(before, after)


def fingerprint_policy(policy: Any) -> str | None:
    """Return a stable policy fingerprint without contacting external services."""

    return _policy_fingerprint(policy)


def policy_fingerprint(policy: Any) -> str | None:
    """Compatibility alias for :func:`fingerprint_policy`."""

    return fingerprint_policy(policy)


def render_redaction_diff(
    diff: RedactionDiff,
    fmt: Literal["text", "markdown", "dict", "json"] = "text",
) -> str | dict[str, Any]:
    """Render a :class:`RedactionDiff` as Markdown, a mapping, or JSON."""

    if not isinstance(diff, RedactionDiff):
        raise TypeError("diff must be a RedactionDiff")
    if fmt in {"text", "markdown"}:
        return diff.to_markdown()
    if fmt == "dict":
        return diff.to_dict()
    if fmt == "json":
        return diff.to_json()
    raise ValueError("fmt must be one of 'text', 'markdown', 'dict', or 'json'")


def render(
    diff: RedactionDiff,
    fmt: Literal["text", "markdown", "dict", "json"] = "text",
) -> str | dict[str, Any]:
    """Short alias for :func:`render_redaction_diff`."""

    return render_redaction_diff(diff, fmt=fmt)


@dataclass(frozen=True)
class _SummaryCounts:
    actions: dict[str, int]
    categories: dict[str, int]
    counts: dict[str, int]


def _coerce_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)

    if isinstance(value, Path):
        return _read_summary_path(value)

    if isinstance(value, str):
        try:
            path = Path(value)
        except (OSError, ValueError):
            raise TypeError(
                "redaction summary must be a mapping, local JSON path, or "
                "object exposing to_dict()"
            ) from None
        return _read_summary_path(path)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            raise TypeError("could not read redaction summary") from None
        if isinstance(payload, Mapping):
            return dict(payload)

    raise TypeError(
        "redaction summary must be a mapping, local JSON path, or object "
        "exposing to_dict()"
    )


def _read_summary_path(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("could not read redaction summary JSON") from None
    if not isinstance(payload, Mapping):
        raise ValueError("redaction summary JSON must contain an object")
    return dict(payload)


def _summary_sources(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [payload]
    for key in _NESTED_SUMMARY_KEYS:
        value = payload.get(key)
        if isinstance(value, Mapping):
            sources.append(value)
    return tuple(sources)


def _extract_counts(payload: Mapping[str, Any]) -> _SummaryCounts:
    sources = _summary_sources(payload)
    categories, derived_actions = _category_counts(sources)
    actions = _action_counts(sources)
    if not actions:
        actions = derived_actions
    return _SummaryCounts(
        actions=actions,
        categories=categories,
        counts=_count_summary(sources),
    )


def _action_counts(sources: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    for source in sources:
        for key in _ACTION_ALIASES:
            parsed = _parse_numeric_mapping(source.get(key), dimension="action")
            if parsed:
                return parsed

        for section_key in _COUNT_SECTION_KEYS:
            section = source.get(section_key)
            if not isinstance(section, Mapping):
                continue
            for key in _ACTION_ALIASES:
                parsed = _parse_numeric_mapping(
                    section.get(key),
                    dimension="action",
                )
                if parsed:
                    return parsed
    return {}


def _category_counts(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    for source in sources:
        for key in _CATEGORY_ALIASES:
            parsed = _parse_category_value(source.get(key))
            if parsed[0] or parsed[1]:
                return parsed

        for section_key in _COUNT_SECTION_KEYS:
            section = source.get(section_key)
            if not isinstance(section, Mapping):
                continue
            for key in _CATEGORY_ALIASES:
                parsed = _parse_category_value(section.get(key))
                if parsed[0] or parsed[1]:
                    return parsed
    return {}, {}


def _count_summary(sources: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    for source in sources:
        for section_key in _COUNT_SECTION_KEYS:
            section = source.get(section_key)
            if isinstance(section, Mapping):
                parsed = _parse_count_section(section)
                if parsed:
                    return parsed

        parsed: dict[str, int] = {}
        for key in _COUNT_ALIASES:
            if key in source:
                parsed[_safe_count_key(key)] = _as_count(source[key])
        if parsed:
            return parsed
    return {}


def _parse_count_section(value: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if isinstance(raw_value, Mapping):
            nested = _parse_count_section(raw_value)
            for key, count in nested.items():
                result[_safe_count_key(f"{raw_key}.{key}")] = count
            continue
        result[_safe_count_key(raw_key)] = _as_count(raw_value)
    return result


def _parse_numeric_mapping(value: Any, *, dimension: Literal["action", "category"]):
    if not isinstance(value, Mapping):
        return {}

    result: dict[str, int] = {}
    for raw_key, raw_value in value.items():
        if (
            isinstance(raw_value, Mapping)
            or isinstance(
                raw_value,
                Sequence,
            )
            and not isinstance(raw_value, (str, bytes))
        ):
            continue
        count = _as_count(raw_value)
        key = (
            _safe_action_key(raw_key)
            if dimension == "action"
            else _safe_category_key(raw_key)
        )
        result[key] = result.get(key, 0) + count
    return result


def _parse_category_value(value: Any) -> tuple[dict[str, int], dict[str, int]]:
    if isinstance(value, Mapping):
        category_counts: dict[str, int] = {}
        action_counts: dict[str, int] = {}
        for raw_category, raw_value in value.items():
            category_key = _safe_category_key(raw_category)
            if not isinstance(raw_value, Mapping):
                category_counts[category_key] = category_counts.get(
                    category_key, 0
                ) + _as_count(raw_value)
                continue

            category_count = _record_count(raw_value)
            nested_actions = _record_action_counts(raw_value)
            if category_count is None and nested_actions:
                category_count = sum(nested_actions.values())
            if category_count is not None:
                category_counts[category_key] = (
                    category_counts.get(category_key, 0) + category_count
                )
            for action, count in nested_actions.items():
                action_counts[action] = action_counts.get(action, 0) + count
        return category_counts, action_counts

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        category_counts = {}
        action_counts: dict[str, int] = {}
        for record in value:
            if not isinstance(record, Mapping):
                continue
            raw_category = record.get("category", record.get("label"))
            if raw_category is None:
                continue
            category_key = _safe_category_key(raw_category)
            category_count = _record_count(record)
            nested_actions = _record_action_counts(record)
            if category_count is None and nested_actions:
                category_count = sum(nested_actions.values())
            if category_count is not None:
                category_counts[category_key] = (
                    category_counts.get(category_key, 0) + category_count
                )
            for action, count in nested_actions.items():
                action_counts[action] = action_counts.get(action, 0) + count
        return category_counts, action_counts

    return {}, {}


def _record_count(record: Mapping[str, Any]) -> int | None:
    for key in _CATEGORY_COUNT_FIELDS:
        if key in record:
            return _as_count(record[key])
    return None


def _record_action_counts(record: Mapping[str, Any]) -> dict[str, int]:
    for key in _ACTION_ALIASES:
        parsed = _parse_numeric_mapping(record.get(key), dimension="action")
        if parsed:
            return parsed

    direct: dict[str, int] = {}
    for raw_key, raw_value in record.items():
        if raw_key in _ACTIONS:
            direct[_safe_action_key(raw_key)] = _as_count(raw_value)
    return direct


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("redaction summary counts must be non-negative integers")
    if isinstance(value, Integral):
        count = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        numeric = float(value)
        if not numeric.is_integer():
            raise ValueError("redaction summary counts must be non-negative integers")
        count = int(numeric)
    else:
        raise ValueError("redaction summary counts must be non-negative integers")
    if count < 0:
        raise ValueError("redaction summary counts must be non-negative integers")
    return count


def _safe_action_key(value: Any) -> str:
    if isinstance(value, str):
        action = value.strip()
        if action in _ACTIONS:
            return action
        return _hashed_key("action", action)
    return _hashed_key("action", _stable_value(value))


def _safe_category_key(value: Any) -> str:
    if isinstance(value, str):
        category = value.strip()
        try:
            canonical = normalize_label(category)
        except (TypeError, ValueError):
            canonical = ""
        if canonical in CANONICAL_LABELS and (
            canonical != "OTHER" or category.upper() == "OTHER"
        ):
            return canonical
        if category in CANONICAL_LABELS:
            return category
        return _hashed_key("category", category)
    return _hashed_key("category", _stable_value(value))


def _safe_count_key(value: Any) -> str:
    text = str(value).strip().lower()
    if text in _SAFE_COUNT_KEYS:
        return text
    return _hashed_key("count", text)


def _hashed_key(namespace: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _count_changes(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> tuple[CountChange, ...]:
    changes: list[CountChange] = []
    for key in sorted(set(before) | set(after)):
        before_count = int(before.get(key, 0))
        after_count = int(after.get(key, 0))
        if before_count == after_count:
            continue
        delta = after_count - before_count
        if before_count == 0:
            classification: ChangeClassification = "added"
        elif after_count == 0:
            classification = "removed"
        elif delta > 0:
            classification = "increased"
        else:
            classification = "decreased"
        changes.append(
            CountChange(
                key=key,
                before=before_count,
                after=after_count,
                delta=delta,
                classification=classification,
            )
        )
    return tuple(changes)


def _summary_policy_fingerprint(payload: Mapping[str, Any]) -> str | None:
    sources = _summary_sources(payload)
    for source in sources:
        for key in ("policy_fingerprint", "policy_hash"):
            if key in source:
                result = _policy_fingerprint(source[key])
                if result is not None:
                    return result

        metadata = source.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("policy_fingerprint", "policy_hash"):
                if key in metadata:
                    result = _policy_fingerprint(metadata[key])
                    if result is not None:
                        return result

        for key in ("policy", "policy_name", "policy_profile"):
            if key in source and source[key] is not None:
                result = _policy_fingerprint(source[key])
                if result is not None:
                    return result
    return None


def _policy_fingerprint(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, PolicyProfile):
        return _profile_fingerprint(value)
    if isinstance(value, PolicyName):
        return _profile_fingerprint(load_policy(value))

    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        normalized = _normalize_fingerprint(candidate)
        if normalized is not None:
            return normalized
        try:
            return _profile_fingerprint(load_policy(candidate))
        except (TypeError, ValueError, OSError):
            return _hashed_key("policy", candidate)

    if isinstance(value, Mapping):
        for key in ("policy_fingerprint", "policy_hash", "fingerprint"):
            if key in value:
                normalized = _normalize_fingerprint(value[key])
                if normalized is not None:
                    return normalized
        name = value.get("name", value.get("policy_name"))
        if isinstance(name, str):
            try:
                profile = load_policy(name)
            except (TypeError, ValueError, OSError):
                profile = None
            if profile is not None and set(value) <= {
                "name",
                "policy_name",
            }:
                return _profile_fingerprint(profile)
        return _hashed_key("policy", _stable_value(value))

    fingerprint = getattr(value, "fingerprint", None)
    if isinstance(fingerprint, str):
        normalized = _normalize_fingerprint(fingerprint)
        if normalized is not None:
            return normalized

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            return _hashed_key("policy", _stable_value(payload))

    return _hashed_key("policy", _stable_value(value))


def _profile_fingerprint(profile: PolicyProfile) -> str:
    encoded = _canonical_json(profile.to_dict()).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _normalize_fingerprint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if _FINGERPRINT_RE.fullmatch(candidate):
        return candidate
    return None


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_stable_value(item) for item in value]
    return {"type": type(value).__name__}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _stable_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _changes_markdown(
    title: str,
    changes: Sequence[CountChange],
) -> list[str]:
    lines = ["", f"### {title}"]
    if not changes:
        lines.append("No changes.")
        return lines
    lines.extend(
        [
            "",
            "| Key | Before | After | Delta | Classification |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for change in changes:
        lines.append(
            "| "
            f"{_markdown_cell(change.key)} | {change.before} | {change.after} | "
            f"{change.delta} | {change.classification} |"
        )
    return lines


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "ACTION_COUNT_CHANGE",
    "ActionCountChange",
    "CategoryCountChange",
    "ChangeClassification",
    "CountChange",
    "REDACTION_DIFF_SCHEMA_VERSION",
    "RedactionCountChange",
    "RedactionDiff",
    "RedactionSummaryInput",
    "diff_redaction_reports",
    "diff_redaction_results",
    "diff_redaction_summaries",
    "fingerprint_policy",
    "policy_fingerprint",
    "render",
    "render_redaction_diff",
]


# Keep the public spelling useful to code that treats action changes as a
# distinct dimension while retaining one stable record implementation.
ACTION_COUNT_CHANGE = ActionCountChange
