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
import stat
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
_HASHED_KEY_RE = re.compile(r"^(?:action|category|count):sha256:[0-9a-f]{64}$")
_MAX_JSON_BYTES = 1024 * 1024
_MAX_CONTAINER_ITEMS = 4096
_MAX_NESTING_DEPTH = 16
_MAX_TEXT_CHARS = 16_384
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
        "total_count",
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
_POLICY_KEYS = (
    "policy_fingerprint",
    "policy_hash",
    "policy",
    "policy_name",
    "policy_profile",
)
_METADATA_KEYS = frozenset({"policy_fingerprint", "policy_hash"})
_SUMMARY_KEYS = frozenset(
    _ACTION_ALIASES
    + _CATEGORY_ALIASES
    + _NESTED_SUMMARY_KEYS
    + _COUNT_SECTION_KEYS
    + _COUNT_ALIASES
    + _POLICY_KEYS
    + ("metadata",)
)
_CATEGORY_RECORD_KEYS = frozenset(
    ("category", "label") + _CATEGORY_COUNT_FIELDS + _ACTION_ALIASES + tuple(_ACTIONS)
)


class _InvalidSummary(ValueError):
    """Internal marker for a safely worded invalid-summary failure."""


class _InvalidSummaryType(TypeError):
    """Internal marker for a safely worded invalid-summary type failure."""


def _is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _classify_change(before: int, after: int) -> ChangeClassification:
    if before == 0:
        return "added"
    if after == 0:
        return "removed"
    return "increased" if after > before else "decreased"


def _is_safe_change_key(value: Any) -> bool:
    return isinstance(value, str) and (
        value in _ACTIONS
        or value in CANONICAL_LABELS
        or value in _SAFE_COUNT_KEYS
        or _HASHED_KEY_RE.fullmatch(value) is not None
    )


def _bounded_text(value: str) -> str:
    if len(value) > _MAX_TEXT_CHARS:
        raise _InvalidSummary("redaction summary exceeds text limits")
    return value


def _validate_fingerprint(value: str | None) -> None:
    if value is not None and (
        not isinstance(value, str) or _FINGERPRINT_RE.fullmatch(value) is None
    ):
        raise ValueError("policy fingerprint is invalid")


def _validate_change_dimension(
    changes: tuple[CountChange, ...], dimension: Literal["action", "category", "count"]
) -> None:
    if type(changes) is not tuple or not all(
        isinstance(change, CountChange) for change in changes
    ):
        raise ValueError("redaction changes must be a tuple of CountChange records")
    keys = [change.key for change in changes]
    if keys != sorted(set(keys)):
        raise ValueError("redaction changes must have unique sorted keys")
    known_keys: frozenset[str]
    if dimension == "action":
        known_keys = _ACTIONS
    elif dimension == "category":
        known_keys = frozenset(CANONICAL_LABELS)
    else:
        known_keys = _SAFE_COUNT_KEYS
    prefix = f"{dimension}:sha256:"
    if any(key not in known_keys and not key.startswith(prefix) for key in keys):
        raise ValueError("redaction change key has the wrong dimension")


@dataclass(frozen=True)
class CountChange:
    """A value-free change for one aggregate key."""

    key: str
    before: int
    after: int
    delta: int
    classification: ChangeClassification

    def __post_init__(self) -> None:
        """Reject records that could carry values or contradict their counts."""

        if not _is_safe_change_key(self.key):
            raise ValueError("change key is not an aggregate identifier")
        if not _is_count(self.before) or not _is_count(self.after):
            raise ValueError("change counts must be non-negative integers")
        if self.before == self.after:
            raise ValueError("change counts must differ")
        expected_delta = self.after - self.before
        if type(self.delta) is not int or self.delta != expected_delta:
            raise ValueError("change delta does not match its counts")
        if self.classification != _classify_change(self.before, self.after):
            raise ValueError("change classification does not match its counts")

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

    def __post_init__(self) -> None:
        """Enforce deterministic, value-free result invariants."""

        _validate_fingerprint(self.before_policy_fingerprint)
        _validate_fingerprint(self.after_policy_fingerprint)
        _validate_change_dimension(self.action_changes, "action")
        _validate_change_dimension(self.category_changes, "category")
        _validate_change_dimension(self.count_changes, "count")

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

    try:
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
    except MemoryError:
        raise
    except _InvalidSummaryType as exc:
        raise TypeError(str(exc)) from None
    except _InvalidSummary as exc:
        raise ValueError(str(exc)) from None
    except Exception:
        raise ValueError("redaction summaries are invalid") from None


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

    try:
        return _policy_fingerprint(policy)
    except MemoryError:
        raise
    except Exception:
        raise ValueError("policy is invalid") from None


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
        payload = _copy_mapping(value)
        _validate_summary_source(payload, allow_nested=True)
        return payload

    if isinstance(value, Path):
        return _read_summary_path(value)

    if isinstance(value, str):
        _bounded_text(value)
        try:
            path = Path(value)
        except (OSError, ValueError):
            raise _InvalidSummaryType(
                "redaction summary must be a mapping, local JSON path, or "
                "object exposing to_dict()"
            ) from None
        return _read_summary_path(path)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            raise _InvalidSummaryType("could not read redaction summary") from None
        if isinstance(payload, Mapping):
            result = _copy_mapping(payload)
            _validate_summary_source(result, allow_nested=True)
            return result

    raise _InvalidSummaryType(
        "redaction summary must be a mapping, local JSON path, or object "
        "exposing to_dict()"
    )


def _read_summary_path(path: Path) -> dict[str, Any]:
    try:
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size > _MAX_JSON_BYTES:
            raise _InvalidSummary("redaction summary JSON exceeds file limits")
        with path.open("rb") as stream:
            raw = stream.read(_MAX_JSON_BYTES + 1)
        if len(raw) > _MAX_JSON_BYTES:
            raise _InvalidSummary("redaction summary JSON exceeds file limits")
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_json_object_no_duplicates
        )
    except _InvalidSummary:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise _InvalidSummary("could not read redaction summary JSON") from None
    if not isinstance(payload, Mapping):
        raise _InvalidSummary("redaction summary JSON must contain an object")
    result = _copy_mapping(payload)
    _validate_summary_source(result, allow_nested=True)
    return result


def _json_object_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) > _MAX_CONTAINER_ITEMS:
        raise _InvalidSummary("redaction summary exceeds container limits")
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidSummary("redaction summary JSON contains duplicate keys")
        result[key] = value
    return result


def _bounded_items(value: Mapping[Any, Any]) -> list[tuple[Any, Any]]:
    result: list[tuple[Any, Any]] = []
    iterator = iter(value.items())
    for _ in range(_MAX_CONTAINER_ITEMS + 1):
        try:
            item = next(iterator)
        except StopIteration:
            return result
        if not isinstance(item, tuple) or len(item) != 2:
            raise _InvalidSummary("redaction summary mapping is invalid")
        result.append(item)
    raise _InvalidSummary("redaction summary exceeds container limits")


def _bounded_sequence(value: Sequence[Any]) -> list[Any]:
    result: list[Any] = []
    iterator = iter(value)
    for _ in range(_MAX_CONTAINER_ITEMS + 1):
        try:
            result.append(next(iterator))
        except StopIteration:
            return result
    raise _InvalidSummary("redaction summary exceeds container limits")


def _copy_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in _bounded_items(value):
        if not isinstance(key, str):
            raise _InvalidSummary("redaction summary keys must be strings")
        if key in result:
            raise _InvalidSummary("redaction summary contains duplicate keys")
        result[key] = item
    return result


def _validate_summary_source(source: Mapping[str, Any], *, allow_nested: bool) -> None:
    items = _bounded_items(source)
    keys = {key for key, _ in items if isinstance(key, str)}
    if len(keys) != len(items) or not keys <= _SUMMARY_KEYS:
        raise _InvalidSummary("redaction summary contains unsupported fields")

    nested = [key for key in _NESTED_SUMMARY_KEYS if key in keys]
    if nested and not allow_nested:
        raise _InvalidSummary("nested redaction summaries are not supported")
    if len(nested) > 1:
        raise _InvalidSummary("redaction summary contains ambiguous sections")
    if nested:
        nested_value = source[nested[0]]
        if not isinstance(nested_value, Mapping):
            raise _InvalidSummary("nested redaction summary must be an object")
        _validate_summary_source(nested_value, allow_nested=False)

    metadata = source.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, Mapping):
            raise _InvalidSummary("redaction summary metadata must be an object")
        metadata_items = _bounded_items(metadata)
        metadata_keys = {key for key, _ in metadata_items if isinstance(key, str)}
        if (
            len(metadata_keys) != len(metadata_items)
            or not metadata_keys <= _METADATA_KEYS
        ):
            raise _InvalidSummary(
                "redaction summary metadata contains unsupported fields"
            )


def _summary_sources(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    sources: list[Mapping[str, Any]] = [payload]
    for key in _NESTED_SUMMARY_KEYS:
        if key in payload:
            value = payload[key]
            if not isinstance(value, Mapping):
                raise _InvalidSummary("nested redaction summary must be an object")
            sources.append(value)
    return tuple(sources)


def _extract_counts(payload: Mapping[str, Any]) -> _SummaryCounts:
    sources = _summary_sources(payload)
    categories, derived_actions = _category_counts(sources)
    actions = _action_counts(sources)
    if actions is None:
        actions = derived_actions
    return _SummaryCounts(
        actions=actions,
        categories=categories,
        counts=_count_summary(sources),
    )


def _single_alias_value(
    sources: Sequence[Mapping[str, Any]], aliases: Sequence[str]
) -> Any:
    matches: list[Any] = []
    for source in sources:
        for key in aliases:
            if key in source:
                matches.append(source[key])
        for section_key in _COUNT_SECTION_KEYS:
            if section_key not in source:
                continue
            section = source[section_key]
            if not isinstance(section, Mapping):
                raise _InvalidSummary("redaction count section must be an object")
            for key in aliases:
                if key in section:
                    matches.append(section[key])
    if len(matches) > 1:
        raise _InvalidSummary("redaction summary contains ambiguous aggregate sections")
    return matches[0] if matches else _MISSING


_MISSING = object()


def _action_counts(
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, int] | None:
    value = _single_alias_value(sources, _ACTION_ALIASES)
    if value is _MISSING:
        return None
    if not isinstance(value, Mapping):
        raise _InvalidSummary("redaction action counts must be an object")
    return _parse_numeric_mapping(value, dimension="action")


def _category_counts(
    sources: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    value = _single_alias_value(sources, _CATEGORY_ALIASES)
    if value is _MISSING:
        return {}, {}
    return _parse_category_value(value)


def _count_summary(sources: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    sections: list[Mapping[str, Any]] = []
    direct: dict[str, int] = {}
    for source in sources:
        for section_key in _COUNT_SECTION_KEYS:
            if section_key not in source:
                continue
            section = source[section_key]
            if not isinstance(section, Mapping):
                raise _InvalidSummary("redaction count section must be an object")
            sections.append(section)
        for key in _COUNT_ALIASES:
            if key in source:
                if key in direct:
                    raise _InvalidSummary("redaction summary contains ambiguous counts")
                direct[_safe_count_key(key)] = _as_count(source[key])
    if len(sections) > 1 or sections and direct:
        raise _InvalidSummary("redaction summary contains ambiguous count sections")
    if sections:
        return _parse_count_section(sections[0])
    return direct


def _parse_count_section(value: Mapping[str, Any], *, depth: int = 0) -> dict[str, int]:
    if depth > _MAX_NESTING_DEPTH:
        raise _InvalidSummary("redaction summary exceeds nesting limits")
    result: dict[str, int] = {}
    for raw_key, raw_value in _bounded_items(value):
        if raw_key in _ACTION_ALIASES or raw_key in _CATEGORY_ALIASES:
            continue
        if isinstance(raw_value, Mapping):
            nested = _parse_count_section(raw_value, depth=depth + 1)
            for key, count in nested.items():
                compound = {"parent": _stable_value(raw_key), "child": key}
                result[_hashed_key("count", compound)] = count
            continue
        result[_safe_count_key(raw_key)] = _as_count(raw_value)
    return result


def _parse_numeric_mapping(value: Any, *, dimension: Literal["action", "category"]):
    if not isinstance(value, Mapping):
        raise _InvalidSummary("redaction aggregate counts must be an object")

    result: dict[str, int] = {}
    for raw_key, raw_value in _bounded_items(value):
        if (
            isinstance(raw_value, Mapping)
            or isinstance(
                raw_value,
                Sequence,
            )
            and not isinstance(raw_value, (str, bytes))
        ):
            raise _InvalidSummary("redaction aggregate counts must be integers")
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
        for raw_category, raw_value in _bounded_items(value):
            category_key = _safe_category_key(raw_category)
            if not isinstance(raw_value, Mapping):
                category_counts[category_key] = category_counts.get(
                    category_key, 0
                ) + _as_count(raw_value)
                continue

            record = _copy_record(raw_value, keyed=True)
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

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        sequence_category_counts: dict[str, int] = {}
        sequence_action_counts: dict[str, int] = {}
        for raw_record in _bounded_sequence(value):
            if not isinstance(raw_record, Mapping):
                raise _InvalidSummary("redaction category records must be objects")
            record = _copy_record(raw_record, keyed=False)
            raw_category = (
                record["category"] if "category" in record else record["label"]
            )
            category_key = _safe_category_key(raw_category)
            category_count = _record_count(record)
            nested_actions = _record_action_counts(record)
            if category_count is None and nested_actions:
                category_count = sum(nested_actions.values())
            if category_count is not None:
                sequence_category_counts[category_key] = (
                    sequence_category_counts.get(category_key, 0) + category_count
                )
            for action, count in nested_actions.items():
                sequence_action_counts[action] = (
                    sequence_action_counts.get(action, 0) + count
                )
        return sequence_category_counts, sequence_action_counts

    raise _InvalidSummary("redaction category counts must be an object or array")


def _copy_record(value: Mapping[Any, Any], *, keyed: bool) -> dict[str, Any]:
    record = _copy_mapping(value)
    keys = set(record)
    if not keys <= _CATEGORY_RECORD_KEYS:
        raise _InvalidSummary("redaction category record contains unsupported fields")
    identity_keys = keys & {"category", "label"}
    if keyed and identity_keys or not keyed and len(identity_keys) != 1:
        raise _InvalidSummary("redaction category record has ambiguous identity")
    if len(keys & set(_CATEGORY_COUNT_FIELDS)) > 1:
        raise _InvalidSummary("redaction category record has ambiguous counts")
    action_aliases = keys & set(_ACTION_ALIASES)
    direct_actions = keys & _ACTIONS
    if len(action_aliases) > 1 or action_aliases and direct_actions:
        raise _InvalidSummary("redaction category record has ambiguous actions")
    if not keys & (set(_CATEGORY_COUNT_FIELDS) | set(_ACTION_ALIASES) | _ACTIONS):
        raise _InvalidSummary("redaction category record has no aggregate counts")
    return record


def _record_count(record: Mapping[str, Any]) -> int | None:
    for key in _CATEGORY_COUNT_FIELDS:
        if key in record:
            return _as_count(record[key])
    return None


def _record_action_counts(record: Mapping[str, Any]) -> dict[str, int]:
    for key in _ACTION_ALIASES:
        if key in record:
            return _parse_numeric_mapping(record[key], dimension="action")

    direct: dict[str, int] = {}
    for raw_key, raw_value in record.items():
        if raw_key in _ACTIONS:
            direct[_safe_action_key(raw_key)] = _as_count(raw_value)
    return direct


def _as_count(value: Any) -> int:
    if isinstance(value, bool):
        raise _InvalidSummary("redaction summary counts must be non-negative integers")
    if isinstance(value, Integral):
        count = int(value)
    elif isinstance(value, Real) and math.isfinite(float(value)):
        numeric = float(value)
        if not numeric.is_integer():
            raise _InvalidSummary(
                "redaction summary counts must be non-negative integers"
            )
        count = int(numeric)
    else:
        raise _InvalidSummary("redaction summary counts must be non-negative integers")
    if count < 0:
        raise _InvalidSummary("redaction summary counts must be non-negative integers")
    return count


def _safe_action_key(value: Any) -> str:
    if isinstance(value, str):
        action = _bounded_text(value).strip()
        if action in _ACTIONS:
            return action
        return _hashed_key("action", action)
    return _hashed_key("action", _stable_value(value))


def _safe_category_key(value: Any) -> str:
    if isinstance(value, str):
        category = _bounded_text(value).strip()
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
    if isinstance(value, str):
        text = _bounded_text(value).strip().lower()
        if text in _SAFE_COUNT_KEYS:
            return text
        return _hashed_key("count", text)
    return _hashed_key("count", _stable_value(value))


def _hashed_key(namespace: str, value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{namespace}:sha256:{digest}"


def _policy_hash(value: Any) -> str:
    encoded = _canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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
        classification = _classify_change(before_count, after_count)
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
    matches: list[tuple[str, Any]] = []
    for source in sources:
        for key in ("policy_fingerprint", "policy_hash"):
            if key in source:
                matches.append(("fingerprint", source[key]))

        metadata = source.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("policy_fingerprint", "policy_hash"):
                if key in metadata:
                    matches.append(("fingerprint", metadata[key]))

        for key in ("policy", "policy_name", "policy_profile"):
            if key in source:
                matches.append(("policy", source[key]))
    if len(matches) > 1:
        raise _InvalidSummary("redaction summary contains ambiguous policy metadata")
    if not matches or matches[0][1] is None:
        return None
    kind, value = matches[0]
    if kind == "fingerprint":
        result = _normalize_fingerprint(value)
        if result is None:
            raise _InvalidSummary("redaction summary policy fingerprint is invalid")
        return result
    return _policy_fingerprint(value)


def _policy_fingerprint(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, PolicyProfile):
        return _profile_fingerprint(value)
    if isinstance(value, PolicyName):
        return _profile_fingerprint(load_policy(value))

    if isinstance(value, str):
        candidate = _bounded_text(value).strip()
        if not candidate:
            return None
        normalized = _normalize_fingerprint(candidate)
        if normalized is not None:
            return normalized
        try:
            return _profile_fingerprint(load_policy(candidate))
        except (TypeError, ValueError, OSError):
            return _policy_hash(candidate)

    if isinstance(value, Mapping):
        policy = _copy_mapping(value)
        fingerprint_keys = set(policy) & {
            "policy_fingerprint",
            "policy_hash",
            "fingerprint",
        }
        if len(fingerprint_keys) > 1:
            raise _InvalidSummary("policy contains ambiguous fingerprints")
        if fingerprint_keys:
            normalized = _normalize_fingerprint(policy[next(iter(fingerprint_keys))])
            if normalized is None:
                raise _InvalidSummary("policy fingerprint is invalid")
            return normalized
        name_keys = set(policy) & {"name", "policy_name"}
        if len(name_keys) > 1:
            raise _InvalidSummary("policy contains ambiguous names")
        name = policy[next(iter(name_keys))] if name_keys else None
        if isinstance(name, str):
            try:
                profile = load_policy(name)
            except (TypeError, ValueError, OSError):
                profile = None
            if profile is not None and set(policy) <= {"name", "policy_name"}:
                return _profile_fingerprint(profile)
        return _policy_hash(policy)

    fingerprint = getattr(value, "fingerprint", _MISSING)
    if fingerprint is not _MISSING:
        normalized = _normalize_fingerprint(fingerprint)
        if normalized is None:
            raise _InvalidSummary("policy fingerprint is invalid")
        return normalized

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            raise _InvalidSummary("could not read policy") from None
        if not isinstance(payload, Mapping):
            raise _InvalidSummary("policy representation must be an object")
        return _policy_hash(payload)

    raise _InvalidSummary("policy type is unsupported")


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


def _stable_value(value: Any, *, depth: int = 0, seen: set[int] | None = None) -> Any:
    if depth > _MAX_NESTING_DEPTH:
        raise _InvalidSummary("redaction summary exceeds nesting limits")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _bounded_text(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        marker = id(value)
        if marker in seen:
            raise _InvalidSummary("redaction summary contains a cycle")
        seen.add(marker)
        try:
            result: dict[str, Any] = {}
            for key, item in _bounded_items(value):
                stable_key = _stable_value(key, depth=depth + 1, seen=seen)
                key_text = (
                    key
                    if isinstance(key, str)
                    else json.dumps(
                        stable_key,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                )
                if key_text in result:
                    raise _InvalidSummary("redaction summary has ambiguous keys")
                result[key_text] = _stable_value(item, depth=depth + 1, seen=seen)
            return {key: result[key] for key in sorted(result)}
        finally:
            seen.remove(marker)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        marker = id(value)
        if marker in seen:
            raise _InvalidSummary("redaction summary contains a cycle")
        seen.add(marker)
        try:
            return [
                _stable_value(item, depth=depth + 1, seen=seen)
                for item in _bounded_sequence(value)
            ]
        finally:
            seen.remove(marker)
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
