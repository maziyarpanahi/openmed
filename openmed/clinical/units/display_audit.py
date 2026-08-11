"""Deterministic, privacy-safe audits for localized unit display labels.

The audit compares a locale's ``canonical_code -> display label`` mapping with
an explicit alias table.  It is intentionally an aggregate artifact: reports
contain counts, normalized locale identifiers, and SHA-256 hashes, but never
the submitted display labels or canonical codes.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openmed.clinical.lexicons.clinical_norm import (
    get_clinical_norm_lexicon,
    normalize_language,
    normalize_unit_surface,
)

UNIT_DISPLAY_AUDIT_SCHEMA_VERSION = "openmed.clinical.unit-display-audit.v1"
UnitDisplayAuditIssueKind = Literal["missing", "duplicate", "conflict"]


def _sha256_text(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _stable_hash(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _canonical_code(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("unit codes must be strings")
    code = unicodedata.normalize("NFKC", value).strip()
    if not code:
        raise ValueError("unit codes must be non-empty strings")
    return code


def _locale_code(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("locale identifiers must be non-empty strings")
    return normalize_language(value)


@dataclass(frozen=True)
class UnitDisplayAuditIssue:
    """One privacy-safe unit-display audit finding.

    ``label_hashes`` and the code hash fields are SHA-256 digests.  They are
    deliberately the only representation of source labels and unit codes in
    the serialized finding.
    """

    kind: UnitDisplayAuditIssueKind
    reason: str
    locale: str
    count: int
    label_hashes: tuple[str, ...] = ()
    canonical_code_hashes: tuple[str, ...] = ()
    resolved_code_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"missing", "duplicate", "conflict"}:
            raise ValueError("unsupported unit-display issue kind")
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("unit-display issue count must be an integer")
        if self.count < 1:
            raise ValueError("unit-display issue count must be positive")
        for field_name in (
            "label_hashes",
            "canonical_code_hashes",
            "resolved_code_hashes",
        ):
            values = tuple(sorted(set(getattr(self, field_name))))
            object.__setattr__(self, field_name, values)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready finding without source strings."""

        return {
            "canonical_code_hashes": list(self.canonical_code_hashes),
            "count": self.count,
            "kind": self.kind,
            "label_hashes": list(self.label_hashes),
            "locale": self.locale,
            "reason": self.reason,
            "resolved_code_hashes": list(self.resolved_code_hashes),
        }


@dataclass(frozen=True)
class UnitDisplayAuditReport:
    """Aggregate, deterministic, and source-free unit-display audit report."""

    locales: tuple[str, ...]
    canonical_unit_count: int
    display_label_count: int
    issues: tuple[UnitDisplayAuditIssue, ...] = ()
    schema_version: str = UNIT_DISPLAY_AUDIT_SCHEMA_VERSION
    repro_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "locales", tuple(sorted(set(self.locales))))
        object.__setattr__(self, "issues", tuple(self.issues))
        for field_name in ("canonical_unit_count", "display_label_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer count")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not self.repro_hash:
            object.__setattr__(self, "repro_hash", self.recompute_repro_hash())

    @property
    def passed(self) -> bool:
        """Return whether no missing, duplicate, or conflict was found."""

        return not self.issues

    @property
    def summary(self) -> dict[str, int]:
        """Return deterministic issue and input counts."""

        counts = {"missing": 0, "duplicate": 0, "conflict": 0}
        for issue in self.issues:
            counts[issue.kind] += issue.count
        return {
            "canonical_units": self.canonical_unit_count,
            "conflict": counts["conflict"],
            "display_labels": self.display_label_count,
            "duplicate": counts["duplicate"],
            "issues": len(self.issues),
            "locales": len(self.locales),
            "missing": counts["missing"],
        }

    def _payload(self, *, include_repro_hash: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "canonical_unit_count": self.canonical_unit_count,
            "display_label_count": self.display_label_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "locales": list(self.locales),
            "passed": self.passed,
            "schema_version": self.schema_version,
            "status": "pass" if self.passed else "fail",
            "summary": self.summary,
        }
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        return payload

    def recompute_repro_hash(self) -> str:
        """Return the stable hash of the source-free report payload."""

        return _stable_hash(self._payload(include_repro_hash=False))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-compatible report."""

        return self._payload(include_repro_hash=True)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report to deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the source-free report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic Markdown summary containing only hashes."""

        lines = [
            "# Unit-display normalization audit",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Locales | {len(self.locales)} |",
            f"| Canonical units | {self.canonical_unit_count} |",
            f"| Display labels | {self.display_label_count} |",
            f"| Status | {'PASS' if self.passed else 'FAIL'} |",
            f"| Repro hash | `{self.repro_hash}` |",
            "",
            "## Findings",
            "",
        ]
        if not self.issues:
            lines.append("No unit-display findings.")
            return "\n".join(lines) + "\n"

        lines.extend(
            [
                "| Locale | Kind | Reason | Count | Label hashes | Code hashes | "
                "Resolved code hashes |",
                "|---|---|---|---:|---|---|---|",
            ]
        )
        for issue in self.issues:
            lines.append(
                "| "
                f"`{issue.locale}` | `{issue.kind}` | `{issue.reason}` | "
                f"{issue.count} | {_hashes_cell(issue.label_hashes)} | "
                f"{_hashes_cell(issue.canonical_code_hashes)} | "
                f"{_hashes_cell(issue.resolved_code_hashes)} |"
            )
        return "\n".join(lines) + "\n"

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for report consumers."""

        return self.to_dict()[key]


def _hashes_cell(values: tuple[str, ...]) -> str:
    if not values:
        return "-"
    return "<br>".join(f"`{value}`" for value in values)


def _canonical_codes(
    canonical_unit_codes: Iterable[str] | Mapping[str, object],
) -> tuple[str, ...]:
    if isinstance(canonical_unit_codes, Mapping):
        values: Iterable[object] = canonical_unit_codes.keys()
    elif isinstance(canonical_unit_codes, (str, bytes)):
        raise TypeError("canonical unit codes must be an iterable of strings")
    else:
        values = canonical_unit_codes

    codes = {_canonical_code(value) for value in values}
    return tuple(sorted(codes))


def _display_entries(
    locale_display_labels: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, str | None]]:
    if not isinstance(locale_display_labels, Mapping):
        raise TypeError("locale display labels must be a mapping")

    entries: dict[str, dict[str, str | None]] = {}
    for raw_locale, raw_labels in locale_display_labels.items():
        locale = _locale_code(raw_locale)
        if locale in entries:
            raise ValueError("locale identifiers must be unique after normalization")
        if not isinstance(raw_labels, Mapping):
            raise TypeError("each locale display-label set must be a mapping")

        locale_entries: dict[str, str | None] = {}
        for raw_code, raw_label in raw_labels.items():
            code = _canonical_code(raw_code)
            if code in locale_entries:
                raise ValueError("unit codes must be unique after normalization")
            if not isinstance(raw_label, str) or not normalize_unit_surface(raw_label):
                locale_entries[code] = None
            else:
                locale_entries[code] = raw_label
        entries[locale] = locale_entries
    return entries


def _alias_entries(
    locale_display_labels: Mapping[str, Mapping[str, object]],
    alias_tables: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
    if alias_tables is None:
        raw_tables: Mapping[str, Mapping[str, str]] = {
            locale: dict(get_clinical_norm_lexicon(locale).unit_aliases)
            for locale in locale_display_labels
        }
    else:
        raw_tables = alias_tables
    if not isinstance(raw_tables, Mapping):
        raise TypeError("alias tables must be a mapping")

    tables: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for raw_locale, raw_aliases in raw_tables.items():
        locale = _locale_code(raw_locale)
        if locale in tables:
            raise ValueError("alias locales must be unique after normalization")
        if not isinstance(raw_aliases, Mapping):
            raise TypeError("each alias table must be a mapping")
        normalized_aliases: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for raw_alias, raw_target in raw_aliases.items():
            if not isinstance(raw_alias, str) or not normalize_unit_surface(raw_alias):
                raise ValueError("alias labels must be non-empty strings")
            target = _canonical_code(raw_target)
            normalized_aliases[normalize_unit_surface(raw_alias)].append(
                (target, _sha256_text(raw_alias))
            )
        tables[locale] = normalized_aliases

    return {
        locale: {
            alias: tuple(sorted(records))
            for alias, records in sorted(normalized.items())
        }
        for locale, normalized in sorted(tables.items())
    }


def _issue(
    kind: UnitDisplayAuditIssueKind,
    reason: str,
    locale: str,
    count: int,
    *,
    label_hashes: Iterable[str] = (),
    canonical_code_hashes: Iterable[str] = (),
    resolved_code_hashes: Iterable[str] = (),
) -> UnitDisplayAuditIssue:
    return UnitDisplayAuditIssue(
        kind=kind,
        reason=reason,
        locale=locale,
        count=count,
        label_hashes=tuple(label_hashes),
        canonical_code_hashes=tuple(canonical_code_hashes),
        resolved_code_hashes=tuple(resolved_code_hashes),
    )


def _issue_sort_key(issue: UnitDisplayAuditIssue) -> tuple[Any, ...]:
    order = {"missing": 0, "duplicate": 1, "conflict": 2}
    return (
        issue.locale,
        order[issue.kind],
        issue.reason,
        issue.label_hashes,
        issue.canonical_code_hashes,
        issue.resolved_code_hashes,
    )


def audit_unit_display_labels(
    locale_display_labels: Mapping[str, Mapping[str, object]],
    canonical_unit_codes: Iterable[str] | Mapping[str, object],
    alias_tables: Mapping[str, Mapping[str, str]] | None = None,
) -> UnitDisplayAuditReport:
    """Audit localized unit labels against canonical codes and aliases.

    Args:
        locale_display_labels: Mapping of locale to ``canonical_code -> label``
            entries.  A missing or empty label is reported as missing.
        canonical_unit_codes: Iterable of allowed canonical unit codes, or a
            mapping whose keys are the allowed codes.
        alias_tables: Optional mapping of locale to ``label -> canonical_code``
            aliases.  When omitted, the registered local clinical normalization
            lexicons are used.  Passing a mapping makes the audit self-contained
            and ensures that only the supplied aliases are considered.

    Returns:
        A deterministic report whose findings contain counts and hashes only.

    Raises:
        TypeError or ValueError: If the audit inputs do not have the documented
            mapping shape.  Error messages never include submitted values.
    """

    canonical_codes = _canonical_codes(canonical_unit_codes)
    canonical_code_set = set(canonical_codes)
    display_entries = _display_entries(locale_display_labels)
    aliases = _alias_entries(display_entries, alias_tables)
    issues: list[UnitDisplayAuditIssue] = []
    display_label_count = 0

    for locale in sorted(display_entries):
        locale_aliases = aliases.get(locale, {})
        used_label_surfaces = {
            normalize_unit_surface(label)
            for label in display_entries[locale].values()
            if label is not None
        }

        for alias_surface, records in locale_aliases.items():
            targets = tuple(sorted({target for target, _ in records}))
            label_hashes = tuple(label_hash for _, label_hash in records)
            if len(targets) > 1:
                issues.append(
                    _issue(
                        "conflict",
                        "alias_table_conflict",
                        locale,
                        len(targets),
                        label_hashes=label_hashes,
                        canonical_code_hashes=(
                            _sha256_text(target) for target in targets
                        ),
                    )
                )
            unknown_targets = tuple(
                target for target in targets if target not in canonical_code_set
            )
            if unknown_targets and alias_surface in used_label_surfaces:
                issues.append(
                    _issue(
                        "conflict",
                        "alias_targets_unknown_code",
                        locale,
                        len(unknown_targets),
                        label_hashes=label_hashes,
                        resolved_code_hashes=(
                            _sha256_text(target) for target in unknown_targets
                        ),
                    )
                )

        labels_by_surface: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for code in sorted(display_entries[locale]):
            label = display_entries[locale][code]
            if label is None:
                continue
            display_label_count += 1
            label_surface = normalize_unit_surface(label)
            labels_by_surface[label_surface].append((code, _sha256_text(label)))

            label_hash = _sha256_text(label)
            if code not in canonical_code_set:
                issues.append(
                    _issue(
                        "conflict",
                        "display_references_unknown_code",
                        locale,
                        1,
                        label_hashes=(label_hash,),
                        canonical_code_hashes=(_sha256_text(code),),
                    )
                )
                continue

            alias_records = locale_aliases.get(label_surface, ())
            alias_targets = tuple(sorted({target for target, _ in alias_records}))
            if len(alias_targets) > 1:
                continue
            if alias_targets:
                target = alias_targets[0]
                if target not in canonical_code_set:
                    continue
                if target != code:
                    issues.append(
                        _issue(
                            "conflict",
                            "alias_resolves_to_different_code",
                            locale,
                            1,
                            label_hashes=(label_hash,),
                            canonical_code_hashes=(_sha256_text(code),),
                            resolved_code_hashes=(_sha256_text(target),),
                        )
                    )
                continue

            if normalize_unit_surface(code) != label_surface:
                issues.append(
                    _issue(
                        "conflict",
                        "label_not_canonical_or_explicit_alias",
                        locale,
                        1,
                        label_hashes=(label_hash,),
                        canonical_code_hashes=(_sha256_text(code),),
                    )
                )

        for label_surface, records in sorted(labels_by_surface.items()):
            if len(records) < 2:
                continue
            codes = tuple(sorted({code for code, _ in records}))
            if len(codes) < 2:
                continue
            issues.append(
                _issue(
                    "duplicate",
                    "display_label_used_for_multiple_codes",
                    locale,
                    len(codes),
                    label_hashes=(label_hash for _, label_hash in records),
                    canonical_code_hashes=(_sha256_text(code) for code in codes),
                )
            )

        missing_codes = tuple(
            code
            for code in canonical_codes
            if display_entries[locale].get(code) is None
        )
        if missing_codes:
            issues.append(
                _issue(
                    "missing",
                    "missing_display_label",
                    locale,
                    len(missing_codes),
                    canonical_code_hashes=(
                        _sha256_text(code) for code in missing_codes
                    ),
                )
            )

    ordered_issues = tuple(sorted(issues, key=_issue_sort_key))
    return UnitDisplayAuditReport(
        locales=tuple(sorted(display_entries)),
        canonical_unit_count=len(canonical_codes),
        display_label_count=display_label_count,
        issues=ordered_issues,
    )


audit_unit_displays = audit_unit_display_labels
run_unit_display_audit = audit_unit_display_labels


__all__ = [
    "UNIT_DISPLAY_AUDIT_SCHEMA_VERSION",
    "UnitDisplayAuditIssue",
    "UnitDisplayAuditIssueKind",
    "UnitDisplayAuditReport",
    "audit_unit_display_labels",
    "audit_unit_displays",
    "run_unit_display_audit",
]
