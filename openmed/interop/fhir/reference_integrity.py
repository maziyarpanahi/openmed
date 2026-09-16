"""Deterministic, counts-only FHIR Bundle reference-integrity reports.

The checker is intentionally structural rather than a complete FHIR validator.
It can be run after a local transformation to detect duplicate Bundle
identities, stale URL identities, and references that cannot be resolved inside
the Bundle.  R4 and R5 use the same JSON shape for the parts inspected here.

Only fixed finding codes, counts, and FHIRPath-style structural paths are
returned.  Full URLs, resource ids, reference strings, and resource payloads
are never placed in a report or an exception.  The implementation uses only
the Python standard library and performs no network access.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlsplit

__all__ = [
    "FHIRReferenceIntegrityReport",
    "REFERENCE_INTEGRITY_SCHEMA_VERSION",
    "ReferenceIntegrityFinding",
    "ReferenceIntegrityReport",
    "check_bundle_reference_integrity",
    "check_fhir_reference_integrity",
    "check_reference_integrity",
    "fhir_reference_integrity_report",
    "reference_integrity_report",
]

REFERENCE_INTEGRITY_SCHEMA_VERSION: Final = "openmed.fhir_reference_integrity.v1"

_FINDING_ORDER: Final[tuple[str, ...]] = (
    "ambiguous_reference",
    "dangling_reference",
    "duplicate_contained_identity",
    "duplicate_full_url",
    "duplicate_resource_identity",
    "full_url_identity_mismatch",
    "invalid_bundle_entry",
    "invalid_contained_resource",
    "invalid_full_url",
    "invalid_reference",
    "invalid_resource_identity",
    "invalid_resource_type",
)
_SUPPORTED_FHIR_VERSIONS: Final[frozenset[str]] = frozenset({"R4", "R5"})


@dataclass(frozen=True)
class ReferenceIntegrityFinding:
    """One aggregate integrity finding.

    ``paths`` contains only structural paths and is ordered deterministically.
    A path is included once for every count represented by the finding; no
    source value is retained.
    """

    code: str
    count: int
    paths: tuple[str, ...]

    @property
    def kind(self) -> str:
        """Return the stable finding code as a descriptive alias."""

        return self.code

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible, payload-free finding."""

        return {
            "code": self.code,
            "count": self.count,
            "paths": list(self.paths),
        }


@dataclass(frozen=True)
class ReferenceIntegrityReport:
    """Counts-only integrity results for one FHIR Bundle.

    The report is immutable and contains no resource values.  ``findings``
    records only non-zero finding codes; an empty tuple means the inspected
    structure is internally consistent under this checker.
    """

    entry_count: int
    resource_count: int
    contained_resource_count: int
    full_url_count: int
    reference_count: int
    findings: tuple[ReferenceIntegrityFinding, ...] = ()
    schema_version: str = REFERENCE_INTEGRITY_SCHEMA_VERSION

    @property
    def valid(self) -> bool:
        """Return whether no integrity findings were emitted."""

        return not self.findings

    @property
    def is_valid(self) -> bool:
        """Return :attr:`valid` as an explicit report-style alias."""

        return self.valid

    @property
    def ok(self) -> bool:
        """Return whether the Bundle passed the structural checks."""

        return self.valid

    @property
    def resources_inspected(self) -> int:
        """Return top-level plus contained resources inspected."""

        return self.resource_count + self.contained_resource_count

    @property
    def references_checked(self) -> int:
        """Return the number of non-empty string references examined."""

        return self.reference_count

    @property
    def finding_count(self) -> int:
        """Return the total number of counted integrity conflicts."""

        return sum(finding.count for finding in self.findings)

    @property
    def counts(self) -> dict[str, int]:
        """Return finding counts keyed by stable code."""

        return {finding.code: finding.count for finding in self.findings}

    @property
    def finding_counts(self) -> dict[str, int]:
        """Return :attr:`counts` as a report-oriented alias."""

        return self.counts

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report without source values."""

        counts = self.counts
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "entry_count": self.entry_count,
            "resource_count": self.resource_count,
            "contained_resource_count": self.contained_resource_count,
            "full_url_count": self.full_url_count,
            "reference_count": self.reference_count,
            "finding_counts": counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with stable key and finding ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True)
class _TopLevelResource:
    resource: Mapping[str, Any]
    path: str
    full_url: str | None
    contained_ids: Mapping[str, tuple[str, ...]]


class _FindingCollector:
    """Collect structural paths without retaining source payload values."""

    def __init__(self) -> None:
        self._paths: dict[str, list[str]] = defaultdict(list)

    def add(self, code: str, path: str) -> None:
        if code not in _FINDING_ORDER:
            raise ValueError("unknown reference-integrity finding code")
        self._paths[code].append(path)

    def add_many(self, code: str, paths: Sequence[str]) -> None:
        for path in paths:
            self.add(code, path)

    def findings(self) -> tuple[ReferenceIntegrityFinding, ...]:
        findings: list[ReferenceIntegrityFinding] = []
        for code in _FINDING_ORDER:
            paths = self._paths.get(code, [])
            if not paths:
                continue
            ordered_paths = tuple(_ordered_unique_paths(paths))
            findings.append(
                ReferenceIntegrityFinding(
                    code=code,
                    count=len(ordered_paths),
                    paths=ordered_paths,
                )
            )
        return tuple(findings)


def reference_integrity_report(
    bundle: Mapping[str, Any],
    *,
    fhir_version: str | None = None,
) -> ReferenceIntegrityReport:
    """Return a deterministic integrity report for a FHIR R4 or R5 Bundle.

    The checker resolves relative ``ResourceType/id`` references, exact
    ``fullUrl`` references, and local ``#contained-id`` references.  Absolute
    URLs are resolved when they match an entry ``fullUrl`` or the trailing
    ``ResourceType/id`` identity of an in-Bundle resource.  References that do
    not resolve to this Bundle are reported as ``dangling_reference``.

    Args:
        bundle: JSON-shaped FHIR Bundle mapping. The mapping is not mutated.
        fhir_version: Optional ``"R4"`` or ``"R5"`` contract marker. The
            inspected Bundle fields are shared by these FHIR versions; when
            omitted, both shapes are accepted.

    Returns:
        An immutable report containing counts and structural paths only.

    Raises:
        TypeError: If ``bundle`` is not mapping-shaped.
        ValueError: If the input is not a Bundle, ``entry`` is not an array,
            or an unsupported FHIR version is supplied.
    """

    _require_bundle(bundle)
    _validate_fhir_version(fhir_version)

    entries_value = bundle.get("entry", ())
    if entries_value is None:
        entries: Sequence[Any] = ()
    elif _is_sequence(entries_value):
        entries = entries_value
    else:
        raise ValueError("FHIR Bundle.entry must be an array when present")

    collector = _FindingCollector()
    full_urls: dict[str, list[str]] = defaultdict(list)
    identities: dict[tuple[str, str], list[str]] = defaultdict(list)
    top_level_resources: list[_TopLevelResource] = []
    resource_count = 0
    contained_resource_count = 0
    full_url_count = 0

    for entry_index, entry in enumerate(entries):
        entry_path = f"Bundle.entry[{entry_index}]"
        if not isinstance(entry, Mapping):
            collector.add("invalid_bundle_entry", entry_path)
            continue

        full_url_path = f"{entry_path}.fullUrl"
        full_url_value = entry.get("fullUrl")
        full_url: str | None = None
        if full_url_value is not None:
            if not isinstance(full_url_value, str) or not full_url_value.strip():
                collector.add("invalid_full_url", full_url_path)
            else:
                full_url = full_url_value
                full_urls[full_url].append(full_url_path)
                full_url_count += 1

        resource_value = entry.get("resource")
        if resource_value is None:
            continue
        if not isinstance(resource_value, Mapping):
            collector.add("invalid_bundle_entry", f"{entry_path}.resource")
            continue

        resource_count += 1
        resource_path = f"{entry_path}.resource"
        resource_type = resource_value.get("resourceType")
        resource_id = resource_value.get("id")
        if not isinstance(resource_type, str) or not resource_type.strip():
            collector.add("invalid_resource_type", f"{resource_path}.resourceType")
        elif resource_id is not None and not isinstance(resource_id, str):
            collector.add("invalid_resource_identity", f"{resource_path}.id")
        elif isinstance(resource_id, str) and not resource_id.strip():
            collector.add("invalid_resource_identity", f"{resource_path}.id")
        elif isinstance(resource_id, str):
            identity = (resource_type, resource_id)
            identities[identity].append(resource_path)

        if full_url is not None:
            _check_full_url_identity(
                full_url,
                resource_type,
                resource_id,
                full_url_path,
                collector,
            )

        contained_ids, contained_count = _collect_contained_resources(
            resource_value,
            resource_path,
            collector,
            identities=None,
        )
        contained_resource_count += contained_count
        top_level_resources.append(
            _TopLevelResource(
                resource=resource_value,
                path=resource_path,
                full_url=full_url,
                contained_ids=contained_ids,
            )
        )

    _add_duplicate_paths(full_urls, "duplicate_full_url", collector)
    _add_duplicate_paths(identities, "duplicate_resource_identity", collector)

    reference_count = 0
    for top_level in top_level_resources:
        reference_count += _inspect_references(
            top_level.resource,
            top_level.path,
            top_level.contained_ids,
            full_urls,
            identities,
            collector,
        )

    findings = collector.findings()
    return ReferenceIntegrityReport(
        entry_count=len(entries),
        resource_count=resource_count,
        contained_resource_count=contained_resource_count,
        full_url_count=full_url_count,
        reference_count=reference_count,
        findings=findings,
    )


def _require_bundle(bundle: Any) -> None:
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a FHIR Bundle mapping")
    if bundle.get("resourceType") != "Bundle":
        raise ValueError("bundle resourceType must be 'Bundle'")


def _validate_fhir_version(fhir_version: str | None) -> None:
    if fhir_version is None:
        return
    if not isinstance(fhir_version, str) or fhir_version.upper() not in (
        _SUPPORTED_FHIR_VERSIONS
    ):
        raise ValueError("fhir_version must be 'R4' or 'R5' when provided")


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _collect_contained_resources(
    resource: Mapping[str, Any],
    resource_path: str,
    collector: _FindingCollector,
    *,
    identities: dict[tuple[str, str], list[str]] | None,
) -> tuple[dict[str, tuple[str, ...]], int]:
    """Collect local ``#id`` targets and validate contained resource shape."""

    contained_value = resource.get("contained")
    if contained_value is None:
        return {}, 0
    if not _is_sequence(contained_value):
        collector.add("invalid_contained_resource", f"{resource_path}.contained")
        return {}, 0

    contained_ids: dict[str, list[str]] = defaultdict(list)
    contained_count = 0
    for contained_index, contained in enumerate(contained_value):
        contained_path = f"{resource_path}.contained[{contained_index}]"
        if not isinstance(contained, Mapping):
            collector.add("invalid_contained_resource", contained_path)
            continue
        contained_count += 1
        resource_type = contained.get("resourceType")
        resource_id = contained.get("id")
        if not isinstance(resource_type, str) or not resource_type.strip():
            collector.add("invalid_resource_type", f"{contained_path}.resourceType")
        elif resource_id is not None and not isinstance(resource_id, str):
            collector.add("invalid_resource_identity", f"{contained_path}.id")
        elif isinstance(resource_id, str) and not resource_id.strip():
            collector.add("invalid_resource_identity", f"{contained_path}.id")
        elif isinstance(resource_id, str):
            contained_ids[resource_id].append(contained_path)
            if identities is not None:
                identities[(resource_type, resource_id)].append(contained_path)

        nested_ids, nested_count = _collect_contained_resources(
            contained,
            contained_path,
            collector,
            identities=identities,
        )
        contained_count += nested_count
        for contained_id, paths in nested_ids.items():
            contained_ids[contained_id].extend(paths)

    _add_duplicate_paths(contained_ids, "duplicate_contained_identity", collector)
    return (
        {key: tuple(paths) for key, paths in contained_ids.items()},
        contained_count,
    )


def _check_full_url_identity(
    full_url: str,
    resource_type: Any,
    resource_id: Any,
    path: str,
    collector: _FindingCollector,
) -> None:
    if not isinstance(resource_type, str) or not isinstance(resource_id, str):
        return
    if not resource_type.strip() or not resource_id.strip():
        return
    identity = _identity_from_url(full_url)
    if identity is not None and identity != (resource_type, resource_id):
        collector.add("full_url_identity_mismatch", path)


def _identity_from_url(value: str) -> tuple[str, str] | None:
    """Return a trailing FHIR ``ResourceType/id`` identity when unambiguous."""

    parsed = urlsplit(value)
    path = parsed.path if parsed.scheme or parsed.netloc else value
    path = path.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = [part for part in path.split("/") if part]
    if len(parts) >= 4 and parts[-2] == "_history":
        parts = parts[:-2]
    if len(parts) < 2:
        return None
    resource_type, resource_id = parts[-2:]
    if not resource_type or not resource_id:
        return None
    if value.startswith("urn:"):
        return None
    return resource_type, resource_id


def _add_duplicate_paths(
    grouped_paths: Mapping[Any, Sequence[str]],
    code: str,
    collector: _FindingCollector,
) -> None:
    for paths in grouped_paths.values():
        if len(paths) > 1:
            collector.add_many(code, paths[1:])


def _inspect_references(
    value: Any,
    path: str,
    contained_ids: Mapping[str, Sequence[str]],
    full_urls: Mapping[str, Sequence[str]],
    identities: Mapping[tuple[str, str], Sequence[str]],
    collector: _FindingCollector,
) -> int:
    if isinstance(value, Mapping):
        reference_count = 0
        for key in _ordered_keys(value):
            child = value[key]
            child_path = _child_path(path, key)
            if key == "reference":
                if not isinstance(child, str) or not child.strip():
                    collector.add("invalid_reference", child_path)
                else:
                    reference_count += 1
                    _check_reference_target(
                        child,
                        child_path,
                        contained_ids,
                        full_urls,
                        identities,
                        collector,
                    )
            reference_count += _inspect_references(
                child,
                child_path,
                contained_ids,
                full_urls,
                identities,
                collector,
            )
        return reference_count
    if _is_sequence(value):
        return sum(
            _inspect_references(
                item,
                f"{path}[{index}]",
                contained_ids,
                full_urls,
                identities,
                collector,
            )
            for index, item in enumerate(value)
        )
    return 0


def _check_reference_target(
    reference: str,
    path: str,
    contained_ids: Mapping[str, Sequence[str]],
    full_urls: Mapping[str, Sequence[str]],
    identities: Mapping[tuple[str, str], Sequence[str]],
    collector: _FindingCollector,
) -> None:
    if reference.startswith("#"):
        contained_id = reference[1:]
        if not contained_id or "/" in contained_id:
            collector.add("invalid_reference", path)
            return
        matches = contained_ids.get(contained_id, ())
        _record_resolution(matches, path, collector)
        return

    exact_matches = full_urls.get(reference, ())
    if exact_matches:
        _record_resolution(exact_matches, path, collector)
        return

    identity = _identity_from_reference(reference)
    if identity is None:
        collector.add("dangling_reference", path)
        return
    matches = identities.get(identity, ())
    _record_resolution(matches, path, collector)


def _identity_from_reference(value: str) -> tuple[str, str] | None:
    candidate = value.split("?", 1)[0].split("#", 1)[0].strip("/")
    parts = [part for part in candidate.split("/") if part]
    if len(parts) >= 4 and parts[-2] == "_history":
        parts = parts[:-2]
    if len(parts) < 2:
        return None
    return parts[-2], parts[-1]


def _record_resolution(
    matches: Sequence[str],
    path: str,
    collector: _FindingCollector,
) -> None:
    if not matches:
        collector.add("dangling_reference", path)
    elif len(matches) > 1:
        collector.add("ambiguous_reference", path)


def _ordered_keys(value: Mapping[Any, Any]) -> tuple[Any, ...]:
    try:
        return tuple(sorted(value, key=lambda item: str(item)))
    except Exception:
        return tuple(value)


def _child_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key:
        return f"{path}.{key}"
    return f"{path}.field"


def _ordered_unique_paths(paths: Sequence[str]) -> list[str]:
    unique = set(paths)
    return sorted(unique, key=_path_sort_key)


def _path_sort_key(path: str) -> tuple[tuple[int, str | int], ...]:
    parts: list[tuple[int, str | int]] = []
    token = ""
    index = 0
    while index < len(path):
        character = path[index]
        if character == "[":
            if token:
                parts.append((0, token))
                token = ""
            end = path.find("]", index)
            if end == -1:
                token += path[index:]
                break
            number = path[index + 1 : end]
            parts.append((1, int(number) if number.isdigit() else number))
            index = end + 1
            continue
        token += character
        index += 1
    if token:
        parts.append((0, token))
    return tuple(parts)


# Descriptive aliases keep the module convenient for callers while preserving
# one implementation and one report schema.
check_reference_integrity = reference_integrity_report
check_fhir_reference_integrity = reference_integrity_report
check_bundle_reference_integrity = reference_integrity_report
fhir_reference_integrity_report = reference_integrity_report

FHIRReferenceIntegrityReport = ReferenceIntegrityReport
