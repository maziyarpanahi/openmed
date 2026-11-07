"""Verify local evidence bundles without exposing protected values.

An evidence bundle is a directory containing a ``manifest.json`` file and
content-addressed evidence files.  The manifest is deliberately small: it
binds the files to a schema version, a policy fingerprint, required evidence
sections, and a safe provenance record.  Verification reads file bytes only
to calculate SHA-256 digests; no file contents, paths, or manifest values are
stored in the returned report.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, TypeAlias, cast

EVIDENCE_BUNDLE_SCHEMA_VERSION = "openmed.evidence_bundle.v1"
MANIFEST_FILENAME = "manifest.json"
DEFAULT_REQUIRED_SECTIONS = ("summary", "metrics", "provenance")
REQUIRED_PROVENANCE_FIELDS = (
    "source_fingerprint",
    "generator",
    "created_at",
)

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/+@-]{0,127}$")
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_MANIFEST_FIELDS = 16
_MAX_REQUIRED_SECTIONS = 128
_MAX_FILE_ENTRIES = 4096
_MAX_FILE_ENTRY_FIELDS = 4
_MAX_PROVENANCE_FIELDS = 10
_ALLOWED_MANIFEST_FIELDS = frozenset(
    {
        "files",
        "manifest_hash",
        "policy",
        "policy_fingerprint",
        "provenance",
        "required_sections",
        "schema_version",
    }
)
_ALLOWED_FILE_ENTRY_FIELDS = frozenset({"hash", "path", "section", "sha256"})
_ALLOWED_PROVENANCE_FIELDS = frozenset(
    {
        "created_at",
        "generated_at",
        "generator",
        "generator_version",
        "input_fingerprint",
        "policy_fingerprint",
        "source_fingerprint",
        "source_hash",
        "timestamp",
        "tool",
    }
)
_ALLOWED_POLICY_FIELDS = frozenset({"fingerprint", "policy_fingerprint"})


class EvidenceFailureCategory(str, Enum):
    """Stable, privacy-safe categories returned by bundle verification."""

    MANIFEST_UNREADABLE = "manifest_unreadable"
    INVALID_MANIFEST = "invalid_manifest"
    SCHEMA_MISMATCH = "schema_mismatch"
    POLICY_MISMATCH = "policy_mismatch"
    MISSING_SECTION = "missing_section"
    INCOMPLETE_PROVENANCE = "incomplete_provenance"
    MISSING_FILE = "missing_file"
    HASH_MISMATCH = "hash_mismatch"
    UNSAFE_PATH = "unsafe_path"
    UNREADABLE_FILE = "unreadable_file"


_CATEGORY_ORDER = tuple(category.value for category in EvidenceFailureCategory)

EvidenceBundleInput: TypeAlias = Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class EvidenceBundleCheck:
    """Aggregate result of a local evidence-bundle integrity check.

    ``failures`` contains unique category names in a stable order.  The
    report intentionally contains no manifest paths, provenance values, or
    file contents.  Use :meth:`to_dict` when serializing it to logs or an
    audit artifact.
    """

    passed: bool
    failures: tuple[str, ...] = ()
    checked_file_count: int = 0
    failure_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        failures = self.failures
        if type(failures) is not tuple or any(
            type(category) is not str for category in failures
        ):
            raise ValueError("evidence bundle check fields are invalid")
        expected_order = tuple(
            category for category in _CATEGORY_ORDER if category in failures
        )
        if (
            type(self.passed) is not bool
            or type(self.checked_file_count) is not int
            or self.checked_file_count < 0
            or len(set(failures)) != len(failures)
            or failures != expected_order
            or self.passed != (not failures)
        ):
            raise ValueError("evidence bundle check fields are invalid")

        if type(self.failure_counts) is not tuple or len(self.failure_counts) != len(
            failures
        ):
            raise ValueError("evidence bundle check fields are invalid")
        for entry, category in zip(self.failure_counts, failures):
            if (
                type(entry) is not tuple
                or len(entry) != 2
                or entry[0] != category
                or type(entry[1]) is not int
                or entry[1] <= 0
            ):
                raise ValueError("evidence bundle check fields are invalid")

    @property
    def valid(self) -> bool:
        """Return whether the bundle passed every requested check."""

        return self.passed

    @property
    def ok(self) -> bool:
        """Return the result under the conventional short property name."""

        return self.passed

    @property
    def failure_categories(self) -> tuple[str, ...]:
        """Return the stable failure-category names."""

        return self.failures

    @property
    def categories(self) -> tuple[str, ...]:
        """Return :attr:`failure_categories` as a concise alias."""

        return self.failures

    def to_dict(self) -> dict[str, Any]:
        """Return an aggregate, path-free JSON-serializable report."""

        return {
            "checked_file_count": self.checked_file_count,
            "failure_categories": list(self.failures),
            "failure_counts": dict(self.failure_counts),
            "passed": self.passed,
        }

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        if self.passed:
            return "Evidence bundle check passed"
        return "Evidence bundle check failed: " + ", ".join(self.failures)


# These aliases keep the result discoverable for callers that use report or
# result terminology while preserving one canonical public implementation.
EvidenceBundleReport = EvidenceBundleCheck
EvidenceBundleResult = EvidenceBundleCheck


@dataclass(frozen=True)
class _FileEntry:
    path: str
    digest: str
    section: str


class _InvalidManifest(Exception):
    """Internal marker; its message must never reach a caller."""


def _check_evidence_bundle(
    bundle: EvidenceBundleInput,
    *,
    root: str | Path | None = None,
    expected_policy_fingerprint: str | None = None,
    required_sections: Iterable[str] | None = None,
    expected_schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION,
    manifest_name: str = MANIFEST_FILENAME,
) -> EvidenceBundleCheck:
    """Check a local evidence bundle and return privacy-safe failure categories.

    Args:
        bundle: A bundle directory, a manifest JSON path, or an in-memory
            manifest mapping.  In-memory manifests require ``root`` so file
            references are resolved against an explicit local directory.
        root: Directory containing evidence files when ``bundle`` is a
            manifest mapping or an explicitly separate manifest path.
        expected_policy_fingerprint: Optional caller-supplied policy digest.
            The manifest must always contain a canonical digest; when this
            argument is supplied it must match it.
        required_sections: Optional allow-list of section names to require.
            When omitted, the manifest's ``required_sections`` field is used,
            falling back to :data:`DEFAULT_REQUIRED_SECTIONS`.
        expected_schema_version: Schema version accepted by this verifier.
        manifest_name: Manifest filename used when ``bundle`` is a directory.

    Returns:
        An aggregate :class:`EvidenceBundleCheck`.  Failure categories and
        counts are deterministic and never include protected values.

    The function performs filesystem reads only.  It does not resolve URLs,
    contact services, or require network access.
    """

    counts: dict[str, int] = {}

    def add(category: EvidenceFailureCategory, amount: int = 1) -> None:
        name = category.value
        counts[name] = counts.get(name, 0) + amount

    manifest, bundle_root = _load_manifest(
        bundle,
        root=root,
        manifest_name=manifest_name,
        on_failure=lambda: add(EvidenceFailureCategory.MANIFEST_UNREADABLE),
    )
    if manifest is None or bundle_root is None:
        return _build_result(counts)

    if not _mapping_has_only_keys(
        manifest,
        allowed=_ALLOWED_MANIFEST_FIELDS,
        limit=_MAX_MANIFEST_FIELDS,
    ):
        add(EvidenceFailureCategory.INVALID_MANIFEST)
        return _build_result(counts)

    if not isinstance(expected_schema_version, str):
        add(EvidenceFailureCategory.SCHEMA_MISMATCH)
    elif manifest.get("schema_version") != expected_schema_version:
        add(EvidenceFailureCategory.SCHEMA_MISMATCH)

    _check_manifest_hash(manifest, add)
    _check_policy_fingerprint(manifest, expected_policy_fingerprint, add)

    try:
        sections = _normalise_required_sections(
            required_sections
            if required_sections is not None
            else manifest.get("required_sections", DEFAULT_REQUIRED_SECTIONS)
        )
        entries = _normalise_file_entries(manifest.get("files"))
    except _InvalidManifest:
        add(EvidenceFailureCategory.INVALID_MANIFEST)
        return _build_result(counts)

    if not _provenance_is_complete(manifest.get("provenance")):
        add(EvidenceFailureCategory.INCOMPLETE_PROVENANCE)

    declared_sections = {entry.section for entry in entries}
    missing_sections = set(sections) - declared_sections
    if missing_sections:
        add(EvidenceFailureCategory.MISSING_SECTION, len(missing_sections))

    checked_file_count = 0
    for entry in entries:
        try:
            file_path = _resolve_evidence_path(bundle_root, entry.path)
        except (OSError, ValueError):
            add(EvidenceFailureCategory.UNSAFE_PATH)
            continue

        if not file_path.exists():
            add(EvidenceFailureCategory.MISSING_FILE)
            continue
        if not file_path.is_file():
            add(EvidenceFailureCategory.UNREADABLE_FILE)
            continue
        try:
            actual_digest = _hash_file(file_path)
        except OSError:
            add(EvidenceFailureCategory.UNREADABLE_FILE)
            continue
        checked_file_count += 1
        if not hmac.compare_digest(actual_digest, entry.digest):
            add(EvidenceFailureCategory.HASH_MISMATCH)

    return _build_result(counts, checked_file_count=checked_file_count)


def check_evidence_bundle(
    bundle: EvidenceBundleInput,
    *,
    root: str | Path | None = None,
    expected_policy_fingerprint: str | None = None,
    required_sections: Iterable[str] | None = None,
    expected_schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION,
    manifest_name: str = MANIFEST_FILENAME,
) -> EvidenceBundleCheck:
    """Check a local evidence bundle without exposing invalid input values."""

    try:
        return _check_evidence_bundle(
            bundle,
            root=root,
            expected_policy_fingerprint=expected_policy_fingerprint,
            required_sections=required_sections,
            expected_schema_version=expected_schema_version,
            manifest_name=manifest_name,
        )
    except MemoryError:
        raise
    except Exception:
        return _build_result({EvidenceFailureCategory.INVALID_MANIFEST.value: 1})


def verify_evidence_bundle(
    bundle: EvidenceBundleInput,
    *,
    root: str | Path | None = None,
    expected_policy_fingerprint: str | None = None,
    required_sections: Iterable[str] | None = None,
    expected_schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION,
    manifest_name: str = MANIFEST_FILENAME,
) -> EvidenceBundleCheck:
    """Alias for :func:`check_evidence_bundle` using verification terminology."""

    return check_evidence_bundle(
        bundle,
        root=root,
        expected_policy_fingerprint=expected_policy_fingerprint,
        required_sections=required_sections,
        expected_schema_version=expected_schema_version,
        manifest_name=manifest_name,
    )


def _load_manifest(
    bundle: EvidenceBundleInput,
    *,
    root: str | Path | None,
    manifest_name: str,
    on_failure: Any,
) -> tuple[Mapping[str, Any] | None, Path | None]:
    if isinstance(bundle, Mapping):
        if root is None:
            on_failure()
            return None, None
        try:
            bundle_root = Path(root).resolve()
        except (OSError, TypeError, ValueError):
            on_failure()
            return None, None
        try:
            if not bundle_root.is_dir():
                on_failure()
                return None, None
        except OSError:
            on_failure()
            return None, None
        return bundle, bundle_root

    try:
        source = Path(bundle)
    except (TypeError, ValueError):
        on_failure()
        return None, None

    try:
        if source.is_dir():
            bundle_root = source.resolve()
            manifest_path = bundle_root / _manifest_name(manifest_name)
        else:
            manifest_path = source.resolve()
            bundle_root = (
                Path(root).resolve() if root is not None else manifest_path.parent
            )
        if not bundle_root.is_dir() or not manifest_path.is_file():
            on_failure()
            return None, None
        with manifest_path.open("rb") as handle:
            encoded = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(encoded) > _MAX_MANIFEST_BYTES:
            raise _InvalidManifest
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
        _InvalidManifest,
    ):
        on_failure()
        return None, None

    if not isinstance(payload, Mapping):
        on_failure()
        return None, None
    return payload, bundle_root


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _InvalidManifest
        result[key] = value
    return result


def _manifest_name(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _InvalidManifest
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value:
        raise _InvalidManifest
    if path.name in {".", ".."} or "\x00" in path.name:
        raise _InvalidManifest
    return value


def _bounded_tuple(value: Iterable[Any], *, limit: int) -> tuple[Any, ...]:
    try:
        iterator = iter(value)
        items: list[Any] = []
        for _ in range(limit + 1):
            try:
                items.append(next(iterator))
            except StopIteration:
                break
    except MemoryError:
        raise
    except Exception:
        raise _InvalidManifest from None
    if len(items) > limit:
        raise _InvalidManifest
    return tuple(items)


def _mapping_keys(value: Mapping[Any, Any], *, limit: int) -> tuple[str, ...]:
    keys = _bounded_tuple(value.keys(), limit=limit)
    if any(type(key) is not str for key in keys):
        raise _InvalidManifest
    return keys


def _mapping_has_only_keys(
    value: Mapping[Any, Any],
    *,
    allowed: frozenset[str],
    limit: int,
) -> bool:
    try:
        return set(_mapping_keys(value, limit=limit)) <= allowed
    except _InvalidManifest:
        return False


def _check_manifest_hash(
    manifest: Mapping[str, Any],
    add: Any,
) -> None:
    if "manifest_hash" not in manifest:
        return
    supplied = manifest.get("manifest_hash")
    if not _is_digest(supplied):
        add(EvidenceFailureCategory.INVALID_MANIFEST)
        return
    supplied_digest = cast(str, supplied)
    payload = {
        key: manifest[key]
        for key in _mapping_keys(manifest, limit=_MAX_MANIFEST_FIELDS)
    }
    payload.pop("manifest_hash", None)
    try:
        expected = _hash_json(payload)
    except (TypeError, ValueError):
        add(EvidenceFailureCategory.INVALID_MANIFEST)
        return
    if not hmac.compare_digest(supplied_digest, expected):
        add(EvidenceFailureCategory.HASH_MISMATCH)


def _check_policy_fingerprint(
    manifest: Mapping[str, Any],
    expected_policy_fingerprint: str | None,
    add: Any,
) -> None:
    has_direct_policy = "policy_fingerprint" in manifest
    has_nested_policy = "policy" in manifest
    if has_direct_policy and has_nested_policy:
        add(EvidenceFailureCategory.INVALID_MANIFEST)
        return

    actual = manifest.get("policy_fingerprint")
    if has_nested_policy:
        policy = manifest.get("policy")
        if not isinstance(policy, Mapping) or not _mapping_has_only_keys(
            policy,
            allowed=_ALLOWED_POLICY_FIELDS,
            limit=len(_ALLOWED_POLICY_FIELDS),
        ):
            add(EvidenceFailureCategory.INVALID_MANIFEST)
            return
        policy_keys = set(_mapping_keys(policy, limit=len(_ALLOWED_POLICY_FIELDS)))
        if len(policy_keys) != 1:
            add(EvidenceFailureCategory.INVALID_MANIFEST)
            return
        actual = policy.get("fingerprint", policy.get("policy_fingerprint"))
    if not _is_digest(actual):
        add(EvidenceFailureCategory.POLICY_MISMATCH)
        return
    actual_digest = cast(str, actual)
    if expected_policy_fingerprint is not None:
        if not _is_digest(expected_policy_fingerprint) or not hmac.compare_digest(
            actual_digest,
            expected_policy_fingerprint,
        ):
            add(EvidenceFailureCategory.POLICY_MISMATCH)

    provenance = manifest.get("provenance")
    if isinstance(provenance, Mapping) and "policy_fingerprint" in provenance:
        provenance_policy = provenance.get("policy_fingerprint")
        if not _is_digest(provenance_policy) or not hmac.compare_digest(
            actual_digest,
            cast(str, provenance_policy),
        ):
            add(EvidenceFailureCategory.POLICY_MISMATCH)


def _normalise_required_sections(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise _InvalidManifest
    sections = _bounded_tuple(value, limit=_MAX_REQUIRED_SECTIONS)
    if not sections or any(not _is_identifier(item) for item in sections):
        raise _InvalidManifest
    if len(set(sections)) != len(sections):
        raise _InvalidManifest
    return tuple(sorted(sections))


def _normalise_file_entries(value: Any) -> tuple[_FileEntry, ...]:
    if isinstance(value, Mapping):
        keys = sorted(_mapping_keys(value, limit=_MAX_FILE_ENTRIES))
        entries: list[_FileEntry] = []
        for raw_path in keys:
            descriptor = value[raw_path]
            if isinstance(descriptor, str):
                descriptor = {
                    "path": raw_path,
                    "sha256": descriptor,
                    "section": _section_from_path(raw_path),
                }
            elif isinstance(descriptor, Mapping):
                descriptor_keys = _mapping_keys(
                    descriptor,
                    limit=_MAX_FILE_ENTRY_FIELDS,
                )
                descriptor = {key: descriptor[key] for key in descriptor_keys}
                if "path" in descriptor and descriptor["path"] != raw_path:
                    raise _InvalidManifest
                descriptor.setdefault("path", raw_path)
            else:
                raise _InvalidManifest
            entries.append(_file_entry_from_mapping(descriptor))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = [
            _file_entry_from_mapping(item)
            for item in _bounded_tuple(value, limit=_MAX_FILE_ENTRIES)
        ]
    else:
        raise _InvalidManifest

    if not entries or len({entry.path for entry in entries}) != len(entries):
        raise _InvalidManifest
    return tuple(sorted(entries, key=lambda entry: (entry.path, entry.section)))


def _file_entry_from_mapping(value: Any) -> _FileEntry:
    if not isinstance(value, Mapping) or not _mapping_has_only_keys(
        value,
        allowed=_ALLOWED_FILE_ENTRY_FIELDS,
        limit=_MAX_FILE_ENTRY_FIELDS,
    ):
        raise _InvalidManifest
    if "sha256" in value and "hash" in value:
        raise _InvalidManifest
    path = value.get("path")
    digest = value.get("sha256", value.get("hash"))
    section = value.get("section")
    if section is None and isinstance(path, str):
        section = _section_from_path(path)
    if (
        not isinstance(path, str)
        or not _is_digest(digest)
        or not _is_identifier(section)
    ):
        raise _InvalidManifest
    return _FileEntry(
        path=_normalise_relative_path(cast(str, path)),
        digest=cast(str, digest),
        section=cast(str, section),
    )


def _normalise_relative_path(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise _InvalidManifest
    path = PurePosixPath(value)
    if path.as_posix() != value:
        raise _InvalidManifest
    if any(part == "" for part in path.parts):
        raise _InvalidManifest
    return value


def _section_from_path(value: str) -> str:
    try:
        name = PurePosixPath(value).name
        stem = PurePosixPath(name).stem
    except (TypeError, ValueError):
        raise _InvalidManifest from None
    if not _is_identifier(stem):
        raise _InvalidManifest
    return stem


def _provenance_is_complete(value: Any) -> bool:
    if not isinstance(value, Mapping) or not _mapping_has_only_keys(
        value,
        allowed=_ALLOWED_PROVENANCE_FIELDS,
        limit=_MAX_PROVENANCE_FIELDS,
    ):
        return False
    keys = set(_mapping_keys(value, limit=_MAX_PROVENANCE_FIELDS))
    source_keys = keys & {
        "source_fingerprint",
        "source_hash",
        "input_fingerprint",
    }
    generator_keys = keys & {"generator", "tool", "generator_version"}
    timestamp_keys = keys & {"created_at", "generated_at", "timestamp"}
    if not (
        len(source_keys) == 1 and len(generator_keys) == 1 and len(timestamp_keys) == 1
    ):
        return False
    source_fingerprint = _first_present(
        value,
        ("source_fingerprint", "source_hash", "input_fingerprint"),
    )
    generator = _first_present(value, ("generator", "tool", "generator_version"))
    created_at = _first_present(value, ("created_at", "generated_at", "timestamp"))
    return (
        _is_digest(source_fingerprint)
        and _is_identifier(generator)
        and isinstance(created_at, str)
        and bool(_TIMESTAMP_RE.fullmatch(created_at))
    )


def _first_present(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _is_identifier(value: Any) -> bool:
    return type(value) is str and bool(_IDENTIFIER_RE.fullmatch(value))


def _is_digest(value: Any) -> bool:
    return type(value) is str and bool(_DIGEST_RE.fullmatch(value))


def _resolve_evidence_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    current = root
    for part in PurePosixPath(relative_path).parts:
        current /= part
        if current.is_symlink():
            raise ValueError
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError from None
    return candidate


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"sha256:{digest}"


def _build_result(
    counts: Mapping[str, int],
    *,
    checked_file_count: int = 0,
) -> EvidenceBundleCheck:
    failures = tuple(
        category for category in _CATEGORY_ORDER if counts.get(category, 0) > 0
    )
    failure_counts = tuple((category, counts[category]) for category in failures)
    return EvidenceBundleCheck(
        passed=not failures,
        failures=failures,
        checked_file_count=checked_file_count,
        failure_counts=failure_counts,
    )


__all__ = [
    "DEFAULT_REQUIRED_SECTIONS",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "EvidenceBundleCheck",
    "EvidenceFailureCategory",
    "MANIFEST_FILENAME",
    "REQUIRED_PROVENANCE_FIELDS",
    "check_evidence_bundle",
    "verify_evidence_bundle",
]
