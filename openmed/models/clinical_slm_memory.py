"""Deterministic memory admission checks for local clinical SLM loading.

This module is the resource gate between a locally prepared clinical small
language-model artifact and a model loader.  It reads bounded artifact
metadata, combines it with an explicit runtime memory profile, and returns a
value-free aggregate report before a tokenizer, runtime, or weight tensor is
constructed.

The calculation is intentionally conservative and explainable:

* weights are the declared weight bytes in the local artifact;
* cache is ``context_tokens * batch_size * cache_bytes_per_token``;
* context is ``context_tokens * batch_size * context_bytes_per_token``;
* batch is ``batch_size * batch_bytes``; and
* the configured runtime overhead is added to the load total.

No network-capable dependency is imported, no model bytes are read, and no
caller-provided model id, path, prompt, or metadata value is copied into an
exception, log, report, or fingerprint.  A report can therefore be persisted
as local operational evidence without becoming a clinical-data artifact.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Final, NoReturn

MEMORY_SCHEMA_VERSION: Final = "openmed.clinical-slm-memory.v1"
CLINICAL_SLM_MEMORY_SCHEMA_VERSION: Final = MEMORY_SCHEMA_VERSION
SCHEMA_VERSION: Final = MEMORY_SCHEMA_VERSION

MAX_MEMORY_BYTES: Final = (1 << 63) - 1
MAX_CONTEXT_TOKENS: Final = (1 << 31) - 1
MAX_BATCH_SIZE: Final = 1 << 20
MAX_COMPONENTS: Final = 4_096
MAX_PROFILE_NAME_LENGTH: Final = 64
MAX_METADATA_BYTES: Final = 16 * 1024 * 1024

# These are explicit engineering defaults for a profile constructed by the
# class.  A caller can override every value; they are not device benchmarks or
# a promise that a particular runtime will remain within the estimate.
DEFAULT_CONTEXT_TOKENS: Final = 2_048
DEFAULT_BATCH_SIZE: Final = 1
DEFAULT_CACHE_BYTES_PER_TOKEN: Final = 65_536
DEFAULT_CONTEXT_BYTES_PER_TOKEN: Final = 16_384
DEFAULT_BATCH_BYTES: Final = 4 * 1024 * 1024

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROFILE_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_ROLE_RE = re.compile(r"[a-z0-9]+")

_ERROR_MESSAGES: Final = {
    "invalid_input": "clinical SLM memory input is invalid",
    "artifact_invalid": "clinical SLM artifact metadata is invalid",
    "artifact_unreadable": "clinical SLM artifact metadata cannot be read",
    "artifact_missing": "clinical SLM artifact metadata is missing",
    "duplicate_field": "clinical SLM memory metadata contains duplicate fields",
    "weights_missing": "clinical SLM artifact weight size is missing",
    "weights_invalid": "clinical SLM artifact weight size is invalid",
    "profile_invalid": "clinical SLM runtime profile is invalid",
    "profile_ambiguous": "clinical SLM runtime profile has conflicting aliases",
    "memory_budget_missing": "clinical SLM runtime memory budget is missing",
    "overflow": "clinical SLM memory estimate exceeds its safety bound",
}

_REASON_ORDER: Final = (
    "memory_budget_exceeded",
    "headroom_insufficient",
)
_REASON_INDEX: Final = {reason: index for index, reason in enumerate(_REASON_ORDER)}

_WEIGHT_SIZE_KEYS: Final = (
    "weights_bytes",
    "weight_bytes",
    "weights_size_bytes",
    "weight_size_bytes",
    "model_bytes",
    "model_size_bytes",
    "parameter_bytes",
    "parameters_bytes",
)
_SIZE_KEYS: Final = (
    "size_bytes",
    "size",
    "bytes",
    "byte_size",
)
_COMPONENT_COLLECTION_KEYS: Final = (
    "components",
    "artifacts",
    "files",
)
_COMPONENT_ROLE_KEYS: Final = (
    "component",
    "kind",
    "role",
    "type",
    "artifact_type",
    "name",
    "path",
)
_ARTIFACT_TOTAL_KEYS: Final = (
    "artifact_bytes",
    "artifact_size_bytes",
    "total_bytes",
    "total_size_bytes",
)


class ClinicalSLMMemoryError(ValueError):
    """Raised for invalid memory-preflight inputs.

    Only a stable reason code and static message are retained.  In
    particular, a path, model identifier, digest, mapping value, or exception
    detail supplied by a caller is never appended to the error.
    """

    def __init__(self, reason_code: str) -> None:
        self.reason_code = (
            reason_code if reason_code in _ERROR_MESSAGES else "invalid_input"
        )
        self.code = self.reason_code
        super().__init__(_ERROR_MESSAGES[self.reason_code])

    def to_dict(self) -> dict[str, str]:
        """Return the machine-readable, value-free error representation."""

        return {"code": self.reason_code, "message": str(self)}


class ClinicalSLMMemoryValidationError(ClinicalSLMMemoryError):
    """Compatibility subtype for callers that distinguish validation errors."""


class MemoryPreflightStatus(str, Enum):
    """Terminal decision of a clinical SLM memory preflight."""

    ACCEPT = "accept"
    REJECT = "reject"


def _fail(reason_code: str) -> NoReturn:
    """Raise a stable memory error without retaining caller values."""

    raise ClinicalSLMMemoryValidationError(reason_code) from None


def _canonical_json(value: Any) -> str:
    """Encode safe, internally generated metadata deterministically."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError, RecursionError):
        _fail("artifact_invalid")
    raise AssertionError("unreachable")


def _bounded_int(
    value: Any,
    *,
    maximum: int,
    positive: bool,
    reason_code: str,
) -> int:
    """Validate one integer without echoing its value in an exception."""

    lower_bound = 1 if positive else 0
    if type(value) is not int or not lower_bound <= value <= maximum:
        _fail(reason_code)
    return value


def _safe_mapping_copy(value: Any, *, reason_code: str) -> dict[str, Any]:
    """Copy a mapping while converting hostile mapping failures to a static error."""

    if not isinstance(value, Mapping) or isinstance(value, (str, bytes, bytearray)):
        _fail(reason_code)
    try:
        copied = dict(value)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail(reason_code)
    if any(type(key) is not str for key in copied):
        _fail(reason_code)
    return copied


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON fields before any metadata is interpreted."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_field")
        result[key] = value
    return result


def _parse_json(payload: str | bytes | bytearray) -> dict[str, Any]:
    """Parse one bounded metadata document into a safe mapping."""

    try:
        if len(payload) > MAX_METADATA_BYTES:
            _fail("artifact_invalid")
        decoded = json.loads(payload, object_pairs_hook=_strict_json_object)
    except ClinicalSLMMemoryError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, RecursionError):
        _fail("artifact_unreadable")
    return _safe_mapping_copy(decoded, reason_code="artifact_invalid")


def _token(value: Any) -> str:
    """Return a non-sensitive role token for internal component matching."""

    if type(value) is not str:
        return ""
    return "".join(_ROLE_RE.findall(value.lower()))


def _is_weight_role(value: Any) -> bool:
    """Return whether a component role describes model weights."""

    role = _token(value)
    return any(
        marker in role
        for marker in ("weight", "model", "parameter", "checkpoint", "tensor")
    )


def _first_present(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    *,
    conflict_reason: str = "profile_ambiguous",
) -> tuple[bool, Any]:
    """Return one alias value, rejecting conflicting aliases."""

    present = tuple(name for name in names if name in mapping)
    if len(present) > 1:
        _fail(conflict_reason)
    if present:
        return True, mapping[present[0]]
    return False, None


def _size_value(value: Any, *, weight: bool) -> int:
    """Read one bounded size field from a mapping or integer."""

    reason = "weights_invalid" if weight else "artifact_invalid"
    if type(value) is int:
        return _bounded_int(
            value,
            maximum=MAX_MEMORY_BYTES,
            positive=True,
            reason_code=reason,
        )
    if not isinstance(value, Mapping):
        _fail(reason)
    copied = _safe_mapping_copy(value, reason_code=reason)
    keys = _WEIGHT_SIZE_KEYS if weight else _SIZE_KEYS
    present, raw = _first_present(
        copied,
        keys,
        conflict_reason=reason,
    )
    if not present:
        _fail("weights_missing" if weight else "artifact_invalid")
    return _bounded_int(
        raw,
        maximum=MAX_MEMORY_BYTES,
        positive=True,
        reason_code=reason,
    )


def _iter_values(value: Any, *, reason_code: str) -> tuple[Any, ...]:
    """Normalize a component collection without exposing its contents."""

    if isinstance(value, Mapping):
        copied = _safe_mapping_copy(value, reason_code=reason_code)
        # A mapping may be one component record or a role-to-components map.
        if any(key in copied for key in (*_SIZE_KEYS, *_WEIGHT_SIZE_KEYS)):
            return (copied,)
        return tuple(copied.items())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        try:
            return tuple(value)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            _fail(reason_code)
    if type(value) is int:
        return (value,)
    _fail(reason_code)
    raise AssertionError("unreachable")


def _component_size_records(
    value: Any,
    *,
    role_hint: Any = None,
) -> tuple[tuple[bool, int], ...]:
    """Collect ``(is_weight, size)`` records from component-shaped metadata."""

    records: list[tuple[bool, int]] = []
    for item in _iter_values(value, reason_code="artifact_invalid"):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            role, members = item
            records.extend(_component_size_records(members, role_hint=role))
            continue
        if type(item) is int:
            records.append((_is_weight_role(role_hint), _size_value(item, weight=True)))
            continue
        if not isinstance(item, Mapping):
            _fail("artifact_invalid")
        record = _safe_mapping_copy(item, reason_code="artifact_invalid")
        role = role_hint
        for key in _COMPONENT_ROLE_KEYS:
            if key in record:
                role = record[key]
                break
        present, raw_size = _first_present(
            record,
            _SIZE_KEYS,
            conflict_reason="artifact_invalid",
        )
        if not present:
            present, raw_size = _first_present(
                record,
                _WEIGHT_SIZE_KEYS,
                conflict_reason="artifact_invalid",
            )
        if not present:
            # Nested ``weights: {size_bytes: ...}`` is common in compact
            # manifests; recurse without retaining the key or value.
            if "weights" in record:
                records.extend(
                    _component_size_records(record["weights"], role_hint="weights")
                )
                continue
            _fail("artifact_invalid")
        records.append(
            (
                _is_weight_role(role),
                _bounded_int(
                    raw_size,
                    maximum=MAX_MEMORY_BYTES,
                    positive=True,
                    reason_code="artifact_invalid",
                ),
            )
        )
        if len(records) > MAX_COMPONENTS:
            _fail("artifact_invalid")
    return tuple(records)


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMArtifactMemory:
    """Bounded artifact metadata used by the memory estimator.

    ``weights_bytes`` is the only artifact quantity needed for the estimate.
    ``artifact_bytes`` and ``component_count`` are aggregate provenance fields
    and never contain a path, model identifier, or artifact contents.
    """

    weights_bytes: int
    artifact_bytes: int | None = None
    component_count: int = 1
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        _bounded_int(
            self.weights_bytes,
            maximum=MAX_MEMORY_BYTES,
            positive=True,
            reason_code="weights_invalid",
        )
        if self.artifact_bytes is not None:
            _bounded_int(
                self.artifact_bytes,
                maximum=MAX_MEMORY_BYTES,
                positive=True,
                reason_code="artifact_invalid",
            )
            if self.artifact_bytes < self.weights_bytes:
                _fail("artifact_invalid")
        _bounded_int(
            self.component_count,
            maximum=MAX_COMPONENTS,
            positive=True,
            reason_code="artifact_invalid",
        )
        if self.fingerprint is None:
            fingerprint = _artifact_fingerprint(
                self.weights_bytes,
                self.artifact_bytes,
                self.component_count,
            )
        elif type(self.fingerprint) is str and _SHA256_RE.fullmatch(self.fingerprint):
            fingerprint = self.fingerprint
        else:
            _fail("artifact_invalid")
        object.__setattr__(self, "fingerprint", fingerprint)

    @property
    def artifact_fingerprint(self) -> str:
        """Return the deterministic aggregate artifact fingerprint."""

        return self.fingerprint  # type: ignore[return-value]

    @property
    def weight_bytes(self) -> int:
        """Compatibility alias for :attr:`weights_bytes`."""

        return self.weights_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate-only metadata suitable for a report."""

        return {
            "weights_bytes": self.weights_bytes,
            "artifact_bytes": self.artifact_bytes,
            "component_count": self.component_count,
            "fingerprint": self.fingerprint,
        }

    def __repr__(self) -> str:
        """Return a content-free diagnostic representation."""

        return "ClinicalSLMArtifactMemory(<aggregate>)"


ArtifactMemory = ClinicalSLMArtifactMemory
ClinicalSLMArtifactMetadata = ClinicalSLMArtifactMemory


def _artifact_fingerprint(
    weights_bytes: int,
    artifact_bytes: int | None,
    component_count: int,
) -> str:
    """Hash only normalized aggregate artifact metadata."""

    payload = _canonical_json(
        {
            "artifact_bytes": artifact_bytes,
            "component_count": component_count,
            "weights_bytes": weights_bytes,
        }
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _normalize_artifact_mapping(
    payload: Mapping[str, Any],
) -> ClinicalSLMArtifactMemory:
    """Normalize a model manifest or compact artifact metadata mapping."""

    fields = _safe_mapping_copy(payload, reason_code="artifact_invalid")
    component_count = 1

    direct_weights, raw_weights = _first_present(
        fields,
        _WEIGHT_SIZE_KEYS,
        conflict_reason="artifact_invalid",
    )
    if direct_weights:
        weights = _size_value(raw_weights, weight=True)
    else:
        weights = 0
        if "weights" in fields:
            weight_values = _component_size_records(
                fields["weights"], role_hint="weights"
            )
            weights = sum(size for _is_weight, size in weight_values)
        if not weights:
            weight_records: list[tuple[bool, int]] = []
            for key in _COMPONENT_COLLECTION_KEYS:
                if key in fields:
                    weight_records.extend(_component_size_records(fields[key]))
                    break
            weights = sum(size for is_weight, size in weight_records if is_weight)
            component_count = len(weight_records)
        else:
            component_count = 1
        if not weights:
            fallback_present, fallback_value = _first_present(
                fields,
                _ARTIFACT_TOTAL_KEYS + _SIZE_KEYS,
                conflict_reason="artifact_invalid",
            )
            if fallback_present:
                weights = _size_value(fallback_value, weight=True)
                component_count = max(component_count, 1)
    if not weights:
        _fail("weights_missing")

    artifact_bytes: int | None = None
    total_present, total_value = _first_present(
        fields,
        _ARTIFACT_TOTAL_KEYS,
        conflict_reason="artifact_invalid",
    )
    if total_present:
        artifact_bytes = _size_value(total_value, weight=False)
    if artifact_bytes is None and not direct_weights:
        artifact_records: tuple[tuple[bool, int], ...] = ()
        for key in _COMPONENT_COLLECTION_KEYS:
            if key in fields:
                artifact_records = _component_size_records(fields[key])
                break
        if artifact_records:
            artifact_bytes = sum(size for _is_weight, size in artifact_records)
            component_count = len(artifact_records)
    return ClinicalSLMArtifactMemory(
        weights_bytes=weights,
        artifact_bytes=artifact_bytes,
        component_count=component_count,
    )


def load_clinical_slm_artifact_memory(
    source: Mapping[str, Any]
    | ClinicalSLMArtifactMemory
    | str
    | Path
    | bytes
    | bytearray,
) -> ClinicalSLMArtifactMemory:
    """Load bounded local artifact metadata without reading model weights.

    Mapping and JSON inputs may use either a compact ``weights_bytes`` field or
    the component/artifact records used by the local clinical SLM manifest.
    A directory reads only one of the fixed metadata filenames below.  A
    non-JSON regular file is treated as one local weight artifact and only its
    size is inspected; its bytes are never opened.
    """

    if isinstance(source, ClinicalSLMArtifactMemory):
        return source
    if isinstance(source, Mapping):
        return _normalize_artifact_mapping(source)
    if isinstance(source, (bytes, bytearray)):
        return _normalize_artifact_mapping(_parse_json(source))
    if isinstance(source, str) and source.lstrip().startswith(("{", "[")):
        return _normalize_artifact_mapping(_parse_json(source))

    try:
        path = Path(source)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail("artifact_unreadable")

    try:
        if path.is_symlink():
            _fail("artifact_unreadable")
        if path.is_dir():
            candidates = (
                "clinical-slm-manifest.json",
                "clinical-slm-memory.json",
                "manifest.json",
            )
            metadata_path = next(
                (
                    path / name
                    for name in candidates
                    if (path / name).is_file() and not (path / name).is_symlink()
                ),
                None,
            )
            if metadata_path is None:
                _fail("artifact_missing")
            path = metadata_path
        if not path.is_file():
            _fail("artifact_missing")
        if path.suffix.lower() == ".json" or path.name in {
            "clinical-slm-manifest.json",
            "clinical-slm-memory.json",
            "manifest.json",
        }:
            if path.stat().st_size > MAX_METADATA_BYTES:
                _fail("artifact_invalid")
            return _normalize_artifact_mapping(_parse_json(path.read_bytes()))
        size = path.stat().st_size
    except ClinicalSLMMemoryError:
        raise
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        _fail("artifact_unreadable")
    return ClinicalSLMArtifactMemory(
        weights_bytes=_bounded_int(
            size,
            maximum=MAX_MEMORY_BYTES,
            positive=True,
            reason_code="weights_invalid",
        )
    )


load_artifact_memory = load_clinical_slm_artifact_memory


@dataclass(frozen=True, slots=True, init=False, repr=False)
class ClinicalSLMRuntimeProfile:
    """Explicit bounded runtime assumptions for a memory preflight.

    ``memory_budget_bytes`` is the local memory capacity available to the
    process before loading.  ``resident_memory_bytes`` is already occupied by
    the host/runtime.  The estimate must fit while leaving
    ``headroom_bytes`` free.  The three per-unit coefficients make the
    calculation deterministic and avoid a runtime import or live memory
    probe.

    The constructor accepts descriptive aliases such as
    ``available_memory_bytes``, ``required_headroom_bytes``,
    ``kv_cache_bytes_per_token``, and ``batch_overhead_bytes`` for embedding
    integrations.  Conflicting aliases are rejected without echoing values.
    """

    memory_budget_bytes: int
    headroom_bytes: int
    context_tokens: int
    batch_size: int
    cache_bytes_per_token: int
    context_bytes_per_token: int
    batch_bytes: int
    runtime_overhead_bytes: int
    resident_memory_bytes: int
    name: str
    version: str

    def __init__(
        self,
        memory_budget_bytes: int | None = None,
        headroom_bytes: int = 0,
        context_tokens: int = DEFAULT_CONTEXT_TOKENS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_bytes_per_token: int = DEFAULT_CACHE_BYTES_PER_TOKEN,
        context_bytes_per_token: int = DEFAULT_CONTEXT_BYTES_PER_TOKEN,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        runtime_overhead_bytes: int = 0,
        resident_memory_bytes: int = 0,
        name: str = "default",
        version: str = "1.0",
        *,
        available_memory_bytes: int | None = None,
        required_headroom_bytes: int | None = None,
        max_context_tokens: int | None = None,
        kv_cache_bytes_per_token: int | None = None,
        bytes_per_cache_token: int | None = None,
        activation_bytes_per_token: int | None = None,
        workspace_bytes_per_token: int | None = None,
        batch_bytes_per_item: int | None = None,
        batch_overhead_bytes: int | None = None,
        baseline_memory_bytes: int | None = None,
        currently_used_bytes: int | None = None,
        runtime_overhead: int | None = None,
    ) -> None:
        """Construct and validate a profile using canonical or alias fields."""

        if memory_budget_bytes is not None and available_memory_bytes is not None:
            _fail("profile_ambiguous")
        if required_headroom_bytes is not None:
            if headroom_bytes != 0:
                _fail("profile_ambiguous")
            headroom_bytes = required_headroom_bytes
        if max_context_tokens is not None:
            if context_tokens != DEFAULT_CONTEXT_TOKENS:
                _fail("profile_ambiguous")
            context_tokens = max_context_tokens
        cache_aliases = tuple(
            value
            for value in (kv_cache_bytes_per_token, bytes_per_cache_token)
            if value is not None
        )
        if len(cache_aliases) > 1:
            _fail("profile_ambiguous")
        if cache_aliases:
            if cache_bytes_per_token != DEFAULT_CACHE_BYTES_PER_TOKEN:
                _fail("profile_ambiguous")
            cache_bytes_per_token = cache_aliases[0]
        context_aliases = tuple(
            value
            for value in (activation_bytes_per_token, workspace_bytes_per_token)
            if value is not None
        )
        if len(context_aliases) > 1:
            _fail("profile_ambiguous")
        if context_aliases:
            if context_bytes_per_token != DEFAULT_CONTEXT_BYTES_PER_TOKEN:
                _fail("profile_ambiguous")
            context_bytes_per_token = context_aliases[0]
        batch_aliases = tuple(
            value
            for value in (batch_bytes_per_item, batch_overhead_bytes)
            if value is not None
        )
        if len(batch_aliases) > 1:
            _fail("profile_ambiguous")
        if batch_aliases:
            if batch_bytes != DEFAULT_BATCH_BYTES:
                _fail("profile_ambiguous")
            batch_bytes = batch_aliases[0]
        resident_aliases = tuple(
            value
            for value in (baseline_memory_bytes, currently_used_bytes)
            if value is not None
        )
        if len(resident_aliases) > 1:
            _fail("profile_ambiguous")
        if resident_aliases:
            if resident_memory_bytes != 0:
                _fail("profile_ambiguous")
            resident_memory_bytes = resident_aliases[0]
        if runtime_overhead is not None:
            if runtime_overhead_bytes != 0:
                _fail("profile_ambiguous")
            runtime_overhead_bytes = runtime_overhead
        if memory_budget_bytes is None:
            memory_budget_bytes = available_memory_bytes
        if memory_budget_bytes is None:
            _fail("memory_budget_missing")
        object.__setattr__(self, "memory_budget_bytes", memory_budget_bytes)
        object.__setattr__(self, "headroom_bytes", headroom_bytes)
        object.__setattr__(self, "context_tokens", context_tokens)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "cache_bytes_per_token", cache_bytes_per_token)
        object.__setattr__(self, "context_bytes_per_token", context_bytes_per_token)
        object.__setattr__(self, "batch_bytes", batch_bytes)
        object.__setattr__(self, "runtime_overhead_bytes", runtime_overhead_bytes)
        object.__setattr__(self, "resident_memory_bytes", resident_memory_bytes)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        self.__post_init__()

    def __post_init__(self) -> None:
        _bounded_int(
            self.memory_budget_bytes,
            maximum=MAX_MEMORY_BYTES,
            positive=True,
            reason_code="profile_invalid",
        )
        _bounded_int(
            self.headroom_bytes,
            maximum=MAX_MEMORY_BYTES,
            positive=False,
            reason_code="profile_invalid",
        )
        _bounded_int(
            self.context_tokens,
            maximum=MAX_CONTEXT_TOKENS,
            positive=True,
            reason_code="profile_invalid",
        )
        _bounded_int(
            self.batch_size,
            maximum=MAX_BATCH_SIZE,
            positive=True,
            reason_code="profile_invalid",
        )
        for value in (
            self.cache_bytes_per_token,
            self.context_bytes_per_token,
        ):
            _bounded_int(
                value,
                maximum=MAX_MEMORY_BYTES,
                positive=True,
                reason_code="profile_invalid",
            )
        for value in (
            self.batch_bytes,
            self.runtime_overhead_bytes,
            self.resident_memory_bytes,
        ):
            _bounded_int(
                value,
                maximum=MAX_MEMORY_BYTES,
                positive=False,
                reason_code="profile_invalid",
            )
        if self.resident_memory_bytes > self.memory_budget_bytes:
            _fail("profile_invalid")
        if (
            type(self.name) is not str
            or not self.name.isascii()
            or len(self.name) > MAX_PROFILE_NAME_LENGTH
            or _PROFILE_NAME_RE.fullmatch(self.name) is None
        ):
            _fail("profile_invalid")
        if type(self.version) is not str or self.version != "1.0":
            _fail("profile_invalid")

    @property
    def available_memory_bytes(self) -> int:
        """Return memory available before model loading."""

        return self.memory_budget_bytes - self.resident_memory_bytes

    @property
    def required_headroom_bytes(self) -> int:
        """Return the configured free-memory requirement after loading."""

        return self.headroom_bytes

    @property
    def kv_cache_bytes_per_token(self) -> int:
        """Return the KV-cache coefficient under its runtime-facing alias."""

        return self.cache_bytes_per_token

    @property
    def activation_bytes_per_token(self) -> int:
        """Return the context-workspace coefficient."""

        return self.context_bytes_per_token

    @property
    def batch_overhead_bytes(self) -> int:
        """Return the per-item batch overhead."""

        return self.batch_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic, non-sensitive profile metadata."""

        return {
            "name": self.name,
            "version": self.version,
            "memory_budget_bytes": self.memory_budget_bytes,
            "resident_memory_bytes": self.resident_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "headroom_bytes": self.headroom_bytes,
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "cache_bytes_per_token": self.cache_bytes_per_token,
            "context_bytes_per_token": self.context_bytes_per_token,
            "batch_bytes": self.batch_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
        }

    def __repr__(self) -> str:
        """Return a bounded diagnostic representation."""

        return "ClinicalSLMRuntimeProfile(<bounded metadata>)"


RuntimeMemoryProfile = ClinicalSLMRuntimeProfile
MemoryRuntimeProfile = ClinicalSLMRuntimeProfile
ClinicalSLMMemoryProfile = ClinicalSLMRuntimeProfile
RuntimeProfile = ClinicalSLMRuntimeProfile


def _normalize_profile_mapping(
    payload: Mapping[str, Any],
) -> ClinicalSLMRuntimeProfile:
    """Normalize a mapping profile through the strict profile constructor."""

    fields = _safe_mapping_copy(payload, reason_code="profile_invalid")
    nested = fields.get("memory")
    if isinstance(nested, Mapping):
        fields = _safe_mapping_copy(nested, reason_code="profile_invalid")

    def alias(names: Sequence[str], *, default: Any = None) -> Any:
        present = tuple(name for name in names if name in fields)
        if len(present) > 1:
            _fail("profile_ambiguous")
        return fields[present[0]] if present else default

    budget = alias(
        (
            "memory_budget_bytes",
            "memory_limit_bytes",
            "available_memory_bytes",
            "device_memory_bytes",
            "device_ram_bytes",
            "total_memory_bytes",
            "max_memory_bytes",
        )
    )
    headroom = alias(
        ("headroom_bytes", "required_headroom_bytes", "minimum_headroom_bytes"),
        default=0,
    )
    context_tokens = alias(
        (
            "context_tokens",
            "max_context_tokens",
            "context_length",
            "max_sequence_length",
        ),
        default=DEFAULT_CONTEXT_TOKENS,
    )
    batch_size = alias(
        ("batch_size", "max_batch_size", "batch"), default=DEFAULT_BATCH_SIZE
    )
    cache = alias(
        ("cache_bytes_per_token", "kv_cache_bytes_per_token", "bytes_per_cache_token"),
        default=DEFAULT_CACHE_BYTES_PER_TOKEN,
    )
    context = alias(
        (
            "context_bytes_per_token",
            "activation_bytes_per_token",
            "workspace_bytes_per_token",
        ),
        default=DEFAULT_CONTEXT_BYTES_PER_TOKEN,
    )
    batch = alias(
        ("batch_bytes", "batch_bytes_per_item", "batch_overhead_bytes"),
        default=DEFAULT_BATCH_BYTES,
    )
    runtime_overhead = alias(
        ("runtime_overhead_bytes", "runtime_overhead", "loader_overhead_bytes"),
        default=0,
    )
    resident = alias(
        ("resident_memory_bytes", "baseline_memory_bytes", "currently_used_bytes"),
        default=0,
    )
    return ClinicalSLMRuntimeProfile(
        memory_budget_bytes=budget,
        headroom_bytes=headroom,
        context_tokens=context_tokens,
        batch_size=batch_size,
        cache_bytes_per_token=cache,
        context_bytes_per_token=context,
        batch_bytes=batch,
        runtime_overhead_bytes=runtime_overhead,
        resident_memory_bytes=resident,
        name=alias(("name", "profile_name"), default="default"),
        version=alias(("version", "profile_version"), default="1.0"),
    )


def normalize_clinical_slm_runtime_profile(
    profile: ClinicalSLMRuntimeProfile | Mapping[str, Any],
) -> ClinicalSLMRuntimeProfile:
    """Validate and return one immutable runtime memory profile."""

    if isinstance(profile, ClinicalSLMRuntimeProfile):
        return profile
    if isinstance(profile, Mapping):
        return _normalize_profile_mapping(profile)
    _fail("profile_invalid")
    raise AssertionError("unreachable")


normalize_runtime_profile = normalize_clinical_slm_runtime_profile


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMMemoryEstimate:
    """Aggregate memory estimate and headroom arithmetic."""

    weights_bytes: int
    cache_bytes: int
    context_bytes: int
    batch_bytes: int
    runtime_overhead_bytes: int
    total_bytes: int
    memory_budget_bytes: int
    resident_memory_bytes: int
    available_memory_bytes: int
    required_headroom_bytes: int
    remaining_headroom_bytes: int
    headroom_deficit_bytes: int
    fits: bool
    artifact_fingerprint: str
    artifact_component_count: int

    def __post_init__(self) -> None:
        for value in (
            self.weights_bytes,
            self.cache_bytes,
            self.context_bytes,
            self.batch_bytes,
            self.runtime_overhead_bytes,
            self.total_bytes,
            self.memory_budget_bytes,
            self.resident_memory_bytes,
            self.required_headroom_bytes,
            self.headroom_deficit_bytes,
            self.artifact_component_count,
        ):
            _bounded_int(
                value,
                maximum=MAX_MEMORY_BYTES,
                positive=False,
                reason_code="overflow",
            )
        if type(self.available_memory_bytes) is not int or not (
            -MAX_MEMORY_BYTES <= self.available_memory_bytes <= MAX_MEMORY_BYTES
        ):
            _fail("overflow")
        if type(self.remaining_headroom_bytes) is not int or not (
            -MAX_MEMORY_BYTES <= self.remaining_headroom_bytes <= MAX_MEMORY_BYTES
        ):
            _fail("overflow")
        if type(self.fits) is not bool or type(self.artifact_fingerprint) is not str:
            _fail("overflow")
        if _SHA256_RE.fullmatch(self.artifact_fingerprint) is None:
            _fail("overflow")
        _bounded_int(
            self.artifact_component_count,
            maximum=MAX_COMPONENTS,
            positive=True,
            reason_code="overflow",
        )

    @property
    def required_bytes(self) -> int:
        """Return the total bytes reserved by this estimate."""

        return self.total_bytes

    @property
    def headroom_met(self) -> bool:
        """Return whether the configured post-load headroom is met."""

        return self.remaining_headroom_bytes >= self.required_headroom_bytes

    @property
    def memory_budget_met(self) -> bool:
        """Return whether the load total fits the available memory."""

        return self.total_bytes <= self.available_memory_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free resource arithmetic."""

        return {
            "weights_bytes": self.weights_bytes,
            "cache_bytes": self.cache_bytes,
            "context_bytes": self.context_bytes,
            "batch_bytes": self.batch_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "total_bytes": self.total_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "resident_memory_bytes": self.resident_memory_bytes,
            "available_memory_bytes": self.available_memory_bytes,
            "required_headroom_bytes": self.required_headroom_bytes,
            "remaining_headroom_bytes": self.remaining_headroom_bytes,
            "headroom_deficit_bytes": self.headroom_deficit_bytes,
            "fits": self.fits,
            "artifact_component_count": self.artifact_component_count,
        }

    def __repr__(self) -> str:
        """Return a stable aggregate representation."""

        return "ClinicalSLMMemoryEstimate(<aggregate>)"


MemoryEstimate = ClinicalSLMMemoryEstimate


def _checked_sum(values: Sequence[int]) -> int:
    """Add bounded values while preventing an unbounded report."""

    total = 0
    for value in values:
        if type(value) is not int or value < 0 or value > MAX_MEMORY_BYTES:
            _fail("overflow")
        total += value
        if total > MAX_MEMORY_BYTES:
            _fail("overflow")
    return total


def estimate_clinical_slm_memory(
    artifact: Mapping[str, Any]
    | ClinicalSLMArtifactMemory
    | str
    | Path
    | bytes
    | bytearray,
    runtime_profile: ClinicalSLMRuntimeProfile | Mapping[str, Any],
) -> ClinicalSLMMemoryEstimate:
    """Estimate local clinical SLM load memory without loading a model.

    The function reads artifact metadata only.  It does not inspect the
    current process, import an optional backend, make a socket call, or open a
    model weight file.  A local non-JSON artifact path is accounted for using
    its filesystem size; JSON manifests are parsed as metadata.
    """

    metadata = load_clinical_slm_artifact_memory(artifact)
    profile = normalize_clinical_slm_runtime_profile(runtime_profile)
    try:
        cache_bytes = profile.context_tokens * profile.batch_size
        cache_bytes *= profile.cache_bytes_per_token
        context_bytes = profile.context_tokens * profile.batch_size
        context_bytes *= profile.context_bytes_per_token
        batch_bytes = profile.batch_size * profile.batch_bytes
    except (OverflowError, MemoryError):
        _fail("overflow")
    cache_bytes = _checked_sum((cache_bytes,))
    context_bytes = _checked_sum((context_bytes,))
    batch_bytes = _checked_sum((batch_bytes,))
    total_bytes = _checked_sum(
        (
            metadata.weights_bytes,
            cache_bytes,
            context_bytes,
            batch_bytes,
            profile.runtime_overhead_bytes,
        )
    )
    available = profile.available_memory_bytes
    remaining = available - total_bytes
    if not -MAX_MEMORY_BYTES <= remaining <= MAX_MEMORY_BYTES:
        _fail("overflow")
    deficit = max(profile.required_headroom_bytes - remaining, 0)
    fits = total_bytes <= available and remaining >= profile.required_headroom_bytes
    return ClinicalSLMMemoryEstimate(
        weights_bytes=metadata.weights_bytes,
        cache_bytes=cache_bytes,
        context_bytes=context_bytes,
        batch_bytes=batch_bytes,
        runtime_overhead_bytes=profile.runtime_overhead_bytes,
        total_bytes=total_bytes,
        memory_budget_bytes=profile.memory_budget_bytes,
        resident_memory_bytes=profile.resident_memory_bytes,
        available_memory_bytes=available,
        required_headroom_bytes=profile.required_headroom_bytes,
        remaining_headroom_bytes=remaining,
        headroom_deficit_bytes=deficit,
        fits=fits,
        artifact_fingerprint=metadata.artifact_fingerprint,
        artifact_component_count=metadata.component_count,
    )


estimate_memory = estimate_clinical_slm_memory
estimate_clinical_slm_load_memory = estimate_clinical_slm_memory


@dataclass(frozen=True, slots=True, repr=False)
class ClinicalSLMMemoryPreflightReport(Mapping[str, Any]):
    """Deterministic, value-free admission report for a local SLM load."""

    status: MemoryPreflightStatus
    estimate: ClinicalSLMMemoryEstimate
    profile: ClinicalSLMRuntimeProfile
    reason_codes: tuple[str, ...] = ()
    schema_version: str = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.status, MemoryPreflightStatus):
            _fail("invalid_input")
        if not isinstance(self.estimate, ClinicalSLMMemoryEstimate):
            _fail("invalid_input")
        if not isinstance(self.profile, ClinicalSLMRuntimeProfile):
            _fail("invalid_input")
        if self.schema_version != MEMORY_SCHEMA_VERSION:
            _fail("invalid_input")
        reasons = tuple(self.reason_codes)
        if any(
            type(reason) is not str or reason not in _REASON_INDEX for reason in reasons
        ):
            _fail("invalid_input")
        if reasons != tuple(sorted(set(reasons), key=_REASON_INDEX.__getitem__)):
            _fail("invalid_input")
        expected_reasons: list[str] = []
        if not self.estimate.memory_budget_met:
            expected_reasons.append("memory_budget_exceeded")
        if not self.estimate.headroom_met:
            expected_reasons.append("headroom_insufficient")
        if tuple(expected_reasons) != reasons:
            _fail("invalid_input")
        expected_status = (
            MemoryPreflightStatus.ACCEPT
            if not expected_reasons
            else MemoryPreflightStatus.REJECT
        )
        if self.status is not expected_status:
            _fail("invalid_input")

    @property
    def ready(self) -> bool:
        """Return whether the model load is admitted."""

        return self.status is MemoryPreflightStatus.ACCEPT

    @property
    def accepted(self) -> bool:
        """Compatibility alias for :attr:`ready`."""

        return self.ready

    @property
    def fits(self) -> bool:
        """Return whether the complete estimate satisfies the profile."""

        return self.estimate.fits

    @property
    def decision(self) -> str:
        """Return the stable human-readable decision token."""

        return "accepted" if self.ready else "rejected"

    @property
    def artifact_fingerprint(self) -> str:
        """Return the aggregate artifact fingerprint."""

        return self.estimate.artifact_fingerprint

    @property
    def resource_report(self) -> dict[str, Any]:
        """Return the aggregate report payload without arbitrary values."""

        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic resource evidence suitable for JSON storage."""

        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "decision": self.decision,
            "ready": self.ready,
            "reason_codes": list(self.reason_codes),
            "artifact": {
                "fingerprint": self.artifact_fingerprint,
                "component_count": self.estimate.artifact_component_count,
            },
            "profile": self.profile.to_dict(),
            "estimate": self.estimate.to_dict(),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize a stable report with no path or caller metadata."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        """Allow mapping-style access to the serialized report."""

        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over stable serialized report keys."""

        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of serialized report keys."""

        return len(self.to_dict())

    def __bool__(self) -> bool:
        """Use the report as its admission decision in boolean contexts."""

        return self.ready

    def __repr__(self) -> str:
        """Return a content-free diagnostic representation."""

        return "ClinicalSLMMemoryPreflightReport(<aggregate>)"


MemoryPreflightReport = ClinicalSLMMemoryPreflightReport
ClinicalSLMMemoryReport = ClinicalSLMMemoryPreflightReport


def preflight_clinical_slm_memory(
    artifact: Mapping[str, Any]
    | ClinicalSLMArtifactMemory
    | str
    | Path
    | bytes
    | bytearray,
    runtime_profile: ClinicalSLMRuntimeProfile | Mapping[str, Any] | None = None,
    *,
    profile: ClinicalSLMRuntimeProfile | Mapping[str, Any] | None = None,
    **profile_fields: Any,
) -> ClinicalSLMMemoryPreflightReport:
    """Run the local memory admission gate before model construction.

    ``runtime_profile`` and its ``profile`` alias are mutually exclusive.  If
    neither is supplied, the remaining keyword fields are used to construct a
    :class:`ClinicalSLMRuntimeProfile`; a memory budget is still required.
    Resource insufficiency returns a rejected report with stable numeric
    aggregates and no caller values.  Invalid inputs raise
    :class:`ClinicalSLMMemoryError` with a static reason.
    """

    if runtime_profile is not None and profile is not None:
        _fail("profile_ambiguous")
    selected = runtime_profile if runtime_profile is not None else profile
    if selected is None:
        selected = profile_fields
    elif profile_fields:
        _fail("profile_ambiguous")
    estimate = estimate_clinical_slm_memory(artifact, selected)
    reasons: list[str] = []
    if not estimate.memory_budget_met:
        reasons.append("memory_budget_exceeded")
    if not estimate.headroom_met:
        reasons.append("headroom_insufficient")
    normalized_profile = normalize_clinical_slm_runtime_profile(selected)
    return ClinicalSLMMemoryPreflightReport(
        status=(
            MemoryPreflightStatus.ACCEPT
            if not reasons
            else MemoryPreflightStatus.REJECT
        ),
        estimate=estimate,
        profile=normalized_profile,
        reason_codes=tuple(reasons),
    )


preflight_memory = preflight_clinical_slm_memory
preflight_clinical_slm_loading = preflight_clinical_slm_memory
check_clinical_slm_memory = preflight_clinical_slm_memory
run_clinical_slm_memory_preflight = preflight_clinical_slm_memory


def render_memory_report(
    report: ClinicalSLMMemoryPreflightReport,
    *,
    indent: int | None = None,
) -> str:
    """Render a validated memory report as deterministic JSON."""

    if not isinstance(report, ClinicalSLMMemoryPreflightReport):
        _fail("invalid_input")
    return report.to_json(indent=indent)


render_json = render_memory_report


__all__ = [
    "ArtifactMemory",
    "CLINICAL_SLM_MEMORY_SCHEMA_VERSION",
    "ClinicalSLMArtifactMemory",
    "ClinicalSLMArtifactMetadata",
    "ClinicalSLMMemoryError",
    "ClinicalSLMMemoryEstimate",
    "ClinicalSLMMemoryPreflightReport",
    "ClinicalSLMMemoryProfile",
    "ClinicalSLMMemoryReport",
    "ClinicalSLMMemoryValidationError",
    "ClinicalSLMRuntimeProfile",
    "DEFAULT_BATCH_BYTES",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_CACHE_BYTES_PER_TOKEN",
    "DEFAULT_CONTEXT_BYTES_PER_TOKEN",
    "DEFAULT_CONTEXT_TOKENS",
    "MAX_BATCH_SIZE",
    "MAX_COMPONENTS",
    "MAX_CONTEXT_TOKENS",
    "MAX_MEMORY_BYTES",
    "MemoryEstimate",
    "MemoryPreflightReport",
    "MemoryPreflightStatus",
    "MemoryRuntimeProfile",
    "MEMORY_SCHEMA_VERSION",
    "RuntimeMemoryProfile",
    "RuntimeProfile",
    "SCHEMA_VERSION",
    "check_clinical_slm_memory",
    "estimate_clinical_slm_load_memory",
    "estimate_clinical_slm_memory",
    "estimate_memory",
    "load_artifact_memory",
    "load_clinical_slm_artifact_memory",
    "normalize_clinical_slm_runtime_profile",
    "normalize_runtime_profile",
    "preflight_clinical_slm_memory",
    "preflight_clinical_slm_loading",
    "preflight_memory",
    "render_json",
    "render_memory_report",
    "run_clinical_slm_memory_preflight",
]
