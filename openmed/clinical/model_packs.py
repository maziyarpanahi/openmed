"""Versioned local model packs and fail-closed clinical task routing.

The router in this module resolves metadata only.  It never downloads an
artifact, imports an optional inference runtime, or invokes a model.  Callers
must bind every manifest alias to an existing local artifact or an explicitly
registered built-in implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

from openmed.clinical.journey_contracts import canonical_json
from openmed.structured.store import StoreResult, StoreState

MODEL_PACK_SCHEMA_VERSION = "1.0.0"
MODEL_PACK_COMPATIBILITY_POLICY = "same_major"
MODEL_PACK_ENTRY_SCHEMA_VERSION = "1.0.0"

TASK_KINDS = frozenset(
    {
        "assertion",
        "classification",
        "pair_scoring",
        "relation_extraction",
        "span_extraction",
        "temporality",
        "token_classification",
    }
)
MODEL_KINDS = frozenset({"bounded", "deterministic", "generative"})
ARTIFACT_KINDS = frozenset({"builtin", "model"})
QUANTIZATION_MODES = frozenset({"none", "fp32", "fp16", "bf16", "int8", "int4"})
CALIBRATION_METHODS = frozenset(
    {"none", "isotonic", "platt", "temperature", "threshold"}
)
PERMISSIVE_LICENSES = frozenset(
    {
        "apache-2.0",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc-by-4.0",
        "cc0-1.0",
        "isc",
        "mit",
        "unlicense",
    }
)

_CONTROLLED_RE = re.compile(r"^[a-z][a-z0-9_.:/-]{0,127}$")
_ALIAS_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PINNED_REVISION_RE = re.compile(
    r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|builtin-v[1-9][0-9]*|local-v[1-9][0-9]*)$"
)
_SCHEMA_VERSION_RE = re.compile(r"^[1-9][0-9]*\.[0-9]+\.[0-9]+$")
_LANGUAGE_RE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})*$")


class ModelPackError(ValueError):
    """Raised when a model pack cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class ModelFamilyCapability:
    """Built-in knowledge about a supported bounded model family class."""

    family: str
    tasks: tuple[str, ...]
    runtimes: tuple[str, ...]
    zero_shot_labels: bool = False

    def __post_init__(self) -> None:
        _require_controlled(self.family, "family")
        tasks = tuple(self.tasks)
        if not tasks or any(task not in TASK_KINDS for task in tasks):
            raise ModelPackError("model family contains an unsupported task")
        if len(set(tasks)) != len(tasks):
            raise ModelPackError("model family tasks must be unique")
        runtimes = tuple(self.runtimes)
        if not runtimes:
            raise ModelPackError("model family must declare a runtime class")
        for runtime in runtimes:
            _require_controlled(runtime, "model family runtime")
        if len(set(runtimes)) != len(runtimes):
            raise ModelPackError("model family runtimes must be unique")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "runtimes", runtimes)


@dataclass(frozen=True, slots=True)
class CalibrationMetadata:
    """Pinned calibration and holdout evidence for a bounded output."""

    method: str
    threshold: float | None = None
    calibration_dataset_digest: str | None = None
    holdout_dataset_digest: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.method not in CALIBRATION_METHODS:
            raise ModelPackError("unsupported calibration method")
        if self.method == "none":
            if any(
                value is not None
                for value in (
                    self.threshold,
                    self.calibration_dataset_digest,
                    self.holdout_dataset_digest,
                    self.seed,
                )
            ):
                raise ModelPackError(
                    "uncalibrated entries cannot claim calibration data"
                )
            return
        if self.threshold is None or not _finite_unit_interval(self.threshold):
            raise ModelPackError("calibration threshold must be between zero and one")
        _require_digest(self.calibration_dataset_digest, "calibration_dataset_digest")
        _require_digest(self.holdout_dataset_digest, "holdout_dataset_digest")
        if type(self.seed) is not int or self.seed < 0:
            raise ModelPackError("calibration seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready calibration metadata."""

        return {
            "calibration_dataset_digest": self.calibration_dataset_digest,
            "holdout_dataset_digest": self.holdout_dataset_digest,
            "method": self.method,
            "seed": self.seed,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class QuantizationMetadata:
    """Quantization mode and measured holdout delta."""

    mode: str
    metric: str | None = None
    observed_delta: float | None = None
    maximum_delta: float | None = None
    evaluation_digest: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in QUANTIZATION_MODES:
            raise ModelPackError("unsupported quantization mode")
        measured = (
            self.metric,
            self.observed_delta,
            self.maximum_delta,
            self.evaluation_digest,
        )
        if self.mode in {"none", "fp32"}:
            if any(value is not None for value in measured):
                raise ModelPackError(
                    "unquantized entries cannot claim a quantized delta"
                )
            return
        if (
            not isinstance(self.metric, str)
            or _CONTROLLED_RE.fullmatch(self.metric) is None
        ):
            raise ModelPackError("quantization metric must be a controlled identifier")
        if not _finite_non_negative(self.observed_delta):
            raise ModelPackError("observed quantization delta must be non-negative")
        if not _finite_non_negative(self.maximum_delta):
            raise ModelPackError("maximum quantization delta must be non-negative")
        _require_digest(self.evaluation_digest, "quantization evaluation_digest")

    @property
    def within_tolerance(self) -> bool:
        """Return whether the measured quantized delta is accepted."""

        if self.mode in {"none", "fp32"}:
            return True
        assert self.observed_delta is not None
        assert self.maximum_delta is not None
        return self.observed_delta <= self.maximum_delta

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready quantization metadata."""

        return {
            "evaluation_digest": self.evaluation_digest,
            "maximum_delta": self.maximum_delta,
            "metric": self.metric,
            "mode": self.mode,
            "observed_delta": self.observed_delta,
        }


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Explicit fallback from one alias to another manifest entry."""

    mode: Literal["none", "alias"] = "none"
    alias: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "none":
            if self.alias is not None:
                raise ModelPackError("fallback alias requires mode='alias'")
            return
        if self.mode != "alias" or self.alias is None:
            raise ModelPackError("alias fallback requires an alias")
        _require_alias(self.alias, "fallback alias")

    def to_dict(self) -> dict[str, str | None]:
        """Return deterministic JSON-ready fallback metadata."""

        return {"alias": self.alias, "mode": self.mode}


@dataclass(frozen=True, slots=True)
class ModelPackEntry:
    """One pinned local specialist or deterministic implementation."""

    alias: str
    task: str
    family: str
    artifact_id: str
    artifact_kind: str
    revision: str
    artifact_digest: str
    license: str
    runtime: str
    model_kind: str
    output_schema: str
    output_schema_version: str
    languages: tuple[str, ...]
    domains: tuple[str, ...]
    calibration: CalibrationMetadata
    quantization: QuantizationMetadata
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)
    priority: int = 100
    experimental: bool = False
    schema_version: str = MODEL_PACK_ENTRY_SCHEMA_VERSION
    compatibility_policy: str = MODEL_PACK_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _require_alias(self.alias, "alias")
        if self.task not in TASK_KINDS:
            raise ModelPackError("unsupported clinical task")
        _require_controlled(self.family, "family")
        _require_controlled(self.artifact_id, "artifact_id")
        if self.artifact_kind not in ARTIFACT_KINDS:
            raise ModelPackError("unsupported artifact kind")
        if _PINNED_REVISION_RE.fullmatch(self.revision) is None:
            raise ModelPackError("artifact revision must be immutable and pinned")
        _require_digest(self.artifact_digest, "artifact_digest")
        if not isinstance(self.license, str) or not self.license.strip():
            raise ModelPackError("license must be non-empty")
        object.__setattr__(self, "license", self.license.strip().lower())
        _require_controlled(self.runtime, "runtime")
        if self.model_kind not in MODEL_KINDS:
            raise ModelPackError("unsupported model kind")
        if self.artifact_kind == "builtin" and self.model_kind != "deterministic":
            raise ModelPackError("built-in artifacts must be deterministic")
        if self.model_kind == "generative" and not self.experimental:
            raise ModelPackError("generative entries must be explicitly experimental")
        _require_controlled(self.output_schema, "output_schema")
        _require_schema_version(self.output_schema_version, "output_schema_version")
        object.__setattr__(
            self,
            "languages",
            _string_tuple(self.languages, "languages", language=True),
        )
        object.__setattr__(self, "domains", _string_tuple(self.domains, "domains"))
        if type(self.priority) is not int or not 0 <= self.priority <= 10_000:
            raise ModelPackError("priority must be an integer from zero to 10000")
        if self.schema_version != MODEL_PACK_ENTRY_SCHEMA_VERSION:
            raise ModelPackError("unsupported model-pack entry schema version")
        if self.compatibility_policy != MODEL_PACK_COMPATIBILITY_POLICY:
            raise ModelPackError("unsupported model-pack compatibility policy")

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready entry metadata."""

        return {
            "alias": self.alias,
            "artifact_digest": self.artifact_digest,
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "calibration": self.calibration.to_dict(),
            "compatibility_policy": self.compatibility_policy,
            "domains": list(self.domains),
            "experimental": self.experimental,
            "fallback": self.fallback.to_dict(),
            "family": self.family,
            "languages": list(self.languages),
            "license": self.license,
            "model_kind": self.model_kind,
            "output_schema": self.output_schema,
            "output_schema_version": self.output_schema_version,
            "priority": self.priority,
            "quantization": self.quantization.to_dict(),
            "revision": self.revision,
            "runtime": self.runtime,
            "schema_version": self.schema_version,
            "task": self.task,
        }


@dataclass(frozen=True, slots=True)
class ModelPackManifest:
    """Immutable manifest for a set of bounded clinical specialists."""

    pack_id: str
    entries: tuple[ModelPackEntry, ...]
    schema_version: str = MODEL_PACK_SCHEMA_VERSION
    compatibility_policy: str = MODEL_PACK_COMPATIBILITY_POLICY

    def __post_init__(self) -> None:
        _require_controlled(self.pack_id, "pack_id")
        if self.schema_version != MODEL_PACK_SCHEMA_VERSION:
            raise ModelPackError("unsupported model-pack schema version")
        if self.compatibility_policy != MODEL_PACK_COMPATIBILITY_POLICY:
            raise ModelPackError("unsupported model-pack compatibility policy")
        entries = tuple(self.entries)
        if not entries:
            raise ModelPackError("model pack must contain at least one entry")
        aliases = [entry.alias for entry in entries]
        if len(set(aliases)) != len(aliases):
            raise ModelPackError("model-pack aliases must be unique")
        known = set(aliases)
        for entry in entries:
            fallback_alias = entry.fallback.alias
            if fallback_alias is None:
                continue
            if fallback_alias == entry.alias:
                raise ModelPackError("an entry cannot fall back to itself")
            if fallback_alias not in known:
                raise ModelPackError("fallback alias is not present in the model pack")
        _reject_fallback_cycles(entries)
        object.__setattr__(self, "entries", entries)

    @property
    def digest(self) -> str:
        """Return the content digest that pins this manifest."""

        return _digest_bytes(canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-ready manifest metadata."""

        return {
            "compatibility_policy": self.compatibility_policy,
            "entries": [entry.to_dict() for entry in self.entries],
            "pack_id": self.pack_id,
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return canonical JSON for persistence or signing."""

        return canonical_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class LocalArtifactBinding:
    """Explicit binding from a manifest alias to a local artifact."""

    alias: str
    runtime: str
    path: Path | None = field(default=None, repr=False)
    builtin_id: str | None = None
    builtin_digest: str | None = None

    def __post_init__(self) -> None:
        _require_alias(self.alias, "binding alias")
        _require_controlled(self.runtime, "binding runtime")
        if (self.path is None) == (self.builtin_id is None):
            raise ModelPackError(
                "a local binding requires exactly one of path or builtin_id"
            )
        if self.path is not None:
            if self.builtin_digest is not None:
                raise ModelPackError("file bindings cannot include a built-in digest")
            object.__setattr__(self, "path", Path(self.path))
            return
        assert self.builtin_id is not None
        _require_controlled(self.builtin_id, "builtin_id")
        _require_digest(self.builtin_digest, "builtin_digest")


@dataclass(frozen=True, slots=True)
class ClinicalTaskRequest:
    """Bounded task requirements used for deterministic route selection."""

    task: str
    output_schema: str
    output_schema_version: str
    language: str
    domain: str
    requested_alias: str | None = None
    runtime_preference: tuple[str, ...] = ()
    allow_experimental_generative: bool = False

    def __post_init__(self) -> None:
        if self.task not in TASK_KINDS:
            raise ModelPackError("unsupported clinical task request")
        _require_controlled(self.output_schema, "output_schema")
        _require_schema_version(self.output_schema_version, "output_schema_version")
        languages = _string_tuple((self.language,), "language", language=True)
        domains = _string_tuple((self.domain,), "domain")
        object.__setattr__(self, "language", languages[0])
        object.__setattr__(self, "domain", domains[0])
        if self.requested_alias is not None:
            _require_alias(self.requested_alias, "requested_alias")
        preferences = _string_tuple(
            self.runtime_preference, "runtime_preference", allow_empty=True
        )
        object.__setattr__(self, "runtime_preference", preferences)


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """Verified local route; the path is deliberately absent from serialization."""

    pack_id: str
    pack_digest: str
    alias: str
    task: str
    artifact_id: str
    artifact_digest: str
    revision: str
    runtime: str
    model_kind: str
    output_schema: str
    output_schema_version: str
    local_reference: str = field(repr=False)
    fallback_from: str | None = None
    schema_version: str = MODEL_PACK_SCHEMA_VERSION
    compatibility_policy: str = MODEL_PACK_COMPATIBILITY_POLICY

    def to_dict(self) -> dict[str, Any]:
        """Return path-free route provenance safe for logs and manifests."""

        return {
            "alias": self.alias,
            "artifact_digest": self.artifact_digest,
            "artifact_id": self.artifact_id,
            "compatibility_policy": self.compatibility_policy,
            "fallback_from": self.fallback_from,
            "model_kind": self.model_kind,
            "output_schema": self.output_schema,
            "output_schema_version": self.output_schema_version,
            "pack_digest": self.pack_digest,
            "pack_id": self.pack_id,
            "revision": self.revision,
            "runtime": self.runtime,
            "schema_version": self.schema_version,
            "task": self.task,
        }


class ClinicalTaskRouter:
    """Resolve bounded tasks to verified local artifacts without network access."""

    def __init__(
        self,
        manifest: ModelPackManifest,
        *,
        bindings: Sequence[LocalArtifactBinding],
        available_runtimes: Sequence[str],
        allowed_licenses: Sequence[str] = tuple(sorted(PERMISSIVE_LICENSES)),
    ) -> None:
        self._manifest = manifest
        self._entries = MappingProxyType(
            {entry.alias: entry for entry in manifest.entries}
        )
        binding_map = {binding.alias: binding for binding in bindings}
        if len(binding_map) != len(tuple(bindings)):
            raise ModelPackError("local binding aliases must be unique")
        unknown = sorted(set(binding_map) - set(self._entries))
        if unknown:
            raise ModelPackError("local binding is not declared by the model pack")
        self._bindings = MappingProxyType(binding_map)
        self._available_runtimes = frozenset(
            _string_tuple(available_runtimes, "available_runtimes", allow_empty=True)
        ) | {"builtin"}
        licenses = tuple(str(value).strip().lower() for value in allowed_licenses)
        if not licenses or any(not value for value in licenses):
            raise ModelPackError("allowed_licenses must contain identifiers")
        self._allowed_licenses = frozenset(licenses)

    @property
    def manifest(self) -> ModelPackManifest:
        """Return the immutable model-pack manifest."""

        return self._manifest

    def route(self, request: ClinicalTaskRequest) -> StoreResult[ModelRoute]:
        """Return a verified route or an explicit typed terminal state."""

        candidates = self._candidates(request)
        if not candidates:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "task_not_supported")
        if request.requested_alias is not None:
            requested = self._entries.get(request.requested_alias)
            if requested is None or requested not in candidates:
                return StoreResult.outcome(
                    StoreState.UNSUPPORTED, "alias_not_supported"
                )
            primary = requested
        else:
            bounded_candidates = tuple(
                entry for entry in candidates if entry.model_kind != "generative"
            )
            if not bounded_candidates:
                return StoreResult.outcome(
                    StoreState.DENIED, "generative_profile_required"
                )
            primary = min(
                bounded_candidates,
                key=lambda item: self._sort_key(item, request),
            )
        if primary.model_kind == "generative" and not (
            request.allow_experimental_generative and primary.experimental
        ):
            return StoreResult.outcome(StoreState.DENIED, "generative_profile_required")
        result = self._resolve_entry(primary, fallback_from=None)
        if result.ok or result.code not in {"runtime_unavailable", "alias_unbound"}:
            return result
        fallback_alias = primary.fallback.alias
        if fallback_alias is None:
            return result
        fallback = self._entries[fallback_alias]
        if fallback.task != request.task:
            return StoreResult.outcome(StoreState.CONFLICT, "fallback_task_mismatch")
        if fallback.model_kind == "generative":
            return StoreResult.outcome(StoreState.DENIED, "generative_profile_required")
        return self._resolve_entry(fallback, fallback_from=primary.alias)

    def _candidates(self, request: ClinicalTaskRequest) -> tuple[ModelPackEntry, ...]:
        return tuple(
            entry
            for entry in self._manifest.entries
            if entry.task == request.task
            and entry.output_schema == request.output_schema
            and _same_major(entry.output_schema_version, request.output_schema_version)
            and (request.language in entry.languages or "mul" in entry.languages)
            and (request.domain in entry.domains or "general" in entry.domains)
        )

    def _sort_key(
        self, entry: ModelPackEntry, request: ClinicalTaskRequest
    ) -> tuple[int, int, int, int, str]:
        language_rank = 0 if request.language in entry.languages else 1
        domain_rank = 0 if request.domain in entry.domains else 1
        try:
            runtime_rank = request.runtime_preference.index(entry.runtime)
        except ValueError:
            runtime_rank = len(request.runtime_preference)
        kind_rank = {"bounded": 0, "deterministic": 1, "generative": 2}[
            entry.model_kind
        ]
        return (
            language_rank,
            domain_rank,
            runtime_rank,
            kind_rank * 10_001 + entry.priority,
            entry.alias,
        )

    def _resolve_entry(
        self, entry: ModelPackEntry, *, fallback_from: str | None
    ) -> StoreResult[ModelRoute]:
        if entry.license not in self._allowed_licenses:
            return StoreResult.outcome(StoreState.DENIED, "license_denied")
        if not entry.quantization.within_tolerance:
            return StoreResult.outcome(StoreState.DENIED, "quantization_delta_exceeded")
        if entry.runtime not in self._available_runtimes:
            return StoreResult.outcome(StoreState.UNSUPPORTED, "runtime_unavailable")
        binding = self._bindings.get(entry.alias)
        if binding is None:
            return StoreResult.outcome(StoreState.UNKNOWN, "alias_unbound")
        if binding.runtime != entry.runtime:
            return StoreResult.outcome(StoreState.CONFLICT, "runtime_binding_mismatch")
        verified = _verify_binding(entry, binding)
        if not verified.ok:
            return verified
        assert verified.value is not None
        return StoreResult.success(
            ModelRoute(
                pack_id=self._manifest.pack_id,
                pack_digest=self._manifest.digest,
                alias=entry.alias,
                task=entry.task,
                artifact_id=entry.artifact_id,
                artifact_digest=entry.artifact_digest,
                revision=entry.revision,
                runtime=entry.runtime,
                model_kind=entry.model_kind,
                output_schema=entry.output_schema,
                output_schema_version=entry.output_schema_version,
                local_reference=verified.value,
                fallback_from=fallback_from,
            )
        )


def load_model_pack_json(payload: str | bytes) -> ModelPackManifest:
    """Load strict model-pack JSON without accepting duplicates or non-finite data."""

    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ModelPackError("model pack is not valid UTF-8 JSON") from exc
    return model_pack_from_dict(decoded)


def load_model_pack(path: str | Path) -> ModelPackManifest:
    """Load a model pack from an explicit local file."""

    model_pack_path = Path(path)
    if not model_pack_path.is_file() or model_pack_path.is_symlink():
        raise ModelPackError("model pack must be a regular local file")
    return load_model_pack_json(model_pack_path.read_bytes())


def model_pack_from_dict(value: Any) -> ModelPackManifest:
    """Build and validate a model pack from a decoded mapping."""

    data = _exact_mapping(
        value,
        "model pack",
        {"compatibility_policy", "entries", "pack_id", "schema_version"},
    )
    raw_entries = data["entries"]
    if not isinstance(raw_entries, list):
        raise ModelPackError("model-pack entries must be an array")
    return ModelPackManifest(
        pack_id=_string(data["pack_id"], "pack_id"),
        entries=tuple(_entry_from_dict(item) for item in raw_entries),
        schema_version=_string(data["schema_version"], "schema_version"),
        compatibility_policy=_string(
            data["compatibility_policy"], "compatibility_policy"
        ),
    )


def load_model_pack_schema() -> dict[str, Any]:
    """Load the bundled public model-pack JSON Schema."""

    path = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "schemas"
        / "json"
        / "clinical_model_pack.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def load_model_route_schema() -> dict[str, Any]:
    """Load the bundled path-free model-route JSON Schema."""

    path = (
        Path(__file__).resolve().parents[1]
        / "core"
        / "schemas"
        / "json"
        / "clinical_model_route.schema.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def digest_local_artifact(path: str | Path) -> str:
    """Return a deterministic digest for a regular file or directory tree."""

    artifact = Path(path)
    if artifact.is_symlink():
        raise ModelPackError("artifact symlinks are not accepted")
    if artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"
    if not artifact.is_dir():
        raise ModelPackError("artifact path does not exist")
    digest = hashlib.sha256()
    files = sorted(item for item in artifact.rglob("*") if item.is_file())
    if not files:
        raise ModelPackError("artifact directory is empty")
    for item in files:
        if item.is_symlink():
            raise ModelPackError("artifact symlinks are not accepted")
        relative = item.relative_to(artifact).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _verify_binding(
    entry: ModelPackEntry, binding: LocalArtifactBinding
) -> StoreResult[str]:
    if binding.builtin_id is not None:
        if entry.artifact_kind != "builtin":
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_kind_mismatch")
        if binding.builtin_id != entry.artifact_id:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_binding_mismatch")
        if binding.builtin_digest != entry.artifact_digest:
            return StoreResult.outcome(StoreState.CONFLICT, "artifact_digest_mismatch")
        return StoreResult.success(f"builtin://{binding.builtin_id}")
    if entry.artifact_kind != "model" or binding.path is None:
        return StoreResult.outcome(StoreState.CONFLICT, "artifact_kind_mismatch")
    if binding.path.is_symlink():
        return StoreResult.outcome(StoreState.DENIED, "artifact_path_unsafe")
    if not binding.path.exists():
        return StoreResult.outcome(StoreState.UNKNOWN, "artifact_missing")
    try:
        actual = digest_local_artifact(binding.path)
    except (OSError, ModelPackError):
        return StoreResult.outcome(StoreState.FAILURE, "artifact_verification_failed")
    if actual != entry.artifact_digest:
        return StoreResult.outcome(StoreState.CONFLICT, "artifact_digest_mismatch")
    return StoreResult.success(str(binding.path.resolve()))


def _entry_from_dict(value: Any) -> ModelPackEntry:
    fields = {
        "alias",
        "artifact_digest",
        "artifact_id",
        "artifact_kind",
        "calibration",
        "compatibility_policy",
        "domains",
        "experimental",
        "fallback",
        "family",
        "languages",
        "license",
        "model_kind",
        "output_schema",
        "output_schema_version",
        "priority",
        "quantization",
        "revision",
        "runtime",
        "schema_version",
        "task",
    }
    data = _exact_mapping(value, "model-pack entry", fields)
    calibration_data = _exact_mapping(
        data["calibration"],
        "calibration",
        {
            "calibration_dataset_digest",
            "holdout_dataset_digest",
            "method",
            "seed",
            "threshold",
        },
    )
    quantization_data = _exact_mapping(
        data["quantization"],
        "quantization",
        {
            "evaluation_digest",
            "maximum_delta",
            "metric",
            "mode",
            "observed_delta",
        },
    )
    fallback_data = _exact_mapping(data["fallback"], "fallback", {"alias", "mode"})
    return ModelPackEntry(
        alias=_string(data["alias"], "alias"),
        task=_string(data["task"], "task"),
        family=_string(data["family"], "family"),
        artifact_id=_string(data["artifact_id"], "artifact_id"),
        artifact_kind=_string(data["artifact_kind"], "artifact_kind"),
        revision=_string(data["revision"], "revision"),
        artifact_digest=_string(data["artifact_digest"], "artifact_digest"),
        license=_string(data["license"], "license"),
        runtime=_string(data["runtime"], "runtime"),
        model_kind=_string(data["model_kind"], "model_kind"),
        output_schema=_string(data["output_schema"], "output_schema"),
        output_schema_version=_string(
            data["output_schema_version"], "output_schema_version"
        ),
        languages=_list_of_strings(data["languages"], "languages"),
        domains=_list_of_strings(data["domains"], "domains"),
        calibration=CalibrationMetadata(
            method=_string(calibration_data["method"], "calibration method"),
            threshold=_optional_float(calibration_data["threshold"]),
            calibration_dataset_digest=_optional_string(
                calibration_data["calibration_dataset_digest"]
            ),
            holdout_dataset_digest=_optional_string(
                calibration_data["holdout_dataset_digest"]
            ),
            seed=_optional_int(calibration_data["seed"]),
        ),
        quantization=QuantizationMetadata(
            mode=_string(quantization_data["mode"], "quantization mode"),
            metric=_optional_string(quantization_data["metric"]),
            observed_delta=_optional_float(quantization_data["observed_delta"]),
            maximum_delta=_optional_float(quantization_data["maximum_delta"]),
            evaluation_digest=_optional_string(quantization_data["evaluation_digest"]),
        ),
        fallback=FallbackPolicy(
            mode=cast(
                Literal["none", "alias"],
                _string(fallback_data["mode"], "fallback mode"),
            ),
            alias=_optional_string(fallback_data["alias"]),
        ),
        priority=_integer(data["priority"], "priority"),
        experimental=_boolean(data["experimental"], "experimental"),
        schema_version=_string(data["schema_version"], "schema_version"),
        compatibility_policy=_string(
            data["compatibility_policy"], "compatibility_policy"
        ),
    )


def _reject_fallback_cycles(entries: Sequence[ModelPackEntry]) -> None:
    targets = {entry.alias: entry.fallback.alias for entry in entries}
    for alias in targets:
        seen: set[str] = set()
        current: str | None = alias
        while current is not None:
            if current in seen:
                raise ModelPackError("model-pack fallback graph contains a cycle")
            seen.add(current)
            current = targets[current]


def _exact_mapping(value: Any, name: str, expected: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelPackError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ModelPackError(f"{name} fields do not match the public contract")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ModelPackError("model pack contains a duplicate JSON key")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ModelPackError(f"model pack contains non-finite number: {value}")


def _require_controlled(value: str, name: str) -> None:
    if not isinstance(value, str) or _CONTROLLED_RE.fullmatch(value) is None:
        raise ModelPackError(f"{name} must be a controlled identifier")


def _require_alias(value: str, name: str) -> None:
    if not isinstance(value, str) or _ALIAS_RE.fullmatch(value) is None:
        raise ModelPackError(f"{name} must be a local alias")


def _require_digest(value: str | None, name: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ModelPackError(f"{name} must be a sha256 digest")


def _require_schema_version(value: str, name: str) -> None:
    if not isinstance(value, str) or _SCHEMA_VERSION_RE.fullmatch(value) is None:
        raise ModelPackError(f"{name} must be a semantic schema version")


def _string_tuple(
    values: Sequence[str],
    name: str,
    *,
    language: bool = False,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ModelPackError(f"{name} must be a sequence")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ModelPackError(f"{name} must contain non-empty strings")
        normalized = value.strip().lower()
        if language:
            if normalized != "mul" and _LANGUAGE_RE.fullmatch(normalized) is None:
                raise ModelPackError(f"{name} contains an invalid language tag")
        elif _CONTROLLED_RE.fullmatch(normalized) is None:
            raise ModelPackError(f"{name} contains an uncontrolled identifier")
        if normalized in result:
            raise ModelPackError(f"{name} must not contain duplicates")
        result.append(normalized)
    if not result and not allow_empty:
        raise ModelPackError(f"{name} must not be empty")
    return tuple(result)


def _same_major(left: str, right: str) -> bool:
    return left.split(".", 1)[0] == right.split(".", 1)[0]


def _finite_unit_interval(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def _finite_non_negative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _digest_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ModelPackError(f"{name} must be a string")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return _string(value, "optional value")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelPackError("optional numeric value must be a number")
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _integer(value, "optional integer")


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ModelPackError(f"{name} must be an integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ModelPackError(f"{name} must be a boolean")
    return value


def _list_of_strings(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ModelPackError(f"{name} must be an array")
    return tuple(_string(item, name) for item in value)


MODEL_FAMILY_CAPABILITIES: Mapping[str, ModelFamilyCapability] = MappingProxyType(
    {
        capability.family: capability
        for capability in (
            ModelFamilyCapability(
                family="gliner2_5",
                tasks=("span_extraction",),
                runtimes=("torch", "mlx"),
                zero_shot_labels=True,
            ),
            ModelFamilyCapability(
                family="deberta_v2",
                tasks=(
                    "assertion",
                    "classification",
                    "pair_scoring",
                    "relation_extraction",
                    "temporality",
                    "token_classification",
                ),
                runtimes=("torch", "mlx", "onnx"),
            ),
            ModelFamilyCapability(
                family="modernbert",
                tasks=(
                    "assertion",
                    "classification",
                    "pair_scoring",
                    "relation_extraction",
                    "temporality",
                    "token_classification",
                ),
                runtimes=("torch", "mlx", "onnx"),
            ),
            ModelFamilyCapability(
                family="openmed_token_classifier",
                tasks=("token_classification",),
                runtimes=("torch", "mlx", "onnx"),
            ),
            ModelFamilyCapability(
                family="deterministic_rules",
                tasks=tuple(sorted(TASK_KINDS)),
                runtimes=("builtin",),
            ),
        )
    }
)


__all__ = [
    "ARTIFACT_KINDS",
    "CALIBRATION_METHODS",
    "MODEL_KINDS",
    "MODEL_PACK_COMPATIBILITY_POLICY",
    "MODEL_PACK_ENTRY_SCHEMA_VERSION",
    "MODEL_FAMILY_CAPABILITIES",
    "MODEL_PACK_SCHEMA_VERSION",
    "PERMISSIVE_LICENSES",
    "QUANTIZATION_MODES",
    "TASK_KINDS",
    "CalibrationMetadata",
    "ClinicalTaskRequest",
    "ClinicalTaskRouter",
    "FallbackPolicy",
    "LocalArtifactBinding",
    "ModelPackEntry",
    "ModelPackError",
    "ModelPackManifest",
    "ModelFamilyCapability",
    "ModelRoute",
    "QuantizationMetadata",
    "digest_local_artifact",
    "load_model_pack",
    "load_model_pack_json",
    "load_model_pack_schema",
    "load_model_route_schema",
    "model_pack_from_dict",
]
