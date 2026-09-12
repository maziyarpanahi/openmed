"""Deterministic, privacy-safe diffs for synthetic evaluation run manifests.

Evaluation reports often need to answer whether two runs used the same model,
tokenizer, policy, and fixture set.  This module compares only immutable
fingerprints, versions, and declared evaluation-slice identifiers.  Extra run
metadata is deliberately ignored, so prompts, notes, generated text, and
other free-form values cannot enter a diff report.

The implementation is local-only.  It hashes normalized JSON with the standard
library and never resolves a model, fixture, or policy reference over the
network.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_PROVENANCE_DIFF_SCHEMA_VERSION = "openmed.eval.model_provenance_diff.v1"
PROVENANCE_COMPONENTS = ("model", "tokenizer", "policy", "fixtures")

DRIFT_UNCHANGED = "unchanged"
DRIFT_FINGERPRINT_CHANGED = "fingerprint_changed"
DRIFT_VERSION_CHANGED = "version_changed"
DRIFT_FINGERPRINT_AND_VERSION_CHANGED = "fingerprint_and_version_changed"
SLICE_ADDED = "added"
SLICE_REMOVED = "removed"
SLICE_CHANGED = "changed"

_MISSING = object()
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+@-]{0,255}\Z")
_RAW_VALUE_KEYS = frozenset(
    {
        "clinical_text",
        "completion",
        "completions",
        "generated_clinical_text",
        "generated_output",
        "input_text",
        "message",
        "messages",
        "note",
        "notes",
        "output",
        "outputs",
        "prompt",
        "prompts",
        "response",
        "responses",
        "text",
        "texts",
    }
)


class ModelProvenanceError(ValueError):
    """Base error for malformed or privacy-unsafe provenance input."""


class ModelProvenanceInputError(ModelProvenanceError):
    """Raised when a run manifest does not match the supported safe shape."""


class ModelProvenancePrivacyError(ModelProvenanceError):
    """Raised when a known provenance value is not a safe identifier."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hash_payload(value: Any) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(value).encode()).hexdigest()}"


def _safe_identifier(value: Any, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ModelProvenancePrivacyError(f"{field} must be a safe identifier")
    text = str(value).strip()
    if not text or not _SAFE_IDENTIFIER.fullmatch(text):
        raise ModelProvenancePrivacyError(f"{field} must be a safe identifier")
    return text


def _optional_identifier(value: Any, field: str) -> str | None:
    if value is None or value is _MISSING:
        return None
    return _safe_identifier(value, field)


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return _MISSING


@dataclass(frozen=True)
class ProvenanceComponent:
    """Immutable identity and version for one run-manifest component."""

    fingerprint: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "fingerprint",
            _safe_identifier(self.fingerprint, "component fingerprint"),
        )
        object.__setattr__(
            self,
            "version",
            _safe_identifier(self.version, "component version"),
        )

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        field: str,
        fallback_version: Any = _MISSING,
    ) -> "ProvenanceComponent":
        """Normalize a component mapping without retaining unknown metadata."""

        if not isinstance(payload, Mapping):
            raise ModelProvenanceInputError(f"{field} provenance must be an object")

        fingerprint = _first_present(
            payload,
            ("fingerprint", "immutable_fingerprint", "digest", "hash"),
        )
        if fingerprint is _MISSING:
            fingerprint = _first_present(
                payload,
                ("content_hash", "fixture_hash", "manifest_hash"),
            )
        version = _first_present(
            payload,
            ("version", "revision", "manifest_version", "schema_version"),
        )
        if version is _MISSING:
            version = fallback_version
        if fingerprint is _MISSING:
            raise ModelProvenanceInputError(f"{field} provenance needs a fingerprint")
        if version is _MISSING:
            raise ModelProvenanceInputError(f"{field} provenance needs a version")
        return cls(
            fingerprint=_safe_identifier(fingerprint, f"{field} fingerprint"),
            version=_safe_identifier(version, f"{field} version"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the only component fields permitted in a diff report."""

        return {"fingerprint": self.fingerprint, "version": self.version}


@dataclass(frozen=True)
class EvaluationSlice:
    """A declared evaluation slice and its optional immutable provenance."""

    name: str
    fingerprint: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _safe_identifier(self.name, "evaluation slice name"),
        )
        object.__setattr__(
            self,
            "fingerprint",
            _optional_identifier(self.fingerprint, "evaluation slice fingerprint"),
        )
        object.__setattr__(
            self,
            "version",
            _optional_identifier(self.version, "evaluation slice version"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "EvaluationSlice":
        """Normalize one slice declaration and discard unknown metadata."""

        if not isinstance(payload, Mapping):
            raise ModelProvenanceInputError("evaluation slice must be an object")
        name = _first_present(payload, ("name", "id", "slice_id", "slice"))
        if name is _MISSING:
            raise ModelProvenanceInputError("evaluation slice needs a name")
        fingerprint = _first_present(
            payload,
            ("fingerprint", "immutable_fingerprint", "digest", "hash"),
        )
        version = _first_present(
            payload,
            ("version", "revision", "schema_version"),
        )
        return cls(
            name=_safe_identifier(name, "evaluation slice name"),
            fingerprint=_optional_identifier(
                fingerprint,
                "evaluation slice fingerprint",
            ),
            version=_optional_identifier(version, "evaluation slice version"),
        )

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic, text-free slice declaration."""

        payload = {"name": self.name}
        if self.fingerprint is not None:
            payload["fingerprint"] = self.fingerprint
        if self.version is not None:
            payload["version"] = self.version
        return payload


@dataclass(frozen=True)
class ModelProvenanceManifest:
    """Normalized provenance for one synthetic evaluation run."""

    model: ProvenanceComponent
    tokenizer: ProvenanceComponent
    policy: ProvenanceComponent
    fixtures: ProvenanceComponent
    evaluation_slices: tuple[EvaluationSlice, ...] = ()
    schema_version: str = MODEL_PROVENANCE_DIFF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in PROVENANCE_COMPONENTS:
            value = getattr(self, field)
            if not isinstance(value, ProvenanceComponent):
                raise ModelProvenanceInputError(
                    f"{field} provenance must be a ProvenanceComponent"
                )
        slices = tuple(self.evaluation_slices)
        if any(not isinstance(item, EvaluationSlice) for item in slices):
            raise ModelProvenanceInputError(
                "evaluation_slices must contain EvaluationSlice values"
            )
        names = tuple(item.name for item in slices)
        if len(names) != len(set(names)):
            raise ModelProvenanceInputError("evaluation slice names must be unique")
        object.__setattr__(
            self, "evaluation_slices", tuple(sorted(slices, key=lambda item: item.name))
        )
        object.__setattr__(
            self,
            "schema_version",
            _safe_identifier(self.schema_version, "manifest schema version"),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ModelProvenanceManifest":
        """Normalize a mapping while dropping all non-provenance fields."""

        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise ModelProvenanceInputError("run manifest must be an object")

        components: dict[str, ProvenanceComponent] = {}
        for field in PROVENANCE_COMPONENTS:
            aliases = (
                (
                    "fixture",
                    "fixtures",
                    "fixture_provenance",
                    "dataset",
                    "dataset_provenance",
                )
                if field == "fixtures"
                else (field, f"{field}_provenance")
            )
            component = _first_present(payload, aliases)
            version_keys = [f"{field}_version"]
            fingerprint_keys = [f"{field}_fingerprint", f"{field}_digest"]
            if field == "fixtures":
                version_keys.append("fixture_version")
                fingerprint_keys.append("fixture_fingerprint")
            fallback_version = _first_present(payload, version_keys)
            if component is _MISSING:
                fingerprint = _first_present(payload, fingerprint_keys)
                if fingerprint is _MISSING:
                    raise ModelProvenanceInputError(f"{field} provenance is required")
                component = {"fingerprint": fingerprint}
            components[field] = _coerce_component(
                component,
                field=field,
                fallback_version=fallback_version,
            )

        raw_slices = _first_present(
            payload,
            (
                "evaluation_slices",
                "declared_evaluation_slices",
                "eval_slices",
                "slices",
            ),
        )
        slices = (
            ()
            if raw_slices is _MISSING or raw_slices is None
            else _coerce_slices(raw_slices)
        )
        return cls(
            model=components["model"],
            tokenizer=components["tokenizer"],
            policy=components["policy"],
            fixtures=components["fixtures"],
            evaluation_slices=slices,
            schema_version=MODEL_PROVENANCE_DIFF_SCHEMA_VERSION,
        )

    @property
    def manifest_fingerprint(self) -> str:
        """Return a content fingerprint of the normalized manifest."""

        return _hash_payload(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, privacy-safe manifest representation."""

        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "tokenizer": self.tokenizer.to_dict(),
            "policy": self.policy.to_dict(),
            "fixtures": self.fixtures.to_dict(),
            "evaluation_slices": [item.to_dict() for item in self.evaluation_slices],
        }

    def to_payload(self) -> dict[str, Any]:
        """Return the manifest plus its derived content fingerprint."""

        payload = self.to_dict()
        payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload


def _coerce_component(
    value: Any,
    *,
    field: str,
    fallback_version: Any = _MISSING,
) -> ProvenanceComponent:
    if isinstance(value, ProvenanceComponent):
        return value
    if isinstance(value, Mapping):
        return ProvenanceComponent.from_mapping(
            value,
            field=field,
            fallback_version=fallback_version,
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            raise ModelProvenanceInputError(f"{field} provenance cannot be empty")
        entries = tuple(
            _coerce_component(
                item,
                field=f"{field} item",
                fallback_version=fallback_version,
            )
            for item in value
        )
        return ProvenanceComponent(
            fingerprint=_hash_payload(
                {"component": field, "items": [item.to_dict() for item in entries]}
            ),
            version=_hash_payload(
                {"component": field, "versions": [item.version for item in entries]}
            ),
        )
    raise ModelProvenanceInputError(f"{field} provenance must be an object")


def _coerce_slice(value: Any) -> EvaluationSlice:
    if isinstance(value, EvaluationSlice):
        return value
    if isinstance(value, Mapping):
        return EvaluationSlice.from_mapping(value)
    if isinstance(value, str):
        return EvaluationSlice(name=value)
    raise ModelProvenanceInputError("evaluation slice must be a safe declaration")


def _coerce_slices(value: Any) -> tuple[EvaluationSlice, ...]:
    if isinstance(value, Mapping):
        descriptor_keys = {"name", "id", "slice_id", "slice"}
        if descriptor_keys.intersection(value):
            values: Sequence[Any] = (value,)
        elif "names" in value:
            values = value["names"]
        elif "slice_names" in value:
            values = value["slice_names"]
        else:
            descriptors: list[dict[str, Any]] = []
            for name, descriptor in sorted(
                value.items(), key=lambda item: str(item[0])
            ):
                if isinstance(descriptor, Mapping):
                    item = dict(descriptor)
                    item["name"] = name
                elif descriptor is None:
                    item = {"name": name}
                else:
                    item = {"name": name, "version": descriptor}
                descriptors.append(item)
            values = descriptors
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = value
    else:
        raise ModelProvenanceInputError("evaluation_slices must be a list or object")

    try:
        slices = tuple(_coerce_slice(item) for item in values)
    except TypeError:
        raise ModelProvenanceInputError(
            "evaluation_slices must contain declarations"
        ) from None
    names = tuple(item.name for item in slices)
    if len(names) != len(set(names)):
        raise ModelProvenanceInputError("evaluation slice names must be unique")
    return tuple(sorted(slices, key=lambda item: item.name))


def _load_manifest(source: Any) -> ModelProvenanceManifest:
    if isinstance(source, ModelProvenanceManifest):
        return source
    if isinstance(source, Mapping):
        return ModelProvenanceManifest.from_mapping(source)
    if isinstance(source, (str, Path)):
        path = Path(source)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ModelProvenanceInputError(
                "run manifest could not be read locally"
            ) from None
        return ModelProvenanceManifest.from_mapping(payload)
    raise ModelProvenanceInputError("run manifest must be an object or local JSON path")


def build_model_provenance_manifest(
    *,
    model: Any,
    tokenizer: Any,
    policy: Any,
    fixtures: Any,
    evaluation_slices: Any = (),
) -> dict[str, Any]:
    """Build a normalized manifest using only local, immutable provenance.

    ``model``, ``tokenizer``, ``policy``, and ``fixtures`` each need a
    ``fingerprint`` and ``version``.  ``evaluation_slices`` accepts safe names
    or mappings containing ``name``, ``fingerprint``, and ``version``.  Any
    other fields supplied by callers are ignored and never copied to the
    returned payload.
    """

    manifest = ModelProvenanceManifest.from_mapping(
        {
            "model": model,
            "tokenizer": tokenizer,
            "policy": policy,
            "fixtures": fixtures,
            "evaluation_slices": evaluation_slices,
        }
    )
    return manifest.to_payload()


@dataclass(frozen=True)
class ProvenanceComponentDiff:
    """Comparison of one required provenance component."""

    component: str
    before: ProvenanceComponent
    after: ProvenanceComponent

    @property
    def fingerprint_changed(self) -> bool:
        return self.before.fingerprint != self.after.fingerprint

    @property
    def version_changed(self) -> bool:
        return self.before.version != self.after.version

    @property
    def changed(self) -> bool:
        return self.fingerprint_changed or self.version_changed

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.fingerprint_changed:
            reasons.append(DRIFT_FINGERPRINT_CHANGED)
        if self.version_changed:
            reasons.append(DRIFT_VERSION_CHANGED)
        return tuple(reasons)

    @property
    def classification(self) -> str:
        if self.fingerprint_changed and self.version_changed:
            return DRIFT_FINGERPRINT_AND_VERSION_CHANGED
        if self.fingerprint_changed:
            return DRIFT_FINGERPRINT_CHANGED
        if self.version_changed:
            return DRIFT_VERSION_CHANGED
        return DRIFT_UNCHANGED

    def to_dict(self) -> dict[str, Any]:
        """Return a diff row containing no unknown manifest metadata."""

        return {
            "changed": self.changed,
            "classification": self.classification,
            "reasons": list(self.reasons),
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationSliceDiff:
    """Comparison of one added, removed, or changed slice declaration."""

    name: str
    change: str
    before: EvaluationSlice | None = None
    after: EvaluationSlice | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _safe_identifier(self.name, "slice name"))
        if self.change not in {SLICE_ADDED, SLICE_REMOVED, SLICE_CHANGED}:
            raise ModelProvenanceInputError("invalid evaluation slice change")
        if self.change == SLICE_ADDED and (
            self.before is not None or self.after is None
        ):
            raise ModelProvenanceInputError("added slice diff has invalid boundaries")
        if self.change == SLICE_REMOVED and (
            self.before is None or self.after is not None
        ):
            raise ModelProvenanceInputError("removed slice diff has invalid boundaries")
        if self.change == SLICE_CHANGED and (self.before is None or self.after is None):
            raise ModelProvenanceInputError("changed slice diff has invalid boundaries")

    @property
    def reasons(self) -> tuple[str, ...]:
        if self.change != SLICE_CHANGED or self.before is None or self.after is None:
            return (self.change,)
        reasons: list[str] = []
        if self.before.fingerprint != self.after.fingerprint:
            reasons.append(DRIFT_FINGERPRINT_CHANGED)
        if self.before.version != self.after.version:
            reasons.append(DRIFT_VERSION_CHANGED)
        return tuple(reasons) or (DRIFT_UNCHANGED,)

    def to_dict(self) -> dict[str, Any]:
        """Return a safe slice change row."""

        return {
            "name": self.name,
            "change": self.change,
            "reasons": list(self.reasons),
            "before": None if self.before is None else self.before.to_dict(),
            "after": None if self.after is None else self.after.to_dict(),
        }


def _diff_slices(
    before: tuple[EvaluationSlice, ...],
    after: tuple[EvaluationSlice, ...],
) -> tuple[EvaluationSliceDiff, ...]:
    before_by_name = {item.name: item for item in before}
    after_by_name = {item.name: item for item in after}
    changes: list[EvaluationSliceDiff] = []
    for name in sorted(set(before_by_name) | set(after_by_name)):
        before_item = before_by_name.get(name)
        after_item = after_by_name.get(name)
        if before_item is None:
            changes.append(
                EvaluationSliceDiff(name=name, change=SLICE_ADDED, after=after_item)
            )
        elif after_item is None:
            changes.append(
                EvaluationSliceDiff(name=name, change=SLICE_REMOVED, before=before_item)
            )
        elif before_item != after_item:
            changes.append(
                EvaluationSliceDiff(
                    name=name,
                    change=SLICE_CHANGED,
                    before=before_item,
                    after=after_item,
                )
            )
    return tuple(changes)


@dataclass(frozen=True)
class ModelProvenanceDiff:
    """Deterministic report comparing two normalized run manifests."""

    before_manifest: ModelProvenanceManifest
    after_manifest: ModelProvenanceManifest
    component_changes: tuple[ProvenanceComponentDiff, ...]
    slice_changes: tuple[EvaluationSliceDiff, ...]

    @property
    def changed(self) -> bool:
        return any(item.changed for item in self.component_changes) or bool(
            self.slice_changes
        )

    @property
    def drift_detected(self) -> bool:
        return self.changed

    @property
    def has_drift(self) -> bool:
        return self.changed

    @property
    def changed_components(self) -> tuple[str, ...]:
        components = [item.component for item in self.component_changes if item.changed]
        if self.slice_changes:
            components.append("evaluation_slices")
        return tuple(components)

    @property
    def drift_categories(self) -> tuple[str, ...]:
        return self.changed_components

    @property
    def components(self) -> Mapping[str, ProvenanceComponentDiff]:
        return {item.component: item for item in self.component_changes}

    @property
    def evaluation_slices(self) -> Mapping[str, Any]:
        added = [item.name for item in self.slice_changes if item.change == SLICE_ADDED]
        removed = [
            item.name for item in self.slice_changes if item.change == SLICE_REMOVED
        ]
        changed = [
            item.to_dict()
            for item in self.slice_changes
            if item.change == SLICE_CHANGED
        ]
        return {
            "changed": changed,
            "added": added,
            "removed": removed,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the report as deterministic JSON-ready data."""

        payload = {
            "schema_version": MODEL_PROVENANCE_DIFF_SCHEMA_VERSION,
            "before_manifest_fingerprint": self.before_manifest.manifest_fingerprint,
            "after_manifest_fingerprint": self.after_manifest.manifest_fingerprint,
            "changed": self.changed,
            "drift_detected": self.drift_detected,
            "changed_components": list(self.changed_components),
            "drift_categories": list(self.drift_categories),
            "components": {
                item.component: item.to_dict() for item in self.component_changes
            },
            "evaluation_slices": dict(self.evaluation_slices),
        }
        assert_no_raw_text(payload)
        return payload

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )


def diff_model_provenance(
    before: Any = None,
    after: Any = None,
    *,
    before_manifest: Any = None,
    after_manifest: Any = None,
    baseline: Any = None,
    candidate: Any = None,
) -> ModelProvenanceDiff:
    """Compare two local run manifests without copying free-form metadata.

    Positional ``before``/``after`` arguments are the canonical API.  The
    keyword aliases ``before_manifest``/``after_manifest`` and
    ``baseline``/``candidate`` are accepted for callers that use the language
    used in evaluation reports.
    """

    if before is None:
        before = before_manifest if before_manifest is not None else baseline
    if after is None:
        after = after_manifest if after_manifest is not None else candidate
    if before is None or after is None:
        raise ModelProvenanceInputError("both before and after manifests are required")

    before_manifest_value = _load_manifest(before)
    after_manifest_value = _load_manifest(after)
    component_changes = tuple(
        ProvenanceComponentDiff(
            component=field,
            before=getattr(before_manifest_value, field),
            after=getattr(after_manifest_value, field),
        )
        for field in PROVENANCE_COMPONENTS
    )
    return ModelProvenanceDiff(
        before_manifest=before_manifest_value,
        after_manifest=after_manifest_value,
        component_changes=component_changes,
        slice_changes=_diff_slices(
            before_manifest_value.evaluation_slices,
            after_manifest_value.evaluation_slices,
        ),
    )


def load_model_provenance_manifest(path: str | Path) -> ModelProvenanceManifest:
    """Load and normalize one manifest from a local JSON file."""

    return _load_manifest(path)


def write_model_provenance_manifest(
    path: str | Path,
    manifest: ModelProvenanceManifest | Mapping[str, Any],
) -> Path:
    """Write a normalized manifest to a local JSON file."""

    normalized = _load_manifest(manifest)
    output_path = Path(path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                normalized.to_payload(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise ModelProvenanceInputError(
            "normalized manifest could not be written locally"
        ) from None
    return output_path


def assert_no_raw_text(payload: Any) -> None:
    """Raise if a report payload contains free-form or sensitive values.

    This guard is intended for report boundaries.  Manifest normalization
    ignores unknown fields, while this function verifies that the resulting
    report consists only of safe identifiers, booleans, numbers, and nulls.
    """

    _assert_safe_payload(payload, path="report")


def _assert_safe_payload(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str) or key.casefold() in _RAW_VALUE_KEYS:
                raise ModelProvenancePrivacyError("report contains a forbidden field")
            if not _SAFE_IDENTIFIER.fullmatch(key):
                raise ModelProvenancePrivacyError("report contains an unsafe field")
            _assert_safe_payload(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_safe_payload(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise ModelProvenancePrivacyError("report contains a free-form value")
        return
    if isinstance(value, (bool, int, float)) or value is None:
        return
    raise ModelProvenancePrivacyError("report contains an unsupported value")


build_provenance_manifest = build_model_provenance_manifest
compute_model_provenance_diff = diff_model_provenance
compare_model_provenance = diff_model_provenance


__all__ = [
    "DRIFT_FINGERPRINT_AND_VERSION_CHANGED",
    "DRIFT_FINGERPRINT_CHANGED",
    "DRIFT_UNCHANGED",
    "DRIFT_VERSION_CHANGED",
    "EvaluationSlice",
    "EvaluationSliceDiff",
    "MODEL_PROVENANCE_DIFF_SCHEMA_VERSION",
    "ModelProvenanceDiff",
    "ModelProvenanceError",
    "ModelProvenanceInputError",
    "ModelProvenanceManifest",
    "ModelProvenancePrivacyError",
    "PROVENANCE_COMPONENTS",
    "ProvenanceComponent",
    "ProvenanceComponentDiff",
    "SLICE_ADDED",
    "SLICE_CHANGED",
    "SLICE_REMOVED",
    "assert_no_raw_text",
    "build_model_provenance_manifest",
    "build_provenance_manifest",
    "compare_model_provenance",
    "compute_model_provenance_diff",
    "diff_model_provenance",
    "load_model_provenance_manifest",
    "write_model_provenance_manifest",
]
