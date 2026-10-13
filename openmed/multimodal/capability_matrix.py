"""Compare versioned provider capability declarations without loading media."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import islice

SCHEMA_VERSION = 1
_CODE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_VERSION = re.compile(r"[0-9]{1,6}\.[0-9]{1,6}\.[0-9]{1,6}\Z")
_MODALITIES = frozenset({"image", "pdf", "dicom", "audio", "waveform", "video"})
_OUTCOMES = frozenset(
    {"exact", "transform_required", "unsupported", "missing_capability"}
)
_REASONS = frozenset(
    {
        "capability_missing",
        "modality_unsupported",
        "feature_unsupported",
        "declared_transform_required",
    }
)


def _code(value: object) -> str:
    if not isinstance(value, str) or _CODE.fullmatch(value) is None:
        raise ValueError("capability_matrix_invalid_code")
    return value


def _bounded(values: Iterable, maximum: int) -> tuple:
    if isinstance(values, (str, bytes)):
        raise ValueError("capability_matrix_invalid_collection")
    try:
        items = tuple(islice(values, maximum + 1))
    except Exception:
        raise ValueError("capability_matrix_invalid_collection") from None
    if len(items) > maximum:
        raise ValueError("capability_matrix_collection_limit")
    return items


def _features(values: Iterable[str]) -> tuple[str, ...]:
    items = tuple(_code(value) for value in _bounded(values, 128))
    if len(set(items)) != len(items):
        raise ValueError("capability_matrix_duplicate_feature")
    return tuple(sorted(items))


def _schema(value: object) -> None:
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError("capability_matrix_schema_unsupported")


def _modality(value: object) -> str:
    if not isinstance(value, str) or value not in _MODALITIES:
        raise ValueError("capability_matrix_modality_unsupported")
    return value


@dataclass(frozen=True, slots=True)
class ModalityProfile:
    """Validated content-free requirements for a declared asset profile.

    Feature codes are application-defined metadata declarations, not results of
    reading, preflighting or clinically validating an asset. Callers must derive
    them from their validated profiles using the same vocabulary as providers.
    """

    profile_id: str
    modality: str
    required_features: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _code(self.profile_id)
        _modality(self.modality)
        object.__setattr__(self, "required_features", _features(self.required_features))


@dataclass(frozen=True, slots=True)
class ModalityCapability:
    """A modality's directly accepted or explicitly transformable feature codes.

    ``transformable_features`` means a provider declaration requires an external
    transformation; this matrix neither verifies nor performs that transformation.
    """

    modality: str
    exact_features: tuple[str, ...] = ()
    transformable_features: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _modality(self.modality)
        exact = _features(self.exact_features)
        transformable = _features(self.transformable_features)
        if set(exact) & set(transformable):
            raise ValueError("capability_matrix_ambiguous_feature")
        object.__setattr__(self, "exact_features", exact)
        object.__setattr__(self, "transformable_features", transformable)


@dataclass(frozen=True, slots=True)
class ProviderCapabilityEnvelope:
    """Versioned local provider declarations with no endpoints or model paths.

    ``capabilities=None`` means declaration evidence is missing. An empty tuple
    is an explicit declaration that no modality is supported. Neither implies
    that a provider has been installed, loaded or qualified on the current host.
    """

    provider_id: str
    capability_version: str
    capabilities: tuple[ModalityCapability, ...] | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _schema(self.schema_version)
        _code(self.provider_id)
        if not isinstance(self.capability_version, str) or not _VERSION.fullmatch(
            self.capability_version
        ):
            raise ValueError("capability_matrix_invalid_version")
        if self.capabilities is None:
            return
        items = _bounded(self.capabilities, len(_MODALITIES))
        if any(not isinstance(item, ModalityCapability) for item in items):
            raise ValueError("capability_matrix_invalid_capability")
        if len({item.modality for item in items}) != len(items):
            raise ValueError("capability_matrix_duplicate_modality")
        object.__setattr__(
            self, "capabilities", tuple(sorted(items, key=lambda item: item.modality))
        )


@dataclass(frozen=True, slots=True)
class CompatibilityEntry:
    """One immutable matrix cell limited to IDs, version, outcome and reasons."""

    provider_id: str
    capability_version: str
    profile_id: str
    outcome: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _code(self.provider_id)
        _code(self.profile_id)
        if not isinstance(self.capability_version, str) or not _VERSION.fullmatch(
            self.capability_version
        ):
            raise ValueError("capability_matrix_invalid_version")
        if not isinstance(self.outcome, str) or self.outcome not in _OUTCOMES:
            raise ValueError("capability_matrix_invalid_outcome")
        reasons = _bounded(self.reasons, 1)
        expected = {
            "exact": (),
            "transform_required": ("declared_transform_required",),
            "unsupported": None,
            "missing_capability": ("capability_missing",),
        }[self.outcome]
        if expected is None:
            valid = reasons in (("modality_unsupported",), ("feature_unsupported",))
        else:
            valid = reasons == expected
        if not valid or any(reason not in _REASONS for reason in reasons):
            raise ValueError("capability_matrix_invalid_reasons")
        object.__setattr__(self, "reasons", reasons)

    def to_dict(self) -> dict[str, object]:
        """Return only the five content-free cell fields."""
        return {
            "provider_id": self.provider_id,
            "capability_version": self.capability_version,
            "profile_id": self.profile_id,
            "outcome": self.outcome,
            "reasons": list(self.reasons),
        }


def capability_matrix(
    providers: Iterable[ProviderCapabilityEnvelope],
    profiles: Iterable[ModalityProfile],
) -> tuple[CompatibilityEntry, ...]:
    """Cross product bounded declarations without choosing or loading providers."""
    providers = _bounded(providers, 100)
    profiles = _bounded(profiles, 100)
    if any(
        not isinstance(provider, ProviderCapabilityEnvelope) for provider in providers
    ):
        raise ValueError("capability_matrix_invalid_provider")
    if any(not isinstance(profile, ModalityProfile) for profile in profiles):
        raise ValueError("capability_matrix_invalid_profile")
    if len({provider.provider_id for provider in providers}) != len(providers):
        raise ValueError("capability_matrix_duplicate_provider")
    if len({profile.profile_id for profile in profiles}) != len(profiles):
        raise ValueError("capability_matrix_duplicate_profile")
    result = []
    for provider in sorted(providers, key=lambda item: item.provider_id):
        declarations = (
            None
            if provider.capabilities is None
            else {item.modality: item for item in provider.capabilities}
        )
        for profile in sorted(profiles, key=lambda item: item.profile_id):
            if declarations is None:
                outcome, reasons = "missing_capability", ("capability_missing",)
            elif profile.modality not in declarations:
                outcome, reasons = "unsupported", ("modality_unsupported",)
            else:
                declared = declarations[profile.modality]
                requirements = set(profile.required_features)
                exact, transforms = (
                    set(declared.exact_features),
                    set(declared.transformable_features),
                )
                if requirements - (exact | transforms):
                    outcome, reasons = "unsupported", ("feature_unsupported",)
                elif requirements - exact:
                    outcome, reasons = (
                        "transform_required",
                        ("declared_transform_required",),
                    )
                else:
                    outcome, reasons = "exact", ()
            result.append(
                CompatibilityEntry(
                    provider.provider_id,
                    provider.capability_version,
                    profile.profile_id,
                    outcome,
                    reasons,
                )
            )
    return tuple(result)


def _entries(entries: Iterable[CompatibilityEntry]) -> tuple[CompatibilityEntry, ...]:
    items = _bounded(entries, 10000)
    if any(not isinstance(item, CompatibilityEntry) for item in items):
        raise ValueError("capability_matrix_invalid_entry")
    if len({(item.provider_id, item.profile_id) for item in items}) != len(items):
        raise ValueError("capability_matrix_duplicate_entry")
    return tuple(sorted(items, key=lambda item: (item.provider_id, item.profile_id)))


def render_capability_matrix_json(entries: Iterable[CompatibilityEntry]) -> str:
    """Serialize deterministic JSON containing only validated matrix cells."""
    return (
        json.dumps(
            [item.to_dict() for item in _entries(entries)],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def render_capability_matrix_markdown(entries: Iterable[CompatibilityEntry]) -> str:
    """Render a deterministic table containing no caller-supplied free text."""
    lines = [
        "| Provider ID | Capability version | Profile ID | Outcome | Reasons |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in _entries(entries):
        lines.append(
            f"| {item.provider_id} | {item.capability_version} | {item.profile_id} | {item.outcome} | {', '.join(item.reasons) or '-'} |"
        )
    return "\n".join(lines) + "\n"
