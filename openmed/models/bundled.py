"""Opt-in, registry-backed loading for a small offline model bundle.

The Python package ships the bundle's *metadata*, not an implicit download
path.  A deployment can place the verified model snapshot in the configured
OpenMed cache and call :func:`load_bundled_model`.  The loader resolves the
model by its ordinary registry key, pins the registry revision and license,
and keeps the complete load inside an offline socket guard.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from openmed.core.config import OpenMedConfig
from openmed.core.model_integrity import ModelIntegrityError
from openmed.core.model_registry import get_model_info
from openmed.core.models import ModelLoader
from openmed.core.offline import OfflineModeError, network_blocked_if_offline

BUNDLED_MODEL_SCHEMA_VERSION = "openmed.bundled-model.v1"
BUNDLED_MODEL_VERSION = "1.0.0"
BUNDLED_MODEL_KEY = "pii_detection"
BUNDLED_MODEL_ID = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
BUNDLED_MODEL_CHECKSUM = (
    "sha256:364fd803fc7830dc655619c8d345508202e3868b66f7357123bb713828eefc9e"
)
BUNDLED_MODEL_LICENSE = "apache-2.0"

_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class BundledModelError(RuntimeError):
    """Base error for invalid or unsafe bundled-model configuration."""


class BundledModelUnavailableError(BundledModelError):
    """Raised when the selected verified snapshot is not available locally."""


@dataclass(frozen=True)
class BundledModelManifest(Mapping[str, Any]):
    """Immutable metadata describing one opt-in bundled model.

    ``checksum`` is the registry's reproducibility digest for the exact model
    snapshot.  The model's artifact-side integrity manifest is still checked
    by the normal :class:`~openmed.core.models.ModelLoader` path.
    """

    schema_version: str
    version: str
    model_key: str
    model_id: str
    checksum: str
    license: str
    opt_in: bool = True
    offline: bool = True

    def __post_init__(self) -> None:
        """Reject metadata that cannot be safely used as a bundle pin."""
        if (
            type(self.schema_version) is not str
            or self.schema_version != BUNDLED_MODEL_SCHEMA_VERSION
        ):
            raise ValueError("unsupported bundled-model schema version")
        if type(self.version) is not str or _SEMVER_RE.fullmatch(self.version) is None:
            raise ValueError("bundled-model version must use MAJOR.MINOR.PATCH")
        for field_name in ("model_key", "model_id", "license"):
            if (
                type(getattr(self, field_name)) is not str
                or not getattr(self, field_name).strip()
            ):
                raise ValueError(f"bundled-model {field_name} must not be empty")
        if (
            type(self.checksum) is not str
            or _SHA256_RE.fullmatch(self.checksum) is None
        ):
            raise ValueError("bundled-model checksum must be a sha256: digest")
        if self.opt_in is not True:
            raise ValueError("bundled models must remain opt-in")
        if self.offline is not True:
            raise ValueError("bundled models must be offline-only")

    @property
    def registry_key(self) -> str:
        """Return the normal model-registry key for this bundle."""
        return self.model_key

    @property
    def sha256(self) -> str:
        """Return the checksum using the registry's conventional name."""
        return self.checksum

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible manifest payload."""
        return {
            "checksum": self.checksum,
            "license": self.license,
            "model_id": self.model_id,
            "model_key": self.model_key,
            "offline": self.offline,
            "opt_in": self.opt_in,
            "schema_version": self.schema_version,
            "version": self.version,
        }

    def __getitem__(self, key: str) -> Any:
        """Allow the immutable manifest to be consumed like a mapping."""
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over stable manifest field names."""
        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of serialized manifest fields."""
        return len(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "BundledModelManifest":
        """Build a validated manifest from a JSON-like mapping."""
        if not isinstance(payload, Mapping):
            raise TypeError("bundled-model manifest must be a mapping")
        model_key = payload.get("model_key", payload.get("registry_key"))
        checksum = payload.get("checksum", payload.get("sha256"))
        required = {
            "schema_version": payload.get("schema_version"),
            "version": payload.get("version"),
            "model_key": model_key,
            "model_id": payload.get("model_id"),
            "checksum": checksum,
            "license": payload.get("license"),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "bundled-model manifest is missing: " + ", ".join(sorted(missing))
            )
        return cls(
            **required,
            opt_in=payload.get("opt_in", True),
            offline=payload.get("offline", True),
        )


BUNDLED_MODEL_MANIFEST = BundledModelManifest(
    schema_version=BUNDLED_MODEL_SCHEMA_VERSION,
    version=BUNDLED_MODEL_VERSION,
    model_key=BUNDLED_MODEL_KEY,
    model_id=BUNDLED_MODEL_ID,
    checksum=BUNDLED_MODEL_CHECKSUM,
    license=BUNDLED_MODEL_LICENSE,
)


def _coerce_manifest(
    manifest: BundledModelManifest | Mapping[str, Any],
) -> BundledModelManifest:
    """Return a validated immutable manifest instance."""
    if isinstance(manifest, BundledModelManifest):
        return manifest
    return BundledModelManifest.from_mapping(manifest)


def validate_bundled_model_manifest(
    manifest: BundledModelManifest | Mapping[str, Any] = BUNDLED_MODEL_MANIFEST,
) -> BundledModelManifest:
    """Validate bundle metadata against the ordinary model registry.

    The check is local and deterministic.  In particular, it does not refresh
    registry data or ask the model hub for a newer revision.
    """
    candidate = _coerce_manifest(manifest)
    registry_info = get_model_info(candidate.model_key)
    if registry_info is None:
        raise BundledModelError("bundled model key is not in the model registry")
    if registry_info.model_id != candidate.model_id:
        raise BundledModelError(
            "bundled model key resolves to an unexpected registry model"
        )
    if registry_info.reproducibility_hash != candidate.checksum:
        raise BundledModelError("bundled model has a stale registry checksum")
    if (registry_info.license or "").lower() != candidate.license.lower():
        raise BundledModelError("bundled model has mismatched license metadata")
    return candidate


def get_bundled_model_manifest(
    model_key: str = BUNDLED_MODEL_KEY,
) -> BundledModelManifest:
    """Return the explicitly supported bundled manifest for ``model_key``.

    Only the one manifest shipped by this module is considered bundled.  A
    registry entry that is merely available remotely is never promoted into a
    bundled model implicitly.
    """
    if not isinstance(model_key, str) or not model_key.strip():
        raise ValueError("model_key must be a non-empty string")
    manifest = BUNDLED_MODEL_MANIFEST
    if model_key not in {manifest.model_key, manifest.model_id}:
        registry_info = get_model_info(model_key)
        if registry_info is None or registry_info.model_id != manifest.model_id:
            raise KeyError("no bundled model is registered for the requested key")
    return validate_bundled_model_manifest(manifest)


def list_bundled_model_manifests() -> tuple[BundledModelManifest, ...]:
    """Return all bundled manifests in deterministic order."""
    return (get_bundled_model_manifest(),)


def _offline_config(config: OpenMedConfig | None) -> OpenMedConfig:
    """Return a config copy that cannot disable local-only loading."""
    if config is None:
        return OpenMedConfig(local_only=True)
    if getattr(config, "local_only", False):
        return config
    try:
        return replace(config, local_only=True)
    except TypeError:
        copied = copy.copy(config)
        copied.local_only = True
        return copied


def load_bundled_model(
    model_key: str = BUNDLED_MODEL_KEY,
    *,
    config: OpenMedConfig | None = None,
    loader: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load the opt-in bundled model through :class:`ModelLoader`.

    The registry key is resolved normally, while ``local_files_only=True`` and
    a process-local socket guard are enforced for the entire load.  A missing
    or invalid local snapshot therefore fails closed; this function never
    performs a first-run download or silently falls back to a remote model.

    Args:
        model_key: Bundled registry key (or its exact registry model id).
        config: Optional OpenMed configuration.  ``local_only`` is forced on a
            copy when the supplied config has it disabled.
        loader: Optional ModelLoader-compatible object, useful for embedding or
            tests.  Its ``load_model`` method receives the ordinary registry
            key and ``local_files_only=True``.
        **kwargs: Other arguments accepted by ``ModelLoader.load_model``.

    Returns:
        The normal model dictionary returned by ``ModelLoader.load_model``.

    Raises:
        BundledModelError: If the manifest is invalid or a caller attempts to
            disable local-only loading.
    """
    manifest = get_bundled_model_manifest(model_key)
    if kwargs.get("local_files_only") is False:
        raise BundledModelError(
            "bundled model loading requires local_files_only=True; "
            "network fallback is disabled"
        )
    if kwargs.get("require_integrity") is False:
        raise BundledModelError(
            "bundled model loading requires verified artifact integrity"
        )

    offline_config = _offline_config(config)
    model_loader = loader if loader is not None else ModelLoader(offline_config)
    load_kwargs = dict(kwargs)
    load_kwargs["local_files_only"] = True
    load_kwargs["require_integrity"] = True

    with network_blocked_if_offline(offline_config, local_only=True):
        try:
            return model_loader.load_model(manifest.registry_key, **load_kwargs)
        except OfflineModeError:
            raise
        except (BundledModelError, ModelIntegrityError):
            raise BundledModelUnavailableError(
                "the bundled model could not be verified locally; "
                "network fallback is disabled"
            ) from None
        except Exception:
            raise BundledModelUnavailableError(
                "the bundled model could not be loaded locally; "
                "network fallback is disabled"
            ) from None


__all__ = [
    "BUNDLED_MODEL_CHECKSUM",
    "BUNDLED_MODEL_ID",
    "BUNDLED_MODEL_KEY",
    "BUNDLED_MODEL_LICENSE",
    "BUNDLED_MODEL_MANIFEST",
    "BUNDLED_MODEL_SCHEMA_VERSION",
    "BUNDLED_MODEL_VERSION",
    "BundledModelError",
    "BundledModelManifest",
    "BundledModelUnavailableError",
    "get_bundled_model_manifest",
    "list_bundled_model_manifests",
    "load_bundled_model",
    "validate_bundled_model_manifest",
]
