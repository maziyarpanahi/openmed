"""Offline provenance qualification for explicitly synthetic image/text pairs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openmed.training.synthetic.spdx_identifier import normalize_spdx_identifier

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9]{1,5}\.[0-9]{1,5}\.[0-9]{1,5}\Z")


def _digest(value: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("provenance references must be SHA-256 digests")


@dataclass(frozen=True)
class PairTransformation:
    """A reproducible transformation link, containing digests instead of prompts."""

    input_digest: str
    output_digest: str
    tool_digest: str
    configuration_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.input_digest,
            self.output_digest,
            self.tool_digest,
            self.configuration_digest,
        ):
            _digest(value)

    def to_dict(self) -> dict[str, str]:
        """Return a canonical content-free transformation link."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PairAssetProvenance:
    """A permissive source declaration and complete source-to-artifact chain."""

    source_digest: str
    artifact_digest: str
    license_identifier: str
    transformations: tuple[PairTransformation, ...] = ()

    def __post_init__(self) -> None:
        _digest(self.source_digest)
        _digest(self.artifact_digest)
        license_record = normalize_spdx_identifier(self.license_identifier)
        if not license_record.is_permissive:
            raise ValueError("pair source requires a permitted SPDX declaration")
        object.__setattr__(self, "license_identifier", license_record.normalized)
        if (
            not isinstance(self.transformations, Sequence)
            or len(self.transformations) > 32
            or not all(
                isinstance(link, PairTransformation) for link in self.transformations
            )
        ):
            raise ValueError("provide a bounded validated transformation chain")
        object.__setattr__(self, "transformations", tuple(self.transformations))
        current = self.source_digest
        for link in self.transformations:
            if link.input_digest != current:
                raise ValueError("pair transformation chain is disconnected")
            current = link.output_digest
        if current != self.artifact_digest:
            raise ValueError("pair transformation chain does not reach its artifact")

    def to_dict(self) -> dict[str, object]:
        """Serialize declarations and reproducibility references only."""
        return {
            "source_digest": self.source_digest,
            "artifact_digest": self.artifact_digest,
            "license_identifier": self.license_identifier,
            "transformations": [link.to_dict() for link in self.transformations],
        }


@dataclass(frozen=True)
class SyntheticPairManifest:
    """A linked image, text and labels record with explicit validation outcome."""

    image: PairAssetProvenance
    text: PairAssetProvenance
    labels: PairAssetProvenance
    generator_digest: str
    generator_version: str
    validation_digest: str
    validation_passed: bool
    synthetic: bool = True

    def __post_init__(self) -> None:
        if not all(
            isinstance(asset, PairAssetProvenance)
            for asset in (self.image, self.text, self.labels)
        ):
            raise ValueError("image, text and labels require complete provenance")
        for value in (self.generator_digest, self.validation_digest):
            _digest(value)
        if not isinstance(self.generator_version, str) or not _VERSION.fullmatch(
            self.generator_version
        ):
            raise ValueError(
                "generator version must be a bounded numeric semantic version"
            )
        if self.synthetic is not True or not isinstance(self.validation_passed, bool):
            raise ValueError(
                "pair requires explicit synthetic and validation declarations"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic manifest without source images, text or labels."""
        return {
            "schema_version": 1,
            "synthetic": True,
            "assets": {
                "image": self.image.to_dict(),
                "text": self.text.to_dict(),
                "labels": self.labels.to_dict(),
            },
            "generator_digest": self.generator_digest,
            "generator_version": self.generator_version,
            "validation_digest": self.validation_digest,
            "validation_passed": self.validation_passed,
        }

    def to_json(self) -> str:
        """Return byte-stable canonical JSON for offline provenance storage."""
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def manifest_digest(self) -> str:
        """Return a deterministic integrity digest, not a signature or truth proof."""
        return hashlib.sha256(self.to_json().encode("ascii")).hexdigest()

    def require_publishable(self) -> None:
        """Refuse unvalidated pairs; passing does not authorize publication."""
        if not self.validation_passed:
            raise ValueError("synthetic pair validation did not pass")


def verify_pair_artifacts(
    manifest: SyntheticPairManifest, artifacts: Mapping[str, bytes]
) -> None:
    """Offline-check all linked artifact bytes before accepting the manifest."""
    if not isinstance(manifest, SyntheticPairManifest) or set(artifacts) != {
        "image",
        "text",
        "labels",
    }:
        raise ValueError("provide exactly the three linked pair artifacts")
    manifest.require_publishable()
    for role in ("image", "text", "labels"):
        payload = artifacts[role]
        if not isinstance(payload, bytes) or len(payload) > 10_000_000:
            raise ValueError("pair artifact must be bounded bytes")
        if (
            hashlib.sha256(payload).hexdigest()
            != getattr(manifest, role).artifact_digest
        ):
            raise ValueError("pair artifact digest mismatch")
