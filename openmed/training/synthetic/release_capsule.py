"""Canonical synthetic release capsules with offline artifact verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from openmed.training.synthetic.spdx_identifier import normalize_spdx_identifier

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_VERSION = re.compile(r"[0-9]{1,5}\.[0-9]{1,5}\.[0-9]{1,5}\Z")


def _digest(value: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("capsule reference must be a SHA-256 digest")


@dataclass(frozen=True)
class CapsuleGenerator:
    """Actual bounded recipe configuration plus seed/provenance commitments."""

    code: str
    version: str
    configuration: Mapping[str, object]
    seed_commitment: str
    provenance_digest: str
    external_model_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.code, str)
            or not _CODE.fullmatch(self.code)
            or not isinstance(self.version, str)
            or not _VERSION.fullmatch(self.version)
        ):
            raise ValueError("invalid controlled generator identity")
        for value in (self.seed_commitment, self.provenance_digest):
            _digest(value)
        if self.external_model_digest is not None:
            _digest(self.external_model_digest)
        if not isinstance(self.configuration, Mapping) or len(self.configuration) > 64:
            raise ValueError("generator configuration must be a bounded mapping")
        for code, value in self.configuration.items():
            if not isinstance(code, str) or not _CODE.fullmatch(code):
                raise ValueError("invalid generator configuration key")
            if isinstance(value, str):
                if not _CODE.fullmatch(value):
                    raise ValueError("configuration strings must be controlled codes")
            elif isinstance(value, bool):
                pass
            elif isinstance(value, int):
                if abs(value) > 2**53:
                    raise ValueError("configuration integer exceeds canonical bound")
            elif isinstance(value, float):
                if not math.isfinite(value):
                    raise ValueError("configuration numbers must be finite")
            else:
                raise ValueError(
                    "configuration values must be controlled scalar options"
                )
        object.__setattr__(
            self, "configuration", MappingProxyType(dict(self.configuration))
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize the controlled recipe without prompts or private paths."""
        return {
            "code": self.code,
            "version": self.version,
            "configuration": dict(sorted(self.configuration.items())),
            "seed_commitment": self.seed_commitment,
            "provenance_digest": self.provenance_digest,
            "external_model_digest": self.external_model_digest,
        }


@dataclass(frozen=True)
class CapsuleCheck:
    """A declared privacy/utility outcome and evidence commitment."""

    status: str
    checked_count: int
    evidence_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, str)
            or self.status not in {"pass", "fail", "not_evaluated"}
            or isinstance(self.checked_count, bool)
            or not isinstance(self.checked_count, int)
            or not 0 <= self.checked_count <= 1_000_000_000
        ):
            raise ValueError("invalid capsule check outcome")
        _digest(self.evidence_digest)
        if self.status == "pass" and not self.checked_count:
            raise ValueError("passing check requires evaluated examples")

    def to_dict(self) -> dict[str, object]:
        """Return controlled outcome metadata only."""
        return {
            "status": self.status,
            "checked_count": self.checked_count,
            "evidence_digest": self.evidence_digest,
        }


def build_release_capsule(
    generators: Sequence[CapsuleGenerator],
    artifacts: Mapping[str, bytes],
    *,
    licenses: Mapping[str, str],
    privacy: CapsuleCheck,
    utility: CapsuleCheck,
) -> bytes:
    """Bundle complete recipe, lineage, check summaries, licenses and artifact hashes."""
    if (
        not 1 <= len(generators) <= 64
        or not all(isinstance(item, CapsuleGenerator) for item in generators)
        or len({item.code for item in generators}) != len(generators)
    ):
        raise ValueError("provide distinct bounded validated generators")
    if not isinstance(privacy, CapsuleCheck) or not isinstance(utility, CapsuleCheck):
        raise ValueError("provide validated privacy and utility outcomes")
    if (
        not isinstance(artifacts, Mapping)
        or not 1 <= len(artifacts) <= 128
        or set(licenses) != set(artifacts)
    ):
        raise ValueError("every bounded artifact requires its license declaration")
    asset_rows = {}
    for code, payload in sorted(artifacts.items()):
        if (
            not isinstance(code, str)
            or not _CODE.fullmatch(code)
            or not isinstance(payload, bytes)
            or len(payload) > 20_000_000
        ):
            raise ValueError("artifact must have a controlled name and bounded bytes")
        license_record = normalize_spdx_identifier(licenses[code])
        if not license_record.is_permissive:
            raise ValueError("artifact requires a permitted SPDX declaration")
        asset_rows[code] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
            "license": license_record.normalized,
        }
    manifest = {
        "schema_version": 1,
        "synthetic": True,
        "generators": [
            item.to_dict() for item in sorted(generators, key=lambda item: item.code)
        ],
        "privacy": privacy.to_dict(),
        "utility": utility.to_dict(),
        "artifacts": asset_rows,
    }
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    wrapper = {
        "manifest": manifest,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    return json.dumps(
        wrapper, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def verify_release_capsule(
    payload: bytes, artifacts: Mapping[str, bytes]
) -> dict[str, object]:
    """Verify capsule and artifact commitments offline, without claiming regeneration."""
    if not isinstance(payload, bytes) or len(payload) > 1_000_000:
        raise ValueError("capsule must be bounded bytes")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate capsule field")
            result[key] = value
        return result

    try:
        wrapper = json.loads(payload, object_pairs_hook=unique_object)
        if set(wrapper) != {"manifest", "manifest_sha256"}:
            raise ValueError
        manifest = wrapper["manifest"]
        if (
            set(manifest)
            != {
                "schema_version",
                "synthetic",
                "generators",
                "privacy",
                "utility",
                "artifacts",
            }
            or manifest["schema_version"] != 1
            or manifest["synthetic"] is not True
        ):
            raise ValueError
        generators = [CapsuleGenerator(**row) for row in manifest["generators"]]
        privacy, utility = (
            CapsuleCheck(**manifest["privacy"]),
            CapsuleCheck(**manifest["utility"]),
        )
        rows = manifest["artifacts"]
        for row in rows.values():
            if set(row) != {"sha256", "byte_size", "license"}:
                raise ValueError
        regenerated = build_release_capsule(
            generators,
            artifacts,
            licenses={code: row["license"] for code, row in rows.items()},
            privacy=privacy,
            utility=utility,
        )
    except Exception:
        raise ValueError("capsule structure or artifact verification failed") from None
    if regenerated != payload:
        raise ValueError("capsule commitment or artifact digest mismatch")
    external = any(item.external_model_digest is not None for item in generators)
    return {
        "integrity_verified": True,
        "artifact_count": len(artifacts),
        "checks_passed": privacy.status == "pass" and utility.status == "pass",
        "reproducibility": "external_model_required"
        if external
        else "recipe_dependencies_declared",
        "generation_reproduced": False,
    }
