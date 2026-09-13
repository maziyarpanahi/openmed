"""Privacy-safe multimodal preflight report.

One pre-decode entry point that runs the committed preflight contracts -- the
asset manifest, bounded media-type detection, the modality manifest profiles,
the pre-decode limit profiles, and the bounded streaming digest -- in a fixed
order and folds the results into a single accept-or-abstain report. Image, PDF, DICOM, waveform, and audio providers can call this helper to
share the same checks and findings. Existing providers are not automatically
routed through it.

The report carries deterministic findings, reason codes, schema versions,
numeric metadata, and digests only. It never opens a decoder, never keeps the
source prefix or content, and never reflects a malformed manifest's values.
Missing evidence never becomes acceptance: a check that cannot be evaluated
abstains.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, BinaryIO, Final

from .abstention import AbstentionReason, AbstentionRecord, AbstentionStage
from .asset_limits import (
    DESKTOP_V1,
    AssetLimitError,
    LimitFinding,
    LimitProfile,
    evaluate_asset_limits,
)
from .asset_manifest import AssetManifest, AssetManifestError
from .digest import (
    AssetDigest,
    DigestLimitExceededError,
    DigestStreamError,
    digest_asset,
)
from .manifest_profiles import (
    AUDIO_V1,
    DICOM_V1,
    IMAGE_V1,
    PDF_V1,
    ManifestProfile,
    ManifestProfileError,
    ValidationFinding,
    validate_manifest_metadata,
)
from .media_type import MAX_MEDIA_TYPE_PREFIX_BYTES, MediaTypeStatus, detect_media_type

__all__ = [
    "PREFLIGHT_CHECKS",
    "PREFLIGHT_SCHEMA_VERSION",
    "PreflightError",
    "PreflightFinding",
    "PreflightReport",
    "PreflightStatus",
    "preflight_asset",
]

PREFLIGHT_SCHEMA_VERSION: Final = 1

# The documented, deterministic order of checks and therefore of findings.
PREFLIGHT_CHECKS: Final = ("manifest", "media_type", "metadata", "limits", "digest")
_CHECK_ORDER: Final = {name: index for index, name in enumerate(PREFLIGHT_CHECKS)}

# Findings with a fixed shape: (check, reason_code) -> (field_name, carries
# numbers). Metadata findings other than ``unsupported_modality`` are validated
# by the manifest-profile contract and limit findings by the limit contract.
_FIXED_FINDINGS: Final = {
    ("manifest", "malformed_manifest"): (None, False),
    ("media_type", "mismatch"): (None, False),
    ("media_type", "unknown"): (None, False),
    ("metadata", "unsupported_modality"): (None, False),
    ("digest", "byte_count_mismatch"): ("byte_size", True),
    ("digest", "sha256_mismatch"): ("sha256", False),
    ("digest", "not_evaluated"): (None, False),
}

_MODALITIES: Final = frozenset({"image", "pdf", "dicom", "audio"})
_DETECTED_MEDIA_TYPES: Final = frozenset(
    {
        "application/pdf",
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/dicom",
        "audio/wav",
    }
)
_METADATA_PROFILES: Final = {
    "image": IMAGE_V1,
    "pdf": PDF_V1,
    "dicom": DICOM_V1,
    "audio": AUDIO_V1,
}


class PreflightError(ValueError):
    """Raised for invalid preflight inputs; messages are stable categories."""


class PreflightStatus(str, Enum):
    """Terminal decision of a preflight report."""

    ACCEPT = "accept"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class PreflightFinding:
    """One deterministic, content-free finding from a preflight check.

    ``check`` names the check that produced the finding and ``reason_code`` is
    allowlisted for that check. ``field_name``, ``limit``, and ``observed`` are
    populated only where the underlying contract defines them: metadata
    findings name the manifest field, limit findings carry the ceiling and the
    observed number, and a byte-count mismatch carries the declared size and
    the number of bytes hashed (``None`` when reading stopped at the declared
    size). Nothing else crosses this boundary.
    """

    check: str
    reason_code: str
    field_name: str | None = None
    limit: int | float | None = None
    observed: int | float | None = None

    def __post_init__(self) -> None:
        if type(self.check) is not str or self.check not in _CHECK_ORDER:
            raise PreflightError("finding check is unsupported")
        if type(self.reason_code) is not str:
            raise PreflightError("finding reason_code is unsupported")
        key = (self.check, self.reason_code)
        if self.check == "limits":
            try:
                LimitFinding(
                    self.field_name, self.reason_code, self.limit, self.observed
                )
            except (AssetLimitError, TypeError):
                raise PreflightError("finding is not a valid limit finding") from None
            return
        if self.check == "metadata" and key not in _FIXED_FINDINGS:
            try:
                ValidationFinding(self.field_name, self.reason_code)
            except ManifestProfileError:
                raise PreflightError(
                    "finding is not a valid metadata finding"
                ) from None
            if self.limit is not None or self.observed is not None:
                raise PreflightError("finding carries values its check does not define")
            return
        if key not in _FIXED_FINDINGS:
            raise PreflightError("finding reason_code is unsupported")
        field_name, carries_numbers = _FIXED_FINDINGS[key]
        if self.field_name != field_name:
            raise PreflightError("finding field_name is unsupported")
        if not carries_numbers:
            if self.limit is not None or self.observed is not None:
                raise PreflightError("finding carries values its check does not define")
            return
        if type(self.limit) is not int or self.limit <= 0:
            raise PreflightError("finding limit must be a positive integer")
        if self.observed is not None and (
            type(self.observed) is not int or self.observed < 0
        ):
            raise PreflightError("finding observed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as a JSON-ready mapping with a fixed key order."""
        return {
            "check": self.check,
            "reason_code": self.reason_code,
            "field_name": self.field_name,
            "limit": self.limit,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """The accept-or-abstain result of running every preflight check.

    The report is composed only of contracts that already enforce the privacy
    boundary independently: :class:`AssetManifest`, :class:`ManifestProfile`,
    :class:`LimitProfile`, :class:`AssetDigest`, :class:`AbstentionRecord`,
    and :class:`PreflightFinding`. A malformed manifest leaves every optional
    field unset rather than reflecting any submitted value.
    """

    status: PreflightStatus
    findings: tuple[PreflightFinding, ...]
    limit_profile: LimitProfile
    manifest: AssetManifest | None = None
    modality: str | None = None
    metadata_profile: ManifestProfile | None = None
    detected_media_type: str | None = None
    media_type_status: MediaTypeStatus | None = None
    digest: AssetDigest | None = None
    abstention: AbstentionRecord | None = None
    schema_version: int = PREFLIGHT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        status = _status(self.status)
        if (
            type(self.schema_version) is not int
            or self.schema_version != PREFLIGHT_SCHEMA_VERSION
        ):
            raise PreflightError("schema_version is unsupported")
        findings = _findings(self.findings)
        if any(not isinstance(finding, PreflightFinding) for finding in findings):
            raise PreflightError("findings must be PreflightFinding values")
        positions = [_CHECK_ORDER[finding.check] for finding in findings]
        if positions != sorted(positions):
            raise PreflightError("findings are not in the documented check order")
        if not isinstance(self.limit_profile, LimitProfile):
            raise PreflightError("limit_profile must be a LimitProfile")
        _require_optional(self.manifest, AssetManifest, "manifest")
        if self.modality is not None and (
            type(self.modality) is not str or self.modality not in _MODALITIES
        ):
            raise PreflightError("modality is unsupported")
        _require_optional(self.metadata_profile, ManifestProfile, "metadata_profile")
        if self.detected_media_type is not None and (
            type(self.detected_media_type) is not str
            or self.detected_media_type not in _DETECTED_MEDIA_TYPES
        ):
            raise PreflightError("detected_media_type must be a string or None")
        _require_optional(self.media_type_status, MediaTypeStatus, "media_type_status")
        _require_optional(self.digest, AssetDigest, "digest")
        _require_optional(self.abstention, AbstentionRecord, "abstention")
        if (
            self.abstention is not None
            and self.abstention.stage is not AbstentionStage.PREFLIGHT
        ):
            raise PreflightError("abstention stage must be preflight")
        if status is PreflightStatus.ACCEPT:
            if (
                findings
                or self.abstention is not None
                or self.manifest is None
                or self.digest is None
                or self.media_type_status is not MediaTypeStatus.MATCH
            ):
                raise PreflightError("an accepted report must have passed every check")
            expected_modality = _modality_for(self.manifest.media_type)
            expected_profile = _METADATA_PROFILES.get(expected_modality)
            if (
                expected_profile is None
                or self.modality != expected_modality
                or self.metadata_profile != expected_profile
                or self.detected_media_type != self.manifest.media_type
                or self.digest.sha256 != self.manifest.sha256
                or self.digest.byte_count != self.manifest.byte_size
                or validate_manifest_metadata(expected_profile, self.manifest)
                or evaluate_asset_limits(
                    self.limit_profile, self.manifest, expected_modality
                )
            ):
                raise PreflightError("an accepted report must have passed every check")
        elif not findings or self.abstention is None:
            raise PreflightError("an abstaining report must explain itself")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "findings", findings)

    def to_dict(self) -> dict[str, Any]:
        """Return the report as a JSON-ready mapping with a fixed key order."""
        media_type = None
        if self.media_type_status is not None:
            media_type = {
                "detected": self.detected_media_type,
                "status": self.media_type_status.value,
            }
        metadata_profile = None
        if self.metadata_profile is not None:
            metadata_profile = {
                "modality": self.metadata_profile.modality,
                "version": self.metadata_profile.version,
            }
        digest = None
        if self.digest is not None:
            digest = {
                "sha256": self.digest.sha256,
                "byte_count": self.digest.byte_count,
            }
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "abstention": None
            if self.abstention is None
            else self.abstention.to_dict(),
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
            "modality": self.modality,
            "metadata_profile": metadata_profile,
            "limit_profile": {
                "name": self.limit_profile.name,
                "version": self.limit_profile.version,
            },
            "media_type": media_type,
            "digest": digest,
            "findings": [finding.to_dict() for finding in self.findings],
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(self.to_dict(), ensure_ascii=True, separators=(",", ":"))


def preflight_asset(
    manifest: Mapping[str, Any] | AssetManifest,
    source: bytes | bytearray | memoryview | BinaryIO,
    *,
    limit_profile: LimitProfile = DESKTOP_V1,
) -> PreflightReport:
    """Run every preflight check over a manifest and its source bytes.

    Checks run in the documented order: the manifest is validated; the
    declared media type is compared with the type detected from a bounded
    prefix of ``source``; the modality's manifest profile validates the
    metadata fields; the limit profile evaluates the resource ceilings; and
    the source is hashed, reading at most one byte past the declared size, to
    confirm the declared digest. The digest pass is skipped, and reported as
    unevaluated, unless the byte-size ceiling was evaluated and passed, so reads
    are bounded by the smaller declared/profile byte ceiling plus one probe byte.

    A malformed manifest ends preflight before the source is touched. Streams
    are read from their current position; seekable streams are restored on
    success or failure and caller-owned streams are never closed. Bytes-like
    values are hashed in memory. Source failures raise :class:`PreflightError`
    with a stable category and no underlying detail.
    """
    if not isinstance(limit_profile, LimitProfile):
        raise TypeError("limit_profile must be a LimitProfile")
    if not isinstance(manifest, (AssetManifest, Mapping)):
        raise TypeError("manifest must be a mapping or AssetManifest")
    if not isinstance(source, (bytes, bytearray, memoryview)) and not callable(
        getattr(source, "read", None)
    ):
        raise TypeError("source must be bytes-like or a binary stream")

    try:
        validated = (
            manifest
            if isinstance(manifest, AssetManifest)
            else AssetManifest.from_dict(manifest)
        )
    except AssetManifestError:
        return _abstain(
            [PreflightFinding("manifest", "malformed_manifest")], limit_profile
        )

    reader = _Source(
        source,
        prefix_limit=min(
            MAX_MEDIA_TYPE_PREFIX_BYTES,
            validated.byte_size + 1,
            limit_profile.max_byte_size + 1,
        ),
    )
    try:
        return _run_checks(validated, reader, limit_profile)
    finally:
        reader.restore()


def _run_checks(
    manifest: AssetManifest, reader: _Source, limit_profile: LimitProfile
) -> PreflightReport:
    findings: list[PreflightFinding] = []

    detected = detect_media_type(reader.prefix())
    if detected is None:
        media_type_status = MediaTypeStatus.UNKNOWN
        findings.append(PreflightFinding("media_type", "unknown"))
    elif detected == manifest.media_type:
        media_type_status = MediaTypeStatus.MATCH
    else:
        media_type_status = MediaTypeStatus.MISMATCH
        findings.append(PreflightFinding("media_type", "mismatch"))

    modality = _modality_for(manifest.media_type)
    metadata_profile = None
    byte_size_passed = False
    if modality is None:
        findings.append(PreflightFinding("metadata", "unsupported_modality"))
    else:
        metadata_profile = _METADATA_PROFILES[modality]
        findings.extend(
            PreflightFinding("metadata", finding.reason_code, finding.field_name)
            for finding in validate_manifest_metadata(metadata_profile, manifest)
        )
        limit_findings = evaluate_asset_limits(limit_profile, manifest, modality)
        findings.extend(
            PreflightFinding(
                "limits",
                finding.reason_code,
                finding.field_name,
                finding.limit,
                finding.observed,
            )
            for finding in limit_findings
        )
        byte_size_passed = all(
            finding.field_name != "byte_size" for finding in limit_findings
        )

    digest = None
    if not byte_size_passed:
        findings.append(PreflightFinding("digest", "not_evaluated"))
    else:
        try:
            digest = reader.digest(max_bytes=manifest.byte_size)
        except DigestLimitExceededError:
            findings.append(
                PreflightFinding(
                    "digest", "byte_count_mismatch", "byte_size", manifest.byte_size
                )
            )
        else:
            if digest.byte_count != manifest.byte_size:
                findings.append(
                    PreflightFinding(
                        "digest",
                        "byte_count_mismatch",
                        "byte_size",
                        manifest.byte_size,
                        digest.byte_count,
                    )
                )
            elif digest.sha256 != manifest.sha256:
                findings.append(PreflightFinding("digest", "sha256_mismatch", "sha256"))

    if findings:
        return _abstain(
            findings,
            limit_profile,
            manifest=manifest,
            modality=modality,
            metadata_profile=metadata_profile,
            detected_media_type=detected,
            media_type_status=media_type_status,
            digest=digest,
        )
    return PreflightReport(
        status=PreflightStatus.ACCEPT,
        findings=(),
        limit_profile=limit_profile,
        manifest=manifest,
        modality=modality,
        metadata_profile=metadata_profile,
        detected_media_type=detected,
        media_type_status=media_type_status,
        digest=digest,
    )


def _abstain(
    findings: list[PreflightFinding], limit_profile: LimitProfile, **fields: Any
) -> PreflightReport:
    # The coarse reason follows the earliest failing check: the limit check
    # abstains for a resource limit, every other check for unsupported media.
    reason = (
        AbstentionReason.RESOURCE_LIMIT
        if findings[0].check == "limits"
        else AbstentionReason.UNSUPPORTED_MEDIA
    )
    return PreflightReport(
        status=PreflightStatus.ABSTAIN,
        findings=tuple(findings),
        limit_profile=limit_profile,
        abstention=AbstentionRecord(stage=AbstentionStage.PREFLIGHT, reason=reason),
        **fields,
    )


def _modality_for(media_type: str) -> str | None:
    if media_type == "application/pdf":
        return "pdf"
    if media_type == "application/dicom":
        return "dicom"
    if media_type.startswith("image/"):
        return "image"
    if media_type.startswith("audio/"):
        return "audio"
    # application/dicom+json is a valid manifest media type with no metadata
    # or limit profile, so nothing beyond the manifest can be evaluated.
    return None


def _require_optional(value: Any, expected: type, name: str) -> None:
    if value is not None and not isinstance(value, expected):
        raise PreflightError(f"{name} has an unsupported type")


class _Source:
    """Bounded access to a bytes value or binary stream.

    Offers exactly two reads -- a media-type prefix and one digest pass -- and
    restores a seekable stream's position afterwards. Failures surface as
    :class:`PreflightError` categories with no underlying detail attached.
    """

    __slots__ = ("_data", "_position", "_prefix", "_prefix_limit", "_stream")

    def __init__(self, source: Any, *, prefix_limit: int) -> None:
        self._prefix: bytes | None = None
        self._prefix_limit = prefix_limit
        if isinstance(source, (bytes, bytearray, memoryview)):
            try:
                data = memoryview(source).cast("B")
            except (TypeError, ValueError):
                pass
            else:
                self._data: memoryview | None = data
                self._stream = None
                self._position: int | None = None
                return
            raise PreflightError("preflight_source_contract_error")
        if not callable(getattr(source, "read", None)):
            raise TypeError("source must be bytes-like or a binary stream")
        self._data = None
        self._stream = source
        self._position = _stream_position(source)

    def prefix(self) -> bytes:
        if self._prefix is None:
            if self._data is not None:
                self._prefix = bytes(self._data[: self._prefix_limit])
            else:
                self._prefix = _read_prefix(self._stream, self._prefix_limit)
        return self._prefix

    def digest(self, *, max_bytes: int) -> AssetDigest:
        if self._data is not None:
            size = self._data.nbytes
            if size > max_bytes:
                raise DigestLimitExceededError(maximum_bytes=max_bytes, bytes_read=size)
            return AssetDigest(hashlib.sha256(self._data).hexdigest(), size)
        try:
            digest = digest_asset(
                _PrefixedStream(self.prefix(), self._stream), max_bytes=max_bytes
            )
        except DigestStreamError:
            pass
        else:
            return digest
        raise PreflightError("preflight_source_read_error")

    def restore(self) -> None:
        if self._position is None:
            return
        try:
            self._stream.seek(self._position)
        except Exception:
            pass
        else:
            return
        raise PreflightError("preflight_source_restore_error")


class _PrefixedStream:
    """Replays an already-read prefix ahead of the remaining stream bytes."""

    __slots__ = ("_prefix", "_stream")

    def __init__(self, prefix: bytes, stream: Any) -> None:
        self._prefix = prefix
        self._stream = stream

    def read(self, size: int) -> bytes:
        if self._prefix:
            chunk, self._prefix = self._prefix[:size], self._prefix[size:]
            return chunk
        return self._stream.read(size)


def _stream_position(stream: Any) -> int | None:
    seekable = getattr(stream, "seekable", None)
    if not callable(seekable):
        return None
    try:
        if not seekable():
            return None
        position = stream.tell()
    except Exception:
        pass
    else:
        if type(position) is int and position >= 0:
            return position
    raise PreflightError("preflight_source_position_error")


def _read_prefix(stream: Any, limit: int) -> bytes:
    prefix = b""
    while len(prefix) < limit:
        request_bytes = limit - len(prefix)
        chunk = _read_chunk(stream, request_bytes)
        if not chunk:
            break
        prefix += chunk
    return prefix


def _read_chunk(stream: Any, request_bytes: int) -> bytes:
    try:
        chunk = stream.read(request_bytes)
    except Exception:
        pass
    else:
        if isinstance(chunk, bytes) and len(chunk) <= request_bytes:
            return chunk
        raise PreflightError("preflight_source_contract_error")
    raise PreflightError("preflight_source_read_error")


def _status(value: Any) -> PreflightStatus:
    try:
        status = PreflightStatus(value)
    except Exception:
        pass
    else:
        return status
    raise PreflightError("status is unsupported")


def _findings(value: Any) -> tuple[Any, ...]:
    try:
        findings = tuple(value)
    except Exception:
        pass
    else:
        return findings
    raise PreflightError("findings could not be read")
