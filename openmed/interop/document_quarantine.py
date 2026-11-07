"""Deterministic, privacy-safe document intake quarantine checks.

The classifier is deliberately metadata-only at its boundary.  It accepts a
local byte payload, a declared MIME type, and an optional filename, then
returns a disposition, stable reason codes, and a SHA-256 digest.  It never
returns the filename, MIME values, archive member names, or payload bytes.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import io
import json
import lzma
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Final, Iterable


class Disposition(str, Enum):
    """Safe routing decision for an incoming document."""

    ACCEPTED = "accepted"
    QUARANTINED = "quarantined"
    REJECTED = "rejected"


QuarantineDisposition = Disposition
QuarantineStatus = Disposition

ACCEPTED: Final = Disposition.ACCEPTED
QUARANTINED: Final = Disposition.QUARANTINED
REJECTED: Final = Disposition.REJECTED

REASON_ACCEPTED: Final = "accepted"
REASON_EMPTY_PAYLOAD: Final = "empty_payload"
REASON_SIZE_LIMIT_EXCEEDED: Final = "size_limit_exceeded"
REASON_DECLARED_MIME_MISSING: Final = "declared_mime_missing"
REASON_DECLARED_MIME_INVALID: Final = "declared_mime_invalid"
REASON_MIME_UNSUPPORTED: Final = "mime_unsupported"
REASON_EXTENSION_MISSING: Final = "extension_missing"
REASON_EXTENSION_UNSUPPORTED: Final = "extension_unsupported"
REASON_SNIFFED_TYPE_UNKNOWN: Final = "sniffed_type_unknown"
REASON_SNIFFED_TYPE_UNSUPPORTED: Final = "sniffed_type_unsupported"
REASON_MIME_EXTENSION_MISMATCH: Final = "mime_extension_mismatch"
REASON_DECLARED_MIME_SNIFF_MISMATCH: Final = "declared_mime_sniff_mismatch"
REASON_EXTENSION_SNIFF_MISMATCH: Final = "extension_sniff_mismatch"
REASON_ARCHIVE_INVALID: Final = "archive_invalid"
REASON_ARCHIVE_DEPTH_EXCEEDED: Final = "archive_depth_exceeded"
REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED: Final = "archive_member_limit_exceeded"
REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED: Final = "archive_size_limit_exceeded"
REASON_ARCHIVE_PATH_TRAVERSAL: Final = "archive_path_traversal"
REASON_ARCHIVE_SYMLINK: Final = "archive_symlink"
REASON_ARCHIVE_MEMBER_UNINSPECTED: Final = "archive_member_uninspected"

# Compatibility aliases make the reason vocabulary easy to discover without
# creating multiple spellings in returned reports.
REASON_MIME_MISMATCH: Final = REASON_DECLARED_MIME_SNIFF_MISMATCH
REASON_EXTENSION_MISMATCH: Final = REASON_EXTENSION_SNIFF_MISMATCH

_MIME_RE: Final = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")

_MIME_BY_EXTENSION: Final[dict[str, frozenset[str]]] = {
    "pdf": frozenset({"application/pdf"}),
    "png": frozenset({"image/png"}),
    "jpg": frozenset({"image/jpeg"}),
    "jpeg": frozenset({"image/jpeg"}),
    "gif": frozenset({"image/gif"}),
    "tif": frozenset({"image/tiff"}),
    "tiff": frozenset({"image/tiff"}),
    "dicom": frozenset({"application/dicom"}),
    "dcm": frozenset({"application/dicom"}),
    "txt": frozenset({"text/plain"}),
    "md": frozenset({"text/plain"}),
    "csv": frozenset({"text/csv"}),
    "tsv": frozenset({"text/tab-separated-values"}),
    "json": frozenset({"application/json"}),
    "xml": frozenset({"application/xml", "text/xml"}),
    "html": frozenset({"text/html"}),
    "htm": frozenset({"text/html"}),
    "zip": frozenset({"application/zip"}),
    "gz": frozenset({"application/gzip"}),
    "tgz": frozenset({"application/gzip"}),
    "tar": frozenset({"application/x-tar"}),
    "tar.gz": frozenset({"application/gzip", "application/x-tar"}),
    "bz2": frozenset({"application/x-bzip2"}),
    "xz": frozenset({"application/x-xz"}),
    "docx": frozenset(
        {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    ),
    "xlsx": frozenset(
        {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}
    ),
    "pptx": frozenset(
        {"application/vnd.openxmlformats-officedocument.presentationml.presentation"}
    ),
}

_DEFAULT_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/dicom",
        "application/gzip",
        "application/json",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/xml",
        "application/zip",
        "application/x-bzip2",
        "application/x-tar",
        "application/x-xz",
        "image/gif",
        "image/jpeg",
        "image/png",
        "image/tiff",
        "text/csv",
        "text/html",
        "text/plain",
        "text/tab-separated-values",
        "text/xml",
    }
)

_DEFAULT_EXTENSIONS: Final[frozenset[str]] = frozenset(_MIME_BY_EXTENSION)

_REJECT_REASONS: Final[frozenset[str]] = frozenset(
    {
        REASON_EMPTY_PAYLOAD,
        REASON_SIZE_LIMIT_EXCEEDED,
        REASON_MIME_UNSUPPORTED,
        REASON_EXTENSION_UNSUPPORTED,
        REASON_SNIFFED_TYPE_UNSUPPORTED,
        REASON_ARCHIVE_INVALID,
        REASON_ARCHIVE_DEPTH_EXCEEDED,
        REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED,
        REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED,
        REASON_ARCHIVE_PATH_TRAVERSAL,
        REASON_ARCHIVE_SYMLINK,
    }
)

_REASON_ORDER: Final[tuple[str, ...]] = (
    REASON_EMPTY_PAYLOAD,
    REASON_SIZE_LIMIT_EXCEEDED,
    REASON_MIME_UNSUPPORTED,
    REASON_EXTENSION_UNSUPPORTED,
    REASON_SNIFFED_TYPE_UNSUPPORTED,
    REASON_ARCHIVE_INVALID,
    REASON_ARCHIVE_DEPTH_EXCEEDED,
    REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED,
    REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED,
    REASON_ARCHIVE_PATH_TRAVERSAL,
    REASON_ARCHIVE_SYMLINK,
    REASON_ARCHIVE_MEMBER_UNINSPECTED,
    REASON_DECLARED_MIME_MISSING,
    REASON_DECLARED_MIME_INVALID,
    REASON_EXTENSION_MISSING,
    REASON_SNIFFED_TYPE_UNKNOWN,
    REASON_MIME_EXTENSION_MISMATCH,
    REASON_DECLARED_MIME_SNIFF_MISMATCH,
    REASON_EXTENSION_SNIFF_MISMATCH,
)

_ARCHIVE_MAGIC_BYTES: Final[int] = 512
_DEFAULT_MAX_SIZE_BYTES: Final[int] = 25_000_000
_DEFAULT_MAX_ARCHIVE_MEMBERS: Final[int] = 1_024
_DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES: Final[int] = 100_000_000
_DEFAULT_MAX_ARCHIVE_PROBE_BYTES: Final[int] = 8_000_000


def _normalize_mime(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.split(";", 1)[0].strip().lower()
    return normalized if _MIME_RE.fullmatch(normalized) else None


def _normalize_extension(value: str) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().lstrip(".")
    if not normalized or "/" in normalized or "\\" in normalized:
        return None
    return normalized


def _normalize_allowed_mimes(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed MIME types must be an iterable of strings")
    try:
        normalized = frozenset(
            mime for value in values if (mime := _normalize_mime(value)) is not None
        )
    except TypeError as exc:
        raise TypeError("allowed MIME types must be an iterable of strings") from exc
    if not normalized:
        raise ValueError("allowed MIME types must not be empty")
    return normalized


def _normalize_allowed_extensions(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed extensions must be an iterable of strings")
    try:
        normalized = frozenset(
            extension
            for value in values
            if (extension := _normalize_extension(value)) is not None
        )
    except TypeError as exc:
        raise TypeError("allowed extensions must be an iterable of strings") from exc
    if not normalized:
        raise ValueError("allowed extensions must not be empty")
    return normalized


@dataclass(frozen=True, slots=True)
class DocumentQuarantinePolicy:
    """Local limits and allowlists used by :func:`classify_document`.

    Args:
        max_size_bytes: Maximum size of the incoming payload.
        max_archive_depth: Maximum number of nested archive layers.  A
            top-level archive has depth one.
        allowed_mime_types: MIME types that may be accepted after local
            content inspection.
        allowed_extensions: Filename extensions that may be accepted.
        max_archive_members: Maximum number of entries in one archive.
        max_archive_uncompressed_bytes: Maximum declared uncompressed size in
            one archive.
        max_archive_probe_bytes: Maximum size of an archive member inspected
            recursively for nested archives.
    """

    max_size_bytes: int = _DEFAULT_MAX_SIZE_BYTES
    max_archive_depth: int = 1
    allowed_mime_types: frozenset[str] = _DEFAULT_MIME_TYPES
    allowed_extensions: frozenset[str] = _DEFAULT_EXTENSIONS
    max_archive_members: int = _DEFAULT_MAX_ARCHIVE_MEMBERS
    max_archive_uncompressed_bytes: int = _DEFAULT_MAX_ARCHIVE_UNCOMPRESSED_BYTES
    max_archive_probe_bytes: int = _DEFAULT_MAX_ARCHIVE_PROBE_BYTES

    def __post_init__(self) -> None:
        for value in (
            self.max_size_bytes,
            self.max_archive_depth,
            self.max_archive_members,
            self.max_archive_uncompressed_bytes,
            self.max_archive_probe_bytes,
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("policy limits must be non-negative integers")
        object.__setattr__(
            self,
            "allowed_mime_types",
            _normalize_allowed_mimes(self.allowed_mime_types),
        )
        object.__setattr__(
            self,
            "allowed_extensions",
            _normalize_allowed_extensions(self.allowed_extensions),
        )


DEFAULT_POLICY: Final = DocumentQuarantinePolicy()


@dataclass(frozen=True, slots=True)
class QuarantineResult:
    """Privacy-safe result returned by the document classifier."""

    disposition: Disposition
    reason_codes: tuple[str, ...]
    sha256: str

    @property
    def status(self) -> str:
        """Return the string disposition for JSON and logging adapters."""

        return self.disposition.value

    @property
    def reason_code(self) -> str:
        """Return the first stable reason code for simple callers."""

        return self.reason_codes[0]

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return all stable reason codes in deterministic order."""

        return self.reason_codes

    @property
    def content_hash(self) -> str:
        """Return the SHA-256 digest without exposing payload content."""

        return self.sha256

    @property
    def payload_hash(self) -> str:
        """Return the SHA-256 digest under the payload-oriented name."""

        return self.sha256

    def to_dict(self) -> dict[str, object]:
        """Return the complete privacy-safe report shape."""

        return {
            "disposition": self.status,
            "reason_codes": list(self.reason_codes),
            "sha256": self.sha256,
        }

    as_dict = to_dict


@dataclass(frozen=True, slots=True)
class _SniffResult:
    mime: str | None
    archive_kind: str | None = None

    @property
    def is_archive(self) -> bool:
        return self.archive_kind is not None


def _extension_from_filename(filename: str | os.PathLike[str] | None) -> str | None:
    if filename is None:
        return None
    if isinstance(filename, os.PathLike):
        filename = os.fspath(filename)
    if not isinstance(filename, str):
        raise TypeError("filename must be a string or path-like value")
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    for extension in sorted(_MIME_BY_EXTENSION, key=len, reverse=True):
        if name.endswith(f".{extension}"):
            return extension
    if "." not in name or name.endswith("."):
        return None
    return _normalize_extension(name.rsplit(".", 1)[-1])


def _zip_mime(payload: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = {
                info.filename.replace("\\", "/").lower() for info in archive.infolist()
            }
            content_types = next(
                (
                    info
                    for info in archive.infolist()
                    if info.filename.replace("\\", "/").lower() == "[content_types].xml"
                ),
                None,
            )
            if content_types is not None and content_types.file_size <= 1_000_000:
                content = archive.read(content_types).lower()
                if b"wordprocessingml.document.main+xml" in content:
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                if b"spreadsheetml.sheet.main+xml" in content:
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if b"presentationml.presentation.main+xml" in content:
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            if "word/document.xml" in names:
                return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if "xl/workbook.xml" in names:
                return (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            if "ppt/presentation.xml" in names:
                return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile):
        pass
    return "application/zip"


def _sniff_payload(payload: bytes, extension: str | None = None) -> _SniffResult:
    if payload.startswith(b"%PDF-"):
        return _SniffResult("application/pdf")
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return _SniffResult("image/png")
    if payload.startswith(b"\xff\xd8\xff"):
        return _SniffResult("image/jpeg")
    if payload.startswith((b"GIF87a", b"GIF89a")):
        return _SniffResult("image/gif")
    if payload.startswith((b"II*\x00", b"MM\x00*")):
        return _SniffResult("image/tiff")
    if len(payload) >= 132 and payload[128:132] == b"DICM":
        return _SniffResult("application/dicom")
    if payload.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return _SniffResult(_zip_mime(payload), "zip")
    if payload.startswith(b"\x1f\x8b"):
        return _SniffResult("application/gzip", "gzip")
    if payload.startswith(b"BZh"):
        return _SniffResult("application/x-bzip2", "bzip2")
    if payload.startswith(b"\xfd7zXZ\x00"):
        return _SniffResult("application/x-xz", "xz")
    if len(payload) >= 262 and payload[257:262] == b"ustar":
        return _SniffResult("application/x-tar", "tar")
    if payload.startswith(b"MZ"):
        return _SniffResult("application/x-dosexec")
    if payload.startswith(b"\x7fELF"):
        return _SniffResult("application/x-executable")

    text = payload.decode("utf-8-sig", errors="ignore")
    if b"\x00" not in payload and text.strip():
        stripped = text.lstrip()
        if extension == "json" and stripped[:1] in {"{", "["}:
            try:
                json.loads(text)
            except (TypeError, ValueError):
                pass
            else:
                return _SniffResult("application/json")
        if extension in {"xml", "html", "htm"} and stripped.startswith("<"):
            if extension in {"html", "htm"} or stripped.lower().startswith(
                ("<!doctype html", "<html")
            ):
                return _SniffResult("text/html")
            return _SniffResult("application/xml")
        if stripped[:1] == "<":
            return _SniffResult("application/xml")
        if extension == "csv":
            return _SniffResult("text/csv")
        if extension == "tsv":
            return _SniffResult("text/tab-separated-values")
        if extension in {"html", "htm"}:
            return _SniffResult("text/html")
        return _SniffResult("text/plain")
    return _SniffResult(None)


def _mime_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    text_mimes = {left, right}
    if all(mime.startswith("text/") for mime in text_mimes):
        return True
    return False


def _member_path_is_unsafe(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return True
    if re.match(r"^[A-Za-z]:", normalized):
        return True
    return any(part == ".." for part in normalized.split("/"))


def _unique_reasons(reasons: Iterable[str]) -> tuple[str, ...]:
    present = set(reasons)
    ordered = [reason for reason in _REASON_ORDER if reason in present]
    ordered.extend(sorted(present.difference(_REASON_ORDER)))
    return tuple(ordered)


def _inspect_zip(
    payload: bytes,
    *,
    depth: int,
    policy: DocumentQuarantinePolicy,
) -> tuple[str, ...]:
    if depth > policy.max_archive_depth:
        return (REASON_ARCHIVE_DEPTH_EXCEEDED,)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except (OSError, ValueError, zipfile.BadZipFile):
        return (REASON_ARCHIVE_INVALID,)

    reasons: list[str] = []
    try:
        try:
            infos = archive.infolist()
        except (OSError, ValueError, zipfile.BadZipFile):
            return (REASON_ARCHIVE_INVALID,)
        if len(infos) > policy.max_archive_members:
            reasons.append(REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED)
        total_size = sum(max(info.file_size, 0) for info in infos)
        if total_size > policy.max_archive_uncompressed_bytes:
            reasons.append(REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED)

        for info in infos:
            if _member_path_is_unsafe(info.filename):
                reasons.append(REASON_ARCHIVE_PATH_TRAVERSAL)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                reasons.append(REASON_ARCHIVE_SYMLINK)
            if info.is_dir() or info.file_size == 0:
                continue

            try:
                with archive.open(info) as member:
                    prefix = member.read(_ARCHIVE_MAGIC_BYTES)
                    nested = _sniff_payload(prefix)
                    if not nested.is_archive:
                        continue
                    nested_depth = depth + 1
                    if nested_depth > policy.max_archive_depth:
                        reasons.append(REASON_ARCHIVE_DEPTH_EXCEEDED)
                        continue
                    if info.file_size > policy.max_archive_probe_bytes:
                        reasons.append(REASON_ARCHIVE_MEMBER_UNINSPECTED)
                        continue
                    remainder = member.read(policy.max_archive_probe_bytes)
                    member_payload = prefix + remainder
                reasons.extend(
                    _inspect_archive(
                        member_payload,
                        nested,
                        depth=nested_depth,
                        policy=policy,
                    )
                )
            except (OSError, RuntimeError, ValueError, EOFError, zipfile.BadZipFile):
                reasons.append(REASON_ARCHIVE_MEMBER_UNINSPECTED)
    finally:
        archive.close()
    return _unique_reasons(reasons)


def _inspect_tar(
    payload: bytes,
    *,
    depth: int,
    policy: DocumentQuarantinePolicy,
    require_tar: bool,
) -> tuple[str, ...]:
    if depth > policy.max_archive_depth:
        return (REASON_ARCHIVE_DEPTH_EXCEEDED,)
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.ReadError:
        return (REASON_ARCHIVE_INVALID,) if require_tar else ()
    except (OSError, ValueError, tarfile.TarError):
        return (REASON_ARCHIVE_INVALID,)

    reasons: list[str] = []
    try:
        try:
            members = archive.getmembers()
        except tarfile.TarError:
            return (REASON_ARCHIVE_INVALID,)
        if len(members) > policy.max_archive_members:
            reasons.append(REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED)
        total_size = sum(max(member.size, 0) for member in members)
        if total_size > policy.max_archive_uncompressed_bytes:
            reasons.append(REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED)
        for member in members:
            if _member_path_is_unsafe(member.name):
                reasons.append(REASON_ARCHIVE_PATH_TRAVERSAL)
            if member.issym() or member.islnk():
                reasons.append(REASON_ARCHIVE_SYMLINK)
            if not member.isfile() or member.size == 0:
                continue
            if member.size > policy.max_archive_probe_bytes:
                try:
                    extracted = archive.extractfile(member)
                    prefix = (
                        extracted.read(_ARCHIVE_MAGIC_BYTES)
                        if extracted is not None
                        else b""
                    )
                    if _sniff_payload(prefix).is_archive:
                        if depth + 1 > policy.max_archive_depth:
                            reasons.append(REASON_ARCHIVE_DEPTH_EXCEEDED)
                        else:
                            reasons.append(REASON_ARCHIVE_MEMBER_UNINSPECTED)
                except (OSError, RuntimeError, ValueError, tarfile.TarError):
                    reasons.append(REASON_ARCHIVE_MEMBER_UNINSPECTED)
                continue
            try:
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                member_payload = extracted.read(policy.max_archive_probe_bytes)
                nested = _sniff_payload(member_payload)
                if nested.is_archive:
                    nested_depth = depth + 1
                    if nested_depth > policy.max_archive_depth:
                        reasons.append(REASON_ARCHIVE_DEPTH_EXCEEDED)
                    else:
                        reasons.extend(
                            _inspect_archive(
                                member_payload,
                                nested,
                                depth=nested_depth,
                                policy=policy,
                            )
                        )
            except (OSError, RuntimeError, ValueError, EOFError, tarfile.TarError):
                reasons.append(REASON_ARCHIVE_MEMBER_UNINSPECTED)
    finally:
        archive.close()
    return _unique_reasons(reasons)


def _decompress_archive_payload(
    payload: bytes,
    *,
    archive_kind: str,
    policy: DocumentQuarantinePolicy,
) -> tuple[bytes | None, tuple[str, ...]]:
    """Decompress one stream without exceeding the configured byte budget."""

    source = io.BytesIO(payload)
    try:
        if archive_kind == "gzip":
            stream = gzip.GzipFile(fileobj=source, mode="rb")
        elif archive_kind == "bzip2":
            stream = bz2.BZ2File(source, mode="rb")
        elif archive_kind == "xz":
            stream = lzma.LZMAFile(source, mode="rb")
        else:  # pragma: no cover - guarded by the archive dispatcher
            return None, (REASON_ARCHIVE_INVALID,)
        with stream:
            decompressed = stream.read(policy.max_archive_uncompressed_bytes + 1)
    except (EOFError, OSError, ValueError, lzma.LZMAError):
        return None, (REASON_ARCHIVE_INVALID,)

    if len(decompressed) > policy.max_archive_uncompressed_bytes:
        return None, (REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED,)
    return decompressed, ()


def _inspect_archive(
    payload: bytes,
    sniffed: _SniffResult,
    *,
    depth: int,
    policy: DocumentQuarantinePolicy,
) -> tuple[str, ...]:
    if depth > policy.max_archive_depth:
        return (REASON_ARCHIVE_DEPTH_EXCEEDED,)
    if sniffed.archive_kind == "zip":
        return _inspect_zip(payload, depth=depth, policy=policy)
    if sniffed.archive_kind == "tar":
        return _inspect_tar(
            payload,
            depth=depth,
            policy=policy,
            require_tar=True,
        )
    if sniffed.archive_kind in {"gzip", "bzip2", "xz"}:
        decompressed, reasons = _decompress_archive_payload(
            payload,
            archive_kind=sniffed.archive_kind,
            policy=policy,
        )
        if reasons or decompressed is None:
            return reasons
        nested = _sniff_payload(decompressed)
        if nested.archive_kind == "tar":
            # Compression is part of the tar container, not another nesting
            # level (for example, a top-level .tar.gz remains depth one).
            return _inspect_tar(
                decompressed,
                depth=depth,
                policy=policy,
                require_tar=True,
            )
        if nested.is_archive:
            return _inspect_archive(
                decompressed,
                nested,
                depth=depth + 1,
                policy=policy,
            )
        return ()
    return ()


def _as_payload_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    try:
        return bytes(payload)
    except (TypeError, ValueError):
        raise TypeError("payload must be bytes-like") from None


def classify_document(
    payload: bytes | bytearray | memoryview,
    declared_mime: str | None = None,
    filename: str | os.PathLike[str] | None = None,
    *,
    policy: DocumentQuarantinePolicy = DEFAULT_POLICY,
) -> QuarantineResult:
    """Classify a local document without retaining or reporting raw values.

    Args:
        payload: Bytes supplied by the local intake boundary.
        declared_mime: MIME type supplied by the caller or transport layer.
        filename: Optional filename used only to derive a lowercase extension.
        policy: Size, archive, and allowlist policy.

    Returns:
        A deterministic :class:`QuarantineResult` containing only the
        disposition, stable reason codes, and SHA-256 payload digest.

    Raises:
        TypeError: If the payload, filename, or policy has an invalid shape.
    """

    if not isinstance(policy, DocumentQuarantinePolicy):
        raise TypeError("policy must be a DocumentQuarantinePolicy")
    data = _as_payload_bytes(payload)
    digest = hashlib.sha256(data).hexdigest()
    if not data:
        return QuarantineResult(
            Disposition.REJECTED,
            (REASON_EMPTY_PAYLOAD,),
            digest,
        )
    if len(data) > policy.max_size_bytes:
        return QuarantineResult(
            Disposition.REJECTED,
            (REASON_SIZE_LIMIT_EXCEEDED,),
            digest,
        )

    extension = _extension_from_filename(filename)
    if declared_mime is not None and not isinstance(declared_mime, str):
        declared = None
        declared_invalid = True
    else:
        declared = _normalize_mime(declared_mime) if declared_mime is not None else None
        declared_invalid = declared_mime is not None and declared is None

    if declared is not None and declared not in policy.allowed_mime_types:
        return QuarantineResult(
            Disposition.REJECTED,
            (REASON_MIME_UNSUPPORTED,),
            digest,
        )
    if extension is not None and extension not in policy.allowed_extensions:
        return QuarantineResult(
            Disposition.REJECTED,
            (REASON_EXTENSION_UNSUPPORTED,),
            digest,
        )

    sniffed = _sniff_payload(data, extension)
    reasons: list[str] = []
    if declared is None:
        reasons.append(
            REASON_DECLARED_MIME_INVALID
            if declared_invalid
            else REASON_DECLARED_MIME_MISSING
        )
    if extension is None:
        reasons.append(REASON_EXTENSION_MISSING)
    if sniffed.mime is None:
        reasons.append(REASON_SNIFFED_TYPE_UNKNOWN)
    elif sniffed.mime not in policy.allowed_mime_types:
        return QuarantineResult(
            Disposition.REJECTED,
            (REASON_SNIFFED_TYPE_UNSUPPORTED,),
            digest,
        )

    expected_mimes = _MIME_BY_EXTENSION.get(extension or "")
    if expected_mimes:
        if declared is not None and not any(
            _mime_matches(declared, expected) for expected in expected_mimes
        ):
            reasons.append(REASON_MIME_EXTENSION_MISMATCH)
        if sniffed.mime is not None and not any(
            _mime_matches(sniffed.mime, expected) for expected in expected_mimes
        ):
            reasons.append(REASON_EXTENSION_SNIFF_MISMATCH)
    if (
        declared is not None
        and sniffed.mime is not None
        and not _mime_matches(declared, sniffed.mime)
    ):
        reasons.append(REASON_DECLARED_MIME_SNIFF_MISMATCH)

    if sniffed.is_archive:
        reasons.extend(_inspect_archive(data, sniffed, depth=1, policy=policy))

    ordered_reasons = _unique_reasons(reasons)
    if any(reason in _REJECT_REASONS for reason in ordered_reasons):
        disposition = Disposition.REJECTED
    elif ordered_reasons:
        disposition = Disposition.QUARANTINED
    else:
        disposition = Disposition.ACCEPTED
        ordered_reasons = (REASON_ACCEPTED,)
    return QuarantineResult(disposition, ordered_reasons, digest)


def quarantine_document(
    payload: bytes | bytearray | memoryview,
    declared_mime: str | None = None,
    filename: str | os.PathLike[str] | None = None,
    *,
    policy: DocumentQuarantinePolicy = DEFAULT_POLICY,
) -> QuarantineResult:
    """Alias for :func:`classify_document` at an intake boundary."""

    return classify_document(
        payload,
        declared_mime,
        filename,
        policy=policy,
    )


inspect_document = classify_document
evaluate_document = classify_document


__all__ = [
    "ACCEPTED",
    "DEFAULT_POLICY",
    "Disposition",
    "DocumentQuarantinePolicy",
    "QUARANTINED",
    "QuarantineDisposition",
    "QuarantineResult",
    "QuarantineStatus",
    "REJECTED",
    "REASON_ACCEPTED",
    "REASON_ARCHIVE_DEPTH_EXCEEDED",
    "REASON_ARCHIVE_INVALID",
    "REASON_ARCHIVE_MEMBER_LIMIT_EXCEEDED",
    "REASON_ARCHIVE_MEMBER_UNINSPECTED",
    "REASON_ARCHIVE_PATH_TRAVERSAL",
    "REASON_ARCHIVE_SIZE_LIMIT_EXCEEDED",
    "REASON_ARCHIVE_SYMLINK",
    "REASON_DECLARED_MIME_INVALID",
    "REASON_DECLARED_MIME_MISSING",
    "REASON_DECLARED_MIME_SNIFF_MISMATCH",
    "REASON_EMPTY_PAYLOAD",
    "REASON_EXTENSION_MISSING",
    "REASON_EXTENSION_MISMATCH",
    "REASON_EXTENSION_SNIFF_MISMATCH",
    "REASON_EXTENSION_UNSUPPORTED",
    "REASON_MIME_EXTENSION_MISMATCH",
    "REASON_MIME_MISMATCH",
    "REASON_MIME_UNSUPPORTED",
    "REASON_SIZE_LIMIT_EXCEEDED",
    "REASON_SNIFFED_TYPE_UNKNOWN",
    "REASON_SNIFFED_TYPE_UNSUPPORTED",
    "classify_document",
    "evaluate_document",
    "inspect_document",
    "quarantine_document",
]
