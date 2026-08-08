"""Deterministic, local privacy guards for dataset upload call sites.

The guard deliberately separates scanning from uploading.  It accepts a
caller-configured scanner that receives decoded file text and returns offsets;
the default scanner is a small offline safety net for common direct
identifiers.  No scanner result stores the matched surface.

Block mode leaves source files untouched and refuses to call the upload
function when a finding is present.  Redaction mode writes UTF-8 content with
generated replacements to a caller-provided staging directory, then passes
only those staged files to the upload function.  Reports contain counts and
stable file digests, never source paths or file content.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

BLOCK_ONLY_MODE = "block_only"
REDACT_TO_STAGING_MODE = "redact_to_staging"
DEFAULT_MODE = BLOCK_ONLY_MODE

GuardMode: TypeAlias = Literal["block_only", "redact_to_staging"]
DatasetPath: TypeAlias = str | os.PathLike[str]
UploadCallable: TypeAlias = Callable[..., Any]

_MODE_ALIASES = {
    "block": BLOCK_ONLY_MODE,
    "block_only": BLOCK_ONLY_MODE,
    "redact": REDACT_TO_STAGING_MODE,
    "redact_to_staging": REDACT_TO_STAGING_MODE,
    "redact_to_staging_directory": REDACT_TO_STAGING_MODE,
    "redact-to-staging-directory": REDACT_TO_STAGING_MODE,
}
_SAFE_LABEL = re.compile(r"^[A-Z][A-Z0-9_.:-]{0,63}$")
_SAFE_EXTENSION = re.compile(r"\.[A-Za-z0-9]{1,12}$")
_EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@"
    r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?![\w.-])"
)
_SSN_PATTERN = re.compile(r"(?<!\w)\d{3}[- ]\d{2}[- ]\d{4}(?!\w)")
_CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-])?(?:\(\d{3}\)|\d{3})"
    r"[ .-]\d{3}[ .-]\d{4}(?!\w)|"
    r"(?<!\w)\d{3}[ .-]\d{4}(?!\w)"
)


@dataclass(frozen=True)
class DatasetFinding:
    """Privacy finding represented only by a safe label and text offsets.

    Args:
        label: Non-sensitive category such as ``EMAIL`` or ``PHONE``.
        start: Inclusive character offset in the scanned text.
        end: Exclusive character offset in the scanned text.

    The matched text is intentionally not accepted as a field on this object.
    """

    label: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _normalize_label(self.label))
        if type(self.start) is not int or type(self.end) is not int:
            raise ValueError("finding offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("finding offsets must be non-negative and ordered")


@dataclass(frozen=True)
class DatasetFileReport:
    """PHI-free counts for one scanned file."""

    file_id: str
    finding_count: int
    finding_counts: Mapping[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report without source metadata."""
        return {
            "file_id": self.file_id,
            "finding_count": self.finding_count,
            "finding_counts": dict(sorted(self.finding_counts.items())),
        }


@dataclass(frozen=True)
class DatasetGuardReport:
    """Aggregate privacy-safe result of scanning selected dataset files."""

    mode: GuardMode
    allowed: bool
    file_count: int
    finding_count: int
    finding_counts: Mapping[str, int]
    files: tuple[DatasetFileReport, ...]
    staged_file_ids: tuple[str, ...] = ()

    @property
    def file_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for the selected source files."""
        return tuple(file_report.file_id for file_report in self.files)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report without source paths or text."""
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "file_count": self.file_count,
            "finding_count": self.finding_count,
            "finding_counts": dict(sorted(self.finding_counts.items())),
            "file_ids": list(self.file_ids),
            "files": [file_report.to_dict() for file_report in self.files],
            "staged_file_ids": list(self.staged_file_ids),
        }


@dataclass(frozen=True)
class DatasetUploadResult:
    """Result of a guarded upload call.

    ``upload_result`` is retained for the caller but deliberately omitted from
    :meth:`to_dict`, because an arbitrary upload client may return sensitive
    content.  The guard-owned report is always safe to serialize.
    """

    report: DatasetGuardReport
    upload_result: Any = field(repr=False, compare=False)

    @property
    def file_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for the uploaded source files."""
        return self.report.file_ids

    def to_dict(self) -> dict[str, Any]:
        """Return only the guard-owned, privacy-safe result fields."""
        return {
            "upload_completed": True,
            "report": self.report.to_dict(),
        }


class DatasetGuardError(RuntimeError):
    """Base error for safe, fail-closed dataset-guard failures."""


class DatasetUploadBlockedError(DatasetGuardError):
    """Raised when block mode finds one or more unsafe file spans."""

    def __init__(self, report: DatasetGuardReport) -> None:
        self.report = report
        super().__init__("dataset upload blocked by the local privacy guard")


class DatasetUploadError(DatasetGuardError):
    """Raised when the wrapped upload fails without exposing its exception."""

    def __init__(self, report: DatasetGuardReport) -> None:
        self.report = report
        super().__init__("dataset upload failed after the local privacy check")


ScannerFinding: TypeAlias = DatasetFinding | Mapping[str, Any] | Sequence[Any]
DatasetScanner: TypeAlias = Callable[[str], Iterable[ScannerFinding] | None]


def scan_text(text: str) -> tuple[DatasetFinding, ...]:
    """Find common direct identifiers using deterministic local patterns.

    The built-in safety net recognizes email addresses, phone numbers, SSN-like
    values, and valid Luhn card numbers.  It is intentionally extensible: a
    deployment with domain-specific identifiers should pass a scanner of its
    own rather than treating this small set as a compliance guarantee.

    Args:
        text: UTF-8-decoded dataset text to scan.

    Returns:
        Findings sorted by offset, with no matched surfaces.
    """

    if not isinstance(text, str):
        raise TypeError("dataset scanner input must be text")

    candidates: list[tuple[int, int, int, DatasetFinding]] = []
    patterns = (
        (0, "EMAIL", _EMAIL_PATTERN),
        (1, "SSN", _SSN_PATTERN),
        (2, "CREDIT_CARD", _CARD_PATTERN),
        (3, "PHONE", _PHONE_PATTERN),
    )
    for priority, label, pattern in patterns:
        for match in pattern.finditer(text):
            if label == "CREDIT_CARD" and not _passes_luhn(match.group(0)):
                continue
            candidates.append(
                (
                    match.start(),
                    match.end(),
                    priority,
                    DatasetFinding(label, match.start(), match.end()),
                )
            )

    findings: list[DatasetFinding] = []
    for _, _, _, candidate in sorted(
        candidates,
        key=lambda item: (item[0], -(item[1] - item[0]), item[2]),
    ):
        if findings and candidate.start < findings[-1].end:
            continue
        findings.append(candidate)
    return tuple(findings)


def redact_text(text: str, findings: Iterable[ScannerFinding]) -> str:
    """Replace scanner offsets with stable, non-sensitive redaction tokens."""

    if not isinstance(text, str):
        raise TypeError("dataset text must be a string")
    normalized = _normalize_findings(findings, text_length=len(text))
    if not normalized:
        return text

    pieces: list[str] = []
    cursor = 0
    for finding in normalized:
        pieces.append(text[cursor : finding.start])
        pieces.append(f"[OPENMED_REDACTED_{finding.label}]")
        cursor = finding.end
    pieces.append(text[cursor:])
    return "".join(pieces)


def scan_dataset_files(
    files: DatasetPath | Iterable[DatasetPath],
    *,
    scanner: DatasetScanner | None = None,
) -> DatasetGuardReport:
    """Scan selected files and return a PHI-free block-mode report.

    The scanner receives file text only.  File paths are hashed for the report
    and are never passed to the scanner or copied into an exception message.
    """

    scans = _scan_files(files, scanner=scanner or scan_text)
    return _build_report(BLOCK_ONLY_MODE, scans, allowed=not _has_findings(scans))


def inspect_dataset_files(
    files: DatasetPath | Iterable[DatasetPath],
    *,
    scanner: DatasetScanner | None = None,
) -> DatasetGuardReport:
    """Compatibility name for :func:`scan_dataset_files`."""

    return scan_dataset_files(files, scanner=scanner)


class DatasetUploadGuard:
    """Wrap a configured upload callable with a local privacy check.

    Args:
        upload: Callable whose first positional argument is a sequence of file
            paths.  It is called only after the guard permits the upload.
        mode: ``"block_only"`` (also ``"block"``) or
            ``"redact_to_staging"`` (also ``"redact"``).
        scanner: Callable receiving decoded file text and returning findings.
            When omitted, :func:`scan_text` is used.
        staging_dir: Required for redaction mode.  Generated names are used so
            source filenames never become staged artifact names.
    """

    def __init__(
        self,
        upload: UploadCallable,
        *,
        mode: str = DEFAULT_MODE,
        scanner: DatasetScanner | None = None,
        staging_dir: DatasetPath | None = None,
    ) -> None:
        if not callable(upload):
            raise TypeError("upload must be callable")
        self._upload = upload
        self.mode = _normalize_mode(mode)
        self.scanner = _validate_scanner(scanner or scan_text)
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        if self.mode == REDACT_TO_STAGING_MODE and self.staging_dir is None:
            raise ValueError("staging_dir is required for redaction mode")

    def inspect(
        self,
        files: DatasetPath | Iterable[DatasetPath],
    ) -> DatasetGuardReport:
        """Scan files without invoking the wrapped upload callable."""

        scans = _scan_files(files, scanner=self.scanner)
        return _build_report(
            self.mode,
            scans,
            allowed=(self.mode == REDACT_TO_STAGING_MODE or not _has_findings(scans)),
        )

    def __call__(
        self,
        files: DatasetPath | Iterable[DatasetPath],
        *upload_args: Any,
        **upload_kwargs: Any,
    ) -> DatasetUploadResult:
        """Check files and call the configured upload function if permitted."""

        paths = _coerce_paths(files)
        scans = _scan_files(paths, scanner=self.scanner)
        report = _build_report(
            self.mode,
            scans,
            allowed=(self.mode == REDACT_TO_STAGING_MODE or not _has_findings(scans)),
        )
        if self.mode == BLOCK_ONLY_MODE and _has_findings(scans):
            raise DatasetUploadBlockedError(report)

        upload_paths = paths
        if self.mode == REDACT_TO_STAGING_MODE:
            try:
                upload_paths = _stage_files(scans, self.staging_dir)
            except DatasetGuardError:
                raise
            except (OSError, UnicodeError):
                raise DatasetGuardError(
                    "dataset staging failed before upload"
                ) from None
            report = replace(report, staged_file_ids=report.file_ids)

        try:
            upload_result = self._upload(
                tuple(upload_paths), *upload_args, **upload_kwargs
            )
        except Exception:
            raise DatasetUploadError(report) from None
        return DatasetUploadResult(report=report, upload_result=upload_result)


def guard_dataset_upload(
    upload: UploadCallable,
    files: DatasetPath | Iterable[DatasetPath],
    *,
    mode: str = DEFAULT_MODE,
    scanner: DatasetScanner | None = None,
    staging_dir: DatasetPath | None = None,
    upload_args: Sequence[Any] = (),
    upload_kwargs: Mapping[str, Any] | None = None,
) -> DatasetUploadResult:
    """Guard one upload call without requiring a persistent wrapper object."""

    guard = DatasetUploadGuard(
        upload,
        mode=mode,
        scanner=scanner,
        staging_dir=staging_dir,
    )
    return guard(
        files,
        *tuple(upload_args),
        **dict(upload_kwargs or {}),
    )


@dataclass(frozen=True)
class _FileScan:
    path: Path
    data: bytes
    text: str
    file_id: str
    findings: tuple[DatasetFinding, ...]


def _scan_files(
    files: DatasetPath | Iterable[DatasetPath],
    *,
    scanner: DatasetScanner,
) -> tuple[_FileScan, ...]:
    paths = _coerce_paths(files)
    scans: list[_FileScan] = []
    for path in paths:
        try:
            if not path.is_file():
                raise OSError
            data = path.read_bytes()
            text = data.decode("utf-8")
        except (OSError, UnicodeError):
            raise DatasetGuardError("dataset file could not be read safely") from None

        try:
            raw_findings = scanner(text)
            findings = _normalize_findings(raw_findings or (), text_length=len(text))
        except DatasetGuardError:
            raise
        except Exception:
            raise DatasetGuardError("dataset scanner failed safely") from None

        scans.append(
            _FileScan(
                path=path,
                data=data,
                text=text,
                file_id=_file_id(path, data),
                findings=findings,
            )
        )
    return tuple(scans)


def _build_report(
    mode: GuardMode,
    scans: Sequence[_FileScan],
    *,
    allowed: bool,
) -> DatasetGuardReport:
    counts: Counter[str] = Counter()
    file_reports: list[DatasetFileReport] = []
    for scan in scans:
        file_counts = Counter(finding.label for finding in scan.findings)
        counts.update(file_counts)
        file_reports.append(
            DatasetFileReport(
                file_id=scan.file_id,
                finding_count=len(scan.findings),
                finding_counts=dict(sorted(file_counts.items())),
            )
        )
    return DatasetGuardReport(
        mode=mode,
        allowed=allowed,
        file_count=len(scans),
        finding_count=sum(counts.values()),
        finding_counts=dict(sorted(counts.items())),
        files=tuple(file_reports),
    )


def _stage_files(
    scans: Sequence[_FileScan], staging_dir: Path | None
) -> tuple[Path, ...]:
    if staging_dir is None:
        raise DatasetGuardError("dataset staging directory is not configured")
    try:
        staging_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise DatasetGuardError("dataset staging directory is unavailable") from None

    staged: list[Path] = []
    for index, scan in enumerate(scans):
        extension = scan.path.suffix.lower()
        if not _SAFE_EXTENSION.fullmatch(extension):
            extension = ""
        name = f"openmed-{index:04d}-{scan.file_id.removeprefix('sha256:')[:16]}{extension}"
        destination = staging_dir / name
        payload = (
            scan.data
            if not scan.findings
            else redact_text(scan.text, scan.findings).encode("utf-8")
        )
        try:
            destination.write_bytes(payload)
        except OSError:
            raise DatasetGuardError("dataset staging write failed") from None
        staged.append(destination)
    return tuple(staged)


def _coerce_paths(files: DatasetPath | Iterable[DatasetPath]) -> tuple[Path, ...]:
    if isinstance(files, (str, os.PathLike)):
        values: Iterable[DatasetPath] = (files,)
    else:
        try:
            values = tuple(files)
        except TypeError:
            raise TypeError("dataset files must be path-like or iterable") from None
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, (str, os.PathLike)):
            raise TypeError("dataset files must contain only path-like values")
        paths.append(Path(value))
    if not paths:
        raise ValueError("at least one dataset file is required")
    return tuple(paths)


def _normalize_findings(
    findings: Iterable[ScannerFinding],
    *,
    text_length: int,
) -> tuple[DatasetFinding, ...]:
    normalized: list[DatasetFinding] = []
    try:
        raw_values = tuple(findings)
    except TypeError:
        raise DatasetGuardError("dataset scanner did not return findings") from None

    for raw in raw_values:
        finding = _coerce_finding(raw)
        if finding.end > text_length:
            raise DatasetGuardError("dataset scanner returned an out-of-range finding")
        normalized.append(finding)

    normalized.sort(key=lambda finding: (finding.start, finding.end, finding.label))
    for previous, current in zip(normalized, normalized[1:]):
        if current.start < previous.end:
            raise DatasetGuardError("dataset scanner returned overlapping findings")
    return tuple(normalized)


def _coerce_finding(raw: ScannerFinding) -> DatasetFinding:
    if isinstance(raw, DatasetFinding):
        return raw
    if isinstance(raw, Mapping):
        start = raw.get("start")
        end = raw.get("end")
        label = raw.get("label", raw.get("entity_type", raw.get("type", "UNKNOWN")))
    elif hasattr(raw, "start") and hasattr(raw, "end"):
        start = raw.start() if callable(raw.start) else raw.start
        end = raw.end() if callable(raw.end) else raw.end
        label = getattr(raw, "label", getattr(raw, "entity_type", "UNKNOWN"))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) < 2:
            raise DatasetGuardError("dataset scanner returned an invalid finding")
        start, end = raw[0], raw[1]
        label = raw[2] if len(raw) > 2 else "UNKNOWN"
    else:
        raise DatasetGuardError("dataset scanner returned an invalid finding")

    if type(start) is not int or type(end) is not int:
        raise DatasetGuardError("dataset scanner returned invalid offsets")
    try:
        return DatasetFinding(str(label), start, end)
    except (TypeError, ValueError):
        raise DatasetGuardError("dataset scanner returned invalid offsets") from None


def _validate_scanner(scanner: DatasetScanner) -> DatasetScanner:
    if not callable(scanner):
        raise TypeError("scanner must be callable")
    return scanner


def _normalize_mode(mode: str) -> GuardMode:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    try:
        return _MODE_ALIASES[mode.strip().lower()]  # type: ignore[return-value]
    except KeyError:
        raise ValueError("mode must be block_only or redact_to_staging") from None


def _normalize_label(label: Any) -> str:
    if not isinstance(label, str):
        return "UNKNOWN"
    candidate = re.sub(r"[^A-Za-z0-9_.:-]+", "_", label.strip().upper()).strip("_")
    if not candidate or not _SAFE_LABEL.fullmatch(candidate):
        return "UNKNOWN"
    return candidate


def _file_id(path: Path, data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"openmed-dataset-file-v1\0")
    digest.update(os.fspath(path).encode("utf-8", "surrogatepass"))
    digest.update(b"\0")
    digest.update(data)
    return f"sha256:{digest.hexdigest()}"


def _has_findings(scans: Sequence[_FileScan]) -> bool:
    return any(scan.findings for scan in scans)


def _passes_luhn(value: str) -> bool:
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


__all__ = [
    "BLOCK_ONLY_MODE",
    "DEFAULT_MODE",
    "REDACT_TO_STAGING_MODE",
    "DatasetFinding",
    "DatasetFileReport",
    "DatasetGuardError",
    "DatasetGuardReport",
    "DatasetUploadBlockedError",
    "DatasetUploadError",
    "DatasetUploadGuard",
    "DatasetUploadResult",
    "guard_dataset_upload",
    "inspect_dataset_files",
    "redact_text",
    "scan_dataset_files",
    "scan_text",
]
