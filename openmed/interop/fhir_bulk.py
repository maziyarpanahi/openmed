"""Streaming FHIR Bulk Data NDJSON de-identification helpers.

FHIR Bulk Data Access exports resources as newline-delimited JSON files, often
one file per resource type. This module keeps that workflow local and
streaming: it reads one line at a time, applies the existing FHIR
``$de-identify`` operation logic, and writes de-identified NDJSON without
loading a whole export file into memory.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from .fhir_operations import Deidentifier, de_identify_resource

__all__ = [
    "BulkExportSummary",
    "FHIRNDJSONLineError",
    "NDJSONFileSummary",
    "NDJSONLineError",
    "deidentify_export",
    "deidentify_ndjson_async",
    "deidentify_ndjson",
    "deidentify_ndjson_stream",
    "iter_ndjson",
]

_DEFAULT_POLICY = "hipaa_safe_harbor"
_DEFAULT_METHOD = "replace"


@dataclass(frozen=True)
class NDJSONLineError:
    """PHI-safe description of one failed NDJSON line.

    Attributes:
        line_number: One-based physical line number in the input file.
        message: Sanitized parse or processing error. The raw line content is
            intentionally excluded because it may contain PHI.
    """

    line_number: int
    message: str


class FHIRNDJSONLineError(ValueError):
    """Raised by :func:`iter_ndjson` when a line cannot be parsed as a resource."""

    def __init__(self, path: Path, line_number: int, message: str) -> None:
        self.path = path
        self.line_number = line_number
        self.message = message
        super().__init__(f"{path}: line {line_number}: {message}")

    def to_summary_error(self) -> NDJSONLineError:
        """Return a PHI-safe summary error for this parse failure."""

        return NDJSONLineError(
            line_number=self.line_number,
            message=self.message,
        )


@dataclass(frozen=True)
class NDJSONFileSummary:
    """Processing summary for one NDJSON input file."""

    source: str
    destination: str
    lines_processed: int = 0
    resources_deidentified: int = 0
    blank_lines: int = 0
    errors: tuple[NDJSONLineError, ...] = ()
    output_sha256: str = ""

    @property
    def error_count(self) -> int:
        """Return the number of malformed or unprocessable input lines."""

        return len(self.errors)

    @property
    def ok(self) -> bool:
        """Return whether the file completed without per-line errors."""

        return not self.errors


@dataclass(frozen=True)
class BulkExportSummary:
    """Aggregate summary for a directory-level bulk export pass."""

    input_dir: str
    output_dir: str
    files: tuple[NDJSONFileSummary, ...] = field(default_factory=tuple)

    @property
    def file_count(self) -> int:
        """Return the number of NDJSON files processed."""

        return len(self.files)

    @property
    def lines_processed(self) -> int:
        """Return the total number of physical lines read."""

        return sum(file.lines_processed for file in self.files)

    @property
    def resources_deidentified(self) -> int:
        """Return the total number of resources written."""

        return sum(file.resources_deidentified for file in self.files)

    @property
    def errors(self) -> tuple[NDJSONLineError, ...]:
        """Return all per-line errors across files."""

        return tuple(error for file in self.files for error in file.errors)


def iter_ndjson(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield parsed FHIR resources from an NDJSON file lazily.

    Blank lines are skipped. Malformed JSON lines and non-object JSON values
    raise :class:`FHIRNDJSONLineError` with the physical line number.

    Args:
        path: File containing one JSON resource per line.

    Yields:
        Parsed resource mappings in input order.
    """

    source = Path(path)
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            yield _parse_resource_line(source, line, line_number)


def deidentify_ndjson(
    in_path: str | Path,
    out_path: str | Path,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> NDJSONFileSummary:
    """Stream one NDJSON file through FHIR resource de-identification.

    Malformed or unprocessable lines are recorded in the returned summary and
    skipped, allowing later valid lines to continue processing. Error messages
    include line numbers but never raw line content.

    Args:
        in_path: Input NDJSON file.
        out_path: Destination NDJSON file.
        policy: Privacy policy profile passed to the FHIR operation wrapper.
        method: De-identification method passed to the FHIR operation wrapper.
        deidentifier: Optional privacy pipeline override, mainly for tests.

    Returns:
        A file summary with line counts, successful resources, and errors.
    """

    source = Path(in_path)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with (
        source.open("r", encoding="utf-8") as input_stream,
        destination.open("w", encoding="utf-8") as output_stream,
    ):
        return deidentify_ndjson_stream(
            input_stream,
            output_stream,
            source=source,
            destination=destination,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
        )


def deidentify_ndjson_stream(
    lines: Iterable[str],
    output_stream: TextIO,
    *,
    source: str | Path = "<stream>",
    destination: str | Path = "<stream>",
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> NDJSONFileSummary:
    """Stream NDJSON text lines through FHIR resource de-identification.

    This is the file-like equivalent of :func:`deidentify_ndjson`. It is used
    by network ingestion paths that must avoid persisting raw pre-deidentified
    resources to disk.

    Args:
        lines: Iterable yielding one NDJSON line at a time.
        output_stream: Text stream that receives de-identified NDJSON.
        source: PHI-safe source label used only in summaries and errors.
        destination: PHI-safe destination label used only in summaries.
        policy: Privacy policy profile passed to the FHIR operation wrapper.
        method: De-identification method passed to the FHIR operation wrapper.
        deidentifier: Optional privacy pipeline override, mainly for tests.

    Returns:
        A file summary with counts, sanitized errors, and the SHA-256 digest of
        the de-identified NDJSON bytes that were written.
    """

    processor = _NDJSONStreamProcessor(
        source=source,
        destination=destination,
        output_stream=output_stream,
        policy=policy,
        method=method,
        deidentifier=deidentifier,
    )
    for line_number, line in enumerate(lines, start=1):
        processor.process_line(line, line_number)
    return processor.summary()


async def deidentify_ndjson_async(
    lines: AsyncIterable[str],
    out_path: str | Path,
    *,
    source: str | Path = "<stream>",
    destination: str | Path | None = None,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> NDJSONFileSummary:
    """Stream async NDJSON lines to a de-identified NDJSON file.

    The input is consumed one line at a time and is never materialized as a
    complete export file. ``destination`` lets callers report a final atomic
    target path even when ``out_path`` is a temporary partial file.
    """

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_destination = output_path if destination is None else destination
    with output_path.open("w", encoding="utf-8") as output_stream:
        processor = _NDJSONStreamProcessor(
            source=source,
            destination=summary_destination,
            output_stream=output_stream,
            policy=policy,
            method=method,
            deidentifier=deidentifier,
        )
        line_number = 0
        async for line in lines:
            line_number += 1
            processor.process_line(line, line_number)
        return processor.summary()


@dataclass
class _NDJSONStreamProcessor:
    source: str | Path
    destination: str | Path
    output_stream: TextIO
    policy: str
    method: str
    deidentifier: Deidentifier | None
    lines_processed: int = 0
    resources_deidentified: int = 0
    blank_lines: int = 0
    errors: list[NDJSONLineError] = field(default_factory=list)
    _digest: Any = field(default_factory=hashlib.sha256)

    def process_line(self, line: str, line_number: int) -> None:
        """Process one physical NDJSON line."""

        self.lines_processed += 1
        if not line.strip():
            self.blank_lines += 1
            return

        try:
            resource = _parse_resource_line(Path(str(self.source)), line, line_number)
            transformed = de_identify_resource(
                resource,
                policy=self.policy,
                method=self.method,
                deidentifier=self.deidentifier,
            )
        except FHIRNDJSONLineError as exc:
            self.errors.append(exc.to_summary_error())
            return
        except (TypeError, ValueError) as exc:
            self.errors.append(
                NDJSONLineError(
                    line_number=line_number,
                    message=_sanitize_error_message(exc),
                )
            )
            return

        encoded = json.dumps(
            transformed,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self.output_stream.write(encoded.decode("utf-8"))
        self.output_stream.write("\n")
        self._digest.update(encoded)
        self._digest.update(b"\n")
        self.resources_deidentified += 1

    def summary(self) -> NDJSONFileSummary:
        """Return the PHI-safe summary for all processed lines."""

        return NDJSONFileSummary(
            source=str(self.source),
            destination=str(self.destination),
            lines_processed=self.lines_processed,
            resources_deidentified=self.resources_deidentified,
            blank_lines=self.blank_lines,
            errors=tuple(self.errors),
            output_sha256=self._digest.hexdigest(),
        )


def deidentify_export(
    in_dir: str | Path,
    out_dir: str | Path,
    *,
    policy: str = _DEFAULT_POLICY,
    method: str = _DEFAULT_METHOD,
    deidentifier: Deidentifier | None = None,
) -> BulkExportSummary:
    """De-identify every ``*.ndjson`` file in a FHIR bulk export directory.

    The output directory mirrors the input directory's relative NDJSON file
    layout. Non-NDJSON files are ignored.

    Args:
        in_dir: Directory containing per-resource-type NDJSON files.
        out_dir: Destination directory for de-identified NDJSON files.
        policy: Privacy policy profile passed to each file processor.
        method: De-identification method passed to each file processor.
        deidentifier: Optional privacy pipeline override, mainly for tests.

    Returns:
        An aggregate summary with one :class:`NDJSONFileSummary` per file.

    Raises:
        ValueError: If ``in_dir`` is not a directory.
    """

    source_root = Path(in_dir)
    destination_root = Path(out_dir)
    if not source_root.is_dir():
        raise ValueError(f"input export path is not a directory: {source_root}")
    destination_root.mkdir(parents=True, exist_ok=True)

    summaries: list[NDJSONFileSummary] = []
    for source in sorted(source_root.rglob("*.ndjson")):
        if not source.is_file():
            continue
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        summaries.append(
            deidentify_ndjson(
                source,
                destination,
                policy=policy,
                method=method,
                deidentifier=deidentifier,
            )
        )

    return BulkExportSummary(
        input_dir=str(source_root),
        output_dir=str(destination_root),
        files=tuple(summaries),
    )


def _parse_resource_line(
    path: Path,
    line: str,
    line_number: int,
) -> dict[str, Any]:
    """Parse one NDJSON line as a FHIR resource mapping."""

    try:
        raw = json.loads(_strip_initial_bom(line, line_number))
    except json.JSONDecodeError as exc:
        raise FHIRNDJSONLineError(
            path,
            line_number,
            f"malformed JSON: {exc.msg}",
        ) from exc

    if not isinstance(raw, dict):
        raise FHIRNDJSONLineError(
            path,
            line_number,
            "line must contain a JSON object",
        )
    return raw


def _strip_initial_bom(line: str, line_number: int) -> str:
    if line_number == 1:
        return line.lstrip("\ufeff")
    return line


def _sanitize_error_message(exc: BaseException) -> str:
    message = str(exc).strip()
    return message or exc.__class__.__name__
