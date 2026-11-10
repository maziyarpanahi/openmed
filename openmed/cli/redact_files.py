"""Offline redaction of ordinary text and line-delimited files."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from openmed.core.config import get_config
from openmed.core.offline import network_blocked_if_offline
from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.core.policy import canonical_policy_name

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, emit

DEFAULT_PII_MODEL = DEFAULT_PII_MODELS["en"]
_FileRedactionMethod = Literal[
    "mask",
    "aadhaar_mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
]
REDACTION_METHODS: tuple[_FileRedactionMethod, ...] = (
    "mask",
    "aadhaar_mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
)
_FORMAT_ALIASES = {
    "auto": "auto",
    "text": "text",
    "txt": "text",
    "lines": "lines",
    "line": "lines",
    "line-delimited": "lines",
    "jsonl": "jsonl",
    "ndjson": "jsonl",
}
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


@dataclass(frozen=True)
class RedactionOffset:
    """PHI-free location metadata for one redacted span."""

    document: int
    start: int
    end: int
    label: str
    replacement_length: int
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return counts and offsets without the source span value."""
        payload: dict[str, Any] = {
            "document": self.document,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "replacement_length": self.replacement_length,
        }
        if self.line is not None:
            payload["line"] = self.line
        return payload


@dataclass
class FileRedactionSummary:
    """Aggregate, PHI-free outcome for a file redaction run."""

    input_format: str
    documents: int
    redacted_documents: int
    total_spans: int
    per_label_counts: dict[str, int]
    offsets: list[RedactionOffset]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the summary without source values or file paths."""
        counts = dict(sorted(self.per_label_counts.items()))
        return {
            "format": self.input_format,
            "input_format": self.input_format,
            "documents": self.documents,
            "redacted_documents": self.redacted_documents,
            "total_spans": self.total_spans,
            "per_label_counts": counts,
            "offsets": [offset.to_dict() for offset in self.offsets],
        }


@dataclass
class FileRedactionResult:
    """Redacted output path and its PHI-free summary."""

    output_path: Path
    summary: FileRedactionSummary


def add_redact_files_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed redact-files`` command."""
    parser = subparsers.add_parser(
        "redact-files",
        aliases=("redact-file", "redact"),
        help="Redact a local text or line-delimited file offline.",
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        help="Input file path. Use --input instead when preferred.",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Output file path. Use --output instead when preferred.",
    )
    parser.add_argument(
        "--input",
        "--input-path",
        dest="input_option",
        type=Path,
        help="Input file path.",
    )
    parser.add_argument(
        "--output",
        "--output-path",
        "-o",
        dest="output_option",
        type=Path,
        help="Output file path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional PHI-free JSON report path.",
    )
    parser.add_argument(
        "--format",
        "--input-format",
        dest="input_format",
        type=_format_arg,
        default="auto",
        help="Input format: auto, text, lines, or jsonl (default: auto).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input and redacted-output text encoding (default: utf-8).",
    )
    parser.add_argument(
        "--policy",
        type=_policy_arg,
        default="hipaa_safe_harbor",
        help="Policy profile to apply.",
    )
    parser.add_argument(
        "--lang",
        "--language",
        default="en",
        help="Language hint for detection and replacement (default: en).",
    )
    parser.add_argument(
        "--method",
        choices=REDACTION_METHODS,
        default="mask",
        help="De-identification method (default: mask).",
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        default=DEFAULT_PII_MODEL,
        help="PII model or local model path.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=_confidence_arg,
        default=0.7,
        help="Minimum confidence for redaction (default: 0.7).",
    )
    parser.add_argument(
        "--keep-year",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep the year in dates where applicable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed for replacement and date shifting (default: 0).",
    )
    parser.add_argument(
        "--locale",
        default=None,
        help="Optional Faker locale for replacement methods.",
    )
    sweep_group = parser.add_mutually_exclusive_group()
    sweep_group.add_argument(
        "--use-safety-sweep",
        "--safety-sweep",
        dest="use_safety_sweep",
        action="store_true",
        help="Enable deterministic structured-identifier detection (default).",
    )
    sweep_group.add_argument(
        "--no-safety-sweep",
        dest="use_safety_sweep",
        action="store_false",
        help="Disable deterministic structured-identifier detection.",
    )
    parser.set_defaults(handler=run_from_args, use_safety_sweep=True)


def run_from_args(args: argparse.Namespace, *, config: Any | None = None) -> int:
    """Run file redaction from parsed CLI arguments."""
    if config is None:
        from .main import _load_and_apply_config

        config = _load_and_apply_config(args)

    input_path, output_path = _resolve_paths(args)
    report_path = getattr(args, "report", None)
    if report_path is not None:
        _ensure_distinct_paths(input_path, report_path, "input and report")
        _ensure_distinct_paths(output_path, report_path, "output and report")

    try:
        result = redact_file(
            input_path,
            output_path,
            format=getattr(args, "input_format", getattr(args, "format", "auto")),
            encoding=getattr(args, "encoding", "utf-8"),
            policy=getattr(args, "policy", "hipaa_safe_harbor"),
            lang=getattr(args, "lang", getattr(args, "language", "en")),
            method=getattr(args, "method", "mask"),
            model_name=getattr(args, "model_name", DEFAULT_PII_MODEL),
            confidence_threshold=getattr(args, "confidence_threshold", 0.7),
            keep_year=getattr(args, "keep_year", False),
            seed=getattr(args, "seed", 0),
            locale=getattr(args, "locale", None),
            use_safety_sweep=getattr(args, "use_safety_sweep", True),
            config=config,
        )
    except FileNotFoundError:
        raise CliError(
            "Input file was not found.",
            code="input_not_found",
            exit_code=EXIT_ERROR,
        ) from None
    except UnicodeError:
        raise CliError(
            "Input file could not be decoded with the selected encoding.",
            code="encoding_error",
            exit_code=EXIT_ERROR,
        ) from None
    except ValueError:
        raise CliError(
            "File redaction arguments or input are invalid.",
            code="invalid_argument",
            exit_code=EXIT_USAGE,
        ) from None
    except OSError:
        raise CliError(
            "File redaction could not read or write the requested files.",
            code="file_error",
            exit_code=EXIT_ERROR,
        ) from None
    except Exception:
        raise CliError(
            "File redaction failed.",
            code="redaction_failed",
            exit_code=EXIT_ERROR,
        ) from None

    payload = result.summary.to_dict()
    if report_path is not None:
        try:
            _write_report(report_path, payload)
        except OSError:
            raise CliError(
                "The redaction report could not be written.",
                code="report_error",
                exit_code=EXIT_ERROR,
            ) from None

    human = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    return emit(args, payload, human=human)


def redact_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    format: str = "auto",
    input_format: str | None = None,
    encoding: str = "utf-8",
    policy: str = "hipaa_safe_harbor",
    lang: str = "en",
    method: _FileRedactionMethod = "mask",
    model_name: str = DEFAULT_PII_MODEL,
    confidence_threshold: float = 0.7,
    keep_year: bool = False,
    seed: int = 0,
    locale: str | None = None,
    use_safety_sweep: bool = True,
    config: Any | None = None,
    deidentify_fn: Callable[..., Any] | None = None,
) -> FileRedactionResult:
    """Redact a local file without making network requests.

    ``text`` format treats the complete file as one document. ``lines`` and
    ``jsonl`` formats treat each non-empty physical line as one document and
    preserve the original line endings. JSONL is intentionally transformed as
    text so the command remains useful for arbitrary support exports without
    guessing a schema or selecting fields.

    The replacement path is always seeded and consistent. Callers must provide
    a cached model or a local model path; the local-only guard prevents an
    implicit model download.
    """
    source = Path(input_path)
    target = Path(output_path)
    _ensure_distinct_paths(source, target, "input and output")
    if input_format is not None:
        if format != "auto" and format != input_format:
            raise ValueError("format and input_format disagree")
        format = input_format
    resolved_format = _resolve_format(source, format)
    if method not in REDACTION_METHODS:
        raise ValueError("unsupported redaction method")
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between 0 and 1")

    with source.open("r", encoding=encoding, newline="") as handle:
        source_text = handle.read()

    if config is None:
        config = get_config()

    offsets: list[RedactionOffset] = []
    label_counts: Counter[str] = Counter()
    redacted_documents = 0
    if resolved_format == "text":
        documents = 1
        if source_text.strip():
            with network_blocked_if_offline(config, local_only=True):
                redacted_text, unit_offsets = _redact_unit(
                    source_text,
                    document=0,
                    line=1,
                    method=method,
                    model_name=model_name,
                    confidence_threshold=confidence_threshold,
                    keep_year=keep_year,
                    seed=seed,
                    locale=locale,
                    policy=policy,
                    lang=lang,
                    use_safety_sweep=use_safety_sweep,
                    config=config,
                    deidentify_fn=deidentify_fn,
                )
            offsets.extend(unit_offsets)
            redacted_documents = int(redacted_text != source_text)
        else:
            redacted_text = source_text
    else:
        redacted_text, documents, redacted_documents, line_offsets = _redact_lines(
            source_text,
            method=method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            keep_year=keep_year,
            seed=seed,
            locale=locale,
            policy=policy,
            lang=lang,
            use_safety_sweep=use_safety_sweep,
            config=config,
            deidentify_fn=deidentify_fn,
        )
        offsets.extend(line_offsets)

    for offset in offsets:
        label_counts[offset.label] += 1

    _write_text(target, redacted_text, encoding)
    summary = FileRedactionSummary(
        input_format=resolved_format,
        documents=documents,
        redacted_documents=redacted_documents,
        total_spans=len(offsets),
        per_label_counts=dict(label_counts),
        offsets=offsets,
    )
    return FileRedactionResult(output_path=target, summary=summary)


def _redact_lines(
    source_text: str,
    *,
    method: _FileRedactionMethod,
    model_name: str,
    confidence_threshold: float,
    keep_year: bool,
    seed: int,
    locale: str | None,
    policy: str,
    lang: str,
    use_safety_sweep: bool,
    config: Any,
    deidentify_fn: Callable[..., Any] | None,
) -> tuple[str, int, int, list[RedactionOffset]]:
    """Redact physical lines while preserving their exact delimiters."""
    output_parts: list[str] = []
    offsets: list[RedactionOffset] = []
    documents = 0
    redacted_documents = 0
    with network_blocked_if_offline(config, local_only=True):
        for line_number, raw_line in enumerate(
            source_text.splitlines(keepends=True),
            start=1,
        ):
            line, ending = _split_line_ending(raw_line)
            if not line.strip():
                output_parts.append(raw_line)
                continue
            documents += 1
            redacted_line, line_offsets = _redact_unit(
                line,
                document=documents - 1,
                line=line_number,
                method=method,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                keep_year=keep_year,
                seed=seed,
                locale=locale,
                policy=policy,
                lang=lang,
                use_safety_sweep=use_safety_sweep,
                config=config,
                deidentify_fn=deidentify_fn,
            )
            output_parts.append(redacted_line + ending)
            offsets.extend(line_offsets)
            redacted_documents += int(redacted_line != line)

    return "".join(output_parts), documents, redacted_documents, offsets


def _redact_unit(
    text: str,
    *,
    document: int,
    line: int | None,
    method: _FileRedactionMethod,
    model_name: str,
    confidence_threshold: float,
    keep_year: bool,
    seed: int,
    locale: str | None,
    policy: str,
    lang: str,
    use_safety_sweep: bool,
    config: Any,
    deidentify_fn: Callable[..., Any] | None,
) -> tuple[str, list[RedactionOffset]]:
    """Redact one text unit and retain only safe span metadata."""
    if deidentify_fn is None:
        from openmed.core.pii import deidentify

        deidentify_fn = deidentify

    result = deidentify_fn(
        text,
        method=method,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        keep_year=keep_year,
        config=config,
        policy=policy,
        lang=lang,
        use_safety_sweep=use_safety_sweep,
        consistent=True,
        seed=seed,
        locale=locale,
    )
    redacted = getattr(result, "deidentified_text", None)
    if not isinstance(redacted, str):
        raise ValueError("de-identification returned no text")

    offset_shift = 0
    original = getattr(result, "original_text", None)
    if isinstance(original, str) and original != text and original == text.strip():
        leading = text[: len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()) :]
        redacted = leading + redacted + trailing
        offset_shift = len(leading)

    offsets: list[RedactionOffset] = []
    for entity in getattr(result, "pii_entities", ()) or ():
        start = _safe_offset(getattr(entity, "start", None))
        end = _safe_offset(getattr(entity, "end", None))
        if start is None or end is None or start < 0 or end < start:
            continue
        start += offset_shift
        end += offset_shift
        if end > len(text):
            continue
        label = _safe_label(getattr(entity, "label", None))
        replacement = getattr(entity, "redacted_text", None)
        replacement_length = len(replacement) if isinstance(replacement, str) else 0
        offsets.append(
            RedactionOffset(
                document=document,
                line=line,
                start=start,
                end=end,
                label=label,
                replacement_length=replacement_length,
            )
        )
    offsets.sort(key=lambda item: (item.document, item.start, item.end, item.label))
    return redacted, offsets


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve positional and option path spellings without echoing values."""
    positional_input = getattr(args, "input_path", None)
    positional_output = getattr(args, "output_path", None)
    option_input = getattr(args, "input_option", getattr(args, "input", None))
    option_output = getattr(args, "output_option", getattr(args, "output", None))
    if positional_input is not None and option_input is not None:
        raise CliError(
            "Specify the input path once, either positionally or with --input.",
            code="duplicate_input",
            exit_code=EXIT_USAGE,
        )
    if positional_output is not None and option_output is not None:
        raise CliError(
            "Specify the output path once, either positionally or with --output.",
            code="duplicate_output",
            exit_code=EXIT_USAGE,
        )
    input_path = option_input or positional_input
    output_path = option_output or positional_output
    if input_path is None or output_path is None:
        raise CliError(
            "Both an input path and an output path are required.",
            code="missing_paths",
            exit_code=EXIT_USAGE,
        )
    return Path(input_path), Path(output_path)


def _ensure_distinct_paths(first: Path, second: Path, description: str) -> None:
    try:
        same_path = first.resolve() == second.resolve()
    except OSError:
        same_path = first.absolute() == second.absolute()
    if same_path:
        raise CliError(
            f"The {description} paths must be different.",
            code="same_path",
            exit_code=EXIT_USAGE,
        )


def _resolve_format(path: Path, requested: str) -> str:
    normalized = _FORMAT_ALIASES.get(str(requested).strip().lower())
    if normalized is None:
        raise ValueError("unsupported input format")
    if normalized != "auto":
        return normalized
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return "jsonl"
    return "text"


def _format_arg(value: str) -> str:
    normalized = _FORMAT_ALIASES.get(value.strip().lower())
    if normalized is None:
        raise argparse.ArgumentTypeError(
            "format must be one of: auto, text, lines, jsonl, ndjson"
        )
    return normalized


def _policy_arg(value: str) -> str:
    try:
        return canonical_policy_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _confidence_arg(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("confidence must be a number") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("confidence must be between 0 and 1")
    return parsed


def _safe_offset(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed


def _safe_label(value: Any) -> str:
    label = str(value or "UNKNOWN")
    return label if _SAFE_LABEL.fullmatch(label) else "UNKNOWN"


def _split_line_ending(value: str) -> tuple[str, str]:
    if value.endswith("\r\n"):
        return value[:-2], "\r\n"
    if value.endswith(("\n", "\r")):
        return value[:-1], value[-1]
    return value, ""


def _write_text(path: Path, value: str, encoding: str) -> None:
    _atomic_write(path, value, encoding=encoding)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    report = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write(path, report, encoding="utf-8")


def _atomic_write(path: Path, value: str, *, encoding: str) -> None:
    parent = path.parent
    temporary_path: Path | None = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=".openmed-redact-",
            suffix=".tmp",
            dir=str(parent),
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(
            file_descriptor,
            "w",
            encoding=encoding,
            newline="",
        ) as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "DEFAULT_PII_MODEL",
    "FileRedactionResult",
    "FileRedactionSummary",
    "RedactionOffset",
    "REDACTION_METHODS",
    "add_redact_files_command",
    "redact_file",
    "run_from_args",
]
