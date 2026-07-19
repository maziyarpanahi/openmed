"""Executable UDF adapter for streaming column redaction over stdin/stdout.

The adapter implements a one-column ClickHouse ``TabSeparated`` transform:
each input row contains one escaped string value and each output row contains
the corresponding redacted value. Input rows are buffered into bounded batches
before they are passed to :func:`openmed.processing.batch.process_batch`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from openmed.core import ModelLoader
from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.processing.batch import process_batch

DEFAULT_EXECUTABLE_UDF_MODEL = DEFAULT_PII_MODELS["en"]

_TSV_DECODE_ESCAPES = {
    "0": "\0",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "\\": "\\",
    "'": "'",
}
_TSV_ENCODE_ESCAPES = {
    "\\": "\\\\",
    "\0": "\\0",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


class ExecutableUDFError(RuntimeError):
    """Raised when an executable UDF batch cannot be redacted safely."""


@dataclass(frozen=True)
class ExecutableUDFConfig:
    """Configuration for the executable UDF stream.

    Args:
        batch_size: Maximum number of rows passed to one batch call.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.
    """

    batch_size: int = 32
    model_name: str = DEFAULT_EXECUTABLE_UDF_MODEL
    method: str = "mask"
    confidence_threshold: float | None = 0.7
    deidentify_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


def redact_tsv_lines(
    lines: Iterable[str],
    *,
    batch_size: int = 32,
    model_name: str = DEFAULT_EXECUTABLE_UDF_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    config: Any | None = None,
    loader: Any | None = None,
    **deidentify_kwargs: Any,
) -> Iterator[str]:
    """Yield redacted one-column TabSeparated rows.

    A single shared model loader is reused for every buffered batch so a model
    is not reloaded as the stream advances. Blank values are emitted as blank
    output rows without calling the model; no input rows are skipped.

    Args:
        lines: Iterable containing one ClickHouse TabSeparated string per row.
        batch_size: Maximum number of rows passed to one batch call.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        config: Optional OpenMed configuration.
        loader: Optional shared model loader. A loader is created once when
            omitted.
        **deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.

    Yields:
        Escaped redacted values with exactly one trailing newline per row.

    Raises:
        ExecutableUDFError: If input is not one-column TSV or batch redaction
            fails, returns the wrong cardinality, or returns invalid results.
    """

    udf_config = ExecutableUDFConfig(
        batch_size=batch_size,
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        deidentify_kwargs=deidentify_kwargs,
    )
    shared_loader = loader if loader is not None else ModelLoader(config)
    batch: list[str] = []

    for line_number, raw_line in enumerate(lines, start=1):
        batch.append(_parse_tsv_row(raw_line, line_number=line_number))
        if len(batch) >= udf_config.batch_size:
            yield from _redact_batch(
                batch,
                udf_config,
                config=config,
                loader=shared_loader,
            )
            batch = []

    if batch:
        yield from _redact_batch(
            batch,
            udf_config,
            config=config,
            loader=shared_loader,
        )


def redact_tsv_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    batch_size: int = 32,
    model_name: str = DEFAULT_EXECUTABLE_UDF_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    config: Any | None = None,
    loader: Any | None = None,
    **deidentify_kwargs: Any,
) -> int:
    """Redact a one-column TabSeparated stream.

    Args:
        input_stream: Text stream containing one escaped string per row.
        output_stream: Text stream receiving one redacted string per row.
        batch_size: Maximum number of rows passed to one batch call.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        config: Optional OpenMed configuration.
        loader: Optional shared model loader.
        **deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.

    Returns:
        Number of rows emitted.
    """

    emitted = 0
    for redacted_line in redact_tsv_lines(
        input_stream,
        batch_size=batch_size,
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        config=config,
        loader=loader,
        **deidentify_kwargs,
    ):
        output_stream.write(redacted_line)
        emitted += 1
        if emitted % batch_size == 0:
            output_stream.flush()
    output_stream.flush()
    return emitted


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the executable UDF stdin/stdout transform.

    Args:
        argv: Optional command-line arguments.
        stdin: Optional input stream override.
        stdout: Optional output stream override.
        stderr: Optional error stream override.

    Returns:
        Zero on success and one when the transform fails.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)
    input_stream = stdin if stdin is not None else sys.stdin
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr

    try:
        redact_tsv_stream(
            input_stream,
            output_stream,
            batch_size=args.batch_size,
            model_name=args.model_name,
            method=args.method,
            confidence_threshold=args.confidence_threshold,
            lang=args.lang,
            policy=args.policy,
            keep_year=args.keep_year,
            use_safety_sweep=args.use_safety_sweep,
        )
    except Exception:
        error_stream.write("executable UDF redaction failed\n")
        return 1

    return 0


def _redact_batch(
    texts: Sequence[str],
    udf_config: ExecutableUDFConfig,
    *,
    config: Any | None,
    loader: Any,
) -> Iterator[str]:
    nonempty_positions = [index for index, text in enumerate(texts) if text]
    if not nonempty_positions:
        yield from ("\n" for _ in texts)
        return

    batch_texts = [texts[index] for index in nonempty_positions]
    kwargs = dict(udf_config.deidentify_kwargs)
    kwargs.update(
        {
            "operation": "deidentify",
            "batch_size": len(batch_texts),
            "continue_on_error": False,
            "loader": loader,
            "method": udf_config.method,
            "confidence_threshold": udf_config.confidence_threshold,
        }
    )

    try:
        result = process_batch(
            batch_texts,
            model_name=udf_config.model_name,
            config=config,
            **kwargs,
        )
    except Exception:
        raise ExecutableUDFError("failed to redact an input batch") from None

    items = list(getattr(result, "items", []))
    if len(items) != len(batch_texts):
        raise ExecutableUDFError("batch result count did not match input row count")

    redacted_values: list[str | None] = [None] * len(texts)
    for index, item in zip(nonempty_positions, items):
        redacted_values[index] = _redacted_text(item)

    for redacted in redacted_values:
        yield "\n" if redacted is None else _encode_tsv_value(redacted) + "\n"


def _redacted_text(item: Any) -> str:
    if not getattr(item, "success", False):
        raise ExecutableUDFError("one or more input rows could not be redacted")
    redacted = getattr(getattr(item, "result", None), "deidentified_text", None)
    if not isinstance(redacted, str):
        raise ExecutableUDFError("batch redaction returned an invalid result")
    return redacted


def _parse_tsv_row(raw_line: str, *, line_number: int) -> str:
    row = raw_line.removesuffix("\n").removesuffix("\r")
    if "\t" in row:
        raise ExecutableUDFError(
            f"input line {line_number} contains more than one TSV column"
        )
    return _decode_tsv_value(row)


def _decode_tsv_value(value: str) -> str:
    decoded: list[str] = []
    index = 0
    while index < len(value):
        character = value[index]
        if character != "\\" or index + 1 == len(value):
            decoded.append(character)
            index += 1
            continue

        escaped = value[index + 1]
        decoded.append(_TSV_DECODE_ESCAPES.get(escaped, escaped))
        index += 2
    return "".join(decoded)


def _encode_tsv_value(value: str) -> str:
    return "".join(_TSV_ENCODE_ESCAPES.get(character, character) for character in value)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redact one ClickHouse TabSeparated string per stdin row.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=32,
        help="Number of rows buffered before batch redaction.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EXECUTABLE_UDF_MODEL,
        help="PII model identifier passed to OpenMed batch de-identification.",
    )
    parser.add_argument(
        "--method",
        default="mask",
        choices=("mask", "remove", "replace", "hash", "shift_dates"),
        help="De-identification method.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum PII detection confidence.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Language hint forwarded to de-identification.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Optional policy profile forwarded to de-identification.",
    )
    parser.add_argument(
        "--keep-year",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep or redact the year in detected dates.",
    )
    parser.add_argument(
        "--use-safety-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable deterministic structured-identifier detection.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
