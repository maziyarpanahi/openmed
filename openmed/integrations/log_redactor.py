"""NDJSON log-event redaction filters for structured event streams."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TextIO

from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.processing.batch import process_batch

DEFAULT_LOG_MESSAGE_FIELDS = ("message",)
DEFAULT_LOG_REDACTION_MODEL = DEFAULT_PII_MODELS["en"]


class LogRedactorError(RuntimeError):
    """Raised when a log stream cannot be redacted safely."""


@dataclass(frozen=True)
class LogRedactorConfig:
    """Configuration for structured log event redaction.

    Args:
        message_fields: Top-level or dotted event fields containing log text.
        batch_size: Number of events to accumulate before redaction.
        model_name: PII model passed to :func:`openmed.process_batch`.
        method: De-identification method passed to batch de-identification.
        confidence_threshold: Detection confidence threshold.
        deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.
    """

    message_fields: tuple[str, ...] = DEFAULT_LOG_MESSAGE_FIELDS
    batch_size: int = 16
    model_name: str = DEFAULT_LOG_REDACTION_MODEL
    method: str = "mask"
    confidence_threshold: float | None = 0.7
    deidentify_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        fields = _normalize_message_fields(self.message_fields)
        object.__setattr__(self, "message_fields", fields)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class _Target:
    event_index: int
    field_name: str
    path: tuple[str, ...]
    text: str


def redact_log_events(
    events: Iterable[Mapping[str, Any]],
    *,
    message_fields: Sequence[str] = DEFAULT_LOG_MESSAGE_FIELDS,
    batch_size: int = 16,
    model_name: str = DEFAULT_LOG_REDACTION_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    **deidentify_kwargs: Any,
) -> Iterator[dict[str, Any]]:
    """Yield redacted copies of structured log events.

    Only configured string fields are redacted. Missing fields and non-string
    fields are preserved unchanged, as are all non-target event fields.

    Args:
        events: Structured log events to redact.
        message_fields: Top-level or dotted event fields containing log text.
        batch_size: Number of events to accumulate before redaction.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        **deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.

    Yields:
        Redacted event copies in the same order as the input events.
    """

    config = LogRedactorConfig(
        message_fields=tuple(message_fields),
        batch_size=batch_size,
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        deidentify_kwargs=deidentify_kwargs,
    )

    batch: list[Mapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            raise TypeError("log events must be mappings")
        batch.append(event)
        if len(batch) >= config.batch_size:
            yield from _redact_event_batch(batch, config)
            batch = []

    if batch:
        yield from _redact_event_batch(batch, config)


def redact_ndjson_stream(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    message_fields: Sequence[str] = DEFAULT_LOG_MESSAGE_FIELDS,
    batch_size: int = 16,
    model_name: str = DEFAULT_LOG_REDACTION_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    **deidentify_kwargs: Any,
) -> int:
    """Redact NDJSON events from ``input_stream`` into ``output_stream``.

    Args:
        input_stream: Text stream containing one JSON object per line.
        output_stream: Text stream receiving redacted JSON objects.
        message_fields: Top-level or dotted event fields containing log text.
        batch_size: Number of events to accumulate before redaction.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        **deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.

    Returns:
        Number of JSON events emitted. Blank input lines are ignored.
    """

    emitted = 0
    for event in redact_log_events(
        _iter_ndjson_events(input_stream),
        message_fields=message_fields,
        batch_size=batch_size,
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        **deidentify_kwargs,
    ):
        output_stream.write(_to_ndjson_line(event))
        emitted += 1
    return emitted


def redact_ndjson_lines(
    lines: Iterable[str],
    *,
    message_fields: Sequence[str] = DEFAULT_LOG_MESSAGE_FIELDS,
    batch_size: int = 16,
    model_name: str = DEFAULT_LOG_REDACTION_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    **deidentify_kwargs: Any,
) -> Iterator[str]:
    """Yield redacted NDJSON lines from an iterable of raw lines.

    Args:
        lines: Raw NDJSON input lines.
        message_fields: Top-level or dotted event fields containing log text.
        batch_size: Number of events to accumulate before redaction.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        **deidentify_kwargs: Extra keyword arguments forwarded to batch
            de-identification.

    Yields:
        Redacted NDJSON lines.
    """

    for event in redact_log_events(
        _iter_ndjson_events(lines),
        message_fields=message_fields,
        batch_size=batch_size,
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        **deidentify_kwargs,
    ):
        yield _to_ndjson_line(event)


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the stdin/stdout NDJSON redaction transform."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    input_stream = stdin if stdin is not None else sys.stdin
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr

    try:
        redact_ndjson_stream(
            input_stream,
            output_stream,
            message_fields=tuple(args.field),
            batch_size=args.batch_size,
            model_name=args.model_name,
            method=args.method,
            confidence_threshold=args.confidence_threshold,
            lang=args.lang,
            policy=args.policy,
            use_safety_sweep=args.use_safety_sweep,
        )
    except (LogRedactorError, TypeError, ValueError) as exc:
        error_stream.write(f"log redaction failed: {exc}\n")
        return 1

    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Redact PHI from configured fields in NDJSON log events.",
    )
    parser.add_argument(
        "--field",
        action="append",
        default=[],
        help=(
            "Log event field to redact. Repeat for multiple fields. "
            "Dotted paths such as error.message are supported."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of events to buffer before redaction.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_LOG_REDACTION_MODEL,
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
        "--use-safety-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable regex safety sweep fallback.",
    )
    return parser


def _iter_ndjson_events(input_stream: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line_number, raw_line in enumerate(input_stream, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogRedactorError(
                f"invalid JSON object at input line {line_number}"
            ) from None
        if not isinstance(event, dict):
            raise LogRedactorError(
                f"input line {line_number} must contain a JSON object"
            )
        yield event


def _to_ndjson_line(event: Mapping[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"


def _redact_event_batch(
    events: Sequence[Mapping[str, Any]],
    config: LogRedactorConfig,
) -> list[dict[str, Any]]:
    outputs = [copy.deepcopy(dict(event)) for event in events]
    targets = _collect_targets(outputs, config.message_fields)
    if not targets:
        return outputs

    texts = [target.text for target in targets]
    ids = [f"event_{target.event_index}:{target.field_name}" for target in targets]

    try:
        batch_result = process_batch(
            texts,
            model_name=config.model_name,
            ids=ids,
            operation="deidentify",
            batch_size=config.batch_size,
            confidence_threshold=config.confidence_threshold,
            method=config.method,
            continue_on_error=False,
            **dict(config.deidentify_kwargs),
        )
    except Exception:
        raise LogRedactorError("failed to redact a log event batch") from None

    items = list(getattr(batch_result, "items", []))
    if len(items) != len(targets):
        raise LogRedactorError("log redaction returned an unexpected result count")

    for target, item in zip(targets, items):
        if not getattr(item, "success", False):
            raise LogRedactorError("failed to redact a configured log field")
        result = getattr(item, "result", None)
        redacted_text = getattr(result, "deidentified_text", None)
        if not isinstance(redacted_text, str):
            raise LogRedactorError("log redaction returned an invalid result")
        _set_path(outputs[target.event_index], target.path, redacted_text)

    return outputs


def _collect_targets(
    events: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[_Target]:
    targets: list[_Target] = []
    for event_index, event in enumerate(events):
        for field_name in fields:
            path = _resolve_path(event, field_name)
            if path is None:
                continue
            value = _get_path(event, path)
            if isinstance(value, str) and value:
                targets.append(
                    _Target(
                        event_index=event_index,
                        field_name=field_name,
                        path=path,
                        text=value,
                    )
                )
    return targets


def _normalize_message_fields(fields: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(field.strip() for field in fields if field.strip())
    return normalized or DEFAULT_LOG_MESSAGE_FIELDS


def _resolve_path(
    event: Mapping[str, Any],
    field_name: str,
) -> tuple[str, ...] | None:
    if field_name in event:
        return (field_name,)

    path = tuple(part for part in field_name.split(".") if part)
    if not path:
        return None
    try:
        _get_path(event, path)
    except (KeyError, TypeError):
        return None
    return path


def _get_path(event: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = event
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(part)
        current = current[part]
    return current


def _set_path(event: dict[str, Any], path: Sequence[str], value: str) -> None:
    current: dict[str, Any] = event
    for part in path[:-1]:
        next_value = current[part]
        if not isinstance(next_value, dict):
            raise LogRedactorError("configured log field path is not an object")
        current = next_value
    current[path[-1]] = value


if __name__ == "__main__":
    raise SystemExit(main())
