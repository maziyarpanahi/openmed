"""Script-processor adapter for in-flow record de-identification.

The public callable accepts one record plus its flow-file attributes and
returns a redacted copy with the attributes preserved. Only configured string
fields are submitted to OpenMed. Model loaders are cached at module scope so a
long-lived script processor can reuse its loaded pipeline across records.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TextIO

from openmed.core.pii_i18n import DEFAULT_PII_MODELS
from openmed.processing.batch import process_batch

DEFAULT_DATAFLOW_TOOL_MODEL = DEFAULT_PII_MODELS["en"]
RECORD_COUNT_ATTRIBUTE = "openmed.redaction.record_count"
FIELD_COUNT_ATTRIBUTE = "openmed.redaction.field_count"
ENTITY_COUNT_ATTRIBUTE = "openmed.redaction.entity_count"

ProcessBatch = Callable[..., Any]
_MISSING = object()
_RESERVED_DEIDENTIFY_KWARGS = frozenset(
    {
        "batch_size",
        "confidence_threshold",
        "continue_on_error",
        "ids",
        "loader",
        "method",
        "model_name",
        "operation",
        "texts",
    }
)


class DataflowToolProcessorError(RuntimeError):
    """Raised when a flow-file record cannot be redacted safely."""


@dataclass(frozen=True)
class DataflowToolConfig:
    """Configuration for record-level in-flow redaction.

    Args:
        fields: Top-level or dotted record fields to redact.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        deidentify_kwargs: Extra keyword arguments forwarded to
            :func:`openmed.processing.batch.process_batch`.
    """

    fields: tuple[str, ...]
    model_name: str = DEFAULT_DATAFLOW_TOOL_MODEL
    method: str = "mask"
    confidence_threshold: float | None = 0.7
    deidentify_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", _normalize_fields(self.fields))
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")

        reserved = _RESERVED_DEIDENTIFY_KWARGS.intersection(self.deidentify_kwargs)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(
                "deidentify kwargs must not override processor options: " + names
            )


@dataclass
class _CachedPipeline:
    loader: Any
    lock: RLock = field(default_factory=RLock)


@dataclass(frozen=True)
class _Target:
    path: tuple[str, ...]
    text: str


_PIPELINE_CACHE: dict[tuple[str, int | None], _CachedPipeline] = {}
_PIPELINE_CACHE_LOCK = RLock()


def process_flow_file(
    record: Mapping[str, Any],
    attributes: Mapping[str, str] | None = None,
    *,
    fields: Sequence[str] | str,
    model_name: str = DEFAULT_DATAFLOW_TOOL_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    config: Any | None = None,
    process_batch_fn: ProcessBatch | None = None,
    **deidentify_kwargs: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Return one redacted record and its PHI-safe flow-file attributes.

    The input mappings are never mutated. Incoming attributes pass through
    unchanged, while OpenMed adds only numeric record, changed-field, and
    detected-entity counts. Missing, null, empty, and non-string configured
    fields pass through unchanged.

    Args:
        record: Record-like flow-file content.
        attributes: Existing string flow-file attributes.
        fields: Top-level or dotted record fields to redact.
        model_name: PII model passed to batch de-identification.
        method: De-identification method.
        confidence_threshold: Minimum confidence for PII detection.
        config: Optional OpenMed configuration used by the cached model loader.
        process_batch_fn: Optional ``process_batch`` replacement, primarily for
            offline tests.
        **deidentify_kwargs: Extra batch de-identification options such as
            ``policy``, ``lang``, or ``use_safety_sweep``.

    Returns:
        A ``(record, attributes)`` tuple containing one output record for the
        one input record.

    Raises:
        TypeError: If the record or attributes are not mappings.
        DataflowToolProcessorError: If batch de-identification fails or returns
            an invalid result.
    """

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")

    processor_config = DataflowToolConfig(
        fields=_normalize_fields(fields),
        model_name=model_name,
        method=method,
        confidence_threshold=confidence_threshold,
        deidentify_kwargs=deidentify_kwargs,
    )
    output = copy.deepcopy(dict(record))
    output_attributes = _copy_attributes(attributes)
    targets = _collect_targets(output, processor_config.fields)

    changed_fields = 0
    entity_count = 0
    if targets:
        pipeline = _get_pipeline(processor_config.model_name, config)
        batch_callable = process_batch_fn or process_batch
        texts = [target.text for target in targets]
        kwargs = dict(processor_config.deidentify_kwargs)
        kwargs.update(
            {
                "operation": "deidentify",
                "batch_size": len(texts),
                "continue_on_error": False,
                "loader": pipeline.loader,
                "method": processor_config.method,
                "confidence_threshold": processor_config.confidence_threshold,
            }
        )

        try:
            with pipeline.lock:
                batch_result = batch_callable(
                    texts,
                    model_name=processor_config.model_name,
                    config=config,
                    ids=[f"field_{index}" for index in range(len(texts))],
                    **kwargs,
                )
        except Exception:
            raise DataflowToolProcessorError(
                "failed to redact configured record fields"
            ) from None

        items = _batch_items(batch_result)
        if len(items) != len(targets):
            raise DataflowToolProcessorError(
                "batch result count did not match configured field count"
            )

        for target, item in zip(targets, items):
            redacted_text, redactions = _redacted_item(item)
            _set_path(output, target.path, redacted_text)
            changed_fields += int(redacted_text != target.text)
            entity_count += redactions

    output_attributes[RECORD_COUNT_ATTRIBUTE] = "1"
    output_attributes[FIELD_COUNT_ATTRIBUTE] = str(changed_fields)
    output_attributes[ENTITY_COUNT_ATTRIBUTE] = str(entity_count)
    return output, output_attributes


def process_record(
    record: Mapping[str, Any],
    attributes: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Alias-style record entrypoint for generic script processors."""

    return process_flow_file(record, attributes, **kwargs)


def script_processor(
    record: Mapping[str, Any],
    attributes: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Process one record using a script-processor-compatible signature."""

    return process_flow_file(record, attributes, **kwargs)


def process_json_lines(
    input_stream: Iterable[str],
    output_stream: TextIO,
    *,
    fields: Sequence[str] | str,
    model_name: str = DEFAULT_DATAFLOW_TOOL_MODEL,
    method: str = "mask",
    confidence_threshold: float | None = 0.7,
    config: Any | None = None,
    process_batch_fn: ProcessBatch | None = None,
    **deidentify_kwargs: Any,
) -> int:
    """Process JSON-line flow-file envelopes through a long-lived worker.

    Each non-blank input line must contain ``record`` and may contain
    ``attributes``. Exactly one output envelope is written per input envelope.
    The long-lived module process reuses its cached model loader across lines.

    Returns:
        Number of output envelopes written.
    """

    emitted = 0
    for line_number, raw_line in enumerate(input_stream, start=1):
        if not raw_line.strip():
            continue
        try:
            envelope = json.loads(raw_line)
        except json.JSONDecodeError:
            raise DataflowToolProcessorError(
                f"invalid JSON envelope at input line {line_number}"
            ) from None
        if not isinstance(envelope, Mapping):
            raise DataflowToolProcessorError(
                f"input line {line_number} must contain a JSON object"
            )
        if "record" not in envelope:
            raise DataflowToolProcessorError(
                f"input line {line_number} is missing the record object"
            )

        try:
            redacted_record, redacted_attributes = process_flow_file(
                envelope["record"],
                envelope.get("attributes"),
                fields=fields,
                model_name=model_name,
                method=method,
                confidence_threshold=confidence_threshold,
                config=config,
                process_batch_fn=process_batch_fn,
                **deidentify_kwargs,
            )
        except (DataflowToolProcessorError, TypeError, ValueError):
            raise DataflowToolProcessorError(
                f"failed to process input line {line_number}"
            ) from None

        output_stream.write(
            json.dumps(
                {
                    "record": redacted_record,
                    "attributes": redacted_attributes,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        output_stream.write("\n")
        emitted += 1

    return emitted


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the persistent JSON-lines script-processor entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    input_stream = stdin if stdin is not None else sys.stdin
    output_stream = stdout if stdout is not None else sys.stdout
    error_stream = stderr if stderr is not None else sys.stderr

    try:
        process_json_lines(
            input_stream,
            output_stream,
            fields=tuple(args.field),
            model_name=args.model_name,
            method=args.method,
            confidence_threshold=args.confidence_threshold,
            lang=args.lang,
            policy=args.policy,
            use_safety_sweep=args.use_safety_sweep,
        )
    except (DataflowToolProcessorError, TypeError, ValueError) as exc:
        error_stream.write(f"dataflow record redaction failed: {exc}\n")
        return 1
    return 0


def clear_pipeline_cache() -> None:
    """Clear cached model loaders used by long-lived script processors."""

    with _PIPELINE_CACHE_LOCK:
        _PIPELINE_CACHE.clear()


def _get_pipeline(model_name: str, config: Any | None) -> _CachedPipeline:
    cache_key = (model_name, None if config is None else id(config))
    with _PIPELINE_CACHE_LOCK:
        pipeline = _PIPELINE_CACHE.get(cache_key)
        if pipeline is None:
            pipeline = _CachedPipeline(loader=_create_pipeline(config))
            _PIPELINE_CACHE[cache_key] = pipeline
        return pipeline


def _create_pipeline(config: Any | None) -> Any:
    from openmed.core import ModelLoader

    return ModelLoader(config)


def _copy_attributes(
    attributes: Mapping[str, str] | None,
) -> dict[str, str]:
    if attributes is None:
        return {}
    if not isinstance(attributes, Mapping):
        raise TypeError("attributes must be a mapping")
    output = dict(attributes)
    if any(not isinstance(key, str) for key in output):
        raise TypeError("attribute names must be strings")
    if any(not isinstance(value, str) for value in output.values()):
        raise TypeError("attribute values must be strings")
    return output


def _collect_targets(record: Mapping[str, Any], fields: Sequence[str]) -> list[_Target]:
    targets: list[_Target] = []
    for field_name in fields:
        path = tuple(field_name.split("."))
        value = _get_path(record, path)
        if isinstance(value, str) and value:
            targets.append(_Target(path=path, text=value))
    return targets


def _get_path(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _set_path(record: dict[str, Any], path: Sequence[str], value: str) -> None:
    parent: dict[str, Any] = record
    for part in path[:-1]:
        child = parent[part]
        if not isinstance(child, dict):
            raise DataflowToolProcessorError(
                "configured record field has an invalid nested path"
            )
        parent = child
    parent[path[-1]] = value


def _batch_items(batch_result: Any) -> list[Any]:
    raw_items = getattr(batch_result, "items", batch_result)
    try:
        return list(raw_items)
    except TypeError:
        raise DataflowToolProcessorError(
            "batch redaction returned a non-iterable result"
        ) from None


def _redacted_item(item: Any) -> tuple[str, int]:
    if getattr(item, "success", True) is False:
        raise DataflowToolProcessorError(
            "one or more configured record fields could not be redacted"
        )

    result = getattr(item, "result", item)
    if isinstance(result, str):
        return result, 0
    if isinstance(result, Mapping):
        redacted = result.get("deidentified_text")
        entities = result.get("pii_entities", result.get("entities", ()))
    else:
        redacted = getattr(result, "deidentified_text", None)
        entities = getattr(
            result,
            "pii_entities",
            getattr(result, "entities", ()),
        )
    if not isinstance(redacted, str):
        raise DataflowToolProcessorError(
            "batch redaction returned an invalid field result"
        )

    try:
        entity_count = len(entities or ())
    except TypeError:
        entity_count = 0
    return redacted, entity_count


def _normalize_fields(fields: Sequence[str] | str) -> tuple[str, ...]:
    raw_fields = (fields,) if isinstance(fields, str) else tuple(fields)
    normalized: list[str] = []
    for field_name in raw_fields:
        if not isinstance(field_name, str) or not field_name.strip():
            raise ValueError("fields must contain non-empty strings")
        field_name = field_name.strip()
        if any(not part for part in field_name.split(".")):
            raise ValueError("fields must use non-empty dotted path segments")
        if field_name not in normalized:
            normalized.append(field_name)
    if not normalized:
        raise ValueError("fields must include at least one field name")
    return tuple(normalized)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Redact configured fields in JSON-line flow-file envelopes.")
    )
    parser.add_argument(
        "--field",
        action="append",
        required=True,
        help=(
            "Record field to redact. Repeat for multiple fields; dotted paths "
            "such as patient.note are supported."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_DATAFLOW_TOOL_MODEL,
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
        help="Enable or disable the regex safety-sweep fallback.",
    )
    return parser


__all__ = [
    "DEFAULT_DATAFLOW_TOOL_MODEL",
    "ENTITY_COUNT_ATTRIBUTE",
    "FIELD_COUNT_ATTRIBUTE",
    "RECORD_COUNT_ATTRIBUTE",
    "DataflowToolConfig",
    "DataflowToolProcessorError",
    "ProcessBatch",
    "clear_pipeline_cache",
    "main",
    "process_flow_file",
    "process_json_lines",
    "process_record",
    "script_processor",
]


if __name__ == "__main__":  # pragma: no cover - exercised via main()
    raise SystemExit(main())
