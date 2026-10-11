"""Offline clinical commands over validated, caller-supplied NER spans."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections import defaultdict, deque
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

from ._output import EXIT_ERROR, EXIT_USAGE, CliError, command_path, emit, emit_error

_NOTE_BYTES = 16_384
_JSON_BYTES = 1_048_576
_MAX_SPANS = 512
_COMMANDS = ("sdoh", "relations", "timeline")
_ATTRIBUTE_LABELS = frozenset(
    {
        "DOSE",
        "DOSAGE",
        "DURATION",
        "FREQUENCY",
        "ROUTE",
        "STRENGTH",
        "FORM",
        "STATUS",
        "INDICATION",
        "SEVERITY",
        "TIMEX",
        "DRUG",
        "MED",
        "MEDICINE",
        "RX",
        "DISEASE",
        "SYMPTOM",
        "DIAGNOSIS",
        "EVENT",
        "FINDING",
        "OBSERVATION",
    }
)
_TIME_LABELS = frozenset({"DATE", "DATE_OF_BIRTH", "TIME", "TIMEX", "DURATION"})
_SDOH_CATEGORIES = frozenset(
    {"tobacco", "alcohol", "drug", "employment", "living_status", "food_insecurity"}
)
_SDOH_STATUSES = frozenset(
    {
        "current",
        "past",
        "none",
        "unknown",
        "employed",
        "unemployed",
        "retired",
        "disabled",
        "student",
        "former",
        "never",
        "housed",
        "homeless",
        "assisted_living",
        "lives_alone",
        "lives_with_family",
    }
)
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_DATETIME = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])\Z"
)


def add_clinical_command(subparsers: argparse._SubParsersAction) -> None:
    """Register local clinical extraction commands without loading models.

    Args:
        subparsers: Root CLI command registrar.
    """

    group = subparsers.add_parser(
        "clinical", help="Extract value-free clinical records from existing spans."
    )
    children = group.add_subparsers(dest="clinical_command")
    for name in _COMMANDS:
        parser = children.add_parser(
            name, help=f"Run offline {name} extraction over validated spans."
        )
        parser.add_argument("--note", required=True, help="Local UTF-8 note file.")
        parser.add_argument(
            "--spans", required=True, help="Local openmed analyze --json result file."
        )
        parser.add_argument(
            "--sections", help="Optional JSON file of contiguous section spans."
        )
        parser.add_argument("--language", default="en", help="Explicit language code.")
        if name == "timeline":
            parser.add_argument(
                "--reference-time", help="Explicit ISO date or aware ISO datetime."
            )
        parser.add_argument(
            "--output", help="Write the value-free envelope to a new private file."
        )
        parser.set_defaults(handler=_handle_clinical)


def parse_clinical_args(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None
) -> argparse.Namespace:
    """Parse clinical invocations without echoing private argument values.

    Other commands retain their existing argparse behavior. Clinical help
    remains normal help; invalid clinical arguments use a controlled error.

    Args:
        parser: Fully registered root parser.
        argv: Explicit arguments, or None to read process arguments.

    Returns:
        Parsed arguments or a controlled clinical usage-error handler.
    """

    values = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(values):
        if values[index] == "--config-path":
            index += 2
        elif values[index].startswith("--config-path="):
            index += 1
        else:
            break
    if index >= len(values) or values[index] != "clinical":
        return parser.parse_args(values)
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stderr(sink):
            try:
                return parser.parse_args(values)
            except SystemExit as exc:
                if exc.code != EXIT_USAGE:
                    raise
    path = "clinical"
    if index + 1 < len(values) and values[index + 1] in _COMMANDS:
        path += " " + values[index + 1]
    return argparse.Namespace(
        command_path=path,
        json_output="--json" in values,
        handler=_invalid_arguments,
    )


def _invalid_arguments(args: argparse.Namespace) -> int:
    return emit_error(args, _invalid("clinical_arguments_invalid"))


def _invalid(code: str) -> CliError:
    return CliError(
        "Clinical input or options are invalid.", code=code, exit_code=EXIT_USAGE
    )


def _read(path: str, limit: int) -> bytes:
    descriptor = None
    try:
        descriptor = os.open(
            path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0)
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise _invalid("clinical_input_not_regular")
        if info.st_size > limit:
            raise _invalid("clinical_input_too_large")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise _invalid("clinical_input_too_large")
        return data
    except CliError:
        raise
    except (OSError, ValueError):
        pass
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
    raise _invalid("clinical_input_unavailable")


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeError:
        pass
    raise _invalid("clinical_input_encoding")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _invalid("clinical_json_invalid")
        result[key] = value
    return result


def _nonfinite(_: str) -> Any:
    raise _invalid("clinical_json_invalid")


def _load_json(data: bytes) -> Any:
    try:
        payload = json.loads(
            _decode(data), object_pairs_hook=_object, parse_constant=_nonfinite
        )
        pending = [(payload, 0)]
        count = 0
        while pending:
            value, depth = pending.pop()
            count += 1
            if depth > 32 or count > 65_536:
                raise _invalid("clinical_json_invalid")
            if isinstance(value, dict):
                pending.extend((v, depth + 1) for v in value.values())
            elif isinstance(value, list):
                pending.extend((v, depth + 1) for v in value)
            elif isinstance(value, float) and not math.isfinite(value):
                raise _invalid("clinical_json_invalid")
        return payload
    except CliError:
        raise
    except (ValueError, RecursionError):
        pass
    raise _invalid("clinical_json_invalid")


def _offsets(start: Any, end: Any, length: int, code: str) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int:
        raise _invalid(code)
    if not 0 <= start < end <= length:
        raise _invalid(code)
    return start, end


def _score(value: Any) -> float:
    if type(value) not in (int, float):
        raise _invalid("clinical_span_invalid")
    if not 0 <= value <= 1 or not math.isfinite(value):
        raise _invalid("clinical_span_invalid")
    return float(value)


def _label(value: Any, language: str) -> str:
    from openmed.clinical.relations.multilingual import CMEIE_ENTITY_TYPES
    from openmed.core.labels import is_recognized_label

    if type(value) is not str or len(value) > 64:
        raise _invalid("clinical_label_unsupported")
    if value in CMEIE_ENTITY_TYPES:
        return value
    # The public normalizer strips arbitrary punctuation and non-ASCII text.
    # Validate the label spelling before using it as output metadata.
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_ -]{0,63}", value) is None:
        raise _invalid("clinical_label_unsupported")
    if is_recognized_label(value, lang=language) or value.upper() in _ATTRIBUTE_LABELS:
        return value
    raise _invalid("clinical_label_unsupported")


def _context(row: dict[str, Any], metadata: dict[str, Any]) -> dict[str, str]:
    from openmed.clinical.context import (
        CERTAINTY_VALUES,
        EXPERIENCER_VALUES,
        NEGATION_VALUES,
        TEMPORALITY_VALUES,
    )

    axes = {
        "certainty": CERTAINTY_VALUES,
        "negation": NEGATION_VALUES,
        "experiencer": EXPERIENCER_VALUES,
        "temporality": TEMPORALITY_VALUES,
    }
    nested = metadata.get("clinical_context", {})
    assertion = row.get("assertion", {})
    if not isinstance(nested, dict) or not isinstance(assertion, dict):
        raise _invalid("clinical_span_invalid")
    result = {}
    for key, allowed in axes.items():
        aliases = (key, "uncertainty") if key == "certainty" else (key,)
        supplied = [
            source[alias]
            for source in (row, nested, assertion)
            for alias in aliases
            if alias in source
        ]
        if supplied:
            if any(type(v) is not str or v not in allowed for v in supplied) or any(
                v != supplied[0] for v in supplied
            ):
                raise _invalid("clinical_span_invalid")
            result[key] = supplied[0]
    return result


def _spans(payload: Any, text: str, language: str) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise _invalid("clinical_spans_invalid")
    if "ok" in payload:
        if payload.get("ok") is not True or payload.get("command") != "analyze":
            raise _invalid("clinical_spans_invalid")
        payload = payload.get("data")
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        raise _invalid("clinical_spans_invalid")
    if "text" in payload and payload["text"] != text:
        raise _invalid("clinical_source_mismatch")
    entities = payload["entities"]
    if len(entities) > _MAX_SPANS:
        raise _invalid("clinical_spans_too_many")
    result = []
    for row in entities:
        if not isinstance(row, dict):
            raise _invalid("clinical_span_invalid")
        start, end = _offsets(
            row.get("start"), row.get("end"), len(text), "clinical_span_invalid"
        )
        if "text" in row and row["text"] != text[start:end]:
            raise _invalid("clinical_source_mismatch")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise _invalid("clinical_span_invalid")
        score = _score(row.get("confidence", row.get("score", 1.0)))
        if "score" in row and "confidence" in row and _score(row["score"]) != score:
            raise _invalid("clinical_span_invalid")
        axes = _context(row, metadata)
        span: dict[str, Any] = {
            "start": start,
            "end": end,
            "text": text[start:end],
            "label": _label(row.get("label"), language),
            "score": score,
            **axes,
            "metadata": {"clinical_context": dict(axes)},
        }
        if "certainty" in axes:
            span["uncertainty"] = axes["certainty"]
            span["metadata"]["clinical_context"]["uncertainty"] = axes["certainty"]
        section = row.get("section", metadata.get("section"))
        if section is not None:
            span["section"] = _section_label(section)
        time_span = metadata.get("clinical_time_span")
        if time_span is not None:
            if not isinstance(time_span, list) or len(time_span) != 2:
                raise _invalid("clinical_time_link_invalid")
            span["clinical_time_span"] = _offsets(
                time_span[0], time_span[1], len(text), "clinical_time_link_invalid"
            )
        result.append(span)
    temporal = {_span_key(s) for s in result if _is_time(s, language)}
    for span in result:
        if "clinical_time_span" in span and span["clinical_time_span"] not in temporal:
            raise _invalid("clinical_time_link_invalid")
    return result


def _section_label(value: Any) -> str:
    from openmed.clinical.sections import SECTION_LOINC_MAP

    if not isinstance(value, str) or value not in (*SECTION_LOINC_MAP, "unsectioned"):
        raise _invalid("clinical_sections_invalid")
    return value


def _sections(data: bytes | None, text: str) -> list[dict[str, Any]] | None:
    if data is None:
        return None
    payload = _load_json(data)
    if not isinstance(payload, list) or len(payload) > 256:
        raise _invalid("clinical_sections_invalid")
    rows = []
    position = 0
    for row in payload:
        if not isinstance(row, dict):
            raise _invalid("clinical_sections_invalid")
        start, end = _offsets(
            row.get("start"), row.get("end"), len(text), "clinical_sections_invalid"
        )
        if start != position:
            raise _invalid("clinical_sections_invalid")
        rows.append(
            {"start": start, "end": end, "label": _section_label(row.get("label"))}
        )
        position = end
    if position != len(text):
        raise _invalid("clinical_sections_invalid")
    return rows


def _reference(value: str | None) -> date | datetime | None:
    from openmed.core.iso_temporal import parse_iso_date, parse_iso_datetime

    if value is None:
        return None
    try:
        if _DATE.fullmatch(value):
            return parse_iso_date(value)
        if _DATETIME.fullmatch(value):
            stamp = parse_iso_datetime(value)
            if stamp.utcoffset() is not None:
                return stamp
    except ValueError:
        pass
    raise _invalid("clinical_reference_invalid")


def _is_time(span: dict[str, Any], language: str) -> bool:
    from openmed.core.labels import normalize_label

    return (
        span["label"].upper() in _TIME_LABELS
        or normalize_label(span["label"], lang=language) in _TIME_LABELS
    )


def _span_key(span: dict[str, Any]) -> tuple[int, int]:
    return span["start"], span["end"]


def _safe_code(value: Any, allowed: Sequence[str] | frozenset[str]) -> Any:
    if value is not None and (type(value) is not str or value not in allowed):
        raise ValueError("unsupported projection code")
    return value


def _sdoh(text: str, spans: list[dict[str, Any]], sections: Any) -> dict[str, Any]:
    from openmed.clinical.context import TEMPORALITY_VALUES
    from openmed.clinical.sdoh import extract_sdoh

    records = []
    for finding in extract_sdoh(text, spans, sections=sections):
        start, end = finding.span
        start, end = _offsets(start, end, len(text), "clinical_result_invalid")
        records.append(
            {
                "span": [start, end],
                "category": _safe_code(finding.category, _SDOH_CATEGORIES),
                "status": _safe_code(finding.status, _SDOH_STATUSES),
                "temporality": _safe_code(finding.temporality, TEMPORALITY_VALUES),
                "score": _score(finding.score),
                "extent_present": finding.extent is not None,
                "review_required": True,
            }
        )
    return {"findings": records}


def _relation_span(span: Any, text: str, language: str) -> dict[str, Any]:
    start, end = _offsets(span.start, span.end, len(text), "clinical_result_invalid")
    if type(span.derived) is not bool:
        raise ValueError("unsupported derived flag")
    return {
        "start": start,
        "end": end,
        "label": _label(span.label, language),
        "derived": span.derived,
    }


def _relations(
    text: str, spans: list[dict[str, Any]], sections: Any, language: str
) -> dict[str, Any]:
    from openmed.clinical.relations import (
        ATTRIBUTE_RELATION_TYPES,
        PROBLEM_ATTRIBUTE_RELATION_TYPES,
        RELATION_ASSERTION_STATUSES,
        extract_relations,
        relation_type_mapping,
    )

    allowed = (
        frozenset(
            (
                *ATTRIBUTE_RELATION_TYPES.values(),
                *PROBLEM_ATTRIBUTE_RELATION_TYPES.values(),
            )
        )
        if language == "en"
        else frozenset(relation_type_mapping(language).values())
    )
    rows = extract_relations(
        text, spans, sections=sections, language=None if language == "en" else language
    )
    records = []
    for relation in rows:
        record = {
            "head": _relation_span(relation.head, text, language),
            "tail": _relation_span(relation.tail, text, language),
            "relation_type": _safe_code(relation.relation_type, allowed),
            "score": _score(relation.score),
            "review_required": True,
        }
        if hasattr(relation, "assertion_status"):
            record["assertion_status"] = _safe_code(
                relation.assertion_status, RELATION_ASSERTION_STATUSES
            )
        records.append(record)
    return {"relations": records}


def _timeline(
    text: str,
    spans: list[dict[str, Any]],
    sections: Any,
    language: str,
    reference: date | datetime | None,
) -> dict[str, Any]:
    from openmed.clinical.context import assert_context
    from openmed.clinical.temporal_normalizer import normalize_temporal
    from openmed.clinical.timeline import build_timeline

    times = normalize_temporal(
        text, [s for s in spans if _is_time(s, language)], reference, language=language
    )
    by_offset = {(t.start, t.end): t for t in times}
    tagged = assert_context(text, spans, language=language, sections=sections)
    for supplied, target in zip(spans, tagged, strict=True):
        target.update(supplied["metadata"]["clinical_context"])
        target["certainty"] = target.get("certainty", target["uncertainty"])
        link = supplied.get("clinical_time_span")
        if link is None and _is_time(supplied, language):
            link = _span_key(supplied)
        normalized = by_offset.get(link) if link is not None else None
        if normalized is not None and normalized.timex_type == "DATE":
            # Only day-precision dates can order an event. Ambiguous/partial
            # expressions remain visible as unresolved temporal records.
            if (
                normalized.value
                and _DATE.fullmatch(normalized.value)
                and "ambiguous" not in normalized.granularity_flags
            ):
                target["normalized_time"] = normalized.value
    timeline = build_timeline(tagged)
    # The API exposes hashes rather than labels. Align each event with its
    # supplied label, including coincident spans in different assertion lanes.
    labels: dict[tuple[Any, ...], deque[str]] = defaultdict(deque)
    for target in tagged:
        single = build_timeline([target]).events[0]
        labels[_event_key(single)].append(target["label"])
    records = [
        {
            "start": event.start,
            "end": event.end,
            "label": _label(labels[_event_key(event)].popleft(), language),
            "event_kind": _safe_code(
                event.event_kind,
                (
                    "condition",
                    "diagnosis",
                    "event",
                    "finding",
                    "medication",
                    "observation",
                    "procedure",
                    "symptom",
                ),
            ),
            "assertion": _context({"assertion": event.assertion.to_dict()}, {}),
            "time_state": "anchored" if event.normalized_time else "unanchored",
            "review_required": True,
        }
        for event in timeline.events
    ]
    time_records = [
        {
            "start": t.start,
            "end": t.end,
            "type": _safe_code(t.timex_type, ("DATE", "TIME", "DURATION", "SET")),
            "state": "unanchored"
            if "unanchored" in t.granularity_flags
            else "unresolved"
            if t.value is None or "ambiguous" in t.granularity_flags
            else "normalized",
            "granularity_flags": list(t.granularity_flags),
            "review_required": True,
        }
        for t in times
    ]
    return {
        "events": records,
        "temporal": time_records,
        "lanes": {
            lane: [
                i
                for i, event in enumerate(timeline.events)
                if event.assertion.temporality == lane
            ]
            for lane in ("historical", "recent", "hypothetical")
        },
        "reference_time_supplied": reference is not None,
    }


def _event_key(event: Any) -> tuple[Any, ...]:
    return (
        event.start,
        event.end,
        event.event_kind,
        event.normalized_time,
        event.assertion.temporality,
        event.assertion.certainty,
        event.assertion.negation,
        event.assertion.experiencer,
    )


def _write_new(path: str, envelope: dict[str, Any]) -> None:
    descriptor = None
    owned = None
    succeeded = False
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = None
            json.dump(
                envelope,
                stream,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
        succeeded = True
    except Exception:
        pass
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
    if succeeded:
        return
    if owned is not None:
        with contextlib.suppress(OSError):
            current = os.lstat(path)
            if (current.st_dev, current.st_ino) == owned:
                os.unlink(path)
    raise CliError(
        "Clinical output could not be created.",
        code="clinical_output_unavailable",
        exit_code=EXIT_ERROR,
    )


def _handle_clinical(args: argparse.Namespace) -> int:
    name = args.clinical_command
    language = args.language
    supported = {
        "sdoh": ("en",),
        "relations": ("en", "hi", "zh"),
        "timeline": ("en", "de"),
    }
    if language not in supported[name]:
        raise _invalid("clinical_language_unsupported")
    if name == "relations" and language != "en" and args.sections is not None:
        raise _invalid("clinical_options_unsupported")
    reference = _reference(getattr(args, "reference_time", None))
    note_data = _read(args.note, _NOTE_BYTES)
    text = _decode(note_data)
    span_data = _read(args.spans, _JSON_BYTES)
    spans = _spans(_load_json(span_data), text, language)
    section_data = None if args.sections is None else _read(args.sections, _JSON_BYTES)
    sections = _sections(section_data, text)
    processed = False
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                if name == "sdoh":
                    result = _sdoh(text, spans, sections)
                elif name == "relations":
                    result = _relations(text, spans, sections, language)
                else:
                    result = _timeline(text, spans, sections, language, reference)
        expected_keys = {
            "sdoh": {"findings"},
            "relations": {"relations"},
            "timeline": {"events", "temporal", "lanes", "reference_time_supplied"},
        }
        if type(result) is not dict or set(result) != expected_keys[name]:
            raise ValueError("unsupported result projection")
        payload = {
            "schema_version": 1,
            "language": language,
            "source_digest": hashlib.sha256(note_data).hexdigest(),
            "spans_digest": hashlib.sha256(span_data).hexdigest(),
            "sections_digest": hashlib.sha256(section_data).hexdigest()
            if section_data is not None
            else None,
            "input_span_count": len(spans),
            "review_required": True,
            **result,
        }
        envelope = {"ok": True, "command": command_path(args), "data": payload}
        # Validate the complete output before creating a file or writing stdout.
        stdout_payload = dict(payload)
        if args.output is not None:
            stdout_payload["output_written"] = True
        stdout_envelope = {**envelope, "data": stdout_payload}
        serialized = json.dumps(
            stdout_envelope,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        if len(serialized.encode("utf-8")) + 1 > _JSON_BYTES:
            raise ValueError("output exceeds clinical envelope bound")
        human = json.dumps(
            stdout_payload, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        processed = True
    except Exception:
        pass
    if not processed:
        raise CliError(
            "Clinical processing failed.",
            code="clinical_processing_failed",
            exit_code=EXIT_ERROR,
        )
    if args.output is not None:
        _write_new(args.output, envelope)
    return emit(args, stdout_payload, human=human)
