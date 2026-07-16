"""Short-message de-identification and SMS export adapters.

The ``short_text`` preset is deliberately deterministic around the identifiers
that dominate SMS-scale clinical data.  It augments the normal OpenMed PII
pipeline with African MSISDN, shortcode, national-ID, and maternal-health
honorific rules while allowing common SMS/clinical abbreviations to remain
ordinary text.  Export helpers preserve source schemas, pseudonymize contact
addresses, coarsen timestamps to calendar dates, and process message text in
bounded batches.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, TextIO

SHORT_TEXT = "short_text"
DEFAULT_SMS_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"

_NAME_TOKEN = r"(?:[A-Z][^\W\d_]{1,}(?:[-'’][A-Z]?[^\W\d_]*)?|[A-Z]{2,})"
_HONORIFIC_NAME_PATTERN = (
    r"(?i:\b(?:mama|maama|mami|sisi|dada|mme|mrs?|ms|dr|ndugu)\.?\s+)"
    + _NAME_TOKEN
    + r"(?:\s+"
    + _NAME_TOKEN
    + r")?"
)

_SHORT_TEXT_DENY_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "pattern": (
            r"(?<![\w+])(?:\+|00)(?:254|256|27)"
            r"(?:[\s().-]*\d){8,10}(?!\d)"
        ),
        "label": "PHONE",
        "rule_id": "short_text_african_e164",
        "confidence": 1.0,
    },
    {
        "pattern": r"(?<!\d)07\d(?:[\s.-]*\d){7}(?!\d)",
        "label": "PHONE",
        "rule_id": "short_text_local_07xx",
        "confidence": 1.0,
    },
    {
        "pattern": r"(?<![\dA-Za-z])\d{4,6}(?![\dA-Za-z])",
        "label": "PHONE",
        "rule_id": "short_text_shortcode",
        "confidence": 0.98,
    },
    {
        "pattern": (
            r"(?i:\b(?:id|nid|mrn|patient\s*(?:id|no)|clinic\s*no)"
            r"\s*[:#-]?\s*)[A-Z0-9][A-Z0-9/.-]{3,}"
        ),
        "label": "ID_NUM",
        "rule_id": "short_text_contextual_id",
        "confidence": 1.0,
    },
    {
        "pattern": r"\b[A-Z]{2}\d{9}[A-Z0-9]{3}\b",
        "label": "ID_NUM",
        "rule_id": "short_text_uganda_nin",
        "confidence": 1.0,
    },
    {
        "pattern": r"(?<!\d)\d{13}(?!\d)",
        "label": "ID_NUM",
        "rule_id": "short_text_south_africa_id",
        "confidence": 1.0,
    },
    {
        "pattern": _HONORIFIC_NAME_PATTERN,
        "label": "PERSON",
        "rule_id": "short_text_maternal_honorific_name",
        "confidence": 1.0,
    },
)

_SHORT_TEXT_ALLOW_PATTERNS: tuple[dict[str, Any], ...] = (
    {
        "pattern": r"\b(?:ANC|EDD|LMP|BP|HIV|ART|CHW|SMS|USSD)\b",
        "rule_id": "short_text_clinical_abbreviations",
    },
    {
        "pattern": r"(?i:\b(?:pls|plz|thx|appt|wk|wks|msg|rply)\b)",
        "rule_id": "short_text_sms_abbreviations",
    },
)

_RECORD_COLLECTION_KEYS = ("messages", "results", "runs", "records", "data")
_CONTACT_VALUE_KEYS = frozenset({"urn", "contact_urn", "contact_address"})
_CONTACT_LIST_KEYS = frozenset({"urns", "contact_urns"})
_CONTACT_CONTAINER_KEYS = frozenset({"contact", "sender", "recipient"})
_CONTACT_MAPPING_KEYS = frozenset({"urn", "urns", "name", "address"})
_TIMESTAMP_KEYS = frozenset(
    {
        "sent_on",
        "received_on",
        "delivered_on",
        "created_on",
        "modified_on",
        "timestamp",
    }
)

TextRedactor = Callable[[str], Any]


@dataclass(frozen=True)
class ShortTextPreset:
    """Configuration for the SMS-scale ``short_text`` pipeline preset."""

    name: str = SHORT_TEXT
    confidence_threshold: float = 0.35
    batch_size: int = 512
    preserve_whitespace: bool = True
    use_smart_merging: bool = True
    use_safety_sweep: bool = True

    def build_recognizer(self) -> Any:
        """Build a fresh deterministic recognizer for this preset."""

        from openmed.core.custom_recognizer import CustomRecognizer

        return CustomRecognizer(
            deny_patterns=_SHORT_TEXT_DENY_PATTERNS,
            allow_patterns=_SHORT_TEXT_ALLOW_PATTERNS,
            case_sensitive=True,
        )


SHORT_TEXT_PRESET = ShortTextPreset()


@dataclass(frozen=True)
class SMSRedactionSummary:
    """PHI-free counts collected while redacting one SMS export."""

    row_count: int = 0
    message_count: int = 0
    redacted_message_count: int = 0
    pseudonymized_contact_count: int = 0
    coarsened_timestamp_count: int = 0
    batch_count: int = 0

    def to_dict(self) -> dict[str, int]:
        """Return a JSON-serializable, raw-value-free summary."""

        return {
            "row_count": self.row_count,
            "message_count": self.message_count,
            "redacted_message_count": self.redacted_message_count,
            "pseudonymized_contact_count": self.pseudonymized_contact_count,
            "coarsened_timestamp_count": self.coarsened_timestamp_count,
            "batch_count": self.batch_count,
        }


@dataclass(frozen=True)
class RedactedSMSExport:
    """Materialized text or streamed path plus a PHI-free summary."""

    format: str
    summary: SMSRedactionSummary
    text: str | None = None
    output_path: Path | None = None


@dataclass
class _SummaryAccumulator:
    row_count: int = 0
    message_count: int = 0
    redacted_message_count: int = 0
    coarsened_timestamp_count: int = 0
    batch_count: int = 0

    def freeze(self, *, pseudonymized_contact_count: int) -> SMSRedactionSummary:
        return SMSRedactionSummary(
            row_count=self.row_count,
            message_count=self.message_count,
            redacted_message_count=self.redacted_message_count,
            pseudonymized_contact_count=pseudonymized_contact_count,
            coarsened_timestamp_count=self.coarsened_timestamp_count,
            batch_count=self.batch_count,
        )


@dataclass
class _TextSlot:
    container: MutableMapping[str, Any] | list[Any]
    key: str | int
    text: str


class _ContactHasher:
    def __init__(self, key: str | bytes | None = None) -> None:
        if key is None:
            self._key = secrets.token_bytes(32)
        elif isinstance(key, str):
            self._key = key.encode("utf-8")
        else:
            self._key = bytes(key)
        if not self._key:
            raise ValueError("contact_hash_key must not be empty")
        self._tokens: dict[str, str] = {}

    def pseudonymize(self, value: str) -> str:
        if not value:
            return value
        token = self._tokens.get(value)
        if token is None:
            digest = hmac.new(
                self._key,
                value.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:24]
            token = f"contact_sha256:{digest}"
            self._tokens[value] = token
        return token

    @property
    def count(self) -> int:
        return len(self._tokens)


def resolve_short_text_preset(
    preset: str | ShortTextPreset = SHORT_TEXT,
) -> ShortTextPreset:
    """Resolve the supported short-message preset.

    Args:
        preset: The ``"short_text"`` name or a customized preset instance.

    Returns:
        The resolved preset.

    Raises:
        ValueError: If a string names an unsupported preset.
    """

    if isinstance(preset, ShortTextPreset):
        return preset
    if preset != SHORT_TEXT:
        raise ValueError(f"unsupported SMS pipeline preset: {preset!r}")
    return SHORT_TEXT_PRESET


def deidentify_short_text(
    text: str,
    *,
    preset: str | ShortTextPreset = SHORT_TEXT,
    method: str = "mask",
    model_name: str = DEFAULT_SMS_MODEL,
    lang: str = "en",
    policy: Any = None,
    keep_mapping: bool = False,
    loader: Any = None,
    model_detector: Callable[..., Any] | None = None,
) -> Any:
    """De-identify one SMS-scale string without trimming or reflowing it.

    The input is never truncated. Whitespace normalization may be used for
    detection internally, but detected offsets are mapped back to the original
    string before replacements are applied.

    Args:
        text: One message payload.
        preset: The ``short_text`` preset or a customized preset instance.
        method: OpenMed de-identification method.
        model_name: PII model registry key or repository identifier.
        lang: Language hint for model and locale-specific recognizers.
        policy: Optional OpenMed de-identification policy.
        keep_mapping: Whether to retain a reversible placeholder mapping.
        loader: Optional warmed model loader.
        model_detector: Optional detector seam for offline callers and tests.

    Returns:
        An OpenMed de-identification result whose offsets refer to ``text``.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    options = resolve_short_text_preset(preset)

    from openmed.core.pipeline import Pipeline

    pipeline = Pipeline(
        model_name=model_name,
        confidence_threshold=options.confidence_threshold,
        use_smart_merging=options.use_smart_merging,
        lang=lang,
        use_safety_sweep=options.use_safety_sweep,
        preserve_whitespace=options.preserve_whitespace,
        loader=loader,
        model_detector=model_detector,
        policy=policy,
        custom_recognizer=options.build_recognizer(),
    )
    result = pipeline.run(
        text,
        method=method,
        keep_mapping=keep_mapping,
    ).deidentification_result
    metadata = dict(getattr(result, "metadata", None) or {})
    metadata["pipeline_preset"] = options.name
    metadata["source_length"] = len(text)
    result.metadata = metadata
    return result


def iter_redacted_sms_records(
    records: Iterable[Mapping[str, Any]],
    *,
    preset: str | ShortTextPreset = SHORT_TEXT,
    method: str = "mask",
    model_name: str = DEFAULT_SMS_MODEL,
    lang: str = "en",
    policy: Any = None,
    batch_size: int | None = None,
    contact_hash_key: str | bytes | None = None,
    text_redactor: TextRedactor | None = None,
    loader: Any = None,
) -> "_SMSRecordIterator":
    """Return a bounded-memory iterator over redacted message records.

    The default path reuses :class:`openmed.processing.batch.BatchProcessor`
    and retains at most ``batch_size`` source records at a time. A
    ``text_redactor`` can be injected for offline tests or an already-warmed
    application-specific detector.

    Args:
        records: Iterable of message record mappings.
        preset: The ``short_text`` preset or a customized preset instance.
        method: OpenMed de-identification method.
        model_name: PII model registry key or repository identifier.
        lang: Language hint for model and locale-specific recognizers.
        policy: Optional OpenMed de-identification policy.
        batch_size: Maximum source records retained per outer batch.
        contact_hash_key: Optional HMAC key for stable cross-run pseudonyms.
        text_redactor: Optional replacement for model-backed text redaction.
        loader: Optional warmed model loader.

    Returns:
        An iterator exposing a PHI-free ``summary`` property.
    """

    options = resolve_short_text_preset(preset)
    resolved_batch_size = options.batch_size if batch_size is None else batch_size
    if not isinstance(resolved_batch_size, int) or resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    accumulator = _SummaryAccumulator()
    hasher = _ContactHasher(contact_hash_key)
    processor = None
    if text_redactor is None:
        processor = _build_batch_processor(
            options,
            method=method,
            model_name=model_name,
            lang=lang,
            policy=policy,
            batch_size=resolved_batch_size,
            loader=loader,
        )
    iterator = _iter_redacted_record_batches(
        records,
        batch_size=resolved_batch_size,
        processor=processor,
        text_redactor=text_redactor,
        hasher=hasher,
        accumulator=accumulator,
    )
    return _SMSRecordIterator(iterator, accumulator, hasher)


class _SMSRecordIterator:
    def __init__(
        self,
        iterator: Iterator[dict[str, Any]],
        accumulator: _SummaryAccumulator,
        hasher: _ContactHasher,
    ) -> None:
        self._iterator = iterator
        self._accumulator = accumulator
        self._hasher = hasher

    def __iter__(self) -> "_SMSRecordIterator":
        return self

    def __next__(self) -> dict[str, Any]:
        return next(self._iterator)

    @property
    def summary(self) -> SMSRedactionSummary:
        """Return counts accumulated through the records consumed so far."""

        return self._accumulator.freeze(pseudonymized_contact_count=self._hasher.count)


def redact_sms_json(
    source: str | os.PathLike[str] | TextIO | Mapping[str, Any] | list[Any],
    output: str | os.PathLike[str] | TextIO | None = None,
    **kwargs: Any,
) -> RedactedSMSExport:
    """Redact a RapidPro-shaped JSON message or flow-results export.

    Args:
        source: JSON mapping/list, text, path, or readable text stream.
        output: Optional destination path or writable text stream.
        **kwargs: Options forwarded to :func:`iter_redacted_sms_records`.

    Returns:
        Materialized output text when ``output`` is omitted, otherwise the
        destination path when available, plus a PHI-free summary.
    """

    payload = copy.deepcopy(_load_json_payload(source))
    records = _locate_records(payload)
    single_root_record = (
        isinstance(payload, MutableMapping)
        and len(records) == 1
        and records[0] is payload
    )
    iterator = iter_redacted_sms_records(records, **kwargs)
    redacted_records = list(iterator)
    if single_root_record:
        payload.clear()
        payload.update(redacted_records[0])
    else:
        records[:] = redacted_records
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    output_path = _write_optional_text_output(output, text)
    return RedactedSMSExport(
        format="json",
        summary=iterator.summary,
        text=text if output is None else None,
        output_path=output_path,
    )


def redact_sms_csv(
    source: str | os.PathLike[str] | TextIO,
    output: str | os.PathLike[str] | TextIO | None = None,
    **kwargs: Any,
) -> RedactedSMSExport:
    """Stream a generic SMS CSV while preserving columns and row order.

    Args:
        source: CSV text, path, or readable text stream.
        output: Optional destination path or writable text stream.
        **kwargs: Options forwarded to :func:`iter_redacted_sms_records`.

    Returns:
        Materialized output text when ``output`` is omitted, otherwise the
        destination path when available, plus a PHI-free summary.
    """

    source_handle, close_source = _open_text_source(source)
    output_handle, close_output, output_path = _open_text_output(output)
    try:
        reader = csv.DictReader(source_handle)
        if not reader.fieldnames:
            raise ValueError("SMS CSV must include a header row")
        if "text" not in reader.fieldnames:
            raise ValueError("SMS CSV must include a 'text' column")
        if not ({"urn", "contact"} & set(reader.fieldnames)):
            raise ValueError("SMS CSV must include an 'urn' or 'contact' column")

        writer = csv.DictWriter(
            output_handle,
            fieldnames=reader.fieldnames,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        iterator = iter_redacted_sms_records(reader, **kwargs)
        for row in iterator:
            writer.writerow(row)
        summary = iterator.summary
        materialized = (
            output_handle.getvalue() if isinstance(output_handle, io.StringIO) else None
        )
    finally:
        if close_source:
            source_handle.close()
        if close_output:
            output_handle.close()

    return RedactedSMSExport(
        format="csv",
        summary=summary,
        text=materialized,
        output_path=output_path,
    )


def redact_sms_export(
    source: str | os.PathLike[str] | TextIO | Mapping[str, Any] | list[Any],
    output: str | os.PathLike[str] | TextIO | None = None,
    *,
    format: str | None = None,
    **kwargs: Any,
) -> RedactedSMSExport:
    """Dispatch a JSON or CSV SMS export through the ``short_text`` preset.

    Args:
        source: Export payload, text, path, or readable stream.
        output: Optional destination path or writable text stream.
        format: Explicit ``"json"`` or ``"csv"`` override.
        **kwargs: Options forwarded to the selected export adapter.

    Returns:
        The redacted export result and PHI-free summary.
    """

    resolved = (format or _infer_export_format(source)).lower()
    if resolved == "json":
        return redact_sms_json(source, output, **kwargs)
    if resolved == "csv":
        if isinstance(source, (Mapping, list)):
            raise TypeError("CSV source must be text, a path, or a text stream")
        return redact_sms_csv(source, output, **kwargs)
    raise ValueError("format must be 'json' or 'csv'")


def coarsen_timestamp(value: Any) -> Any:
    """Coarsen an ISO-like timestamp to ``YYYY-MM-DD`` when parseable.

    Args:
        value: Date, datetime, ISO-like string, or value to leave untouched.

    Returns:
        A calendar-date string for parseable timestamps, otherwise ``value``.
    """

    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str) or not value:
        return value

    candidate = value.strip()
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError:
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            return value
    return parsed.date().isoformat()


def _build_batch_processor(
    preset: ShortTextPreset,
    *,
    method: str,
    model_name: str,
    lang: str,
    policy: Any,
    batch_size: int,
    loader: Any,
) -> Any:
    from openmed.processing.batch import BatchProcessor

    return BatchProcessor(
        model_name=model_name,
        operation="deidentify",
        batch_size=min(batch_size, 100),
        confidence_threshold=preset.confidence_threshold,
        continue_on_error=False,
        loader=loader,
        method=method,
        lang=lang,
        policy=policy,
        use_smart_merging=preset.use_smart_merging,
        use_safety_sweep=preset.use_safety_sweep,
        preserve_whitespace=preset.preserve_whitespace,
        custom_recognizer=preset.build_recognizer(),
    )


def _iter_redacted_record_batches(
    records: Iterable[Mapping[str, Any]],
    *,
    batch_size: int,
    processor: Any,
    text_redactor: TextRedactor | None,
    hasher: _ContactHasher,
    accumulator: _SummaryAccumulator,
) -> Iterator[dict[str, Any]]:
    batch: list[Mapping[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("SMS export records must be mappings")
        batch.append(record)
        if len(batch) >= batch_size:
            yield from _redact_record_batch(
                batch,
                processor=processor,
                text_redactor=text_redactor,
                hasher=hasher,
                accumulator=accumulator,
            )
            batch = []
    if batch:
        yield from _redact_record_batch(
            batch,
            processor=processor,
            text_redactor=text_redactor,
            hasher=hasher,
            accumulator=accumulator,
        )


def _redact_record_batch(
    records: list[Mapping[str, Any]],
    *,
    processor: Any,
    text_redactor: TextRedactor | None,
    hasher: _ContactHasher,
    accumulator: _SummaryAccumulator,
) -> list[dict[str, Any]]:
    copied = [copy.deepcopy(dict(record)) for record in records]
    slots: list[_TextSlot] = []
    for record in copied:
        _collect_text_slots(record, slots)
        _pseudonymize_contacts(record, hasher)
        accumulator.coarsened_timestamp_count += _coarsen_timestamps(record)

    nonempty_slots = [slot for slot in slots if slot.text]
    if nonempty_slots:
        if text_redactor is not None:
            redacted_texts = [
                _coerce_redacted_text(text_redactor(slot.text))
                for slot in nonempty_slots
            ]
        else:
            batch_result = processor.process_texts(
                [slot.text for slot in nonempty_slots],
                ids=[f"sms_{index}" for index in range(len(nonempty_slots))],
            )
            if batch_result.failed_items:
                raise RuntimeError("short_text batch de-identification failed")
            redacted_texts = [
                _coerce_redacted_text(item.result) for item in batch_result.items
            ]
        for slot, redacted in zip(nonempty_slots, redacted_texts):
            slot.container[slot.key] = redacted
            if redacted != slot.text:
                accumulator.redacted_message_count += 1

    accumulator.row_count += len(copied)
    accumulator.message_count += len(slots)
    accumulator.batch_count += 1
    return copied


def _collect_text_slots(value: Any, slots: list[_TextSlot]) -> None:
    if isinstance(value, MutableMapping):
        for key, item in list(value.items()):
            if str(key).lower() == "text" and isinstance(item, str):
                slots.append(_TextSlot(value, key, item))
            elif isinstance(item, (MutableMapping, list)):
                _collect_text_slots(item, slots)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if isinstance(item, (MutableMapping, list)):
                _collect_text_slots(item, slots)


def _pseudonymize_contacts(value: Any, hasher: _ContactHasher) -> None:
    if isinstance(value, MutableMapping):
        for key, item in list(value.items()):
            normalized_key = str(key).lower()
            if normalized_key in _CONTACT_VALUE_KEYS and isinstance(item, str):
                value[key] = hasher.pseudonymize(item)
            elif normalized_key in _CONTACT_LIST_KEYS and isinstance(item, list):
                value[key] = [
                    hasher.pseudonymize(entry) if isinstance(entry, str) else entry
                    for entry in item
                ]
            elif normalized_key in _CONTACT_CONTAINER_KEYS:
                if isinstance(item, str):
                    value[key] = hasher.pseudonymize(item)
                elif isinstance(item, MutableMapping):
                    _pseudonymize_contact_mapping(item, hasher)
                elif isinstance(item, list):
                    _pseudonymize_contacts(item, hasher)
            elif isinstance(item, (MutableMapping, list)):
                _pseudonymize_contacts(item, hasher)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (MutableMapping, list)):
                _pseudonymize_contacts(item, hasher)


def _pseudonymize_contact_mapping(
    contact: MutableMapping[str, Any],
    hasher: _ContactHasher,
) -> None:
    for key, item in list(contact.items()):
        normalized_key = str(key).lower()
        if normalized_key not in _CONTACT_MAPPING_KEYS:
            continue
        if isinstance(item, str):
            contact[key] = hasher.pseudonymize(item)
        elif isinstance(item, list):
            contact[key] = [
                hasher.pseudonymize(entry) if isinstance(entry, str) else entry
                for entry in item
            ]


def _coarsen_timestamps(value: Any) -> int:
    changed = 0
    if isinstance(value, MutableMapping):
        for key, item in list(value.items()):
            if str(key).lower() in _TIMESTAMP_KEYS:
                coarsened = coarsen_timestamp(item)
                value[key] = coarsened
                changed += int(coarsened != item)
            elif isinstance(item, (MutableMapping, list)):
                changed += _coarsen_timestamps(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, (MutableMapping, list)):
                changed += _coarsen_timestamps(item)
    return changed


def _coerce_redacted_text(result: Any) -> str:
    if hasattr(result, "deidentified_text"):
        return str(result.deidentified_text)
    if isinstance(result, str):
        return result
    raise TypeError("text redactor must return a string or de-identification result")


def _load_json_payload(
    source: str | os.PathLike[str] | TextIO | Mapping[str, Any] | list[Any],
) -> Mapping[str, Any] | list[Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, list):
        return source
    if hasattr(source, "read"):
        payload = json.load(source)
    elif isinstance(source, os.PathLike):
        with Path(source).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    elif isinstance(source, str):
        path = Path(source)
        if "\n" not in source and "\r" not in source and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = json.loads(source)
    else:
        raise TypeError("JSON source must be a mapping, list, path, or text stream")
    if not isinstance(payload, (Mapping, list)):
        raise ValueError("SMS JSON root must be an object or array")
    return payload


def _locate_records(payload: Mapping[str, Any] | list[Any]) -> list[Any]:
    if isinstance(payload, list):
        return payload
    for key in _RECORD_COLLECTION_KEYS:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate
        if isinstance(candidate, Mapping):
            try:
                return _locate_records(candidate)
            except ValueError:
                pass
    if isinstance(payload, MutableMapping):
        return [payload]
    raise ValueError("SMS JSON object does not contain a supported record collection")


def _open_text_source(
    source: str | os.PathLike[str] | TextIO,
) -> tuple[TextIO, bool]:
    if hasattr(source, "read"):
        return source, False
    if isinstance(source, os.PathLike):
        return Path(source).open("r", encoding="utf-8", newline=""), True
    path = Path(source)
    if "\n" not in source and "\r" not in source and path.exists():
        return path.open("r", encoding="utf-8", newline=""), True
    return io.StringIO(source, newline=""), True


def _open_text_output(
    output: str | os.PathLike[str] | TextIO | None,
) -> tuple[TextIO, bool, Path | None]:
    if output is None:
        return io.StringIO(newline=""), False, None
    if hasattr(output, "write"):
        return output, False, None
    path = Path(output)
    return path.open("w", encoding="utf-8", newline=""), True, path


def _write_optional_text_output(
    output: str | os.PathLike[str] | TextIO | None,
    text: str,
) -> Path | None:
    if output is None:
        return None
    if hasattr(output, "write"):
        output.write(text)
        return None
    path = Path(output)
    path.write_text(text, encoding="utf-8")
    return path


def _infer_export_format(
    source: str | os.PathLike[str] | TextIO | Mapping[str, Any] | list[Any],
) -> str:
    if isinstance(source, (Mapping, list)):
        return "json"
    if isinstance(source, os.PathLike):
        suffix = Path(source).suffix.lower()
        if suffix in {".json", ".csv"}:
            return suffix[1:]
    if isinstance(source, str):
        path = Path(source)
        if "\n" not in source and "\r" not in source and path.exists():
            suffix = path.suffix.lower()
            if suffix in {".json", ".csv"}:
                return suffix[1:]
        return "json" if source.lstrip().startswith(("{", "[")) else "csv"
    raise ValueError("format is required for streams without a recognizable suffix")


__all__ = [
    "DEFAULT_SMS_MODEL",
    "SHORT_TEXT",
    "SHORT_TEXT_PRESET",
    "RedactedSMSExport",
    "SMSRedactionSummary",
    "ShortTextPreset",
    "coarsen_timestamp",
    "deidentify_short_text",
    "iter_redacted_sms_records",
    "redact_sms_csv",
    "redact_sms_export",
    "redact_sms_json",
    "resolve_short_text_preset",
]
