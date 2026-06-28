"""Streaming incremental de-identification over bounded text windows."""

from __future__ import annotations

import copy
import hashlib
import hmac
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from .pii import DeidentificationMethod
from .pipeline import DEFAULT_HASH_SECRET, STAGE_NAMES, Pipeline
from .schemas.span import OpenMedSpan


@dataclass(frozen=True)
class StreamingDeidentificationEvent:
    """A safely-emittable redacted stream fragment.

    The final event carries the aggregate audit record and final global-offset
    span records. Non-final events intentionally expose only redacted text so
    provisional window doc ids are not leaked into downstream audit handling.
    """

    redacted_text: str
    final: bool = False
    spans: tuple[OpenMedSpan, ...] = ()
    audit_record: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _WindowResult:
    redacted_text: str
    entities: tuple[Any, ...]
    spans: tuple[OpenMedSpan, ...]
    audit_record: Mapping[str, Any] | None


class StreamingBufferError(ValueError):
    """Raised when a single unsafe tail exceeds the configured carry buffer."""


class StreamingDeidentifier:
    """Incrementally de-identify chunked text without splitting identifiers.

    Args mirror :func:`openmed.core.pii.deidentify` where applicable. The
    instance emits redacted text for prefixes that can be finalized while
    retaining at most ``max_buffer`` source characters as carry-over context.
    """

    def __init__(
        self,
        *,
        max_buffer: int = 4096,
        method: DeidentificationMethod = "mask",
        model_name: str | None = None,
        confidence_threshold: float = 0.7,
        keep_year: bool = False,
        shift_dates: Optional[bool] = None,
        date_shift_days: Optional[int] = None,
        keep_mapping: bool = False,
        config: Any = None,
        use_smart_merging: bool = True,
        lang: str = "en",
        normalize_accents: Optional[bool] = None,
        use_safety_sweep: bool = True,
        consistent: bool = False,
        seed: Optional[int] = None,
        locale: Optional[str] = None,
        loader: Any = None,
        policy: Optional[str] = None,
        calibration_thresholds_path: Optional[str | Path] = None,
        pipeline: Pipeline | None = None,
    ) -> None:
        if max_buffer < 1:
            raise ValueError("max_buffer must be positive")

        self.max_buffer = int(max_buffer)
        self.method = method
        self.keep_year = keep_year
        self.shift_dates = shift_dates
        self.date_shift_days = date_shift_days
        self.keep_mapping = keep_mapping
        self.consistent = consistent
        self.seed = seed
        self.locale = locale

        self.pipeline = pipeline or Pipeline(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            config=config,
            use_smart_merging=use_smart_merging,
            lang=lang,
            normalize_accents=normalize_accents,
            use_safety_sweep=use_safety_sweep,
            loader=loader,
            policy=policy,
            calibration_thresholds_path=(
                str(calibration_thresholds_path)
                if calibration_thresholds_path is not None
                else None
            ),
        )
        secret = getattr(self.pipeline, "hmac_secret", DEFAULT_HASH_SECRET)
        self._input_hmac = _new_hmac(secret)
        self._redacted_hmac = _new_hmac(secret)

        self._pending = ""
        self._base_offset = 0
        self._normalized_length = 0
        self._redacted_length = 0
        self._seen_content = False
        self._closed = False
        self._chunk_count = 0
        self._max_observed_buffer = 0
        self._spans: tuple[OpenMedSpan, ...] = ()
        self._audit_record: Mapping[str, Any] | None = None
        self._last_window_audit: Mapping[str, Any] | None = None

    @property
    def carry_buffer_length(self) -> int:
        """Return the current carry-over buffer length without exposing text."""

        return len(self._pending)

    @property
    def max_observed_buffer(self) -> int:
        """Return the largest carry-over buffer length observed so far."""

        return self._max_observed_buffer

    @property
    def spans(self) -> tuple[OpenMedSpan, ...]:
        """Return final global-offset spans after :meth:`flush` completes."""

        return self._spans

    @property
    def audit_record(self) -> Mapping[str, Any] | None:
        """Return the aggregate stream audit record after :meth:`flush`."""

        return self._audit_record

    def feed(self, chunk: str) -> tuple[StreamingDeidentificationEvent, ...]:
        """Feed one source chunk and return safely-finalized redacted events."""

        if self._closed:
            raise RuntimeError("cannot feed chunks after flush")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if not chunk:
            return ()

        self._chunk_count += 1
        chunk = self._drop_leading_document_whitespace(chunk)
        if not chunk:
            self._remember_buffer_size()
            return ()

        self._pending += chunk
        if len(self._pending) <= self.max_buffer:
            self._remember_buffer_size()
            return ()

        window = self._run_window(self._pending)
        cutoff = self._safe_cutoff(window.spans, window.entities)
        if cutoff <= 0:
            if len(self._pending) > self.max_buffer:
                raise StreamingBufferError(
                    "carry-over buffer exceeded max_buffer before any prefix "
                    "could be finalized; increase max_buffer for this stream"
                )
            return ()
        if len(self._pending) - cutoff > self.max_buffer:
            raise StreamingBufferError(
                "carry-over buffer exceeded max_buffer because the unsafe tail "
                "could not be finalized; increase max_buffer for this stream"
            )

        return (self._emit_prefix(cutoff, window),)

    def flush(self) -> tuple[StreamingDeidentificationEvent, ...]:
        """Drain the stream tail and return the final event with audit data."""

        if self._closed:
            return ()
        self._closed = True

        tail = self._pending.rstrip()
        events: list[StreamingDeidentificationEvent] = []
        if tail:
            window = self._run_window(tail)
            events.append(self._emit_prefix(len(tail), window, final=False))
        self._pending = ""
        self._remember_buffer_size()

        audit_record = self._finalize_audit_record()
        events.append(
            StreamingDeidentificationEvent(
                redacted_text="",
                final=True,
                spans=self._spans,
                audit_record=audit_record,
            )
        )
        return tuple(events)

    def _drop_leading_document_whitespace(self, chunk: str) -> str:
        if self._seen_content:
            return chunk
        stripped = chunk.lstrip()
        if stripped:
            self._seen_content = True
        return stripped

    def _run_window(self, text: str) -> _WindowResult:
        leading_length = len(text) - len(text.lstrip())
        stripped = text.strip()
        trailing_length = len(text) - len(text.rstrip())

        if not stripped:
            return _WindowResult(
                redacted_text=text,
                entities=(),
                spans=(),
                audit_record=None,
            )

        result = self.pipeline.run(
            stripped,
            method=self.method,
            keep_year=self.keep_year,
            shift_dates=self.shift_dates,
            date_shift_days=self.date_shift_days,
            keep_mapping=self.keep_mapping,
            consistent=self.consistent,
            seed=self.seed,
            locale=self.locale,
        )
        self._last_window_audit = result.audit_record

        prefix = text[:leading_length]
        suffix = text[len(text) - trailing_length :] if trailing_length else ""
        entities = tuple(
            _shift_entity(entity, leading_length)
            for entity in result.deidentification_result.pii_entities
        )
        spans = tuple(
            replace(
                span,
                start=span.start + leading_length,
                end=span.end + leading_length,
            )
            for span in result.spans
        )
        return _WindowResult(
            redacted_text=f"{prefix}{result.redacted_text}{suffix}",
            entities=entities,
            spans=spans,
            audit_record=result.audit_record,
        )

    def _safe_cutoff(
        self,
        spans: Sequence[OpenMedSpan],
        entities: Sequence[Any],
    ) -> int:
        cutoff = len(self._pending) - self.max_buffer
        cutoff = _advance_cutoff_past_redactions(self._pending, cutoff, entities)
        cutoff = _adjust_text_boundary(self._pending, cutoff)

        changed = True
        while changed:
            changed = False
            advanced = _advance_cutoff_past_redactions(
                self._pending,
                cutoff,
                entities,
            )
            if advanced != cutoff:
                cutoff = advanced
                changed = True
            for span in spans:
                if span.start < cutoff < span.end:
                    cutoff = span.start
                    changed = True
            adjusted = _adjust_text_boundary(self._pending, cutoff)
            if adjusted != cutoff:
                cutoff = adjusted
                changed = True

        return max(0, cutoff)

    def _emit_prefix(
        self,
        cutoff: int,
        window: _WindowResult,
        *,
        final: bool = False,
    ) -> StreamingDeidentificationEvent:
        original_text = self._pending[:cutoff]
        entities = tuple(entity for entity in window.entities if entity.end <= cutoff)
        redacted_text = _render_redacted_prefix(original_text, entities, cutoff)
        spans = tuple(span for span in window.spans if span.end <= cutoff)

        global_spans = tuple(
            replace(
                span,
                start=span.start + self._base_offset,
                end=span.end + self._base_offset,
            )
            for span in spans
        )
        self._spans = (*self._spans, *global_spans)
        self._update_hashes(original_text, redacted_text)

        self._pending = self._pending[cutoff:]
        self._base_offset += cutoff
        self._remember_buffer_size()
        return StreamingDeidentificationEvent(
            redacted_text=redacted_text,
            final=final,
        )

    def _update_hashes(self, original_text: str, redacted_text: str) -> None:
        if original_text:
            normalized = self.pipeline.stage1_normalize(original_text).normalized_text
            self._input_hmac.update(normalized.encode("utf-8"))
            self._normalized_length += len(normalized)
        if redacted_text:
            self._redacted_hmac.update(redacted_text.encode("utf-8"))
            self._redacted_length += len(redacted_text)

    def _finalize_audit_record(self) -> Mapping[str, Any]:
        input_hash = _digest_text_hash(self._input_hmac)
        redacted_hash = _digest_text_hash(self._redacted_hmac)
        doc_id = input_hash
        self._spans = tuple(replace(span, doc_id=doc_id) for span in self._spans)

        last = dict(self._last_window_audit or {})
        record: dict[str, Any] = {
            "doc_id": doc_id,
            "language": last.get("language", getattr(self.pipeline, "lang", "en")),
            "script": last.get("script", "latin"),
            "model_name": last.get("model_name", getattr(self.pipeline, "model_name")),
            "input_text_hash": input_hash,
            "redacted_text_hash": redacted_hash,
            "normalized_length": self._normalized_length,
            "redacted_length": self._redacted_length,
            "span_count": len(self._spans),
            "spans": [span.to_dict() for span in self._spans],
            "stages": [
                {
                    "stage": 10,
                    "name": STAGE_NAMES[9],
                    "span_count": len(self._spans),
                    "metadata": {
                        "redacted_text_hash": redacted_hash,
                        "audit_stage_count": 10,
                    },
                }
            ],
            "stream": {
                "chunks": self._chunk_count,
                "max_buffer": self.max_buffer,
                "max_observed_buffer": self._max_observed_buffer,
            },
        }
        if "policy" in last:
            record["policy"] = last["policy"]
        self._audit_record = record
        return record

    def _remember_buffer_size(self) -> None:
        self._max_observed_buffer = max(
            self._max_observed_buffer,
            len(self._pending),
        )


def deidentify_stream(
    chunks: Iterable[str],
    **kwargs: Any,
) -> Iterator[StreamingDeidentificationEvent]:
    """Yield streaming de-identification events for ``chunks``."""

    streamer = StreamingDeidentifier(**kwargs)
    for chunk in chunks:
        yield from streamer.feed(chunk)
    yield from streamer.flush()


def _render_redacted_prefix(
    text: str,
    entities: Sequence[Any],
    cutoff: int,
) -> str:
    rendered: list[str] = []
    cursor = 0
    for entity in sorted(entities, key=lambda item: (item.start, item.end)):
        start = int(entity.start)
        end = int(entity.end)
        if start < cursor or start >= cutoff or end > cutoff:
            continue
        rendered.append(text[cursor:start])
        rendered.append(entity.redacted_text or "")
        cursor = end
    rendered.append(text[cursor:cutoff])
    return "".join(rendered)


def _shift_entity(entity: Any, offset: int) -> Any:
    shifted = copy.copy(entity)
    shifted.start = int(entity.start) + offset
    shifted.end = int(entity.end) + offset
    return shifted


def _adjust_text_boundary(text: str, cutoff: int) -> int:
    if cutoff <= 0 or cutoff >= len(text):
        return cutoff

    while cutoff > 0 and cutoff < len(text) and unicodedata.combining(text[cutoff]):
        cutoff -= 1

    if cutoff > 0 and cutoff < len(text):
        left = text[cutoff - 1]
        right = text[cutoff]
        if _is_identifier_token_char(left) and _is_identifier_token_char(right):
            token_start = cutoff - 1
            while token_start > 0 and _is_identifier_token_char(text[token_start - 1]):
                token_start -= 1
            token_end = cutoff
            while token_end < len(text) and _is_identifier_token_char(text[token_end]):
                token_end += 1
            if _is_identifier_like_token(text[token_start:token_end]):
                cutoff = token_start
                left = text[cutoff - 1] if cutoff > 0 else ""
                right = text[cutoff] if cutoff < len(text) else ""
        if left.isspace() and right.isspace():
            run_start = cutoff - 1
            while run_start > 0 and text[run_start - 1].isspace():
                run_start -= 1
            run_end = cutoff
            while run_end < len(text) and text[run_end].isspace():
                run_end += 1
            cutoff = run_end if run_end < len(text) else run_start

    last_non_whitespace = -1
    for index, char in enumerate(text):
        if not char.isspace():
            last_non_whitespace = index
    if last_non_whitespace >= 0 and cutoff > last_non_whitespace + 1:
        cutoff = last_non_whitespace + 1

    return cutoff


def _advance_cutoff_past_redactions(
    text: str,
    cutoff: int,
    entities: Sequence[Any],
) -> int:
    changed = True
    while changed:
        changed = False
        for entity in entities:
            start = int(entity.start)
            end = int(entity.end)
            if start <= cutoff < end and _has_safe_redaction_boundary(text, end):
                cutoff = end
                changed = True
    return cutoff


def _has_safe_redaction_boundary(text: str, end: int) -> bool:
    return end < len(text) and not _is_identifier_token_char(text[end])


def _is_identifier_token_char(char: str) -> bool:
    return char.isalnum() or char in "._%+-@:/#?&=~"


def _is_identifier_like_token(token: str) -> bool:
    return any(char.isdigit() or char in "._%+-@:/#?&=~" for char in token)


def _new_hmac(secret: str | bytes) -> hmac.HMAC:
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(key, digestmod=hashlib.sha256)


def _digest_text_hash(digest: hmac.HMAC) -> str:
    return f"hmac-sha256:{digest.copy().hexdigest()}"


__all__ = [
    "StreamingBufferError",
    "StreamingDeidentificationEvent",
    "StreamingDeidentifier",
    "deidentify_stream",
]
