"""Local JSON-lines sidecar for desktop de-identification.

The protocol reserves stdout for one JSON response per request and stderr for
structured, PHI-free lifecycle records.  The runtime is deliberately
local-only: model files must already be present in the OpenMed cache or be
addressed by a local path.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import signal
import sys
from collections import OrderedDict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Callable, Mapping, TextIO

from openmed.utils.gateway import (
    DEFAULT_MAX_TEXT_BYTES,
    DEFAULT_MAX_TEXT_CHARS,
    GatewayLimits,
    InputValidationError,
    normalize_text,
)

PROTOCOL_VERSION = 1
SIDECAR_HASH_SECRET_ENV_VAR = "OPENMED_SIDECAR_HASH_SECRET"
DEFAULT_MODEL_ENV_VAR = "OPENMED_SIDECAR_MODEL"
MAX_REQUEST_ID_LENGTH = 128
MAX_DOC_ID_LENGTH = 256
MAX_MODEL_NAME_LENGTH = 4_096
MAX_POLICY_LENGTH = 128
MAX_RESPONSE_SPANS = 65_536
MAX_DEIDENTIFIED_TEXT_CHARS = 8_000_000
MAX_DEIDENTIFIED_TEXT_BYTES = 32_000_000
MAX_RESPONSE_LINE_BYTES = 64_000_000
MAX_CACHED_PIPELINES = 8
MAX_REQUEST_LINE_CHARS = DEFAULT_MAX_TEXT_CHARS * 12 + 65_536
_DEFAULT_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_METHODS = frozenset({"mask", "remove", "replace", "hash", "format_preserve"})
_OPTION_KEYS = frozenset(
    {
        "confidence_threshold",
        "deterministic_only",
        "doc_id",
        "lang",
        "method",
        "policy",
        "use_safety_sweep",
        "use_smart_merging",
    }
)
_LOG_FIELDS = frozenset(
    {
        "duration_ms",
        "error_code",
        "event",
        "input_length",
        "operation",
        "protocol_version",
        "request_id_hash",
        "span_count",
    }
)
_REQUEST_KEYS = {
    "ping": frozenset({"id", "operation"}),
    "shutdown": frozenset({"id", "operation"}),
    "deidentify": frozenset({"id", "operation", "options", "text"}),
}


class SidecarRequestError(ValueError):
    """A safe validation error suitable for the sidecar error envelope."""


class _DiscardStream:
    """Drop dependency output so protocol and lifecycle streams stay PHI-free."""

    encoding = "utf-8"

    def write(self, value: str) -> int:
        return len(value)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


_DISCARD_STREAM = _DiscardStream()


@dataclass(frozen=True)
class DeidentifyOptions:
    """Validated options used to construct and cache one privacy pipeline."""

    model_name: str = _DEFAULT_MODEL
    policy: str = "hipaa_safe_harbor"
    method: str = "mask"
    confidence_threshold: float = 0.7
    lang: str = "en"
    doc_id: str | None = None
    use_smart_merging: bool = True
    use_safety_sweep: bool = True
    deterministic_only: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "DeidentifyOptions":
        """Validate a protocol ``options`` object without echoing its values."""

        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise SidecarRequestError("options must be a JSON object")
        if set(payload) - _OPTION_KEYS:
            raise SidecarRequestError("options contains unsupported fields")

        model_name = _configured_model_name()
        policy = _optional_string(
            payload.get("policy", "hipaa_safe_harbor"),
            field="policy",
            max_length=MAX_POLICY_LENGTH,
        )
        if policy is None:
            policy = "hipaa_safe_harbor"
        method = _optional_string(
            payload.get("method", "mask"),
            field="method",
            max_length=32,
        )
        if method not in _METHODS:
            raise SidecarRequestError("method is not supported by the sidecar")
        lang = _optional_string(
            payload.get("lang", "en"),
            field="lang",
            max_length=16,
        )
        if lang is None:
            raise SidecarRequestError("lang must be a short language identifier")
        doc_id = _optional_string(
            payload.get("doc_id"),
            field="doc_id",
            max_length=MAX_DOC_ID_LENGTH,
        )

        threshold = payload.get("confidence_threshold", 0.7)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise SidecarRequestError("confidence_threshold must be a number")
        confidence_threshold = float(threshold)
        if not 0.0 <= confidence_threshold <= 1.0:
            raise SidecarRequestError(
                "confidence_threshold must be between 0.0 and 1.0"
            )

        return cls(
            model_name=model_name,
            policy=policy,
            method=method,
            confidence_threshold=confidence_threshold,
            lang=lang,
            doc_id=doc_id,
            use_smart_merging=_boolean_option(payload, "use_smart_merging", True),
            use_safety_sweep=_boolean_option(payload, "use_safety_sweep", True),
            deterministic_only=_boolean_option(payload, "deterministic_only", False),
        )

    def pipeline_key(self) -> tuple[Any, ...]:
        """Return the fields that affect reusable pipeline construction."""

        return (
            self.model_name,
            self.policy,
            self.confidence_threshold,
            self.lang,
            self.use_smart_merging,
            self.use_safety_sweep,
            self.deterministic_only,
        )


class SidecarRuntime:
    """Own a local-only model loader and reusable de-identification pipelines."""

    def __init__(
        self,
        *,
        loader_factory: Callable[[Any], Any] | None = None,
        hmac_secret: str | bytes | None = None,
    ) -> None:
        from openmed.core.config import OpenMedConfig

        self.config = OpenMedConfig(local_only=True)
        self._loader_factory = loader_factory
        self.loader: Any = None
        self.hmac_secret = hmac_secret or _sidecar_hash_secret()
        self._pipelines: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._closed = False

    def deidentify(self, text: str, options: DeidentifyOptions) -> dict[str, Any]:
        """Run one request without permitting network egress."""

        from openmed.core.offline import network_blocked_if_offline

        with (
            redirect_stdout(_DISCARD_STREAM),
            redirect_stderr(_DISCARD_STREAM),
            network_blocked_if_offline(self.config, local_only=True),
        ):
            pipeline = self._pipeline_for(options)
            result = pipeline.run(
                text,
                method=options.method,
                doc_id=options.doc_id,
            )
        redacted_text = result.redacted_text
        if (
            not isinstance(redacted_text, str)
            or len(redacted_text) > MAX_DEIDENTIFIED_TEXT_CHARS
        ):
            raise RuntimeError("sidecar pipeline returned invalid redacted text")
        try:
            redacted_bytes = redacted_text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "sidecar pipeline returned invalid redacted text"
            ) from exc
        if len(redacted_bytes) > MAX_DEIDENTIFIED_TEXT_BYTES:
            raise RuntimeError("sidecar pipeline returned invalid redacted text")
        spans = _response_spans(result)
        if not _valid_response_spans(spans, len(text)):
            raise RuntimeError("sidecar pipeline returned too many spans")
        return {
            "deidentified_text": redacted_text,
            "spans": [span.to_dict() for span in spans],
        }

    def close(self) -> None:
        """Release cached model objects exactly once."""

        if self._closed:
            return
        self._closed = True
        unload = getattr(self.loader, "unload_all_models", None)
        if callable(unload):
            from openmed.core.offline import network_blocked_if_offline

            try:
                with (
                    redirect_stdout(_DISCARD_STREAM),
                    redirect_stderr(_DISCARD_STREAM),
                    network_blocked_if_offline(self.config, local_only=True),
                ):
                    unload()
            except Exception:
                # Cleanup is best-effort and must not emit dependency details.
                pass
        self._pipelines.clear()

    def _pipeline_for(self, options: DeidentifyOptions) -> Any:
        key = options.pipeline_key()
        cached = self._pipelines.get(key)
        if cached is not None:
            self._pipelines.move_to_end(key)
            return cached

        from openmed.core.pipeline import Pipeline

        model_detector = _empty_model_result if options.deterministic_only else None
        loader = None if options.deterministic_only else self._model_loader()
        pipeline = Pipeline(
            model_name=options.model_name,
            confidence_threshold=options.confidence_threshold,
            config=self.config,
            use_smart_merging=options.use_smart_merging,
            lang=options.lang,
            use_safety_sweep=options.use_safety_sweep,
            loader=loader,
            model_detector=model_detector,
            policy=options.policy,
            hmac_secret=self.hmac_secret,
        )
        self._pipelines[key] = pipeline
        self._pipelines.move_to_end(key)
        while len(self._pipelines) > MAX_CACHED_PIPELINES:
            self._pipelines.popitem(last=False)
        return pipeline

    def _model_loader(self) -> Any:
        if self.loader is None:
            if self._loader_factory is None:
                from openmed.core.models import ModelLoader

                factory = ModelLoader
            else:
                factory = self._loader_factory
            self.loader = factory(self.config)
        return self.loader


class SidecarServer:
    """Serve the versioned JSON-lines protocol over text streams."""

    def __init__(
        self,
        runtime: SidecarRuntime,
        *,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
    ) -> None:
        self.runtime = runtime
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self._stopping = False
        self._closed = False
        self._log_secret = secrets.token_bytes(32)

    def serve(self) -> int:
        """Process requests until EOF, shutdown, or process interruption."""

        previous_logging_level = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self._log(event="sidecar_started", protocol_version=PROTOCOL_VERSION)
        try:
            while not self._stopping:
                line = self.stdin.readline(MAX_REQUEST_LINE_CHARS + 1)
                if line == "":
                    break
                if len(line) > MAX_REQUEST_LINE_CHARS:
                    self._drain_oversized_line(line)
                    self._write_error(
                        None,
                        "INVALID_REQUEST",
                        "Invalid sidecar request.",
                    )
                    self._log(
                        event="request_failed",
                        operation="unknown",
                        request_id_hash=self._request_id_hash(None),
                        error_code="INVALID_REQUEST",
                        duration_ms=0.0,
                    )
                    continue
                if not line.strip():
                    continue
                self._handle_line(line)
            return 0
        finally:
            self.close()
            logging.disable(previous_logging_level)

    def stop(self) -> None:
        """Request a graceful stop after the current request."""

        self._stopping = True

    def close(self) -> None:
        """Release runtime resources and emit the terminal lifecycle event."""

        if self._closed:
            return
        self._closed = True
        self.runtime.close()
        self._log(event="sidecar_stopped", protocol_version=PROTOCOL_VERSION)

    def _handle_line(self, line: str) -> None:
        started = perf_counter()
        request_id: str | None = None
        operation = "unknown"
        try:
            payload = _decode_request(line)
            if not isinstance(payload, Mapping):
                raise SidecarRequestError("request must be a JSON object")
            request_id = _request_id(payload.get("id"))
            requested_operation = payload.get("operation")
            if not isinstance(requested_operation, str):
                raise SidecarRequestError("operation must be a string")
            if requested_operation not in {"ping", "shutdown", "deidentify"}:
                raise SidecarRequestError("operation is not supported")
            operation = requested_operation
            if set(payload) - _REQUEST_KEYS[operation]:
                raise SidecarRequestError("request contains unsupported fields")

            if operation == "ping":
                result: dict[str, Any] = {
                    "offline": True,
                    "protocol_version": PROTOCOL_VERSION,
                }
                span_count = 0
                input_length = 0
            elif operation == "shutdown":
                result = {"shutdown": True}
                span_count = 0
                input_length = 0
                self.stop()
            elif operation == "deidentify":
                text = payload.get("text")
                if not isinstance(text, str):
                    raise SidecarRequestError("text must be a string")
                try:
                    text = normalize_text(
                        text,
                        limits=GatewayLimits(
                            max_chars=DEFAULT_MAX_TEXT_CHARS,
                            max_bytes=DEFAULT_MAX_TEXT_BYTES,
                        ),
                        min_length=1,
                        allow_empty=False,
                        strip=False,
                    )
                except InputValidationError as exc:
                    raise SidecarRequestError("text is invalid") from exc
                options = DeidentifyOptions.from_payload(payload.get("options"))
                result = self.runtime.deidentify(text, options)
                span_count = len(result["spans"])
                input_length = len(text)
            self._write_response({"id": request_id, "ok": True, "result": result})
            self._log(
                event="request_completed",
                operation=operation,
                request_id_hash=self._request_id_hash(request_id),
                input_length=input_length,
                span_count=span_count,
                duration_ms=round((perf_counter() - started) * 1000.0, 3),
            )
        except SidecarRequestError:
            self._write_error(request_id, "INVALID_REQUEST", "Invalid sidecar request.")
            self._log(
                event="request_failed",
                operation=operation,
                request_id_hash=self._request_id_hash(request_id),
                error_code="INVALID_REQUEST",
                duration_ms=round((perf_counter() - started) * 1000.0, 3),
            )
        except Exception:
            self._write_error(
                request_id,
                "PROCESSING_FAILED",
                "De-identification failed; verify the local model bundle.",
            )
            self._log(
                event="request_failed",
                operation=operation,
                request_id_hash=self._request_id_hash(request_id),
                error_code="PROCESSING_FAILED",
                duration_ms=round((perf_counter() - started) * 1000.0, 3),
            )

    def _drain_oversized_line(self, first_chunk: str) -> None:
        if first_chunk.endswith(("\n", "\r")):
            return
        while True:
            chunk = self.stdin.readline(MAX_REQUEST_LINE_CHARS + 1)
            if chunk == "" or chunk.endswith(("\n", "\r")):
                return

    def _write_error(self, request_id: str | None, code: str, message: str) -> None:
        self._write_response(
            {
                "id": request_id,
                "ok": False,
                "error": {"code": code, "message": message},
            }
        )

    def _write_response(self, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(encoded.encode("utf-8", errors="strict")) + 1 > MAX_RESPONSE_LINE_BYTES:
            raise RuntimeError("sidecar response exceeds the protocol limit")
        self.stdout.write(encoded + "\n")
        self.stdout.flush()

    def _request_id_hash(self, request_id: str | None) -> str:
        encoded = (request_id or "missing").encode("utf-8")
        return (
            "hmac-sha256:"
            + hmac.new(
                self._log_secret,
                encoded,
                hashlib.sha256,
            ).hexdigest()
        )

    def _log(self, **payload: Any) -> None:
        if set(payload) - _LOG_FIELDS:
            raise RuntimeError("sidecar log payload contains a non-allow-listed field")
        try:
            self.stderr.write(
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            )
            self.stderr.flush()
        except (OSError, ValueError):
            # Logging is observational and must never change protocol behavior.
            pass


def _decode_request(line: str) -> Any:
    try:
        return json.loads(
            line,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (RecursionError, ValueError) as exc:
        raise SidecarRequestError("request is not valid JSON") from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SidecarRequestError("request contains duplicate fields")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> Any:
    raise SidecarRequestError("request contains a non-finite number")


def _sidecar_hash_secret() -> bytes:
    configured = os.getenv(SIDECAR_HASH_SECRET_ENV_VAR)
    if configured:
        return configured.encode("utf-8")
    return secrets.token_bytes(32)


def _empty_model_result(text: str, **_: Any) -> Any:
    del text
    return SimpleNamespace(entities=[], model_name="deterministic-only")


def _response_spans(result: Any) -> tuple[Any, ...]:
    entities = getattr(result.deidentification_result, "pii_entities", ()) or ()
    by_bounds = {
        (int(entity.start), int(entity.end)): entity
        for entity in entities
        if entity.start is not None and entity.end is not None
    }
    response = []
    for span in result.spans:
        entity = by_bounds.get((span.start, span.end))
        if entity is None:
            response.append(span)
            continue
        action = getattr(entity, "action", None) or span.action
        if action == "remove":
            action = "redact"
        response.append(
            replace(
                span,
                action=action,
                replacement=getattr(entity, "redacted_text", None),
                reversible_id=getattr(entity, "reversible_id", None),
            )
        )
    return tuple(response)


def _valid_response_spans(spans: tuple[Any, ...], text_length: int) -> bool:
    if len(spans) > MAX_RESPONSE_SPANS or len(spans) > text_length:
        return False
    previous_end = 0
    for span in sorted(spans, key=lambda value: (value.start, value.end)):
        if (
            isinstance(span.start, bool)
            or isinstance(span.end, bool)
            or not isinstance(span.start, int)
            or not isinstance(span.end, int)
            or span.start < previous_end
            or span.end <= span.start
            or span.end > text_length
        ):
            return False
        previous_end = span.end
    return True


def _request_id(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise SidecarRequestError("id must be a string or integer")
    request_id = str(value)
    if not request_id or len(request_id) > MAX_REQUEST_ID_LENGTH:
        raise SidecarRequestError("id is empty or too long")
    return request_id


def _configured_model_name() -> str:
    value = _optional_string(
        os.getenv(DEFAULT_MODEL_ENV_VAR, _DEFAULT_MODEL),
        field="configured model",
        max_length=MAX_MODEL_NAME_LENGTH,
    )
    if value is None:
        raise SidecarRequestError("configured model must be a non-empty string")
    return value


def _optional_string(
    value: Any,
    *,
    field: str,
    max_length: int,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SidecarRequestError(f"{field} must be a non-empty string")
    if len(value) > max_length:
        raise SidecarRequestError(f"{field} is too long")
    return value


def _boolean_option(payload: Mapping[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise SidecarRequestError(f"{name} must be a boolean")
    return value


def main() -> int:
    """Run the standalone sidecar process."""

    previous_logging_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        runtime = SidecarRuntime()
        server = SidecarServer(runtime)
    except Exception:
        _write_terminal_error("SIDECAR_START_FAILED")
        logging.disable(previous_logging_level)
        return 1

    def stop_server(_signum: int, _frame: Any) -> None:
        server.stop()
        raise SystemExit(0)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_server)
    try:
        return server.serve()
    except (BrokenPipeError, KeyboardInterrupt):
        return 0
    except Exception:
        _write_terminal_error("SIDECAR_RUNTIME_FAILED")
        return 1
    finally:
        server.close()
        logging.disable(previous_logging_level)


def _write_terminal_error(error_code: str) -> None:
    try:
        sys.stderr.write(
            json.dumps(
                {"error_code": error_code, "event": "sidecar_failed"},
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    raise SystemExit(main())
