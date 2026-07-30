"""gRPC service surface for OpenMed analysis and de-identification."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from concurrent import futures
from typing import Any, Callable

import grpc
from google.protobuf import struct_pb2

from openmed.core.streaming import deidentify_stream

from .app import _analyze_payload, _pii_deidentify_payload, _pii_extract_payload
from .proto.generated import openmed_pb2, openmed_pb2_grpc
from .runtime import ServiceRuntime
from .schemas import AnalyzeRequest, PIIDeidentifyRequest, PIIExtractRequest

DEFAULT_GRPC_ADDRESS = "[::]:50051"
DEFAULT_GRPC_MAX_WORKERS = 10


class OpenMedGrpcServicer(openmed_pb2_grpc.OpenMedServiceServicer):
    """Synchronous grpcio servicer backed by the shared service runtime."""

    def __init__(self, runtime: ServiceRuntime | None = None) -> None:
        self.runtime = runtime or ServiceRuntime.from_env()

    def Analyze(self, request, context):
        """Run the OpenMed analyze endpoint over gRPC."""
        return _run_unary(
            context,
            lambda: self._analyze(request),
        )

    def Extract(self, request, context):
        """Run the OpenMed PII extraction endpoint over gRPC."""
        return _run_unary(
            context,
            lambda: self._extract(request),
        )

    def Deidentify(self, request, context):
        """Run the OpenMed de-identification endpoint over gRPC."""
        return _run_unary(
            context,
            lambda: self._deidentify(request),
        )

    def StreamDeidentify(self, request, context):
        """Stream redacted text fragments for chunked de-identification."""
        chunks = _stream_chunks(request)
        fallback_text = "".join(chunks) if chunks else request.request.text
        try:
            payload = _parse_request(
                lambda proto: _deidentify_request_from_proto(
                    proto,
                    fallback_text=fallback_text,
                ),
                request.request,
            )
            max_buffer = int(request.max_buffer or 4096)
            if max_buffer < 1:
                raise ValueError("Invalid max_buffer")
        except ValueError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid request")

        model_key = self.runtime.begin_model_request(payload.model_name)
        try:
            for event in deidentify_stream(
                chunks,
                max_buffer=max_buffer,
                method=payload.method,
                model_name=payload.model_name,
                confidence_threshold=payload.confidence_threshold,
                keep_year=payload.keep_year,
                shift_dates=payload.shift_dates,
                date_shift_days=payload.date_shift_days,
                keep_mapping=payload.keep_mapping,
                config=self.runtime.config,
                use_smart_merging=payload.use_smart_merging,
                lang=payload.lang,
                normalize_accents=payload.normalize_accents,
                use_safety_sweep=payload.use_safety_sweep,
                loader=self.runtime.get_loader(),
                policy=payload.policy,
            ):
                yield _stream_response(event)
        except ValueError:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid request")
        except Exception:
            context.abort(grpc.StatusCode.INTERNAL, "Internal server error")
        finally:
            self.runtime.finish_model_request(model_key, payload.keep_alive)

    def _analyze(self, request) -> openmed_pb2.AnalyzeResponse:
        payload = _parse_request(_analyze_request_from_proto, request)
        return _analyze_response(
            self.runtime.run_model_request(
                payload.model_name,
                payload.keep_alive,
                lambda: _analyze_payload(payload, self.runtime),
            )
        )

    def _extract(self, request) -> openmed_pb2.ExtractResponse:
        payload = _parse_request(_extract_request_from_proto, request)
        return _extract_response(
            self.runtime.run_model_request(
                payload.model_name,
                payload.keep_alive,
                lambda: _pii_extract_payload(payload, self.runtime),
            )
        )

    def _deidentify(self, request) -> openmed_pb2.DeidentifyResponse:
        payload = _parse_request(_deidentify_request_from_proto, request)
        return _deidentify_response(
            self.runtime.run_model_request(
                payload.model_name,
                payload.keep_alive,
                lambda: _pii_deidentify_payload(payload, self.runtime),
            )
        )


def create_grpc_server(
    *,
    runtime: ServiceRuntime | None = None,
    max_workers: int = DEFAULT_GRPC_MAX_WORKERS,
) -> grpc.Server:
    """Create a grpcio server with the OpenMed service registered."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    openmed_pb2_grpc.add_OpenMedServiceServicer_to_server(
        OpenMedGrpcServicer(runtime),
        server,
    )
    return server


def serve(
    address: str = DEFAULT_GRPC_ADDRESS,
    *,
    runtime: ServiceRuntime | None = None,
    max_workers: int = DEFAULT_GRPC_MAX_WORKERS,
) -> grpc.Server:
    """Start an insecure local gRPC server and return the running server."""
    server = create_grpc_server(runtime=runtime, max_workers=max_workers)
    server.add_insecure_port(address)
    server.start()
    return server


def _run_unary(context: grpc.ServicerContext, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except ValueError:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid request")
    except Exception:
        context.abort(grpc.StatusCode.INTERNAL, "Internal server error")


def _parse_request(factory: Callable[[Any], Any], request: Any) -> Any:
    try:
        return factory(request)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Invalid request") from exc


def _analyze_request_from_proto(request) -> AnalyzeRequest:
    payload = _request_payload(
        request,
        (
            "model_name",
            "confidence_threshold",
            "group_entities",
            "aggregation_strategy",
            "sentence_detection",
            "sentence_language",
            "sentence_clean",
            "use_fast_tokenizer",
            "keep_alive",
        ),
    )
    return AnalyzeRequest(**payload)


def _extract_request_from_proto(request) -> PIIExtractRequest:
    payload = _request_payload(
        request,
        (
            "model_name",
            "confidence_threshold",
            "use_smart_merging",
            "lang",
            "normalize_accents",
            "keep_alive",
        ),
    )
    return PIIExtractRequest(**payload)


def _deidentify_request_from_proto(
    request,
    *,
    fallback_text: str | None = None,
) -> PIIDeidentifyRequest:
    payload = _request_payload(
        request,
        (
            "method",
            "model_name",
            "confidence_threshold",
            "keep_year",
            "shift_dates",
            "date_shift_days",
            "keep_mapping",
            "policy",
            "use_smart_merging",
            "use_safety_sweep",
            "lang",
            "normalize_accents",
            "keep_alive",
        ),
        fallback_text=fallback_text,
    )
    return PIIDeidentifyRequest(**payload)


def _request_payload(
    request,
    optional_fields: Sequence[str],
    *,
    fallback_text: str | None = None,
) -> dict[str, Any]:
    text = request.text if request.text else fallback_text
    payload: dict[str, Any] = {"text": text}
    for field_name in optional_fields:
        if _has_field(request, field_name):
            payload[field_name] = getattr(request, field_name)
    return payload


def _has_field(message, field_name: str) -> bool:
    try:
        return bool(message.HasField(field_name))
    except ValueError:
        return getattr(message, field_name) is not None


def _optional_text(message, field_name: str) -> str | None:
    if _has_field(message, field_name):
        value = getattr(message, field_name)
        if value != "":
            return str(value)
    return None


def _analyze_response(payload: Mapping[str, Any]) -> openmed_pb2.AnalyzeResponse:
    response = openmed_pb2.AnalyzeResponse(
        text=str(payload.get("text") or ""),
        model_name=str(payload.get("model_name") or ""),
        timestamp=str(payload.get("timestamp") or ""),
    )
    _copy_optional_float(response, "processing_time", payload.get("processing_time"))
    response.metadata.CopyFrom(_struct_from_mapping(payload.get("metadata") or {}))
    response.entities.extend(
        _entity_prediction(entity) for entity in payload.get("entities") or ()
    )
    response.spans.extend(_openmed_span(span) for span in payload.get("spans") or ())
    return response


def _extract_response(payload: Mapping[str, Any]) -> openmed_pb2.ExtractResponse:
    response = openmed_pb2.ExtractResponse(
        text=str(payload.get("text") or ""),
        model_name=str(payload.get("model_name") or ""),
        timestamp=str(payload.get("timestamp") or ""),
    )
    _copy_optional_float(response, "processing_time", payload.get("processing_time"))
    response.metadata.CopyFrom(_struct_from_mapping(payload.get("metadata") or {}))
    response.entities.extend(
        _entity_prediction(entity) for entity in payload.get("entities") or ()
    )
    response.spans.extend(_openmed_span(span) for span in payload.get("spans") or ())
    return response


def _deidentify_response(
    payload: Mapping[str, Any],
) -> openmed_pb2.DeidentifyResponse:
    response = openmed_pb2.DeidentifyResponse(
        original_text=str(payload.get("original_text") or ""),
        deidentified_text=str(payload.get("deidentified_text") or ""),
        method=str(payload.get("method") or ""),
        timestamp=str(payload.get("timestamp") or ""),
        num_entities_redacted=int(payload.get("num_entities_redacted") or 0),
    )
    response.metadata.CopyFrom(_struct_from_mapping(payload.get("metadata") or {}))
    response.pii_entities.extend(
        _pii_entity(entity) for entity in payload.get("pii_entities") or ()
    )
    response.spans.extend(_openmed_span(span) for span in payload.get("spans") or ())
    mapping = payload.get("mapping")
    if isinstance(mapping, Mapping):
        response.mapping.CopyFrom(_struct_from_mapping(mapping))
    return response


def _stream_response(event: Any) -> openmed_pb2.DeidentifyStreamResponse:
    response = openmed_pb2.DeidentifyStreamResponse(
        redacted_text=str(getattr(event, "redacted_text", "")),
        final=bool(getattr(event, "final", False)),
    )
    response.spans.extend(_openmed_span(span) for span in getattr(event, "spans", ()))
    audit_record = getattr(event, "audit_record", None)
    if isinstance(audit_record, Mapping):
        response.audit_record.CopyFrom(_struct_from_mapping(audit_record))
    return response


def _entity_prediction(item: Any) -> openmed_pb2.EntityPrediction:
    data = _plain_mapping(item)
    message = openmed_pb2.EntityPrediction(
        text=str(data.get("text") or ""),
        label=str(data.get("label") or ""),
        confidence=float(data.get("confidence") or 0.0),
    )
    _copy_optional_int(message, "start", data.get("start"))
    _copy_optional_int(message, "end", data.get("end"))
    message.metadata.CopyFrom(_struct_from_mapping(data.get("metadata") or {}))
    return message


def _pii_entity(item: Any) -> openmed_pb2.PIIEntity:
    data = _plain_mapping(item)
    message = openmed_pb2.PIIEntity(
        text=str(data.get("text") or ""),
        label=str(data.get("label") or ""),
        entity_type=str(data.get("entity_type") or data.get("label") or ""),
        confidence=float(data.get("confidence") or 0.0),
    )
    _copy_optional_int(message, "start", data.get("start"))
    _copy_optional_int(message, "end", data.get("end"))
    _copy_optional_text(message, "redacted_text", data.get("redacted_text"))
    _copy_optional_text(message, "canonical_label", data.get("canonical_label"))
    message.sources.extend(str(source) for source in data.get("sources") or ())
    message.evidence.CopyFrom(_struct_from_mapping(data.get("evidence") or {}))
    _copy_optional_float(message, "threshold", data.get("threshold"))
    _copy_optional_text(message, "action", data.get("action"))
    _copy_optional_text(message, "surrogate", data.get("surrogate"))
    _copy_optional_text(message, "reversible_id", data.get("reversible_id"))
    message.metadata.CopyFrom(_struct_from_mapping(data.get("metadata") or {}))
    return message


def _openmed_span(item: Any) -> openmed_pb2.OpenMedSpan:
    data = _plain_mapping(item)
    message = openmed_pb2.OpenMedSpan(
        schema_version=int(data.get("schema_version") or 1),
        doc_id=str(data.get("doc_id") or ""),
        start=int(data.get("start") or 0),
        end=int(data.get("end") or 0),
        text_hash=str(data.get("text_hash") or ""),
        entity_type=str(data.get("entity_type") or ""),
        canonical_label=str(data.get("canonical_label") or ""),
        action=str(data.get("action") or "keep"),
    )
    _copy_optional_text(message, "policy_label", data.get("policy_label"))
    message.regulatory_tags.extend(
        str(tag) for tag in data.get("regulatory_tags") or ()
    )
    _copy_optional_float(message, "score", data.get("score"))
    _copy_optional_text(message, "detector", data.get("detector"))
    message.evidence.CopyFrom(_struct_from_mapping(data.get("evidence") or {}))
    _copy_optional_text(message, "replacement", data.get("replacement"))
    _copy_optional_text(message, "reversible_id", data.get("reversible_id"))
    _copy_optional_text(message, "section", data.get("section"))
    message.metadata.CopyFrom(_struct_from_mapping(data.get("metadata") or {}))
    return message


def _copy_optional_text(message: Any, field_name: str, value: Any) -> None:
    if value is not None:
        setattr(message, field_name, str(value))


def _copy_optional_int(message: Any, field_name: str, value: Any) -> None:
    if value is not None:
        setattr(message, field_name, int(value))


def _copy_optional_float(message: Any, field_name: str, value: Any) -> None:
    if value is not None:
        setattr(message, field_name, float(value))


def _struct_from_mapping(value: Mapping[str, Any]) -> struct_pb2.Struct:
    structure = struct_pb2.Struct()
    if value:
        structure.update(_json_safe(value))
    return structure


def _plain_mapping(item: Any) -> dict[str, Any]:
    if hasattr(item, "to_dict") and callable(item.to_dict):
        item = item.to_dict()
    if isinstance(item, Mapping):
        return dict(item)
    return dict(vars(item))


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, item in value.items():
        safe[str(key)] = _json_safe_value(item)
    return safe


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _stream_chunks(request) -> tuple[str, ...]:
    if request.chunks:
        return tuple(str(chunk) for chunk in request.chunks)
    text = request.request.text
    chunk_size = int(request.chunk_size or 0)
    if chunk_size <= 0 or chunk_size >= len(text):
        return (text,)
    return tuple(_chunk_text(text, chunk_size))


def _chunk_text(text: str, chunk_size: int) -> Iterable[str]:
    for start in range(0, len(text), chunk_size):
        yield text[start : start + chunk_size]


__all__ = [
    "DEFAULT_GRPC_ADDRESS",
    "DEFAULT_GRPC_MAX_WORKERS",
    "OpenMedGrpcServicer",
    "create_grpc_server",
    "serve",
]
