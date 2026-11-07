"""gRPC service surface for OpenMed analysis and de-identification.

Every server built here installs
:class:`~openmed.service.grpc_policy.GrpcSecurityInterceptor` ahead of any
caller interceptors and applies the message-size limits of its
:class:`~openmed.service.grpc_policy.GrpcLimits`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent import futures
from typing import Any, Callable

import grpc
from google.protobuf import struct_pb2

from openmed.core.streaming import deidentify_stream

from .app import _analyze_payload, _pii_deidentify_payload, _pii_extract_payload
from .grpc_policy import (
    REASON_INTERNAL_ERROR,
    REASON_INVALID_REQUEST,
    REASON_MODEL_NOT_SERVED,
    GrpcLimits,
    GrpcPolicyError,
    GrpcSecurityInterceptor,
    GrpcServicePolicy,
    abort_rpc,
    listener_client_ca_certificates,
    require_authenticated_listener,
    validate_grpc_rpc_policies,
)
from .grpc_security import (
    GrpcListener,
    GrpcServerCredentials,
    GrpcStartupError,
    GrpcTLSConfig,
    bind_grpc_listener,
    resolve_grpc_listener,
)
from .proto.generated import openmed_pb2, openmed_pb2_grpc
from .runtime import ModelNotServedError, ServiceRuntime
from .schemas import AnalyzeRequest, PIIDeidentifyRequest, PIIExtractRequest

DEFAULT_GRPC_ADDRESS = "127.0.0.1:50051"
DEFAULT_GRPC_MAX_WORKERS = 10


class OpenMedGrpcServicer(openmed_pb2_grpc.OpenMedServiceServicer):
    """Synchronous grpcio servicer backed by the shared service runtime."""

    def __init__(
        self,
        runtime: ServiceRuntime | None = None,
        *,
        limits: GrpcLimits | None = None,
    ) -> None:
        """Create the servicer.

        Args:
            runtime: Shared service runtime; defaults to ``ServiceRuntime.from_env``.
            limits: Text and stream budgets checked before any parsing or
                model work; defaults to :meth:`GrpcLimits.from_env`.
        """
        self.runtime = runtime or ServiceRuntime.from_env()
        self.limits = (
            limits if limits is not None else GrpcLimits.from_env(runtime=self.runtime)
        )

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
        """Stream redacted text fragments for chunked de-identification.

        Chunk count, chunk size, total text and the caller's ``max_buffer``
        are checked against the server limits before chunks are joined,
        parsed, or handed to a model.
        """
        try:
            chunks = self.limits.stream_chunks(request)
            max_buffer = self.limits.stream_buffer(request.max_buffer)
            fallback_text = "".join(chunks) if chunks else request.request.text
            payload = _parse_request(
                lambda proto: _deidentify_request_from_proto(
                    proto,
                    fallback_text=fallback_text,
                ),
                request.request,
            )
        except GrpcPolicyError as exc:
            abort_rpc(context, exc.reason)
        except ValueError:
            abort_rpc(context, REASON_INVALID_REQUEST)

        try:
            model_key = self.runtime.begin_model_request(payload.model_name)
        except ModelNotServedError:
            abort_rpc(context, REASON_MODEL_NOT_SERVED)
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
            abort_rpc(context, REASON_INVALID_REQUEST)
        except Exception:
            abort_rpc(context, REASON_INTERNAL_ERROR)
        finally:
            self.runtime.finish_model_request(model_key, payload.keep_alive)

    def _analyze(self, request) -> openmed_pb2.AnalyzeResponse:
        self.limits.check_text(request.text)
        payload = _parse_request(_analyze_request_from_proto, request)
        return _analyze_response(
            self.runtime.run_model_request(
                payload.model_name,
                payload.keep_alive,
                lambda: _analyze_payload(payload, self.runtime),
            )
        )

    def _extract(self, request) -> openmed_pb2.ExtractResponse:
        self.limits.check_text(request.text)
        payload = _parse_request(_extract_request_from_proto, request)
        return _extract_response(
            self.runtime.run_model_request(
                payload.model_name,
                payload.keep_alive,
                lambda: _pii_extract_payload(payload, self.runtime),
            )
        )

    def _deidentify(self, request) -> openmed_pb2.DeidentifyResponse:
        self.limits.check_text(request.text)
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
    interceptors: Sequence[grpc.ServerInterceptor] = (),
    policy: GrpcServicePolicy | None = None,
    listener: GrpcListener | None = None,
) -> grpc.Server:
    """Create an unbound grpcio server with the OpenMed service registered.

    The server always runs :class:`GrpcSecurityInterceptor` first, then the
    caller's interceptors, and applies the policy's message-size limits.

    Args:
        runtime: Shared service runtime; defaults to ``ServiceRuntime.from_env``.
        max_workers: Worker threads for the synchronous server.
        interceptors: Extra server interceptors applied to every RPC, in order,
            after authentication and admission.
        policy: Authentication, throttle and budget policy; defaults to
            :meth:`GrpcServicePolicy.from_env`, which reads the same
            environment as the REST service.
        listener: The listener this server will be bound to. When given, a
            listener that other hosts can reach is refused unless the policy
            authenticates callers, and a mutual-TLS listener's client CAs
            enable certificate principals.

    Returns:
        A server with no listener. Bind it with
        :func:`openmed.service.grpc_security.bind_grpc_listener`.

    Raises:
        GrpcStartupError: ``grpc_auth_required`` when ``listener`` is
            reachable from other hosts and nothing authenticates callers.
    """
    validate_grpc_rpc_policies()
    runtime = runtime if runtime is not None else ServiceRuntime.from_env()
    policy = (
        policy if policy is not None else GrpcServicePolicy.from_env(runtime=runtime)
    )
    if listener is not None:
        require_authenticated_listener(listener, policy)
    security = GrpcSecurityInterceptor(
        policy,
        client_ca_certificates=listener_client_ca_certificates(listener),
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=(security, *interceptors),
        options=policy.limits.server_options(),
    )
    openmed_pb2_grpc.add_OpenMedServiceServicer_to_server(
        OpenMedGrpcServicer(runtime, limits=policy.limits),
        server,
    )
    return server


def serve(
    address: str = DEFAULT_GRPC_ADDRESS,
    *,
    runtime: ServiceRuntime | None = None,
    max_workers: int = DEFAULT_GRPC_MAX_WORKERS,
    tls: GrpcTLSConfig | GrpcServerCredentials | None = None,
    allow_insecure_remote_dev: bool = False,
    interceptors: Sequence[grpc.ServerInterceptor] = (),
    policy: GrpcServicePolicy | None = None,
) -> grpc.Server:
    """Validate the listener, bind it, start the server and return it.

    The default address is loopback-only plaintext. Any non-loopback address
    requires ``tls`` unless ``allow_insecure_remote_dev`` is set under the
    ``dev`` or ``test`` profile, and a non-loopback TLS listener also needs
    caller authentication: service API keys or JWTs, or mutual TLS.

    Args:
        address: The gRPC bind address. Defaults to ``127.0.0.1:50051``.
        runtime: Shared service runtime; defaults to ``ServiceRuntime.from_env``.
        max_workers: Worker threads for the synchronous server.
        tls: Server TLS files (or pre-validated credentials). Supplying a
            client CA enables mutual TLS.
        allow_insecure_remote_dev: Development-only override for plaintext on
            a non-loopback address; refused outside ``dev``/``test``.
        interceptors: Extra server interceptors applied to every RPC, in order,
            after authentication and admission.
        policy: Authentication, throttle and budget policy; defaults to
            :meth:`GrpcServicePolicy.from_env`.

    Returns:
        A started server that is listening on the requested address.

    Raises:
        GrpcStartupError: Before serving, when the listener policy refuses the
            configuration, credentials are invalid, a reachable listener has
            no caller authentication, or binding fails.
    """
    listener = resolve_grpc_listener(
        address,
        tls=tls,
        allow_insecure_remote_dev=allow_insecure_remote_dev,
        profile=runtime.profile if runtime is not None else None,
    )
    server = create_grpc_server(
        runtime=runtime,
        max_workers=max_workers,
        interceptors=interceptors,
        policy=policy,
        listener=listener,
    )
    try:
        bind_grpc_listener(server, listener)
    except GrpcStartupError:
        server.stop(None)
        raise
    server.start()
    return server


def _run_unary(context: grpc.ServicerContext, operation: Callable[[], Any]) -> Any:
    try:
        return operation()
    except GrpcPolicyError as exc:
        abort_rpc(context, exc.reason)
    except ModelNotServedError:
        abort_rpc(context, REASON_MODEL_NOT_SERVED)
    except ValueError:
        abort_rpc(context, REASON_INVALID_REQUEST)
    except Exception:
        abort_rpc(context, REASON_INTERNAL_ERROR)


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


__all__ = [
    "DEFAULT_GRPC_ADDRESS",
    "DEFAULT_GRPC_MAX_WORKERS",
    "OpenMedGrpcServicer",
    "create_grpc_server",
    "serve",
]
