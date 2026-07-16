"""Typed sync client for the OpenMed REST service."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Mapping, Optional

import httpx

JsonDict = dict[str, Any]
KeepAliveValue = int | float | str
AggregationStrategy = Literal["simple", "first", "average", "max"]
DeidentificationMethod = Literal["mask", "remove", "replace", "hash", "shift_dates"]
PIILanguage = Literal[
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "te",
    "pt",
    "ar",
    "ja",
    "tr",
    "he",
    "id",
    "th",
    "ko",
    "ro",
    "sw",
    "zh",
]

_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_REQUEST_ID_HEADER = "X-Request-ID"


@dataclass(frozen=True)
class AnalyzeRequest:
    """Typed request body for the ``POST /analyze`` endpoint."""

    text: str
    model_name: str = "disease_detection_superclinical"
    confidence_threshold: Optional[float] = 0.0
    group_entities: bool = False
    aggregation_strategy: Optional[AggregationStrategy] = "simple"
    sentence_detection: bool = True
    sentence_language: str = "en"
    sentence_clean: bool = False
    use_fast_tokenizer: bool = True
    keep_alive: Optional[KeepAliveValue] = None


@dataclass(frozen=True)
class PIIExtractRequest:
    """Typed request body for the ``POST /pii/extract`` endpoint."""

    text: str
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.5
    use_smart_merging: bool = True
    lang: PIILanguage = "en"
    normalize_accents: Optional[bool] = None
    keep_alive: Optional[KeepAliveValue] = None


@dataclass(frozen=True)
class PIIExtractStreamRequest(PIIExtractRequest):
    """Typed request body for the ``POST /pii/extract/stream`` endpoint."""

    chunk_size: int = 1024
    window_chars: int = 4096
    tokenizer_context_chars: int = 128
    max_entity_chars: int = 512
    include_text: bool = True


@dataclass(frozen=True)
class PIIDeidentifyRequest:
    """Typed request body for the ``POST /pii/deidentify`` endpoint."""

    text: str
    method: DeidentificationMethod = "mask"
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.7
    keep_year: bool = False
    shift_dates: Optional[bool] = None
    date_shift_days: Optional[int] = None
    keep_mapping: bool = False
    policy: Optional[str] = None
    use_smart_merging: bool = True
    use_safety_sweep: bool = True
    lang: PIILanguage = "en"
    normalize_accents: Optional[bool] = None
    keep_alive: Optional[KeepAliveValue] = None


@dataclass(frozen=True)
class PrivacyGatewayRequest:
    """Typed request body for the ``POST /privacy-gateway/complete`` endpoint."""

    text: str
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.85
    detector_confidence_floor: float = 0.0
    policy: str = "strict"
    disallowed_entity_categories: tuple[str, ...] = ()
    use_smart_merging: bool = True
    lang: PIILanguage = "en"
    normalize_accents: Optional[bool] = None
    keep_alive: Optional[KeepAliveValue] = None


@dataclass(frozen=True)
class ModelUnloadRequest:
    """Typed request body for the ``POST /models/unload`` endpoint."""

    model_name: Optional[str] = None
    all: bool = False


@dataclass(frozen=True)
class ClientEndpoint:
    """Spec-verifiable endpoint metadata for one client method."""

    method: Literal["GET", "POST"]
    path: str
    request_fields: frozenset[str] = frozenset()


def _request_field_names(request_type: type[Any]) -> frozenset[str]:
    return frozenset(field.name for field in fields(request_type))


CLIENT_ENDPOINTS: Mapping[str, ClientEndpoint] = {
    "analyze": ClientEndpoint(
        method="POST",
        path="/analyze",
        request_fields=_request_field_names(AnalyzeRequest),
    ),
    "extract_pii": ClientEndpoint(
        method="POST",
        path="/pii/extract",
        request_fields=_request_field_names(PIIExtractRequest),
    ),
    "extract_pii_stream": ClientEndpoint(
        method="POST",
        path="/pii/extract/stream",
        request_fields=_request_field_names(PIIExtractStreamRequest),
    ),
    "deidentify": ClientEndpoint(
        method="POST",
        path="/pii/deidentify",
        request_fields=_request_field_names(PIIDeidentifyRequest),
    ),
    "privacy_gateway": ClientEndpoint(
        method="POST",
        path="/privacy-gateway/complete",
        request_fields=_request_field_names(PrivacyGatewayRequest),
    ),
    "loaded_models": ClientEndpoint(method="GET", path="/models/loaded"),
    "unload_model": ClientEndpoint(
        method="POST",
        path="/models/unload",
        request_fields=_request_field_names(ModelUnloadRequest),
    ),
    "unload_all_models": ClientEndpoint(
        method="POST",
        path="/models/unload",
        request_fields=_request_field_names(ModelUnloadRequest),
    ),
}


class OpenMedAPIError(RuntimeError):
    """Raised when the REST service returns a non-2xx error envelope."""

    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: Any = None,
        request_id: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details
        self.request_id = request_id
        suffix = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"{status_code} {code}: {message}{suffix}")


class OpenMedClient:
    """Small typed sync client for the OpenMed REST service."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        *,
        timeout: Optional[float] = 30.0,
        headers: Optional[Mapping[str, str]] = None,
        request_id: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        if client is not None and transport is not None:
            raise ValueError("Pass either client or transport, not both.")

        self._request_id = request_id
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
            transport=transport,
        )

    def __enter__(self) -> "OpenMedClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying ``httpx.Client`` if this object owns it."""
        if self._owns_client:
            self._client.close()

    def analyze(
        self,
        text: str,
        *,
        model_name: str = "disease_detection_superclinical",
        confidence_threshold: Optional[float] = 0.0,
        group_entities: bool = False,
        aggregation_strategy: Optional[AggregationStrategy] = "simple",
        sentence_detection: bool = True,
        sentence_language: str = "en",
        sentence_clean: bool = False,
        use_fast_tokenizer: bool = True,
        keep_alive: Optional[KeepAliveValue] = None,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        """Analyze text with ``POST /analyze``."""
        return self._post(
            "/analyze",
            AnalyzeRequest(
                text=text,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                group_entities=group_entities,
                aggregation_strategy=aggregation_strategy,
                sentence_detection=sentence_detection,
                sentence_language=sentence_language,
                sentence_clean=sentence_clean,
                use_fast_tokenizer=use_fast_tokenizer,
                keep_alive=keep_alive,
            ),
            request_id=request_id,
        )

    def extract_pii(
        self,
        text: str,
        *,
        model_name: str = _DEFAULT_PII_MODEL,
        confidence_threshold: float = 0.5,
        use_smart_merging: bool = True,
        lang: PIILanguage = "en",
        normalize_accents: Optional[bool] = None,
        keep_alive: Optional[KeepAliveValue] = None,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        """Extract PII entities with ``POST /pii/extract``."""
        return self._post(
            "/pii/extract",
            PIIExtractRequest(
                text=text,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                use_smart_merging=use_smart_merging,
                lang=lang,
                normalize_accents=normalize_accents,
                keep_alive=keep_alive,
            ),
            request_id=request_id,
        )

    def extract_pii_stream(
        self,
        text: str,
        *,
        model_name: str = _DEFAULT_PII_MODEL,
        confidence_threshold: float = 0.5,
        use_smart_merging: bool = True,
        lang: PIILanguage = "en",
        normalize_accents: Optional[bool] = None,
        keep_alive: Optional[KeepAliveValue] = None,
        chunk_size: int = 1024,
        window_chars: int = 4096,
        tokenizer_context_chars: int = 128,
        max_entity_chars: int = 512,
        include_text: bool = True,
        request_id: Optional[str] = None,
    ):
        """Stream PII extraction events with ``POST /pii/extract/stream``."""
        yield from self._post_lines(
            "/pii/extract/stream",
            PIIExtractStreamRequest(
                text=text,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                use_smart_merging=use_smart_merging,
                lang=lang,
                normalize_accents=normalize_accents,
                keep_alive=keep_alive,
                chunk_size=chunk_size,
                window_chars=window_chars,
                tokenizer_context_chars=tokenizer_context_chars,
                max_entity_chars=max_entity_chars,
                include_text=include_text,
            ),
            request_id=request_id,
        )

    def deidentify(
        self,
        text: str,
        *,
        method: DeidentificationMethod = "mask",
        model_name: str = _DEFAULT_PII_MODEL,
        confidence_threshold: float = 0.7,
        keep_year: bool = False,
        shift_dates: Optional[bool] = None,
        date_shift_days: Optional[int] = None,
        keep_mapping: bool = False,
        policy: Optional[str] = None,
        use_smart_merging: bool = True,
        use_safety_sweep: bool = True,
        lang: PIILanguage = "en",
        normalize_accents: Optional[bool] = None,
        keep_alive: Optional[KeepAliveValue] = None,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        """De-identify text with ``POST /pii/deidentify``."""
        return self._post(
            "/pii/deidentify",
            PIIDeidentifyRequest(
                text=text,
                method=method,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                keep_year=keep_year,
                shift_dates=shift_dates,
                date_shift_days=date_shift_days,
                keep_mapping=keep_mapping,
                policy=policy,
                use_smart_merging=use_smart_merging,
                use_safety_sweep=use_safety_sweep,
                lang=lang,
                normalize_accents=normalize_accents,
                keep_alive=keep_alive,
            ),
            request_id=request_id,
        )

    def privacy_gateway(
        self,
        text: str,
        *,
        model_name: str = _DEFAULT_PII_MODEL,
        confidence_threshold: float = 0.85,
        detector_confidence_floor: float = 0.0,
        policy: str = "strict",
        disallowed_entity_categories: tuple[str, ...] = (),
        use_smart_merging: bool = True,
        lang: PIILanguage = "en",
        normalize_accents: Optional[bool] = None,
        keep_alive: Optional[KeepAliveValue] = None,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        """Complete a redacted external LLM round-trip."""
        return self._post(
            "/privacy-gateway/complete",
            PrivacyGatewayRequest(
                text=text,
                model_name=model_name,
                confidence_threshold=confidence_threshold,
                detector_confidence_floor=detector_confidence_floor,
                policy=policy,
                disallowed_entity_categories=disallowed_entity_categories,
                use_smart_merging=use_smart_merging,
                lang=lang,
                normalize_accents=normalize_accents,
                keep_alive=keep_alive,
            ),
            request_id=request_id,
        )

    def loaded_models(self, *, request_id: Optional[str] = None) -> JsonDict:
        """Return loaded model and warm-pool state."""
        return self._request("GET", "/models/loaded", request_id=request_id)

    def unload_model(
        self,
        model_name: str,
        *,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        """Unload one inactive model from the service."""
        return self._post(
            "/models/unload",
            ModelUnloadRequest(model_name=model_name),
            request_id=request_id,
        )

    def unload_all_models(self, *, request_id: Optional[str] = None) -> JsonDict:
        """Unload every inactive model from the service."""
        return self._post(
            "/models/unload",
            ModelUnloadRequest(all=True),
            request_id=request_id,
        )

    def _post(
        self,
        path: str,
        payload: object,
        *,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        return self._request(
            "POST",
            path,
            json=asdict(payload),
            request_id=request_id,
        )

    def _post_lines(
        self,
        path: str,
        payload: object,
        *,
        request_id: Optional[str] = None,
    ):
        active_request_id = request_id or self._request_id
        headers = {"Accept": "application/x-ndjson"}
        if active_request_id:
            headers[_REQUEST_ID_HEADER] = active_request_id
        with self._client.stream(
            "POST",
            path,
            json=asdict(payload),
            headers=headers,
        ) as response:
            if response.is_error:
                response.read()
                self._raise_api_error(response, request_id=active_request_id)
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[JsonDict] = None,
        request_id: Optional[str] = None,
    ) -> JsonDict:
        active_request_id = request_id or self._request_id
        headers = {_REQUEST_ID_HEADER: active_request_id} if active_request_id else None
        response = self._client.request(method, path, json=json, headers=headers)
        if response.is_error:
            self._raise_api_error(response, request_id=active_request_id)

        payload = response.json()
        if not isinstance(payload, dict):
            raise OpenMedAPIError(
                status_code=response.status_code,
                code="invalid_response",
                message="Expected JSON object response from OpenMed REST service",
                details=payload,
                request_id=response.headers.get(_REQUEST_ID_HEADER)
                or active_request_id,
            )
        return payload

    def _raise_api_error(
        self,
        response: httpx.Response,
        *,
        request_id: Optional[str],
    ) -> None:
        code = "http_error"
        message = response.reason_phrase or "HTTP error"
        details: Any = None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error = payload["error"]
            code = str(error.get("code") or code)
            message = str(error.get("message") or message)
            details = error.get("details")

        raise OpenMedAPIError(
            status_code=response.status_code,
            code=code,
            message=message,
            details=details,
            request_id=response.headers.get(_REQUEST_ID_HEADER) or request_id,
        )


__all__ = [
    "AnalyzeRequest",
    "CLIENT_ENDPOINTS",
    "ClientEndpoint",
    "ModelUnloadRequest",
    "OpenMedAPIError",
    "OpenMedClient",
    "PIIDeidentifyRequest",
    "PIIExtractRequest",
    "PIIExtractStreamRequest",
]
