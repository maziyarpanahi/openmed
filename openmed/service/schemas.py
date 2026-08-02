"""Pydantic schemas for the OpenMed REST service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Optional, Union

from openmed.core.policy import canonical_policy_name
from openmed.interop.tools import (
    AnalyzeTextArgs,
    DeidentifyArgs,
    ExtractPIIArgs,
    PIILanguage,
    UnloadModelArgs,
)
from openmed.utils.gateway import normalize_text, validate_language
from openmed.utils.validation import (
    validate_confidence_threshold,
    validate_model_name,
)

from .keep_alive import parse_keep_alive
from .limits import get_max_text_length

try:
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    PYDANTIC_V2 = True
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field, root_validator, validator

    ConfigDict = None  # type: ignore[assignment]
    field_validator = None  # type: ignore[assignment]
    model_validator = None  # type: ignore[assignment]
    PYDANTIC_V2 = False


_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_DEFAULT_STREAM_CHUNK_SIZE = 1024
_DEFAULT_STREAM_WINDOW_CHARS = 4096
_DEFAULT_STREAM_TOKENIZER_CONTEXT_CHARS = 128
_DEFAULT_STREAM_MAX_ENTITY_CHARS = 512
KeepAliveValue = Union[int, float, str]


def _normalize_text(value: Any) -> str:
    # Route through the shared gateway so REST requests validate text identically
    # to the library and MCP surfaces: same character cap (still driven by
    # ``OPENMED_SERVICE_MAX_TEXT_LENGTH``), plus byte-size and UTF-8 encoding
    # guardrails. Errors never echo the (possibly PHI) text.
    return normalize_text(value)


def _normalize_pii_language(value: Any) -> str:
    """Normalize API PII languages through the shared gateway."""
    return validate_language(value, include_national_id=False)


def _normalize_model_name(value: str) -> str:
    return validate_model_name(value)


def _normalize_confidence_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return validate_confidence_threshold(value)


def _normalize_policy_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    return canonical_policy_name(str(value))


def _normalize_gateway_policy(value: Any) -> str:
    if value is None:
        return "strict"
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Policy must not be blank")
    return normalized


def _normalize_entity_categories(value: Any) -> list[str]:
    from .privacy_gateway import normalize_entity_category

    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [item.strip() for item in value.split(",")]
    elif isinstance(value, Sequence):
        raw_values = [str(item).strip() for item in value]
    else:
        raise ValueError("Entity categories must be a list of strings")

    categories = {
        normalize_entity_category(item) for item in raw_values if item.strip()
    }
    return sorted(categories)


def _normalize_webhook_url(value: Any) -> str:
    if value is None:
        raise ValueError("Webhook URL is required")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Webhook URL must not be blank")
    if not normalized.startswith(("http://", "https://")):
        raise ValueError("Webhook URL must start with http:// or https://")
    return normalized


def _normalize_webhook_secret(value: Any) -> str:
    if value is None:
        raise ValueError("Webhook secret is required")
    normalized = str(value)
    if not normalized:
        raise ValueError("Webhook secret must not be blank")
    return normalized


def _normalize_document_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _validate_job_documents(value: list[Any]) -> list[Any]:
    if not value:
        raise ValueError("At least one document is required")
    return value


def _validate_keep_alive_value(value: Any) -> Any:
    parse_keep_alive(value)
    return value


def _normalize_nonblank_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} is required")
    if not isinstance(value, str):
        value = str(value)
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    return value


def _normalize_optional_nonblank_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _normalize_nonblank_string(value, field_name)


def _normalize_records_jsonl(value: Any) -> str:
    if value is None:
        raise ValueError("records_jsonl is required")
    if not isinstance(value, str):
        raise ValueError("records_jsonl must be a string")
    if not value.strip():
        raise ValueError("records_jsonl must not be blank")
    max_text_length = get_max_text_length()
    if len(value) > max_text_length:
        raise ValueError(
            f"records_jsonl exceeds the maximum length of {max_text_length} characters"
        )
    return value


def _normalize_shift_dates_payload(values: dict[str, Any]) -> dict[str, Any]:
    method = values.get("method", "mask")
    shift_dates = values.get("shift_dates")
    date_shift_days = values.get("date_shift_days")

    if shift_dates is True and method != "shift_dates":
        values["method"] = "shift_dates"
        method = "shift_dates"
    elif shift_dates is False and method == "shift_dates":
        raise ValueError("shift_dates=false conflicts with method='shift_dates'")

    if date_shift_days is not None and method != "shift_dates":
        raise ValueError("date_shift_days requires method='shift_dates'")

    return values


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover

        class Config:
            extra = "forbid"

    if PYDANTIC_V2:

        @field_validator("lang", mode="before", check_fields=False)
        @classmethod
        def _validate_pii_language(cls, value: Any) -> str:
            return _normalize_pii_language(value)

    else:  # pragma: no cover

        @validator("lang", pre=True, check_fields=False)
        def _validate_pii_language(cls, value: Any) -> str:
            return _normalize_pii_language(value)


if PYDANTIC_V2:

    class AnalyzeRequest(AnalyzeTextArgs):
        """Request schema for /analyze."""

        keep_alive: Optional[KeepAliveValue] = None

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PIIExtractRequest(ExtractPIIArgs):
        """Request schema for /pii/extract."""

        keep_alive: Optional[KeepAliveValue] = None

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PIIExtractStreamRequest(PIIExtractRequest):
        """Request schema for /pii/extract/stream."""

        chunk_size: int = Field(default=_DEFAULT_STREAM_CHUNK_SIZE, ge=1, le=32768)
        window_chars: int = Field(default=_DEFAULT_STREAM_WINDOW_CHARS, ge=64)
        tokenizer_context_chars: int = Field(
            default=_DEFAULT_STREAM_TOKENIZER_CONTEXT_CHARS,
            ge=0,
        )
        max_entity_chars: int = Field(
            default=_DEFAULT_STREAM_MAX_ENTITY_CHARS,
            ge=1,
        )
        include_text: bool = True

    class PIIDeidentifyRequest(DeidentifyArgs):
        """Request schema for /pii/deidentify."""

        keep_alive: Optional[KeepAliveValue] = None

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PrivacyGatewayRequest(_StrictModel):
        """Request schema for /privacy-gateway/complete."""

        text: str
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
        detector_confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0)
        policy: str = "strict"
        disallowed_entity_categories: list[str] = Field(default_factory=list)
        use_smart_merging: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None
        keep_alive: Optional[KeepAliveValue] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold", "detector_confidence_floor")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must not be None")
            return normalized

        @field_validator("policy", mode="before")
        @classmethod
        def _validate_gateway_policy(cls, value: Any) -> str:
            return _normalize_gateway_policy(value)

        @field_validator("disallowed_entity_categories", mode="before")
        @classmethod
        def _validate_disallowed_entity_categories(cls, value: Any) -> list[str]:
            return _normalize_entity_categories(value)

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class ModelUnloadRequest(UnloadModelArgs):
        """Request schema for /models/unload."""

    class SMARTBackendIngestionRequest(_StrictModel):
        """Request schema for starting SMART backend-services ingestion."""

        fhir_base_url: str
        token_url: str
        client_id: str
        private_key_pem: str
        output_dir: str
        checkpoint_path: Optional[str] = None
        key_id: Optional[str] = None
        scope: str = "system/*.read"
        export_path: str = "$export"
        max_inflight_downloads: int = Field(default=2, ge=1)
        poll_interval_seconds: float = Field(default=1.0, ge=0.0)
        request_timeout_seconds: float = Field(default=30.0, gt=0.0)
        policy: Optional[str] = "hipaa_safe_harbor"
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "replace"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

        use_smart_merging: bool = True
        use_safety_sweep: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None
        keep_alive: Optional[KeepAliveValue] = None

        @field_validator(
            "fhir_base_url",
            "token_url",
            "client_id",
            "private_key_pem",
            "output_dir",
            "scope",
            "export_path",
            mode="before",
        )
        @classmethod
        def _validate_required_text(cls, value: Any, info: Any) -> str:
            return _normalize_nonblank_string(value, info.field_name)

        @field_validator("checkpoint_path", "key_id", mode="before")
        @classmethod
        def _validate_optional_text(cls, value: Any, info: Any) -> Optional[str]:
            return _normalize_optional_nonblank_string(value, info.field_name)

        @field_validator("policy", mode="before")
        @classmethod
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy_name(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class JobWebhookRequest(_StrictModel):
        """Webhook callback configuration for async jobs."""

        url: str
        secret: str
        max_attempts: int = Field(default=3, ge=1, le=10)
        backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)

        @field_validator("url", mode="before")
        @classmethod
        def _validate_url(cls, value: Any) -> str:
            return _normalize_webhook_url(value)

        @field_validator("secret", mode="before")
        @classmethod
        def _validate_secret(cls, value: Any) -> str:
            return _normalize_webhook_secret(value)

    class DeidentifyJobDocument(_StrictModel):
        """One document in an async de-identification job."""

        text: str
        id: Optional[str] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @field_validator("id", mode="before")
        @classmethod
        def _validate_id(cls, value: Any) -> Optional[str]:
            return _normalize_document_id(value)

    class DeidentifyJobRequest(_StrictModel):
        """Request schema for POST /jobs."""

        documents: list[DeidentifyJobDocument]
        webhook: Optional[JobWebhookRequest] = None
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
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

        @field_validator("documents", mode="before")
        @classmethod
        def _validate_documents(cls, value: Any) -> list[DeidentifyJobDocument]:
            if not isinstance(value, list):
                raise ValueError("documents must be a list")
            _validate_job_documents(value)
            return [
                item
                if isinstance(item, DeidentifyJobDocument)
                else DeidentifyJobDocument(**item)
                for item in value
            ]

        @field_validator("webhook", mode="before")
        @classmethod
        def _validate_webhook(cls, value: Any) -> Optional[JobWebhookRequest]:
            if value is None or isinstance(value, JobWebhookRequest):
                return value
            if isinstance(value, dict):
                return JobWebhookRequest(**value)
            raise ValueError("webhook must be an object")

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @field_validator("policy", mode="before")
        @classmethod
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy_name(value)

        @field_validator("keep_alive", mode="before")
        @classmethod
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

        @model_validator(mode="after")
        def _validate_shift_dates(self) -> "DeidentifyJobRequest":
            values = _normalize_shift_dates_payload(
                {
                    "method": self.method,
                    "shift_dates": self.shift_dates,
                    "date_shift_days": self.date_shift_days,
                }
            )
            for field_name, value in values.items():
                setattr(self, field_name, value)
            return self

    class OmopLoadRequest(_StrictModel):
        """Request schema for /omop/load."""

        records_jsonl: str
        vocabulary_version: Optional[str] = None
        validate_constraints: bool = False

        @field_validator("records_jsonl", mode="before")
        @classmethod
        def _validate_records_jsonl(cls, value: Any) -> str:
            return _normalize_records_jsonl(value)

        @field_validator("vocabulary_version", mode="before")
        @classmethod
        def _validate_vocabulary_version(cls, value: Any) -> Optional[str]:
            return _normalize_optional_nonblank_string(value, "vocabulary_version")

else:

    class AnalyzeRequest(AnalyzeTextArgs):
        """Request schema for /analyze."""

        keep_alive: Optional[KeepAliveValue] = None

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PIIExtractRequest(ExtractPIIArgs):
        """Request schema for /pii/extract."""

        keep_alive: Optional[KeepAliveValue] = None

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PIIExtractStreamRequest(PIIExtractRequest):
        """Request schema for /pii/extract/stream."""

        chunk_size: int = Field(default=_DEFAULT_STREAM_CHUNK_SIZE, ge=1, le=32768)
        window_chars: int = Field(default=_DEFAULT_STREAM_WINDOW_CHARS, ge=64)
        tokenizer_context_chars: int = Field(
            default=_DEFAULT_STREAM_TOKENIZER_CONTEXT_CHARS,
            ge=0,
        )
        max_entity_chars: int = Field(
            default=_DEFAULT_STREAM_MAX_ENTITY_CHARS,
            ge=1,
        )
        include_text: bool = True

    class PIIDeidentifyRequest(DeidentifyArgs):
        """Request schema for /pii/deidentify."""

        keep_alive: Optional[KeepAliveValue] = None

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class PrivacyGatewayRequest(_StrictModel):
        """Request schema for /privacy-gateway/complete."""

        text: str
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
        detector_confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0)
        policy: str = "strict"
        disallowed_entity_categories: list[str] = Field(default_factory=list)
        use_smart_merging: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None
        keep_alive: Optional[KeepAliveValue] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold", "detector_confidence_floor")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must not be None")
            return normalized

        @validator("policy", pre=True)
        def _validate_gateway_policy(cls, value: Any) -> str:
            return _normalize_gateway_policy(value)

        @validator("disallowed_entity_categories", pre=True)
        def _validate_disallowed_entity_categories(cls, value: Any) -> list[str]:
            return _normalize_entity_categories(value)

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class ModelUnloadRequest(UnloadModelArgs):
        """Request schema for /models/unload."""

    class SMARTBackendIngestionRequest(_StrictModel):
        """Request schema for starting SMART backend-services ingestion."""

        fhir_base_url: str
        token_url: str
        client_id: str
        private_key_pem: str
        output_dir: str
        checkpoint_path: Optional[str] = None
        key_id: Optional[str] = None
        scope: str = "system/*.read"
        export_path: str = "$export"
        max_inflight_downloads: int = Field(default=2, ge=1)
        poll_interval_seconds: float = Field(default=1.0, ge=0.0)
        request_timeout_seconds: float = Field(default=30.0, gt=0.0)
        policy: Optional[str] = "hipaa_safe_harbor"
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "replace"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

        use_smart_merging: bool = True
        use_safety_sweep: bool = True
        lang: PIILanguage = "en"
        normalize_accents: Optional[bool] = None
        keep_alive: Optional[KeepAliveValue] = None

        @validator(
            "fhir_base_url",
            "token_url",
            "client_id",
            "private_key_pem",
            "output_dir",
            "scope",
            "export_path",
            pre=True,
        )
        def _validate_required_text(cls, value: Any, field: Any) -> str:
            return _normalize_nonblank_string(value, field.name)

        @validator("checkpoint_path", "key_id", pre=True)
        def _validate_optional_text(cls, value: Any, field: Any) -> Optional[str]:
            return _normalize_optional_nonblank_string(value, field.name)

        @validator("policy", pre=True)
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy_name(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

    class JobWebhookRequest(_StrictModel):
        """Webhook callback configuration for async jobs."""

        url: str
        secret: str
        max_attempts: int = Field(default=3, ge=1, le=10)
        backoff_seconds: float = Field(default=0.5, ge=0.0, le=60.0)

        @validator("url", pre=True)
        def _validate_url(cls, value: Any) -> str:
            return _normalize_webhook_url(value)

        @validator("secret", pre=True)
        def _validate_secret(cls, value: Any) -> str:
            return _normalize_webhook_secret(value)

    class DeidentifyJobDocument(_StrictModel):
        """One document in an async de-identification job."""

        text: str
        id: Optional[str] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @validator("id", pre=True)
        def _validate_id(cls, value: Any) -> Optional[str]:
            return _normalize_document_id(value)

    class DeidentifyJobRequest(_StrictModel):
        """Request schema for POST /jobs."""

        documents: list[DeidentifyJobDocument]
        webhook: Optional[JobWebhookRequest] = None
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
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

        @validator("documents", pre=True)
        def _validate_documents(cls, value: Any) -> list[DeidentifyJobDocument]:
            if not isinstance(value, list):
                raise ValueError("documents must be a list")
            _validate_job_documents(value)
            return [
                item
                if isinstance(item, DeidentifyJobDocument)
                else DeidentifyJobDocument(**item)
                for item in value
            ]

        @validator("webhook", pre=True)
        def _validate_webhook(cls, value: Any) -> Optional[JobWebhookRequest]:
            if value is None or isinstance(value, JobWebhookRequest):
                return value
            if isinstance(value, dict):
                return JobWebhookRequest(**value)
            raise ValueError("webhook must be an object")

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            if normalized is None:
                raise ValueError("confidence_threshold must be a valid number")
            return normalized

        @validator("policy", pre=True)
        def _validate_policy(cls, value: Any) -> Optional[str]:
            return _normalize_policy_name(value)

        @validator("keep_alive", pre=True)
        def _validate_keep_alive(cls, value: Any) -> Any:
            return _validate_keep_alive_value(value)

        @root_validator
        def _validate_shift_dates(cls, values: dict[str, Any]) -> dict[str, Any]:
            return _normalize_shift_dates_payload(values)

    class OmopLoadRequest(_StrictModel):
        """Request schema for /omop/load."""

        records_jsonl: str
        vocabulary_version: Optional[str] = None
        validate_constraints: bool = False

        @validator("records_jsonl", pre=True)
        def _validate_records_jsonl(cls, value: Any) -> str:
            return _normalize_records_jsonl(value)

        @validator("vocabulary_version", pre=True)
        def _validate_vocabulary_version(cls, value: Any) -> Optional[str]:
            return _normalize_optional_nonblank_string(value, "vocabulary_version")
