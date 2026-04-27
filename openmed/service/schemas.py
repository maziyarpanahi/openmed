"""Pydantic schemas for the OpenMed REST service."""

from __future__ import annotations

from typing import Any, Literal, Optional

from openmed.utils.validation import (
    validate_confidence_threshold,
    validate_model_name,
)

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


def _normalize_text(value: Any) -> str:
    if value is None:
        raise ValueError("Text is required")
    if not isinstance(value, str):
        value = str(value)

    normalized = value.strip()
    if not normalized:
        raise ValueError("Text must not be blank")
    return normalized


def _normalize_model_name(value: str) -> str:
    return validate_model_name(value)


def _normalize_confidence_threshold(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return validate_confidence_threshold(value)


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

    class AnalyzeRequest(_StrictModel):
        """Request schema for /analyze."""

        text: str
        model_name: str = "disease_detection_superclinical"
        confidence_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
        group_entities: bool = False
        aggregation_strategy: Optional[Literal["simple", "first", "average", "max"]] = "simple"
        sentence_detection: bool = True
        sentence_language: str = "en"
        sentence_clean: bool = False
        use_fast_tokenizer: bool = True

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: Optional[float]) -> Optional[float]:
            return _normalize_confidence_threshold(value)


    class PIIExtractRequest(_StrictModel):
        """Request schema for /pii/extract."""

        text: str
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        use_smart_merging: bool = True
        lang: Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"] = "en"
        normalize_accents: Optional[bool] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            assert normalized is not None
            return normalized


    class PIIDeidentifyRequest(_StrictModel):
        """Request schema for /pii/deidentify."""

        text: str
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        keep_year: bool = True
        shift_dates: Optional[bool] = None
        date_shift_days: Optional[int] = None
        keep_mapping: bool = False
        use_smart_merging: bool = True
        lang: Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"] = "en"
        normalize_accents: Optional[bool] = None

        @field_validator("text", mode="before")
        @classmethod
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @field_validator("model_name")
        @classmethod
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @field_validator("confidence_threshold")
        @classmethod
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            assert normalized is not None
            return normalized

        @model_validator(mode="after")
        def _validate_shift_dates(self) -> "PIIDeidentifyRequest":
            values = _normalize_shift_dates_payload(self.model_dump())
            for field_name, value in values.items():
                setattr(self, field_name, value)
            return self

else:

    class AnalyzeRequest(_StrictModel):
        """Request schema for /analyze."""

        text: str
        model_name: str = "disease_detection_superclinical"
        confidence_threshold: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
        group_entities: bool = False
        aggregation_strategy: Optional[Literal["simple", "first", "average", "max"]] = "simple"
        sentence_detection: bool = True
        sentence_language: str = "en"
        sentence_clean: bool = False
        use_fast_tokenizer: bool = True

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: Optional[float]) -> Optional[float]:
            return _normalize_confidence_threshold(value)


    class PIIExtractRequest(_StrictModel):
        """Request schema for /pii/extract."""

        text: str
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
        use_smart_merging: bool = True
        lang: Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"] = "en"
        normalize_accents: Optional[bool] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            assert normalized is not None
            return normalized


    class PIIDeidentifyRequest(_StrictModel):
        """Request schema for /pii/deidentify."""

        text: str
        method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
        model_name: str = _DEFAULT_PII_MODEL
        confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        keep_year: bool = True
        shift_dates: Optional[bool] = None
        date_shift_days: Optional[int] = None
        keep_mapping: bool = False
        use_smart_merging: bool = True
        lang: Literal["en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"] = "en"
        normalize_accents: Optional[bool] = None

        @validator("text", pre=True)
        def _validate_text(cls, value: Any) -> str:
            return _normalize_text(value)

        @validator("model_name")
        def _validate_model_name(cls, value: str) -> str:
            return _normalize_model_name(value)

        @validator("confidence_threshold")
        def _validate_confidence_threshold(cls, value: float) -> float:
            normalized = _normalize_confidence_threshold(value)
            assert normalized is not None
            return normalized

        @root_validator
        def _validate_shift_dates(cls, values: dict[str, Any]) -> dict[str, Any]:
            return _normalize_shift_dates_payload(values)
