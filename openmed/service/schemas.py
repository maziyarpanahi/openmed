"""Pydantic schemas for the OpenMed REST service."""

from typing import Literal, Optional

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    from pydantic import BaseModel, Field
    ConfigDict = None  # type: ignore[assignment]


_DEFAULT_PII_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover
        class Config:
            extra = "forbid"


class AnalyzeRequest(_StrictModel):
    """Request schema for /analyze."""

    text: str = Field(..., min_length=1)
    model_name: str = "disease_detection_superclinical"
    confidence_threshold: Optional[float] = 0.0
    group_entities: bool = False
    aggregation_strategy: Optional[str] = "simple"
    sentence_detection: bool = True
    sentence_language: str = "en"
    sentence_clean: bool = False
    use_fast_tokenizer: bool = True


class PIIExtractRequest(_StrictModel):
    """Request schema for /pii/extract."""

    text: str = Field(..., min_length=1)
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.5
    use_smart_merging: bool = True
    lang: Literal["en", "fr", "de", "it", "es"] = "en"
    normalize_accents: Optional[bool] = None


class PIIDeidentifyRequest(_StrictModel):
    """Request schema for /pii/deidentify."""

    text: str = Field(..., min_length=1)
    method: Literal["mask", "remove", "replace", "hash", "shift_dates"] = "mask"
    model_name: str = _DEFAULT_PII_MODEL
    confidence_threshold: float = 0.7
    keep_year: bool = True
    shift_dates: bool = False
    date_shift_days: Optional[int] = None
    keep_mapping: bool = False
    use_smart_merging: bool = True
    lang: Literal["en", "fr", "de", "it", "es"] = "en"
    normalize_accents: Optional[bool] = None
