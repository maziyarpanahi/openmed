"""Model family abstractions for zero-shot NER."""

from __future__ import annotations

from .base import EncoderOutput, ModelFamily, SupportsEncoding
from .gliner import ensure_gliner_available, is_gliner_available
from .gliner2 import (
    clear_gliner2_cache,
    ensure_gliner2_available,
    is_gliner2_available,
    load_gliner2_handle,
)
from .indic import (
    INDIC_ENCODER_SPECS,
    IndicEncoderHandle,
    IndicEncoderLoadResult,
    IndicEncoderSpec,
    IndicNerAdapter,
    IndicNerPrediction,
    IndicNerWeightsUnavailable,
    configured_indic_ner_model,
    get_indic_encoder_spec,
    is_indic_encoder_available,
    is_indic_ner_configured,
    load_indic_encoder,
    load_indic_ner_adapter,
)

__all__ = [
    "ModelFamily",
    "EncoderOutput",
    "SupportsEncoding",
    "ensure_gliner_available",
    "is_gliner_available",
    "ensure_gliner2_available",
    "is_gliner2_available",
    "load_gliner2_handle",
    "clear_gliner2_cache",
    "INDIC_ENCODER_SPECS",
    "IndicEncoderHandle",
    "IndicEncoderLoadResult",
    "IndicEncoderSpec",
    "IndicNerAdapter",
    "IndicNerPrediction",
    "IndicNerWeightsUnavailable",
    "configured_indic_ner_model",
    "get_indic_encoder_spec",
    "is_indic_ner_configured",
    "is_indic_encoder_available",
    "load_indic_encoder",
    "load_indic_ner_adapter",
]
