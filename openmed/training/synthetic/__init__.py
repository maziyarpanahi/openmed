"""Synthetic training data generators."""

from .locale_phi import (
    LOCALE_PHI_LABELS,
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    LocalePhiExample,
    LocalePhiGenerator,
    SyntheticPhiSpan,
    generate_locale_phi_examples,
)
from .offset_projection import (
    SpanAnnotation,
    SpanProjectionError,
    normalize_span_annotations,
    realign_translated_spans,
    span_corruption_count,
    validate_span_integrity,
)
from .translation_augment import (
    DEFAULT_TARGET_LANGUAGES,
    SYNTHETIC_SOURCE,
    DictionaryTranslator,
    ModelBackedTranslator,
    TranslationAugmentedExample,
    augment_span_annotated_examples,
    load_span_jsonl,
    write_augmented_jsonl,
)

__all__ = [
    "DEFAULT_TARGET_LANGUAGES",
    "LOCALE_PHI_LABELS",
    "SYNTHETIC_SOURCE",
    "SUPPORTED_LOCALE_PHI_LANGUAGES",
    "DictionaryTranslator",
    "LocalePhiExample",
    "LocalePhiGenerator",
    "ModelBackedTranslator",
    "SpanAnnotation",
    "SpanProjectionError",
    "SyntheticPhiSpan",
    "TranslationAugmentedExample",
    "augment_span_annotated_examples",
    "generate_locale_phi_examples",
    "load_span_jsonl",
    "normalize_span_annotations",
    "realign_translated_spans",
    "span_corruption_count",
    "validate_span_integrity",
    "write_augmented_jsonl",
]
