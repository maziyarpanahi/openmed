"""Versioned schemas used by OpenMed trace and training adapters."""

from .preference import (
    CONTENT_FIELDS,
    PREFERENCE_SCHEMA_VERSION,
    PreferencePair,
    PreferencePairAdapter,
    PreferenceRedactionError,
    PreferenceRedactionReport,
    PreferenceRedactionResult,
    PreferenceRedactionState,
    PreferenceSchemaAdapter,
    PreferenceSchemaError,
    PreferenceSpan,
    SensitiveSpan,
    adapt_preference_pair,
    redact_preference_dataset,
    redact_preference_pair,
)

__all__ = [
    "CONTENT_FIELDS",
    "PREFERENCE_SCHEMA_VERSION",
    "PreferencePair",
    "PreferencePairAdapter",
    "PreferenceRedactionError",
    "PreferenceRedactionReport",
    "PreferenceRedactionResult",
    "PreferenceRedactionState",
    "PreferenceSchemaAdapter",
    "PreferenceSchemaError",
    "PreferenceSpan",
    "SensitiveSpan",
    "adapt_preference_pair",
    "redact_preference_dataset",
    "redact_preference_pair",
]
