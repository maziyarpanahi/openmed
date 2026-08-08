"""Trace and training-record schema helpers."""

from .schemas.preference import (
    CONTENT_FIELDS,
    PREFERENCE_SCHEMA_VERSION,
    PreferencePair,
    PreferencePairAdapter,
    PreferenceRedactionError,
    PreferenceRedactionReport,
    PreferenceRedactionResult,
    PreferenceRedactionState,
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
    "PreferenceSchemaError",
    "PreferenceSpan",
    "SensitiveSpan",
    "adapt_preference_pair",
    "redact_preference_dataset",
    "redact_preference_pair",
]
