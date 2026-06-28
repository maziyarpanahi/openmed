"""Synthetic training data generators."""

from .locale_phi import (
    LOCALE_PHI_LABELS,
    SUPPORTED_LOCALE_PHI_LANGUAGES,
    LocalePhiExample,
    LocalePhiGenerator,
    SyntheticPhiSpan,
    generate_locale_phi_examples,
)

__all__ = [
    "LOCALE_PHI_LABELS",
    "SUPPORTED_LOCALE_PHI_LANGUAGES",
    "LocalePhiExample",
    "LocalePhiGenerator",
    "SyntheticPhiSpan",
    "generate_locale_phi_examples",
]
