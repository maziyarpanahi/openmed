"""Faker-backed PII anonymization engine.

Provides locale-aware, optionally deterministic surrogate generation for
detected PII entities. Replaces the legacy hardcoded ``LANGUAGE_FAKE_DATA``
lookup with a real Faker pipeline.

Public surface:

  - :class:`Anonymizer` — main runtime entry point; instantiate per
    document or per session.
  - :class:`AnonymizerConfig` — dataclass for advanced configuration.
  - :func:`register_label_generator` — extend or override per-canonical-
    label generators.
  - :func:`register_clinical_provider` — register a custom Faker provider
    on every cached locale (alias for ``Anonymizer``'s constructor option).
  - :data:`LANG_TO_LOCALE` — language -> Faker locale lookup table.

Typical usage::

    from openmed.core.anonymizer import Anonymizer

    anon = Anonymizer(lang="pt", consistent=True, seed=42)
    fake = anon.surrogate("Pedro Almeida", "FIRSTNAME")  # canonical label-aware
"""

from typing import Any

from .engine import Anonymizer, AnonymizerConfig
from .locales import LANG_TO_LOCALE, resolve_locale
from .registry import LABEL_GENERATORS, Generator, register_label_generator


def register_clinical_provider(provider: Any) -> None:
    """Register a custom Faker ``BaseProvider`` for every new Anonymizer.

    Use this to add e.g. proprietary patient-ID formats. The provider is
    registered on each fresh Faker instance built by the engine; existing
    instances are not retroactively updated. Pass via
    ``AnonymizerConfig.custom_providers`` instead when you need
    per-instance scoping.
    """
    from .providers import clinical_ids
    if not hasattr(clinical_ids, "_extra_providers"):
        clinical_ids._extra_providers = []  # type: ignore[attr-defined]
    clinical_ids._extra_providers.append(provider)  # type: ignore[attr-defined]

    original = clinical_ids.register_clinical_providers

    def _augmented(faker):
        original(faker)
        for extra in getattr(clinical_ids, "_extra_providers", ()):
            faker.add_provider(extra)

    clinical_ids.register_clinical_providers = _augmented  # type: ignore[assignment]


__all__ = [
    "Anonymizer",
    "AnonymizerConfig",
    "Generator",
    "LABEL_GENERATORS",
    "LANG_TO_LOCALE",
    "register_clinical_provider",
    "register_label_generator",
    "resolve_locale",
]
