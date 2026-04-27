"""Faker-backed anonymization engine.

The :class:`Anonymizer` is the single entry point for generating fake
surrogate values for detected PII entities. It supports:

  - **Locale-aware generation**: surrogates are drawn from a Faker locale
    matched to the input language (``de`` -> ``de_DE``, ``pt`` ->
    ``pt_PT``, ...). Override per-call via ``locale=`` or per-instance
    via the constructor.
  - **Deterministic mode**: when ``consistent=True``, the same
    ``(canonical_label, original_value)`` pair always yields the same
    surrogate within a session — solves "John Doe appearing twice gets
    two different fakes" without sacrificing realism. Cross-session
    determinism is opt-in via ``seed=``.
  - **Format preservation**: phone digit groups, date separators, email
    domains, and ID shapes are kept stable so downstream regexes and
    template renderers don't break.
  - **Custom generators**: extend or override per canonical label via
    :func:`openmed.core.anonymizer.register_label_generator`. Add custom
    Faker providers (e.g. proprietary patient ID formats) via
    :func:`openmed.core.anonymizer.register_clinical_provider`.

This module is the runtime engine. The label-to-generator mapping lives
in :mod:`registry`; the locale resolution in :mod:`locales`; format
helpers in :mod:`format_preserve`; clinical-ID providers in
:mod:`providers.clinical_ids`.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..labels import normalize_label
from .locales import resolve_locale
from .providers import register_clinical_providers
from .registry import LABEL_GENERATORS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AnonymizerConfig:
    """Per-call/per-instance configuration for :class:`Anonymizer`.

    Attributes:
        lang: ISO 639-1 language code controlling default Faker locale.
        locale: Explicit Faker locale (``pt_BR``, ``en_GB``); overrides
            the ``lang`` -> locale lookup.
        consistent: When True, identical ``(canonical_label, original_value)``
            pairs always produce the same surrogate. Use for within-document
            consistency (so "John Doe" appearing twice gets one surrogate).
        seed: Optional integer seed. When set together with
            ``consistent=True``, surrogates are stable across sessions.
        custom_providers: Additional Faker providers to register on every
            new locale instance.
    """

    lang: str = "en"
    locale: Optional[str] = None
    consistent: bool = False
    seed: Optional[int] = None
    custom_providers: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Anonymizer
# ---------------------------------------------------------------------------

class Anonymizer:
    """Generate locale-aware, optionally deterministic surrogate PII values."""

    def __init__(
        self,
        lang: str = "en",
        *,
        locale: Optional[str] = None,
        consistent: bool = False,
        seed: Optional[int] = None,
        config: Optional[AnonymizerConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = AnonymizerConfig(
                lang=lang,
                locale=locale,
                consistent=consistent,
                seed=seed,
            )
        self._faker_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Faker management
    # ------------------------------------------------------------------

    def _build_faker(self, locale: str):
        try:
            from faker import Faker
        except ImportError as exc:  # pragma: no cover - faker is a hard dep
            raise ImportError(
                "Faker is required for openmed.core.anonymizer. "
                "Install with `pip install faker` (or upgrade openmed)."
            ) from exc

        faker = Faker(locale)
        register_clinical_providers(faker)
        for provider in self.config.custom_providers:
            faker.add_provider(provider)
        return faker

    def _get_faker(self, locale: str):
        cached = self._faker_cache.get(locale)
        if cached is None:
            cached = self._build_faker(locale)
            self._faker_cache[locale] = cached
        return cached

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def _derive_seed(self, canonical_label: str, original_value: str) -> int:
        """Map ``(label, value)`` -> 64-bit integer seed."""
        base = self.config.seed if self.config.seed is not None else 0
        material = f"{base}|{canonical_label}|{original_value}".encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def surrogate(
        self,
        original: str,
        label: str,
        *,
        lang: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> str:
        """Return a surrogate for ``original`` of type ``label``.

        Args:
            original: The detected PII string. Used for format
                preservation and (in deterministic mode) seed derivation.
            label: Source label as emitted by the model. Run through
                :func:`openmed.core.labels.normalize_label` internally.
            lang: Override the configured language for this call.
            locale: Override the configured Faker locale for this call.

        Returns:
            A locale-appropriate fake value of the same type as the
            detected PII. Falls back to a format-preserving substitution
            when no specific generator is registered.
        """
        effective_lang = lang or self.config.lang
        effective_locale = resolve_locale(effective_lang, locale or self.config.locale)
        canonical = normalize_label(label, effective_lang)

        faker = self._get_faker(effective_locale)
        if self.config.consistent:
            faker.seed_instance(self._derive_seed(canonical, original))

        generator = LABEL_GENERATORS.get(canonical, LABEL_GENERATORS["OTHER"])
        try:
            return generator(faker, original, locale=effective_locale)
        except Exception as exc:  # noqa: BLE001 — never let a single label kill the doc
            warnings.warn(
                f"Anonymizer fallback for label {label!r} (canonical "
                f"{canonical!r}) at locale {effective_locale!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return f"[{label}]"


__all__ = ["Anonymizer", "AnonymizerConfig"]
