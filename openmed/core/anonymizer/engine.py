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

from .. import labels as L
from ..labels import normalize_label
from ..name_order import CJK_LANGUAGES, normalize_person_span
from .format_preserve import (
    preserve_date_format,
    preserve_email_pattern,
    preserve_id_pattern,
    preserve_phone_format,
)
from .locales import resolve_faker_backend_locale, resolve_locale
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

        backend_locale = resolve_faker_backend_locale(locale)
        try:
            faker = Faker(backend_locale)
        except (AttributeError, KeyError):
            warnings.warn(
                f"OpenMed: Faker locale {backend_locale!r} is unavailable; "
                "falling back to 'en_US'. Pass locale=... to override.",
                UserWarning,
                stacklevel=2,
            )
            faker = Faker("en_US")
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

        # CJK PERSON spans: peel a trailing honorific (さん/様/씨/님/先生/…) so
        # the name is swapped while the honorific is re-attached verbatim.
        # Only ja/ko/zh PERSON spans take this path; all other languages and
        # labels are unaffected. See ``openmed.core.name_order`` (OM-291).
        honorific_suffix = ""
        seed_value = original
        generator_input = original
        if canonical == L.PERSON and effective_lang in CJK_LANGUAGES:
            core_name, honorific_suffix = normalize_person_span(
                original, effective_lang
            )
            if honorific_suffix:
                if not core_name:
                    return f"[{label}]{honorific_suffix}"
                # Seed on the bare name so a name with and without an honorific
                # map to the same core surrogate in deterministic mode.
                seed_value = core_name
                generator_input = core_name

        faker = self._get_faker(effective_locale)
        if self.config.consistent:
            faker.seed_instance(self._derive_seed(canonical, seed_value))

        source_key = str(label).strip().upper().replace("-", "_").replace(" ", "_")
        generator = LABEL_GENERATORS.get(
            source_key,
            LABEL_GENERATORS.get(canonical, LABEL_GENERATORS["OTHER"]),
        )
        try:
            generated = generator(faker, generator_input, locale=effective_locale)
            return f"{generated}{honorific_suffix}"
        except Exception as exc:  # noqa: BLE001 — never let a single label kill the doc
            warnings.warn(
                f"Anonymizer fallback for label {label!r} (canonical "
                f"{canonical!r}) at locale {effective_locale!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return f"[{label}]{honorific_suffix}"

    def can_format_preserve(
        self,
        original: str,
        label: str,
        *,
        lang: Optional[str] = None,
    ) -> bool:
        """Return whether ``label`` has a format-preserving generator."""
        canonical = normalize_label(label, lang or self.config.lang)
        value = original or ""
        if canonical == L.PHONE:
            return any(ch.isdigit() for ch in value)
        if canonical in _FORMAT_PRESERVE_DATE_LABELS:
            return any(ch.isdigit() for ch in value)
        if canonical == L.EMAIL:
            return "@" in value
        if canonical in _FORMAT_PRESERVE_GENERIC_ID_LABELS:
            return any(ch.isalnum() for ch in value)
        return False

    def format_preserving_surrogate(
        self,
        original: str,
        label: str,
        *,
        lang: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> str | None:
        """Return a synthetic surrogate that preserves ``original``'s shape.

        Only structured labels with an explicit format-preserving generator are
        supported. ``None`` signals callers to use their normal mask fallback.
        """
        effective_lang = lang or self.config.lang
        effective_locale = resolve_locale(effective_lang, locale or self.config.locale)
        canonical = normalize_label(label, effective_lang)
        original_value = original or ""
        if not self.can_format_preserve(original_value, canonical, lang=effective_lang):
            return None

        faker = self._get_faker(effective_locale)
        if self.config.consistent:
            faker.seed_instance(self._derive_seed(canonical, original_value))

        if canonical == L.PHONE:
            return preserve_phone_format(original_value, rng=faker.random)
        if canonical in _FORMAT_PRESERVE_DATE_LABELS:
            day_first = effective_locale in _FORMAT_PRESERVE_DAY_FIRST_LOCALES
            return _non_identical_surrogate(
                original_value,
                lambda: preserve_date_format(
                    original_value,
                    day_first=day_first,
                    rng=faker.random,
                ),
            )
        if canonical == L.EMAIL:
            return _non_identical_surrogate(
                original_value,
                lambda: preserve_email_pattern(original_value, faker.email()),
            )
        if canonical in _FORMAT_PRESERVE_GENERIC_ID_LABELS:
            return preserve_id_pattern(original_value, rng=faker.random)
        return None


_FORMAT_PRESERVE_DATE_LABELS = frozenset({L.DATE, L.DATE_OF_BIRTH})

_FORMAT_PRESERVE_GENERIC_ID_LABELS = frozenset(
    {
        L.ACCOUNT_NUMBER,
        L.API_KEY,
        L.BIC,
        L.BITCOIN_ADDRESS,
        L.CREDIT_CARD,
        L.CVV,
        L.ETHEREUM_ADDRESS,
        L.IBAN,
        L.ID_NUM,
        L.IMEI,
        L.IP_ADDRESS,
        L.LITECOIN_ADDRESS,
        L.MAC_ADDRESS,
        L.MASKED_NUMBER,
        L.PASSWORD,
        L.PIN,
        L.SSN,
        L.VEHICLE_REGISTRATION,
        L.VIN,
        L.ZIPCODE,
    }
)

_FORMAT_PRESERVE_DAY_FIRST_LOCALES = frozenset(
    {
        "fr_FR",
        "de_DE",
        "it_IT",
        "es_ES",
        "nl_NL",
        "hi_IN",
        "en_IN",
        "pt_PT",
        "pt_BR",
        "cs_CZ",
        "sw",
        "zu_ZA",
        "xh_ZA",
    }
)


def _non_identical_surrogate(original: str, generator: Any) -> str | None:
    for _ in range(10):
        surrogate = generator()
        if surrogate != original:
            return surrogate
    return None


__all__ = ["Anonymizer", "AnonymizerConfig"]
