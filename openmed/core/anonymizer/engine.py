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
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

from .. import labels as L
from ..indic_name_match import (
    DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD,
    IndicNameNormalizer,
    is_indic_name_candidate,
)
from ..labels import normalize_label
from ..language_pack import get_language_pack
from ..name_order import CJK_LANGUAGES, normalize_person_span
from ..script_detect import detect_script
from .format_preserve import (
    mask_aadhaar,
    preserve_date_format,
    preserve_email_pattern,
    preserve_id_pattern,
    preserve_phone_format,
)
from .locales import resolve_faker_backend_locale, resolve_locale
from .providers import register_clinical_providers
from .registry import resolve_label_generator

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
        transliteration_aware_name_matching: Link Indic-script and Latin
            PERSON spellings through one canonical surrogate identity.
        indic_name_similarity_threshold: Collision-safety threshold used by
            the stdlib romanization fallback.
        indic_name_normalizer: Optional preconfigured normalizer carrying a
            caller-supplied local transliterator.
    """

    lang: str = "en"
    locale: Optional[str] = None
    consistent: bool = False
    seed: Optional[int] = None
    custom_providers: list[Any] = field(default_factory=list)
    transliteration_aware_name_matching: bool = False
    indic_name_similarity_threshold: float = DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD
    indic_name_normalizer: IndicNameNormalizer | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.transliteration_aware_name_matching, bool):
            raise TypeError("transliteration_aware_name_matching must be a boolean")
        IndicNameNormalizer(similarity_threshold=self.indic_name_similarity_threshold)


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
        transliteration_aware_name_matching: bool = False,
        indic_name_similarity_threshold: float = (
            DEFAULT_INDIC_NAME_SIMILARITY_THRESHOLD
        ),
        indic_name_normalizer: IndicNameNormalizer | None = None,
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
                transliteration_aware_name_matching=(
                    transliteration_aware_name_matching
                ),
                indic_name_similarity_threshold=indic_name_similarity_threshold,
                indic_name_normalizer=indic_name_normalizer,
            )
        self._indic_name_normalizer = self.config.indic_name_normalizer or (
            IndicNameNormalizer(
                similarity_threshold=self.config.indic_name_similarity_threshold
            )
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

    @staticmethod
    def _seed_value_for_identifier(
        canonical_label: str,
        original_value: str,
        locale: str,
    ) -> str:
        """Canonicalize alternate renderings of the same Tanzania NIDA."""
        if canonical_label != L.ID_NUM or locale not in {"en_TZ", "sw", "sw_TZ"}:
            return original_value

        from ..pii_i18n import validate_tanzania_nida

        if validate_tanzania_nida(original_value):
            return re.sub(r"[^0-9]", "", original_value)
        return original_value

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
        canonical = normalize_label(label, effective_lang)

        if (
            canonical == L.PERSON
            and self.config.transliteration_aware_name_matching
            and is_indic_name_candidate(original, lang=effective_lang)
        ):
            identity = self._indic_name_identity(
                original,
                canonical_label=canonical,
            )
            return self.render_name_surrogate(identity, source_surface=original)

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

        script = detect_script(generator_input)
        language_pack = get_language_pack(effective_lang)
        source_key = str(label).strip().upper().replace("-", "_").replace(" ", "_")
        generator, is_script_specific = resolve_label_generator(
            canonical,
            language_pack=language_pack,
            script=script,
            source_label=source_key,
        )
        effective_locale = resolve_locale(
            effective_lang,
            locale or self.config.locale,
            warn_approximation=not is_script_specific,
        )
        if canonical == L.ID_NUM and _is_aadhaar_surrogate_source(original):
            # Aadhaar stays checksum-valid even when an English/code-mixed
            # note routes through the generic en_US locale.
            effective_locale = "en_IN"
        seed_value = self._seed_value_for_identifier(
            canonical,
            seed_value,
            effective_locale,
        )
        faker = self._get_faker(effective_locale)
        if self.config.consistent:
            faker.seed_instance(self._derive_seed(canonical, seed_value))

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

    def anonymize_aadhaar(
        self,
        original: str,
        *,
        strategy: Literal["uidai_mask", "surrogate"] = "uidai_mask",
    ) -> str:
        """Apply a dedicated Aadhaar anonymization strategy.

        Args:
            original: A checksum-valid Aadhaar value.
            strategy: ``"uidai_mask"`` preserves only the last four digits;
                ``"surrogate"`` returns a checksum-valid synthetic Aadhaar.

        Returns:
            A UIDAI-masked value or a Verhoeff-valid synthetic Aadhaar.

        Raises:
            ValueError: If the source is invalid or the strategy is unknown.
        """

        if not _is_valid_aadhaar(original):
            raise ValueError("Aadhaar must have a valid Verhoeff checksum")
        if strategy == "uidai_mask":
            return mask_aadhaar(original)
        if strategy == "surrogate":
            return self.surrogate(
                original,
                "aadhaar",
                lang="en",
                locale="en_IN",
            )
        raise ValueError("strategy must be 'uidai_mask' or 'surrogate'")

    def surrogate_identity(
        self,
        original: str,
        label: str,
        *,
        lang: Optional[str] = None,
        locale: Optional[str] = None,
        attempt: int = 0,
    ) -> str:
        """Return the stored Latin identity for a transliteration-aware name.

        Non-Indic names and disabled configurations retain :meth:`surrogate`
        behavior. The method is primarily used by :class:`SurrogateVault`,
        which stores one identity and renders it for each source script.
        """

        effective_lang = lang or self.config.lang
        canonical = normalize_label(label, effective_lang)
        if (
            canonical == L.PERSON
            and self.config.transliteration_aware_name_matching
            and is_indic_name_candidate(original, lang=effective_lang)
        ):
            return self._indic_name_identity(
                original,
                canonical_label=canonical,
                attempt=attempt,
            )
        return self.surrogate(
            original,
            label,
            lang=effective_lang,
            locale=locale,
        )

    def render_name_surrogate(self, identity: str, *, source_surface: str) -> str:
        """Render a stored Latin PERSON identity in ``source_surface``'s script."""

        if not self.config.transliteration_aware_name_matching:
            return identity
        return self._indic_name_normalizer.render_surrogate(
            identity,
            source_surface=source_surface,
        )

    def _indic_name_identity(
        self,
        original: str,
        *,
        canonical_label: str,
        attempt: int = 0,
    ) -> str:
        """Generate one Latin surrogate identity without retaining raw names."""

        faker = self._get_faker("en_IN")
        canonical_key = self._indic_name_normalizer.canonical_key(original)
        if self.config.consistent:
            faker.seed_instance(
                self._derive_seed(canonical_label, f"{canonical_key}|{attempt}")
            )
        generator, _ = resolve_label_generator(
            canonical_label,
            language_pack=get_language_pack("en"),
            script="Latin",
            source_label=canonical_label,
        )
        try:
            return generator(faker, "", locale="en_IN")
        except Exception as exc:  # noqa: BLE001 - retain safe anonymizer fallback
            warnings.warn(
                "Anonymizer fallback for transliteration-aware PERSON at "
                f"locale 'en_IN': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return "[PERSON]"

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
            if hasattr(faker, "african_phone"):
                african_surrogate = faker.african_phone(original_value)
                if african_surrogate is not None:
                    return african_surrogate
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
        "as_IN",
        "hi_IN",
        "en_IN",
        "pt_PT",
        "pt_BR",
        "uk_UA",
        "cs_CZ",
        "sw",
        "zu_ZA",
        "xh_ZA",
        "el_GR",
    }
)


def _non_identical_surrogate(original: str, generator: Any) -> str | None:
    for _ in range(10):
        surrogate = generator()
        if surrogate != original:
            return surrogate
    return None


def _is_valid_aadhaar(value: str) -> bool:
    from ..pii_i18n import validate_aadhaar

    return validate_aadhaar(value)


def _is_aadhaar_surrogate_source(value: str) -> bool:
    if _is_valid_aadhaar(value):
        return True
    source, separator, attempt = value.rpartition("|")
    return bool(separator and attempt.isdigit() and _is_valid_aadhaar(source))


__all__ = ["Anonymizer", "AnonymizerConfig"]
