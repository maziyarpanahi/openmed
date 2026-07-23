"""Declarative built-in language packs and live legacy-map adapters.

The language-pack registry owns model-backed language declarations.  This
module projects its current snapshot into the public ``set`` and ``dict``
objects that predate language packs, keeping their existing runtime types while
making one registry update visible across every downstream surface.

National-ID-only languages remain explicit compatibility declarations because
they do not yet satisfy the complete :class:`LanguagePack` contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .language_pack import LANGUAGE_PACK_REGISTRY, LanguagePack, LanguagePackRegistry

UNKNOWN_SCRIPT = "Unknown"
UNROUTED_SCRIPT = "Unrouted"

REGISTERED_SEGMENTERS = frozenset({"jieba", "pysbd", "unicode-sentence"})

# These built-in routes intentionally use a named fallback until a dedicated
# PII model is published. They must not be represented as trained/model-backed
# languages in release manifests.
DEFAULT_MODEL_PLACEHOLDER_LANGUAGES = frozenset({"zh"})


def is_registered_segmenter(segmenter_id: str) -> bool:
    """Return whether ``segmenter_id`` names an available sentence segmenter."""

    return segmenter_id in REGISTERED_SEGMENTERS


def _pack(
    code: str,
    model: str,
    locale: str,
    scripts: Sequence[str],
    *,
    national_id_provider: tuple[str, str] | None = None,
    context_scripts: Sequence[str] = (),
) -> LanguagePack:
    providers: dict[str, str] = {}
    if national_id_provider is not None:
        provider_locale, method = national_id_provider
        providers[method] = provider_locale
    return LanguagePack(
        code=code,
        scripts=tuple(scripts),
        default_model=model,
        segmenter_id="pysbd",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale=locale,
        national_id_providers=providers,
        context_scripts=tuple(context_scripts),
    )


BUILTIN_LANGUAGE_PACKS: tuple[LanguagePack, ...] = (
    _pack(
        "en",
        "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        "en_US",
        ("Latin", "Cyrillic", "Greek", "Hebrew", "Thai", UNKNOWN_SCRIPT),
        national_id_provider=("en_US", "ssn"),
    ),
    _pack(
        "fr",
        "OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1",
        "fr_FR",
        ("Latin",),
        national_id_provider=("fr_FR", "ssn"),
    ),
    _pack(
        "de",
        "OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1",
        "de_DE",
        ("Latin",),
        national_id_provider=("de_DE", "german_steuer_id"),
    ),
    _pack(
        "it",
        "OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1",
        "it_IT",
        ("Latin",),
        national_id_provider=("it_IT", "ssn"),
    ),
    _pack(
        "es",
        "OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1",
        "es_ES",
        ("Latin",),
        national_id_provider=("es_ES", "nie"),
    ),
    _pack(
        "nl",
        "OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1",
        "nl_NL",
        ("Latin",),
        national_id_provider=("nl_NL", "ssn"),
    ),
    _pack(
        "hi",
        "OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1",
        "hi_IN",
        ("Devanagari",),
        national_id_provider=("hi_IN", "aadhaar"),
    ),
    _pack(
        "te",
        "OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1",
        "en_IN",
        ("Telugu",),
        national_id_provider=("en_IN", "aadhaar"),
    ),
    _pack(
        "am",
        "OpenMed/privacy-filter-multilingual",
        "am_ET",
        ("Ethiopic",),
        national_id_provider=("am_ET", "ethiopia_fayda"),
    ),
    _pack(
        "pt",
        "OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1",
        "pt_PT",
        ("Latin",),
        national_id_provider=("pt_BR", "cpf"),
    ),
    _pack(
        "ar",
        "OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1",
        "ar_EG",
        ("Arabic",),
    ),
    _pack(
        "he",
        "OpenMed/privacy-filter-multilingual",
        "he_IL",
        (UNROUTED_SCRIPT,),
        national_id_provider=("he_IL", "teudat_zehut"),
    ),
    _pack(
        "ja",
        "OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1",
        "ja_JP",
        ("Han", "Hiragana/Katakana"),
        context_scripts=("Hiragana/Katakana",),
    ),
    LanguagePack(
        code="zh",
        scripts=("Han",),
        default_model="OpenMed/privacy-filter-multilingual",
        segmenter_id="jieba",
        recognizers=("builtin-patterns", "model"),
        surrogate_locale="zh_CN",
        national_id_providers={"chinese_resident_id": "zh_CN"},
    ),
    _pack(
        "tr",
        "OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1",
        "tr_TR",
        ("Latin",),
        national_id_provider=("tr_TR", "ssn"),
    ),
    _pack(
        "id",
        "OpenMed/privacy-filter-multilingual",
        "id_ID",
        (UNROUTED_SCRIPT,),
        national_id_provider=("id_ID", "indonesian_nik"),
    ),
    _pack(
        "th",
        "OpenMed/privacy-filter-multilingual",
        "th_TH",
        (UNROUTED_SCRIPT,),
        national_id_provider=("th_TH", "thai_national_id"),
    ),
    _pack(
        "ko",
        "OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1",
        "ko_KR",
        ("Hangul",),
        national_id_provider=("ko_KR", "korean_rrn"),
    ),
    _pack(
        "ro",
        "OpenMed/privacy-filter-multilingual",
        "ro_RO",
        (UNROUTED_SCRIPT,),
        national_id_provider=("ro_RO", "romanian_cnp"),
    ),
    _pack(
        "ru",
        "OpenMed/privacy-filter-multilingual",
        "ru_RU",
        ("Cyrillic",),
        national_id_provider=("ru_RU", "snils"),
    ),
    _pack(
        "sw",
        "OpenMed/privacy-filter-multilingual",
        "sw",
        ("Latin",),
        national_id_provider=("sw", "kenya_national_id"),
    ),
    _pack(
        "zu",
        "OpenMed/privacy-filter-multilingual",
        "zu_ZA",
        ("Latin",),
        national_id_provider=("zu_ZA", "south_african_id"),
    ),
    _pack(
        "xh",
        "OpenMed/privacy-filter-multilingual",
        "xh_ZA",
        ("Latin",),
        national_id_provider=("xh_ZA", "south_african_id"),
    ),
    _pack(
        "sv",
        "OpenMed/privacy-filter-multilingual",
        "sv_SE",
        ("Latin",),
        national_id_provider=("sv_SE", "ssn"),
    ),
    _pack(
        "da",
        "OpenMed/privacy-filter-multilingual",
        "da_DK",
        ("Latin",),
        national_id_provider=("da_DK", "danish_cpr"),
    ),
    _pack(
        "no",
        "OpenMed/privacy-filter-multilingual",
        "no_NO",
        ("Latin",),
        national_id_provider=("no_NO", "ssn"),
    ),
)


@dataclass(frozen=True, slots=True)
class NationalIdOnlyCapability:
    """Compatibility declaration for a language without a default PII model."""

    locale: str
    provider: tuple[str, str]


NATIONAL_ID_ONLY_CAPABILITIES: Mapping[str, NationalIdOnlyCapability] = {
    "af": NationalIdOnlyCapability("af_ZA", ("af_ZA", "south_african_id")),
    "ha": NationalIdOnlyCapability("ha_NG", ("ha_NG", "nigeria_nin")),
    "ig": NationalIdOnlyCapability("ig_NG", ("ig_NG", "nigeria_nin")),
    "yo": NationalIdOnlyCapability("yo_NG", ("yo_NG", "nigeria_nin")),
    "pl": NationalIdOnlyCapability("pl_PL", ("pl_PL", "pesel")),
    "lv": NationalIdOnlyCapability("lv_LV", ("lv_LV", "personas_kods")),
    "sk": NationalIdOnlyCapability("sk_SK", ("sk_SK", "rodne_cislo")),
    "ms": NationalIdOnlyCapability("ms_MY", ("ms_MY", "mykad")),
    "tl": NationalIdOnlyCapability("fil_PH", ("fil_PH", "philsys_psn")),
    "hu": NationalIdOnlyCapability("hu_HU", ("hu_HU", "hungarian_taj")),
    "et": NationalIdOnlyCapability("et_EE", ("et_EE", "isikukood")),
    "sr": NationalIdOnlyCapability("sr_RS", ("sr_RS", "jmbg")),
    "hr": NationalIdOnlyCapability("hr_HR", ("hr_HR", "ssn")),
    "bg": NationalIdOnlyCapability("bg_BG", ("bg_BG", "egn")),
    "fi": NationalIdOnlyCapability("fi_FI", ("fi_FI", "ssn")),
    "cs": NationalIdOnlyCapability("cs_CZ", ("cs_CZ", "rodne_cislo")),
    "el": NationalIdOnlyCapability("el_GR", ("el_GR", "ssn")),
    "vi": NationalIdOnlyCapability("vi_VN", ("vi_VN", "vietnamese_cccd")),
    "ur": NationalIdOnlyCapability("ur_PK", ("ur_PK", "cnic")),
    "rw": NationalIdOnlyCapability("rw_RW", ("rw_RW", "rwanda_id")),
}

SUPPLEMENTAL_LOCALES: Mapping[str, str] = {
    "as": "as_IN",
    "bn": "bn_BD",
    "gu": "gu_IN",
    "kn": "kn_IN",
    "ml": "ml_IN",
    "mr": "mr_IN",
    "or": "or_IN",
    "pa": "pa_IN",
    "ta": "ta_IN",
}

# Languages surfaced by script routing before a bundled default PII model or
# complete language pack is available. Callers must supply their own model for
# these codes; keeping them separate from ``SUPPORTED_LANGUAGES`` avoids
# advertising model support that OpenMed does not ship yet.
USER_SUPPLIED_MODEL_LANGUAGES: set[str] = {
    "as",
    "bn",
    "gu",
    "kn",
    "ml",
    "mr",
    "ne",
    "or",
    "pa",
    "ta",
    "ur",
}

_SCRIPT_ORDER = (
    "Latin",
    "Arabic",
    "Ethiopic",
    "Han",
    "Hiragana/Katakana",
    "Hangul",
    "Cyrillic",
    "Devanagari",
    "Bengali",
    "Gurmukhi",
    "Gujarati",
    "Odia",
    "Tamil",
    "Telugu",
    "Kannada",
    "Malayalam",
    "Greek",
    "Hebrew",
    "Thai",
    UNKNOWN_SCRIPT,
)

# Exact routing candidates for scripts that cover more languages than the
# current bundled model-backed language packs. These codes are routing hints;
# entries in ``USER_SUPPLIED_MODEL_LANGUAGES`` do not gain default models.
_SCRIPT_LANGUAGE_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "Latin": (
        "en",
        "fr",
        "de",
        "it",
        "es",
        "nl",
        "pt",
        "tr",
        "sw",
        "ig",
        "yo",
        "zu",
        "xh",
    ),
    "Arabic": ("ar", "ha", "ur"),
    "Cyrillic": ("ru"),
    "Han": ("zh", "ja"),
    "Devanagari": ("hi", "mr", "ne"),
    "Bengali": ("bn", "as"),
    "Gurmukhi": ("pa",),
    "Gujarati": ("gu",),
    "Odia": ("or",),
    "Tamil": ("ta",),
    "Telugu": ("te",),
    "Kannada": ("kn",),
    "Malayalam": ("ml",),
}

_LOCALE_ORDER = (
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "te",
    "am",
    "pt",
    "ar",
    "he",
    "ja",
    "zh",
    "tr",
    "id",
    "th",
    "ha",
    "ig",
    "yo",
    "pl",
    "lv",
    "ko",
    "cs",
    "sk",
    "ms",
    "tl",
    "da",
    "ro",
    "ru",
    "sw",
    "rw",
    "af",
    "fi",
    "bg",
    "hr",
    "sr",
    "hu",
    "et",
    "el",
    "vi",
    "ur",
)

_NATIONAL_ID_PROVIDER_ORDER = (
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "te",
    "am",
    "pt",
    "tr",
    "he",
    "id",
    "th",
    "ha",
    "ig",
    "yo",
    "pl",
    "lv",
    "ko",
    "cs",
    "sk",
    "ms",
    "tl",
    "da",
    "ro",
    "ru",
    "sw",
    "rw",
    "af",
    "fi",
    "bg",
    "hr",
    "sr",
    "hu",
    "et",
    "el",
    "vi",
    "ur",
)


def _ordered_items(
    values: Mapping[str, object],
    order: Sequence[str],
) -> tuple[tuple[str, object], ...]:
    known = set(order)
    items = [(code, values[code]) for code in order if code in values]
    items.extend((code, values[code]) for code in sorted(values) if code not in known)
    return tuple(items)


class LanguagePackAdapters:
    """Maintain compatible live ``set`` and ``dict`` registry projections."""

    def __init__(
        self,
        registry: LanguagePackRegistry,
        *,
        builtin_order: Sequence[str] = (),
        national_id_only: Mapping[str, NationalIdOnlyCapability] | None = None,
        supplemental_locales: Mapping[str, str] | None = None,
        user_supplied_model_languages: Sequence[str] = (),
        script_language_candidates: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        """Create and subscribe a live adapter bundle."""

        self.registry = registry
        self._builtin_order = tuple(builtin_order)
        self._national_id_only = dict(national_id_only or {})
        self._supplemental_locales = dict(supplemental_locales or {})
        self._user_supplied_model_languages = set(user_supplied_model_languages)
        self._script_language_candidates = {
            script: tuple(codes)
            for script, codes in (script_language_candidates or {}).items()
        }
        self.supported_languages: set[str] = set()
        self.default_pii_models: dict[str, str] = {}
        self.lang_to_locale: dict[str, str] = {}
        self.national_id_providers: dict[str, tuple[str, str]] = {}
        self.script_language_hints: dict[str, tuple[str, ...]] = {}
        registry._add_listener(self.refresh)

    def _ordered_codes(self) -> tuple[str, ...]:
        codes = tuple(self.registry.iter_codes())
        known = set(codes)
        ordered = [code for code in self._builtin_order if code in known]
        ordered.extend(code for code in codes if code not in self._builtin_order)
        return tuple(ordered)

    def refresh(self) -> None:
        """Refresh every public projection from one registry snapshot."""

        codes = self._ordered_codes()
        packs = tuple(self.registry.get(code) for code in codes)

        self.supported_languages.clear()
        self.supported_languages.update(codes)

        self.default_pii_models.clear()
        self.default_pii_models.update(
            (pack.code, pack.default_model) for pack in packs
        )

        locale_values = dict((pack.code, pack.surrogate_locale) for pack in packs)
        locale_values.update(self._supplemental_locales)
        locale_values.update(
            (code, capability.locale)
            for code, capability in self._national_id_only.items()
        )
        self.lang_to_locale.clear()
        self.lang_to_locale.update(_ordered_items(locale_values, _LOCALE_ORDER))

        provider_values: dict[str, tuple[str, str]] = {}
        for pack in packs:
            if pack.national_id_providers:
                method, locale = next(iter(pack.national_id_providers.items()))
                provider_values[pack.code] = (locale, method)
        provider_values.update(
            (code, capability.provider)
            for code, capability in self._national_id_only.items()
        )
        self.national_id_providers.clear()
        self.national_id_providers.update(
            _ordered_items(provider_values, _NATIONAL_ID_PROVIDER_ORDER)
        )

        script_names = list(_SCRIPT_ORDER)
        script_names.extend(
            sorted(
                {
                    script
                    for pack in packs
                    for script in pack.scripts
                    if script not in _SCRIPT_ORDER and script != UNROUTED_SCRIPT
                }
            )
        )
        self.script_language_hints.clear()
        routable_languages = (
            self.supported_languages
            | set(self._national_id_only)
            | self._user_supplied_model_languages
        )
        for script in script_names:
            configured_hints = self._script_language_candidates.get(script)
            if configured_hints is None:
                hints = tuple(pack.code for pack in packs if script in pack.scripts)
            else:
                hints = tuple(
                    code for code in configured_hints if code in routable_languages
                )
            if hints:
                self.script_language_hints[script] = hints


for _builtin_pack in BUILTIN_LANGUAGE_PACKS:
    try:
        _registered_pack = LANGUAGE_PACK_REGISTRY.get(_builtin_pack.code)
    except KeyError:
        LANGUAGE_PACK_REGISTRY.register(_builtin_pack)
    else:
        if _registered_pack != _builtin_pack:
            raise RuntimeError(
                f"built-in language pack {_builtin_pack.code!r} was replaced "
                "before catalog initialization"
            )

LANGUAGE_PACK_ADAPTERS = LanguagePackAdapters(
    LANGUAGE_PACK_REGISTRY,
    builtin_order=tuple(pack.code for pack in BUILTIN_LANGUAGE_PACKS),
    national_id_only=NATIONAL_ID_ONLY_CAPABILITIES,
    supplemental_locales=SUPPLEMENTAL_LOCALES,
    user_supplied_model_languages=USER_SUPPLIED_MODEL_LANGUAGES,
    script_language_candidates=_SCRIPT_LANGUAGE_CANDIDATES,
)

SUPPORTED_LANGUAGES = LANGUAGE_PACK_ADAPTERS.supported_languages
DEFAULT_PII_MODELS = LANGUAGE_PACK_ADAPTERS.default_pii_models
LANG_TO_LOCALE = LANGUAGE_PACK_ADAPTERS.lang_to_locale
NATIONAL_ID_PROVIDERS = LANGUAGE_PACK_ADAPTERS.national_id_providers
SCRIPT_LANGUAGE_HINTS = LANGUAGE_PACK_ADAPTERS.script_language_hints
NATIONAL_ID_ONLY_LANGUAGES = set(NATIONAL_ID_ONLY_CAPABILITIES)


__all__ = [
    "BUILTIN_LANGUAGE_PACKS",
    "DEFAULT_MODEL_PLACEHOLDER_LANGUAGES",
    "DEFAULT_PII_MODELS",
    "LANG_TO_LOCALE",
    "LANGUAGE_PACK_ADAPTERS",
    "LanguagePackAdapters",
    "NATIONAL_ID_ONLY_CAPABILITIES",
    "NATIONAL_ID_ONLY_LANGUAGES",
    "NATIONAL_ID_PROVIDERS",
    "REGISTERED_SEGMENTERS",
    "SCRIPT_LANGUAGE_HINTS",
    "SUPPORTED_LANGUAGES",
    "USER_SUPPLIED_MODEL_LANGUAGES",
    "is_registered_segmenter",
]
