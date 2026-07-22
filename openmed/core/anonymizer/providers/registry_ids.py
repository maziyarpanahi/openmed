"""Locale national-ID validator provider registry.

Language packs add validator-backed national IDs here by following these steps:

1. Implement or import a deterministic validator that accepts the generated
   surrogate shape and returns ``True`` only for valid identifiers.
2. Ensure Faker can generate matching surrogates, either by using a built-in
   method or by adding a ``BaseProvider`` method in
   :mod:`openmed.core.anonymizer.providers.clinical_ids`.
3. Register a NationalIdSpec with the OpenMed language code or Faker locale,
   the stable ``id_type`` key, the validator, the Faker method name, and an
   optional formatter for final presentation.
4. Register every needed alias explicitly, such as both ``"it"`` and
   ``"it_IT"``, so callers can discover the same ID type from language or
   locale context.
5. Add unit tests that look up the spec and verify a freshly generated
   surrogate passes the registered validator.

Lookups normalize case, hyphens, and whitespace to underscores. The registry
stores no PHI and only keeps callables/classes needed to validate and generate
synthetic identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from openmed.core.pii_i18n import (
    validate_aadhaar,
    validate_bulgarian_egn,
    validate_chinese_passport,
    validate_chinese_resident_id,
    validate_croatian_oib,
    validate_czech_rodne_cislo,
    validate_czechoslovak_rodne_cislo,
    validate_danish_cpr,
    validate_dutch_bsn,
    validate_egyptian_national_id,
    validate_estonian_isikukood,
    validate_finnish_hetu,
    validate_french_nir,
    validate_german_steuer_id,
    validate_ghana_card_pin,
    validate_greek_amka,
    validate_hong_kong_macau_permit,
    validate_hungarian_taj,
    validate_indonesian_nik,
    validate_israeli_teudat_zehut,
    validate_italian_codice_fiscale,
    validate_jmbg,
    validate_kenya_maisha_namba,
    validate_kenya_national_id,
    validate_korean_rrn,
    validate_latvian_personas_kods,
    validate_malaysian_mykad,
    validate_moroccan_cin,
    validate_nigeria_bvn,
    validate_nigeria_nin,
    validate_pakistani_cnic,
    validate_philhealth_pin,
    validate_philsys_psn,
    validate_polish_pesel,
    validate_portuguese_cnpj,
    validate_portuguese_cpf,
    validate_portuguese_nif,
    validate_romanian_cnp,
    validate_spanish_dni,
    validate_spanish_nie,
    validate_taiwan_compatriot_permit,
    validate_thai_national_id,
    validate_turkish_tckn,
    validate_uk_nhs_number,
    validate_uk_nino,
    validate_vietnamese_cccd,
    validate_vietnamese_cmnd,
    validate_za_id_number,
)

from .clinical_ids import (
    AadhaarProvider,
    ABDMProvider,
    AustralianMedicareProvider,
    AustralianTFNProvider,
    BCPHNProvider,
    BulgarianEgnProvider,
    CanadianSINProvider,
    ChineseIdentifierProvider,
    ChineseResidentIdProvider,
    DanishCPRProvider,
    EgyptMoroccoIdProvider,
    EstonianIsikukoodProvider,
    GermanSteuerIdProvider,
    GhanaKenyaIdProvider,
    HungarianTAJProvider,
    IndonesianNIKProvider,
    IsraeliTeudatZehutProvider,
    KoreanRRNProvider,
    LatvianPersonasKodsProvider,
    MalaysianMyKadProvider,
    NigeriaIdProvider,
    NPIProvider,
    OntarioHealthCardProvider,
    PakistaniCnicProvider,
    PhilippinesIdProvider,
    PolishPeselProvider,
    PortugueseNIFProvider,
    RodneCisloProvider,
    RomanianCNPProvider,
    SerbianJmbgProvider,
    SouthAfricanIdProvider,
    SpanishDNIProvider,
    SpanishNIEProvider,
    ThaiNationalIdProvider,
    UKNHSNumberProvider,
    UKNINOProvider,
    VietnameseIdProvider,
    validate_abdm_registry_id,
    validate_abha_address,
    validate_abha_number,
    validate_australian_medicare,
    validate_australian_tfn,
    validate_bc_phn,
    validate_canadian_sin,
    validate_npi,
    validate_ontario_health_card,
    validate_pan,
)

NationalIdValidator = Callable[[str], bool]
NationalIdFormatter = Callable[[str], str]
NationalIdProvider = type[Any]
RegistryKey = tuple[str, str]


def _normalize_key_part(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")

    normalized = re.sub(r"[\s-]+", "_", value.strip().casefold())
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


@dataclass(frozen=True)
class NationalIdSpec:
    """National-ID validator and matching Faker generation contract.

    Attributes:
        lang_or_locale: OpenMed language code, country alias, or Faker locale
            this ID type belongs to.
        id_type: Stable lookup key, such as ``"aadhaar"`` or
            ``"codice_fiscale"``.
        validate: Callable that returns whether a candidate identifier is
            valid for this type.
        faker_method: Faker method name that generates matching surrogates.
        formatter: Optional presentation formatter applied after generation.
        faker_provider: Optional Faker ``BaseProvider`` class that supplies
            ``faker_method`` when Faker does not include it natively.
    """

    lang_or_locale: str
    id_type: str
    validate: NationalIdValidator
    faker_method: str
    formatter: NationalIdFormatter | None = None
    faker_provider: NationalIdProvider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lang_or_locale",
            _normalize_key_part(self.lang_or_locale, field_name="lang_or_locale"),
        )
        object.__setattr__(
            self,
            "id_type",
            _normalize_key_part(self.id_type, field_name="id_type"),
        )
        if not callable(self.validate):
            raise TypeError("validate must be callable")
        if not isinstance(self.faker_method, str) or not self.faker_method.strip():
            raise ValueError("faker_method must be a non-empty string")

    @property
    def key(self) -> RegistryKey:
        """Return the normalized registry key for this spec."""
        return (self.lang_or_locale, self.id_type)

    def format(self, value: str) -> str:
        """Apply the optional formatter to ``value``."""
        if self.formatter is None:
            return value
        return self.formatter(value)


ID_PROVIDER_REGISTRY: dict[RegistryKey, NationalIdSpec] = {}


def register_national_id(spec: NationalIdSpec) -> NationalIdSpec:
    """Register ``spec`` under its normalized ``(lang_or_locale, id_type)`` key.

    Raises:
        ValueError: If that key is already registered.
    """
    key = spec.key
    if key in ID_PROVIDER_REGISTRY:
        raise ValueError(
            "national ID provider already registered for "
            f"{spec.lang_or_locale!r}/{spec.id_type!r}"
        )
    ID_PROVIDER_REGISTRY[key] = spec
    return spec


def get_national_id(lang_or_locale: str, id_type: str) -> NationalIdSpec | None:
    """Return the registered spec for ``lang_or_locale`` and ``id_type``."""
    key = (
        _normalize_key_part(lang_or_locale, field_name="lang_or_locale"),
        _normalize_key_part(id_type, field_name="id_type"),
    )
    return ID_PROVIDER_REGISTRY.get(key)


def iter_national_ids() -> tuple[NationalIdSpec, ...]:
    """Return all registered national-ID specs in registration order."""
    return tuple(ID_PROVIDER_REGISTRY.values())


def national_id_faker_provider_classes() -> tuple[NationalIdProvider, ...]:
    """Return custom Faker provider classes referenced by registered specs."""
    providers: list[NationalIdProvider] = []
    seen: set[NationalIdProvider] = set()
    for spec in ID_PROVIDER_REGISTRY.values():
        provider = spec.faker_provider
        if provider is not None and provider not in seen:
            providers.append(provider)
            seen.add(provider)
    return tuple(providers)


def _register_aliases(
    aliases: tuple[str, ...],
    *,
    id_type: str,
    validate: NationalIdValidator,
    faker_method: str,
    formatter: NationalIdFormatter | None = None,
    faker_provider: NationalIdProvider | None = None,
) -> None:
    for alias in aliases:
        register_national_id(
            NationalIdSpec(
                alias,
                id_type,
                validate,
                faker_method,
                formatter=formatter,
                faker_provider=faker_provider,
            )
        )


def _register_builtin_specs() -> None:
    chinese_aliases = ("zh", "zh_CN", "cn")
    _register_aliases(
        chinese_aliases,
        id_type="chinese_passport",
        validate=validate_chinese_passport,
        faker_method="chinese_passport",
        faker_provider=ChineseIdentifierProvider,
    )
    _register_aliases(
        chinese_aliases,
        id_type="hong_kong_macau_permit",
        validate=validate_hong_kong_macau_permit,
        faker_method="hong_kong_macau_permit",
        faker_provider=ChineseIdentifierProvider,
    )
    _register_aliases(
        chinese_aliases,
        id_type="taiwan_permit",
        validate=validate_taiwan_compatriot_permit,
        faker_method="taiwan_compatriot_permit",
        faker_provider=ChineseIdentifierProvider,
    )
    _register_aliases(
        ("eg", "ar", "ar_EG"),
        id_type="egyptian_national_id",
        validate=validate_egyptian_national_id,
        faker_method="egyptian_national_id",
        faker_provider=EgyptMoroccoIdProvider,
    )
    _register_aliases(
        ("ma", "ar_MA", "fr_MA"),
        id_type="moroccan_cin",
        validate=validate_moroccan_cin,
        faker_method="moroccan_cin",
        faker_provider=EgyptMoroccoIdProvider,
    )
    _register_aliases(
        ("za", "en_ZA", "af", "af_ZA", "zu", "zu_ZA", "xh", "xh_ZA"),
        id_type="sa_id_number",
        validate=validate_za_id_number,
        faker_method="south_african_id",
        faker_provider=SouthAfricanIdProvider,
    )
    nigeria_aliases = ("ng", "en_NG", "ha", "ha_NG", "ig", "ig_NG", "yo", "yo_NG")
    _register_aliases(
        nigeria_aliases,
        id_type="nin",
        validate=validate_nigeria_nin,
        faker_method="nigeria_nin",
        faker_provider=NigeriaIdProvider,
    )
    _register_aliases(
        nigeria_aliases,
        id_type="bvn",
        validate=validate_nigeria_bvn,
        faker_method="nigeria_bvn",
        faker_provider=NigeriaIdProvider,
    )
    _register_aliases(
        ("gh", "en_GH"),
        id_type="ghana_card_pin",
        validate=validate_ghana_card_pin,
        faker_method="ghana_card_pin",
        faker_provider=GhanaKenyaIdProvider,
    )
    _register_aliases(
        ("ke", "en_KE", "sw"),
        id_type="kenya_national_id",
        validate=validate_kenya_national_id,
        faker_method="kenya_national_id",
        faker_provider=GhanaKenyaIdProvider,
    )
    _register_aliases(
        ("ke", "en_KE", "sw"),
        id_type="maisha_namba",
        validate=validate_kenya_maisha_namba,
        faker_method="kenya_maisha_namba",
        faker_provider=GhanaKenyaIdProvider,
    )
    _register_aliases(
        ("fr", "fr_FR"),
        id_type="nir",
        validate=validate_french_nir,
        faker_method="ssn",
    )
    _register_aliases(
        ("de", "de_DE"),
        id_type="steuer_id",
        validate=validate_german_steuer_id,
        faker_method="german_steuer_id",
        faker_provider=GermanSteuerIdProvider,
    )
    _register_aliases(
        ("it", "it_IT"),
        id_type="codice_fiscale",
        validate=validate_italian_codice_fiscale,
        faker_method="ssn",
    )
    _register_aliases(
        ("es", "es_ES"),
        id_type="dni",
        validate=validate_spanish_dni,
        faker_method="dni",
        faker_provider=SpanishDNIProvider,
    )
    _register_aliases(
        ("es", "es_ES"),
        id_type="nie",
        validate=validate_spanish_nie,
        faker_method="nie",
        faker_provider=SpanishNIEProvider,
    )
    _register_aliases(
        ("nl", "nl_NL"),
        id_type="bsn",
        validate=validate_dutch_bsn,
        faker_method="ssn",
    )
    _register_aliases(
        ("in", "india", "en", "hi", "te", "en_IN", "hi_IN", "te_IN"),
        id_type="aadhaar",
        validate=validate_aadhaar,
        faker_method="aadhaar",
        faker_provider=AadhaarProvider,
    )
    for id_type, validate, faker_method in (
        ("abha_number", validate_abha_number, "abha_number"),
        ("abha_address", validate_abha_address, "abha_address"),
        ("pan", validate_pan, "pan"),
        ("abdm_hpr_id", validate_abdm_registry_id, "abdm_hpr_id"),
        ("abdm_hfr_id", validate_abdm_registry_id, "abdm_hfr_id"),
    ):
        _register_aliases(
            ("in", "hi", "te", "en_IN", "hi_IN"),
            id_type=id_type,
            validate=validate,
            faker_method=faker_method,
            faker_provider=ABDMProvider,
        )
    _register_aliases(
        ("zh", "zh_CN", "cn"),
        id_type="resident_id",
        validate=validate_chinese_resident_id,
        faker_method="chinese_resident_id",
        faker_provider=ChineseResidentIdProvider,
    )
    _register_aliases(
        ("id", "id_ID"),
        id_type="nik",
        validate=validate_indonesian_nik,
        faker_method="indonesian_nik",
        faker_provider=IndonesianNIKProvider,
    )
    _register_aliases(
        ("ms", "ms_MY"),
        id_type="mykad",
        validate=validate_malaysian_mykad,
        faker_method="mykad",
        faker_provider=MalaysianMyKadProvider,
    )
    _register_aliases(
        ("tl", "fil_PH", "tl_PH", "ph"),
        id_type="philsys_psn",
        validate=validate_philsys_psn,
        faker_method="philsys_psn",
        faker_provider=PhilippinesIdProvider,
    )
    _register_aliases(
        ("tl", "fil_PH", "tl_PH", "ph"),
        id_type="philhealth_pin",
        validate=validate_philhealth_pin,
        faker_method="philhealth_pin",
        faker_provider=PhilippinesIdProvider,
    )
    _register_aliases(
        ("da", "da_DK", "dk"),
        id_type="cpr",
        validate=validate_danish_cpr,
        faker_method="danish_cpr",
        faker_provider=DanishCPRProvider,
    )
    _register_aliases(
        ("th", "th_TH"),
        id_type="thai_national_id",
        validate=validate_thai_national_id,
        faker_method="thai_national_id",
        faker_provider=ThaiNationalIdProvider,
    )
    _register_aliases(
        ("he", "he_IL"),
        id_type="teudat_zehut",
        validate=validate_israeli_teudat_zehut,
        faker_method="teudat_zehut",
        faker_provider=IsraeliTeudatZehutProvider,
    )
    _register_aliases(
        ("pl", "pl_PL"),
        id_type="pesel",
        validate=validate_polish_pesel,
        faker_method="pesel",
        faker_provider=PolishPeselProvider,
    )
    _register_aliases(
        ("lv", "lv_LV"),
        id_type="personas_kods",
        validate=validate_latvian_personas_kods,
        faker_method="personas_kods",
        faker_provider=LatvianPersonasKodsProvider,
    )
    _register_aliases(
        ("fi", "fi_FI"),
        id_type="hetu",
        validate=validate_finnish_hetu,
        faker_method="ssn",
    )
    _register_aliases(
        ("bg", "bg_BG"),
        id_type="egn",
        validate=validate_bulgarian_egn,
        faker_method="egn",
        faker_provider=BulgarianEgnProvider,
    )
    _register_aliases(
        ("hr", "hr_HR"),
        id_type="oib",
        validate=validate_croatian_oib,
        faker_method="ssn",
    )
    _register_aliases(
        ("sr", "sr_RS"),
        id_type="jmbg",
        validate=validate_jmbg,
        faker_method="jmbg",
        faker_provider=SerbianJmbgProvider,
    )
    _register_aliases(
        ("et", "et_EE"),
        id_type="isikukood",
        validate=validate_estonian_isikukood,
        faker_method="isikukood",
        faker_provider=EstonianIsikukoodProvider,
    )
    _register_aliases(
        ("el", "el_GR"),
        id_type="amka",
        validate=validate_greek_amka,
        faker_method="ssn",
    )
    _register_aliases(
        ("ko", "ko_KR"),
        id_type="rrn",
        validate=validate_korean_rrn,
        faker_method="korean_rrn",
        faker_provider=KoreanRRNProvider,
    )
    _register_aliases(
        ("ko", "ko_KR"),
        id_type="korean_rrn",
        validate=validate_korean_rrn,
        faker_method="korean_rrn",
        faker_provider=KoreanRRNProvider,
    )
    _register_aliases(
        ("sk", "sk_SK"),
        id_type="rodne_cislo",
        validate=validate_czechoslovak_rodne_cislo,
        faker_method="rodne_cislo",
        faker_provider=RodneCisloProvider,
    )
    _register_aliases(
        ("cs", "cs_CZ"),
        id_type="rodne_cislo",
        validate=validate_czech_rodne_cislo,
        faker_method="rodne_cislo",
        faker_provider=RodneCisloProvider,
    )
    _register_aliases(
        ("ro", "ro_RO"),
        id_type="cnp",
        validate=validate_romanian_cnp,
        faker_method="romanian_cnp",
        faker_provider=RomanianCNPProvider,
    )
    _register_aliases(
        ("hu", "hu_HU"),
        id_type="taj",
        validate=validate_hungarian_taj,
        faker_method="hungarian_taj",
        faker_provider=HungarianTAJProvider,
    )
    _register_aliases(
        ("vi", "vi_VN", "vn"),
        id_type="cccd",
        validate=validate_vietnamese_cccd,
        faker_method="vietnamese_cccd",
        faker_provider=VietnameseIdProvider,
    )
    _register_aliases(
        ("vi", "vi_VN", "vn"),
        id_type="cmnd",
        validate=validate_vietnamese_cmnd,
        faker_method="vietnamese_cmnd",
        faker_provider=VietnameseIdProvider,
    )
    _register_aliases(
        ("pt", "pt_BR"),
        id_type="cpf",
        validate=validate_portuguese_cpf,
        faker_method="cpf",
    )
    _register_aliases(
        ("pt", "pt_BR"),
        id_type="cnpj",
        validate=validate_portuguese_cnpj,
        faker_method="cnpj",
    )
    _register_aliases(
        ("pt_PT",),
        id_type="nif",
        validate=validate_portuguese_nif,
        faker_method="nif",
        faker_provider=PortugueseNIFProvider,
    )
    _register_aliases(
        ("tr", "tr_TR"),
        id_type="tckn",
        validate=validate_turkish_tckn,
        faker_method="ssn",
    )
    _register_aliases(
        ("us", "en", "en_US"),
        id_type="npi",
        validate=validate_npi,
        faker_method="npi",
        faker_provider=NPIProvider,
    )
    _register_aliases(
        ("en", "en_GB", "gb", "uk"),
        id_type="nhs_number",
        validate=validate_uk_nhs_number,
        faker_method="nhs_number",
        faker_provider=UKNHSNumberProvider,
    )
    _register_aliases(
        ("en", "en_GB", "gb", "uk"),
        id_type="nino",
        validate=validate_uk_nino,
        faker_method="nino",
        faker_provider=UKNINOProvider,
    )
    _register_aliases(
        ("en", "en_AU", "au"),
        id_type="medicare",
        validate=validate_australian_medicare,
        faker_method="australian_medicare",
        faker_provider=AustralianMedicareProvider,
    )
    _register_aliases(
        ("en", "en_AU", "au"),
        id_type="tfn",
        validate=validate_australian_tfn,
        faker_method="australian_tfn",
        faker_provider=AustralianTFNProvider,
    )
    _register_aliases(
        ("en", "en_CA", "fr_CA", "ca"),
        id_type="sin",
        validate=validate_canadian_sin,
        faker_method="canadian_sin",
        faker_provider=CanadianSINProvider,
    )
    _register_aliases(
        ("en", "en_CA", "fr_CA", "ca"),
        id_type="on_health_card",
        validate=validate_ontario_health_card,
        faker_method="ontario_health_card",
        faker_provider=OntarioHealthCardProvider,
    )
    _register_aliases(
        ("en", "en_CA", "fr_CA", "ca"),
        id_type="bc_phn",
        validate=validate_bc_phn,
        faker_method="bc_phn",
        faker_provider=BCPHNProvider,
    )
    _register_aliases(
        ("ur", "ur_PK"),
        id_type="cnic",
        validate=validate_pakistani_cnic,
        faker_method="cnic",
        faker_provider=PakistaniCnicProvider,
    )


_register_builtin_specs()


__all__ = [
    "ID_PROVIDER_REGISTRY",
    "NationalIdFormatter",
    "NationalIdProvider",
    "NationalIdSpec",
    "NationalIdValidator",
    "get_national_id",
    "iter_national_ids",
    "national_id_faker_provider_classes",
    "register_national_id",
]
