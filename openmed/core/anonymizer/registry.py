"""Per-canonical-label generator registry.

Each generator takes ``(faker, original, locale)`` and returns a string
surrogate. Generators are responsible for:

  1. Picking the right Faker method or custom provider for ``locale``
     (e.g. ``cpf()`` for ``pt_BR``, ``nie()`` for ``es_ES``).
  2. Format-preserving the output where it makes downstream tooling
     happier (phone digit groups, date separators, email domains).

The registry is keyed off canonical labels from :mod:`openmed.core.labels`,
so callers should run ``normalize_label(model_label)`` before lookup.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib import resources
from typing import Callable, Dict

from .. import labels as L
from ..language_pack import (
    LanguagePack,
    get_language_pack,
    register_language_pack,
)
from .format_preserve import (
    preserve_date_format,
    preserve_email_pattern,
    preserve_id_pattern,
    preserve_phone_format,
)
from .locales import ZH_CN_ADDRESS_LOCALE

Generator = Callable[..., str]
"""Signature: ``(faker, original: str, *, locale: str) -> str``."""

_INDIA_LOCALES = frozenset({"en_IN", "hi_IN", "pa_IN"})


def _contains_original_fragment(original: str, candidate: str) -> bool:
    """Return whether a meaningful original token survives in ``candidate``."""

    fragments = re.findall(
        r"[A-Za-z]{3,}|\d{4,}|[\u0900-\u097f]{3,}",
        original,
    )
    folded_candidate = candidate.casefold()
    return any(fragment.casefold() in folded_candidate for fragment in fragments)


def _draw_distinct(faker, original: str, method: str) -> str:
    """Draw from ``method`` without retaining meaningful original fragments."""

    candidate = ""
    for _ in range(20):
        candidate = str(getattr(faker, method)())
        if not _contains_original_fragment(original, candidate):
            return candidate
    return candidate


def _locale_fake_value(faker, locale: str, key: str, original: str):
    """Draw a curated value for an OpenMed conceptual locale, if present."""

    from ..pii_i18n import LOCALE_FAKE_DATA

    values = LOCALE_FAKE_DATA.get(locale, {}).get(key, ())
    if not values:
        return None
    alternatives = tuple(value for value in values if value != original)
    return faker.random_element(alternatives or tuple(values))


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


_HAN_NAME_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ZH_GIVEN_NAME_FALLBACK = tuple("清宁安和嘉悦晨星岚澄涵瑞瑶璟")


def _is_zh_cn(locale: str) -> bool:
    return locale.replace("-", "_").casefold() == "zh_cn"


def _han_characters(value: str) -> str:
    return "".join(_HAN_NAME_RE.findall(value or ""))


def _zh_surname_pools():
    from ..pii_i18n import CHINESE_COMPOUND_SURNAMES, CHINESE_SINGLE_SURNAMES

    return (
        tuple(sorted(CHINESE_SINGLE_SURNAMES)),
        tuple(sorted(CHINESE_COMPOUND_SURNAMES)),
    )


def _draw_zh_surname(faker, *, compound: bool, forbidden: set[str]) -> str:
    single_surnames, compound_surnames = _zh_surname_pools()
    pool = compound_surnames if compound else single_surnames
    eligible = tuple(
        surname for surname in pool if not set(surname).intersection(forbidden)
    )
    return faker.random_element(eligible or pool)


def _draw_zh_given_name(faker, *, length: int, forbidden: set[str]) -> str:
    sampled_characters: list[str] = []
    for _ in range(128):
        candidate = _han_characters(faker.first_name())
        sampled_characters.extend(candidate)
        if len(candidate) == length and not set(candidate).intersection(forbidden):
            return candidate

    eligible = [
        char
        for char in (*sampled_characters, *_ZH_GIVEN_NAME_FALLBACK)
        if char not in forbidden
    ]
    if not eligible:
        raise RuntimeError("could not generate a non-leaking Chinese given name")
    return "".join(faker.random.choice(eligible) for _ in range(length))


def _gen_zh_person(faker, original: str) -> str:
    source = _han_characters(original)
    source_characters = set(source)
    _single_surnames, compound_surnames = _zh_surname_pools()
    compound = any(source.startswith(surname) for surname in compound_surnames)
    target_length = 3 if compound else max(2, min(len(source) or 2, 3))
    given_length = target_length - (2 if compound else 1)

    for _ in range(64):
        surname = _draw_zh_surname(
            faker,
            compound=compound,
            forbidden=source_characters,
        )
        given_name = _draw_zh_given_name(
            faker,
            length=given_length,
            forbidden=source_characters | set(surname),
        )
        candidate = f"{surname}{given_name}"
        if candidate != source and not set(candidate).intersection(source_characters):
            return candidate
    raise RuntimeError("could not generate a distinct Chinese person surrogate")


def _gen_person(faker, original, *, locale):
    if _is_zh_cn(locale):
        return _gen_zh_person(faker, original)
    return _locale_fake_value(faker, locale, "NAME", original) or faker.name()


def _gen_first_name(faker, original, *, locale):
    if _is_zh_cn(locale):
        source = _han_characters(original)
        return _draw_zh_given_name(
            faker,
            length=max(1, min(len(source) or 1, 2)),
            forbidden=set(source),
        )
    return (
        _locale_fake_value(faker, locale, "FIRST_NAME", original) or faker.first_name()
    )


def _gen_last_name(faker, original, *, locale):
    if _is_zh_cn(locale):
        source = _han_characters(original)
        return _draw_zh_surname(
            faker,
            compound=len(source) >= 2,
            forbidden=set(source),
        )
    return _locale_fake_value(faker, locale, "LAST_NAME", original) or faker.last_name()


def _gen_middle_name(faker, original, *, locale):
    return faker.first_name()


def _gen_india_person(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_name"):
        return _draw_distinct(faker, original, "indian_name")
    return _gen_person(faker, original, locale=locale)


def _gen_india_first_name(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_first_name"):
        return _draw_distinct(faker, original, "indian_first_name")
    return _gen_first_name(faker, original, locale=locale)


def _gen_india_last_name(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_last_name"):
        return _draw_distinct(faker, original, "indian_last_name")
    return _gen_last_name(faker, original, locale=locale)


def _gen_india_middle_name(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_first_name"):
        return _draw_distinct(faker, original, "indian_first_name")
    return _gen_middle_name(faker, original, locale=locale)


def _gen_prefix(faker, original, *, locale):
    return faker.prefix()


def _gen_username(faker, original, *, locale):
    return faker.user_name()


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------


def _gen_email(faker, original, *, locale):
    fake = faker.email()
    return preserve_email_pattern(original, fake)


def _gen_phone(faker, original, *, locale):
    curated = _locale_fake_value(faker, locale, "PHONE", original)
    if curated is not None:
        return curated
    if _is_zh_cn(locale):
        from openmed.core.pii_i18n import validate_chinese_mobile_number

        if validate_chinese_mobile_number(original):
            from .providers.clinical_ids import generate_chinese_mobile_number

            return generate_chinese_mobile_number(original, rng=faker.random)
    if locale in {"en_NG", "ha_NG", "ig_NG", "yo_NG"} and hasattr(
        faker, "ng_mobile_number"
    ):
        return faker.ng_mobile_number(original)
    if locale in {"af_ZA", "en_ZA", "zu_ZA", "xh_ZA"} and hasattr(
        faker, "za_mobile_number"
    ):
        return faker.za_mobile_number(original)
    if any(ch.isdigit() for ch in original):
        if hasattr(faker, "african_phone"):
            african_surrogate = faker.african_phone(original)
            if african_surrogate is not None:
                return african_surrogate
        return preserve_phone_format(original, rng=faker.random)
    return faker.phone_number()


def _gen_india_phone(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_phone_number"):
        return _draw_distinct(faker, original, "indian_phone_number")
    return _gen_phone(faker, original, locale=locale)


def _gen_url(faker, original, *, locale):
    return faker.url()


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


_ZH_ADDRESS_RESOURCE = "data/zh_cn_administrative_divisions.json"
_ZH_ADMINISTRATIVE_PREFIX_RE = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fff]{1,12}?省"
    r"[\u3400-\u4dbf\u4e00-\u9fff]{1,12}?市"
)
_ZH_SYNTHETIC_STREETS = (
    "安澜路",
    "和景街",
    "康悦路",
    "清禾大道",
    "瑞宁街",
    "新岚路",
    "云栖大道",
    "竹安路",
)


@lru_cache(maxsize=1)
def _zh_address_divisions() -> tuple[tuple[str, str, str], ...]:
    """Load the bundled province/city/district hierarchy."""
    resource = resources.files("openmed.clinical").joinpath(_ZH_ADDRESS_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    divisions: list[tuple[str, str, str]] = []
    for province_record in payload["divisions"]:
        province = province_record["province"]
        for city_record in province_record["cities"]:
            city = city_record["city"]
            for district in city_record["districts"]:
                divisions.append((province, city, district))
    if not divisions:
        raise ValueError("Chinese administrative-division resource is empty")
    return tuple(divisions)


def _is_zh_address_locale(locale: str) -> bool:
    return locale.replace("-", "_").casefold() == ZH_CN_ADDRESS_LOCALE.casefold()


def _pick_zh_division(faker, original: str) -> tuple[str, str, str]:
    divisions = _zh_address_divisions()
    # Do not accidentally retain an original administrative token. The source
    # list is deliberately broad enough that a different hierarchy is always
    # available for normal address inputs.
    candidates = [
        division
        for division in divisions
        if all(part not in original for part in division)
    ]
    return faker.random_element(candidates or divisions)


def _zh_random_digits(faker, *, width: int, original: str) -> str:
    source_numbers = set(re.findall(r"\d+", original))
    minimum = 10 ** (width - 1)
    maximum = (10**width) - 1

    def is_safe(value: str) -> bool:
        return all(source not in value for source in source_numbers)

    for _ in range(12):
        value = str(faker.random_int(min=minimum, max=maximum))
        if is_safe(value):
            return value

    # Exhaustive fallback makes the privacy property deterministic even when
    # several random candidates contain a short source number such as 17 or 88.
    for candidate in range(minimum, maximum + 1):
        value = str(candidate)
        if is_safe(value):
            return value
    raise ValueError("Could not generate Chinese address digits without source reuse")


def _gen_location(faker, original, *, locale):
    if _is_zh_address_locale(locale):
        return "".join(_pick_zh_division(faker, original))
    # Prefer city-level granularity since most "LOCATION" detections are cities
    return _locale_fake_value(faker, locale, "LOCATION", original) or faker.city()


def _gen_street_address(faker, original, *, locale):
    if _is_zh_address_locale(locale):
        streets = [street for street in _ZH_SYNTHETIC_STREETS if street not in original]
        street = faker.random_element(streets or _ZH_SYNTHETIC_STREETS)
        building = _zh_random_digits(faker, width=3, original=original).lstrip("0")
        postcode = _zh_random_digits(faker, width=6, original=original)
        # The deterministic address assist deliberately emits only the
        # district/street/building tail beside an existing LOCATION prefix.
        # Avoid inserting a second, unrelated province/city hierarchy when
        # that tail is replaced independently during de-identification.
        if _ZH_ADMINISTRATIVE_PREFIX_RE.match(original) is None:
            return f"{street}{building}号，邮编{postcode}"
        province, city, district = _pick_zh_division(faker, original)
        return f"{province}{city}{district}{street}{building}号，邮编{postcode}"
    return (
        _locale_fake_value(faker, locale, "STREET_ADDRESS", original)
        or faker.street_address()
    )


def _gen_india_street_address(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_address"):
        return _draw_distinct(faker, original, "indian_address")
    return _gen_street_address(faker, original, locale=locale)


def _gen_building_number(faker, original, *, locale):
    if _is_zh_address_locale(locale):
        number = _zh_random_digits(faker, width=3, original=original).lstrip("0")
        return f"{number}号"
    return faker.building_number()


def _gen_zipcode(faker, original, *, locale):
    if _is_zh_address_locale(locale):
        return _zh_random_digits(faker, width=6, original=original)
    if any(ch.isdigit() or ch.isalpha() for ch in original):
        return preserve_id_pattern(original, rng=faker.random)
    return faker.postcode()


def _gen_india_zipcode(faker, original, *, locale):
    if locale in _INDIA_LOCALES and hasattr(faker, "indian_pin"):
        return _draw_distinct(faker, original, "indian_pin")
    return _gen_zipcode(faker, original, locale=locale)


def _gen_gps(faker, original, *, locale):
    lat, lon = faker.latitude(), faker.longitude()
    return f"{float(lat):.6f}, {float(lon):.6f}"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

# Day-first locales — same set as openmed.core.pii._DAY_FIRST_LANGS but
# expressed in Faker locale terms.
_DAY_FIRST_LOCALES = frozenset(
    {
        "fr_FR",
        "de_DE",
        "it_IT",
        "es_ES",
        "nl_NL",
        "hi_IN",
        "pa_IN",
        "en_IN",
        "pt_PT",
        "pt_BR",
        "he_IL",
        "id_ID",
        "ms_MY",
        "fil_PH",
        "sv_SE",
        "da_DK",
        "no_NO",
        "th_TH",
        "uk_UA",
        "cs_CZ",
        "sk_SK",
        "af_ZA",
        "en_ZA",
        "zu_ZA",
        "xh_ZA",
        "el_GR",
    }
)


def _gen_date(faker, original, *, locale):
    day_first = locale in _DAY_FIRST_LOCALES
    return preserve_date_format(original, day_first=day_first, rng=faker.random)


def _gen_date_of_birth(faker, original, *, locale):
    day_first = locale in _DAY_FIRST_LOCALES
    return preserve_date_format(original, day_first=day_first, rng=faker.random)


def _gen_time(faker, original, *, locale):
    return faker.time()


def _gen_age(faker, original, *, locale):
    return str(faker.random_int(min=0, max=120))


# ---------------------------------------------------------------------------
# Identifiers — locale-aware dispatch
# ---------------------------------------------------------------------------

# Maps locale -> ``(faker_method_name, validator_module_attr or None)``.
# When the locale-appropriate ID method exists, we call it; otherwise we
# format-preserve the original.
_LOCALE_ID_METHODS = {
    "af_ZA": "south_african_id",
    "am_ET": "ethiopia_fayda",
    "ar_EG": "egyptian_national_id",
    "ar_MA": "moroccan_cin",
    "en_ZA": "south_african_id",
    "en_NG": "nigeria_nin",
    "ha_NG": "nigeria_nin",
    "ig_NG": "nigeria_nin",
    "yo_NG": "nigeria_nin",
    "en_GH": "ghana_card_pin",
    "en_KE": "kenya_national_id",
    "sw": "kenya_national_id",
    "zu_ZA": "south_african_id",
    "xh_ZA": "south_african_id",
    "pt_BR": "cpf",
    "pt_PT": "nif",
    "fr_FR": "ssn",
    "fr_MA": "moroccan_cin",
    "it_IT": "ssn",
    "es_ES": "nie",
    "nl_NL": "ssn",
    "en_IN": "aadhaar",
    "hi_IN": "aadhaar",
    "pa_IN": "aadhaar",
    "zh_CN": "chinese_resident_id",
    "de_DE": "german_steuer_id",
    "en_US": "ssn",
    "en_GB": "nino",
    "en_ET": "ethiopia_fayda",
    "en_TZ": "tanzania_nida",
    "en_UG": "uganda_nin",
    "tr_TR": "ssn",
    "he_IL": "teudat_zehut",
    "id_ID": "indonesian_nik",
    "ms_MY": "mykad",
    "fil_PH": "philsys_psn",
    "da_DK": "danish_cpr",
    "pl_PL": "pesel",
    "lv_LV": "personas_kods",
    "ko_KR": "korean_rrn",
    "sv_SE": "ssn",
    "no_NO": "ssn",
    "th_TH": "thai_national_id",
    "uk_UA": "rnokpp",
    "sk_SK": "rodne_cislo",
    "cs_CZ": "rodne_cislo",
    "ro_RO": "romanian_cnp",
    "ru_RU": "snils",
    "fi_FI": "ssn",
    "bg_BG": "egn",
    "hr_HR": "ssn",
    "sr_RS": "jmbg",
    "hu_HU": "hungarian_taj",
    "et_EE": "isikukood",
    "el_GR": "ssn",
    "vi_VN": "vietnamese_cccd",
    "rw_RW": "rwanda_id",
    "sw_TZ": "tanzania_nida",
    "ur_PK": "cnic",
}

_INDIA_ID_METHODS = {
    "ABHA_NUMBER": "abha_number",
    "ABHA_ADDRESS": "abha_address",
    "AADHAAR": "aadhaar",
    "PAN": "pan",
    "ABDM_HPR_ID": "abdm_hpr_id",
    "ABDM_HFR_ID": "abdm_hfr_id",
}

_FIELD_PRESERVING_ID_METHODS = frozenset({"rwanda_id", "tanzania_nida", "uganda_nin"})


_MRZ_CHARSET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")


def _mrz_surrogate(faker, original):
    """Return a valid MRZ surrogate when ``original`` looks like an MRZ block."""
    if not original or "\n" not in original:
        return None
    lines = [line.strip() for line in original.strip().splitlines()]
    if not lines or any(set(line) - _MRZ_CHARSET for line in lines):
        return None
    lengths = {len(line) for line in lines}
    from openmed.core.pii_i18n import generate_mrz_td1, generate_mrz_td3

    if len(lines) == 2 and lengths == {44}:
        return generate_mrz_td3(faker.random)
    if len(lines) == 3 and lengths == {30}:
        return generate_mrz_td1(faker.random)
    return None


def _uscc_surrogate(faker, original):
    """Return a valid USCC surrogate when ``original`` is a valid USCC."""
    if not original:
        return None
    from openmed.core.pii_i18n import validate_unified_social_credit_code

    if not validate_unified_social_credit_code(original.strip()):
        return None
    from openmed.core.anonymizer.providers.clinical_ids import (
        generate_unified_social_credit_code,
    )

    return generate_unified_social_credit_code(rng=faker.random)


def _mpesa_surrogate(faker, original):
    """Return an M-Pesa surrogate when ``original`` is a valid code."""

    if not original:
        return None
    from openmed.core.pii_i18n import validate_mpesa_transaction_code

    if not validate_mpesa_transaction_code(original):
        return None
    if not hasattr(faker, "mpesa_transaction_code"):
        return None
    return faker.mpesa_transaction_code(original)


def _gen_mobile_money_identifier(
    faker,
    original: str,
    *,
    method: str,
    validator: Callable[[str], bool],
) -> str:
    """Generate a provider-specific billing identifier when shape-valid."""

    if validator(original) and hasattr(faker, method):
        return str(getattr(faker, method)(original))
    return preserve_id_pattern(original, rng=faker.random)


def _gen_mobile_money_paybill(faker, original, *, locale):
    from openmed.core.pii_i18n import validate_mobile_money_paybill

    return _gen_mobile_money_identifier(
        faker,
        original,
        method="mobile_money_paybill",
        validator=validate_mobile_money_paybill,
    )


def _gen_mobile_money_till(faker, original, *, locale):
    from openmed.core.pii_i18n import validate_mobile_money_till

    return _gen_mobile_money_identifier(
        faker,
        original,
        method="mobile_money_till",
        validator=validate_mobile_money_till,
    )


def _gen_mobile_money_agent(faker, original, *, locale):
    from openmed.core.pii_i18n import validate_mobile_money_paybill

    return _gen_mobile_money_identifier(
        faker,
        original,
        method="mobile_money_agent",
        validator=validate_mobile_money_paybill,
    )


def _gen_momo_reference(faker, original, *, locale):
    from openmed.core.pii_i18n import validate_momo_reference

    return _gen_mobile_money_identifier(
        faker,
        original,
        method="momo_reference",
        validator=validate_momo_reference,
    )


def _gen_facility_id(faker, original, *, locale):
    """Generate a validator-compatible Kenya or Nigeria facility surrogate."""

    from openmed.core.pii_i18n import (
        validate_kenya_mfl_code,
        validate_nigeria_hfr_code,
    )

    if validate_kenya_mfl_code(original) and hasattr(faker, "kmhfl_code"):
        return faker.kmhfl_code(original)
    if validate_nigeria_hfr_code(original) and hasattr(faker, "hfr_facility_code"):
        return faker.hfr_facility_code(original)
    return preserve_id_pattern(original, rng=faker.random)


def _generate_distinct_chinese_resident_id(faker, original):
    """Return a valid Chinese Resident ID that differs from ``original``."""
    from openmed.core.pii_i18n import validate_chinese_resident_id

    generate = getattr(faker, "chinese_resident_id")
    for _ in range(10):
        candidate = generate()
        if candidate != original and validate_chinese_resident_id(candidate):
            return candidate
    raise RuntimeError("could not generate a distinct Chinese Resident ID")


def _india_health_id_surrogate(faker, original):
    """Return a validator-compatible surrogate for an Indian health ID."""

    if not original:
        return None
    from openmed.core.pii_i18n import (
        validate_abha_address,
        validate_abha_number,
        validate_indian_ration_card,
        validate_upi_id,
    )

    candidate = original.strip()
    if validate_abha_address(candidate):
        return faker.abha_address()
    if validate_abha_number(candidate):
        return faker.abha_number()
    if validate_upi_id(candidate):
        return faker.upi_id()
    if re.fullmatch(
        r"[A-Za-z]{1,3}[\s/-]\d{8,12}(?:[\s/-][A-Za-z0-9]{1,4})?",
        candidate,
    ) and validate_indian_ration_card(candidate):
        return faker.indian_ration_card()
    return None


def _gen_id_num(faker, original, *, locale):
    method = _LOCALE_ID_METHODS.get(locale)
    if locale == "zh_CN" and method and hasattr(faker, method) and original:
        from openmed.core.pii_i18n import validate_chinese_resident_id

        if validate_chinese_resident_id(original.strip()):
            return _generate_distinct_chinese_resident_id(faker, original)
    mrz = _mrz_surrogate(faker, original)
    if mrz is not None:
        return mrz
    uscc = _uscc_surrogate(faker, original)
    if uscc is not None:
        return uscc
    mpesa = _mpesa_surrogate(faker, original)
    if mpesa is not None:
        return mpesa
    if _is_zh_cn(locale):
        from openmed.core.pii_i18n import (
            validate_chinese_passport,
            validate_hong_kong_macau_permit,
            validate_taiwan_compatriot_permit,
        )

        if validate_chinese_passport(original):
            return faker.chinese_passport(original)
        if validate_hong_kong_macau_permit(original):
            return faker.hong_kong_macau_permit(original)
        if validate_taiwan_compatriot_permit(original):
            return faker.taiwan_compatriot_permit(original)
    if locale in {"en_IN", "hi_IN", "pa_IN", "te_IN"}:
        india_health_id = _india_health_id_surrogate(faker, original)
        if india_health_id is not None:
            return india_health_id
    if locale == "sw":
        from openmed.core.pii_i18n import validate_tanzania_nida

        if validate_tanzania_nida(original):
            return faker.tanzania_nida(original)
    method = _LOCALE_ID_METHODS.get(locale)
    if method and hasattr(faker, method):
        if locale == "zh_CN":
            return _generate_distinct_chinese_resident_id(faker, original)
        if method in _FIELD_PRESERVING_ID_METHODS | {
            "egyptian_national_id",
            "moroccan_cin",
            "nigeria_nin",
            "south_african_id",
        }:
            return getattr(faker, method)(original)
        if locale in {"en_GH", "en_KE", "sw"}:
            if locale in {"en_KE", "sw"} and re.fullmatch(
                r"[0-9]{9}", original.strip()
            ):
                return faker.kenya_maisha_namba(original)
            return getattr(faker, method)(original)
        return getattr(faker, method)()
    return preserve_id_pattern(original, rng=faker.random)


def _gen_india_id(faker, original, *, locale, id_type):
    method = _INDIA_ID_METHODS[id_type]
    original_fingerprint = "".join(
        char.casefold() for char in original if char.isalnum()
    )
    generated = ""
    for _ in range(20):
        generated = getattr(faker, method)()
        generated_fingerprint = "".join(
            char.casefold() for char in generated if char.isalnum()
        )
        if generated_fingerprint != original_fingerprint:
            return generated
    return generated


def _gen_abha_number(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="ABHA_NUMBER")


def _gen_abha_address(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="ABHA_ADDRESS")


def _gen_aadhaar(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="AADHAAR")


def _gen_pan(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="PAN")


def _gen_abdm_hpr_id(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="ABDM_HPR_ID")


def _gen_abdm_hfr_id(faker, original, *, locale):
    return _gen_india_id(faker, original, locale=locale, id_type="ABDM_HFR_ID")


def _gen_ng_nin(faker, original, *, locale):
    return faker.nigeria_nin(original)


def _gen_ng_bvn(faker, original, *, locale):
    return faker.nigeria_bvn(original)


def _gen_ghana_card(faker, original, *, locale):
    return faker.ghana_card_pin(original)


def _gen_ke_national_id(faker, original, *, locale):
    return faker.kenya_national_id(original)


def _gen_ke_maisha_namba(faker, original, *, locale):
    return faker.kenya_maisha_namba(original)


def _indian_id_surrogate(faker, original):
    """Return a validator-compatible Indian ID surrogate when recognized."""

    if not original:
        return None

    from openmed.core.anonymizer.providers.clinical_ids import (
        validate_abha,
        validate_gstin,
        validate_pan,
    )
    from openmed.core.pii_i18n import (
        validate_aadhaar,
        validate_ifsc,
        validate_indian_driving_licence,
        validate_indian_passport,
        validate_voter_id_epic,
    )

    validators_and_methods = (
        (validate_gstin, "gstin"),
        (validate_pan, "pan"),
        (validate_ifsc, "ifsc"),
        (validate_indian_driving_licence, "indian_driving_licence"),
        (validate_indian_passport, "indian_passport"),
        (validate_voter_id_epic, "voter_id_epic"),
        (validate_abha, "abha"),
        (validate_aadhaar, "aadhaar"),
    )
    for validator, method in validators_and_methods:
        if validator(original) and hasattr(faker, method):
            return _draw_distinct(faker, original, method)
    return None


def _gen_india_id_num(faker, original, *, locale):
    if locale in _INDIA_LOCALES:
        surrogate = _indian_id_surrogate(faker, original)
        if surrogate is not None:
            return surrogate
    return _gen_id_num(faker, original, locale=locale)


def _gen_ssn(faker, original, *, locale):
    method = _LOCALE_ID_METHODS.get(locale, "ssn")
    if hasattr(faker, method):
        return getattr(faker, method)()
    return faker.ssn()


def _gen_account_number(faker, original, *, locale):
    return faker.bban() if hasattr(faker, "bban") else faker.iban()


def _gen_password(faker, original, *, locale):
    length = max(8, min(len(original), 32)) if original else 12
    return faker.password(length=length)


def _gen_pin(faker, original, *, locale):
    length = max(3, min(len(original), 8)) if original else 4
    return "".join(str(faker.random.randint(0, 9)) for _ in range(length))


def _gen_api_key(faker, original, *, locale):
    return faker.sha256()


# ---------------------------------------------------------------------------
# Financial
# ---------------------------------------------------------------------------


def _gen_credit_card(faker, original, *, locale):
    if _is_zh_cn(locale):
        from openmed.core.pii_i18n import validate_chinese_bank_card

        if validate_chinese_bank_card(original):
            from .providers.clinical_ids import generate_chinese_bank_card

            return generate_chinese_bank_card(original, rng=faker.random)
    return faker.credit_card_number()


def _gen_credit_card_issuer(faker, original, *, locale):
    return faker.credit_card_provider()


def _gen_cvv(faker, original, *, locale):
    return faker.credit_card_security_code()


def _gen_iban(faker, original, *, locale):
    if hasattr(faker, "financial_iban"):
        return faker.financial_iban()

    from .providers import clinical_ids

    value = faker.iban()
    if clinical_ids.validate_iban(value):
        return value
    return clinical_ids.generate_iban(rng=faker.random)


def _gen_bic(faker, original, *, locale):
    if hasattr(faker, "financial_bic"):
        return faker.financial_bic()

    from .providers import clinical_ids

    if hasattr(faker, "swift11"):
        value = faker.swift11()
        if clinical_ids.validate_bic(value):
            return value
    return clinical_ids.generate_bic(include_branch=True, rng=faker.random)


def _gen_amount(faker, original, *, locale):
    return f"{faker.pyfloat(positive=True, min_value=10, max_value=100000):.2f}"


def _gen_currency(faker, original, *, locale):
    return faker.currency_code()


def _gen_bitcoin_address(faker, original, *, locale):
    if hasattr(faker, "ascii_email"):
        return faker.bothify(
            "1?????????????????????????????????",
            letters="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789",
        )
    return faker.sha256()[:34]


def _gen_ethereum_address(faker, original, *, locale):
    return "0x" + faker.hexify(text="^" * 40, upper=False)


def _gen_litecoin_address(faker, original, *, locale):
    return faker.bothify(
        "L?????????????????????????????????",
        letters="ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789",
    )


def _gen_masked_number(faker, original, *, locale):
    return (
        preserve_id_pattern(original, rng=faker.random)
        if original
        else faker.numerify("****-****-####")
    )


# ---------------------------------------------------------------------------
# Demographics / work / tech
# ---------------------------------------------------------------------------


def _gen_gender(faker, original, *, locale):
    return faker.random_element(("Female", "Male", "Non-binary"))


def _gen_eye_color(faker, original, *, locale):
    return faker.random_element(("brown", "blue", "green", "hazel", "amber", "gray"))


def _gen_height(faker, original, *, locale):
    cm = faker.random_int(min=140, max=210)
    return f"{cm} cm"


def _gen_organization(faker, original, *, locale):
    return faker.company()


def _gen_job_title(faker, original, *, locale):
    return faker.job()


def _gen_job_department(faker, original, *, locale):
    return faker.random_element(
        (
            "Cardiology",
            "Oncology",
            "Radiology",
            "Emergency",
            "Pediatrics",
            "Neurology",
            "Surgery",
            "Internal Medicine",
            "Dermatology",
            "Orthopedics",
            "Psychiatry",
            "Anesthesiology",
        )
    )


def _gen_occupation(faker, original, *, locale):
    return faker.job()


def _gen_ip_address(faker, original, *, locale):
    return faker.ipv4()


def _gen_mac_address(faker, original, *, locale):
    return faker.mac_address()


def _gen_user_agent(faker, original, *, locale):
    return faker.user_agent()


def _gen_vin(faker, original, *, locale):
    return faker.bothify(
        "?????????????????", letters="ABCDEFGHJKLMNPRSTUVWXYZ0123456789"
    ).upper()


def _gen_vehicle_registration(faker, original, *, locale):
    from openmed.core.pii_i18n import validate_vehicle_registration

    if validate_vehicle_registration(original) and hasattr(
        faker, "indian_vehicle_registration"
    ):
        return _draw_distinct(faker, original, "indian_vehicle_registration")
    return (
        faker.license_plate()
        if hasattr(faker, "license_plate")
        else faker.bothify("???-####")
    )


def _gen_imei(faker, original, *, locale):
    return faker.numerify("###############")


def _gen_ordinal_direction(faker, original, *, locale):
    return faker.random_element(
        (
            "North",
            "South",
            "East",
            "West",
            "Northeast",
            "Northwest",
            "Southeast",
            "Southwest",
        )
    )


# ---------------------------------------------------------------------------
# Default fallback
# ---------------------------------------------------------------------------


def _gen_other(faker, original, *, locale):
    """Last-resort surrogate when no specific generator fits.

    Prefer format-preserving substitution over a random word so the
    surrogate is at least the same shape as the original.
    """
    if original:
        return preserve_id_pattern(original, rng=faker.random)
    return faker.word()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LABEL_GENERATORS: Dict[str, Generator] = {
    L.PERSON: _gen_person,
    L.FIRST_NAME: _gen_first_name,
    L.LAST_NAME: _gen_last_name,
    L.MIDDLE_NAME: _gen_middle_name,
    L.PREFIX: _gen_prefix,
    L.USERNAME: _gen_username,
    L.EMAIL: _gen_email,
    L.PHONE: _gen_phone,
    L.URL: _gen_url,
    L.LOCATION: _gen_location,
    L.STREET_ADDRESS: _gen_street_address,
    L.BUILDING_NUMBER: _gen_building_number,
    L.ZIPCODE: _gen_zipcode,
    L.GPS_COORDINATES: _gen_gps,
    L.ORDINAL_DIRECTION: _gen_ordinal_direction,
    L.DATE: _gen_date,
    L.DATE_OF_BIRTH: _gen_date_of_birth,
    L.TIME: _gen_time,
    L.AGE: _gen_age,
    L.ID_NUM: _gen_id_num,
    L.SSN: _gen_ssn,
    L.ACCOUNT_NUMBER: _gen_account_number,
    L.PASSWORD: _gen_password,
    L.PIN: _gen_pin,
    L.API_KEY: _gen_api_key,
    L.CREDIT_CARD: _gen_credit_card,
    L.CREDIT_CARD_ISSUER: _gen_credit_card_issuer,
    L.CVV: _gen_cvv,
    L.IBAN: _gen_iban,
    L.BIC: _gen_bic,
    L.AMOUNT: _gen_amount,
    L.CURRENCY: _gen_currency,
    L.BITCOIN_ADDRESS: _gen_bitcoin_address,
    L.ETHEREUM_ADDRESS: _gen_ethereum_address,
    L.LITECOIN_ADDRESS: _gen_litecoin_address,
    L.MASKED_NUMBER: _gen_masked_number,
    L.GENDER: _gen_gender,
    L.EYE_COLOR: _gen_eye_color,
    L.HEIGHT: _gen_height,
    L.ORGANIZATION: _gen_organization,
    L.JOB_TITLE: _gen_job_title,
    L.JOB_DEPARTMENT: _gen_job_department,
    L.OCCUPATION: _gen_occupation,
    L.IP_ADDRESS: _gen_ip_address,
    L.MAC_ADDRESS: _gen_mac_address,
    L.USER_AGENT: _gen_user_agent,
    L.VIN: _gen_vin,
    L.VEHICLE_REGISTRATION: _gen_vehicle_registration,
    L.IMEI: _gen_imei,
    L.OTHER: _gen_other,
    "ABHA_NUMBER": _gen_abha_number,
    "ABHA_ADDRESS": _gen_abha_address,
    "AADHAAR": _gen_aadhaar,
    "PAN": _gen_pan,
    "ABDM_HPR_ID": _gen_abdm_hpr_id,
    "ABDM_HFR_ID": _gen_abdm_hfr_id,
    "NG_NIN": _gen_ng_nin,
    "NG_BVN": _gen_ng_bvn,
    "GH_GHANA_CARD": _gen_ghana_card,
    "KE_NATIONAL_ID": _gen_ke_national_id,
    "KE_MAISHA_NAMBA": _gen_ke_maisha_namba,
    "MOBILE_MONEY_PAYBILL": _gen_mobile_money_paybill,
    "MOBILE_MONEY_TILL": _gen_mobile_money_till,
    "MOBILE_MONEY_AGENT": _gen_mobile_money_agent,
    "MOMO_REFERENCE": _gen_momo_reference,
    "FACILITY_ID": _gen_facility_id,
}

LANGUAGE_PACK_GENERATORS: Dict[tuple[str, str, str], Generator] = {}
"""Script-specific generators keyed by ``(pack code, script, label)``."""


def register_label_generator(
    canonical_label: str,
    generator: Generator,
    *,
    language_pack: LanguagePack | None = None,
    script: str | None = None,
) -> None:
    """Register or override a generator for a canonical label.

    Use to extend coverage (new label types) or to swap in a domain-
    specific generator (e.g. project-specific medical record format). Pass a
    ``language_pack`` and one of its ``script`` values to register a
    script-specific generator without changing the global Faker fallback.
    """

    if language_pack is None:
        if script is not None:
            raise ValueError("script requires a language_pack")
        LABEL_GENERATORS[canonical_label] = generator
        return

    if script is None:
        raise ValueError("language_pack generators require a script")
    if script not in language_pack.scripts:
        raise ValueError(
            f"script {script!r} is not declared by language pack {language_pack.code!r}"
        )
    registered_pack = get_language_pack(language_pack.code)
    if registered_pack is None:
        register_language_pack(language_pack)
    elif registered_pack != language_pack:
        raise ValueError(f"language pack {language_pack.code!r} is already registered")
    LANGUAGE_PACK_GENERATORS[(language_pack.code, script, canonical_label)] = generator


def resolve_label_generator(
    canonical_label: str,
    *,
    language_pack: LanguagePack | None,
    script: str,
    source_label: str | None = None,
) -> tuple[Generator, bool]:
    """Resolve a script-aware generator before the global Faker fallback.

    Returns:
        A ``(generator, is_script_specific)`` pair. The flag lets locale
        resolution suppress approximation warnings only when Faker's locale
        data is not being used for the selected provider.
    """

    if language_pack is not None and script in language_pack.scripts:
        generator = LANGUAGE_PACK_GENERATORS.get(
            (language_pack.code, script, canonical_label)
        )
        if generator is not None:
            return generator, True
    fallback_label = source_label or canonical_label
    return (
        LABEL_GENERATORS.get(
            fallback_label,
            LABEL_GENERATORS.get(canonical_label, LABEL_GENERATORS[L.OTHER]),
        ),
        False,
    )


def _register_builtin_script_name_generators() -> None:
    from .providers.script_names import SCRIPT_NAME_PACKS

    name_labels = (L.PERSON, L.FIRST_NAME, L.LAST_NAME, L.MIDDLE_NAME)
    for language_pack, script, generator in SCRIPT_NAME_PACKS:
        for canonical_label in name_labels:
            selected_generator = generator
            if language_pack.code == "zh" and canonical_label != L.MIDDLE_NAME:
                # Preserve the established Chinese surname-aware generators;
                # they already enforce Han-only, shape-correct, disjoint output.
                selected_generator = LABEL_GENERATORS[canonical_label]
            register_label_generator(
                canonical_label,
                selected_generator,
                language_pack=language_pack,
                script=script,
            )


_register_builtin_script_name_generators()


def register_india_label_generators() -> None:
    """Install locale-gated India surrogate generators in the label registry."""

    india_generators = {
        L.PERSON: _gen_india_person,
        L.FIRST_NAME: _gen_india_first_name,
        L.LAST_NAME: _gen_india_last_name,
        L.MIDDLE_NAME: _gen_india_middle_name,
        L.PHONE: _gen_india_phone,
        L.STREET_ADDRESS: _gen_india_street_address,
        L.ZIPCODE: _gen_india_zipcode,
        L.ID_NUM: _gen_india_id_num,
    }
    for canonical_label, generator in india_generators.items():
        register_label_generator(canonical_label, generator)


register_india_label_generators()


__all__ = [
    "Generator",
    "LANGUAGE_PACK_GENERATORS",
    "LABEL_GENERATORS",
    "register_india_label_generators",
    "register_label_generator",
    "resolve_label_generator",
]
