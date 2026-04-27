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

from typing import Callable, Dict

from .. import labels as L
from .format_preserve import (
    preserve_date_format,
    preserve_email_pattern,
    preserve_id_pattern,
    preserve_phone_format,
)


Generator = Callable[..., str]
"""Signature: ``(faker, original: str, *, locale: str) -> str``."""


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------

def _gen_person(faker, original, *, locale):
    return faker.name()


def _gen_first_name(faker, original, *, locale):
    return faker.first_name()


def _gen_last_name(faker, original, *, locale):
    return faker.last_name()


def _gen_middle_name(faker, original, *, locale):
    return faker.first_name()


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
    if any(ch.isdigit() for ch in original):
        return preserve_phone_format(original, rng=faker.random)
    return faker.phone_number()


def _gen_url(faker, original, *, locale):
    return faker.url()


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------

def _gen_location(faker, original, *, locale):
    # Prefer city-level granularity since most "LOCATION" detections are cities
    return faker.city()


def _gen_street_address(faker, original, *, locale):
    return faker.street_address()


def _gen_building_number(faker, original, *, locale):
    return faker.building_number()


def _gen_zipcode(faker, original, *, locale):
    if any(ch.isdigit() or ch.isalpha() for ch in original):
        return preserve_id_pattern(original, rng=faker.random)
    return faker.postcode()


def _gen_gps(faker, original, *, locale):
    lat, lon = faker.latitude(), faker.longitude()
    return f"{float(lat):.6f}, {float(lon):.6f}"


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

# Day-first locales — same set as openmed.core.pii._DAY_FIRST_LANGS but
# expressed in Faker locale terms.
_DAY_FIRST_LOCALES = frozenset({
    "fr_FR", "de_DE", "it_IT", "es_ES", "nl_NL",
    "hi_IN", "en_IN", "pt_PT", "pt_BR",
})


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
    "pt_BR": "cpf",
    "pt_PT": "vat_id",
    "fr_FR": "ssn",
    "it_IT": "ssn",
    "es_ES": "nie",
    "nl_NL": "ssn",
    "en_IN": "aadhaar",
    "hi_IN": "aadhaar",
    "de_DE": "german_steuer_id",
    "en_US": "ssn",
    "en_GB": "ssn",
}


def _gen_id_num(faker, original, *, locale):
    method = _LOCALE_ID_METHODS.get(locale)
    if method and hasattr(faker, method):
        return getattr(faker, method)()
    return preserve_id_pattern(original, rng=faker.random)


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
    return faker.credit_card_number()


def _gen_credit_card_issuer(faker, original, *, locale):
    return faker.credit_card_provider()


def _gen_cvv(faker, original, *, locale):
    return faker.credit_card_security_code()


def _gen_iban(faker, original, *, locale):
    return faker.iban()


def _gen_bic(faker, original, *, locale):
    return faker.swift11() if hasattr(faker, "swift11") else faker.bothify("########XXX")


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
    return preserve_id_pattern(original, rng=faker.random) if original else faker.numerify("****-****-####")


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
    return faker.random_element((
        "Cardiology", "Oncology", "Radiology", "Emergency", "Pediatrics",
        "Neurology", "Surgery", "Internal Medicine", "Dermatology",
        "Orthopedics", "Psychiatry", "Anesthesiology",
    ))


def _gen_occupation(faker, original, *, locale):
    return faker.job()


def _gen_ip_address(faker, original, *, locale):
    return faker.ipv4()


def _gen_mac_address(faker, original, *, locale):
    return faker.mac_address()


def _gen_user_agent(faker, original, *, locale):
    return faker.user_agent()


def _gen_vin(faker, original, *, locale):
    return faker.bothify("?????????????????", letters="ABCDEFGHJKLMNPRSTUVWXYZ0123456789").upper()


def _gen_vehicle_registration(faker, original, *, locale):
    return faker.license_plate() if hasattr(faker, "license_plate") else faker.bothify("???-####")


def _gen_imei(faker, original, *, locale):
    return faker.numerify("###############")


def _gen_ordinal_direction(faker, original, *, locale):
    return faker.random_element(("North", "South", "East", "West",
                                 "Northeast", "Northwest", "Southeast", "Southwest"))


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
}


def register_label_generator(canonical_label: str, generator: Generator) -> None:
    """Register or override a generator for a canonical label.

    Use to extend coverage (new label types) or to swap in a domain-
    specific generator (e.g. project-specific medical record format).
    """
    LABEL_GENERATORS[canonical_label] = generator


__all__ = [
    "Generator",
    "LABEL_GENERATORS",
    "register_label_generator",
]
