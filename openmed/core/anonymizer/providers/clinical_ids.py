"""Faker providers for clinical / national IDs.

Each provider produces values that pass the corresponding validator in
:mod:`openmed.core.pii_i18n`. We rely on Faker's built-in providers where
the checksum and format already match (verified empirically against our
validators):

  - ``pt_BR.cpf()``               valid
  - ``pt_BR.cnpj()``              valid
  - ``nl_NL.ssn()``  (BSN)        valid
  - ``fr_FR.ssn()``  (NIR)        valid
  - ``it_IT.ssn()``  (Codice Fiscale) valid

Custom providers below cover the gaps where Faker either has no built-in
or emits a US-style format unrelated to the requested locale's actual ID.
They also replace generators whose checksums are valid but not instance-seed
deterministic:

  - German Steuer-ID (Faker's ``de_DE.ssn`` is US-format)
  - Aadhaar with Verhoeff checksum (Faker's ``en_IN.aadhaar_id`` rarely
    passes the official Verhoeff check — only ~1 in 20 by sampling)
  - Spanish NIE (Faker's built-in uses non-instance randomness)
  - Spanish DNI (Faker's ``es_ES`` provider exposes NIE but not DNI)
  - Israeli Teudat Zehut (Faker has no built-in)
  - Indonesian NIK with a decodable embedded birth date
  - Malaysian MyKad / NRIC with a decodable embedded birth date
  - Philippine PhilSys PSN and PhilHealth PIN structural formats
  - Danish CPR / personnummer with a decodable embedded birth date
  - Thai national ID (13 digits with a weighted mod-11 checksum)
  - Polish PESEL, Latvian personas kods, South Korean RRN, and Slovak rodne
    cislo
  - UK NHS Number, a patient health identifier validated with the NHS
    Modulus 11 check
  - UK National Insurance Number (NINO)
  - Generic medical record numbers (MRN-XXXXXXX style)
  - US National Provider Identifier (Luhn over a "80840" prefix)
  - IBAN and SWIFT/BIC financial identifiers with deterministic validation
"""

from __future__ import annotations

import random
import re
from datetime import date
from typing import Sequence

from faker.providers import BaseProvider

from openmed.core.labels import id_subtype_for


def id_subtype_for_entity_type(entity_type: str) -> str | None:
    """Return ID_NUM subtype metadata for regex/checksum entity labels."""

    return id_subtype_for(entity_type)


# ---------------------------------------------------------------------------
# Shared deterministic validators
# ---------------------------------------------------------------------------


def _digits_only(text: str) -> str:
    return re.sub(r"[^0-9]", "", text)


def validate_ssn(ssn_text: str) -> bool:
    """Validate a US SSN's format and basic impossible-number rules."""
    digits = _digits_only(ssn_text)

    if len(digits) != 9:
        return False

    area = digits[0:3]
    group = digits[3:5]
    serial = digits[5:9]

    if area == "000" or area == "666" or area[0] == "9":
        return False
    if group == "00":
        return False
    if serial == "0000":
        return False

    return True


def validate_phone_us(phone_text: str) -> bool:
    """Validate US phone numbers accepted by the PII detector."""
    digits = _digits_only(phone_text)

    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]

    if len(digits) != 10:
        return False

    area_code = digits[0:3]
    exchange = digits[3:6]
    if area_code[0] in "01":
        return False
    if exchange[0] == "0":
        return False
    return True


def validate_luhn(number_text: str) -> bool:
    """Validate a numeric identifier with the Luhn checksum."""
    digits = _digits_only(number_text)

    if len(digits) < 13:
        return False

    body = [int(digit) for digit in digits[:-1]]
    return _luhn_check_digit(body) == int(digits[-1])


def validate_npi(npi_text: str) -> bool:
    """Validate a 10-digit US National Provider Identifier."""
    digits = _digits_only(npi_text)

    if len(digits) != 10:
        return False

    body = [int(digit) for digit in digits[:-1]]
    prefixed = [8, 0, 8, 4, 0, *body]
    return _luhn_check_digit(prefixed) == int(digits[-1])


def generate_luhn_identifier(
    *,
    length: int = 16,
    prefix: str = "",
    rng: random.Random | None = None,
) -> str:
    """Generate a numeric identifier that passes :func:`validate_luhn`."""

    if length < 13:
        raise ValueError("length must be at least 13 for Luhn validation")
    if not prefix.isdigit() and prefix:
        raise ValueError("prefix must contain only digits")
    if len(prefix) >= length:
        raise ValueError("prefix must leave room for body and check digit")

    source = rng or random.Random()
    body = [int(digit) for digit in prefix]
    body.extend(source.randint(0, 9) for _ in range(length - len(prefix) - 1))
    check_digit = _luhn_check_digit(body)
    return "".join(str(digit) for digit in body) + str(check_digit)


def generate_npi(*, rng: random.Random | None = None) -> str:
    """Generate a 10-digit US NPI that passes :func:`validate_npi`."""

    source = rng or random.Random()
    body = [source.randint(0, 9) for _ in range(9)]
    prefixed = [8, 0, 8, 4, 0, *body]
    check_digit = _luhn_check_digit(prefixed)
    return "".join(str(digit) for digit in body) + str(check_digit)


_SPANISH_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
_SPANISH_NIE_PREFIX_VALUES = {"X": "0", "Y": "1", "Z": "2"}


def generate_spanish_nie(*, rng: random.Random | None = None) -> str:
    """Generate a Spanish NIE with a valid modulo-23 check letter."""

    source = rng or random.Random()
    prefix = source.choice(tuple(_SPANISH_NIE_PREFIX_VALUES))
    digits = f"{source.randint(0, 9999999):07d}"
    number = int(_SPANISH_NIE_PREFIX_VALUES[prefix] + digits)
    check = _SPANISH_DNI_LETTERS[number % len(_SPANISH_DNI_LETTERS)]
    return f"{prefix}{digits}{check}"


def generate_ssn(*, rng: random.Random | None = None) -> str:
    """Generate a US SSN-shaped value accepted by :func:`validate_ssn`."""

    source = rng or random.Random()
    area = source.randint(1, 899)
    while area == 666:
        area = source.randint(1, 899)
    group = source.randint(1, 99)
    serial = source.randint(1, 9999)
    return f"{area:03d}-{group:02d}-{serial:04d}"


_IBAN_LENGTHS = {
    "AD": 24,
    "AE": 23,
    "AL": 28,
    "AT": 20,
    "AZ": 28,
    "BA": 20,
    "BE": 16,
    "BG": 22,
    "BH": 22,
    "BR": 29,
    "BY": 28,
    "CH": 21,
    "CR": 22,
    "CY": 28,
    "CZ": 24,
    "DE": 22,
    "DK": 18,
    "DO": 28,
    "EE": 20,
    "EG": 29,
    "ES": 24,
    "FI": 18,
    "FO": 18,
    "FR": 27,
    "GB": 22,
    "GE": 22,
    "GI": 23,
    "GL": 18,
    "GR": 27,
    "GT": 28,
    "HR": 21,
    "HU": 28,
    "IE": 22,
    "IL": 23,
    "IS": 26,
    "IT": 27,
    "JO": 30,
    "KW": 30,
    "KZ": 20,
    "LB": 28,
    "LC": 32,
    "LI": 21,
    "LT": 20,
    "LU": 20,
    "LV": 21,
    "MC": 27,
    "MD": 24,
    "ME": 22,
    "MK": 19,
    "MR": 27,
    "MT": 31,
    "MU": 30,
    "NL": 18,
    "NO": 15,
    "PK": 24,
    "PL": 28,
    "PS": 29,
    "PT": 25,
    "QA": 29,
    "RO": 24,
    "RS": 22,
    "SA": 24,
    "SC": 31,
    "SE": 24,
    "SI": 19,
    "SK": 24,
    "SM": 27,
    "TN": 24,
    "TR": 26,
    "UA": 29,
    "VA": 22,
    "VG": 24,
    "XK": 20,
}


_IBAN_SURROGATE_COUNTRIES = ("GB", "DE", "FR", "ES", "NL", "BE", "IT", "PL")
_BIC_SURROGATE_COUNTRIES = ("GB", "DE", "FR", "ES", "NL", "BE", "IT", "US")
_BIC_COUNTRY_CODES = frozenset(_IBAN_LENGTHS) | {
    "AG",
    "AI",
    "AM",
    "AO",
    "AQ",
    "AR",
    "AS",
    "AU",
    "AW",
    "BB",
    "BD",
    "BF",
    "BI",
    "BJ",
    "BM",
    "BN",
    "BO",
    "BS",
    "BT",
    "BV",
    "BW",
    "BZ",
    "CA",
    "CC",
    "CD",
    "CF",
    "CG",
    "CI",
    "CK",
    "CL",
    "CM",
    "CN",
    "CO",
    "CU",
    "CV",
    "CW",
    "CX",
    "DJ",
    "DM",
    "DZ",
    "ER",
    "ET",
    "FJ",
    "FM",
    "GA",
    "GD",
    "GF",
    "GH",
    "GM",
    "GN",
    "GP",
    "GQ",
    "GW",
    "GY",
    "HM",
    "HK",
    "ID",
    "IN",
    "IO",
    "IQ",
    "IR",
    "JM",
    "JP",
    "KE",
    "KG",
    "KH",
    "KI",
    "KM",
    "KN",
    "KP",
    "KR",
    "LA",
    "LK",
    "LR",
    "LS",
    "LY",
    "MA",
    "MG",
    "MH",
    "ML",
    "MM",
    "MN",
    "MO",
    "MP",
    "MQ",
    "MS",
    "MV",
    "MW",
    "MX",
    "MY",
    "MZ",
    "NA",
    "NC",
    "NE",
    "NF",
    "NG",
    "NI",
    "NP",
    "NR",
    "NU",
    "NZ",
    "OM",
    "PA",
    "PE",
    "PF",
    "PG",
    "PH",
    "PM",
    "PN",
    "PR",
    "PW",
    "PY",
    "RE",
    "RU",
    "RW",
    "SB",
    "SD",
    "SG",
    "SH",
    "SL",
    "SN",
    "SO",
    "SR",
    "SS",
    "ST",
    "SV",
    "SX",
    "SY",
    "SZ",
    "TC",
    "TD",
    "TF",
    "TG",
    "TJ",
    "TK",
    "TM",
    "TO",
    "TT",
    "TV",
    "TW",
    "TZ",
    "UG",
    "UM",
    "US",
    "UY",
    "UZ",
    "VC",
    "VE",
    "VI",
    "VN",
    "VU",
    "WF",
    "WS",
    "YE",
    "YT",
    "ZA",
    "ZM",
    "ZW",
}
_UPPER_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_UPPER_ALNUM = _UPPER_ALPHA + "0123456789"


def _mod97_remainder(value: str) -> int:
    remainder = 0
    for char in value:
        if char.isdigit():
            values = char
        else:
            values = str(ord(char) - 55)
        for digit in values:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder


def _random_alnum(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(_UPPER_ALNUM) for _ in range(length))


def generate_iban(
    *,
    country_code: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic IBAN that passes :func:`validate_iban`."""

    source = rng or random.Random()
    country = (country_code or source.choice(_IBAN_SURROGATE_COUNTRIES)).upper()
    expected_length = _IBAN_LENGTHS.get(country)
    if expected_length is None:
        raise ValueError(f"unsupported IBAN surrogate country: {country!r}")

    bban = _random_alnum(expected_length - 4, source)
    provisional = f"{country}00{bban}"
    check_digits = 98 - _mod97_remainder(provisional[4:] + provisional[:4])
    return f"{country}{check_digits:02d}{bban}"


def validate_iban(iban_text: str) -> bool:
    """Validate an IBAN with the ISO 13616 mod-97 checksum."""
    cleaned = re.sub(r"[\s-]", "", iban_text).upper()

    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", cleaned):
        return False

    expected_length = _IBAN_LENGTHS.get(cleaned[:2])
    if expected_length is not None and len(cleaned) != expected_length:
        return False
    if expected_length is None and not 15 <= len(cleaned) <= 34:
        return False

    rearranged = cleaned[4:] + cleaned[:4]
    return _mod97_remainder(rearranged) == 1


def validate_bic(bic_text: str) -> bool:
    """Validate a SWIFT/BIC code's 8- or 11-character structure."""

    cleaned = re.sub(r"\s", "", bic_text).upper()
    if not re.fullmatch(r"[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?", cleaned):
        return False
    return cleaned[4:6] in _BIC_COUNTRY_CODES


def generate_bic(
    *,
    country_code: str | None = None,
    include_branch: bool | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic SWIFT/BIC accepted by :func:`validate_bic`."""

    source = rng or random.Random()
    country = (country_code or source.choice(_BIC_SURROGATE_COUNTRIES)).upper()
    if not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError("country_code must be a two-letter ISO code")

    bank = "".join(source.choice(_UPPER_ALPHA) for _ in range(4))
    location = _random_alnum(2, source)
    should_include_branch = source.choice((False, True))
    if include_branch is not None:
        should_include_branch = include_branch
    branch = _random_alnum(3, source) if should_include_branch else ""
    return f"{bank}{country}{location}{branch}"


class FinancialIdentifierProvider(BaseProvider):
    """Generates checksum-valid IBANs and structurally valid SWIFT/BIC codes."""

    def financial_iban(self) -> str:
        return generate_iban(rng=self.generator.random)

    def financial_bic(self) -> str:
        return generate_bic(include_branch=True, rng=self.generator.random)


# ---------------------------------------------------------------------------
# Aadhaar (12 digits, Verhoeff checksum)
# ---------------------------------------------------------------------------

# Verhoeff multiplication, permutation and inverse tables, transcribed from
# https://en.wikipedia.org/wiki/Verhoeff_algorithm.
_VERHOEFF_D: Sequence[Sequence[int]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P: Sequence[Sequence[int]] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV: Sequence[int] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def _verhoeff_checksum(digits: Sequence[int]) -> int:
    """Compute the Verhoeff check digit for ``digits`` (without the check)."""
    c = 0
    for i, n in enumerate(reversed(digits), start=1):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][n]]
    return _VERHOEFF_INV[c]


class AadhaarProvider(BaseProvider):
    """Generates 12-digit Aadhaar numbers with valid Verhoeff checksums."""

    def aadhaar(self) -> str:
        # First digit cannot be 0 or 1 per UIDAI spec.
        digits = [self.generator.random.randint(2, 9)]
        digits.extend(self.generator.random.randint(0, 9) for _ in range(10))
        digits.append(_verhoeff_checksum(digits))
        return "".join(str(d) for d in digits)


# ---------------------------------------------------------------------------
# Spanish NIE (prefix + 7 digits + modulo-23 check letter)
# ---------------------------------------------------------------------------


class SpanishNIEProvider(BaseProvider):
    """Generates Spanish NIE values using Faker's instance RNG."""

    def nie(self) -> str:
        return generate_spanish_nie(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Portuguese NIF / NIPC (9 digits with weighted mod-11 checksum)
# ---------------------------------------------------------------------------

# Leading digits and two-digit prefixes the generator may draw from; kept in
# sync with ``validate_portuguese_nif`` so surrogates round-trip.
_PORTUGUESE_NIF_LEADING = ("1", "2", "3", "5", "6", "8", "9")


def generate_portuguese_nif(*, rng: random.Random | None = None) -> str:
    """Generate a Portuguese NIF accepted by :func:`validate_portuguese_nif`."""
    source = rng or random.Random()

    first = source.choice(_PORTUGUESE_NIF_LEADING)
    body = first + "".join(str(source.randint(0, 9)) for _ in range(7))
    total = sum(int(body[index]) * (9 - index) for index in range(8))
    check = 11 - (total % 11)
    if check >= 10:
        check = 0
    return body + str(check)


class PortugueseNIFProvider(BaseProvider):
    """Generates Portuguese NIF values using Faker's instance RNG."""

    def nif(self) -> str:
        return generate_portuguese_nif(rng=self.generator.random)


# ---------------------------------------------------------------------------
# German Steuer-ID (11 digits with mod-11 checksum and digit-frequency rules)
# ---------------------------------------------------------------------------


class GermanSteuerIdProvider(BaseProvider):
    """Generates 11-digit German Steuer-IDs that pass our validator.

    The Steuer-ID rules are subtle enough that we generate by trial and
    delegate to the validator from :mod:`openmed.core.pii_i18n`. Bounded
    retries keep this from looping indefinitely on adversarial random
    states; in practice ~1 in 50 random 11-digit strings satisfies all
    constraints, so we typically succeed in <100 tries.
    """

    _MAX_TRIES = 500

    def german_steuer_id(self) -> str:
        from openmed.core.pii_i18n import validate_german_steuer_id

        rng = self.generator.random
        for _ in range(self._MAX_TRIES):
            # First digit must not be 0
            digits = [rng.randint(1, 9)]
            digits.extend(rng.randint(0, 9) for _ in range(10))
            candidate = "".join(str(d) for d in digits)
            if validate_german_steuer_id(candidate):
                return candidate
        # Fallback: return any 11 digits; validator may reject downstream.
        return self.numerify("###########")


# ---------------------------------------------------------------------------
# Israeli Teudat Zehut (9 digits, alternating 1/2 checksum)
# ---------------------------------------------------------------------------


def _teudat_zehut_check_digit(body_digits: Sequence[int]) -> int:
    """Return the final check digit for the first eight Teudat Zehut digits."""
    if len(body_digits) != 8:
        raise ValueError("Teudat Zehut body must contain exactly eight digits")

    total = 0
    for index, digit in enumerate(body_digits):
        product = digit * (1 if index % 2 == 0 else 2)
        total += product if product < 10 else (product // 10) + (product % 10)
    return (10 - total % 10) % 10


def generate_teudat_zehut(*, rng: random.Random | None = None) -> str:
    """Generate an Israeli Teudat Zehut that passes its checksum validator."""
    source = rng or random.Random()
    body = [source.randint(0, 9) for _ in range(8)]
    if all(digit == 0 for digit in body):
        body[-1] = 1
    body.append(_teudat_zehut_check_digit(body))
    return "".join(str(digit) for digit in body)


class IsraeliTeudatZehutProvider(BaseProvider):
    """Generates 9-digit Israeli Teudat Zehut numbers."""

    def teudat_zehut(self) -> str:
        return generate_teudat_zehut(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Spanish DNI (8 digits + modulo-23 letter)
# ---------------------------------------------------------------------------

_SPANISH_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


class SpanishDNIProvider(BaseProvider):
    """Generates Spanish DNI values that pass the existing validator."""

    def dni(self) -> str:
        number = self.generator.random.randint(0, 99_999_999)
        return f"{number:08d}{_SPANISH_DNI_LETTERS[number % 23]}"


# ---------------------------------------------------------------------------
# Medical Record Number (opaque, but recognizably MRN-shaped)
# ---------------------------------------------------------------------------


class MedicalRecordNumberProvider(BaseProvider):
    """Generates plausible medical record numbers (``MRN-1234567``)."""

    def medical_record_number(self) -> str:
        return f"MRN-{self.numerify('#######')}"


# ---------------------------------------------------------------------------
# US National Provider Identifier (10 digits, Luhn over "80840" prefix)
# ---------------------------------------------------------------------------


def _luhn_check_digit(digits: Sequence[int]) -> int:
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


class NPIProvider(BaseProvider):
    """Generates valid 10-digit US NPI numbers.

    The NPI uses Luhn over the digits prefixed with ``80840``. We generate
    9 random digits, prepend the prefix for checksumming, compute the
    check digit, and emit the original 9 digits + the check digit.
    """

    def npi(self) -> str:
        return generate_npi(rng=self.generator.random)


# ---------------------------------------------------------------------------
# UK NHS Number and National Insurance Number
# ---------------------------------------------------------------------------

_UK_NINO_DISALLOWED_PREFIX_CHARS = frozenset("DFIQUV")
_UK_NINO_DISALLOWED_SECOND_CHARS = _UK_NINO_DISALLOWED_PREFIX_CHARS | {"O"}
_UK_NINO_DISALLOWED_PREFIXES = frozenset({"BG", "GB", "KN", "NK", "NT", "TN", "ZZ"})


def validate_uk_nhs_number(text: str) -> bool:
    """Validate a 10-digit UK NHS Number with the Modulus 11 checksum."""

    digits = _digits_only(text)
    if len(digits) != 10:
        return False

    numbers = [int(digit) for digit in digits]
    total = sum(weight * value for weight, value in zip(range(10, 1, -1), numbers[:9]))
    check = 11 - (total % 11)
    if check == 11:
        check = 0
    if check == 10:
        return False

    return numbers[9] == check


def generate_uk_nhs_number(*, rng: random.Random | None = None) -> str:
    """Generate a UK NHS Number accepted by :func:`validate_uk_nhs_number`."""

    source = rng or random.Random()
    for _ in range(200):
        body_digits = [source.randint(0, 9) for _ in range(9)]
        if all(digit == 0 for digit in body_digits):
            continue

        total = sum(
            weight * value for weight, value in zip(range(10, 1, -1), body_digits)
        )
        check = 11 - (total % 11)
        if check == 11:
            check = 0
        if check == 10:
            continue

        return "".join(str(digit) for digit in body_digits) + str(check)

    return "9434765919"


def validate_uk_nino(text: str) -> bool:
    """Validate a UK National Insurance Number's current structure."""

    cleaned = re.sub(r"[\s-]", "", text).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{6}[A-D]", cleaned):
        return False

    prefix = cleaned[:2]
    if prefix in _UK_NINO_DISALLOWED_PREFIXES:
        return False
    if prefix[0] in _UK_NINO_DISALLOWED_PREFIX_CHARS:
        return False
    if prefix[1] in _UK_NINO_DISALLOWED_SECOND_CHARS:
        return False

    return True


class UKNHSNumberProvider(BaseProvider):
    """Generates valid 10-digit UK NHS Numbers."""

    def nhs_number(self) -> str:
        return generate_uk_nhs_number(rng=self.generator.random)


class UKNINOProvider(BaseProvider):
    """Generates structurally valid UK National Insurance Numbers."""

    _FIRST_LETTERS = tuple(
        letter
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if letter not in _UK_NINO_DISALLOWED_PREFIX_CHARS
    )
    _SECOND_LETTERS = tuple(
        letter
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if letter not in _UK_NINO_DISALLOWED_SECOND_CHARS
    )

    def nino(self) -> str:
        rng = self.generator.random
        while True:
            prefix = rng.choice(self._FIRST_LETTERS) + rng.choice(self._SECOND_LETTERS)
            if prefix not in _UK_NINO_DISALLOWED_PREFIXES:
                break

        digits = f"{rng.randint(0, 999999):06d}"
        suffix = rng.choice("ABCD")
        return f"{prefix} {digits[:2]} {digits[2:4]} {digits[4:]} {suffix}"


# ---------------------------------------------------------------------------
# Polish PESEL (11 digits, weighted checksum with embedded birth date)
# ---------------------------------------------------------------------------


def generate_pesel(*, rng: random.Random | None = None) -> str:
    """Generate a Polish PESEL that passes :func:`validate_polish_pesel`.

    Constructs a PESEL for a random birth date between 1920 and 2029.
    """
    source = rng or random.Random()

    # Random date between 1920-01-01 and 2029-12-31.
    year = source.randint(1920, 2029)
    month = source.randint(1, 12)
    max_day = 28  # safe default
    if month in (1, 3, 5, 7, 8, 10, 12):
        max_day = 31
    elif month in (4, 6, 9, 11):
        max_day = 30
    else:
        import calendar

        max_day = calendar.monthrange(year, month)[1]
    day = source.randint(1, max_day)

    # Encode century offset in the month.
    if year >= 2200:
        encoded_month = month + 60
    elif year >= 2100:
        encoded_month = month + 40
    elif year >= 2000:
        encoded_month = month + 20
    else:
        encoded_month = month

    yy = year % 100
    body_digits = [
        yy // 10,
        yy % 10,
        encoded_month // 10,
        encoded_month % 10,
        day // 10,
        day % 10,
    ]
    # 3-digit serial + 1 sex digit.
    body_digits.extend(source.randint(0, 9) for _ in range(4))

    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(w * d for w, d in zip(weights, body_digits))
    check = (10 - total % 10) % 10

    return "".join(str(d) for d in body_digits) + str(check)


class PolishPeselProvider(BaseProvider):
    """Generates valid Polish PESEL numbers."""

    def pesel(self) -> str:
        return generate_pesel(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Latvian Personas Kods
# ---------------------------------------------------------------------------


def generate_latvian_personas_kods(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic Latvian personas kods accepted by its validator."""
    import calendar

    source = rng or random.Random()

    if source.random() < 0.5:
        # Legacy form: DDMMYY-CNNNQ.
        year = source.randint(1900, 2029)
        month = source.randint(1, 12)
        day = source.randint(1, calendar.monthrange(year, month)[1])
        century = (year - 1800) // 100

        body = [
            day // 10,
            day % 10,
            month // 10,
            month % 10,
            (year % 100) // 10,
            year % 10,
            century,
        ]
        body.extend(source.randint(0, 9) for _ in range(3))
        check = _latvian_personas_kods_check_digit(body)

        digits = "".join(str(digit) for digit in body) + str(check)
        return f"{digits[:6]}-{digits[6:]}"

    # New 32-prefixed format: 32 + 8 random digits + check digit.
    body = [3, 2]
    body.extend(source.randint(0, 9) for _ in range(8))
    check = _latvian_personas_kods_check_digit(body)

    return "".join(str(digit) for digit in body) + str(check)


def _latvian_personas_kods_check_digit(digits: list[int]) -> int:
    weights = (1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    return (
        (1101 - sum(weight * digit for weight, digit in zip(weights, digits))) % 11 % 10
    )


class LatvianPersonasKodsProvider(BaseProvider):
    """Generate synthetic Latvian personas kods values."""

    def personas_kods(self) -> str:
        return generate_latvian_personas_kods(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Bulgarian EGN
# ---------------------------------------------------------------------------


def generate_bulgarian_egn(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic Bulgarian EGN accepted by its validator."""
    import calendar

    source = rng or random.Random()

    year = source.randint(1800, 2099)
    month = source.randint(1, 12)
    day = source.randint(1, calendar.monthrange(year, month)[1])
    if year >= 2000:
        month_code = month + 40
    elif year < 1900:
        month_code = month + 20
    else:
        month_code = month

    body = [
        (year % 100) // 10,
        year % 10,
        month_code // 10,
        month_code % 10,
        day // 10,
        day % 10,
    ]
    body.extend(source.randint(0, 9) for _ in range(3))
    check = _bulgarian_egn_check_digit(body)

    return "".join(str(digit) for digit in body) + str(check)


def _bulgarian_egn_check_digit(digits: list[int]) -> int:
    weights = (2, 4, 8, 5, 10, 9, 7, 3, 6)
    remainder = sum(weight * digit for weight, digit in zip(weights, digits)) % 11
    return 0 if remainder == 10 else remainder


class BulgarianEgnProvider(BaseProvider):
    """Generate synthetic Bulgarian EGN values."""

    def egn(self) -> str:
        return generate_bulgarian_egn(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Serbian / ex-Yugoslav JMBG
# ---------------------------------------------------------------------------


def generate_jmbg(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic JMBG accepted by its validator."""
    import calendar

    source = rng or random.Random()

    year = source.randint(1900, 2029)
    month = source.randint(1, 12)
    day = source.randint(1, calendar.monthrange(year, month)[1])
    region = source.randint(70, 89)
    serial = source.randint(0, 999)

    body = [
        day // 10,
        day % 10,
        month // 10,
        month % 10,
        (year % 1000) // 100,
        (year % 100) // 10,
        year % 10,
        region // 10,
        region % 10,
        serial // 100,
        (serial % 100) // 10,
        serial % 10,
    ]
    check = _jmbg_check_digit(body)

    return "".join(str(digit) for digit in body) + str(check)


def _jmbg_check_digit(digits: list[int]) -> int:
    weights = (7, 6, 5, 4, 3, 2)
    total = sum(
        weight * (digits[index] + digits[index + 6])
        for index, weight in enumerate(weights)
    )
    remainder = 11 - (total % 11)
    return 0 if remainder > 9 else remainder


class SerbianJmbgProvider(BaseProvider):
    """Generate synthetic Serbian / ex-Yugoslav JMBG values."""

    def jmbg(self) -> str:
        return generate_jmbg(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Estonian Isikukood
# ---------------------------------------------------------------------------


def generate_estonian_isikukood(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic Estonian isikukood accepted by its validator."""
    import calendar

    source = rng or random.Random()

    year = source.randint(1800, 2099)
    month = source.randint(1, 12)
    day = source.randint(1, calendar.monthrange(year, month)[1])
    sex = source.randint(0, 1)  # even century digit = female, odd = male
    century_digit = 1 + ((year - 1800) // 100) * 2 + sex

    body = [
        century_digit,
        (year % 100) // 10,
        year % 10,
        month // 10,
        month % 10,
        day // 10,
        day % 10,
    ]
    body.extend(source.randint(0, 9) for _ in range(3))
    check = _estonian_isikukood_check_digit(body)

    return "".join(str(digit) for digit in body) + str(check)


def _estonian_isikukood_check_digit(digits: list[int]) -> int:
    for weights in (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 1),
        (3, 4, 5, 6, 7, 8, 9, 1, 2, 3),
    ):
        remainder = sum(weight * digit for weight, digit in zip(weights, digits)) % 11
        if remainder < 10:
            return remainder
    return 0


class EstonianIsikukoodProvider(BaseProvider):
    """Generate synthetic Estonian isikukood values."""

    def isikukood(self) -> str:
        return generate_estonian_isikukood(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Indonesian NIK (16 digits with province/regency/district + birth date)
# ---------------------------------------------------------------------------


def generate_indonesian_nik(*, rng: random.Random | None = None) -> str:
    """Generate an Indonesian NIK accepted by ``validate_indonesian_nik``."""
    import calendar

    source = rng or random.Random()

    province = source.randint(11, 94)
    regency = source.randint(1, 99)
    district = source.randint(1, 99)

    year = source.randint(1940, 2009)
    month = source.randint(1, 12)
    max_day = calendar.monthrange(year, month)[1]
    day = source.randint(1, max_day)
    if source.choice((False, True)):
        day += 40

    serial = source.randint(1, 9999)
    candidate = (
        f"{province:02d}{regency:02d}{district:02d}"
        f"{day:02d}{month:02d}{year % 100:02d}{serial:04d}"
    )

    from openmed.core.pii_i18n import validate_indonesian_nik

    if validate_indonesian_nik(candidate):
        return candidate
    return "3174051708850001"


class IndonesianNIKProvider(BaseProvider):
    """Generates valid Indonesian NIK numbers."""

    def indonesian_nik(self) -> str:
        return generate_indonesian_nik(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Malaysian MyKad / NRIC (YYMMDD-PB-XXXX with embedded birth date)
# ---------------------------------------------------------------------------


def generate_malaysian_mykad(*, rng: random.Random | None = None) -> str:
    """Generate a Malaysian MyKad accepted by ``validate_malaysian_mykad``."""
    import calendar

    source = rng or random.Random()

    year = source.randint(1940, 2009)
    month = source.randint(1, 12)
    day = source.randint(1, calendar.monthrange(year, month)[1])
    place_code = source.randint(1, 99)
    serial = source.randint(1, 9999)

    candidate = f"{year % 100:02d}{month:02d}{day:02d}-{place_code:02d}-{serial:04d}"

    from openmed.core.pii_i18n import validate_malaysian_mykad

    if validate_malaysian_mykad(candidate):
        return candidate
    return "850817-14-5678"


class MalaysianMyKadProvider(BaseProvider):
    """Generates valid Malaysian MyKad / NRIC numbers."""

    def mykad(self) -> str:
        return generate_malaysian_mykad(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Philippine PhilSys PSN / PhilHealth PIN
# ---------------------------------------------------------------------------


def generate_philsys_psn(*, rng: random.Random | None = None) -> str:
    """Generate a Philippine PhilSys PSN accepted by its structural validator."""
    source = rng or random.Random()

    for _ in range(100):
        digits = "".join(str(source.randint(0, 9)) for _ in range(12))
        candidate = f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"

        from openmed.core.pii_i18n import validate_philsys_psn

        if validate_philsys_psn(candidate):
            return candidate

    return "1234-5678-9012"


def generate_philhealth_pin(*, rng: random.Random | None = None) -> str:
    """Generate a Philippine PhilHealth PIN accepted by its structural validator."""
    source = rng or random.Random()

    for _ in range(100):
        prefix = source.randint(1, 99)
        serial = source.randint(1, 999_999_999)
        suffix = source.randint(0, 9)
        candidate = f"{prefix:02d}-{serial:09d}-{suffix}"

        from openmed.core.pii_i18n import validate_philhealth_pin

        if validate_philhealth_pin(candidate):
            return candidate

    return "98-765432109-8"


class PhilippinesIdProvider(BaseProvider):
    """Generates structurally valid Philippine PhilSys and PhilHealth IDs."""

    def philsys_psn(self) -> str:
        return generate_philsys_psn(rng=self.generator.random)

    def philhealth_pin(self) -> str:
        return generate_philhealth_pin(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Danish CPR / personnummer (DDMMYY-SSSS with century digit)
# ---------------------------------------------------------------------------


def generate_danish_cpr(*, rng: random.Random | None = None) -> str:
    """Generate a Danish CPR accepted by ``validate_danish_cpr``."""
    import calendar

    source = rng or random.Random()

    for _ in range(200):
        year = source.randint(1940, 2009)
        month = source.randint(1, 12)
        day = source.randint(1, calendar.monthrange(year, month)[1])

        if year >= 2000:
            century_digit = source.choice((4, 5, 6, 7, 8, 9))
        else:
            century_digit = source.choice((0, 1, 2, 3))
        serial_tail = source.randint(1, 999)

        candidate = (
            f"{day:02d}{month:02d}{year % 100:02d}-{century_digit}{serial_tail:03d}"
        )

        from openmed.core.pii_i18n import validate_danish_cpr

        if validate_danish_cpr(candidate):
            return candidate

    return "170885-1234"


class DanishCPRProvider(BaseProvider):
    """Generates structurally valid Danish CPR / personnummer values."""

    def danish_cpr(self) -> str:
        return generate_danish_cpr(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Hungarian TAJ (9 digits, alternating 3/7 weighted checksum)
# ---------------------------------------------------------------------------

_RESERVED_HUNGARIAN_TAJ = {"900000007"}


def generate_hungarian_taj(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic TAJ accepted by ``validate_hungarian_taj``."""
    source = rng or random.Random()
    while True:
        body = [source.randint(0, 9) for _ in range(8)]
        if not any(body):
            body[-1] = 1

        total = sum(
            digit * (3 if index % 2 == 0 else 7) for index, digit in enumerate(body)
        )
        candidate = "".join(str(digit) for digit in body) + str(total % 10)
        if candidate in _RESERVED_HUNGARIAN_TAJ:
            continue

        from openmed.core.pii_i18n import validate_hungarian_taj

        if not validate_hungarian_taj(candidate):  # pragma: no cover
            raise RuntimeError("generated Hungarian TAJ failed checksum validation")
        return candidate


class HungarianTAJProvider(BaseProvider):
    """Generates valid synthetic Hungarian TAJ identifiers."""

    def hungarian_taj(self) -> str:
        return generate_hungarian_taj(rng=self.generator.random)


# ---------------------------------------------------------------------------
# South Korean RRN (13 digits, weighted mod-11 with embedded birth date)
# ---------------------------------------------------------------------------


def generate_korean_rrn(*, rng: random.Random | None = None) -> str:
    """Generate a Korean RRN that passes :func:`validate_korean_rrn`.

    Constructs an RRN for a random birth date between 1920 and 2029.
    Gender code 1/2 for 1900s, 3/4 for 2000s.
    """
    source = rng or random.Random()

    year = source.randint(1920, 2029)
    month = source.randint(1, 12)
    if month in (1, 3, 5, 7, 8, 10, 12):
        max_day = 31
    elif month in (4, 6, 9, 11):
        max_day = 30
    else:
        import calendar

        max_day = calendar.monthrange(year, month)[1]
    day = source.randint(1, max_day)

    if year >= 2000:
        gender_code = source.choice((3, 4))
    else:
        gender_code = source.choice((1, 2))

    yy = year % 100
    body_digits = [
        yy // 10,
        yy % 10,
        month // 10,
        month % 10,
        day // 10,
        day % 10,
        gender_code,
    ]
    # 5 more serial digits.
    body_digits.extend(source.randint(0, 9) for _ in range(5))

    weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    total = sum(w * d for w, d in zip(weights, body_digits[:12]))
    check = (11 - total % 11) % 10

    return "".join(str(d) for d in body_digits[:12]) + str(check)


class KoreanRRNProvider(BaseProvider):
    """Generates valid South Korean Resident Registration Numbers."""

    def korean_rrn(self) -> str:
        return generate_korean_rrn(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Czech/Slovak rodne cislo (YYMMDD/XXXX, modulo-11)
# ---------------------------------------------------------------------------


def generate_rodne_cislo(*, rng: random.Random | None = None) -> str:
    """Generate a Czech/Slovak rodne cislo accepted by its checksum validator."""
    import calendar

    source = rng or random.Random()

    for _ in range(500):
        year = source.randint(1954, 2029)
        month = source.randint(1, 12)
        day = source.randint(1, calendar.monthrange(year, month)[1])

        encoded_month = month
        if source.choice((False, True)):
            encoded_month += 50
        elif year >= 2004 and source.randint(0, 9) == 0:
            encoded_month += 20

        yy = year % 100
        serial = source.randint(0, 999)
        first_nine = f"{yy:02d}{encoded_month:02d}{day:02d}{serial:03d}"
        check = (-int(first_nine) * 10) % 11
        if check == 10:
            continue

        candidate = f"{first_nine[:6]}/{first_nine[6:]}{check}"
        from openmed.core.pii_i18n import validate_czechoslovak_rodne_cislo

        if validate_czechoslovak_rodne_cislo(candidate):
            return candidate

    return "850505/1236"


class RodneCisloProvider(BaseProvider):
    """Generates valid Czech/Slovak rodne cislo birth numbers."""

    def rodne_cislo(self) -> str:
        return generate_rodne_cislo(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Thai national ID (13 digits, weighted mod-11 checksum)
# ---------------------------------------------------------------------------


def generate_thai_national_id(*, rng: random.Random | None = None) -> str:
    """Generate a Thai national ID accepted by ``validate_thai_national_id``."""
    source = rng or random.Random()
    body_digits = [source.randint(1, 9)]
    body_digits.extend(source.randint(0, 9) for _ in range(11))

    total = sum(weight * value for weight, value in zip(range(13, 1, -1), body_digits))
    check = (11 - total % 11) % 10

    return "".join(str(digit) for digit in body_digits) + str(check)


class ThaiNationalIdProvider(BaseProvider):
    """Generates valid 13-digit Thai national IDs."""

    def thai_national_id(self) -> str:
        return generate_thai_national_id(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Romanian CNP (13 digits, weighted mod-11 checksum with embedded birth date)
# ---------------------------------------------------------------------------


def generate_romanian_cnp(*, rng: random.Random | None = None) -> str:
    """Generate a Romanian CNP that passes :func:`validate_romanian_cnp`.

    Builds a 13-digit ``S YY MM DD JJ NNN C`` identifier for a random,
    non-future birth date and location code, then appends the documented
    ``279146358279`` weighted modulo-11 control digit (a remainder of 10 maps
    to control digit 1).
    """
    source = rng or random.Random()

    today = date.today()
    documented_ranges = {
        1: (date(1900, 1, 1), date(1999, 12, 31)),
        2: (date(1900, 1, 1), date(1999, 12, 31)),
        3: (date(1800, 1, 1), date(1899, 12, 31)),
        4: (date(1800, 1, 1), date(1899, 12, 31)),
        5: (date(2000, 1, 1), date(2099, 12, 31)),
        6: (date(2000, 1, 1), date(2099, 12, 31)),
    }
    available_ranges = {
        code: (start, min(end, today))
        for code, (start, end) in documented_ranges.items()
        if start <= today
    }
    gender_code = source.choice(tuple(available_ranges))
    start, end = available_ranges[gender_code]
    birth_date = date.fromordinal(source.randint(start.toordinal(), end.toordinal()))

    # Legacy county/sector codes 01-46 and 51-52, or current SIIEASC code 70.
    county = source.choice(tuple(range(1, 47)) + (51, 52, 70))
    serial = source.randint(1, 999)  # non-zero sequence number

    body = (
        f"{gender_code}{birth_date.year % 100:02d}{birth_date.month:02d}"
        f"{birth_date.day:02d}{county:02d}{serial:03d}"
    )
    weights = (2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9)
    total = sum(weight * int(digit) for weight, digit in zip(weights, body))
    control = total % 11
    if control == 10:
        control = 1

    return body + str(control)


class RomanianCNPProvider(BaseProvider):
    """Generates valid Romanian CNP (Cod Numeric Personal) identifiers."""

    def romanian_cnp(self) -> str:
        return generate_romanian_cnp(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Canadian Social Insurance Number and provincial health-card numbers
#
# These identify Canadian residents and patients:
#   - SIN: nine-digit Social Insurance Number with a Luhn (mod-10) check.
#   - Ontario health card: ten-digit number with a Luhn check, optionally
#     suffixed by a two-letter version code (a health identifier).
#   - British Columbia Personal Health Number (PHN): ten digits beginning
#     with ``9`` and validated with a weighted modulus-11 check (a health
#     identifier).
# ---------------------------------------------------------------------------

# SIN and BC PHN validators accept only their plain digit form or the canonical
# consistently spaced/hyphenated grouping used on forms and cards.
_CANADIAN_SIN_RE = re.compile(
    r"^(?:\d{9}|\d{3}(?P<sin_sep>[ -])\d{3}(?P=sin_sep)\d{3})$"
)
_BC_PHN_RE = re.compile(r"^(?:9\d{9}|9\d{3}(?P<bc_sep>[ -])\d{3}(?P=bc_sep)\d{3})$")

# Ontario health-card numbers start with 1-9, contain ten digits, and may be
# grouped 4-3-3. The official HCV schema permits a one- or two-letter version
# code only at the end.
_ONTARIO_HEALTH_CARD_RE = re.compile(
    r"^(?P<number>[1-9]\d{3}(?P<ontario_sep>[ -]?)\d{3}"
    r"(?P=ontario_sep)\d{3})(?:[ -]?(?P<version>[A-Za-z]{1,2}))?$"
)

# British Columbia PHN weights applied to digits two through nine.
_BC_PHN_WEIGHTS = (2, 4, 8, 5, 10, 9, 7, 3)


def _split_ontario_health_card(text: str) -> tuple[str, str]:
    """Return the ``(digits, version_code)`` parts of an Ontario health card."""

    match = _ONTARIO_HEALTH_CARD_RE.fullmatch(text.strip())
    if match is None:
        return "", ""
    digits = _digits_only(match.group("number"))
    version = (match.group("version") or "").upper()
    return digits, version


def validate_canadian_sin(text: str) -> bool:
    """Validate a nine-digit Canadian Social Insurance Number (SIN).

    The SIN carries a Luhn (mod-10) check digit over all nine digits. A SIN
    starting with ``0`` is not assigned to a person, so it is rejected.

    Args:
        text: Candidate SIN, optionally spaced or hyphenated (``NNN-NNN-NNN``).

    Returns:
        ``True`` when ``text`` is a Luhn-valid, non-zero-prefixed SIN.
    """

    candidate = text.strip()
    if _CANADIAN_SIN_RE.fullmatch(candidate) is None:
        return False
    digits = _digits_only(candidate)
    if len(digits) != 9:
        return False
    if digits[0] == "0":
        return False

    body = [int(digit) for digit in digits[:-1]]
    return _luhn_check_digit(body) == int(digits[-1])


def generate_canadian_sin(*, rng: random.Random | None = None) -> str:
    """Generate a Canadian SIN accepted by :func:`validate_canadian_sin`."""

    source = rng or random.Random()
    body = [source.randint(1, 9)]
    body.extend(source.randint(0, 9) for _ in range(7))
    check_digit = _luhn_check_digit(body)
    return "".join(str(digit) for digit in body) + str(check_digit)


def validate_ontario_health_card(text: str) -> bool:
    """Validate an Ontario (OHIP) health-card number.

    The core is a ten-digit number carrying a Luhn (mod-10) check digit. An
    optional one- or two-letter version code may follow the ten digits. Ontario
    health cards are health identifiers.

    Args:
        text: Candidate health card, optionally spaced or hyphenated and
            optionally suffixed by a two-letter version code.

    Returns:
        ``True`` when the ten-digit core is Luhn-valid and any version code is
        well formed.
    """

    digits, version = _split_ontario_health_card(text)
    if len(digits) != 10:
        return False
    body = [int(digit) for digit in digits[:-1]]
    return _luhn_check_digit(body) == int(digits[-1])


def generate_ontario_health_card(
    *,
    rng: random.Random | None = None,
    with_version: bool = True,
) -> str:
    """Generate an Ontario health card accepted by the validator.

    Args:
        rng: Optional deterministic random source.
        with_version: When ``True``, append a synthetic two-letter version code.

    Returns:
        A ten-digit Luhn-valid Ontario health-card number, optionally suffixed
        with a two-letter version code.
    """

    source = rng or random.Random()
    body = [source.randint(1, 9)]
    body.extend(source.randint(0, 9) for _ in range(8))
    check_digit = _luhn_check_digit(body)
    number = "".join(str(digit) for digit in body) + str(check_digit)
    if not with_version:
        return number
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    version = source.choice(letters) + source.choice(letters)
    return f"{number}-{version}"


def validate_bc_phn(text: str) -> bool:
    """Validate a British Columbia Personal Health Number (PHN).

    The PHN is ten digits beginning with ``9``. Digits two through nine are
    weighted by ``(2, 4, 8, 5, 10, 9, 7, 3)``; the check digit is
    ``11 - (sum mod 11)``. A remainder that yields a check of ``10`` or ``11``
    marks an unissued number and is rejected. BC PHNs are health identifiers.

    Args:
        text: Candidate PHN, optionally spaced or hyphenated.

    Returns:
        ``True`` when ``text`` is a valid, issued BC PHN.
    """

    candidate = text.strip()
    if _BC_PHN_RE.fullmatch(candidate) is None:
        return False
    digits = _digits_only(candidate)
    if len(digits) != 10:
        return False

    numbers = [int(digit) for digit in digits]
    if numbers[0] != 9:
        return False

    total = sum(weight * value for weight, value in zip(_BC_PHN_WEIGHTS, numbers[1:9]))
    check = 11 - (total % 11)
    if check in (10, 11):
        return False

    return numbers[9] == check


def generate_bc_phn(*, rng: random.Random | None = None) -> str:
    """Generate a BC PHN accepted by :func:`validate_bc_phn`."""

    source = rng or random.Random()
    for _ in range(200):
        body_digits = [source.randint(0, 9) for _ in range(8)]
        total = sum(
            weight * value for weight, value in zip(_BC_PHN_WEIGHTS, body_digits)
        )
        check = 11 - (total % 11)
        if check in (10, 11):
            continue
        return "9" + "".join(str(digit) for digit in body_digits) + str(check)

    return "9999999998"


class CanadianSINProvider(BaseProvider):
    """Generates valid nine-digit Canadian Social Insurance Numbers."""

    def canadian_sin(self) -> str:
        return generate_canadian_sin(rng=self.generator.random)


class OntarioHealthCardProvider(BaseProvider):
    """Generates valid Ontario health-card numbers with a version code."""

    def ontario_health_card(self) -> str:
        return generate_ontario_health_card(rng=self.generator.random)


class BCPHNProvider(BaseProvider):
    """Generates valid British Columbia Personal Health Numbers."""

    def bc_phn(self) -> str:
        return generate_bc_phn(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Australian Medicare card number (10 digits, weighted checksum + issue digit)
# and Tax File Number (TFN, 9-digit weighted mod-11)
# ---------------------------------------------------------------------------

# Weights applied to the first eight Medicare digits before the mod-10 check
# digit (the ninth digit). The tenth digit is the card issue number and is not
# part of the checksum; a person's separate IRN may follow the full card number.
_MEDICARE_WEIGHTS: Sequence[int] = (1, 3, 7, 9, 1, 3, 7, 9)

# ATO Tax File Number weighting factors for the published nine-digit format.
_TFN_WEIGHTS_9: Sequence[int] = (1, 4, 3, 7, 5, 8, 6, 9, 10)

# Medicare accepts the 10-digit card in plain or printed 4-5-1 grouping. A
# separate one-digit Individual Reference Number (IRN) may follow contiguously,
# after a space/hyphen, or after a slash. TFNs accept only their plain or
# canonically spaced 3-3-3 form.
_AUSTRALIAN_MEDICARE_RE = re.compile(
    r"^(?P<card>[2-6]\d{9}|[2-6]\d{3} \d{5} \d)"
    r"(?:(?:[ ]*/[ ]*|[ -]?)(?P<irn>[1-9]))?$"
)
_AUSTRALIAN_TFN_RE = re.compile(r"^(?:\d{9}|\d{3} \d{3} \d{3})$")


def validate_australian_medicare(text: str) -> bool:
    """Validate an Australian Medicare card number.

    Medicare card numbers are ten digits. The first eight digits are weighted
    by ``1, 3, 7, 9, 1, 3, 7, 9`` and summed modulo 10; that remainder must
    equal the ninth digit (the check digit). The tenth digit is the card issue
    number and does not participate in the checksum. A separate one-digit
    Individual Reference Number (IRN) may follow the card number. The leading
    digit is constrained to ``2``-``6`` by Services Australia's published
    numbering scheme.

    This is a health identifier under HIPAA cross-mapping: it identifies an
    individual's enrolment in the Australian Medicare scheme.

    Args:
        text: Medicare card number, with or without printed spaces
            (``NNNN NNNNN N``), optionally followed by a one-digit IRN.

    Returns:
        True when the value has a valid Medicare shape and checksum.
    """

    match = _AUSTRALIAN_MEDICARE_RE.fullmatch(text.strip())
    if match is None:
        return False
    digits = _digits_only(match.group("card"))

    numbers = [int(digit) for digit in digits]
    total = sum(weight * value for weight, value in zip(_MEDICARE_WEIGHTS, numbers[:8]))
    return total % 10 == numbers[8]


def generate_australian_medicare(*, rng: random.Random | None = None) -> str:
    """Generate a Medicare number accepted by :func:`validate_australian_medicare`.

    Returns the digits-only ``NNNNNNNNNN`` form; the ninth digit is a valid
    checksum and the tenth is a non-zero issue number.
    """

    source = rng or random.Random()
    body = [source.randint(2, 6)]
    body.extend(source.randint(0, 9) for _ in range(7))
    check = sum(weight * value for weight, value in zip(_MEDICARE_WEIGHTS, body)) % 10
    issue = source.randint(1, 9)
    return "".join(str(digit) for digit in body) + str(check) + str(issue)


def _tfn_checksum_ok(digits: str) -> bool:
    if len(digits) != 9 or digits == "000000000":
        return False
    numbers = [int(digit) for digit in digits]
    total = sum(weight * value for weight, value in zip(_TFN_WEIGHTS_9, numbers))
    return total % 11 == 0


def validate_australian_tfn(text: str) -> bool:
    """Validate an Australian Tax File Number (TFN).

    A TFN is nine digits guarded by a weighted modulus-11 checksum. The weights
    are ``1, 4, 3, 7, 5, 8, 6, 9, 10`` and the weighted sum must be divisible
    by 11. The all-zero value is not a valid issued TFN.

    Args:
        text: TFN string, with or without spaces (``NNN NNN NNN``).

    Returns:
        True when the value has a valid TFN length and checksum.
    """

    candidate = text.strip()
    if _AUSTRALIAN_TFN_RE.fullmatch(candidate) is None:
        return False
    digits = _digits_only(candidate)
    return _tfn_checksum_ok(digits)


def generate_australian_tfn(*, rng: random.Random | None = None) -> str:
    """Generate a nine-digit TFN accepted by :func:`validate_australian_tfn`."""

    source = rng or random.Random()
    for _ in range(200):
        body = [source.randint(0, 9) for _ in range(9)]
        if any(body):
            digits = "".join(str(digit) for digit in body)
            if _tfn_checksum_ok(digits):
                return digits

    return "123456782"


class AustralianMedicareProvider(BaseProvider):
    """Generates valid 10-digit Australian Medicare card numbers."""

    def australian_medicare(self) -> str:
        return generate_australian_medicare(rng=self.generator.random)


class AustralianTFNProvider(BaseProvider):
    """Generates valid Australian Tax File Numbers."""

    def australian_tfn(self) -> str:
        return generate_australian_tfn(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Bulk registration helper
# ---------------------------------------------------------------------------


class MrzProvider(BaseProvider):
    """Faker provider for ICAO 9303 passport/ID MRZ surrogates."""

    def passport_mrz(self) -> str:
        from openmed.core.pii_i18n import generate_mrz_td3

        return generate_mrz_td3(self.generator.random)


def generate_unified_social_credit_code(*, rng: random.Random | None = None) -> str:
    """Generate a checksum-valid China Unified Social Credit Code surrogate.

    The 18-character result uses only the permitted 31-character alphabet (never
    the excluded letters I/O/S/V/Z), carries a 6-digit administrative-region
    segment, and always passes ``validate_unified_social_credit_code`` by
    construction.
    """
    from openmed.core.pii_i18n import (
        USCC_ALPHABET,
        USCC_DEPARTMENT_CATEGORY_CODES,
        uscc_check_char,
    )

    source = rng or random.Random()
    department = source.choice(tuple(USCC_DEPARTMENT_CATEGORY_CODES))
    category = source.choice(tuple(sorted(USCC_DEPARTMENT_CATEGORY_CODES[department])))
    region = "".join(str(source.randint(0, 9)) for _ in range(6))
    organization = "".join(source.choice(USCC_ALPHABET) for _ in range(9))
    body = f"{department}{category}{region}{organization}"
    return body + uscc_check_char(body)


class UnifiedSocialCreditCodeProvider(BaseProvider):
    """Faker provider for China Unified Social Credit Code surrogates."""

    def unified_social_credit_code(self) -> str:
        """Return a checksum-valid synthetic Unified Social Credit Code."""

        return generate_unified_social_credit_code(rng=self.generator.random)


def register_clinical_providers(faker) -> None:
    """Add every custom provider in this module to ``faker``."""
    from .registry_ids import national_id_faker_provider_classes

    for provider in national_id_faker_provider_classes():
        faker.add_provider(provider)
    faker.add_provider(MedicalRecordNumberProvider)
    faker.add_provider(FinancialIdentifierProvider)
    faker.add_provider(MrzProvider)
    faker.add_provider(UnifiedSocialCreditCodeProvider)


__all__ = [
    "AadhaarProvider",
    "AustralianMedicareProvider",
    "AustralianTFNProvider",
    "BCPHNProvider",
    "BulgarianEgnProvider",
    "CanadianSINProvider",
    "DanishCPRProvider",
    "EstonianIsikukoodProvider",
    "FinancialIdentifierProvider",
    "GermanSteuerIdProvider",
    "HungarianTAJProvider",
    "IndonesianNIKProvider",
    "IsraeliTeudatZehutProvider",
    "KoreanRRNProvider",
    "LatvianPersonasKodsProvider",
    "MalaysianMyKadProvider",
    "MedicalRecordNumberProvider",
    "MrzProvider",
    "NPIProvider",
    "UnifiedSocialCreditCodeProvider",
    "PhilippinesIdProvider",
    "PolishPeselProvider",
    "RomanianCNPProvider",
    "RodneCisloProvider",
    "SerbianJmbgProvider",
    "ThaiNationalIdProvider",
    "SpanishDNIProvider",
    "PortugueseNIFProvider",
    "SpanishNIEProvider",
    "UKNHSNumberProvider",
    "UKNINOProvider",
    "generate_australian_medicare",
    "generate_australian_tfn",
    "generate_bc_phn",
    "generate_bic",
    "generate_bulgarian_egn",
    "generate_canadian_sin",
    "generate_danish_cpr",
    "generate_hungarian_taj",
    "generate_estonian_isikukood",
    "generate_iban",
    "generate_ontario_health_card",
    "generate_indonesian_nik",
    "generate_jmbg",
    "generate_teudat_zehut",
    "generate_korean_rrn",
    "generate_luhn_identifier",
    "generate_npi",
    "generate_pesel",
    "generate_latvian_personas_kods",
    "generate_malaysian_mykad",
    "generate_philhealth_pin",
    "generate_philsys_psn",
    "generate_rodne_cislo",
    "generate_romanian_cnp",
    "generate_portuguese_nif",
    "generate_spanish_nie",
    "generate_ssn",
    "generate_thai_national_id",
    "generate_uk_nhs_number",
    "generate_unified_social_credit_code",
    "id_subtype_for_entity_type",
    "register_clinical_providers",
    "validate_australian_medicare",
    "validate_australian_tfn",
    "validate_bc_phn",
    "validate_bic",
    "validate_canadian_sin",
    "validate_iban",
    "validate_luhn",
    "validate_npi",
    "validate_ontario_health_card",
    "validate_phone_us",
    "validate_ssn",
    "validate_uk_nhs_number",
    "validate_uk_nino",
]
