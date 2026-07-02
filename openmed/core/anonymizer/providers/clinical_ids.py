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
  - Thai national ID (13 digits with a weighted mod-11 checksum)
  - Polish PESEL, Latvian personas kods, South Korean RRN, and Slovak rodne
    cislo
  - UK NHS Number, a patient health identifier validated with the NHS
    Modulus 11 check
  - UK National Insurance Number (NINO)
  - Generic medical record numbers (MRN-XXXXXXX style)
  - US National Provider Identifier (Luhn over a "80840" prefix)
"""

from __future__ import annotations

import random
import re
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
    remainder = 0
    for char in rearranged:
        if char.isdigit():
            values = char
        else:
            values = str(ord(char) - 55)
        for digit in values:
            remainder = (remainder * 10 + int(digit)) % 97

    return remainder == 1


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
    from datetime import date

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
# Slovak rodne cislo (YYMMDD/XXXX, modulo-11)
# ---------------------------------------------------------------------------


def generate_rodne_cislo(*, rng: random.Random | None = None) -> str:
    """Generate a Slovak rodne cislo accepted by its checksum validator."""
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
    """Generates valid Slovak rodne cislo birth numbers."""

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
# Bulk registration helper
# ---------------------------------------------------------------------------


def register_clinical_providers(faker) -> None:
    """Add every custom provider in this module to ``faker``."""
    from .registry_ids import national_id_faker_provider_classes

    for provider in national_id_faker_provider_classes():
        faker.add_provider(provider)
    faker.add_provider(MedicalRecordNumberProvider)


__all__ = [
    "AadhaarProvider",
    "GermanSteuerIdProvider",
    "IndonesianNIKProvider",
    "IsraeliTeudatZehutProvider",
    "KoreanRRNProvider",
    "LatvianPersonasKodsProvider",
    "MedicalRecordNumberProvider",
    "NPIProvider",
    "PolishPeselProvider",
    "RodneCisloProvider",
    "ThaiNationalIdProvider",
    "SpanishDNIProvider",
    "SpanishNIEProvider",
    "UKNHSNumberProvider",
    "UKNINOProvider",
    "generate_indonesian_nik",
    "generate_teudat_zehut",
    "generate_korean_rrn",
    "generate_luhn_identifier",
    "generate_npi",
    "generate_pesel",
    "generate_latvian_personas_kods",
    "generate_rodne_cislo",
    "generate_spanish_nie",
    "generate_ssn",
    "generate_thai_national_id",
    "generate_uk_nhs_number",
    "id_subtype_for_entity_type",
    "register_clinical_providers",
    "validate_iban",
    "validate_luhn",
    "validate_npi",
    "validate_phone_us",
    "validate_ssn",
    "validate_uk_nhs_number",
    "validate_uk_nino",
]
