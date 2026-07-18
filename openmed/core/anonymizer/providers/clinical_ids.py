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
  - ABDM identifiers: 14-digit ABHA numbers, ABHA addresses,
    PAN-shaped tax identifiers, and synthetic HPR/HFR registry identifiers
  - Indian PIN codes, mobile numbers, PAN, GSTIN, and ABHA identifiers
  - Spanish NIE (Faker's built-in uses non-instance randomness)
  - Spanish DNI (Faker's ``es_ES`` provider exposes NIE but not DNI)
  - Chinese Resident Identity Card (18 characters with MOD 11-2 checksum)
  - Israeli Teudat Zehut (Faker has no built-in)
  - Indonesian NIK with a decodable embedded birth date
  - East African national IDs for Tanzania, Uganda, Rwanda, and Ethiopia
  - Malaysian MyKad / NRIC with a decodable embedded birth date
  - Philippine PhilSys PSN and PhilHealth PIN structural formats
  - Danish CPR / personnummer with a decodable embedded birth date
  - Thai national ID (13 digits with a weighted mod-11 checksum)
  - Nigerian NIN/BVN values and mobile numbers with prefix-class preservation
  - Ghana Card PIN and Kenyan legacy/Maisha identity numbers
  - South African ID (13 digits with an embedded birth date and Luhn checksum)
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


def _matches_mobile_prefix(number: str, prefix: str) -> bool:
    if len(number) < len(prefix):
        return False
    return all(
        expected == "x" or expected == actual
        for actual, expected in zip(number, prefix)
    )


def _match_african_mobile_phone(phone_text: str) -> tuple[int, int] | None:
    """Return preserved digit count and total digit count for a known plan."""

    if not isinstance(phone_text, str):
        return None
    stripped = phone_text.strip()
    if not stripped or re.fullmatch(r"[+0-9\s.-]+", stripped) is None:
        return None

    from openmed.core.pii_i18n import AFRICAN_MOBILE_PLANS

    digits = _digits_only(stripped)
    candidates: list[tuple[int, int]] = []
    for plan in AFRICAN_MOBILE_PLANS.values():
        if stripped.startswith("+"):
            international_prefix = plan.country_code
            if not digits.startswith(international_prefix):
                continue
            nsn = digits[len(international_prefix) :]
            rendering_prefix_length = len(international_prefix)
        elif stripped.startswith("00"):
            international_prefix = f"00{plan.country_code}"
            if not digits.startswith(international_prefix):
                continue
            nsn = digits[len(international_prefix) :]
            rendering_prefix_length = len(international_prefix)
        elif stripped.startswith("0") and not stripped.startswith("00"):
            nsn = digits[1:]
            rendering_prefix_length = 1
        else:
            continue

        if len(nsn) != plan.nsn_length:
            continue
        for prefix in plan.mobile_prefixes:
            if _matches_mobile_prefix(nsn, prefix):
                candidates.append(
                    (
                        rendering_prefix_length + len(prefix),
                        len(digits),
                    )
                )

    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])


def _different_phone_digit(original: str, *, rng: random.Random) -> str:
    candidate = rng.randint(0, 8)
    if candidate >= int(original):
        candidate += 1
    return str(candidate)


def generate_african_phone(
    original: str,
    *,
    rng: random.Random | None = None,
) -> str | None:
    """Generate a prefix-preserving surrogate for a registered African mobile.

    Country/trunk digits, the matched operator-prefix class, separators, and
    total length stay unchanged. Every subscriber digit is replaced, so a
    successful surrogate can never equal its input.

    Args:
        original: Candidate E.164, ``00``-international, or national number.
        rng: Optional seeded random source for deterministic generation.

    Returns:
        A non-identical surrogate, or ``None`` when no registered plan matches.
    """

    match = _match_african_mobile_phone(original)
    if match is None:
        return None
    preserved_digits, _total_digits = match
    source = rng or random.Random()

    digit_index = 0
    surrogate: list[str] = []
    for char in original:
        if not char.isdigit():
            surrogate.append(char)
            continue
        if digit_index < preserved_digits:
            surrogate.append(char)
        else:
            surrogate.append(_different_phone_digit(char, rng=source))
        digit_index += 1
    return "".join(surrogate)


class AfricanPhoneProvider(BaseProvider):
    """Faker provider for operator-prefix-preserving African mobile phones."""

    def african_phone(self, original: str) -> str | None:
        """Return a deterministic surrogate for a registered mobile number."""

        return generate_african_phone(original, rng=self.generator.random)


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


def generate_aadhaar(*, rng: random.Random | None = None) -> str:
    """Generate a 12-digit Aadhaar value with a valid Verhoeff checksum."""

    source = rng or random.Random()
    digits = [source.randint(2, 9)]
    digits.extend(source.randint(0, 9) for _ in range(10))
    digits.append(_verhoeff_checksum(digits))
    return "".join(str(digit) for digit in digits)


class AadhaarProvider(BaseProvider):
    """Generates 12-digit Aadhaar numbers with valid Verhoeff checksums."""

    def aadhaar(self) -> str:
        return generate_aadhaar(rng=self.generator.random)


# ---------------------------------------------------------------------------
# Pakistani CNIC
# ---------------------------------------------------------------------------
class PakistaniCnicProvider(BaseProvider):
    """Generate structurally valid synthetic Pakistani CNIC values."""

    def cnic(self) -> str:
        """Return a dashed 13-digit synthetic CNIC."""
        province = self.random_int(1, 7)
        district_tehsil = f"{self.random_int(1000, 9999):04d}"
        region = f"{province}{district_tehsil}"
        family = f"{self.random_int(0, 9999999):07d}"
        sex_digit = self.random_int(0, 9)
        return f"{region}-{family}-{sex_digit}"

    def cnic_undashed(self) -> str:
        """Return an undashed 13-digit synthetic CNIC."""
        return self.cnic().replace("-", "")


# ---------------------------------------------------------------------------
# India ABDM / ABHA identifiers
# ---------------------------------------------------------------------------

_ABHA_ADDRESS_RE = re.compile(
    r"^[a-z][a-z0-9]*(?:\.[a-z0-9]+)*@[a-z][a-z0-9-]{1,31}$",
    re.IGNORECASE,
)
_ABDM_REGISTRY_ID_RE = re.compile(r"^(?:HPR|HFR)-[A-Z0-9][A-Z0-9-]{5,31}$")


def validate_abha_number(text: str) -> bool:
    """Validate the publicly documented 14-digit ABHA number shape.

    This structural validator preserves the existing ABDM recognizer contract;
    it does not query ABDM or claim that a matching value was issued.
    """

    value = str(text or "").strip()
    if not re.fullmatch(r"\d(?:[\s-]?\d){13}", value):
        return False
    digits = _digits_only(value)
    return digits != "0" * 14


def validate_abha_address(text: str) -> bool:
    """Validate the structural shape used for synthetic ABHA addresses."""

    value = str(text or "").strip()
    if not _ABHA_ADDRESS_RE.fullmatch(value):
        return False
    handle, _domain = value.split("@", 1)
    return 4 <= len(handle) <= 32 and not handle.endswith(".")


def validate_abdm_registry_id(text: str) -> bool:
    """Validate OpenMed's synthetic HPR/HFR surrogate shape."""

    return bool(_ABDM_REGISTRY_ID_RE.fullmatch(str(text or "").strip().upper()))


class ABDMProvider(BaseProvider):
    """Generate synthetic, validator-backed India ABDM identifiers."""

    def abha_number(self) -> str:
        return generate_abha_number(rng=self.generator.random)

    def abha_address(self) -> str:
        rng = self.generator.random
        handle_length = rng.randint(6, 14)
        first = rng.choice("abcdefghijklmnopqrstuvwxyz")
        tail = "".join(
            rng.choice("abcdefghijklmnopqrstuvwxyz0123456789")
            for _ in range(handle_length - 1)
        )
        return f"{first}{tail}@{rng.choice(('abdm', 'sbx'))}"

    def pan(self) -> str:
        rng = self.generator.random
        prefix = "".join(rng.choice(_UPPER_ALPHA) for _ in range(3))
        holder_type = rng.choice(_PAN_HOLDER_TYPES)
        surname_initial = rng.choice(_UPPER_ALPHA)
        serial = f"{rng.randint(0, 9999):04d}"
        check = rng.choice(_UPPER_ALPHA)
        return f"{prefix}{holder_type}{surname_initial}{serial}{check}"

    def abdm_hpr_id(self) -> str:
        return self._registry_id("HPR")

    def abdm_hfr_id(self) -> str:
        return self._registry_id("HFR")

    def _registry_id(self, prefix: str) -> str:
        rng = self.generator.random
        suffix = "".join(rng.choice(_UPPER_ALPHA + "0123456789") for _ in range(12))
        return f"{prefix}-{suffix}"


# ---------------------------------------------------------------------------
# India health identifiers (ABHA, UPI, ration card)
# ---------------------------------------------------------------------------

_UPI_SURROGATE_PROVIDERS = (
    "okaxis",
    "okhdfcbank",
    "oksbi",
    "paytm",
    "upi",
    "ybl",
)
_RATION_CARD_STATE_PREFIXES = (
    "AP",
    "AS",
    "BR",
    "DL",
    "GJ",
    "KA",
    "KL",
    "MH",
    "PB",
    "RJ",
    "TN",
    "TS",
    "UP",
    "WB",
)


def generate_abha_number(*, rng: random.Random | None = None) -> str:
    """Generate a 14-digit ABHA surrogate with a Verhoeff check digit."""

    source = rng or random.Random()
    digits = [source.randint(1, 9)]
    digits.extend(source.randint(0, 9) for _ in range(12))
    digits.append(_verhoeff_checksum(digits))
    return "".join(str(digit) for digit in digits)


def generate_abha_address(*, rng: random.Random | None = None) -> str:
    """Generate a structurally valid synthetic ABHA Address."""

    source = rng or random.Random()
    stem = source.choice(("patient", "health", "record", "member"))
    suffix = "".join(str(source.randint(0, 9)) for _ in range(6))
    domain = source.choice(("abdm", "sbx"))
    return f"{stem}.{suffix}@{domain}"


def generate_upi_id(*, rng: random.Random | None = None) -> str:
    """Generate a structurally valid synthetic UPI virtual payment address."""

    source = rng or random.Random()
    stem = source.choice(("patient", "refund", "member", "account"))
    suffix = "".join(str(source.randint(0, 9)) for _ in range(6))
    provider = source.choice(_UPI_SURROGATE_PROVIDERS)
    return f"{stem}.{suffix}@{provider}"


def generate_indian_ration_card(*, rng: random.Random | None = None) -> str:
    """Generate a conservative synthetic Indian ration-card identifier."""

    source = rng or random.Random()
    prefix = source.choice(_RATION_CARD_STATE_PREFIXES)
    digits = "".join(str(source.randint(0, 9)) for _ in range(10))
    if len(set(digits)) == 1:
        digits = f"{digits[:-1]}{(int(digits[-1]) + 1) % 10}"
    return f"{prefix}-{digits}"


class IndiaHealthIdProvider(BaseProvider):
    """Faker provider for synthetic Indian health-adjacent identifiers."""

    def abha_number(self) -> str:
        return generate_abha_number(rng=self.generator.random)

    def abha_address(self) -> str:
        return generate_abha_address(rng=self.generator.random)

    def upi_id(self) -> str:
        return generate_upi_id(rng=self.generator.random)

    def indian_ration_card(self) -> str:
        return generate_indian_ration_card(rng=self.generator.random)


# ---------------------------------------------------------------------------
# India locale bundle (PIN, phone, PAN, GSTIN, and ABHA)
# ---------------------------------------------------------------------------

_INDIA_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_INDIA_ALPHANUMERIC = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PAN_HOLDER_TYPES = "ABCFGHJLPT"
_PAN_SYNTHETIC_PREFIX = "OMD"
_PAN_CHECK_WEIGHTS = (3, 7, 1, 3, 7, 1, 3, 7, 1)
_GSTIN_VALUE = {character: index for index, character in enumerate(_INDIA_ALPHANUMERIC)}


def validate_indian_pin(text: str) -> bool:
    """Validate the public six-digit Indian Postal Index Number shape."""

    return (
        isinstance(text, str) and re.fullmatch(r"[1-9]\d{5}", text.strip()) is not None
    )


def generate_indian_pin(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic six-digit Indian PIN-shaped value."""

    source = rng or random.Random()
    return str(source.randint(100_000, 999_999))


def validate_indian_phone(text: str) -> bool:
    """Validate an Indian mobile number with an optional ``+91``/``0`` prefix."""

    if not isinstance(text, str):
        return False
    digits = _digits_only(text)
    if text.strip().startswith("+"):
        return len(digits) == 12 and digits.startswith("91") and digits[2] in "6789"
    if len(digits) == 11 and digits.startswith("0"):
        return digits[1] in "6789"
    return len(digits) == 10 and digits[0] in "6789"


def generate_indian_phone(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic ``+91`` Indian mobile number."""

    source = rng or random.Random()
    national = source.choice("6789") + "".join(
        str(source.randint(0, 9)) for _ in range(9)
    )
    return f"+91 {national[:5]} {national[5:]}"


def pan_check_letter(body9: str) -> str:
    """Return the check letter for OpenMed's synthetic PAN series.

    India's allocated PAN check-letter algorithm is not public. Synthetic PANs
    therefore use the project-specific ``OMD`` prefix and this deterministic
    local checksum; non-synthetic PANs are validated by public structure only.
    """

    body = str(body9).strip().upper()
    if re.fullmatch(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}", body) is None:
        raise ValueError("PAN body must match AAAAA9999 with a valid holder type")
    total = sum(
        _GSTIN_VALUE[character] * weight
        for character, weight in zip(body, _PAN_CHECK_WEIGHTS)
    )
    return _INDIA_UPPERCASE[total % len(_INDIA_UPPERCASE)]


def validate_pan(text: str) -> bool:
    """Validate an Indian Permanent Account Number offline."""

    if not isinstance(text, str):
        return False
    code = text.strip().upper()
    if re.fullmatch(r"[A-Z]{3}[ABCFGHJLPT][A-Z]\d{4}[A-Z]", code) is None:
        return False
    if code[5:9] == "0000":
        return False
    if code.startswith(_PAN_SYNTHETIC_PREFIX):
        return code[-1] == pan_check_letter(code[:9])
    return True


def generate_pan(*, rng: random.Random | None = None) -> str:
    """Generate a structurally valid PAN in OpenMed's synthetic series."""

    source = rng or random.Random()
    body = (
        _PAN_SYNTHETIC_PREFIX
        + source.choice(_PAN_HOLDER_TYPES)
        + source.choice(_INDIA_UPPERCASE)
        + f"{source.randint(1, 9999):04d}"
    )
    return body + pan_check_letter(body)


def gstin_check_char(body14: str) -> str:
    """Return the Luhn mod-36 check character for a GSTIN body."""

    body = str(body14).strip().upper()
    if len(body) != 14 or any(character not in _GSTIN_VALUE for character in body):
        raise ValueError("GSTIN body must contain 14 uppercase alphanumeric characters")

    factor = 2
    total = 0
    for character in reversed(body):
        addend = factor * _GSTIN_VALUE[character]
        total += (addend // 36) + (addend % 36)
        factor = 1 if factor == 2 else 2
    return _INDIA_ALPHANUMERIC[(36 - (total % 36)) % 36]


def validate_gstin(text: str) -> bool:
    """Validate an Indian GSTIN, including embedded PAN and checksum."""

    if not isinstance(text, str):
        return False
    code = text.strip().upper()
    if re.fullmatch(r"\d{2}[A-Z0-9]{10}[1-9A-Z]Z[0-9A-Z]", code) is None:
        return False
    if not 1 <= int(code[:2]) <= 38:
        return False
    if not validate_pan(code[2:12]):
        return False
    return code[-1] == gstin_check_char(code[:14])


def generate_gstin(*, rng: random.Random | None = None) -> str:
    """Generate a state-, PAN-, and mod-36-valid synthetic GSTIN."""

    source = rng or random.Random()
    state = f"{source.randint(1, 38):02d}"
    pan = generate_pan(rng=source)
    entity = source.choice(_INDIA_ALPHANUMERIC[1:])
    body = f"{state}{pan}{entity}Z"
    return body + gstin_check_char(body)


def validate_abha(text: str) -> bool:
    """Validate OpenMed's deterministic 14-digit synthetic ABHA contract."""

    if not isinstance(text, str):
        return False
    digits = re.sub(r"[\s-]", "", text)
    if re.fullmatch(r"\d{14}", digits) is None or len(set(digits)) == 1:
        return False

    checksum = 0
    for index, digit in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(digit)]]
    return checksum == 0


def generate_abha(*, rng: random.Random | None = None) -> str:
    """Generate a 14-digit synthetic ABHA with a Verhoeff check digit."""

    source = rng or random.Random()
    digits = [source.randint(1, 9)]
    digits.extend(source.randint(0, 9) for _ in range(12))
    digits.append(_verhoeff_checksum(digits))
    return "".join(str(digit) for digit in digits)


class IndiaSurrogateProvider(BaseProvider):
    """Faker bundle for locale-correct synthetic Indian surrogates."""

    def indian_name(self) -> str:
        """Return a name from the active Indian Faker locale."""

        return self.generator.name()

    def indian_first_name(self) -> str:
        """Return a first name from the active Indian Faker locale."""

        return self.generator.first_name()

    def indian_last_name(self) -> str:
        """Return a last name from the active Indian Faker locale."""

        return self.generator.last_name()

    def indian_pin(self) -> str:
        """Return a synthetic six-digit Indian PIN."""

        return generate_indian_pin(rng=self.generator.random)

    def indian_address(self) -> str:
        """Return a localized Indian street address carrying a PIN."""

        return (
            f"{self.generator.street_address()}, {self.generator.city()} "
            f"{self.indian_pin()}"
        )

    def indian_phone_number(self) -> str:
        """Return a synthetic Indian mobile number."""

        return generate_indian_phone(rng=self.generator.random)

    def pan(self) -> str:
        """Return a synthetic checksum-bearing PAN."""

        return generate_pan(rng=self.generator.random)

    def gstin(self) -> str:
        """Return a synthetic checksum-valid GSTIN."""

        return generate_gstin(rng=self.generator.random)

    def abha(self) -> str:
        """Return a synthetic checksum-valid ABHA number."""

        return generate_abha(rng=self.generator.random)


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
# Chinese mobile, bank-card, passport, and travel-permit identifiers
# ---------------------------------------------------------------------------


def _distinct_digit_candidate(original: str, candidate: str) -> str:
    """Return a same-length digit candidate that differs from ``original``."""

    if candidate != original:
        return candidate
    final_digit = (int(candidate[-1]) + 1) % 10
    return candidate[:-1] + str(final_digit)


def generate_chinese_mobile_number(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic mainland China mobile number.

    The result always uses the domestic 11-digit ``1[3-9]#########`` shape,
    even when ``original`` includes the optional ``+86`` country prefix.
    """

    source = rng or random.Random()
    original_digits = _digits_only(original or "")
    if len(original_digits) == 13 and original_digits.startswith("86"):
        original_digits = original_digits[2:]

    for _ in range(100):
        candidate = (
            "1"
            + str(source.randint(3, 9))
            + "".join(str(source.randint(0, 9)) for _ in range(9))
        )
        if candidate != original_digits:
            return candidate

    return _distinct_digit_candidate(original_digits, candidate)


def generate_chinese_bank_card(
    original: str | None = None,
    *,
    length: int | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a length-preserving, Luhn-valid synthetic bank-card number."""

    source = rng or random.Random()
    original_digits = _digits_only(original or "")
    card_length = length or len(original_digits) or 16
    if card_length not in range(16, 20):
        raise ValueError("Chinese bank-card length must be between 16 and 19 digits")

    for _ in range(100):
        candidate = generate_luhn_identifier(length=card_length, rng=source)
        if candidate != original_digits:
            return candidate

    body = [int(digit) for digit in candidate[:-2]]
    body.append((int(candidate[-2]) + 1) % 10)
    return "".join(str(digit) for digit in body) + str(_luhn_check_digit(body))


def generate_chinese_passport(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic PRC passport number in letter-plus-eight-digit form."""

    source = rng or random.Random()
    original_value = (original or "").strip().upper()
    prefix = (
        original_value[0]
        if re.fullmatch(r"[EGDSP][0-9]{8}", original_value)
        else source.choice("EGDSP")
    )
    for _ in range(100):
        candidate = prefix + "".join(str(source.randint(0, 9)) for _ in range(8))
        if candidate != original_value:
            return candidate
    return prefix + _distinct_digit_candidate(original_value[1:], candidate[1:])


def generate_hong_kong_macau_permit(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic Hong Kong/Macau resident Home Return Permit."""

    source = rng or random.Random()
    original_value = (original or "").strip().upper()
    prefix = (
        original_value[0]
        if re.fullmatch(r"[HM][0-9]{8}", original_value)
        else source.choice("HM")
    )
    for _ in range(100):
        candidate = prefix + "".join(str(source.randint(0, 9)) for _ in range(8))
        if candidate != original_value:
            return candidate
    return prefix + _distinct_digit_candidate(original_value[1:], candidate[1:])


def generate_taiwan_compatriot_permit(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a synthetic eight-digit Taiwan Compatriot Permit."""

    source = rng or random.Random()
    original_digits = _digits_only(original or "")
    for _ in range(100):
        candidate = "".join(str(source.randint(0, 9)) for _ in range(8))
        if candidate != original_digits:
            return candidate
    return _distinct_digit_candidate(original_digits, candidate)


class ChineseIdentifierProvider(BaseProvider):
    """Generate Chinese mobile, bank-card, passport, and permit surrogates."""

    def chinese_mobile_number(self, original: str | None = None) -> str:
        """Return a synthetic mainland China mobile number."""
        return generate_chinese_mobile_number(original, rng=self.generator.random)

    def chinese_bank_card(self, original: str | None = None) -> str:
        """Return a length-preserving, Luhn-valid bank-card number."""
        return generate_chinese_bank_card(original, rng=self.generator.random)

    def chinese_passport(self, original: str | None = None) -> str:
        """Return a synthetic PRC passport number."""
        return generate_chinese_passport(original, rng=self.generator.random)

    def hong_kong_macau_permit(self, original: str | None = None) -> str:
        """Return a synthetic Hong Kong/Macau resident Home Return Permit."""
        return generate_hong_kong_macau_permit(original, rng=self.generator.random)

    def taiwan_compatriot_permit(self, original: str | None = None) -> str:
        """Return a synthetic Taiwan Compatriot Permit."""
        return generate_taiwan_compatriot_permit(original, rng=self.generator.random)


# ---------------------------------------------------------------------------
# South African identity number and mobile phone
# ---------------------------------------------------------------------------


def _south_african_birth_year(digits: str) -> int:
    """Resolve the most recent non-future century for an encoded ``YYMMDD``."""
    today = date.today()
    year = int(digits[:2])
    month = int(digits[2:4])
    day = int(digits[4:6])
    modern_year = 2000 + year
    try:
        modern_date = date(modern_year, month, day)
    except ValueError:
        modern_date = None
    if modern_date is not None and modern_date <= today:
        return modern_year
    return 1900 + year


def _shares_digit_substring(source: str, candidate: str, *, length: int = 6) -> bool:
    """Return whether two digit strings share a substring of ``length``."""
    if len(source) < length:
        return False
    return any(
        source[index : index + length] in candidate
        for index in range(len(source) - length + 1)
    )


def generate_south_african_id(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a valid South African ID surrogate.

    When a 13-digit source is supplied, the surrogate preserves its decoded
    decade of birth, gender band, and citizenship digit. The generated value
    always has a valid calendar date and Luhn check digit and never equals the
    source.

    Args:
        original: Optional source ID whose protected structural attributes are
            preserved.
        rng: Optional deterministic random source.

    Returns:
        A checksum-valid 13-digit synthetic identity number.
    """
    import calendar

    source = rng or random.Random()
    original_digits = _digits_only(original or "")
    has_source_shape = len(original_digits) == 13
    today = date.today()

    if has_source_shape:
        birth_year = _south_african_birth_year(original_digits)
        decade_start = birth_year - birth_year % 10
        gender_is_male = int(original_digits[6:10]) >= 5000
        citizenship = (
            int(original_digits[10])
            if original_digits[10] in {"0", "1"}
            else source.randint(0, 1)
        )
    else:
        decade_start = source.randrange(1940, 2020, 10)
        gender_is_male = bool(source.randint(0, 1))
        citizenship = source.randint(0, 1)

    latest_year = min(decade_start + 9, today.year)
    if latest_year < decade_start:
        decade_start = today.year - today.year % 10
        latest_year = today.year

    candidate = ""
    for _ in range(100):
        year = source.randint(decade_start, latest_year)
        month = source.randint(1, 12)
        day = source.randint(1, calendar.monthrange(year, month)[1])
        birth_date = date(year, month, day)
        if birth_date > today:
            continue

        gender_sequence = (
            source.randint(5000, 9999) if gender_is_male else source.randint(0, 4999)
        )
        body = (
            f"{year % 100:02d}{month:02d}{day:02d}"
            f"{gender_sequence:04d}{citizenship}{source.randint(0, 9)}"
        )
        candidate = body + str(_luhn_check_digit([int(digit) for digit in body]))
        if candidate != original_digits and not _shares_digit_substring(
            original_digits,
            candidate,
        ):
            return candidate

    if not candidate:
        year = decade_start
        body = (
            f"{year % 100:02d}0101{'5000' if gender_is_male else '0000'}{citizenship}0"
        )
        candidate = body + str(_luhn_check_digit([int(digit) for digit in body]))
    body = candidate[:11] + str((int(candidate[11]) + 1) % 10)
    return body + str(_luhn_check_digit([int(digit) for digit in body]))


def generate_za_id_number(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a valid South African ID, preserving source structure."""
    return generate_south_african_id(original, rng=rng)


def generate_za_mobile_number(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a South African mobile surrogate preserving its prefix class."""
    source = rng or random.Random()
    original_text = (original or "").strip()
    original_digits = _digits_only(original_text)

    if len(original_digits) == 11 and original_digits.startswith("27"):
        national = "0" + original_digits[2:]
    elif len(original_digits) == 10 and original_digits.startswith("0"):
        national = original_digits
    else:
        national = ""

    if len(national) == 10 and national[:2] in {"06", "07", "08"}:
        prefix = national[:3]
    else:
        prefix = f"0{source.choice('678')}{source.randint(0, 9)}"

    domestic = ""
    for _ in range(100):
        domestic = prefix + "".join(str(source.randint(0, 9)) for _ in range(7))
        if domestic != national and not _shares_digit_substring(national, domestic):
            break
    if domestic == national:
        domestic = domestic[:-1] + str((int(domestic[-1]) + 1) % 10)

    if original_text.startswith("+27"):
        return f"+27 {domestic[1:3]} {domestic[3:6]} {domestic[6:]}"
    if len(original_digits) == 11 and original_digits.startswith("27"):
        return "27" + domestic[1:]
    if any(separator in original_text for separator in (" ", "-", ".")):
        return f"{domestic[:3]} {domestic[3:6]} {domestic[6:]}"
    return domestic


class SouthAfricanIdProvider(BaseProvider):
    """Generate deterministic South African ID and mobile surrogates."""

    def south_african_id(self, original: str | None = None) -> str:
        """Return a Luhn-valid South African identity-number surrogate."""
        return generate_south_african_id(original, rng=self.generator.random)

    def za_mobile_number(self, original: str | None = None) -> str:
        """Return a South African mobile surrogate with a stable prefix class."""
        return generate_za_mobile_number(original, rng=self.generator.random)


# ---------------------------------------------------------------------------
# Egyptian national ID and Moroccan CIN
# ---------------------------------------------------------------------------

_MOROCCAN_CIN_DEFAULT_PREFIXES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + (
    "AA",
    "BK",
    "BE",
    "BH",
    "BJ",
)


def generate_egyptian_national_id(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a structurally valid Egyptian national ID surrogate.

    When ``original`` is valid, its century, birth date, governorate, and
    gender parity are retained while its four-digit serial is changed. The
    unpublished final check digit is generated but intentionally not validated.

    Args:
        original: Optional source ID in ASCII or Arabic-Indic digits.
        rng: Optional deterministic random source.

    Returns:
        A distinct 14-digit Egyptian national ID surrogate.
    """
    import calendar

    from openmed.core.pii_i18n import (
        EGYPTIAN_GOVERNORATE_CODES,
        normalize_arabic_indic_digits,
        validate_egyptian_national_id,
    )

    source = rng or random.Random()
    original_text = (
        normalize_arabic_indic_digits(original.strip())
        if isinstance(original, str)
        else ""
    )

    if validate_egyptian_national_id(original_text):
        prefix = original_text[:9]
        gender_parity = int(original_text[12]) % 2
        original_serial = original_text[9:13]
    else:
        century_digit = source.choice(("2", "3"))
        first_year = 1900 if century_digit == "2" else 2000
        last_year = 1999 if century_digit == "2" else date.today().year
        year = source.randint(first_year, last_year)
        month = source.randint(1, 12)
        day = source.randint(1, calendar.monthrange(year, month)[1])
        governorate = source.choice(tuple(sorted(EGYPTIAN_GOVERNORATE_CODES)))
        prefix = f"{century_digit}{year % 100:02d}{month:02d}{day:02d}{governorate}"
        gender_parity = source.randint(0, 1)
        original_serial = ""

    serial = ""
    for _ in range(100):
        serial = "".join(str(source.randint(0, 9)) for _ in range(3))
        serial += str(source.choice(tuple(range(gender_parity, 10, 2))))
        if serial != original_serial:
            break
    else:  # pragma: no cover - hostile RNG fallback
        serial = f"{(int(serial or '0') + 2) % 10_000:04d}"

    check_digit = source.randint(0, 9)
    return f"{prefix}{serial}{check_digit}"


def generate_moroccan_cin(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a Moroccan CIN surrogate preserving its regional prefix.

    A valid source retains its exact one- or two-letter prefix and digit width.
    Without a valid source, a published regional prefix and six-digit serial are
    generated.

    Args:
        original: Optional source CIN in ASCII or Arabic-Indic digits.
        rng: Optional deterministic random source.

    Returns:
        A structurally valid CIN distinct from ``original``.
    """
    from openmed.core.pii_i18n import (
        normalize_arabic_indic_digits,
        validate_moroccan_cin,
    )

    source = rng or random.Random()
    original_text = (
        normalize_arabic_indic_digits(original.strip()).upper()
        if isinstance(original, str)
        else ""
    )
    if validate_moroccan_cin(original_text):
        match = re.fullmatch(r"([A-Z]{1,2})([0-9]{5,7})", original_text)
        if match is None:  # pragma: no cover - validator contract
            raise RuntimeError("validated Moroccan CIN could not be parsed")
        prefix, original_serial = match.groups()
        serial_length = len(original_serial)
    else:
        prefix = source.choice(_MOROCCAN_CIN_DEFAULT_PREFIXES)
        original_serial = ""
        serial_length = 6

    serial = ""
    for _ in range(100):
        serial = "".join(str(source.randint(0, 9)) for _ in range(serial_length))
        if serial != original_serial:
            return prefix + serial

    value = (int(serial or "0") + 1) % (10**serial_length)
    return prefix + f"{value:0{serial_length}d}"


class EgyptMoroccoIdProvider(BaseProvider):
    """Generate format-preserving Egyptian and Moroccan ID surrogates."""

    def egyptian_national_id(self, original: str | None = None) -> str:
        """Return a structurally valid Egyptian national ID surrogate."""
        return generate_egyptian_national_id(original, rng=self.generator.random)

    def moroccan_cin(self, original: str | None = None) -> str:
        """Return a format-preserving Moroccan CIN surrogate."""
        return generate_moroccan_cin(original, rng=self.generator.random)


# ---------------------------------------------------------------------------
# Nigerian NIN, BVN, and mobile phone
# ---------------------------------------------------------------------------


def generate_nigeria_nin(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a structurally valid Nigerian NIN surrogate.

    Args:
        original: Optional source NIN that the surrogate must not equal.
        rng: Optional deterministic random source.

    Returns:
        A non-trivial synthetic 11-digit NIN.
    """
    source = rng or random.Random()
    original_digits = _digits_only(original or "")

    from openmed.core.pii_i18n import validate_nigeria_nin

    for _ in range(100):
        candidate = "".join(str(source.randint(0, 9)) for _ in range(11))
        if candidate != original_digits and validate_nigeria_nin(candidate):
            return candidate
    return "52740618395"


def generate_nigeria_bvn(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a structurally valid Nigerian BVN surrogate.

    Args:
        original: Optional source BVN that the surrogate must not equal.
        rng: Optional deterministic random source.

    Returns:
        A non-trivial synthetic 11-digit BVN.
    """
    source = rng or random.Random()
    original_digits = _digits_only(original or "")

    from openmed.core.pii_i18n import validate_nigeria_bvn

    for _ in range(100):
        candidate = "".join(str(source.randint(0, 9)) for _ in range(11))
        if candidate != original_digits and validate_nigeria_bvn(candidate):
            return candidate
    return "28471390652"


_NIGERIA_MOBILE_PREFIX_CLASSES = ("070", "080", "081", "090", "091")


def generate_ng_mobile_number(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a Nigerian mobile surrogate preserving its prefix class.

    Recognized domestic prefixes retain their ``070x``, ``080x``, ``081x``,
    ``090x``, or ``091x`` class. International ``+234`` presentation and common
    digit grouping are retained as well.

    Args:
        original: Optional source mobile number whose prefix class and display
            style should be preserved.
        rng: Optional deterministic random source.

    Returns:
        A synthetic Nigerian mobile number.
    """
    source = rng or random.Random()
    original_text = (original or "").strip()
    original_digits = _digits_only(original_text)

    if len(original_digits) == 13 and original_digits.startswith("234"):
        national = "0" + original_digits[3:]
    elif len(original_digits) == 11 and original_digits.startswith("0"):
        national = original_digits
    else:
        national = ""

    prefix_class = (
        national[:3]
        if national[:3] in _NIGERIA_MOBILE_PREFIX_CLASSES
        else source.choice(_NIGERIA_MOBILE_PREFIX_CLASSES)
    )

    domestic = ""
    for _ in range(100):
        domestic = prefix_class + "".join(str(source.randint(0, 9)) for _ in range(8))
        if domestic != national:
            break
    if domestic == national:
        domestic = domestic[:-1] + str((int(domestic[-1]) + 1) % 10)

    if original_text.startswith("+234"):
        return f"+234 {domestic[1:4]} {domestic[4:7]} {domestic[7:]}"
    if len(original_digits) == 13 and original_digits.startswith("234"):
        return "234" + domestic[1:]
    if any(separator in original_text for separator in (" ", "-", ".")):
        return f"{domestic[:4]} {domestic[4:7]} {domestic[7:]}"
    return domestic


class NigeriaIdProvider(BaseProvider):
    """Generate deterministic Nigerian NIN, BVN, and mobile surrogates."""

    def nigeria_nin(self, original: str | None = None) -> str:
        """Return a structurally valid Nigerian NIN surrogate."""
        return generate_nigeria_nin(original, rng=self.generator.random)

    def nigeria_bvn(self, original: str | None = None) -> str:
        """Return a structurally valid Nigerian BVN surrogate."""
        return generate_nigeria_bvn(original, rng=self.generator.random)

    def ng_mobile_number(self, original: str | None = None) -> str:
        """Return a Nigerian mobile surrogate with a stable prefix class."""
        return generate_ng_mobile_number(original, rng=self.generator.random)


# ---------------------------------------------------------------------------
# Ghana Card PIN and Kenyan identity numbers
# ---------------------------------------------------------------------------


def _numeric_surrogate(
    original: str | None,
    *,
    length: int,
    rng: random.Random,
) -> str:
    """Return a deterministic numeric surrogate distinct from ``original``."""
    original_text = (original or "").strip()
    candidate = ""
    for _ in range(100):
        candidate = "".join(str(rng.randint(0, 9)) for _ in range(length))
        if candidate != original_text:
            return candidate

    value = (int(candidate or "0") + 1) % (10**length)
    candidate = f"{value:0{length}d}"
    if candidate == original_text:  # pragma: no cover - defensive wraparound
        candidate = f"{(value + 1) % (10**length):0{length}d}"
    return candidate


def generate_ghana_card_pin(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a structurally valid Ghana Card PIN surrogate.

    A valid source's ICAO prefix is retained; otherwise ``GHA`` is used. The
    generated presentation always preserves both hyphens and the documented
    ``GHA-#########-#`` card form. NIA does not publish an offline checksum.

    Args:
        original: Optional source PIN whose country prefix should be retained.
        rng: Optional deterministic random source.

    Returns:
        A distinct, structurally valid Ghana Card PIN.
    """
    source = rng or random.Random()
    original_text = (original or "").strip().upper()
    match = re.fullmatch(r"([A-Z]{3})-([0-9]{9})-([0-9])", original_text)
    prefix = match.group(1) if match is not None else "GHA"
    original_digits = "" if match is None else f"{match.group(2)}{match.group(3)}"
    digits = _numeric_surrogate(original_digits, length=10, rng=source)
    return f"{prefix}-{digits[:9]}-{digits[9]}"


def generate_kenya_national_id(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a distinct seven- or eight-digit Kenyan national ID."""
    original_text = (original or "").strip()
    length = len(original_text) if re.fullmatch(r"[0-9]{7,8}", original_text) else 8
    return _numeric_surrogate(original_text, length=length, rng=rng or random.Random())


def generate_kenya_maisha_namba(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a distinct nine-digit Kenya Maisha Namba surrogate."""
    return _numeric_surrogate(
        (original or "").strip(),
        length=9,
        rng=rng or random.Random(),
    )


class GhanaKenyaIdProvider(BaseProvider):
    """Generate deterministic Ghanaian and Kenyan identity surrogates."""

    def ghana_card_pin(self, original: str | None = None) -> str:
        """Return a structurally valid Ghana Card PIN surrogate."""
        return generate_ghana_card_pin(original, rng=self.generator.random)

    def kenya_national_id(self, original: str | None = None) -> str:
        """Return a seven- or eight-digit Kenyan national ID surrogate."""
        return generate_kenya_national_id(original, rng=self.generator.random)

    def kenya_maisha_namba(self, original: str | None = None) -> str:
        """Return a nine-digit Kenya Maisha Namba surrogate."""
        return generate_kenya_maisha_namba(original, rng=self.generator.random)


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
# Chinese Resident Identity Card (18 characters, ISO 7064 MOD 11-2)
# ---------------------------------------------------------------------------

_CHINESE_RESIDENT_ID_PREFECTURE_CODES = (*range(1, 71), 90)


def generate_chinese_resident_id(*, rng: random.Random | None = None) -> str:
    """Generate a checksum-valid synthetic Chinese Resident Identity Card ID.

    The six-digit region portion uses a stable mainland GB/T 2260
    province-level prefix and synthetic non-zero subdivision digits. Birth
    dates and sequence values are generated rather than copied from any
    person or dataset.
    """
    import calendar

    from openmed.core.pii_i18n import (
        CHINESE_RESIDENT_ID_REGION_PREFIXES,
        chinese_resident_id_check_character,
    )

    source = rng or random.Random()
    province_prefix = source.choice(tuple(sorted(CHINESE_RESIDENT_ID_REGION_PREFIXES)))
    prefecture_code = source.choice(_CHINESE_RESIDENT_ID_PREFECTURE_CODES)
    county_code = source.randint(1, 99)

    year = source.randint(1940, 2020)
    month = source.randint(1, 12)
    day = source.randint(1, calendar.monthrange(year, month)[1])
    sequence = source.randint(1, 999)

    body = (
        f"{province_prefix}{prefecture_code:02d}{county_code:02d}"
        f"{year:04d}{month:02d}{day:02d}{sequence:03d}"
    )
    return f"{body}{chinese_resident_id_check_character(body)}"


class ChineseResidentIdProvider(BaseProvider):
    """Generates synthetic Chinese Resident IDs with valid MOD 11-2 checks."""

    def chinese_resident_id(self) -> str:
        return generate_chinese_resident_id(rng=self.generator.random)


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
# Vietnamese CCCD / legacy CMND
# ---------------------------------------------------------------------------


def generate_vietnamese_cccd(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic CCCD accepted by its structural validator."""
    from openmed.core.pii_i18n import validate_vietnamese_cccd

    source = rng or random.Random()
    for _ in range(20):
        candidate = "".join(str(source.randint(0, 9)) for _ in range(12))
        if len(set(candidate)) > 1 and validate_vietnamese_cccd(candidate):
            return candidate
    return "001203123456"


def generate_vietnamese_cmnd(*, rng: random.Random | None = None) -> str:
    """Generate a synthetic legacy CMND accepted by its structural validator."""
    from openmed.core.pii_i18n import validate_vietnamese_cmnd

    source = rng or random.Random()
    for _ in range(20):
        candidate = str(source.randint(1, 9)) + "".join(
            str(source.randint(0, 9)) for _ in range(8)
        )
        if validate_vietnamese_cmnd(candidate):
            return candidate
    return "123456789"


class VietnameseIdProvider(BaseProvider):
    """Generates synthetic Vietnamese CCCD and legacy CMND values."""

    def vietnamese_cccd(self) -> str:
        return generate_vietnamese_cccd(rng=self.generator.random)

    def vietnamese_cmnd(self) -> str:
        return generate_vietnamese_cmnd(rng=self.generator.random)


# ---------------------------------------------------------------------------
# East African national IDs (Tanzania, Uganda, Rwanda, Ethiopia)
# ---------------------------------------------------------------------------


def generate_tanzania_nida(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a Tanzania NIDA surrogate, preserving the birth decade."""
    from openmed.core.pii_i18n import validate_tanzania_nida

    source = rng or random.Random()
    today = date.today()
    if original is not None and validate_tanzania_nida(original):
        original_year = int(re.sub(r"[^0-9]", "", original)[:4])
        decade_start = (original_year // 10) * 10
        first_date = date(decade_start, 1, 1)
        last_date = min(date(decade_start + 9, 12, 31), today)
    else:
        first_date = date(1940, 1, 1)
        last_date = today

    birth_date = date.fromordinal(
        source.randint(first_date.toordinal(), last_date.toordinal())
    )
    suffix = "".join(str(source.randint(0, 9)) for _ in range(12))
    return f"{birth_date:%Y%m%d}{suffix}"


def generate_uganda_nin(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a Uganda NIN surrogate, preserving class and gender."""
    from openmed.core.pii_i18n import validate_uganda_nin

    source = rng or random.Random()
    if original is not None and validate_uganda_nin(original):
        prefix = original.strip().upper()[:2]
    else:
        prefix = f"C{source.choice(('M', 'F'))}"
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    suffix = "".join(source.choice(alphabet) for _ in range(12))
    return f"{prefix}{suffix}"


def generate_rwanda_id(
    original: str | None = None,
    *,
    rng: random.Random | None = None,
) -> str:
    """Generate a Rwanda ID preserving status, birth decade, and gender."""
    from openmed.core.pii_i18n import validate_rwanda_id

    source = rng or random.Random()
    current_year = date.today().year
    if original is not None and validate_rwanda_id(original):
        digits = original.strip()
        status = digits[0]
        gender = digits[5]
        decade_start = (int(digits[1:5]) // 10) * 10
        birth_year = source.randint(
            decade_start,
            min(decade_start + 9, current_year),
        )
    else:
        status = str(source.randint(1, 9))
        gender = source.choice(("7", "8"))
        birth_year = source.randint(1940, current_year)

    suffix = "".join(str(source.randint(0, 9)) for _ in range(10))
    return f"{status}{birth_year:04d}{gender}{suffix}"


def generate_ethiopia_fayda(*, rng: random.Random | None = None) -> str:
    """Generate a 12-digit Fayda FAN with a valid Verhoeff checksum."""
    source = rng or random.Random()
    digits = [source.randint(2, 9)]
    digits.extend(source.randint(0, 9) for _ in range(10))
    digits.append(_verhoeff_checksum(digits))
    return "".join(str(digit) for digit in digits)


class EastAfricanIdProvider(BaseProvider):
    """Generate structurally valid East African national-ID surrogates."""

    def tanzania_nida(self, original: str | None = None) -> str:
        return generate_tanzania_nida(original, rng=self.generator.random)

    def uganda_nin(self, original: str | None = None) -> str:
        return generate_uganda_nin(original, rng=self.generator.random)

    def rwanda_id(self, original: str | None = None) -> str:
        return generate_rwanda_id(original, rng=self.generator.random)

    def ethiopia_fayda(self) -> str:
        return generate_ethiopia_fayda(rng=self.generator.random)


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


_MPESA_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_MPESA_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MPESA_DIGITS = "0123456789"


def generate_mpesa_transaction_code(
    original: str | None = None,
    *,
    leading_character: str | None = None,
    rng: random.Random | None = None,
) -> str:
    """Generate a structurally valid M-Pesa transaction-code surrogate.

    When an original code or explicit leading character is supplied, the
    first character is retained. M-Pesa uses that position as part of its date
    encoding, so preserving it keeps receipt ordering signals without copying
    the identifying transaction code.

    Args:
        original: Optional original transaction code whose first character is
            preserved.
        leading_character: Optional explicit first character. Takes precedence
            over ``original``.
        rng: Optional seeded random-number generator.

    Returns:
        A ten-character uppercase alphanumeric transaction code.
    """

    source = rng or random.Random()
    preserved = leading_character
    if preserved is None and original:
        preserved = original[0]
    if preserved is not None and re.fullmatch(r"[A-Z0-9]", preserved) is None:
        raise ValueError("leading_character must be one uppercase letter or digit")

    characters = [preserved or source.choice(_MPESA_ALPHABET)]
    characters.extend(source.choice(_MPESA_ALPHABET) for _ in range(2))
    characters.append(source.choice(_MPESA_DIGITS))
    characters.extend(source.choice(_MPESA_ALPHABET) for _ in range(6))

    if not any(character.isalpha() for character in characters):
        index = source.choice((1, 2, 4, 5, 6, 7, 8, 9))
        characters[index] = source.choice(_MPESA_LETTERS)

    candidate = "".join(characters)
    if original is not None and candidate == original:
        alternatives = _MPESA_LETTERS.replace(original[1], "")
        characters[1] = source.choice(alternatives)
        candidate = "".join(characters)
    return candidate


class MpesaProvider(BaseProvider):
    """Generates seeded M-Pesa transaction-code surrogates."""

    def mpesa_transaction_code(self, original: str | None = None) -> str:
        """Return a valid code, preserving the original leading character."""

        return generate_mpesa_transaction_code(
            original,
            rng=self.generator.random,
        )


def _generate_mobile_money_digits(
    original: str | None,
    *,
    allowed_lengths: range,
    default_length: int,
    rng: random.Random,
) -> str:
    """Generate a distinct, length-preserving numeric billing identifier."""

    if original is None:
        length = default_length
    else:
        if not isinstance(original, str) or not original.isascii():
            raise ValueError("mobile-money identifiers must contain ASCII digits")
        if not original.isdigit() or len(original) not in allowed_lengths:
            raise ValueError("mobile-money identifier has an unsupported digit length")
        length = len(original)

    candidate = "".join(str(rng.randint(0, 9)) for _ in range(length))
    if candidate == original:
        replacement = str((int(candidate[-1]) + 1) % 10)
        candidate = f"{candidate[:-1]}{replacement}"
    return candidate


class MobileMoneyProvider(BaseProvider):
    """Generate seeded mobile-money billing identifier surrogates."""

    def mobile_money_paybill(self, original: str | None = None) -> str:
        """Return a five- to seven-digit paybill surrogate."""

        return _generate_mobile_money_digits(
            original,
            allowed_lengths=range(5, 8),
            default_length=6,
            rng=self.generator.random,
        )

    def mobile_money_till(self, original: str | None = None) -> str:
        """Return a five- to seven-digit till surrogate."""

        return _generate_mobile_money_digits(
            original,
            allowed_lengths=range(5, 8),
            default_length=6,
            rng=self.generator.random,
        )

    def mobile_money_agent(self, original: str | None = None) -> str:
        """Return a five- to seven-digit agent-number surrogate."""

        return _generate_mobile_money_digits(
            original,
            allowed_lengths=range(5, 8),
            default_length=6,
            rng=self.generator.random,
        )

    def momo_reference(self, original: str | None = None) -> str:
        """Return a 10- to 12-digit MTN MoMo reference surrogate."""

        return _generate_mobile_money_digits(
            original,
            allowed_lengths=range(10, 13),
            default_length=10,
            rng=self.generator.random,
        )


# The Kenya Ministry of Health defines MFL codes as five-digit sequential
# numbers. The high five-digit band and high Nigeria serial band keep generated
# identifiers visibly synthetic without consulting either country's registry.
KENYA_MFL_SYNTHETIC_MIN = 90_000
KENYA_MFL_SYNTHETIC_MAX = 99_999
NIGERIA_HFR_SYNTHETIC_SERIAL_MIN = 9_000
NIGERIA_HFR_SYNTHETIC_SERIAL_MAX = 9_999


class HealthFacilityCodeProvider(BaseProvider):
    """Generate deterministic Kenya KMHFL and Nigeria HFR surrogates."""

    def kmhfl_code(self, original: str | None = None) -> str:
        """Return a five-digit Kenya MFL code from the synthetic high band."""

        for _ in range(20):
            candidate = str(
                self.generator.random.randint(
                    KENYA_MFL_SYNTHETIC_MIN,
                    KENYA_MFL_SYNTHETIC_MAX,
                )
            )
            if candidate != original:
                return candidate

        fallback = KENYA_MFL_SYNTHETIC_MIN
        if str(fallback) == original:
            fallback += 1
        return str(fallback)

    def hfr_facility_code(self, original: str | None = None) -> str:
        """Return a structurally valid Nigeria HFR code with a synthetic serial."""

        for _ in range(20):
            state = self.generator.random.randint(1, 37)
            lga = self.generator.random.randint(1, 44)
            ownership = self.generator.random.randint(1, 2)
            level_of_care = self.generator.random.randint(1, 3)
            serial = self.generator.random.randint(
                NIGERIA_HFR_SYNTHETIC_SERIAL_MIN,
                NIGERIA_HFR_SYNTHETIC_SERIAL_MAX,
            )
            candidate = f"{state:02d}{lga:02d}{ownership}{level_of_care}{serial:04d}"
            if candidate != original:
                return candidate

        fallback = f"010111{NIGERIA_HFR_SYNTHETIC_SERIAL_MIN:04d}"
        if fallback == original:
            fallback = f"010111{NIGERIA_HFR_SYNTHETIC_SERIAL_MIN + 1:04d}"
        return fallback


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


_extra_providers: list[type[BaseProvider]] = []


def register_extra_clinical_provider(provider: type[BaseProvider]) -> None:
    """Register a provider class for every subsequently built Faker instance."""

    if provider not in _extra_providers:
        _extra_providers.append(provider)


def register_clinical_providers(faker) -> None:
    """Add every built-in and caller-registered provider to ``faker``."""
    from .registry_ids import clinical_faker_provider_classes

    providers = (
        *clinical_faker_provider_classes(),
        MedicalRecordNumberProvider,
        FinancialIdentifierProvider,
        MrzProvider,
        UnifiedSocialCreditCodeProvider,
        *_extra_providers,
    )
    seen: set[type[BaseProvider]] = set()
    for provider in providers:
        if provider not in seen:
            faker.add_provider(provider)
            seen.add(provider)


__all__ = [
    "ABDMProvider",
    "AadhaarProvider",
    "AfricanPhoneProvider",
    "AustralianMedicareProvider",
    "AustralianTFNProvider",
    "BCPHNProvider",
    "BulgarianEgnProvider",
    "CanadianSINProvider",
    "ChineseIdentifierProvider",
    "ChineseResidentIdProvider",
    "DanishCPRProvider",
    "EastAfricanIdProvider",
    "EgyptMoroccoIdProvider",
    "EstonianIsikukoodProvider",
    "FinancialIdentifierProvider",
    "GermanSteuerIdProvider",
    "HealthFacilityCodeProvider",
    "GhanaKenyaIdProvider",
    "HungarianTAJProvider",
    "IndiaHealthIdProvider",
    "IndiaSurrogateProvider",
    "IndonesianNIKProvider",
    "IsraeliTeudatZehutProvider",
    "KoreanRRNProvider",
    "KENYA_MFL_SYNTHETIC_MAX",
    "KENYA_MFL_SYNTHETIC_MIN",
    "LatvianPersonasKodsProvider",
    "MalaysianMyKadProvider",
    "MedicalRecordNumberProvider",
    "MobileMoneyProvider",
    "MpesaProvider",
    "MrzProvider",
    "NPIProvider",
    "NigeriaIdProvider",
    "NIGERIA_HFR_SYNTHETIC_SERIAL_MAX",
    "NIGERIA_HFR_SYNTHETIC_SERIAL_MIN",
    "UnifiedSocialCreditCodeProvider",
    "PhilippinesIdProvider",
    "PolishPeselProvider",
    "RomanianCNPProvider",
    "RodneCisloProvider",
    "SerbianJmbgProvider",
    "SouthAfricanIdProvider",
    "ThaiNationalIdProvider",
    "VietnameseIdProvider",
    "SpanishDNIProvider",
    "PortugueseNIFProvider",
    "SpanishNIEProvider",
    "UKNHSNumberProvider",
    "UKNINOProvider",
    "generate_australian_medicare",
    "generate_australian_tfn",
    "generate_aadhaar",
    "generate_african_phone",
    "generate_bc_phn",
    "generate_bic",
    "generate_bulgarian_egn",
    "generate_canadian_sin",
    "generate_chinese_bank_card",
    "generate_chinese_mobile_number",
    "generate_chinese_passport",
    "generate_chinese_resident_id",
    "generate_danish_cpr",
    "generate_egyptian_national_id",
    "generate_hungarian_taj",
    "generate_abha_address",
    "generate_abha_number",
    "generate_abha",
    "generate_gstin",
    "generate_indian_phone",
    "generate_indian_pin",
    "generate_pan",
    "generate_estonian_isikukood",
    "generate_ethiopia_fayda",
    "generate_ghana_card_pin",
    "generate_hong_kong_macau_permit",
    "generate_iban",
    "generate_ontario_health_card",
    "generate_rwanda_id",
    "generate_indonesian_nik",
    "generate_indian_ration_card",
    "generate_jmbg",
    "generate_kenya_maisha_namba",
    "generate_kenya_national_id",
    "generate_teudat_zehut",
    "generate_korean_rrn",
    "generate_luhn_identifier",
    "generate_npi",
    "generate_nigeria_bvn",
    "generate_nigeria_nin",
    "generate_pesel",
    "generate_latvian_personas_kods",
    "generate_malaysian_mykad",
    "generate_mpesa_transaction_code",
    "generate_moroccan_cin",
    "generate_ng_mobile_number",
    "generate_philhealth_pin",
    "generate_philsys_psn",
    "generate_rodne_cislo",
    "generate_romanian_cnp",
    "generate_portuguese_nif",
    "generate_south_african_id",
    "generate_spanish_nie",
    "generate_ssn",
    "generate_za_id_number",
    "generate_za_mobile_number",
    "generate_thai_national_id",
    "generate_taiwan_compatriot_permit",
    "generate_tanzania_nida",
    "generate_uganda_nin",
    "generate_upi_id",
    "generate_vietnamese_cccd",
    "generate_vietnamese_cmnd",
    "generate_uk_nhs_number",
    "generate_unified_social_credit_code",
    "id_subtype_for_entity_type",
    "register_clinical_providers",
    "register_extra_clinical_provider",
    "validate_australian_medicare",
    "validate_australian_tfn",
    "validate_abdm_registry_id",
    "validate_abha_address",
    "validate_abha_number",
    "validate_bc_phn",
    "validate_bic",
    "validate_canadian_sin",
    "validate_iban",
    "validate_abha",
    "validate_gstin",
    "validate_indian_phone",
    "validate_indian_pin",
    "validate_luhn",
    "validate_npi",
    "validate_ontario_health_card",
    "validate_pan",
    "validate_phone_us",
    "validate_ssn",
    "validate_uk_nhs_number",
    "validate_uk_nino",
]
