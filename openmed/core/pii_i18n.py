"""Multilingual PII detection support.

Language-specific data, validators, regex patterns, and fake data
for multilingual PII detection and de-identification.

Health identifiers for HIPAA cross-map consumers:
    Several national IDs surfaced here double as patient health identifiers and
    should be treated as PHI. These include the UK NHS Number, the Australian
    Medicare card number (:func:`validate_australian_medicare`), and the
    Canadian provincial health-card numbers: the Ontario (OHIP) health card
    (:func:`validate_ontario_health_card`) and the British Columbia Personal
    Health Number (:func:`validate_bc_phn`). The Australian Tax File Number
    (:func:`validate_australian_tfn`) and Canadian Social Insurance Number
    (:func:`validate_canadian_sin`) are direct identifiers rather than health
    identifiers, but are redacted alongside them in clinical text.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .anonymizer.providers.clinical_ids import (
    validate_australian_medicare,
    validate_australian_tfn,
    validate_bc_phn,
    validate_canadian_sin,
    validate_ontario_health_card,
    validate_uk_nhs_number,
    validate_uk_nino,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Set[str] = {
    "en",
    "fr",
    "de",
    "it",
    "es",
    "nl",
    "hi",
    "te",
    "pt",
    "ar",
    "he",
    "ja",
    "tr",
    "id",
    "th",
    "ko",
}

# Languages with validator-backed national-ID coverage but no bundled default
# PII model or full language pack yet.
NATIONAL_ID_ONLY_LANGUAGES: Set[str] = {"pl", "lv", "sk", "ms", "tl", "da"}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
    "hi": "Hindi",
    "te": "Telugu",
    "pt": "Portuguese",
    "ar": "Arabic",
    "he": "Hebrew",
    "ja": "Japanese",
    "tr": "Turkish",
    "id": "Indonesian",
    "th": "Thai",
    "ko": "Korean",
}

LANGUAGE_MODEL_PREFIX: Dict[str, str] = {
    "en": "",
    "fr": "French-",
    "de": "German-",
    "it": "Italian-",
    "es": "Spanish-",
    "nl": "Dutch-",
    "hi": "Hindi-",
    "te": "Telugu-",
    "pt": "Portuguese-",
    "ar": "Arabic-",
    "he": "Hebrew-",
    "ja": "Japanese-",
    "tr": "Turkish-",
    "id": "Indonesian-",
    "th": "Thai-",
    "ko": "Korean-",
}

DEFAULT_PII_MODELS: Dict[str, str] = {
    "en": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    "fr": "OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1",
    "de": "OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1",
    "it": "OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1",
    "es": "OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1",
    "nl": "OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1",
    "hi": "OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1",
    "te": "OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1",
    "pt": "OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1",
    "ar": "OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1",
    "he": "OpenMed/privacy-filter-multilingual",
    "ja": "OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1",
    "tr": "OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1",
    "id": "OpenMed/privacy-filter-multilingual",
    "th": "OpenMed/privacy-filter-multilingual",
    "ko": "OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1",
}


# ---------------------------------------------------------------------------
# Financial Identifier Validators
# ---------------------------------------------------------------------------


def validate_iban(text: str) -> bool:
    """Validate an IBAN using ISO 7064 MOD-97-10 checksum rules."""

    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_iban(text)


def validate_bic(text: str) -> bool:
    """Validate a SWIFT/BIC code's 8- or 11-character structure."""

    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_bic(text)


# ---------------------------------------------------------------------------
# National ID Validators
# ---------------------------------------------------------------------------


def validate_french_nir(text: str) -> bool:
    """Validate French NIR/INSEE number.

    The NIR (Numero d'Inscription au Repertoire) is 15 characters.
    Format: S AA MM DDD CCC OOO KK
    Key (last 2 digits) = 97 - (first 13 digits mod 97)
    Corsica department codes 2A and 2B are normalized to 19 and 18
    respectively before computing the checksum.

    Args:
        text: NIR string (may contain spaces)

    Returns:
        True if valid NIR format and checksum
    """
    cleaned = re.sub(r"[\s.-]", "", text).upper()

    if len(cleaned) != 15:
        return False

    # First digit must be 1 or 2
    if cleaned[0] not in ("1", "2"):
        return False

    try:
        body = cleaned[:13]
        key = int(cleaned[13:15])
        if "A" in body or "B" in body:
            if not re.match(r"^[12]\d{4}2[AB]\d{6}$", body):
                return False
            body = body.replace("2A", "19").replace("2B", "18")
        elif not body.isdigit():
            return False

        number = int(body)
        return key == 97 - (number % 97)
    except (ValueError, IndexError):
        return False


def validate_german_steuer_id(text: str) -> bool:
    """Validate German Steuer-ID (tax identification number).

    The Steuer-ID is 11 digits with a checksum validation.
    Rules:
    - Exactly 11 digits
    - First digit cannot be 0
    - Exactly one digit appears twice or three times;
      remaining digits appear exactly once

    Args:
        text: Steuer-ID string (may contain spaces)

    Returns:
        True if valid Steuer-ID format
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False

    # First digit cannot be 0
    if digits[0] == "0":
        return False

    # Check digit frequency: in the first 10 digits, exactly one digit
    # must appear 2 or 3 times, rest appear once or zero times
    first_ten = digits[:10]
    from collections import Counter

    counts = Counter(first_ten)
    multi_count = sum(1 for c in counts.values() if c >= 2)
    if multi_count != 1:
        return False

    return True


def validate_italian_codice_fiscale(text: str) -> bool:
    """Validate Italian Codice Fiscale format.

    Format: AAABBB00A00A000A (16 alphanumeric characters)
    - 3 letters: surname consonants
    - 3 letters: first name consonants
    - 2 digits: year of birth
    - 1 letter: month of birth (A-T mapping)
    - 2 digits: day of birth (+ 40 for females)
    - 1 letter: municipality code letter
    - 3 digits: municipality code number
    - 1 letter: check character

    Args:
        text: Codice Fiscale string

    Returns:
        True if valid format
    """
    cleaned = re.sub(r"\s", "", text).upper()

    if len(cleaned) != 16:
        return False

    # Check format: LLLLLLDDLDDLDDDL
    pattern = r"^[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]$"
    return bool(re.match(pattern, cleaned))


_DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def validate_spanish_dni(text: str) -> bool:
    """Validate Spanish DNI (Documento Nacional de Identidad).

    Format: 8 digits + 1 check letter.
    The check letter is determined by number mod 23 mapped to a lookup table.

    Args:
        text: DNI string (may contain spaces)

    Returns:
        True if valid DNI format and check letter
    """
    cleaned = re.sub(r"\s", "", text).upper()

    if len(cleaned) != 9:
        return False

    match = re.match(r"^(\d{8})([A-Z])$", cleaned)
    if not match:
        return False

    number = int(match.group(1))
    letter = match.group(2)
    return letter == _DNI_LETTERS[number % 23]


def validate_spanish_nie(text: str) -> bool:
    """Validate Spanish NIE (N\u00famero de Identidad de Extranjero).

    Format: X/Y/Z prefix + 7 digits + 1 check letter.
    The prefix is replaced with 0/1/2, then the same DNI mod-23 check applies.

    Args:
        text: NIE string (may contain spaces)

    Returns:
        True if valid NIE format and check letter
    """
    cleaned = re.sub(r"\s", "", text).upper()

    if len(cleaned) != 9:
        return False

    match = re.match(r"^([XYZ])(\d{7})([A-Z])$", cleaned)
    if not match:
        return False

    prefix_map = {"X": "0", "Y": "1", "Z": "2"}
    prefix = match.group(1)
    digits = match.group(2)
    letter = match.group(3)

    number = int(prefix_map[prefix] + digits)
    return letter == _DNI_LETTERS[number % 23]


def validate_dutch_bsn(text: str) -> bool:
    """Validate Dutch BSN (Burgerservicenummer).

    The BSN uses the Elfproef checksum over 9 digits. Legacy 8-digit values are
    left-padded with a zero for validation.

    Args:
        text: BSN string (may contain spaces or separators)

    Returns:
        True if the BSN passes the checksum test
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) not in (8, 9):
        return False

    digits = digits.zfill(9)
    if digits == "000000000":
        return False

    weights = [9, 8, 7, 6, 5, 4, 3, 2, -1]
    checksum = sum(int(digit) * weight for digit, weight in zip(digits, weights))
    return checksum % 11 == 0


# Verhoeff tables for Aadhaar checksum validation
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def validate_aadhaar(text: str) -> bool:
    """Validate Indian Aadhaar number using the Verhoeff checksum.

    Aadhaar is a 12-digit unique identity number. The last digit is a
    Verhoeff check digit.

    Args:
        text: Aadhaar string (may contain spaces or separators)

    Returns:
        True if the Aadhaar passes the Verhoeff checksum
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 12:
        return False

    # First digit cannot be 0 or 1
    if digits[0] in ("0", "1"):
        return False

    # Verhoeff checksum
    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


_CHINESE_RESIDENT_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_CHINESE_RESIDENT_ID_CHECK_DIGITS = "10X98765432"


def validate_chinese_resident_identity_card(text: str) -> bool:
    """Validate a mainland China 18-digit resident identity card number.

    The second-generation resident identity card stores a six-digit address
    code, an eight-digit Gregorian birth date, a three-digit sequence, and a
    MOD 11-2 checksum digit. OpenMed validates only these offline structural
    and checksum properties; it does not bundle or query a region registry.
    """
    cleaned = re.sub(r"[\s-]", "", text).upper()
    if re.fullmatch(r"\d{17}[\dX]", cleaned) is None:
        return False

    if cleaned[:6] == "000000":
        return False

    try:
        year = int(cleaned[6:10])
        month = int(cleaned[10:12])
        day = int(cleaned[12:14])
    except ValueError:
        return False

    import calendar

    if month < 1 or month > 12 or day < 1:
        return False
    try:
        if day > calendar.monthrange(year, month)[1]:
            return False
    except (ValueError, calendar.IllegalMonthError):
        return False

    total = sum(
        int(digit) * weight
        for digit, weight in zip(cleaned[:17], _CHINESE_RESIDENT_ID_WEIGHTS)
    )
    return cleaned[-1] == _CHINESE_RESIDENT_ID_CHECK_DIGITS[total % 11]


def validate_portuguese_cpf(text: str) -> bool:
    """Validate Brazilian CPF number.

    CPF is an 11-digit taxpayer identifier. The final two digits are
    check digits calculated from the first nine and ten digits.

    Args:
        text: CPF string (may contain dots and hyphen)

    Returns:
        True if the CPF passes format and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False
    if len(set(digits)) == 1:
        return False

    numbers = [int(digit) for digit in digits]

    first_sum = sum(numbers[i] * (10 - i) for i in range(9))
    first_check = (first_sum * 10) % 11
    if first_check == 10:
        first_check = 0

    second_sum = sum(numbers[i] * (11 - i) for i in range(10))
    second_check = (second_sum * 10) % 11
    if second_check == 10:
        second_check = 0

    return numbers[9] == first_check and numbers[10] == second_check


def validate_portuguese_cnpj(text: str) -> bool:
    """Validate Brazilian CNPJ number.

    CNPJ is a 14-digit business identifier. The final two digits use the
    standard weighted modulo-11 checksum.

    Args:
        text: CNPJ string (may contain dots, slash, and hyphen)

    Returns:
        True if the CNPJ passes format and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 14:
        return False
    if len(set(digits)) == 1:
        return False

    numbers = [int(digit) for digit in digits]

    def calculate_check(values: List[int], weights: List[int]) -> int:
        remainder = sum(value * weight for value, weight in zip(values, weights)) % 11
        return 0 if remainder < 2 else 11 - remainder

    first_check = calculate_check(
        numbers[:12],
        [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2],
    )
    second_check = calculate_check(
        numbers[:13],
        [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2],
    )

    return numbers[12] == first_check and numbers[13] == second_check


def validate_turkish_tckn(text: str) -> bool:
    """Validate Turkish T.C. Kimlik No (TCKN).

    The Turkish national identity number is 11 digits. The first digit cannot
    be zero; digit 10 and digit 11 are checksum digits derived from the first
    nine and first ten digits.

    Args:
        text: TCKN string (may contain spaces)

    Returns:
        True if the TCKN passes format and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False
    if digits[0] == "0":
        return False

    numbers = [int(digit) for digit in digits]
    odd_sum = sum(numbers[i] for i in (0, 2, 4, 6, 8))
    even_sum = sum(numbers[i] for i in (1, 3, 5, 7))
    tenth = ((odd_sum * 7) - even_sum) % 10
    eleventh = sum(numbers[:10]) % 10

    return numbers[9] == tenth and numbers[10] == eleventh


def validate_israeli_teudat_zehut(text: str) -> bool:
    """Validate an Israeli Teudat Zehut identity number.

    The Teudat Zehut is a numeric identifier validated with a Luhn-style
    checksum. Values shorter than nine digits are left-padded with zeros before
    applying alternating 1/2 weights from left to right.

    Args:
        text: Teudat Zehut string (may contain spaces or hyphens)

    Returns:
        True if the identifier passes format and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if not 1 <= len(digits) <= 9:
        return False

    digits = digits.zfill(9)
    if digits == "000000000":
        return False

    total = 0
    for index, digit in enumerate(digits):
        product = int(digit) * (1 if index % 2 == 0 else 2)
        total += product if product < 10 else (product // 10) + (product % 10)

    return total % 10 == 0


def validate_indonesian_nik(text: str) -> bool:
    """Validate Indonesian NIK (Nomor Induk Kependudukan) structure.

    NIK is a 16-digit civil-registration identifier. This validator checks the
    stable structural parts OpenMed can verify without bundling a restricted
    registry:

    - PP: province code, using Indonesia's numeric 11-94 province-code range.
    - RR: regency/city code, non-zero two-digit shape.
    - DD: district code, non-zero two-digit shape.
    - DDMMYY: embedded birth date. Female identifiers add 40 to the day.
    - SSSS: non-zero sequence number.

    Args:
        text: NIK string (may contain spaces or separators).

    Returns:
        True if the NIK has a valid structure and decodable birth date.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 16:
        return False

    province = int(digits[0:2])
    regency = int(digits[2:4])
    district = int(digits[4:6])
    if not (11 <= province <= 94 and 1 <= regency <= 99 and 1 <= district <= 99):
        return False

    raw_day = int(digits[6:8])
    month = int(digits[8:10])
    year = int(digits[10:12])
    serial = int(digits[12:16])

    day = raw_day - 40 if raw_day > 40 else raw_day
    if not (1 <= day <= 31 and 1 <= month <= 12 and serial > 0):
        return False

    import calendar

    # NIK stores a two-digit year without a century. Accept the date if it is
    # possible in either common civil-registration century.
    try:
        return any(
            day <= calendar.monthrange(century + year, month)[1]
            for century in (1900, 2000)
        )
    except (ValueError, calendar.IllegalMonthError):
        return False


def validate_thai_national_id(text: str) -> bool:
    """Validate Thai 13-digit national ID with its mod-11 checksum.

    The check digit is calculated from the first 12 digits weighted 13..2:
    ``check = (11 - sum % 11) % 10``.

    Args:
        text: Thai national ID string (may contain spaces or hyphens)

    Returns:
        True if the ID has valid shape and checksum.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 13:
        return False
    if digits[0] == "0":
        return False

    numbers = [int(digit) for digit in digits]
    total = sum(weight * value for weight, value in zip(range(13, 1, -1), numbers[:12]))
    check = (11 - total % 11) % 10

    return numbers[12] == check


def validate_malaysian_mykad(text: str) -> bool:
    """Validate Malaysian MyKad / NRIC structure.

    MyKad values are 12 digits, commonly written as ``YYMMDD-PB-XXXX``. OpenMed
    validates only offline structural properties: an embedded birth date, a
    non-zero place-of-birth code, and a non-zero serial. It does not bundle or
    query Malaysian registry data.

    Args:
        text: MyKad string, with or without dashes.

    Returns:
        True when the value has a plausible MyKad structure and embedded date.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 12:
        return False

    year_suffix = int(digits[0:2])
    month = int(digits[2:4])
    day = int(digits[4:6])
    place_code = int(digits[6:8])
    serial = int(digits[8:12])

    if place_code == 0 or serial == 0:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False

    import calendar

    try:
        return any(
            day <= calendar.monthrange(century + year_suffix, month)[1]
            for century in (1900, 2000)
        )
    except (ValueError, calendar.IllegalMonthError):
        return False


def _matches_digit_grouping(text: str, group_sizes: tuple[int, ...]) -> bool:
    """Return whether ``text`` has contiguous digits or the requested grouping."""
    stripped = text.strip()
    total_digits = sum(group_sizes)
    if re.fullmatch(rf"\d{{{total_digits}}}", stripped):
        return True

    grouped_pattern = r"[\s-]+".join(rf"\d{{{size}}}" for size in group_sizes)
    return re.fullmatch(grouped_pattern, stripped) is not None


def validate_philsys_psn(text: str) -> bool:
    """Validate Philippine PhilSys PSN structure.

    PhilSys PSNs are 12 digits, commonly written as ``NNNN-NNNN-NNNN``.
    OpenMed performs offline structural validation only: expected length,
    expected grouping when separators are present, and non-trivial digits.
    It does not query Philippine registry data.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 12:
        return False
    if len(set(digits)) == 1:
        return False
    return _matches_digit_grouping(text, (4, 4, 4))


def validate_philhealth_pin(text: str) -> bool:
    """Validate Philippine PhilHealth Identification Number structure.

    PhilHealth PINs are 12 digits, commonly written as ``NN-NNNNNNNNN-N``.
    OpenMed checks only local structural properties: expected length,
    expected grouping when separators are present, and non-trivial groups.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 12:
        return False
    if len(set(digits)) == 1:
        return False
    if digits[:2] == "00" or digits[2:11] == "000000000":
        return False
    return _matches_digit_grouping(text, (2, 9, 1))


def _danish_cpr_candidate_years(
    year_suffix: int, century_digit: int
) -> tuple[int, ...]:
    """Return possible birth years encoded by a Danish CPR century digit."""
    if 0 <= century_digit <= 3:
        return (1900 + year_suffix,)
    if century_digit in (4, 9):
        if year_suffix <= 36:
            return (2000 + year_suffix,)
        return (1900 + year_suffix,)
    if 5 <= century_digit <= 8:
        if year_suffix <= 57:
            return (2000 + year_suffix,)
        return (1800 + year_suffix,)
    return ()


def validate_danish_cpr(text: str) -> bool:
    """Validate Danish CPR/personnummer structure.

    CPR values are 10 digits, commonly written as ``DDMMYY-SSSS``. The first
    six digits encode date of birth, and the first serial digit disambiguates
    the century. OpenMed intentionally does not require the historical
    modulus-11 checksum because Danish CPR numbers without that check have been
    validly issued since 2007.
    """
    stripped = text.strip()
    if re.fullmatch(r"\d{6}(?:[-\s]?\d{4})", stripped) is None:
        return False

    digits = re.sub(r"[^0-9]", "", stripped)
    day = int(digits[0:2])
    month = int(digits[2:4])
    year_suffix = int(digits[4:6])
    century_digit = int(digits[6])
    serial = int(digits[6:10])

    if serial == 0:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False

    import calendar

    for year in _danish_cpr_candidate_years(year_suffix, century_digit):
        try:
            if day <= calendar.monthrange(year, month)[1]:
                return True
        except (ValueError, calendar.IllegalMonthError):
            continue
    return False


def validate_polish_pesel(text: str) -> bool:
    """Validate Polish PESEL number.

    The PESEL is an 11-digit number with an embedded birth date and a
    weighted checksum.  Structure: YYMMDDZZZSC.

    - YYMMDD: date of birth.  Month has century offsets: +20 for 2000s,
      +40 for 2100s, +60 for 2200s, +80 for 1800s.
    - ZZZ: serial number.
    - S: check digit.

    Checksum weights over the first ten digits: 1, 3, 7, 9, 1, 3, 7, 9, 1, 3.
    Check digit = (10 - sum mod 10) mod 10.

    Args:
        text: PESEL string (may contain spaces or hyphens)

    Returns:
        True if the PESEL passes format, date, and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False

    numbers = [int(d) for d in digits]

    # --- checksum ---
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(w * n for w, n in zip(weights, numbers[:10]))
    if numbers[10] != (10 - total % 10) % 10:
        return False

    # --- embedded birth date ---
    year = numbers[0] * 10 + numbers[1]
    month_raw = numbers[2] * 10 + numbers[3]
    day = numbers[4] * 10 + numbers[5]

    # Century offset encoded in the month tens digit.
    if month_raw > 92 or month_raw == 0:
        return False

    if month_raw > 80:
        year += 1800
        month = month_raw - 80
    elif month_raw > 60:
        year += 2200
        month = month_raw - 60
    elif month_raw > 40:
        year += 2100
        month = month_raw - 40
    elif month_raw > 20:
        year += 2000
        month = month_raw - 20
    else:
        year += 1900
        month = month_raw

    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False

    import calendar

    try:
        max_day = calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False
    if day > max_day:
        return False

    return True


def validate_latvian_personas_kods(text: str) -> bool:
    """Validate Latvian personas kods.

    Legacy values encode birth date as ``DDMMYY-CNNNQ``. New values issued
    from 2017 use an opaque ``32`` prefix, but both formats keep an 11-digit
    modulo-11 check digit.
    """

    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False

    numbers = [int(digit) for digit in digits]
    check_digit = _latvian_personas_kods_check_digit(numbers[:10])
    if numbers[10] != check_digit:
        return False

    if digits.startswith("32"):
        return True

    day = int(digits[0:2])
    month = int(digits[2:4])
    year_suffix = int(digits[4:6])
    century_digit = int(digits[6])
    if century_digit not in (0, 1, 2):
        return False

    import calendar

    year = 1800 + century_digit * 100 + year_suffix
    try:
        return day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


def _latvian_personas_kods_check_digit(digits: list[int]) -> int:
    """Return the Latvian personas kods check digit for the first 10 digits."""
    weights = (1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
    return (
        (1101 - sum(weight * digit for weight, digit in zip(weights, digits))) % 11 % 10
    )


def validate_korean_rrn(text: str) -> bool:
    """Validate South Korean Resident Registration Number (RRN).

    The RRN is a 13-digit number in the format YYMMDDGXXXXXX:

    - YYMMDD: date of birth.
    - G (position 6, zero-indexed): century and gender code.
        1/2 = 1900s (1=male, 2=female),
        3/4 = 2000s (3=male, 4=female),
        5/6 = 1900s foreign residents,
        7/8 = 2000s foreign residents,
        9/0 = 1800s (9=male, 0=female).
    - Positions 7-11: serial number.
    - Position 12: check digit.

    Checksum weights: 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5.
    Check digit = (11 - (sum mod 11)) mod 10.

    Args:
        text: RRN string (may contain hyphens or spaces)

    Returns:
        True if the RRN passes format, date, and checksum validation.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 13:
        return False

    numbers = [int(d) for d in digits]

    # --- century/gender code ---
    gender_code = numbers[6]
    century_map = {
        1: 1900,
        2: 1900,
        3: 2000,
        4: 2000,
        5: 1900,
        6: 1900,
        7: 2000,
        8: 2000,
        9: 1800,
        0: 1800,
    }
    if gender_code not in century_map:
        return False
    century = century_map[gender_code]

    # --- embedded birth date ---
    year = century + numbers[0] * 10 + numbers[1]
    month = numbers[2] * 10 + numbers[3]
    day = numbers[4] * 10 + numbers[5]

    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False

    import calendar

    try:
        max_day = calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False
    if day > max_day:
        return False

    # --- checksum ---
    weights = (2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5)
    total = sum(w * n for w, n in zip(weights, numbers[:12]))
    check = (11 - total % 11) % 10

    return numbers[12] == check


def validate_czechoslovak_rodne_cislo(text: str) -> bool:
    """Validate a Czech/Slovak rodne cislo birth number.

    Modern Czech and Slovak birth numbers use ten digits in the shape
    ``YYMMDDXXXX`` with a modulo-11 checksum over the whole value. Female
    identifiers add 50 to the birth month; overflow series may add 20 for
    men or 70 for women.

    Args:
        text: Birth number string, optionally containing a slash, spaces, or
            hyphen separators.

    Returns:
        True if the identifier has a decodable birth date and modulo-11 check.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 10:
        return False
    if int(digits) % 11 != 0:
        return False

    year_suffix = int(digits[0:2])
    encoded_month = int(digits[2:4])
    day = int(digits[4:6])

    overflow_series = False
    if 1 <= encoded_month <= 12:
        month = encoded_month
    elif 21 <= encoded_month <= 32:
        month = encoded_month - 20
        overflow_series = True
    elif 51 <= encoded_month <= 62:
        month = encoded_month - 50
    elif 71 <= encoded_month <= 82:
        month = encoded_month - 70
        overflow_series = True
    else:
        return False

    if day < 1 or day > 31:
        return False

    import calendar

    # Ten-digit rodne cislo values were introduced in 1954. The two-digit year
    # is century-ambiguous, so accept either plausible civil-registration
    # century while keeping the overflow series in the post-2004 era.
    for century in (1900, 2000):
        year = century + year_suffix
        if year < 1954:
            continue
        if overflow_series and year < 2004:
            continue
        try:
            if day <= calendar.monthrange(year, month)[1]:
                return True
        except (ValueError, calendar.IllegalMonthError):
            continue
    return False


# ---------------------------------------------------------------------------
# Language-specific month names (for date parsing/formatting)
# ---------------------------------------------------------------------------

LANGUAGE_MONTH_NAMES: Dict[str, List[str]] = {
    "en": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "fr": [
        "janvier",
        "f\u00e9vrier",
        "mars",
        "avril",
        "mai",
        "juin",
        "juillet",
        "ao\u00fbt",
        "septembre",
        "octobre",
        "novembre",
        "d\u00e9cembre",
    ],
    "de": [
        "Januar",
        "Februar",
        "M\u00e4rz",
        "April",
        "Mai",
        "Juni",
        "Juli",
        "August",
        "September",
        "Oktober",
        "November",
        "Dezember",
    ],
    "it": [
        "gennaio",
        "febbraio",
        "marzo",
        "aprile",
        "maggio",
        "giugno",
        "luglio",
        "agosto",
        "settembre",
        "ottobre",
        "novembre",
        "dicembre",
    ],
    "es": [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ],
    "pt": [
        "janeiro",
        "fevereiro",
        "mar\u00e7o",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ],
    "nl": [
        "januari",
        "februari",
        "maart",
        "april",
        "mei",
        "juni",
        "juli",
        "augustus",
        "september",
        "oktober",
        "november",
        "december",
    ],
    "hi": [
        "\u091c\u0928\u0935\u0930\u0940",
        "\u092b\u093c\u0930\u0935\u0930\u0940",
        "\u092e\u093e\u0930\u094d\u091a",
        "\u0905\u092a\u094d\u0930\u0948\u0932",
        "\u092e\u0908",
        "\u091c\u0942\u0928",
        "\u091c\u0941\u0932\u093e\u0908",
        "\u0905\u0917\u0938\u094d\u0924",
        "\u0938\u093f\u0924\u0902\u092c\u0930",
        "\u0905\u0915\u094d\u091f\u0942\u092c\u0930",
        "\u0928\u0935\u0902\u092c\u0930",
        "\u0926\u093f\u0938\u0902\u092c\u0930",
    ],
    "te": [
        "\u0c1c\u0c28\u0c35\u0c30\u0c3f",
        "\u0c2b\u0c3f\u0c2c\u0c4d\u0c30\u0c35\u0c30\u0c3f",
        "\u0c2e\u0c3e\u0c30\u0c4d\u0c1a\u0c3f",
        "\u0c0f\u0c2a\u0c4d\u0c30\u0c3f\u0c32\u0c4d",
        "\u0c2e\u0c47",
        "\u0c1c\u0c42\u0c28\u0c4d",
        "\u0c1c\u0c42\u0c32\u0c48",
        "\u0c06\u0c17\u0c38\u0c4d\u0c1f\u0c41",
        "\u0c38\u0c46\u0c2a\u0c4d\u0c1f\u0c46\u0c02\u0c2c\u0c30\u0c4d",
        "\u0c05\u0c15\u0c4d\u0c1f\u0c4b\u0c2c\u0c30\u0c4d",
        "\u0c28\u0c35\u0c02\u0c2c\u0c30\u0c4d",
        "\u0c21\u0c3f\u0c38\u0c46\u0c02\u0c2c\u0c30\u0c4d",
    ],
    "ar": [
        "\u064a\u0646\u0627\u064a\u0631",
        "\u0641\u0628\u0631\u0627\u064a\u0631",
        "\u0645\u0627\u0631\u0633",
        "\u0623\u0628\u0631\u064a\u0644",
        "\u0645\u0627\u064a\u0648",
        "\u064a\u0648\u0646\u064a\u0648",
        "\u064a\u0648\u0644\u064a\u0648",
        "\u0623\u063a\u0633\u0637\u0633",
        "\u0633\u0628\u062a\u0645\u0628\u0631",
        "\u0623\u0643\u062a\u0648\u0628\u0631",
        "\u0646\u0648\u0641\u0645\u0628\u0631",
        "\u062f\u064a\u0633\u0645\u0628\u0631",
    ],
    "he": [
        "\u05d9\u05e0\u05d5\u05d0\u05e8",
        "\u05e4\u05d1\u05e8\u05d5\u05d0\u05e8",
        "\u05de\u05e8\u05e5",
        "\u05d0\u05e4\u05e8\u05d9\u05dc",
        "\u05de\u05d0\u05d9",
        "\u05d9\u05d5\u05e0\u05d9",
        "\u05d9\u05d5\u05dc\u05d9",
        "\u05d0\u05d5\u05d2\u05d5\u05e1\u05d8",
        "\u05e1\u05e4\u05d8\u05de\u05d1\u05e8",
        "\u05d0\u05d5\u05e7\u05d8\u05d5\u05d1\u05e8",
        "\u05e0\u05d5\u05d1\u05de\u05d1\u05e8",
        "\u05d3\u05e6\u05de\u05d1\u05e8",
    ],
    "ja": [
        "1\u6708",
        "2\u6708",
        "3\u6708",
        "4\u6708",
        "5\u6708",
        "6\u6708",
        "7\u6708",
        "8\u6708",
        "9\u6708",
        "10\u6708",
        "11\u6708",
        "12\u6708",
    ],
    "tr": [
        "Ocak",
        "\u015eubat",
        "Mart",
        "Nisan",
        "May\u0131s",
        "Haziran",
        "Temmuz",
        "A\u011fustos",
        "Eyl\u00fcl",
        "Ekim",
        "Kas\u0131m",
        "Aral\u0131k",
    ],
    "id": [
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ],
    "ms": [
        "Januari",
        "Februari",
        "Mac",
        "April",
        "Mei",
        "Jun",
        "Julai",
        "Ogos",
        "September",
        "Oktober",
        "November",
        "Disember",
    ],
    "th": [
        "มกราคม",
        "กุมภาพันธ์",
        "มีนาคม",
        "เมษายน",
        "พฤษภาคม",
        "มิถุนายน",
        "กรกฎาคม",
        "สิงหาคม",
        "กันยายน",
        "ตุลาคม",
        "พฤศจิกายน",
        "ธันวาคม",
    ],
    "ko": [
        "1\uc6d4",
        "2\uc6d4",
        "3\uc6d4",
        "4\uc6d4",
        "5\uc6d4",
        "6\uc6d4",
        "7\uc6d4",
        "8\uc6d4",
        "9\uc6d4",
        "10\uc6d4",
        "11\uc6d4",
        "12\uc6d4",
    ],
}


# ---------------------------------------------------------------------------
# ICAO 9303 machine-readable zone (MRZ) validation
# ---------------------------------------------------------------------------

_MRZ_LINE_RE = re.compile(r"[A-Z0-9<]+")


def _mrz_char_value(char: str) -> int:
    """ICAO 9303 character value: digits 0-9, A-Z = 10-35, filler '<' = 0."""
    if char == "<":
        return 0
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    return -1


def _mrz_check_digit(field: str) -> Optional[int]:
    """Compute the ICAO 9303 modulo-10 check digit (7-3-1 weighting)."""
    weights = (7, 3, 1)
    total = 0
    for index, char in enumerate(field):
        value = _mrz_char_value(char)
        if value < 0:
            return None
        total += value * weights[index % 3]
    return total % 10


def _mrz_check_ok(field: str, check_char: str) -> bool:
    expected = _mrz_check_digit(field)
    if expected is None:
        return False
    if check_char.isdigit():
        return expected == int(check_char)
    if check_char == "<":  # filler check digit for all-filler fields
        return expected == 0
    return False


def _mrz_lines(text: str, width: int, count: int) -> Optional[List[str]]:
    """Return the ``count`` MRZ lines of exactly ``width`` chars, else None."""
    lines = [line.strip() for line in text.strip().splitlines()]
    candidates = [
        line for line in lines if len(line) == width and _MRZ_LINE_RE.fullmatch(line)
    ]
    if len(candidates) != count:
        return None
    return candidates


def validate_mrz_td3(text: str) -> bool:
    """Validate an ICAO 9303 TD3 (passport) MRZ: two 44-character lines.

    Confirms the document-number, date-of-birth, expiry, personal-number and
    composite modulo-10 check digits. Date fields must be numeric (YYMMDD).
    """
    lines = _mrz_lines(text, 44, 2)
    if lines is None:
        return False
    line2 = lines[1]
    checks = (
        (line2[0:9], line2[9]),  # document number
        (line2[13:19], line2[19]),  # date of birth
        (line2[21:27], line2[27]),  # expiry date
        (line2[28:42], line2[42]),  # optional personal number
        (line2[0:10] + line2[13:20] + line2[21:43], line2[43]),  # composite
    )
    if not all(_mrz_check_ok(field, check) for field, check in checks):
        return False
    return line2[13:19].isdigit() and line2[21:27].isdigit()


def validate_mrz_td1(text: str) -> bool:
    """Validate an ICAO 9303 TD1 (ID card) MRZ: three 30-character lines.

    Confirms the document-number, date-of-birth, expiry and composite
    modulo-10 check digits. Date fields must be numeric (YYMMDD).
    """
    lines = _mrz_lines(text, 30, 3)
    if lines is None:
        return False
    line1, line2 = lines[0], lines[1]
    composite = line1[5:30] + line2[0:7] + line2[8:15] + line2[18:29]
    checks = (
        (line1[5:14], line1[14]),  # document number
        (line2[0:6], line2[6]),  # date of birth
        (line2[8:14], line2[14]),  # expiry date
        (composite, line2[29]),  # composite
    )
    if not all(_mrz_check_ok(field, check) for field, check in checks):
        return False
    return line2[0:6].isdigit() and line2[8:14].isdigit()


_MRZ_FILLER = "<"
_MRZ_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_MRZ_ALPHANUM = _MRZ_LETTERS + "0123456789"


def _mrz_pad(text: str, length: int) -> str:
    return (text + _MRZ_FILLER * length)[:length]


def generate_mrz_td3(rng=None) -> str:
    """Generate a synthetic but check-digit-valid TD3 (passport) MRZ block."""
    import random as _random

    rng = rng or _random.Random()
    digits = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))  # noqa: E731
    country = "".join(rng.choice(_MRZ_LETTERS) for _ in range(3))
    docnum = "".join(rng.choice(_MRZ_ALPHANUM) for _ in range(9))
    dob, expiry = digits(6), digits(6)
    sex = rng.choice("MF<")
    personal = _MRZ_FILLER * 14
    partial = (
        f"{docnum}{_mrz_check_digit(docnum)}{country}"
        f"{dob}{_mrz_check_digit(dob)}{sex}"
        f"{expiry}{_mrz_check_digit(expiry)}"
        f"{personal}{_mrz_check_digit(personal)}"
    )
    composite = partial[0:10] + partial[13:20] + partial[21:43]
    line2 = partial + str(_mrz_check_digit(composite))
    line1 = "P<" + country + _mrz_pad("SPECIMEN<<TRAVELLER", 39)
    return f"{line1}\n{line2}"


def generate_mrz_td1(rng=None) -> str:
    """Generate a synthetic but check-digit-valid TD1 (ID-card) MRZ block."""
    import random as _random

    rng = rng or _random.Random()
    digits = lambda n: "".join(str(rng.randint(0, 9)) for _ in range(n))  # noqa: E731
    country = "".join(rng.choice(_MRZ_LETTERS) for _ in range(3))
    docnum = "".join(rng.choice(_MRZ_ALPHANUM) for _ in range(9))
    line1 = _mrz_pad(f"I<{country}{docnum}{_mrz_check_digit(docnum)}", 30)
    dob, expiry = digits(6), digits(6)
    sex = rng.choice("MF<")
    middle = (
        f"{dob}{_mrz_check_digit(dob)}{sex}"
        f"{expiry}{_mrz_check_digit(expiry)}{country}{_MRZ_FILLER * 11}"
    )
    composite = line1[5:30] + middle[0:7] + middle[8:15] + middle[18:29]
    line2 = middle + str(_mrz_check_digit(composite))
    line3 = _mrz_pad("SPECIMEN<<TRAVELLER", 30)
    return f"{line1}\n{line2}\n{line3}"


# ---------------------------------------------------------------------------
# Language-specific PII patterns
# ---------------------------------------------------------------------------

from .pii_entity_merger import PIIPattern  # noqa: E402

_UK_ENGLISH_PII_PATTERNS: List[PIIPattern] = [
    # UK NHS Number (10 digits, optional 3-3-4 spacing, Modulus 11 check).
    PIIPattern(
        r"\b\d{3}\s?\d{3}\s?\d{4}\b",
        "national_id",
        priority=11,
        base_score=0.45,
        context_words=[
            "nhs",
            "nhs number",
            "nhs no",
            "patient number",
            "health identifier",
        ],
        context_boost=0.45,
        validator=validate_uk_nhs_number,
    ),
    # UK National Insurance Number (NINO).
    PIIPattern(
        r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "national insurance",
            "national insurance number",
            "nino",
            "ni number",
        ],
        context_boost=0.45,
        validator=validate_uk_nino,
        flags=re.IGNORECASE,
    ),
]

_CANADIAN_ENGLISH_PII_PATTERNS: List[PIIPattern] = [
    # Ontario (OHIP) health card: 10 digits beginning with 1-9 (4-3-3 spacing)
    # plus an optional one- or two-letter version code, Luhn-checked. Health
    # identifier. Checked before the SIN so the longer 10-digit match wins.
    PIIPattern(
        r"\b[1-9]\d{3}(?P<ohip_sep>[ -]?)\d{3}(?P=ohip_sep)\d{3}"
        r"(?:[ -]?[A-Za-z]{1,2})?\b",
        "national_id",
        priority=12,
        base_score=0.45,
        context_words=[
            "health card",
            "health card number",
            "ohip",
            "ohip number",
            "ontario health",
            "health number",
            "health identifier",
        ],
        context_boost=0.45,
        validator=validate_ontario_health_card,
        flags=re.IGNORECASE,
    ),
    # British Columbia Personal Health Number (PHN): 10 digits beginning with 9
    # (4-3-3 spacing), weighted mod-11. Health identifier.
    PIIPattern(
        r"\b9\d{3}[\s-]?\d{3}[\s-]?\d{3}\b",
        "national_id",
        priority=12,
        base_score=0.45,
        context_words=[
            "phn",
            "personal health number",
            "bc phn",
            "health number",
            "health identifier",
        ],
        context_boost=0.45,
        validator=validate_bc_phn,
    ),
    # Canadian Social Insurance Number (SIN): 9 digits (3-3-3 spacing), Luhn.
    PIIPattern(
        r"\b\d{3}[\s-]?\d{3}[\s-]?\d{3}\b",
        "national_id",
        priority=11,
        base_score=0.45,
        context_words=[
            "sin",
            "social insurance",
            "social insurance number",
            "numero d'assurance sociale",
            "nas",
        ],
        context_boost=0.45,
        validator=validate_canadian_sin,
    ),
]


_AU_ENGLISH_PII_PATTERNS: List[PIIPattern] = [
    # Australian Medicare card number (exactly 10 digits, ``NNNN NNNNN N``;
    # weighted mod-10 checksum on the first eight digits guards it).
    PIIPattern(
        r"\b\d{4}\s?\d{5}\s?\d\b(?!\s?/?\s?\d)",
        "national_id",
        priority=12,
        base_score=0.45,
        context_words=[
            "medicare",
            "medicare number",
            "medicare card",
            "medicare no",
            "health identifier",
        ],
        context_boost=0.45,
        validator=validate_australian_medicare,
    ),
    # Australian Tax File Number (TFN, 8-9 digits, ``NNN NNN NNN`` spacing;
    # weighted mod-11 checksum).
    PIIPattern(
        r"\b\d{3}\s?\d{3}\s?\d{2,3}\b",
        "national_id",
        priority=11,
        base_score=0.45,
        context_words=[
            "tfn",
            "tax file number",
            "tax file no",
        ],
        context_boost=0.45,
        validator=validate_australian_tfn,
    ),
]


# Language-agnostic ICAO 9303 machine-readable-zone patterns, guarded by the
# check-digit validators and always included in the universal base set.
MRZ_PII_PATTERNS: List[PIIPattern] = [
    # TD3 passport MRZ: two 44-character lines.
    PIIPattern(
        r"^[A-Z0-9<]{44}\n[A-Z0-9<]{44}$",
        "passport_mrz",
        priority=15,
        flags=re.MULTILINE,
        base_score=0.7,
        context_words=["passport", "mrz", "machine readable zone"],
        validator=validate_mrz_td3,
    ),
    # TD1 identity-card MRZ: three 30-character lines.
    PIIPattern(
        r"^[A-Z0-9<]{30}\n[A-Z0-9<]{30}\n[A-Z0-9<]{30}$",
        "passport_mrz",
        priority=15,
        flags=re.MULTILINE,
        base_score=0.7,
        context_words=["passport", "identity card", "mrz"],
        validator=validate_mrz_td1,
    ),
]

_FRENCH_PII_PATTERNS: List[PIIPattern] = [
    # French dates DD/MM/YYYY
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "n\u00e9",
            "n\u00e9e",
            "naissance",
            "date de naissance",
            "dob",
            "d\u00e9c\u00e8s",
            "d\u00e9c\u00e9d\u00e9",
            "admis",
            "sorti",
        ],
        context_boost=0.3,
    ),
    # French dates with month names
    PIIPattern(
        r"\b\d{1,2}\s+(?:janvier|f\u00e9vrier|mars|avril|mai|juin|juillet|ao\u00fbt|septembre|octobre|novembre|d\u00e9cembre)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "n\u00e9",
            "n\u00e9e",
            "naissance",
            "date de naissance",
            "d\u00e9c\u00e8s",
            "admis",
            "sorti",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # French phone numbers: +33, 06, 07
    PIIPattern(
        r"(?<!\w)(?:\+33\s?|0)[1-9](?:[\s.-]?\d{2}){4}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "t\u00e9l\u00e9phone",
            "t\u00e9l",
            "portable",
            "mobile",
            "num\u00e9ro",
            "appeler",
            "contact",
            "fax",
        ],
        context_boost=0.3,
    ),
    # French NIR/INSEE (15 characters, possibly spaced; includes Corsica 2A/2B)
    PIIPattern(
        r"\b[12]\s?\d{2}\s?\d{2}\s?(?:\d{2}|2[ABab])\s?\d{3}\s?\d{3}\s?\d{2}\b",
        "national_id",
        priority=10,
        base_score=0.55,
        context_words=[
            "nir",
            "insee",
            "s\u00e9curit\u00e9 sociale",
            "num\u00e9ro de s\u00e9curit\u00e9",
        ],
        context_boost=0.45,
        validator=validate_french_nir,
    ),
    # French street addresses
    PIIPattern(
        r"\b\d{1,5}\s+(?:rue|boulevard|avenue|place|impasse|all\u00e9e|chemin|passage|quai)\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+(?:\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+)*\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "adresse",
            "domicile",
            "r\u00e9side",
            "habite",
            "situ\u00e9",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # French postal code (valid department prefixes 01-95 + DOM-TOM 971-976)
    PIIPattern(
        r"\b(?:(?:0[1-9]|[1-8]\d|9[0-5])\d{3}|97[1-6]\d{2})\b",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "code postal",
            "cp",
            "cedex",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
]

_GERMAN_PII_PATTERNS: List[PIIPattern] = [
    # German dates DD.MM.YYYY
    PIIPattern(
        r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "Geburtsdatum",
            "geboren",
            "geb",
            "verstorben",
            "aufgenommen",
            "entlassen",
            "Datum",
        ],
        context_boost=0.3,
    ),
    # German dates with month names
    PIIPattern(
        r"\b\d{1,2}\.?\s+(?:Januar|Februar|M\u00e4rz|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "Geburtsdatum",
            "geboren",
            "geb",
            "verstorben",
            "aufgenommen",
            "entlassen",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # German phone numbers: +49, 0xxx (require at least 7 digits after prefix)
    PIIPattern(
        r"(?<!\w)(?:\+49\s?|0)\d{2,4}[\s/-]?\d{4,8}\b",
        "phone_number",
        priority=8,
        base_score=0.5,
        context_words=[
            "Telefon",
            "Tel",
            "Handy",
            "Mobil",
            "Fax",
            "Rufnummer",
            "Nummer",
            "anrufen",
            "Kontakt",
        ],
        context_boost=0.35,
    ),
    # German Steuer-ID (11 digits, first digit non-zero)
    PIIPattern(
        r"\b[1-9]\d{10}\b",
        "national_id",
        priority=9,
        base_score=0.35,
        context_words=[
            "Steuer-ID",
            "Steueridentifikationsnummer",
            "Steuernummer",
            "IdNr",
            "Identifikationsnummer",
        ],
        context_boost=0.6,
        validator=validate_german_steuer_id,
    ),
    # German street addresses
    PIIPattern(
        r"\b[A-Z\u00c4\u00d6\u00dc][a-z\u00e4\u00f6\u00fc\u00df]+(?:stra\u00dfe|strasse|str\.|weg|platz|allee|gasse|ring|damm)\s+\d{1,5}[a-z]?\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "Adresse",
            "Anschrift",
            "wohnhaft",
            "wohnt",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # German PLZ (valid range 01000-99999, excluding 00xxx)
    PIIPattern(
        r"\b(?:0[1-9]|[1-9]\d)\d{3}\b",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "PLZ",
            "Postleitzahl",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
]

_ITALIAN_PII_PATTERNS: List[PIIPattern] = [
    # Italian dates DD/MM/YYYY
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "nato",
            "nata",
            "nascita",
            "data di nascita",
            "decesso",
            "deceduto",
            "ricovero",
            "dimissione",
        ],
        context_boost=0.3,
    ),
    # Italian dates with month names
    PIIPattern(
        r"\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "nato",
            "nata",
            "nascita",
            "data di nascita",
            "decesso",
            "ricovero",
            "dimissione",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Italian phone numbers: +39, 3xx
    PIIPattern(
        r"\b(?:\+39\s?)?3\d{2}[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "telefono",
            "tel",
            "cellulare",
            "mobile",
            "numero",
            "chiamare",
            "contatto",
            "fax",
        ],
        context_boost=0.3,
    ),
    # Italian Codice Fiscale (16 alphanumeric)
    PIIPattern(
        r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b",
        "national_id",
        priority=10,
        base_score=0.7,
        context_words=[
            "codice fiscale",
            "cf",
            "c.f.",
        ],
        context_boost=0.25,
        validator=validate_italian_codice_fiscale,
    ),
    # Italian street addresses
    PIIPattern(
        r"\b(?:via|piazza|corso|viale|vicolo|largo|piazzale|lungomare)\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+(?:\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+)*(?:\s*,?\s*\d{1,5})?\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "indirizzo",
            "domicilio",
            "residente",
            "risiede",
            "abitazione",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # Italian CAP (5 digits)
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "CAP",
            "codice postale",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
]

_SPANISH_PII_PATTERNS: List[PIIPattern] = [
    # Spanish dates DD/MM/YYYY
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "nacido",
            "nacida",
            "nacimiento",
            "fecha de nacimiento",
            "fallecimiento",
            "fallecido",
            "ingreso",
            "alta",
        ],
        context_boost=0.3,
    ),
    # Spanish dates with month names (unique "de" connector)
    PIIPattern(
        r"\b\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+de\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "nacido",
            "nacida",
            "nacimiento",
            "fecha de nacimiento",
            "fallecimiento",
            "ingreso",
            "alta",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Spanish phone numbers: +34, mobile 6xx/7xx, landline 9xx
    PIIPattern(
        r"(?<!\w)(?:\+34\s?)?[679]\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "tel\u00e9fono",
            "tel",
            "m\u00f3vil",
            "celular",
            "n\u00famero",
            "llamar",
            "contacto",
            "fax",
        ],
        context_boost=0.3,
    ),
    # Spanish DNI (8 digits + letter)
    PIIPattern(
        r"\b\d{8}[A-Za-z]\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "dni",
            "documento nacional",
            "identidad",
            "documento de identidad",
        ],
        context_boost=0.4,
        validator=validate_spanish_dni,
    ),
    # Spanish NIE (X/Y/Z + 7 digits + letter)
    PIIPattern(
        r"\b[XYZxyz]\d{7}[A-Za-z]\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "nie",
            "n\u00famero de identidad de extranjero",
            "extranjero",
            "residencia",
        ],
        context_boost=0.4,
        validator=validate_spanish_nie,
    ),
    # Spanish street addresses
    PIIPattern(
        r"\b(?:calle|avenida|paseo|plaza|camino|carretera|ronda|traves\u00eda|glorieta)\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+(?:\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+)*(?:\s*,?\s*\d{1,5})?\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "direcci\u00f3n",
            "domicilio",
            "residente",
            "reside",
            "ubicaci\u00f3n",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # Spanish postal codes (01000–52999)
    PIIPattern(
        r"\b(?:0[1-9]|[1-4]\d|5[0-2])\d{3}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "c\u00f3digo postal",
            "cp",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
]

_PORTUGUESE_PII_PATTERNS: List[PIIPattern] = [
    # Portuguese dates DD/MM/YYYY and DD-MM-YYYY
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "nascido",
            "nascida",
            "nascimento",
            "data de nascimento",
            "internado",
            "internada",
            "admiss\u00e3o",
            "alta",
            "falecimento",
        ],
        context_boost=0.3,
    ),
    # Portuguese dates with month names: 15 de mar\u00e7o de 1985
    PIIPattern(
        r"\b\d{1,2}\s+de\s+(?:janeiro|fevereiro|mar\u00e7o|marco|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "nascido",
            "nascida",
            "nascimento",
            "data de nascimento",
            "internado",
            "internada",
            "admiss\u00e3o",
            "alta",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Portugal and Brazil phone numbers.
    PIIPattern(
        r"(?<!\w)(?:(?:\+351\s?)?9[1236]\d(?:[\s.-]?\d{3}){2}|(?:\+55\s?)?(?:\(?\d{2}\)?[\s.-]?)?9?\d{4}[\s.-]?\d{4})\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telefone",
            "tel",
            "telem\u00f3vel",
            "telemovel",
            "celular",
            "n\u00famero",
            "numero",
            "contato",
            "contacto",
            "fax",
        ],
        context_boost=0.35,
    ),
    # Brazilian CPF (11 digits)
    PIIPattern(
        r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "cpf",
            "cadastro de pessoas f\u00edsicas",
            "cadastro de pessoas fisicas",
            "identifica\u00e7\u00e3o",
            "identificacao",
        ],
        context_boost=0.45,
        validator=validate_portuguese_cpf,
    ),
    # Brazilian CNPJ (14 digits)
    PIIPattern(
        r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "cnpj",
            "cadastro nacional",
            "empresa",
            "pessoa jur\u00eddica",
            "pessoa juridica",
        ],
        context_boost=0.45,
        validator=validate_portuguese_cnpj,
    ),
    # Portuguese street addresses
    PIIPattern(
        r"\b(?:rua|avenida|av\.?|travessa|pra\u00e7a|praca|alameda|estrada|rodovia|largo)\s+(?:[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+|d[aeo]s?)(?:\s+(?:[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]+|d[aeo]s?))*\s*,?\s*\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "endere\u00e7o",
            "endereco",
            "morada",
            "resid\u00eancia",
            "residencia",
            "domic\u00edlio",
            "domicilio",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # Portugal postcode (1200-195) and Brazil CEP (01310-100)
    PIIPattern(
        r"\b(?:\d{4}-\d{3}|\d{5}-?\d{3})\b",
        "postcode",
        priority=6,
        base_score=0.35,
        context_words=[
            "c\u00f3digo postal",
            "codigo postal",
            "cep",
            "endere\u00e7o",
            "endereco",
            "morada",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
    ),
]

_DUTCH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "geboren",
            "geboortedatum",
            "opname",
            "ontslag",
            "datum",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "geboren",
            "geboortedatum",
            "opname",
            "ontslag",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:(?:\+31\s?6|06)[\s.-]?\d{8}|(?:\+31\s?|0)\d{1,3}(?:[\s.-]?\d{2,4}){2,3})\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "telefoon",
            "tel",
            "mobiel",
            "nummer",
            "contact",
            "fax",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{9}\b",
        "national_id",
        priority=9,
        base_score=0.25,
        context_words=[
            "bsn",
            "burgerservicenummer",
            "service nummer",
        ],
        context_boost=0.55,
        validator=validate_dutch_bsn,
    ),
    PIIPattern(
        r"\b[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]*(?:\s+[A-Z\u00c0-\u00ff][a-z\u00e0-\u00ff]*)*(?:straat|laan|weg|plein|gracht|dreef|kade)\s+\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.7,
        context_words=[
            "adres",
            "woonadres",
            "woont",
            "verblijft",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{4}\s?[A-Z]{2}\b",
        "postcode",
        priority=6,
        base_score=0.45,
        context_words=[
            "postcode",
            "post code",
            "adres",
        ],
        context_boost=0.4,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

_HINDI_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u091c\u0928\u094d\u092e",
            "\u091c\u0928\u094d\u092e\u0924\u093f\u0925\u093f",
            "\u092d\u0930\u094d\u0924\u0940",
            "\u0924\u093e\u0930\u0940\u0916",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u091c\u0928\u0935\u0930\u0940|\u092b(?:\u093c)?\u0930\u0935\u0930\u0940|\u092e\u093e\u0930\u094d\u091a|\u0905\u092a\u094d\u0930\u0948\u0932|\u092e\u0908|\u091c\u0942\u0928|\u091c\u0941\u0932\u093e\u0908|\u0905\u0917\u0938\u094d\u0924|\u0938\u093f\u0924\u0902\u092c\u0930|\u0905\u0915\u094d\u091f\u0942\u092c\u0930|\u0928\u0935\u0902\u092c\u0930|\u0926\u093f\u0938\u0902\u092c\u0930)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u091c\u0928\u094d\u092e",
            "\u091c\u0928\u094d\u092e\u0924\u093f\u0925\u093f",
            "\u092d\u0930\u094d\u0924\u0940",
            "\u0924\u093e\u0930\u0940\u0916",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}\b|(?<!\w)(?:\+91[\s-]?)?[6-9]\d{1}[\s.-]?\d{4}[\s.-]?\d{5}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "\u092b\u094b\u0928",
            "\u092e\u094b\u092c\u093e\u0907\u0932",
            "\u0928\u0902\u092c\u0930",
            "\u0938\u0902\u092a\u0930\u094d\u0915",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"\b(?:\d{1,5}\s+)?(?:[\u0900-\u097F]+\s+)*(?:\u0917\u0932\u0940|\u0938\u0921\u093c\u0915|\u0928\u0917\u0930|\u092e\u093e\u0930\u094d\u0917|Road|Street|Nagar)\s*(?:\u0938\u0902\u0916\u094d\u092f\u093e\s*)?\d+[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u092a\u0924\u093e",
            "\u0928\u093f\u0935\u093e\u0938",
            "\u091a\u093f\u0930\u0941\u0928\u093e\u092e\u093e",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{6}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "\u092a\u093f\u0928",
            "\u092a\u093f\u0928\u0915\u094b\u0921",
            "\u0921\u093e\u0915",
            "\u092a\u0924\u093e",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
    # Aadhaar (12 digits, possibly spaced as 4-4-4)
    PIIPattern(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "national_id",
        priority=9,
        base_score=0.4,
        context_words=[
            "\u0906\u0927\u093e\u0930",
            "aadhaar",
            "\u092f\u0942\u0906\u0908\u0921\u0940\u090f\u0906\u0908",
            "uid",
            "\u092a\u0939\u091a\u093e\u0928",
        ],
        context_boost=0.45,
        validator=validate_aadhaar,
    ),
]

_TELUGU_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u0c1c\u0c28\u0c4d\u0c2e",
            "\u0c1c\u0c28\u0c4d\u0c2e\u0c24\u0c47\u0c26\u0c40",
            "\u0c1a\u0c47\u0c30\u0c4d\u0c2a\u0c41",
            "\u0c24\u0c47\u0c26\u0c40",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u0c1c\u0c28\u0c35\u0c30\u0c3f|\u0c2b\u0c3f\u0c2c\u0c4d\u0c30\u0c35\u0c30\u0c3f|\u0c2e\u0c3e\u0c30\u0c4d\u0c1a\u0c3f|\u0c0f\u0c2a\u0c4d\u0c30\u0c3f\u0c32\u0c4d|\u0c2e\u0c47|\u0c1c\u0c42\u0c28\u0c4d|\u0c1c\u0c42\u0c32\u0c48|\u0c06\u0c17\u0c38\u0c4d\u0c1f\u0c41|\u0c38\u0c46\u0c2a\u0c4d\u0c1f\u0c46\u0c02\u0c2c\u0c30\u0c4d|\u0c05\u0c15\u0c4d\u0c1f\u0c4b\u0c2c\u0c30\u0c4d|\u0c28\u0c35\u0c02\u0c2c\u0c30\u0c4d|\u0c21\u0c3f\u0c38\u0c46\u0c02\u0c2c\u0c30\u0c4d)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u0c1c\u0c28\u0c4d\u0c2e",
            "\u0c1c\u0c28\u0c4d\u0c2e\u0c24\u0c47\u0c26\u0c40",
            "\u0c24\u0c47\u0c26\u0c40",
            "\u0c1a\u0c47\u0c30\u0c4d\u0c2a\u0c41",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}\b|(?<!\w)(?:\+91[\s-]?)?[6-9]\d{1}[\s.-]?\d{4}[\s.-]?\d{5}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "\u0c2b\u0c4b\u0c28\u0c4d",
            "\u0c2e\u0c4a\u0c2c\u0c48\u0c32\u0c4d",
            "\u0c28\u0c02\u0c2c\u0c30\u0c4d",
            "\u0c38\u0c02\u0c2a\u0c30\u0c4d\u0c15\u0c02",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"\b(?:\d{1,5}\s+)?(?:[\u0c00-\u0c7F]+\s+)*(?:\u0c35\u0c40\u0c27\u0c3f|\u0c30\u0c4b\u0c21\u0c4d|\u0c28\u0c17\u0c30\u0c02|\u0c2e\u0c3e\u0c30\u0c4d\u0c17\u0c02|Road|Street|Nagar)\s*\d+[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e",
            "\u0c35\u0c3f\u0c32\u0c3e\u0c38\u0c02",
            "\u0c28\u0c3f\u0c35\u0c3e\u0c38\u0c02",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{6}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "\u0c2a\u0c3f\u0c28\u0c4d",
            "\u0c2a\u0c3f\u0c28\u0c4d \u0c15\u0c4b\u0c21\u0c4d",
            "\u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
    # Aadhaar (12 digits, possibly spaced as 4-4-4)
    PIIPattern(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "national_id",
        priority=9,
        base_score=0.4,
        context_words=[
            "\u0c06\u0c27\u0c3e\u0c30\u0c4d",
            "aadhaar",
            "uid",
            "\u0c17\u0c41\u0c30\u0c4d\u0c24\u0c3f\u0c02\u0c2a\u0c41",
        ],
        context_boost=0.45,
        validator=validate_aadhaar,
    ),
]

_ARABIC_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u062a\u0627\u0631\u064a\u062e",
            "\u0645\u064a\u0644\u0627\u062f",
            "\u0627\u0644\u0645\u064a\u0644\u0627\u062f",
            "\u0648\u0644\u062f",
            "\u0627\u0644\u062f\u062e\u0648\u0644",
            "\u062e\u0631\u0648\u062c",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u064a\u0646\u0627\u064a\u0631|\u0641\u0628\u0631\u0627\u064a\u0631|\u0645\u0627\u0631\u0633|\u0623\u0628\u0631\u064a\u0644|\u0627\u0628\u0631\u064a\u0644|\u0645\u0627\u064a\u0648|\u064a\u0648\u0646\u064a\u0648|\u064a\u0648\u0644\u064a\u0648|\u0623\u063a\u0633\u0637\u0633|\u0627\u063a\u0633\u0637\u0633|\u0633\u0628\u062a\u0645\u0628\u0631|\u0623\u0643\u062a\u0648\u0628\u0631|\u0627\u0643\u062a\u0648\u0628\u0631|\u0646\u0648\u0641\u0645\u0628\u0631|\u062f\u064a\u0633\u0645\u0628\u0631)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u062a\u0627\u0631\u064a\u062e",
            "\u0645\u064a\u0644\u0627\u062f",
            "\u0627\u0644\u0645\u064a\u0644\u0627\u062f",
            "\u0648\u0644\u062f",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        # Require either an international prefix (+CC) or a leading 0 so the
        # pattern doesn't fire on every 5–13-digit number string (e.g. the
        # 14-digit national-ID format in the same clinical note).
        r"(?<!\w)(?:\+(?:20|966|971|962|961|212|216|213|964|965|974|968|973)[\s.-]?|0)\d{1,3}(?:[\s.-]?\d{2,4}){2,3}\b",
        "phone_number",
        priority=8,
        base_score=0.45,
        context_words=[
            "\u0647\u0627\u062a\u0641",
            "\u062c\u0648\u0627\u0644",
            "\u0645\u0648\u0628\u0627\u064a\u0644",
            "\u062a\u0644\u064a\u0641\u0648\u0646",
            "\u0631\u0642\u0645",
            "\u0627\u062a\u0635\u0627\u0644",
        ],
        context_boost=0.4,
    ),
    PIIPattern(
        # Egyptian 14-digit national ID (starts with 2 or 3 for the
        # century digit). Other Arabic locales have different ID formats
        # (e.g. Saudi 10 digits starting 1/2, UAE 15 digits starting 784)
        # and are not currently matched here.
        r"\b[23]\d{13}\b",
        "national_id",
        priority=9,
        base_score=0.35,
        context_words=[
            "\u0627\u0644\u0631\u0642\u0645 \u0627\u0644\u0642\u0648\u0645\u064a",
            "\u0628\u0637\u0627\u0642\u0629",
            "\u0647\u0648\u064a\u0629",
            "\u0631\u0642\u0645 \u0627\u0644\u0647\u0648\u064a\u0629",
        ],
        context_boost=0.5,
    ),
    PIIPattern(
        r"\b(?:\u0634\u0627\u0631\u0639|\u0637\u0631\u064a\u0642|\u062d\u064a|\u0645\u064a\u062f\u0627\u0646|\u062c\u0627\u062f\u0629)\s+[\u0600-\u06FF0-9\s]{3,60}\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u0639\u0646\u0648\u0627\u0646",
            "\u0627\u0644\u0639\u0646\u0648\u0627\u0646",
            "\u0633\u0643\u0646",
            "\u064a\u0642\u064a\u0645",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "\u0627\u0644\u0631\u0645\u0632 \u0627\u0644\u0628\u0631\u064a\u062f\u064a",
            "\u0631\u0645\u0632 \u0628\u0631\u064a\u062f\u064a",
            "\u0639\u0646\u0648\u0627\u0646",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
    ),
]

_HEBREW_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u05ea\u05d0\u05e8\u05d9\u05da",
            "\u05ea\u05d0\u05e8\u05d9\u05da \u05dc\u05d9\u05d3\u05d4",
            "\u05dc\u05d9\u05d3\u05d4",
            "\u05e0\u05d5\u05dc\u05d3",
            "\u05e0\u05d5\u05dc\u05d3\u05d4",
            "\u05d0\u05e9\u05e4\u05d5\u05d6",
            "\u05e9\u05d7\u05e8\u05d5\u05e8",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u05d9\u05e0\u05d5\u05d0\u05e8|\u05e4\u05d1\u05e8\u05d5\u05d0\u05e8|\u05de\u05e8\u05e5|\u05d0\u05e4\u05e8\u05d9\u05dc|\u05de\u05d0\u05d9|\u05d9\u05d5\u05e0\u05d9|\u05d9\u05d5\u05dc\u05d9|\u05d0\u05d5\u05d2\u05d5\u05e1\u05d8|\u05e1\u05e4\u05d8\u05de\u05d1\u05e8|\u05d0\u05d5\u05e7\u05d8\u05d5\u05d1\u05e8|\u05e0\u05d5\u05d1\u05de\u05d1\u05e8|\u05d3\u05e6\u05de\u05d1\u05e8)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u05ea\u05d0\u05e8\u05d9\u05da",
            "\u05ea\u05d0\u05e8\u05d9\u05da \u05dc\u05d9\u05d3\u05d4",
            "\u05dc\u05d9\u05d3\u05d4",
            "\u05e0\u05d5\u05dc\u05d3",
            "\u05e0\u05d5\u05dc\u05d3\u05d4",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+972[\s.-]?|0)5\d[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "\u05d8\u05dc\u05e4\u05d5\u05df",
            "\u05e0\u05d9\u05d9\u05d3",
            "\u05de\u05e1\u05e4\u05e8",
            "\u05d9\u05e6\u05d9\u05e8\u05ea \u05e7\u05e9\u05e8",
            "\u05e7\u05e9\u05e8",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\d)\d{3}[\s-]?\d{3}[\s-]?\d{3}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.35,
        context_words=[
            "\u05ea\u05e2\u05d5\u05d3\u05ea \u05d6\u05d4\u05d5\u05ea",
            "\u05ea.\u05d6",
            "\u05de\u05e1\u05e4\u05e8 \u05d6\u05d4\u05d5\u05ea",
            "\u05d6\u05d4\u05d5\u05ea",
        ],
        context_boost=0.55,
        validator=validate_israeli_teudat_zehut,
    ),
    PIIPattern(
        r"(?<!\d)\d{5}(?:\d{2})?(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "\u05de\u05d9\u05e7\u05d5\u05d3",
            "\u05e7\u05d5\u05d3 \u05d3\u05d5\u05d0\u05e8",
            "\u05d3\u05d5\u05d0\u05e8",
            "\u05db\u05ea\u05d5\u05d1\u05ea",
        ],
        context_boost=0.5,
    ),
    PIIPattern(
        r"(?<!\w)(?:\u05e8\u05d7\u05d5\u05d1|\u05e8\u05d7'|\u05e9\u05d3\u05e8\u05d5\u05ea|\u05e9\u05d3'|\u05d3\u05e8\u05da|\u05db\u05d9\u05db\u05e8)\s+[\u0590-\u05FF\"'\s.-]{2,50}\s+\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u05db\u05ea\u05d5\u05d1\u05ea",
            "\u05de\u05e2\u05df",
            "\u05de\u05d2\u05d5\u05e8\u05d9\u05dd",
            "\u05e8\u05d7\u05d5\u05d1",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
]

_JAPANESE_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u751f\u5e74\u6708\u65e5",
            "\u8a95\u751f\u65e5",
            "\u751f\u307e\u308c",
            "\u5165\u9662",
            "\u9000\u9662",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{4}\u5e74\d{1,2}\u6708\d{1,2}\u65e5\b",
        "date",
        priority=9,
        base_score=0.7,
        context_words=[
            "\u751f\u5e74\u6708\u65e5",
            "\u8a95\u751f\u65e5",
            "\u751f\u307e\u308c",
            "\u5165\u9662",
            "\u9000\u9662",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+81[\s-]?)?(?:0\d{1,4}|\d{1,4})[\s-]?\d{1,4}[\s-]?\d{3,4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "\u96fb\u8a71",
            "\u643a\u5e2f",
            "\u756a\u53f7",
            "\u9023\u7d61",
            "\u30d5\u30a1\u30c3\u30af\u30b9",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "national_id",
        priority=9,
        base_score=0.35,
        context_words=[
            "\u30de\u30a4\u30ca\u30f3\u30d0\u30fc",
            "\u500b\u4eba\u756a\u53f7",
            "\u8eab\u5206\u8a3c",
            "\u756a\u53f7",
        ],
        context_boost=0.5,
    ),
    PIIPattern(
        r"\b(?:\d{3}-\d{4}|\u3012\d{3}-\d{4})\b",
        "postcode",
        priority=6,
        base_score=0.45,
        context_words=[
            "\u90f5\u4fbf\u756a\u53f7",
            "\u4f4f\u6240",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"\b[\u4e00-\u9fff]{2,12}(?:\u90fd|\u9053|\u5e9c|\u770c)[\u4e00-\u9fff0-9\u4e01\u76ee\u756a\u5730\u53f7\s-]{3,60}\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u4f4f\u6240",
            "\u6240\u5728\u5730",
            "\u81ea\u5b85",
        ],
        context_boost=0.25,
    ),
]

_TURKISH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "do\u011fum",
            "do\u011fum tarihi",
            "tarih",
            "yat\u0131\u015f",
            "\u00e7\u0131k\u0131\u015f",
            "taburcu",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:Ocak|\u015eubat|Subat|Mart|Nisan|May\u0131s|Mayis|Haziran|Temmuz|A\u011fustos|Agustos|Eyl\u00fcl|Eylul|Ekim|Kas\u0131m|Kasim|Aral\u0131k|Aralik)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "do\u011fum",
            "do\u011fum tarihi",
            "tarih",
            "yat\u0131\u015f",
            "\u00e7\u0131k\u0131\u015f",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+90\s?)?(?:5\d{2}|0\d{3})[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "telefon",
            "tel",
            "cep",
            "mobil",
            "numara",
            "ileti\u015fim",
            "faks",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b[1-9]\d{10}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "tc kimlik",
            "t.c. kimlik",
            "kimlik no",
            "tckn",
            "vatanda\u015fl\u0131k",
        ],
        context_boost=0.4,
        validator=validate_turkish_tckn,
    ),
    PIIPattern(
        # Latin Extended-A (\u0100-\u017f) covers Turkish-specific letters
        # (\u015e \u015f \u011e \u011f \u0130 \u0131) that fall outside Latin-1 Supplement.
        (
            r"\b(?:"
            r"(?:cadde|cad\.|sokak|sok\.|mahalle|mah\.|bulvar|bulv\.|apartman|apt\.)"
            r"\s+[A-Z\u00c0-\u00ff\u0100-\u017f][A-Za-z\u00c0-\u00ff\u0100-\u017f\s.-]{2,50}"
            r"\s+\d{1,5}[A-Za-z]?"
            r"|[A-Z\u00c0-\u00ff\u0100-\u017f][A-Za-z\u00c0-\u00ff\u0100-\u017f\s.-]{2,50}"
            r"\s+(?:caddesi|cadde|cad\.|soka\u011f\u0131|sokak|sok\.|mahallesi|mahalle|mah\."
            r"|bulvar\u0131|bulvar|bulv\.|apartman\u0131|apartman|apt\.)"
            r"\s+\d{1,5}[A-Za-z]?"
            r")\b"
        ),
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "adres",
            "ikamet",
            "oturuyor",
            "mahallesi",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "posta kodu",
            "pk",
            "adres",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Indonesian PII patterns
# ---------------------------------------------------------------------------

_INDONESIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "tanggal",
            "tanggal lahir",
            "lahir",
            "masuk",
            "keluar",
            "rawat",
            "kontrol",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "tanggal",
            "tanggal lahir",
            "lahir",
            "masuk",
            "keluar",
            "rawat",
            "kontrol",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+62\s?|0)8\d{1,3}(?:[\s.-]?\d{3,4}){2}\b",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "telepon",
            "telp",
            "hp",
            "ponsel",
            "nomor",
            "kontak",
            "whatsapp",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{16}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "nik",
            "nomor induk kependudukan",
            "ktp",
            "identitas",
            "kependudukan",
        ],
        context_boost=0.45,
        validator=validate_indonesian_nik,
    ),
    PIIPattern(
        r"\b(?:Jl\.|Jalan)\s+[A-Z][A-Za-z0-9 .'-]{2,60}\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "alamat",
            "domisili",
            "tinggal",
            "jalan",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "kode pos",
            "pos",
            "alamat",
            "domisili",
        ],
        context_boost=0.5,
        flags=re.IGNORECASE,
    ),
]


_THAI_MONTH_PATTERN = (
    r"มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|"
    r"กันยายน|ตุลาคม|พฤศจิกายน|ธันวาคม|ม\.ค\.|ก\.พ\.|มี\.ค\.|เม\.ย\.|"
    r"พ\.ค\.|มิ\.ย\.|ก\.ค\.|ส\.ค\.|ก\.ย\.|ต\.ค\.|พ\.ย\.|ธ\.ค\."
)

_THAI_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<!\d)\d{1,2}[/-]\d{1,2}[/-](?:25\d{2}|\d{2,4})(?!\d)",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "วันที่",
            "วันเกิด",
            "เกิด",
            "รับไว้",
            "จำหน่าย",
            "นัด",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"(?<!\d)\d{{1,2}}\s+(?:{_THAI_MONTH_PATTERN})\s+(?:พ\.ศ\.\s*)?(?:25\d{{2}}|\d{{4}})(?!\d)",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "วันที่",
            "วันเกิด",
            "เกิด",
            "รับไว้",
            "จำหน่าย",
            "นัด",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)(?:\+66[\s.-]?|0)[689]\d[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "โทรศัพท์",
            "โทร",
            "มือถือ",
            "เบอร์",
            "ติดต่อ",
            "หมายเลข",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)[1-9](?:[\s-]?\d){12}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "เลขบัตรประชาชน",
            "บัตรประชาชน",
            "เลขประจำตัวประชาชน",
            "ประจำตัวประชาชน",
            "ประชาชน",
        ],
        context_boost=0.4,
        validator=validate_thai_national_id,
    ),
    PIIPattern(
        r"(?<!\d)\d{5}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "รหัสไปรษณีย์",
            "ไปรษณีย์",
            "ที่อยู่",
            "แขวง",
            "เขต",
            "จังหวัด",
        ],
        context_boost=0.5,
    ),
    PIIPattern(
        r"(?<!\w)\d{1,5}\s+(?:ถนน|ถ\.|ซอย|ซ\.|หมู่|ม\.|แขวง|เขต|ตำบล|ต\.|อำเภอ|อ\.|จังหวัด|จ\.)[\u0E00-\u0E7F0-9\s./-]{3,80}",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "ที่อยู่",
            "อยู่ที่",
            "บ้านเลขที่",
            "ถนน",
            "ซอย",
            "แขวง",
            "เขต",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Polish PII patterns
# ---------------------------------------------------------------------------

_POLISH_PII_PATTERNS: List[PIIPattern] = [
    # PESEL (11-digit national ID)
    PIIPattern(
        r"\b\d{11}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "pesel",
            "numer pesel",
            "nr pesel",
            "pesel:",
            "tozsamo",
            "dowód osobisty",
            "dowod osobisty",
        ],
        context_boost=0.4,
        validator=validate_polish_pesel,
    ),
]

# ---------------------------------------------------------------------------
# Latvian PII patterns
# ---------------------------------------------------------------------------

_LATVIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "datums",
            "dzimsanas",
            "dzimšanas",
            "uznemts",
            "uzņemts",
            "izrakstits",
            "izrakstīts",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+371\s?)?[267]\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefons", "talrunis", "tālrunis", "mobilais", "kontakts"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{6}[-\s]?\d{5}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "personas kods",
            "personas kods:",
            "pk",
        ],
        context_boost=0.4,
        validator=validate_latvian_personas_kods,
    ),
    PIIPattern(
        r"\b(?:[A-Z][A-Za-z.'-]+\s+(?:iela|gatve|bulvaris|prospekts)\s+\d{1,5}[A-Za-z]?|(?:iela|gatve|bulvaris|prospekts)\s+[A-Z][A-Za-z .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["adrese", "dzivesvieta", "dzīvesvieta", "iela"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\bLV[-\s]?\d{4}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["pasta indekss", "indekss", "adrese"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]


_KOREAN_PII_PATTERNS: List[PIIPattern] = [
    # Korean dates: YYYY년 MM월 DD일
    PIIPattern(
        r"\b\d{4}\s?\ub144\s?\d{1,2}\s?\uc6d4\s?\d{1,2}\s?\uc77c\b",
        "date",
        priority=9,
        base_score=0.7,
        context_words=[
            "\uc0dd\ub144\uc6d4\uc77c",
            "\ucd9c\uc0dd",
            "\ud0dc\uc5b4\ub09c",
            "\uc0dd\uc77c",
            "\uc0ac\ub9dd",
            "\uc0ac\ub9dd\uc77c",
            "\uc785\uc6d0",
            "\ud1f4\uc6d0",
            "DOB",
        ],
        context_boost=0.25,
    ),
    # Korean dates: YYYY.MM.DD or YYYY-MM-DD or YYYY/MM/DD
    PIIPattern(
        r"\b\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}\b",
        "date",
        priority=8,
        base_score=0.55,
        context_words=[
            "\uc0dd\ub144\uc6d4\uc77c",
            "\ucd9c\uc0dd",
            "\ud0dc\uc5b4\ub09c",
            "\uc0dd\uc77c",
            "DOB",
            "\uc0ac\ub9dd",
            "\uc0ac\ub9dd\uc77c",
            "\uc785\uc6d0",
            "\ud1f4\uc6d0",
        ],
        context_boost=0.3,
    ),
    # Korean phone numbers: mobile and landline
    PIIPattern(
        r"(?<!\d)(?:\+82[-\s]?)?0?(?:2|1[016789]|[3-6]\d)[-\s]?\d{3,4}[-\s]?\d{4}(?!\d)",
        "phone_number",
        priority=8,
        base_score=0.6,
        context_words=[
            "\uc804\ud654",
            "\uc804\ud654\ubc88\ud638",
            "\ud734\ub300\ud3f0",
            "\ud578\ub4dc\ud3f0",
            "\uc5f0\ub77d\ucc98",
            "\ud329\uc2a4",
            "call",
            "contact",
        ],
        context_boost=0.3,
    ),
    # Korean RRN (13-digit Resident Registration Number)
    PIIPattern(
        r"(?<!\d)\d{6}[-\s]?\d{7}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "\uc8fc\ubbfc\ub4f1\ub85d\ubc88\ud638",
            "\uc8fc\ubbfc\ubc88\ud638",
            "\ub4f1\ub85d\ubc88\ud638",
            "rrn",
            "resident registration",
            "jumin",
        ],
        context_boost=0.4,
        validator=validate_korean_rrn,
    ),
    # Korean street addresses
    PIIPattern(
        r"(?<![\uac00-\ud7a30-9])(?:[\uac00-\ud7a3]{1,4}(?:\ud2b9\ubcc4\uc2dc|\uad11\uc5ed\uc2dc|\ud2b9\ubcc4\uc790\uce58\uc2dc|\ud2b9\ubcc4\uc790\uce58\ub3c4|\ub3c4|\uc2dc|\uad70|\uad6c)\s*)+[\uac00-\ud7a30-9]{1,5}(?:\ub85c|\uae38|\ub3d9)\s?\d+(?:-\d+)?(?!\d)",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\uc8fc\uc18c",
            "\uac70\uc8fc",
            "\uc790\ud0dd",
            "\uc704\uce58",
            "\uc18c\uc7ac\uc9c0",
        ],
        context_boost=0.25,
    ),
    # Korean postal code (5-digit)
    PIIPattern(
        r"(?<!\d)\d{5}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.15,
        context_words=[
            "\uc6b0\ud3b8\ubc88\ud638",
            "\uc6b0\ud3b8",
            "zip",
            "postal",
        ],
        context_boost=0.55,
    ),
]


_MALAY_MONTH_PATTERN = (
    r"Januari|Februari|Mac|April|Mei|Jun|Julai|Ogos|September|"
    r"Oktober|November|Disember"
)

_MALAY_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "tarikh",
            "tarikh lahir",
            "lahir",
            "masuk",
            "keluar",
            "rawatan",
            "temu janji",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\s+(?:{_MALAY_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "tarikh",
            "tarikh lahir",
            "lahir",
            "masuk",
            "keluar",
            "rawatan",
            "temu janji",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+60[\s.-]?|0)1\d(?:[\s.-]?\d{3,4}){2}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telefon",
            "tel",
            "nombor telefon",
            "bimbit",
            "mudah alih",
            "hubungi",
            "kontak",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)\d{6}(?:-\d{2}-|\d{2})\d{4}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "mykad",
            "nric",
            "kad pengenalan",
            "no kad pengenalan",
            "nombor kad pengenalan",
            "no kp",
            "ic",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        validator=validate_malaysian_mykad,
    ),
    PIIPattern(
        r"\b(?:Jalan|Jl\.?|Lorong|Lrg\.?|Taman|Persiaran|Lebuh|Kampung)\s+[A-Z][A-Za-z0-9 .'-]{2,60}\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "alamat",
            "tinggal",
            "kediaman",
            "jalan",
            "lorong",
            "taman",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
]


_TAGALOG_MONTH_PATTERN = (
    r"Enero|Pebrero|Marso|Abril|Mayo|Hunyo|Hulyo|Agosto|Setyembre|"
    r"Oktubre|Nobyembre|Disyembre"
)

_TAGALOG_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "petsa",
            "petsa ng kapanganakan",
            "kapanganakan",
            "ipinanganak",
            "isinilang",
            "admit",
            "discharge",
            "nilabas",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\s+(?:{_TAGALOG_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "petsa",
            "petsa ng kapanganakan",
            "kapanganakan",
            "ipinanganak",
            "isinilang",
            "admit",
            "discharge",
            "nilabas",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+63[\s.-]?|0)9\d{2}[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telepono",
            "tel",
            "mobile",
            "cellphone",
            "numero",
            "numero ng telepono",
            "kontak",
            "tawagan",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)\d{4}(?:[\s-]?\d{4}){2}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "psn",
            "philsys",
            "philippine identification",
            "philippine national id",
            "pambansang id",
            "national id",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        validator=validate_philsys_psn,
    ),
    PIIPattern(
        r"(?<!\d)\d{2}[\s-]?\d{9}[\s-]?\d(?!\d)",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "philhealth",
            "philhealth pin",
            "philhealth number",
            "philhealth no",
            "pin",
            "numero ng philhealth",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        validator=validate_philhealth_pin,
    ),
    PIIPattern(
        r"\b(?:Barangay|Brgy\.?|Bgy\.?|Kalye|Kalsada|Daang|Daan|Avenida|Sitio|Purok)\s+[A-Z][A-Za-z0-9 .'-]{2,60}\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "tirahan",
            "address",
            "adres",
            "barangay",
            "kalye",
            "kalsada",
            "sitio",
            "purok",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
]


_DANISH_MONTH_PATTERN = (
    r"januar|februar|marts|april|maj|juni|juli|august|september|"
    r"oktober|november|december"
)

_DANISH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "dato",
            "født",
            "foedt",
            "fodt",
            "fødselsdato",
            "foedselsdato",
            "indlagt",
            "udskrevet",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\.?\s+(?:{_DANISH_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "dato",
            "født",
            "foedt",
            "fodt",
            "fødselsdato",
            "foedselsdato",
            "indlagt",
            "udskrevet",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+45[\s.-]?)?(?:\d{2}[\s.-]?){3}\d{2}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telefon",
            "tlf",
            "mobil",
            "kontakt",
            "ring",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)\d{6}[-\s]?\d{4}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "cpr",
            "cpr-nummer",
            "cpr nummer",
            "personnummer",
            "person nr",
            "person-id",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        validator=validate_danish_cpr,
    ),
    PIIPattern(
        r"\b(?!(?:adresse|bopæl|bopael)\b)[A-ZÆØÅ][A-Za-zÆØÅæøå .'-]{2,60}(?:gade|vej|all[eé]|plads|torv|str[æa]de)\s+\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "adresse",
            "bopæl",
            "bopael",
            "vej",
            "gade",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)(?:DK[-\s]?)?\d{4}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=["postnummer", "postnr", "postkode", "adresse"],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]


_SLOVAK_MONTH_PATTERN = (
    r"janu[aá]r(?:a)?|febru[aá]r(?:a)?|marec|marca|apr[ií]l(?:a)?|"
    r"m[aá]j(?:a)?|j[uú]n(?:a)?|j[uú]l(?:a)?|august(?:a)?|"
    r"september|septembra|okt[oó]ber|okt[oó]bra|november|novembra|"
    r"december|decembra"
)

_SLOVAK_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "datum",
            "d\u00e1tum",
            "narodenia",
            "narodeny",
            "naroden\u00fd",
            "prijaty",
            "prijat\u00fd",
            "prepusteny",
            "prepusten\u00fd",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\.?\s+(?:{_SLOVAK_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "datum",
            "d\u00e1tum",
            "narodenia",
            "narodeny",
            "naroden\u00fd",
            "prijaty",
            "prijat\u00fd",
            "prepusteny",
            "prepusten\u00fd",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+421\s?|0)9\d{2}(?:[\s.-]?\d{3}){2}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telefon",
            "telef\u00f3n",
            "tel",
            "mobil",
            "cislo",
            "\u010d\u00edslo",
            "kontakt",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)\d{2}(?:0[1-9]|1[0-2]|2[1-9]|3[0-2]|5[1-9]|6[0-2]|7[1-9]|8[0-2])(?:0[1-9]|[12]\d|3[01])[\s/-]?\d{4}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "rodne cislo",
            "rodn\u00e9 \u010d\u00edslo",
            "rc",
            "r\u010d",
            "identifikacne cislo",
            "identifika\u010dn\u00e9 \u010d\u00edslo",
        ],
        context_boost=0.45,
        validator=validate_czechoslovak_rodne_cislo,
    ),
    PIIPattern(
        r"\b(?!(?:adresa|bydlisko|trvale)\b)(?:[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f.'-]+(?:\s+[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f.'-]+)*?\s+(?:ulica|ul\.|trieda|n[aá]mestie|cesta)\s+\d{1,5}[A-Za-z]?|(?:ulica|ul\.|trieda|n[aá]mestie|cesta)\s+[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "adresa",
            "bydlisko",
            "trvale bydlisko",
            "trval\u00e9 bydlisko",
            "ulica",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)\d{3}\s?\d{2}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "psc",
            "ps\u010d",
            "postove smerovacie cislo",
            "po\u0161tov\u00e9 smerovacie \u010d\u00edslo",
            "adresa",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

LANGUAGE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "fr": _FRENCH_PII_PATTERNS,
    "de": _GERMAN_PII_PATTERNS,
    "it": _ITALIAN_PII_PATTERNS,
    "es": _SPANISH_PII_PATTERNS,
    "pt": _PORTUGUESE_PII_PATTERNS,
    "nl": _DUTCH_PII_PATTERNS,
    "hi": _HINDI_PII_PATTERNS,
    "te": _TELUGU_PII_PATTERNS,
    "ar": _ARABIC_PII_PATTERNS,
    "he": _HEBREW_PII_PATTERNS,
    "ja": _JAPANESE_PII_PATTERNS,
    "tr": _TURKISH_PII_PATTERNS,
    "id": _INDONESIAN_PII_PATTERNS,
    "th": _THAI_PII_PATTERNS,
    "lv": _LATVIAN_PII_PATTERNS,
    "pl": _POLISH_PII_PATTERNS,
    "ko": _KOREAN_PII_PATTERNS,
    "sk": _SLOVAK_PII_PATTERNS,
    "ms": _MALAY_PII_PATTERNS,
    "tl": _TAGALOG_PII_PATTERNS,
    "da": _DANISH_PII_PATTERNS,
}

LOCALE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "en_gb": _UK_ENGLISH_PII_PATTERNS,
    "en_au": _AU_ENGLISH_PII_PATTERNS,
    "en_ca": _CANADIAN_ENGLISH_PII_PATTERNS,
    "fr_ca": _CANADIAN_ENGLISH_PII_PATTERNS,
}


# ---------------------------------------------------------------------------
# Language-specific fake data
# ---------------------------------------------------------------------------

LANGUAGE_FAKE_DATA: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "NAME": ["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"],
        "FIRST_NAME": ["Jane", "John", "Alex", "Sam"],
        "LAST_NAME": ["Smith", "Doe", "Johnson", "Taylor"],
        "EMAIL": ["patient@example.com", "contact@example.org"],
        "PHONE": ["555-0123", "555-0456", "555-0789"],
        "ID_NUM": ["XXX-XX-1234", "MRN-987654"],
        "STREET_ADDRESS": ["123 Main St", "456 Oak Ave"],
        "URL_PERSONAL": ["https://example.com"],
        "USERNAME": ["user123", "patient456"],
        "DATE": ["01/01/2000", "12/31/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["New York", "Los Angeles"],
        "ZIPCODE": ["10001", "90210", "60601"],
    },
    "fr": {
        "NAME": ["Marie Dupont", "Jean Martin", "Sophie Bernard", "Pierre Durand"],
        "FIRST_NAME": ["Marie", "Jean", "Sophie", "Pierre"],
        "LAST_NAME": ["Dupont", "Martin", "Bernard", "Durand"],
        "EMAIL": ["patient@exemple.fr", "contact@exemple.org"],
        "PHONE": ["+33 6 12 34 56 78", "+33 7 98 76 54 32", "01 23 45 67 89"],
        "ID_NUM": ["1 85 05 78 006 084 36"],
        "STREET_ADDRESS": ["12 rue de la Paix", "45 avenue Victor Hugo"],
        "URL_PERSONAL": ["https://exemple.fr"],
        "USERNAME": ["utilisateur123", "patient456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Paris", "Lyon", "Marseille"],
        "ZIPCODE": ["75001", "69002", "13001"],
    },
    "de": {
        "NAME": ["Anna M\u00fcller", "Hans Schmidt", "Petra Weber", "Klaus Fischer"],
        "FIRST_NAME": ["Anna", "Hans", "Petra", "Klaus"],
        "LAST_NAME": ["M\u00fcller", "Schmidt", "Weber", "Fischer"],
        "EMAIL": ["patient@beispiel.de", "kontakt@beispiel.org"],
        "PHONE": ["+49 30 1234567", "+49 89 9876543", "+49 170 1234567"],
        "ID_NUM": ["12345678901"],
        "STREET_ADDRESS": ["Hauptstra\u00dfe 12", "Berliner Allee 45"],
        "URL_PERSONAL": ["https://beispiel.de"],
        "USERNAME": ["benutzer123", "patient456"],
        "DATE": ["01.01.2000", "31.12.1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Berlin", "M\u00fcnchen", "Hamburg"],
        "ZIPCODE": ["10115", "80331", "20095"],
    },
    "it": {
        "NAME": ["Maria Rossi", "Marco Bianchi", "Giulia Russo", "Luca Ferrari"],
        "FIRST_NAME": ["Maria", "Marco", "Giulia", "Luca"],
        "LAST_NAME": ["Rossi", "Bianchi", "Russo", "Ferrari"],
        "EMAIL": ["paziente@esempio.it", "contatto@esempio.org"],
        "PHONE": ["+39 333 1234567", "+39 06 12345678", "+39 348 9876543"],
        "ID_NUM": ["RSSMRA85M01H501Z"],
        "STREET_ADDRESS": ["Via Roma 12", "Piazza Garibaldi 3"],
        "URL_PERSONAL": ["https://esempio.it"],
        "USERNAME": ["utente123", "paziente456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Roma", "Milano", "Napoli"],
        "ZIPCODE": ["00100", "20121", "80100"],
    },
    "es": {
        "NAME": [
            "Mar\u00eda L\u00f3pez",
            "Carlos Garc\u00eda",
            "Ana Mart\u00ednez",
            "Pedro S\u00e1nchez",
        ],
        "FIRST_NAME": ["Mar\u00eda", "Carlos", "Ana", "Pedro"],
        "LAST_NAME": ["L\u00f3pez", "Garc\u00eda", "Mart\u00ednez", "S\u00e1nchez"],
        "EMAIL": ["paciente@ejemplo.es", "contacto@ejemplo.org"],
        "PHONE": ["+34 612 345 678", "+34 934 567 890", "+34 711 234 567"],
        "ID_NUM": ["12345678Z", "X1234567L"],
        "STREET_ADDRESS": ["Calle Serrano 42", "Avenida de la Constituci\u00f3n 10"],
        "URL_PERSONAL": ["https://ejemplo.es"],
        "USERNAME": ["usuario123", "paciente456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Madrid", "Barcelona", "Sevilla"],
        "ZIPCODE": ["28001", "08001", "41001"],
    },
    "pt": {
        "NAME": ["Ana Silva", "Pedro Almeida", "Mariana Costa", "Jo\u00e3o Santos"],
        "FIRST_NAME": ["Ana", "Pedro", "Mariana", "Jo\u00e3o"],
        "LAST_NAME": ["Silva", "Almeida", "Costa", "Santos"],
        "EMAIL": ["paciente@exemplo.pt", "contato@exemplo.org"],
        "PHONE": ["+351 912 345 678", "+55 11 91234-5678"],
        "ID_NUM": ["123.456.789-09", "11.222.333/0001-81"],
        "STREET_ADDRESS": ["Rua das Flores 25", "Avenida da Liberdade 42"],
        "URL_PERSONAL": ["https://exemplo.pt"],
        "USERNAME": ["usuario123", "paciente456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Lisboa", "Porto", "S\u00e3o Paulo"],
        "ZIPCODE": ["1200-195", "1250-096", "01310-100"],
    },
    "nl": {
        "NAME": ["Sanne de Vries", "Daan Jansen", "Lotte Bakker", "Milan Visser"],
        "FIRST_NAME": ["Sanne", "Daan", "Lotte", "Milan"],
        "LAST_NAME": ["de Vries", "Jansen", "Bakker", "Visser"],
        "EMAIL": ["patient@voorbeeld.nl", "contact@voorbeeld.org"],
        "PHONE": ["+31 6 12345678", "06 87654321"],
        "ID_NUM": ["123456782"],
        "STREET_ADDRESS": ["Keizersgracht 123", "Stationsweg 45A"],
        "URL_PERSONAL": ["https://voorbeeld.nl"],
        "USERNAME": ["gebruiker123", "patient456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Amsterdam", "Utrecht", "Rotterdam"],
        "ZIPCODE": ["1012 AB", "3511 CC", "3011 AA"],
    },
    "hi": {
        "NAME": [
            "\u0905\u0928\u093f\u0924\u093e \u0936\u0930\u094d\u092e\u093e",
            "\u0930\u093e\u091c\u0947\u0936 \u0915\u0941\u092e\u093e\u0930",
            "\u092a\u094d\u0930\u093f\u092f\u093e \u0935\u0930\u094d\u092e\u093e",
            "\u0905\u092e\u093f\u0924 \u0938\u093f\u0902\u0939",
        ],
        "FIRST_NAME": [
            "\u0905\u0928\u093f\u0924\u093e",
            "\u0930\u093e\u091c\u0947\u0936",
            "\u092a\u094d\u0930\u093f\u092f\u093e",
            "\u0905\u092e\u093f\u0924",
        ],
        "LAST_NAME": [
            "\u0936\u0930\u094d\u092e\u093e",
            "\u0915\u0941\u092e\u093e\u0930",
            "\u0935\u0930\u094d\u092e\u093e",
            "\u0938\u093f\u0902\u0939",
        ],
        "EMAIL": ["patient@example.in", "sampark@example.org"],
        "PHONE": ["+91 9876543210", "9123456789"],
        "ID_NUM": ["MRN-982341"],
        "STREET_ADDRESS": [
            "12 \u0917\u0932\u0940 \u0938\u0902\u0916\u094d\u092f\u093e 5",
            "45 Green Park Road",
        ],
        "URL_PERSONAL": ["https://udaharan.in"],
        "USERNAME": ["rogi123", "parichay456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": [
            "\u0926\u093f\u0932\u094d\u0932\u0940",
            "\u092e\u0941\u0902\u092c\u0908",
            "\u0932\u0916\u0928\u090a",
        ],
        "ZIPCODE": ["110001", "400001", "226001"],
    },
    "te": {
        "NAME": [
            "\u0c38\u0c3f\u0c24\u0c3e \u0c30\u0c46\u0c21\u0c4d\u0c21\u0c3f",
            "\u0c30\u0c3e\u0c2e\u0c4d \u0c15\u0c41\u0c2e\u0c3e\u0c30\u0c4d",
            "\u0c2a\u0c4d\u0c30\u0c3f\u0c2f\u0c3e \u0c26\u0c47\u0c35\u0c3f",
            "\u0c05\u0c30\u0c41\u0c23\u0c4d \u0c35\u0c30\u0c4d\u0c2e",
        ],
        "FIRST_NAME": [
            "\u0c38\u0c3f\u0c24\u0c3e",
            "\u0c30\u0c3e\u0c2e\u0c4d",
            "\u0c2a\u0c4d\u0c30\u0c3f\u0c2f\u0c3e",
            "\u0c05\u0c30\u0c41\u0c23\u0c4d",
        ],
        "LAST_NAME": [
            "\u0c30\u0c46\u0c21\u0c4d\u0c21\u0c3f",
            "\u0c15\u0c41\u0c2e\u0c3e\u0c30\u0c4d",
            "\u0c26\u0c47\u0c35\u0c3f",
            "\u0c35\u0c30\u0c4d\u0c2e",
        ],
        "EMAIL": ["patient@example.in", "sampark@example.org"],
        "PHONE": ["+91 9876543210", "9988776655"],
        "ID_NUM": ["MRN-548231"],
        "STREET_ADDRESS": [
            "12 \u0c35\u0c40\u0c27\u0c3f 5",
            "45 Jubilee Hills Road",
        ],
        "URL_PERSONAL": ["https://udaharanam.in"],
        "USERNAME": ["rogi123", "samacharam456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": [
            "\u0c39\u0c48\u0c26\u0c30\u0c3e\u0c2c\u0c3e\u0c26\u0c4d",
            "\u0c35\u0c3f\u0c1c\u0c2f\u0c35\u0c3e\u0c21",
            "\u0c17\u0c41\u0c02\u0c1f\u0c42\u0c30\u0c41",
        ],
        "ZIPCODE": ["500001", "520001", "522001"],
    },
    "ar": {
        "NAME": [
            "\u0644\u064a\u0644\u0649 \u062d\u0633\u0646",
            "\u0623\u062d\u0645\u062f \u0639\u0644\u064a",
            "\u0645\u0631\u064a\u0645 \u0645\u062d\u0645\u0648\u062f",
            "\u064a\u0648\u0633\u0641 \u0625\u0628\u0631\u0627\u0647\u064a\u0645",
        ],
        "FIRST_NAME": [
            "\u0644\u064a\u0644\u0649",
            "\u0623\u062d\u0645\u062f",
            "\u0645\u0631\u064a\u0645",
            "\u064a\u0648\u0633\u0641",
        ],
        "LAST_NAME": [
            "\u062d\u0633\u0646",
            "\u0639\u0644\u064a",
            "\u0645\u062d\u0645\u0648\u062f",
            "\u0625\u0628\u0631\u0627\u0647\u064a\u0645",
        ],
        "EMAIL": ["patient@example.eg", "contact@example.org"],
        "PHONE": ["+20 10 1234 5678", "+966 50 123 4567"],
        "ID_NUM": ["29801011234567"],
        "STREET_ADDRESS": [
            "\u0634\u0627\u0631\u0639 \u0627\u0644\u0646\u064a\u0644 12",
            "\u0637\u0631\u064a\u0642 \u0627\u0644\u0645\u0644\u0643 \u0641\u0647\u062f 45",
        ],
        "URL_PERSONAL": ["https://example.eg"],
        "USERNAME": ["mareed123", "mostakhdem456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": [
            "\u0627\u0644\u0642\u0627\u0647\u0631\u0629",
            "\u0627\u0644\u0631\u064a\u0627\u0636",
            "\u062f\u0628\u064a",
        ],
        "ZIPCODE": ["11511", "12345", "54321"],
    },
    "he": {
        "NAME": [
            "\u05d3\u05e0\u05d4 \u05db\u05d4\u05df",
            "\u05d9\u05d5\u05e0\u05ea\u05df \u05dc\u05d5\u05d9",
            "\u05de\u05d9\u05db\u05dc \u05d0\u05d1\u05e8\u05d4\u05dd",
            "\u05e2\u05de\u05d9\u05ea \u05e4\u05e8\u05e5",
        ],
        "FIRST_NAME": [
            "\u05d3\u05e0\u05d4",
            "\u05d9\u05d5\u05e0\u05ea\u05df",
            "\u05de\u05d9\u05db\u05dc",
            "\u05e2\u05de\u05d9\u05ea",
        ],
        "LAST_NAME": [
            "\u05db\u05d4\u05df",
            "\u05dc\u05d5\u05d9",
            "\u05d0\u05d1\u05e8\u05d4\u05dd",
            "\u05e4\u05e8\u05e5",
        ],
        "EMAIL": ["patient@example.co.il", "contact@example.org"],
        "PHONE": ["+972 54-123-4567", "054-987-6543"],
        "ID_NUM": ["123456782", "000000018"],
        "STREET_ADDRESS": [
            "\u05e8\u05d7\u05d5\u05d1 \u05d4\u05e8\u05e6\u05dc 12",
            "\u05e9\u05d3\u05e8\u05d5\u05ea \u05e8\u05d5\u05d8\u05e9\u05d9\u05dc\u05d3 45",
        ],
        "URL_PERSONAL": ["https://example.co.il"],
        "USERNAME": ["metupal123", "patient456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": [
            "\u05ea\u05dc \u05d0\u05d1\u05d9\u05d1",
            "\u05d9\u05e8\u05d5\u05e9\u05dc\u05d9\u05dd",
            "\u05d7\u05d9\u05e4\u05d4",
        ],
        "ZIPCODE": ["6423905", "9414101", "3100001"],
    },
    "ja": {
        "NAME": [
            "\u4f50\u85e4 \u82b1\u5b50",
            "\u7530\u4e2d \u592a\u90ce",
            "\u9234\u6728 \u7f8e\u54b2",
            "\u9ad8\u6a4b \u5065",
        ],
        "FIRST_NAME": [
            "\u82b1\u5b50",
            "\u592a\u90ce",
            "\u7f8e\u54b2",
            "\u5065",
        ],
        "LAST_NAME": [
            "\u4f50\u85e4",
            "\u7530\u4e2d",
            "\u9234\u6728",
            "\u9ad8\u6a4b",
        ],
        "EMAIL": ["patient@example.jp", "renraku@example.org"],
        "PHONE": ["+81 90 1234 5678", "03-1234-5678"],
        "ID_NUM": ["1234 5678 9012"],
        "STREET_ADDRESS": [
            "\u6771\u4eac\u90fd\u65b0\u5bbf\u533a\u897f\u65b0\u5bbf2\u4e01\u76ee8\u756a1\u53f7",
            "\u5927\u962a\u5e9c\u5927\u962a\u5e02\u5317\u533a\u6885\u75301\u4e01\u76ee1\u756a",
        ],
        "URL_PERSONAL": ["https://example.jp"],
        "USERNAME": ["kanja123", "riyousha456"],
        "DATE": ["2000/01/01", "1999/12/31"],
        "AGE": ["45", "62", "38"],
        "LOCATION": [
            "\u6771\u4eac",
            "\u5927\u962a",
            "\u4eac\u90fd",
        ],
        "ZIPCODE": ["160-0023", "530-0001", "100-0001"],
    },
    "tr": {
        "NAME": [
            "Ay\u015fe Y\u0131lmaz",
            "Mehmet Kaya",
            "Zeynep Demir",
            "Emre \u015eahin",
        ],
        "FIRST_NAME": ["Ay\u015fe", "Mehmet", "Zeynep", "Emre"],
        "LAST_NAME": ["Y\u0131lmaz", "Kaya", "Demir", "\u015eahin"],
        "EMAIL": ["hasta@ornek.tr", "iletisim@ornek.org"],
        "PHONE": ["+90 532 123 45 67", "0532 987 65 43"],
        "ID_NUM": ["10000000146"],
        "STREET_ADDRESS": ["Atat\u00fcrk Caddesi 12", "\u0130stiklal Sokak 45"],
        "URL_PERSONAL": ["https://ornek.tr"],
        "USERNAME": ["hasta123", "kullanici456"],
        "DATE": ["01.01.2000", "31.12.1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["\u0130stanbul", "Ankara", "\u0130zmir"],
        "ZIPCODE": ["34000", "06000", "35000"],
    },
    "id": {
        "NAME": ["Siti Aminah", "Budi Santoso", "Dewi Lestari", "Agus Pratama"],
        "FIRST_NAME": ["Siti", "Budi", "Dewi", "Agus"],
        "LAST_NAME": ["Aminah", "Santoso", "Lestari", "Pratama"],
        "EMAIL": ["pasien@contoh.id", "kontak@contoh.org"],
        "PHONE": ["+62 812 3456 7890", "0812-9876-5432"],
        "ID_NUM": ["3174055708850001"],
        "STREET_ADDRESS": ["Jl. Merdeka No. 10", "Jl. Sudirman 25"],
        "URL_PERSONAL": ["https://contoh.id"],
        "USERNAME": ["pasien123", "pengguna456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Jakarta", "Bandung", "Surabaya"],
        "ZIPCODE": ["10110", "40123", "60234"],
    },
    "ms": {
        "NAME": ["Nur Aisyah", "Ahmad Hakim", "Siti Farah", "Lim Wei Han"],
        "FIRST_NAME": ["Nur", "Ahmad", "Siti", "Wei Han"],
        "LAST_NAME": ["Aisyah", "Hakim", "Farah", "Lim"],
        "EMAIL": ["pesakit@contoh.my", "hubungi@contoh.org"],
        "PHONE": ["+60 12-345 6789", "012-987 6543"],
        "ID_NUM": ["850817-14-5678", "900101145678"],
        "STREET_ADDRESS": ["Jalan Merdeka 10", "Lorong Damai 5"],
        "URL_PERSONAL": ["https://contoh.my"],
        "USERNAME": ["pesakit123", "pengguna456"],
        "DATE": ["17/08/1985", "1 Januari 2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Kuala Lumpur", "Johor Bahru", "George Town"],
        "ZIPCODE": ["50000", "80000", "10300"],
    },
    "tl": {
        "NAME": [
            "Maria Santos",
            "Jose Reyes",
            "Ana Cruz",
            "Juan dela Cruz",
        ],
        "FIRST_NAME": ["Maria", "Jose", "Ana", "Juan"],
        "LAST_NAME": ["Santos", "Reyes", "Cruz", "dela Cruz"],
        "EMAIL": ["pasyente@example.ph", "kontak@example.org"],
        "PHONE": ["+63 917 123 4567", "0917-987-6543"],
        "ID_NUM": ["1234-5678-9012", "98-765432109-8"],
        "STREET_ADDRESS": ["Barangay Maligaya", "Kalye Rizal 12"],
        "URL_PERSONAL": ["https://example.ph"],
        "USERNAME": ["pasyente123", "gumagamit456"],
        "DATE": ["17/08/1985", "17 Agosto 1985"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Manila", "Quezon City", "Cebu City"],
        "ZIPCODE": ["1000", "1100", "6000"],
    },
    "da": {
        "NAME": ["Anna Nielsen", "Peter Jensen", "Mette Hansen", "Lars Andersen"],
        "FIRST_NAME": ["Anna", "Peter", "Mette", "Lars"],
        "LAST_NAME": ["Nielsen", "Jensen", "Hansen", "Andersen"],
        "EMAIL": ["patient@example.dk", "kontakt@example.org"],
        "PHONE": ["+45 20 12 34 56", "30 45 67 89"],
        "ID_NUM": ["170885-1234", "010101-4001"],
        "STREET_ADDRESS": ["Bredgade 12", "Roskildevej 45"],
        "URL_PERSONAL": ["https://example.dk"],
        "USERNAME": ["patient123", "bruger456"],
        "DATE": ["17/08/1985", "17 august 1985"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Kobenhavn", "Aarhus", "Odense"],
        "ZIPCODE": ["1260", "8000", "5000"],
    },
    "th": {
        "NAME": [
            "สมชาย ใจดี",
            "สุดา แก้วใส",
            "อนันต์ วิชัย",
            "มาลี พรหมมา",
        ],
        "FIRST_NAME": ["สมชาย", "สุดา", "อนันต์", "มาลี"],
        "LAST_NAME": ["ใจดี", "แก้วใส", "วิชัย", "พรหมมา"],
        "EMAIL": ["patient@example.th", "contact@example.org"],
        "PHONE": ["+66 81 234 5678", "081-234-5678"],
        "ID_NUM": ["1101700203450"],
        "STREET_ADDRESS": [
            "123 ถนนสุขุมวิท",
            "45 ซอยสีลม",
        ],
        "URL_PERSONAL": ["https://example.th"],
        "USERNAME": ["rogi123", "phuphuai456"],
        "DATE": ["15 มกราคม 2567", "01/01/2567"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["กรุงเทพมหานคร", "เชียงใหม่", "ภูเก็ต"],
        "ZIPCODE": ["10110", "50000", "83000"],
    },
    "lv": {
        "NAME": ["Anna Kalnina", "Janis Berzins", "Ilze Ozolina", "Peteris Liepa"],
        "FIRST_NAME": ["Anna", "Janis", "Ilze", "Peteris"],
        "LAST_NAME": ["Kalnina", "Berzins", "Ozolina", "Liepa"],
        "EMAIL": ["pacients@example.lv", "kontakts@example.org"],
        "PHONE": ["+371 2123 4567", "2912 3456"],
        "ID_NUM": ["161175-19997", "32867300679"],
        "STREET_ADDRESS": ["Brivibas iela 12", "Dzirnavu iela 45"],
        "URL_PERSONAL": ["https://example.lv"],
        "USERNAME": ["pacients123", "lietotajs456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Riga", "Daugavpils", "Liepaja"],
        "ZIPCODE": ["LV-1010", "LV-5401", "LV-3401"],
    },
    "sk": {
        "NAME": [
            "Jana Kovacova",
            "Peter Novak",
            "Marta Horvathova",
            "Tomas Kral",
        ],
        "FIRST_NAME": ["Jana", "Peter", "Marta", "Tomas"],
        "LAST_NAME": ["Kovacova", "Novak", "Horvathova", "Kral"],
        "EMAIL": ["pacient@example.sk", "kontakt@example.org"],
        "PHONE": ["+421 903 123 456", "0903 987 654"],
        "ID_NUM": ["850505/1236", "855505/1230"],
        "STREET_ADDRESS": ["Hlavna ulica 12", "Namestie SNP 5"],
        "URL_PERSONAL": ["https://example.sk"],
        "USERNAME": ["pacient123", "pouzivatel456"],
        "DATE": ["05.05.1985", "5. maja 1985"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Bratislava", "Kosice", "Zilina"],
        "ZIPCODE": ["81101", "04001", "01001"],
    },
    "ko": {
        "NAME": [
            "\uae40\ubbfc\uc900",
            "\uc774\uc11c\uc5f0",
            "\ubc15\uc9c0\ud6c8",
        ],
        "FIRST_NAME": [
            "\ubbfc\uc900",
            "\uc11c\uc5f0",
            "\uc9c0\ud6c8",
        ],
        "LAST_NAME": [
            "\uae40",
            "\uc774",
            "\ubc15",
        ],
        "EMAIL": ["hwanja@example.co.kr", "contact@example.kr", "user@example.co.kr"],
        "PHONE": ["010-1234-5678", "02-123-4567", "+82 10 9876 5432"],
        "ID_NUM": ["000101-3123456", "991231-1123456", "MRN-123456"],
        "STREET_ADDRESS": [
            "\uc11c\uc6b8\ud2b9\ubcc4\uc2dc \ub9c8\ud3ec\uad6c \ub9c8\ud3ec\ub300\ub85c 25",
            "\uc11c\uc6b8\ud2b9\ubcc4\uc2dc \uc6a9\uc0b0\uad6c \ub0a8\uc0b0\uacf5\uc6d0\uae38 105",
            "\ubd80\uc0b0\uad11\uc5ed\uc2dc \ud574\uc6b4\ub300\uad6c \ud574\uc6b4\ub300\ud574\ubcc0\ub85c 45",
        ],
        "URL_PERSONAL": ["https://example.kr"],
        "USERNAME": ["miinji88", "songgu_1990", "jiun_10"],
        "DATE": ["2000-01-01", "2009/05/19", "2020-10-10"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["\uc11c\uc6b8", "\ubd80\uc0b0", "\uc778\ucc9c"],
        "ZIPCODE": ["01007", "46007", "21307"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_pattern_language(lang: str) -> str:
    return lang.strip().replace("-", "_").split("_", 1)[0].casefold()


def _normalize_pattern_locale(locale: str) -> str:
    return locale.strip().replace("-", "_").casefold()


def _locale_pattern_keys(lang: str, locale: str | None) -> list[str]:
    keys: list[str] = []
    if locale:
        keys.append(_normalize_pattern_locale(locale))
    if "_" in lang or "-" in lang:
        keys.append(_normalize_pattern_locale(lang))

    deduped: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def get_patterns_for_language(lang: str, locale: str | None = None) -> List[PIIPattern]:
    """Return combined PII patterns for the given language.

    English patterns (email, URL, IP, etc.) are universal and always
    included. Language-specific patterns (dates, phones, national IDs,
    addresses) are added on top. Locale-specific overlays, such as
    ``en_GB`` UK identifiers, are appended when ``locale`` is provided or
    when ``lang`` includes a region code.

    Args:
        lang: ISO 639-1 language code, optionally with a region suffix for
            pattern lookup. Model-backed languages are listed in
            :data:`SUPPORTED_LANGUAGES`; national-ID-only languages are listed
            in :data:`NATIONAL_ID_ONLY_LANGUAGES`.
        locale: Optional locale override (for example, ``"en_GB"``) whose
            locale-specific deterministic patterns should also be active.

    Returns:
        List of PIIPattern instances for the language

    Raises:
        ValueError: If the language is not supported
    """
    supported_pattern_languages = SUPPORTED_LANGUAGES | NATIONAL_ID_ONLY_LANGUAGES
    base_lang = _normalize_pattern_language(lang)
    if base_lang not in supported_pattern_languages:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {sorted(supported_pattern_languages)}"
        )

    from .pii_entity_merger import PII_PATTERNS

    # English patterns serve as universal base
    # MRZ patterns are language-agnostic, so they join the universal base.
    base = list(PII_PATTERNS) + MRZ_PII_PATTERNS

    combined = base
    if base_lang != "en":
        combined = combined + LANGUAGE_PII_PATTERNS.get(base_lang, [])

    for locale_key in _locale_pattern_keys(lang, locale):
        combined = combined + LOCALE_PII_PATTERNS.get(locale_key, [])

    return combined
