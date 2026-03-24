"""Multilingual PII detection support.

Language-specific data, validators, regex patterns, and fake data
for multilingual PII detection and de-identification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Set[str] = {"en", "fr", "de", "it", "es", "nl", "hi", "te"}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
    "hi": "Hindi",
    "te": "Telugu",
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
}


# ---------------------------------------------------------------------------
# National ID Validators
# ---------------------------------------------------------------------------

def validate_french_nir(text: str) -> bool:
    """Validate French NIR/INSEE number.

    The NIR (Numero d'Inscription au Repertoire) is 15 digits.
    Format: S AA MM DDD CCC OOO KK
    Key (last 2 digits) = 97 - (first 13 digits mod 97)

    Args:
        text: NIR string (may contain spaces)

    Returns:
        True if valid NIR format and checksum
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 15:
        return False

    # First digit must be 1 or 2
    if digits[0] not in ("1", "2"):
        return False

    try:
        number = int(digits[:13])
        key = int(digits[13:15])
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


# ---------------------------------------------------------------------------
# Language-specific month names (for date parsing/formatting)
# ---------------------------------------------------------------------------

LANGUAGE_MONTH_NAMES: Dict[str, List[str]] = {
    "en": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "fr": [
        "janvier", "f\u00e9vrier", "mars", "avril", "mai", "juin",
        "juillet", "ao\u00fbt", "septembre", "octobre", "novembre", "d\u00e9cembre",
    ],
    "de": [
        "Januar", "Februar", "M\u00e4rz", "April", "Mai", "Juni",
        "Juli", "August", "September", "Oktober", "November", "Dezember",
    ],
    "it": [
        "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
        "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
    ],
    "es": [
        "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    ],
    "nl": [
        "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november", "december",
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
}


# ---------------------------------------------------------------------------
# Language-specific PII patterns
# ---------------------------------------------------------------------------

from .pii_entity_merger import PIIPattern  # noqa: E402

_FRENCH_PII_PATTERNS: List[PIIPattern] = [
    # French dates DD/MM/YYYY
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "n\u00e9", "n\u00e9e", "naissance", "date de naissance", "dob",
            "d\u00e9c\u00e8s", "d\u00e9c\u00e9d\u00e9", "admis", "sorti",
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
            "n\u00e9", "n\u00e9e", "naissance", "date de naissance",
            "d\u00e9c\u00e8s", "admis", "sorti",
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
            "t\u00e9l\u00e9phone", "t\u00e9l", "portable", "mobile",
            "num\u00e9ro", "appeler", "contact", "fax",
        ],
        context_boost=0.3,
    ),
    # French NIR/INSEE (15 digits, possibly spaced)
    PIIPattern(
        r"\b[12]\s?\d{2}\s?\d{2}\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{2}\b",
        "national_id",
        priority=10,
        base_score=0.55,
        context_words=[
            "nir", "insee", "s\u00e9curit\u00e9 sociale",
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
            "adresse", "domicile", "r\u00e9side", "habite", "situ\u00e9",
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
            "code postal", "cp", "cedex",
        ],
        context_boost=0.5,
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
            "Geburtsdatum", "geboren", "geb", "verstorben",
            "aufgenommen", "entlassen", "Datum",
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
            "Geburtsdatum", "geboren", "geb", "verstorben",
            "aufgenommen", "entlassen",
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
            "Telefon", "Tel", "Handy", "Mobil", "Fax",
            "Rufnummer", "Nummer", "anrufen", "Kontakt",
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
            "Steuer-ID", "Steueridentifikationsnummer", "Steuernummer",
            "IdNr", "Identifikationsnummer",
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
            "Adresse", "Anschrift", "wohnhaft", "wohnt",
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
            "PLZ", "Postleitzahl",
        ],
        context_boost=0.5,
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
            "nato", "nata", "nascita", "data di nascita",
            "decesso", "deceduto", "ricovero", "dimissione",
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
            "nato", "nata", "nascita", "data di nascita",
            "decesso", "ricovero", "dimissione",
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
            "telefono", "tel", "cellulare", "mobile",
            "numero", "chiamare", "contatto", "fax",
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
            "codice fiscale", "cf", "c.f.",
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
            "indirizzo", "domicilio", "residente", "risiede", "abitazione",
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
            "CAP", "codice postale",
        ],
        context_boost=0.5,
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
            "nacido", "nacida", "nacimiento", "fecha de nacimiento",
            "fallecimiento", "fallecido", "ingreso", "alta",
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
            "nacido", "nacida", "nacimiento", "fecha de nacimiento",
            "fallecimiento", "ingreso", "alta",
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
            "tel\u00e9fono", "tel", "m\u00f3vil", "celular",
            "n\u00famero", "llamar", "contacto", "fax",
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
            "dni", "documento nacional", "identidad",
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
            "nie", "n\u00famero de identidad de extranjero",
            "extranjero", "residencia",
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
            "direcci\u00f3n", "domicilio", "residente", "reside", "ubicaci\u00f3n",
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
            "c\u00f3digo postal", "cp",
        ],
        context_boost=0.5,
    ),
]

_DUTCH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "geboren", "geboortedatum", "opname", "ontslag", "datum",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:januari|februari|maart|april|mei|juni|juli|augustus|september|oktober|november|december)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "geboren", "geboortedatum", "opname", "ontslag",
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
            "telefoon", "tel", "mobiel", "nummer", "contact", "fax",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{9}\b",
        "national_id",
        priority=9,
        base_score=0.25,
        context_words=[
            "bsn", "burgerservicenummer", "service nummer",
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
            "adres", "woonadres", "woont", "verblijft",
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
            "postcode", "post code", "adres",
        ],
        context_boost=0.4,
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
            "\u091c\u0928\u094d\u092e", "\u091c\u0928\u094d\u092e\u0924\u093f\u0925\u093f",
            "\u092d\u0930\u094d\u0924\u0940", "\u0924\u093e\u0930\u0940\u0916",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u091c\u0928\u0935\u0930\u0940|\u092b(?:\u093c)?\u0930\u0935\u0930\u0940|\u092e\u093e\u0930\u094d\u091a|\u0905\u092a\u094d\u0930\u0948\u0932|\u092e\u0908|\u091c\u0942\u0928|\u091c\u0941\u0932\u093e\u0908|\u0905\u0917\u0938\u094d\u0924|\u0938\u093f\u0924\u0902\u092c\u0930|\u0905\u0915\u094d\u091f\u0942\u092c\u0930|\u0928\u0935\u0902\u092c\u0930|\u0926\u093f\u0938\u0902\u092c\u0930)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u091c\u0928\u094d\u092e", "\u091c\u0928\u094d\u092e\u0924\u093f\u0925\u093f",
            "\u092d\u0930\u094d\u0924\u0940", "\u0924\u093e\u0930\u0940\u0916",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}\b|(?<!\w)(?:\+91[\s-]?)?[6-9]\d{1}[\s.-]?\d{4}[\s.-]?\d{5}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "\u092b\u094b\u0928", "\u092e\u094b\u092c\u093e\u0907\u0932",
            "\u0928\u0902\u092c\u0930", "\u0938\u0902\u092a\u0930\u094d\u0915",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"\b(?:\d{1,5}\s+)?(?:[\u0900-\u097F]+\s+)*(?:\u0917\u0932\u0940|\u0938\u0921\u093c\u0915|\u0928\u0917\u0930|\u092e\u093e\u0930\u094d\u0917|Road|Street|Nagar)\s*(?:\u0938\u0902\u0916\u094d\u092f\u093e\s*)?\d+[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u092a\u0924\u093e", "\u0928\u093f\u0935\u093e\u0938", "\u091a\u093f\u0930\u0941\u0928\u093e\u092e\u093e",
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
            "\u092a\u093f\u0928", "\u092a\u093f\u0928\u0915\u094b\u0921",
            "\u0921\u093e\u0915", "\u092a\u0924\u093e",
        ],
        context_boost=0.5,
    ),
    # Aadhaar (12 digits, possibly spaced as 4-4-4)
    PIIPattern(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "national_id",
        priority=9,
        base_score=0.4,
        context_words=[
            "\u0906\u0927\u093e\u0930", "aadhaar", "\u092f\u0942\u0906\u0908\u0921\u0940\u090f\u0906\u0908",
            "uid", "\u092a\u0939\u091a\u093e\u0928",
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
            "\u0c1c\u0c28\u0c4d\u0c2e", "\u0c1c\u0c28\u0c4d\u0c2e\u0c24\u0c47\u0c26\u0c40",
            "\u0c1a\u0c47\u0c30\u0c4d\u0c2a\u0c41", "\u0c24\u0c47\u0c26\u0c40",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}\s+(?:\u0c1c\u0c28\u0c35\u0c30\u0c3f|\u0c2b\u0c3f\u0c2c\u0c4d\u0c30\u0c35\u0c30\u0c3f|\u0c2e\u0c3e\u0c30\u0c4d\u0c1a\u0c3f|\u0c0f\u0c2a\u0c4d\u0c30\u0c3f\u0c32\u0c4d|\u0c2e\u0c47|\u0c1c\u0c42\u0c28\u0c4d|\u0c1c\u0c42\u0c32\u0c48|\u0c06\u0c17\u0c38\u0c4d\u0c1f\u0c41|\u0c38\u0c46\u0c2a\u0c4d\u0c1f\u0c46\u0c02\u0c2c\u0c30\u0c4d|\u0c05\u0c15\u0c4d\u0c1f\u0c4b\u0c2c\u0c30\u0c4d|\u0c28\u0c35\u0c02\u0c2c\u0c30\u0c4d|\u0c21\u0c3f\u0c38\u0c46\u0c02\u0c2c\u0c30\u0c4d)\s+\d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "\u0c1c\u0c28\u0c4d\u0c2e", "\u0c1c\u0c28\u0c4d\u0c2e\u0c24\u0c47\u0c26\u0c40",
            "\u0c24\u0c47\u0c26\u0c40", "\u0c1a\u0c47\u0c30\u0c4d\u0c2a\u0c41",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+91[\s-]?)?[6-9]\d{9}\b|(?<!\w)(?:\+91[\s-]?)?[6-9]\d{1}[\s.-]?\d{4}[\s.-]?\d{5}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "\u0c2b\u0c4b\u0c28\u0c4d", "\u0c2e\u0c4a\u0c2c\u0c48\u0c32\u0c4d",
            "\u0c28\u0c02\u0c2c\u0c30\u0c4d", "\u0c38\u0c02\u0c2a\u0c30\u0c4d\u0c15\u0c02",
        ],
        context_boost=0.35,
    ),
    PIIPattern(
        r"\b(?:\d{1,5}\s+)?(?:[\u0c00-\u0c7F]+\s+)*(?:\u0c35\u0c40\u0c27\u0c3f|\u0c30\u0c4b\u0c21\u0c4d|\u0c28\u0c17\u0c30\u0c02|\u0c2e\u0c3e\u0c30\u0c4d\u0c17\u0c02|Road|Street|Nagar)\s*\d+[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "\u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e", "\u0c35\u0c3f\u0c32\u0c3e\u0c38\u0c02", "\u0c28\u0c3f\u0c35\u0c3e\u0c38\u0c02",
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
            "\u0c2a\u0c3f\u0c28\u0c4d", "\u0c2a\u0c3f\u0c28\u0c4d \u0c15\u0c4b\u0c21\u0c4d",
            "\u0c1a\u0c3f\u0c30\u0c41\u0c28\u0c3e\u0c2e\u0c3e",
        ],
        context_boost=0.5,
    ),
    # Aadhaar (12 digits, possibly spaced as 4-4-4)
    PIIPattern(
        r"\b\d{4}\s?\d{4}\s?\d{4}\b",
        "national_id",
        priority=9,
        base_score=0.4,
        context_words=[
            "\u0c06\u0c27\u0c3e\u0c30\u0c4d", "aadhaar", "uid",
            "\u0c17\u0c41\u0c30\u0c4d\u0c24\u0c3f\u0c02\u0c2a\u0c41",
        ],
        context_boost=0.45,
        validator=validate_aadhaar,
    ),
]

LANGUAGE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "fr": _FRENCH_PII_PATTERNS,
    "de": _GERMAN_PII_PATTERNS,
    "it": _ITALIAN_PII_PATTERNS,
    "es": _SPANISH_PII_PATTERNS,
    "nl": _DUTCH_PII_PATTERNS,
    "hi": _HINDI_PII_PATTERNS,
    "te": _TELUGU_PII_PATTERNS,
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
        "NAME": ["Mar\u00eda L\u00f3pez", "Carlos Garc\u00eda", "Ana Mart\u00ednez", "Pedro S\u00e1nchez"],
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
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_patterns_for_language(lang: str) -> List[PIIPattern]:
    """Return combined PII patterns for the given language.

    English patterns (email, URL, IP, etc.) are universal and always
    included. Language-specific patterns (dates, phones, national IDs,
    addresses) are added on top.

    Args:
        lang: ISO 639-1 language code (en, fr, de, it, es, nl, hi, te)

    Returns:
        List of PIIPattern instances for the language

    Raises:
        ValueError: If the language is not supported
    """
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    from .pii_entity_merger import PII_PATTERNS

    # English patterns serve as universal base
    base = list(PII_PATTERNS)

    if lang == "en":
        return base

    # Add language-specific patterns
    lang_patterns = LANGUAGE_PII_PATTERNS.get(lang, [])
    return base + lang_patterns
