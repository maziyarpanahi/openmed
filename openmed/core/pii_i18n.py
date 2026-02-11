"""Multilingual PII detection support.

Language-specific data, validators, regex patterns, and fake data
for French, German, and Italian PII detection and de-identification.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: Set[str] = {"en", "fr", "de", "it"}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
}

LANGUAGE_MODEL_PREFIX: Dict[str, str] = {
    "en": "",
    "fr": "French-",
    "de": "German-",
    "it": "Italian-",
}

DEFAULT_PII_MODELS: Dict[str, str] = {
    "en": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    "fr": "OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1",
    "de": "OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1",
    "it": "OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1",
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
        base_score=0.4,
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
    # French postal code (5 digits)
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
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
    # German phone numbers: +49, 0xxx
    PIIPattern(
        r"(?<!\w)(?:\+49\s?|0)\d{2,4}[\s/-]?\d{3,8}\b",
        "phone_number",
        priority=8,
        base_score=0.5,
        context_words=[
            "Telefon", "Tel", "Handy", "Mobil", "Fax",
            "Rufnummer", "Nummer", "anrufen", "Kontakt",
        ],
        context_boost=0.35,
    ),
    # German Steuer-ID (11 digits)
    PIIPattern(
        r"\b\d{11}\b",
        "national_id",
        priority=9,
        base_score=0.2,
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
    # German PLZ (5 digits)
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
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

LANGUAGE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "fr": _FRENCH_PII_PATTERNS,
    "de": _GERMAN_PII_PATTERNS,
    "it": _ITALIAN_PII_PATTERNS,
}


# ---------------------------------------------------------------------------
# Language-specific fake data
# ---------------------------------------------------------------------------

LANGUAGE_FAKE_DATA: Dict[str, Dict[str, List[str]]] = {
    "en": {
        "NAME": ["Jane Smith", "John Doe", "Alex Johnson", "Sam Taylor"],
        "EMAIL": ["patient@example.com", "contact@example.org"],
        "PHONE": ["555-0123", "555-0456", "555-0789"],
        "ID_NUM": ["XXX-XX-1234", "MRN-987654"],
        "STREET_ADDRESS": ["123 Main St", "456 Oak Ave"],
        "URL_PERSONAL": ["https://example.com"],
        "USERNAME": ["user123", "patient456"],
        "DATE": ["01/01/2000", "12/31/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["New York", "Los Angeles"],
    },
    "fr": {
        "NAME": ["Marie Dupont", "Jean Martin", "Sophie Bernard", "Pierre Durand"],
        "EMAIL": ["patient@exemple.fr", "contact@exemple.org"],
        "PHONE": ["+33 6 12 34 56 78", "+33 7 98 76 54 32", "01 23 45 67 89"],
        "ID_NUM": ["1 85 05 78 006 084 36"],
        "STREET_ADDRESS": ["12 rue de la Paix", "45 avenue Victor Hugo"],
        "URL_PERSONAL": ["https://exemple.fr"],
        "USERNAME": ["utilisateur123", "patient456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Paris", "Lyon", "Marseille"],
    },
    "de": {
        "NAME": ["Anna M\u00fcller", "Hans Schmidt", "Petra Weber", "Klaus Fischer"],
        "EMAIL": ["patient@beispiel.de", "kontakt@beispiel.org"],
        "PHONE": ["+49 30 1234567", "+49 89 9876543", "+49 170 1234567"],
        "ID_NUM": ["12345678901"],
        "STREET_ADDRESS": ["Hauptstra\u00dfe 12", "Berliner Allee 45"],
        "URL_PERSONAL": ["https://beispiel.de"],
        "USERNAME": ["benutzer123", "patient456"],
        "DATE": ["01.01.2000", "31.12.1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Berlin", "M\u00fcnchen", "Hamburg"],
    },
    "it": {
        "NAME": ["Maria Rossi", "Marco Bianchi", "Giulia Russo", "Luca Ferrari"],
        "EMAIL": ["paziente@esempio.it", "contatto@esempio.org"],
        "PHONE": ["+39 333 1234567", "+39 06 12345678", "+39 348 9876543"],
        "ID_NUM": ["RSSMRA85M01H501Z"],
        "STREET_ADDRESS": ["Via Roma 12", "Piazza Garibaldi 3"],
        "URL_PERSONAL": ["https://esempio.it"],
        "USERNAME": ["utente123", "paziente456"],
        "DATE": ["01/01/2000", "31/12/1999"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Roma", "Milano", "Napoli"],
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
        lang: ISO 639-1 language code (en, fr, de, it)

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
