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
from datetime import date
from importlib import resources
from typing import Dict, List, Optional

from .anonymizer.providers.clinical_ids import (
    validate_australian_medicare,
    validate_australian_tfn,
    validate_bc_phn,
    validate_canadian_sin,
    validate_luhn,
    validate_ontario_health_card,
    validate_uk_nhs_number,
    validate_uk_nino,
)
from .language_pack_catalog import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
    DEFAULT_PII_MODELS,
    NATIONAL_ID_ONLY_LANGUAGES,
    SUPPORTED_LANGUAGES,
    USER_SUPPLIED_MODEL_LANGUAGES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "es": "Spanish",
    "nl": "Dutch",
    "hi": "Hindi",
    "te": "Telugu",
    "am": "Amharic",
    "pt": "Portuguese",
    "ar": "Arabic",
    "he": "Hebrew",
    "ja": "Japanese",
    "tr": "Turkish",
    "id": "Indonesian",
    "th": "Thai",
    "ko": "Korean",
    "ro": "Romanian",
    "zh": "Chinese",
    "sw": "Swahili",
    "zu": "isiZulu",
    "xh": "isiXhosa",
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
    "am": "Amharic-",
    "pt": "Portuguese-",
    "ar": "Arabic-",
    "he": "Hebrew-",
    "ja": "Japanese-",
    "tr": "Turkish-",
    "id": "Indonesian-",
    "th": "Thai-",
    "ko": "Korean-",
    "ro": "Romanian-",
    "zh": "Chinese-",
    "sw": "Swahili-",
    "zu": "isiZulu-",
    "xh": "isiXhosa-",
}

# ---------------------------------------------------------------------------
# Chinese personal-name assist data
# ---------------------------------------------------------------------------

_CHINESE_SURNAME_RESOURCE = "data/chinese_surnames.txt"
_HAN_CHARACTER_CLASS = "\u3400-\u4dbf\u4e00-\u9fff"


def _load_chinese_surnames() -> frozenset[str]:
    """Load the packaged public-domain/CC0 Chinese surname gazetteer."""

    resource = resources.files("openmed.clinical").joinpath(_CHINESE_SURNAME_RESOURCE)
    with resource.open("r", encoding="utf-8") as handle:
        surnames = {
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        }
    if not surnames or any(
        re.fullmatch(rf"[{_HAN_CHARACTER_CLASS}]{{1,2}}", surname) is None
        for surname in surnames
    ):
        raise RuntimeError("invalid packaged Chinese surname gazetteer")
    return frozenset(surnames)


CHINESE_SURNAMES = _load_chinese_surnames()
"""Permissively licensed Chinese surname vocabulary with no patient data."""

CHINESE_COMPOUND_SURNAMES = frozenset(
    surname for surname in CHINESE_SURNAMES if len(surname) == 2
)
"""Common two-character Chinese surnames, such as ``欧阳`` and ``司马``."""

CHINESE_SINGLE_SURNAMES = CHINESE_SURNAMES - CHINESE_COMPOUND_SURNAMES
"""Single-character Chinese surnames used by the name-recognition assist."""

# Synthetic, non-patient probes used to audit tokenizer representation before a
# PII model is advertised for a writing system.  The shape intentionally mirrors
# LANGUAGE_FAKE_DATA below while keeping the audit focused on names, addresses,
# and identifiers that would create direct leakage if tokenized as unknowns.
TOKENIZER_SCRIPT_FAKE_DATA: Dict[str, Dict[str, List[str]]] = {
    "han_simplified": {
        "NAME": ["张伟", "李娜", "王芳"],
        "STREET_ADDRESS": [
            "北京市朝阳区建国路88号",
            "上海市浦东新区世纪大道100号",
            "广州市天河区体育西路12号",
        ],
        "ID_NUM": ["身份证110105199001012345", "病历号京医20260018"],
    },
    "han_traditional": {
        "NAME": ["王小明", "陳美玲", "林志豪"],
        "STREET_ADDRESS": [
            "臺北市中正區忠孝東路一段10號",
            "高雄市苓雅區中正一路25號",
            "香港九龍彌敦道88號",
        ],
        "ID_NUM": ["身分證A123456789", "病歷號臺醫20260027"],
    },
    "devanagari": {
        "NAME": ["अनिता शर्मा", "राजेश कुमार", "प्रिया वर्मा"],
        "STREET_ADDRESS": [
            "१२ जनपथ मार्ग नई दिल्ली",
            "४५ एम जी रोड पुणे",
            "८८ अस्पताल मार्ग लखनऊ",
        ],
        "ID_NUM": ["रोगी क्रमांक १२३४५६", "पहचान ९८७६५४३२१०"],
    },
    "bengali": {
        "NAME": ["অনন্যা সেন", "রাহুল দাস", "মেঘা রায়"],
        "STREET_ADDRESS": [
            "১২ কলেজ স্ট্রিট কলকাতা",
            "৪৫ হাসপাতাল রোড ঢাকা",
            "৮৮ রবীন্দ্র সরণি শিলিগুড়ি",
        ],
        "ID_NUM": ["রোগী নম্বর ১২৩৪৫৬", "পরিচয় ৯৮৭৬৫৪৩২১০"],
    },
    "tamil": {
        "NAME": ["அனிதா குமார்", "ரவி சங்கர்", "மீனா தேவி"],
        "STREET_ADDRESS": [
            "௧௨ அண்ணா சாலை சென்னை",
            "௪௫ மருத்துவமனை வீதி மதுரை",
            "௮௮ காந்தி சாலை கோயம்புத்தூர்",
        ],
        "ID_NUM": ["நோயாளர் எண் ௧௨௩௪௫௬", "அடையாளம் ௯௮௭௬௫௪௩௨௧௦"],
    },
    "telugu": {
        "NAME": ["సీతా రెడ్డి", "రామ్ కుమార్", "ప్రియా దేవి"],
        "STREET_ADDRESS": [
            "౧౨ బంజారా హిల్స్ హైదరాబాద్",
            "౪౫ ఆసుపత్రి వీధి విజయవాడ",
            "౮౮ గాంధీ రోడ్ గుంటూరు",
        ],
        "ID_NUM": ["రోగి సంఖ్య ౧౨౩౪౫౬", "గుర్తింపు ౯౮౭౬౫౪౩౨౧౦"],
    },
    "kannada": {
        "NAME": ["ಅನಿತಾ ರಾವ್", "ರವಿ ಕುಮಾರ್", "ಮೀನಾ ದೇವಿ"],
        "STREET_ADDRESS": [
            "೧೨ ಎಂ ಜಿ ರಸ್ತೆ ಬೆಂಗಳೂರು",
            "೪೫ ಆಸ್ಪತ್ರೆ ರಸ್ತೆ ಮೈಸೂರು",
            "೮೮ ಗಾಂಧಿ ನಗರ ಹುಬ್ಬಳ್ಳಿ",
        ],
        "ID_NUM": ["ರೋಗಿ ಸಂಖ್ಯೆ ೧೨೩೪೫೬", "ಗುರುತು ೯೮೭೬೫೪೩೨೧೦"],
    },
    "malayalam": {
        "NAME": ["അനിത നായർ", "രവി കുമാർ", "മീന ദേവി"],
        "STREET_ADDRESS": [
            "൧൨ എം ജി റോഡ് കൊച്ചി",
            "൪൫ ആശുപത്രി റോഡ് കോഴിക്കോട്",
            "൮൮ ഗാന്ധി നഗർ തിരുവനന്തപുരം",
        ],
        "ID_NUM": ["രോഗി നമ്പർ ൧൨൩൪൫൬", "തിരിച്ചറിയൽ ൯൮൭൬൫൪൩൨൧൦"],
    },
    "gujarati": {
        "NAME": ["અનિતા પટેલ", "રવિ શાહ", "મીના દેસાઈ"],
        "STREET_ADDRESS": [
            "૧૨ આશ્રમ રોડ અમદાવાદ",
            "૪૫ હોસ્પિટલ માર્ગ વડોદરા",
            "૮૮ ગાંધી નગર સુરત",
        ],
        "ID_NUM": ["દર્દી નંબર ૧૨૩૪૫૬", "ઓળખ ૯૮૭૬૫૪૩૨૧૦"],
    },
    "gurmukhi": {
        "NAME": ["ਅਨੀਤਾ ਕੌਰ", "ਰਵੀ ਸਿੰਘ", "ਮੀਨਾ ਗਿੱਲ"],
        "STREET_ADDRESS": [
            "੧੨ ਮਾਲ ਰੋਡ ਅੰਮ੍ਰਿਤਸਰ",
            "੪੫ ਹਸਪਤਾਲ ਮਾਰਗ ਲੁਧਿਆਣਾ",
            "੮੮ ਗਾਂਧੀ ਨਗਰ ਪਟਿਆਲਾ",
        ],
        "ID_NUM": ["ਮਰੀਜ਼ ਨੰਬਰ ੧੨੩੪੫੬", "ਪਛਾਣ ੯੮੭੬੫੪੩੨੧੦"],
    },
    "odia": {
        "NAME": ["ଅନିତା ଦାସ", "ରବି କୁମାର", "ମୀନା ପଟ୍ଟନାୟକ"],
        "STREET_ADDRESS": [
            "୧୨ ଜନପଥ ଭୁବନେଶ୍ୱର",
            "୪୫ ଡାକ୍ତରଖାନା ରୋଡ କଟକ",
            "୮୮ ଗାନ୍ଧୀ ନଗର ପୁରୀ",
        ],
        "ID_NUM": ["ରୋଗୀ ନମ୍ବର ୧୨୩୪୫୬", "ପରିଚୟ ୯୮୭୬୫୪୩୨୧୦"],
    },
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
# Chinese Contact, Financial, and Travel Identifiers
# ---------------------------------------------------------------------------


def validate_chinese_mobile_number(text: str) -> bool:
    """Validate a mainland China mobile number with an optional ``+86`` prefix."""

    if not isinstance(text, str):
        return False
    return re.fullmatch(r"(?:\+86[ -]?)?1[3-9][0-9]{9}", text.strip()) is not None


def validate_chinese_bank_card(text: str) -> bool:
    """Validate a 16-19 digit Chinese bank-card candidate using Luhn."""

    if not isinstance(text, str):
        return False
    candidate = text.strip()
    if re.fullmatch(r"[0-9](?:[ -]?[0-9]){15,18}", candidate) is None:
        return False
    digits = re.sub(r"[ -]", "", candidate)
    return 16 <= len(digits) <= 19 and validate_luhn(digits)


def validate_chinese_passport(text: str) -> bool:
    """Validate the offline structure of a PRC passport number."""

    return bool(
        isinstance(text, str) and re.fullmatch(r"[EGDSP][0-9]{8}", text.strip().upper())
    )


def validate_hong_kong_macau_permit(text: str) -> bool:
    """Validate a Hong Kong/Macau resident Home Return Permit number."""

    return bool(
        isinstance(text, str) and re.fullmatch(r"[HM][0-9]{8}", text.strip().upper())
    )


def validate_taiwan_compatriot_permit(text: str) -> bool:
    """Validate the eight-digit Taiwan Compatriot Permit structure."""

    return bool(isinstance(text, str) and re.fullmatch(r"[0-9]{8}", text.strip()))


# Descriptive aliases matching the official travel-permit names.
validate_chinese_mobile = validate_chinese_mobile_number
validate_hk_macau_permit = validate_hong_kong_macau_permit
validate_mainland_travel_permit_hong_kong_macau = validate_hong_kong_macau_permit
validate_mainland_travel_permit_taiwan = validate_taiwan_compatriot_permit
validate_taiwan_permit = validate_taiwan_compatriot_permit


_ARABIC_INDIC_DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def normalize_arabic_indic_digits(text: str) -> str:
    """Fold Arabic-Indic decimal digits to ASCII without changing offsets.

    Both the Arabic-Indic block (U+0660-U+0669) and the extended block commonly
    used in Arabic-script text (U+06F0-U+06F9) are mapped one code point at a
    time. ASCII text and all non-digit characters are left unchanged.

    Args:
        text: Text that may contain Arabic-Indic decimal digits.

    Returns:
        Length-preserving text with supported decimal digits rendered as ASCII.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return text.translate(_ARABIC_INDIC_DIGIT_TRANSLATION)


EGYPTIAN_GOVERNORATE_CODES = frozenset(
    {
        "01",
        "02",
        "03",
        "04",
        "11",
        "12",
        "13",
        "14",
        "15",
        "16",
        "17",
        "18",
        "19",
        "21",
        "22",
        "23",
        "24",
        "25",
        "26",
        "27",
        "28",
        "29",
        "31",
        "32",
        "33",
        "34",
        "35",
        "88",
    }
)


def validate_egyptian_national_id(text: str) -> bool:
    """Validate the published structure of an Egyptian national ID.

    The official final check-digit algorithm is not public, so this validator
    deliberately checks only stable offline structure: 14 decimal digits, the
    1900s/2000s century marker, a real embedded Gregorian birth date, and a
    published governorate code.
    """
    if not isinstance(text, str):
        return False
    normalized = normalize_arabic_indic_digits(text.strip())
    if re.fullmatch(r"[23][0-9]{13}", normalized) is None:
        return False

    century = 1900 if normalized[0] == "2" else 2000
    if not _is_valid_calendar_date(
        century + int(normalized[1:3]),
        int(normalized[3:5]),
        int(normalized[5:7]),
    ):
        return False
    return normalized[7:9] in EGYPTIAN_GOVERNORATE_CODES


MOROCCAN_CIN_REGION_LETTERS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def validate_moroccan_cin(text: str) -> bool:
    """Validate the offline structure of a Moroccan CIN identifier.

    A CIN contains a one- or two-letter issuing-region prefix followed by five
    to seven decimal digits. There is no public checksum, so detection also
    requires nearby identity-card context.
    """
    if not isinstance(text, str):
        return False
    normalized = normalize_arabic_indic_digits(text.strip()).upper()
    match = re.fullmatch(r"([A-Z]{1,2})[0-9]{5,7}", normalized)
    return match is not None and all(
        letter in MOROCCAN_CIN_REGION_LETTERS for letter in match.group(1)
    )


def validate_za_id_number(text: str) -> bool:
    """Validate a South African 13-digit identity number.

    South African identity numbers use ``YYMMDDSSSSCAZ``: an embedded birth
    date, a four-digit sequence, a citizenship digit (``0`` or ``1``), a
    legacy classification digit, and a final Luhn check digit. The two-digit
    year has no century marker, so the date is accepted when it exists in
    either the 1900s or 2000s.

    Args:
        text: Candidate containing exactly 13 ASCII digits.

    Returns:
        ``True`` when the shape, date, citizenship digit, and checksum are
        valid.
    """
    if not isinstance(text, str):
        return False

    digits = text.strip()
    if re.fullmatch(r"[0-9]{13}", digits) is None:
        return False

    year = int(digits[:2])
    month = int(digits[2:4])
    day = int(digits[4:6])
    if not any(
        _is_valid_calendar_date(century + year, month, day) for century in (1900, 2000)
    ):
        return False
    if digits[10] not in {"0", "1"}:
        return False

    total = 0
    for index, value in enumerate(digits):
        digit = int(value)
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _is_valid_calendar_date(year: int, month: int, day: int) -> bool:
    """Return whether ``year``/``month``/``day`` form a Gregorian date."""
    try:
        date(year, month, day)
    except ValueError:
        return False
    return True


validate_south_african_id = validate_za_id_number


# ---------------------------------------------------------------------------
# National ID Validators
# ---------------------------------------------------------------------------


def _is_nigeria_sequential_run(digits: str) -> bool:
    """Return whether an 11-digit value contains a full ascending/descending run."""

    return "0123456789" in digits or "9876543210" in digits


def validate_nigeria_nin(text: str) -> bool:
    """Validate the offline structural rules for a Nigerian NIN.

    Nigerian National Identification Numbers contain exactly 11 ASCII digits
    and have no public checksum. OpenMed therefore rejects only unmistakable
    placeholder values: repeated single digits and complete ascending or
    descending digit runs. Detection remains context-gated to avoid treating
    arbitrary 11-digit values as identifiers.

    Args:
        text: Candidate containing exactly 11 ASCII digits.

    Returns:
        ``True`` when the candidate has a non-trivial NIN structure.
    """
    if not isinstance(text, str):
        return False
    digits = text.strip()
    if re.fullmatch(r"[0-9]{11}", digits) is None:
        return False
    if len(set(digits)) == 1:
        return False
    return not _is_nigeria_sequential_run(digits)


def validate_nigeria_bvn(text: str) -> bool:
    """Validate the offline structural rules for a Nigerian BVN.

    Bank Verification Numbers contain exactly 11 ASCII digits. No public
    checksum or leading-digit allocation rule is documented, so OpenMed uses
    the same conservative non-triviality checks as NIN validation and keeps
    detection context-gated.

    Args:
        text: Candidate containing exactly 11 ASCII digits.

    Returns:
        ``True`` when the candidate has a non-trivial BVN structure.
    """
    return validate_nigeria_nin(text)


def validate_ghana_card_pin(text: str) -> bool:
    """Validate the documented offline structure of a Ghana Card PIN.

    NIA documents the ``GHA-#########-#`` form and three-letter nationality
    prefixes for resident cards, but does not publish an offline checksum
    algorithm. Detection therefore remains context-gated; authoritative card
    verification requires the NIA Identity Verification System Platform.

    Args:
        text: Candidate Ghana Card PIN.

    Returns:
        ``True`` when the candidate has the documented ASCII shape.
    """
    return bool(
        isinstance(text, str)
        and re.fullmatch(
            r"[A-Z]{3}-[0-9]{9}-[0-9]",
            text.strip().upper(),
        )
    )


def validate_kenya_national_id(text: str) -> bool:
    """Validate the offline structure of a legacy Kenyan national ID.

    Kenya's second-generation identity numbers contain seven or eight digits
    and have no public checksum. Recognition therefore requires identity
    context at the pattern layer.

    Args:
        text: Candidate containing exactly seven or eight ASCII digits.

    Returns:
        ``True`` when the candidate has the expected structure.
    """
    return (
        isinstance(text, str) and re.fullmatch(r"[0-9]{7,8}", text.strip()) is not None
    )


def validate_kenya_maisha_namba(text: str) -> bool:
    """Validate the documented nine-digit structure of a Kenya Maisha Namba.

    The identifier has no public checksum, so recognition also requires nearby
    Maisha, Huduma, or UPI context at the pattern layer.

    Args:
        text: Candidate containing exactly nine ASCII digits.

    Returns:
        ``True`` when the candidate has the expected structure.
    """
    return isinstance(text, str) and re.fullmatch(r"[0-9]{9}", text.strip()) is not None


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


def validate_hungarian_taj(text: str) -> bool:
    """Validate a Hungarian TAJ social-security identifier.

    TAJ values contain eight serial digits followed by a check digit. The
    first eight digits use alternating weights 3 and 7, starting with 3; the
    weighted sum modulo 10 must equal the ninth digit, as specified by Annex 2
    of Hungary's Act XX of 1996. Plain nine-digit values and the commonly
    grouped ``NNN NNN NNN`` / ``NNN-NNN-NNN`` forms are accepted.

    Args:
        text: Candidate TAJ value.

    Returns:
        True when the value has a supported format and a valid check digit.
    """
    if not isinstance(text, str):
        return False

    stripped = text.strip()
    if re.fullmatch(r"[0-9]{9}|[0-9]{3}([ -])[0-9]{3}\1[0-9]{3}", stripped) is None:
        return False

    digits = re.sub(r"[ -]", "", stripped)
    numbers = [int(digit) for digit in digits]
    total = sum(
        digit * (3 if index % 2 == 0 else 7) for index, digit in enumerate(numbers[:8])
    )
    return numbers[8] == total % 10


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
        text: Aadhaar string, either as 12 digits or in the UIDAI 4-4-4
            display form.

    Returns:
        True if the Aadhaar passes the Verhoeff checksum
    """
    candidate = text.strip()
    if (
        re.fullmatch(
            r"[2-9][0-9]{11}|[2-9][0-9]{3} [0-9]{4} [0-9]{4}",
            candidate,
        )
        is None
    ):
        return False
    digits = candidate.replace(" ", "")

    # Verhoeff checksum
    c = 0
    for i, digit in enumerate(reversed(digits)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(digit)]]
    return c == 0


CHINESE_RESIDENT_ID_REGION_PREFIXES = frozenset(
    {
        "11",
        "12",
        "13",
        "14",
        "15",
        "21",
        "22",
        "23",
        "31",
        "32",
        "33",
        "34",
        "35",
        "36",
        "37",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "50",
        "51",
        "52",
        "53",
        "54",
        "61",
        "62",
        "63",
        "64",
        "65",
    }
)
"""Mainland province-level prefixes from the GB/T 2260 code hierarchy.

The full county-level table changes over time. OpenMed deliberately bundles
only this small, stable first-level set and validates the remaining four
digits structurally, avoiding a stale or restrictively licensed data asset.
"""

_CHINESE_RESIDENT_ID_WEIGHTS = (
    7,
    9,
    10,
    5,
    8,
    4,
    2,
    1,
    6,
    3,
    7,
    9,
    10,
    5,
    8,
    4,
    2,
)
_CHINESE_RESIDENT_ID_CHECK_DIGITS = "10X98765432"

_PAKISTANI_CNIC_EASTERN_DIGITS = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹",
    "0123456789",
)
_PAKISTANI_CNIC_RE = re.compile(r"(?:[0-9]{5}-[0-9]{7}-[0-9]|[0-9]{13})")


def validate_pakistani_cnic(text: str) -> bool:
    """Validate the public dashed or undashed Pakistani CNIC format.

    Persian/Eastern Arabic-Indic digits are normalized to ASCII before the
    structural check. CNIC has no public checksum, so this intentionally does
    not claim that a matching value was issued by NADRA.
    """
    if not isinstance(text, str):
        return False
    normalized = text.translate(_PAKISTANI_CNIC_EASTERN_DIGITS)
    return _PAKISTANI_CNIC_RE.fullmatch(normalized) is not None


def chinese_resident_id_check_character(body: str) -> str:
    """Return the ISO 7064 MOD 11-2 check character for a 17-digit body.

    Args:
        body: The first 17 digits of a Chinese Resident Identity Card number.

    Returns:
        The decimal check digit or uppercase ``"X"``.

    Raises:
        ValueError: If ``body`` is not exactly 17 ASCII digits.
    """
    if re.fullmatch(r"[0-9]{17}", body) is None:
        raise ValueError("Chinese Resident ID body must contain 17 ASCII digits")

    total = sum(
        int(digit) * weight for digit, weight in zip(body, _CHINESE_RESIDENT_ID_WEIGHTS)
    )
    return _CHINESE_RESIDENT_ID_CHECK_DIGITS[total % 11]


def validate_chinese_resident_id(text: str) -> bool:
    """Validate a mainland China 18-digit Resident Identity Card number.

    The second-generation resident identity card stores a six-digit address
    code, an eight-digit Gregorian birth date, a three-digit sequence, and a
    MOD 11-2 checksum character. Region validation deliberately uses the
    stable province-level GB/T 2260 prefixes plus structural county digits,
    rather than bundling a mutable full administrative-code dataset.

    Args:
        text: Candidate identifier. The final check character is
            case-insensitive, but separators are not accepted.

    Returns:
        ``True`` only when format, region, birth date, sequence, and checksum
        are all valid.
    """
    if not isinstance(text, str):
        return False

    cleaned = text.upper()
    if re.fullmatch(r"[0-9]{17}[0-9X]", cleaned) is None:
        return False

    region_code = cleaned[:6]
    prefecture_code = int(region_code[2:4])
    if (
        region_code[:2] not in CHINESE_RESIDENT_ID_REGION_PREFIXES
        or (prefecture_code > 70 and prefecture_code != 90)
        or prefecture_code == 0
        or region_code[4:] == "00"
    ):
        return False

    try:
        birth_date = date.fromisoformat(
            f"{cleaned[6:10]}-{cleaned[10:12]}-{cleaned[12:14]}"
        )
    except ValueError:
        return False

    if birth_date > date.today() or cleaned[14:17] == "000":
        return False

    return cleaned[-1] == chinese_resident_id_check_character(cleaned[:17])


def validate_chinese_resident_identity_card(text: str) -> bool:
    """Compatibility alias for :func:`validate_chinese_resident_id`."""

    return validate_chinese_resident_id(text)


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


# Portuguese NIF/NIPC valid leading digits (individuals, companies, public
# bodies, sole traders) and two-digit prefixes (non-resident individuals and
# special taxpayer categories).
_PORTUGUESE_NIF_FIRST_DIGITS: frozenset[str] = frozenset("1235689")
_PORTUGUESE_NIF_PREFIXES: frozenset[str] = frozenset(
    {"45", "70", "71", "72", "74", "75", "77", "79", "90", "91", "98", "99"}
)


def validate_portuguese_nif(text: str) -> bool:
    """Validate a Portuguese NIF/NIPC tax identification number.

    The NIF (individuals) and NIPC (entities) share a nine-digit format
    with a weighted modulo-11 check digit. The leading digit — or, for
    some taxpayer categories, the leading two digits — must fall in the
    documented issuance set. This is distinct from the Brazilian CPF/CNPJ
    validated elsewhere in the Portuguese pack.
    """
    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 9:
        return False
    if (
        digits[0] not in _PORTUGUESE_NIF_FIRST_DIGITS
        and digits[:2] not in _PORTUGUESE_NIF_PREFIXES
    ):
        return False

    total = sum(int(digits[index]) * (9 - index) for index in range(8))
    check = 11 - (total % 11)
    if check >= 10:
        check = 0
    return check == int(digits[8])


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


def validate_vietnamese_cccd(text: str) -> bool:
    """Validate the offline-verifiable structure of a Vietnamese CCCD.

    Article 12 of Vietnam's 2023 Law on Identification publicly guarantees a
    natural-number sequence of 12 digits. Earlier province, century, gender,
    and birth-year encoding rules expired with Circular 59/2021/TT-BCA on
    1 July 2024, and no public checksum is available. OpenMed therefore checks
    only the current public length and digit contract; detection patterns
    require nearby CCCD context to avoid treating arbitrary clinical numbers
    as identifiers.

    Args:
        text: CCCD value, contiguous or grouped as four three-digit blocks.

    Returns:
        True when the candidate contains exactly 12 digits in a supported
        presentation.
    """
    stripped = text.strip()
    supported_shape = r"\d{12}|\d{3}(?:\s+\d{3}){3}|\d{3}(?:-\d{3}){3}"
    return re.fullmatch(supported_shape, stripped) is not None


def validate_vietnamese_cmnd(text: str) -> bool:
    """Validate the local structure of a legacy Vietnamese CMND.

    Legacy CMND values have nine digits and no public checksum. To keep this
    structural validator conservative, all-zero and repeated-digit values are
    rejected. The corresponding detection pattern additionally requires an
    explicit CMND or ``chung minh nhan dan`` context cue.

    Args:
        text: CMND value, contiguous or grouped as three three-digit blocks.

    Returns:
        True when the candidate has a plausible legacy CMND structure.
    """
    stripped = text.strip()
    if re.fullmatch(r"\d{9}|\d{3}(?:[\s-]+\d{3}){2}", stripped) is None:
        return False

    digits = re.sub(r"[^0-9]", "", stripped)
    return len(set(digits)) > 1


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


def validate_greek_amka(text: str) -> bool:
    """Validate a Greek AMKA social-security number.

    The AMKA is an 11-digit code whose first six digits encode a birth
    date as ``DDMMYY`` and whose full value carries a Luhn check digit.
    The two-digit year is century-ambiguous, so a date valid in either
    the 1900s or the 2000s is accepted.
    """

    digits = re.sub(r"[^0-9]", "", text)

    if len(digits) != 11:
        return False
    if not _passes_luhn(digits):
        return False

    day = int(digits[0:2])
    month = int(digits[2:4])
    year_suffix = int(digits[4:6])
    if month < 1 or month > 12:
        return False

    import calendar

    for century in (1900, 2000):
        year = century + year_suffix
        try:
            if 1 <= day <= calendar.monthrange(year, month)[1]:
                return True
        except (ValueError, calendar.IllegalMonthError):
            continue
    return False


def _passes_luhn(digits: str) -> bool:
    """Return whether an all-digit string satisfies the Luhn checksum."""
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_FINNISH_HETU_CHECK_ALPHABET = "0123456789ABCDEFHJKLMNPRSTUVWXY"
_FINNISH_HETU_CENTURY_SIGNS = {
    **{"+": 1800},
    **{sign: 1900 for sign in "-YXWVU"},
    **{sign: 2000 for sign in "ABCDEF"},
}
_FINNISH_HETU_RE = re.compile(
    r"^(\d{2})(\d{2})(\d{2})([-+YXWVUABCDEF])(\d{3})([0-9ABCDEFHJKLMNPRSTUVWXY])$"
)


def validate_finnish_hetu(text: str) -> bool:
    """Validate Finnish HETU personal identity code.

    The HETU is ``DDMMYYCZZZQ``:

    - DDMMYY: birth date within the century selected by C.
    - C: century sign — ``+`` for the 1800s, ``-`` (or reform signs
      ``Y``/``X``/``W``/``V``/``U``) for the 1900s, and ``A``-``F`` for
      the 2000s.
    - ZZZ: individual serial number.
    - Q: modulo-31 check character over the nine digits ``DDMMYYZZZ``,
      mapped through the alphabet ``0-9ABCDEFHJKLMNPRSTUVWXY``.
    """

    if not isinstance(text, str):
        return False

    match = _FINNISH_HETU_RE.fullmatch(text)
    if match is None:
        return False

    day_text, month_text, year_text, sign, serial, check = match.groups()
    if not 2 <= int(serial) <= 899:
        return False

    expected = _FINNISH_HETU_CHECK_ALPHABET[
        int(day_text + month_text + year_text + serial) % 31
    ]
    if check != expected:
        return False

    year = _FINNISH_HETU_CENTURY_SIGNS[sign] + int(year_text)
    month = int(month_text)
    day = int(day_text)

    import calendar

    try:
        return 1 <= day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


def validate_bulgarian_egn(text: str) -> bool:
    """Validate Bulgarian EGN unified civil number.

    The EGN is a 10-digit code ``YYMMDDRRRC``:

    - YYMMDD: birth date with a century-offset month — months 01-12 are
      births in the 1900s, 21-32 (month + 20) the 1800s, and 41-52
      (month + 40) the 2000s.
    - RRR: region/serial digits; the ninth digit encodes sex.
    - C: weighted check digit; see :func:`_bulgarian_egn_check_digit`.
    """

    if not isinstance(text, str) or re.fullmatch(r"[0-9]{10}", text) is None:
        return False

    digits = text
    numbers = [int(digit) for digit in digits]
    if numbers[9] != _bulgarian_egn_check_digit(numbers[:9]):
        return False

    year = int(digits[0:2])
    month_raw = int(digits[2:4])
    day = int(digits[4:6])
    if 41 <= month_raw <= 52:
        year += 2000
        month = month_raw - 40
    elif 21 <= month_raw <= 32:
        year += 1800
        month = month_raw - 20
    elif 1 <= month_raw <= 12:
        year += 1900
        month = month_raw
    else:
        return False

    import calendar

    try:
        return 1 <= day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


def _bulgarian_egn_check_digit(digits: list[int]) -> int:
    """Return the Bulgarian EGN check digit for the first 9 digits.

    Weighted sum with weights 2,4,8,5,10,9,7,3,6 modulo 11; a remainder
    of 10 yields check digit 0.
    """
    weights = (2, 4, 8, 5, 10, 9, 7, 3, 6)
    remainder = sum(weight * digit for weight, digit in zip(weights, digits)) % 11
    return 0 if remainder == 10 else remainder


def validate_croatian_oib(text: str) -> bool:
    """Validate Croatian OIB personal identification number.

    The OIB is an 11-digit code with no embedded structure; the final
    digit is an ISO 7064 MOD 11,10 check over the first 10 digits; see
    :func:`_croatian_oib_check_digit`.
    """

    if not isinstance(text, str) or re.fullmatch(r"[0-9]{11}", text) is None:
        return False

    digits = text
    numbers = [int(digit) for digit in digits]
    return numbers[10] == _croatian_oib_check_digit(numbers[:10])


def _croatian_oib_check_digit(digits: list[int]) -> int:
    """Return the ISO 7064 MOD 11,10 check digit for the first 10 digits."""
    partial = 10
    for digit in digits:
        partial = (partial + digit) % 10
        if partial == 0:
            partial = 10
        partial = (partial * 2) % 11
    return (11 - partial) % 10


def validate_jmbg(text: str) -> bool:
    """Validate Serbian / ex-Yugoslav JMBG unique master citizen number.

    The JMBG is a 13-digit code ``DDMMYYYRRSSSK``:

    - DDMMYYY: birth date; YYY holds the last three digits of the year
      (values of 800 and above map to the 1800/1900s, lower values to
      the 2000s).
    - RR: political region of registration (all values accepted).
    - SSS: serial number; 000-499 male, 500-999 female.
    - K: modulo-11 check digit; see :func:`_jmbg_check_digit`.
    """

    if not isinstance(text, str) or re.fullmatch(r"[0-9]{13}", text) is None:
        return False

    digits = text
    numbers = [int(digit) for digit in digits]
    if numbers[12] != _jmbg_check_digit(numbers[:12]):
        return False

    day = int(digits[0:2])
    month = int(digits[2:4])
    year_suffix = int(digits[4:7])
    year = 1000 + year_suffix if year_suffix >= 800 else 2000 + year_suffix

    import calendar

    try:
        return 1 <= day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


def _jmbg_check_digit(digits: list[int]) -> int:
    """Return the JMBG check digit for the first 12 digits.

    ``m = 11 - ((7*(d1+d7) + 6*(d2+d8) + 5*(d3+d9) + 4*(d4+d10) +
    3*(d5+d11) + 2*(d6+d12)) mod 11)``; remainders of 10 or 11 yield
    check digit 0.
    """
    weights = (7, 6, 5, 4, 3, 2)
    total = sum(
        weight * (digits[index] + digits[index + 6])
        for index, weight in enumerate(weights)
    )
    remainder = 11 - (total % 11)
    return 0 if remainder > 9 else remainder


def validate_estonian_isikukood(text: str) -> bool:
    """Validate Estonian isikukood.

    The isikukood is an 11-digit personal code ``GYYMMDDSSSC``:

    - G: century and sex (1/2 = 1800s, 3/4 = 1900s, 5/6 = 2000s;
      odd = male, even = female).
    - YYMMDD: date of birth within that century.
    - SSS: serial number.
    - C: two-pass modulo-11 check digit; see
      :func:`_estonian_isikukood_check_digit`.
    """

    if not isinstance(text, str) or re.fullmatch(r"[0-9]{11}", text) is None:
        return False

    digits = text
    numbers = [int(digit) for digit in digits]
    if not 1 <= numbers[0] <= 6:
        return False
    if numbers[10] != _estonian_isikukood_check_digit(numbers[:10]):
        return False

    year = 1800 + ((numbers[0] - 1) // 2) * 100 + int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])

    import calendar

    try:
        return 1 <= day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


def _estonian_isikukood_check_digit(digits: list[int]) -> int:
    """Return the Estonian isikukood check digit for the first 10 digits.

    First pass uses weights 1..9,1 modulo 11; a remainder of 10 triggers a
    second pass with weights 3..9,1,2,3, and a second remainder of 10 yields
    check digit 0.
    """
    for weights in (
        (1, 2, 3, 4, 5, 6, 7, 8, 9, 1),
        (3, 4, 5, 6, 7, 8, 9, 1, 2, 3),
    ):
        remainder = sum(weight * digit for weight, digit in zip(weights, digits)) % 11
        if remainder < 10:
            return remainder
    return 0


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

    Modern values use ten digits in the shape ``YYMMDDXXXX`` and the whole
    value must be divisible by 11. Legacy values assigned to people born
    before 1954 use nine digits and no checksum. Female identifiers add 50
    to the birth month; post-2004 overflow series may add 20 for men or 70
    for women.

    Args:
        text: Birth number string, optionally containing a slash, spaces, or
            hyphen separators.

    Returns:
        True if the identifier has a valid shape, birth date, serial, and
        (for ten-digit values) modulo-11 checksum.
    """
    if not isinstance(text, str):
        return False

    stripped = text.strip()
    if re.fullmatch(r"[0-9]{9,10}|[0-9]{6}[ /-][0-9]{3,4}", stripped) is None:
        return False

    digits = re.sub(r"[^0-9]", "", stripped)
    legacy = len(digits) == 9
    if legacy:
        if digits[6:] == "000":
            return False
    elif int(digits) % 11 != 0:
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

    if legacy and overflow_series:
        return False
    if day < 1 or day > 31:
        return False

    import calendar

    if legacy:
        year = (1900 if year_suffix <= 53 else 1800) + year_suffix
    else:
        year = (2000 if year_suffix <= 53 else 1900) + year_suffix
        if overflow_series and year < 2004:
            return False

    try:
        return 1 <= day <= calendar.monthrange(year, month)[1]
    except (ValueError, calendar.IllegalMonthError):
        return False


# Czech and Slovak share the Czechoslovak birth-number system. Keep a
# Czech-facing public name for issue-specific callers without duplicating logic.
validate_czech_rodne_cislo = validate_czechoslovak_rodne_cislo


# Romanian CNP location/sequence codes: legacy values 01-46 cover the counties,
# Bucharest, and its sectors; 51-52 cover Calarasi and Giurgiu. Current SIIEASC
# issuance uses 70 nationwide. Codes 47-50 are unassigned and must not be
# accepted as a numeric range shortcut.
_ROMANIAN_CNP_COUNTY_CODES: frozenset[int] = frozenset(set(range(1, 47)) | {51, 52, 70})

# Documented CNP control-digit weights ("279146358279").
_ROMANIAN_CNP_WEIGHTS: tuple[int, ...] = (2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9)

# CNP first digit (S) encodes century and gender.
_ROMANIAN_CNP_CENTURY: Dict[int, int] = {
    1: 1900,  # male, 1900-1999
    2: 1900,  # female, 1900-1999
    3: 1800,  # male, 1800-1899
    4: 1800,  # female, 1800-1899
    5: 2000,  # male, 2000-2099
    6: 2000,  # female, 2000-2099
}


def validate_romanian_cnp(text: str) -> bool:
    """Validate a Romanian CNP (Cod Numeric Personal).

    The CNP is a 13-digit national identifier with the layout
    ``S YY MM DD JJ NNN C``:

    - ``S``: documented century and gender code (1-6). Codes 1/3/5 are male
      and 2/4/6 female; 1/2 map to the 1900s, 3/4 to the 1800s, and 5/6 to
      the 2000s.
    - ``YYMMDD``: non-future date of birth. The century is taken from ``S``.
    - ``JJ``: legacy county/sector code 01-46 or 51-52, or the current
      nationwide SIIEASC sequence code 70. Unassigned codes 47-50 are invalid.
    - ``NNN``: non-zero sequence number.
    - ``C``: control digit, computed as the sum of the first twelve digits
      weighted by the documented constant ``279146358279`` taken modulo 11.
      A remainder of 10 maps to a control digit of 1.

    Args:
        text: CNP string containing exactly 13 ASCII digits. Surrounding
            whitespace is ignored, but internal separators are invalid.

    Returns:
        True if the CNP has a valid structure, birth date, county code, and
        control digit.
    """
    if not isinstance(text, str):
        return False
    digits = text.strip()
    if re.fullmatch(r"[0-9]{13}", digits) is None:
        return False

    numbers = [int(digit) for digit in digits]

    # --- century/gender code ---
    gender_code = numbers[0]
    if gender_code not in _ROMANIAN_CNP_CENTURY:
        return False
    century = _ROMANIAN_CNP_CENTURY[gender_code]

    # --- embedded birth date ---
    year = century + numbers[1] * 10 + numbers[2]
    month = numbers[3] * 10 + numbers[4]
    day = numbers[5] * 10 + numbers[6]

    try:
        birth_date = date(year, month, day)
    except ValueError:
        return False
    if birth_date > date.today():
        return False

    # --- county code ---
    county = numbers[7] * 10 + numbers[8]
    if county not in _ROMANIAN_CNP_COUNTY_CODES:
        return False

    # --- non-zero sequence number ---
    serial = numbers[9] * 100 + numbers[10] * 10 + numbers[11]
    if serial == 0:
        return False

    # --- control digit ---
    total = sum(w * n for w, n in zip(_ROMANIAN_CNP_WEIGHTS, numbers[:12]))
    control = total % 11
    if control == 10:
        control = 1

    return numbers[12] == control


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
    # Gregorian month names used by the existing date parser. Ethiopian-calendar
    # conversion is intentionally outside this language pack's scope.
    "am": [
        "ጃንዋሪ",
        "ፌብሩዋሪ",
        "ማርች",
        "ኤፕሪል",
        "ሜይ",
        "ጁን",
        "ጁላይ",
        "ኦገስት",
        "ሴፕቴምበር",
        "ኦክቶበር",
        "ኖቬምበር",
        "ዲሴምበር",
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
    "sw": [
        "Januari",
        "Februari",
        "Machi",
        "Aprili",
        "Mei",
        "Juni",
        "Julai",
        "Agosti",
        "Septemba",
        "Oktoba",
        "Novemba",
        "Desemba",
    ],
    "zu": [
        "Januwari",
        "Februwari",
        "Mashi",
        "Ephreli",
        "Meyi",
        "Juni",
        "Julayi",
        "Agasti",
        "Septhemba",
        "Okthoba",
        "Novemba",
        "Disemba",
    ],
    "xh": [
        "Janyuwari",
        "Februwari",
        "Matshi",
        "Epreli",
        "Meyi",
        "Juni",
        "Julayi",
        "Agasti",
        "Septemba",
        "Okthobha",
        "Novemba",
        "Disemba",
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
    "ro": [
        "ianuarie",
        "februarie",
        "martie",
        "aprilie",
        "mai",
        "iunie",
        "iulie",
        "august",
        "septembrie",
        "octombrie",
        "noiembrie",
        "decembrie",
    ],
    "zh": [
        "一月",
        "二月",
        "三月",
        "四月",
        "五月",
        "六月",
        "七月",
        "八月",
        "九月",
        "十月",
        "十一月",
        "十二月",
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

_EGYPT_NATIONAL_ID_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<!\d)[23٢٣۲۳]\d{13}(?!\d)",
        "national_id",
        priority=15,
        base_score=0.8,
        context_words=[
            "national id",
            "national id number",
            "egyptian id",
            "الرقم القومي",
            "رقم قومي",
            "بطاقة الرقم القومي",
            "رقم الهوية",
        ],
        context_boost=0.15,
        validator=validate_egyptian_national_id,
    ),
]


_MOROCCO_CIN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<![A-Z0-9])[A-Z]{1,2}\d{5,7}(?![A-Z0-9])",
        "national_id",
        priority=14,
        base_score=0.5,
        context_words=[
            "cin",
            "cin number",
            "carte nationale",
            "carte d'identité nationale",
            "carte d identite nationale",
            "بطاقة التعريف الوطنية",
            "بطاقة وطنية",
            "بيطاقة التعريف الوطنية",
            "bitaqa",
            "bitaqa watania",
        ],
        context_boost=0.45,
        validator=validate_moroccan_cin,
        requires_context=True,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

_NIGERIAN_PII_PATTERNS: List[PIIPattern] = [
    # Nigeria's NIN and BVN have no public checksums, so deterministic sweeps
    # require explicit identifier context. NIN precedes phone detection so an
    # explicitly labeled mobile-shaped identifier is consumed as an ID first.
    PIIPattern(
        r"(?<![0-9])[0-9]{11}(?![0-9])",
        "NG_NIN",
        priority=14,
        base_score=0.5,
        context_words=[
            "nin",
            "nimc",
            "national identification",
            "national identification number",
            "national identity number",
        ],
        context_boost=0.45,
        validator=validate_nigeria_nin,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"(?<![0-9])[0-9]{11}(?![0-9])",
        "NG_BVN",
        priority=13,
        base_score=0.5,
        context_words=[
            "bvn",
            "nibss",
            "bank verification",
            "bank verification number",
        ],
        context_boost=0.45,
        validator=validate_nigeria_bvn,
        safety_sweep_requires_context=True,
    ),
    # Nigerian mobile numbers use 070x/080x/081x/090x/091x domestically and
    # drop the leading zero after the +234 country code.
    PIIPattern(
        r"(?<![0-9])(?:\+234[\s.-]?(?:70|80|81|90|91)[0-9]|"
        r"0(?:70|80|81|90|91)[0-9])[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}(?![0-9])",
        "NG_PHONE",
        priority=12,
        base_score=0.7,
        context_words=[
            "phone",
            "mobile",
            "telephone",
            "contact",
            "waya",
            "ekwentị",
            "foonu",
        ],
        context_boost=0.25,
    ),
]


_GHANA_CARD_PII_PATTERNS: List[PIIPattern] = [
    # GH_GHANA_CARD: documented country prefix + ten digits. With no published
    # offline checksum, require explicit Ghana Card or NIA context.
    PIIPattern(
        r"(?<![A-Z0-9])[A-Z]{3}-[0-9]{9}-[0-9](?![A-Z0-9])",
        "GH_GHANA_CARD",
        priority=15,
        base_score=0.5,
        context_words=[
            "ghana card",
            "ghana card pin",
            "national identification authority",
            "nia pin",
        ],
        context_boost=0.45,
        validator=validate_ghana_card_pin,
        safety_sweep_requires_context=True,
    ),
]


_KENYA_ID_PII_PATTERNS: List[PIIPattern] = [
    # KE_MAISHA_NAMBA: structural UPI with mandatory nearby identifier context.
    PIIPattern(
        r"(?<![0-9])[0-9]{9}(?![0-9])",
        "KE_MAISHA_NAMBA",
        priority=15,
        base_score=0.5,
        context_words=[
            "maisha namba",
            "maisha number",
            "maisha card",
            "huduma namba",
            "huduma number",
            "unique personal identifier",
            "upi number",
            "nambari ya maisha",
        ],
        context_boost=0.45,
        validator=validate_kenya_maisha_namba,
        safety_sweep_requires_context=True,
    ),
    # KE_NATIONAL_ID: legacy seven/eight-digit number. Without an identity
    # keyword this shape is too common in labs, MRNs, and other clinical data.
    PIIPattern(
        r"(?<![0-9])[0-9]{7,8}(?![0-9])",
        "KE_NATIONAL_ID",
        priority=14,
        base_score=0.5,
        context_words=[
            "id no",
            "id number",
            "national id",
            "national identification number",
            "identity card number",
            "kitambulisho",
            "nambari ya kitambulisho",
            "nambari ya id",
        ],
        context_boost=0.45,
        validator=validate_kenya_national_id,
        safety_sweep_requires_context=True,
    ),
]

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
    # Australian Medicare card number (10 digits, ``NNNN NNNNN N``) with an
    # optional separate one-digit IRN. The main card's first eight digits carry
    # a weighted mod-10 checksum; the full matched identifier is protected.
    PIIPattern(
        r"\b(?:[2-6]\d{9}|[2-6]\d{3} \d{5} \d)"
        r"(?:(?:[ ]*/[ ]*|[ -]?)[1-9])?\b",
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
    # Australian Tax File Number (TFN, exactly 9 digits, ``NNN NNN NNN``
    # spacing; weighted mod-11 checksum).
    PIIPattern(
        r"\b(?:\d{9}|\d{3} \d{3} \d{3})\b",
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


# China Unified Social Credit Code (GB 32100-2015), ISO 7064 MOD 31-3 over a
# restricted 31-character alphabet that excludes the ambiguous letters I, O, S,
# V and Z. Position values are the index into this ordered alphabet.
USCC_ALPHABET = "0123456789ABCDEFGHJKLMNPQRTUWXY"
_USCC_VALUE = {char: index for index, char in enumerate(USCC_ALPHABET)}
_USCC_WEIGHTS = tuple(pow(3, i, 31) for i in range(17))
# Department/category pairs from GB 32100-2015 Amendment No. 1. Keeping the
# pairs together prevents a checksum-valid but structurally impossible prefix.
USCC_DEPARTMENT_CATEGORY_CODES = {
    "1": frozenset("1239"),
    "2": frozenset("19"),
    "3": frozenset("123459"),
    "4": frozenset("19"),
    "5": frozenset("1239"),
    "6": frozenset("129"),
    "7": frozenset("129"),
    "8": frozenset("19"),
    "9": frozenset("123"),
    "A": frozenset("19"),
    "N": frozenset("1239"),
    "Y": frozenset("1"),
}


def uscc_check_char(body17: str) -> str:
    """Return the ISO 7064 MOD 31-3 check character for a USCC body.

    Args:
        body17: Seventeen characters from :data:`USCC_ALPHABET`.

    Returns:
        The single checksum character.

    Raises:
        ValueError: If the body has the wrong length or contains a forbidden
            character.
    """

    if len(body17) != 17 or any(char not in _USCC_VALUE for char in body17):
        raise ValueError("USCC body must contain 17 characters from USCC_ALPHABET")
    total = sum(_USCC_VALUE[char] * _USCC_WEIGHTS[i] for i, char in enumerate(body17))
    return USCC_ALPHABET[(31 - (total % 31)) % 31]


def validate_unified_social_credit_code(text: str) -> bool:
    """Validate a China Unified Social Credit Code (18-char, ISO 7064 MOD 31-3).

    Checks the length, the restricted alphabet (excluding I/O/S/V/Z), the
    numeric administrative-region segment (positions 3-8), and the MOD 31-3
    check character. Returns ``False`` for any non-conforming input.
    """
    if not isinstance(text, str):
        return False
    code = text.strip()
    if len(code) != 18 or any(char not in _USCC_VALUE for char in code):
        return False
    categories = USCC_DEPARTMENT_CATEGORY_CODES.get(code[0])
    if categories is None or code[1] not in categories:
        return False
    # Positions 3-8 (0-indexed 2:8) are the 6-digit administrative region code.
    if not code[2:8].isdigit():
        return False
    return uscc_check_char(code[:17]) == code[17]


# Language-agnostic China Unified Social Credit Code pattern, guarded by the
# MOD 31-3 validator and always included in the universal base set.
USCC_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<![0-9A-Z])[0-9A-HJ-NP-RTUW-Y]{18}(?![0-9A-Z])",
        "social_credit_code",
        priority=15,
        base_score=0.6,
        context_words=[
            "统一社会信用代码",
            "信用代码",
            "unified social credit",
            "social credit code",
            "uscc",
        ],
        context_boost=0.4,
        validator=validate_unified_social_credit_code,
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

# Aadhaar is language-agnostic in Indian clinical notes: English labels and
# Romanized code-mixed text are common alongside Devanagari or Telugu cues.
# The strict validator gate prevents phone-like or random 12-digit values from
# being promoted to a national identifier by this deterministic recognizer.
AADHAAR_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<![0-9])[2-9][0-9]{3}(?P<aadhaar_sep> ?)[0-9]{4}"
        r"(?P=aadhaar_sep)[0-9]{4}(?![0-9])",
        "national_id",
        priority=12,
        base_score=0.55,
        context_words=[
            "आधार",
            "यूआईडीएआई",
            "पहचान",
            "aadhaar",
            "aadhar",
            "uid",
            "uidai",
            "unique identification",
            "ఆధార్",
            "గుర్తింపు",
        ],
        context_boost=0.4,
        validator=validate_aadhaar,
        reject_on_validation_failure=True,
        safety_sweep_requires_context=True,
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
    # Portuguese NIF / NIPC (9 digits, pt_PT)
    PIIPattern(
        r"\b\d{3}\s?\d{3}\s?\d{3}\b",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "nif",
            "nipc",
            "contribuinte",
            "n\u00famero de contribuinte",
            "numero de contribuinte",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        validator=validate_portuguese_nif,
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
]

_ETHIOPIC_SCRIPT_RANGES = (
    r"\u1200-\u135F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F"
    r"\U0001E7E0-\U0001E7FF"
)
_ETHIOPIC_NUMERAL_RANGE = r"\u1369-\u137C"

_AMHARIC_NAME_CONTEXT = [
    "ስም",
    "የታካሚ ስም",
    "ታካሚ",
    "name",
    "patient name",
]
_AMHARIC_DATE_CONTEXT = [
    "ቀን",
    "የትውልድ ቀን",
    "ትውልድ",
    "የቀጠሮ ቀን",
    "date",
    "date of birth",
    "dob",
]
_AMHARIC_AGE_CONTEXT = ["ዕድሜ", "እድሜ", "ዓመት", "age", "years old"]
_AMHARIC_ID_CONTEXT = [
    "ፋይዳ",
    "የፋይዳ መለያ ቁጥር",
    "መለያ ቁጥር",
    "መታወቂያ",
    "fayda",
    "fayda id",
    "id number",
]
_AMHARIC_PHONE_CONTEXT = [
    "ስልክ",
    "ስልክ ቁጥር",
    "ሞባይል",
    "phone",
    "mobile",
    "contact",
]
_AMHARIC_ADDRESS_CONTEXT = [
    "አድራሻ",
    "መኖሪያ",
    "ቦታ",
    "address",
    "location",
]
_AMHARIC_POSTCODE_CONTEXT = [
    "የፖስታ ኮድ",
    "ፖስታ ኮድ",
    "postal code",
    "postcode",
    "zip code",
]

# Ethiopic has no letter case, so case-insensitive matching is a no-op for the
# native-script patterns. ``re.IGNORECASE`` is used only by the Latin name
# overlay that protects explicitly labelled names in code-mixed notes.
_AMHARIC_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        rf"(?<=ስም[፡:] )[{_ETHIOPIC_SCRIPT_RANGES}]+"
        rf"(?:\s+[{_ETHIOPIC_SCRIPT_RANGES}]+){{1,3}}(?=[\s፡።,.;]|$)",
        "name",
        priority=12,
        base_score=0.65,
        context_words=_AMHARIC_NAME_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=0,
    ),
    PIIPattern(
        r"(?<=Patient name: )[A-Z][A-Za-z'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-z'’-]{1,30}){1,3}\b",
        "name",
        priority=12,
        base_score=0.65,
        context_words=_AMHARIC_NAME_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        rf"(?<!\w)[0-9{_ETHIOPIC_NUMERAL_RANGE}]{{1,12}}[./-]"
        rf"[0-9{_ETHIOPIC_NUMERAL_RANGE}]{{1,12}}[./-]"
        rf"[0-9{_ETHIOPIC_NUMERAL_RANGE}]{{1,16}}(?!\w)",
        "date",
        priority=11,
        base_score=0.65,
        context_words=_AMHARIC_DATE_CONTEXT,
        context_boost=0.3,
        flags=0,
    ),
    PIIPattern(
        rf"(?<=ዕድሜ[፡:] )[0-9{_ETHIOPIC_NUMERAL_RANGE}]{{1,8}}"
        r"(?=[\s፡።,.;]|$)",
        "age",
        priority=12,
        base_score=0.65,
        context_words=_AMHARIC_AGE_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=0,
    ),
    # Fayda recognition is format-only: exactly 12 decimal digits. No checksum
    # or validity claim is made because none is publicly specified.
    PIIPattern(
        r"(?<!\w)[0-9]{12}(?!\w)",
        "national_id",
        priority=14,
        base_score=0.5,
        context_words=_AMHARIC_ID_CONTEXT,
        context_boost=0.45,
        requires_context=True,
        safety_sweep_requires_context=True,
        flags=0,
    ),
    # Traditional Ethiopic numerals are non-positional Unicode ``No`` values,
    # not decimal digits. Preserve labelled tokens without treating them as
    # 12-digit Fayda values.
    PIIPattern(
        rf"(?<=መለያ ቁጥር[፡:] )[{_ETHIOPIC_NUMERAL_RANGE}]{{2,24}}"
        r"(?=[\s፡።,.;]|$)",
        "national_id",
        priority=13,
        base_score=0.5,
        context_words=_AMHARIC_ID_CONTEXT,
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=0,
    ),
    PIIPattern(
        r"(?<!\d)(?:\+251[\s.-]?9\d{2}|09\d{2})"
        r"[\s.-]?\d{3}[\s.-]?\d{3}(?!\d)",
        "phone_number",
        priority=12,
        base_score=0.7,
        context_words=_AMHARIC_PHONE_CONTEXT,
        context_boost=0.25,
        flags=0,
    ),
    PIIPattern(
        rf"(?<=አድራሻ[፡:] )[{_ETHIOPIC_SCRIPT_RANGES}]+"
        rf"(?:\s+[{_ETHIOPIC_SCRIPT_RANGES}]+){{1,6}}"
        r"(?:\s+[0-9]{1,5})?(?=[፡።,.;]|$)",
        "street_address",
        priority=10,
        base_score=0.65,
        context_words=_AMHARIC_ADDRESS_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=0,
    ),
    PIIPattern(
        r"(?<=የፖስታ ኮድ[፡:] )[0-9]{4}(?![0-9])",
        "postcode",
        priority=8,
        base_score=0.35,
        context_words=_AMHARIC_POSTCODE_CONTEXT,
        context_boost=0.45,
        requires_context=True,
        safety_sweep_requires_context=True,
        flags=0,
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

_URDU_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "\u062a\u0627\u0631\u06cc\u062e",
            "\u067e\u06cc\u062f\u0627\u0626\u0634",
            "\u062a\u0627\u0631\u06cc\u062e \u067e\u06cc\u062f\u0627\u0626\u0634",
            "\u062f\u0627\u062e\u0644\u06c1",
            "\u0688\u0633\u0686\u0627\u0631\u062c",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+92[\s.-]?|0)\d{2,3}(?:[\s.-]?\d{3,4}){1,2}\b",
        "phone_number",
        priority=8,
        base_score=0.45,
        context_words=[
            "\u0641\u0648\u0646",
            "\u0645\u0648\u0628\u0627\u0626\u0644",
            "\u0631\u0627\u0628\u0637\u06c1",
        ],
        context_boost=0.4,
    ),
    PIIPattern(
        # Pakistani CNIC: 13 digits, formatted XXXXX-XXXXXXX-X or undashed.
        # No public checksum exists (NADRA does not publish one); this is a
        # format-only match. Also accepts Eastern Arabic-Indic digits since
        # Urdu-script notes may be hand-typed with them, though issued CNIC
        # cards themselves use Western digits.
        r"\b[0-9\u06F0-\u06F9]{5}-[0-9\u06F0-\u06F9]{7}-[0-9\u06F0-\u06F9]\b"
        r"|\b[0-9\u06F0-\u06F9]{13}\b",
        "national_id",
        priority=9,
        base_score=0.4,
        context_words=[
            "\u0634\u0646\u0627\u062e\u062a\u06cc \u06a9\u0627\u0631\u0688",
            "\u0642\u0648\u0645\u06cc \u0634\u0646\u0627\u062e\u062a\u06cc \u06a9\u0627\u0631\u0688",
            "\u06a9\u0627\u0631\u0688 \u0646\u0645\u0628\u0631",
        ],
        context_boost=0.5,
        validator=validate_pakistani_cnic,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"\b(?:\u0645\u062d\u0644\u06c1|\u06af\u0644\u06cc|\u0633\u0691\u06a9|\u0628\u0644\u0627\u06a9)\s+[\u0600-\u06D3\u06D5-\u06FF0-9\s]{3,40}(?=[۔،\n]|$)",
        "street_address",
        priority=7,
        base_score=0.6,
        context_words=[
            "\u067e\u062a\u06c1",
            "\u0631\u06c1\u0627\u0626\u0634",
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
            "\u067e\u0648\u0633\u0679\u0644 \u06a9\u0648\u0688",
            "\u0632\u067e \u06a9\u0648\u0688",
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

_CHINESE_SURNAME_ALTERNATION = "|".join(
    re.escape(surname)
    for surname in sorted(CHINESE_SURNAMES, key=lambda value: (-len(value), value))
)

# Chinese names are contiguous, so ordinary ``\b`` boundaries cannot separate a
# family name from surrounding Han text. The left side accepts either a true Han
# boundary or a strong clinical/name-field cue; the right side accepts normal
# punctuation/whitespace plus common clinical continuations. The latter makes
# ``患者王伟今日复诊`` resolve to ``王伟`` rather than ``王伟今``.
_CHINESE_NAME_LEFT_BOUNDARY = (
    rf"(?:(?<![{_HAN_CHARACTER_CLASS}])|(?<=患者)|(?<=病人)|(?<=病患)|"
    r"(?<=姓名为)|(?<=姓名是)|(?<=姓名：)|(?<=姓名:))"
)
_CHINESE_NAME_RIGHT_BOUNDARY = (
    rf"(?=$|[^{_HAN_CHARACTER_CLASS}]|今日|因|于|来|现|复诊|就诊|主诉|"
    r"报告|表示|诉|接受|返回)"
)

_CHINESE_IDENTIFIER_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<![0-9])(?:\+86[ -]?)?1[3-9][0-9]{9}(?![0-9])",
        "phone_number",
        priority=14,
        base_score=0.75,
        context_words=["手机", "手机号", "电话", "联系电话"],
        context_boost=0.2,
        validator=validate_chinese_mobile_number,
    ),
    PIIPattern(
        r"(?<![0-9])(?:[0-9][ -]?){15,18}[0-9](?![0-9])",
        "credit_card",
        priority=15,
        base_score=0.55,
        context_words=["银行卡", "银行卡号", "卡号", "银联卡"],
        context_boost=0.4,
        validator=validate_chinese_bank_card,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"(?<![0-9A-Z])[EGDSP][0-9]{8}(?![0-9A-Z])",
        "chinese_passport",
        priority=14,
        base_score=0.6,
        context_words=["护照", "护照号", "护照号码", "旅行证件"],
        context_boost=0.35,
        validator=validate_chinese_passport,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<![0-9A-Z])[HM][0-9]{8}(?![0-9A-Z])",
        "home_return_permit",
        priority=14,
        base_score=0.6,
        context_words=["回乡证", "港澳居民来往内地通行证", "港澳居民通行证"],
        context_boost=0.35,
        validator=validate_hong_kong_macau_permit,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<![0-9A-Z])[0-9]{8}(?![0-9])",
        "taiwan_compatriot_permit",
        priority=13,
        base_score=0.55,
        context_words=["台胞证", "台湾居民来往大陆通行证", "台湾居民通行证"],
        context_boost=0.4,
        validator=validate_taiwan_compatriot_permit,
        safety_sweep_requires_context=True,
    ),
]

_CHINESE_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<![0-9])[0-9]{17}[0-9Xx](?![0-9A-Za-z])",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "居民身份证",
            "公民身份号码",
            "身份证号码",
            "身份证号",
            "身份证",
            "身份号码",
            "证件号码",
        ],
        context_boost=0.5,
        validator=validate_chinese_resident_id,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        _CHINESE_NAME_LEFT_BOUNDARY
        + rf"(?:{_CHINESE_SURNAME_ALTERNATION})"
        + rf"[{_HAN_CHARACTER_CLASS}]{{1,2}}"
        + _CHINESE_NAME_RIGHT_BOUNDARY,
        "person",
        priority=12,
        flags=0,
        base_score=0.75,
        context_words=[
            "患者",
            "病人",
            "病患",
            "姓名",
            "就诊者",
            "联系人",
            "家属",
        ],
        context_boost=0.2,
    ),
]

_ZH_ADDRESS_NAME = r"[\u3400-\u4dbf\u4e00-\u9fff]"
_ZH_ADDRESS_TAIL_CORE = (
    rf"{_ZH_ADDRESS_NAME}{{1,12}}?(?:区|县)\s*"
    rf"{_ZH_ADDRESS_NAME}{{1,12}}?(?:街道|镇|乡)\s*"
    rf"{_ZH_ADDRESS_NAME}{{1,12}}?(?:大道|路|街|巷|弄)\s*"
    r"\d{1,5}号(?:\d{1,4}(?:室|房)|[一二三四五六七八九十]{1,3}单元)?"
)
_ZH_ADDRESS_TAIL_PATTERN = rf"(?<=市){_ZH_ADDRESS_TAIL_CORE}"
_ZH_FULL_ADDRESS_PATTERN = (
    rf"(?:(?<=[：:，,。；;\s])|^)"
    rf"{_ZH_ADDRESS_NAME}{{1,12}}?省"
    rf"{_ZH_ADDRESS_NAME}{{1,12}}?市"
    rf"{_ZH_ADDRESS_TAIL_CORE}"
)

_CHINESE_ADDRESS_PII_PATTERNS: List[PIIPattern] = [
    # When the model misses the entire address, cover the full hierarchy so
    # province/city tokens cannot survive a tail-only replacement.
    PIIPattern(
        _ZH_FULL_ADDRESS_PATTERN,
        "street_address",
        priority=10,
        base_score=0.9,
        context_words=["地址", "住址", "现住址", "联系地址", "居住地"],
        context_boost=0.1,
    ),
    # Assist an under-segmented model span by adding only the strong
    # district/street/building tail. The leading boundary deliberately starts
    # after a city suffix so the safety sweep can add this span without
    # overlapping or moving an existing province/city span.
    PIIPattern(
        _ZH_ADDRESS_TAIL_PATTERN,
        "street_address",
        priority=9,
        base_score=0.9,
        context_words=["地址", "住址", "现住址", "联系地址", "居住地"],
        context_boost=0.1,
    ),
    PIIPattern(
        r"(?<!\d)\d{6}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["邮编", "邮政编码", "邮政", "地址", "住址"],
        context_boost=0.55,
        safety_sweep_requires_context=True,
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

# ---------------------------------------------------------------------------
# Finnish PII patterns
# ---------------------------------------------------------------------------

_FINNISH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "syntymäaika",
            "syntymaaika",
            "syntynyt",
            "päivämäärä",
            "paivamaara",
            "saapunut",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+358[\s.-]?\d{1,2}|0\d{1,2})[\s.-]?\d{3}[\s.-]?\d{2,4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["puhelin", "matkapuhelin", "gsm", "yhteystiedot"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{6}[-+YXWVUABCDEF]\d{3}[0-9ABCDEFHJKLMNPRSTUVWXY]\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "henkilötunnus",
            "henkilotunnus",
            "hetu",
        ],
        context_boost=0.4,
        validator=validate_finnish_hetu,
    ),
    PIIPattern(
        r"\b[A-ZÅÄÖ][a-zåäö.'-]*(?:katu|tie|kuja|polku|raitti)\s+\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["osoite", "asuu", "katu"],
        context_boost=0.25,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["postinumero", "osoite"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Bulgarian PII patterns (Cyrillic script)
# ---------------------------------------------------------------------------

_BULGARIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "дата",
            "раждане",
            "роден",
            "родена",
            "приет",
            "приета",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+359[\s.-]?\d{1,2}|0\d{1,2})[\s.-]?\d{3}[\s.-]?\d{3,4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["телефон", "тел", "мобилен", "gsm"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{2}[0-5]\d[0-3]\d{5}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "егн",
            "егн:",
            "egn",
            "единен граждански номер",
        ],
        context_boost=0.4,
        validator=validate_bulgarian_egn,
    ),
    PIIPattern(
        r"\b(?:[А-Я][а-яА-Я.'-]+\s+(?:улица|булевард|площад)\s+\d{1,5}[А-Яа-яA-Za-z]?|(?:улица|булевард|площад|ул\.|бул\.)\s+[А-Я][а-яА-Я .'-]{2,60}\s+\d{1,5}[А-Яа-яA-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["адрес", "живее", "улица"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{4}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["пощенски код", "адрес"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Croatian PII patterns
# ---------------------------------------------------------------------------

_CROATIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "datum",
            "rodjenja",
            "rođenja",
            "primljen",
            "otpusten",
            "otpušten",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+385[\s.-]?\d{1,2}|0\d{1,2})[\s.-]?\d{3}[\s.-]?\d{3,4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefon", "mobitel", "kontakt"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{11}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "oib",
            "oib:",
            "osobni identifikacijski broj",
        ],
        context_boost=0.4,
        validator=validate_croatian_oib,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"\b(?:[A-ZČĆŠŽĐ][A-Za-zčćšžđ.'-]+\s+(?:ulica|trg|avenija|cesta)\s+\d{1,5}[A-Za-z]?|(?:ulica|trg|avenija|cesta)\s+[A-ZČĆŠŽĐ][A-Za-zčćšžđ .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["adresa", "stanuje", "ulica"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["postanski broj", "poštanski broj", "adresa"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Serbian PII patterns (Latin and Cyrillic scripts)
# ---------------------------------------------------------------------------

_SERBIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "datum",
            "rodjenja",
            "rođenja",
            "датум",
            "рођења",
            "primljen",
            "примљен",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+381[\s.-]?\d{1,2}|0\d{1,2})[\s.-]?\d{3}[\s.-]?\d{3,4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefon", "mobilni", "kontakt", "телефон", "мобилни"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b[0-3]\d[01]\d{10}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "jmbg",
            "jmbg:",
            "јмбг",
            "maticni broj",
            "matični broj",
            "матични број",
        ],
        context_boost=0.4,
        validator=validate_jmbg,
    ),
    PIIPattern(
        r"\b(?:[A-ZČĆŠŽĐЀ-Я][A-Za-zčćšžđа-џ.'-]+\s+(?:ulica|bulevar|trg|улица|булевар|трг)\s+\d{1,5}[A-Za-z]?|(?:ulica|bulevar|trg|улица|булевар|трг)\s+[A-ZČĆŠŽĐЀ-Я][A-Za-zčćšžđа-џ .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["adresa", "stanuje", "ulica", "адреса", "улица"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "postanski broj",
            "poštanski broj",
            "поштански број",
            "adresa",
            "адреса",
        ],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Estonian PII patterns
# ---------------------------------------------------------------------------

_ESTONIAN_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "kuupaev",
            "kuupäev",
            "sunniaeg",
            "sünniaeg",
            "sundinud",
            "sündinud",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+372\s?)?[3-8]\d{2,3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefon", "mobiil", "kontakt"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b[1-6]\d{2}[01]\d[0-3]\d{5}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "isikukood",
            "isikukood:",
            "ik",
        ],
        context_boost=0.4,
        validator=validate_estonian_isikukood,
    ),
    PIIPattern(
        r"\b(?:[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f.'-]+\s+(?:tanav|tänav|maantee|puiestee|tee)\s+\d{1,5}[A-Za-z]?|(?:tanav|tänav|maantee|puiestee|tee)\s+[A-Z\u00c0-\u024f][A-Za-z\u00c0-\u024f .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["aadress", "elukoht", "tanav", "tänav"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{5}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["postiindeks", "indeks", "aadress"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Czech PII patterns
# ---------------------------------------------------------------------------

# Czech month names in both nominative and genitive (dates read
# "16. listopadu 1975"). Longer forms are listed first so the alternation
# prefers them.
_CZECH_MONTH_PATTERN = (
    r"ledna|leden|února|únor|března|březen|dubna|duben|"
    r"května|květen|července|červenec|června|červen|"
    r"srpna|srpen|září|října|říjen|listopadu|listopad|prosince|prosinec"
)

_CZECH_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "datum",
            "narozeni",
            "narození",
            "narozen",
            "narozena",
            "prijat",
            "přijat",
            "přijata",
            "propusten",
            "propuštěn",
            "propuštěna",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\.?\s+(?:{_CZECH_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "datum",
            "narození",
            "narozeni",
            "narozen",
            "narozena",
            "přijat",
            "prijat",
            "přijata",
            "propuštěn",
            "propusten",
            "propuštěna",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+420[\s.-]?)?[2-7]\d{2}(?:[\s.-]?\d{3}){2}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefon", "tel", "mobil", "číslo", "cislo", "kontakt"],
        context_boost=0.35,
    ),
    PIIPattern(
        r"(?<!\d)\d{2}(?:0[1-9]|1[0-2]|2[1-9]|3[0-2]|5[1-9]|6[0-2]|7[1-9]|8[0-2])(?:0[1-9]|[12]\d|3[01])[\s/-]?\d{3,4}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "rodne cislo",
            "rodné číslo",
            "rc",
            "rč",
            "identifikacni cislo",
            "identifikační číslo",
        ],
        context_boost=0.45,
        validator=validate_czech_rodne_cislo,
    ),
    PIIPattern(
        r"\b(?!(?:adresa|bydliště|bydliste|trvalé)\b)(?:[A-ZÀ-ɏ][A-Za-zÀ-ɏ.'-]+(?:\s+[A-ZÀ-ɏ][A-Za-zÀ-ɏ.'-]+)*?\s+(?:ulice|ul\.|trida|třída|namesti|náměstí|nábřeží)\s+\d{1,5}[A-Za-z]?|(?:ulice|ul\.|trida|třída|namesti|náměstí|nábřeží)\s+[A-ZÀ-ɏ][A-Za-zÀ-ɏ .'-]{2,60}\s+\d{1,5}[A-Za-z]?)\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "adresa",
            "bydliste",
            "bydliště",
            "trvalé bydliště",
            "ulice",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)\d{3}\s?\d{2}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=[
            "psc",
            "psč",
            "postovni smerovaci cislo",
            "poštovní směrovací číslo",
            "adresa",
        ],
        context_boost=0.5,
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


_SWAHILI_MONTH_PATTERN = (
    r"Januari|Februari|Machi|Aprili|Mei|Juni|Julai|Agosti|Septemba|"
    r"Oktoba|Novemba|Desemba"
)

_SWAHILI_DATE_CONTEXT = [
    "tarehe",
    "tarehe ya kuzaliwa",
    "alizaliwa",
    "kuzaliwa",
    "date",
    "date of birth",
    "dob",
    "born",
    "admitted",
    "discharged",
]

_SWAHILI_AGE_CONTEXT = [
    "umri",
    "umri wa miaka",
    "miaka",
    "age",
    "aged",
    "years old",
]

_SWAHILI_ID_CONTEXT = [
    "nambari ya kitambulisho",
    "namba ya kitambulisho",
    "kitambulisho",
    "nambari ya nida",
    "namba ya nida",
    "nida",
    "national id",
    "national identification number",
    "identity number",
    "id number",
    "id no",
]

_SWAHILI_NHIF_CONTEXT = [
    "nambari ya nhif",
    "namba ya nhif",
    "nhif nambari",
    "nhif namba",
    "nhif",
    "nhif number",
    "nhif member number",
    "member number",
    "health insurance number",
]

_SWAHILI_PHONE_CONTEXT = [
    "simu",
    "nambari ya simu",
    "namba ya simu",
    "piga simu",
    "wasiliana",
    "phone",
    "phone number",
    "mobile",
    "call",
    "contact",
]

_SWAHILI_NAME_CONTEXT = [
    "jina",
    "jina la mgonjwa",
    "mgonjwa",
    "name",
    "patient name",
    "patient",
]

_SWAHILI_PII_PATTERNS: List[PIIPattern] = [
    # Explicitly labelled names are safe to sweep without guessing which
    # capitalised words in a Latin-script, code-mixed note denote a person.
    PIIPattern(
        r"(?:(?<=Jina: )|(?<=Name: )|(?<=Jina la mgonjwa: )|"
        r"(?<=Patient name: ))[A-Z][A-Za-z'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-z'’-]{1,30}){1,3}\b",
        "name",
        priority=12,
        base_score=0.65,
        context_words=_SWAHILI_NAME_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=_SWAHILI_DATE_CONTEXT,
        context_boost=0.3,
    ),
    PIIPattern(
        rf"\b\d{{1,2}}\s+(?:{_SWAHILI_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=_SWAHILI_DATE_CONTEXT,
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Match only ages immediately attached to a bilingual age cue. A generic
    # one-to-three-digit pattern would over-redact labs elsewhere in the note.
    PIIPattern(
        r"(?:(?<=Umri: )|(?<=Age: )|(?<=Umri wa miaka )|(?<=Aged ))"
        r"(?:1[01]\d|[1-9]?\d)\b",
        "age",
        priority=11,
        base_score=0.65,
        context_words=_SWAHILI_AGE_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    # This offline pack has no checksum validation for NHIF member numbers.
    # Restrict recognition to explicitly labelled six-to-ten-digit values to
    # avoid treating clinical measurements as IDs.
    PIIPattern(
        r"(?<!\d)\d{6,10}(?!\d)",
        "national_id",
        priority=13,
        base_score=0.5,
        context_words=_SWAHILI_NHIF_CONTEXT,
        context_boost=0.45,
        safety_sweep_requires_context=True,
    ),
    # Tanzanian NIDA NIN: format-only recognition of the 20-digit value,
    # including its commonly grouped 8-5-5-2 rendering. No checksum is claimed.
    PIIPattern(
        r"(?<!\d)\d{8}(?P<nida_sep>[ -]?)\d{5}(?P=nida_sep)"
        r"\d{5}(?P=nida_sep)\d{2}(?!\d)",
        "national_id",
        priority=15,
        base_score=0.5,
        context_words=_SWAHILI_ID_CONTEXT,
        context_boost=0.45,
        safety_sweep_requires_context=True,
    ),
    PIIPattern(
        r"(?:(?<=Anwani: )|(?<=Address: ))[A-Z][A-Za-z'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-z'’-]{1,30}){0,4}\s+\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=8,
        base_score=0.65,
        context_words=["anwani", "address", "barabara", "mtaa"],
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)\d{5}(?!\d)",
        "postcode",
        priority=7,
        base_score=0.25,
        context_words=["msimbo wa posta", "postal code", "postcode", "posta"],
        context_boost=0.55,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    # East African mobile numbers in international form for Kenya, Tanzania,
    # and Uganda. Each has a nine-digit national significant number.
    PIIPattern(
        r"(?<!\d)\+(?:254[\s.-]?(?:1|7)\d{2}|"
        r"255[\s.-]?[67]\d{2}|256[\s.-]?7\d{2})"
        r"[\s.-]?\d{3}[\s.-]?\d{3}(?!\d)",
        "phone_number",
        priority=12,
        base_score=0.7,
        context_words=_SWAHILI_PHONE_CONTEXT,
        context_boost=0.25,
    ),
]

_SWAHILI_AND_KENYA_PII_PATTERNS = [
    *_SWAHILI_PII_PATTERNS,
    *_KENYA_ID_PII_PATTERNS,
]

_NGUNI_NAME_CONTEXT = [
    "igama",
    "igama lesiguli",
    "igama lesigulane",
    "name",
    "patient name",
]

_NGUNI_DATE_CONTEXT = [
    "usuku lokuzalwa",
    "umhla wokuzalwa",
    "wazalwa",
    "date of birth",
    "dob",
    "born",
]

_NGUNI_AGE_CONTEXT = ["iminyaka", "ubudala", "age", "aged", "years old"]

_NGUNI_ID_CONTEXT = [
    "identiteitsnommer",
    "inombolo kamazisi",
    "umazisi",
    "inombolo yesazisi",
    "isazisi",
    "south african id",
    "sa id",
    "identity number",
    "id number",
]

_NGUNI_MEDICAL_AID_CONTEXT = [
    "inombolo yosizo lwezempilo",
    "usizo lwezempilo",
    "inombolo yoncedo lwezonyango",
    "uncedo lwezonyango",
    "medical aid",
    "medical aid number",
    "medical aid member number",
    "membership number",
]

_NGUNI_PHONE_CONTEXT = [
    "selfoon",
    "ucingo",
    "umakhalekhukhwini",
    "ifowuni",
    "inombolo yefowuni",
    "phone",
    "mobile",
    "call",
    "contact",
]

_NGUNI_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?:(?<=Igama: )|(?<=Igama lesiguli: )|(?<=Igama lesigulane: )|"
        r"(?<=Name: )|(?<=Patient name: ))[A-Z][A-Za-z'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-z'’-]{1,30}){1,3}\b",
        "name",
        priority=12,
        base_score=0.65,
        context_words=_NGUNI_NAME_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b(?:0?[1-9]|[12][0-9]|3[01])[./-]"
        r"(?:0?[1-9]|1[0-2])[./-](?:19|20)[0-9]{2}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=_NGUNI_DATE_CONTEXT,
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?:(?<=Iminyaka )|(?<=Ubudala: )|(?<=Age: )|(?<=Aged ))"
        r"(?:1[01][0-9]|120|[1-9]?[0-9])\b",
        "age",
        priority=11,
        base_score=0.65,
        context_words=_NGUNI_AGE_CONTEXT,
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<![0-9])[0-9]{13}(?![0-9])",
        "national_id",
        priority=15,
        base_score=0.75,
        context_words=_NGUNI_ID_CONTEXT,
        context_boost=0.2,
        validator=validate_za_id_number,
    ),
    PIIPattern(
        r"(?<![A-Z0-9])(?:[A-Z]{2,5}[- ]?)?[0-9]{6,12}(?![A-Z0-9])",
        "national_id",
        priority=13,
        base_score=0.5,
        context_words=_NGUNI_MEDICAL_AID_CONTEXT,
        context_boost=0.45,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?:(?<=Ikheli: )|(?<=Idilesi: )|(?<=Address: ))"
        r"[0-9]{1,5}\s+[A-Z][A-Za-z'’-]{1,30}"
        r"(?:\s+[A-Z][A-Za-z'’-]{1,30}){0,4}\b",
        "street_address",
        priority=8,
        base_score=0.65,
        context_words=["ikheli", "idilesi", "address"],
        context_boost=0.3,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<![0-9])[0-9]{4}(?![0-9])",
        "postcode",
        priority=7,
        base_score=0.25,
        context_words=[
            "ikhodi yeposi",
            "ikhowudi yeposi",
            "postal code",
            "postcode",
        ],
        context_boost=0.55,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<![0-9])(?:\+?27[\s.-]?[678][0-9]|0[678][0-9])"
        r"[\s.-]?[0-9]{3}[\s.-]?[0-9]{4}(?![0-9])",
        "phone_number",
        priority=12,
        base_score=0.7,
        context_words=_NGUNI_PHONE_CONTEXT,
        context_boost=0.25,
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


_ROMANIAN_MONTH_PATTERN = (
    r"ianuarie|februarie|martie|aprilie|mai|iunie|iulie|august|"
    r"septembrie|octombrie|noiembrie|decembrie"
)

_ROMANIAN_PII_PATTERNS: List[PIIPattern] = [
    # Romanian dates DD.MM.YYYY (also tolerate / or - separators).
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "data",
            "nascut",
            "născut",
            "nascuta",
            "născută",
            "data nasterii",
            "data nașterii",
            "internat",
            "externat",
            "decedat",
        ],
        context_boost=0.3,
    ),
    # Romanian dates with month names ("12 martie 1985").
    PIIPattern(
        rf"\b\d{{1,2}}\s+(?:{_ROMANIAN_MONTH_PATTERN})\s+\d{{4}}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "data",
            "nascut",
            "născut",
            "nascuta",
            "născută",
            "data nasterii",
            "data nașterii",
            "internat",
            "externat",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Romanian mobile / landline numbers: +40 or national 0 prefix.
    PIIPattern(
        r"(?<!\w)(?:\+40[\s.-]?|0)7\d{2}[\s.-]?\d{3}[\s.-]?\d{3}\b"
        r"|(?<!\w)(?:\+40[\s.-]?|0)\d{2}[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "telefon",
            "tel",
            "mobil",
            "contact",
            "gsm",
        ],
        context_boost=0.35,
    ),
    # CNP (13-digit Cod Numeric Personal), guarded by the checksum validator.
    PIIPattern(
        r"\b\d{13}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "cnp",
            "cod numeric personal",
            "cnp:",
            "act de identitate",
            "buletin",
            "carte de identitate",
        ],
        context_boost=0.4,
        validator=validate_romanian_cnp,
    ),
    # Romanian street addresses ("Str. Mihai Eminescu 12"). Accept modern
    # comma-below, legacy cedilla, and decomposed combining-mark spellings.
    PIIPattern(
        r"\b(?:Str\.?|Strada|Bd\.?|Bdul|Bulevardul|Calea|Aleea|"
        r"(?:[SȘŞ]|S[\u0326\u0327])oseaua|Splaiul|"
        r"Pia(?:[TȚŢ]|T[\u0326\u0327])a|Intrarea)\s+"
        r"[^\W\d_](?:[\u0300-\u036f])?"
        r"(?:[^\W_]|[\u0300-\u036f .'-]){1,60}\s+"
        r"(?:nr\.?\s*)?\d{1,5}[A-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "adresa",
            "adresă",
            "domiciliu",
            "strada",
            "resedinta",
            "reședința",
            "reşedinţa",
            "res\u0326edint\u0326a",
            "res\u0327edint\u0327a",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Romanian postal codes: six contiguous digits.
    PIIPattern(
        r"(?<!\d)\d{6}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "cod postal",
            "cod poștal",
            "cod poştal",
            "cod pos\u0326tal",
            "cod pos\u0327tal",
            "cp",
            "adresa",
            "adresă",
            "adresa\u0306",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
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
        r"(?<!\d)\d{2}(?:0[1-9]|1[0-2]|2[1-9]|3[0-2]|5[1-9]|6[0-2]|7[1-9]|8[0-2])(?:0[1-9]|[12]\d|3[01])[\s/-]?\d{3,4}(?!\d)",
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


_HUNGARIAN_MONTH_PATTERN = (
    r"január|február|március|április|május|június|július|augusztus|"
    r"szeptember|október|november|december"
)

_HUNGARIAN_PII_PATTERNS: List[PIIPattern] = [
    # Hungarian civil dates conventionally use year-month-day order.
    PIIPattern(
        r"(?<!\d)\d{4}[./-]\s?\d{1,2}[./-]\s?\d{1,2}\.?(?!\d)",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "dátum",
            "született",
            "születési dátum",
            "felvétel",
            "elbocsátás",
            "kontroll",
        ],
        context_boost=0.3,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        rf"(?<!\d)\d{{4}}\.\s?(?:{_HUNGARIAN_MONTH_PATTERN})\s+\d{{1,2}}\.?(?!\d)",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "dátum",
            "született",
            "születési dátum",
            "felvétel",
            "elbocsátás",
            "kontroll",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    # Hungarian E.164 (+36) and domestic (06) geographic/mobile forms.
    PIIPattern(
        r"(?<!\w)(?:\+36|06)[\s.-]?(?:"
        r"1[\s.-]?\d{3}[\s.-]?\d{4}|"
        r"(?:20|30|31|50|70)[\s.-]?\d{3}[\s.-]?\d{4}|"
        r"[2-9]\d[\s.-]?\d{3}[\s.-]?\d{3})(?!\d)",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["telefon", "telefonszám", "mobil", "elérhetőség", "tel"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)(?:[0-9]{9}|[0-9]{3}(?:[ -][0-9]{3}){2})(?!\d)",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "taj",
            "taj-szám",
            "taj szám",
            "társadalombiztosítási azonosító jel",
            "biztosítási azonosító",
        ],
        context_boost=0.4,
        safety_sweep_requires_context=True,
        validator=validate_hungarian_taj,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b[^\W\d_](?:[^\W_]|[ .'-]){1,60}\s+"
        r"(?:utca|út|tér|körút|köz|sétány|fasor|park|sor)\s+"
        r"\d{1,4}[A-Za-z]?(?:/\d+)?\.?(?!\w)",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "cím",
            "lakcím",
            "lakóhely",
            "tartózkodási hely",
            "utca",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)[1-9]\d{3}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=["irányítószám", "postai irányítószám", "cím", "lakcím"],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Greek PII patterns (Greek script)
# ---------------------------------------------------------------------------

_GREEK_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "ημερομηνία",
            "γέννησης",
            "γεννήθηκε",
            "εισαγωγή",
            "εξιτήριο",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+30[\s.-]?)?(?:69\d|2\d{2})[\s.-]?\d{3}[\s.-]?\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=["τηλέφωνο", "τηλ", "κινητό", "επικοινωνία"],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{11}\b",
        "national_id",
        priority=10,
        base_score=0.5,
        context_words=[
            "αμκα",
            "α.μ.κ.α",
            "αριθμός μητρώου κοινωνικής ασφάλισης",
        ],
        context_boost=0.4,
        validator=validate_greek_amka,
    ),
    PIIPattern(
        r"\b(?:οδός|λεωφόρος|πλατεία)\s+[Α-ΩΆΈΉΊΌΎΏ][Α-Ωα-ωάέήίόύώϊϋΐΰ.'-]{2,40}\s+\d{1,4}[Α-Ωα-ωA-Za-z]?\b",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=["διεύθυνση", "κατοικία", "οδός"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"\b\d{3}\s?\d{2}\b",
        "postcode",
        priority=6,
        base_score=0.3,
        context_words=["ταχυδρομικός κώδικας", "τ.κ", "διεύθυνση"],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# Vietnamese PII patterns
# ---------------------------------------------------------------------------

_VIETNAMESE_PII_PATTERNS: List[PIIPattern] = [
    PIIPattern(
        r"(?<!\d)(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}(?!\d)",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "ngày sinh",
            "ngay sinh",
            "sinh ngày",
            "sinh ngay",
            "ngày khám",
            "ngay kham",
            "ngày nhập viện",
            "ngay nhap vien",
            "ngày xuất viện",
            "ngay xuat vien",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"(?<!\w)ngày\s+(?:0?[1-9]|[12]\d|3[01])\s+tháng\s+"
        r"(?:0?[1-9]|1[0-2])\s+năm\s+(?:19|20)\d{2}(?!\d)",
        "date",
        priority=8,
        base_score=0.7,
        context_words=["ngày sinh", "sinh ngày", "ngày khám", "ngày nhập viện"],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+84[\s.-]?|0)[35789]\d(?:[\s.-]?\d){7}(?!\d)",
        "phone_number",
        priority=8,
        base_score=0.55,
        context_words=[
            "điện thoại",
            "dien thoai",
            "số điện thoại",
            "so dien thoai",
            "di động",
            "di dong",
            "liên hệ",
            "lien he",
            "đt",
            "sđt",
        ],
        context_boost=0.35,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\w)(?:\+84[\s.-]?|0)2\d{1,2}(?:[\s.-]?\d){7,8}(?!\d)",
        "phone_number",
        priority=8,
        base_score=0.45,
        context_words=[
            "điện thoại",
            "dien thoai",
            "số điện thoại",
            "so dien thoai",
            "liên hệ",
            "lien he",
            "đt",
        ],
        context_boost=0.4,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)\d{3}(?:[\s-]?\d{3}){3}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.45,
        context_words=[
            "cccd",
            "căn cước công dân",
            "can cuoc cong dan",
            "căn cước",
            "can cuoc",
            "số định danh cá nhân",
            "so dinh danh ca nhan",
        ],
        context_boost=0.5,
        safety_sweep_requires_context=True,
        validator=validate_vietnamese_cccd,
    ),
    PIIPattern(
        r"(?<!\d)\d{3}(?:[\s-]?\d{3}){2}(?!\d)",
        "national_id",
        priority=10,
        base_score=0.35,
        context_words=[
            "cmnd",
            "chứng minh nhân dân",
            "chung minh nhan dan",
            "số chứng minh",
            "so chung minh",
        ],
        # Keep a contextual CMND below a fully contextual Vietnamese phone so
        # a spaced nine-digit subscriber number wins overlap resolution.
        context_boost=0.5,
        safety_sweep_requires_context=True,
        validator=validate_vietnamese_cmnd,
    ),
    PIIPattern(
        r"(?<!\w)(?:số\s+)?\d{1,5}[A-Za-z]?(?:[/.-]\d{1,5}[A-Za-z]?)?\s+"
        r"(?:đường|duong|phố|pho|ngõ|ngo|hẻm|hem)\s+[^\n,;]{2,80}",
        "street_address",
        priority=7,
        base_score=0.65,
        context_words=[
            "địa chỉ",
            "dia chi",
            "thường trú",
            "thuong tru",
            "tạm trú",
            "tam tru",
            "đường",
            "duong",
            "phường",
            "phuong",
        ],
        context_boost=0.25,
        flags=re.IGNORECASE,
    ),
    PIIPattern(
        r"(?<!\d)\d{5}(?!\d)",
        "postcode",
        priority=6,
        base_score=0.25,
        context_words=[
            "mã bưu chính",
            "ma buu chinh",
            "mã bưu điện",
            "ma buu dien",
            "bưu chính",
            "buu chinh",
            "địa chỉ",
            "dia chi",
        ],
        context_boost=0.55,
        safety_sweep_requires_context=True,
        flags=re.IGNORECASE,
    ),
]


LANGUAGE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "af": _NGUNI_PII_PATTERNS,
    "ha": _NIGERIAN_PII_PATTERNS,
    "ig": _NIGERIAN_PII_PATTERNS,
    "yo": _NIGERIAN_PII_PATTERNS,
    "fr": _FRENCH_PII_PATTERNS,
    "de": _GERMAN_PII_PATTERNS,
    "it": _ITALIAN_PII_PATTERNS,
    "es": _SPANISH_PII_PATTERNS,
    "pt": _PORTUGUESE_PII_PATTERNS,
    "nl": _DUTCH_PII_PATTERNS,
    # Keep Aadhaar discoverable through the historical per-language mapping
    # while get_patterns_for_language deduplicates the universal rule.
    "hi": [*_HINDI_PII_PATTERNS, *AADHAAR_PII_PATTERNS],
    "te": [*_TELUGU_PII_PATTERNS, *AADHAAR_PII_PATTERNS],
    "am": _AMHARIC_PII_PATTERNS,
    "ar": _ARABIC_PII_PATTERNS,
    "he": _HEBREW_PII_PATTERNS,
    "ja": _JAPANESE_PII_PATTERNS,
    "zh": [
        *_CHINESE_PII_PATTERNS,
        *_CHINESE_ADDRESS_PII_PATTERNS,
        *_CHINESE_IDENTIFIER_PII_PATTERNS,
    ],
    "tr": _TURKISH_PII_PATTERNS,
    "id": _INDONESIAN_PII_PATTERNS,
    "th": _THAI_PII_PATTERNS,
    "lv": _LATVIAN_PII_PATTERNS,
    "pl": _POLISH_PII_PATTERNS,
    "ko": _KOREAN_PII_PATTERNS,
    "sk": _SLOVAK_PII_PATTERNS,
    "ms": _MALAY_PII_PATTERNS,
    "tl": _TAGALOG_PII_PATTERNS,
    "sw": _SWAHILI_AND_KENYA_PII_PATTERNS,
    "zu": _NGUNI_PII_PATTERNS,
    "xh": _NGUNI_PII_PATTERNS,
    "da": _DANISH_PII_PATTERNS,
    "ro": _ROMANIAN_PII_PATTERNS,
    "fi": _FINNISH_PII_PATTERNS,
    "bg": _BULGARIAN_PII_PATTERNS,
    "hr": _CROATIAN_PII_PATTERNS,
    "sr": _SERBIAN_PII_PATTERNS,
    "hu": _HUNGARIAN_PII_PATTERNS,
    "et": _ESTONIAN_PII_PATTERNS,
    "el": _GREEK_PII_PATTERNS,
    "cs": _CZECH_PII_PATTERNS,
    "vi": _VIETNAMESE_PII_PATTERNS,
    "ur": _URDU_PII_PATTERNS,
}

LOCALE_PII_PATTERNS: Dict[str, List[PIIPattern]] = {
    "zh_cn": _CHINESE_IDENTIFIER_PII_PATTERNS,
    "ar": _EGYPT_NATIONAL_ID_PII_PATTERNS + _MOROCCO_CIN_PII_PATTERNS,
    "ar_eg": _EGYPT_NATIONAL_ID_PII_PATTERNS,
    "ar_ma": _MOROCCO_CIN_PII_PATTERNS,
    "en_za": _NGUNI_PII_PATTERNS,
    "af": _NGUNI_PII_PATTERNS,
    "en_ng": _NIGERIAN_PII_PATTERNS,
    "ha": _NIGERIAN_PII_PATTERNS,
    "ig": _NIGERIAN_PII_PATTERNS,
    "yo": _NIGERIAN_PII_PATTERNS,
    "en_gh": _GHANA_CARD_PII_PATTERNS,
    "en_ke": _KENYA_ID_PII_PATTERNS,
    "sw": _SWAHILI_AND_KENYA_PII_PATTERNS,
    "zu": _NGUNI_PII_PATTERNS,
    "xh": _NGUNI_PII_PATTERNS,
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
    "am": {
        "NAME": [
            "ሰላም ተስፋዬ",
            "ዳዊት ከበደ",
            "ሜሮን አለሙ",
            "ሚካኤል ገብረ",
        ],
        "FIRST_NAME": ["ሰላም", "ዳዊት", "ሜሮን", "ሚካኤል"],
        "LAST_NAME": ["ተስፋዬ", "ከበደ", "አለሙ", "ገብረ"],
        "EMAIL": ["takami@example.et", "contact@example.org"],
        "PHONE": ["+251 911 234 567", "0911 765 432"],
        "ID_NUM": ["123456789012", "987654321098"],
        "STREET_ADDRESS": ["አዲስ አበባ ቦሌ 12", "ባሕር ዳር ቀበሌ 5"],
        "URL_PERSONAL": ["https://example.et"],
        "USERNAME": ["takami123", "fayda456"],
        "DATE": ["፲፬/፭/፲፱፻፹፰", "01/01/2000"],
        "AGE": ["፳፱", "፴፭", "፵፯"],
        "LOCATION": ["አዲስ አበባ", "ባሕር ዳር", "ሀዋሳ"],
        "ZIPCODE": ["1000", "3000", "6000"],
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
    "ur": {
        "NAME": [
            "\u0627\u062d\u0645\u062f \u0639\u0644\u06cc",
            "\u0641\u0627\u0637\u0645\u06c1 \u062e\u0627\u0646",
            "\u0626\u0644\u0627\u0644 \u062d\u0633\u06cc\u0646",
            "\u0633\u0627\u0631\u0627 \u0627\u062d\u0645\u062f",
        ],
        "FIRST_NAME": [
            "\u0627\u062d\u0645\u062f",
            "\u0641\u0627\u0637\u0645\u06c1",
            "\u0626\u0644\u0627\u0644",
            "\u0633\u0627\u0631\u0627",
        ],
        "LAST_NAME": [
            "\u0639\u0644\u06cc",
            "\u062e\u0627\u0646",
            "\u062d\u0633\u06cc\u0646",
            "\u0627\u062d\u0645\u062f",
        ],
        "EMAIL": ["patient@example.pk", "contact@example.org"],
        "PHONE": ["+92 300 1234567", "021 34567890"],
        "ID_NUM": ["12345-6789012-3", "42101-1234567-9"],
        "STREET_ADDRESS": [
            "\u06af\u0644\u06cc \u0646\u0645\u0628\u0631 5 \u0645\u062d\u0644\u06c1 \u0627\u0633\u0644\u0627\u0645 \u0622\u0626\u0627\u062f 12"
        ],
        "URL_PERSONAL": ["https://example.pk"],
        "USERNAME": ["patient123", "user456"],
        "DATE": [
            "\u06f1\u06f6.\u06f1\u06f1.\u06f1\u06f9\u06f7\u06f5",
            "16.11.1975",
        ],
        "AGE": ["\u06f4\u06f5", "62", "38"],
        "LOCATION": [
            "\u06a9\u0631\u0627\u0686\u06cc",
            "\u0644\u0627\u06c1\u0648\u0631",
            "\u0627\u0633\u0644\u0627\u0645 \u0622\u0626\u0627\u062f",
        ],
        "ZIPCODE": ["74200", "54000", "44000"],
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
    "sw": {
        "NAME": [
            "Amina Hassan",
            "Daniel Otieno",
            "Wanjiku Njeri",
            "Baraka Mushi",
        ],
        "FIRST_NAME": ["Amina", "Daniel", "Wanjiku", "Baraka"],
        "LAST_NAME": ["Hassan", "Otieno", "Njeri", "Mushi"],
        "EMAIL": ["mgonjwa@example.ke", "mawasiliano@example.tz"],
        "PHONE": [
            "+254 712 345 678",
            "+255 754 321 098",
            "+256 772 456 789",
        ],
        "ID_NUM": [
            "12345678",
            "987654321",
            "19791103-12345-67890-12",
        ],
        "STREET_ADDRESS": ["Kenyatta Avenue 12", "Barabara ya Nyerere 45"],
        "URL_PERSONAL": ["https://example.ke"],
        "USERNAME": ["mgonjwa123", "mtumiaji456"],
        "DATE": ["14/05/1988", "3 Novemba 1979"],
        "AGE": ["29", "38", "47"],
        "LOCATION": ["Nairobi", "Dar es Salaam", "Kampala", "Mombasa"],
        "ZIPCODE": ["00100", "11101", "10101"],
    },
    "zu": {
        "NAME": [
            "Nomcebo Dlamini",
            "Xolani Khumalo",
            "Qhawe Ndlovu",
            "Cebisile Zulu",
        ],
        "FIRST_NAME": ["Nomcebo", "Xolani", "Qhawe", "Cebisile"],
        "LAST_NAME": ["Dlamini", "Khumalo", "Ndlovu", "Zulu"],
        "EMAIL": ["isiguli@example.co.za", "xhumana@example.org"],
        "PHONE": ["+27 82 123 4567", "071 987 6543"],
        "ID_NUM": ["8001015009087", "9003030123082"],
        "STREET_ADDRESS": ["12 Umgeni Road", "45 Vilakazi Street"],
        "URL_PERSONAL": ["https://example.co.za"],
        "USERNAME": ["isiguli123", "nomcebo88"],
        "DATE": ["14/05/1988", "03/11/1979"],
        "AGE": ["38", "47", "29"],
        "LOCATION": ["Durban", "Umlazi", "East London", "Gqeberha"],
        "ZIPCODE": ["4001", "4066", "5201", "6001"],
    },
    "xh": {
        "NAME": [
            "Xolani Qwabe",
            "Qhawe Mbeki",
            "Nomcebo Gcaleka",
            "Zukiswa Nqatha",
        ],
        "FIRST_NAME": ["Xolani", "Qhawe", "Nomcebo", "Zukiswa"],
        "LAST_NAME": ["Qwabe", "Mbeki", "Gcaleka", "Nqatha"],
        "EMAIL": ["isigulane@example.co.za", "qhagamshelana@example.org"],
        "PHONE": ["+27 71 234 5678", "083 765 4321"],
        "ID_NUM": ["7903116001080", "0102034000186"],
        "STREET_ADDRESS": ["18 Oxford Street", "27 Govan Mbeki Avenue"],
        "URL_PERSONAL": ["https://example.co.za"],
        "USERNAME": ["isigulane123", "qhawe79"],
        "DATE": ["03/11/1979", "21/06/1991"],
        "AGE": ["47", "35", "29"],
        "LOCATION": ["East London", "Gqeberha", "Durban", "Umlazi"],
        "ZIPCODE": ["5201", "6001", "4001", "4066"],
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
    "vi": {
        "NAME": [
            "Nguyễn Minh Anh",
            "Trần Hoàng Nam",
            "Lê Thu Hà",
            "Phạm Quang Huy",
        ],
        "FIRST_NAME": ["Minh Anh", "Hoàng Nam", "Thu Hà", "Quang Huy"],
        "LAST_NAME": ["Nguyễn", "Trần", "Lê", "Phạm"],
        "EMAIL": ["benhnhan@example.vn", "lienhe@example.org"],
        "PHONE": ["+84 912 345 678", "0912-345-678", "028 3822 1234"],
        "ID_NUM": ["001203123456", "024 203 654 321", "123456789"],
        "STREET_ADDRESS": [
            "12 đường Nguyễn Trãi",
            "45 phố Trần Hưng Đạo",
        ],
        "URL_PERSONAL": ["https://example.vn"],
        "USERNAME": ["benhnhan123", "nguoidung456"],
        "DATE": ["17/08/1985", "ngày 1 tháng 1 năm 2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Hà Nội", "Thành phố Hồ Chí Minh", "Đà Nẵng"],
        "ZIPCODE": ["10000", "70000", "50000"],
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
    "ro": {
        "NAME": [
            "Ana Popescu",
            "Ion Ionescu",
            "Maria Dumitru",
            "Andrei Popa",
        ],
        "FIRST_NAME": ["Ana", "Ion", "Maria", "Andrei"],
        "LAST_NAME": ["Popescu", "Ionescu", "Dumitru", "Popa"],
        "EMAIL": ["pacient@exemplu.ro", "contact@exemplu.org"],
        "PHONE": ["+40 721 234 567", "0721 234 567", "+40 21 123 4567"],
        "ID_NUM": ["1800101400181", "2990312072589"],
        "STREET_ADDRESS": ["Str. Mihai Eminescu 12", "Bd. Unirii 45"],
        "URL_PERSONAL": ["https://exemplu.ro"],
        "USERNAME": ["pacient123", "utilizator456"],
        "DATE": ["01.01.2000", "12 martie 1985"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Bucuresti", "Cluj-Napoca", "Timisoara"],
        "ZIPCODE": ["010011", "400001", "300001"],
    },
    "fi": {
        "NAME": ["Matti Virtanen", "Liisa Korhonen", "Juha Nieminen", "Anna Laine"],
        "FIRST_NAME": ["Matti", "Liisa", "Juha", "Anna"],
        "LAST_NAME": ["Virtanen", "Korhonen", "Nieminen", "Laine"],
        "EMAIL": ["potilas@example.fi", "yhteys@example.org"],
        "PHONE": ["+358 40 123 4567", "09 1234 567"],
        "ID_NUM": ["210582-0179", "030904A501C"],
        "STREET_ADDRESS": ["Mannerheimintie 12", "Aleksanterinkatu 5"],
        "URL_PERSONAL": ["https://example.fi"],
        "USERNAME": ["potilas123", "kayttaja456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Helsinki", "Tampere", "Turku"],
        "ZIPCODE": ["00100", "33100", "20100"],
    },
    "bg": {
        "NAME": ["Иван Петров", "Мария Иванова", "Георги Димитров", "Елена Тодорова"],
        "FIRST_NAME": ["Иван", "Мария", "Георги", "Елена"],
        "LAST_NAME": ["Петров", "Иванова", "Димитров", "Тодорова"],
        "EMAIL": ["patsient@example.bg", "kontakt@example.org"],
        "PHONE": ["+359 88 123 4567", "02 987 6543"],
        "ID_NUM": ["8205210172", "0449035017"],
        "STREET_ADDRESS": ["улица Раковски 35", "булевард Витоша 18"],
        "URL_PERSONAL": ["https://example.bg"],
        "USERNAME": ["patsient123", "potrebitel456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["София", "Пловдив", "Варна"],
        "ZIPCODE": ["1000", "4000", "9000"],
    },
    "hr": {
        "NAME": ["Ivan Horvat", "Ana Kovacevic", "Marko Babic", "Petra Novak"],
        "FIRST_NAME": ["Ivan", "Ana", "Marko", "Petra"],
        "LAST_NAME": ["Horvat", "Kovacevic", "Babic", "Novak"],
        "EMAIL": ["pacijent@example.hr", "kontakt@example.org"],
        "PHONE": ["+385 91 234 5678", "01 234 5678"],
        "ID_NUM": ["12345678903", "55512345672"],
        "STREET_ADDRESS": ["Savska ulica 12", "Vukovarska ulica 45"],
        "URL_PERSONAL": ["https://example.hr"],
        "USERNAME": ["pacijent123", "korisnik456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Zagreb", "Split", "Rijeka"],
        "ZIPCODE": ["10000", "21000", "51000"],
    },
    "sr": {
        "NAME": ["Marko Petrovic", "Jelena Jovanovic", "Nikola Nikolic", "Ana Ilic"],
        "FIRST_NAME": ["Marko", "Jelena", "Nikola", "Ana"],
        "LAST_NAME": ["Petrovic", "Jovanovic", "Nikolic", "Ilic"],
        "EMAIL": ["pacijent@example.rs", "kontakt@example.org"],
        "PHONE": ["+381 64 123 4567", "011 234 5678"],
        "ID_NUM": ["2105982710174", "0309004715013"],
        "STREET_ADDRESS": ["Bulevar Oslobodjenja 45", "Ulica Kneza Milosa 12"],
        "URL_PERSONAL": ["https://example.rs"],
        "USERNAME": ["pacijent123", "korisnik456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Beograd", "Novi Sad", "Nis"],
        "ZIPCODE": ["11000", "21000", "18000"],
    },
    "hu": {
        "NAME": ["Kovács Anna", "Nagy Péter", "Tóth Júlia", "Szabó Gábor"],
        "FIRST_NAME": ["Anna", "Péter", "Júlia", "Gábor"],
        "LAST_NAME": ["Kovács", "Nagy", "Tóth", "Szabó"],
        "EMAIL": ["beteg@example.hu", "kapcsolat@example.org"],
        "PHONE": ["+36 30 123 4567", "06 1 234 5678"],
        "ID_NUM": ["123456788", "123 456 788"],
        "STREET_ADDRESS": ["Kossuth Lajos utca 12", "Andrássy út 45"],
        "URL_PERSONAL": ["https://example.hu"],
        "USERNAME": ["beteg123", "felhasznalo456"],
        "DATE": ["1985. május 5.", "2000.01.01."],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Budapest", "Szeged", "Debrecen"],
        "ZIPCODE": ["1051", "6720", "4024"],
    },
    "et": {
        "NAME": ["Mari Tamm", "Jaan Kask", "Anu Saar", "Peeter Kivi"],
        "FIRST_NAME": ["Mari", "Jaan", "Anu", "Peeter"],
        "LAST_NAME": ["Tamm", "Kask", "Saar", "Kivi"],
        "EMAIL": ["patsient@example.ee", "kontakt@example.org"],
        "PHONE": ["+372 5123 4567", "644 1234"],
        "ID_NUM": ["38205210123", "60409032208"],
        "STREET_ADDRESS": ["Pikk tanav 12", "Narva maantee 45"],
        "URL_PERSONAL": ["https://example.ee"],
        "USERNAME": ["patsient123", "kasutaja456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Tallinn", "Tartu", "Parnu"],
        "ZIPCODE": ["10115", "50090", "80010"],
    },
    "cs": {
        "NAME": ["Jana Nováková", "Petr Novák", "Marie Svobodová", "Tomáš Král"],
        "FIRST_NAME": ["Jana", "Petr", "Marie", "Tomáš"],
        "LAST_NAME": ["Nováková", "Novák", "Svobodová", "Král"],
        "EMAIL": ["pacient@example.cz", "kontakt@example.org"],
        "PHONE": ["+420 601 234 567", "702 345 678"],
        "ID_NUM": ["820521/0002", "485305/123"],
        "STREET_ADDRESS": ["Hlavní ulice 12", "Náměstí Míru 5"],
        "URL_PERSONAL": ["https://example.cz"],
        "USERNAME": ["pacient123", "uzivatel456"],
        "DATE": ["16.11.1975", "16. listopadu 1975"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Praha", "Brno", "Ostrava"],
        "ZIPCODE": ["110 00", "602 00", "702 00"],
    },
    "el": {
        "NAME": [
            "Γιώργος Παπαδόπουλος",
            "Μαρία Νικολάου",
            "Δημήτρης Ιωάννου",
            "Ελένη Γεωργίου",
        ],
        "FIRST_NAME": ["Γιώργος", "Μαρία", "Δημήτρης", "Ελένη"],
        "LAST_NAME": ["Παπαδόπουλος", "Νικολάου", "Ιωάννου", "Γεωργίου"],
        "EMAIL": ["asthenis@example.gr", "epikoinonia@example.org"],
        "PHONE": ["+30 691 234 5678", "210 123 4567"],
        "ID_NUM": ["21058200177", "03090450119"],
        "STREET_ADDRESS": ["οδός Ερμού 15", "λεωφόρος Κηφισίας 42"],
        "URL_PERSONAL": ["https://example.gr"],
        "USERNAME": ["asthenis123", "christis456"],
        "DATE": ["16.11.1975", "01.01.2000"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["Αθήνα", "Θεσσαλονίκη", "Πάτρα"],
        "ZIPCODE": ["104 31", "546 21", "262 21"],
    },
    "zh": {
        "NAME": ["王芳", "李雷", "张伟", "刘洋"],
        "FIRST_NAME": ["芳", "雷", "伟", "洋"],
        "LAST_NAME": ["王", "李", "张", "刘"],
        "EMAIL": ["patient@example.cn", "contact@example.org"],
        "PHONE": ["13800138000", "13900139000"],
        "ID_NUM": ["CN123456", "MRN-987654"],
        "STREET_ADDRESS": ["北京市朝阳区健康路12号", "上海市和平路45号"],
        "URL_PERSONAL": ["https://example.cn"],
        "USERNAME": ["patient123", "user456"],
        "DATE": ["2000年1月1日", "1985年3月15日"],
        "AGE": ["45", "62", "38"],
        "LOCATION": ["北京", "上海", "广州"],
        "ZIPCODE": ["100000", "200000", "510000"],
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
    elif "_" in lang or "-" in lang:
        keys.append(_normalize_pattern_locale(lang))
    else:
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

    # English patterns serve as universal base. MRZ, USCC, and Aadhaar patterns
    # are language-agnostic, so they join the universal base for every route.
    base = (
        list(PII_PATTERNS) + MRZ_PII_PATTERNS + USCC_PII_PATTERNS + AADHAAR_PII_PATTERNS
    )

    combined = base
    if base_lang != "en":
        language_patterns = LANGUAGE_PII_PATTERNS.get(base_lang, [])
        combined = combined + [
            pattern
            for pattern in language_patterns
            if not any(pattern is existing for existing in combined)
        ]

    for locale_key in _locale_pattern_keys(lang, locale):
        combined = combined + [
            pattern
            for pattern in LOCALE_PII_PATTERNS.get(locale_key, [])
            if not any(pattern is existing for existing in combined)
        ]

    return combined
