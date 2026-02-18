"""PII extraction and de-identification for HIPAA compliance.

This module provides production-ready tools for detecting and redacting Personally
Identifiable Information (PII) from clinical notes, enabling HIPAA-compliant
processing of medical records.

Key Features:
    - Token-level PII detection for 18+ entity types
    - Multiple de-identification strategies (mask, remove, replace, hash, shift_dates)
    - HIPAA Safe Harbor method support
    - Reversible de-identification with secure mapping
    - Integration with OpenMed's existing NER infrastructure

Example:
    >>> from openmed import extract_pii, deidentify
    >>>
    >>> # Extract PII entities
    >>> result = extract_pii("Dr. Smith called John Doe at 555-1234")
    >>> for entity in result.entities:
    ...     print(f"{entity.label}: {entity.text}")
    NAME: Dr. Smith
    NAME: John Doe
    PHONE: 555-1234

    >>> # De-identify with masking
    >>> deid = deidentify(
    ...     "Patient John Doe (DOB: 01/15/1970) at 555-123-4567",
    ...     method="mask",
    ...     keep_year=True
    ... )
    >>> print(deid.deidentified_text)
    Patient [NAME] (DOB: [DATE]/1970) at [PHONE]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Literal
from datetime import datetime, timedelta
import hashlib
import random
import re
import unicodedata

from .config import OpenMedConfig
from ..processing.outputs import EntityPrediction

# Type alias for de-identification methods
DeidentificationMethod = Literal["mask", "remove", "replace", "hash", "shift_dates"]


@dataclass
class PIIEntity(EntityPrediction):
    """Extended Entity with PII-specific metadata.

    Attributes:
        text: The entity text span
        label: PII category (NAME, EMAIL, PHONE, etc.)
        start: Character start position
        end: Character end position
        confidence: Model confidence score (0-1)
        entity_type: PII category (same as label)
        redacted_text: Replacement text after de-identification
        original_text: Original text before redaction
        hash_value: Consistent hash for entity linking
    """

    entity_type: str = ""
    redacted_text: Optional[str] = None
    original_text: Optional[str] = None
    hash_value: Optional[str] = None

    def __post_init__(self):
        """Initialize entity_type from label if not set."""
        if not self.entity_type:
            self.entity_type = self.label


@dataclass
class DeidentificationResult:
    """Result of de-identification operation.

    Attributes:
        original_text: Input text before de-identification
        deidentified_text: Output text with PII redacted
        pii_entities: List of detected and redacted PII entities
        method: De-identification method used
        timestamp: When de-identification was performed
        mapping: Optional mapping for re-identification (redacted -> original)
    """

    original_text: str
    deidentified_text: str
    pii_entities: list[PIIEntity]
    method: str
    timestamp: datetime
    mapping: Optional[dict[str, str]] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary format.

        Returns:
            Dictionary with all result fields and metadata
        """
        return {
            "original_text": self.original_text,
            "deidentified_text": self.deidentified_text,
            "pii_entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "entity_type": e.entity_type,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "redacted_text": e.redacted_text,
                }
                for e in self.pii_entities
            ],
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
            "num_entities_redacted": len(self.pii_entities),
        }


# Languages whose PII models were trained on accent-free text.
# For these, input is automatically stripped of accents before model
# inference and entity positions are mapped back to the original text.
_ACCENT_NORMALIZE_LANGS = frozenset({"es"})

_DEFAULT_EN_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"


def _strip_accents(text: str) -> str:
    """Remove combining diacritical marks from *text*.

    The input is first NFC-normalised so that pre-composed characters like
    ``\u00e9`` are handled consistently.  After NFD decomposition every
    combining mark (Unicode category ``Mn``) is dropped and the result is
    NFC-normalised again.

    For common Latin-script accented characters this is a 1-to-1 character
    mapping, so ``len(result) == len(text)`` and character positions are
    preserved — critical for mapping model entity offsets back to the
    original text.

    Args:
        text: Arbitrary Unicode string.

    Returns:
        Accent-free copy with the same character count.
    """
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", nfc)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def extract_pii(
    text: str,
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
):
    """Extract PII entities from text with intelligent entity merging.

    Uses token classification models to detect personally identifiable information
    including names, emails, phone numbers, addresses, and other HIPAA-protected
    identifiers.

    The smart merging feature uses regex patterns to identify semantic units
    (dates, SSN, phone numbers, etc.) and merges fragmented model predictions
    into complete entities with dominant label selection.

    Args:
        text: Input text to analyze
        model_name: PII detection model (registry key or HuggingFace ID).
            When the default is used and ``lang`` is not ``"en"``, the
            language-appropriate default model is selected automatically.
        confidence_threshold: Minimum confidence score (0-1)
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)
        lang: ISO 639-1 language code (en, fr, de, it, es). Controls which
            default model and regex patterns are used.
        normalize_accents: Strip diacritical marks before model inference so
            that models trained on accent-free text still detect accented
            names.  Entity spans in the result reference the *original*
            (accented) text.  ``None`` (default) auto-enables for languages
            in ``_ACCENT_NORMALIZE_LANGS`` (currently Spanish).

    Returns:
        AnalysisResult with detected PII entities

    Example:
        >>> result = extract_pii("DOB: 01/15/1970, SSN: 123-45-6789")
        >>> for entity in result.entities:
        ...     print(f"{entity.label}: {entity.text}")
        date_of_birth: 01/15/1970
        ssn: 123-45-6789

        >>> # French PII detection
        >>> result = extract_pii("Né le 15/01/1970", lang="fr")
    """
    from .pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES

    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    # Resolve language-appropriate default model
    effective_model = model_name
    if model_name == _DEFAULT_EN_MODEL and lang != "en":
        effective_model = DEFAULT_PII_MODELS[lang]

    # Decide whether to strip accents before inference
    do_normalize = normalize_accents if normalize_accents is not None else (lang in _ACCENT_NORMALIZE_LANGS)

    original_text = text
    if do_normalize:
        text = _strip_accents(text)

    # Import here to avoid circular dependency
    from .. import analyze_text

    result = analyze_text(
        text,
        model_name=effective_model,
        confidence_threshold=confidence_threshold,
        config=config,
        group_entities=True,  # Group multi-token PII entities
    )

    # Map entity spans back to the original (possibly accented) text
    if do_normalize and original_text != text:
        result.text = original_text
        from ..processing.outputs import EntityPrediction as _EP
        result.entities = [
            _EP(
                text=original_text[e.start:e.end],
                label=e.label,
                start=e.start,
                end=e.end,
                confidence=e.confidence,
            )
            for e in result.entities
        ]

    # Apply smart merging if enabled
    if use_smart_merging:
        from .pii_entity_merger import merge_entities_with_semantic_units
        from .pii_i18n import get_patterns_for_language

        # Get language-specific patterns
        lang_patterns = get_patterns_for_language(lang)

        # Convert entities to dict format for merging
        entity_dicts = [
            {
                'entity_type': e.label,
                'score': e.confidence,
                'start': e.start,
                'end': e.end,
                'word': e.text
            }
            for e in result.entities
        ]

        # Merge using semantic patterns
        # IMPORTANT: Use result.text (validated/processed text) not original text
        # because entity positions are based on the processed text
        merged_dicts = merge_entities_with_semantic_units(
            entity_dicts,
            result.text,
            patterns=lang_patterns,
            use_semantic_patterns=True,
            prefer_model_labels=True  # Prefer model's more specific labels
        )

        # Convert back to EntityPrediction objects
        from ..processing.outputs import EntityPrediction
        merged_entities = [
            EntityPrediction(
                text=e['word'],
                label=e['entity_type'],
                start=e['start'],
                end=e['end'],
                confidence=e['score']
            )
            for e in merged_dicts
        ]

        # Update result
        result.entities = merged_entities
        result.num_entities = len(merged_entities)

    return result


def deidentify(
    text: str,
    method: DeidentificationMethod = "mask",
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.7,  # Higher threshold for safety
    keep_year: bool = True,
    shift_dates: bool = False,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
) -> DeidentificationResult:
    """De-identify text by detecting and redacting PII with intelligent merging.

    Implements multiple de-identification strategies for HIPAA compliance:

    - **mask**: Replace with placeholders like [NAME], [EMAIL], etc.
    - **remove**: Remove PII text entirely (empty string)
    - **replace**: Replace with fake but realistic data
    - **hash**: Replace with consistent hashed values for entity linking
    - **shift_dates**: Shift dates by random offset while preserving intervals

    Smart merging uses regex patterns to merge fragmented entities (e.g., dates
    split into '01' and '/15/1970' are merged into complete '01/15/1970').

    Args:
        text: Input text to de-identify
        method: De-identification method (mask, remove, replace, hash, shift_dates)
        model_name: PII detection model
        confidence_threshold: Minimum confidence for redaction (default 0.7 for safety)
        keep_year: For dates, keep the year unchanged
        shift_dates: Shift all dates by a consistent random offset
        date_shift_days: Specific number of days to shift (random if None)
        keep_mapping: Keep mapping for re-identification
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)
        lang: ISO 639-1 language code (en, fr, de, it, es). Controls model
            selection, regex patterns, and fake data for replacement.
        normalize_accents: Strip diacritical marks before model inference.
            ``None`` (default) auto-enables for Spanish.

    Returns:
        DeidentificationResult with original and de-identified text

    Example:
        >>> result = deidentify(
        ...     "Patient John Doe (DOB: 01/15/1970) called from 555-1234",
        ...     method="mask",
        ...     keep_year=True
        ... )
        >>> print(result.deidentified_text)
        Patient [NAME] (DOB: [DATE]/1970) called from [PHONE]

        >>> result = deidentify(text, method="replace", lang="de")
    """
    # Extract PII entities with smart merging
    pii_result = extract_pii(
        text, model_name, confidence_threshold, config, use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
    )

    # Convert to PIIEntity with metadata
    pii_entities = [
        PIIEntity(
            text=e.text,
            label=e.label,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            entity_type=e.label,  # Use label as entity_type
            original_text=e.text,
        )
        for e in pii_result.entities
    ]

    # Sort by position (reverse order for safe replacement)
    pii_entities.sort(key=lambda e: e.start, reverse=True)

    # Generate date shift offset if needed
    if shift_dates and date_shift_days is None:
        date_shift_days = random.randint(-365, 365)

    # Apply de-identification
    deidentified = text
    mapping = {} if keep_mapping else None

    for entity in pii_entities:
        redacted = _redact_entity(
            entity,
            method,
            keep_year=keep_year,
            date_shift_days=date_shift_days if shift_dates else None,
            lang=lang,
        )
        entity.redacted_text = redacted

        # Replace in text (working backwards to preserve offsets)
        deidentified = (
            deidentified[: entity.start] + redacted + deidentified[entity.end :]
        )

        # Store mapping
        if keep_mapping and mapping is not None:
            mapping[redacted] = entity.original_text or entity.text

    return DeidentificationResult(
        original_text=text,
        deidentified_text=deidentified,
        pii_entities=pii_entities,
        method=method,
        timestamp=datetime.now(),
        mapping=mapping,
    )


def _redact_entity(
    entity: PIIEntity,
    method: DeidentificationMethod,
    keep_year: bool = True,
    date_shift_days: Optional[int] = None,
    lang: str = "en",
) -> str:
    """Redact a single PII entity based on method.

    Args:
        entity: PIIEntity to redact
        method: De-identification method
        keep_year: Keep year in dates
        date_shift_days: Days to shift dates
        lang: Language code for fake data and date formatting

    Returns:
        Redacted text replacement
    """
    if method == "mask":
        # Replace with placeholder
        return f"[{entity.entity_type}]"

    elif method == "remove":
        # Remove entirely (replace with empty string)
        return ""

    elif method == "replace":
        # Replace with fake but realistic data
        return _generate_fake_pii(entity.entity_type, lang=lang)

    elif method == "hash":
        # Generate consistent hash
        hash_val = hashlib.sha256(entity.text.encode()).hexdigest()[:8]
        entity.hash_value = hash_val
        return f"{entity.entity_type}_{hash_val}"

    elif method == "shift_dates":
        # Shift dates by offset
        if entity.entity_type == "DATE" and date_shift_days is not None:
            return _shift_date(entity.text, date_shift_days, keep_year, lang=lang)
        else:
            # Non-date entities get masked
            return f"[{entity.entity_type}]"

    return entity.text


_LABEL_TO_FAKE_KEY: Dict[str, str] = {
    # Name variants
    "first_name": "FIRST_NAME",
    "FIRSTNAME": "FIRST_NAME",
    "firstname": "FIRST_NAME",
    "last_name": "LAST_NAME",
    "LASTNAME": "LAST_NAME",
    "lastname": "LAST_NAME",
    "name": "NAME",
    "NAME": "NAME",
    "patient": "NAME",
    "PATIENT": "NAME",
    "doctor": "NAME",
    "DOCTOR": "NAME",

    # Phone variants
    "phone_number": "PHONE",
    "PHONE": "PHONE",
    "phone": "PHONE",
    "PHONENUMBER": "PHONE",

    # Location variants
    "city": "LOCATION",
    "CITY": "LOCATION",
    "state": "LOCATION",
    "STATE": "LOCATION",
    "country": "LOCATION",
    "COUNTRY": "LOCATION",
    "location": "LOCATION",
    "LOCATION": "LOCATION",

    # Address variants
    "street_address": "STREET_ADDRESS",
    "STREET": "STREET_ADDRESS",
    "street": "STREET_ADDRESS",
    "STREETADDRESS": "STREET_ADDRESS",
    "address": "STREET_ADDRESS",
    "ADDRESS": "STREET_ADDRESS",

    # Date variants
    "date": "DATE",
    "DATE": "DATE",
    "date_of_birth": "DATE",
    "DATEOFBIRTH": "DATE",
    "dateofbirth": "DATE",
    "dob": "DATE",
    "DOB": "DATE",

    # ID variants
    "id_num": "ID_NUM",
    "ID_NUM": "ID_NUM",
    "ssn": "ID_NUM",
    "SSN": "ID_NUM",
    "national_id": "ID_NUM",
    "NATIONAL_ID": "ID_NUM",
    "medical_record_number": "ID_NUM",
    "MEDICAL_RECORD_NUMBER": "ID_NUM",

    # Other
    "email": "EMAIL",
    "EMAIL": "EMAIL",
    "age": "AGE",
    "AGE": "AGE",
    "username": "USERNAME",
    "USERNAME": "USERNAME",
    "url_personal": "URL_PERSONAL",
    "URL_PERSONAL": "URL_PERSONAL",
    "zipcode": "ZIPCODE",
    "ZIPCODE": "ZIPCODE",
    "zip": "ZIPCODE",
    "ZIP": "ZIPCODE",
    "postal_code": "ZIPCODE",
}


def _generate_fake_pii(entity_type: str, lang: str = "en") -> str:
    """Generate fake but realistic PII data.

    Args:
        entity_type: Type of PII entity
        lang: Language code for language-appropriate fake data

    Returns:
        Fake replacement text
    """
    from .pii_i18n import LANGUAGE_FAKE_DATA

    fake_data = LANGUAGE_FAKE_DATA.get(lang, LANGUAGE_FAKE_DATA["en"])

    # Resolve the model label to a fake-data key
    key = _LABEL_TO_FAKE_KEY.get(entity_type, entity_type.upper())

    if key in fake_data:
        return random.choice(fake_data[key])

    # Fall back to English if the entity type isn't in the language-specific data
    en_data = LANGUAGE_FAKE_DATA["en"]
    if key in en_data:
        return random.choice(en_data[key])

    return f"[{entity_type}]"


def _shift_date(
    date_str: str, shift_days: int, keep_year: bool = True, lang: str = "en",
) -> str:
    """Shift a date string by specified number of days.

    Supports multiple date formats commonly found in clinical documents:
    - MM/DD/YYYY, MM-DD-YYYY (US/English)
    - DD/MM/YYYY, DD-MM-YYYY (French/Italian)
    - DD.MM.YYYY (German)
    - YYYY-MM-DD (ISO)
    - Month DD, YYYY / DD Month YYYY (with localized month names)

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift (positive = future, negative = past)
        keep_year: Keep the year unchanged (only shift month/day)
        lang: Language code for date format conventions

    Returns:
        Shifted date string in the same format as input
    """
    # Try to parse and shift using dateutil if available
    try:
        from dateutil import parser as date_parser
        from dateutil.relativedelta import relativedelta
    except ImportError:
        # Fallback without dateutil - basic pattern matching
        return _shift_date_basic(date_str, shift_days, keep_year, lang=lang)

    try:
        # For European languages, try day-first parsing
        dayfirst = lang in ("fr", "de", "it", "es")
        parsed_date = date_parser.parse(date_str, fuzzy=False, dayfirst=dayfirst)
        original_year = parsed_date.year

        # Shift the date
        shifted_date = parsed_date + timedelta(days=shift_days)

        # If keep_year is True, restore the original year
        if keep_year:
            shifted_date = shifted_date.replace(year=original_year)

        # Try to preserve the original format
        return _format_date_like_original(date_str, shifted_date, lang=lang)

    except (ValueError, OverflowError):
        # If parsing fails, return a masked placeholder
        return "[DATE_SHIFTED]"


def _shift_date_basic(
    date_str: str, shift_days: int, keep_year: bool = True, lang: str = "en",
) -> str:
    """Basic date shifting without dateutil dependency.

    Handles common date formats using regex and datetime.

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift
        keep_year: Keep the year unchanged
        lang: Language code for date format conventions

    Returns:
        Shifted date string or placeholder
    """
    # Order patterns based on language convention
    if lang in ("fr", "it", "es"):
        # European: DD/MM/YYYY first
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    elif lang == "de":
        # German: DD.MM.YYYY
        patterns = [
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", "dmy"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    else:
        # US/English: MM/DD/YYYY first
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "mdy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
        ]

    for pattern, order in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            groups = match.groups()
            try:
                if order == "mdy":
                    month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                elif order == "ymd":
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # dmy
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])

                # Handle 2-digit years
                if year < 100:
                    year += 2000 if year < 50 else 1900

                # Validate and create date
                original_date = datetime(year, month, day)
                original_year = original_date.year

                # Shift
                shifted = original_date + timedelta(days=shift_days)

                # Keep year if requested
                if keep_year:
                    shifted = shifted.replace(year=original_year)

                # Format back preserving separator
                if "." in date_str:
                    sep = "."
                elif "/" in date_str:
                    sep = "/"
                else:
                    sep = "-"

                if order == "mdy":
                    return f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{shifted.year}"
                elif order == "ymd":
                    return f"{shifted.year}{sep}{shifted.month:02d}{sep}{shifted.day:02d}"
                else:
                    return f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{shifted.year}"

            except (ValueError, OverflowError):
                continue

    return "[DATE_SHIFTED]"


def _format_date_like_original(
    original: str, new_date: datetime, lang: str = "en",
) -> str:
    """Format a datetime to match the original string's format.

    Args:
        original: Original date string (for format detection)
        new_date: New datetime to format
        lang: Language code for date format conventions

    Returns:
        Formatted date string
    """
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    original_stripped = original.strip()

    # ISO format: YYYY-MM-DD
    if re.match(r"\d{4}-\d{2}-\d{2}", original_stripped):
        return new_date.strftime("%Y-%m-%d")

    # German dot-separated: DD.MM.YYYY
    if re.match(r"\d{1,2}\.\d{1,2}\.\d{2,4}", original_stripped):
        return new_date.strftime("%d.%m.%Y")

    # Slash-separated dates: interpretation depends on language
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", original_stripped):
        if lang in ("fr", "de", "it", "es"):
            # European: DD/MM/YYYY
            return new_date.strftime("%d/%m/%Y")
        else:
            # US: MM/DD/YYYY
            return new_date.strftime("%m/%d/%Y")

    # Dash-separated dates
    if re.match(r"\d{1,2}-\d{1,2}-\d{4}", original_stripped):
        if lang in ("fr", "de", "it", "es"):
            return new_date.strftime("%d-%m-%Y")
        else:
            return new_date.strftime("%m-%d-%Y")

    # Month name formats - check all supported languages
    month_names_flat = []
    for month_list in LANGUAGE_MONTH_NAMES.values():
        month_names_flat.extend(m.lower() for m in month_list)

    original_lower = original_stripped.lower()
    for month in month_names_flat:
        if month in original_lower:
            # Use language-specific month name
            lang_months = LANGUAGE_MONTH_NAMES.get(lang, LANGUAGE_MONTH_NAMES["en"])
            month_name = lang_months[new_date.month - 1]

            # "15 janvier 2020" or "January 15, 2020" format
            if re.match(r"\d+\.?\s+[A-Za-z\u00c0-\u00ff]+\s+\d{4}", original_stripped):
                return f"{new_date.day} {month_name} {new_date.year}"
            if re.match(r"[A-Za-z\u00c0-\u00ff]+\s+\d+,?\s+\d{4}", original_stripped):
                return f"{month_name} {new_date.day}, {new_date.year}"
            break

    # Default to ISO format
    return new_date.strftime("%Y-%m-%d")


def reidentify(
    deidentified_text: str,
    mapping: dict[str, str],
) -> str:
    """Re-identify text using stored mapping.

    Restores original PII from de-identified text using the mapping created
    during de-identification. Only works if keep_mapping=True was used.

    Args:
        deidentified_text: De-identified text
        mapping: Mapping from redacted to original text

    Returns:
        Re-identified text with original PII restored

    Example:
        >>> result = deidentify(text, method="mask", keep_mapping=True)
        >>> original = reidentify(result.deidentified_text, result.mapping)
        >>> assert original == text

    Note:
        Only works if keep_mapping=True was used during de-identification.
        Requires proper authorization and audit logging in production.
    """
    result = deidentified_text

    for redacted, original in mapping.items():
        result = result.replace(redacted, original)

    return result
