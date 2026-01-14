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
from typing import Optional, Literal
from datetime import datetime, timedelta
import hashlib
import random
import re

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


def extract_pii(
    text: str,
    model_name: str = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
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
        model_name: PII detection model (registry key or HuggingFace ID)
        confidence_threshold: Minimum confidence score (0-1)
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)

    Returns:
        AnalysisResult with detected PII entities

    Example:
        >>> result = extract_pii("DOB: 01/15/1970, SSN: 123-45-6789")
        >>> for entity in result.entities:
        ...     print(f"{entity.label}: {entity.text}")
        date_of_birth: 01/15/1970
        ssn: 123-45-6789

        Without smart merging, the date might be split into fragments:
        >>> result = extract_pii(text, use_smart_merging=False)
        >>> # May produce: date: '01', date_of_birth: '/15/1970'
    """
    # Import here to avoid circular dependency
    from .. import analyze_text

    result = analyze_text(
        text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        group_entities=True,  # Group multi-token PII entities
    )

    # Apply smart merging if enabled
    if use_smart_merging:
        from .pii_entity_merger import merge_entities_with_semantic_units

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
    model_name: str = "pii_detection",
    confidence_threshold: float = 0.7,  # Higher threshold for safety
    keep_year: bool = True,
    shift_dates: bool = False,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
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

        >>> result = deidentify(text, method="replace")
        >>> print(result.deidentified_text)
        Patient Jane Smith (DOB: 03/22/1970) called from 555-9876
    """
    # Extract PII entities with smart merging
    pii_result = extract_pii(text, model_name, confidence_threshold, config, use_smart_merging)

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
) -> str:
    """Redact a single PII entity based on method.

    Args:
        entity: PIIEntity to redact
        method: De-identification method
        keep_year: Keep year in dates
        date_shift_days: Days to shift dates

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
        return _generate_fake_pii(entity.entity_type)

    elif method == "hash":
        # Generate consistent hash
        hash_val = hashlib.sha256(entity.text.encode()).hexdigest()[:8]
        entity.hash_value = hash_val
        return f"{entity.entity_type}_{hash_val}"

    elif method == "shift_dates":
        # Shift dates by offset
        if entity.entity_type == "DATE" and date_shift_days is not None:
            return _shift_date(entity.text, date_shift_days, keep_year)
        else:
            # Non-date entities get masked
            return f"[{entity.entity_type}]"

    return entity.text


def _generate_fake_pii(entity_type: str) -> str:
    """Generate fake but realistic PII data.

    Args:
        entity_type: Type of PII entity

    Returns:
        Fake replacement text
    """
    fake_data = {
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
    }

    if entity_type in fake_data:
        return random.choice(fake_data[entity_type])

    return f"[{entity_type}]"


def _shift_date(date_str: str, shift_days: int, keep_year: bool = True) -> str:
    """Shift a date string by specified number of days.

    Supports multiple date formats commonly found in clinical documents:
    - MM/DD/YYYY, MM-DD-YYYY
    - DD/MM/YYYY, DD-MM-YYYY (European)
    - YYYY-MM-DD (ISO)
    - Month DD, YYYY (e.g., January 15, 2020)
    - DD Month YYYY (e.g., 15 January 2020)

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift (positive = future, negative = past)
        keep_year: Keep the year unchanged (only shift month/day)

    Returns:
        Shifted date string in the same format as input
    """
    # Try to parse and shift using dateutil if available
    try:
        from dateutil import parser as date_parser
        from dateutil.relativedelta import relativedelta
    except ImportError:
        # Fallback without dateutil - basic pattern matching
        return _shift_date_basic(date_str, shift_days, keep_year)

    try:
        # Parse the date
        parsed_date = date_parser.parse(date_str, fuzzy=False)
        original_year = parsed_date.year

        # Shift the date
        shifted_date = parsed_date + timedelta(days=shift_days)

        # If keep_year is True, restore the original year
        if keep_year:
            shifted_date = shifted_date.replace(year=original_year)

        # Try to preserve the original format
        return _format_date_like_original(date_str, shifted_date)

    except (ValueError, OverflowError):
        # If parsing fails, return a masked placeholder
        return "[DATE_SHIFTED]"


def _shift_date_basic(date_str: str, shift_days: int, keep_year: bool = True) -> str:
    """Basic date shifting without dateutil dependency.

    Handles common date formats using regex and datetime.

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift
        keep_year: Keep the year unchanged

    Returns:
        Shifted date string or placeholder
    """
    # Common date patterns
    patterns = [
        # MM/DD/YYYY or MM-DD-YYYY
        (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "%m/%d/%Y", "mdy"),
        # YYYY-MM-DD (ISO)
        (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "%Y-%m-%d", "ymd"),
        # DD/MM/YYYY (allow European interpretation based on values)
        (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "%d/%m/%Y", "dmy"),
    ]

    for pattern, fmt, order in patterns:
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

                # Validate and create date
                original_date = datetime(year, month, day)
                original_year = original_date.year

                # Shift
                shifted = original_date + timedelta(days=shift_days)

                # Keep year if requested
                if keep_year:
                    shifted = shifted.replace(year=original_year)

                # Format back
                sep = "/" if "/" in date_str else "-"
                if order == "mdy":
                    return f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{shifted.year}"
                elif order == "ymd":
                    return f"{shifted.year}{sep}{shifted.month:02d}{sep}{shifted.day:02d}"
                else:
                    return f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{shifted.year}"

            except (ValueError, OverflowError):
                continue

    return "[DATE_SHIFTED]"


def _format_date_like_original(original: str, new_date: datetime) -> str:
    """Format a datetime to match the original string's format.

    Args:
        original: Original date string (for format detection)
        new_date: New datetime to format

    Returns:
        Formatted date string
    """
    # Detect format from original string
    original_stripped = original.strip()

    # ISO format: YYYY-MM-DD
    if re.match(r"\d{4}-\d{2}-\d{2}", original_stripped):
        return new_date.strftime("%Y-%m-%d")

    # US format: MM/DD/YYYY
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", original_stripped):
        return new_date.strftime("%m/%d/%Y")

    # US with dashes: MM-DD-YYYY
    if re.match(r"\d{1,2}-\d{1,2}-\d{4}", original_stripped):
        return new_date.strftime("%m-%d-%Y")

    # European format: DD/MM/YYYY
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", original_stripped):
        return new_date.strftime("%d/%m/%Y")

    # Month name formats
    month_names = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december"
    ]

    original_lower = original_stripped.lower()
    for month in month_names:
        if month in original_lower:
            # January 15, 2020 format
            if re.match(r"[A-Za-z]+ \d+,? \d{4}", original_stripped):
                return new_date.strftime("%B %d, %Y")
            # 15 January 2020 format
            if re.match(r"\d+ [A-Za-z]+ \d{4}", original_stripped):
                return new_date.strftime("%d %B %Y")
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
