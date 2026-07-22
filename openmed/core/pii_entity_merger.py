"""Advanced entity merging strategies for PII detection.

This module provides intelligent entity merging that combines:
1. Regex-based semantic unit detection (dates, SSN, phone, email, etc.)
2. Model prediction aggregation with dominant label selection
3. BIO-aware post-processing

This solves the common problem where tokenizers split semantic units like
"01/15/1970" into multiple sub-tokens, leading to fragmented entity predictions.

Example:
    Model predictions:
        - [date] '01' (confidence: 0.711)
        - [date_of_birth] '/15/1970' (confidence: 0.751)

    After merging:
        - [date_of_birth] '01/15/1970' (confidence: 0.731)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class PIIPattern:
    """A regex pattern for detecting PII semantic units with context-aware scoring.

    Inspired by Microsoft Presidio's PatternRecognizer approach:
    - base_score: Low confidence for pattern-only matches (like Presidio's 0.01-0.3)
    - context_words: Keywords that boost confidence when found nearby
    - requires_context: Reject pattern-only matches without nearby context
    - validator: Optional checksum/validation function to confirm matches
    - reject_on_validation_failure: Drop regex matches whose validator fails,
      rather than retaining them with a reduced confidence score
    - context_required: Require nearby context for semantic-only recognition
    - safety_sweep_requires_context: Require nearby context before the
      deterministic safety sweep accepts this pattern

    Example:
        PIIPattern(
            pattern=r'\\b\\d{3}-\\d{2}-\\d{4}\\b',
            entity_type='ssn',
            base_score=0.3,
            context_words=['ssn', 'social security', 'social security number'],
            validator=validate_ssn
        )
    """

    pattern: str
    entity_type: str
    priority: int = 0  # Higher priority patterns checked first
    flags: int = re.IGNORECASE
    base_score: float = 0.5  # Score when pattern matches without context
    context_boost: float = 0.35  # Additional score when context words found
    context_words: List[str] = field(
        default_factory=list
    )  # Keywords that boost confidence
    requires_context: bool = False
    validator: Optional[Callable[[str], bool]] = (
        None  # Validation function (e.g., checksum)
    )
    context_required: bool = False
    safety_sweep_requires_context: bool = False
    reject_on_validation_failure: bool = False


# ============================================================================
# Validation Functions (inspired by Presidio's checksum approach)
# ============================================================================


def validate_ssn(ssn_text: str) -> bool:
    """Validate SSN format and basic rules.

    SSN rules:
    - Cannot have all zeros in any group (000-XX-XXXX, XXX-00-XXXX, XXX-XX-0000)
    - Area code (first 3) cannot be 666 or 900-999

    Note: We allow sequential patterns like 123-45-6789 for testing purposes.

    Args:
        ssn_text: SSN string (may have hyphens or spaces)

    Returns:
        True if valid SSN format
    """
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_ssn(ssn_text)


def validate_luhn(number_text: str) -> bool:
    """Validate using Luhn algorithm (for credit cards, some IDs).

    The Luhn algorithm is used to validate credit card numbers and some
    other identification numbers.

    Args:
        number_text: Numeric string (may contain spaces/hyphens)

    Returns:
        True if passes Luhn checksum
    """
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_luhn(number_text)


def validate_npi(npi_text: str) -> bool:
    """Validate NPI (National Provider Identifier) using Luhn algorithm.

    NPIs are 10-digit numbers that use Luhn checksum with a prefix of 80840.

    Args:
        npi_text: 10-digit NPI string

    Returns:
        True if valid NPI
    """
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_npi(npi_text)


def validate_iban(iban_text: str) -> bool:
    """Validate an IBAN using the shared clinical identifier validator."""
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_iban(iban_text)


def validate_bic(bic_text: str) -> bool:
    """Validate a SWIFT/BIC using the shared clinical identifier validator."""
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_bic(bic_text)


def validate_phone_us(phone_text: str) -> bool:
    """Validate US phone number format.

    Checks:
    - Area code cannot start with 0 or 1
    - Exchange (middle 3 digits) cannot start with 0 (allows 1 for testing)

    Note: We relax some rules for common test numbers like 555-123-4567.

    Args:
        phone_text: Phone number string

    Returns:
        True if valid US phone format
    """
    from .anonymizer.providers import clinical_ids

    return clinical_ids.validate_phone_us(phone_text)


# Comprehensive PII regex patterns with context-aware scoring
# Following Presidio's philosophy: low base scores, boosted by context
PII_PATTERNS = [
    # Dates (highest priority - most specific first)
    # Dates are common, so moderate base score
    PIIPattern(
        r"\b\d{4}-\d{2}-\d{2}\b",
        "date",
        priority=10,
        base_score=0.6,
        context_words=[
            "dob",
            "birth",
            "born",
            "date of birth",
            "birthdate",
            "deceased",
            "died",
            "admitted",
            "discharged",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "dob",
            "birth",
            "born",
            "date of birth",
            "birthdate",
            "deceased",
            "died",
            "admitted",
            "discharged",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
        "date",
        priority=9,
        base_score=0.6,
        context_words=[
            "dob",
            "birth",
            "born",
            "date of birth",
            "birthdate",
            "deceased",
            "died",
            "admitted",
            "discharged",
        ],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "dob",
            "birth",
            "born",
            "date of birth",
            "birthdate",
            "deceased",
            "died",
            "admitted",
            "discharged",
        ],
        context_boost=0.25,
    ),
    PIIPattern(
        r"\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b",
        "date",
        priority=8,
        base_score=0.7,
        context_words=[
            "dob",
            "birth",
            "born",
            "date of birth",
            "birthdate",
            "deceased",
            "died",
            "admitted",
            "discharged",
        ],
        context_boost=0.25,
    ),
    # SSN (very specific pattern with validation)
    PIIPattern(
        r"\b\d{3}-\d{2}-\d{4}\b",
        "ssn",
        priority=10,
        base_score=0.3,  # Low without context - could be other IDs
        context_words=[
            "ssn",
            "social security",
            "social security number",
            "ss#",
            "ss number",
        ],
        context_boost=0.55,  # High boost with context (0.3 + 0.55 = 0.85)
        validator=validate_ssn,
    ),
    PIIPattern(
        r"\b\d{3}\s\d{2}\s\d{4}\b",
        "ssn",
        priority=9,
        base_score=0.3,
        context_words=[
            "ssn",
            "social security",
            "social security number",
            "ss#",
            "ss number",
        ],
        context_boost=0.55,
        validator=validate_ssn,
    ),
    # Phone numbers with validation
    PIIPattern(
        r"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b",
        "phone_number",
        priority=9,
        base_score=0.6,  # Format is pretty specific
        context_words=[
            "phone",
            "tel",
            "telephone",
            "cell",
            "mobile",
            "fax",
            "call",
            "contact",
        ],
        context_boost=0.3,
        validator=validate_phone_us,
    ),
    PIIPattern(
        r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b",
        "phone_number",
        priority=8,
        base_score=0.5,
        context_words=[
            "phone",
            "tel",
            "telephone",
            "cell",
            "mobile",
            "fax",
            "call",
            "contact",
        ],
        context_boost=0.35,
        validator=validate_phone_us,
    ),
    PIIPattern(
        r"\b\d{10}\b",
        "phone_number",
        priority=5,  # Low priority - ambiguous (could be NPI, account, etc.)
        base_score=0.2,  # Very low base score
        context_words=[
            "phone",
            "tel",
            "telephone",
            "cell",
            "mobile",
            "fax",
            "call",
            "contact",
        ],
        context_boost=0.5,  # Needs context to be confident
        validator=validate_phone_us,
    ),
    # NPI - 10 digit with specific validation
    PIIPattern(
        r"\b\d{10}\b",
        "npi",
        priority=6,  # Slightly higher than phone for 10-digit
        base_score=0.15,  # Very low - needs context OR validation
        context_words=[
            "npi",
            "national provider",
            "provider number",
            "provider id",
            "provider identifier",
        ],
        context_boost=0.65,  # Strong boost with context
        validator=validate_npi,
    ),
    # Email addresses (very specific pattern)
    PIIPattern(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "email",
        priority=10,
        base_score=0.9,  # Email pattern is very specific
        context_words=["email", "e-mail", "contact", "mail"],
        context_boost=0.1,
    ),
    # ZIP codes (US)
    PIIPattern(
        r"\b\d{5}(?:-\d{4})?\b",
        "postcode",
        priority=7,
        base_score=0.4,  # Could be other 5-digit numbers
        context_words=["zip", "zipcode", "zip code", "postal", "postal code"],
        context_boost=0.45,
        safety_sweep_requires_context=True,
    ),
    # Credit card with Luhn validation
    PIIPattern(
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "credit_debit_card",
        priority=8,
        base_score=0.4,
        context_words=[
            "card",
            "credit",
            "debit",
            "visa",
            "mastercard",
            "amex",
            "discover",
            "payment",
        ],
        context_boost=0.4,
        validator=validate_luhn,
    ),
    # IBANs with ISO 13616 checksum validation
    PIIPattern(
        r"\b[A-Z]{2}\d{2}(?:[\s-]?[A-Z0-9]){11,30}\b",
        "iban",
        priority=9,
        flags=0,
        base_score=0.85,
        context_words=["iban", "bank account", "account", "routing", "wire"],
        context_boost=0.1,
        validator=validate_iban,
    ),
    # SWIFT/BIC codes have fixed 8/11-character structure but no checksum, so
    # require context during deterministic safety sweeps.
    PIIPattern(
        r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b",
        "bic",
        priority=8,
        flags=0,
        base_score=0.25,
        context_words=[
            "bic",
            "swift",
            "swift/bic",
            "swift code",
            "bank identifier",
            "wire",
            "wire transfer",
        ],
        context_boost=0.65,
        validator=validate_bic,
        safety_sweep_requires_context=True,
    ),
    # Account numbers when explicitly labeled in context
    PIIPattern(
        r"\b(?:account number|account #|acct\.|acct|account)[:\s#-]*[A-Z0-9][A-Z0-9-]{5,23}\b",
        "account_number",
        priority=8,
        base_score=0.75,
        context_words=["account", "acct", "account number", "bank", "billing"],
        context_boost=0.15,
        flags=re.IGNORECASE,
    ),
    # Medical record numbers (common formats)
    PIIPattern(
        r"\b(?:MRN|mrn)[:\s#]*\d{6,10}\b",
        "medical_record_number",
        priority=9,
        base_score=0.8,  # "MRN" prefix is strong indicator
        context_words=[
            "medical record",
            "patient id",
            "patient number",
            "record number",
        ],
        context_boost=0.15,
    ),
    PIIPattern(
        r"\b[A-Z]{2,3}\d{6,9}\b",
        "medical_record_number",
        priority=5,
        base_score=0.3,  # Generic pattern
        context_words=[
            "mrn",
            "medical record",
            "patient id",
            "patient number",
            "record number",
        ],
        context_boost=0.5,
    ),
    # Street addresses (basic - number + street)
    PIIPattern(
        r"\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b",
        "street_address",
        priority=7,
        base_score=0.7,  # Street suffix is good indicator
        context_words=[
            "address",
            "street",
            "resides",
            "residence",
            "lives at",
            "located at",
        ],
        context_boost=0.2,
        flags=re.IGNORECASE,
    ),
    # URLs
    PIIPattern(
        r"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b",
        "url",
        priority=8,
        base_score=0.8,
        context_words=["url", "website", "link", "webpage"],
        context_boost=0.15,
    ),
    # IP addresses
    PIIPattern(
        r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ipv4",
        priority=7,
        base_score=0.6,
        context_words=["ip", "ip address", "address", "server", "host"],
        context_boost=0.3,
    ),
    PIIPattern(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b",
        "ipv6",
        priority=8,
        base_score=0.85,  # IPv6 pattern is very specific
        context_words=["ip", "ipv6", "ip address", "address", "server", "host"],
        context_boost=0.15,
    ),
    # MAC addresses
    PIIPattern(
        r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
        "mac_address",
        priority=8,
        base_score=0.75,
        context_words=["mac", "mac address", "hardware address"],
        context_boost=0.2,
    ),
]


# ============================================================================
# Context Detection (inspired by Presidio's LemmaContextAwareEnhancer)
# ============================================================================


def find_context_words(
    text: str,
    start: int,
    end: int,
    context_words: List[str],
    context_window: int = 100,
    *,
    require_boundaries: bool = False,
) -> bool:
    """Check if context words appear near the matched entity.

    Inspired by Presidio's context-aware enhancement. Looks for lemmatized
    context words within a window before/after the match.

    Args:
        text: Full input text
        start: Entity start position
        end: Entity end position
        context_words: List of keywords to search for
        context_window: Number of characters to search before/after (default: 100)
        require_boundaries: Require context phrases to begin and end outside
            word characters, preventing short keywords from matching inside
            unrelated words.

    Returns:
        True if any context word found within window

    Example:
        >>> text = "Patient SSN: 123-45-6789"
        >>> find_context_words(text, 13, 24, ['ssn', 'social security'])
        True
    """
    if not context_words:
        return False

    # Extract context window
    window_start = max(0, start - context_window)
    window_end = min(len(text), end + context_window)
    context_text = text[window_start:window_end].lower()

    # Simple lemmatization: strip common suffixes and check
    # More sophisticated would use spaCy/nltk lemmatizer
    for word in context_words:
        word_lower = word.lower()

        # Direct match
        if not require_boundaries and word_lower in context_text:
            return True

        # Check word boundaries (avoid partial matches like "ssn" in "assign")
        # Use word boundary pattern
        if re.search(r"\b" + re.escape(word_lower) + r"\b", context_text):
            return True

    return False


def find_semantic_units(
    text: str, patterns: Optional[List[PIIPattern]] = None
) -> List[Tuple[int, int, str, float, PIIPattern]]:
    """Find semantic units in text using regex patterns with context-aware scoring.

    Args:
        text: Input text to analyze
        patterns: Optional custom patterns (uses PII_PATTERNS if None)

    Returns:
        List of tuples (start, end, entity_type, score, pattern) sorted by start position.
        Score is calculated based on:
        - base_score from pattern
        - context_boost if context words found nearby
        - Validation penalty if validator exists and fails

    Example:
        >>> text = "DOB: 01/15/1970, SSN: 123-45-6789"
        >>> units = find_semantic_units(text)
        >>> units[0]
        (5, 15, 'date', 0.5, <PIIPattern>)
        >>> units[1]  # SSN with context "SSN:"
        (22, 33, 'ssn', 0.85, <PIIPattern>)
    """
    if patterns is None:
        patterns = PII_PATTERNS

    units = []

    # Sort patterns by priority (highest first)
    sorted_patterns = sorted(patterns, key=lambda p: p.priority, reverse=True)

    for pii_pattern in sorted_patterns:
        for match in re.finditer(pii_pattern.pattern, text, pii_pattern.flags):
            # Check for overlap with higher-priority existing units
            overlaps = False
            for existing in units:
                existing_start, existing_end = existing[0], existing[1]
                if match.start() < existing_end and match.end() > existing_start:
                    overlaps = True
                    break

            if overlaps:
                continue

            matched_text = text[match.start() : match.end()]

            has_context = bool(
                pii_pattern.context_words
                and find_context_words(
                    text,
                    match.start(),
                    match.end(),
                    pii_pattern.context_words,
                )
            )
            if pii_pattern.requires_context:
                has_required_context = find_context_words(
                    text,
                    match.start(),
                    match.end(),
                    pii_pattern.context_words,
                    require_boundaries=True,
                )
                if not has_required_context:
                    continue

            # Calculate score with context awareness
            score = pii_pattern.base_score

            # Check for context words (like Presidio)
            if has_context:
                score = min(1.0, score + pii_pattern.context_boost)
            if pii_pattern.context_required and not has_context:
                continue

            # Validate if validator exists
            validated = True
            if pii_pattern.validator:
                is_valid = pii_pattern.validator(matched_text)
                if not is_valid:
                    if pii_pattern.reject_on_validation_failure:
                        continue
                    # Failed validation - significantly reduce score
                    score = score * 0.3  # Reduce to 30% confidence
                    validated = False

            units.append(
                (
                    match.start(),
                    match.end(),
                    pii_pattern.entity_type,
                    score,
                    pii_pattern,
                    validated,
                )
            )

    # Sort by start position
    units.sort(key=lambda x: x[0])
    return units


def calculate_dominant_label(
    entities: List[Dict[str, Any]], tie_breaker: str = "confidence"
) -> Tuple[str, float]:
    """Calculate the dominant label from a list of entities.

    Args:
        entities: List of entity dicts with 'entity_type' and 'score' keys
        tie_breaker: How to break ties ('confidence' or 'first')

    Returns:
        Tuple of (dominant_label, average_confidence)

    Example:
        >>> entities = [
        ...     {'entity_type': 'date', 'score': 0.7},
        ...     {'entity_type': 'date_of_birth', 'score': 0.9},
        ...     {'entity_type': 'date_of_birth', 'score': 0.8}
        ... ]
        >>> calculate_dominant_label(entities)
        ('date_of_birth', 0.8)
    """
    if not entities:
        raise ValueError("Cannot calculate dominant label from empty entity list")

    # Count occurrences
    label_counts = {}
    label_confidences = {}

    for entity in entities:
        label = entity["entity_type"]
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in label_confidences:
            label_confidences[label] = []
        label_confidences[label].append(entity["score"])

    # Find most frequent
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]

    if len(candidates) == 1:
        dominant_label = candidates[0]
    elif tie_breaker == "confidence":
        # Break tie by highest average confidence
        avg_confidences = {
            label: sum(label_confidences[label]) / len(label_confidences[label])
            for label in candidates
        }
        dominant_label = max(avg_confidences, key=avg_confidences.get)
    else:  # tie_breaker == 'first'
        # Use first occurrence
        for entity in entities:
            if entity["entity_type"] in candidates:
                dominant_label = entity["entity_type"]
                break

    # Calculate average confidence
    avg_confidence = sum(e["score"] for e in entities) / len(entities)

    return dominant_label, avg_confidence


def merge_entities_with_semantic_units(
    entities: List[Dict[str, Any]],
    text: str,
    use_semantic_patterns: bool = True,
    patterns: Optional[List[PIIPattern]] = None,
    prefer_model_labels: bool = False,
    allow_semantic_only_matches: bool = True,
    allow_label_expansion: bool = True,
    india_clinical: bool = False,
) -> List[Dict[str, Any]]:
    """Merge entity predictions using semantic unit patterns.

    This is the main merging function that combines regex-based semantic units
    with model predictions to produce clean, complete entities.

    Args:
        entities: List of entity dicts from model (with keys: entity_type, score, start, end, word)
        text: Original text
        use_semantic_patterns: Whether to use regex patterns for semantic units
        patterns: Optional custom patterns (uses PII_PATTERNS if None)
        prefer_model_labels: If True, prefer model's label over pattern's label
        allow_semantic_only_matches: If False, regex-only matches are not added
        allow_label_expansion: If False, keep the model label taxonomy unchanged
        india_clinical: Normalize Indian-English clinical abbreviations before
            merging and bridge eligible PERSON/LOCATION spans across Latin and
            Indic script boundaries.

    Returns:
        List of merged entity dicts

    Example:
        >>> entities = [
        ...     {'entity_type': 'date', 'score': 0.7, 'start': 5, 'end': 7, 'word': '01'},
        ...     {'entity_type': 'date_of_birth', 'score': 0.9, 'start': 7, 'end': 15, 'word': '/15/1970'}
        ... ]
        >>> text = "DOB: 01/15/1970"
        >>> merged = merge_entities_with_semantic_units(entities, text)
        >>> merged[0]
        {'entity_type': 'date_of_birth', 'score': 0.8, 'start': 5, 'end': 15,
         'word': '01/15/1970', 'merged_from': 2}
    """
    if india_clinical:
        from ..clinical.normalization import normalize_indian_clinical_entities

        entities = normalize_indian_clinical_entities(entities)

    if not use_semantic_patterns:
        # Just return entities as-is if not using patterns
        ordered = sorted(entities, key=lambda x: x["start"])
        return (
            merge_india_code_mixed_spans(ordered, text) if india_clinical else ordered
        )

    # Find semantic units
    semantic_units = find_semantic_units(text, patterns)

    if not semantic_units:
        # No semantic units found, return original entities
        ordered = sorted(entities, key=lambda x: x["start"])
        return (
            merge_india_code_mixed_spans(ordered, text) if india_clinical else ordered
        )

    merged = []
    used_entities = set()

    # Process each semantic unit (includes score, pattern, and validation flag)
    for unit_tuple in semantic_units:
        # Unpack with backwards-compat for 5-element tuples (pre-v0.6.4)
        if len(unit_tuple) >= 6:
            (
                unit_start,
                unit_end,
                unit_type,
                unit_score,
                unit_pattern,
                unit_validated,
            ) = unit_tuple[:6]
        else:
            unit_start, unit_end, unit_type, unit_score, unit_pattern = unit_tuple[:5]
            unit_validated = True
        # Find all entities that overlap with this semantic unit
        overlapping = []
        for i, entity in enumerate(entities):
            if entity["start"] < unit_end and entity["end"] > unit_start:
                overlapping.append((i, entity))

        if overlapping:
            # Calculate dominant label from model predictions
            overlapping_entities = [e for _, e in overlapping]
            dominant_label, model_avg_confidence = calculate_dominant_label(
                overlapping_entities
            )

            # Decide which label to use
            if not allow_label_expansion:
                final_label = dominant_label
            elif prefer_model_labels:
                if normalize_label(dominant_label) == normalize_label(
                    unit_type
                ) or is_more_specific(dominant_label, unit_type):
                    final_label = dominant_label
                else:
                    final_label = unit_type
            else:
                # Use pattern's label if it matches any model prediction
                model_labels = set(e["entity_type"] for e in overlapping_entities)
                # Normalize labels for comparison (handle variations)
                if any(
                    normalize_label(unit_type) == normalize_label(ml)
                    for ml in model_labels
                ):
                    final_label = dominant_label
                else:
                    # Pattern type doesn't match model - prefer more specific label
                    # e.g., 'date_of_birth' is more specific than 'date'
                    final_label = (
                        dominant_label
                        if is_more_specific(dominant_label, unit_type)
                        else unit_type
                    )

            source_labels = sorted(
                {
                    str(unit_type),
                    *(str(entity["entity_type"]) for entity in overlapping_entities),
                }
            )
            mixed_label_union = (
                len({normalize_label(label) for label in source_labels}) > 1
            )

            # Combine model confidence with pattern confidence.
            # When pattern validation failed, heavily discount the pattern
            # contribution to avoid high-confidence invalid entities.
            if unit_validated:
                # Normal blend: 60% model, 40% pattern
                final_confidence = (0.6 * model_avg_confidence) + (0.4 * unit_score)
            else:
                # Unvalidated: 90% model, 10% pattern
                final_confidence = (0.9 * model_avg_confidence) + (0.1 * unit_score)

            # Semantic patterns may expand a fragmented model span, but they
            # must never shrink away any text the model already classified as
            # PII. Use the union of the regex unit and every overlapping model
            # entity so a narrower semantic submatch (for example a postcode
            # inside an MRN) cannot create a raw-identifier leak.
            merged_start = min(
                unit_start,
                *(int(entity["start"]) for entity in overlapping_entities),
            )
            merged_end = max(
                unit_end,
                *(int(entity["end"]) for entity in overlapping_entities),
            )

            # Create merged entity
            merged.append(
                {
                    "entity_type": final_label,
                    "score": final_confidence,
                    "start": merged_start,
                    "end": merged_end,
                    "word": text[merged_start:merged_end],
                    "merged_from": len(overlapping),
                    "source_labels": source_labels,
                    "mixed_label_union": mixed_label_union,
                }
            )

            # Mark entities as used
            for i, _ in overlapping:
                used_entities.add(i)
        elif allow_semantic_only_matches:
            merged.append(
                {
                    "entity_type": unit_type,
                    "score": unit_score,
                    "start": unit_start,
                    "end": unit_end,
                    "word": text[unit_start:unit_end],
                    "merged_from": 0,
                    "source_labels": [str(unit_type)],
                    "mixed_label_union": False,
                }
            )

    # Add non-overlapping entities as-is
    for i, entity in enumerate(entities):
        if i not in used_entities:
            merged.append(entity)

    coalesced = _coalesce_overlapping_merged_entities(merged, text)
    if india_clinical:
        return merge_india_code_mixed_spans(coalesced, text)
    return coalesced


def _coalesce_overlapping_merged_entities(
    entities: List[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    """Union transitive overlaps without losing any detected source text.

    Separate, non-overlapping semantic units can both overlap the same wider
    model span. After semantic expansion that produces overlapping output
    entities; simply choosing one would discard model-detected text. Collapse
    each transitive overlap cluster to its full union before downstream
    arbitration so every contributing detection remains covered.
    """
    if not entities:
        return []

    ordered = sorted(entities, key=lambda entity: (entity["start"], entity["end"]))
    clusters: List[List[Dict[str, Any]]] = []
    current = [ordered[0]]
    current_end = int(ordered[0]["end"])
    for entity in ordered[1:]:
        start = int(entity["start"])
        end = int(entity["end"])
        if start < current_end:
            current.append(entity)
            current_end = max(current_end, end)
            continue
        clusters.append(current)
        current = [entity]
        current_end = end
    clusters.append(current)

    result: List[Dict[str, Any]] = []
    for cluster in clusters:
        if len(cluster) == 1:
            result.append(cluster[0])
            continue

        start = min(int(entity["start"]) for entity in cluster)
        end = max(int(entity["end"]) for entity in cluster)
        winner = max(
            cluster,
            key=lambda entity: (
                int(entity["end"]) - int(entity["start"]),
                float(entity.get("score", 0.0)),
                str(entity.get("entity_type", "")),
            ),
        )
        source_labels = sorted(
            {
                str(label)
                for entity in cluster
                for label in entity.get(
                    "source_labels",
                    [entity.get("entity_type", "")],
                )
                if label
            }
        )
        result.append(
            {
                "entity_type": winner["entity_type"],
                "score": max(float(entity.get("score", 0.0)) for entity in cluster),
                "start": start,
                "end": end,
                "word": text[start:end],
                "merged_from": sum(
                    max(int(entity.get("merged_from", 1)), 1) for entity in cluster
                ),
                "source_labels": source_labels,
                "mixed_label_union": len(
                    {normalize_label(label) for label in source_labels}
                )
                > 1,
            }
        )

    return result


_INDIA_TRANSLITERATION_GAP_RE = re.compile(r"(?:\s*|\s*[-/|]\s*|\s*[\[(]\s*)")
_INDIA_PERSON_LABELS = frozenset(
    {"name", "person", "patient", "per", "patient_name", "full_name"}
)
_INDIA_LOCATION_LABELS = frozenset({"location", "loc", "place", "city"})


def merge_india_code_mixed_spans(
    entities: List[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    """Bridge PERSON/LOCATION fragments split only by an Indic script change.

    The rule is intentionally narrow and must be opted into by the India
    clinical route. It unions adjacent same-family spans only when one surface
    is Latin and the other is Devanagari or Telugu. This captures mixed-script
    names such as ``राहुल Sharma`` and parenthesized transliterations without
    widening ordinary English entities.
    """

    if not entities:
        return []

    from ..clinical.normalization import normalize_indian_clinical_entities

    ordered = sorted(
        (dict(entity) for entity in entities),
        key=lambda entity: (int(entity["start"]), int(entity["end"])),
    )
    merged: list[Dict[str, Any]] = []
    current = ordered[0]
    for candidate in ordered[1:]:
        if not _should_bridge_india_spans(current, candidate, text):
            merged.append(current)
            current = candidate
            continue
        current = _bridge_india_spans(current, candidate, text)
    merged.append(current)
    return normalize_indian_clinical_entities(merged)


def _should_bridge_india_spans(
    left: Dict[str, Any],
    right: Dict[str, Any],
    text: str,
) -> bool:
    left_family = _india_entity_family(str(left.get("entity_type", "")))
    right_family = _india_entity_family(str(right.get("entity_type", "")))
    if left_family is None or left_family != right_family:
        return False

    left_end = int(left["end"])
    right_start = int(right["start"])
    if right_start < left_end or right_start - left_end > 12:
        return False
    gap = text[left_end:right_start]
    if _INDIA_TRANSLITERATION_GAP_RE.fullmatch(gap) is None:
        return False

    from .script_detect import detect_script

    scripts = {
        detect_script(text[int(left["start"]) : left_end]),
        detect_script(text[right_start : int(right["end"])]),
    }
    return "Latin" in scripts and bool(scripts & {"Devanagari", "Telugu"})


def _india_entity_family(label: str) -> str | None:
    normalized = label.casefold().replace("b-", "").replace("i-", "")
    if normalized in _INDIA_PERSON_LABELS:
        return "person"
    if normalized in _INDIA_LOCATION_LABELS:
        return "location"
    return None


def _bridge_india_spans(
    left: Dict[str, Any],
    right: Dict[str, Any],
    text: str,
) -> Dict[str, Any]:
    start = int(left["start"])
    end = int(right["end"])
    gap = text[int(left["end"]) : int(right["start"])]
    if "(" in gap and text[end : end + 1] == ")":
        end += 1
    elif "[" in gap and text[end : end + 1] == "]":
        end += 1

    source_labels = sorted(
        {
            str(label)
            for entity in (left, right)
            for label in entity.get(
                "source_labels",
                [entity.get("entity_type", "")],
            )
            if label
        }
    )
    return {
        "entity_type": max(
            (left, right),
            key=lambda entity: float(entity.get("score", 0.0)),
        )["entity_type"],
        "score": max(float(left.get("score", 0.0)), float(right.get("score", 0.0))),
        "start": start,
        "end": end,
        "word": text[start:end],
        "merged_from": max(int(left.get("merged_from", 1)), 1)
        + max(int(right.get("merged_from", 1)), 1),
        "source_labels": source_labels,
        "mixed_label_union": len({normalize_label(label) for label in source_labels})
        > 1,
        "india_clinical_merge": {
            "script_boundary_bridge": True,
            "transliteration_pair": True,
        },
    }


def normalize_label(label: str) -> str:
    """Normalize entity label for comparison.

    Examples:
        >>> normalize_label('date_of_birth')
        'date'
        >>> normalize_label('phone_number')
        'phone'
        >>> normalize_label('email')
        'email'
    """
    label_lower = label.lower()

    # Normalize date variants
    if "date" in label_lower:
        return "date"

    # Normalize phone variants
    if "phone" in label_lower or "fax" in label_lower:
        return "phone"

    # Normalize address variants
    if "address" in label_lower:
        return "address"

    # Normalize ID variants
    if label_lower in ("ssn", "social_security", "social_security_number"):
        return "ssn"

    # Normalize national ID variants
    if label_lower in (
        "national_id",
        "nir",
        "insee",
        "steuer_id",
        "steuernummer",
        "codice_fiscale",
        "bsn",
        "dni",
        "nie",
        "aadhaar",
        "cpf",
        "cnpj",
        "teudat_zehut",
        "teudatzehut",
        "tz",
    ):
        return "national_id"

    # Normalize postal code variants
    if label_lower in ("postcode", "zipcode", "zip", "postal_code"):
        return "postcode"

    # Normalize medical record variants
    if label_lower in ("medical_record_number", "mrn", "medical_record"):
        return "medical_record"

    # Normalize account variants
    if label_lower in ("account_number", "account"):
        return "account"

    # Normalize payment card variants
    if label_lower in (
        "credit_debit_card",
        "credit_card",
        "debit_card",
        "payment_card",
    ):
        return "payment_card"

    return label_lower


def is_more_specific(label1: str, label2: str) -> bool:
    """Check if label1 is more specific than label2.

    Examples:
        >>> is_more_specific('date_of_birth', 'date')
        True
        >>> is_more_specific('date', 'date_of_birth')
        False
        >>> is_more_specific('first_name', 'name')
        True
    """
    label1_lower = label1.lower()
    label2_lower = label2.lower()

    # More specific if it contains the general label plus additional info
    if label2_lower in label1_lower and label1_lower != label2_lower:
        return True

    # Specific label hierarchies
    specificity_hierarchy = {
        "date": ["date_of_birth", "date_time"],
        "name": ["first_name", "last_name", "full_name"],
        "phone": ["phone_number", "fax_number", "mobile_number"],
        "address": ["street_address", "home_address", "billing_address"],
        "id": ["ssn", "medical_record_number", "account_number", "employee_id"],
        "national_id": [
            "nir",
            "insee",
            "steuer_id",
            "steuernummer",
            "codice_fiscale",
            "cpf",
            "cnpj",
            "teudat_zehut",
        ],
    }

    for general, specific_list in specificity_hierarchy.items():
        if normalize_label(label2) == general and label1_lower in [
            s.lower() for s in specific_list
        ]:
            return True

    return False
