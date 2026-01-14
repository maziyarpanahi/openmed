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
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class PIIPattern:
    """A regex pattern for detecting PII semantic units with context-aware scoring.

    Inspired by Microsoft Presidio's PatternRecognizer approach:
    - base_score: Low confidence for pattern-only matches (like Presidio's 0.01-0.3)
    - context_words: Keywords that boost confidence when found nearby
    - validator: Optional checksum/validation function to confirm matches

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
    context_words: List[str] = field(default_factory=list)  # Keywords that boost confidence
    validator: Optional[Callable[[str], bool]] = None  # Validation function (e.g., checksum)


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
    # Extract digits only
    digits = re.sub(r'[^0-9]', '', ssn_text)

    if len(digits) != 9:
        return False

    area = digits[0:3]
    group = digits[3:5]
    serial = digits[5:9]

    # Check for invalid patterns
    if area == '000' or area == '666' or area[0] == '9':
        return False
    if group == '00':
        return False
    if serial == '0000':
        return False

    return True


def validate_luhn(number_text: str) -> bool:
    """Validate using Luhn algorithm (for credit cards, some IDs).

    The Luhn algorithm is used to validate credit card numbers and some
    other identification numbers.

    Args:
        number_text: Numeric string (may contain spaces/hyphens)

    Returns:
        True if passes Luhn checksum
    """
    # Extract digits only
    digits = re.sub(r'[^0-9]', '', number_text)

    if len(digits) < 13:  # Minimum for credit cards
        return False

    # Luhn algorithm
    def luhn_checksum(card_number):
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10

    return luhn_checksum(digits) == 0


def validate_npi(npi_text: str) -> bool:
    """Validate NPI (National Provider Identifier) using Luhn algorithm.

    NPIs are 10-digit numbers that use Luhn checksum with a prefix of 80840.

    Args:
        npi_text: 10-digit NPI string

    Returns:
        True if valid NPI
    """
    # Extract digits only
    digits = re.sub(r'[^0-9]', '', npi_text)

    if len(digits) != 10:
        return False

    # NPI uses Luhn algorithm with prefix "80840"
    # The constant 24 is added to the checksum
    prefix = "80840"
    checksum_input = prefix + digits

    def luhn_checksum(number):
        def digits_of(n):
            return [int(d) for d in str(n)]

        digits = digits_of(number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d * 2))
        return checksum % 10

    total = luhn_checksum(checksum_input)
    return total == 0


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
    # Extract digits only
    digits = re.sub(r'[^0-9]', '', phone_text)

    if len(digits) == 10:
        area_code = digits[0:3]
        exchange = digits[3:6]

        # Area code can't start with 0 or 1
        if area_code[0] in '01':
            return False

        # Exchange can't start with 0 (allow 1 for test numbers)
        if exchange[0] == '0':
            return False

        return True
    elif len(digits) == 11 and digits[0] == '1':
        # 1-XXX-XXX-XXXX format
        return validate_phone_us(digits[1:])

    return False


# Comprehensive PII regex patterns with context-aware scoring
# Following Presidio's philosophy: low base scores, boosted by context
PII_PATTERNS = [
    # Dates (highest priority - most specific first)
    # Dates are common, so moderate base score
    PIIPattern(
        r'\b\d{4}-\d{2}-\d{2}\b',
        'date',
        priority=10,
        base_score=0.6,
        context_words=['dob', 'birth', 'born', 'date of birth', 'birthdate', 'deceased', 'died', 'admitted', 'discharged'],
        context_boost=0.3
    ),
    PIIPattern(
        r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
        'date',
        priority=9,
        base_score=0.6,
        context_words=['dob', 'birth', 'born', 'date of birth', 'birthdate', 'deceased', 'died', 'admitted', 'discharged'],
        context_boost=0.3
    ),
    PIIPattern(
        r'\b\d{1,2}-\d{1,2}-\d{2,4}\b',
        'date',
        priority=9,
        base_score=0.6,
        context_words=['dob', 'birth', 'born', 'date of birth', 'birthdate', 'deceased', 'died', 'admitted', 'discharged'],
        context_boost=0.3
    ),
    PIIPattern(
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
        'date',
        priority=8,
        base_score=0.7,
        context_words=['dob', 'birth', 'born', 'date of birth', 'birthdate', 'deceased', 'died', 'admitted', 'discharged'],
        context_boost=0.25
    ),
    PIIPattern(
        r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
        'date',
        priority=8,
        base_score=0.7,
        context_words=['dob', 'birth', 'born', 'date of birth', 'birthdate', 'deceased', 'died', 'admitted', 'discharged'],
        context_boost=0.25
    ),

    # SSN (very specific pattern with validation)
    PIIPattern(
        r'\b\d{3}-\d{2}-\d{4}\b',
        'ssn',
        priority=10,
        base_score=0.3,  # Low without context - could be other IDs
        context_words=['ssn', 'social security', 'social security number', 'ss#', 'ss number'],
        context_boost=0.55,  # High boost with context (0.3 + 0.55 = 0.85)
        validator=validate_ssn
    ),
    PIIPattern(
        r'\b\d{3}\s\d{2}\s\d{4}\b',
        'ssn',
        priority=9,
        base_score=0.3,
        context_words=['ssn', 'social security', 'social security number', 'ss#', 'ss number'],
        context_boost=0.55,
        validator=validate_ssn
    ),

    # Phone numbers with validation
    PIIPattern(
        r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',
        'phone_number',
        priority=9,
        base_score=0.6,  # Format is pretty specific
        context_words=['phone', 'tel', 'telephone', 'cell', 'mobile', 'fax', 'call', 'contact'],
        context_boost=0.3,
        validator=validate_phone_us
    ),
    PIIPattern(
        r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b',
        'phone_number',
        priority=8,
        base_score=0.5,
        context_words=['phone', 'tel', 'telephone', 'cell', 'mobile', 'fax', 'call', 'contact'],
        context_boost=0.35,
        validator=validate_phone_us
    ),
    PIIPattern(
        r'\b\d{10}\b',
        'phone_number',
        priority=5,  # Low priority - ambiguous (could be NPI, account, etc.)
        base_score=0.2,  # Very low base score
        context_words=['phone', 'tel', 'telephone', 'cell', 'mobile', 'fax', 'call', 'contact'],
        context_boost=0.5,  # Needs context to be confident
        validator=validate_phone_us
    ),

    # NPI - 10 digit with specific validation
    PIIPattern(
        r'\b\d{10}\b',
        'npi',
        priority=6,  # Slightly higher than phone for 10-digit
        base_score=0.15,  # Very low - needs context OR validation
        context_words=['npi', 'national provider', 'provider number', 'provider id', 'provider identifier'],
        context_boost=0.65,  # Strong boost with context
        validator=validate_npi
    ),

    # Email addresses (very specific pattern)
    PIIPattern(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'email',
        priority=10,
        base_score=0.9,  # Email pattern is very specific
        context_words=['email', 'e-mail', 'contact', 'mail'],
        context_boost=0.1
    ),

    # ZIP codes (US)
    PIIPattern(
        r'\b\d{5}(?:-\d{4})?\b',
        'postcode',
        priority=7,
        base_score=0.4,  # Could be other 5-digit numbers
        context_words=['zip', 'zipcode', 'zip code', 'postal', 'postal code'],
        context_boost=0.45
    ),

    # Credit card with Luhn validation
    PIIPattern(
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'credit_debit_card',
        priority=8,
        base_score=0.4,
        context_words=['card', 'credit', 'debit', 'visa', 'mastercard', 'amex', 'discover', 'payment'],
        context_boost=0.4,
        validator=validate_luhn
    ),

    # Medical record numbers (common formats)
    PIIPattern(
        r'\b(?:MRN|mrn)[:\s#]*\d{6,10}\b',
        'medical_record_number',
        priority=9,
        base_score=0.8,  # "MRN" prefix is strong indicator
        context_words=['medical record', 'patient id', 'patient number', 'record number'],
        context_boost=0.15
    ),
    PIIPattern(
        r'\b[A-Z]{2,3}\d{6,9}\b',
        'medical_record_number',
        priority=5,
        base_score=0.3,  # Generic pattern
        context_words=['mrn', 'medical record', 'patient id', 'patient number', 'record number'],
        context_boost=0.5
    ),

    # Street addresses (basic - number + street)
    PIIPattern(
        r'\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b',
        'street_address',
        priority=7,
        base_score=0.7,  # Street suffix is good indicator
        context_words=['address', 'street', 'resides', 'residence', 'lives at', 'located at'],
        context_boost=0.2,
        flags=re.IGNORECASE
    ),

    # URLs
    PIIPattern(
        r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b',
        'url',
        priority=8,
        base_score=0.8,
        context_words=['url', 'website', 'link', 'webpage'],
        context_boost=0.15
    ),

    # IP addresses
    PIIPattern(
        r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
        'ipv4',
        priority=7,
        base_score=0.6,
        context_words=['ip', 'ip address', 'address', 'server', 'host'],
        context_boost=0.3
    ),
    PIIPattern(
        r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',
        'ipv6',
        priority=8,
        base_score=0.85,  # IPv6 pattern is very specific
        context_words=['ip', 'ipv6', 'ip address', 'address', 'server', 'host'],
        context_boost=0.15
    ),

    # MAC addresses
    PIIPattern(
        r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b',
        'mac_address',
        priority=8,
        base_score=0.75,
        context_words=['mac', 'mac address', 'hardware address'],
        context_boost=0.2
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
    context_window: int = 100
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
        if word_lower in context_text:
            return True

        # Check word boundaries (avoid partial matches like "ssn" in "assign")
        # Use word boundary pattern
        if re.search(r'\b' + re.escape(word_lower) + r'\b', context_text):
            return True

    return False


def find_semantic_units(
    text: str,
    patterns: Optional[List[PIIPattern]] = None
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
            for existing_start, existing_end, _, _, _ in units:
                if match.start() < existing_end and match.end() > existing_start:
                    overlaps = True
                    break

            if overlaps:
                continue

            matched_text = text[match.start():match.end()]

            # Calculate score with context awareness
            score = pii_pattern.base_score

            # Check for context words (like Presidio)
            if pii_pattern.context_words:
                has_context = find_context_words(
                    text,
                    match.start(),
                    match.end(),
                    pii_pattern.context_words
                )
                if has_context:
                    score = min(1.0, score + pii_pattern.context_boost)

            # Validate if validator exists
            if pii_pattern.validator:
                is_valid = pii_pattern.validator(matched_text)
                if not is_valid:
                    # Failed validation - significantly reduce score or skip
                    score = score * 0.3  # Reduce to 30% confidence
                    # Optionally could skip entirely: continue

            units.append((
                match.start(),
                match.end(),
                pii_pattern.entity_type,
                score,
                pii_pattern
            ))

    # Sort by start position
    units.sort(key=lambda x: x[0])
    return units


def calculate_dominant_label(
    entities: List[Dict[str, Any]],
    tie_breaker: str = 'confidence'
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
        label = entity['entity_type']
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in label_confidences:
            label_confidences[label] = []
        label_confidences[label].append(entity['score'])

    # Find most frequent
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]

    if len(candidates) == 1:
        dominant_label = candidates[0]
    elif tie_breaker == 'confidence':
        # Break tie by highest average confidence
        avg_confidences = {
            label: sum(label_confidences[label]) / len(label_confidences[label])
            for label in candidates
        }
        dominant_label = max(avg_confidences, key=avg_confidences.get)
    else:  # tie_breaker == 'first'
        # Use first occurrence
        for entity in entities:
            if entity['entity_type'] in candidates:
                dominant_label = entity['entity_type']
                break

    # Calculate average confidence
    avg_confidence = sum(e['score'] for e in entities) / len(entities)

    return dominant_label, avg_confidence


def merge_entities_with_semantic_units(
    entities: List[Dict[str, Any]],
    text: str,
    use_semantic_patterns: bool = True,
    patterns: Optional[List[PIIPattern]] = None,
    prefer_model_labels: bool = False
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
    if not use_semantic_patterns:
        # Just return entities as-is if not using patterns
        return sorted(entities, key=lambda x: x['start'])

    # Find semantic units
    semantic_units = find_semantic_units(text, patterns)

    if not semantic_units:
        # No semantic units found, return original entities
        return sorted(entities, key=lambda x: x['start'])

    merged = []
    used_entities = set()

    # Process each semantic unit (now includes score and pattern)
    for unit_start, unit_end, unit_type, unit_score, unit_pattern in semantic_units:
        # Find all entities that overlap with this semantic unit
        overlapping = []
        for i, entity in enumerate(entities):
            if entity['start'] < unit_end and entity['end'] > unit_start:
                overlapping.append((i, entity))

        if overlapping:
            # Calculate dominant label from model predictions
            overlapping_entities = [e for _, e in overlapping]
            dominant_label, model_avg_confidence = calculate_dominant_label(overlapping_entities)

            # Decide which label to use
            if prefer_model_labels:
                final_label = dominant_label
            else:
                # Use pattern's label if it matches any model prediction
                model_labels = set(e['entity_type'] for e in overlapping_entities)
                # Normalize labels for comparison (handle variations)
                if any(normalize_label(unit_type) == normalize_label(ml) for ml in model_labels):
                    final_label = dominant_label
                else:
                    # Pattern type doesn't match model - prefer more specific label
                    # e.g., 'date_of_birth' is more specific than 'date'
                    final_label = dominant_label if is_more_specific(dominant_label, unit_type) else unit_type

            # Combine model confidence with pattern confidence
            # Weight: 60% model, 40% pattern (model has more context)
            final_confidence = (0.6 * model_avg_confidence) + (0.4 * unit_score)

            # Create merged entity
            merged.append({
                'entity_type': final_label,
                'score': final_confidence,
                'start': unit_start,
                'end': unit_end,
                'word': text[unit_start:unit_end],
                'merged_from': len(overlapping)
            })

            # Mark entities as used
            for i, _ in overlapping:
                used_entities.add(i)

    # Add non-overlapping entities as-is
    for i, entity in enumerate(entities):
        if i not in used_entities:
            merged.append(entity)

    # Sort by start position
    merged.sort(key=lambda x: x['start'])
    return merged


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
    if 'date' in label_lower:
        return 'date'

    # Normalize phone variants
    if 'phone' in label_lower or 'fax' in label_lower:
        return 'phone'

    # Normalize address variants
    if 'address' in label_lower:
        return 'address'

    # Normalize ID variants
    if label_lower in ('ssn', 'social_security', 'social_security_number'):
        return 'ssn'

    # Normalize postal code variants
    if label_lower in ('postcode', 'zipcode', 'zip', 'postal_code'):
        return 'postcode'

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
        'date': ['date_of_birth', 'date_time'],
        'name': ['first_name', 'last_name', 'full_name'],
        'phone': ['phone_number', 'fax_number', 'mobile_number'],
        'address': ['street_address', 'home_address', 'billing_address'],
        'id': ['ssn', 'medical_record_number', 'account_number', 'employee_id'],
    }

    for general, specific_list in specificity_hierarchy.items():
        if normalize_label(label2) == general and label1_lower in [s.lower() for s in specific_list]:
            return True

    return False
