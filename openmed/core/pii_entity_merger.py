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
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PIIPattern:
    """A regex pattern for detecting PII semantic units."""

    pattern: str
    entity_type: str
    priority: int = 0  # Higher priority patterns checked first
    flags: int = re.IGNORECASE


# Comprehensive PII regex patterns
PII_PATTERNS = [
    # Dates (highest priority - most specific first)
    PIIPattern(r'\b\d{4}-\d{2}-\d{2}\b', 'date', priority=10),  # ISO: YYYY-MM-DD
    PIIPattern(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', 'date', priority=9),  # US: MM/DD/YYYY
    PIIPattern(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', 'date', priority=9),  # US: MM-DD-YYYY
    PIIPattern(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',
               'date', priority=8),  # Month DD, YYYY
    PIIPattern(r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',
               'date', priority=8),  # DD Month YYYY

    # SSN (very specific)
    PIIPattern(r'\b\d{3}-\d{2}-\d{4}\b', 'ssn', priority=10),
    PIIPattern(r'\b\d{3}\s\d{2}\s\d{4}\b', 'ssn', priority=9),

    # Phone numbers
    PIIPattern(r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b', 'phone_number', priority=9),  # (555) 123-4567
    PIIPattern(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', 'phone_number', priority=8),    # 555-123-4567
    PIIPattern(r'\b\d{10}\b', 'phone_number', priority=5),  # 5551234567 (lower priority - could be other ID)

    # Email addresses
    PIIPattern(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'email', priority=10),

    # ZIP codes (US)
    PIIPattern(r'\b\d{5}(?:-\d{4})?\b', 'postcode', priority=7),

    # Credit card (basic pattern)
    PIIPattern(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'credit_debit_card', priority=8),

    # Medical record numbers (common formats)
    PIIPattern(r'\b(?:MRN|mrn)[:\s#]*\d{6,10}\b', 'medical_record_number', priority=9),
    PIIPattern(r'\b[A-Z]{2,3}\d{6,9}\b', 'medical_record_number', priority=5),  # AA123456 format

    # Street addresses (basic - number + street)
    PIIPattern(r'\b\d{1,5}\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b',
               'street_address', priority=7, flags=re.IGNORECASE),

    # URLs
    PIIPattern(r'\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?\b',
               'url', priority=8),

    # IP addresses
    PIIPattern(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', 'ipv4', priority=7),
    PIIPattern(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b', 'ipv6', priority=8),

    # MAC addresses
    PIIPattern(r'\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b', 'mac_address', priority=8),
]


def find_semantic_units(text: str, patterns: Optional[List[PIIPattern]] = None) -> List[Tuple[int, int, str]]:
    """Find semantic units in text using regex patterns.

    Args:
        text: Input text to analyze
        patterns: Optional custom patterns (uses PII_PATTERNS if None)

    Returns:
        List of tuples (start, end, entity_type) sorted by start position

    Example:
        >>> text = "DOB: 01/15/1970, SSN: 123-45-6789"
        >>> units = find_semantic_units(text)
        >>> units
        [(5, 15, 'date'), (22, 33, 'ssn')]
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
            for existing_start, existing_end, _ in units:
                if match.start() < existing_end and match.end() > existing_start:
                    overlaps = True
                    break

            if not overlaps:
                units.append((match.start(), match.end(), pii_pattern.entity_type))

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

    # Process each semantic unit
    for unit_start, unit_end, unit_type in semantic_units:
        # Find all entities that overlap with this semantic unit
        overlapping = []
        for i, entity in enumerate(entities):
            if entity['start'] < unit_end and entity['end'] > unit_start:
                overlapping.append((i, entity))

        if overlapping:
            # Calculate dominant label from model predictions
            overlapping_entities = [e for _, e in overlapping]
            dominant_label, avg_confidence = calculate_dominant_label(overlapping_entities)

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

            # Create merged entity
            merged.append({
                'entity_type': final_label,
                'score': avg_confidence,
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
