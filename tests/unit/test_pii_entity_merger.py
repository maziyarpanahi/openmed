"""Tests for PII entity merger with context-aware scoring."""

import pytest
from openmed.core.pii_entity_merger import (
    PIIPattern,
    validate_ssn,
    validate_luhn,
    validate_npi,
    validate_phone_us,
    find_context_words,
    find_semantic_units,
    merge_entities_with_semantic_units,
)


class TestValidators:
    """Test validation functions."""

    def test_validate_ssn_valid(self):
        """Test valid SSN."""
        assert validate_ssn("123-45-6789") == True
        assert validate_ssn("123 45 6789") == True

    def test_validate_ssn_invalid(self):
        """Test invalid SSN patterns."""
        assert validate_ssn("000-45-6789") == False  # Area code 000
        assert validate_ssn("666-45-6789") == False  # Area code 666
        assert validate_ssn("123-00-6789") == False  # Group 00
        assert validate_ssn("123-45-0000") == False  # Serial 0000
        assert validate_ssn("900-45-6789") == False  # Area code starts with 9

    def test_validate_phone_us_valid(self):
        """Test valid US phone numbers."""
        assert validate_phone_us("(555) 123-4567") == True
        assert validate_phone_us("555-123-4567") == True
        assert validate_phone_us("5551234567") == True
        assert validate_phone_us("1-555-123-4567") == True

    def test_validate_phone_us_invalid(self):
        """Test invalid US phone numbers."""
        assert validate_phone_us("(055) 123-4567") == False  # Area code starts with 0
        assert validate_phone_us("(155) 123-4567") == False  # Area code starts with 1
        assert validate_phone_us("555-023-4567") == False    # Exchange starts with 0
        assert validate_phone_us("555-123-456") == False     # Too short

    def test_validate_luhn_valid(self):
        """Test valid credit card numbers with Luhn checksum."""
        # Valid test credit card numbers
        assert validate_luhn("4532015112830366") == True  # Visa
        assert validate_luhn("6011111111111117") == True  # Discover

    def test_validate_luhn_invalid(self):
        """Test invalid credit card numbers."""
        assert validate_luhn("4532015112830367") == False  # Wrong checksum
        assert validate_luhn("1234567890123456") == False  # Wrong checksum

    def test_validate_npi_valid(self):
        """Test valid NPI numbers."""
        # Valid NPI with correct checksum
        assert validate_npi("1234567893") == True

    def test_validate_npi_invalid(self):
        """Test invalid NPI numbers."""
        assert validate_npi("1234567890") == False  # Wrong checksum
        assert validate_npi("123456789") == False   # Too short


class TestContextDetection:
    """Test context word detection."""

    def test_find_context_words_found(self):
        """Test finding context words near entity."""
        text = "Patient SSN: 123-45-6789"
        # SSN is at positions 13-24
        assert find_context_words(text, 13, 24, ['ssn', 'social security']) == True

    def test_find_context_words_not_found(self):
        """Test when context words are not present."""
        text = "Patient ID: 123-45-6789"
        assert find_context_words(text, 12, 23, ['ssn', 'social security']) == False

    def test_find_context_words_outside_window(self):
        """Test context words outside the search window."""
        text = "A" * 150 + "SSN" + "B" * 150 + "123-45-6789"
        ssn_start = 303
        ssn_end = 314
        # SSN keyword is 150 chars away, outside default 100-char window
        assert find_context_words(text, ssn_start, ssn_end, ['ssn'], context_window=100) == False
        # But found with larger window
        assert find_context_words(text, ssn_start, ssn_end, ['ssn'], context_window=200) == True

    def test_find_context_words_word_boundary(self):
        """Test that context matching respects word boundaries."""
        text = "Assignment number: 123-45-6789"
        # Should NOT match "ssn" inside "assignment"
        assert find_context_words(text, 19, 30, ['ssn']) == False


class TestSemanticUnitsWithScoring:
    """Test semantic unit detection with context-aware scoring."""

    def test_ssn_with_context(self):
        """Test SSN detection with context boosts score."""
        text = "Patient SSN: 123-45-6789"

        patterns = [
            PIIPattern(
                r'\b\d{3}-\d{2}-\d{4}\b',
                'ssn',
                priority=10,
                base_score=0.3,
                context_words=['ssn', 'social security'],
                context_boost=0.55,
                validator=validate_ssn
            )
        ]

        units = find_semantic_units(text, patterns)
        assert len(units) == 1
        start, end, entity_type, score, pattern = units[0]

        assert entity_type == 'ssn'
        assert text[start:end] == '123-45-6789'
        # Should have boosted score: 0.3 + 0.55 = 0.85
        assert score >= 0.8  # Allow small tolerance

    def test_ssn_without_context(self):
        """Test SSN detection without context has lower score."""
        text = "The number is 123-45-6789"

        patterns = [
            PIIPattern(
                r'\b\d{3}-\d{2}-\d{4}\b',
                'ssn',
                priority=10,
                base_score=0.3,
                context_words=['ssn', 'social security'],
                context_boost=0.55,
                validator=validate_ssn
            )
        ]

        units = find_semantic_units(text, patterns)
        assert len(units) == 1
        start, end, entity_type, score, pattern = units[0]

        assert entity_type == 'ssn'
        # Should have only base score: 0.3
        assert 0.25 <= score <= 0.35

    def test_ssn_invalid_checksum(self):
        """Test that invalid SSN gets reduced score."""
        text = "Patient SSN: 000-45-6789"  # Invalid area code

        patterns = [
            PIIPattern(
                r'\b\d{3}-\d{2}-\d{4}\b',
                'ssn',
                priority=10,
                base_score=0.3,
                context_words=['ssn', 'social security'],
                context_boost=0.55,
                validator=validate_ssn
            )
        ]

        units = find_semantic_units(text, patterns)
        assert len(units) == 1
        start, end, entity_type, score, pattern = units[0]

        # Failed validation, so score reduced to 30% of (base + context)
        # (0.3 + 0.55) * 0.3 = 0.255
        assert score < 0.3  # Significantly reduced

    def test_phone_vs_npi_ambiguity(self):
        """Test that 10-digit numbers prioritize by context and validation."""
        # Test NPI with context
        text_npi = "Provider NPI: 1234567893"

        patterns = [
            PIIPattern(
                r'\b\d{10}\b',
                'phone_number',
                priority=5,
                base_score=0.2,
                context_words=['phone', 'tel', 'telephone'],
                context_boost=0.5,
                validator=validate_phone_us
            ),
            PIIPattern(
                r'\b\d{10}\b',
                'npi',
                priority=6,  # Higher priority
                base_score=0.15,
                context_words=['npi', 'national provider', 'provider'],
                context_boost=0.65,
                validator=validate_npi
            ),
        ]

        units = find_semantic_units(text_npi, patterns)
        # Should detect as NPI (higher priority, has context)
        assert len(units) == 1
        assert units[0][2] == 'npi'  # entity_type
        assert units[0][3] >= 0.7  # Good score with context


class TestMergeWithContextAwareScoring:
    """Test entity merging with context-aware pattern scoring."""

    def test_merge_combines_model_and_pattern_scores(self):
        """Test that merged entities combine model and pattern confidence."""
        text = "Patient SSN: 123-45-6789"

        # Model predictions (fragmented)
        entities = [
            {'entity_type': 'ssn', 'score': 0.9, 'start': 13, 'end': 16, 'word': '123'},
            {'entity_type': 'ssn', 'score': 0.85, 'start': 17, 'end': 19, 'word': '45'},
            {'entity_type': 'ssn', 'score': 0.88, 'start': 20, 'end': 24, 'word': '6789'},
        ]

        patterns = [
            PIIPattern(
                r'\b\d{3}-\d{2}-\d{4}\b',
                'ssn',
                priority=10,
                base_score=0.3,
                context_words=['ssn', 'social security'],
                context_boost=0.55,
                validator=validate_ssn
            )
        ]

        merged = merge_entities_with_semantic_units(entities, text, patterns=patterns)

        assert len(merged) == 1
        assert merged[0]['entity_type'] == 'ssn'
        assert merged[0]['word'] == '123-45-6789'
        assert merged[0]['start'] == 13
        assert merged[0]['end'] == 24

        # Score should be combination of model (~0.877) and pattern (0.85 with context)
        # final = 0.6 * 0.877 + 0.4 * 0.85 = 0.526 + 0.34 = 0.866
        assert 0.8 <= merged[0]['score'] <= 0.95

    def test_merge_with_default_patterns(self):
        """Test merging with default PII_PATTERNS from module."""
        text = "DOB: 01/15/1970, Email: john@example.com"

        # Model predictions
        entities = [
            {'entity_type': 'date', 'score': 0.8, 'start': 5, 'end': 7, 'word': '01'},
            {'entity_type': 'date', 'score': 0.75, 'start': 7, 'end': 15, 'word': '/15/1970'},
            {'entity_type': 'email', 'score': 0.95, 'start': 24, 'end': 40, 'word': 'john@example.com'},
        ]

        # Use default patterns (PII_PATTERNS)
        merged = merge_entities_with_semantic_units(entities, text, use_semantic_patterns=True)

        # Should have merged the date fragments
        date_entities = [e for e in merged if e['entity_type'] == 'date']
        email_entities = [e for e in merged if e['entity_type'] == 'email']

        assert len(date_entities) == 1
        assert date_entities[0]['word'] == '01/15/1970'

        assert len(email_entities) == 1
        assert email_entities[0]['word'] == 'john@example.com'


class TestMultilingualPatterns:
    """Test multilingual pattern matching and label normalization."""

    def test_normalize_label_national_id_variants(self):
        """Test normalize_label handles national ID types."""
        from openmed.core.pii_entity_merger import normalize_label

        assert normalize_label('national_id') == 'national_id'
        assert normalize_label('nir') == 'national_id'
        assert normalize_label('insee') == 'national_id'
        assert normalize_label('steuer_id') == 'national_id'
        assert normalize_label('steuernummer') == 'national_id'
        assert normalize_label('codice_fiscale') == 'national_id'

    def test_normalize_label_postcode_variants(self):
        """Test normalize_label handles postcode variants."""
        from openmed.core.pii_entity_merger import normalize_label

        assert normalize_label('postcode') == 'postcode'
        assert normalize_label('zipcode') == 'postcode'
        assert normalize_label('zip') == 'postcode'
        assert normalize_label('postal_code') == 'postcode'

    def test_is_more_specific_national_id(self):
        """Test is_more_specific with national_id hierarchy."""
        from openmed.core.pii_entity_merger import is_more_specific

        assert is_more_specific('nir', 'national_id') is True
        assert is_more_specific('steuer_id', 'national_id') is True
        assert is_more_specific('codice_fiscale', 'national_id') is True
        assert is_more_specific('national_id', 'nir') is False

    def test_french_date_pattern_with_context(self):
        """Test French date pattern detection with context scoring."""
        from openmed.core.pii_i18n import LANGUAGE_PII_PATTERNS

        text = "Patient né le 15/01/1970"
        fr_date_patterns = [
            p for p in LANGUAGE_PII_PATTERNS["fr"]
            if p.entity_type == "date" and "\\d{1,2}/\\d{1,2}" in p.pattern
        ]

        units = find_semantic_units(text, fr_date_patterns)
        assert len(units) >= 1
        # Should detect "15/01/1970"
        matched_text = text[units[0][0]:units[0][1]]
        assert "15/01/1970" in matched_text

    def test_german_date_pattern_dot_format(self):
        """Test German DD.MM.YYYY date pattern."""
        from openmed.core.pii_i18n import LANGUAGE_PII_PATTERNS

        text = "Geburtsdatum: 15.01.1970"
        de_date_patterns = [
            p for p in LANGUAGE_PII_PATTERNS["de"]
            if p.entity_type == "date"
        ]

        units = find_semantic_units(text, de_date_patterns)
        assert len(units) >= 1
        matched_text = text[units[0][0]:units[0][1]]
        assert "15.01.1970" in matched_text

    def test_italian_codice_fiscale_pattern(self):
        """Test Italian Codice Fiscale pattern detection."""
        from openmed.core.pii_i18n import LANGUAGE_PII_PATTERNS

        text = "Codice fiscale: RSSMRA85M01H501Z"
        it_patterns = [
            p for p in LANGUAGE_PII_PATTERNS["it"]
            if p.entity_type == "national_id"
        ]

        units = find_semantic_units(text, it_patterns)
        assert len(units) >= 1
        matched_text = text[units[0][0]:units[0][1]]
        assert matched_text == "RSSMRA85M01H501Z"

    def test_merge_with_french_patterns(self):
        """Test entity merging with French language patterns."""
        from openmed.core.pii_i18n import get_patterns_for_language

        text = "Né le 15/01/1970, email: patient@exemple.fr"
        entities = [
            {'entity_type': 'date', 'score': 0.8, 'start': 6, 'end': 8, 'word': '15'},
            {'entity_type': 'date', 'score': 0.75, 'start': 8, 'end': 16, 'word': '/01/1970'},
            {'entity_type': 'email', 'score': 0.95, 'start': 25, 'end': 43, 'word': 'patient@exemple.fr'},
        ]

        fr_patterns = get_patterns_for_language("fr")
        merged = merge_entities_with_semantic_units(
            entities, text, patterns=fr_patterns, use_semantic_patterns=True,
        )

        date_entities = [e for e in merged if 'date' in e['entity_type'].lower()]
        email_entities = [e for e in merged if e['entity_type'] == 'email']

        assert len(date_entities) == 1
        assert date_entities[0]['word'] == '15/01/1970'
        assert len(email_entities) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
