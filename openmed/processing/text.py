"""Text processing utilities."""

import re
import string
from typing import List, Dict, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text preprocessing and cleaning for medical text analysis."""

    def __init__(self,
                 lowercase: bool = False,
                 remove_punctuation: bool = False,
                 remove_numbers: bool = False,
                 normalize_whitespace: bool = True):
        """Initialize text processor.

        Args:
            lowercase: Whether to convert text to lowercase.
            remove_punctuation: Whether to remove punctuation.
            remove_numbers: Whether to remove numbers.
            normalize_whitespace: Whether to normalize whitespace.
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.normalize_whitespace = normalize_whitespace

        # Medical abbreviations that should be preserved
        self.medical_abbreviations = {
            "mg", "ml", "kg", "lb", "oz", "cm", "mm", "hr", "min",
            "bp", "hr", "rr", "temp", "o2", "co2", "hiv", "aids",
            "icu", "er", "or", "cbc", "ekg", "ecg", "mri", "ct",
            "x-ray", "ultrasound", "bmi", "copd", "chf", "mi",
            "stroke", "tia", "dvt", "pe", "uti", "copd"
        }

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text.

        Args:
            text: Input text to clean.

        Returns:
            Cleaned text.
        """
        if not isinstance(text, str):
            text = str(text)

        original_text = text

        # Normalize whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())

        # Handle medical abbreviations before other processing
        protected_abbrevs = {}
        if not self.remove_punctuation:
            for i, abbrev in enumerate(self.medical_abbreviations):
                placeholder = f"__ABBREV_{i}__"
                text = re.sub(
                    rf'\b{re.escape(abbrev)}\b',
                    placeholder,
                    text,
                    flags=re.IGNORECASE
                )
                protected_abbrevs[placeholder] = abbrev

        # Remove or clean numbers
        if self.remove_numbers:
            # Preserve medical measurements (e.g., "120/80", "98.6°F")
            text = re.sub(r'\b\d+(?:[./]\d+)*\b(?![°%])', ' ', text)

        # Remove punctuation
        if self.remove_punctuation:
            # Keep hyphens in compound medical terms
            text = re.sub(r'[^\w\s\-]', ' ', text)

        # Convert to lowercase
        if self.lowercase:
            text = text.lower()

        # Restore protected abbreviations
        for placeholder, abbrev in protected_abbrevs.items():
            text = text.replace(placeholder, abbrev)

        # Final whitespace normalization
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())

        logger.debug(f"Text cleaning: '{original_text[:50]}...' -> '{text[:50]}...'")
        return text

    def segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences using medical text-aware rules.

        Args:
            text: Input text to segment.

        Returns:
            List of sentences.
        """
        # Medical abbreviations that shouldn't trigger sentence breaks
        abbrev_pattern = r'\b(?:' + '|'.join(self.medical_abbreviations) + r')\.'

        # Temporarily replace medical abbreviations
        text_modified = re.sub(abbrev_pattern, lambda m: m.group().replace('.', '___DOT___'), text, flags=re.IGNORECASE)

        # Simple sentence segmentation
        sentences = re.split(r'[.!?]+\s+', text_modified)

        # Restore dots in abbreviations
        sentences = [s.replace('___DOT___', '.') for s in sentences if s.strip()]

        return sentences

    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic medical entities using regex patterns.

        Args:
            text: Input text.

        Returns:
            Dictionary of entity types and their matches.
        """
        entities = {
            "medications": [],
            "dosages": [],
            "vital_signs": [],
            "lab_values": [],
            "symptoms": [],
        }

        # Dosage patterns
        dosage_patterns = [
            r'\b\d+\s*(?:mg|ml|g|kg|mcg|units?)\b',
            r'\b\d+\.\d+\s*(?:mg|ml|g|kg|mcg|units?)\b',
        ]

        for pattern in dosage_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["dosages"].extend(matches)

        # Vital signs patterns
        vital_patterns = [
            r'\b(?:bp|blood pressure):?\s*\d+/\d+\b',
            r'\b(?:hr|heart rate):?\s*\d+\b',
            r'\b(?:temp|temperature):?\s*\d+\.?\d*\s*[°]?[fF]?\b',
            r'\b(?:rr|respiratory rate):?\s*\d+\b',
        ]

        for pattern in vital_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities["vital_signs"].extend(matches)

        # Clean up duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

        return entities


def preprocess_text(
    text: str,
    lowercase: bool = False,
    remove_punctuation: bool = False,
    remove_numbers: bool = False,
    normalize_whitespace: bool = True
) -> str:
    """Convenience function for text preprocessing.

    Args:
        text: Input text.
        lowercase: Whether to convert to lowercase.
        remove_punctuation: Whether to remove punctuation.
        remove_numbers: Whether to remove numbers.
        normalize_whitespace: Whether to normalize whitespace.

    Returns:
        Preprocessed text.
    """
    processor = TextProcessor(
        lowercase=lowercase,
        remove_punctuation=remove_punctuation,
        remove_numbers=remove_numbers,
        normalize_whitespace=normalize_whitespace
    )
    return processor.clean_text(text)


def postprocess_text(text: str, capitalize_first: bool = True) -> str:
    """Postprocess text for better readability.

    Args:
        text: Input text.
        capitalize_first: Whether to capitalize the first letter.

    Returns:
        Postprocessed text.
    """
    if not text:
        return text

    text = text.strip()

    if capitalize_first and text:
        text = text[0].upper() + text[1:]

    return text
