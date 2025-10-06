"""Input validation utilities for OpenMed."""

import re
from typing import Any, Optional, List, Union


def validate_input(
    text: Any,
    min_length: int = 1,
    max_length: Optional[int] = None,
    allow_empty: bool = False
) -> str:
    """Validate and clean input text.

    Args:
        text: Input text to validate.
        min_length: Minimum allowed text length.
        max_length: Maximum allowed text length.
        allow_empty: Whether to allow empty strings.

    Returns:
        Validated and cleaned text.

    Raises:
        ValueError: If validation fails.
    """
    if text is None:
        if allow_empty:
            return ""
        raise ValueError("Input text cannot be None")

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Basic cleaning
    text = text.strip()

    # Check empty string
    if not text and not allow_empty:
        raise ValueError("Input text cannot be empty")

    # Check length constraints
    if len(text) < min_length:
        if allow_empty and len(text) == 0:
            return text
        raise ValueError(f"Input text too short. Minimum length: {min_length}")

    if max_length and len(text) > max_length:
        raise ValueError(f"Input text too long. Maximum length: {max_length}")

    # Check for suspicious content
    if _contains_suspicious_content(text):
        raise ValueError("Input text contains suspicious content")

    return text


def validate_model_name(model_name: str) -> str:
    """Validate model name format.

    Args:
        model_name: Model name to validate.

    Returns:
        Validated model name.

    Raises:
        ValueError: If model name is invalid.
    """
    if not isinstance(model_name, str):
        raise ValueError("Model name must be a string")

    model_name = model_name.strip()

    if not model_name:
        raise ValueError("Model name cannot be empty")

    # Check format (organization/model or just model)
    if "/" in model_name:
        parts = model_name.split("/")
        if len(parts) != 2:
            raise ValueError("Invalid model name format. Use 'org/model' or 'model'")

        org, model = parts
        if not org or not model:
            raise ValueError("Organization and model name cannot be empty")

        # Validate characters
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', org):
            raise ValueError("Invalid characters in organization name")
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model):
            raise ValueError("Invalid characters in model name")
    else:
        # Just model name
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            raise ValueError("Invalid characters in model name")

    return model_name


def validate_confidence_threshold(threshold: float) -> float:
    """Validate confidence threshold value.

    Args:
        threshold: Confidence threshold to validate.

    Returns:
        Validated threshold.

    Raises:
        ValueError: If threshold is invalid.
    """
    if not isinstance(threshold, (int, float)):
        raise ValueError("Confidence threshold must be a number")

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("Confidence threshold must be between 0.0 and 1.0")

    return float(threshold)


def validate_output_format(format_name: str) -> str:
    """Validate output format name.

    Args:
        format_name: Output format to validate.

    Returns:
        Validated format name.

    Raises:
        ValueError: If format is not supported.
    """
    valid_formats = ["dict", "json", "html", "csv"]

    if not isinstance(format_name, str):
        raise ValueError("Output format must be a string")

    format_name = format_name.lower().strip()

    if format_name not in valid_formats:
        raise ValueError(f"Unsupported output format. Valid formats: {valid_formats}")

    return format_name


def validate_batch_size(batch_size: int, max_batch_size: int = 100) -> int:
    """Validate batch size for processing.

    Args:
        batch_size: Batch size to validate.
        max_batch_size: Maximum allowed batch size.

    Returns:
        Validated batch size.

    Raises:
        ValueError: If batch size is invalid.
    """
    if not isinstance(batch_size, int):
        raise ValueError("Batch size must be an integer")

    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    if batch_size > max_batch_size:
        raise ValueError(f"Batch size too large. Maximum: {max_batch_size}")

    return batch_size


def _contains_suspicious_content(text: str) -> bool:
    """Check if text contains suspicious content.

    Args:
        text: Text to check.

    Returns:
        True if suspicious content is found.
    """
    # Check for extremely long repeated characters
    if re.search(r'(.)\1{100,}', text):
        return True

    # Check for excessive special characters
    special_char_ratio = len(re.findall(r'[^\w\s]', text)) / len(text) if text else 0
    if special_char_ratio > 0.5:
        return True

    # Check for binary or encoded content
    if re.search(r'[^\x00-\x7F]{50,}', text):
        return True

    return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations.

    Args:
        filename: Filename to sanitize.

    Returns:
        Sanitized filename.
    """
    if not isinstance(filename, str):
        filename = str(filename)

    # Remove path separators and dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove control characters
    filename = re.sub(r'[\x00-\x1f\x7f]', '', filename)

    # Limit length
    if len(filename) > 255:
        filename = filename[:255]

    # Ensure not empty
    if not filename.strip():
        filename = "output"

    return filename.strip()
