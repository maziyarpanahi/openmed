"""Unit tests for utility functions."""

import pytest
from unittest.mock import patch, Mock
import logging
from openmed.utils.logging import setup_logging, get_logger, OpenMedLogger
from openmed.utils.validation import (
    validate_input, validate_model_name, validate_confidence_threshold,
    validate_output_format, validate_batch_size, sanitize_filename
)


class TestLogging:
    """Test cases for logging utilities."""

    def test_setup_logging_default(self):
        """Test setup_logging with default settings."""
        setup_logging()
        logger = logging.getLogger("openmed")
        assert logger.level == logging.INFO

    def test_setup_logging_custom_level(self):
        """Test setup_logging with custom level."""
        setup_logging(level="DEBUG")
        logger = logging.getLogger("openmed")
        assert logger.level == logging.DEBUG

    def test_get_logger(self):
        """Test get_logger function."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "openmed.test_module"

    def test_openmed_logger(self):
        """Test OpenMedLogger class."""
        logger = OpenMedLogger("test")
        assert hasattr(logger, 'logger')
        assert hasattr(logger, 'log_model_loading')
        assert hasattr(logger, 'log_processing')
        assert hasattr(logger, 'log_predictions')

    def test_openmed_logger_methods(self):
        """Test OpenMedLogger methods don't raise errors."""
        logger = OpenMedLogger("test")

        # These should not raise exceptions
        logger.log_model_loading("test-model", "started")
        logger.log_model_loading("test-model", "completed")
        logger.log_model_loading("test-model", "failed")
        logger.log_processing(100, 0.5)
        logger.log_predictions(5, "test-model")


class TestValidation:
    """Test cases for validation utilities."""

    def test_validate_input_valid(self):
        """Test validate_input with valid input."""
        text = "This is valid medical text."
        result = validate_input(text)
        assert result == text

    def test_validate_input_strip_whitespace(self):
        """Test validate_input strips whitespace."""
        text = "  This is valid text.  "
        result = validate_input(text)
        assert result == "This is valid text."

    def test_validate_input_none(self):
        """Test validate_input with None input."""
        with pytest.raises(ValueError, match="Input text cannot be None"):
            validate_input(None)

    def test_validate_input_empty_string(self):
        """Test validate_input with empty string."""
        with pytest.raises(ValueError, match="Input text cannot be empty"):
            validate_input("")

    def test_validate_input_empty_string_allowed(self):
        """Test validate_input with empty string when allowed."""
        result = validate_input("", allow_empty=True)
        assert result == ""

    def test_validate_input_too_short(self):
        """Test validate_input with text too short."""
        with pytest.raises(ValueError, match="Input text too short"):
            validate_input("hi", min_length=5)

    def test_validate_input_too_long(self):
        """Test validate_input with text too long."""
        text = "a" * 1000
        with pytest.raises(ValueError, match="Input text too long"):
            validate_input(text, max_length=100)

    def test_validate_input_non_string(self):
        """Test validate_input converts non-string to string."""
        result = validate_input(123)
        assert result == "123"

    def test_validate_model_name_valid(self):
        """Test validate_model_name with valid names."""
        assert validate_model_name("model-name") == "model-name"
        assert validate_model_name("org/model-name") == "org/model-name"
        assert validate_model_name("OpenMed/medical-ner") == "OpenMed/medical-ner"

    def test_validate_model_name_invalid_type(self):
        """Test validate_model_name with invalid type."""
        with pytest.raises(ValueError, match="Model name must be a string"):
            validate_model_name(123)

    def test_validate_model_name_empty(self):
        """Test validate_model_name with empty string."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            validate_model_name("")

    def test_validate_model_name_invalid_format(self):
        """Test validate_model_name with invalid format."""
        with pytest.raises(ValueError, match="Invalid model name format"):
            validate_model_name("org/model/extra")

    def test_validate_model_name_empty_parts(self):
        """Test validate_model_name with empty organization or model."""
        with pytest.raises(ValueError, match="Organization and model name cannot be empty"):
            validate_model_name("/model")
        with pytest.raises(ValueError, match="Organization and model name cannot be empty"):
            validate_model_name("org/")

    def test_validate_model_name_invalid_characters(self):
        """Test validate_model_name with invalid characters."""
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_model_name("org@invalid/model")
        with pytest.raises(ValueError, match="Invalid characters"):
            validate_model_name("org/model@invalid")

    def test_validate_confidence_threshold_valid(self):
        """Test validate_confidence_threshold with valid values."""
        assert validate_confidence_threshold(0.0) == 0.0
        assert validate_confidence_threshold(0.5) == 0.5
        assert validate_confidence_threshold(1.0) == 1.0
        assert validate_confidence_threshold(1) == 1.0  # int to float conversion

    def test_validate_confidence_threshold_invalid_type(self):
        """Test validate_confidence_threshold with invalid type."""
        with pytest.raises(ValueError, match="Confidence threshold must be a number"):
            validate_confidence_threshold("0.5")

    def test_validate_confidence_threshold_out_of_range(self):
        """Test validate_confidence_threshold with out of range values."""
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            validate_confidence_threshold(-0.1)
        with pytest.raises(ValueError, match="Confidence threshold must be between 0.0 and 1.0"):
            validate_confidence_threshold(1.1)

    def test_validate_output_format_valid(self):
        """Test validate_output_format with valid formats."""
        assert validate_output_format("json") == "json"
        assert validate_output_format("JSON") == "json"  # case insensitive
        assert validate_output_format("html") == "html"
        assert validate_output_format("csv") == "csv"
        assert validate_output_format("dict") == "dict"

    def test_validate_output_format_invalid_type(self):
        """Test validate_output_format with invalid type."""
        with pytest.raises(ValueError, match="Output format must be a string"):
            validate_output_format(123)

    def test_validate_output_format_unsupported(self):
        """Test validate_output_format with unsupported format."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            validate_output_format("xml")

    def test_validate_batch_size_valid(self):
        """Test validate_batch_size with valid values."""
        assert validate_batch_size(1) == 1
        assert validate_batch_size(10) == 10
        assert validate_batch_size(100) == 100

    def test_validate_batch_size_invalid_type(self):
        """Test validate_batch_size with invalid type."""
        with pytest.raises(ValueError, match="Batch size must be an integer"):
            validate_batch_size(10.5)
        with pytest.raises(ValueError, match="Batch size must be an integer"):
            validate_batch_size("10")

    def test_validate_batch_size_non_positive(self):
        """Test validate_batch_size with non-positive values."""
        with pytest.raises(ValueError, match="Batch size must be positive"):
            validate_batch_size(0)
        with pytest.raises(ValueError, match="Batch size must be positive"):
            validate_batch_size(-1)

    def test_validate_batch_size_too_large(self):
        """Test validate_batch_size with too large value."""
        with pytest.raises(ValueError, match="Batch size too large"):
            validate_batch_size(1000, max_batch_size=100)

    def test_sanitize_filename_valid(self):
        """Test sanitize_filename with valid filename."""
        assert sanitize_filename("model_results.json") == "model_results.json"

    def test_sanitize_filename_invalid_characters(self):
        """Test sanitize_filename removes invalid characters."""
        result = sanitize_filename("model<>results.json")
        assert "<" not in result
        assert ">" not in result
        assert result == "model__results.json"

    def test_sanitize_filename_long_name(self):
        """Test sanitize_filename truncates long names."""
        long_name = "a" * 300
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_sanitize_filename_empty(self):
        """Test sanitize_filename with empty string."""
        result = sanitize_filename("")
        assert result == "output"

    def test_sanitize_filename_non_string(self):
        """Test sanitize_filename converts non-string to string."""
        result = sanitize_filename(123)
        assert result == "123"


class TestSuspiciousContentDetection:
    """Test cases for suspicious content detection."""

    @patch('openmed.utils.validation._contains_suspicious_content')
    def test_validate_input_suspicious_content(self, mock_suspicious):
        """Test validate_input detects suspicious content."""
        mock_suspicious.return_value = True

        with pytest.raises(ValueError, match="Input text contains suspicious content"):
            validate_input("suspicious text")

    def test_suspicious_content_repeated_chars(self):
        """Test detection of repeated characters."""
        from openmed.utils.validation import _contains_suspicious_content

        # Normal text should be fine
        assert not _contains_suspicious_content("Normal medical text.")

        # Very long repeated characters should be detected
        suspicious_text = "a" * 150
        assert _contains_suspicious_content(suspicious_text)

    def test_suspicious_content_special_chars(self):
        """Test detection of excessive special characters."""
        from openmed.utils.validation import _contains_suspicious_content

        # Normal punctuation should be fine
        assert not _contains_suspicious_content("Patient has diabetes, hypertension.")

        # Excessive special characters should be detected
        suspicious_text = "!@#$%^&*()_+" * 10
        assert _contains_suspicious_content(suspicious_text)
