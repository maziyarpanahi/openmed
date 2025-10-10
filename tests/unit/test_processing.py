"""Unit tests for processing functionality."""

import pytest
from unittest.mock import Mock, patch
from openmed.processing.text import TextProcessor, preprocess_text, postprocess_text
from openmed.processing.tokenization import TokenizationHelper
from openmed.processing.outputs import OutputFormatter, EntityPrediction, PredictionResult, format_predictions


class TestTextProcessor:
    """Test cases for TextProcessor."""

    def test_init_default(self):
        """Test TextProcessor initialization with defaults."""
        processor = TextProcessor()
        assert not processor.lowercase
        assert not processor.remove_punctuation
        assert not processor.remove_numbers
        assert processor.normalize_whitespace

    def test_init_custom(self):
        """Test TextProcessor initialization with custom settings."""
        processor = TextProcessor(
            lowercase=True,
            remove_punctuation=True,
            remove_numbers=True,
            normalize_whitespace=False
        )
        assert processor.lowercase
        assert processor.remove_punctuation
        assert processor.remove_numbers
        assert not processor.normalize_whitespace

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        processor = TextProcessor(normalize_whitespace=True)
        text = "  Patient   has    diabetes.  "
        result = processor.clean_text(text)
        assert result == "Patient has diabetes."

    def test_clean_text_lowercase(self):
        """Test text cleaning with lowercase."""
        processor = TextProcessor(lowercase=True)
        text = "Patient Has Diabetes"
        result = processor.clean_text(text)
        assert result == "patient has diabetes"

    def test_clean_text_remove_punctuation(self):
        """Test text cleaning with punctuation removal."""
        processor = TextProcessor(remove_punctuation=True)
        text = "Patient has diabetes, hypertension!"
        result = processor.clean_text(text)
        assert "," not in result
        assert "!" not in result

    def test_clean_text_preserve_medical_abbreviations(self):
        """Test that medical abbreviations are preserved."""
        processor = TextProcessor()
        text = "Patient's BP is 120/80 mmHg and HR is 85 bpm."
        result = processor.clean_text(text)
        assert "BP" in result or "bp" in result
        assert "HR" in result or "hr" in result

    def test_segment_sentences(self):
        """Test sentence segmentation."""
        processor = TextProcessor()
        text = "Patient has diabetes. BP is normal. Follow up needed."
        sentences = processor.segment_sentences(text)
        assert len(sentences) == 3
        assert "diabetes" in sentences[0]
        assert "BP" in sentences[1]
        assert "Follow up" in sentences[2]

    def test_extract_medical_entities(self):
        """Test basic medical entity extraction."""
        processor = TextProcessor()
        text = "Patient takes metformin 500mg daily. BP: 120/80."
        entities = processor.extract_medical_entities(text)

        assert "dosages" in entities
        assert "vital_signs" in entities
        assert len(entities["dosages"]) > 0
        assert len(entities["vital_signs"]) > 0


class TestTokenizationHelper:
    """Test cases for TokenizationHelper."""

    def test_init_without_tokenizer(self):
        """Test initialization without tokenizer."""
        helper = TokenizationHelper()
        assert helper.tokenizer is None

    def test_init_with_tokenizer(self, mock_tokenizer):
        """Test initialization with tokenizer."""
        helper = TokenizationHelper(mock_tokenizer)
        assert helper.tokenizer == mock_tokenizer

    def test_tokenize_with_alignment(self, mock_tokenizer):
        """Test tokenization with alignment."""
        helper = TokenizationHelper(mock_tokenizer)
        result = helper.tokenize_with_alignment("test text")

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "tokens" in result

    def test_align_predictions_to_words(self):
        """Test aligning predictions to words."""
        helper = TokenizationHelper()
        predictions = [0.9, 0.8, 0.7]
        word_ids = [0, 1, 2]
        text = "patient has diabetes"

        result = helper.align_predictions_to_words(predictions, word_ids, text)
        assert len(result) == 3
        assert result[0][0] == "patient"
        assert result[0][1] == 0.9

    def test_create_attention_masks(self):
        """Test attention mask creation."""
        helper = TokenizationHelper()
        input_ids = [[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]]
        masks = helper.create_attention_masks(input_ids, pad_token_id=0)

        assert len(masks) == 2
        assert masks[0] == [1, 1, 1, 0, 0]
        assert masks[1] == [1, 1, 0, 0, 0]


class TestEntityPrediction:
    """Test cases for EntityPrediction."""

    def test_creation(self):
        """Test EntityPrediction creation."""
        entity = EntityPrediction(
            text="diabetes",
            label="CONDITION",
            confidence=0.95,
            start=10,
            end=18
        )
        assert entity.text == "diabetes"
        assert entity.label == "CONDITION"
        assert entity.confidence == 0.95
        assert entity.start == 10
        assert entity.end == 18

    def test_to_dict(self):
        """Test EntityPrediction to_dict method."""
        entity = EntityPrediction("diabetes", "CONDITION", 0.95)
        result = entity.to_dict()

        assert isinstance(result, dict)
        assert result["text"] == "diabetes"
        assert result["label"] == "CONDITION"
        assert result["confidence"] == 0.95


class TestOutputFormatter:
    """Test cases for OutputFormatter."""

    def test_init_default(self):
        """Test OutputFormatter initialization with defaults."""
        formatter = OutputFormatter()
        assert formatter.include_confidence
        assert formatter.confidence_threshold == 0.0
        assert not formatter.group_entities

    def test_init_custom(self):
        """Test OutputFormatter initialization with custom settings."""
        formatter = OutputFormatter(
            include_confidence=False,
            confidence_threshold=0.5,
            group_entities=True
        )
        assert not formatter.include_confidence
        assert formatter.confidence_threshold == 0.5
        assert formatter.group_entities

    def test_format_predictions(self, sample_predictions, sample_text):
        """Test prediction formatting."""
        formatter = OutputFormatter()
        result = formatter.format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model"
        )

        assert isinstance(result, PredictionResult)
        assert result.text == sample_text
        assert result.model_name == "test-model"
        assert len(result.entities) == len(sample_predictions)

    def test_format_predictions_with_threshold(self, sample_predictions, sample_text):
        """Test prediction formatting with confidence threshold."""
        formatter = OutputFormatter(confidence_threshold=0.9)
        result = formatter.format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model"
        )

        # Only predictions with confidence >= 0.9 should be included
        assert len(result.entities) == 1  # Only diabetes with 0.95 confidence
        assert result.entities[0].text == "diabetes"

    def test_to_json(self, test_helpers, sample_predictions, sample_text):
        """Test JSON output generation."""
        formatter = OutputFormatter()
        result = test_helpers.create_prediction_result(sample_text, sample_predictions)
        json_output = formatter.to_json(result)

        assert isinstance(json_output, str)
        assert "diabetes" in json_output
        assert "test-model" in json_output

    def test_to_html(self, test_helpers, sample_predictions, sample_text):
        """Test HTML output generation."""
        formatter = OutputFormatter()
        result = test_helpers.create_prediction_result(sample_text, sample_predictions)
        html_output = formatter.to_html(result)

        assert isinstance(html_output, str)
        assert "<div" in html_output
        assert "diabetes" in html_output
        assert "test-model" in html_output

    def test_to_csv_rows(self, test_helpers, sample_predictions, sample_text):
        """Test CSV rows generation."""
        formatter = OutputFormatter()
        result = test_helpers.create_prediction_result(sample_text, sample_predictions)
        csv_rows = formatter.to_csv_rows(result)

        assert isinstance(csv_rows, list)
        assert len(csv_rows) == len(sample_predictions)
        assert all(isinstance(row, dict) for row in csv_rows)
        assert "text" in csv_rows[0]
        assert "label" in csv_rows[0]


class TestFormatPredictionsFunction:
    """Test cases for the format_predictions function."""

    def test_format_predictions_dict(self, sample_predictions, sample_text):
        """Test format_predictions function with dict output."""
        result = format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model",
            output_format="dict"
        )
        assert isinstance(result, PredictionResult)

    def test_format_predictions_json(self, sample_predictions, sample_text):
        """Test format_predictions function with JSON output."""
        result = format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model",
            output_format="json"
        )
        assert isinstance(result, str)
        assert "diabetes" in result

    def test_format_predictions_html(self, sample_predictions, sample_text):
        """Test format_predictions function with HTML output."""
        result = format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model",
            output_format="html"
        )
        assert isinstance(result, str)
        assert "<div" in result

    def test_format_predictions_pass_through_metadata(self, sample_predictions, sample_text):
        """Additional kwargs like processing_time should end up in the result metadata."""
        result = format_predictions(
            sample_predictions,
            sample_text,
            model_name="test-model",
            output_format="dict",
            processing_time=0.123,
        )

        assert isinstance(result, PredictionResult)
        assert result.processing_time == 0.123

    def test_format_predictions_invalid_format(self, sample_predictions, sample_text):
        """Test format_predictions function with invalid format."""
        with pytest.raises(ValueError, match="Unsupported output format"):
            format_predictions(
                sample_predictions,
                sample_text,
                model_name="test-model",
                output_format="invalid"
            )


class TestPreprocessFunction:
    """Test cases for the preprocess_text function."""

    def test_preprocess_text_defaults(self):
        """Test preprocess_text with default settings."""
        text = "  Patient   has    diabetes.  "
        result = preprocess_text(text)
        assert result == "Patient has diabetes."

    def test_preprocess_text_lowercase(self):
        """Test preprocess_text with lowercase."""
        text = "Patient Has Diabetes"
        result = preprocess_text(text, lowercase=True)
        assert result == "patient has diabetes"

    def test_preprocess_text_remove_punctuation(self):
        """Test preprocess_text with punctuation removal."""
        text = "Patient has diabetes, hypertension!"
        result = preprocess_text(text, remove_punctuation=True)
        assert "," not in result
        assert "!" not in result


class TestPostprocessFunction:
    """Test cases for the postprocess_text function."""

    def test_postprocess_text_default(self):
        """Test postprocess_text with default settings."""
        text = "patient has diabetes"
        result = postprocess_text(text)
        assert result == "Patient has diabetes"

    def test_postprocess_text_no_capitalize(self):
        """Test postprocess_text without capitalization."""
        text = "patient has diabetes"
        result = postprocess_text(text, capitalize_first=False)
        assert result == "patient has diabetes"

    def test_postprocess_text_empty(self):
        """Test postprocess_text with empty string."""
        result = postprocess_text("")
        assert result == ""
