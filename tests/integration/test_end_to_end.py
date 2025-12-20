"""End-to-end integration tests for OpenMed."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import openmed
from openmed import analyze_text, list_models


class TestEndToEndAnalysis:
    """Test end-to-end analysis workflows."""

    @pytest.mark.integration
    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.pipeline')
    @patch('openmed.core.models.AutoConfig')
    @patch('openmed.core.models.AutoTokenizer')
    @patch('openmed.core.models.AutoModelForTokenClassification')
    def test_analyze_text_full_pipeline(
        self,
        mock_model_class,
        mock_tokenizer_class,
        mock_config_class,
        mock_pipeline,
        sample_text
    ):
        """Test full text analysis pipeline."""
        # Setup mocks
        mock_config = Mock()
        mock_config.num_labels = 5
        mock_config.problem_type = "token_classification"
        mock_config.architectures = ["BertForTokenClassification"]
        mock_config_class.from_pretrained.return_value = mock_config

        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.config = mock_config
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock pipeline predictions
        mock_pipeline_instance = Mock()
        mock_pipeline_instance.return_value = [
            {
                "entity": "B-CONDITION",
                "score": 0.95,
                "word": "diabetes",
                "start": 17,
                "end": 25
            },
            {
                "entity": "B-CONDITION",
                "score": 0.89,
                "word": "hypertension",
                "start": 30,
                "end": 42
            }
        ]
        mock_pipeline.return_value = mock_pipeline_instance

        # Run analysis
        from openmed.core.config import OpenMedConfig

        result = analyze_text(
            sample_text,
            model_name="medical-ner",
            config=OpenMedConfig(use_medical_tokenizer=False),
        )

        # Verify result structure
        assert hasattr(result, 'text')
        assert hasattr(result, 'entities')
        assert hasattr(result, 'model_name')
        assert result.text == sample_text
        assert result.model_name == "medical-ner"
        assert len(result.entities) == 2

    @pytest.mark.integration
    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.get_all_models', return_value={})
    @patch('openmed.core.models.list_models')
    def test_list_models_integration(self, mock_list_models, mock_get_all_models):
        """Test model listing integration."""
        # Mock model info objects
        mock_model1 = Mock()
        mock_model1.modelId = "OpenMed/medical-ner"
        mock_model2 = Mock()
        mock_model2.modelId = "OpenMed/medication-extraction"
        mock_list_models.return_value = [mock_model1, mock_model2]

        # Test list_models function
        models = list_models()

        assert isinstance(models, list)
        assert len(models) == 2
        assert "OpenMed/medical-ner" in models
        assert "OpenMed/medication-extraction" in models

    @pytest.mark.integration
    def test_package_imports(self):
        """Test that all main package components can be imported."""
        # Test core imports
        from openmed import ModelLoader, load_model, OpenMedConfig
        from openmed import TextProcessor, TokenizationHelper, OutputFormatter
        from openmed import setup_logging, get_logger, validate_input
        from openmed import analyze_text, list_models

        # Verify classes can be instantiated
        config = OpenMedConfig()
        assert config is not None

        processor = TextProcessor()
        assert processor is not None

        formatter = OutputFormatter()
        assert formatter is not None

    @pytest.mark.integration
    @patch('openmed.core.models.HF_AVAILABLE', True)
    def test_configuration_flow(self):
        """Test configuration management flow."""
        from openmed.core.config import get_config, set_config, OpenMedConfig

        # Test default config
        default_config = get_config()
        assert default_config.default_org == "OpenMed"

        # Test custom config
        custom_config = OpenMedConfig(
            default_org="TestOrg",
            log_level="DEBUG"
        )
        set_config(custom_config)

        retrieved_config = get_config()
        assert retrieved_config.default_org == "TestOrg"
        assert retrieved_config.log_level == "DEBUG"

    @pytest.mark.integration
    def test_text_processing_pipeline(self, sample_long_text):
        """Test text processing pipeline."""
        from openmed.processing import preprocess_text, TextProcessor

        # Test preprocessing
        processed_text = preprocess_text(
            sample_long_text,
            normalize_whitespace=True
        )
        assert len(processed_text) <= len(sample_long_text)

        # Test advanced processing
        processor = TextProcessor(normalize_whitespace=True)
        clean_text = processor.clean_text(sample_long_text)
        sentences = processor.segment_sentences(clean_text)
        entities = processor.extract_medical_entities(clean_text)

        assert isinstance(sentences, list)
        assert len(sentences) > 0
        assert isinstance(entities, dict)
        assert "dosages" in entities

    @pytest.mark.integration
    def test_output_formatting_pipeline(self, sample_predictions, sample_text):
        """Test output formatting pipeline."""
        from openmed.processing import format_predictions

        # Test different output formats
        dict_result = format_predictions(
            sample_predictions,
            sample_text,
            output_format="dict"
        )
        assert hasattr(dict_result, 'entities')

        json_result = format_predictions(
            sample_predictions,
            sample_text,
            output_format="json"
        )
        assert isinstance(json_result, str)
        assert "diabetes" in json_result

        html_result = format_predictions(
            sample_predictions,
            sample_text,
            output_format="html"
        )
        assert isinstance(html_result, str)
        assert "<div" in html_result

    @pytest.mark.integration
    def test_error_handling_flow(self):
        """Test error handling throughout the pipeline."""
        from openmed.utils.validation import validate_input, validate_model_name

        # Test input validation errors
        with pytest.raises(ValueError):
            validate_input(None)

        with pytest.raises(ValueError):
            validate_input("")

        # Test model name validation errors
        with pytest.raises(ValueError):
            validate_model_name("")

        with pytest.raises(ValueError):
            validate_model_name("invalid@model")


class TestPackageMetadata:
    """Test package metadata and version info."""

    def test_version_available(self):
        """Test that version information is available."""
        assert hasattr(openmed, '__version__')
        assert isinstance(openmed.__version__, str)
        assert len(openmed.__version__) > 0

    def test_package_structure(self):
        """Test that main package structure is correct."""
        # Test that main modules are accessible
        assert hasattr(openmed, 'ModelLoader')
        assert hasattr(openmed, 'TextProcessor')
        assert hasattr(openmed, 'OutputFormatter')
        assert hasattr(openmed, 'analyze_text')
        assert hasattr(openmed, 'list_models')

    def test_all_exports(self):
        """Test that __all__ exports are accessible."""
        if hasattr(openmed, '__all__'):
            for export in openmed.__all__:
                assert hasattr(openmed, export), f"Export {export} not found"


@pytest.mark.slow
class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""

    @pytest.mark.integration
    def test_text_processing_performance(self, sample_long_text):
        """Test text processing performance with large text."""
        from openmed.processing import TextProcessor
        import time

        processor = TextProcessor(normalize_whitespace=True)

        start_time = time.time()
        result = processor.clean_text(sample_long_text)
        processing_time = time.time() - start_time

        # Should process reasonably quickly
        assert processing_time < 1.0  # Less than 1 second
        assert len(result) > 0

    @pytest.mark.integration
    def test_batch_processing_simulation(self):
        """Test simulated batch processing."""
        from openmed.processing import preprocess_text, format_predictions

        # Simulate multiple texts
        texts = [
            "Patient has diabetes.",
            "Blood pressure is elevated.",
            "Prescribed metformin 500mg.",
            "Follow up in 2 weeks.",
            "Lab results pending."
        ]

        # Process each text
        processed_texts = []
        for text in texts:
            processed = preprocess_text(text, normalize_whitespace=True)
            processed_texts.append(processed)

        assert len(processed_texts) == len(texts)
        assert all(len(text) > 0 for text in processed_texts)
