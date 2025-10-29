"""Unit tests for core functionality."""

from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
from openmed.core.config import OpenMedConfig, get_config, set_config
from openmed.core.models import ModelLoader, load_model
from openmed.processing.sentences import SentenceSpan


class TestOpenMedConfig:
    """Test cases for OpenMedConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OpenMedConfig()
        assert config.default_org == "OpenMed"
        assert config.log_level == "INFO"
        assert config.timeout == 300
        assert config.cache_dir is not None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OpenMedConfig(
            default_org="TestOrg",
            log_level="DEBUG",
            timeout=60
        )
        assert config.default_org == "TestOrg"
        assert config.log_level == "DEBUG"
        assert config.timeout == 60

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "default_org": "TestOrg",
            "log_level": "DEBUG",
            "timeout": 120
        }
        config = OpenMedConfig.from_dict(config_dict)
        assert config.default_org == "TestOrg"
        assert config.log_level == "DEBUG"
        assert config.timeout == 120

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = OpenMedConfig(default_org="TestOrg")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["default_org"] == "TestOrg"

    def test_global_config_management(self, sample_config):
        """Test global configuration management."""
        set_config(sample_config)
        retrieved_config = get_config()
        assert retrieved_config.default_org == sample_config.default_org


class TestModelLoader:
    """Test cases for ModelLoader."""

    @patch('openmed.core.models.HF_AVAILABLE', True)
    def test_init_with_config(self, sample_config):
        """Test ModelLoader initialization with config."""
        loader = ModelLoader(sample_config)
        assert loader.config == sample_config

    @patch('openmed.core.models.HF_AVAILABLE', False)
    def test_init_without_transformers(self):
        """Test ModelLoader initialization without transformers."""
        with pytest.raises(ImportError, match="HuggingFace transformers is required"):
            ModelLoader()

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.get_all_models', return_value={})
    @patch('openmed.core.models.list_models')
    def test_list_available_models(self, mock_list_models, mock_get_all_models):
        """Test listing available models."""
        # Mock the model info objects
        mock_model1 = Mock()
        mock_model1.modelId = "OpenMed/model1"
        mock_model2 = Mock()
        mock_model2.modelId = "OpenMed/model2"
        mock_list_models.return_value = [mock_model1, mock_model2]

        loader = ModelLoader()
        models = loader.list_available_models()

        assert len(models) == 2
        assert "OpenMed/model1" in models
        assert "OpenMed/model2" in models
        mock_list_models.assert_called_once()

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.get_all_models', return_value={})
    @patch('openmed.core.models.list_models')
    def test_list_models_error_handling(self, mock_list_models, mock_get_all_models):
        """Test error handling when listing models fails."""
        mock_list_models.side_effect = Exception("API Error")

        loader = ModelLoader()
        models = loader.list_available_models()

        assert models == []

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.get_all_models')
    @patch('openmed.core.models.list_models')
    def test_list_available_models_offline(self, mock_list_models, mock_get_all_models):
        """Ensure registry listing works without remote fetch."""
        mock_get_all_models.return_value = {
            "disease_detection": Mock(model_id="OpenMed/model1")
        }

        loader = ModelLoader()
        models = loader.list_available_models(include_remote=False)

        assert models == ["OpenMed/model1"]
        mock_list_models.assert_not_called()

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.AutoTokenizer')
    def test_get_max_sequence_length_from_tokenizer(
        self,
        mock_auto_tokenizer,
    ):
        """Tokenizer-derived max length should be returned when available."""

        tokenizer = Mock()
        tokenizer.model_max_length = 384
        tokenizer.init_kwargs = {}
        tokenizer.config = None
        mock_auto_tokenizer.from_pretrained.return_value = tokenizer
        loader = ModelLoader()
        max_len = loader.get_max_sequence_length("test-model")

        assert max_len == 384

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.AutoTokenizer')
    @patch('openmed.core.models.AutoConfig')
    def test_get_max_sequence_length_falls_back_to_config(
        self,
        mock_auto_config,
        mock_auto_tokenizer,
        *_
    ):
        """Configuration attributes provide a fallback when tokenizer lacks data."""

        config = Mock()
        config.max_position_embeddings = 1024
        mock_auto_config.from_pretrained.return_value = config
        mock_auto_tokenizer.from_pretrained.side_effect = Exception("no tokenizer")

        loader = ModelLoader()
        max_len = loader.get_max_sequence_length("test-model")

        assert max_len == 1024
        mock_auto_config.from_pretrained.assert_called_once()


class TestGetModelMaxLengthFunction:
    """Tests for the top-level get_model_max_length helper."""

    @patch('openmed.ModelLoader')
    def test_get_model_max_length_wrapper(self, mock_loader_cls):
        instance = Mock()
        instance.get_max_sequence_length.return_value = 256
        mock_loader_cls.return_value = instance

        from openmed import get_model_max_length

        assert get_model_max_length("model") == 256
        instance.get_max_sequence_length.assert_called_once_with("model")


class TestAnalyzeTextBehaviour:
    """Behavioural tests for the public analyze_text helper."""

    @patch('openmed.processing.sentences.segment_text')
    @patch('openmed.format_predictions')
    @patch('openmed.ModelLoader')
    def test_analyze_text_attaches_max_length(
        self,
        mock_loader_cls,
        mock_format_predictions,
        mock_segment_text,
    ):
        loader = Mock()
        pipeline = Mock(return_value=[{"entity": "LABEL", "score": 0.9, "word": "foo"}])
        pipeline.tokenizer = Mock()
        loader.create_pipeline.return_value = pipeline
        loader.get_max_sequence_length.return_value = 256
        mock_loader_cls.return_value = loader
        mock_format_predictions.return_value = "ok"
        mock_segment_text.return_value = [SentenceSpan("sample text", 0, len("sample text"))]

        from openmed import analyze_text

        analyze_text("sample text", model_name="model")

        pipeline.assert_called_once()
        call_args, call_kwargs = pipeline.call_args
        assert call_args == (["sample text"],)
        assert call_kwargs == {}
        assert mock_format_predictions.called
        kwargs = mock_format_predictions.call_args.kwargs
        assert kwargs["metadata"]["max_length"] == 256
        assert kwargs["metadata"]["sentence_detection"] is True
        assert pipeline.tokenizer.model_max_length == 256

    @patch('openmed.processing.sentences.segment_text')
    @patch('openmed.format_predictions')
    @patch('openmed.ModelLoader')
    def test_analyze_text_respects_truncation_flag(
        self,
        mock_loader_cls,
        mock_format_predictions,
        mock_segment_text,
    ):
        loader = Mock()
        pipeline = Mock(return_value=[{"entity": "LABEL", "score": 0.9, "word": "foo"}])
        loader.create_pipeline.return_value = pipeline
        loader.get_max_sequence_length.return_value = 1024
        mock_loader_cls.return_value = loader
        mock_format_predictions.return_value = "ok"
        mock_segment_text.return_value = [SentenceSpan("sample text", 0, len("sample text"))]
        pipeline.tokenizer = Mock()

        from openmed import analyze_text

        analyze_text(
            "sample text",
            model_name="model",
            max_length=128,
            truncation=False,
            sentence_detection=False,
        )

        pipeline.assert_called_once()
        call_args, call_kwargs = pipeline.call_args
        assert call_args == ("sample text",)
        assert call_kwargs == {}
        loader.get_max_sequence_length.assert_not_called()
        assert pipeline.tokenizer.model_max_length == 0

    @patch('openmed.processing.sentences.segment_text')
    @patch('openmed.ModelLoader')
    def test_sentence_detection_attaches_metadata(
        self,
        mock_loader_cls,
        mock_segment_text,
    ):
        loader = Mock()
        pipeline = Mock(
            return_value=[
                {
                    "entity": "COND",
                    "score": 0.9,
                    "word": "First",
                    "start": 0,
                    "end": 5,
                },
                {
                    "entity": "COND",
                    "score": 0.85,
                    "word": "Second",
                    "start": 17,
                    "end": 23,
                },
            ]
        )
        pipeline.tokenizer = Mock()
        loader.create_pipeline.return_value = pipeline
        loader.get_max_sequence_length.return_value = 384
        mock_loader_cls.return_value = loader

        text = "First sentence. Second sentence."
        mock_segment_text.return_value = [
            SentenceSpan("First sentence.", 0, 15),
            SentenceSpan("Second sentence.", 16, 32),
        ]

        from openmed import analyze_text

        result = analyze_text(
            text,
            model_name="model",
            output_format="dict",
        )

        assert len(result.entities) == 2
        first, second = result.entities
        assert first.start == 0
        assert first.metadata["sentence_index"] == 0
        assert first.metadata["sentence_text"] == "First sentence."
        assert second.metadata["sentence_index"] == 1
        assert second.metadata["sentence_text"] == "Second sentence."
        assert second.start >= second.metadata["sentence_start"]
        assert result.metadata["sentence_count"] == 2
        assert result.metadata["sentence_detection"] is True
        assert pipeline.tokenizer.model_max_length == 384

    @patch('openmed.processing.sentences.segment_text')
    @patch('openmed.format_predictions')
    @patch('openmed.ModelLoader')
    def test_sentence_detection_batches_large_inputs(
        self,
        mock_loader_cls,
        mock_format_predictions,
        mock_segment_text,
    ):
        loader = Mock()
        sentences_fixture = Path(__file__).resolve().parents[1] / "fixtures" / "long_clinical_note.txt"
        long_text = sentences_fixture.read_text().strip()
        sentence_lines = [line.strip() for line in long_text.splitlines() if line.strip()]

        segments = []
        cursor = 0
        for sentence in sentence_lines:
            start = long_text.index(sentence, cursor)
            end = start + len(sentence)
            segments.append(SentenceSpan(sentence, start, end))
            cursor = end

        mock_segment_text.return_value = segments

        pipeline = Mock(return_value=[[] for _ in segments])
        pipeline.tokenizer = Mock()
        loader.create_pipeline.return_value = pipeline
        loader.get_max_sequence_length.return_value = 512
        mock_loader_cls.return_value = loader
        mock_format_predictions.return_value = "ok"

        from openmed import analyze_text

        analyze_text(long_text, model_name="model")

        pipeline.assert_called_once()
        call_args, call_kwargs = pipeline.call_args
        assert isinstance(call_args[0], list)
        assert 0 < len(call_args[0]) < len(segments)
        assert call_kwargs == {}

        fmt_kwargs = mock_format_predictions.call_args.kwargs
        assert fmt_kwargs["metadata"]["sentence_count"] == len(segments)

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.AutoConfig')
    @patch('openmed.core.models.AutoTokenizer')
    @patch('openmed.core.models.AutoModelForTokenClassification')
    def test_load_model_success(self, mock_model_class, mock_tokenizer_class, mock_config_class):
        """Test successful model loading."""
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

        loader = ModelLoader()
        result = loader.load_model("test-model")

        assert "model" in result
        assert "tokenizer" in result
        assert "config" in result
        assert result["model"] == mock_model
        assert result["tokenizer"] == mock_tokenizer
        assert result["config"] == mock_config

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.AutoConfig')
    def test_load_model_failure(self, mock_config_class):
        """Test model loading failure."""
        mock_config_class.from_pretrained.side_effect = Exception("Model not found")

        loader = ModelLoader()
        with pytest.raises(ValueError, match="Could not load model"):
            loader.load_model("nonexistent-model")

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.pipeline')
    def test_create_pipeline(self, mock_pipeline):
        """Test pipeline creation."""
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance

        loader = ModelLoader()

        # Mock the load_model method
        with patch.object(loader, 'load_model') as mock_load:
            mock_load.return_value = {
                "model": Mock(),
                "tokenizer": Mock(),
                "config": Mock()
            }

            result = loader.create_pipeline("test-model")

            assert result == mock_pipeline_instance
            mock_pipeline.assert_called_once()

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.hf_model_info')
    def test_get_model_info_success(self, mock_model_info):
        """Test successful model info retrieval."""
        mock_info = Mock()
        mock_model_info.return_value = mock_info

        loader = ModelLoader()
        result = loader.get_model_info("test-model")

        assert result == mock_info
        mock_model_info.assert_called_once()

    @patch('openmed.core.models.HF_AVAILABLE', True)
    @patch('openmed.core.models.hf_model_info')
    def test_get_model_info_failure(self, mock_model_info):
        """Test model info retrieval failure."""
        mock_model_info.side_effect = Exception("Model not found")

        loader = ModelLoader()
        result = loader.get_model_info("nonexistent-model")

        assert result is None


class TestLoadModelFunction:
    """Test cases for the load_model convenience function."""

    @patch('openmed.core.models.ModelLoader')
    def test_load_model_function(self, mock_loader_class):
        """Test the load_model convenience function."""
        mock_loader = Mock()
        mock_result = {"model": Mock(), "tokenizer": Mock(), "config": Mock()}
        mock_loader.load_model.return_value = mock_result
        mock_loader_class.return_value = mock_loader

        result = load_model("test-model")

        assert result == mock_result
        mock_loader_class.assert_called_once()
        mock_loader.load_model.assert_called_once_with("test-model")
