"""Unit tests for core functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from openmed.core.config import OpenMedConfig, get_config, set_config
from openmed.core.models import ModelLoader, load_model


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
