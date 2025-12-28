"""Unit tests for the OpenMed TUI module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from openmed.tui.app import (
    Entity,
    ENTITY_COLORS,
    get_entity_color,
    OpenMedTUI,
    InputPanel,
    AnnotatedView,
    EntityTable,
    StatusBar,
)


class TestEntityColors:
    """Tests for entity color mapping."""

    def test_disease_colors(self):
        """Test that disease-related labels get red color."""
        assert get_entity_color("DISEASE") == "#ef4444"
        assert get_entity_color("CONDITION") == "#ef4444"
        assert get_entity_color("PROBLEM") == "#ef4444"
        assert get_entity_color("DIAGNOSIS") == "#ef4444"

    def test_drug_colors(self):
        """Test that drug-related labels get blue color."""
        assert get_entity_color("DRUG") == "#3b82f6"
        assert get_entity_color("MEDICATION") == "#3b82f6"
        assert get_entity_color("TREATMENT") == "#3b82f6"
        assert get_entity_color("CHEMICAL") == "#3b82f6"

    def test_anatomy_colors(self):
        """Test that anatomy-related labels get green color."""
        assert get_entity_color("ANATOMY") == "#22c55e"
        assert get_entity_color("BODY_PART") == "#22c55e"
        assert get_entity_color("ORGAN") == "#22c55e"

    def test_procedure_colors(self):
        """Test that procedure-related labels get purple color."""
        assert get_entity_color("PROCEDURE") == "#a855f7"
        assert get_entity_color("TEST") == "#a855f7"
        assert get_entity_color("LAB") == "#a855f7"

    def test_gene_colors(self):
        """Test that gene-related labels get amber color."""
        assert get_entity_color("GENE") == "#f59e0b"
        assert get_entity_color("PROTEIN") == "#f59e0b"
        assert get_entity_color("GENE_OR_GENE_PRODUCT") == "#f59e0b"

    def test_species_colors(self):
        """Test that species-related labels get cyan color."""
        assert get_entity_color("SPECIES") == "#06b6d4"
        assert get_entity_color("ORGANISM") == "#06b6d4"

    def test_unknown_label_returns_default(self):
        """Test that unknown labels get default gray color."""
        assert get_entity_color("UNKNOWN") == "#9ca3af"
        assert get_entity_color("FOOBAR") == "#9ca3af"

    def test_case_insensitive(self):
        """Test that color lookup is case insensitive."""
        assert get_entity_color("disease") == "#ef4444"
        assert get_entity_color("Drug") == "#3b82f6"
        assert get_entity_color("ANATOMY") == "#22c55e"


class TestEntity:
    """Tests for Entity dataclass."""

    def test_basic_creation(self):
        """Test creating an entity with basic attributes."""
        entity = Entity(
            text="leukemia",
            label="DISEASE",
            start=10,
            end=18,
            confidence=0.95,
        )
        assert entity.text == "leukemia"
        assert entity.label == "DISEASE"
        assert entity.start == 10
        assert entity.end == 18
        assert entity.confidence == 0.95

    def test_from_prediction_with_text_key(self):
        """Test creating entity from prediction dict with 'text' key."""
        pred = {
            "text": "imatinib",
            "label": "DRUG",
            "start": 5,
            "end": 13,
            "confidence": 0.88,
        }
        entity = Entity.from_prediction(pred)
        assert entity.text == "imatinib"
        assert entity.label == "DRUG"
        assert entity.start == 5
        assert entity.end == 13
        assert entity.confidence == 0.88

    def test_from_prediction_with_word_key(self):
        """Test creating entity from prediction dict with 'word' key."""
        pred = {
            "word": "diabetes",
            "entity_group": "DISEASE",
            "start": 0,
            "end": 8,
            "score": 0.92,
        }
        entity = Entity.from_prediction(pred)
        assert entity.text == "diabetes"
        assert entity.label == "DISEASE"
        assert entity.confidence == 0.92

    def test_from_prediction_missing_keys(self):
        """Test creating entity from prediction dict with missing keys."""
        pred = {}
        entity = Entity.from_prediction(pred)
        assert entity.text == ""
        assert entity.label == "UNKNOWN"
        assert entity.start == 0
        assert entity.end == 0
        assert entity.confidence == 0.0


class TestOpenMedTUI:
    """Tests for the main TUI application class."""

    def test_init_defaults(self):
        """Test TUI initialization with default values."""
        app = OpenMedTUI()
        assert app._model_name is None
        assert app._confidence_threshold == 0.5
        assert app._analyze_func is None
        assert app._entities == []
        assert app._is_analyzing is False

    def test_init_with_model(self):
        """Test TUI initialization with model name."""
        app = OpenMedTUI(model_name="disease_detection_superclinical")
        assert app._model_name == "disease_detection_superclinical"

    def test_init_with_threshold(self):
        """Test TUI initialization with custom threshold."""
        app = OpenMedTUI(confidence_threshold=0.75)
        assert app._confidence_threshold == 0.75

    def test_init_with_analyze_func(self):
        """Test TUI initialization with custom analyze function."""
        mock_func = MagicMock()
        app = OpenMedTUI(analyze_func=mock_func)
        assert app._analyze_func is mock_func

    def test_title_and_subtitle(self):
        """Test TUI has correct title and subtitle."""
        app = OpenMedTUI()
        assert app.TITLE == "OpenMed TUI"
        assert app.SUB_TITLE == "Interactive Clinical NER Workbench"

    def test_bindings_defined(self):
        """Test TUI has key bindings defined."""
        app = OpenMedTUI()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "ctrl+q" in binding_keys
        assert "ctrl+enter" in binding_keys
        assert "ctrl+l" in binding_keys
        assert "f1" in binding_keys


class TestStatusBar:
    """Tests for StatusBar widget."""

    def test_init_defaults(self):
        """Test StatusBar initialization with defaults."""
        status = StatusBar()
        assert status._model_name == "No model"
        assert status._threshold == 0.5
        assert status._inference_time is None

    def test_init_with_values(self):
        """Test StatusBar initialization with custom values."""
        status = StatusBar(
            model_name="test_model",
            threshold=0.8,
            inference_time=42.5,
        )
        assert status._model_name == "test_model"
        assert status._threshold == 0.8
        assert status._inference_time == 42.5

    def test_get_status_text_without_inference_time(self):
        """Test status text without inference time."""
        status = StatusBar(model_name="test", threshold=0.6)
        text = status._get_status_text()
        assert "Model: test" in text
        assert "Threshold: 0.60" in text
        assert "Inference:" not in text

    def test_get_status_text_with_inference_time(self):
        """Test status text with inference time."""
        status = StatusBar(model_name="test", threshold=0.6, inference_time=123.4)
        text = status._get_status_text()
        assert "Model: test" in text
        assert "Threshold: 0.60" in text
        assert "Inference: 123ms" in text


class TestRunTui:
    """Tests for run_tui function."""

    def test_run_tui_creates_app(self):
        """Test run_tui creates OpenMedTUI with correct params."""
        from openmed.tui.app import run_tui

        with patch.object(OpenMedTUI, "run") as mock_run:
            # We can't actually run the TUI in tests, so we patch the run method
            with patch("openmed.tui.app.OpenMedTUI") as MockApp:
                mock_instance = MagicMock()
                MockApp.return_value = mock_instance

                run_tui(model_name="test_model", confidence_threshold=0.7)

                MockApp.assert_called_once_with(
                    model_name="test_model",
                    confidence_threshold=0.7,
                )
                mock_instance.run.assert_called_once()


class TestCLIIntegration:
    """Tests for TUI CLI command integration."""

    def test_cli_has_tui_command(self):
        """Test CLI parser includes tui command."""
        from openmed.cli.main import build_parser

        parser = build_parser()
        # Parse with tui subcommand
        args = parser.parse_args(["tui"])
        assert args.command == "tui"

    def test_cli_tui_default_threshold(self):
        """Test CLI tui command has default threshold."""
        from openmed.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["tui"])
        assert args.confidence_threshold == 0.5

    def test_cli_tui_custom_threshold(self):
        """Test CLI tui command accepts custom threshold."""
        from openmed.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["tui", "--confidence-threshold", "0.8"])
        assert args.confidence_threshold == 0.8

    def test_cli_tui_custom_model(self):
        """Test CLI tui command accepts custom model."""
        from openmed.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["tui", "--model", "my_custom_model"])
        assert args.model == "my_custom_model"

    def test_cli_tui_default_model_is_none(self):
        """Test CLI tui command model defaults to None."""
        from openmed.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["tui"])
        assert args.model is None
