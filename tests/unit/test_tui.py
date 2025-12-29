"""Unit tests for the OpenMed TUI module."""

from __future__ import annotations

import pytest

# Skip all tests in this module if TUI dependencies are not installed
pytest.importorskip("rich")
pytest.importorskip("textual")

from unittest.mock import MagicMock, patch

from openmed.tui.app import (
    Entity,
    HistoryItem,
    PROFILE_PRESETS,
    get_entity_color,
    get_available_models,
    OpenMedTUI,
    StatusBar,
    ModelSwitcherScreen,
    ProfileSwitcherScreen,
    ConfigPanelScreen,
    HistoryScreen,
    ExportScreen,
    FileNavigationScreen,
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


class TestProfilePresets:
    """Tests for profile presets."""

    def test_dev_profile(self):
        """Test dev profile settings."""
        assert "dev" in PROFILE_PRESETS
        dev = PROFILE_PRESETS["dev"]
        assert dev["threshold"] == 0.3
        assert dev["group_entities"] is False
        assert dev["medical_tokenizer"] is True

    def test_prod_profile(self):
        """Test prod profile settings."""
        assert "prod" in PROFILE_PRESETS
        prod = PROFILE_PRESETS["prod"]
        assert prod["threshold"] == 0.7
        assert prod["group_entities"] is True
        assert prod["medical_tokenizer"] is True

    def test_test_profile(self):
        """Test test profile settings."""
        assert "test" in PROFILE_PRESETS
        test = PROFILE_PRESETS["test"]
        assert test["threshold"] == 0.5
        assert test["group_entities"] is False
        assert test["medical_tokenizer"] is False

    def test_fast_profile(self):
        """Test fast profile settings."""
        assert "fast" in PROFILE_PRESETS
        fast = PROFILE_PRESETS["fast"]
        assert fast["threshold"] == 0.5
        assert fast["group_entities"] is True
        assert fast["medical_tokenizer"] is False

    def test_all_profiles_have_required_keys(self):
        """Test all profiles have required keys."""
        required_keys = {"threshold", "group_entities", "medical_tokenizer"}
        for name, profile in PROFILE_PRESETS.items():
            assert required_keys <= set(profile.keys()), f"Profile {name} missing keys"


class TestGetAvailableModels:
    """Tests for get_available_models function."""

    def test_returns_list(self):
        """Test that get_available_models returns a list."""
        models = get_available_models()
        assert isinstance(models, list)

    def test_returns_non_empty(self):
        """Test that get_available_models returns non-empty list."""
        models = get_available_models()
        assert len(models) > 0

    def test_fallback_models(self):
        """Test fallback models when registry unavailable."""
        with patch.dict("sys.modules", {"openmed.core.model_registry": None}):
            # The function should return fallback models
            models = get_available_models()
            assert isinstance(models, list)


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
        assert app._group_entities is False
        assert app._use_medical_tokenizer is True
        assert app._current_profile is None

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

    def test_init_with_group_entities(self):
        """Test TUI initialization with group_entities."""
        app = OpenMedTUI(group_entities=True)
        assert app._group_entities is True

    def test_init_with_medical_tokenizer(self):
        """Test TUI initialization with use_medical_tokenizer."""
        app = OpenMedTUI(use_medical_tokenizer=False)
        assert app._use_medical_tokenizer is False

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

    def test_phase2_bindings(self):
        """Test Phase 2 key bindings are defined."""
        app = OpenMedTUI()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "f2" in binding_keys  # Model switcher
        assert "f3" in binding_keys  # Config panel
        assert "f4" in binding_keys  # Profile switcher


class TestStatusBar:
    """Tests for StatusBar widget."""

    def test_init_defaults(self):
        """Test StatusBar initialization with defaults."""
        status = StatusBar()
        assert status._model_name == "No model"
        assert status._threshold == 0.5
        assert status._inference_time is None
        assert status._profile is None
        assert status._group_entities is False
        assert status._medical_tokenizer is True

    def test_init_with_values(self):
        """Test StatusBar initialization with custom values."""
        status = StatusBar(
            model_name="test_model",
            threshold=0.8,
            inference_time=42.5,
            profile="dev",
            group_entities=True,
            medical_tokenizer=False,
        )
        assert status._model_name == "test_model"
        assert status._threshold == 0.8
        assert status._inference_time == 42.5
        assert status._profile == "dev"
        assert status._group_entities is True
        assert status._medical_tokenizer is False

    def test_get_status_text_without_inference_time(self):
        """Test status text without inference time."""
        status = StatusBar(model_name="test", threshold=0.6)
        text = status._get_status_text()
        assert "Model: test" in text
        assert "Thresh: 0.60" in text

    def test_get_status_text_with_inference_time(self):
        """Test status text with inference time."""
        status = StatusBar(model_name="test", threshold=0.6, inference_time=123.4)
        text = status._get_status_text()
        assert "Model: test" in text
        assert "123ms" in text

    def test_get_status_text_with_profile(self):
        """Test status text includes profile when set."""
        status = StatusBar(model_name="test", threshold=0.6, profile="prod")
        text = status._get_status_text()
        assert "Profile: prod" in text

    def test_get_status_text_with_grouped(self):
        """Test status text includes Grouped when enabled."""
        status = StatusBar(model_name="test", threshold=0.6, group_entities=True)
        text = status._get_status_text()
        assert "Grouped" in text

    def test_get_status_text_with_medtok(self):
        """Test status text includes MedTok when enabled."""
        status = StatusBar(model_name="test", threshold=0.6, medical_tokenizer=True)
        text = status._get_status_text()
        assert "MedTok" in text

    def test_get_status_text_without_medtok(self):
        """Test status text excludes MedTok when disabled."""
        status = StatusBar(model_name="test", threshold=0.6, medical_tokenizer=False)
        text = status._get_status_text()
        assert "MedTok" not in text


class TestModelSwitcherScreen:
    """Tests for ModelSwitcherScreen modal."""

    def test_init_with_current_model(self):
        """Test initialization with current model."""
        screen = ModelSwitcherScreen(current_model="disease_detection_superclinical")
        assert screen._current_model == "disease_detection_superclinical"

    def test_init_without_current_model(self):
        """Test initialization without current model."""
        screen = ModelSwitcherScreen()
        assert screen._current_model is None

    def test_models_list_populated(self):
        """Test that models list is populated."""
        screen = ModelSwitcherScreen()
        assert len(screen._models) > 0

    def test_bindings(self):
        """Test modal bindings."""
        screen = ModelSwitcherScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "enter" in binding_keys


class TestProfileSwitcherScreen:
    """Tests for ProfileSwitcherScreen modal."""

    def test_init_with_current_profile(self):
        """Test initialization with current profile."""
        screen = ProfileSwitcherScreen(current_profile="dev")
        assert screen._current_profile == "dev"

    def test_init_without_current_profile(self):
        """Test initialization without current profile."""
        screen = ProfileSwitcherScreen()
        assert screen._current_profile is None

    def test_profiles_list_populated(self):
        """Test that profiles list is populated from presets."""
        screen = ProfileSwitcherScreen()
        assert screen._profiles == list(PROFILE_PRESETS.keys())

    def test_bindings(self):
        """Test modal bindings."""
        screen = ProfileSwitcherScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "enter" in binding_keys


class TestConfigPanelScreen:
    """Tests for ConfigPanelScreen modal."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        screen = ConfigPanelScreen()
        assert screen._threshold == 0.5
        assert screen._group_entities is False
        assert screen._medical_tokenizer is True

    def test_init_with_values(self):
        """Test initialization with custom values."""
        screen = ConfigPanelScreen(
            threshold=0.8,
            group_entities=True,
            medical_tokenizer=False,
        )
        assert screen._threshold == 0.8
        assert screen._group_entities is True
        assert screen._medical_tokenizer is False

    def test_bindings(self):
        """Test modal bindings."""
        screen = ConfigPanelScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys


class TestRunTui:
    """Tests for run_tui function."""

    def test_run_tui_creates_app(self):
        """Test run_tui creates OpenMedTUI with correct params."""
        from openmed.tui.app import run_tui

        with patch.object(OpenMedTUI, "run"):
            with patch("openmed.tui.app.OpenMedTUI") as MockApp:
                mock_instance = MagicMock()
                MockApp.return_value = mock_instance

                run_tui(
                    model_name="test_model",
                    confidence_threshold=0.7,
                    group_entities=True,
                    use_medical_tokenizer=False,
                )

                MockApp.assert_called_once_with(
                    model_name="test_model",
                    confidence_threshold=0.7,
                    group_entities=True,
                    use_medical_tokenizer=False,
                )
                mock_instance.run.assert_called_once()


class TestCLIIntegration:
    """Tests for TUI CLI command integration."""

    def test_cli_has_tui_command(self):
        """Test CLI parser includes tui command."""
        from openmed.cli.main import build_parser

        parser = build_parser()
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


# ---------------------------------------------------------------------------
# Phase 3 Tests
# ---------------------------------------------------------------------------


class TestHistoryItem:
    """Tests for HistoryItem dataclass."""

    def test_basic_creation(self):
        """Test creating a history item."""
        from datetime import datetime

        entities = [
            Entity(text="leukemia", label="DISEASE", start=0, end=8, confidence=0.95)
        ]
        item = HistoryItem(
            id="test-1",
            text="Test text",
            entities=entities,
            model_name="test_model",
            threshold=0.5,
            inference_time=42.5,
        )
        assert item.id == "test-1"
        assert item.text == "Test text"
        assert len(item.entities) == 1
        assert item.model_name == "test_model"
        assert item.threshold == 0.5
        assert item.inference_time == 42.5
        assert isinstance(item.timestamp, datetime)

    def test_to_dict(self):
        """Test converting history item to dictionary."""
        entities = [
            Entity(text="imatinib", label="DRUG", start=0, end=8, confidence=0.88)
        ]
        item = HistoryItem(
            id="test-2",
            text="Patient on imatinib",
            entities=entities,
            model_name="disease_model",
            threshold=0.6,
            inference_time=100.0,
        )
        result = item.to_dict()

        assert result["id"] == "test-2"
        assert result["text"] == "Patient on imatinib"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["text"] == "imatinib"
        assert result["entities"][0]["label"] == "DRUG"
        assert result["model_name"] == "disease_model"
        assert result["threshold"] == 0.6
        assert result["inference_time"] == 100.0
        assert "timestamp" in result

    def test_to_dict_with_multiple_entities(self):
        """Test to_dict with multiple entities."""
        entities = [
            Entity(text="diabetes", label="DISEASE", start=0, end=8, confidence=0.9),
            Entity(text="metformin", label="DRUG", start=10, end=19, confidence=0.85),
        ]
        item = HistoryItem(
            id="test-3",
            text="diabetes metformin",
            entities=entities,
            model_name="model",
            threshold=0.5,
            inference_time=50.0,
        )
        result = item.to_dict()
        assert len(result["entities"]) == 2


class TestHistoryScreen:
    """Tests for HistoryScreen modal."""

    def test_init_with_empty_history(self):
        """Test initialization with empty history."""
        screen = HistoryScreen([])
        assert screen._history == []

    def test_init_with_history(self):
        """Test initialization with history items."""
        entities = [Entity(text="test", label="TEST", start=0, end=4, confidence=0.9)]
        item = HistoryItem(
            id="h-1",
            text="Sample text",
            entities=entities,
            model_name="model",
            threshold=0.5,
            inference_time=25.0,
        )
        screen = HistoryScreen([item])
        assert len(screen._history) == 1
        assert screen._history[0].id == "h-1"

    def test_bindings(self):
        """Test modal bindings."""
        screen = HistoryScreen([])
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "enter" in binding_keys
        assert "delete" in binding_keys


class TestExportScreen:
    """Tests for ExportScreen modal."""

    def test_init(self):
        """Test initialization."""
        entities = [
            Entity(text="cancer", label="DISEASE", start=0, end=6, confidence=0.92)
        ]
        screen = ExportScreen(
            text="Patient has cancer",
            entities=entities,
            model_name="test_model",
        )
        assert screen._text == "Patient has cancer"
        assert len(screen._entities) == 1
        assert screen._model_name == "test_model"

    def test_init_without_model(self):
        """Test initialization without model name."""
        screen = ExportScreen(text="test", entities=[])
        assert screen._text == "test"
        assert screen._entities == []
        assert screen._model_name is None

    def test_get_json_output(self):
        """Test JSON export output."""
        import json

        entities = [
            Entity(text="aspirin", label="DRUG", start=0, end=7, confidence=0.95)
        ]
        screen = ExportScreen(
            text="Take aspirin",
            entities=entities,
            model_name="pharma_model",
        )
        output = screen._get_json_output()
        data = json.loads(output)

        assert data["text"] == "Take aspirin"
        assert data["model"] == "pharma_model"
        assert len(data["entities"]) == 1
        assert data["entities"][0]["text"] == "aspirin"
        assert data["entities"][0]["label"] == "DRUG"

    def test_get_csv_output(self):
        """Test CSV export output."""
        entities = [
            Entity(text="diabetes", label="DISEASE", start=0, end=8, confidence=0.9),
            Entity(text="insulin", label="DRUG", start=10, end=17, confidence=0.85),
        ]
        screen = ExportScreen(text="test", entities=entities)
        output = screen._get_csv_output()

        lines = output.split("\n")
        assert lines[0] == "text,label,start,end,confidence"
        assert len(lines) == 3  # header + 2 entities
        assert "diabetes" in lines[1]
        assert "insulin" in lines[2]

    def test_get_csv_handles_quotes(self):
        """Test CSV export handles quotes in text."""
        entities = [
            Entity(text='test "quote"', label="TEST", start=0, end=12, confidence=0.9)
        ]
        screen = ExportScreen(text="test", entities=entities)
        output = screen._get_csv_output()
        assert '""quote""' in output  # Escaped quotes

    def test_get_export_content_json(self):
        """Test get_export_content returns JSON."""
        screen = ExportScreen(text="test", entities=[])
        content = screen.get_export_content("json")
        assert "{" in content
        assert '"text"' in content

    def test_get_export_content_csv(self):
        """Test get_export_content returns CSV."""
        screen = ExportScreen(text="test", entities=[])
        content = screen.get_export_content("csv")
        assert "text,label" in content

    def test_bindings(self):
        """Test modal bindings."""
        screen = ExportScreen(text="test", entities=[])
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys


class TestFileNavigationScreen:
    """Tests for FileNavigationScreen modal."""

    def test_init_default_path(self):
        """Test initialization with default path."""
        from pathlib import Path

        screen = FileNavigationScreen()
        assert screen._start_path == Path.cwd()
        assert screen._selected_path is None

    def test_init_custom_path(self):
        """Test initialization with custom path."""
        from pathlib import Path

        custom_path = Path("/tmp")
        screen = FileNavigationScreen(start_path=custom_path)
        assert screen._start_path == custom_path

    def test_bindings(self):
        """Test modal bindings."""
        screen = FileNavigationScreen()
        binding_keys = [b.key for b in screen.BINDINGS]
        assert "escape" in binding_keys
        assert "enter" in binding_keys


class TestOpenMedTUIPhase3:
    """Tests for Phase 3 features of OpenMedTUI."""

    def test_init_has_history(self):
        """Test TUI initialization includes history list."""
        app = OpenMedTUI()
        assert hasattr(app, "_history")
        assert app._history == []
        assert hasattr(app, "_history_counter")
        assert app._history_counter == 0

    def test_phase3_bindings(self):
        """Test Phase 3 key bindings are defined."""
        app = OpenMedTUI()
        binding_keys = [b.key for b in app.BINDINGS]
        assert "f5" in binding_keys  # History
        assert "f6" in binding_keys  # Export
        assert "ctrl+o" in binding_keys  # Open file

    def test_all_bindings_present(self):
        """Test all bindings from all phases are present."""
        app = OpenMedTUI()
        binding_keys = [b.key for b in app.BINDINGS]

        # Phase 1 bindings
        assert "ctrl+q" in binding_keys
        assert "ctrl+enter" in binding_keys
        assert "ctrl+l" in binding_keys
        assert "f1" in binding_keys

        # Phase 2 bindings
        assert "f2" in binding_keys
        assert "f3" in binding_keys
        assert "f4" in binding_keys

        # Phase 3 bindings
        assert "f5" in binding_keys
        assert "f6" in binding_keys
        assert "ctrl+o" in binding_keys
