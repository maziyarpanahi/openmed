"""OpenMed TUI Application - Interactive clinical NER workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.style import Style
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    OptionList,
    Static,
    Switch,
    TextArea,
)
from textual.widgets.option_list import Option


# Entity type color mapping
ENTITY_COLORS: dict[str, str] = {
    # Diseases and conditions
    "DISEASE": "#ef4444",  # red
    "CONDITION": "#ef4444",
    "PROBLEM": "#ef4444",
    "DIAGNOSIS": "#ef4444",
    # Drugs and treatments
    "DRUG": "#3b82f6",  # blue
    "MEDICATION": "#3b82f6",
    "TREATMENT": "#3b82f6",
    "CHEMICAL": "#3b82f6",
    # Anatomy
    "ANATOMY": "#22c55e",  # green
    "BODY_PART": "#22c55e",
    "ORGAN": "#22c55e",
    # Procedures
    "PROCEDURE": "#a855f7",  # purple
    "TEST": "#a855f7",
    "LAB": "#a855f7",
    # Genes and proteins
    "GENE": "#f59e0b",  # amber
    "PROTEIN": "#f59e0b",
    "GENE_OR_GENE_PRODUCT": "#f59e0b",
    # Species
    "SPECIES": "#06b6d4",  # cyan
    "ORGANISM": "#06b6d4",
    # Default
    "DEFAULT": "#9ca3af",  # gray
}

# Built-in profiles
PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "dev": {"threshold": 0.3, "group_entities": False, "medical_tokenizer": True},
    "prod": {"threshold": 0.7, "group_entities": True, "medical_tokenizer": True},
    "test": {"threshold": 0.5, "group_entities": False, "medical_tokenizer": False},
    "fast": {"threshold": 0.5, "group_entities": True, "medical_tokenizer": False},
}


def get_entity_color(label: str) -> str:
    """Get color for an entity type."""
    return ENTITY_COLORS.get(label.upper(), ENTITY_COLORS["DEFAULT"])


def get_available_models() -> list[str]:
    """Get list of available models from registry."""
    try:
        from openmed.core.model_registry import MODEL_REGISTRY
        return list(MODEL_REGISTRY.keys())
    except ImportError:
        return ["disease_detection_superclinical", "pharma_detection_superclinical"]


@dataclass
class Entity:
    """Represents a detected entity."""

    text: str
    label: str
    start: int
    end: int
    confidence: float

    @classmethod
    def from_prediction(cls, pred: dict[str, Any]) -> "Entity":
        """Create Entity from prediction dict."""
        return cls(
            text=pred.get("text", pred.get("word", "")),
            label=pred.get("label", pred.get("entity_group", "UNKNOWN")),
            start=pred.get("start", 0),
            end=pred.get("end", 0),
            confidence=pred.get("confidence", pred.get("score", 0.0)),
        )


# ---------------------------------------------------------------------------
# Modal Screens
# ---------------------------------------------------------------------------


class ModelSwitcherScreen(ModalScreen[str | None]):
    """Modal screen for selecting a model."""

    CSS = """
    ModelSwitcherScreen {
        align: center middle;
    }

    #model-dialog {
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
    }

    #model-dialog > Label {
        width: 100%;
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #model-list {
        height: auto;
        max-height: 15;
        margin-bottom: 1;
    }

    #model-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #model-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, current_model: str | None = None) -> None:
        super().__init__()
        self._current_model = current_model
        self._models = get_available_models()

    def compose(self) -> ComposeResult:
        with Vertical(id="model-dialog"):
            yield Label("Select Model")
            option_list = OptionList(id="model-list")
            for model in self._models:
                marker = " [current]" if model == self._current_model else ""
                option_list.add_option(Option(f"{model}{marker}", id=model))
            yield option_list
            with Horizontal(id="model-buttons"):
                yield Button("Select", variant="primary", id="select-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        """Focus the option list and select current model."""
        option_list = self.query_one("#model-list", OptionList)
        option_list.focus()
        # Try to highlight current model
        if self._current_model and self._current_model in self._models:
            idx = self._models.index(self._current_model)
            option_list.highlighted = idx

    @on(Button.Pressed, "#select-btn")
    def on_select_pressed(self) -> None:
        self.action_select()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(OptionList.OptionSelected)
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(str(event.option.id))

    def action_select(self) -> None:
        option_list = self.query_one("#model-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option.id:
                self.dismiss(str(option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ProfileSwitcherScreen(ModalScreen[str | None]):
    """Modal screen for selecting a configuration profile."""

    CSS = """
    ProfileSwitcherScreen {
        align: center middle;
    }

    #profile-dialog {
        width: 50;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #profile-dialog > Label {
        width: 100%;
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    #profile-list {
        height: auto;
        max-height: 10;
        margin-bottom: 1;
    }

    #profile-info {
        height: 4;
        border: solid $primary-lighten-2;
        padding: 0 1;
        margin-bottom: 1;
    }

    #profile-buttons {
        width: 100%;
        height: auto;
        align: center middle;
    }

    #profile-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("enter", "select", "Select"),
    ]

    def __init__(self, current_profile: str | None = None) -> None:
        super().__init__()
        self._current_profile = current_profile
        self._profiles = list(PROFILE_PRESETS.keys())

    def compose(self) -> ComposeResult:
        with Vertical(id="profile-dialog"):
            yield Label("Select Profile")
            option_list = OptionList(id="profile-list")
            for profile in self._profiles:
                marker = " [active]" if profile == self._current_profile else ""
                option_list.add_option(Option(f"{profile}{marker}", id=profile))
            yield option_list
            yield Static("", id="profile-info")
            with Horizontal(id="profile-buttons"):
                yield Button("Apply", variant="primary", id="apply-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def on_mount(self) -> None:
        option_list = self.query_one("#profile-list", OptionList)
        option_list.focus()
        if self._current_profile and self._current_profile in self._profiles:
            idx = self._profiles.index(self._current_profile)
            option_list.highlighted = idx
        self._update_info()

    @on(OptionList.OptionHighlighted)
    def on_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._update_info()

    def _update_info(self) -> None:
        option_list = self.query_one("#profile-list", OptionList)
        info = self.query_one("#profile-info", Static)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option.id and option.id in PROFILE_PRESETS:
                settings = PROFILE_PRESETS[str(option.id)]
                info_text = (
                    f"Threshold: {settings['threshold']:.1f}  "
                    f"Grouped: {'Yes' if settings['group_entities'] else 'No'}  "
                    f"MedTok: {'Yes' if settings['medical_tokenizer'] else 'No'}"
                )
                info.update(info_text)

    @on(Button.Pressed, "#apply-btn")
    def on_apply_pressed(self) -> None:
        self.action_select()

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    @on(OptionList.OptionSelected)
    def on_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id:
            self.dismiss(str(event.option.id))

    def action_select(self) -> None:
        option_list = self.query_one("#profile-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            if option.id:
                self.dismiss(str(option.id))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfigPanelScreen(ModalScreen[dict[str, Any] | None]):
    """Modal screen for adjusting configuration."""

    CSS = """
    ConfigPanelScreen {
        align: center middle;
    }

    #config-dialog {
        width: 55;
        height: auto;
        border: thick $secondary;
        background: $surface;
        padding: 1 2;
    }

    #config-dialog > Label.title {
        width: 100%;
        text-align: center;
        text-style: bold;
        padding-bottom: 1;
    }

    .config-row {
        width: 100%;
        height: 3;
        padding: 0 1;
    }

    .config-row > Label {
        width: 25;
    }

    .config-row > Static {
        width: 1fr;
    }

    #threshold-display {
        color: $success;
        text-style: bold;
    }

    #threshold-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #threshold-buttons Button {
        min-width: 5;
        margin: 0 1;
    }

    .switch-row {
        width: 100%;
        height: 3;
        align: left middle;
        padding: 0 1;
    }

    .switch-row > Label {
        width: 25;
    }

    #config-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding-top: 1;
    }

    #config-buttons Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(
        self,
        threshold: float = 0.5,
        group_entities: bool = False,
        medical_tokenizer: bool = True,
    ) -> None:
        super().__init__()
        self._threshold = threshold
        self._group_entities = group_entities
        self._medical_tokenizer = medical_tokenizer

    def compose(self) -> ComposeResult:
        with Vertical(id="config-dialog"):
            yield Label("Configuration", classes="title")

            # Threshold control
            with Horizontal(classes="config-row"):
                yield Label("Confidence Threshold:")
                yield Static(f"{self._threshold:.2f}", id="threshold-display")

            with Horizontal(id="threshold-buttons"):
                yield Button("−0.1", id="thresh-down-big")
                yield Button("−", id="thresh-down")
                yield Button("+", id="thresh-up")
                yield Button("+0.1", id="thresh-up-big")

            # Toggle switches
            with Horizontal(classes="switch-row"):
                yield Label("Group Entities:")
                yield Switch(value=self._group_entities, id="group-switch")

            with Horizontal(classes="switch-row"):
                yield Label("Medical Tokenizer:")
                yield Switch(value=self._medical_tokenizer, id="medtok-switch")

            # Action buttons
            with Horizontal(id="config-buttons"):
                yield Button("Apply", variant="primary", id="apply-btn")
                yield Button("Cancel", variant="default", id="cancel-btn")

    def _update_threshold_display(self) -> None:
        self.query_one("#threshold-display", Static).update(f"{self._threshold:.2f}")

    @on(Button.Pressed, "#thresh-down-big")
    def on_thresh_down_big(self) -> None:
        self._threshold = max(0.0, self._threshold - 0.1)
        self._update_threshold_display()

    @on(Button.Pressed, "#thresh-down")
    def on_thresh_down(self) -> None:
        self._threshold = max(0.0, self._threshold - 0.05)
        self._update_threshold_display()

    @on(Button.Pressed, "#thresh-up")
    def on_thresh_up(self) -> None:
        self._threshold = min(1.0, self._threshold + 0.05)
        self._update_threshold_display()

    @on(Button.Pressed, "#thresh-up-big")
    def on_thresh_up_big(self) -> None:
        self._threshold = min(1.0, self._threshold + 0.1)
        self._update_threshold_display()

    @on(Button.Pressed, "#apply-btn")
    def on_apply_pressed(self) -> None:
        result = {
            "threshold": self._threshold,
            "group_entities": self.query_one("#group-switch", Switch).value,
            "medical_tokenizer": self.query_one("#medtok-switch", Switch).value,
        }
        self.dismiss(result)

    @on(Button.Pressed, "#cancel-btn")
    def on_cancel_pressed(self) -> None:
        self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------
# Main Widgets
# ---------------------------------------------------------------------------


class InputPanel(Static):
    """Panel for text input."""

    DEFAULT_CSS = """
    InputPanel {
        height: auto;
        min-height: 5;
        max-height: 12;
        border: solid $primary;
        padding: 0 1;
    }

    InputPanel > Label {
        color: $text-muted;
        padding: 0 0 0 0;
    }

    InputPanel > TextArea {
        height: auto;
        min-height: 3;
        max-height: 10;
        border: none;
        padding: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Input (Ctrl+Enter to analyze)")
        yield TextArea(id="input-text")

    def get_text(self) -> str:
        """Get the current input text."""
        return self.query_one("#input-text", TextArea).text

    def set_text(self, text: str) -> None:
        """Set the input text."""
        self.query_one("#input-text", TextArea).text = text


class AnnotatedView(Static):
    """Panel showing text with highlighted entities."""

    DEFAULT_CSS = """
    AnnotatedView {
        height: auto;
        min-height: 5;
        max-height: 15;
        border: solid $secondary;
        padding: 1;
        overflow-y: auto;
    }

    AnnotatedView > Label {
        color: $text-muted;
        padding: 0 0 1 0;
    }

    AnnotatedView > #annotated-text {
        height: auto;
        min-height: 3;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Annotated")
        yield Static("", id="annotated-text")

    def update_annotated(self, text: str, entities: list[Entity]) -> None:
        """Update the annotated text view with highlighted entities."""
        if not text:
            self.query_one("#annotated-text", Static).update("")
            return

        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda e: e.start)

        # Build rich text with highlighting
        rich_text = Text()
        last_end = 0

        for entity in sorted_entities:
            # Add text before this entity
            if entity.start > last_end:
                rich_text.append(text[last_end : entity.start])

            # Add highlighted entity
            color = get_entity_color(entity.label)
            style = Style(color=color, bold=True)
            rich_text.append(f"[{entity.text}]", style=style)

            last_end = entity.end

        # Add remaining text
        if last_end < len(text):
            rich_text.append(text[last_end:])

        self.query_one("#annotated-text", Static).update(rich_text)


class EntityTable(Static):
    """Table displaying detected entities."""

    DEFAULT_CSS = """
    EntityTable {
        height: 1fr;
        min-height: 8;
        border: solid $accent;
        padding: 0;
    }

    EntityTable > Label {
        color: $text-muted;
        padding: 0 1;
    }

    EntityTable > DataTable {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Entities")
        table = DataTable(id="entity-table")
        table.cursor_type = "row"
        table.zebra_stripes = True
        yield table

    def on_mount(self) -> None:
        """Set up the table columns."""
        table = self.query_one("#entity-table", DataTable)
        table.add_column("Label", width=15, key="label")
        table.add_column("Entity", width=35, key="entity")
        table.add_column("Confidence", width=25, key="confidence")

    def update_entities(self, entities: list[Entity]) -> None:
        """Update the entity table."""
        table = self.query_one("#entity-table", DataTable)
        table.clear()

        # Update header label
        self.query_one("Label", Label).update(f"Entities ({len(entities)})")

        # Sort by confidence descending
        sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)

        for entity in sorted_entities:
            color = get_entity_color(entity.label)

            # Create styled label
            label_text = Text(entity.label)
            label_text.stylize(Style(color=color, bold=True))

            # Create confidence bar
            bar_width = 15
            filled = int(entity.confidence * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            confidence_text = Text(f"{bar} {entity.confidence:.2f}")
            confidence_text.stylize(Style(color=color))

            table.add_row(label_text, entity.text, confidence_text)


class StatusBar(Static):
    """Status bar showing current configuration."""

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        dock: bottom;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        model_name: str = "No model",
        threshold: float = 0.5,
        inference_time: float | None = None,
        profile: str | None = None,
        group_entities: bool = False,
        medical_tokenizer: bool = True,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._threshold = threshold
        self._inference_time = inference_time
        self._profile = profile
        self._group_entities = group_entities
        self._medical_tokenizer = medical_tokenizer

    def compose(self) -> ComposeResult:
        yield Label(self._get_status_text(), id="status-label")

    def _get_status_text(self) -> str:
        parts = [f"Model: {self._model_name}"]
        if self._profile:
            parts.append(f"Profile: {self._profile}")
        parts.append(f"Thresh: {self._threshold:.2f}")
        if self._group_entities:
            parts.append("Grouped")
        if self._medical_tokenizer:
            parts.append("MedTok")
        if self._inference_time is not None:
            parts.append(f"{self._inference_time:.0f}ms")
        return " │ ".join(parts)

    def update_status(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        inference_time: float | None = None,
        profile: str | None = None,
        group_entities: bool | None = None,
        medical_tokenizer: bool | None = None,
    ) -> None:
        """Update status bar values."""
        if model_name is not None:
            self._model_name = model_name
        if threshold is not None:
            self._threshold = threshold
        if inference_time is not None:
            self._inference_time = inference_time
        if profile is not None:
            self._profile = profile
        if group_entities is not None:
            self._group_entities = group_entities
        if medical_tokenizer is not None:
            self._medical_tokenizer = medical_tokenizer
        self.query_one("#status-label", Label).update(self._get_status_text())

    def clear_profile(self) -> None:
        """Clear the profile indicator."""
        self._profile = None
        self.query_one("#status-label", Label).update(self._get_status_text())


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------


class OpenMedTUI(App):
    """OpenMed Terminal User Interface for interactive clinical NER analysis."""

    TITLE = "OpenMed TUI"
    SUB_TITLE = "Interactive Clinical NER Workbench"

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        height: 1fr;
        padding: 1;
    }

    #left-panel {
        width: 1fr;
        height: 1fr;
    }

    #loading-indicator {
        display: none;
        height: 1;
        content-align: center middle;
        color: $warning;
    }

    #loading-indicator.visible {
        display: block;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+enter", "analyze", "Analyze", show=True),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("f1", "help", "Help"),
        Binding("f2", "switch_model", "Model"),
        Binding("f3", "config_panel", "Config"),
        Binding("f4", "switch_profile", "Profile"),
    ]

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float = 0.5,
        analyze_func: Callable[..., Any] | None = None,
        group_entities: bool = False,
        use_medical_tokenizer: bool = True,
    ) -> None:
        """Initialize the TUI.

        Args:
            model_name: Model to use for analysis (optional, loads default if None).
            confidence_threshold: Minimum confidence threshold for entities.
            analyze_func: Custom analysis function (defaults to openmed.analyze_text).
            group_entities: Whether to group adjacent entities.
            use_medical_tokenizer: Whether to use medical tokenizer.
        """
        super().__init__()
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._analyze_func = analyze_func
        self._group_entities = group_entities
        self._use_medical_tokenizer = use_medical_tokenizer
        self._current_profile: str | None = None
        self._entities: list[Entity] = []
        self._last_text: str = ""
        self._is_analyzing = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="left-panel"):
                yield InputPanel(id="input-panel")
                yield Static("Analyzing...", id="loading-indicator")
                yield AnnotatedView(id="annotated-view")
                yield EntityTable(id="entity-table")
        yield StatusBar(
            model_name=self._model_name or "default",
            threshold=self._confidence_threshold,
            group_entities=self._group_entities,
            medical_tokenizer=self._use_medical_tokenizer,
        )
        yield Footer()

    def on_mount(self) -> None:
        """Focus the input when app starts."""
        self.query_one("#input-text", TextArea).focus()

    def _get_analyze_func(self) -> Callable[..., Any]:
        """Get the analysis function, importing lazily if needed."""
        if self._analyze_func is not None:
            return self._analyze_func

        # Lazy import to avoid loading models at TUI import time
        from openmed import analyze_text

        return analyze_text

    @work(exclusive=True, thread=True)
    def _run_analysis(self, text: str) -> None:
        """Run analysis in a background thread."""
        import time

        start_time = time.perf_counter()

        try:
            analyze = self._get_analyze_func()

            # Build kwargs
            kwargs: dict[str, Any] = {
                "confidence_threshold": self._confidence_threshold,
                "group_entities": self._group_entities,
            }
            if self._model_name:
                kwargs["model_name"] = self._model_name

            result = analyze(text, **kwargs)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract entities from result
            entities: list[Entity] = []
            if hasattr(result, "entities"):
                # PredictionResult object
                for e in result.entities:
                    entities.append(
                        Entity(
                            text=e.text,
                            label=e.label,
                            start=e.start,
                            end=e.end,
                            confidence=e.confidence,
                        )
                    )
            elif isinstance(result, dict) and "entities" in result:
                # Dict result
                for pred in result["entities"]:
                    entities.append(Entity.from_prediction(pred))
            elif isinstance(result, list):
                # Raw list of predictions
                for pred in result:
                    entities.append(Entity.from_prediction(pred))

            # Post results back to main thread
            self.call_from_thread(self._update_results, text, entities, elapsed_ms)

        except Exception as e:
            self.call_from_thread(self._show_error, str(e))

    def _update_results(
        self, text: str, entities: list[Entity], elapsed_ms: float
    ) -> None:
        """Update UI with analysis results (called from main thread)."""
        self._entities = entities
        self._last_text = text
        self._is_analyzing = False

        # Hide loading indicator
        self.query_one("#loading-indicator").remove_class("visible")

        # Update views
        self.query_one("#annotated-view", AnnotatedView).update_annotated(
            text, entities
        )
        self.query_one("#entity-table", EntityTable).update_entities(entities)
        self.query_one(StatusBar).update_status(inference_time=elapsed_ms)

    def _show_error(self, message: str) -> None:
        """Show error message."""
        self._is_analyzing = False
        self.query_one("#loading-indicator").remove_class("visible")
        self.notify(f"Error: {message}", severity="error", timeout=5)

    def _reanalyze_if_needed(self) -> None:
        """Re-analyze last text if available."""
        if self._last_text and not self._is_analyzing:
            self._is_analyzing = True
            self.query_one("#loading-indicator").add_class("visible")
            self._run_analysis(self._last_text)

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_analyze(self) -> None:
        """Analyze the current input text."""
        if self._is_analyzing:
            return

        text = self.query_one("#input-panel", InputPanel).get_text().strip()
        if not text:
            self.notify("Please enter text to analyze", severity="warning")
            return

        self._is_analyzing = True
        self.query_one("#loading-indicator").add_class("visible")
        self._run_analysis(text)

    def action_clear(self) -> None:
        """Clear input and results."""
        self.query_one("#input-panel", InputPanel).set_text("")
        self.query_one("#annotated-view", AnnotatedView).update_annotated("", [])
        self.query_one("#entity-table", EntityTable).update_entities([])
        self._entities = []
        self._last_text = ""
        self.query_one("#input-text", TextArea).focus()

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
OpenMed TUI - Keyboard Shortcuts

Ctrl+Enter  Analyze current text
Ctrl+L      Clear input and results
F1          Show this help
F2          Switch model
F3          Configuration panel
F4          Switch profile
Ctrl+Q      Quit application

Tips:
- Paste clinical notes into the input area
- Entities are color-coded by type
- Table shows entities sorted by confidence
- Changing config re-analyzes automatically
        """
        self.notify(help_text.strip(), timeout=10)

    def action_switch_model(self) -> None:
        """Open model switcher modal."""

        def on_model_selected(model: str | None) -> None:
            if model and model != self._model_name:
                self._model_name = model
                self.query_one(StatusBar).update_status(model_name=model)
                self.query_one(StatusBar).clear_profile()
                self._current_profile = None
                self.notify(f"Model switched to: {model}", timeout=3)
                self._reanalyze_if_needed()

        self.push_screen(ModelSwitcherScreen(self._model_name), on_model_selected)

    def action_config_panel(self) -> None:
        """Open configuration panel modal."""

        def on_config_applied(config: dict[str, Any] | None) -> None:
            if config:
                changed = False
                if config["threshold"] != self._confidence_threshold:
                    self._confidence_threshold = config["threshold"]
                    changed = True
                if config["group_entities"] != self._group_entities:
                    self._group_entities = config["group_entities"]
                    changed = True
                if config["medical_tokenizer"] != self._use_medical_tokenizer:
                    self._use_medical_tokenizer = config["medical_tokenizer"]
                    changed = True

                self.query_one(StatusBar).update_status(
                    threshold=self._confidence_threshold,
                    group_entities=self._group_entities,
                    medical_tokenizer=self._use_medical_tokenizer,
                )

                if changed:
                    self.query_one(StatusBar).clear_profile()
                    self._current_profile = None
                    self.notify("Configuration updated", timeout=2)
                    self._reanalyze_if_needed()

        self.push_screen(
            ConfigPanelScreen(
                threshold=self._confidence_threshold,
                group_entities=self._group_entities,
                medical_tokenizer=self._use_medical_tokenizer,
            ),
            on_config_applied,
        )

    def action_switch_profile(self) -> None:
        """Open profile switcher modal."""

        def on_profile_selected(profile: str | None) -> None:
            if profile and profile in PROFILE_PRESETS:
                settings = PROFILE_PRESETS[profile]
                self._confidence_threshold = settings["threshold"]
                self._group_entities = settings["group_entities"]
                self._use_medical_tokenizer = settings["medical_tokenizer"]
                self._current_profile = profile

                self.query_one(StatusBar).update_status(
                    threshold=self._confidence_threshold,
                    group_entities=self._group_entities,
                    medical_tokenizer=self._use_medical_tokenizer,
                    profile=profile,
                )

                self.notify(f"Profile applied: {profile}", timeout=3)
                self._reanalyze_if_needed()

        self.push_screen(
            ProfileSwitcherScreen(self._current_profile), on_profile_selected
        )


def run_tui(
    model_name: str | None = None,
    confidence_threshold: float = 0.5,
    group_entities: bool = False,
    use_medical_tokenizer: bool = True,
) -> None:
    """Run the OpenMed TUI.

    Args:
        model_name: Model to use for analysis.
        confidence_threshold: Minimum confidence threshold.
        group_entities: Whether to group adjacent entities.
        use_medical_tokenizer: Whether to use medical tokenizer.
    """
    app = OpenMedTUI(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        group_entities=group_entities,
        use_medical_tokenizer=use_medical_tokenizer,
    )
    app.run()


if __name__ == "__main__":
    run_tui()
