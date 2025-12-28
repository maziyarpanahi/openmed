"""OpenMed TUI Application - Interactive clinical NER workbench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from rich.style import Style
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Static,
    TextArea,
)


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


def get_entity_color(label: str) -> str:
    """Get color for an entity type."""
    return ENTITY_COLORS.get(label.upper(), ENTITY_COLORS["DEFAULT"])


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
        yield Label("📝 Input (Ctrl+Enter to analyze)")
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
        yield Label("🔍 Annotated")
        yield Static("", id="annotated-text")

    def update_annotated(self, text: str, entities: list[Entity]) -> None:
        """Update the annotated text view with highlighted entities."""
        if not text:
            self.query_one("#annotated-text", Static).update("")
            return

        # Sort entities by start position (reverse to handle overlaps)
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
        yield Label("📋 Entities")
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
        self.query_one("Label", Label).update(f"📋 Entities ({len(entities)})")

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
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._threshold = threshold
        self._inference_time = inference_time

    def compose(self) -> ComposeResult:
        yield Label(self._get_status_text(), id="status-label")

    def _get_status_text(self) -> str:
        parts = [
            f"Model: {self._model_name}",
            f"Threshold: {self._threshold:.2f}",
        ]
        if self._inference_time is not None:
            parts.append(f"Inference: {self._inference_time:.0f}ms")
        return " │ ".join(parts)

    def update_status(
        self,
        model_name: str | None = None,
        threshold: float | None = None,
        inference_time: float | None = None,
    ) -> None:
        """Update status bar values."""
        if model_name is not None:
            self._model_name = model_name
        if threshold is not None:
            self._threshold = threshold
        if inference_time is not None:
            self._inference_time = inference_time
        self.query_one("#status-label", Label).update(self._get_status_text())


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
    ]

    def __init__(
        self,
        model_name: str | None = None,
        confidence_threshold: float = 0.5,
        analyze_func: Callable[..., Any] | None = None,
    ) -> None:
        """Initialize the TUI.

        Args:
            model_name: Model to use for analysis (optional, loads default if None).
            confidence_threshold: Minimum confidence threshold for entities.
            analyze_func: Custom analysis function (defaults to openmed.analyze_text).
        """
        super().__init__()
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._analyze_func = analyze_func
        self._entities: list[Entity] = []
        self._is_analyzing = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="left-panel"):
                yield InputPanel(id="input-panel")
                yield Static("⏳ Analyzing...", id="loading-indicator")
                yield AnnotatedView(id="annotated-view")
                yield EntityTable(id="entity-table")
        yield StatusBar(
            model_name=self._model_name or "default",
            threshold=self._confidence_threshold,
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
        self.query_one("#input-text", TextArea).focus()

    def action_help(self) -> None:
        """Show help information."""
        help_text = """
OpenMed TUI - Keyboard Shortcuts

Ctrl+Enter  Analyze current text
Ctrl+L      Clear input and results
Ctrl+Q      Quit application
F1          Show this help

Tips:
- Paste clinical notes into the input area
- Entities are color-coded by type
- Table shows entities sorted by confidence
        """
        self.notify(help_text.strip(), timeout=10)


def run_tui(
    model_name: str | None = None,
    confidence_threshold: float = 0.5,
) -> None:
    """Run the OpenMed TUI.

    Args:
        model_name: Model to use for analysis.
        confidence_threshold: Minimum confidence threshold.
    """
    app = OpenMedTUI(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
    )
    app.run()


if __name__ == "__main__":
    run_tui()
