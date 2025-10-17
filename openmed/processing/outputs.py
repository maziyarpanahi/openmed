"""Output formatting utilities for OpenMed."""

import json
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert value to built-in float if possible."""
    if value is None:
        return default

    if isinstance(value, (int, float)):
        return float(value)

    if hasattr(value, "item"):
        try:
            return float(value.item())
        except (TypeError, ValueError):
            pass

    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to convert %r to float", value)
        return default


def _to_int(value: Any) -> Optional[int]:
    """Convert value to built-in int if possible."""
    if value is None:
        return None

    if isinstance(value, int):
        return value

    if hasattr(value, "item"):
        try:
            return int(value.item())
        except (TypeError, ValueError):
            pass

    try:
        return int(value)
    except (TypeError, ValueError):
        logger.debug("Failed to convert %r to int", value)
        return None


@dataclass
class EntityPrediction:
    """Represents a single entity prediction."""
    text: str
    label: str
    confidence: float
    start: Optional[int] = None
    end: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "label": self.label,
            "confidence": _to_float(self.confidence, 0.0),
            "start": _to_int(self.start),
            "end": _to_int(self.end),
        }


@dataclass
class PredictionResult:
    """Represents the complete prediction result."""
    text: str
    entities: List[EntityPrediction]
    model_name: str
    timestamp: str
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["entities"] = [entity.to_dict() for entity in self.entities]
        result["processing_time"] = _to_float(result.get("processing_time"))
        return result


class OutputFormatter:
    """Formats model predictions into various output formats."""

    def __init__(self,
                 include_confidence: bool = True,
                 confidence_threshold: float = 0.0,
                 group_entities: bool = False):
        """Initialize output formatter.

        Args:
            include_confidence: Whether to include confidence scores.
            confidence_threshold: Minimum confidence to include predictions.
            group_entities: Whether to group adjacent entities of same type.
        """
        self.include_confidence = include_confidence
        self.confidence_threshold = confidence_threshold
        self.group_entities = group_entities
        self._current_text: Optional[str] = None

    def format_predictions(
        self,
        predictions: List[Dict[str, Any]],
        original_text: str,
        model_name: str = "unknown",
        **kwargs
    ) -> PredictionResult:
        """Format raw model predictions into structured output.

        Args:
            predictions: Raw predictions from model.
            original_text: Original input text.
            model_name: Name of the model used.
            **kwargs: Additional metadata.

        Returns:
            Formatted prediction result.
        """
        entities = []

        self._current_text = original_text

        for pred in predictions:
            score = _to_float(pred.get("score", 0.0), 0.0) or 0.0

            if score < self.confidence_threshold:
                continue

            start = pred.get("start")
            end = pred.get("end")
            raw_word = pred.get("word", "")
            token_text = ""
            adjusted_start = start if isinstance(start, int) else None
            adjusted_end = end if isinstance(end, int) else None

            if isinstance(start, int) and isinstance(end, int):
                token_text = original_text[start:end]

                if token_text:
                    leading_ws = len(token_text) - len(token_text.lstrip())
                    trailing_ws = len(token_text) - len(token_text.rstrip())

                    if leading_ws:
                        adjusted_start = start + leading_ws
                    if trailing_ws:
                        adjusted_end = end - trailing_ws

                    token_text = token_text.strip()

            if not token_text and raw_word:
                token_text = raw_word

            normalized_text = self._normalize_token_text(token_text)

            if not normalized_text and raw_word:
                normalized_text = self._normalize_token_text(raw_word)

            entity_text = normalized_text

            raw_label = (
                pred.get("entity_group")
                or pred.get("entity")
                or ""
            )
            clean_label = raw_label.replace("B-", "").replace("I-", "")
            label = clean_label or raw_label or "UNKNOWN"

            entity = EntityPrediction(
                text=entity_text,
                label=label,
                confidence=score,
                start=_to_int(adjusted_start if adjusted_start is not None else start),
                end=_to_int(adjusted_end if adjusted_end is not None else end)
            )
            entities.append(entity)

        if self.group_entities:
            entities = self._group_adjacent_entities(entities)

        result = PredictionResult(
            text=original_text,
            entities=entities,
            model_name=model_name,
            timestamp=datetime.now().isoformat(),
            processing_time=kwargs.get("processing_time"),
            metadata=kwargs.get("metadata", {})
        )

        # Reset reference to avoid leaking state across calls
        self._current_text = None

        return result

    def _normalize_token_text(self, text: Optional[str]) -> str:
        """Clean token text produced by different tokenization strategies."""
        if not text:
            return ""

        cleaned = text.replace("▁", " ").replace("Ġ", " ").replace("Ċ", " ").replace("@@", " ")

        while cleaned.startswith("##"):
            cleaned = cleaned[2:]

        cleaned = cleaned.strip()

        if not cleaned:
            return ""

        return " ".join(cleaned.split())

    def _group_adjacent_entities(
        self,
        entities: List[EntityPrediction]
    ) -> List[EntityPrediction]:
        """Group adjacent entities of the same type.

        Args:
            entities: List of entity predictions.

        Returns:
            Grouped entities.
        """
        if not entities:
            return entities

        grouped = []
        current_group = [entities[0]]

        for entity in entities[1:]:
            last_entity = current_group[-1]

            # Check if entities are adjacent and same label
            if (entity.label == last_entity.label and
                entity.start is not None and
                last_entity.end is not None and
                entity.start <= last_entity.end + 2):  # Allow small gaps
                current_group.append(entity)
            else:
                # Finalize current group
                if len(current_group) > 1:
                    grouped_entity = self._merge_entities(current_group)
                    grouped.append(grouped_entity)
                else:
                    grouped.append(current_group[0])

                current_group = [entity]

        # Handle last group
        if len(current_group) > 1:
            grouped_entity = self._merge_entities(current_group)
            grouped.append(grouped_entity)
        else:
            grouped.append(current_group[0])

        return grouped

    def _merge_entities(
        self,
        entities: List[EntityPrediction]
    ) -> EntityPrediction:
        """Merge multiple entities into one.

        Args:
            entities: List of entities to merge.

        Returns:
            Merged entity.
        """
        if not entities:
            raise ValueError("Cannot merge empty entity list")

        start = entities[0].start
        end = entities[-1].end

        if (
            self._current_text is not None
            and isinstance(start, int)
            and isinstance(end, int)
        ):
            text = self._current_text
            start_idx = start
            end_idx = end

            while start_idx > 0 and text[start_idx - 1].isalnum():
                start_idx -= 1
            while end_idx < len(text) and text[end_idx].isalnum():
                end_idx += 1

            merged_text = text[start_idx:end_idx].strip()
            start = start_idx
            end = end_idx
        else:
            text_parts = [entity.text for entity in entities if entity.text]
            merged_text = " ".join(text_parts)

        # Use the label and average confidence
        label = entities[0].label
        avg_confidence = sum(e.confidence for e in entities) / len(entities)

        # Use start of first and end of last
        return EntityPrediction(
            text=merged_text,
            label=label,
            confidence=avg_confidence,
            start=start,
            end=end
        )

    def to_json(self, result: PredictionResult, indent: int = 2) -> str:
        """Convert result to JSON string.

        Args:
            result: Prediction result to convert.
            indent: JSON indentation.

        Returns:
            JSON string.
        """
        return json.dumps(result.to_dict(), indent=indent)

    def to_html(self, result: PredictionResult) -> str:
        """Convert result to HTML format with highlighted entities.

        Args:
            result: Prediction result to convert.

        Returns:
            HTML string.
        """
        html = f'<div class="openmed-result">\n'
        html += f'<h3>Analysis Results</h3>\n'
        html += f'<p><strong>Model:</strong> {result.model_name}</p>\n'
        html += f'<p><strong>Timestamp:</strong> {result.timestamp}</p>\n'

        if result.processing_time:
            html += f'<p><strong>Processing Time:</strong> {result.processing_time:.3f}s</p>\n'

        html += f'<div class="text-content">\n'

        # Highlight entities in text
        highlighted_text = result.text
        offset = 0

        # Sort entities by start position
        sorted_entities = sorted(
            [e for e in result.entities if e.start is not None and e.end is not None],
            key=lambda x: x.start
        )

        for entity in sorted_entities:
            start = entity.start + offset
            end = entity.end + offset

            color = self._get_entity_color(entity.label)

            highlight_start = f'<span class="entity entity-{entity.label.lower()}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="Label: {entity.label}, Confidence: {entity.confidence:.3f}">'
            highlight_end = '</span>'

            highlighted_text = (
                highlighted_text[:start] +
                highlight_start +
                highlighted_text[start:end] +
                highlight_end +
                highlighted_text[end:]
            )

            offset += len(highlight_start) + len(highlight_end)

        html += f'<p>{highlighted_text}</p>\n'
        html += f'</div>\n'

        # Entity summary
        if result.entities:
            html += f'<div class="entity-summary">\n'
            html += f'<h4>Detected Entities ({len(result.entities)})</h4>\n'
            html += f'<ul>\n'

            for entity in result.entities:
                confidence_str = f" (confidence: {entity.confidence:.3f})" if self.include_confidence else ""
                html += f'<li><strong>{entity.label}:</strong> {entity.text}{confidence_str}</li>\n'

            html += f'</ul>\n'
            html += f'</div>\n'

        html += f'</div>\n'
        return html

    def _get_entity_color(self, label: str) -> str:
        """Get color for entity label.

        Args:
            label: Entity label.

        Returns:
            CSS color string.
        """
        colors = {
            "person": "#FFE4E1",
            "organization": "#E6F3FF",
            "location": "#E6FFE6",
            "date": "#FFF0E6",
            "time": "#F0E6FF",
            "money": "#FFFFE6",
            "medication": "#FFE4F0",
            "condition": "#E4F0FF",
            "procedure": "#F0FFE4",
            "anatomy": "#F5F5DC"
        }
        return colors.get(label.lower(), "#F0F0F0")

    def to_csv_rows(self, result: PredictionResult) -> List[Dict[str, Any]]:
        """Convert result to CSV-compatible rows.

        Args:
            result: Prediction result to convert.

        Returns:
            List of dictionaries for CSV writing.
        """
        rows = []
        for entity in result.entities:
            row = {
                "text": entity.text,
                "label": entity.label,
                "confidence": entity.confidence,
                "start": entity.start,
                "end": entity.end,
                "model_name": result.model_name,
                "timestamp": result.timestamp,
                "processing_time": result.processing_time,
                "original_text": result.text
            }
            rows.append(row)
        return rows


def format_predictions(
    predictions: List[Dict[str, Any]],
    original_text: str,
    model_name: str = "unknown",
    output_format: str = "dict",
    **kwargs
) -> Union[PredictionResult, str, List[Dict[str, Any]]]:
    """Convenience function to format predictions.

    Args:
        predictions: Raw model predictions.
        original_text: Original input text.
        model_name: Name of the model used.
        output_format: Output format ("dict", "json", "html", "csv").
        **kwargs: Additional formatting options.

    Returns:
        Formatted output in requested format.
    """
    formatter_keys = {"include_confidence", "confidence_threshold", "group_entities"}
    formatter_kwargs = {k: v for k, v in kwargs.items() if k in formatter_keys}
    downstream_kwargs = {k: v for k, v in kwargs.items() if k not in formatter_keys}

    formatter = OutputFormatter(**formatter_kwargs)
    result = formatter.format_predictions(
        predictions,
        original_text,
        model_name,
        **downstream_kwargs,
    )

    if output_format == "dict":
        return result
    elif output_format == "json":
        return formatter.to_json(result)
    elif output_format == "html":
        return formatter.to_html(result)
    elif output_format == "csv":
        return formatter.to_csv_rows(result)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
