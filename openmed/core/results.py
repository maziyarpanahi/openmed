"""Typed result objects returned by public OpenMed helpers."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from ..processing.outputs import EntityPrediction, PredictionResult


def _to_float(value: Any) -> Optional[float]:
    """Convert numeric-like values to built-in floats for JSON payloads."""
    if value is None:
        return None
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
        return None


@dataclass(frozen=True)
class AnalyzeResult(Mapping[str, Any]):
    """Typed result returned by :func:`openmed.analyze_text`.

    ``to_dict()`` intentionally preserves the historical ``PredictionResult``
    payload used by the REST service: ``text``, ``entities``, ``model_name``,
    ``timestamp``, ``processing_time``, and ``metadata``.
    """

    text: str
    entities: list[EntityPrediction]
    model: str
    timestamp: str
    processing_time: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None
    text_length: int = field(init=False)

    def __post_init__(self) -> None:
        """Record the validated input length without changing dict output."""
        object.__setattr__(self, "text_length", len(self.text))

    @classmethod
    def from_prediction_result(cls, result: PredictionResult) -> "AnalyzeResult":
        """Build an analyze result from the formatter's prediction result."""
        metadata = deepcopy(result.metadata) if result.metadata is not None else None
        return cls(
            text=result.text,
            entities=list(result.entities),
            model=result.model_name,
            timestamp=result.timestamp,
            processing_time=_to_float(result.processing_time),
            metadata=metadata,
        )

    @property
    def model_name(self) -> str:
        """Backward-compatible alias for the serialized model key."""
        return self.model

    def to_dict(self) -> dict[str, Any]:
        """Return the legacy dict payload used by API and CLI callers."""
        metadata = deepcopy(self.metadata) if self.metadata is not None else None
        return {
            "text": self.text,
            "entities": [entity.to_dict() for entity in self.entities],
            "model_name": self.model,
            "timestamp": self.timestamp,
            "processing_time": _to_float(self.processing_time),
            "metadata": metadata,
        }

    def __getitem__(self, key: str) -> Any:
        """Support mapping-style access to the legacy result payload."""
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over legacy payload keys."""
        return iter(self.to_dict())

    def __len__(self) -> int:
        """Return the number of legacy payload keys."""
        return len(self.to_dict())
