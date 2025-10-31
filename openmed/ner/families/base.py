"""Shared definitions for model family integrations."""

from __future__ import annotations

from enum import Enum
from typing import Protocol


class ModelFamily(str, Enum):
    """Enumeration of supported model families."""

    GLINER = "gliner"
    OTHER = "other"


class SupportsPrediction(Protocol):  # pragma: no cover - interface only
    """Protocol for minimal predictor objects."""

    def predict(self, text: str, *, labels: list[str] | None = None) -> object:
        ...


__all__ = ["ModelFamily", "SupportsPrediction"]
