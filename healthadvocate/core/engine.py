"""OpenMed model management and inference wrapper."""

from __future__ import annotations

import sys
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Ensure the openmed-fixes directory is on sys.path so `import openmed` works
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import openmed
from openmed import ModelLoader, OpenMedConfig


@dataclass
class EntityMatch:
    text: str
    label: str
    confidence: float
    start: int
    end: int
    category: str


@dataclass
class AnalysisResult:
    entities: list[EntityMatch] = field(default_factory=list)
    model_used: str = ""
    processing_time: float = 0.0
    error: str = ""


def _extract_entities(raw_result: object, category: str) -> list[EntityMatch]:
    """Extract EntityMatch list from an openmed PredictionResult or dict."""
    entities: list[EntityMatch] = []

    # PredictionResult has .entities attribute
    if hasattr(raw_result, "entities"):
        items = raw_result.entities
    elif isinstance(raw_result, dict):
        items = raw_result.get("entities", [])
        if not items:
            items = raw_result.get("predictions", [])
    elif isinstance(raw_result, list):
        items = raw_result
    else:
        return entities

    for item in items:
        if isinstance(item, dict):
            entities.append(EntityMatch(
                text=item.get("text", item.get("word", "")),
                label=item.get("label", item.get("entity_group", item.get("entity", ""))),
                confidence=float(item.get("confidence", item.get("score", 0.0))),
                start=int(item.get("start", 0)),
                end=int(item.get("end", 0)),
                category=category,
            ))
        elif hasattr(item, "text"):
            entities.append(EntityMatch(
                text=item.text,
                label=getattr(item, "label", getattr(item, "entity_group", "")),
                confidence=float(getattr(item, "confidence", getattr(item, "score", 0.0))),
                start=int(getattr(item, "start", 0)),
                end=int(getattr(item, "end", 0)),
                category=category,
            ))
    return entities


class HealthEngine:
    """Central OpenMed inference wrapper shared by all feature modules."""

    def __init__(self) -> None:
        self._loader: Optional[ModelLoader] = None

    @property
    def loader(self) -> ModelLoader:
        if self._loader is None:
            self._loader = ModelLoader()
        return self._loader

    def preload(self) -> None:
        """Pre-warm the most commonly used models (call at startup)."""
        logger.info("Pre-loading OpenMed models...")
        start = time.time()
        # Trigger model loading by creating a pipeline for each key model
        for model_name in (
            "disease_detection_superclinical",
            "pharma_detection_superclinical",
            "anatomy_detection_electramed",
        ):
            try:
                self.loader.create_pipeline(
                    model_name,
                    task="token-classification",
                    aggregation_strategy="simple",
                )
                logger.info("  Loaded: %s", model_name)
            except Exception as exc:
                logger.warning("  Failed to preload %s: %s", model_name, exc)
        elapsed = time.time() - start
        logger.info("Model pre-loading complete (%.1fs)", elapsed)

    def analyze(
        self,
        text: str,
        model_name: str,
        confidence: float = 0.5,
        category: str = "unknown",
    ) -> AnalysisResult:
        """Run NER analysis and return normalized results."""
        start = time.time()
        try:
            raw = openmed.analyze_text(
                text,
                model_name,
                loader=self.loader,
                output_format="dict",
                confidence_threshold=confidence,
                group_entities=True,
            )
        except Exception as exc:
            logger.error("analyze_text failed with %s: %s", model_name, exc)
            return AnalysisResult(model_used=model_name, processing_time=0.0, error=str(exc))

        elapsed = time.time() - start
        entities = _extract_entities(raw, category)
        return AnalysisResult(
            entities=entities,
            model_used=model_name,
            processing_time=round(elapsed, 3),
        )

    def extract_diseases(self, text: str, confidence: float = 0.65) -> AnalysisResult:
        return self.analyze(text, "disease_detection_superclinical", confidence, "disease")

    def extract_drugs(self, text: str, confidence: float = 0.70) -> AnalysisResult:
        return self.analyze(text, "pharma_detection_superclinical", confidence, "drug")

    def extract_anatomy(self, text: str, confidence: float = 0.60) -> AnalysisResult:
        return self.analyze(text, "anatomy_detection_electramed", confidence, "anatomy")

    def extract_pii(self, text: str, lang: str = "en") -> AnalysisResult:
        start = time.time()
        try:
            raw = openmed.extract_pii(text, lang=lang, loader=self.loader)
        except Exception as exc:
            logger.error("extract_pii failed: %s", exc)
            return AnalysisResult(model_used="pii", processing_time=0.0, error=str(exc))

        elapsed = time.time() - start
        entities = _extract_entities(raw, "pii")
        return AnalysisResult(
            entities=entities,
            model_used="pii",
            processing_time=round(elapsed, 3),
        )

    def deidentify(self, text: str, method: str = "mask") -> str:
        try:
            result = openmed.deidentify(text, method=method, loader=self.loader)
            return result.deidentified_text
        except Exception as exc:
            logger.error("deidentify failed: %s", exc)
            return text

    def deidentify_for_llm(self, text: str, method: str = "mask") -> tuple[str, dict[str, str]]:
        try:
            result = openmed.deidentify(text, method=method, loader=self.loader, keep_mapping=True)
            pii_map = getattr(result, "mapping", {}) or {}
            return result.deidentified_text, pii_map
        except Exception as exc:
            logger.error("deidentify_for_llm failed: %s", exc)
            return text, {}


def format_entities_with_confidence(entities: list[EntityMatch]) -> str:
    if not entities:
        return "No medical entities detected by NER."
    parts = []
    for e in entities:
        if e.confidence >= 0.85:
            cert = "very high certainty"
        elif e.confidence >= 0.70:
            cert = "high certainty"
        elif e.confidence >= 0.50:
            cert = "moderate certainty, may need verification"
        else:
            cert = "low certainty, possibly incidental"
        parts.append(f"'{e.text}' ({e.label}, {e.confidence:.0%} confidence - {cert})")
    return "; ".join(parts)
