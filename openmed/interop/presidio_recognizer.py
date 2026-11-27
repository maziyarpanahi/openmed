"""Presidio ``EntityRecognizer`` backed by a locally cached OpenMed PII model.

Add OpenMed to a Presidio ``AnalyzerEngine``::

    from presidio_analyzer import AnalyzerEngine
    from openmed.interop.presidio_recognizer import OpenMedRecognizer

    analyzer = AnalyzerEngine()
    analyzer.registry.add_recognizer(OpenMedRecognizer())

The recognizer only reports entity names Presidio already defines, runs the
model from the local cache (no download at analyze time) and never copies
source text into results, explanations or logs.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, Callable, Iterable, Sequence

from openmed.core.capabilities import raise_missing_backend
from openmed.core.labels import normalize_label
from openmed.interop.presidio import _CANONICAL_TO_PRESIDIO, _load_presidio_result_cls

DEFAULT_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
RECOGNIZER_NAME = "OpenMedRecognizer"
SUPPORTED_ENTITIES: tuple[str, ...] = tuple(
    sorted(set(_CANONICAL_TO_PRESIDIO.values()))
)

Extractor = Callable[[str], Any]


def create_openmed_recognizer(
    model_name: str = DEFAULT_MODEL,
    *,
    supported_language: str = "en",
    confidence_threshold: float = 0.5,
    supported_entities: Sequence[str] | None = None,
    extractor: Extractor | None = None,
    base_cls: type[Any] | None = None,
    result_cls: type[Any] | None = None,
    explanation_cls: type[Any] | None = None,
) -> Any:
    """Build an OpenMed recognizer instance for a Presidio ``AnalyzerEngine``.

    Args:
        model_name: OpenMed PII model id; it must already be in the local cache.
        supported_language: Presidio language code the recognizer serves.
        confidence_threshold: Minimum OpenMed confidence for a reported span.
        supported_entities: Subset of ``SUPPORTED_ENTITIES`` to report.
        extractor: Callable ``text -> PredictionResult``-like object. Defaults
            to local-only ``openmed.extract_pii``. Injected for tests.
        base_cls: Recognizer base class; defaults to Presidio's ``EntityRecognizer``.
        result_cls: Result class; defaults to Presidio's ``RecognizerResult``.
        explanation_cls: Explanation class; defaults to Presidio's
            ``AnalysisExplanation`` for a Presidio base, else no explanation.

    Returns:
        An ``OpenMedRecognizer`` instance subclassing ``base_cls``.
    """

    cls = _recognizer_class(base_cls or _load_presidio_attr("EntityRecognizer"))
    return cls(
        model_name=model_name,
        supported_language=supported_language,
        confidence_threshold=confidence_threshold,
        supported_entities=supported_entities,
        extractor=extractor,
        result_cls=result_cls,
        explanation_cls=explanation_cls,
    )


@lru_cache(maxsize=None)
def _recognizer_class(base: type[Any]) -> type[Any]:
    class OpenMedRecognizer(base):  # type: ignore[misc, valid-type]
        """Presidio recognizer that runs an OpenMed PII model locally."""

        def __init__(
            self,
            model_name: str = DEFAULT_MODEL,
            *,
            supported_language: str = "en",
            confidence_threshold: float = 0.5,
            supported_entities: Sequence[str] | None = None,
            extractor: Extractor | None = None,
            result_cls: type[Any] | None = None,
            explanation_cls: type[Any] | None = None,
        ) -> None:
            entities = list(supported_entities or SUPPORTED_ENTITIES)
            unknown = sorted(set(entities) - set(SUPPORTED_ENTITIES))
            if unknown:
                raise ValueError(f"Unsupported Presidio entity names: {unknown}")
            self.model_name = model_name
            self.confidence_threshold = float(confidence_threshold)
            self._extractor = extractor
            self._result_cls = result_cls or _load_presidio_result_cls()
            self._explanation_cls = explanation_cls
            if explanation_cls is None and base.__module__.startswith(
                "presidio_analyzer"
            ):
                self._explanation_cls = _load_presidio_attr("AnalysisExplanation")
            # Presidio's base __init__ calls load().
            super().__init__(
                supported_entities=entities,
                name=RECOGNIZER_NAME,
                supported_language=supported_language,
            )

        def load(self) -> None:
            """Bind the local-only extractor; the model loads on first analyze."""
            if self._extractor is None:
                self._extractor = _local_extractor(
                    self.model_name, self.confidence_threshold
                )

        def analyze(
            self,
            text: str,
            entities: Sequence[str] | None = None,
            nlp_artifacts: Any = None,
        ) -> list[Any]:
            """Return Presidio results for OpenMed spans in ``text``."""
            wanted = set(entities or self.supported_entities)
            wanted &= set(self.supported_entities)
            if not text or not wanted:
                return []
            if self._extractor is None:
                self.load()
            prediction = self._extractor(text)
            return [
                self._to_result(entity_type, start, end, score)
                for entity_type, start, end, score in _spans(
                    prediction, wanted, self.confidence_threshold, len(text)
                )
            ]

        def _to_result(self, entity_type: str, start: int, end: int, score: float):
            explanation = None
            if self._explanation_cls is not None:
                # Model id only, never source text.
                explanation = self._explanation_cls(
                    recognizer=self.name,
                    original_score=score,
                    textual_explanation=f"Identified by OpenMed model {self.model_name}",
                )
            return self._result_cls(
                entity_type=entity_type,
                start=start,
                end=end,
                score=score,
                analysis_explanation=explanation,
                recognition_metadata={
                    "recognizer_name": self.name,
                    "recognizer_identifier": getattr(self, "id", self.name),
                    "openmed_model": self.model_name,
                },
            )

    return OpenMedRecognizer


def _spans(
    prediction: Any, wanted: set[str], threshold: float, text_length: int
) -> Iterable[tuple[str, int, int, float]]:
    entities = getattr(prediction, "entities", prediction) or []
    for entity in entities:
        start, end = getattr(entity, "start", None), getattr(entity, "end", None)
        if start is None or end is None or not 0 <= start < end <= text_length:
            continue
        score = float(getattr(entity, "confidence", 0.0) or 0.0)
        if score < threshold:
            continue
        mapped = _CANONICAL_TO_PRESIDIO.get(normalize_label(str(entity.label)))
        if mapped in wanted:
            yield mapped, int(start), int(end), score


def _local_extractor(model_name: str, threshold: float) -> Extractor:
    from openmed.core.config import OpenMedConfig

    config = OpenMedConfig(local_only=True)
    loaders: list[Any] = []

    def extract(text: str) -> Any:
        from openmed.core.models import ModelLoader
        from openmed.core.pii import extract_pii

        if not loaders:
            loaders.append(ModelLoader(config))
        return extract_pii(
            text,
            model_name=model_name,
            confidence_threshold=threshold,
            config=config,
            loader=loaders[0],
        )

    return extract


def _load_presidio_attr(name: str) -> Any:
    try:
        module = import_module("presidio_analyzer")
    except ImportError as exc:
        raise_missing_backend("presidio", feature="Presidio support", cause=exc)
    return getattr(module, name)


OpenMedRecognizer: type[Any]  # resolved lazily by __getattr__ (needs Presidio)


def __getattr__(name: str) -> Any:
    if name == "OpenMedRecognizer":
        return _recognizer_class(_load_presidio_attr("EntityRecognizer"))
    raise AttributeError(name)


__all__ = [
    "DEFAULT_MODEL",
    "OpenMedRecognizer",
    "SUPPORTED_ENTITIES",
    "create_openmed_recognizer",
]
