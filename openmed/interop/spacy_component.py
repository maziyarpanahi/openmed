"""spaCy pipeline component for OpenMed PII de-identification spans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any

from openmed.core.capabilities import raise_missing_backend

try:
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    from spacy.util import filter_spans
except ImportError as exc:  # pragma: no cover - exercised without the extra
    raise_missing_backend("spacy", feature="spaCy support", cause=exc)


Extractor = Callable[..., Any]
AlignmentMode = str
_ALIGNMENT_MODES = {"strict", "contract", "expand"}


@dataclass(frozen=True)
class OpenMedPiiSpan:
    """Dependency-light PII span stored on ``Doc._.openmed_pii``."""

    label: str
    start: int
    end: int
    score: float | None = None


@dataclass(frozen=True)
class OpenMedDeidConfig:
    """Runtime options for the ``openmed_deid`` spaCy component."""

    model_name: str | None = None
    confidence_threshold: float = 0.5
    lang: str = "en"
    policy: str | None = None
    target: str = "openmed_pii"
    merge_ents: bool = False
    alignment_mode: AlignmentMode = "expand"

    def __post_init__(self) -> None:
        if self.alignment_mode not in _ALIGNMENT_MODES:
            known = ", ".join(sorted(_ALIGNMENT_MODES))
            raise ValueError(f"alignment_mode must be one of: {known}")
        if not self.target:
            raise ValueError("target must be a non-empty doc.spans key")

    def extract_kwargs(self, extractor: Extractor) -> dict[str, Any]:
        """Return keyword arguments supported by ``extractor``."""

        kwargs: dict[str, Any] = {
            "confidence_threshold": float(self.confidence_threshold),
            "lang": self.lang,
        }
        if self.model_name is not None:
            kwargs["model_name"] = self.model_name
        if self.policy is not None and _accepts_keyword(extractor, "policy"):
            kwargs["policy"] = self.policy
        return kwargs


class OpenMedDeidComponent:
    """Project OpenMed PII detections onto a spaCy ``Doc``."""

    def __init__(
        self,
        *,
        config: OpenMedDeidConfig | None = None,
        extractor: Extractor | None = None,
    ) -> None:
        self.config = config or OpenMedDeidConfig()
        self._extractor = extractor
        _ensure_doc_extension()

    def __call__(self, doc: Doc) -> Doc:
        """Detect PII in ``doc.text`` and attach spaCy spans."""

        extractor = self._extractor or _load_extract_pii()
        result = extractor(doc.text, **self.config.extract_kwargs(extractor))
        raw_spans = _raw_spans_from_result(result)
        doc._.openmed_pii = raw_spans

        projected = _project_spans(
            doc,
            raw_spans,
            alignment_mode=self.config.alignment_mode,
        )
        doc.spans[self.config.target] = projected
        if self.config.merge_ents:
            doc.ents = _resolve_entity_overlaps((*doc.ents, *projected))
        return doc


@Language.factory(
    "openmed_deid",
    default_config={
        "model_name": None,
        "confidence_threshold": 0.5,
        "lang": "en",
        "policy": None,
        "target": "openmed_pii",
        "merge_ents": False,
        "alignment_mode": "expand",
    },
)
def make_openmed_deid(
    nlp: Language,
    name: str,
    model_name: str | None,
    confidence_threshold: float,
    lang: str,
    policy: str | None,
    target: str,
    merge_ents: bool,
    alignment_mode: AlignmentMode,
) -> OpenMedDeidComponent:
    """Create the ``openmed_deid`` spaCy pipeline component."""

    del nlp, name
    return OpenMedDeidComponent(
        config=OpenMedDeidConfig(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            lang=lang,
            policy=policy,
            target=target,
            merge_ents=merge_ents,
            alignment_mode=alignment_mode,
        )
    )


def _ensure_doc_extension() -> None:
    if not Doc.has_extension("openmed_pii"):
        Doc.set_extension("openmed_pii", default=())


def _load_extract_pii() -> Extractor:
    from openmed import extract_pii

    return extract_pii


def _raw_spans_from_result(result: Any) -> tuple[OpenMedPiiSpan, ...]:
    items = getattr(result, "entities", result)
    if items is None:
        return ()

    spans: list[OpenMedPiiSpan] = []
    for item in _as_sequence(items):
        start = _optional_int(_field_value(item, "start"))
        end = _optional_int(_field_value(item, "end"))
        label = _field_value(item, "canonical_label", "label", "entity_type")
        if start is None or end is None or end <= start or label is None:
            continue

        score = _optional_float(_field_value(item, "score", "confidence"))
        spans.append(
            OpenMedPiiSpan(
                label=str(label),
                start=start,
                end=end,
                score=score,
            )
        )
    return tuple(spans)


def _project_spans(
    doc: Doc,
    raw_spans: Sequence[OpenMedPiiSpan],
    *,
    alignment_mode: AlignmentMode,
) -> list[Span]:
    spans: list[Span] = []
    for raw_span in raw_spans:
        span = doc.char_span(
            raw_span.start,
            raw_span.end,
            label=raw_span.label,
            alignment_mode=alignment_mode,
        )
        if span is not None:
            spans.append(span)
    return spans


def _resolve_entity_overlaps(spans: Sequence[Span]) -> tuple[Span, ...]:
    return tuple(filter_spans(spans))


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (str, bytes)):
        return (value,)
    if isinstance(value, Sequence):
        return value
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _field_value(item: Any, *names: str) -> Any:
    for name in names:
        if isinstance(item, Mapping) and name in item:
            value = item[name]
            if value is not None:
                return value
        if hasattr(item, name):
            value = getattr(item, name)
            if value is not None:
                return value
    return None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _accepts_keyword(func: Extractor, name: str) -> bool:
    try:
        parameters = signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind is Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


__all__ = [
    "OpenMedDeidComponent",
    "OpenMedDeidConfig",
    "OpenMedPiiSpan",
    "make_openmed_deid",
]
