"""Optional learned section-boundary refinement.

The default path is a small deterministic fallback built from synthetic and
weak-label-friendly lexical cues.  A caller may provide a local MLX
token-classification artifact or an injected predictor.  MLX and tokenizer
imports stay inside the first prediction call so importing the clinical
section package never adds a heavy dependency or performs I/O.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from openmed.clinical.data.section_loinc_map import section_codes, section_codings
from openmed.clinical.lexicons import normalize_section_header

SECTION_MODEL_ENV = "OPENMED_SECTION_MODEL"
LEARNED_SOURCE = "learned"


class SectionPredictor(Protocol):
    """Callable contract for an injected local section predictor."""

    def __call__(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> Iterable[Any]: ...


@dataclass(frozen=True)
class LearnedSectionCandidate:
    """A learned section boundary before rules-first reconciliation."""

    label: str
    start: int
    end: int
    confidence: float
    header_start: int | None = None
    header_end: int | None = None
    content_start: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready candidate mapping."""

        payload: dict[str, Any] = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source": LEARNED_SOURCE,
            "codes": section_codes(self.label),
        }
        codings = section_codings(self.label)
        if codings:
            payload["coding"] = codings
            payload["loinc_code"] = codings[0]["code"]
        if self.header_start is not None and self.header_end is not None:
            payload["header_start"] = self.header_start
            payload["header_end"] = self.header_end
        if self.content_start is not None:
            payload["content_start"] = self.content_start
        return payload


class SectionHead:
    """Lazy local section head with an offline lexical fallback.

    Args:
        model_path: Optional local MLX artifact directory. The artifact is
            loaded only when the first prediction is requested. No remote
            model name is accepted here, which keeps this API local-first.
        predictor: Optional callable used for tests or another on-device
            runtime. It must return mappings or tuples containing section
            labels and character offsets.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        predictor: SectionPredictor | Callable[..., Iterable[Any]] | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.predictor = predictor
        self._pipeline: Any | None = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        """Return whether an external local model has been initialized."""

        return self._loaded

    def __call__(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Predict section boundary candidates for *text*."""

        return self.predict(text, language=language)

    def predict(
        self,
        text: str,
        *,
        language: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Run the injected, local MLX, or offline fallback head."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if self.predictor is not None:
            raw = _call_predictor(self.predictor, text, language=language)
            return _normalize_predictions(raw, text)

        model_path = self.model_path or _model_path_from_environment()
        if model_path is not None:
            raw = self._predict_with_mlx(model_path, text)
            return _normalize_predictions(raw, text)

        return _fallback_predictions(text)

    def _predict_with_mlx(self, model_path: Path, text: str) -> Any:
        if not model_path.exists() or not model_path.is_dir():
            raise FileNotFoundError(
                f"section model artifact directory does not exist: {model_path}"
            )
        if self._pipeline is None:
            # Optional imports deliberately live on this lazy path.  The
            # caller can use the offline fallback without MLX installed.
            from openmed.mlx.inference import MLXTokenClassificationPipeline

            self._pipeline = MLXTokenClassificationPipeline(
                model_path,
                aggregation_strategy="simple",
            )
            self._loaded = True
        return self._pipeline(text)


LearnedSectionHead = SectionHead


def predict_section_candidates(
    text: str,
    *,
    language: str | None = None,
    head: SectionHead | SectionPredictor | Callable[..., Iterable[Any]] | None = None,
    model_path: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return normalized candidates from an optional section head."""

    active_head = head or SectionHead(model_path=model_path)
    if isinstance(active_head, SectionHead):
        return active_head.predict(text, language=language)
    raw = _call_predictor(active_head, text, language=language)
    return _normalize_predictions(raw, text)


def load_section_head(
    model_path: str | Path | None = None,
    *,
    predictor: SectionPredictor | Callable[..., Iterable[Any]] | None = None,
) -> SectionHead:
    """Create a lazy section head without importing or loading MLX."""

    return SectionHead(model_path=model_path, predictor=predictor)


def _call_predictor(
    predictor: Callable[..., Iterable[Any]],
    text: str,
    *,
    language: str | None,
) -> Iterable[Any]:
    try:
        return predictor(text, language=language)
    except TypeError as exc:
        if "language" not in str(exc):
            raise
        return predictor(text)


def _model_path_from_environment() -> Path | None:
    raw_path = os.environ.get(SECTION_MODEL_ENV, "").strip()
    return Path(raw_path).expanduser() if raw_path else None


def _normalize_predictions(
    raw_predictions: Iterable[Any],
    text: str,
) -> tuple[dict[str, Any], ...]:
    if isinstance(raw_predictions, Mapping):
        raw_predictions = (raw_predictions,)
    elif isinstance(raw_predictions, (str, bytes)):
        raise TypeError("section predictor must return span mappings")

    candidates: list[LearnedSectionCandidate] = []
    for raw in raw_predictions:
        candidate = _candidate_from_raw(raw, text)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda item: (item.start, -item.confidence, item.label))
    selected: list[LearnedSectionCandidate] = []
    for candidate in candidates:
        if selected and candidate.start == selected[-1].start:
            if candidate.confidence > selected[-1].confidence:
                selected[-1] = candidate
            continue
        selected.append(candidate)
    return tuple(candidate.to_dict() for candidate in selected)


def _candidate_from_raw(
    raw: Any,
    text: str,
) -> LearnedSectionCandidate | None:
    if isinstance(raw, Mapping):
        label_value = raw.get("label")
        if label_value is None:
            label_value = raw.get("entity_group", raw.get("entity"))
        start_value = raw.get("start")
        end_value = raw.get("end")
        score_value = raw.get("confidence", raw.get("score", 0.75))
        metadata = raw
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 3:
            return None
        if isinstance(raw[0], str):
            label_value, start_value, end_value = raw
        else:
            start_value, end_value, label_value = raw
        score_value = 0.75
        metadata = {}
    else:
        label_value = _raw_field(raw, "label")
        if label_value is None:
            label_value = _raw_field(raw, "entity_group")
        start_value = _raw_field(raw, "start")
        end_value = _raw_field(raw, "end")
        score_value = _raw_field(raw, "confidence")
        if score_value is None:
            score_value = _raw_field(raw, "score")
        if score_value is None:
            score_value = 0.75
        metadata = raw

    label = _normalize_section_label(label_value)
    if label is None or label == "unsectioned":
        return None
    try:
        start = int(start_value)
        end = int(end_value) if end_value is not None else len(text)
        confidence = min(max(float(score_value), 0.0), 1.0)
    except (TypeError, ValueError):
        return None
    if isinstance(start_value, bool) or isinstance(end_value, bool):
        return None
    if start < 0 or start >= len(text) or end <= start or end > len(text):
        return None

    header_start = _optional_int(_raw_field(metadata, "header_start"))
    header_end = _optional_int(_raw_field(metadata, "header_end"))
    content_start = _optional_int(_raw_field(metadata, "content_start"))
    if header_start is None:
        header_start = start
    if header_end is None:
        header_end = max(start + 1, min(end, start + _surface_width(text, start)))
    if not start <= header_start < header_end <= end:
        header_start, header_end = start, min(end, start + 1)
    if content_start is not None and not header_end <= content_start <= end:
        content_start = header_end

    return LearnedSectionCandidate(
        label=label,
        start=start,
        end=end,
        confidence=confidence,
        header_start=header_start,
        header_end=header_end,
        content_start=content_start,
    )


def _raw_field(raw: Any, key: str) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key)
    return getattr(raw, key, None)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _surface_width(text: str, start: int) -> int:
    match = re.match(r"\S+", text[start:])
    return len(match.group(0)) if match else 1


_SECTION_ALIASES = {
    "allergy": "allergies",
    "allergies": "allergies",
    "a p": "assessment_and_plan",
    "assessment plan": "assessment_and_plan",
    "assessment and plan": "assessment_and_plan",
    "assessment": "assessment",
    "chief complaint": "chief_complaint",
    "cc": "chief_complaint",
    "family history": "family_history",
    "findings": "findings",
    "history": "history",
    "history of present illness": "history_of_present_illness",
    "hpi": "history_of_present_illness",
    "impression": "impression",
    "medication": "medications",
    "medications": "medications",
    "medication list": "medications",
    "current medications": "medications",
    "past history": "past_medical_history",
    "past medical history": "past_medical_history",
    "medical history": "past_medical_history",
    "pmh": "past_medical_history",
    "plan": "plan",
    "plan of care": "plan",
    "problem list": "problem_list",
    "problems": "problem_list",
    "review of systems": "review_of_systems",
    "ros": "review_of_systems",
    "social history": "social_history",
    "social hx": "social_history",
    "sh": "social_history",
}


def _normalize_section_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = value.strip()
    label = re.sub(r"^(?:bio|section|label)[-_]", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^[bieus][-_]", "", label, flags=re.IGNORECASE)
    normalized = normalize_section_header(label)
    return _SECTION_ALIASES.get(normalized, normalized or None)


_CUE_PATTERNS: tuple[tuple[str, re.Pattern[str], float], ...] = (
    (
        "assessment_and_plan",
        re.compile(r"\b(?:assessment\s*(?:and|/|&)\s*plan)\b", re.IGNORECASE),
        0.94,
    ),
    (
        "history_of_present_illness",
        re.compile(
            r"\b(?:history\s+of\s+present\s+illness|present\s+illness|hpi)\b",
            re.IGNORECASE,
        ),
        0.91,
    ),
    (
        "past_medical_history",
        re.compile(
            r"\b(?:past\s+medical\s+history|past\s+history|medical\s+history|pmh)\b",
            re.IGNORECASE,
        ),
        0.9,
    ),
    (
        "family_history",
        re.compile(
            r"\b(?:family\s+history|family\s+medical\s+history|fh)\b", re.IGNORECASE
        ),
        0.9,
    ),
    (
        "social_history",
        re.compile(r"\b(?:social\s+history|social\s+hx|sh)\b", re.IGNORECASE),
        0.9,
    ),
    (
        "review_of_systems",
        re.compile(r"\b(?:review\s+of\s+systems|ros)\b", re.IGNORECASE),
        0.88,
    ),
    (
        "chief_complaint",
        re.compile(r"\b(?:chief\s+complaint|cc)\b", re.IGNORECASE),
        0.88,
    ),
    (
        "medications",
        re.compile(
            r"\b(?:current\s+medications?|home\s+medications?|medication\s+list|medications?)\b",
            re.IGNORECASE,
        ),
        0.87,
    ),
    (
        "allergies",
        re.compile(r"\b(?:drug\s+allergies|allergy|allergies)\b", re.IGNORECASE),
        0.86,
    ),
    (
        "problem_list",
        re.compile(r"\b(?:problem\s+list|active\s+problems|problems)\b", re.IGNORECASE),
        0.86,
    ),
    (
        "assessment",
        re.compile(r"\bassessment\b", re.IGNORECASE),
        0.84,
    ),
    (
        "impression",
        re.compile(r"\bimpression\b", re.IGNORECASE),
        0.84,
    ),
    (
        "plan",
        re.compile(r"\b(?:plan\s+of\s+care|plan)\b", re.IGNORECASE),
        0.82,
    ),
)


def _fallback_predictions(text: str) -> tuple[dict[str, Any], ...]:
    matches: list[LearnedSectionCandidate] = []
    for label, pattern, confidence in _CUE_PATTERNS:
        for match in pattern.finditer(text):
            if _looks_like_inline_content(text, match.start(), match.end()):
                matches.append(
                    LearnedSectionCandidate(
                        label=label,
                        start=match.start(),
                        end=len(text),
                        confidence=confidence,
                        header_start=match.start(),
                        header_end=match.end(),
                        content_start=match.end(),
                    )
                )

    matches = _select_fallback_matches(matches)
    if not any(candidate.start == 0 for candidate in matches) and _looks_like_hpi(text):
        matches.insert(
            0,
            LearnedSectionCandidate(
                label="history_of_present_illness",
                start=0,
                end=len(text),
                confidence=0.72,
                header_start=0,
                header_end=1,
                content_start=0,
            ),
        )
    matches.sort(key=lambda candidate: (candidate.start, -candidate.confidence))
    return tuple(candidate.to_dict() for candidate in matches)


def _looks_like_inline_content(text: str, start: int, end: int) -> bool:
    before = text[:start]
    after = text[end:]
    if not before or not after:
        return True
    return any(char in after[:2] for char in ":,;-") or after[:1].isspace()


def _looks_like_hpi(text: str) -> bool:
    prefix = text[:240].casefold()
    return bool(
        re.search(
            r"\b(?:patient|reports?|presents?|complains?|seen\s+for|history)\b",
            prefix,
        )
    )


def _select_fallback_matches(
    candidates: Sequence[LearnedSectionCandidate],
) -> list[LearnedSectionCandidate]:
    ordered = sorted(candidates, key=lambda item: (item.start, -item.confidence))
    selected: list[LearnedSectionCandidate] = []
    for candidate in ordered:
        if selected and candidate.start == selected[-1].start:
            if candidate.confidence > selected[-1].confidence:
                selected[-1] = candidate
            continue
        if candidate.label == "plan" and any(
            previous.label == "assessment_and_plan"
            and 0 <= candidate.start - previous.start <= 48
            for previous in selected
        ):
            continue
        selected.append(candidate)
    return selected


__all__ = [
    "LEARNED_SOURCE",
    "SECTION_MODEL_ENV",
    "LearnedSectionCandidate",
    "LearnedSectionHead",
    "SectionHead",
    "SectionPredictor",
    "load_section_head",
    "predict_section_candidates",
]
