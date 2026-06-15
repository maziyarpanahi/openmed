"""PII extraction and de-identification for HIPAA compliance.

This module provides production-ready tools for detecting and redacting Personally
Identifiable Information (PII) from clinical notes, enabling HIPAA-compliant
processing of medical records.

Key Features:
    - Token-level PII detection for 18+ entity types
    - Multiple de-identification strategies (mask, remove, replace, hash, shift_dates)
    - HIPAA Safe Harbor method support
    - Reversible de-identification with secure mapping
    - Integration with OpenMed's existing NER infrastructure

Example:
    >>> from openmed import extract_pii, deidentify
    >>>
    >>> # Extract PII entities
    >>> result = extract_pii("Dr. Smith called John Doe at 555-1234")
    >>> for entity in result.entities:
    ...     print(f"{entity.label}: {entity.text}")
    NAME: Dr. Smith
    NAME: John Doe
    PHONE: 555-1234

    >>> # De-identify with masking
    >>> deid = deidentify(
    ...     "Patient John Doe (DOB: 01/15/1970) at 555-123-4567",
    ...     method="mask",
    ...     keep_year=True
    ... )
    >>> print(deid.deidentified_text)
    Patient [NAME] (DOB: [DATE]/1970) at [PHONE]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Literal, TYPE_CHECKING, Sequence
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path

from .config import OpenMedConfig
from ..processing.outputs import EntityPrediction

if TYPE_CHECKING:
    from .audit import AuditReport
    from .anonymizer import Anonymizer
    from .models import ModelLoader

# Type alias for de-identification methods
DeidentificationMethod = Literal["mask", "remove", "replace", "hash", "shift_dates"]


@dataclass
class PIIEntity(EntityPrediction):
    """Extended Entity with PII-specific metadata.

    Attributes:
        text: The entity text span
        label: PII category (NAME, EMAIL, PHONE, etc.)
        start: Character start position
        end: Character end position
        confidence: Model confidence score (0-1)
        entity_type: PII category (same as label)
        redacted_text: Replacement text after de-identification
        original_text: Original text before redaction
        hash_value: Consistent hash for entity linking
        reversible_id: Optional reversible pseudonymization handle
    """

    entity_type: str = ""
    redacted_text: Optional[str] = None
    original_text: Optional[str] = None
    hash_value: Optional[str] = None
    reversible_id: Optional[str] = None
    canonical_label: Optional[str] = None
    sources: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    action: Optional[str] = None
    surrogate: Optional[str] = None

    def __post_init__(self):
        """Initialize entity_type from label if not set."""
        if not self.entity_type:
            self.entity_type = self.label


@dataclass
class DeidentificationResult:
    """Result of de-identification operation.

    Attributes:
        original_text: Input text before de-identification
        deidentified_text: Output text with PII redacted
        pii_entities: List of detected and redacted PII entities
        method: De-identification method used
        timestamp: When de-identification was performed
        mapping: Optional mapping for re-identification (redacted -> original)
    """

    original_text: str
    deidentified_text: str
    pii_entities: list[PIIEntity]
    method: str
    timestamp: datetime
    mapping: Optional[dict[str, str]] = None
    metadata: Optional[dict[str, Any]] = None
    audit_report: Optional["AuditReport"] = None

    def to_dict(self) -> dict:
        """Convert result to dictionary format.

        Returns:
            Dictionary with all result fields and metadata
        """
        return {
            "original_text": self.original_text,
            "deidentified_text": self.deidentified_text,
            "pii_entities": [
                {
                    "text": e.text,
                    "label": e.label,
                    "entity_type": e.entity_type,
                    "start": e.start,
                    "end": e.end,
                    "confidence": e.confidence,
                    "redacted_text": e.redacted_text,
                    "canonical_label": e.canonical_label,
                    "sources": list(e.sources),
                    "evidence": e.evidence,
                    "threshold": e.threshold,
                    "action": e.action,
                    "surrogate": e.surrogate,
                    "metadata": e.metadata or {},
                    **(
                        {"reversible_id": e.reversible_id}
                        if e.reversible_id is not None
                        else {}
                    ),
                }
                for e in self.pii_entities
            ],
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
            "num_entities_redacted": len(self.pii_entities),
            "metadata": self.metadata or {},
            "audit_report": (
                self.audit_report.to_dict() if self.audit_report is not None else None
            ),
        }


# Languages whose PII models were trained on accent-free text.
# For these, input is automatically stripped of accents before model
# inference and entity positions are mapped back to the original text.
_ACCENT_NORMALIZE_LANGS = frozenset({"es"})

_DEFAULT_EN_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_DAY_FIRST_LANGS = frozenset({"fr", "de", "it", "es", "nl", "hi", "te", "pt", "ar", "tr"})
_PRIVACY_FILTER_FAMILY_ALIASES = frozenset({"openai-privacy-filter", "privacy-filter"})

# Repository-prefix allowlist for org/model identifiers that route through the
# privacy-filter dispatcher. The dispatcher loads these via Transformers'
# custom-code path (trust_remote_code), so only first-party orgs are matched.
# An identifier qualifies if it is exactly one of the prefixes (with any
# trailing hyphen stripped) or starts with the prefix. Untrusted names whose
# substring contains "privacy-filter" (e.g. attacker/foo-privacy-filter-bar)
# are intentionally NOT matched and fall through to the standard PII loader,
# which never enables trust_remote_code.
_TRUSTED_PRIVACY_FILTER_PREFIXES = (
    "openai/privacy-filter",
    "openmed/privacy-filter-",
)


def _normalize_model_family(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().lower().replace("_", "-")


def _looks_like_privacy_filter_identifier(value: Optional[str]) -> bool:
    normalized = _normalize_model_family(value)
    if not normalized:
        return False
    if normalized in _PRIVACY_FILTER_FAMILY_ALIASES:
        return True
    for prefix in _TRUSTED_PRIVACY_FILTER_PREFIXES:
        bare = prefix.rstrip("-")
        if normalized == bare or normalized.startswith(prefix):
            return True
    return False


@lru_cache(maxsize=32)
def _is_privacy_filter_artifact_path(model_name: str) -> bool:
    path = Path(model_name).expanduser()
    if path.is_file():
        path = path.parent

    if not path.exists() or not path.is_dir():
        return False

    for file_name in ("openmed-mlx.json", "config.json"):
        candidate = path / file_name
        if not candidate.is_file():
            continue

        try:
            payload = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        for key in ("family", "_mlx_family", "_mlx_model_type", "model_type", "source_model_id", "_name_or_path"):
            if _looks_like_privacy_filter_identifier(payload.get(key)):
                return True

    return False


def _uses_model_led_pii_merging(*model_identifiers: Optional[str]) -> bool:
    for identifier in model_identifiers:
        if _looks_like_privacy_filter_identifier(identifier):
            return True

    for identifier in model_identifiers:
        if identifier and _is_privacy_filter_artifact_path(identifier):
            return True

    return False


def _prediction_result_from_privacy_filter_raw(
    raw: Sequence[dict[str, Any]],
    text: str,
    *,
    model_name: str,
    confidence_threshold: float,
    original_text: str,
    do_normalize: bool,
):
    """Convert privacy-filter raw output into a PredictionResult."""
    from ..processing.outputs import EntityPrediction, PredictionResult

    entities: list[EntityPrediction] = []
    for item in raw:
        score = float(item.get("score", 0.0))
        if score < confidence_threshold:
            continue
        label = item.get("entity_group") or item.get("entity") or ""
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        # When accent normalization happened upstream, span indices match
        # the stripped text. The pipeline ran on ``text`` so spans align
        # with ``text``; remap to ``original_text`` if they're equal-length.
        span_text = (original_text if do_normalize else text)[start:end] if end > start else item.get("word", "")
        entities.append(
            EntityPrediction(
                text=span_text,
                label=label,
                start=start,
                end=end,
                confidence=score,
            )
        )

    return PredictionResult(
        text=original_text if do_normalize else text,
        entities=entities,
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def _coerce_batched_raw_outputs(
    raw_outputs: Any,
    expected_count: int,
) -> list[list[dict[str, Any]]]:
    """Normalize backend output for one or more input texts."""
    if expected_count == 0:
        return []

    if raw_outputs is None:
        return [[] for _ in range(expected_count)]

    if expected_count == 1:
        if isinstance(raw_outputs, list):
            if not raw_outputs:
                return [[]]
            if all(isinstance(item, dict) for item in raw_outputs):
                return [raw_outputs]
            if len(raw_outputs) == 1 and isinstance(raw_outputs[0], list):
                return [raw_outputs[0]]
        return [[raw_outputs]]

    if isinstance(raw_outputs, list) and len(raw_outputs) == expected_count:
        normalized: list[list[dict[str, Any]]] = []
        for item in raw_outputs:
            if item is None:
                normalized.append([])
            elif isinstance(item, list):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append([item])
            else:
                normalized.append(list(item) if item else [])
        return normalized

    raise ValueError(
        "Privacy-filter batch output length did not match input length "
        f"({expected_count})"
    )


def _extract_pii_via_privacy_filter(
    text: str,
    *,
    model_name: str,
    confidence_threshold: float,
    original_text: str,
    do_normalize: bool,
    pipeline: Optional[Any] = None,
):
    """Run privacy-filter inference via the MLX/Torch backend dispatcher.

    Returns a ``PredictionResult`` with the same shape callers expect from
    ``analyze_text``. Confidence filtering is applied here since the
    privacy-filter pipelines don't know the user's threshold.
    """
    if pipeline is None:
        from .backends import create_privacy_filter_pipeline

        pipeline = create_privacy_filter_pipeline(model_name)

    raw = pipeline(text)
    return _prediction_result_from_privacy_filter_raw(
        raw,
        text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        original_text=original_text,
        do_normalize=do_normalize,
    )


def _strip_accents(text: str) -> str:
    """Remove combining diacritical marks from *text*.

    The input is first NFC-normalised so that pre-composed characters like
    ``\u00e9`` are handled consistently.  After NFD decomposition every
    combining mark (Unicode category ``Mn``) is dropped and the result is
    NFC-normalised again.

    For common Latin-script accented characters this is a 1-to-1 character
    mapping, so ``len(result) == len(text)`` and character positions are
    preserved — critical for mapping model entity offsets back to the
    original text.

    Args:
        text: Arbitrary Unicode string.

    Returns:
        Accent-free copy with the same character count.
    """
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", nfc)
    stripped = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def _resolve_effective_pii_model(model_name: str, lang: str) -> str:
    """Validate language and resolve language-specific default PII model."""
    from .pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES

    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language '{lang}'. "
            f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if model_name == _DEFAULT_EN_MODEL and lang != "en":
        return DEFAULT_PII_MODELS[lang]
    return model_name


def _prepare_pii_text(
    text: str,
    *,
    lang: str,
    normalize_accents: Optional[bool],
) -> tuple[str, str, bool]:
    """Return original stripped text, inference text, and normalization flag."""
    do_normalize = (
        normalize_accents
        if normalize_accents is not None
        else (lang in _ACCENT_NORMALIZE_LANGS)
    )

    original_text = text.strip()
    inference_text = _strip_accents(original_text) if do_normalize else original_text
    return original_text, inference_text, do_normalize


def _apply_pii_smart_merging(result: Any, effective_model: str, lang: str) -> None:
    """Apply semantic-unit PII merging in place."""
    from .pii_entity_merger import merge_entities_with_semantic_units
    from .pii_i18n import get_patterns_for_language
    from ..processing.outputs import EntityPrediction

    lang_patterns = get_patterns_for_language(lang)
    entity_dicts = [
        {
            "entity_type": e.label,
            "score": e.confidence,
            "start": e.start,
            "end": e.end,
            "word": e.text,
        }
        for e in result.entities
    ]

    model_led_merging = _uses_model_led_pii_merging(
        effective_model,
        getattr(result, "model_name", None),
    )

    merged_dicts = merge_entities_with_semantic_units(
        entity_dicts,
        result.text,
        patterns=lang_patterns,
        use_semantic_patterns=True,
        prefer_model_labels=True,
        allow_semantic_only_matches=not model_led_merging,
        allow_label_expansion=not model_led_merging,
    )

    result.entities = [
        EntityPrediction(
            text=e["word"],
            label=e["entity_type"],
            start=e["start"],
            end=e["end"],
            confidence=e["score"],
        )
        for e in merged_dicts
    ]
    result.num_entities = len(result.entities)


def _extract_pii_batch(
    texts: Sequence[str],
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    *,
    loader: Optional["ModelLoader"] = None,
    privacy_filter_pipeline: Optional[Any] = None,
    **pipeline_kwargs: Any,
) -> list[Any]:
    """Extract PII for multiple texts while reusing the same backend resources."""
    effective_model = _resolve_effective_pii_model(model_name, lang)
    prepared = [
        _prepare_pii_text(
            text,
            lang=lang,
            normalize_accents=normalize_accents,
        )
        for text in texts
    ]

    if not prepared:
        return []

    uses_privacy_filter = (
        _looks_like_privacy_filter_identifier(effective_model)
        or _is_privacy_filter_artifact_path(effective_model)
    )

    if uses_privacy_filter:
        from .backends import create_privacy_filter_pipeline

        pipeline = privacy_filter_pipeline or create_privacy_filter_pipeline(
            effective_model
        )
        inference_texts = [item[1] for item in prepared]
        privacy_call_kwargs = {
            key: pipeline_kwargs[key]
            for key in ("batch_size", "num_workers")
            if key in pipeline_kwargs and pipeline_kwargs[key] is not None
        }
        raw_outputs = pipeline(inference_texts, **privacy_call_kwargs)
        batched_raw = _coerce_batched_raw_outputs(raw_outputs, len(prepared))
        results = [
            _prediction_result_from_privacy_filter_raw(
                raw,
                inference_text,
                model_name=effective_model,
                confidence_threshold=confidence_threshold,
                original_text=original_text,
                do_normalize=do_normalize,
            )
            for raw, (original_text, inference_text, do_normalize) in zip(
                batched_raw, prepared
            )
        ]
    else:
        from .. import analyze_text
        from .models import ModelLoader

        shared_loader = loader
        if shared_loader is None and len(prepared) > 1:
            shared_loader = ModelLoader(config)
        results = []
        for original_text, inference_text, do_normalize in prepared:
            result = analyze_text(
                inference_text,
                model_name=effective_model,
                confidence_threshold=confidence_threshold,
                config=config,
                loader=shared_loader,
                group_entities=True,
                **pipeline_kwargs,
            )

            if do_normalize and original_text != inference_text:
                result.text = original_text
                result.entities = [
                    EntityPrediction(
                        text=original_text[e.start:e.end],
                        label=e.label,
                        start=e.start,
                        end=e.end,
                        confidence=e.confidence,
                    )
                    for e in result.entities
                ]

            results.append(result)

    if use_smart_merging and not uses_privacy_filter:
        for result in results:
            _apply_pii_smart_merging(result, effective_model, lang)

    from .quality_gates import validate_entity_spans

    for result in results:
        validate_entity_spans(result.entities, result.text)

    return results


def extract_pii(
    text: str,
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    *,
    loader: Optional["ModelLoader"] = None,
):
    """Extract PII entities from text with intelligent entity merging.

    Uses token classification models to detect personally identifiable information
    including names, emails, phone numbers, addresses, and other HIPAA-protected
    identifiers.

    The smart merging feature uses regex patterns to identify semantic units
    (dates, SSN, phone numbers, etc.) and merges fragmented model predictions
    into complete entities with dominant label selection.

    Args:
        text: Input text to analyze
        model_name: PII detection model (registry key or HuggingFace ID).
            When the default is used and ``lang`` is not ``"en"``, the
            language-appropriate default model is selected automatically.
        confidence_threshold: Minimum confidence score (0-1)
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)
        lang: ISO 639-1 language code (en, fr, de, it, es, nl, hi, te, pt,
            ar, ja, tr). Controls which
            default model and regex patterns are used.
        normalize_accents: Strip diacritical marks before model inference so
            that models trained on accent-free text still detect accented
            names.  Entity spans in the result reference the *original*
            (accented) text.  ``None`` (default) auto-enables for languages
            in ``_ACCENT_NORMALIZE_LANGS`` (currently Spanish).
        loader: Optional shared model loader to reuse warmed pipelines.

    Returns:
        AnalysisResult with detected PII entities

    Example:
        >>> result = extract_pii("DOB: 01/15/1970, SSN: 123-45-6789")
        >>> for entity in result.entities:
        ...     print(f"{entity.label}: {entity.text}")
        date_of_birth: 01/15/1970
        ssn: 123-45-6789

        >>> # French PII detection
        >>> result = extract_pii("Né le 15/01/1970", lang="fr")
    """
    return _extract_pii_batch(
        [text],
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        loader=loader,
    )[0]


def _resolve_deidentification_method(
    method: DeidentificationMethod,
    shift_dates: Optional[bool],
    date_shift_days: Optional[int],
) -> DeidentificationMethod:
    """Resolve method aliases and validate date-shift-only parameters."""
    effective_method = method
    if shift_dates is True and method != "shift_dates":
        effective_method = "shift_dates"
    elif shift_dates is False and method == "shift_dates":
        raise ValueError("shift_dates=false conflicts with method='shift_dates'")

    if date_shift_days is not None and effective_method != "shift_dates":
        raise ValueError("date_shift_days requires method='shift_dates'")

    return effective_method


def _copy_metadata(metadata: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None
    return dict(metadata)


def _apply_safety_sweep_to_result(
    text: str,
    pii_result: Any,
    *,
    lang: str,
) -> int:
    """Run the deterministic sweep and record its net span contribution."""
    from .safety_sweep import (
        SAFETY_SWEEP_PATTERNS_VERSION,
        SAFETY_SWEEP_SOURCE,
        safety_sweep,
    )
    from .quality_gates import validate_entity_spans

    before_count = len(pii_result.entities)
    pii_result.entities = safety_sweep(text, pii_result.entities, lang=lang)
    added_count = len(pii_result.entities) - before_count

    metadata = dict(getattr(pii_result, "metadata", None) or {})
    metadata["safety_sweep"] = {
        "source": SAFETY_SWEEP_SOURCE,
        "patterns_version": SAFETY_SWEEP_PATTERNS_VERSION,
        "spans_added": added_count,
    }
    pii_result.metadata = metadata
    pii_result.num_entities = len(pii_result.entities)
    validate_entity_spans(pii_result.entities, text)
    return added_count


_AUDIT_TEXT_KEYS = {
    "text",
    "word",
    "value",
    "surface",
    "replacement",
    "original_text",
    "deidentified_text",
}


def _sanitize_audit_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _sanitize_audit_evidence(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key).lower() not in _AUDIT_TEXT_KEYS
        }
    if isinstance(value, list):
        return [_sanitize_audit_evidence(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_audit_evidence(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _entity_sources(entity: Any) -> list[str]:
    metadata = getattr(entity, "metadata", None) or {}
    source = str(metadata.get("source", "")).strip()
    if source == "safety_sweep":
        return ["safety_sweep"]
    if source == "locale_rule":
        return ["locale_rule"]
    return ["ml"]


def _entity_evidence(
    entity: Any,
    *,
    pii_result: Any,
    model_name: str,
    lang: str,
) -> dict[str, Any]:
    metadata = _sanitize_audit_evidence(getattr(entity, "metadata", None) or {})
    evidence = {
        "raw_label": getattr(entity, "label", ""),
        "language": lang,
        "metadata": metadata,
    }
    if "ml" in _entity_sources(entity):
        evidence["model_id"] = getattr(pii_result, "model_name", None) or model_name
    return evidence


def _model_format(model_id: str) -> str:
    if not model_id:
        return "unknown"
    lowered = model_id.lower()
    if lowered.endswith(".mlmodel") or "coreml" in lowered:
        return "coreml"
    if "mlx" in lowered or _is_privacy_filter_artifact_path(model_id):
        return "mlx"
    if _looks_like_privacy_filter_identifier(model_id):
        return "privacy_filter"
    return "transformers"


def _detector_infos(
    pii_result: Any,
    *,
    model_name: str,
    use_safety_sweep: bool,
) -> list[Any]:
    from .audit import DetectorInfo
    from .safety_sweep import SAFETY_SWEEP_PATTERNS_VERSION

    metadata = getattr(pii_result, "metadata", None) or {}
    result_model = str(getattr(pii_result, "model_name", None) or model_name)
    detectors = [
        DetectorInfo(
            source="ml",
            model_id=result_model,
            model_format=_model_format(result_model),
            commit=metadata.get("model_commit"),
        )
    ]
    if use_safety_sweep:
        detectors.append(
            DetectorInfo(
                source="safety_sweep",
                model_id="safety_sweep",
                model_format="rules",
                commit=None,
                metadata={"patterns_version": SAFETY_SWEEP_PATTERNS_VERSION},
            )
        )
    return detectors


def _context_window(text: str, start: int, end: int, *, size: int = 32) -> dict[str, str]:
    return {
        "before": text[max(0, start - size):start],
        "after": text[end:min(len(text), end + size)],
    }


def _audit_span_from_entity(text: str, entity: PIIEntity) -> Any:
    from .audit import AuditSpan, hash_text

    start = int(entity.start or 0)
    end = int(entity.end or start)
    return AuditSpan(
        start=start,
        end=end,
        label=entity.label,
        canonical_label=entity.canonical_label or entity.entity_type,
        sources=list(entity.sources),
        confidence=float(entity.confidence),
        threshold=float(entity.threshold or 0.0),
        action=entity.action or "",
        surrogate=entity.surrogate,
        text_hash=hash_text(text[start:end]),
        evidence=dict(entity.evidence),
        context=_context_window(text, start, end),
    )


def _span_record(entity: PIIEntity) -> dict[str, Any]:
    return {
        "label": entity.label,
        "canonical_label": entity.canonical_label or entity.entity_type,
        "start": entity.start,
        "end": entity.end,
        "metadata": {
            "source": list(entity.sources),
            **(entity.metadata or {}),
        },
    }


def _projected_leakage(spans: Sequence[PIIEntity]) -> float:
    if not spans:
        return 0.0
    residual = sum(max(0.0, 1.0 - float(span.confidence)) for span in spans)
    return round(min(1.0, residual / len(spans)), 6)


def _risk_summary(
    *,
    original_text: str,
    deidentified_text: str,
    spans: Sequence[PIIEntity],
) -> dict[str, Any]:
    from ..risk import risk_report

    report = risk_report(
        {"text": deidentified_text},
        original={
            "text": original_text,
            "entities": [_span_record(entity) for entity in spans],
        },
    )
    record_score = max(
        float(report.get("leakage_rate", 0.0)),
        float(report.get("reid_rate", 0.0)),
    )
    if report.get("k_min") == 1:
        record_score = max(record_score, 1.0)
    return {
        "projected_leakage": _projected_leakage(spans),
        "risk_report_record_score": round(record_score, 6),
        "risk_report": report,
    }


def _resolved_audit_profile(
    *,
    method: DeidentificationMethod,
    model_name: str,
    confidence_threshold: float,
    keep_year: bool,
    keep_mapping: bool,
    lang: str,
    normalize_accents: Optional[bool],
    use_smart_merging: bool,
    use_safety_sweep: bool,
) -> dict[str, Any]:
    return {
        "method": method,
        "model_name": model_name,
        "confidence_threshold": float(confidence_threshold),
        "keep_year": bool(keep_year),
        "keep_mapping": bool(keep_mapping),
        "language": lang,
        "normalize_accents": normalize_accents,
        "use_smart_merging": bool(use_smart_merging),
        "use_safety_sweep": bool(use_safety_sweep),
    }


def _build_audit_report(
    *,
    original_text: str,
    deidentified_text: str,
    pii_result: Any,
    pii_entities: Sequence[PIIEntity],
    effective_method: DeidentificationMethod,
    model_name: str,
    confidence_threshold: float,
    keep_year: bool,
    keep_mapping: bool,
    lang: str,
    normalize_accents: Optional[bool],
    use_smart_merging: bool,
    use_safety_sweep: bool,
    policy: str,
) -> Any:
    from .audit import AuditReport, hash_text, manifest_hash
    from .safety_sweep import SAFETY_SWEEP_PATTERNS_VERSION, SAFETY_SWEEP_SOURCE
    from .labels import normalize_label
    from ..__about__ import __version__

    thresholds = {
        normalize_label(entity.label, lang): float(confidence_threshold)
        for entity in pii_entities
    }
    if not thresholds:
        thresholds["DEFAULT"] = float(confidence_threshold)

    metadata = getattr(pii_result, "metadata", None) or {}
    sweep_metadata = dict(metadata.get("safety_sweep") or {})
    sweep_metadata.setdefault("source", SAFETY_SWEEP_SOURCE)
    sweep_metadata.setdefault("patterns_version", SAFETY_SWEEP_PATTERNS_VERSION)
    sweep_metadata.setdefault("enabled", bool(use_safety_sweep))

    return AuditReport(
        policy=policy,
        resolved_profile=_resolved_audit_profile(
            method=effective_method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            keep_year=keep_year,
            keep_mapping=keep_mapping,
            lang=lang,
            normalize_accents=normalize_accents,
            use_smart_merging=use_smart_merging,
            use_safety_sweep=use_safety_sweep,
        ),
        detectors=_detector_infos(
            pii_result,
            model_name=model_name,
            use_safety_sweep=use_safety_sweep,
        ),
        safety_sweep=sweep_metadata,
        spans=[_audit_span_from_entity(original_text, entity) for entity in pii_entities],
        thresholds=thresholds,
        residual_risk=_risk_summary(
            original_text=original_text,
            deidentified_text=deidentified_text,
            spans=pii_entities,
        ),
        openmed_version=__version__,
        manifest_hash=manifest_hash(),
        document_length=len(original_text),
        input_hash=hash_text(original_text),
        deidentified_text_hash=hash_text(deidentified_text),
    )


def _build_deidentification_result(
    text: str,
    pii_result: Any,
    *,
    effective_method: DeidentificationMethod,
    keep_year: bool,
    date_shift_days: Optional[int],
    keep_mapping: bool,
    lang: str,
    consistent: bool,
    seed: Optional[int],
    locale: Optional[str],
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.7,
    normalize_accents: Optional[bool] = None,
    use_smart_merging: bool = True,
    use_safety_sweep: bool = True,
    reversible_ids: bool = False,
    policy_name: Optional[str] = None,
    policy: str = "hipaa_safe_harbor",
    audit: bool = False,
) -> DeidentificationResult:
    """Build a de-identification result from an existing PII result."""
    from .labels import normalize_label
    from .quality_gates import resolve_overlapping_entities

    resolved_entities = resolve_overlapping_entities(list(pii_result.entities))
    pii_result.entities = resolved_entities
    if hasattr(pii_result, "num_entities"):
        pii_result.num_entities = len(resolved_entities)

    pii_entities = [
        PIIEntity(
            text=e.text,
            label=e.label,
            start=e.start,
            end=e.end,
            confidence=e.confidence,
            metadata=_copy_metadata(getattr(e, "metadata", None)),
            entity_type=e.label,
            original_text=e.text,
            canonical_label=normalize_label(e.label, lang),
            sources=_entity_sources(e),
            evidence=_entity_evidence(
                e,
                pii_result=pii_result,
                model_name=model_name,
                lang=lang,
            ),
            threshold=confidence_threshold,
        )
        for e in pii_result.entities
    ]

    redaction_entities = sorted(pii_entities, key=lambda e: e.start, reverse=True)

    if effective_method == "shift_dates" and date_shift_days is None:
        date_shift_days = random.randint(-365, 365)

    anonymizer = None
    if effective_method == "replace" or any(
        _entity_policy_action(entity) == "replace" for entity in pii_entities
    ):
        from .anonymizer import Anonymizer

        effective_consistent = consistent or seed is not None
        anonymizer = Anonymizer(
            lang=lang,
            locale=locale,
            consistent=effective_consistent,
            seed=seed,
        )

    deidentified = text
    mapping = {} if keep_mapping else None

    for entity in redaction_entities:
        entity_method = _entity_redaction_method(entity, effective_method)
        redacted = _redact_entity(
            entity,
            entity_method,
            keep_year=keep_year,
            date_shift_days=(
                date_shift_days if entity_method == "shift_dates" else None
            ),
            lang=lang,
            anonymizer=anonymizer,
        )
        entity.redacted_text = redacted
        if reversible_ids:
            entity.reversible_id = _build_reversible_id(
                entity,
                policy_name=policy_name,
            )
        entity.action = entity_method
        entity.surrogate = redacted if redacted else None

        deidentified = (
            deidentified[: entity.start] + redacted + deidentified[entity.end :]
        )

        if keep_mapping and mapping is not None:
            mapping[redacted] = entity.original_text or entity.text

    audit_report = None
    if audit:
        audit_report = _build_audit_report(
            original_text=text,
            deidentified_text=deidentified,
            pii_result=pii_result,
            pii_entities=pii_entities,
            effective_method=effective_method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            keep_year=keep_year,
            keep_mapping=keep_mapping,
            lang=lang,
            normalize_accents=normalize_accents,
            use_smart_merging=use_smart_merging,
            use_safety_sweep=use_safety_sweep,
            policy=policy,
        )

    return DeidentificationResult(
        original_text=text,
        deidentified_text=deidentified,
        pii_entities=pii_entities,
        method=effective_method,
        timestamp=datetime.now(),
        mapping=mapping,
        metadata=_copy_metadata(getattr(pii_result, "metadata", None)),
        audit_report=audit_report,
    )


def _entity_policy_action(entity: PIIEntity) -> Optional[str]:
    metadata = entity.metadata or {}
    policy_action = metadata.get("policy_action")
    if not isinstance(policy_action, dict):
        return None
    action = policy_action.get("action")
    return str(action) if action is not None else None


def _entity_redaction_method(
    entity: PIIEntity,
    fallback: DeidentificationMethod,
) -> DeidentificationMethod:
    action = _entity_policy_action(entity)
    if action == "replace":
        return "replace"
    if action == "hash":
        return "hash"
    if action in {"mask", "redact"}:
        return "mask"
    return fallback


def _build_reversible_id(
    entity: PIIEntity,
    *,
    policy_name: Optional[str],
) -> str:
    original = entity.original_text or entity.text
    material = (
        f"{policy_name or 'policy'}|{entity.entity_type}|"
        f"{entity.start}|{entity.end}|{original}"
    )
    return "rev_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _deidentify_batch(
    texts: Sequence[str],
    method: DeidentificationMethod = "mask",
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.7,
    keep_year: bool = True,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    use_safety_sweep: bool = True,
    *,
    consistent: bool = False,
    seed: Optional[int] = None,
    locale: Optional[str] = None,
    loader: Optional["ModelLoader"] = None,
    privacy_filter_pipeline: Optional[Any] = None,
    **pipeline_kwargs: Any,
) -> list[DeidentificationResult]:
    """De-identify multiple texts after one batched PII extraction pass."""
    effective_method = _resolve_deidentification_method(
        method,
        shift_dates,
        date_shift_days,
    )
    stripped_texts = [text.strip() for text in texts]
    pii_results = _extract_pii_batch(
        stripped_texts,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        loader=loader,
        privacy_filter_pipeline=privacy_filter_pipeline,
        **pipeline_kwargs,
    )

    if use_safety_sweep:
        for stripped_text, pii_result in zip(stripped_texts, pii_results):
            _apply_safety_sweep_to_result(stripped_text, pii_result, lang=lang)

    return [
        _build_deidentification_result(
            text,
            pii_result,
            effective_method=effective_method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            keep_year=keep_year,
            date_shift_days=date_shift_days,
            keep_mapping=keep_mapping,
            lang=lang,
            normalize_accents=normalize_accents,
            use_smart_merging=use_smart_merging,
            use_safety_sweep=use_safety_sweep,
            consistent=consistent,
            seed=seed,
            locale=locale,
            policy="hipaa_safe_harbor",
        )
        for text, pii_result in zip(stripped_texts, pii_results)
    ]


def deidentify(
    text: str,
    method: DeidentificationMethod = "mask",
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.7,  # Higher threshold for safety
    keep_year: bool = True,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    use_safety_sweep: bool = True,
    *,
    consistent: bool = False,
    seed: Optional[int] = None,
    locale: Optional[str] = None,
    loader: Optional["ModelLoader"] = None,
    policy: Optional[str] = None,
    audit: bool = False,
) -> DeidentificationResult | "AuditReport":
    """De-identify text by detecting and redacting PII with intelligent merging.

    Implements multiple de-identification strategies for HIPAA compliance:

    - **mask**: Replace with placeholders like [NAME], [EMAIL], etc.
    - **remove**: Remove PII text entirely (empty string)
    - **replace**: Replace with fake but realistic data
    - **hash**: Replace with consistent hashed values for entity linking
    - **shift_dates**: Shift dates by random offset while preserving intervals

    Smart merging uses regex patterns to merge fragmented entities (e.g., dates
    split into '01' and '/15/1970' are merged into complete '01/15/1970').

    Args:
        text: Input text to de-identify
        method: De-identification method (mask, remove, replace, hash, shift_dates)
        model_name: PII detection model
        confidence_threshold: Minimum confidence for redaction (default 0.7 for safety)
        keep_year: For dates, keep the year unchanged
        shift_dates: Deprecated alias for ``method="shift_dates"``.
        date_shift_days: Specific number of days to shift (random if None)
        keep_mapping: Keep mapping for re-identification
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)
        use_safety_sweep: Run a deterministic structured-identifier sweep
            after model detection and before redaction.
        lang: ISO 639-1 language code (en, fr, de, it, es, nl, hi, te, pt,
            ar, ja, tr). Controls model
            selection, regex patterns, and fake data for replacement.
        normalize_accents: Strip diacritical marks before model inference.
            ``None`` (default) auto-enables for Spanish.
        loader: Optional shared model loader to reuse warmed pipelines.
        consistent: When ``method="replace"``, generate stable surrogates
            (same input -> same surrogate within the call). Lets repeated
            mentions of the same name resolve to one fake identity instead
            of a different one each time.
        seed: Optional integer seed for cross-run reproducibility of
            ``consistent=True`` replacements. Implies ``consistent=True``.
        locale: Faker locale override (``pt_BR``, ``en_GB``, ...) for
            ``method="replace"``. When ``None``, derived from ``lang``.
        policy: Optional policy profile name controlling arbitration, action
            selection, mandatory safety sweep behavior, and reversible mapping.
        audit: Return a deterministic AuditReport instead of the
            DeidentificationResult.

    Returns:
        DeidentificationResult with original and de-identified text, or
        AuditReport when ``audit=True``.

    Example:
        >>> result = deidentify(
        ...     "Patient John Doe (DOB: 01/15/1970) called from 555-1234",
        ...     method="mask",
        ...     keep_year=True
        ... )
        >>> print(result.deidentified_text)
        Patient [NAME] (DOB: [DATE]/1970) called from [PHONE]

        >>> result = deidentify(text, method="replace", lang="de")
        >>> result = deidentify(text, method="replace", lang="pt",
        ...                    locale="pt_BR", consistent=True, seed=42)
    """
    from .pipeline import Pipeline

    pipeline = Pipeline(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        use_safety_sweep=use_safety_sweep,
        loader=loader,
        policy=policy,
    )
    result = pipeline.run(
        text,
        method=method,
        keep_year=keep_year,
        shift_dates=shift_dates,
        date_shift_days=date_shift_days,
        keep_mapping=keep_mapping,
        consistent=consistent,
        seed=seed,
        locale=locale,
        audit=audit,
    )
    if audit and result.deidentification_result.audit_report is not None:
        return result.deidentification_result.audit_report
    return result.deidentification_result


def _redact_entity(
    entity: PIIEntity,
    method: DeidentificationMethod,
    keep_year: bool = True,
    date_shift_days: Optional[int] = None,
    lang: str = "en",
    anonymizer: Optional["Anonymizer"] = None,
) -> str:
    """Redact a single PII entity based on method.

    Args:
        entity: PIIEntity to redact
        method: De-identification method
        keep_year: Keep year in dates
        date_shift_days: Days to shift dates
        lang: Language code for fake data and date formatting
        anonymizer: Pre-built ``Anonymizer`` instance for ``method="replace"``.
            When ``None``, a fresh per-call instance is built using the
            language default (random, non-deterministic).

    Returns:
        Redacted text replacement
    """
    if method == "mask":
        # Replace with placeholder
        return f"[{entity.entity_type}]"

    elif method == "remove":
        # Remove entirely (replace with empty string)
        return ""

    elif method == "replace":
        if anonymizer is not None:
            return anonymizer.surrogate(
                entity.original_text or entity.text,
                entity.entity_type,
                lang=lang,
            )
        return _generate_fake_pii(entity.entity_type, lang=lang)

    elif method == "hash":
        # Generate consistent hash
        hash_val = hashlib.sha256(entity.text.encode()).hexdigest()[:8]
        entity.hash_value = hash_val
        return f"{entity.entity_type}_{hash_val}"

    elif method == "shift_dates":
        # Shift dates by offset
        if entity.entity_type == "DATE" and date_shift_days is not None:
            return _shift_date(entity.text, date_shift_days, keep_year, lang=lang)
        else:
            # Non-date entities get masked
            return f"[{entity.entity_type}]"

    return entity.text


_LABEL_TO_FAKE_KEY: Dict[str, str] = {
    # Name variants
    "first_name": "FIRST_NAME",
    "FIRSTNAME": "FIRST_NAME",
    "firstname": "FIRST_NAME",
    "last_name": "LAST_NAME",
    "LASTNAME": "LAST_NAME",
    "lastname": "LAST_NAME",
    "name": "NAME",
    "NAME": "NAME",
    "patient": "NAME",
    "PATIENT": "NAME",
    "doctor": "NAME",
    "DOCTOR": "NAME",

    # Phone variants
    "phone_number": "PHONE",
    "PHONE": "PHONE",
    "phone": "PHONE",
    "PHONENUMBER": "PHONE",

    # Location variants
    "city": "LOCATION",
    "CITY": "LOCATION",
    "state": "LOCATION",
    "STATE": "LOCATION",
    "country": "LOCATION",
    "COUNTRY": "LOCATION",
    "location": "LOCATION",
    "LOCATION": "LOCATION",

    # Address variants
    "street_address": "STREET_ADDRESS",
    "STREET": "STREET_ADDRESS",
    "street": "STREET_ADDRESS",
    "STREETADDRESS": "STREET_ADDRESS",
    "address": "STREET_ADDRESS",
    "ADDRESS": "STREET_ADDRESS",

    # Date variants
    "date": "DATE",
    "DATE": "DATE",
    "date_of_birth": "DATE",
    "DATEOFBIRTH": "DATE",
    "dateofbirth": "DATE",
    "dob": "DATE",
    "DOB": "DATE",

    # ID variants
    "id_num": "ID_NUM",
    "ID_NUM": "ID_NUM",
    "ssn": "ID_NUM",
    "SSN": "ID_NUM",
    "national_id": "ID_NUM",
    "NATIONAL_ID": "ID_NUM",
    "cpf": "ID_NUM",
    "CPF": "ID_NUM",
    "cnpj": "ID_NUM",
    "CNPJ": "ID_NUM",
    "medical_record_number": "ID_NUM",
    "MEDICAL_RECORD_NUMBER": "ID_NUM",

    # Other
    "email": "EMAIL",
    "EMAIL": "EMAIL",
    "age": "AGE",
    "AGE": "AGE",
    "username": "USERNAME",
    "USERNAME": "USERNAME",
    "url_personal": "URL_PERSONAL",
    "URL_PERSONAL": "URL_PERSONAL",
    "zipcode": "ZIPCODE",
    "ZIPCODE": "ZIPCODE",
    "zip": "ZIPCODE",
    "ZIP": "ZIPCODE",
    "postal_code": "ZIPCODE",
}


# Map canonical taxonomy (from openmed.core.labels) to LANGUAGE_FAKE_DATA keys.
# Canonical labels that don't have a fake-data key fall through to the
# placeholder, the same as labels that aren't mapped at all.
_CANONICAL_TO_FAKE_KEY: Dict[str, str] = {
    "PERSON": "NAME",
    "FIRST_NAME": "FIRST_NAME",
    "LAST_NAME": "LAST_NAME",
    "MIDDLE_NAME": "FIRST_NAME",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "LOCATION": "LOCATION",
    "STREET_ADDRESS": "STREET_ADDRESS",
    "DATE": "DATE",
    "DATE_OF_BIRTH": "DATE",
    "ID_NUM": "ID_NUM",
    "SSN": "ID_NUM",
    "ACCOUNT_NUMBER": "ID_NUM",
    "AGE": "AGE",
    "USERNAME": "USERNAME",
    "URL": "URL_PERSONAL",
    "ZIPCODE": "ZIPCODE",
}


def _resolve_fake_data_key(entity_type: str, lang: str = "en") -> str:
    """Resolve a model entity label to a LANGUAGE_FAKE_DATA key.

    Tries the legacy ``_LABEL_TO_FAKE_KEY`` table first to preserve exact
    behavior for labels it already covers. Labels outside the legacy table
    fall through to the canonical taxonomy in :mod:`openmed.core.labels`,
    which covers Portuguese UPPERCASE labels and BIOES-tagged privacy-filter
    labels too.
    """
    direct = _LABEL_TO_FAKE_KEY.get(entity_type)
    if direct is not None:
        return direct

    from .labels import normalize_label
    canonical = normalize_label(entity_type, lang)
    return _CANONICAL_TO_FAKE_KEY.get(canonical, entity_type.upper())


def _generate_fake_pii(entity_type: str, lang: str = "en") -> str:
    """Generate fake but realistic PII data.

    Args:
        entity_type: Type of PII entity
        lang: Language code for language-appropriate fake data

    Returns:
        Fake replacement text
    """
    from .pii_i18n import LANGUAGE_FAKE_DATA

    fake_data = LANGUAGE_FAKE_DATA.get(lang, LANGUAGE_FAKE_DATA["en"])
    key = _resolve_fake_data_key(entity_type, lang)

    if key in fake_data:
        return random.choice(fake_data[key])

    # Fall back to English if the entity type isn't in the language-specific data
    en_data = LANGUAGE_FAKE_DATA["en"]
    if key in en_data:
        return random.choice(en_data[key])

    return f"[{entity_type}]"


def _parse_localized_month_date(
    date_str: str, lang: str,
) -> tuple[datetime, str] | None:
    """Parse localized month-name dates that dateutil may not understand."""
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    month_names = LANGUAGE_MONTH_NAMES.get(lang)
    if not month_names:
        return None

    month_alts = "|".join(re.escape(name) for name in month_names)
    text = date_str.strip()

    if lang in {"es", "pt"}:
        pattern = rf"^(?P<day>\d{{1,2}})\s+de\s+(?P<month>{month_alts})\s+de\s+(?P<year>\d{{4}})$"
        style = "day_month_year_de"
    elif lang == "de":
        pattern = rf"^(?P<day>\d{{1,2}})(?P<dot>\.)?\s+(?P<month>{month_alts})\s+(?P<year>\d{{4}})$"
        style = "day_month_year_dot"
    else:
        pattern = rf"^(?P<day>\d{{1,2}})\s+(?P<month>{month_alts})\s+(?P<year>\d{{4}})$"
        style = "day_month_year"

    match = re.match(pattern, text, re.IGNORECASE)
    if not match:
        return None

    month_lookup = {name.casefold(): index + 1 for index, name in enumerate(month_names)}
    month_name = match.group("month").casefold()
    month = month_lookup.get(month_name)
    if month is None:
        return None

    try:
        parsed = datetime(
            int(match.group("year")),
            month,
            int(match.group("day")),
        )
    except ValueError:
        return None

    if lang == "de" and match.groupdict().get("dot"):
        style = "day_month_year_dot"
    return parsed, style


def _format_localized_month_date(
    new_date: datetime, lang: str, style: str,
) -> str:
    """Render a localized month-name date using the language month table."""
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    month_name = LANGUAGE_MONTH_NAMES.get(lang, LANGUAGE_MONTH_NAMES["en"])[new_date.month - 1]

    if style == "day_month_year_de":
        return f"{new_date.day} de {month_name} de {new_date.year}"
    if style == "day_month_year_dot":
        return f"{new_date.day}. {month_name} {new_date.year}"
    return f"{new_date.day} {month_name} {new_date.year}"


def _replace_year_safe(date_value: datetime, year: int) -> datetime:
    """Return ``date_value`` with its year set to ``year``.

    ``datetime.replace(year=...)`` raises ``ValueError`` for Feb 29 when the
    target year is not a leap year. Clamp to Feb 28 in that case so keep_year
    date shifting degrades gracefully instead of falling through to the
    ``[DATE_SHIFTED]`` placeholder.
    """
    try:
        return date_value.replace(year=year)
    except ValueError:
        # The only date that can fail here is Feb 29 -> a non-leap year.
        return date_value.replace(year=year, month=2, day=28)


def _shift_date(
    date_str: str, shift_days: int, keep_year: bool = True, lang: str = "en",
) -> str:
    """Shift a date string by specified number of days.

    Supports multiple date formats commonly found in clinical documents:
    - MM/DD/YYYY, MM-DD-YYYY (US/English)
    - DD/MM/YYYY, DD-MM-YYYY (French/Italian)
    - DD.MM.YYYY (German)
    - YYYY-MM-DD (ISO)
    - Month DD, YYYY / DD Month YYYY (with localized month names)

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift (positive = future, negative = past)
        keep_year: Keep the year unchanged (only shift month/day)
        lang: Language code for date format conventions

    Returns:
        Shifted date string in the same format as input
    """
    localized = _parse_localized_month_date(date_str, lang)
    if localized is not None:
        try:
            parsed_date, localized_style = localized
            original_year = parsed_date.year
            shifted_date = parsed_date + timedelta(days=shift_days)

            if keep_year:
                shifted_date = _replace_year_safe(shifted_date, original_year)

            return _format_localized_month_date(shifted_date, lang, localized_style)
        except (ValueError, OverflowError):
            return "[DATE_SHIFTED]"

    # Try to parse and shift using dateutil if available
    try:
        from dateutil import parser as date_parser
    except ImportError:
        # Fallback without dateutil - basic pattern matching
        return _shift_date_basic(date_str, shift_days, keep_year, lang=lang)

    try:
        # For European languages, try day-first parsing
        dayfirst = lang in _DAY_FIRST_LANGS
        parsed_date = date_parser.parse(date_str, fuzzy=False, dayfirst=dayfirst)
        original_year = parsed_date.year

        # Shift the date
        shifted_date = parsed_date + timedelta(days=shift_days)

        # If keep_year is True, restore the original year
        if keep_year:
            shifted_date = _replace_year_safe(shifted_date, original_year)

        # Try to preserve the original format
        return _format_date_like_original(date_str, shifted_date, lang=lang)

    except (ValueError, OverflowError):
        # If parsing fails, return a masked placeholder
        return "[DATE_SHIFTED]"


def _shift_date_basic(
    date_str: str, shift_days: int, keep_year: bool = True, lang: str = "en",
) -> str:
    """Basic date shifting without dateutil dependency.

    Handles common date formats using regex and datetime.

    Args:
        date_str: Date string to shift
        shift_days: Number of days to shift
        keep_year: Keep the year unchanged
        lang: Language code for date format conventions

    Returns:
        Shifted date string or placeholder
    """
    # Order patterns based on language convention
    if lang in _DAY_FIRST_LANGS - {"de"}:
        # European: DD/MM/YYYY first
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    elif lang == "de":
        # German: DD.MM.YYYY
        patterns = [
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", "dmy"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    elif lang == "ja":
        # Japanese: YYYY/MM/DD (kanji-form 年月日 is handled by the
        # JAPANESE_PII_PATTERNS regex, not here).
        patterns = [
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
        ]
    else:
        # US/English: MM/DD/YYYY first
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "mdy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})", "dmy"),
        ]

    for pattern, order in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            groups = match.groups()
            try:
                if order == "mdy":
                    month, day, year = int(groups[0]), int(groups[1]), int(groups[2])
                elif order == "ymd":
                    year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                else:  # dmy
                    day, month, year = int(groups[0]), int(groups[1]), int(groups[2])

                # Handle 2-digit years
                if year < 100:
                    year += 2000 if year < 50 else 1900

                # Validate and create date
                original_date = datetime(year, month, day)
                original_year = original_date.year

                # Shift
                shifted = original_date + timedelta(days=shift_days)

                # Keep year if requested
                if keep_year:
                    shifted = _replace_year_safe(shifted, original_year)

                # Format back preserving separator
                if "." in date_str:
                    sep = "."
                elif "/" in date_str:
                    sep = "/"
                else:
                    sep = "-"

                if order == "mdy":
                    return f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{shifted.year}"
                elif order == "ymd":
                    return f"{shifted.year}{sep}{shifted.month:02d}{sep}{shifted.day:02d}"
                else:
                    return f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{shifted.year}"

            except (ValueError, OverflowError):
                continue

    return "[DATE_SHIFTED]"


def _format_date_like_original(
    original: str, new_date: datetime, lang: str = "en",
) -> str:
    """Format a datetime to match the original string's format.

    Args:
        original: Original date string (for format detection)
        new_date: New datetime to format
        lang: Language code for date format conventions

    Returns:
        Formatted date string
    """
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    original_stripped = original.strip()

    # ISO format: YYYY-MM-DD
    if re.match(r"\d{4}-\d{2}-\d{2}", original_stripped):
        return new_date.strftime("%Y-%m-%d")

    # German dot-separated: DD.MM.YYYY
    if re.match(r"\d{1,2}\.\d{1,2}\.\d{2,4}", original_stripped):
        return new_date.strftime("%d.%m.%Y")

    # Slash-separated dates: interpretation depends on language
    if re.match(r"\d{1,2}/\d{1,2}/\d{4}", original_stripped):
        if lang in _DAY_FIRST_LANGS:
            # European: DD/MM/YYYY
            return new_date.strftime("%d/%m/%Y")
        else:
            # US: MM/DD/YYYY
            return new_date.strftime("%m/%d/%Y")

    # Dash-separated dates
    if re.match(r"\d{1,2}-\d{1,2}-\d{4}", original_stripped):
        if lang in _DAY_FIRST_LANGS:
            return new_date.strftime("%d-%m-%Y")
        else:
            return new_date.strftime("%m-%d-%Y")

    # Month name formats - check all supported languages
    month_names_flat = []
    for month_list in LANGUAGE_MONTH_NAMES.values():
        month_names_flat.extend(m.lower() for m in month_list)

    original_lower = original_stripped.lower()
    for month in month_names_flat:
        if month in original_lower:
            # Use language-specific month name
            lang_months = LANGUAGE_MONTH_NAMES.get(lang, LANGUAGE_MONTH_NAMES["en"])
            month_name = lang_months[new_date.month - 1]

            # "15 januari 2020" / "15. Januar 2020" / localized day-month-year
            if re.match(r"\d+\.?\s+[^\W\d_]+\s+\d{4}", original_stripped, re.UNICODE):
                return f"{new_date.day} {month_name} {new_date.year}"
            if re.match(r"\d+\s+de\s+[^\W\d_]+\s+de\s+\d{4}", original_stripped, re.UNICODE):
                return f"{new_date.day} de {month_name} de {new_date.year}"
            if re.match(r"[^\W\d_]+\s+\d+,?\s+\d{4}", original_stripped, re.UNICODE):
                return f"{month_name} {new_date.day}, {new_date.year}"
            break

    # Default to ISO format
    return new_date.strftime("%Y-%m-%d")


def reidentify(
    deidentified_text: str,
    mapping: dict[str, str],
) -> str:
    """Re-identify text using stored mapping.

    Restores original PII from de-identified text using the mapping created
    during de-identification. Only works if keep_mapping=True was used.

    Args:
        deidentified_text: De-identified text
        mapping: Mapping from redacted to original text

    Returns:
        Re-identified text with original PII restored

    Example:
        >>> result = deidentify(text, method="mask", keep_mapping=True)
        >>> original = reidentify(result.deidentified_text, result.mapping)
        >>> assert original == text

    Note:
        Only works if keep_mapping=True was used during de-identification.
        Requires proper authorization and audit logging in production.
    """
    result = deidentified_text

    for redacted, original in mapping.items():
        result = result.replace(redacted, original)

    return result
