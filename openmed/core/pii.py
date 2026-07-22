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
    >>> from openmed import reidentify
    >>> reidentify(
    ...     "Patient [NAME] called [PHONE]",
    ...     {"[NAME]": "Casey Example", "[PHONE]": "555-0100"},
    ... )
    'Patient Casey Example called 555-0100'
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, NoReturn, Optional, Sequence

from ..processing.outputs import EntityPrediction, PredictionResult
from .config import OpenMedConfig
from .custom_recognizer import (
    CUSTOM_DENY_DETECTOR,
    abdm_mode_enabled,
    coerce_custom_recognizer,
    with_abdm_recognizer,
)
from .date_shift import (
    DEFAULT_DATE_SHIFT_MAX_DAYS,
    stable_offset_for,
)
from .decoding import remap_normalized_span, snap_span_to_grapheme_boundaries
from .offline import network_blocked_if_offline
from .script_detect import (
    DetectionNormalization,
    india_clinical_script_windows,
    normalize_for_pii_detection,
)

if TYPE_CHECKING:
    from .anonymizer import Anonymizer
    from .audit import AuditReport
    from .models import ModelLoader
    from .surrogate_vault import SurrogateVault

from .result_cache import (
    get_result_cache,
    make_cache_key,
)

# Type alias for de-identification methods
DEIDENTIFICATION_METHODS = (
    "mask",
    "aadhaar_mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
)
DeidentificationMethod = Literal[
    "mask",
    "aadhaar_mask",
    "remove",
    "replace",
    "hash",
    "shift_dates",
    "format_preserve",
]


class MissingOptionalDependencyError(ImportError):
    """Raised when a requested optional capability needs an unavailable package."""

    def __init__(
        self,
        *,
        package: str,
        feature: str,
        extra: str | None = None,
    ) -> None:
        instruction = _optional_dependency_install_instruction(package, extra)
        super().__init__(
            f"{feature} requires optional dependency '{package}'. {instruction}"
        )
        self.package = package
        self.feature = feature
        self.extra = extra


def _optional_dependency_install_instruction(
    package: str,
    extra: str | None = None,
) -> str:
    if extra:
        return (
            f"Install it with `pip install openmed[{extra}]` "
            f"or `pip install {package}`."
        )
    return f"Install it with `pip install {package}`."


def _optional_dependency_status(
    *,
    package: str,
    feature: str,
    available: bool,
    skipped: bool,
    extra: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "feature": feature,
        "dependency": package,
        "available": available,
        "skipped": skipped,
        "install": _optional_dependency_install_instruction(package, extra),
    }
    if extra is not None:
        status["extra"] = extra
    if reason is not None:
        status["reason"] = reason
    return status


def _raise_missing_optional_dependency(
    *,
    package: str,
    feature: str,
    extra: str | None = None,
    cause: ImportError | None = None,
) -> NoReturn:
    raise MissingOptionalDependencyError(
        package=package,
        feature=feature,
        extra=extra,
    ) from cause


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
        mapping: Optional mapping for re-identification. Colliding replacement
            surfaces use private occurrence keys so separate source spellings
            remain reversible without changing the de-identified text.
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

    def _repr_html_(self) -> str:
        """Render a highlighted-span HTML view for Jupyter/IPython notebooks.

        IPython calls this automatically to display the result inline,
        highlighting each detected PII span over the original text. Confidence
        values are suppressed for this implicit representation; callers can opt
        into them explicitly with :func:`openmed.processing.show`. The helper is
        imported lazily so importing this module never requires display
        dependencies.
        """
        from ..processing.display import render_spans_html

        return render_spans_html(
            self.original_text,
            self.pii_entities,
            title=f"deidentify · {self.method}",
            show_confidence=False,
        )

    def to_dataframe(self) -> Any:
        """Convert detected PII entities to a pandas DataFrame.

        Returns:
            A pandas DataFrame with one row per detected entity and columns
            ``text``, ``label``, ``entity_type``, ``start``, ``end``,
            ``confidence``, ``action``, and ``result_id``.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required to use DeidentificationResult.to_dataframe(). "
                "Install it with `pip install pandas`."
            ) from exc

        columns = [
            "text",
            "label",
            "entity_type",
            "start",
            "end",
            "confidence",
            "action",
            "result_id",
        ]
        payload = self.to_dict()
        result_id_source = {
            "deidentified_text": payload["deidentified_text"],
            "method": payload["method"],
            "num_entities_redacted": payload["num_entities_redacted"],
            "timestamp": payload["timestamp"],
        }
        result_id = hashlib.sha256(
            json.dumps(
                result_id_source,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

        records = [
            {
                "text": entity.get("text"),
                "label": entity.get("label"),
                "entity_type": entity.get("entity_type"),
                "start": entity.get("start"),
                "end": entity.get("end"),
                "confidence": entity.get("confidence"),
                "action": entity.get("action"),
                "result_id": result_id,
            }
            for entity in payload["pii_entities"]
        ]
        return pd.DataFrame.from_records(records, columns=columns)


# Languages whose PII models were trained on accent-free text.
# For these, input is automatically stripped of accents before model
# inference and entity positions are mapped back to the original text.
_ACCENT_NORMALIZE_LANGS = frozenset({"es"})
_OCCURRENCE_MAPPING_PREFIX = "__openmed_occurrence_v1__:"

_DEFAULT_EN_MODEL = "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"
_DAY_FIRST_LANGS = frozenset(
    {
        "fr",
        "de",
        "it",
        "es",
        "nl",
        "hi",
        "te",
        "pt",
        "ar",
        "tr",
        "cs",
        "zu",
        "xh",
    }
)
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
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        for key in (
            "family",
            "_mlx_family",
            "_mlx_model_type",
            "model_type",
            "source_model_id",
            "_name_or_path",
        ):
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
        span_text = (
            (original_text if do_normalize else text)[start:end]
            if end > start
            else item.get("word", "")
        )
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
    config: Optional[OpenMedConfig] = None,
):
    """Run privacy-filter inference via the MLX/Torch backend dispatcher.

    Returns a ``PredictionResult`` with the same shape callers expect from
    ``analyze_text``. Confidence filtering is applied here since the
    privacy-filter pipelines don't know the user's threshold.
    """
    if pipeline is None:
        from .backends import create_privacy_filter_pipeline

        if config is None:
            pipeline = create_privacy_filter_pipeline(model_name)
        else:
            pipeline = create_privacy_filter_pipeline(model_name, config=config)

    with network_blocked_if_offline(config):
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
    from .model_registry import get_default_pii_model
    from .pii_i18n import (
        INDIC_NER_LANGUAGES,
        INDIC_NER_MODEL_ENV,
        SUPPORTED_LANGUAGES,
    )

    accepted_languages = SUPPORTED_LANGUAGES | INDIC_NER_LANGUAGES
    if lang not in accepted_languages:
        raise ValueError(
            f"Unsupported language '{lang}'. Supported: {sorted(accepted_languages)}"
        )

    if model_name == _DEFAULT_EN_MODEL and lang != "en":
        resolved = get_default_pii_model(lang)
        if resolved is None:
            raise ValueError(
                f"Language '{lang}' uses optional Indic NER weights; pass an "
                f"explicit model_name or set {INDIC_NER_MODEL_ENV}"
            )
        return resolved
    return model_name


@dataclass(frozen=True)
class _PreparedPIIText:
    original_text: str
    inference_text: str
    do_normalize: bool
    detection_normalization: DetectionNormalization


def _prepare_pii_text(
    text: str,
    *,
    lang: str,
    normalize_accents: Optional[bool],
    config: Optional[OpenMedConfig] = None,
    preserve_whitespace: bool = False,
) -> _PreparedPIIText:
    """Return source text, inference text, and normalization metadata."""
    do_normalize = (
        normalize_accents
        if normalize_accents is not None
        else (lang in _ACCENT_NORMALIZE_LANGS)
    )

    original_text = text if preserve_whitespace else text.strip()
    width_convention = config.cjk_width_convention if config is not None else "cjk"
    detection_normalization = normalize_for_pii_detection(
        original_text,
        width_convention=width_convention,
    )
    if lang.strip().replace("-", "_").split("_", 1)[0].casefold() == "ar":
        from .pii_i18n import normalize_arabic_indic_digits

        normalized_arabic = normalize_arabic_indic_digits(detection_normalization.text)
        folded_arabic_digits = sum(
            before != after
            for before, after in zip(
                detection_normalization.text,
                normalized_arabic,
            )
        )
        if folded_arabic_digits:
            detection_normalization = replace(
                detection_normalization,
                text=normalized_arabic,
                folded_native_digits=(
                    detection_normalization.folded_native_digits + folded_arabic_digits
                ),
            )
    inference_text = detection_normalization.text
    if do_normalize:
        inference_text = _strip_accents(inference_text)
    return _PreparedPIIText(
        original_text=original_text,
        inference_text=inference_text,
        do_normalize=bool(do_normalize)
        or detection_normalization.changed
        or detection_normalization.mixed_script,
        detection_normalization=detection_normalization,
    )


def _replace_analysis_result(result: Any, **updates: Any) -> Any:
    """Return ``result`` with analysis-result updates applied compatibly."""
    from .results import AnalyzeResult

    if isinstance(result, AnalyzeResult):
        return replace(result, **updates)

    for key, value in updates.items():
        setattr(result, key, value)
    if "entities" in updates:
        result.num_entities = len(updates["entities"])
    return result


def _mutable_prediction_result(result: Any) -> Any:
    """Return a mutable prediction result for internal PII post-processing."""
    from .results import AnalyzeResult

    if not isinstance(result, AnalyzeResult):
        return result
    return PredictionResult(
        text=result.text,
        entities=list(result.entities),
        model_name=result.model_name,
        timestamp=result.timestamp,
        processing_time=result.processing_time,
        metadata=dict(result.metadata) if result.metadata is not None else None,
    )


def _remap_prepared_pii_result(result: Any, prepared: _PreparedPIIText) -> Any:
    """Map normalized inference spans back to the original input text."""
    normalization = prepared.detection_normalization
    if (
        result.text == prepared.original_text
        and not normalization.changed
        and not normalization.mixed_script
    ):
        return result

    entities: list[EntityPrediction] = []
    for entity in result.entities:
        start = int(entity.start or 0)
        end = int(entity.end or start)
        original_start, original_end, original_surface = remap_normalized_span(
            start,
            end,
            prepared.original_text,
            normalization,
        )
        metadata = dict(entity.metadata or {})
        metadata.setdefault("unicode_defense", normalization.to_metadata())
        entities.append(
            EntityPrediction(
                text=original_surface,
                label=entity.label,
                start=original_start,
                end=original_end,
                confidence=entity.confidence,
                metadata=metadata,
            )
        )

    metadata = dict(getattr(result, "metadata", None) or {})
    metadata["unicode_defense"] = normalization.to_metadata()
    return _replace_analysis_result(
        result,
        text=prepared.original_text,
        entities=entities,
        metadata=metadata,
    )


def _snap_entities_to_grapheme_boundaries(
    text: str,
    entities: list[EntityPrediction],
) -> list[EntityPrediction]:
    """Return entities whose source spans cannot bisect grapheme clusters."""

    snapped_entities: list[EntityPrediction] = []
    for entity in entities:
        if entity.start is None or entity.end is None:
            snapped_entities.append(entity)
            continue

        start, end = snap_span_to_grapheme_boundaries(
            int(entity.start),
            int(entity.end),
            text,
        )
        if start == end:
            snapped_entities.append(entity)
            continue

        if (start, end) == (entity.start, entity.end):
            snapped_entities.append(entity)
            continue

        metadata = dict(entity.metadata or {})
        metadata["grapheme_boundary_adjustment"] = {
            "start_codepoints": int(entity.start) - start,
            "end_codepoints": end - int(entity.end),
        }
        snapped_entities.append(
            replace(
                entity,
                text=text[start:end],
                start=start,
                end=end,
                metadata=metadata,
            )
        )
    return snapped_entities


def _extract_india_clinical_via_script_windows(
    text: str,
    *,
    lang: str,
    effective_model: str,
    user_model: str | None,
    confidence_threshold: float,
    config: Optional[OpenMedConfig],
    loader: Optional["ModelLoader"],
    pipeline_kwargs: Mapping[str, Any],
) -> Any:
    """Run Latin and Indic context windows through their configured models."""

    from .. import analyze_text
    from ..clinical.context import INDIA_CLINICAL_NER_DISCLAIMER
    from ..processing.outputs import EntityPrediction
    from .models import HF_AVAILABLE, ModelLoader
    from .pii_i18n import get_india_clinical_model_route

    windows = india_clinical_script_windows(text, lang)
    if not windows:
        raise ValueError("India clinical routing requires mixed Latin/Indic text")

    route = get_india_clinical_model_route(lang)
    # Keep the analyzer injection boundary usable in core-only installs. The
    # real ``analyze_text`` path will still raise its normal optional-dependency
    # error when Transformers is absent, while test/custom analyzers can run
    # without constructing a Hugging Face loader they do not use.
    shared_loader = loader
    if shared_loader is None and HF_AVAILABLE:
        shared_loader = ModelLoader(config)
    routed_results: list[Any] = []
    routed_entities: list[EntityPrediction] = []
    route_metadata: list[dict[str, Any]] = []
    seen_requests: set[tuple[int, int, str, str]] = set()

    for window in windows:
        selected_model = route.model_for_script(
            window.script,
            user_model=user_model,
        )
        request_key = (window.start, window.end, selected_model, window.script)
        if request_key in seen_requests:
            continue
        seen_requests.add(request_key)

        window_text = window.extract(text)
        result = analyze_text(
            window_text,
            model_name=selected_model,
            confidence_threshold=confidence_threshold,
            config=config,
            loader=shared_loader,
            group_entities=True,
            **pipeline_kwargs,
        )
        result = _mutable_prediction_result(result)
        routed_results.append(result)

        # Tests and custom adapters sometimes return a full-note result even
        # when invoked with a window. Accept that shape without double-offsetting.
        base_offset = 0 if result.text == text else window.start
        for entity in result.entities:
            local_start = int(entity.start or 0)
            local_end = int(entity.end or local_start)
            start = base_offset + local_start
            end = base_offset + local_end
            if start < 0 or end < start or end > len(text):
                raise ValueError("India clinical model returned an invalid span")
            metadata = dict(entity.metadata or {})
            metadata["india_clinical_route"] = {
                "core_end": window.core_end,
                "core_start": window.core_start,
                "model": selected_model,
                "script": window.script,
                "window_end": window.end,
                "window_start": window.start,
            }
            routed_entities.append(
                EntityPrediction(
                    text=text[start:end],
                    label=entity.label,
                    start=start,
                    end=end,
                    confidence=entity.confidence,
                    metadata=metadata,
                )
            )
        route_metadata.append(
            {
                "core_end": window.core_end,
                "core_start": window.core_start,
                "model": selected_model,
                "script": window.script,
                "window_end": window.end,
                "window_start": window.start,
            }
        )

    if not routed_results:
        raise RuntimeError("India clinical routing produced no inference requests")

    base_result = routed_results[0]
    metadata = dict(getattr(base_result, "metadata", None) or {})
    metadata["india_clinical"] = {
        "active": True,
        "disclaimer": INDIA_CLINICAL_NER_DISCLAIMER,
        "fallback_model": route.fallback_model,
        "language": lang,
        "routes": route_metadata,
    }
    processing_times = [
        float(result.processing_time)
        for result in routed_results
        if result.processing_time is not None
    ]
    return _replace_analysis_result(
        base_result,
        text=text,
        entities=_coalesce_india_routed_entities(routed_entities, text),
        model_name=effective_model,
        processing_time=sum(processing_times) if processing_times else None,
        metadata=metadata,
    )


def _coalesce_india_routed_entities(
    entities: Sequence[EntityPrediction],
    text: str,
) -> list[EntityPrediction]:
    """Union duplicate same-label detections emitted by overlapping windows."""

    from .pii_entity_merger import normalize_label

    ordered = sorted(
        entities,
        key=lambda entity: (int(entity.start or 0), int(entity.end or 0)),
    )
    coalesced: list[EntityPrediction] = []
    for entity in ordered:
        if not coalesced:
            coalesced.append(entity)
            continue
        previous = coalesced[-1]
        previous_start = int(previous.start or 0)
        previous_end = int(previous.end or previous_start)
        start = int(entity.start or 0)
        end = int(entity.end or start)
        same_family = normalize_label(previous.label) == normalize_label(entity.label)
        if start >= previous_end or not same_family:
            coalesced.append(entity)
            continue

        merged_start = min(previous_start, start)
        merged_end = max(previous_end, end)
        winner = max((previous, entity), key=lambda item: float(item.confidence))
        metadata = dict(winner.metadata or {})
        metadata["india_clinical_window_union"] = True
        coalesced[-1] = EntityPrediction(
            text=text[merged_start:merged_end],
            label=winner.label,
            start=merged_start,
            end=merged_end,
            confidence=max(float(previous.confidence), float(entity.confidence)),
            metadata=metadata,
        )
    return coalesced


def _apply_pii_smart_merging(
    result: Any,
    effective_model: str,
    lang: str,
    *,
    locale: Optional[str] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    india_clinical: bool = False,
) -> Any:
    """Apply semantic-unit PII merging to a prediction result."""
    from ..processing.outputs import EntityPrediction
    from .pii_entity_merger import merge_entities_with_semantic_units
    from .pii_i18n import (
        get_patterns_for_code_mixed_tags,
        get_patterns_for_language,
    )

    if code_mixed:
        if token_language_tags is None:
            raise ValueError("code_mixed=True requires token_language_tags")
        lang_patterns = get_patterns_for_code_mixed_tags(
            result.text,
            token_language_tags,
            base_lang=lang,
            locale=locale,
        )
    else:
        lang_patterns = get_patterns_for_language(lang, locale=locale)
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
        india_clinical=india_clinical,
    )

    merged_entities = [
        EntityPrediction(
            text=e["word"],
            label=e["entity_type"],
            start=e["start"],
            end=e["end"],
            confidence=e["score"],
            metadata=_pii_merge_metadata(e),
        )
        for e in merged_dicts
    ]
    return _replace_analysis_result(result, entities=merged_entities)


def _pii_merge_metadata(entity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return structured metadata retained from semantic/clinical merging."""

    metadata: dict[str, Any] = {}
    if entity.get("source_labels"):
        metadata["semantic_merge"] = {
            "source_labels": list(entity.get("source_labels", ())),
            "mixed_label_union": bool(entity.get("mixed_label_union", False)),
        }
    if entity.get("normalized_word"):
        # Keep result/audit metadata surface-free. The merger's internal
        # ``normalized_word`` drives span logic, while public provenance records
        # only the registered abbreviation keys and normalization version.
        metadata["clinical_normalization"] = dict(
            entity.get("clinical_normalization") or {}
        )
    if entity.get("india_clinical_merge"):
        metadata["india_clinical_merge"] = dict(entity["india_clinical_merge"])
    return metadata or None


def _extract_pii_batch(
    texts: Sequence[str],
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.5,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    transliterated_name_config: Any = None,
    *,
    preserve_whitespace: bool = False,
    locale: Optional[str] = None,
    loader: Optional["ModelLoader"] = None,
    privacy_filter_pipeline: Optional[Any] = None,
    **pipeline_kwargs: Any,
) -> list[Any]:
    """Extract PII for multiple texts while reusing the same backend resources."""
    if code_mixed and len(texts) != 1:
        raise ValueError("code-mixed token tags currently require one input text")
    if code_mixed and token_language_tags is None:
        raise ValueError("code_mixed=True requires token_language_tags")
    effective_model = _resolve_effective_pii_model(model_name, lang)
    prepared = [
        _prepare_pii_text(
            text,
            lang=lang,
            normalize_accents=normalize_accents,
            config=config,
            preserve_whitespace=preserve_whitespace,
        )
        for text in texts
    ]

    if not prepared:
        return []

    india_clinical_flags = [
        bool(india_clinical_script_windows(item.inference_text, lang))
        for item in prepared
    ]

    uses_privacy_filter = _looks_like_privacy_filter_identifier(
        effective_model
    ) or _is_privacy_filter_artifact_path(effective_model)

    if uses_privacy_filter:
        from .backends import create_privacy_filter_pipeline

        if privacy_filter_pipeline is not None:
            pipeline = privacy_filter_pipeline
        elif config is None:
            pipeline = create_privacy_filter_pipeline(effective_model)
        else:
            pipeline = create_privacy_filter_pipeline(effective_model, config=config)
        inference_texts = [item.inference_text for item in prepared]
        privacy_call_kwargs = {
            key: pipeline_kwargs[key]
            for key in ("batch_size", "num_workers")
            if key in pipeline_kwargs and pipeline_kwargs[key] is not None
        }
        with network_blocked_if_offline(config):
            raw_outputs = pipeline(inference_texts, **privacy_call_kwargs)
        batched_raw = _coerce_batched_raw_outputs(raw_outputs, len(prepared))
        results = [
            _prediction_result_from_privacy_filter_raw(
                raw,
                item.inference_text,
                model_name=effective_model,
                confidence_threshold=confidence_threshold,
                original_text=item.inference_text,
                do_normalize=False,
            )
            for raw, item in zip(batched_raw, prepared)
        ]
    else:
        from .. import analyze_text
        from .models import ModelLoader

        shared_loader = loader
        if shared_loader is None and len(prepared) > 1:
            shared_loader = ModelLoader(config)
        results = []
        for item, india_clinical in zip(prepared, india_clinical_flags):
            if india_clinical:
                result = _extract_india_clinical_via_script_windows(
                    item.inference_text,
                    lang=lang,
                    effective_model=effective_model,
                    user_model=(
                        None if model_name == _DEFAULT_EN_MODEL else effective_model
                    ),
                    confidence_threshold=confidence_threshold,
                    config=config,
                    loader=shared_loader,
                    pipeline_kwargs=pipeline_kwargs,
                )
            else:
                result = analyze_text(
                    item.inference_text,
                    model_name=effective_model,
                    confidence_threshold=confidence_threshold,
                    config=config,
                    loader=shared_loader,
                    group_entities=True,
                    **pipeline_kwargs,
                )
            result = _mutable_prediction_result(result)
            results.append(result)

    if use_smart_merging:
        results = [
            (
                _apply_pii_smart_merging(
                    result,
                    effective_model,
                    lang,
                    locale=locale,
                    code_mixed=code_mixed,
                    token_language_tags=token_language_tags,
                    india_clinical=india_clinical,
                )
                if not uses_privacy_filter or india_clinical
                else result
            )
            for result, india_clinical in zip(results, india_clinical_flags)
        ]

    results = [
        _remap_prepared_pii_result(result, item)
        for result, item in zip(results, prepared)
    ]

    recognizer_config = custom_recognizer
    if abdm_mode_enabled(abdm, lang=lang, locale=locale):
        recognizer_config = with_abdm_recognizer(recognizer_config)
    recognizer = coerce_custom_recognizer(recognizer_config)
    if recognizer is not None:
        for result in results:
            recognizer.apply_to_prediction_result(result)

    if code_mixed:
        from .custom_recognizer import build_transliterated_name_recognizer
        from .pii_i18n import code_mixed_route_active

        if token_language_tags is not None and code_mixed_route_active(
            results[0].text,
            token_language_tags,
        ):
            name_recognizer = build_transliterated_name_recognizer(
                transliterated_name_config
            )
            previous_metadata = dict(getattr(results[0], "metadata", None) or {})
            name_recognizer.apply_to_prediction_result(results[0])
            metadata = dict(getattr(results[0], "metadata", None) or {})
            bridge_metadata = metadata.pop("custom_recognizer", {})
            metadata.update(previous_metadata)
            metadata["transliterated_name_bridge"] = bridge_metadata
            results[0].metadata = metadata

    for result in results:
        _apply_clinical_protection_to_result(
            result.text,
            result,
            config=config,
            lang=lang,
        )

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
    cache_results: bool = False,
    max_cache_entries: int = 128,
    normalize_accents: Optional[bool] = None,
    *,
    preserve_whitespace: bool = False,
    locale: Optional[str] = None,
    loader: Optional["ModelLoader"] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    transliterated_name_config: Any = None,
) -> PredictionResult:
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
            default model and regex patterns are used. Mixed Latin/Devanagari
            or Latin/Telugu notes automatically use script-aware India
            clinical routing for ``hi`` and ``te``.
        normalize_accents: Strip diacritical marks before model inference so
            that models trained on accent-free text still detect accented
            names.  Entity spans in the result reference the *original*
            (accented) text.  ``None`` (default) auto-enables for languages
            in ``_ACCENT_NORMALIZE_LANGS`` (currently Spanish).
        preserve_whitespace: Preserve leading and trailing source whitespace
            so returned entity offsets refer to the exact input string.
        loader: Optional shared model loader to reuse warmed pipelines.
        batch_size: Optional backend inference batch size.
        num_workers: Optional backend inference worker count.
        custom_recognizer: Optional deny-list/allow-list recognizer config,
            ``CustomRecognizer`` instance, or JSON/YAML config path. Deny-list
            matches are added with ``custom:deny`` provenance; allow-list
            matches suppress overlapping spans from any detector.
        abdm: Enable the India ABDM identifier bundle. ``None`` auto-enables
            it for Hindi/Telugu and India locales; ``False`` explicitly
            disables that automatic activation.
        code_mixed: Enable the explicit English/Hinglish route. This preserves
            English model detection while adding Roman-Hindi context patterns.
        token_language_tags: Required with ``code_mixed=True``. Ordered,
            non-overlapping records with ``start``, ``end``, and ``label``
            (``en``, ``hi``, ``ne``, ``univ``, or ``other``). Tags are consumed
            as offsets/labels only and are never copied with raw token surfaces.
        transliterated_name_config: Optional configuration for the conservative
            Latin-script Indian given/family-name allow/deny bridge.
        cache_results: Whether to cache this result in the in-process LRU
            cache. Cached results may contain PHI, but are never saved to disk.
        max_cache_entries: Maximum number of cached results.

    Returns:
        PredictionResult with detected PII entities

    Example:
        >>> from unittest.mock import patch
        >>> from openmed.core.pii import extract_pii
        >>> from openmed.processing.outputs import EntityPrediction, PredictionResult
        >>> fake_result = PredictionResult(
        ...     text="Patient Casey Example called.",
        ...     entities=[
        ...         EntityPrediction(
        ...             text="Casey Example",
        ...             label="NAME",
        ...             confidence=0.98,
        ...             start=8,
        ...             end=21,
        ...         )
        ...     ],
        ...     model_name="fixture-pii-model",
        ...     timestamp="2026-01-01T00:00:00",
        ... )
        >>> with patch("openmed.analyze_text", return_value=fake_result):
        ...     result = extract_pii(
        ...         "Patient Casey Example called.",
        ...         model_name="fixture-pii-model",
        ...         use_smart_merging=False,
        ...     )
        >>> next((entity.text, entity.label) for entity in result.entities)
        ('Casey Example', 'NAME')
    """
    if cache_results:
        params = dict(locals())
        cache_key = make_cache_key("extract_pii", params)
        cache = get_result_cache(max_entries=max_cache_entries)
        final_result = cache.get(cache_key)
        if final_result is not None:
            return final_result
    runtime_kwargs = {}
    if batch_size is not None:
        runtime_kwargs["batch_size"] = batch_size
    if num_workers is not None:
        runtime_kwargs["num_workers"] = num_workers
    final_result = _extract_pii_batch(
        [text],
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        preserve_whitespace=preserve_whitespace,
        locale=locale,
        loader=loader,
        custom_recognizer=custom_recognizer,
        abdm=abdm,
        code_mixed=code_mixed,
        token_language_tags=token_language_tags,
        transliterated_name_config=transliterated_name_config,
        **runtime_kwargs,
    )[0]
    if cache_results:
        cache.set(cache_key, final_result)
    return final_result


def _resolve_deidentification_method(
    method: DeidentificationMethod,
    shift_dates: Optional[bool],
    date_shift_days: Optional[int],
    *,
    patient_key: Optional[str | bytes] = None,
    date_shift_max_days: Optional[int] = None,
    date_shift_secret: Optional[str | bytes] = None,
) -> DeidentificationMethod:
    """Resolve method aliases and validate date-shift-only parameters."""
    effective_method = method
    if shift_dates is True and method != "shift_dates":
        effective_method = "shift_dates"
    elif shift_dates is False and method == "shift_dates":
        raise ValueError("shift_dates=false conflicts with method='shift_dates'")

    if date_shift_days is not None and effective_method != "shift_dates":
        raise ValueError("date_shift_days requires method='shift_dates'")
    if patient_key is not None and effective_method != "shift_dates":
        raise ValueError("patient_key requires method='shift_dates'")
    if date_shift_max_days is not None and effective_method != "shift_dates":
        raise ValueError("date_shift_max_days requires method='shift_dates'")
    if date_shift_secret is not None and effective_method != "shift_dates":
        raise ValueError("date_shift_secret requires method='shift_dates'")
    if date_shift_secret is not None and patient_key is None:
        raise ValueError("date_shift_secret requires patient_key")
    if patient_key is not None and date_shift_secret is None:
        raise ValueError("patient_key requires date_shift_secret")
    if effective_method not in DEIDENTIFICATION_METHODS:
        raise ValueError(f"method must be one of {DEIDENTIFICATION_METHODS!r}")

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
    locale: Optional[str] = None,
    patterns: Optional[Sequence[Any]] = None,
) -> tuple[Any, int]:
    """Run the deterministic sweep and record its net span contribution."""
    from .quality_gates import validate_entity_spans
    from .safety_sweep import (
        SAFETY_SWEEP_PATTERNS_VERSION,
        SAFETY_SWEEP_SOURCE,
        safety_sweep,
    )

    before_count = len(pii_result.entities)
    entities = safety_sweep(
        text,
        pii_result.entities,
        lang=lang,
        locale=locale,
        patterns=patterns,
    )
    added_count = len(entities) - before_count

    metadata = dict(getattr(pii_result, "metadata", None) or {})
    metadata["safety_sweep"] = {
        "enabled": True,
        "source": SAFETY_SWEEP_SOURCE,
        "patterns_version": SAFETY_SWEEP_PATTERNS_VERSION,
        "spans_added": added_count,
    }
    pii_result = _replace_analysis_result(
        pii_result,
        entities=entities,
        metadata=metadata,
    )
    validate_entity_spans(pii_result.entities, text)
    return pii_result, added_count


_AUDIT_TEXT_KEYS = {
    "canonical_transliteration_key",
    "text",
    "transliteration_key",
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
    detector = str(metadata.get("detector", "")).strip()
    if source == CUSTOM_DENY_DETECTOR or detector == CUSTOM_DENY_DETECTOR:
        return [CUSTOM_DENY_DETECTOR]
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
    custom_metadata = metadata.get("custom_recognizer")
    if isinstance(custom_metadata, Mapping):
        detectors.append(
            DetectorInfo(
                source=CUSTOM_DENY_DETECTOR,
                model_id="custom_recognizer",
                model_format="rules",
                commit=None,
                metadata=dict(custom_metadata),
            )
        )
    return detectors


def _add_custom_recognizer_metadata(
    pii_result: Any,
    *,
    allow_spans: int,
    suppressed: int,
) -> None:
    metadata = dict(getattr(pii_result, "metadata", None) or {})
    custom_metadata = dict(metadata.get("custom_recognizer") or {})
    custom_metadata.setdefault("detector", CUSTOM_DENY_DETECTOR)
    custom_metadata["allow_spans"] = allow_spans
    custom_metadata["spans_suppressed_by_allow"] = (
        int(custom_metadata.get("spans_suppressed_by_allow", 0)) + suppressed
    )
    metadata["custom_recognizer"] = custom_metadata
    pii_result.metadata = metadata


def _suppress_custom_allowed_entities(
    text: str,
    pii_result: Any,
    custom_recognizer: Any,
) -> int:
    recognizer = coerce_custom_recognizer(custom_recognizer)
    if recognizer is None:
        return 0
    retained, suppressed = recognizer.suppress_entities(
        text,
        list(getattr(pii_result, "entities", ()) or ()),
    )
    if suppressed:
        pii_result.entities = retained
        if hasattr(pii_result, "num_entities"):
            pii_result.num_entities = len(retained)
    _add_custom_recognizer_metadata(
        pii_result,
        allow_spans=len(recognizer.allow_matches(text)),
        suppressed=suppressed,
    )
    return suppressed


def _apply_clinical_protection_to_result(
    text: str,
    pii_result: Any,
    *,
    config: Optional[OpenMedConfig] = None,
    options: Optional[Mapping[str, Any]] = None,
    lang: str = "en",
) -> int:
    """Suppress ambiguous PII entities that exactly match clinical terms."""
    from .clinical_protect import (
        filter_protected_spans,
        protection_options_from_config,
    )

    protection_options = dict(options or protection_options_from_config(config))
    result = filter_protected_spans(
        list(getattr(pii_result, "entities", ()) or ()),
        text,
        lang=lang,
        **protection_options,
    )
    pii_result.entities = result.spans
    if hasattr(pii_result, "num_entities"):
        pii_result.num_entities = len(result.spans)

    metadata = dict(getattr(pii_result, "metadata", None) or {})
    previous = metadata.get("clinical_protection")
    previous_metadata = previous if isinstance(previous, Mapping) else {}
    clinical_metadata = dict(result.metadata["clinical_protection"])
    clinical_metadata["checked_spans"] += int(previous_metadata.get("checked_spans", 0))
    clinical_metadata["suppressed_spans"] += int(
        previous_metadata.get("suppressed_spans", 0)
    )
    clinical_metadata["protected_term_count"] = max(
        int(clinical_metadata["protected_term_count"]),
        int(previous_metadata.get("protected_term_count", 0)),
    )
    clinical_metadata["enabled"] = bool(protection_options.get("enabled", True))
    metadata["clinical_protection"] = clinical_metadata
    pii_result.metadata = metadata
    return result.suppressed_count


def _context_window(
    text: str, start: int, end: int, *, size: int = 32
) -> dict[str, dict[str, int | str]]:
    from .audit import hash_text

    before_start = max(0, start - size)
    after_end = min(len(text), end + size)
    before = text[before_start:start]
    after = text[end:after_end]
    return {
        "before": {
            "start": before_start,
            "end": start,
            "length": len(before),
            "text_hash": hash_text(before),
        },
        "after": {
            "start": end,
            "end": after_end,
            "length": len(after),
            "text_hash": hash_text(after),
        },
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


def _active_calibration_thresholds(
    pii_result: Any,
    *,
    lang: str,
) -> dict[str, float]:
    from .labels import normalize_label

    metadata = getattr(pii_result, "metadata", None) or {}
    calibration = metadata.get("calibration_thresholds")
    if not isinstance(calibration, Mapping):
        return {}
    active = calibration.get("active")
    if not isinstance(active, Mapping):
        return {}

    thresholds: dict[str, float] = {}
    for label, value in active.items():
        try:
            thresholds[normalize_label(str(label), lang)] = float(value)
        except (TypeError, ValueError):
            continue
    return thresholds


def _entity_threshold(
    entity: Any,
    *,
    canonical_label: str,
    active_thresholds: Mapping[str, float],
    fallback: float,
) -> float:
    metadata = getattr(entity, "metadata", None) or {}
    calibration = metadata.get("calibration_threshold")
    if isinstance(calibration, Mapping):
        try:
            return float(calibration["threshold"])
        except (KeyError, TypeError, ValueError):
            pass
    return float(active_thresholds.get(canonical_label, fallback))


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
    from ..__about__ import __version__
    from .audit import AuditReport, hash_text, manifest_hash
    from .labels import normalize_label
    from .safety_sweep import SAFETY_SWEEP_PATTERNS_VERSION, SAFETY_SWEEP_SOURCE

    thresholds = _active_calibration_thresholds(pii_result, lang=lang)
    thresholds.update(
        {
            normalize_label(entity.label, lang): float(
                entity.threshold or confidence_threshold
            )
            for entity in pii_entities
        }
    )
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
        spans=[
            _audit_span_from_entity(original_text, entity) for entity in pii_entities
        ],
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
    patient_key: Optional[str | bytes] = None,
    date_shift_max_days: Optional[int] = None,
    date_shift_secret: Optional[str | bytes] = None,
    keep_mapping: bool,
    lang: str,
    consistent: bool,
    seed: Optional[int],
    locale: Optional[str],
    surrogate_vault: Optional["SurrogateVault"] = None,
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

    grapheme_safe_entities = _snap_entities_to_grapheme_boundaries(
        text,
        list(pii_result.entities),
    )
    resolved_entities = resolve_overlapping_entities(grapheme_safe_entities)
    pii_result = _replace_analysis_result(pii_result, entities=resolved_entities)

    active_thresholds = _active_calibration_thresholds(pii_result, lang=lang)
    pii_entities = []
    for e in pii_result.entities:
        canonical_label = normalize_label(e.label, lang)
        pii_entities.append(
            PIIEntity(
                text=e.text,
                label=e.label,
                start=e.start,
                end=e.end,
                confidence=e.confidence,
                metadata=_copy_metadata(getattr(e, "metadata", None)),
                entity_type=e.label,
                original_text=e.text,
                canonical_label=canonical_label,
                sources=_entity_sources(e),
                evidence=_entity_evidence(
                    e,
                    pii_result=pii_result,
                    model_name=model_name,
                    lang=lang,
                ),
                threshold=_entity_threshold(
                    e,
                    canonical_label=canonical_label,
                    active_thresholds=active_thresholds,
                    fallback=confidence_threshold,
                ),
            ),
        )

    redaction_entities = sorted(pii_entities, key=lambda e: e.start, reverse=True)

    if effective_method == "shift_dates":
        date_shift_days = _resolve_date_shift_days(
            date_shift_days=date_shift_days,
            patient_key=patient_key,
            date_shift_max_days=date_shift_max_days,
            date_shift_secret=date_shift_secret,
        )

    anonymizer = None
    if effective_method in {"replace", "format_preserve"} or any(
        _entity_policy_action(entity) in {"replace", "format_preserve"}
        for entity in pii_entities
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
    mapping: dict[str, str] | None = None
    mapping_occurrences: dict[str, list[tuple[int, str]]] = {}
    source_surrogates: dict[tuple[str, str], str] = {}
    entity_occurrence_indexes: dict[int, int] = {}
    if keep_mapping:
        entity_type_counts: dict[str, int] = {}
        for entity in sorted(pii_entities, key=lambda e: e.start):
            entity_method = _entity_redaction_method(entity, effective_method)
            if (
                entity_method in {"mask", "remove"}
                or (
                    entity_method == "shift_dates" and not _is_date_entity(entity, lang)
                )
                or (
                    entity_method == "format_preserve"
                    and _format_preserve_uses_mask_fallback(entity, anonymizer, lang)
                )
            ):
                entity_type_counts[entity.entity_type] = (
                    entity_type_counts.get(entity.entity_type, 0) + 1
                )
                entity_occurrence_indexes[id(entity)] = entity_type_counts[
                    entity.entity_type
                ]

    for entity in redaction_entities:
        entity_method = _entity_redaction_method(entity, effective_method)
        actual_entity_method = entity_method
        if entity_method == "format_preserve" and _format_preserve_uses_mask_fallback(
            entity,
            anonymizer,
            lang,
        ):
            actual_entity_method = "mask"
        source_key = (
            entity.canonical_label or entity.entity_type,
            entity.original_text or entity.text,
        )
        if (
            keep_mapping
            and actual_entity_method in {"replace", "format_preserve"}
            and source_key in source_surrogates
        ):
            redacted = source_surrogates[source_key]
        else:
            redacted = _redact_entity(
                entity,
                actual_entity_method,
                keep_year=keep_year,
                date_shift_days=(
                    date_shift_days if actual_entity_method == "shift_dates" else None
                ),
                lang=lang,
                anonymizer=anonymizer,
                require_dateutil=effective_method == "shift_dates",
                surrogate_vault=surrogate_vault,
            )
        if entity_method == "format_preserve" and redacted == _mask_placeholder(entity):
            actual_entity_method = "mask"
        if (
            keep_mapping
            and actual_entity_method in {"replace", "format_preserve"}
            and redacted
        ):
            source_surrogates.setdefault(source_key, redacted)

        if keep_mapping and actual_entity_method == "remove":
            redacted = f"[{entity.entity_type}_REMOVED]"

        # Only make repeated placeholders unique for reversible mappings. Plain
        # masking/removal without keep_mapping keeps the legacy redacted text.
        occurrence_index = entity_occurrence_indexes.get(id(entity), 1)
        if keep_mapping and redacted and occurrence_index > 1:
            if redacted.endswith("]"):
                redacted = f"{redacted[:-1]}_{occurrence_index}]"
            else:
                redacted = f"{redacted}_{occurrence_index}"

        entity.redacted_text = redacted
        if reversible_ids:
            entity.reversible_id = _build_reversible_id(
                entity,
                policy_name=policy_name,
            )
        entity.action = actual_entity_method
        entity.surrogate = redacted if redacted else None

        deidentified = (
            deidentified[: entity.start] + redacted + deidentified[entity.end :]
        )

        if keep_mapping:
            mapping_occurrences.setdefault(redacted, []).append(
                (entity.start, entity.original_text or entity.text)
            )

    if keep_mapping:
        mapping = _build_reidentification_mapping(mapping_occurrences)

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
    if action == "format_preserve":
        return "format_preserve"
    if action in {"mask", "redact"}:
        return "mask"
    return fallback


def _is_mixed_label_union(entity: PIIEntity) -> bool:
    """Return whether smart merging joined distinct semantic label families."""
    metadata = entity.metadata or {}
    semantic_merge = metadata.get("semantic_merge")
    return isinstance(semantic_merge, Mapping) and bool(
        semantic_merge.get("mixed_label_union", False)
    )


def _format_preserve_uses_mask_fallback(
    entity: PIIEntity,
    anonymizer: Optional["Anonymizer"],
    lang: str,
) -> bool:
    if anonymizer is None or _is_mixed_label_union(entity):
        return True
    return not anonymizer.can_format_preserve(
        entity.original_text or entity.text,
        entity.entity_type,
        lang=lang,
    )


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
    keep_year: bool = False,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    patient_key: Optional[str | bytes] = None,
    date_shift_max_days: Optional[int] = None,
    date_shift_secret: Optional[str | bytes] = None,
    keep_mapping: bool = False,
    config: Optional[OpenMedConfig] = None,
    use_smart_merging: bool = True,
    lang: str = "en",
    normalize_accents: Optional[bool] = None,
    use_safety_sweep: bool = True,
    preserve_whitespace: bool = False,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    transliterated_name_config: Any = None,
    policy: Optional[str] = None,
    *,
    consistent: bool = False,
    seed: Optional[int] = None,
    locale: Optional[str] = None,
    surrogate_vault: Optional["SurrogateVault"] = None,
    loader: Optional["ModelLoader"] = None,
    privacy_filter_pipeline: Optional[Any] = None,
    **pipeline_kwargs: Any,
) -> list[DeidentificationResult]:
    """De-identify multiple texts after one batched PII extraction pass."""
    effective_method = _resolve_deidentification_method(
        method,
        shift_dates,
        date_shift_days,
        patient_key=patient_key,
        date_shift_max_days=date_shift_max_days,
        date_shift_secret=date_shift_secret,
    )
    source_texts = [text if preserve_whitespace else text.strip() for text in texts]
    recognizer = coerce_custom_recognizer(custom_recognizer)
    pii_results = _extract_pii_batch(
        source_texts,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        config=config,
        use_smart_merging=use_smart_merging,
        lang=lang,
        normalize_accents=normalize_accents,
        preserve_whitespace=preserve_whitespace,
        locale=locale,
        custom_recognizer=recognizer,
        abdm=abdm_mode_enabled(
            abdm,
            policy=policy,
            lang=lang,
            locale=locale,
        ),
        code_mixed=code_mixed,
        token_language_tags=token_language_tags,
        transliterated_name_config=transliterated_name_config,
        loader=loader,
        privacy_filter_pipeline=privacy_filter_pipeline,
        **pipeline_kwargs,
    )

    if use_safety_sweep:
        swept_results = []
        for source_text, pii_result in zip(source_texts, pii_results):
            sweep_patterns = None
            if code_mixed:
                if token_language_tags is None:
                    raise ValueError("code_mixed=True requires token_language_tags")
                from .pii_i18n import get_patterns_for_code_mixed_tags

                sweep_patterns = get_patterns_for_code_mixed_tags(
                    source_text,
                    token_language_tags,
                    base_lang=lang,
                    locale=locale,
                )
            pii_result, _ = _apply_safety_sweep_to_result(
                source_text,
                pii_result,
                lang=lang,
                locale=locale,
                patterns=sweep_patterns,
            )
            _suppress_custom_allowed_entities(source_text, pii_result, recognizer)
            swept_results.append(pii_result)
        pii_results = swept_results

    return [
        _build_deidentification_result(
            text,
            pii_result,
            effective_method=effective_method,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            keep_year=keep_year,
            date_shift_days=date_shift_days,
            patient_key=patient_key,
            date_shift_max_days=date_shift_max_days,
            date_shift_secret=date_shift_secret,
            keep_mapping=keep_mapping,
            lang=lang,
            normalize_accents=normalize_accents,
            use_smart_merging=use_smart_merging,
            use_safety_sweep=use_safety_sweep,
            consistent=consistent,
            seed=seed,
            locale=locale,
            surrogate_vault=surrogate_vault,
            policy=policy or "hipaa_safe_harbor",
        )
        for text, pii_result in zip(source_texts, pii_results)
    ]


def deidentify(
    text: str,
    method: DeidentificationMethod = "mask",
    model_name: str = _DEFAULT_EN_MODEL,
    confidence_threshold: float = 0.7,  # Higher threshold for safety
    keep_year: bool = False,
    shift_dates: Optional[bool] = None,
    date_shift_days: Optional[int] = None,
    patient_key: Optional[str | bytes] = None,
    date_shift_max_days: Optional[int] = None,
    date_shift_secret: Optional[str | bytes] = None,
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
    surrogate_vault: Optional["SurrogateVault"] = None,
    loader: Optional["ModelLoader"] = None,
    policy: Optional[str] = None,
    calibration_thresholds_path: Optional[str | Path] = None,
    custom_recognizer: Any = None,
    abdm: Optional[bool] = None,
    code_mixed: bool = False,
    token_language_tags: Optional[Sequence[Any]] = None,
    transliterated_name_config: Any = None,
    audit: bool = False,
    cache_results: bool = False,
    max_cache_entries: int = 128,
) -> DeidentificationResult | "AuditReport":
    """De-identify text by detecting and redacting PII with intelligent merging.

    Implements multiple de-identification strategies for HIPAA compliance:

    - **mask**: Replace with placeholders like [NAME], [EMAIL], etc.
    - **aadhaar_mask**: Render valid Aadhaar values as ``XXXX XXXX NNNN``;
      use ordinary placeholders for every other entity
    - **remove**: Remove PII text entirely (empty string)
    - **replace**: Replace with fake but realistic data
    - **hash**: Replace with consistent hashed values for entity linking
    - **format_preserve**: Replace structured identifiers with synthetic
      values that keep shape and separators, masking unsupported labels
    - **shift_dates**: Shift dates by random offset while preserving intervals

    Smart merging uses regex patterns to merge fragmented entities (e.g., dates
    split into '01' and '/15/1970' are merged into complete '01/15/1970').

    Code-mixed mode is explicit and offset driven. With ``code_mixed=True`` and
    per-token language tags, the English NER path remains active while a
    separate Roman-script Hindi pattern bank detects cues such as ``naam``,
    ``umar``, ``pata``, ``mobile``, and ``janm``. The combined spans pass
    through the normal entity merger and final safety sweep before redaction.

    Args:
        text: Input text to de-identify
        method: De-identification method (mask, aadhaar_mask, remove, replace,
            hash, shift_dates, format_preserve)
        model_name: PII detection model
        confidence_threshold: Minimum confidence for redaction (default 0.7 for safety)
        keep_year: For dates, keep the year unchanged
        shift_dates: Deprecated alias for ``method="shift_dates"``.
        date_shift_days: Specific number of days to shift when ``patient_key``
            is omitted. When ``patient_key`` is supplied, this is treated as a
            legacy maximum absolute offset bound unless ``date_shift_max_days``
            is also supplied.
        patient_key: Optional stable patient identifier used only to derive a
            deterministic HMAC date-shift offset. Raw keys are not logged,
            persisted, or returned.
        date_shift_max_days: Maximum absolute offset for random or
            patient-keyed date shifting. Defaults to 365 when ``patient_key``
            is supplied and neither this nor ``date_shift_days`` is set.
        date_shift_secret: Required HMAC key material for patient-keyed
            offsets. Reuse the same value across sessions to keep offsets
            stable.
        keep_mapping: Keep mapping for re-identification
        config: Optional configuration override
        use_smart_merging: Enable regex-based semantic unit merging (recommended)
        use_safety_sweep: Run a deterministic structured-identifier sweep
            after model detection and before redaction.
        lang: ISO 639-1 language code (en, fr, de, it, es, nl, hi, te, pt,
            ar, ja, tr). Controls model
            selection, regex patterns, and fake data for replacement. Mixed
            Latin/Devanagari or Latin/Telugu notes automatically use the
            script-aware India clinical route for ``hi`` and ``te``.
        normalize_accents: Strip diacritical marks before model inference.
            ``None`` (default) auto-enables for Spanish.
        loader: Optional shared model loader to reuse warmed pipelines.
        consistent: When ``method="replace"`` or
            ``method="format_preserve"``, generate stable surrogates
            (same input -> same surrogate within the call). Lets repeated
            mentions of the same name resolve to one fake identity instead
            of a different one each time.
        seed: Optional integer seed for cross-run reproducibility of
            ``consistent=True`` replacements. Implies ``consistent=True``.
        locale: Faker locale override (``pt_BR``, ``en_GB``, ...) for
            ``method="replace"`` and ``method="format_preserve"``. When
            ``None``, derived from ``lang``.
        surrogate_vault: Optional cross-document surrogate vault. When provided
            with ``method="replace"``, OpenMed stores only HMAC source hashes.
            Indian names in Devanagari, Tamil, or opted-in Romanization use one
            HMAC of an in-memory phonetic fold and render the reused identity in
            the input script; the fold itself is never persisted or audited.
        policy: Optional policy profile name controlling arbitration, action
            selection, mandatory safety sweep behavior, and reversible mapping.
        calibration_thresholds_path: Optional thresholds.json artifact path
            or artifact directory. When provided, per-label calibrated
            thresholds filter model detections and appear in audit output.
        custom_recognizer: Optional deny-list/allow-list recognizer config,
            ``CustomRecognizer`` instance, or JSON/YAML config path. Deny-list
            matches are redacted with ``custom:deny`` provenance; allow-list
            matches suppress overlapping spans from any detector.
        abdm: Enable the India ABDM recognizer bundle. ``None`` auto-enables
            it for ``policy="india_dpdp_act"``, Hindi/Telugu, or an India
            locale. Pass ``False`` to opt out of automatic activation.
        code_mixed: Enable the explicit English/Hinglish de-identification path.
        token_language_tags: Required with ``code_mixed=True``. Ordered token
            records with exact ``start``/``end`` offsets and an ``en``, ``hi``,
            ``ne``, ``univ``, or ``other`` label. A pure-English tag stream does
            not activate Roman-Hindi patterns.
        transliterated_name_config: Optional configuration for the
            Latin-script Indian name allow/deny bridge. The default bridge is
            conservative and can be replaced or extended by configuration.
        audit: Return a deterministic AuditReport instead of the
            DeidentificationResult.
        cache_results: Whether to cache this result in the in-process LRU cache. Cached results may contain PHI, but are never saved to disk.
        max_cache_entries: Maximum number of cached results.

    Returns:
        DeidentificationResult with original and de-identified text, or
        AuditReport when ``audit=True``.

    Example:
        >>> from datetime import datetime
        >>> from types import SimpleNamespace
        >>> from unittest.mock import patch
        >>> from openmed.core.pii import (
        ...     DeidentificationResult,
        ...     PIIEntity,
        ...     deidentify,
        ... )
        >>> fixture = DeidentificationResult(
        ...     original_text="Patient Casey Example",
        ...     deidentified_text="Patient [NAME]",
        ...     pii_entities=[
        ...         PIIEntity(
        ...             text="Casey Example",
        ...             label="NAME",
        ...             start=8,
        ...             end=21,
        ...             confidence=0.98,
        ...             redacted_text="[NAME]",
        ...         )
        ...     ],
        ...     method="mask",
        ...     timestamp=datetime(2026, 1, 1, 0, 0, 0),
        ...     mapping={"[NAME]": "Casey Example"},
        ... )
        >>> with patch("openmed.core.pipeline.Pipeline") as pipeline_cls:
        ...     pipeline_cls.return_value.run.return_value = SimpleNamespace(
        ...         deidentification_result=fixture
        ...     )
        ...     result = deidentify(
        ...         "Patient Casey Example",
        ...         method="mask",
        ...         keep_mapping=True,
        ...     )
        >>> result.deidentified_text
        'Patient [NAME]'
        >>> result.mapping
        {'[NAME]': 'Casey Example'}
    """

    if cache_results:
        params = dict(locals())
        cache_key = make_cache_key("deidentify", params)
        cache = get_result_cache(max_entries=max_cache_entries)
        final_result = cache.get(cache_key)
        if final_result is not None:
            return final_result
    from .pipeline import Pipeline

    recognizer_config = custom_recognizer
    if abdm_mode_enabled(abdm, policy=policy, lang=lang, locale=locale):
        recognizer_config = with_abdm_recognizer(recognizer_config)

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
        calibration_thresholds_path=(
            str(calibration_thresholds_path)
            if calibration_thresholds_path is not None
            else None
        ),
        custom_recognizer=recognizer_config,
        code_mixed=code_mixed,
        token_language_tags=token_language_tags,
        transliterated_name_config=transliterated_name_config,
    )
    result = pipeline.run(
        text,
        method=method,
        keep_year=keep_year,
        shift_dates=shift_dates,
        date_shift_days=date_shift_days,
        patient_key=patient_key,
        date_shift_max_days=date_shift_max_days,
        date_shift_secret=date_shift_secret,
        keep_mapping=keep_mapping,
        consistent=consistent,
        seed=seed,
        locale=locale,
        surrogate_vault=surrogate_vault,
        audit=audit,
    )

    if audit and result.deidentification_result.audit_report is not None:
        final_result = result.deidentification_result.audit_report
    else:
        final_result = result.deidentification_result
    if cache_results:
        cache.set(cache_key, final_result)
    return final_result


def _is_date_entity(entity: PIIEntity, lang: str = "en") -> bool:
    """Return True if ``entity`` is a DATE span.

    ``entity.entity_type`` holds the raw model label, whose spelling and case
    vary across models (the default English model emits lowercase ``"date"``),
    so comparing it to the literal ``"DATE"`` misses dates for most models. The
    canonical label normalizes these to a date label; fall back to normalizing
    the raw label when ``canonical_label`` is unset.
    """
    from .labels import DATE, DATE_OF_BIRTH, normalize_label

    canonical = entity.canonical_label or normalize_label(entity.entity_type, lang)
    return canonical in {DATE, DATE_OF_BIRTH}


def _redact_entity(
    entity: PIIEntity,
    method: DeidentificationMethod,
    keep_year: bool = False,
    date_shift_days: Optional[int] = None,
    lang: str = "en",
    anonymizer: Optional["Anonymizer"] = None,
    require_dateutil: bool = False,
    surrogate_vault: Optional["SurrogateVault"] = None,
) -> str:
    """Redact a single PII entity based on method.

    Args:
        entity: PIIEntity to redact
        method: De-identification method
        keep_year: Keep year in dates
        date_shift_days: Days to shift dates
        lang: Language code for fake data and date formatting
        anonymizer: Pre-built ``Anonymizer`` instance for ``method="replace"``
            or ``method="format_preserve"``.
            When ``None``, a fresh per-call instance is built using the
            language default (random, non-deterministic).
        require_dateutil: Require python-dateutil instead of silently falling
            back to the basic regex shifter.
        surrogate_vault: Optional vault for stable cross-document replacements.

    Returns:
        Redacted text replacement
    """
    if method == "mask":
        # Replace with placeholder
        return _mask_placeholder(entity)

    elif method == "aadhaar_mask":
        original = entity.original_text or entity.text
        from .pii_i18n import validate_aadhaar

        if validate_aadhaar(original):
            from .anonymizer.format_preserve import mask_aadhaar

            return mask_aadhaar(original)
        return _mask_placeholder(entity)

    elif method == "remove":
        # Remove entirely (replace with empty string)
        return ""

    elif method == "replace":
        if anonymizer is not None:
            original = entity.original_text or entity.text
            surrogate_label = (
                "OTHER" if _is_mixed_label_union(entity) else entity.entity_type
            )
            if surrogate_vault is not None:
                label = (
                    "OTHER"
                    if _is_mixed_label_union(entity)
                    else entity.canonical_label or entity.entity_type
                )

                def _create_surrogate(attempt: int) -> str:
                    source = original if attempt == 0 else f"{original}|{attempt}"
                    return anonymizer.surrogate(
                        source,
                        surrogate_label,
                        lang=lang,
                    )

                return surrogate_vault.get_or_create(
                    original,
                    label=label,
                    lang=lang,
                    create_surrogate=_create_surrogate,
                    required_script=_surrogate_script_constraint(entity),
                )
            return anonymizer.surrogate(
                original,
                surrogate_label,
                lang=lang,
            )
        return _generate_fake_pii(entity.entity_type, lang=lang)

    elif method == "format_preserve":
        if _is_mixed_label_union(entity):
            return _mask_placeholder(entity)
        if anonymizer is not None:
            surrogate = anonymizer.format_preserving_surrogate(
                entity.original_text or entity.text,
                entity.entity_type,
                lang=lang,
            )
            if surrogate is not None:
                return surrogate
        return _mask_placeholder(entity)

    elif method == "hash":
        # Generate consistent hash
        hash_val = hashlib.sha256(entity.text.encode()).hexdigest()[:8]
        entity.hash_value = hash_val
        return f"{entity.entity_type}_{hash_val}"

    elif method == "shift_dates":
        # Shift dates by offset
        if _is_mixed_label_union(entity):
            return _mask_placeholder(entity)
        if _is_date_entity(entity, lang) and date_shift_days is not None:
            return _shift_date(
                entity.text,
                date_shift_days,
                keep_year,
                lang=lang,
                require_dateutil=require_dateutil,
            )
        else:
            # Non-date entities get masked
            return _mask_placeholder(entity)

    return entity.text


def _surrogate_script_constraint(entity: PIIEntity) -> Optional[str]:
    """Keep Roman-script personal names in their source script run."""
    label = str(entity.canonical_label or entity.entity_type).upper()
    if label not in {"NAME", "PATIENT", "PERSON"}:
        return None
    source = entity.original_text or entity.text
    letters = [char for char in source if char.isalpha()]
    if letters and all("LATIN" in unicodedata.name(char, "") for char in letters):
        return "Latin"
    return None


def _mask_placeholder(entity: PIIEntity) -> str:
    return f"[{entity.entity_type}]"


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
    "teudat_zehut": "ID_NUM",
    "TEUDAT_ZEHUT": "ID_NUM",
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
    date_str: str,
    lang: str,
) -> tuple[datetime, str] | None:
    """Parse localized month-name dates that dateutil may not understand."""
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    month_names = LANGUAGE_MONTH_NAMES.get(lang)
    if not month_names:
        return None

    month_alts = "|".join(re.escape(name) for name in month_names)
    text = date_str.strip()

    if lang in {"es", "pt"}:
        patterns = [
            (
                rf"^(?P<day>\d{{1,2}})\s+de\s+(?P<month>{month_alts})\s+de\s+(?P<year>\d{{4}})$",
                "day_month_year_de",
            )
        ]
    elif lang == "de":
        patterns = [
            (
                rf"^(?P<day>\d{{1,2}})(?P<dot>\.)?\s+(?P<month>{month_alts})\s+(?P<year>\d{{4}})$",
                "day_month_year_dot",
            )
        ]
    else:
        patterns = [
            (
                rf"^(?P<day>\d{{1,2}})\s+(?P<month>{month_alts})\s+(?P<year>\d{{4}})$",
                "day_month_year",
            ),
            (
                rf"^(?P<month>{month_alts})\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})$",
                "month_day_year",
            ),
        ]

    match = None
    style = ""
    for pattern, candidate_style in patterns:
        match = re.match(pattern, text, re.IGNORECASE)
        if match:
            style = candidate_style
            break
    if match is None:
        return None

    month_lookup = {
        name.casefold(): index + 1 for index, name in enumerate(month_names)
    }
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
    new_date: datetime,
    lang: str,
    style: str,
) -> str:
    """Render a localized month-name date using the language month table."""
    from .pii_i18n import LANGUAGE_MONTH_NAMES

    month_name = LANGUAGE_MONTH_NAMES.get(lang, LANGUAGE_MONTH_NAMES["en"])[
        new_date.month - 1
    ]

    if style == "day_month_year_de":
        return f"{new_date.day} de {month_name} de {new_date.year}"
    if style == "day_month_year_dot":
        return f"{new_date.day}. {month_name} {new_date.year}"
    if style == "month_day_year":
        return f"{month_name} {new_date.day}, {new_date.year}"
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


def _random_nonzero_shift(low: int = -365, high: int = 365) -> int:
    """Return a random non-zero day offset in ``[low, high]`` excluding 0.

    A zero-day shift would leave clinical dates unchanged, silently defeating
    de-identification, so the auto-selected offset must never be 0.

    Args:
        low: Minimum (most-negative) shift in days, inclusive.
        high: Maximum (most-positive) shift in days, inclusive.

    Returns:
        A non-zero integer day offset within the range.
    """
    if low > high:
        raise ValueError("low must be less than or equal to high")
    if low == high == 0:
        raise ValueError("range must contain at least one non-zero shift")

    while True:
        shift_days = random.randint(low, high)
        if shift_days != 0:
            return shift_days


def _validate_date_shift_max_days(max_days: int) -> int:
    if isinstance(max_days, bool) or not isinstance(max_days, int):
        raise TypeError("date_shift_max_days must be an integer")
    if max_days <= 0:
        raise ValueError("date_shift_max_days must be positive")
    return max_days


def _stable_date_shift_days(
    patient_key: str | bytes,
    *,
    date_shift_days: Optional[int],
    date_shift_max_days: Optional[int],
    date_shift_secret: Optional[str | bytes],
) -> int:
    """Resolve a patient-keyed stable date shift without retaining the key."""
    if date_shift_max_days is not None:
        max_days = _validate_date_shift_max_days(date_shift_max_days)
    elif date_shift_days is not None:
        max_days = _validate_date_shift_max_days(abs(date_shift_days))
    else:
        max_days = DEFAULT_DATE_SHIFT_MAX_DAYS

    return stable_offset_for(
        patient_key,
        max_days=max_days,
        secret=date_shift_secret,
    )


def _resolve_date_shift_days(
    *,
    date_shift_days: Optional[int],
    patient_key: Optional[str | bytes],
    date_shift_max_days: Optional[int],
    date_shift_secret: Optional[str | bytes],
) -> int:
    if patient_key is not None:
        return _stable_date_shift_days(
            patient_key,
            date_shift_days=date_shift_days,
            date_shift_max_days=date_shift_max_days,
            date_shift_secret=date_shift_secret,
        )

    if date_shift_days is not None:
        return date_shift_days

    if date_shift_max_days is not None:
        max_days = _validate_date_shift_max_days(date_shift_max_days)
        return _random_nonzero_shift(-max_days, max_days)

    return _random_nonzero_shift()


def _shift_date(
    date_str: str,
    shift_days: int,
    keep_year: bool = False,
    lang: str = "en",
    *,
    require_dateutil: bool = False,
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
        require_dateutil: Raise a clear optional-dependency error when
            python-dateutil is unavailable. When false, fall back to the basic
            regex shifter.

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
    except ImportError as exc:
        if require_dateutil:
            _raise_missing_optional_dependency(
                package="python-dateutil",
                extra="dev",
                feature="method='shift_dates' date parsing",
                cause=exc,
            )
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
    date_str: str,
    shift_days: int,
    keep_year: bool = False,
    lang: str = "en",
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
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    elif lang == "de":
        # German: DD.MM.YYYY
        patterns = [
            (r"(\d{1,2})\.(\d{1,2})\.(\d{2,4})", "dmy"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "dmy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
        ]
    elif lang == "ja":
        # Japanese: YYYY/MM/DD (kanji-form 年月日 is handled by the
        # JAPANESE_PII_PATTERNS regex, not here).
        patterns = [
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "dmy"),
        ]
    else:
        # US/English: MM/DD/YYYY first
        patterns = [
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "mdy"),
            (r"(\d{4})[/\-](\d{1,2})[/\-](\d{1,2})", "ymd"),
            (r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", "dmy"),
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
                    return (
                        f"{shifted.month:02d}{sep}{shifted.day:02d}{sep}{shifted.year}"
                    )
                elif order == "ymd":
                    return (
                        f"{shifted.year}{sep}{shifted.month:02d}{sep}{shifted.day:02d}"
                    )
                else:
                    return (
                        f"{shifted.day:02d}{sep}{shifted.month:02d}{sep}{shifted.year}"
                    )

            except (ValueError, OverflowError):
                continue

    return "[DATE_SHIFTED]"


def _format_date_like_original(
    original: str,
    new_date: datetime,
    lang: str = "en",
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
    if re.match(r"\d{1,2}/\d{1,2}/\d{2,4}", original_stripped):
        if lang in _DAY_FIRST_LANGS:
            # European: DD/MM/YYYY
            return new_date.strftime("%d/%m/%Y")
        else:
            # US: MM/DD/YYYY
            return new_date.strftime("%m/%d/%Y")

    # Dash-separated dates
    if re.match(r"\d{1,2}-\d{1,2}-\d{2,4}", original_stripped):
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
            if re.match(
                r"\d+\s+de\s+[^\W\d_]+\s+de\s+\d{4}", original_stripped, re.UNICODE
            ):
                return f"{new_date.day} de {month_name} de {new_date.year}"
            if re.match(r"[^\W\d_]+\s+\d+,?\s+\d{4}", original_stripped, re.UNICODE):
                return f"{month_name} {new_date.day}, {new_date.year}"
            break

    # Default to ISO format
    return new_date.strftime("%Y-%m-%d")


def reidentify(
    deidentified_text: str,
    mapping: Mapping[str, str],
) -> str:
    """Re-identify text using stored mapping.

    Restores original PII from de-identified text using the mapping created
    during de-identification. Only works if keep_mapping=True was used.

    Args:
        deidentified_text: De-identified text
        mapping: Mapping from redacted to original text, optionally including
            occurrence-aware entries emitted for colliding replacement values.

    Returns:
        Re-identified text with original PII restored

    Example:
        >>> from openmed.core.pii import reidentify
        >>> reidentify(
        ...     "Patient [NAME] has record [ID]",
        ...     {"[NAME]": "Casey Example", "[ID]": "MRN-0001"},
        ... )
        'Patient Casey Example has record MRN-0001'

    Note:
        Only works if keep_mapping=True was used during de-identification.
        Requires proper authorization and audit logging in production.
    """
    result = deidentified_text
    regular_mapping: dict[str, str] = {}
    occurrence_mapping: dict[str, list[tuple[int, str]]] = {}
    for redacted, original in mapping.items():
        parsed = _parse_occurrence_mapping_key(redacted)
        if parsed is None:
            regular_mapping[redacted] = original
            continue
        ordinal, surface = parsed
        occurrence_mapping.setdefault(surface, []).append((ordinal, original))

    for redacted in sorted(occurrence_mapping, key=len, reverse=True):
        originals = [original for _, original in sorted(occurrence_mapping[redacted])]
        occurrence_index = 0

        def restore_occurrence(match: re.Match[str]) -> str:
            nonlocal occurrence_index
            if occurrence_index >= len(originals):
                return match.group(0)
            original = originals[occurrence_index]
            occurrence_index += 1
            return original

        result = re.sub(re.escape(redacted), restore_occurrence, result)

    for redacted, original in regular_mapping.items():
        result = result.replace(redacted, original)

    return result


def _build_reidentification_mapping(
    occurrences: Mapping[str, list[tuple[int, str]]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for redacted, values in occurrences.items():
        ordered = sorted(values)
        originals = [original for _, original in ordered]
        if len(set(originals)) == 1:
            mapping[redacted] = originals[0]
            continue
        for ordinal, original in enumerate(originals, start=1):
            key = f"{_OCCURRENCE_MAPPING_PREFIX}{ordinal:08d}:{redacted}"
            mapping[key] = original
    return mapping


def _parse_occurrence_mapping_key(key: str) -> tuple[int, str] | None:
    if not key.startswith(_OCCURRENCE_MAPPING_PREFIX):
        return None
    payload = key[len(_OCCURRENCE_MAPPING_PREFIX) :]
    ordinal_text, separator, redacted = payload.partition(":")
    if not separator or not ordinal_text.isdigit() or not redacted:
        return None
    return int(ordinal_text), redacted
