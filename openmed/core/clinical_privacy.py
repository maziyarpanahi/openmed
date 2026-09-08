"""Batched clinical privacy orchestration over the shared SDK pipeline.

Model batching is separate from each document's normalization, policy, language
and emission state. Qualification is supplied by the deployment's reviewed
manifest; successful execution alone never qualifies a model or language.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Sequence

from openmed.processing.outputs import EntityPrediction, PredictionResult

from .clinical_identifiers import detect_clinical_identifiers
from .clinical_label_map import (
    CLINICAL_LABEL_MAP_VERSION,
    clinical_label,
    clinical_label_map,
)
from .clinical_language import ClinicalLanguage, resolve_clinical_language
from .clinical_policy import ClinicalPolicy, resolve_clinical_policy
from .config import OpenMedConfig
from .labels import normalize_label
from .pipeline import Pipeline


@dataclass(frozen=True)
class ClinicalPrivacyOptions:
    """Per-document controls; secrets belong to the processor, not this record."""

    language: str = "auto"
    locale: str | None = None
    method: str = "mask"
    confidence_threshold: float = 0.7
    redact_categories: tuple[str, ...] | None = None
    redact_roles: tuple[str, ...] = ("patient", "clinician")
    keep_labels: tuple[str, ...] = ()
    keep_terms: tuple[str, ...] = field(default=(), repr=False)
    date_shift_days: int | None = None
    pseudonym_scope: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.method not in {"mask", "remove", "replace", "hash", "shift_dates"}:
            raise ValueError("unsupported clinical redaction method")
        if (
            isinstance(self.confidence_threshold, bool)
            or not 0 <= self.confidence_threshold <= 1
        ):
            raise ValueError("confidence_threshold must be between zero and one")
        if (
            isinstance(self.keep_terms, (str, bytes))
            or len(self.keep_terms) > 256
            or any(
                not isinstance(term, str) or not term.strip() or len(term) > 128
                for term in self.keep_terms
            )
        ):
            raise ValueError(
                "keep_terms must contain at most 256 non-empty terms of at most 128 characters"
            )
        if self.method == "shift_dates":
            if (
                isinstance(self.date_shift_days, bool)
                or not isinstance(self.date_shift_days, int)
                or not 1 <= abs(self.date_shift_days) <= 3650
            ):
                raise ValueError(
                    "shift_dates requires a non-zero date_shift_days within ten years"
                )
        elif self.date_shift_days is not None:
            raise ValueError("date_shift_days requires method shift_dates")
        if self.method == "hash" and (
            not isinstance(self.pseudonym_scope, str)
            or not 1 <= len(self.pseudonym_scope) <= 256
        ):
            raise ValueError("hash requires an explicit bounded pseudonym_scope")


@dataclass(frozen=True)
class ClinicalPrivacyDocument:
    """One identified source document and its requested controls."""

    id: str
    text: str = field(repr=False)
    options: ClinicalPrivacyOptions = field(default_factory=ClinicalPrivacyOptions)


@dataclass(frozen=True)
class ClinicalPrivacyResult:
    """Safe metadata and transformed text; no plaintext identifier mapping."""

    id: str
    status: str
    complete: bool
    deidentified_text: str | None = field(default=None, repr=False)
    spans: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    language: dict[str, Any] = field(default_factory=dict)
    policy: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the response without source text or replacement mappings."""
        return asdict(self)


class _ClinicalPipeline(Pipeline):
    def __init__(self, document, language, policy, processor):
        self.document = document
        self.language_choice = language
        self.clinical_policy = policy
        self.processor = processor
        self.prepared = None
        self.prediction = None
        self.review_reasons: set[str] = set()
        super().__init__(
            lang=language.language,
            model_name=processor.model_id,
            model_detector=self._model_prediction,
            clinical_model_detector=self._context_prediction,
            policy=policy.profile,
            preserve_whitespace=True,
            config=OpenMedConfig(
                clinical_protect_terms=list(document.options.keep_terms)
            ),
            confidence_threshold=document.options.confidence_threshold,
            hmac_secret=processor._audit_key,
            telemetry_enabled=False,
        )
        self.clinical_protect_options["protect_word_fragments"] = True

    def stage1_normalize(self, text):
        if self.prepared is None:
            self.prepared = super().stage1_normalize(text)
        if text != self.prepared.original_text:
            raise RuntimeError("prepared document does not match source")
        return self.prepared

    def _model_prediction(self, text, **kwargs):
        if self.prediction is None or text != self.prepared.normalized_text:
            raise RuntimeError("model batch does not match the prepared document")
        entities = [
            EntityPrediction(
                e.text,
                clinical_label(e.label),
                e.score,
                e.start,
                e.end,
                metadata={"source_label": e.label},
            )
            for e in self.prediction.entities
            if e.score >= self.document.options.confidence_threshold
        ]
        return self._prediction_result(text, entities, self.processor.model_id)

    def _context_prediction(self, text, **kwargs):
        entities = []
        if self.language_choice.language in {"de", "en"}:
            entities = detect_clinical_identifiers(
                text, language=self.language_choice.language
            )
        return self._prediction_result(text, entities, "clinical_context")

    @staticmethod
    def _prediction_result(text, entities, model):
        return PredictionResult(
            text, entities, model, datetime.now(timezone.utc).isoformat()
        )

    def _decide(self, label, start, end, metadata):
        decision = self.clinical_policy.decide(
            label,
            text=self.prepared.normalized_text,
            start=start,
            end=end,
            role=(metadata or {}).get("clinical_role"),
        )
        if decision.needs_review:
            self.review_reasons.add(decision.reason)
        return decision

    def _policy_spans(self, spans):
        result = []
        for span in spans:
            decision = self._decide(
                span.canonical_label, span.start, span.end, span.metadata
            )
            metadata = dict(span.metadata)
            metadata["policy_action"] = {
                "action": decision.action,
                "source": "clinical_policy",
            }
            if decision.role:
                metadata["clinical_role"] = decision.role
            result.append(replace(span, action=decision.action, metadata=metadata))
        return tuple(result)

    def stage8_policy_actions(self, spans, context, **kwargs):
        return self._policy_spans(
            super().stage8_policy_actions(spans, context, **kwargs)
        )

    def stage9_safety_sweep(self, text, pii_result, context):
        spans, metadata = super().stage9_safety_sweep(text, pii_result, context)
        resolved = self._policy_spans(spans)
        retained = []
        for entity in pii_result.entities:
            decision = self._decide(
                entity.label, entity.start, entity.end, entity.metadata
            )
            if decision.action != "keep":
                retained.append(entity)
        pii_result.entities = retained
        return tuple(span for span in resolved if span.action != "keep"), metadata

    def stage10_emit(self, text, pii_result, **kwargs):
        method = self.document.options.method
        # The profile selects spans. The explicitly requested method controls
        # emission, including remove/date-shift which are not policy actions.
        for entity in pii_result.entities:
            entity.metadata = {
                key: value
                for key, value in (entity.metadata or {}).items()
                if key != "policy_action"
            }
        kwargs["effective_method"] = "mask" if method == "hash" else method
        result = super().stage10_emit(text, pii_result, **kwargs)
        if method == "hash":
            output = text
            for entity in sorted(
                result.pii_entities, key=lambda e: e.start, reverse=True
            ):
                payload = json.dumps(
                    [
                        "openmed-clinical-pseudonym-v1",
                        self.document.options.pseudonym_scope,
                        entity.canonical_label,
                        unicodedata.normalize("NFC", entity.text),
                    ],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
                digest = hmac.new(
                    self.processor._pseudonym_key, payload, hashlib.sha256
                ).hexdigest()[:32]
                replacement = f"[{entity.canonical_label}_{digest}]"
                output = output[: entity.start] + replacement + output[entity.end :]
                entity.redacted_text = entity.surrogate = replacement
                entity.action = "hash"
            result.deidentified_text = output
            result.method = "hash"
        if method == "shift_dates":
            for entity in result.pii_entities:
                if entity.canonical_label in {"DATE", "DATE_OF_BIRTH"} and (
                    entity.redacted_text or ""
                ).startswith("["):
                    self.review_reasons.add("date_shift_unresolved")
        return result


class ClinicalPrivacyProcessor:
    """Run one model batch, then the shared privacy pipeline for each document."""

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        revision: str,
        qualified_languages: Sequence[str] = (),
        pseudonym_key: bytes | None = None,
        batch_size: int = 8,
        max_batch_tokens: int = 4096,
        max_documents: int = 64,
        max_document_chars: int = 100_000,
        max_total_chars: int = 500_000,
    ) -> None:
        if pseudonym_key is not None and (
            not isinstance(pseudonym_key, bytes) or len(pseudonym_key) < 32
        ):
            raise ValueError("pseudonym_key must contain at least 32 secret bytes")
        for limit in (
            batch_size,
            max_batch_tokens,
            max_documents,
            max_document_chars,
            max_total_chars,
        ):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("clinical processor limits must be positive integers")
        self.model, self.model_id, self.revision = model, model_id, revision
        self.label_map = (
            clinical_label_map(model.id2label.values())
            if hasattr(model, "id2label")
            else None
        )
        self.qualified_languages = frozenset(qualified_languages)
        self._pseudonym_key = pseudonym_key
        self._audit_key = secrets.token_bytes(32)
        self.batch_size, self.max_batch_tokens = batch_size, max_batch_tokens
        self.max_documents, self.max_document_chars = max_documents, max_document_chars
        self.max_total_chars = max_total_chars

    def process_batch(
        self,
        documents: Sequence[ClinicalPrivacyDocument],
        *,
        execution_control: Any = None,
    ) -> list[ClinicalPrivacyResult]:
        """Preserve IDs/order and return explicit per-item failure or review state."""
        documents = list(documents)
        if len(documents) > self.max_documents:
            raise ValueError("request exceeds the document limit")
        ids = [document.id for document in documents]
        if any(
            not isinstance(value, str) or not 1 <= len(value) <= 128 for value in ids
        ) or len(set(ids)) != len(ids):
            raise ValueError("document IDs must be unique bounded non-empty strings")
        if (
            sum(
                len(document.text)
                for document in documents
                if isinstance(document.text, str)
            )
            > self.max_total_chars
        ):
            raise ValueError("request exceeds the total character limit")
        results: list[ClinicalPrivacyResult | None] = [None] * len(documents)
        prepared = []
        for index, document in enumerate(documents):
            try:
                if (
                    not isinstance(document.text, str)
                    or not document.text.strip()
                    or len(document.text) > self.max_document_chars
                ):
                    raise ValueError("invalid document")
                if document.options.method == "hash" and self._pseudonym_key is None:
                    raise ValueError("pseudonym key is unavailable")
                language = resolve_clinical_language(
                    document.text,
                    language=document.options.language,
                    locale=document.options.locale,
                )
                if language.language == "und":
                    raise ValueError("language could not be determined")
                categories = document.options.redact_categories
                if document.options.method == "shift_dates":
                    from .clinical_policy import DEFAULT_CATEGORIES

                    categories = tuple(
                        dict.fromkeys(
                            [
                                *(
                                    categories
                                    if categories is not None
                                    else DEFAULT_CATEGORIES
                                ),
                                "dates",
                            ]
                        )
                    )
                policy = resolve_clinical_policy(
                    redact_categories=categories,
                    redact_roles=document.options.redact_roles,
                    keep_labels=document.options.keep_labels,
                )
                pipeline = _ClinicalPipeline(document, language, policy, self)
                pipeline.stage1_normalize(document.text)
                prepared.append((index, pipeline))
            except (TypeError, ValueError):
                results[index] = ClinicalPrivacyResult(
                    document.id, "failed", False, error="invalid_document_or_controls"
                )
        if prepared:
            try:
                execution_kwargs = (
                    {"execution_control": execution_control}
                    if execution_control is not None
                    else {}
                )
                predictions = self.model.predict_batch_detailed(
                    [pipeline.prepared.normalized_text for _, pipeline in prepared],
                    threshold=0.0,
                    batch_size=self.batch_size,
                    max_batch_tokens=self.max_batch_tokens,
                    **execution_kwargs,
                )
                if len(predictions) != len(prepared):
                    raise RuntimeError("incomplete model batch")
            except Exception as exc:
                from openmed.onnx.execution import OnnxExecutionCancelled

                error = (
                    "model_timeout"
                    if isinstance(exc, TimeoutError)
                    else "model_cancelled"
                    if isinstance(exc, OnnxExecutionCancelled)
                    else "model_batch_failed"
                )
                for index, _pipeline in prepared:
                    results[index] = ClinicalPrivacyResult(
                        documents[index].id, "failed", False, error=error
                    )
                return [result for result in results if result is not None]
            for (index, pipeline), prediction in zip(
                prepared, predictions, strict=True
            ):
                try:
                    if execution_control is not None:
                        execution_control.check()
                    if (
                        not prediction.complete
                        or prediction.processed_tokens != prediction.token_count
                    ):
                        raise RuntimeError("incomplete token coverage")
                    pipeline.prediction = prediction
                    results[index] = self._finish(pipeline)
                    if execution_control is not None:
                        execution_control.check()
                except Exception:
                    results[index] = ClinicalPrivacyResult(
                        documents[index].id,
                        "failed",
                        False,
                        error="privacy_pipeline_failed",
                    )
        return [result for result in results if result is not None]

    def _finish(self, pipeline: _ClinicalPipeline) -> ClinicalPrivacyResult:
        document, language, policy = (
            pipeline.document,
            pipeline.language_choice,
            pipeline.clinical_policy,
        )
        result = pipeline.run(
            document.text,
            method=document.options.method,
            keep_year=False,
            date_shift_days=document.options.date_shift_days,
            consistent=True,
            locale=language.locale,
            keep_mapping=False,
        )
        warnings = pipeline.review_reasons
        if language.needs_review:
            warnings.add("language_requires_review")
        if language.language not in self.qualified_languages:
            warnings.add("model_language_not_qualified")
        if policy.narrowed:
            warnings.add("policy_narrowed")
        if document.options.keep_terms:
            from .clinical_protect import load_bundled_terms, normalize_term

            if any(
                normalize_term(term) not in load_bundled_terms()
                for term in document.options.keep_terms
            ):
                warnings.add("custom_protection_requires_review")
        for term in document.options.keep_terms:
            for match in re.finditer(
                r"(?<!\w)" + re.escape(term) + r"(?!\w)", document.text, re.I
            ):
                if any(
                    entity.start < match.end() and match.start() < entity.end
                    for entity in result.deidentification_result.pii_entities
                ):
                    warnings.add("keep_term_conflicts_with_identifier")
        prediction = pipeline.prediction
        return ClinicalPrivacyResult(
            id=document.id,
            status="needs_review" if warnings else "complete",
            complete=True,
            deidentified_text=result.redacted_text,
            spans=tuple(
                {
                    "start": e.start,
                    "end": e.end,
                    "label": normalize_label(e.label),
                    "score": e.confidence,
                    "action": e.action,
                }
                for e in result.deidentification_result.pii_entities
            ),
            warnings=tuple(sorted(warnings)),
            language={
                "language": language.language,
                "locale": language.locale,
                "confidence": language.confidence,
                "source": language.source,
                "mixed": language.mixed,
            },
            policy=policy.metadata(),
            coverage={
                "token_count": prediction.token_count,
                "processed_tokens": prediction.processed_tokens,
                "window_count": prediction.window_count,
            },
            provenance={
                "model_id": self.model_id,
                "revision": self.revision,
                "variant": self.model.variant,
                "pipeline": "clinical-privacy-v1",
                "label_map_version": CLINICAL_LABEL_MAP_VERSION,
            },
        )


__all__ = [
    "ClinicalPrivacyOptions",
    "ClinicalPrivacyDocument",
    "ClinicalPrivacyResult",
    "ClinicalPrivacyProcessor",
]
