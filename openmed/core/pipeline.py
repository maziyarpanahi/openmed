"""Staged privacy de-identification pipeline runtime."""

from __future__ import annotations

import copy
import unicodedata
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from .labels import hipaa_class_for, normalize_label, policy_label_for
from .pii_entity_merger import PII_PATTERNS, PIIPattern
from .schemas.span import ACTION_KEEP, OpenMedSpan, hmac_text_hash

STAGE_NAMES: tuple[str, ...] = (
    "normalize",
    "language_script",
    "doc_type_section",
    "deterministic_detectors",
    "fast_pii_model",
    "clinical_phi_model",
    "span_arbitration",
    "policy_actions",
    "safety_sweep",
    "emit",
)

DEFAULT_HASH_SECRET = b"openmed-pipeline-v1"


@dataclass(frozen=True)
class OffsetMap:
    """Bidirectional character offset map from source to normalized text."""

    original_to_normalized: tuple[int | None, ...]
    normalized_to_original: tuple[int, ...]
    normalized_to_original_span: tuple[tuple[int, int], ...]

    def original_index_to_normalized(self, index: int) -> int | None:
        return self.original_to_normalized[index]

    def normalized_index_to_original(self, index: int) -> int:
        return self.normalized_to_original[index]

    def original_span_to_normalized(self, start: int, end: int) -> tuple[int, int]:
        mapped = [
            value
            for value in self.original_to_normalized[start:end]
            if value is not None
        ]
        if not mapped:
            cursor = min(max(start, 0), len(self.original_to_normalized))
            while cursor < len(self.original_to_normalized):
                value = self.original_to_normalized[cursor]
                if value is not None:
                    return value, value
                cursor += 1
            return len(self.normalized_to_original), len(self.normalized_to_original)
        return min(mapped), max(mapped) + 1

    def normalized_span_to_original_offsets(
        self, start: int, end: int
    ) -> tuple[int, int]:
        if start == end:
            if start >= len(self.normalized_to_original_span):
                terminal = (
                    self.normalized_to_original_span[-1][1]
                    if self.normalized_to_original_span
                    else 0
                )
                return terminal, terminal
            original = self.normalized_to_original_span[start][0]
            return original, original

        spans = self.normalized_to_original_span[start:end]
        if not spans:
            return 0, 0
        return min(span[0] for span in spans), max(span[1] for span in spans)


@dataclass(frozen=True)
class NormalizedDocument:
    original_text: str
    normalized_text: str
    offset_map: OffsetMap
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageRoute:
    lang: str
    script: str
    model_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineStageResult:
    stage: int
    name: str
    spans: tuple[OpenMedSpan, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineContext:
    doc_id: str
    original_text: str
    normalized_text: str
    offset_map: OffsetMap
    route: LanguageRoute
    section_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    original_text: str
    normalized_text: str
    offset_map: OffsetMap
    route: LanguageRoute
    spans: tuple[OpenMedSpan, ...]
    stage_results: tuple[PipelineStageResult, ...]
    redacted_text: str
    audit_record: Mapping[str, Any]
    deidentification_result: Any = None

    def stage(self, name: str) -> PipelineStageResult:
        for result in self.stage_results:
            if result.name == name:
                return result
        raise KeyError(name)


SpanHook = Callable[[Sequence[OpenMedSpan], PipelineContext], Sequence[OpenMedSpan]]
ModelDetector = Callable[..., Any]


class Pipeline:
    """Orchestrate the ten-stage privacy detection pipeline."""

    stage_names = STAGE_NAMES

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        confidence_threshold: float = 0.7,
        config: Any = None,
        use_smart_merging: bool = True,
        lang: str = "en",
        normalize_accents: Optional[bool] = None,
        use_safety_sweep: bool = True,
        loader: Any = None,
        privacy_filter_pipeline: Any = None,
        model_detector: ModelDetector | None = None,
        clinical_model_detector: ModelDetector | None = None,
        cascade_router: Any = None,
        arbitration: SpanHook | None = None,
        arbitration_mode: str | None = None,
        strict_no_leak: bool = False,
        policy_profile: str | None = None,
        policy: Any = None,
        threshold_matrix: Mapping[str, Any] | None = None,
        calibration_thresholds_path: str | None = None,
        calibration_thresholds: Mapping[str, Any] | Any | None = None,
        score_calibrator: Any = None,
        arbitration_label_floors: Mapping[str, float] | None = None,
        high_recall_label_floors: Mapping[str, float] | None = None,
        policy_actions: SpanHook | None = None,
        section_detector: Callable[..., Mapping[str, Any]] | None = None,
        hmac_secret: str | bytes = DEFAULT_HASH_SECRET,
    ) -> None:
        from . import pii
        from .policy import load_policy

        resolved_policy = load_policy(policy) if policy is not None else None
        if resolved_policy is not None:
            policy_profile = policy_profile or resolved_policy.threshold_profile
            arbitration_mode = resolved_policy.arbitration_mode
            strict_no_leak = strict_no_leak or resolved_policy.strict_no_leak
            use_safety_sweep = (
                use_safety_sweep or resolved_policy.safety_sweep_mandatory
            )

        self.policy = resolved_policy
        self.model_name = model_name or pii._DEFAULT_EN_MODEL
        self.confidence_threshold = confidence_threshold
        self.config = config
        self.use_smart_merging = use_smart_merging
        self.lang = lang
        self.normalize_accents = normalize_accents
        self.use_safety_sweep = use_safety_sweep
        self.loader = loader
        self.privacy_filter_pipeline = privacy_filter_pipeline
        self.model_detector = model_detector
        self.clinical_model_detector = clinical_model_detector
        self.cascade_router = cascade_router
        self.arbitration = arbitration
        self.arbitration_mode = arbitration_mode
        self.strict_no_leak = strict_no_leak
        self.policy_profile = policy_profile
        self.threshold_matrix = threshold_matrix
        self.calibration_thresholds = _load_calibration_thresholds(
            calibration_thresholds=calibration_thresholds,
            calibration_thresholds_path=calibration_thresholds_path,
        )
        self.score_calibrator = score_calibrator
        self.arbitration_label_floors = arbitration_label_floors
        self.high_recall_label_floors = high_recall_label_floors
        self.policy_actions = policy_actions
        self.section_detector = section_detector
        self.hmac_secret = hmac_secret

    def run(
        self,
        text: str,
        *,
        method: str = "mask",
        keep_year: bool = True,
        shift_dates: Optional[bool] = None,
        date_shift_days: Optional[int] = None,
        keep_mapping: bool = False,
        consistent: bool = False,
        seed: Optional[int] = None,
        locale: Optional[str] = None,
        doc_id: str | None = None,
        audit: bool = False,
    ) -> PipelineResult:
        from . import pii

        effective_method = pii._resolve_deidentification_method(
            method,
            shift_dates,
            date_shift_days,
        )
        original_text = text.strip()
        normalized = self.stage1_normalize(original_text)
        route = self.stage2_language_script(normalized.normalized_text)
        resolved_doc_id = doc_id or hmac_text_hash(
            normalized.normalized_text,
            self.hmac_secret,
        )
        section_metadata = self.stage3_doc_type_section(normalized.normalized_text)
        context = PipelineContext(
            doc_id=resolved_doc_id,
            original_text=normalized.original_text,
            normalized_text=normalized.normalized_text,
            offset_map=normalized.offset_map,
            route=route,
            section_metadata=section_metadata,
        )

        stage_results: list[PipelineStageResult] = [
            PipelineStageResult(
                1,
                STAGE_NAMES[0],
                metadata=normalized.metadata,
            ),
            PipelineStageResult(
                2,
                STAGE_NAMES[1],
                metadata={
                    "lang": route.lang,
                    "script": route.script,
                    "model_name": route.model_name,
                    **dict(route.metadata),
                },
            ),
            PipelineStageResult(
                3,
                STAGE_NAMES[2],
                metadata=section_metadata,
            ),
        ]

        cascade_driven = self.cascade_router is not None
        if cascade_driven:
            cascade_result = self.cascade_router.run(
                normalized.normalized_text,
                context=context,
                strict_no_leak=self.strict_no_leak,
                language=route.lang,
                policy_profile=self.policy_profile,
            )
            deterministic_spans = _cascade_stage_spans(cascade_result, {"R0"})
            model_spans = _cascade_stage_spans(cascade_result, {"R1", "R2"})
            clinical_spans = _cascade_stage_spans(cascade_result, {"R3", "R4"})
            pii_result = _prediction_result_from_spans(
                normalized.normalized_text,
                cascade_result.spans,
                model_name="cascade",
            )
            cascade_metadata = {
                "cascade_mode": cascade_result.mode,
                "routes": [
                    {
                        "route": stage.route,
                        "name": stage.name,
                        "reason": stage.reason,
                        "span_count": len(stage.spans),
                    }
                    for stage in cascade_result.stage_results
                ],
            }
            stage_results.append(
                PipelineStageResult(
                    4,
                    STAGE_NAMES[3],
                    spans=deterministic_spans,
                    metadata=cascade_metadata,
                )
            )
            stage_results.append(
                PipelineStageResult(5, STAGE_NAMES[4], spans=model_spans)
            )
            stage_results.append(
                PipelineStageResult(6, STAGE_NAMES[5], spans=clinical_spans)
            )
        else:
            deterministic_spans = self.stage4_deterministic_detectors(
                normalized.normalized_text,
                context,
            )
            stage_results.append(
                PipelineStageResult(4, STAGE_NAMES[3], spans=deterministic_spans)
            )

            pii_result = self.stage5_fast_pii_model(normalized.normalized_text, route)
            self._apply_calibration_thresholds(pii_result, route)
            model_spans = self._entities_to_spans(
                getattr(pii_result, "entities", ()),
                normalized.normalized_text,
                context,
                default_detector=(
                    f"model:{getattr(pii_result, 'model_name', route.model_name)}"
                ),
                stage=STAGE_NAMES[4],
            )
            stage_results.append(
                PipelineStageResult(5, STAGE_NAMES[4], spans=model_spans)
            )

            clinical_spans = self.stage6_clinical_phi_model(
                normalized.normalized_text,
                context,
            )
            stage_results.append(
                PipelineStageResult(6, STAGE_NAMES[5], spans=clinical_spans)
            )

        merged_spans = self.stage7_arbitration(
            (*deterministic_spans, *model_spans, *clinical_spans),
            context,
        )
        if cascade_driven:
            pii_result = _prediction_result_from_spans(
                normalized.normalized_text,
                merged_spans,
                model_name="cascade",
            )
        stage_results.append(PipelineStageResult(7, STAGE_NAMES[6], spans=merged_spans))

        policy_spans = self.stage8_policy_actions(merged_spans, context)
        stage_results.append(PipelineStageResult(8, STAGE_NAMES[7], spans=policy_spans))

        if self.policy is not None:
            previous_metadata = dict(getattr(pii_result, "metadata", None) or {})
            pii_result = _prediction_result_from_spans(
                normalized.normalized_text,
                [span for span in policy_spans if span.action != ACTION_KEEP],
                model_name=f"policy:{self.policy.name}",
            )
            _attach_policy_metadata(pii_result, self.policy)
            if "calibration_thresholds" in previous_metadata:
                metadata = dict(getattr(pii_result, "metadata", None) or {})
                metadata["calibration_thresholds"] = previous_metadata[
                    "calibration_thresholds"
                ]
                pii_result.metadata = metadata

        sweep_spans, sweep_metadata = self.stage9_safety_sweep(
            normalized.normalized_text,
            pii_result,
            context,
        )
        stage_results.append(
            PipelineStageResult(
                9,
                STAGE_NAMES[8],
                spans=sweep_spans,
                metadata=sweep_metadata,
            )
        )

        effective_keep_mapping = keep_mapping or bool(
            self.policy is not None and self.policy.keep_mapping
        )
        emission_pii_result = _remap_pii_result_to_original(
            pii_result,
            normalized,
        )
        deidentified = self.stage10_emit(
            normalized.original_text,
            emission_pii_result,
            effective_method=effective_method,
            keep_year=keep_year,
            date_shift_days=date_shift_days,
            keep_mapping=effective_keep_mapping,
            lang=route.lang,
            consistent=consistent,
            seed=seed,
            locale=locale,
            model_name=route.model_name,
            confidence_threshold=self.confidence_threshold,
            normalize_accents=self.normalize_accents,
            use_smart_merging=self.use_smart_merging,
            use_safety_sweep=self.use_safety_sweep,
            reversible_ids=bool(self.policy is not None and self.policy.reversible_id),
            policy_name=self.policy.name if self.policy is not None else None,
            policy=self.policy.name if self.policy is not None else "hipaa_safe_harbor",
            audit=audit,
        )
        final_spans = (
            sweep_spans if self.policy is not None else (sweep_spans or policy_spans)
        )
        stage_results.append(
            PipelineStageResult(
                10,
                STAGE_NAMES[9],
                spans=final_spans,
                metadata={
                    "redacted_text_hash": hmac_text_hash(
                        deidentified.deidentified_text,
                        self.hmac_secret,
                    ),
                    "audit_stage_count": 10,
                },
            )
        )
        audit_record = self._audit_record(
            context,
            stage_results,
            final_spans,
            redacted_text=deidentified.deidentified_text,
        )

        return PipelineResult(
            original_text=normalized.original_text,
            normalized_text=normalized.normalized_text,
            offset_map=normalized.offset_map,
            route=route,
            spans=final_spans,
            stage_results=tuple(stage_results),
            redacted_text=deidentified.deidentified_text,
            audit_record=audit_record,
            deidentification_result=deidentified,
        )

    def stage1_normalize(self, text: str) -> NormalizedDocument:
        normalized_parts: list[str] = []
        original_to_normalized: list[int | None] = [None] * len(text)
        normalized_to_original: list[int] = []
        normalized_to_original_span: list[tuple[int, int]] = []
        normalized_length = 0

        for start, end, segment, is_whitespace in _iter_normalization_segments(text):
            normalized_start = normalized_length
            if is_whitespace:
                normalized_segment = " "
            else:
                normalized_segment = unicodedata.normalize(
                    "NFC",
                    _repair_encoding_segment(segment),
                )

            if not normalized_segment:
                continue

            for index in range(start, end):
                original_to_normalized[index] = normalized_start

            normalized_parts.append(normalized_segment)
            normalized_length += len(normalized_segment)
            for _ in normalized_segment:
                normalized_to_original.append(start)
                normalized_to_original_span.append((start, end))

        normalized_text = "".join(normalized_parts)
        offset_map = OffsetMap(
            original_to_normalized=tuple(original_to_normalized),
            normalized_to_original=tuple(normalized_to_original),
            normalized_to_original_span=tuple(normalized_to_original_span),
        )
        return NormalizedDocument(
            original_text=text,
            normalized_text=normalized_text,
            offset_map=offset_map,
            metadata={
                "unicode_normalization": "NFC",
                "whitespace_collapsed": normalized_text != text,
                "original_length": len(text),
                "normalized_length": len(normalized_text),
            },
        )

    def stage2_language_script(self, text: str) -> LanguageRoute:
        from . import pii
        from .pii_i18n import DEFAULT_PII_MODELS, SUPPORTED_LANGUAGES

        script = _detect_script(text)
        lang = _lang_from_script(script) if self.lang == "auto" else self.lang
        if lang not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language '{lang}'. "
                f"Supported: {sorted(SUPPORTED_LANGUAGES)}"
            )
        model_name = pii._resolve_effective_pii_model(self.model_name, lang)
        return LanguageRoute(
            lang=lang,
            script=script,
            model_name=model_name,
            metadata={"available_default_model": DEFAULT_PII_MODELS.get(lang)},
        )

    def stage3_doc_type_section(self, text: str) -> Mapping[str, Any]:
        if self.section_detector is not None:
            return dict(self.section_detector(text))

        try:
            from openmed.clinical import sections
        except ImportError:
            return {"section_hook": "unavailable", "sections": ()}

        if hasattr(sections, "detect_sections"):
            return {
                "section_hook": "detect_sections",
                "sections": sections.detect_sections(text),
            }
        return {"section_hook": "unavailable", "sections": ()}

    def stage4_deterministic_detectors(
        self,
        text: str,
        context: PipelineContext,
    ) -> tuple[OpenMedSpan, ...]:
        from .safety_sweep import safety_sweep

        entities = safety_sweep(
            text,
            [],
            lang=context.route.lang,
            patterns=_deterministic_patterns(context.route.lang),
        )
        return self._entities_to_spans(
            entities,
            text,
            context,
            default_detector="rules:regex",
            stage=STAGE_NAMES[3],
        )

    def stage5_fast_pii_model(self, text: str, route: LanguageRoute) -> Any:
        if self.model_detector is not None:
            return self.model_detector(
                text,
                model_name=route.model_name,
                confidence_threshold=self.confidence_threshold,
                config=self.config,
                use_smart_merging=self.use_smart_merging,
                lang=route.lang,
                normalize_accents=self.normalize_accents,
                loader=self.loader,
            )

        from . import pii

        return pii.extract_pii(
            text,
            self.model_name,
            self.confidence_threshold,
            self.config,
            self.use_smart_merging,
            lang=route.lang,
            normalize_accents=self.normalize_accents,
            loader=self.loader,
        )

    def stage6_clinical_phi_model(
        self,
        text: str,
        context: PipelineContext,
    ) -> tuple[OpenMedSpan, ...]:
        if self.clinical_model_detector is None:
            return ()

        result = self.clinical_model_detector(text, context=context)
        return self._entities_to_spans(
            getattr(result, "entities", ()),
            text,
            context,
            default_detector=f"model:{getattr(result, 'model_name', 'clinical_phi')}",
            stage=STAGE_NAMES[5],
        )

    def _apply_calibration_thresholds(
        self,
        pii_result: Any,
        route: LanguageRoute,
    ) -> None:
        if self.calibration_thresholds is None:
            return

        result_model = str(getattr(pii_result, "model_name", None) or route.model_name)
        active = self.calibration_thresholds.active_for(
            model_id=result_model,
            language=route.lang,
        )
        retained = []
        filtered = 0
        for entity in getattr(pii_result, "entities", ()):
            label = normalize_label(str(getattr(entity, "label", "") or ""), route.lang)
            threshold = self.calibration_thresholds.lookup(
                label,
                route.lang,
                model_id=result_model,
                default=self.confidence_threshold,
            )
            active[label] = threshold
            metadata = dict(getattr(entity, "metadata", None) or {})
            metadata["calibration_threshold"] = {
                "threshold": threshold,
                "source": self.calibration_thresholds.source_path or "inline",
                "schema_version": self.calibration_thresholds.schema_version,
                "model_id": result_model,
                "language": route.lang,
            }
            entity.metadata = metadata
            if float(getattr(entity, "confidence", 0.0) or 0.0) >= threshold:
                retained.append(entity)
            else:
                filtered += 1

        pii_result.entities = retained
        if hasattr(pii_result, "num_entities"):
            pii_result.num_entities = len(retained)
        metadata = dict(getattr(pii_result, "metadata", None) or {})
        metadata["calibration_thresholds"] = {
            "active": active,
            "filtered_entities": filtered,
            "source": self.calibration_thresholds.source_path or "inline",
            "schema_version": self.calibration_thresholds.schema_version,
            "model_id": result_model,
            "suite": self.calibration_thresholds.suite,
        }
        pii_result.metadata = metadata

    def stage7_arbitration(
        self,
        spans: Sequence[OpenMedSpan],
        context: PipelineContext,
    ) -> tuple[OpenMedSpan, ...]:
        if self.arbitration is not None:
            return tuple(self.arbitration(spans, context))
        from .arbitration import arbitrate

        return arbitrate(
            spans,
            mode=self.arbitration_mode,
            strict_no_leak=self.strict_no_leak,
            language=context.route.lang,
            policy_profile=self.policy_profile,
            calibrator=self.score_calibrator,
            label_floors=self.arbitration_label_floors,
            high_recall_label_floors=self.high_recall_label_floors,
            threshold_matrix=self.threshold_matrix,
        )

    def stage8_policy_actions(
        self,
        spans: Sequence[OpenMedSpan],
        context: PipelineContext,
    ) -> tuple[OpenMedSpan, ...]:
        if self.policy_actions is not None:
            return tuple(self.policy_actions(spans, context))
        if self.policy is not None:
            return tuple(
                _apply_policy_action(span, self.policy, language=context.route.lang)
                for span in spans
            )
        return tuple(
            _apply_threshold_action(
                span,
                language=context.route.lang,
                policy_profile=self.policy_profile,
                strict_no_leak=self.strict_no_leak,
                matrix=self.threshold_matrix,
            )
            for span in spans
        )

    def stage9_safety_sweep(
        self,
        text: str,
        pii_result: Any,
        context: PipelineContext,
    ) -> tuple[tuple[OpenMedSpan, ...], Mapping[str, Any]]:
        before = _redacted_char_count(getattr(pii_result, "entities", ()))
        spans_added = 0
        if self.use_safety_sweep:
            from . import pii

            spans_added = pii._apply_safety_sweep_to_result(
                text,
                pii_result,
                lang=context.route.lang,
            )
        after = _redacted_char_count(getattr(pii_result, "entities", ()))
        if after < before:
            raise RuntimeError("safety sweep must not reduce redacted character count")

        spans = self._entities_to_spans(
            getattr(pii_result, "entities", ()),
            text,
            context,
            default_detector=(
                f"model:{getattr(pii_result, 'model_name', context.route.model_name)}"
            ),
            stage=STAGE_NAMES[8],
        )
        return spans, {
            "enabled": self.use_safety_sweep,
            "spans_added": spans_added,
            "redacted_chars_before": before,
            "redacted_chars_after": after,
        }

    def stage10_emit(
        self,
        text: str,
        pii_result: Any,
        *,
        effective_method: str,
        keep_year: bool,
        date_shift_days: Optional[int],
        keep_mapping: bool,
        lang: str,
        consistent: bool,
        seed: Optional[int],
        locale: Optional[str],
        model_name: str,
        confidence_threshold: float,
        normalize_accents: Optional[bool],
        use_smart_merging: bool,
        use_safety_sweep: bool,
        reversible_ids: bool = False,
        policy_name: Optional[str] = None,
        policy: str = "hipaa_safe_harbor",
        audit: bool = False,
    ) -> Any:
        from . import pii

        return pii._build_deidentification_result(
            text,
            pii_result,
            effective_method=effective_method,
            keep_year=keep_year,
            date_shift_days=date_shift_days,
            keep_mapping=keep_mapping,
            lang=lang,
            consistent=consistent,
            seed=seed,
            locale=locale,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            normalize_accents=normalize_accents,
            use_smart_merging=use_smart_merging,
            use_safety_sweep=use_safety_sweep,
            reversible_ids=reversible_ids,
            policy_name=policy_name,
            policy=policy,
            audit=audit,
        )

    def _entities_to_spans(
        self,
        entities: Sequence[Any],
        text: str,
        context: PipelineContext,
        *,
        default_detector: str,
        stage: str,
    ) -> tuple[OpenMedSpan, ...]:
        spans: list[OpenMedSpan] = []
        for entity in entities:
            bounds = _entity_bounds(entity, text)
            if bounds is None:
                continue
            start, end = bounds
            surface = text[start:end]
            label = str(getattr(entity, "label", "") or "")
            canonical = normalize_label(label, lang=context.route.lang)
            detector = _detector_for_entity(entity, default_detector)
            metadata = dict(getattr(entity, "metadata", None) or {})
            metadata.setdefault("pipeline_stage", stage)
            spans.append(
                OpenMedSpan(
                    doc_id=context.doc_id,
                    start=start,
                    end=end,
                    text_hash=hmac_text_hash(surface, self.hmac_secret),
                    entity_type=label or canonical,
                    canonical_label=canonical,
                    policy_label=policy_label_for(canonical, lang=context.route.lang),
                    regulatory_tags=(
                        hipaa_class_for(canonical, lang=context.route.lang),
                    ),
                    score=float(getattr(entity, "confidence", 0.0) or 0.0),
                    detector=detector,
                    evidence=_evidence_for_entity(entity),
                    section=_section_for_span(start, end, context.section_metadata),
                    metadata=metadata,
                )
            )
        return tuple(spans)

    def _audit_record(
        self,
        context: PipelineContext,
        stage_results: Sequence[PipelineStageResult],
        final_spans: Sequence[OpenMedSpan],
        *,
        redacted_text: str,
    ) -> Mapping[str, Any]:
        record = {
            "doc_id": context.doc_id,
            "language": context.route.lang,
            "script": context.route.script,
            "model_name": context.route.model_name,
            "input_text_hash": hmac_text_hash(
                context.normalized_text, self.hmac_secret
            ),
            "redacted_text_hash": hmac_text_hash(redacted_text, self.hmac_secret),
            "normalized_length": len(context.normalized_text),
            "redacted_length": len(redacted_text),
            "span_count": len(final_spans),
            "stages": [
                {
                    "stage": result.stage,
                    "name": result.name,
                    "span_count": len(result.spans),
                    "metadata": dict(result.metadata),
                }
                for result in stage_results
            ],
        }
        if self.policy is not None:
            record["policy"] = {
                "name": self.policy.name,
                "schema_version": self.policy.schema_version,
                "arbitration_mode": self.policy.arbitration_mode,
                "threshold_profile": self.policy.threshold_profile,
            }
        return record


def _iter_normalization_segments(text: str):
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            start = index
            index += 1
            while index < len(text) and text[index].isspace():
                index += 1
            yield start, index, text[start:index], True
            continue

        start = index
        index += 1
        while index < len(text) and unicodedata.combining(text[index]):
            index += 1
        yield start, index, text[start:index], False


def _repair_encoding_segment(segment: str) -> str:
    try:
        from ftfy import fix_text
    except ImportError:
        return segment
    return fix_text(segment)


def _detect_script(text: str) -> str:
    for char in text:
        codepoint = ord(char)
        if 0x0600 <= codepoint <= 0x06FF:
            return "arabic"
        if 0x3040 <= codepoint <= 0x30FF or 0x4E00 <= codepoint <= 0x9FFF:
            return "japanese"
        if 0x0900 <= codepoint <= 0x097F:
            return "devanagari"
        if 0x0C00 <= codepoint <= 0x0C7F:
            return "telugu"
    return "latin"


def _lang_from_script(script: str) -> str:
    return {
        "arabic": "ar",
        "japanese": "ja",
        "devanagari": "hi",
        "telugu": "te",
    }.get(script, "en")


def _deterministic_patterns(lang: str) -> list[PIIPattern]:
    from .anonymizer.providers import clinical_ids

    luhn_mrn = PIIPattern(
        r"\b(?:MRN|mrn)[:\s#-]*(?:\d[\s-]?){13,19}\b",
        "medical_record_number",
        priority=20,
        base_score=0.95,
        context_words=["mrn", "medical record", "patient id", "record number"],
        context_boost=0.05,
        validator=clinical_ids.validate_luhn,
    )
    if lang == "en":
        return [luhn_mrn, *PII_PATTERNS]

    from .pii_i18n import get_patterns_for_language

    return [luhn_mrn, *get_patterns_for_language(lang)]


def _entity_bounds(entity: Any, text: str) -> tuple[int, int] | None:
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start <= end <= len(text)
    ):
        return start, end

    surface = str(getattr(entity, "text", "") or "")
    if not surface:
        return None
    found = text.find(surface)
    if found < 0:
        return None
    return found, found + len(surface)


def _detector_for_entity(entity: Any, default_detector: str) -> str:
    metadata = dict(getattr(entity, "metadata", None) or {})
    detector = metadata.get("detector") or metadata.get("source")
    if detector and str(detector).startswith(("rules:", "model:")):
        return str(detector)

    sweep = metadata.get("safety_sweep") if isinstance(metadata, Mapping) else None
    pattern = str((sweep or {}).get("pattern", ""))
    label = str(getattr(entity, "label", "") or "")
    if label == "medical_record_number" and "13,19" in pattern:
        return "rules:mrn_luhn"
    if label == "medical_record_number":
        return "rules:mrn_regex"
    if label in {"credit_debit_card", "credit_card"}:
        return "rules:luhn"
    if label == "npi":
        return "rules:npi_luhn"
    if label == "iban":
        return "rules:iban_mod97"
    if label == "ssn":
        return "rules:ssn"
    if detector == "safety_sweep":
        return "rules:regex"
    return default_detector


def _evidence_for_entity(entity: Any) -> Mapping[str, Any]:
    metadata = dict(getattr(entity, "metadata", None) or {})
    evidence: dict[str, Any] = {}
    if "safety_sweep" in metadata:
        evidence["rule"] = metadata["safety_sweep"]
    if "patterns_version" in metadata:
        evidence["patterns_version"] = metadata["patterns_version"]
    return evidence


def _section_for_span(
    start: int,
    end: int,
    section_metadata: Mapping[str, Any],
) -> str | None:
    sections = section_metadata.get("sections")
    if not sections:
        return None
    for section in sections:
        if not isinstance(section, Mapping):
            continue
        section_start = section.get("start")
        section_end = section.get("end")
        if isinstance(section_start, int) and isinstance(section_end, int):
            if start >= section_start and end <= section_end:
                return str(section.get("label") or section.get("name") or "")
    return None


def _redacted_char_count(entities: Sequence[Any]) -> int:
    intervals: list[tuple[int, int]] = []
    for entity in entities:
        start = getattr(entity, "start", None)
        end = getattr(entity, "end", None)
        if isinstance(start, int) and isinstance(end, int) and start < end:
            intervals.append((start, end))
    if not intervals:
        return 0

    intervals.sort()
    total = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total


def _cascade_stage_spans(
    cascade_result: Any, routes: set[str]
) -> tuple[OpenMedSpan, ...]:
    spans: list[OpenMedSpan] = []
    for stage in getattr(cascade_result, "stage_results", ()):
        if getattr(stage, "route", None) in routes:
            spans.extend(getattr(stage, "spans", ()) or ())
    return tuple(spans)


def _prediction_result_from_spans(
    text: str,
    spans: Sequence[OpenMedSpan],
    *,
    model_name: str,
) -> Any:
    from datetime import datetime

    from ..processing.outputs import EntityPrediction, PredictionResult

    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text=text[span.start : span.end],
                label=span.entity_type or span.canonical_label,
                start=span.start,
                end=span.end,
                confidence=float(span.score or 0.0),
                metadata={
                    **dict(span.metadata),
                    "detector": span.detector,
                    "canonical_label": span.canonical_label,
                },
            )
            for span in spans
        ],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def _remap_pii_result_to_original(pii_result: Any, document: NormalizedDocument) -> Any:
    """Clone a normalized-text PII result with entities mapped to original text."""
    remapped_result = copy.copy(pii_result)
    remapped_entities = []
    for entity in getattr(pii_result, "entities", ()):
        remapped = _remap_entity_to_original(entity, document)
        if remapped is not None:
            remapped_entities.append(remapped)
    remapped_result.entities = remapped_entities
    if hasattr(remapped_result, "text"):
        remapped_result.text = document.original_text
    if hasattr(remapped_result, "num_entities"):
        remapped_result.num_entities = len(remapped_entities)
    return remapped_result


def _remap_entity_to_original(
    entity: Any,
    document: NormalizedDocument,
) -> Any | None:
    bounds = _entity_bounds(entity, document.normalized_text)
    if bounds is None:
        return None

    normalized_start, normalized_end = bounds
    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(
            normalized_start,
            normalized_end,
        )
    )
    original_surface = document.original_text[original_start:original_end]

    remapped = copy.copy(entity)
    remapped.start = original_start
    remapped.end = original_end
    remapped.text = original_surface

    metadata = dict(getattr(entity, "metadata", None) or {})
    metadata.setdefault(
        "normalized_text",
        document.normalized_text[normalized_start:normalized_end],
    )
    metadata.setdefault("normalized_start", normalized_start)
    metadata.setdefault("normalized_end", normalized_end)
    remapped.metadata = metadata
    return remapped


def _load_calibration_thresholds(
    *,
    calibration_thresholds: Mapping[str, Any] | Any | None,
    calibration_thresholds_path: str | None,
) -> Any | None:
    if calibration_thresholds is not None:
        from openmed.eval.calibrate import coerce_calibration_thresholds

        return coerce_calibration_thresholds(calibration_thresholds)
    if calibration_thresholds_path is None:
        return None

    from openmed.eval.calibrate import load_calibration_thresholds

    return load_calibration_thresholds(calibration_thresholds_path)


def _attach_policy_metadata(pii_result: Any, policy: Any) -> None:
    metadata = dict(getattr(pii_result, "metadata", None) or {})
    metadata["policy"] = {
        "name": policy.name,
        "schema_version": policy.schema_version,
        "arbitration_mode": policy.arbitration_mode,
        "threshold_profile": policy.threshold_profile,
        "keep_mapping": policy.keep_mapping,
        "reversible_id": policy.reversible_id,
    }
    pii_result.metadata = metadata


def _apply_policy_action(
    span: OpenMedSpan,
    policy: Any,
    *,
    language: str,
) -> OpenMedSpan:
    action = policy.action_for(span.canonical_label, lang=language)
    metadata = dict(span.metadata)
    metadata["policy_action"] = {
        "policy": policy.name,
        "schema_version": policy.schema_version,
        "action": action,
        "source": "policy_profile",
    }
    if span.action == action:
        return replace(span, metadata=metadata)
    return replace(span, action=action, metadata=metadata)


def _apply_threshold_action(
    span: OpenMedSpan,
    *,
    language: str,
    policy_profile: str | None,
    strict_no_leak: bool,
    matrix: Mapping[str, Any] | None,
) -> OpenMedSpan:
    from .thresholds import lookup_threshold

    profile = policy_profile or ("strict_no_leak" if strict_no_leak else "balanced")
    threshold = lookup_threshold(
        span.canonical_label,
        language,
        profile,
        matrix=matrix,
    )
    action = str(threshold["action"])
    if span.action == action:
        return span

    metadata = dict(span.metadata)
    metadata["threshold_action"] = {
        "policy_profile": threshold["policy_profile"],
        "source": threshold["source"],
        "schema_version": threshold["schema_version"],
    }
    return replace(span, action=action, metadata=metadata)


__all__ = [
    "LanguageRoute",
    "NormalizedDocument",
    "OffsetMap",
    "Pipeline",
    "PipelineContext",
    "PipelineResult",
    "PipelineStageResult",
    "STAGE_NAMES",
]
