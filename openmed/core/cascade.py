"""Cheapest-first local detector cascade for privacy spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .arbitration import (
    MODE_BALANCED,
    MODE_HIGH_RECALL_UNION,
    ScoreCalibrator,
    arbitrate,
    arbitration_mode,
)
from .schemas.span import OpenMedSpan


Detector = Callable[..., Sequence[OpenMedSpan]]
OfflineHook = Callable[[str, Any], bool | None]

R0_RULES = "R0"
R1_TINY = "R1"
R2_BASE = "R2"
R3_ACCURATE = "R3"
R4_LOCAL_SLM = "R4"


@dataclass(frozen=True)
class CascadeStageResult:
    route: str
    name: str
    spans: tuple[OpenMedSpan, ...]
    reason: str


@dataclass(frozen=True)
class CascadeResult:
    spans: tuple[OpenMedSpan, ...]
    stage_results: tuple[CascadeStageResult, ...]
    mode: str

    def reached(self, route: str) -> bool:
        return any(stage.route == route for stage in self.stage_results)

    def stage(self, route: str) -> CascadeStageResult:
        for stage in self.stage_results:
            if stage.route == route:
                return stage
        raise KeyError(route)


class CascadeRouter:
    """Run local detectors R0..R4 and arbitrate the resulting span union."""

    def __init__(
        self,
        *,
        rules_detector: Detector | None = None,
        tiny_detector: Detector | None = None,
        base_detector: Detector | None = None,
        accurate_detector: Detector | None = None,
        local_slm_detector: Detector | None = None,
        strict_no_leak: bool = False,
        high_recall: bool = False,
        allow_local_slm: bool = False,
        low_confidence_threshold: float = 0.5,
        clinical_doc_types: Sequence[str] = ("clinical", "note", "ehr", "medical"),
        offline_hook: OfflineHook | None = None,
        calibrator: ScoreCalibrator | None = None,
        label_floors: Mapping[str, float] | None = None,
        high_recall_label_floors: Mapping[str, float] | None = None,
        arbitration_mode_override: str | None = None,
    ) -> None:
        self.rules_detector = rules_detector
        self.tiny_detector = tiny_detector
        self.base_detector = base_detector
        self.accurate_detector = accurate_detector
        self.local_slm_detector = local_slm_detector
        self.strict_no_leak = strict_no_leak
        self.high_recall = high_recall
        self.allow_local_slm = allow_local_slm
        self.low_confidence_threshold = low_confidence_threshold
        self.clinical_doc_types = tuple(item.lower() for item in clinical_doc_types)
        self.offline_hook = offline_hook
        self.calibrator = calibrator
        self.label_floors = label_floors
        self.high_recall_label_floors = high_recall_label_floors
        self.arbitration_mode_override = arbitration_mode_override

    def run(
        self,
        text: str,
        *,
        context: Any = None,
        doc_type: str | None = None,
        strict_no_leak: bool | None = None,
        high_recall: bool | None = None,
        audit_flag: bool = False,
    ) -> CascadeResult:
        strict = self.strict_no_leak if strict_no_leak is None else strict_no_leak
        recall = self.high_recall if high_recall is None else high_recall
        selected_mode = (
            self.arbitration_mode_override
            or arbitration_mode(strict_no_leak=strict or recall)
        )

        stage_results: list[CascadeStageResult] = []
        all_spans: list[OpenMedSpan] = []

        r0_spans = self._run_detector(
            R0_RULES,
            "rules",
            self.rules_detector,
            text,
            context=context,
            reason="always",
        )
        stage_results.append(
            CascadeStageResult(R0_RULES, "rules", r0_spans, "always")
        )
        all_spans.extend(r0_spans)

        r1_spans = self._run_detector(
            R1_TINY,
            "tiny",
            self.tiny_detector,
            text,
            context=context,
            reason="always",
        )
        stage_results.append(CascadeStageResult(R1_TINY, "tiny", r1_spans, "always"))
        all_spans.extend(r1_spans)

        doc_type_value = self._doc_type(context, doc_type)
        r2_reason = self._r2_reason(all_spans, doc_type_value)
        r2_spans: tuple[OpenMedSpan, ...] = ()
        if r2_reason is not None:
            r2_spans = self._run_detector(
                R2_BASE,
                "base",
                self.base_detector,
                text,
                context=context,
                reason=r2_reason,
            )
            stage_results.append(CascadeStageResult(R2_BASE, "base", r2_spans, r2_reason))
            all_spans.extend(r2_spans)

        r3_reason = self._r3_reason(
            strict=strict,
            high_recall=recall,
            before_r2=(*r0_spans, *r1_spans),
            r2_spans=r2_spans,
        )
        if r3_reason is not None:
            r3_spans = self._run_detector(
                R3_ACCURATE,
                "accurate",
                self.accurate_detector,
                text,
                context=context,
                reason=r3_reason,
            )
            stage_results.append(
                CascadeStageResult(R3_ACCURATE, "accurate", r3_spans, r3_reason)
            )
            all_spans.extend(r3_spans)

        r4_reason = self._r4_reason(all_spans, audit_flag=audit_flag)
        if r4_reason is not None:
            r4_spans = self._run_detector(
                R4_LOCAL_SLM,
                "local_slm",
                self.local_slm_detector,
                text,
                context=context,
                reason=r4_reason,
            )
            stage_results.append(
                CascadeStageResult(R4_LOCAL_SLM, "local_slm", r4_spans, r4_reason)
            )
            all_spans.extend(r4_spans)

        final_spans = arbitrate(
            all_spans,
            mode=selected_mode,
            strict_no_leak=strict,
            calibrator=self.calibrator,
            label_floors=self.label_floors,
            high_recall_label_floors=self.high_recall_label_floors,
        )
        return CascadeResult(
            spans=final_spans,
            stage_results=tuple(stage_results),
            mode=selected_mode,
        )

    def _run_detector(
        self,
        route: str,
        name: str,
        detector: Detector | None,
        text: str,
        *,
        context: Any,
        reason: str,
    ) -> tuple[OpenMedSpan, ...]:
        if detector is None:
            return ()
        self._assert_local(route, detector)
        try:
            result = detector(text, context=context, route=route, reason=reason)
        except TypeError:
            try:
                result = detector(text, context=context)
            except TypeError:
                result = detector(text)
        return tuple(result or ())

    def _assert_local(self, route: str, detector: Any) -> None:
        if bool(getattr(detector, "network_egress", False)):
            raise RuntimeError(f"{route} detector is not local-only")
        if bool(getattr(detector, "allows_network", False)):
            raise RuntimeError(f"{route} detector is not local-only")
        if self.offline_hook is not None and self.offline_hook(route, detector) is False:
            raise RuntimeError(f"{route} detector failed offline assertion")

    def _doc_type(self, context: Any, doc_type: str | None) -> str | None:
        if doc_type:
            return doc_type.lower()
        section_metadata = getattr(context, "section_metadata", None)
        if isinstance(section_metadata, Mapping):
            value = section_metadata.get("doc_type")
            if value:
                return str(value).lower()
        return None

    def _r2_reason(
        self,
        spans: Sequence[OpenMedSpan],
        doc_type: str | None,
    ) -> str | None:
        if doc_type and doc_type.lower() in self.clinical_doc_types:
            return "clinical_doc_type"
        if _has_low_confidence(spans, self.low_confidence_threshold):
            return "low_confidence"
        return None

    def _r3_reason(
        self,
        *,
        strict: bool,
        high_recall: bool,
        before_r2: Sequence[OpenMedSpan],
        r2_spans: Sequence[OpenMedSpan],
    ) -> str | None:
        if strict:
            return "strict_no_leak"
        if high_recall:
            return "high_recall"
        if r2_spans and _has_disagreement(before_r2, r2_spans):
            return "r2_disagreement"
        return None

    def _r4_reason(
        self,
        spans: Sequence[OpenMedSpan],
        *,
        audit_flag: bool,
    ) -> str | None:
        if not self.allow_local_slm:
            return None
        if audit_flag:
            return "audit_flag"
        if _has_low_confidence(spans, self.low_confidence_threshold):
            return "low_confidence"
        return None


def _has_low_confidence(
    spans: Sequence[OpenMedSpan],
    threshold: float,
) -> bool:
    if not spans:
        return True
    return any(span.score is None or float(span.score) < threshold for span in spans)


def _has_disagreement(
    left: Sequence[OpenMedSpan],
    right: Sequence[OpenMedSpan],
) -> bool:
    for left_span in left:
        for right_span in right:
            if left_span.start >= right_span.end or left_span.end <= right_span.start:
                continue
            if (
                left_span.start != right_span.start
                or left_span.end != right_span.end
                or left_span.canonical_label != right_span.canonical_label
            ):
                return True
    return False


__all__ = [
    "CascadeResult",
    "CascadeRouter",
    "CascadeStageResult",
    "MODE_BALANCED",
    "MODE_HIGH_RECALL_UNION",
    "R0_RULES",
    "R1_TINY",
    "R2_BASE",
    "R3_ACCURATE",
    "R4_LOCAL_SLM",
]
