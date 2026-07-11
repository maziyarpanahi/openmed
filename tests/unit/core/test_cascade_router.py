import pytest

from openmed.core.cascade import (
    R0_RULES,
    R1_TINY,
    R2_BASE,
    R3_ACCURATE,
    CascadeRouter,
)
from openmed.core.pipeline import STAGE_NAMES, Pipeline
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash


def _span(
    *,
    start: int = 0,
    end: int = 8,
    label: str = "PERSON",
    detector: str = "model:tiny",
    score: float = 0.9,
) -> OpenMedSpan:
    surface = f"{start}:{end}:{label}"
    return OpenMedSpan(
        doc_id="doc-1",
        start=start,
        end=end,
        text_hash=hmac_text_hash(surface, "test-secret"),
        entity_type=label,
        canonical_label=label,
        score=score,
        detector=detector,
    )


def _detector(route_calls, route, spans):
    def detect(text, **kwargs):
        route_calls.append((route, text, kwargs.get("reason")))
        return spans

    return detect


def test_router_reaches_accurate_route_under_strict_no_leak():
    calls = []
    router = CascadeRouter(
        tiny_detector=_detector(calls, R1_TINY, [_span(score=0.95)]),
        accurate_detector=_detector(
            calls,
            R3_ACCURATE,
            [_span(start=16, end=22, detector="model:accurate", score=0.9)],
        ),
        strict_no_leak=True,
    )

    result = router.run("Patient John Doe")

    assert result.reached(R0_RULES)
    assert result.reached(R1_TINY)
    assert result.reached(R3_ACCURATE)
    assert result.mode == "high_recall_union"
    assert [call[0] for call in calls] == [R1_TINY, R3_ACCURATE]


def test_router_stays_at_base_route_without_strict_policy():
    calls = []
    r1_span = _span(detector="model:tiny", score=0.2)
    r2_span = _span(detector="model:base", score=0.8)
    router = CascadeRouter(
        tiny_detector=_detector(calls, R1_TINY, [r1_span]),
        base_detector=_detector(calls, R2_BASE, [r2_span]),
        accurate_detector=_detector(
            calls,
            R3_ACCURATE,
            [_span(detector="model:accurate", score=0.9)],
        ),
    )

    result = router.run("Patient John Doe")

    assert result.reached(R1_TINY)
    assert result.reached(R2_BASE)
    assert not result.reached(R3_ACCURATE)
    assert result.mode == "balanced"
    assert [call[0] for call in calls] == [R1_TINY, R2_BASE]


def test_router_rejects_detector_when_offline_hook_fails():
    router = CascadeRouter(
        tiny_detector=lambda text, **kwargs: [],
        offline_hook=lambda route, detector: False,
    )

    with pytest.raises(RuntimeError, match="offline assertion"):
        router.run("Patient John Doe")


def test_pipeline_can_use_cascade_router_for_detection_stages():
    rules_span = _span(start=0, end=4, detector="rules:regex", score=0.95)
    tiny_span = _span(start=8, end=12, detector="model:tiny", score=0.95)
    router = CascadeRouter(
        rules_detector=lambda text, **kwargs: [rules_span],
        tiny_detector=lambda text, **kwargs: [tiny_span],
    )

    result = Pipeline(
        cascade_router=router,
        use_safety_sweep=False,
    ).run("John met Jane")

    assert result.stage("deterministic_detectors").spans == (rules_span,)
    assert result.stage("fast_pii_model").spans == (tiny_span,)
    assert result.stage("span_arbitration").spans == (rules_span, tiny_span)


def test_pipeline_reports_complete_stage_and_shared_cascade_durations(monkeypatch):
    ticks = iter(index / 1000 for index in range(100))
    monkeypatch.setattr(
        "openmed.core.pipeline.perf_counter",
        lambda: next(ticks),
    )
    router = CascadeRouter(
        rules_detector=lambda text, **kwargs: [],
        tiny_detector=lambda text, **kwargs: [],
    )

    result = Pipeline(
        cascade_router=router,
        use_safety_sweep=False,
    ).run("Synthetic clinical note without identifiers.")

    assert set(result.stage_durations_ms) == set(STAGE_NAMES)
    assert all(result.stage_duration_ms(name) > 0.0 for name in STAGE_NAMES)
    assert result.cascade_duration_ms == pytest.approx(1.0)
    assert (
        "reported separately" in result.stage("fast_pii_model").metadata["timing_scope"]
    )
    assert "stage_durations_ms" not in result.audit_record
    assert "cascade_duration_ms" not in result.audit_record
