import logging
from datetime import datetime

import pytest

from openmed.core import detector_plugins
from openmed.core.detector_plugins import (
    DetectorSpec,
    iter_detectors,
    register_detector,
)
from openmed.core.pipeline import Pipeline
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import PredictionResult


@pytest.fixture(autouse=True)
def reset_detector_plugins():
    detector_plugins._reset_detector_registry_for_tests()
    yield
    detector_plugins._reset_detector_registry_for_tests()


def _empty_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def _plugin_span(
    text: str,
    surface: str,
    *,
    canonical_label: str = "PERSON",
) -> OpenMedSpan:
    start = text.index(surface)
    return OpenMedSpan(
        doc_id="plugin-placeholder",
        start=start,
        end=start + len(surface),
        text_hash=hmac_text_hash(surface, "plugin-placeholder"),
        entity_type=canonical_label,
        canonical_label=canonical_label,
        score=0.99,
        detector="local",
        evidence={"pattern": "synthetic", "surface": surface},
        metadata={
            "safe": "ok",
            "surface": surface,
            "normalized_text": surface,
            "nested": {"text": surface, "rule": "custom"},
        },
    )


def test_register_iter_and_pipeline_route_plugin_span_through_arbitration():
    def detect(text: str, **kwargs):
        return (_plugin_span(text, "Alpha Beta"),)

    register_detector(
        DetectorSpec(
            name="custom_person",
            stage="fast_pii",
            languages=("en",),
            detect=detect,
        )
    )

    assert [spec.name for spec in iter_detectors("fast_pii", "en")] == ["custom_person"]
    assert iter_detectors("fast_pii", "es") == ()

    text = "Patient   Alpha Beta"
    result = Pipeline(
        model_detector=lambda text, **kwargs: _empty_prediction(
            text,
            kwargs["model_name"],
        ),
        use_safety_sweep=False,
    ).run(text, method="mask")

    normalized_start = result.normalized_text.index("Alpha Beta")
    plugin_stage_span = result.stage("fast_pii_model").spans[0]
    arbitration_span = result.stage("span_arbitration").spans[0]
    emitted_span = result.spans[0]

    assert plugin_stage_span.detector == "plugin:custom_person"
    assert arbitration_span.detector == "plugin:custom_person"
    assert plugin_stage_span.start == normalized_start
    assert emitted_span.start == text.index("Alpha Beta")
    assert emitted_span.metadata["normalized_start"] == normalized_start
    assert emitted_span.metadata["normalized_text_hash"].startswith("hmac-sha256:")
    assert emitted_span.metadata["safe"] == "ok"
    assert "surface" not in plugin_stage_span.metadata
    assert "normalized_text" not in plugin_stage_span.metadata
    assert "surface" not in plugin_stage_span.evidence
    assert result.redacted_text == "Patient   [PERSON]"
    assert "Alpha Beta" not in str(result.audit_record)


def test_fake_entry_point_is_discovered_once_without_explicit_registration(
    monkeypatch,
):
    calls = 0

    class FakeEntryPoint:
        name = "fake_clinical"

        def load(self):
            return lambda: DetectorSpec(
                name="fake_clinical",
                stage="clinical_phi",
                languages=("*",),
                detect=lambda text, **kwargs: (),
            )

    def fake_entry_points(*, group=None):
        nonlocal calls
        calls += 1
        assert group == detector_plugins.DETECTOR_ENTRY_POINT_GROUP
        return (FakeEntryPoint(),)

    monkeypatch.setattr(
        detector_plugins.importlib_metadata,
        "entry_points",
        fake_entry_points,
    )

    assert [spec.name for spec in iter_detectors("clinical_phi", "en")] == [
        "fake_clinical"
    ]
    assert [spec.name for spec in iter_detectors("clinical_phi", "fr")] == [
        "fake_clinical"
    ]
    assert calls == 1


def test_broken_entry_point_logs_warning_and_pipeline_continues(
    monkeypatch,
    caplog,
):
    class BrokenEntryPoint:
        name = "broken_detector"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        detector_plugins.importlib_metadata,
        "entry_points",
        lambda *, group=None: (BrokenEntryPoint(),),
    )
    caplog.set_level(logging.WARNING, logger="openmed.core.detector_plugins")

    result = Pipeline(
        model_detector=lambda text, **kwargs: _empty_prediction(
            text,
            kwargs["model_name"],
        ),
        use_safety_sweep=False,
    ).run("No identifiers here")

    assert result.spans == ()
    assert "broken_detector" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "No identifiers here" not in caplog.text
