import builtins
import importlib
import unicodedata
from datetime import datetime

import pytest

from openmed.core.pii import MissingOptionalDependencyError
from openmed.core.pipeline import Pipeline, _iter_normalization_segments
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _empty_prediction(text: str, model_name: str = "stub") -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[],
        model_name=model_name,
        timestamp=datetime.now().isoformat(),
    )


def test_normalized_offsets_remap_combining_characters_to_original_positions():
    text = "Cafe\u0301  MRN"
    document = Pipeline().stage1_normalize(text)

    assert document.normalized_text == "Café MRN"

    original_e = text.index("e")
    original_combining = original_e + 1
    normalized_e = document.normalized_text.index("é")

    assert document.offset_map.original_index_to_normalized(original_e) == normalized_e
    assert (
        document.offset_map.original_index_to_normalized(original_combining)
        == normalized_e
    )
    assert document.offset_map.normalized_index_to_original(normalized_e) == original_e
    assert document.offset_map.normalized_span_to_original_offsets(
        normalized_e,
        normalized_e + 1,
    ) == (original_e, original_combining + 1)


def test_stage1_records_skipped_encoding_repair_when_ftfy_missing(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ftfy" or name.startswith("ftfy."):
            raise ImportError("ftfy unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    document = Pipeline().stage1_normalize("Patient Caf\u00e9")
    metadata = document.metadata["encoding_repair"]

    assert metadata["available"] is False
    assert metadata["skipped"] is True
    assert metadata["dependency"] == "ftfy"
    assert "missing optional dependency" in metadata["reason"]
    assert "pip install ftfy" in metadata["install"]


def test_stage1_identity_repair_shortcut_preserves_normalized_output():
    # When encoding repair is unavailable the per-segment repair call is skipped
    # as a latency optimisation; the normalized text and offset map must be
    # identical to a plain NFC normalisation of each segment (OM-364).
    text = "Café  Zoë   Straße\tnaı̈ve"
    document = Pipeline().stage1_normalize(text)

    expected = []
    for start, end, segment, is_whitespace in _iter_normalization_segments(text):
        expected.append(" " if is_whitespace else unicodedata.normalize("NFC", segment))
    expected_text = "".join(part for part in expected if part)

    assert document.normalized_text == expected_text
    assert len(document.offset_map.normalized_to_original) == len(
        document.normalized_text
    )


def test_stage1_applies_active_encoding_repair_per_segment(monkeypatch):
    # When a real repairer is active it must still run on each non-whitespace
    # segment (the optimisation only skips the identity no-op case).
    calls = []

    def fake_repairer():
        def repair(segment):
            calls.append(segment)
            return segment.replace("�", "?")

        return repair, {"feature": "encoding repair", "available": True}

    monkeypatch.setattr(
        "openmed.core.pipeline._encoding_repairer",
        fake_repairer,
    )

    document = Pipeline().stage1_normalize("A�B")

    assert document.normalized_text == "A?B"
    # Repairer invoked once per non-whitespace segment (three characters here).
    assert calls == ["A", "�", "B"]


def test_run_exposes_per_stage_latency_measurements():
    result = Pipeline(
        model_detector=lambda text, **kwargs: _empty_prediction(
            text, model_name=kwargs["model_name"]
        )
    ).run("Patient note without identifiers.", method="mask")

    for stage_name in Pipeline.stage_names:
        assert result.stage_duration_ms(stage_name) >= 0.0

    with pytest.raises(KeyError):
        result.stage_duration_ms("nonexistent_stage")

    # Durations are latency-only floats keyed by every stage name, and are kept
    # off the reproducible audit record because wall-clock time is
    # non-deterministic.
    assert set(result.stage_durations_ms) == set(Pipeline.stage_names)
    assert all(isinstance(value, float) for value in result.stage_durations_ms.values())
    assert "stage_durations_ms" not in result.audit_record


def test_stage3_records_unavailable_section_hook_metadata(monkeypatch):
    original_import_module = importlib.import_module

    def fake_import_module(name, package=None):
        if name == "openmed.clinical.sections":
            raise ImportError("sections unavailable")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    metadata = Pipeline().stage3_doc_type_section("Assessment: stable")
    section_detection = metadata["section_detection"]

    assert metadata["section_hook"] == "unavailable"
    assert metadata["sections"] == ()
    assert section_detection["available"] is False
    assert section_detection["skipped"] is True
    assert section_detection["dependency"] == "openmed.clinical.sections"
    assert "missing optional capability" in section_detection["reason"]


def test_normalized_span_round_trips_nfc_length_change_to_original_surface():
    text = "Patient Jose\u0301 Garcia visited"
    document = Pipeline().stage1_normalize(text)
    normalized_surface = "José Garcia"
    normalized_start = document.normalized_text.index(normalized_surface)
    normalized_end = normalized_start + len(normalized_surface)

    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(
            normalized_start,
            normalized_end,
        )
    )
    original_surface = text[original_start:original_end]

    assert original_surface == "Jose\u0301 Garcia"
    assert unicodedata.normalize("NFC", original_surface) == normalized_surface
    assert document.offset_map.original_span_to_normalized(
        original_start,
        original_end,
    ) == (normalized_start, normalized_end)


def test_normalized_span_round_trips_collapsed_whitespace_to_original_surface():
    text = "Patient John   Doe visited"
    document = Pipeline().stage1_normalize(text)
    normalized_surface = "John Doe"
    normalized_start = document.normalized_text.index(normalized_surface)
    normalized_end = normalized_start + len(normalized_surface)

    original_start, original_end = (
        document.offset_map.normalized_span_to_original_offsets(
            normalized_start,
            normalized_end,
        )
    )
    original_surface = text[original_start:original_end]

    assert original_surface == "John   Doe"
    assert Pipeline().stage1_normalize(original_surface).normalized_text == (
        normalized_surface
    )
    assert document.offset_map.original_span_to_normalized(
        original_start,
        original_end,
    ) == (normalized_start, normalized_end)


def test_deidentification_redacts_original_nfc_changed_entity_surface():
    text = "Patient Jose\u0301 Garcia visited"

    def model_detector(text, **kwargs):
        entity_text = "José Garcia"
        start = text.index(entity_text)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=entity_text,
                    label="NAME",
                    start=start,
                    end=start + len(entity_text),
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")
    entity = result.deidentification_result.pii_entities[0]
    span = result.spans[0]

    assert result.deidentification_result.original_text == text
    assert entity.original_text == "Jose\u0301 Garcia"
    assert entity.text == "Jose\u0301 Garcia"
    assert (entity.start, entity.end) == (
        text.index("Jose\u0301 Garcia"),
        text.index("Jose\u0301 Garcia") + len("Jose\u0301 Garcia"),
    )
    assert (span.start, span.end) == (entity.start, entity.end)
    assert "normalized_text" not in (entity.metadata or {})
    assert "normalized_text" not in span.metadata
    assert (entity.metadata or {})["normalized_text_hash"].startswith("hmac-sha256:")
    assert span.metadata["normalized_text_hash"].startswith("hmac-sha256:")
    assert result.redacted_text == "Patient [NAME] visited"


def test_deidentification_redacts_original_collapsed_whitespace_entity_surface():
    text = "Patient John   Doe visited"

    def model_detector(text, **kwargs):
        entity_text = "John Doe"
        start = text.index(entity_text)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=entity_text,
                    label="NAME",
                    start=start,
                    end=start + len(entity_text),
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask")
    entity = result.deidentification_result.pii_entities[0]
    span = result.spans[0]

    assert result.deidentification_result.original_text == text
    assert entity.original_text == "John   Doe"
    assert entity.text == "John   Doe"
    assert (entity.start, entity.end) == (
        text.index("John   Doe"),
        text.index("John   Doe") + len("John   Doe"),
    )
    assert (span.start, span.end) == (entity.start, entity.end)
    assert "normalized_text" not in (entity.metadata or {})
    assert "normalized_text" not in span.metadata
    assert (entity.metadata or {})["normalized_text_hash"].startswith("hmac-sha256:")
    assert span.metadata["normalized_text_hash"].startswith("hmac-sha256:")
    assert result.redacted_text == "Patient [NAME] visited"


def test_shift_dates_missing_dateutil_raises_clear_optional_extra_error(monkeypatch):
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "dateutil" or name.startswith("dateutil."):
            raise ImportError("dateutil unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    text = "DOB 01/15/2020"

    def model_detector(text, **kwargs):
        entity_text = "01/15/2020"
        start = text.index(entity_text)
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text=entity_text,
                    label="DATE",
                    start=start,
                    end=start + len(entity_text),
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    with pytest.raises(MissingOptionalDependencyError) as excinfo:
        Pipeline(
            model_detector=model_detector,
            use_safety_sweep=False,
        ).run(text, method="shift_dates", date_shift_days=30)

    message = str(excinfo.value)
    assert "method='shift_dates'" in message
    assert "python-dateutil" in message
    assert "pip install openmed[dev]" in message


def test_luhn_valid_mrn_is_detected_before_model_stage_runs():
    model_calls = []

    def model_detector(text, **kwargs):
        model_calls.append(("model", text, kwargs))
        return _empty_prediction(text, model_name=kwargs["model_name"])

    text = "MRN: 4111111111111111"
    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text)

    deterministic_stage = result.stage("deterministic_detectors")
    model_stage = result.stage("fast_pii_model")

    assert len(model_calls) == 1
    assert model_calls[0][1] == text
    assert model_stage.spans == ()
    assert deterministic_stage.spans
    assert deterministic_stage.stage < model_stage.stage
    assert deterministic_stage.spans[0].detector.startswith("rules:")
    assert deterministic_stage.spans[0].detector == "rules:mrn_luhn"


def test_stage9_safety_sweep_only_increases_redacted_character_count():
    text = "Patient John Doe email jane.patient@example.com"

    def model_detector(text, **kwargs):
        return PredictionResult(
            text=text,
            entities=[
                EntityPrediction(
                    text="John Doe",
                    label="NAME",
                    start=8,
                    end=16,
                    confidence=0.95,
                )
            ],
            model_name=kwargs["model_name"],
            timestamp=datetime.now().isoformat(),
        )

    result = Pipeline(model_detector=model_detector).run(text, method="mask")
    stage9 = result.stage("safety_sweep")

    assert (
        stage9.metadata["redacted_chars_after"]
        >= stage9.metadata["redacted_chars_before"]
    )
    assert stage9.metadata["spans_added"] == 1
    assert result.redacted_text == "Patient [NAME] email [email]"
