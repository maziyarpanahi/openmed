import json
from datetime import datetime

from openmed import ExplainReport, explain
from openmed.core.explain import render
from openmed.core.pipeline import Pipeline
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _prediction_result(text, entities):
    return PredictionResult(
        text=text,
        entities=list(entities),
        model_name="unit-model",
        timestamp=datetime.now().isoformat(),
    )


def test_explain_reports_emitted_span_trace_from_stage_metadata():
    text = "Patient Jane Roe visited"
    identifier = "Jane Roe"

    def model_detector(text, **kwargs):
        start = text.index(identifier)
        return _prediction_result(
            text,
            [
                EntityPrediction(
                    text=identifier,
                    label="PERSON",
                    start=start,
                    end=start + len(identifier),
                    confidence=0.96,
                    metadata={"detector": "model:unit"},
                )
            ],
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask", explain=True)

    report = explain(result)

    assert isinstance(report, ExplainReport)
    assert len(report.entries) == len(result.spans) == 1
    entry = report.entries[0]
    assert "fast_pii_model" in {detection.stage for detection in entry.detections}
    assert "model:unit" in {detection.detector for detection in entry.detections}
    assert entry.arbitration.outcome == "winner"
    assert entry.threshold is not None
    assert entry.threshold.keep_floor == 0.5
    assert entry.threshold.action == "mask"
    assert entry.policy is not None
    assert entry.policy.policy_label == "DIRECT_IDENTIFIER"
    assert entry.policy.rule == "threshold_matrix"
    assert entry.final_action == "mask"


def test_explain_reports_arbitration_loser_with_winner_and_tie_break_rule():
    text = "Patient Alicia visited"
    identifier = "Alicia"

    def model_detector(text, **kwargs):
        start = text.index(identifier)
        end = start + len(identifier)
        return _prediction_result(
            text,
            [
                EntityPrediction(
                    text=identifier,
                    label="PERSON",
                    start=start,
                    end=end,
                    confidence=0.99,
                    metadata={"detector": "model:person"},
                ),
                EntityPrediction(
                    text=identifier,
                    label="FIRST_NAME",
                    start=start,
                    end=end,
                    confidence=0.60,
                    metadata={"detector": "model:first_name"},
                ),
            ],
        )

    result = Pipeline(
        model_detector=model_detector,
        policy="clinical_minimal_redaction",
        use_safety_sweep=False,
    ).run(text, method="mask", explain=True)

    report = result.explain()

    assert len(report.entries) == 1
    assert {
        detection.canonical_label for detection in report.entries[0].detections
    } == {"PERSON", "FIRST_NAME"}
    assert report.entries[0].arbitration.losing_spans
    assert len(report.dropped_spans) == 1
    dropped = report.dropped_spans[0]
    assert dropped.emitted is False
    assert dropped.arbitration.outcome == "loser"
    assert dropped.arbitration.rule == "label_specificity"
    assert dropped.arbitration.winning_span in (
        report.entries[0].normalized_key,
        report.entries[0].key,
    )
    assert dropped.final_action == "dropped"


def test_rendered_explain_outputs_do_not_include_raw_identifiers():
    text = "Patient Jane Roe email jane.roe@example.test"
    raw_values = ["Jane Roe", "jane.roe@example.test"]

    def model_detector(text, **kwargs):
        name_start = text.index(raw_values[0])
        email_start = text.index(raw_values[1])
        return _prediction_result(
            text,
            [
                EntityPrediction(
                    text=raw_values[0],
                    label="PERSON",
                    start=name_start,
                    end=name_start + len(raw_values[0]),
                    confidence=0.91,
                    metadata={"detector": "model:name"},
                ),
                EntityPrediction(
                    text=raw_values[1],
                    label="EMAIL",
                    start=email_start,
                    end=email_start + len(raw_values[1]),
                    confidence=0.97,
                    metadata={"detector": "model:email"},
                ),
            ],
        )

    result = Pipeline(
        model_detector=model_detector,
        use_safety_sweep=False,
    ).run(text, method="mask", explain=True)
    report = result.explain()

    text_output = render(report, fmt="text")
    dict_output = render(report, fmt="dict")
    serialized_dict = json.dumps(dict_output, sort_keys=True)

    for raw_value in [text, *raw_values]:
        assert raw_value not in text_output
        assert raw_value not in serialized_dict
    assert "hmac-sha256:" in text_output
    assert "hmac-sha256:" in serialized_dict


def test_explain_defaults_off_preserves_normal_run_output_and_audit_record():
    text = "Patient Jane Roe visited"
    identifier = "Jane Roe"

    def model_detector(text, **kwargs):
        start = text.index(identifier)
        return _prediction_result(
            text,
            [
                EntityPrediction(
                    text=identifier,
                    label="PERSON",
                    start=start,
                    end=start + len(identifier),
                    confidence=0.96,
                    metadata={"detector": "model:unit"},
                )
            ],
        )

    pipeline = Pipeline(model_detector=model_detector, use_safety_sweep=False)
    normal = pipeline.run(text, method="mask")
    explicit_default = pipeline.run(text, method="mask", explain=False)

    assert normal.redacted_text == explicit_default.redacted_text
    assert [span.to_dict() for span in normal.spans] == [
        span.to_dict() for span in explicit_default.spans
    ]
    assert json.dumps(normal.audit_record, sort_keys=True) == json.dumps(
        explicit_default.audit_record,
        sort_keys=True,
    )
    assert "arbitration_trace" not in normal.stage("span_arbitration").metadata
