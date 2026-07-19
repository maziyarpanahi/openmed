"""Offline tests for the Apache Beam/Dataflow batch processor."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import apache_beam as beam
import pytest
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline
from apache_beam.testing.util import assert_that, equal_to

from openmed.integrations.dataflow_processor import (
    DataflowBatchProcessor,
    DataflowDeadLetter,
)


def _redact(text: str) -> str:
    return text.replace("Jane Roe", "[NAME]").replace("555-0101", "[PHONE]")


class _ResidentProcessor:
    batch_sizes: list[int] = []

    def process_texts(self, texts: list[str]) -> Any:
        type(self).batch_sizes.append(len(texts))
        return SimpleNamespace(
            items=[
                SimpleNamespace(
                    success=True,
                    result=SimpleNamespace(deidentified_text=_redact(text)),
                )
                for text in texts
            ]
        )


class _ProcessorFactory:
    setup_calls = 0
    options: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> _ResidentProcessor:
        type(self).setup_calls += 1
        type(self).options.append(dict(kwargs))
        return _ResidentProcessor()


@pytest.fixture(autouse=True)
def _reset_probes() -> None:
    _ProcessorFactory.setup_calls = 0
    _ProcessorFactory.options = []
    _ResidentProcessor.batch_sizes = []


def _direct_runner_options() -> PipelineOptions:
    return PipelineOptions(
        [
            "--runner=DirectRunner",
            "--direct_num_workers=1",
            "--direct_running_mode=in_memory",
        ]
    )


def test_local_runner_redacts_in_batches_with_one_bundle_scoped_setup(
    caplog: pytest.LogCaptureFixture,
) -> None:
    records = [
        {
            "record_id": "synthetic-1",
            "note": "Patient Jane Roe called 555-0101.",
            "facility": "example-clinic",
        },
        {
            "record_id": "synthetic-2",
            "note": "Jane Roe requested a follow-up.",
            "facility": "example-clinic",
        },
        {
            "record_id": "synthetic-3",
            "note": "No seeded identifiers.",
            "facility": "example-clinic",
        },
    ]

    with BeamTestPipeline(options=_direct_runner_options()) as pipeline:
        outputs = (
            pipeline
            | "Create synthetic records" >> beam.Create(records)
            | "Apply OpenMed processor"
            >> DataflowBatchProcessor(
                "note",
                policy="hipaa_safe_harbor",
                batch_size=2,
                processor_factory=_ProcessorFactory(),
                use_safety_sweep=True,
            )
        )
        assert_that(
            outputs.cleaned,
            equal_to(
                [
                    {
                        "record_id": "synthetic-1",
                        "note": "Patient [NAME] called [PHONE].",
                        "facility": "example-clinic",
                    },
                    {
                        "record_id": "synthetic-2",
                        "note": "[NAME] requested a follow-up.",
                        "facility": "example-clinic",
                    },
                    {
                        "record_id": "synthetic-3",
                        "note": "No seeded identifiers.",
                        "facility": "example-clinic",
                    },
                ]
            ),
            label="Assert cleaned records",
        )
        assert_that(
            outputs.dead_letter,
            equal_to([]),
            label="Assert no dead letters",
        )

    assert _ProcessorFactory.setup_calls == 1
    assert sorted(_ResidentProcessor.batch_sizes) == [1, 2]
    assert _ProcessorFactory.options == [
        {
            "model_name": "disease_detection_superclinical",
            "operation": "deidentify",
            "batch_size": 2,
            "continue_on_error": True,
            "method": "mask",
            "use_safety_sweep": True,
            "policy": "hipaa_safe_harbor",
        }
    ]
    assert "Jane Roe" not in caplog.text
    assert "555-0101" not in caplog.text


def test_none_and_unprocessable_elements_route_to_dead_letter() -> None:
    records = [
        None,
        {"record_id": "missing-note"},
        {"record_id": "null-note", "note": None},
        {"record_id": "numeric-note", "note": 42},
        {"record_id": "valid", "note": "Patient Jane Roe"},
    ]

    with BeamTestPipeline(options=_direct_runner_options()) as pipeline:
        outputs = (
            pipeline
            | "Create mixed records" >> beam.Create(records)
            | "Apply processor with dead letters"
            >> DataflowBatchProcessor(
                "note",
                batch_size=8,
                processor_factory=_ProcessorFactory(),
            )
        )
        assert_that(
            outputs.cleaned,
            equal_to([{"record_id": "valid", "note": "Patient [NAME]"}]),
            label="Assert one cleaned record",
        )
        assert_that(
            outputs.dead_letter,
            equal_to(
                [
                    DataflowDeadLetter(None, "invalid_element"),
                    DataflowDeadLetter(
                        {"record_id": "missing-note"}, "missing_text_field"
                    ),
                    DataflowDeadLetter(
                        {"record_id": "null-note", "note": None}, "null_text"
                    ),
                    DataflowDeadLetter(
                        {"record_id": "numeric-note", "note": 42},
                        "invalid_text_type",
                    ),
                ]
            ),
            label="Assert dead-letter records",
        )


def test_batch_failure_is_phi_safe_and_does_not_fail_bundle(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_text = "Patient Jane Roe called 555-0101."

    def fail_batch(texts: list[str], **kwargs: Any) -> None:
        raise RuntimeError(f"failed to process {texts[0]}")

    with BeamTestPipeline(options=_direct_runner_options()) as pipeline:
        outputs = (
            pipeline
            | "Create failing record"
            >> beam.Create([{"record_id": "synthetic-4", "note": raw_text}])
            | "Apply failing processor"
            >> DataflowBatchProcessor("note", process_batch_fn=fail_batch)
        )
        assert_that(
            outputs.cleaned,
            equal_to([]),
            label="Assert failed record is not cleaned",
        )
        assert_that(
            outputs.dead_letter,
            equal_to(
                [
                    DataflowDeadLetter(
                        {"record_id": "synthetic-4", "note": raw_text},
                        "processing_error",
                    )
                ]
            ),
            label="Assert failure dead letter",
        )

    assert raw_text not in caplog.text
    assert "Jane Roe" not in caplog.text
    assert "555-0101" not in caplog.text


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"text_field": ""}, ValueError, "text_field must be a non-empty string"),
        ({"batch_size": 0}, ValueError, "batch_size must be a positive integer"),
        ({"processor_factory": object()}, TypeError, "must be callable"),
    ],
)
def test_constructor_rejects_invalid_options(
    kwargs: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        DataflowBatchProcessor(**kwargs)
