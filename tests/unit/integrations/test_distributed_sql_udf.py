from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations.distributed_sql_udf import (
    OPENMED_DEIDENTIFY_DESCRIPTOR,
    DistributedSQLDeidentifyUDF,
    DistributedSQLUDFConfig,
)
from openmed.processing.batch import BatchItemResult, BatchResult


def _synthetic_process_batch(calls: list[dict[str, Any]]):
    def process_batch(texts: list[str], **kwargs: Any) -> BatchResult:
        calls.append({"texts": list(texts), **kwargs})
        replacements = {
            "Jane Roe": "[PERSON]",
            "john.roe@example.test": "[EMAIL]",
            "555-0101": "[PHONE]",
        }
        items = []
        for index, text in enumerate(texts):
            redacted = text
            for identifier, placeholder in replacements.items():
                redacted = redacted.replace(identifier, placeholder)
            items.append(
                BatchItemResult(
                    id=f"item_{index}",
                    result=SimpleNamespace(deidentified_text=redacted),
                )
            )
        return BatchResult(items=items)

    return process_batch


def test_udf_callable_redacts_synthetic_vector_in_row_windows() -> None:
    calls: list[dict[str, Any]] = []
    loader = object()
    loader_count = 0

    def loader_factory() -> object:
        nonlocal loader_count
        loader_count += 1
        return loader

    udf = DistributedSQLDeidentifyUDF(
        config=DistributedSQLUDFConfig(batch_size=2),
        process_batch_fn=_synthetic_process_batch(calls),
        loader_factory=loader_factory,
    )
    rows = [
        "Jane Roe has hypertension and takes metformin.",
        "Email john.roe@example.test about the diabetes follow-up.",
        "Call 555-0101 after the cardiology appointment.",
    ]

    assert loader_count == 0
    output = udf(rows, "hipaa_safe_harbor")

    assert output == [
        "[PERSON] has hypertension and takes metformin.",
        "Email [EMAIL] about the diabetes follow-up.",
        "Call [PHONE] after the cardiology appointment.",
    ]
    assert [len(call["texts"]) for call in calls] == [2, 1]
    assert all(call["operation"] == "deidentify" for call in calls)
    assert all(call["policy"] == "hipaa_safe_harbor" for call in calls)
    assert all(call["loader"] is loader for call in calls)
    assert loader_count == 1


def test_null_and_empty_inputs_bypass_loader_and_process_batch() -> None:
    calls: list[dict[str, Any]] = []

    def unexpected_loader() -> object:
        raise AssertionError("loader must remain lazy for null and empty rows")

    udf = DistributedSQLDeidentifyUDF(
        process_batch_fn=_synthetic_process_batch(calls),
        loader_factory=unexpected_loader,
    )

    assert udf([None, ""], ["not_a_profile", "also_invalid"]) == [None, ""]
    assert udf.deidentify(None, "not_a_profile") is None
    assert udf.deidentify("", "not_a_profile") == ""
    assert calls == []


def test_loader_is_reused_across_worker_calls_and_profiles_are_grouped() -> None:
    calls: list[dict[str, Any]] = []
    loaders: list[object] = []

    def loader_factory() -> object:
        loader = object()
        loaders.append(loader)
        return loader

    udf = DistributedSQLDeidentifyUDF(
        process_batch_fn=_synthetic_process_batch(calls),
        loader_factory=loader_factory,
    )

    first = udf(
        ["Jane Roe has asthma.", "Call 555-0101 for hypertension."],
        ["hipaa", "gdpr"],
    )
    second = udf(["Jane Roe has diabetes."], "safe_harbor")

    assert first == ["[PERSON] has asthma.", "Call [PHONE] for hypertension."]
    assert second == ["[PERSON] has diabetes."]
    assert len(loaders) == 1
    assert all(call["loader"] is loaders[0] for call in calls)
    assert [call["policy"] for call in calls] == [
        "hipaa_safe_harbor",
        "gdpr_pseudonymization",
        "hipaa_safe_harbor",
    ]


def test_profiles_must_align_with_text_rows() -> None:
    udf = DistributedSQLDeidentifyUDF(
        process_batch_fn=_synthetic_process_batch([]),
        loader_factory=object,
    )

    with pytest.raises(ValueError, match="1 values for 2 text rows"):
        udf(["one", "two"], ["hipaa_safe_harbor"])


def test_registration_descriptor_maps_scalar_sql_to_vector_entrypoint() -> None:
    assert OPENMED_DEIDENTIFY_DESCRIPTOR == {
        "name": "openmed_deidentify",
        "language": "python",
        "entrypoint": ("openmed.integrations.distributed_sql_udf:deidentify_batch"),
        "arguments": [
            {"name": "text", "sql_type": "VARCHAR", "python_batch": "texts"},
            {
                "name": "profile",
                "sql_type": "VARCHAR",
                "python_batch": "profiles",
            },
        ],
        "return_type": "VARCHAR",
        "vectorized": True,
        "null_handling": "called_on_null_input",
        "default_batch_size": 64,
    }
