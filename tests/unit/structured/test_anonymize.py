"""End-to-end tests for role-directed structured anonymization."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmed.interop.bridges.arx import ArxResult
from openmed.risk.kanon import kanon_report
from openmed.structured import (
    MODEL_DP,
    MODEL_L_DIVERSITY,
    MODEL_T_CLOSENESS,
    TableAnonymizationResult,
    anonymize_table,
    read_table,
    write_table,
)


def _rows() -> list[dict[str, object]]:
    return [
        {
            "name": f"Synthetic Person {index}",
            "age": 30 + index % 4,
            "zip": f"1000{index % 2}",
            "condition": ("A", "B")[index % 2],
            "lab_value": 10.0 + index,
            "note": f"Synthetic narrative {index} containing a patient name.",
        }
        for index in range(8)
    ]


def test_python_fallback_routes_columns_and_emits_risk_report() -> None:
    seen: list[str] = []

    def deidentify_text(value: str) -> str:
        seen.append(value)
        return "[DEIDENTIFIED NOTE]"

    result = anonymize_table(
        _rows(),
        {"age": "age", "zip": "zip"},
        k=2,
        text_columns=["note"],
        text_deidentifier=deidentify_text,
        engine="auto",
        profile_backend="native",
    )

    assert isinstance(result, TableAnonymizationResult)
    assert len(seen) == len(_rows())
    assert all("name" not in row for row in result.records)
    assert {row["note"] for row in result.records} == {"[DEIDENTIFIED NOTE]"}
    assert result.manifest["engine"] == "python"
    assert result.manifest["column_routing"]["direct_identifiers_removed"] == ["name"]
    assert result.risk_report["k_min"] >= 2


def test_path_input_infers_supported_qis_and_writes_transformed_table(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jsonl"
    output = tmp_path / "release.jsonl"
    rows = [
        {"age": 30 + index % 4, "zip": f"1000{index % 2}", "status": "ok"}
        for index in range(8)
    ]
    write_table(source, rows)

    result = anonymize_table(
        source,
        k=2,
        output_path=output,
        profile_backend="native",
    )

    assert result.output_path == output
    assert read_table(output) == [dict(row) for row in result.records]
    assert result.manifest["column_routing"]["structured_quasi_identifiers"] == {
        "age": "age",
        "zip": "zip",
    }
    measured = kanon_report(result.records, quasi_identifiers=["age", "zip"])
    assert measured["k"] >= 2


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [
        (MODEL_L_DIVERSITY, {"l": 2}),
        (MODEL_T_CLOSENESS, {"t": 0.1}),
    ],
)
def test_l_diversity_and_t_closeness_models_use_sensitive_attributes(
    model: str,
    kwargs: dict[str, object],
) -> None:
    result = anonymize_table(
        _rows(),
        {"age": "age"},
        model=model,
        k=2,
        sensitive_attributes=["condition"],
        text_columns=[],
        profile_backend="native",
        **kwargs,
    )

    assert result.manifest["model"] == model
    assert result.risk_report["k_min"] >= 2


def test_dp_model_uses_public_bounds_and_preserves_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openmed.structured.anonymize._laplace_noise",
        lambda scale: scale,
    )

    result = anonymize_table(
        _rows(),
        {"age": "age"},
        model=MODEL_DP,
        k=2,
        text_columns=[],
        dp_bounds={"lab_value": (0.0, 20.0)},
        epsilon=2.0,
        delta=1e-6,
        profile_backend="native",
    )

    assert result.records[0]["lab_value"] == pytest.approx(20.0)
    assert result.manifest["differential_privacy"] == {
        "mechanism": "local_laplace",
        "epsilon": 2.0,
        "delta": 1e-6,
        "guaranteed_delta": 0.0,
        "composition": "sequential_per_record",
        "per_column_epsilon": 2.0,
        "columns": [{"column": "lab_value", "lower": 0.0, "upper": 20.0}],
    }
    assert result.risk_report["k_min"] >= 2


def test_transient_subject_identifier_is_removed_without_date_processing() -> None:
    result = anonymize_table(
        [
            {"patient_id": "synthetic-1", "age": 30},
            {"patient_id": "synthetic-2", "age": 31},
        ],
        {"age": "age"},
        k=2,
        subject_id_column="patient_id",
        text_columns=[],
        profile_backend="native",
    )

    assert all("patient_id" not in row for row in result.records)
    assert result.manifest["column_routing"]["direct_identifiers_removed"] == [
        "patient_id"
    ]


def test_arx_output_is_independently_rejected_when_target_k_is_missed() -> None:
    class BadBridge:
        available = True

        def anonymize(self, records, **kwargs):
            return ArxResult(
                records=tuple(dict(row) for row in records),
                manifest={"engine": "arx"},
            )

    with pytest.raises(ValueError, match="did not reach target k"):
        anonymize_table(
            [{"age": 30}, {"age": 31}],
            {"age": "age"},
            k=2,
            engine="arx",
            arx_bridge=BadBridge(),  # type: ignore[arg-type]
            text_columns=[],
            profile_backend="native",
        )


def test_text_deidentification_error_never_echoes_cell_value() -> None:
    source_canary = "synthetic-private-cell"

    def fail(value: str) -> str:
        raise RuntimeError(value)

    with pytest.raises(ValueError) as raised:
        anonymize_table(
            [
                {"age": 30, "note": source_canary},
                {"age": 31, "note": source_canary},
            ],
            {"age": "age"},
            text_columns=["note"],
            text_deidentifier=fail,
            profile_backend="native",
        )

    assert source_canary not in str(raised.value)
    assert raised.value.__cause__ is None
