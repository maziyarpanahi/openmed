from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

try:
    import tomllib as _toml
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as _toml  # type: ignore[no-redef]

from openmed.interop import adapter_spec, available_adapters, get_adapter, spark_udf
from openmed.interop.spark_udf import _deidentify_series

ROOT = Path(__file__).resolve().parents[3]


def fake_deidentifier(text: str, **kwargs):
    assert kwargs["policy"] == "hipaa_safe_harbor"
    redacted = text.replace("Jane Roe", "[PERSON]").replace("555-0100", "[PHONE]")
    return SimpleNamespace(deidentified_text=redacted)


def test_deidentify_series_redacts_fixture_series():
    texts = pd.Series(["Patient Jane Roe, call 555-0100", "no PII here"])

    result = _deidentify_series(
        texts,
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentifier,
    )

    assert "Jane Roe" not in result[0]
    assert "555-0100" not in result[0]
    assert "[PERSON]" in result[0]
    assert "[PHONE]" in result[0]
    assert result[1] == "no PII here"


def test_deidentify_series_passes_none_through():
    texts = pd.Series(["Jane Roe", None])

    result = _deidentify_series(
        texts,
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentifier,
    )

    assert pd.isna(result[1])


def test_deidentify_series_raises_clear_error_for_malformed_result():
    def broken_deidentifier(text: str, **kwargs):
        return {"redacted": text}

    texts = pd.Series(["Jane Roe"])

    with pytest.raises(TypeError, match="deidentified_text"):
        _deidentify_series(texts, deidentifier=broken_deidentifier)


def test_deidentify_series_forwards_extra_kwargs_to_deidentifier():
    captured: list[dict] = []

    def capturing_deidentifier(text: str, **kwargs):
        captured.append(kwargs)
        return SimpleNamespace(deidentified_text=text)

    texts = pd.Series(["Jane Roe"])

    _deidentify_series(
        texts,
        policy="hipaa_safe_harbor",
        deidentifier=capturing_deidentifier,
        method="mask",
        confidence_threshold=0.9,
    )

    assert captured == [
        {
            "policy": "hipaa_safe_harbor",
            "method": "mask",
            "confidence_threshold": 0.9,
        }
    ]


def test_deidentify_series_defaults_to_core_deidentify(monkeypatch):
    calls: list[str] = []

    def fake_core_deidentify(text: str, **kwargs):
        calls.append(text)
        return SimpleNamespace(deidentified_text=f"[REDACTED:{text}]")

    monkeypatch.setattr(
        spark_udf, "_default_deidentifier", lambda: fake_core_deidentify
    )
    monkeypatch.setattr(spark_udf, "_cached_model_loader", lambda: object())

    texts = pd.Series(["Jane Roe"])

    result = _deidentify_series(texts, policy="hipaa_safe_harbor")

    assert calls == ["Jane Roe"]
    assert result[0] == "[REDACTED:Jane Roe]"


def test_default_deidentifier_reuses_cached_loader(monkeypatch):
    loader = object()
    captured_loaders: list[object] = []

    def deidentifier(text: str, **kwargs):
        captured_loaders.append(kwargs["loader"])
        return SimpleNamespace(deidentified_text=f"[REDACTED:{text}]")

    monkeypatch.setattr(spark_udf, "_default_deidentifier", lambda: deidentifier)
    monkeypatch.setattr(spark_udf, "_cached_model_loader", lambda: loader)

    texts = pd.Series(["Jane Roe", "John Doe"])

    _deidentify_series(texts, policy="hipaa_safe_harbor")

    assert captured_loaders == [loader, loader]


def test_registry_loads_spark_adapter_lazily():
    adapter = get_adapter("spark")

    assert adapter is spark_udf
    assert "spark" in available_adapters()
    assert adapter_spec("spark").extra == "spark"
    assert hasattr(adapter, "make_deidentify_udf")


def test_spark_extra_installs_all_pandas_udf_runtime_dependencies():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        dependencies = _toml.load(handle)["project"]["optional-dependencies"]["spark"]

    assert any(requirement.startswith("pyspark") for requirement in dependencies)
    assert any(requirement.startswith("pandas") for requirement in dependencies)
    assert any(requirement.startswith("pyarrow") for requirement in dependencies)


def test_make_deidentify_udf_raises_clear_error_without_pyspark(monkeypatch):
    def missing_dependency(name: str):
        raise ImportError(name)

    monkeypatch.setattr(spark_udf, "_import_module", missing_dependency)

    with pytest.raises(ImportError, match=r"openmed\[spark\]"):
        spark_udf.make_deidentify_udf()


class _FakeDataFrame:
    def __init__(self, columns: dict[str, str]) -> None:
        self.columns = dict(columns)

    def __getitem__(self, name: str) -> str:
        return self.columns[name]

    def withColumn(self, name: str, value: str) -> "_FakeDataFrame":
        return _FakeDataFrame({**self.columns, name: value})


def test_deidentify_columns_applies_udf_to_each_named_column(monkeypatch):
    applied: list[str] = []

    def fake_udf(column: str) -> str:
        applied.append(column)
        return f"redacted({column})"

    monkeypatch.setattr(spark_udf, "make_deidentify_udf", lambda **kwargs: fake_udf)

    df = _FakeDataFrame({"note": "raw_note", "comment": "raw_comment", "id": "1"})

    result = spark_udf.deidentify_columns(
        df, ["note", "comment"], policy="hipaa_safe_harbor"
    )

    assert applied == ["raw_note", "raw_comment"]
    assert result.columns["note"] == "redacted(raw_note)"
    assert result.columns["comment"] == "redacted(raw_comment)"
    assert result.columns["id"] == "1"
