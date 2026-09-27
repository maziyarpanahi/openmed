"""Exercise the Spark adapter's Series transform with real pandas, no JVM."""

from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from openmed.interop import spark_udf


@pytest.mark.parametrize("values", [[], [None], [None, float("nan"), pd.NA]])
def test_null_only_batches_do_not_initialize_defaults(monkeypatch, values):
    default = Mock(side_effect=AssertionError("unused deidentifier initialized"))
    loader = Mock(side_effect=AssertionError("unused loader initialized"))
    monkeypatch.setattr(spark_udf, "_default_deidentifier", default)
    monkeypatch.setattr(spark_udf, "_cached_model_loader", loader)
    series = pd.Series(
        values, index=range(10, 10 + len(values)), name="synthetic", dtype=object
    )
    result = spark_udf._deidentify_series(series)
    assert result.tolist() == [None] * len(values)
    assert result.index.equals(series.index)
    assert result.name == series.name
    default.assert_not_called()
    loader.assert_not_called()


def test_mixed_batch_initializes_only_once_on_first_text(monkeypatch):
    events = []
    model_loader = object()

    def redact(text, **kwargs):
        events.append(("redact", text, kwargs))
        return SimpleNamespace(deidentified_text="synthetic-redacted")

    default = Mock(side_effect=lambda: events.append(("default",)) or redact)
    loader = Mock(side_effect=lambda: events.append(("loader",)) or model_loader)
    monkeypatch.setattr(spark_udf, "_default_deidentifier", default)
    monkeypatch.setattr(spark_udf, "_cached_model_loader", loader)
    series = pd.Series(
        [None, "synthetic-first", pd.NA, "synthetic-second"],
        index=[9, 9, 4, 1],
        name="synthetic",
        dtype=object,
    )
    before = series.copy(deep=True)
    result = spark_udf._deidentify_series(series, method="mask")
    assert result.tolist() == [None, "synthetic-redacted", None, "synthetic-redacted"]
    assert result.index.equals(series.index)
    pd.testing.assert_series_equal(series, before)
    default.assert_called_once_with()
    loader.assert_called_once_with()
    assert events[0:2] == [("default",), ("loader",)]
    assert events[2][2] == {
        "policy": "hipaa_safe_harbor",
        "method": "mask",
        "loader": model_loader,
    }


def test_custom_deidentifier_does_not_create_loader(monkeypatch):
    loader = Mock(side_effect=AssertionError("unexpected loader"))
    monkeypatch.setattr(spark_udf, "_cached_model_loader", loader)
    custom = Mock(return_value="redacted")
    result = spark_udf._deidentify_series(
        pd.Series([None, "synthetic"]), deidentifier=custom
    )
    assert result.tolist() == [None, "redacted"]
    custom.assert_called_once_with("synthetic", policy="hipaa_safe_harbor")
    loader.assert_not_called()


def test_nonempty_input_still_surfaces_dependency_failure(monkeypatch):
    default = Mock(side_effect=ImportError("synthetic missing dependency"))
    monkeypatch.setattr(spark_udf, "_default_deidentifier", default)
    with pytest.raises(ImportError):
        spark_udf._deidentify_series(pd.Series([None, "synthetic"]))


def test_empty_string_is_processed_not_treated_as_missing(monkeypatch):
    custom = Mock(return_value="")
    result = spark_udf._deidentify_series(pd.Series([""]), deidentifier=custom)
    assert result.tolist() == [""]
    custom.assert_called_once_with("", policy="hipaa_safe_harbor")
