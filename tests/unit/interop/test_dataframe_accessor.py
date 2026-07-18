from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from openmed.interop import adapter_spec, available_adapters, get_adapter


def fake_deidentify(text: str, **kwargs):
    assert kwargs["method"] == "mask"
    assert kwargs["policy"] == "hipaa_safe_harbor"
    return SimpleNamespace(
        deidentified_text=text.replace("Jane Roe", "[PERSON]").replace(
            "555-0100",
            "[PHONE]",
        )
    )


def test_registry_lists_dataframe_adapters_lazily():
    assert "pandas" in available_adapters()
    assert "polars" in available_adapters()
    assert adapter_spec("pandas").extra == "pandas"
    assert adapter_spec("polars").extra == "polars"


def test_importing_openmed_does_not_import_pandas_or_polars():
    code = """
import sys

for name in list(sys.modules):
    if name == "pandas" or name.startswith("pandas."):
        sys.modules.pop(name, None)
    if name == "polars" or name.startswith("polars."):
        sys.modules.pop(name, None)

import openmed  # noqa: F401
from openmed.interop import available_adapters

assert "pandas" in available_adapters()
assert "polars" in available_adapters()
if any(name == "pandas" or name.startswith("pandas.") for name in sys.modules):
    raise SystemExit("pandas was imported")
if any(name == "polars" or name.startswith("polars.") for name in sys.modules):
    raise SystemExit("polars was imported")
"""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=False,
        cwd=".",
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_pandas_accessor_deidentifies_selected_text_columns_only():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "note": ["Patient Jane Roe called 555-0100.", "No identifiers."],
            "age": [42, 73],
        }
    )

    redacted = frame.openmed.deidentify(
        columns=["note"],
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentify,
    )

    assert redacted is not frame
    assert redacted["note"].tolist() == [
        "Patient [PERSON] called [PHONE].",
        "No identifiers.",
    ]
    assert redacted["age"].tolist() == [42, 73]
    assert frame["note"].tolist()[0] == "Patient Jane Roe called 555-0100."


def test_pandas_accessor_risk_report_returns_om009_shape():
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    get_adapter("pandas")
    frame = pd.DataFrame(
        [
            {"record_id": "a", "age": 73, "city": "Riverton"},
            {"record_id": "b", "age": 73, "city": "Riverton"},
            {"record_id": "unique", "age": 94, "city": "Smallville"},
        ]
    )

    report = frame.openmed.risk_report(qi_columns=["age", "city"])

    assert set(report) == {
        "leakage_rate",
        "reid_rate",
        "k_min",
        "singleton_records",
        "quasi_identifiers",
    }
    assert report["k_min"] == 1


def test_polars_helper_deidentifies_selected_text_columns_only():
    pl = pytest.importorskip("polars", exc_type=ImportError)
    polars_adapter = get_adapter("polars")
    frame = pl.DataFrame(
        {
            "record_id": ["a", "b"],
            "note": ["Patient Jane Roe called 555-0100.", "No identifiers."],
            "age": [42, 73],
        }
    )

    redacted = polars_adapter.deidentify_frame(
        frame,
        columns=["note"],
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentify,
    )

    assert redacted is not frame
    assert redacted["note"].to_list() == [
        "Patient [PERSON] called [PHONE].",
        "No identifiers.",
    ]
    assert redacted["age"].to_list() == [42, 73]
    assert frame["note"].to_list()[0] == "Patient Jane Roe called 555-0100."


def test_polars_risk_report_returns_om009_shape():
    pl = pytest.importorskip("polars", exc_type=ImportError)
    polars_adapter = get_adapter("polars")
    frame = pl.DataFrame(
        [
            {"record_id": "a", "age": 73, "city": "Riverton"},
            {"record_id": "b", "age": 73, "city": "Riverton"},
            {"record_id": "unique", "age": 94, "city": "Smallville"},
        ]
    )

    report = polars_adapter.risk_report(frame, qi_columns=["age", "city"])

    assert set(report) == {
        "leakage_rate",
        "reid_rate",
        "k_min",
        "singleton_records",
        "quasi_identifiers",
    }
    assert report["k_min"] == 1


def test_polars_namespace_is_registered_when_supported():
    pl = pytest.importorskip("polars", exc_type=ImportError)
    get_adapter("polars")
    frame = pl.DataFrame({"note": ["Patient Jane Roe called 555-0100."]})

    if not hasattr(frame, "openmed"):
        pytest.skip("installed polars does not support DataFrame namespaces")

    redacted = frame.openmed.deidentify(
        columns="note",
        policy="hipaa_safe_harbor",
        deidentifier=fake_deidentify,
    )

    assert redacted["note"].to_list() == ["Patient [PERSON] called [PHONE]."]
