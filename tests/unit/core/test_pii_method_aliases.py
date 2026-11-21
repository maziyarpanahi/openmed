"""Date shifting warns at the alias call site and keeps its results."""

import warnings
from types import SimpleNamespace

import openmed
from openmed.processing.outputs import EntityPrediction, PredictionResult
from openmed.utils.deprecation import SHIFT_DATES_NOTICE


def test_python_alias_warns_once_per_callsite(monkeypatch):
    text = "DOB 01/15/2020"
    result = PredictionResult(
        text=text,
        entities=[EntityPrediction("01/15/2020", "DATE", 1.0, 4, 14)],
        model_name="test",
        timestamp="2026-01-01",
    )
    monkeypatch.setattr("openmed.core.pii.extract_pii", lambda *a, **kw: result)
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("default")
        for _ in range(2):
            shifted = openmed.deidentify(text, shift_dates=True, date_shift_days=30)
        canonical = openmed.deidentify(text, method="shift_dates", date_shift_days=30)
    assert shifted.deidentified_text == canonical.deidentified_text == "DOB 02/14/2020"
    matching = [w for w in notices if 'Use method="shift_dates"' in str(w.message)]
    assert len(matching) == 1 and matching[0].filename == __file__


def test_mcp_notice_only_for_explicit_boolean(monkeypatch):
    from openmed.mcp.server import openmed_deidentify

    runtime = SimpleNamespace(
        config=None,
        get_loader=lambda: None,
        run_model_request=lambda model, keep_alive, fn: fn(),
    )
    monkeypatch.setattr(
        openmed,
        "deidentify",
        lambda text, **kw: {
            "original_text": text,
            "deidentified_text": "sample",
            "pii_entities": [],
            "method": kw["method"],
            "timestamp": "2026-01-01",
            "num_entities_redacted": 0,
            "metadata": {"kept": True},
            "audit_report": None,
        },
    )
    for flag in (True, False):
        result = openmed_deidentify(
            "sample", shift_dates=flag, runtime_provider=lambda: runtime
        )
        assert result["metadata"] == {
            "kept": True,
            "deprecation_notice": SHIFT_DATES_NOTICE,
        }
    assert openmed_deidentify(
        "sample", method="shift_dates", runtime_provider=lambda: runtime
    )["metadata"] == {"kept": True}
