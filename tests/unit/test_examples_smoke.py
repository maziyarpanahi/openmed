"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import clinical_ner_families, datasets_walkthrough, gradio_deid_app


def test_clinical_ner_families_example_is_syntactically_valid():
    source = Path("examples/clinical_ner_families.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_clinical_ner_families_selects_three_registry_families():
    assert {"Disease", "Pharmaceutical", "Oncology"}.issubset(
        clinical_ner_families.NER_FAMILIES
    )

    selections = clinical_ner_families.selected_families()

    assert {selection.family for selection in selections} == {
        "Disease",
        "Pharmaceutical",
        "Oncology",
    }
    assert all(selection.source == "accurate tier" for selection in selections)
    assert all(
        selection.model.model_id.startswith("OpenMed/") for selection in selections
    )


def test_clinical_ner_families_uses_mocked_analyzer_without_network(
    capsys,
    monkeypatch,
):
    monkeypatch.delenv(clinical_ner_families.ALLOW_DOWNLOAD_ENV, raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    calls = []

    def fake_analyzer(text, *, model_name, confidence_threshold, group_entities):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        calls.append(
            {
                "text": text,
                "model_name": model_name,
                "confidence_threshold": confidence_threshold,
                "group_entities": group_entities,
            }
        )
        return SimpleNamespace(
            entities=[
                SimpleNamespace(
                    label="MOCK_ENTITY",
                    text="synthetic span",
                    confidence=0.99,
                )
            ]
        )

    clinical_ner_families.run_family_extraction(analyzer=fake_analyzer)

    assert len(calls) == 3
    assert all(call["group_entities"] is True for call in calls)
    assert all(call["model_name"].startswith("OpenMed/") for call in calls)
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    captured = capsys.readouterr().out
    assert "Disease" in captured
    assert "Pharmaceutical" in captured
    assert "Oncology" in captured
    assert "MOCK_ENTITY" in captured


def test_clinical_ner_families_reports_offline_unavailable(capsys):
    def unavailable_analyzer(*args, **kwargs):
        raise OSError("cached files not found")

    clinical_ner_families.run_family_extraction(
        families=("Disease",),
        analyzer=unavailable_analyzer,
    )

    captured = capsys.readouterr().out
    assert "Disease: model unavailable offline" in captured
    assert "cached files not found" in captured


def test_gradio_deid_app_is_syntactically_valid():
    source = Path("examples/gradio_deid_app.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_gradio_deid_app_exposes_public_surface():
    assert gradio_deid_app.DEIDENTIFICATION_METHODS == ("mask", "replace", "hash")
    assert callable(gradio_deid_app.build_demo)
    assert callable(gradio_deid_app.run_deidentification)
    assert "Synthetic note" in gradio_deid_app.SYNTHETIC_CLINICAL_TEXT


def test_gradio_deid_app_builds_entity_rows():
    entities = [
        SimpleNamespace(label="NAME", text="John Doe", start=0, end=8, confidence=0.97),
        SimpleNamespace(
            label="EMAIL", text="j@x.org", start=20, end=27, confidence=0.5
        ),
    ]

    rows = gradio_deid_app.entities_to_rows(entities)

    assert rows == [
        ["NAME", "John Doe", "0", "8", "0.97"],
        ["EMAIL", "j@x.org", "20", "27", "0.50"],
    ]


def test_gradio_deid_app_run_uses_deidentify_without_network(monkeypatch):
    calls = []

    def fake_deidentify(text, *, method):
        calls.append((text, method))
        return SimpleNamespace(
            deidentified_text="[NAME] was seen.",
            pii_entities=[
                SimpleNamespace(
                    label="NAME", text="John Doe", start=0, end=8, confidence=0.9
                )
            ],
        )

    monkeypatch.setattr(gradio_deid_app, "deidentify", fake_deidentify)

    view = gradio_deid_app.run_deidentification("John Doe was seen.", "mask")

    assert calls == [("John Doe was seen.", "mask")]
    assert view.deidentified_text == "[NAME] was seen."
    assert view.entity_rows == [["NAME", "John Doe", "0", "8", "0.90"]]


def test_gradio_deid_app_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unsupported method"):
        gradio_deid_app.run_deidentification("text", "encrypt")


def test_gradio_deid_app_missing_gradio_prints_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "gradio", None)

    with pytest.raises(SystemExit) as excinfo:
        gradio_deid_app.build_demo()

    assert "pip install gradio" in str(excinfo.value)


def test_datasets_walkthrough_import_and_fixture_load_are_network_safe():
    fixture = datasets_walkthrough.load_fixture()

    assert datasets_walkthrough.fixture_path().is_file()
    assert fixture["id"] == datasets_walkthrough.FIXTURE_ID
    assert fixture["metadata"]["synthetic"] is True
    assert fixture["text"]


def test_datasets_walkthrough_reports_offline_unavailable(capsys, monkeypatch):
    def unavailable_extractor(*args, **kwargs):
        raise OSError("cached files not found")

    monkeypatch.setattr(datasets_walkthrough, "extract_pii", unavailable_extractor)
    datasets_walkthrough.run_walkthrough()

    captured = capsys.readouterr().out
    assert "Synthetic data: yes" in captured
    assert "model unavailable offline" in captured
