"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace

from examples import clinical_ner_families


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
    assert all(selection.model.model_id.startswith("OpenMed/") for selection in selections)


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
