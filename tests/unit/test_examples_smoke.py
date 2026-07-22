"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import (
    clinical_ner_families,
    datasets_walkthrough,
    gradio_deid_app,
    onboarding_china_mirrors,
    onboarding_india_dpdp,
)


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


def test_onboarding_china_mirrors_example_is_syntactically_valid():
    source = Path("examples/onboarding_china_mirrors.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_onboarding_india_dpdp_example_is_syntactically_valid():
    source = Path("examples/onboarding_india_dpdp.py").read_text(encoding="utf-8")

    ast.parse(source)


def test_onboarding_china_mirrors_configures_download_endpoint(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert "HF_HUB_OFFLINE" not in os.environ
        assert "TRANSFORMERS_OFFLINE" not in os.environ
        calls.append(kwargs)
        return "/cache/downloaded-snapshot"

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    path = onboarding_china_mirrors.predownload_model(loader=fake_snapshot_download)

    assert path == Path("/cache/downloaded-snapshot")
    assert calls == [
        {
            "repo_id": onboarding_china_mirrors.DEFAULT_MODEL_ID,
            "repo_type": "model",
            "local_files_only": False,
        }
    ]
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_onboarding_china_mirrors_loads_cache_without_network(monkeypatch):
    calls = []

    def fake_snapshot_download(**kwargs):
        assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        calls.append(kwargs)
        return "/cache/offline-snapshot"

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    path = onboarding_china_mirrors.load_cached_model(loader=fake_snapshot_download)

    assert path == Path("/cache/offline-snapshot")
    assert calls == [
        {
            "repo_id": onboarding_china_mirrors.DEFAULT_MODEL_ID,
            "repo_type": "model",
            "local_files_only": True,
        }
    ]
    assert "HF_ENDPOINT" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_onboarding_china_mirrors_run_defaults_to_offline_cache(
    capsys,
    monkeypatch,
):
    def fake_snapshot_download(**kwargs):
        assert kwargs["local_files_only"] is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return "/cache/offline-snapshot"

    monkeypatch.delenv(onboarding_china_mirrors.ALLOW_DOWNLOAD_ENV, raising=False)

    path = onboarding_china_mirrors.run(loader=fake_snapshot_download)

    assert path == Path("/cache/offline-snapshot")
    captured = capsys.readouterr().out
    assert "Offline cache-only mode" in captured
    assert "Resolved cached model" in captured


def test_onboarding_india_dpdp_recognizers_cover_synthetic_direct_identifiers():
    from openmed.core.custom_recognizer import ABDMRecognizer, CustomRecognizer

    name_recognizer = CustomRecognizer.from_config(
        onboarding_india_dpdp.SYNTHETIC_NAME_RECOGNIZER
    )
    abdm_recognizer = ABDMRecognizer()

    assert {
        (entity.text, entity.label)
        for entity in name_recognizer.detect_entities(
            onboarding_india_dpdp.SYNTHETIC_HINGLISH_NOTE
        )
    } == {
        (onboarding_india_dpdp.SYNTHETIC_PERSON, "PERSON"),
    }
    assert {
        (entity.text, entity.label)
        for entity in abdm_recognizer.detect_entities(
            onboarding_india_dpdp.SYNTHETIC_HINGLISH_NOTE
        )
    } == {
        (onboarding_india_dpdp.SYNTHETIC_AADHAAR, "AADHAAR"),
        (onboarding_india_dpdp.SYNTHETIC_ABHA, "ABHA_NUMBER"),
    }


def test_onboarding_india_dpdp_runs_policy_pipeline_without_network(monkeypatch):
    from openmed.core import pii
    from openmed.processing.outputs import PredictionResult

    calls = []

    def fake_extract_pii(text, model_name, *args, **kwargs):
        calls.append(
            {
                "text": text,
                "model_name": model_name,
                "lang": kwargs["lang"],
            }
        )
        return PredictionResult(
            text=text,
            entities=[],
            model_name=model_name,
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)

    result = onboarding_india_dpdp.run()

    assert calls == [
        {
            "text": onboarding_india_dpdp.SYNTHETIC_HINGLISH_NOTE,
            "model_name": onboarding_india_dpdp.HINDI_MODEL_ID,
            "lang": "hi",
        }
    ]
    onboarding_india_dpdp.assert_synthetic_pii_is_deidentified(result.deidentified_text)
    protected_values = {entity.text for entity in result.pii_entities}
    assert {
        onboarding_india_dpdp.SYNTHETIC_PERSON,
        onboarding_india_dpdp.SYNTHETIC_AADHAAR,
        onboarding_india_dpdp.SYNTHETIC_ABHA,
    }.issubset(protected_values)
    assert all(entity.action == "replace" for entity in result.pii_entities)
