"""Smoke tests for import-safe example scripts."""

from __future__ import annotations

import ast
import importlib
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from jsonschema import validate

from examples import (
    clinical_ner_families,
    datasets_walkthrough,
    deid_chinese_clinical_note,
    deid_hindi_hinglish_note,
    gradio_deid_app,
    onboarding_china_mirrors,
    onboarding_india_dpdp,
)

deid_demo = importlib.import_module("examples.spaces.deid_demo.app")


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
    assert all(call["group_entities"] is False for call in calls)
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


@pytest.mark.parametrize(
    "example_path",
    [
        Path("examples/deid_chinese_clinical_note.py"),
        Path("examples/deid_hindi_hinglish_note.py"),
    ],
)
def test_chinese_hindi_deid_examples_are_syntactically_valid(example_path):
    ast.parse(example_path.read_text(encoding="utf-8"))


def test_chinese_deid_recognizer_covers_every_synthetic_identifier():
    from openmed.core.custom_recognizer import CustomRecognizer

    recognizer = CustomRecognizer.from_config(
        deid_chinese_clinical_note.CHINESE_CUSTOM_RECOGNIZER
    )

    detected = {
        entity.text
        for entity in recognizer.detect_entities(
            deid_chinese_clinical_note.SYNTHETIC_CHINESE_NOTE
        )
    }

    assert detected == set(deid_chinese_clinical_note.SYNTHETIC_IDENTIFIERS)


def test_chinese_deid_example_runs_without_network(tmp_path, monkeypatch):
    from openmed.core import pii
    from openmed.processing.outputs import PredictionResult

    calls = []

    def fake_extract_pii(text, model_name, *args, **kwargs):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        calls.append((text, model_name, kwargs["lang"]))
        return PredictionResult(
            text=text,
            entities=[],
            model_name=model_name,
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)
    output_path = tmp_path / "zh" / "redacted.txt"

    result = deid_chinese_clinical_note.run_chinese_deidentification(output_path)

    assert calls == [
        (
            deid_chinese_clinical_note.SYNTHETIC_CHINESE_NOTE,
            deid_chinese_clinical_note.MODEL_KEY,
            "zh",
        )
    ]
    deid_chinese_clinical_note.assert_synthetic_identifiers_removed(
        result.deidentified_text
    )
    assert output_path.read_text(encoding="utf-8") == result.deidentified_text + "\n"
    rows = deid_chinese_clinical_note.structured_entities(result)
    assert {row["text"] for row in rows} == set(
        deid_chinese_clinical_note.SYNTHETIC_IDENTIFIERS
    )


def test_chinese_deid_example_fails_closed_on_synthetic_leak():
    with pytest.raises(AssertionError, match="not redacted"):
        deid_chinese_clinical_note.assert_synthetic_identifiers_removed(
            deid_chinese_clinical_note.SYNTHETIC_CHINESE_NOTE
        )


@pytest.mark.parametrize(
    ("note_name", "note"),
    list(deid_hindi_hinglish_note.NOTES.items()),
)
def test_hindi_hinglish_recognizer_covers_every_synthetic_identifier(
    note_name,
    note,
):
    from openmed.core.custom_recognizer import ABDMRecognizer, CustomRecognizer

    recognizer = CustomRecognizer.from_config(
        deid_hindi_hinglish_note.INDIA_CUSTOM_RECOGNIZER
    )

    custom_detected = {entity.text for entity in recognizer.detect_entities(note)}
    abdm_detected = {
        (entity.text, entity.label) for entity in ABDMRecognizer().detect_entities(note)
    }

    expected_custom = {
        deid_hindi_hinglish_note.SYNTHETIC_MEDICAL_RECORD_NUMBER,
        deid_hindi_hinglish_note.SYNTHETIC_PHONE,
        deid_hindi_hinglish_note.SYNTHETIC_HINDI_NAME
        if note_name == "hindi"
        else deid_hindi_hinglish_note.SYNTHETIC_HINGLISH_NAME,
    }
    if note_name == "hindi":
        expected_custom.add(deid_hindi_hinglish_note.SYNTHETIC_CLINICIAN_NAME)

    assert custom_detected == expected_custom
    assert abdm_detected == {
        (deid_hindi_hinglish_note.SYNTHETIC_AADHAAR, "AADHAAR"),
        (deid_hindi_hinglish_note.SYNTHETIC_ABHA, "ABHA_NUMBER"),
    }


def test_hindi_hinglish_deid_example_runs_without_network(tmp_path, monkeypatch):
    from openmed.core import pii
    from openmed.processing.outputs import PredictionResult

    calls = []

    def fake_extract_pii(text, model_name, *args, **kwargs):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        calls.append((text, model_name, kwargs["lang"]))
        return PredictionResult(
            text=text,
            entities=[],
            model_name=model_name,
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)
    output_dir = tmp_path / "hi"

    results = deid_hindi_hinglish_note.run_hindi_hinglish_deidentification(output_dir)

    assert calls == [
        (
            note,
            deid_hindi_hinglish_note.HINDI_MODEL_ID,
            "hi",
        )
        for note in deid_hindi_hinglish_note.NOTES.values()
    ]
    assert set(results) == set(deid_hindi_hinglish_note.NOTES)
    for note_name, result in results.items():
        deid_hindi_hinglish_note.assert_synthetic_identifiers_removed(
            note_name,
            result.deidentified_text,
        )
        saved = output_dir / f"{note_name}_note_redacted.txt"
        assert saved.read_text(encoding="utf-8") == result.deidentified_text + "\n"
        rows = deid_hindi_hinglish_note.structured_entities(result)
        assert {row["text"] for row in rows} == set(
            deid_hindi_hinglish_note.SYNTHETIC_IDENTIFIERS_BY_NOTE[note_name]
        )


@pytest.mark.parametrize(
    ("note_name", "note"),
    list(deid_hindi_hinglish_note.NOTES.items()),
)
def test_hindi_hinglish_deid_example_fails_closed_on_synthetic_leak(
    note_name,
    note,
):
    with pytest.raises(AssertionError, match="survived"):
        deid_hindi_hinglish_note.assert_synthetic_identifiers_removed(
            note_name,
            note,
        )


@pytest.mark.parametrize(
    ("text", "expected_language"),
    [
        ("患者李晓雯今天复诊。", "zh"),
        ("रोगी आज जाँच के लिए आए।", "hi"),
        ("Patient Alex Example was seen today.", "en"),
        ("Patient Ananya ka follow-up aaj hai.", "en"),
        ("1234 / +1 555 0100", "en"),
    ],
)
def test_deid_demo_detects_unicode_script(text, expected_language):
    assert deid_demo.detect_script(text) == expected_language


@pytest.mark.parametrize("override", ["zh", "hi", "en"])
def test_deid_demo_manual_override_wins(override):
    route = deid_demo.resolve_model_route("患者 हिन्दी Patient", override)

    assert route.language == override
    assert route.model_id == deid_demo.MODEL_IDS[override]


@pytest.mark.parametrize(
    ("sample_language", "openmed_language"),
    [("zh", "zh"), ("hi", "hi"), ("en", "en")],
)
def test_deid_demo_routes_samples_without_network(
    sample_language,
    openmed_language,
):
    calls = []

    def fake_deidentify(text, **kwargs):
        calls.append((text, kwargs))
        return SimpleNamespace(deidentified_text=f"[{sample_language.upper()} MASKED]")

    sample = deid_demo.SAMPLES[sample_language]
    result = deid_demo.run_deidentification(
        sample.text,
        deidentifier=fake_deidentify,
    )

    assert result.deidentified_text == f"[{sample_language.upper()} MASKED]"
    assert result.route.language == sample_language
    assert calls[0][0] == sample.text
    assert calls[0][1] == {
        "method": "mask",
        "model_name": deid_demo.MODEL_IDS[sample_language],
        "lang": openmed_language,
        "policy": "strict_no_leak",
        "use_safety_sweep": True,
        "custom_recognizer": deid_demo._sample_recognizer(sample_language),
        "keep_mapping": False,
        "audit": False,
        "cache_results": False,
    }


def test_deid_demo_models_are_registered():
    from openmed.core.model_registry import get_model_info

    assert all(
        get_model_info(model_id) is not None
        for model_id in deid_demo.MODEL_IDS.values()
    )


def test_deid_demo_samples_run_through_pipeline_without_network(monkeypatch):
    from openmed.core import pii
    from openmed.processing.outputs import PredictionResult

    calls = []

    def fake_extract_pii(text, model_name, *args, **kwargs):
        calls.append((text, model_name, kwargs["lang"]))
        return PredictionResult(
            text=text,
            entities=[],
            model_name=model_name,
            timestamp="2026-01-01T00:00:00",
        )

    monkeypatch.setattr(pii, "extract_pii", fake_extract_pii)

    for language, sample in deid_demo.SAMPLES.items():
        result = deid_demo.run_deidentification(sample.text, language)

        assert all(
            identifier not in result.deidentified_text
            for identifier, _label in sample.identifiers
        )

    assert calls == [
        (sample.text, deid_demo.MODEL_IDS[language], language)
        for language, sample in deid_demo.SAMPLES.items()
    ]


def test_deid_demo_rejects_unknown_override():
    with pytest.raises(ValueError, match="Unsupported language override"):
        deid_demo.resolve_model_route("Synthetic note", "fr")


def test_deid_demo_does_not_log_or_persist_input(caplog, monkeypatch, tmp_path):
    raw_input = "Synthetic secret ZH-709-NEVER-LOG"
    monkeypatch.chdir(tmp_path)

    def fake_deidentify(text, **kwargs):
        del text, kwargs
        return SimpleNamespace(deidentified_text="Synthetic secret [ID_NUM]")

    result = deid_demo.run_deidentification(
        raw_input,
        language_override="en",
        deidentifier=fake_deidentify,
    )

    assert result.deidentified_text == "Synthetic secret [ID_NUM]"
    assert raw_input not in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_deid_demo_disclaimer_contains_all_locales():
    assert set(deid_demo.DISCLAIMERS) == {"zh", "hi", "en"}
    assert all(
        disclaimer in deid_demo.DISCLAIMER_MARKDOWN
        for disclaimer in deid_demo.DISCLAIMERS.values()
    )
    assert "Never paste real patient data" in deid_demo.DISCLAIMERS["en"]
    assert "真实患者数据" in deid_demo.DISCLAIMERS["zh"]
    assert "वास्तविक रोगी डेटा" in deid_demo.DISCLAIMERS["hi"]


def test_deid_demo_space_frontmatter_matches_schema():
    readme = Path("examples/spaces/deid_demo/README.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", readme, flags=re.DOTALL)
    assert match is not None
    metadata = yaml.safe_load(match.group(1))
    schema = {
        "type": "object",
        "required": ["title", "sdk", "sdk_version", "app_file", "models"],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "sdk": {"const": "gradio"},
            "sdk_version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "python_version": {"type": "string", "pattern": r"^3\.\d+(?:\.\d+)?$"},
            "app_file": {"const": "app.py"},
            "models": {
                "type": "array",
                "minItems": 3,
                "items": {"type": "string", "pattern": r"^OpenMed/"},
            },
        },
    }

    validate(instance=metadata, schema=schema)
    assert Path("examples/spaces/deid_demo", metadata["app_file"]).is_file()
    assert metadata["sdk_version"] == "6.20.0"


def test_deid_demo_requirements_are_pinned():
    requirements = Path("examples/spaces/deid_demo/requirements.txt").read_text(
        encoding="utf-8"
    )
    lines = [
        line for line in requirements.splitlines() if line and not line.startswith("#")
    ]

    assert lines
    assert all(
        re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[^\s]+", line)
        for line in lines
    )


def test_deid_demo_readme_has_local_run_instructions():
    readme = Path("examples/spaces/deid_demo/README.md").read_text(encoding="utf-8")

    assert "python -m pip install -r requirements.txt" in readme
    assert "python app.py" in readme
