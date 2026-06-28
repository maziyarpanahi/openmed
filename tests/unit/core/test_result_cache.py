"""Tests for the LRU Cache (opt-out is the default)"""

import os
from datetime import datetime
from types import SimpleNamespace

import pytest

import openmed
import openmed.core.pii as pii
import openmed.core.result_cache as result_cache
from openmed.core import ModelLoader, OpenMedConfig
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.processing.outputs import PredictionResult


def reset_cache() -> None:
    result_cache.RESULT_CACHE = None


def test_result_cache_lru_and_keys() -> None:
    cache = result_cache.ResultCache(max_entries=2)
    cache.set("a", "A")
    cache.set("b", "B")
    assert cache.get("a") == "A"
    cache.set("c", "C")
    assert cache.get("b") is None
    assert cache.get("a") == "A"

    base = result_cache.make_cache_key(
        "extract_pii",
        {"text": " Patient John ", "model_name": "pii", "threshold": 0.5},
    )
    same = result_cache.make_cache_key(
        "extract_pii",
        {
            "text": "Patient John",
            "model_name": "pii",
            "threshold": 0.5,
            "cache_results": False,
            "max_cache_entries": 999,
        },
    )
    different = result_cache.make_cache_key(
        "extract_pii",
        {"text": "Patient John", "model_name": "pii", "threshold": 0.9},
    )
    assert base == same
    assert base != different
    assert "Patient John" not in base


def test_analyze_text_cache_with_five_calls() -> None:
    reset_cache()

    class FakeLoader:
        def __init__(self):
            self.create_calls = 0

        def create_pipeline(self, *args, **kwargs):
            self.create_calls += 1

            def fake_pipeline(text, **kwargs):
                return [
                    {
                        "word": "John",
                        "entity_group": "NAME",
                        "score": 0.95,
                        "start": 8,
                        "end": 12,
                    }
                ]

            return fake_pipeline

        def get_max_sequence_length(self, *args, **kwargs):
            return 128

    loader = FakeLoader()
    kwargs = dict(loader=loader, sentence_detection=False, cache_results=True)

    first = openmed.analyze_text("Patient John", model_name="unit", **kwargs)
    second = openmed.analyze_text("Patient John", model_name="unit", **kwargs)
    third = openmed.analyze_text(" Patient John ", model_name="unit", **kwargs)
    fourth = openmed.analyze_text("Patient John", model_name="other", **kwargs)
    fifth = openmed.analyze_text(
        "Patient John", model_name="unit", loader=loader, sentence_detection=False
    )

    assert second is first
    assert third is first
    assert fourth is not first
    assert fifth is not first
    assert loader.create_calls == 3


@pytest.mark.integration
@pytest.mark.slow
def test_real_analyze_text_cache_reuses_loaded_model(tmp_path) -> None:
    model_name = os.environ.get("OPENMED_RESULT_CACHE_REAL_MODEL")
    if not model_name:
        pytest.skip("Set OPENMED_RESULT_CACHE_REAL_MODEL to run real cache smoke test")
    pytest.importorskip("transformers")

    class CountingLoader:
        def __init__(self):
            self.inner = ModelLoader(
                OpenMedConfig(cache_dir=str(tmp_path), device="cpu")
            )
            self.create_calls = 0

        def create_pipeline(self, *args, **kwargs):
            self.create_calls += 1
            return self.inner.create_pipeline(*args, **kwargs)

        def get_max_sequence_length(self, *args, **kwargs):
            return self.inner.get_max_sequence_length(*args, **kwargs)

    reset_cache()
    loader = CountingLoader()
    kwargs = dict(
        model_name=model_name,
        loader=loader,
        sentence_detection=False,
        cache_results=True,
        max_cache_entries=4,
    )

    first = openmed.analyze_text("Patient John has diabetes.", **kwargs)
    second = openmed.analyze_text("Patient John has diabetes.", **kwargs)
    third = openmed.analyze_text("Patient Jane has diabetes.", **kwargs)

    assert second is first
    assert third is not first
    assert loader.create_calls == 2


def test_extract_pii_cache_with_five_calls(monkeypatch) -> None:
    reset_cache()
    calls = {"count": 0}

    def fake_batch(texts, **kwargs):
        calls["count"] += 1
        return [
            PredictionResult(
                texts[0], [], kwargs["model_name"], datetime.now().isoformat()
            )
        ]

    monkeypatch.setattr(pii, "_extract_pii_batch", fake_batch)
    kwargs = dict(model_name="pii", confidence_threshold=0.5, cache_results=True)

    first = pii.extract_pii("Patient John", **kwargs)
    second = pii.extract_pii("Patient John", **kwargs)
    third = pii.extract_pii(" Patient John ", **kwargs)
    fourth = pii.extract_pii(
        "Patient John", model_name="pii", confidence_threshold=0.9, cache_results=True
    )
    fifth = pii.extract_pii("Patient John", model_name="pii", confidence_threshold=0.5)

    assert second is first
    assert third is first
    assert fourth is not first
    assert fifth is not first
    assert calls["count"] == 3


def test_deidentify_cache_with_five_calls(monkeypatch) -> None:
    reset_cache()
    calls = {"count": 0}

    class FakePipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, text, **kwargs):
            calls["count"] += 1
            entity = PIIEntity("John", "NAME", 0.95, 8, 12, redacted_text="[NAME]")
            result = DeidentificationResult(
                text, "Patient [NAME]", [entity], kwargs["method"], datetime.now()
            )
            return SimpleNamespace(deidentification_result=result)

    import openmed.core.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "Pipeline", FakePipeline)
    kwargs = dict(method="mask", model_name="pii", cache_results=True)

    first = pii.deidentify("Patient John", **kwargs)
    second = pii.deidentify("Patient John", **kwargs)
    third = pii.deidentify(" Patient John ", **kwargs)
    fourth = pii.deidentify(
        "Patient John", method="hash", model_name="pii", cache_results=True
    )
    fifth = pii.deidentify("Patient John", method="mask", model_name="pii")

    assert second is first
    assert third is first
    assert fourth is not first
    assert fifth is not first
    assert calls["count"] == 3
