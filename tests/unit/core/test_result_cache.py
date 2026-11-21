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


def test_cache_key_includes_model_revision() -> None:
    params = {"text": "Patient John", "model_name": "pii"}

    unpinned = result_cache.make_cache_key("extract_pii", params)
    rev_a = result_cache.make_cache_key("extract_pii", {**params, "revision": "a" * 40})
    rev_b = result_cache.make_cache_key("extract_pii", {**params, "revision": "b" * 40})
    via_kwargs = result_cache.make_cache_key(
        "analyze_text", {**params, "pipeline_kwargs": {"revision": "a" * 40}}
    )

    assert len({unpinned, rev_a, rev_b}) == 3
    assert via_kwargs != result_cache.make_cache_key("analyze_text", params)
    for key in (unpinned, rev_a, rev_b, via_kwargs):
        assert "Patient John" not in key
        assert "a" * 40 not in key


def test_model_generation_change_invalidates_cache() -> None:
    reset_cache()
    params = {"text": "Patient John", "model_name": "pii"}
    cache = result_cache.get_result_cache(max_entries=4)
    before = result_cache.make_cache_key("extract_pii", params)
    cache.set(before, "old weights")

    generation = result_cache.notify_model_generation_change()

    after = result_cache.make_cache_key("extract_pii", params)
    assert generation == result_cache.current_model_generation()
    assert after != before
    assert cache.get(before) is None
    assert cache.get(after) is None
    assert len(cache) == 0


def test_analyze_text_misses_cache_after_model_generation_change() -> None:
    reset_cache()

    class FakeLoader:
        config = None

        def __init__(self):
            self.create_calls = 0

        def create_pipeline(self, *args, **kwargs):
            self.create_calls += 1
            return lambda text, **call_kwargs: []

        def get_max_sequence_length(self, model_name, tokenizer=None):
            return 128

    loader = FakeLoader()
    kwargs = dict(
        model_name="fixture-ner-model",
        loader=loader,
        sentence_detection=False,
        cache_results=True,
    )

    openmed.analyze_text("History of asthma.", **kwargs)
    openmed.analyze_text("History of asthma.", **kwargs)
    assert loader.create_calls == 1

    result_cache.notify_model_generation_change()
    openmed.analyze_text("History of asthma.", **kwargs)
    assert loader.create_calls == 2


def test_scoped_cache_withdrawal_uses_confirmed_revision_and_isolates_subjects():
    import json

    from openmed.compliance.withdrawal import ParticipantCode, WithdrawalCoordinator

    a, b = "hmac-sha256:" + "a" * 64, "sha256:" + "b" * 64
    cache = result_cache.ResultCache(4)
    cache.set("private-key", "private-untagged")
    cache.set("private-key", "private-result-a", subject_scope_fingerprint=a)
    cache.set("private-key", "private-result-b", subject_scope_fingerprint=b)
    withdrawal = WithdrawalCoordinator({ParticipantCode.RESULT_CACHE: cache})
    initial = withdrawal.plan(a)
    cache.set("private-key", "private-new", subject_scope_fingerprint=a)
    with pytest.raises(ValueError, match="confirmation"):
        withdrawal.withdraw(a, confirmation=initial.confirmation_token)
    epoch = cache.withdrawal_epoch
    plan = withdrawal.plan(a)
    receipt = withdrawal.withdraw(a, confirmation=plan.confirmation_token)
    assert receipt.state == "success"
    assert (
        receipt.participants[0].applied == receipt.participants[0].unattributable == 1
    )
    assert "private" not in json.dumps(receipt.to_dict())
    cache.set(
        "inflight-key", "private-old", subject_scope_fingerprint=a, expected_epoch=epoch
    )
    assert cache.get("private-key", subject_scope_fingerprint=a) is None
    assert cache.get("inflight-key", subject_scope_fingerprint=a) is None
    assert cache.get("private-key", subject_scope_fingerprint=b) == "private-result-b"
    assert cache.get("private-key") == "private-untagged"
    cache.set("new", 1, subject_scope_fingerprint=a)
    cache.set("next", 2)
    cache.set("another", 3)
    assert cache.plan(b).count == 0
    cache.clear()
    assert cache.plan(a).count == cache.plan(a).unattributable_count == 0


def test_extract_pii_cache_subject_isolation(monkeypatch):
    reset_cache()
    monkeypatch.setattr(
        pii,
        "_extract_pii_batch",
        lambda texts, **kwargs: [PredictionResult(texts[0], [], "fixture", "time")],
    )
    args = {"cache_results": True, "model_name": "test"}
    scope = "sha256:" + "c" * 64
    first = pii.extract_pii("synthetic", subject_scope_fingerprint=scope, **args)
    assert (
        pii.extract_pii("synthetic", subject_scope_fingerprint=scope, **args) is first
    )
    assert pii.extract_pii("synthetic", **args) is not first
    assert result_cache.get_result_cache().plan(scope).count == 1
    with pytest.raises(ValueError, match="^invalid subject scope fingerprint$"):
        pii.extract_pii("synthetic", subject_scope_fingerprint="patient-name", **args)
