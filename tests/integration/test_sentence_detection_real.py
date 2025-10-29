"""Slow integration tests that exercise sentence detection with real models."""

import os
from pathlib import Path
from typing import Tuple

import pytest


def _skip_if_env_disabled() -> None:
    """Skip the test suite if networked Hugging Face models are unavailable."""
    if os.environ.get("OPENMED_SKIP_HF_TESTS") == "1":
        pytest.skip("Skipping Hugging Face dependent tests via OPENMED_SKIP_HF_TESTS")


@pytest.mark.slow
def test_sentence_detection_short_text_consistency(tmp_path):
    """Sentence detection should behave identically on short inputs."""
    transformers = pytest.importorskip("transformers")  # noqa: F841
    torch = pytest.importorskip("torch")  # noqa: F841
    _skip_if_env_disabled()

    from openmed import analyze_text, OpenMedConfig, ModelLoader

    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = OpenMedConfig(cache_dir=str(cache_dir), device="cpu")
    loader = ModelLoader(config)
    model_id = "dslim/bert-base-NER"

    text = "John Smith visited London in April."

    common_kwargs = dict(
        model_name=model_id,
        loader=loader,
        config=config,
        output_format="dict",
        max_length=128,
        batch_size=8,
    )

    result_sd = analyze_text(text, sentence_detection=True, **common_kwargs)
    result_no = analyze_text(text, sentence_detection=False, **common_kwargs)

    def _to_span(result) -> Tuple[Tuple[int, int, str], ...]:
        return tuple(sorted((ent.start or -1, ent.end or -1, ent.label) for ent in result.entities))

    assert _to_span(result_sd) == _to_span(result_no)


@pytest.mark.slow
def test_sentence_detection_filters_placeholders(tmp_path):
    """Placeholder-only segments should not yield entities when sentence detection is enabled."""
    transformers = pytest.importorskip("transformers")  # noqa: F841
    torch = pytest.importorskip("torch")  # noqa: F841
    _skip_if_env_disabled()

    from openmed import analyze_text, OpenMedConfig, ModelLoader

    cache_dir = tmp_path / "hf-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    config = OpenMedConfig(cache_dir=str(cache_dir), device="cpu")
    loader = ModelLoader(config)
    model_id = "OpenMed/OpenMed-NER-OncologyDetect-SuperMedical-355M"

    note_path = Path("tests/fixtures/clinical_note.txt")
    text = note_path.read_text().strip()

    result_sd = analyze_text(
        text,
        model_name=model_id,
        loader=loader,
        config=config,
        output_format="dict",
        sentence_detection=True,
        confidence_threshold=0.8,
        group_entities=True,
        max_length=512,
        batch_size=8,
    )

    result_no = analyze_text(
        text,
        model_name=model_id,
        loader=loader,
        config=config,
        output_format="dict",
        sentence_detection=False,
        confidence_threshold=0.8,
        group_entities=True,
        max_length=512,
        batch_size=8,
    )

    def _entity_fingerprint(result):
        return sorted(
            (ent.text, ent.label, ent.start, ent.end)
            for ent in result.entities
        )

    assert _entity_fingerprint(result_sd) == _entity_fingerprint(result_no)
    assert all(ent.text.strip("_- \n\t") for ent in result_sd.entities)
