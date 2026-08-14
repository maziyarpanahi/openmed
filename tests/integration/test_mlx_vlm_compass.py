"""Real-model parity tests for the OpenMed Cohere Compass MLX runtime."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_VARIANTS = ("4bit", "5bit", "6bit", "8bit", "bf16")
_PRIVACY_PROMPT = (
    "In one concise sentence, explain how running a vision-language model "
    "entirely on-device can improve privacy for clinical documents."
)
_FACT_PROMPT = (
    'A synthetic note states: "The follow-up appointment is scheduled for '
    'Tuesday at 10:30 AM." What day is the follow-up? Answer with only the day.'
)
_DOCUMENT_PROMPT = (
    "This is synthetic test data. In one concise sentence, report the exact "
    "patient name, record ID, medication with dose and frequency, and allergy "
    "shown in the image."
)
_CHART_PROMPT = (
    "Which category has the tallest bar, and what exact value is printed above "
    "it? Answer concisely."
)


def _configured_directory(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        pytest.skip(f"set {variable} to run real Compass model tests")
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        pytest.fail(f"{variable} is not a directory: {directory}")
    return directory


def _reference_responses(directory: Path, variant: str) -> dict[str, str]:
    report = json.loads((directory / f"{variant}.json").read_text())
    return {case["id"]: case["response"] for case in report["cases"]}


@pytest.mark.parametrize("variant", _VARIANTS)
def test_compass_text_and_image_generation_matches_contract(variant: str) -> None:
    """Strictly load each precision and validate deterministic clinical tasks."""

    artifact_root = _configured_directory("OPENMED_COMPASS_MLX_ARTIFACT_ROOT")
    fixture_directory = _configured_directory("OPENMED_COMPASS_FIXTURE_DIRECTORY")
    report_directory = _configured_directory(
        "OPENMED_COMPASS_REFERENCE_REPORT_DIRECTORY"
    )
    artifact = artifact_root / f"North-Micro-Vision-Instruct-{variant}-mlx"
    if not artifact.is_dir():
        pytest.fail(f"missing {variant} artifact: {artifact}")

    from openmed.mlx import OpenMedMLXVisionLanguageModel

    model = OpenMedMLXVisionLanguageModel(artifact, strict=True)
    expected = _reference_responses(report_directory, variant)
    try:
        privacy = model.generate_with_metadata(_PRIVACY_PROMPT, max_tokens=96)
        assert privacy.prompt_tokens == 30
        privacy_text = privacy.text.casefold()
        assert "private" in privacy_text or "privacy" in privacy_text
        assert all(term in privacy_text for term in ("clinical", "device", "sensitive"))
        assert any(
            term in privacy_text for term in ("cloud", "network", "transmit", "local")
        )

        fact = model.generate_with_metadata(_FACT_PROMPT, max_tokens=32)
        assert fact.text == expected["text_fact_extraction"] == "Tuesday"
        assert fact.token_ids == (29_445,)
        assert fact.prompt_tokens == 44

        document = model.generate_with_metadata(
            _DOCUMENT_PROMPT,
            image=fixture_directory / "synthetic_clinical_document.png",
            max_tokens=96,
        )
        assert document.prompt_tokens == 1_161
        document_text = document.text.casefold()
        for value in (
            "Alex Rivera",
            "SYN-2048",
            "Metformin",
            "500",
            "twice",
            "Penicillin",
        ):
            assert value.casefold() in document_text

        chart = model.generate_with_metadata(
            _CHART_PROMPT,
            image=fixture_directory / "synthetic_clinic_chart.png",
            max_tokens=32,
        )
        assert chart.text == expected["image_chart"] == "Screening, 42"
        assert chart.token_ids == (198_759, 16, 225, 3_304)
        assert chart.prompt_tokens == 1_053
    finally:
        del model
        gc.collect()
        import mlx.core as mx

        mx.clear_cache()
