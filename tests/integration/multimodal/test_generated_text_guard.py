"""Offline MLX generation-boundary tests with synthetic in-memory inputs."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from openmed.mlx import vlm
from openmed.multimodal.generated_text_guard import (
    AssetIdentifierDigest,
    GeneratedTextGuard,
    PHILabel,
    identifier_digest,
)

pytestmark = pytest.mark.integration


def fake_runner(monkeypatch, output):
    core = ModuleType("mlx.core")
    core.eval = lambda _: None
    core.array = np.array
    core.int32 = np.int32
    core.logsumexp = lambda x, **_: 0
    core.get_peak_memory = lambda: 0
    mlx = ModuleType("mlx")
    mlx.core = core
    sample = ModuleType("mlx_lm.sample_utils")
    sample.make_sampler = lambda **_: lambda _: np.array(7)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", sample)
    runner = object.__new__(vlm.OpenMedMLXVisionLanguageModel)

    class Model:
        def reset_generation_state(self):
            pass

        def make_cache(self):
            return []

        def __call__(self, *args, **kwargs):
            return np.array([[[0.0, 1.0]]])

    runner.model = Model()
    runner.config = {}
    runner.processor = SimpleNamespace(
        prepare=lambda *args, **kwargs: SimpleNamespace(
            input_ids=np.array([[1]]),
            pixel_values=None,
            image_grid_thw=None,
            attention_mask=None,
        ),
        tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: output),
    )
    monkeypatch.setattr(vlm, "OpenMedMLXVisionLanguageModel", lambda _: runner)
    return runner


def test_fake_generation_cannot_release_identifier_or_original_tokens(monkeypatch):
    runner = fake_runner(monkeypatch, "MIRA VALE")
    guard = GeneratedTextGuard(lambda _: ())
    evidence = [
        AssetIdentifierDigest(
            "a" * 64, PHILabel.NAME, identifier_digest("Mira Vale"), 8
        )
    ]
    result = runner.generate_with_metadata(
        "Read label",
        max_tokens=1,
        release_guard=guard,
        asset_digest="a" * 64,
        identifiers=evidence,
    )
    assert result.text == "[REDACTED]"
    assert result.token_ids == ()
    assert result.release_envelope.guard_metadata["decision"] == "redact"
    assert (
        vlm.generate_vision_text(
            "Read label",
            max_tokens=1,
            release_guard=guard,
            asset_digest="a" * 64,
            identifiers=evidence,
        )
        == "[REDACTED]"
    )


def test_default_missing_detector_fails_closed(monkeypatch):
    runner = fake_runner(monkeypatch, "Mira Vale")
    result = runner.generate_with_metadata("Read label", max_tokens=1)
    assert result.text == ""
    assert result.token_ids == ()
    assert result.release_envelope.abstention_code.value == "phi_uncertainty"
    assert "Mira Vale" not in repr(result)
