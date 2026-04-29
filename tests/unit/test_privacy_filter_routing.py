"""Backend dispatch for the privacy-filter family.

These tests assert that when a privacy-filter model is requested via
``extract_pii()`` / ``deidentify()``:

- On Apple Silicon with MLX importable: the MLX pipeline runs.
- Elsewhere (Linux, Intel Mac, MLX missing): an ``openai/privacy-filter``
  PyTorch pipeline runs, with a one-time ``UserWarning`` substituted for
  any MLX-only artifact name.

We mock both pipelines so the tests pass without downloading either model.
"""

from __future__ import annotations

import sys
import types
import warnings
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_pipeline(entities: List[dict]):
    """Return a callable that mimics a privacy-filter pipeline."""
    def _call(text: str) -> List[dict]:
        return entities
    return _call


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class TestSelectPrivacyFilterBackend:
    def test_mlx_artifact_on_mac_with_mlx(self):
        from openmed.core.backends import select_privacy_filter_backend
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=True):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-mlx-8bit") == "mlx"
            assert select_privacy_filter_backend("OpenMed/privacy-filter-mlx") == "mlx"

    def test_mlx_artifact_on_linux_falls_back_to_torch(self):
        from openmed.core.backends import select_privacy_filter_backend
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-mlx-8bit") == "torch"

    def test_torch_artifact_always_uses_torch(self):
        from openmed.core.backends import select_privacy_filter_backend
        # Even if MLX is available, requesting the upstream PyTorch model
        # should stay on Torch (no point converting).
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=True):
            assert select_privacy_filter_backend("openai/privacy-filter") == "torch"
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False):
            assert select_privacy_filter_backend("openai/privacy-filter") == "torch"

    def test_nemotron_mlx_artifact_on_mac_with_mlx(self):
        from openmed.core.backends import select_privacy_filter_backend
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=True):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron-mlx") == "mlx"
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron-mlx-8bit") == "mlx"

    def test_nemotron_mlx_artifact_on_linux_falls_back_to_torch(self):
        from openmed.core.backends import select_privacy_filter_backend
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron-mlx") == "torch"
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron-mlx-8bit") == "torch"

    def test_nemotron_torch_artifact_always_uses_torch(self):
        from openmed.core.backends import select_privacy_filter_backend
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=True):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron") == "torch"
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False):
            assert select_privacy_filter_backend("OpenMed/privacy-filter-nemotron") == "torch"


class TestResolvePrivacyFilterModel:
    def test_mlx_keeps_name(self):
        from openmed.core.backends import resolve_privacy_filter_model
        assert (
            resolve_privacy_filter_model("OpenMed/privacy-filter-mlx-8bit", "mlx")
            == "OpenMed/privacy-filter-mlx-8bit"
        )

    def test_torch_substitutes_mlx_only_artifact(self):
        from openmed.core.backends import (
            PRIVACY_FILTER_TORCH_FALLBACK,
            resolve_privacy_filter_model,
        )
        # Reset the warning cache so we can observe it
        from openmed.core import backends
        backends._warned_substitutions.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = resolve_privacy_filter_model("OpenMed/privacy-filter-mlx-8bit", "torch")
        assert actual == PRIVACY_FILTER_TORCH_FALLBACK
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_torch_keeps_native_torch_name(self):
        from openmed.core.backends import resolve_privacy_filter_model
        assert resolve_privacy_filter_model("openai/privacy-filter", "torch") == "openai/privacy-filter"

    def test_nemotron_mlx_substitutes_to_nemotron_torch_repo(self):
        """An MLX-only Nemotron request must fall back to the Nemotron PyTorch
        repo (NOT the unrelated default ``openai/privacy-filter``)."""
        from openmed.core.backends import resolve_privacy_filter_model
        from openmed.core import backends
        backends._warned_substitutions.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            actual = resolve_privacy_filter_model(
                "OpenMed/privacy-filter-nemotron-mlx-8bit", "torch",
            )
        assert actual == "OpenMed/privacy-filter-nemotron"
        # And it warns once.
        assert any(
            issubclass(w.category, UserWarning)
            and "OpenMed/privacy-filter-nemotron" in str(w.message)
            for w in caught
        )

    def test_nemotron_torch_repo_keeps_name(self):
        from openmed.core.backends import resolve_privacy_filter_model
        assert (
            resolve_privacy_filter_model("OpenMed/privacy-filter-nemotron", "torch")
            == "OpenMed/privacy-filter-nemotron"
        )

    def test_torch_fallback_for_helper(self):
        """The substring matcher picks the right family fallback."""
        from openmed.core.backends import _torch_fallback_for, PRIVACY_FILTER_TORCH_FALLBACK
        assert _torch_fallback_for("OpenMed/privacy-filter-nemotron-mlx") == "OpenMed/privacy-filter-nemotron"
        assert _torch_fallback_for("OpenMed/privacy-filter-mlx") == PRIVACY_FILTER_TORCH_FALLBACK
        assert _torch_fallback_for("openai/privacy-filter") == PRIVACY_FILTER_TORCH_FALLBACK


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

class TestCreatePrivacyFilterPipeline:
    def test_mlx_route_calls_create_mlx_pipeline(self):
        from openmed.core.backends import create_privacy_filter_pipeline

        fake_mlx = types.ModuleType("openmed.mlx")
        fake_mlx.__path__ = []
        fake_inference = types.ModuleType("openmed.mlx.inference")
        fake_inference.create_mlx_pipeline = MagicMock(return_value=_fake_pipeline([]))
        fake_mlx.inference = fake_inference

        with patch("openmed.core.backends.MLXBackend.is_available", return_value=True), \
             patch.dict(sys.modules, {
                 "openmed.mlx": fake_mlx,
                 "openmed.mlx.inference": fake_inference,
             }):
            pipeline = create_privacy_filter_pipeline("OpenMed/privacy-filter-mlx-8bit")
            fake_inference.create_mlx_pipeline.assert_called_once_with("OpenMed/privacy-filter-mlx-8bit")
            assert pipeline("hello") == []

    def test_torch_route_constructs_torch_pipeline(self):
        from openmed.core.backends import create_privacy_filter_pipeline
        sentinel = _fake_pipeline([{"entity_group": "NAME", "score": 0.9, "word": "X", "start": 0, "end": 1}])
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False), \
             patch("openmed.torch.privacy_filter.PrivacyFilterTorchPipeline") as MockPF:
            MockPF.return_value = sentinel
            pipeline = create_privacy_filter_pipeline("openai/privacy-filter")
            MockPF.assert_called_once_with("openai/privacy-filter")
            assert pipeline("hi") == sentinel("hi")

    def test_mlx_request_on_linux_substitutes_torch_model(self):
        """An MLX-only request on Linux should resolve to openai/privacy-filter."""
        from openmed.core import backends
        backends._warned_substitutions.clear()
        from openmed.core.backends import create_privacy_filter_pipeline
        with patch("openmed.core.backends.MLXBackend.is_available", return_value=False), \
             patch("openmed.torch.privacy_filter.PrivacyFilterTorchPipeline") as MockPF, \
             warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            MockPF.return_value = _fake_pipeline([])
            create_privacy_filter_pipeline("OpenMed/privacy-filter-mlx-8bit")
            MockPF.assert_called_once_with("openai/privacy-filter")
            assert any(issubclass(w.category, UserWarning) for w in caught)


# ---------------------------------------------------------------------------
# extract_pii() integration
# ---------------------------------------------------------------------------

class TestExtractPiiViaPrivacyFilter:
    def test_extract_pii_routes_privacy_filter_to_dispatcher(self):
        """``extract_pii(model_name='OpenMed/privacy-filter-mlx-8bit')`` should
        call the dispatcher and skip ``analyze_text``."""
        from openmed.core.pii import extract_pii

        fake_entities = [
            {"entity_group": "NAME", "score": 0.92, "word": "John Doe",
             "start": 8, "end": 16},
            {"entity_group": "DATE", "score": 0.88, "word": "1970-01-15",
             "start": 25, "end": 35},
        ]

        with patch("openmed.core.pii.create_privacy_filter_pipeline",
                   create=True) as mock_factory, \
             patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be:
            mock_be.return_value = _fake_pipeline(fake_entities)

            with patch("openmed.analyze_text") as mock_analyze:
                result = extract_pii(
                    "Patient John Doe born on 1970-01-15",
                    model_name="OpenMed/privacy-filter-mlx-8bit",
                    confidence_threshold=0.5,
                )

            # analyze_text must NOT have been called for privacy-filter
            mock_analyze.assert_not_called()

        assert len(result.entities) == 2
        assert result.entities[0].label == "NAME"
        assert result.entities[0].text == "John Doe"
        assert result.entities[1].label == "DATE"

    def test_extract_pii_drops_below_threshold(self):
        from openmed.core.pii import extract_pii

        fake_entities = [
            {"entity_group": "NAME", "score": 0.92, "word": "John", "start": 0, "end": 4},
            {"entity_group": "EMAIL", "score": 0.30, "word": "x", "start": 6, "end": 7},
        ]

        with patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be, \
             patch("openmed.analyze_text"):
            mock_be.return_value = _fake_pipeline(fake_entities)
            result = extract_pii(
                "John, x",
                model_name="openai/privacy-filter",
                confidence_threshold=0.5,
            )
        assert len(result.entities) == 1
        assert result.entities[0].label == "NAME"

    def test_non_privacy_filter_model_takes_normal_path(self):
        """Regular PII models should still route through ``analyze_text``."""
        from openmed.core.pii import extract_pii
        from openmed.processing.outputs import EntityPrediction, PredictionResult

        with patch("openmed.analyze_text") as mock_analyze, \
             patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be:
            mock_analyze.return_value = PredictionResult(
                text="John Doe", entities=[], model_name="test",
                timestamp="now",
            )
            extract_pii(
                "John Doe",
                model_name="OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
            )
            mock_analyze.assert_called_once()
            mock_be.assert_not_called()


# ---------------------------------------------------------------------------
# deidentify() integration via privacy-filter
# ---------------------------------------------------------------------------

class TestDeidentifyViaPrivacyFilter:
    def test_deidentify_replace_with_privacy_filter_consistent(self):
        from openmed.core.pii import deidentify

        fake_entities = [
            {"entity_group": "NAME", "score": 0.95, "word": "John Doe", "start": 8, "end": 16},
        ]

        with patch("openmed.core.backends.create_privacy_filter_pipeline") as mock_be:
            mock_be.return_value = _fake_pipeline(fake_entities)
            r1 = deidentify(
                "Patient John Doe came in.",
                method="replace",
                model_name="openai/privacy-filter",
                lang="en",
                consistent=True,
                seed=42,
                confidence_threshold=0.5,
            )
            r2 = deidentify(
                "Patient John Doe came in.",
                method="replace",
                model_name="openai/privacy-filter",
                lang="en",
                consistent=True,
                seed=42,
                confidence_threshold=0.5,
            )
        assert r1.deidentified_text == r2.deidentified_text
        assert "John Doe" not in r1.deidentified_text
