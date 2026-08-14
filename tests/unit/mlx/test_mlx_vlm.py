"""Tests for the OpenMed-owned MLX vision-language runtime."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.mlx.vlm import (
    CohereCompassProcessor,
    OpenMedMLXVLMArtifactError,
    resolve_mlx_vlm_model,
    smart_resize,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.messages = None

    def apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        content = messages[-1]["content"]
        if isinstance(content, str):
            return content
        rendered = []
        for item in content:
            rendered.append(
                "<|VISION_START|><|IMAGE_PAD|><|VISION_END|>"
                if item["type"] == "image"
                else item["text"]
            )
        return "".join(rendered)


def _artifact(directory: Path, *, model_type: str = "cohere_compass") -> Path:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"model_type": model_type}))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ):
        (directory / name).write_bytes(b"{}")
    return directory


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (900, 1280, (896, 1280)),
        (850, 1200, (864, 1216)),
        (10, 10, (128, 128)),
    ],
)
def test_smart_resize_matches_compass_patch_grid(height, width, expected):
    assert smart_resize(height, width) == expected


def test_smart_resize_rejects_pathological_aspect_ratio():
    with pytest.raises(ValueError, match="aspect ratio"):
        smart_resize(1, 201)


def test_resolve_local_compass_artifact(tmp_path):
    artifact = _artifact(tmp_path / "model")

    assert resolve_mlx_vlm_model(artifact) == artifact.resolve()


def test_resolve_rejects_wrong_architecture(tmp_path):
    artifact = _artifact(tmp_path / "model", model_type="qwen2_vl")

    with pytest.raises(OpenMedMLXVLMArtifactError, match="model_type"):
        resolve_mlx_vlm_model(artifact)


def test_resolve_downloads_only_data_artifacts(monkeypatch, tmp_path):
    artifact = _artifact(tmp_path / "downloaded")
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(artifact)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    resolved = resolve_mlx_vlm_model(
        "OpenMed/North-Micro-Vision-Instruct-4bit-mlx",
        revision="012345",
    )

    assert resolved == artifact
    assert calls[0]["revision"] == "012345"
    assert "*.py" not in calls[0]["allow_patterns"]
    assert "model*.safetensors" in calls[0]["allow_patterns"]


def test_processor_places_images_before_user_text():
    tokenizer = _FakeTokenizer()
    processor = CohereCompassProcessor(tokenizer, {"vision_config": {}})

    rendered = processor.format_prompt("Read this chart", image_count=2)

    assert rendered == (
        "<|VISION_START|><|IMAGE_PAD|><|VISION_END|>"
        "<|VISION_START|><|IMAGE_PAD|><|VISION_END|>"
        "Read this chart"
    )
    assert tokenizer.messages[-1]["content"][-1]["text"] == "Read this chart"


def test_processor_uses_model_patch_configuration():
    processor = CohereCompassProcessor(
        _FakeTokenizer(),
        {
            "vision_config": {
                "patch_size": 14,
                "temporal_patch_size": 4,
                "spatial_merge_size": 3,
            },
            "min_pixels": 4096,
            "max_pixels": 1_000_000,
        },
    )

    assert processor.patch_size == 14
    assert processor.temporal_patch_size == 4
    assert processor.merge_size == 3
    assert processor.min_pixels == 4096
    assert processor.max_pixels == 1_000_000
