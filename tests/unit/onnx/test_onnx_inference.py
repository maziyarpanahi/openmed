"""Tests for the public OpenMed ONNX inference API."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openmed.onnx import inference


class FakeTokenizer:
    """Small tokenizer fixture returning deterministic source offsets."""

    loaded_from: tuple[str, dict] | None = None

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        cls.loaded_from = (model_id, kwargs)
        return cls()

    def __call__(self, text: str, **kwargs):
        assert text == "Alice Nguyen"
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_tensors"] == "np"
        return {
            "input_ids": np.array([[101, 11, 12, 102]], dtype=np.int64),
            "attention_mask": np.ones((1, 4), dtype=np.int64),
            "offset_mapping": np.array(
                [[[0, 0], [0, 5], [6, 12], [0, 0]]],
                dtype=np.int64,
            ),
        }


class FakeSession:
    """ONNX Runtime session fixture with BIO person logits."""

    def __init__(self, model_path, sess_options=None, providers=None):
        self.model_path = Path(model_path)
        self.sess_options = sess_options
        self.providers = providers
        self.last_feed = None

    def get_inputs(self):
        return [
            SimpleNamespace(name="input_ids", type="tensor(int64)"),
            SimpleNamespace(name="attention_mask", type="tensor(int64)"),
            SimpleNamespace(name="token_type_ids", type="tensor(int64)"),
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, feed):
        assert output_names == ["logits"]
        self.last_feed = feed
        return [
            np.array(
                [
                    [
                        [9.0, 0.0, 0.0],
                        [0.0, 9.0, 0.0],
                        [0.0, 0.0, 8.0],
                        [9.0, 0.0, 0.0],
                    ]
                ],
                dtype=np.float32,
            )
        ]


class FakeOrt:
    InferenceSession = FakeSession


def _write_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "model.onnx").write_bytes(b"fp32")
    (artifact_dir / "model_int8.onnx").write_bytes(b"int8")
    (artifact_dir / "config.json").write_text(
        json.dumps({"id2label": {"0": "O", "1": "B-PERSON", "2": "E-PERSON"}}),
        encoding="utf-8",
    )
    (artifact_dir / "id2label.json").write_text(
        json.dumps({"0": "O", "1": "B-PERSON", "2": "E-PERSON"}),
        encoding="utf-8",
    )
    (artifact_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return artifact_dir


def _install_dependencies(monkeypatch, snapshot_download):
    monkeypatch.setattr(
        inference,
        "_load_runtime_dependencies",
        lambda: (np, FakeOrt, FakeTokenizer, snapshot_download),
    )


def test_from_pretrained_loads_cpu_int8_and_predicts_entities(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_dir = _write_artifact(tmp_path)
    _install_dependencies(monkeypatch, lambda **kwargs: str(artifact_dir))

    model = inference.OnnxModel.from_pretrained(artifact_dir)
    entities = model("Alice Nguyen", threshold=0.9)

    assert model.variant == "int8"
    assert model.model_path == artifact_dir / "model_int8.onnx"
    assert model.session.providers == ["CPUExecutionProvider"]
    assert np.array_equal(
        model.session.last_feed["token_type_ids"],
        np.zeros((1, 4), dtype=np.int64),
    )
    assert [entity.to_dict() for entity in entities] == [
        {
            "label": "PERSON",
            "score": pytest.approx(0.99954, abs=1e-4),
            "start": 0,
            "end": 12,
            "text": "Alice Nguyen",
        }
    ]


def test_from_pretrained_downloads_only_runtime_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifact_dir = _write_artifact(tmp_path)
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(artifact_dir)

    _install_dependencies(monkeypatch, snapshot_download)

    model = inference.OnnxModel.from_pretrained(
        "OpenMed/example-v1-onnx-android",
        variant="fp32",
        revision="release",
        token="read-token",
        local_files_only=True,
    )

    assert model.variant == "fp32"
    assert calls[0]["repo_id"] == "OpenMed/example-v1-onnx-android"
    assert calls[0]["revision"] == "release"
    assert calls[0]["token"] == "read-token"
    assert calls[0]["local_files_only"] is True
    assert "model.onnx" in calls[0]["allow_patterns"]
    assert "model_fp16.onnx" not in calls[0]["allow_patterns"]


def test_direct_onnx_path_uses_sibling_metadata(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = _write_artifact(tmp_path)
    _install_dependencies(monkeypatch, lambda **kwargs: str(artifact_dir))

    model = inference.load_onnx_model(artifact_dir / "model.onnx")

    assert model.variant == "fp32"
    assert model.artifact_dir == artifact_dir


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_predict_rejects_invalid_threshold(
    tmp_path: Path,
    monkeypatch,
    threshold: float,
) -> None:
    artifact_dir = _write_artifact(tmp_path)
    _install_dependencies(monkeypatch, lambda **kwargs: str(artifact_dir))
    model = inference.OnnxModel.from_pretrained(artifact_dir)

    with pytest.raises(ValueError, match="threshold"):
        model.predict("Alice Nguyen", threshold=threshold)


def test_explicit_missing_variant_is_actionable(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = _write_artifact(tmp_path)
    _install_dependencies(monkeypatch, lambda **kwargs: str(artifact_dir))

    with pytest.raises(FileNotFoundError, match="model_fp16.onnx"):
        inference.OnnxModel.from_pretrained(artifact_dir, variant="fp16")
