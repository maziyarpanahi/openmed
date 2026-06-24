from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

from openmed.mlx.artifact import (
    load_artifact_config,
    resolve_weight_candidates,
    write_manifest,
)


def _write_fixture(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "id": "stub-note",
                        "text": "Patient John Doe arrived today.",
                        "gold_spans": [
                            {
                                "start": 8,
                                "end": 16,
                                "label": "PERSON",
                                "text": "John Doe",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_stub_int4_artifact(path: Path) -> Path:
    path.mkdir()
    (path / "weights.npz").write_bytes(b"stub")
    config = {
        "num_labels": 2,
        "id2label": {"0": "O", "1": "PERSON"},
        "_mlx_task": "token-classification",
        "_mlx_model_type": "bert",
        "_mlx_weights_format": "npz",
        "_mlx_quantization": {
            "bits": 4,
            "format": "mlx-4bit",
            "group_size": 32,
        },
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    write_manifest(
        path,
        source_model_id="OpenMed/stub-token-classifier",
        config=config,
        tokenizer_files=[],
    )
    return path


def _perfect_runner(fixture: Any, model_name: str, device: str) -> list[dict[str, Any]]:
    del model_name, device
    start = fixture.text.index("John Doe")
    return [
        {
            "entity_group": "PERSON",
            "score": 0.99,
            "start": start,
            "end": start + len("John Doe"),
            "word": "John Doe",
        }
    ]


def _miss_runner(fixture: Any, model_name: str, device: str) -> list[dict[str, Any]]:
    del fixture, model_name, device
    return []


def test_int4_report_certifies_artifact_within_recall_budget(tmp_path: Path) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = write_quant_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        quantize_group_size=32,
        parent_runner=_perfect_runner,
        candidate_runner=_perfect_runner,
    )

    assert report["certified"] is True
    assert report["quant_recall_delta"] == 0.0
    assert report["per_label"]["PERSON"]["fp_recall"] == 1.0
    assert report["per_label"]["PERSON"]["int4_recall"] == 1.0
    assert (artifact / "recall_delta.json").exists()

    manifest, config = load_artifact_config(artifact)
    assert manifest is not None
    assert manifest["formats"] == ["mlx-4bit"]
    assert manifest["quant_recall_delta"] == 0.0
    assert manifest["certified"] is True
    assert manifest["quantization"]["group_size"] == 32
    assert config["quant_recall_delta"] == 0.0
    assert config["certified"] is True
    assert resolve_weight_candidates(artifact, config, manifest)[0].exists()


def test_int4_report_marks_over_threshold_delta_uncertified(
    tmp_path: Path,
) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = write_quant_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        quantize_group_size=32,
        parent_runner=_perfect_runner,
        candidate_runner=_miss_runner,
    )

    assert report["certified"] is False
    assert report["quant_recall_delta"] == 1.0
    assert report["delta"]["blocking_format"] == "mlx-4bit"

    manifest, config = load_artifact_config(artifact)
    assert manifest is not None
    assert manifest["certified"] is False
    assert manifest["quantization"]["certified"] is False
    assert config["_mlx_quantization"]["certified"] is False


def test_save_mlx_model_records_int4_group_size_and_loadable_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openmed.mlx import convert as convert_module

    quantize_calls: list[dict[str, int]] = []

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_nn = types.ModuleType("mlx.nn")
    fake_utils = types.ModuleType("mlx.utils")

    fake_core.array = lambda value: value

    def save_safetensors(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("force npz fallback")

    def savez(path: str, **weights: Any) -> None:
        del weights
        Path(path).write_bytes(b"stub-npz")

    fake_core.save_safetensors = save_safetensors
    fake_core.savez = savez

    class FakeModel:
        def __init__(self) -> None:
            self._params: dict[str, Any] = {}

        def load_weights(self, weights: list[tuple[str, Any]]) -> None:
            self._params = dict(weights)

        def parameters(self) -> dict[str, Any]:
            return {"classifier.weight": self._params["classifier.weight"]}

    def quantize(model: FakeModel, **kwargs: int) -> None:
        del model
        quantize_calls.append(kwargs)

    fake_nn.quantize = quantize
    fake_utils.tree_flatten = lambda params: list(params.items())

    fake_mlx.core = fake_core
    fake_mlx.nn = fake_nn
    fake_mlx.utils = fake_utils
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(sys.modules, "mlx.nn", fake_nn)
    monkeypatch.setitem(sys.modules, "mlx.utils", fake_utils)

    class FakeTokenizer:
        def save_pretrained(self, output_dir: str | Path) -> None:
            Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(convert_module, "build_model", lambda config: FakeModel())

    output = convert_module.save_mlx_model(
        weights={"classifier.weight": [[0.0, 1.0]]},
        config={"num_labels": 2, "_mlx_task": "token-classification"},
        output_dir=tmp_path / "artifact",
        quantize_bits=4,
        quantize_group_size=32,
        source_model_id="OpenMed/stub-token-classifier",
    )

    manifest, config = load_artifact_config(output)
    assert manifest is not None
    assert quantize_calls == [{"bits": 4, "group_size": 32}]
    assert manifest["formats"] == ["mlx-4bit"]
    assert manifest["quantization"]["bits"] == 4
    assert manifest["quantization"]["group_size"] == 32
    assert config["_mlx_quantization"]["format"] == "mlx-4bit"
    assert resolve_weight_candidates(output, config, manifest)[0].exists()
