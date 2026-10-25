from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

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


def _write_boundary_fixture(path: Path) -> Path:
    text = "A" * 100
    path.write_text(
        json.dumps(
            {
                "fixtures": [
                    {
                        "id": "boundary-note",
                        "text": text,
                        "gold_spans": [
                            {
                                "start": 0,
                                "end": len(text),
                                "label": "PERSON",
                                "text": text,
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
    span = fixture.gold_spans[0]
    return [
        {
            "entity_group": span.label,
            "score": 0.99,
            "start": span.start,
            "end": span.end,
            "word": fixture.text[span.start : span.end],
        }
    ]


def _miss_runner(fixture: Any, model_name: str, device: str) -> list[dict[str, Any]]:
    del fixture, model_name, device
    return []


def _one_point_loss_runner(
    fixture: Any, model_name: str, device: str
) -> list[dict[str, Any]]:
    del model_name, device
    return [
        {
            "entity_group": "PERSON",
            "score": 0.99,
            "start": 0,
            "end": len(fixture.text) - 1,
            "word": fixture.text[:-1],
        }
    ]


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


def test_int4_report_marks_exact_threshold_delta_uncertified(
    tmp_path: Path,
) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_boundary_fixture(tmp_path / "fixtures.json")

    report = write_quant_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        quantize_group_size=32,
        parent_runner=_perfect_runner,
        candidate_runner=_one_point_loss_runner,
    )

    assert report["certified"] is False
    assert report["quant_recall_delta"] == pytest.approx(0.010)
    assert report["delta"]["offending_labels"]["PERSON"]["limit"] == 0.010


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


def test_external_recall_delta_report_path_does_not_leak_absolute_paths(
    tmp_path: Path,
) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    report_path = tmp_path / "reports" / "recall_delta.json"

    report = write_quant_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        output_path=report_path,
        quantize_group_size=32,
        parent_runner=_perfect_runner,
        candidate_runner=_perfect_runner,
    )

    manifest, config = load_artifact_config(artifact)
    assert manifest is not None
    assert report["report_path"] == "recall_delta.json"
    assert manifest["recall_delta_path"] == "recall_delta.json"
    assert config["_mlx_quantization"]["recall_delta_path"] == "recall_delta.json"
    assert str(tmp_path) not in report["report_path"]
    assert str(tmp_path) not in manifest["certification"]["report_path"]
    assert (artifact / "recall_delta.json").exists()


@pytest.mark.parametrize(
    "filename",
    ["config.json", "openmed-mlx.json", "tokenizer.json", "vocab.txt", "weights.npz"],
)
def test_external_quant_report_rejects_runtime_projection_collisions(
    tmp_path: Path, filename: str
) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    (artifact / "tokenizer.json").write_text("{}")
    (artifact / "vocab.txt").write_text("synthetic vocab")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}
    external = tmp_path / "reports" / filename
    with pytest.raises(ValueError, match="report destination"):
        write_quant_recall_delta_report(
            source_model_id="OpenMed/stub-token-classifier",
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=external,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )
    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before
    assert not external.exists()
    assert not (tmp_path / "artifact_mlx_observations.json").exists()


def test_external_custom_quant_report_can_update_its_declared_report(
    tmp_path: Path,
) -> None:
    from openmed.core.export_provenance import verify_export_provenance
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    external = tmp_path / "reports" / "custom-recall.json"
    for _ in range(2):
        write_quant_recall_delta_report(
            source_model_id="OpenMed/stub-token-classifier",
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=external,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )
        verify_export_provenance(artifact, "openmed-mlx.json")
    assert external.read_bytes() == (artifact / "custom-recall.json").read_bytes()
    assert json.loads((artifact / "config.json").read_text())["num_labels"] == 2


def test_report_update_rejects_partial_report_shaped_json(tmp_path: Path) -> None:
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    target = artifact / "custom.json"
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "metric": "character_recall",
                "report_path": "custom.json",
                "certified": True,
                "quantization": {},
                "delta": {},
            }
        )
    )
    before = target.read_bytes()
    with pytest.raises(ValueError, match="report destination"):
        write_quant_recall_delta_report(
            source_model_id="OpenMed/stub-token-classifier",
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=target,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )
    assert target.read_bytes() == before


@pytest.mark.parametrize("mutation", ["before", "during"])
def test_restored_local_quant_source_checks_raw_reference_before_and_after_eval(
    tmp_path: Path, mutation: str
) -> None:
    from openmed.core.export_provenance import resolve_source_context
    from openmed.mlx.convert import write_quant_recall_delta_report

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight_path = checkpoint / "weights.bin"
    weight_path.write_bytes(b"AA")
    source = resolve_source_context(str(checkpoint))
    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    config = json.loads((artifact / "config.json").read_text())
    write_manifest(
        artifact, source_model_id=source.source_model_id, config=config, source=source
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    called = []

    def parent(fixture, model_name, device):
        called.append(True)
        if mutation == "during":
            weight_path.write_bytes(b"BB")
        return _perfect_runner(fixture, model_name, device)

    if mutation == "before":
        weight_path.write_bytes(b"BB")
    with pytest.raises(ValueError, match="local source changed during conversion"):
        write_quant_recall_delta_report(
            source_model_id=str(checkpoint),
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            parent_runner=parent,
            candidate_runner=_perfect_runner,
        )
    if mutation == "before":
        assert not called
    assert not (artifact / "recall_delta.json").exists()


@pytest.mark.parametrize("custom_runner", [False, True])
def test_restored_local_quant_source_without_live_reference_is_explicit(
    tmp_path: Path, custom_runner: bool
) -> None:
    from openmed.core.export_provenance import resolve_source_context
    from openmed.mlx.convert import write_quant_recall_delta_report

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"AA")
    source = resolve_source_context(str(checkpoint))
    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    config = json.loads((artifact / "config.json").read_text())
    write_manifest(
        artifact, source_model_id=source.source_model_id, config=config, source=source
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    kwargs = dict(
        source_model_id=source.source_model_id,
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        candidate_runner=_perfect_runner,
    )
    if custom_runner:
        report = write_quant_recall_delta_report(
            parent_runner=_perfect_runner, **kwargs
        )
        assert report["local_source_verification"] == "caller-managed"
    else:
        with pytest.raises(ValueError, match="local source reference is required"):
            write_quant_recall_delta_report(**kwargs)
        assert not (artifact / "recall_delta.json").exists()


def test_relative_current_directory_keeps_observations_outside_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    from openmed.core.export_provenance import verify_export_provenance
    from openmed.mlx.convert import write_quant_recall_delta_report

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    monkeypatch.chdir(artifact)
    manifests = []
    for _ in range(2):
        write_quant_recall_delta_report(
            source_model_id="OpenMed/stub-token-classifier",
            artifact_dir=".",
            eval_suite_path=fixture_path,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )
        verify_export_provenance(artifact, "openmed-mlx.json")
        manifests.append((artifact / "openmed-mlx.json").read_bytes())
    assert manifests[0] == manifests[1]
    assert (tmp_path / "artifact_mlx_observations.json").exists()
    assert not (artifact / "_mlx_observations.json").exists()


@pytest.mark.parametrize("report_name", ["recall_delta.json", "custom-recall.json"])
def test_reused_output_conversion_and_mock_certification_remain_canonical(
    tmp_path: Path, monkeypatch, report_name: str
) -> None:
    from openmed.core.export_provenance import verify_export_provenance
    from openmed.mlx import convert as convert_module

    np = pytest.importorskip("numpy")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"AA")
    fake_transformers = types.ModuleType("transformers")

    class Tokenizer:
        def save_pretrained(self, output_dir):
            Path(output_dir, "tokenizer.json").write_text("{}")

    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: Tokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setattr(
        convert_module,
        "convert_weights",
        lambda *args, **kwargs: (
            {"weight": np.array([1.0])},
            {"num_labels": 2, "model_type": "bert"},
        ),
    )
    artifact = tmp_path / "artifact"
    output_path = (
        None
        if report_name == "recall_delta.json"
        else tmp_path / "reports" / report_name
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    manifests = []
    for _ in range(2):
        convert_module.convert(str(checkpoint), artifact)
        convert_module.write_quant_recall_delta_report(
            source_model_id=str(checkpoint),
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=output_path,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )
        verify_export_provenance(artifact, "openmed-mlx.json")
        manifests.append((artifact / "openmed-mlx.json").read_bytes())
    assert manifests[0] == manifests[1]
    assert json.loads(manifests[1])["tokenizer"]["files"] == ["tokenizer.json"]


def test_quant_report_keeps_observations_outside_deterministic_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    from openmed.core.export_provenance import (
        resolve_source_context,
        verify_export_provenance,
    )
    from openmed.mlx import convert as convert_module

    checkpoint = tmp_path / "SYNTHETIC_USER" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type":"bert"}')
    (checkpoint / "weights.bin").write_bytes(b"synthetic checkpoint")
    source = resolve_source_context(str(checkpoint))
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    timestamps = iter(("2026-10-25T12:00:00Z", "2026-10-25T13:00:00Z"))
    monkeypatch.setattr(convert_module, "_utc_now", lambda: next(timestamps))
    loaded = []

    def parent_runner(model_id, *, cache_dir=None, revision=None):
        loaded.append(model_id)
        return _perfect_runner

    monkeypatch.setattr(
        convert_module, "_hf_token_classification_runner", parent_runner
    )
    outputs = []
    for name in ("first", "second"):
        artifact = _write_stub_int4_artifact(tmp_path / name)
        config = json.loads((artifact / "config.json").read_text())
        write_manifest(
            artifact,
            source_model_id=source.source_model_id,
            config=config,
            source=source,
        )
        convert_module.write_quant_recall_delta_report(
            source_model_id=source.source_model_id,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            source=source,
            candidate_runner=_perfect_runner,
        )
        provenance = verify_export_provenance(artifact, "openmed-mlx.json")
        assert provenance["source"]["content_digest"] == source.source_digest
        assert "recall_delta.json" in {item["path"] for item in provenance["files"]}
        outputs.append(artifact)

    first, second = outputs
    assert loaded == [str(checkpoint)] * 2
    for name in ("recall_delta.json", "openmed-mlx.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    report = json.loads((first / "recall_delta.json").read_text())
    assert "generated_at" not in report
    assert report["artifact_path"] == "."
    assert report["eval_suite_path"].startswith("sha256:")
    first_observations = json.loads(
        (tmp_path / "first_mlx_observations.json").read_text()
    )
    second_observations = json.loads(
        (tmp_path / "second_mlx_observations.json").read_text()
    )
    assert first_observations["generated_at"] == "2026-10-25T12:00:00Z"
    assert second_observations["generated_at"] == "2026-10-25T13:00:00Z"
    assert first_observations["quant_recall_delta"] == report["quant_recall_delta"]
    assert "SYNTHETIC_USER" not in json.dumps(first_observations)
    assert str(tmp_path) not in json.dumps(first_observations)


def test_standalone_quant_report_pins_saved_revision_for_real_loaders(
    tmp_path: Path, monkeypatch
) -> None:
    from openmed.core.export_provenance import SourceContext
    from openmed.mlx import convert as convert_module

    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    config = json.loads((artifact / "config.json").read_text())
    revision = "c" * 40
    source = SourceContext(
        source_model_id="fixture/tiny-bert",
        load_reference="fixture/tiny-bert",
        revision=revision,
    )
    write_manifest(
        artifact, source_model_id=source.source_model_id, config=config, source=source
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    loaded = []

    def tokenizer_loader(model_id, **kwargs):
        loaded.append(("tokenizer", model_id, kwargs.get("revision")))
        return object()

    def model_loader(model_id, **kwargs):
        loaded.append(("model", model_id, kwargs.get("revision")))
        return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer_loader)
    transformers.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=model_loader
    )
    transformers.pipeline = lambda *args, **kwargs: (
        lambda text: [
            {"entity_group": "PERSON", "start": 8, "end": 16, "word": text[8:16]}
        ]
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    convert_module.write_quant_recall_delta_report(
        source_model_id="fixture/tiny-bert",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        candidate_runner=_perfect_runner,
    )
    assert loaded == [
        ("tokenizer", "fixture/tiny-bert", revision),
        ("model", "fixture/tiny-bert", revision),
    ]


def test_quant_report_rejects_source_changed_during_evaluation(tmp_path: Path) -> None:
    from openmed.core.export_provenance import resolve_source_context
    from openmed.mlx.convert import write_quant_recall_delta_report

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    weight_path = checkpoint / "weights.bin"
    weight_path.write_bytes(b"AA")
    source = resolve_source_context(str(checkpoint))
    artifact = _write_stub_int4_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    def modifying_parent(fixture, model_name, device):
        weight_path.write_bytes(b"BB")
        return _perfect_runner(fixture, model_name, device)

    with pytest.raises(ValueError, match="local source changed during conversion"):
        write_quant_recall_delta_report(
            source_model_id=source.source_model_id,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            source=source,
            parent_runner=modifying_parent,
            candidate_runner=_perfect_runner,
        )
    assert not (artifact / "recall_delta.json").exists()


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


def test_convert_runs_quant_recall_certification_for_eval_suite(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from openmed.core.export_provenance import SourceContext
    from openmed.mlx import convert as convert_module

    monkeypatch.setattr(
        convert_module,
        "resolve_source_context",
        lambda model_id, **kwargs: SourceContext(
            source_model_id=model_id,
            load_reference=model_id,
            revision="a" * 40,
        ),
    )

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setattr(
        convert_module,
        "convert_weights",
        lambda model_id, cache_dir=None: (
            {"classifier.weight": [[0.0, 1.0]]},
            {"num_labels": 2, "_mlx_task": "token-classification"},
        ),
    )

    def fake_save_mlx_model(**kwargs: Any) -> Path:
        return _write_stub_int4_artifact(Path(kwargs["output_dir"]))

    report_calls: list[dict[str, Any]] = []

    def fake_write_quant_recall_delta_report(**kwargs: Any) -> dict[str, Any]:
        report_calls.append(kwargs)
        return {"certified": True, "quant_recall_delta": 0.0}

    monkeypatch.setattr(convert_module, "save_mlx_model", fake_save_mlx_model)
    monkeypatch.setattr(
        convert_module,
        "write_quant_recall_delta_report",
        fake_write_quant_recall_delta_report,
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    report_path = tmp_path / "reports" / "recall_delta.json"

    output = convert_module.convert(
        model_id="OpenMed/stub-token-classifier",
        output_dir=tmp_path / "artifact",
        quantize_bits=4,
        eval_suite_path=fixture_path,
        recall_delta_report_path=report_path,
        quantize_group_size=32,
    )

    assert output == tmp_path / "artifact"
    assert len(report_calls) == 1
    assert report_calls[0]["artifact_dir"] == output
    assert report_calls[0]["eval_suite_path"] == fixture_path
    assert report_calls[0]["output_path"] == report_path
    assert report_calls[0]["format_name"] == "mlx-4bit"
    assert report_calls[0]["quantize_bits"] == 4
    assert report_calls[0]["quantize_group_size"] == 32
