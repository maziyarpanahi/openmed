"""Tests for Android ONNX INT8 quantization and recall certification."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from openmed.core.export_provenance import SourceContext


def _quant_module():
    return importlib.import_module("openmed.onnx.quantize_int8")


def _convert_module():
    return importlib.import_module("openmed.onnx.convert")


def test_quantize_dynamic_int8_writes_qint8_arm_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _quant_module()
    source = tmp_path / "model.onnx"
    output = tmp_path / "model_int8.onnx"
    source.write_bytes(b"onnx")
    calls: list[dict[str, Any]] = []

    onnx_mod = types.ModuleType("onnx")
    onnx_mod.load = lambda path, **kwargs: {"path": path}
    onnx_mod.checker = types.SimpleNamespace(check_model=lambda model: None)

    runtime_mod = types.ModuleType("onnxruntime")
    quantization_mod = types.ModuleType("onnxruntime.quantization")
    quantization_mod.QuantType = types.SimpleNamespace(QInt8="qint8")

    def fake_quantize_dynamic(input_model, output_model, **kwargs):
        calls.append(
            {
                "input_model": input_model,
                "output_model": output_model,
                **kwargs,
            }
        )
        Path(output_model).write_bytes(b"int8")

    quantization_mod.quantize_dynamic = fake_quantize_dynamic
    monkeypatch.setitem(sys.modules, "onnx", onnx_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization_mod)

    result = module.quantize_dynamic_int8(source, output)

    assert result == output
    assert output.read_bytes() == b"int8"
    assert calls == [
        {
            "input_model": str(source),
            "output_model": str(output),
            "weight_type": "qint8",
            "op_types_to_quantize": ["MatMul", "Gemm"],
            "use_external_data_format": False,
        }
    ]


def test_quantize_dynamic_int8_preserves_external_data_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _quant_module()
    source = tmp_path / "model.onnx"
    output = tmp_path / "model_int8.onnx"
    source.write_bytes(b"onnx")
    calls = []

    quantization_mod = types.ModuleType("onnxruntime.quantization")
    quantization_mod.QuantType = types.SimpleNamespace(QInt8="qint8")

    def fake_quantize_dynamic(input_model, output_model, **kwargs):
        calls.append(kwargs)
        Path(output_model).write_bytes(b"int8")

    quantization_mod.quantize_dynamic = fake_quantize_dynamic
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quantization_mod)
    monkeypatch.setattr(module, "_uses_external_data", lambda path: True)
    monkeypatch.setattr(module, "_repair_onnx_topological_order", lambda path: None)
    monkeypatch.setattr(module, "_check_onnx_model", lambda path: None)

    module.quantize_dynamic_int8(source, output)

    assert calls[0]["use_external_data_format"] is True


def test_topological_repair_accounts_for_if_branch_captures() -> None:
    module = _quant_module()
    fake_onnx = types.SimpleNamespace(
        AttributeProto=types.SimpleNamespace(GRAPH=1, GRAPHS=2)
    )
    branch = _FakeGraph(
        [
            _FakeNode(
                "branch_identity",
                inputs=["cast_out"],
                outputs=["branch_out"],
            )
        ]
    )
    graph = _FakeGraph(
        [
            _FakeNode(
                "if_node",
                inputs=["condition"],
                outputs=["if_out"],
                attributes=[_FakeAttribute(fake_onnx.AttributeProto.GRAPH, branch)],
            ),
            _FakeNode("cast_node", inputs=["input_ids"], outputs=["cast_out"]),
        ]
    )

    changed = module._sort_onnx_graph(graph, fake_onnx)

    assert changed is True
    assert [node.name for node in graph.node] == ["cast_node", "if_node"]


def test_topological_repair_detects_reordered_unnamed_nodes() -> None:
    module = _quant_module()
    fake_onnx = types.SimpleNamespace(
        AttributeProto=types.SimpleNamespace(GRAPH=1, GRAPHS=2)
    )
    graph = _FakeGraph(
        [
            _FakeNode("", inputs=["cast_out"], outputs=["result"]),
            _FakeNode("", inputs=["input_ids"], outputs=["cast_out"]),
        ]
    )

    changed = module._sort_onnx_graph(graph, fake_onnx)

    assert changed is True
    assert [node.output for node in graph.node] == [["cast_out"], ["result"]]


def test_int8_report_certifies_and_omits_record_text(tmp_path: Path) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = module.write_int8_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        parent_runner=_perfect_runner,
        candidate_runner=_perfect_runner,
    )

    assert report["format"] == "onnx-int8"
    assert report["certified"] is True
    assert report["quant_recall_delta"] == 0.0
    assert report["per_label"]["PERSON"]["fp_recall"] == 1.0
    assert report["per_label"]["PERSON"]["int8_recall"] == 1.0
    assert report["per_label"]["PERSON"]["gold_span_count"] == 1

    raw_report = (artifact / "recall_delta.json").read_text(encoding="utf-8")
    assert "John Doe" not in raw_report
    assert "Patient John Doe arrived today." not in raw_report
    _assert_no_record_text_fields(json.loads(raw_report))

    manifest = json.loads((artifact / "openmed-onnx.json").read_text())
    config = json.loads((artifact / "config.json").read_text())
    int8_artifact = next(
        item for item in manifest["artifacts"] if item["path"] == "model_int8.onnx"
    )
    assert manifest["formats"] == ["onnx-android", "onnx-int8"]
    assert manifest["certified"] is True
    assert manifest["quant_recall_delta"] == 0.0
    assert int8_artifact["format"] == "onnx-int8"
    assert int8_artifact["metadata"]["quant_recall_delta"] == 0.0
    assert int8_artifact["metadata"]["format"] == "onnx-int8"
    assert config["_onnx_quantization"]["format"] == "onnx-int8"
    assert config["certified"] is True


def test_int8_report_marks_over_budget_delta_uncertified(tmp_path: Path) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")

    report = module.write_int8_recall_delta_report(
        source_model_id="OpenMed/stub-token-classifier",
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        parent_runner=_perfect_runner,
        candidate_runner=_miss_runner,
    )

    assert report["certified"] is False
    assert report["quant_recall_delta"] == 1.0
    assert report["delta"]["blocking_format"] == "onnx-int8"

    manifest = json.loads((artifact / "openmed-onnx.json").read_text())
    assert manifest["certified"] is False
    assert manifest["quantization"]["certified"] is False


def test_int8_certification_refreshes_existing_provenance(tmp_path: Path) -> None:
    from openmed.core.export_provenance import (
        SourceContext,
        verify_export_provenance,
        write_canonical_manifest,
    )

    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    manifest_path = artifact / "openmed-onnx.json"
    source = SourceContext("OpenMed/synthetic", "unused", revision="a" * 40)
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    module.apply_int8_recall_certification(
        artifact,
        {
            "limit": 0.005,
            "certified": True,
            "format": "onnx-int8",
            "quant_recall_delta": 0.0,
            "metric": "character_recall",
        },
        report_relpath="recall_delta.json",
    )

    provenance = verify_export_provenance(artifact, manifest_path.name)
    assert provenance["source"]["revision"] == source.revision


def test_int8_report_uses_private_reference_and_separates_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmed.core.export_provenance import (
        resolve_source_context,
        verify_export_provenance,
        write_canonical_manifest,
    )

    module = _quant_module()
    checkpoint = tmp_path / "synthetic-user" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text('{"model_type": "bert"}')
    source = resolve_source_context(str(checkpoint))
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    manifest_path = artifact / "openmed-onnx.json"
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    observation_path = artifact.with_name("artifact_onnx_observations.json")
    observation_path.write_text('{"validation": {"latency": {"p50": 1.25}}}')
    loads = []
    monkeypatch.setattr(
        module,
        "_hf_token_classification_runner",
        lambda model_id, **kwargs: loads.append(model_id) or _perfect_runner,
    )
    outputs = []
    for clock in ("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"):
        monkeypatch.setattr(module, "_utc_now", lambda value=clock: value)
        payload = module.write_int8_recall_delta_report(
            source_model_id=source.source_model_id,
            source_context=source,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            candidate_runner=_perfect_runner,
        )
        assert payload["generated_at"] == clock
        stable = json.loads((artifact / "recall_delta.json").read_text())
        assert "generated_at" not in stable
        assert stable["delta"] == payload["delta"]
        assert stable["per_label"] == payload["per_label"]
        assert str(checkpoint) not in json.dumps(stable)
        observations = json.loads(observation_path.read_text())
        assert observations["validation"]["latency"]["p50"] == 1.25
        assert observations["int8_recall"]["generated_at"] == clock
        verify_export_provenance(artifact, manifest_path.name)
        outputs.append(
            (manifest_path.read_bytes(), (artifact / "recall_delta.json").read_bytes())
        )
    assert loads == [source.load_reference, source.load_reference]
    assert outputs[0] == outputs[1]


def test_int8_report_requires_real_local_load_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    monkeypatch.setattr(
        module,
        "_hf_token_classification_runner",
        lambda *args, **kwargs: pytest.fail("local digest must never reach loader"),
    )

    with pytest.raises(ValueError, match="load reference"):
        module.write_int8_recall_delta_report(
            source_model_id="local:sha256:" + "a" * 64,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            candidate_runner=_perfect_runner,
        )


def test_standalone_int8_report_pins_model_and_tokenizer_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmed.core.export_provenance import write_canonical_manifest

    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    source = SourceContext("OpenMed/synthetic", "OpenMed/synthetic", revision="a" * 40)
    manifest_path = artifact / "openmed-onnx.json"
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    loads = []

    def load_tokenizer(model_id, **kwargs):
        loads.append(("tokenizer", model_id, kwargs))
        return object()

    def load_model(model_id, **kwargs):
        loads.append(("model", model_id, kwargs))
        return object()

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=load_tokenizer)
    transformers.AutoModelForTokenClassification = types.SimpleNamespace(
        from_pretrained=load_model
    )
    transformers.pipeline = lambda *args, **kwargs: (
        lambda text: [{"entity_group": "PERSON", "start": 8, "end": 16, "score": 0.99}]
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    report = module.write_int8_recall_delta_report(
        source_model_id=source.source_model_id,
        artifact_dir=artifact,
        eval_suite_path=fixture_path,
        candidate_runner=_perfect_runner,
    )

    assert report["certified"] is True
    assert loads == [
        (
            "tokenizer",
            source.source_model_id,
            {"cache_dir": None, "revision": source.revision},
        ),
        (
            "model",
            source.source_model_id,
            {"cache_dir": None, "revision": source.revision},
        ),
    ]


@pytest.mark.parametrize("parent_component", [False, True])
def test_int8_external_report_has_an_inventoried_stable_copy(
    tmp_path: Path, parent_component: bool
) -> None:
    from openmed.core.export_provenance import (
        verify_export_provenance,
        write_canonical_manifest,
    )

    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    source = SourceContext("OpenMed/synthetic", "unused", revision="a" * 40)
    manifest_path = artifact / "openmed-onnx.json"
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    external_path = (
        artifact / ".." / "reports" / "custom-recall.json"
        if parent_component
        else tmp_path / "reports" / "custom-recall.json"
    )

    for runner in (_perfect_runner, _miss_runner):
        report = module.write_int8_recall_delta_report(
            source_model_id=source.source_model_id,
            source_context=source,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=external_path,
            parent_runner=_perfect_runner,
            candidate_runner=runner,
        )

        assert json.loads(external_path.read_text()) == report
        stable_path = artifact / report["report_path"]
        stable = json.loads(stable_path.read_text())
        assert "generated_at" not in stable
        assert stable["delta"] == report["delta"]
        assert stable["certified"] is (runner is _perfect_runner)
        manifest = json.loads(manifest_path.read_text())
        assert manifest["recall_delta_path"] == stable_path.name
        provenance = verify_export_provenance(artifact, manifest_path.name)
        assert any(item["path"] == stable_path.name for item in provenance["files"])


@pytest.mark.parametrize(
    "filename",
    ["config.json", "tokenizer.json", "openmed-onnx.json", "model_int8.onnx"],
)
def test_external_int8_report_rejects_runtime_file_collisions(
    tmp_path: Path, filename: str
) -> None:
    from openmed.core.export_provenance import write_canonical_manifest

    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    (artifact / "tokenizer.json").write_text('{"synthetic": true}')
    manifest_path = artifact / "openmed-onnx.json"
    manifest = json.loads(manifest_path.read_text())
    # A stale or malformed report declaration must not turn a tokenizer into a
    # certification report that can be overwritten.
    manifest["recall_delta_path"] = "tokenizer.json"
    source = SourceContext("OpenMed/synthetic", "unused", revision="a" * 40)
    write_canonical_manifest(
        artifact, manifest_path.name, manifest, source=source, backend="onnx"
    )
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}
    external_path = tmp_path / "reports" / filename

    with pytest.raises(ValueError, match="report.*(overwrite|collision)"):
        module.write_int8_recall_delta_report(
            source_model_id=source.source_model_id,
            source_context=source,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=external_path,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )

    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before
    assert not external_path.exists()
    assert not artifact.with_name("artifact_onnx_observations.json").exists()


@pytest.mark.parametrize("change_when", ["before", "during"])
def test_standalone_int8_report_rejects_changed_local_source(
    tmp_path: Path, change_when: str
) -> None:
    from openmed.core.export_provenance import (
        resolve_source_context,
        write_canonical_manifest,
    )

    module = _quant_module()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text('{"model_type": "bert"}')
    source = resolve_source_context(str(checkpoint))
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    manifest_path = artifact / "openmed-onnx.json"
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    if change_when == "before":
        config_path.write_text('{"model_type": "changed"}')

    def parent_runner(fixture, model_name, device):
        if change_when == "during":
            config_path.write_text('{"model_type": "changed"}')
        return _perfect_runner(fixture, model_name, device)

    before = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="local source changed"):
        module.write_int8_recall_delta_report(
            source_model_id=str(checkpoint),
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            parent_runner=parent_runner,
            candidate_runner=_perfect_runner,
        )

    assert manifest_path.read_bytes() == before
    assert not (artifact / "recall_delta.json").exists()
    assert not artifact.with_name("artifact_onnx_observations.json").exists()


def test_int8_report_supports_current_directory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openmed.core.export_provenance import (
        verify_export_provenance,
        write_canonical_manifest,
    )

    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    source = SourceContext("OpenMed/synthetic", "unused", revision="a" * 40)
    manifest_path = artifact / "openmed-onnx.json"
    write_canonical_manifest(
        artifact,
        manifest_path.name,
        json.loads(manifest_path.read_text()),
        source=source,
        backend="onnx",
    )
    monkeypatch.chdir(artifact)

    report = module.write_int8_recall_delta_report(
        source_model_id=source.source_model_id,
        source_context=source,
        artifact_dir=".",
        eval_suite_path=fixture_path,
        parent_runner=_perfect_runner,
        candidate_runner=_perfect_runner,
    )

    assert report["report_path"] == "recall_delta.json"
    assert (tmp_path / "artifact_onnx_observations.json").is_file()
    verify_export_provenance(".", manifest_path.name)


def test_int8_report_cannot_overwrite_managed_observations(tmp_path: Path) -> None:
    module = _quant_module()
    artifact = _write_stub_onnx_artifact(tmp_path / "artifact")
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    source = SourceContext("OpenMed/synthetic", "unused", revision="a" * 40)
    observations = tmp_path / "artifact_onnx_observations.json"
    original = '{"validation": {"latency": {"p50": 1.25}}}'
    observations.write_text(original)
    before = {path.name: path.read_bytes() for path in artifact.iterdir()}

    with pytest.raises(ValueError, match="report.*collision"):
        module.write_int8_recall_delta_report(
            source_model_id=source.source_model_id,
            source_context=source,
            artifact_dir=artifact,
            eval_suite_path=fixture_path,
            output_path=observations,
            parent_runner=_perfect_runner,
            candidate_runner=_perfect_runner,
        )

    assert observations.read_text() == original
    assert {path.name: path.read_bytes() for path in artifact.iterdir()} == before


def test_convert_android_profile_emits_int8_and_runs_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _convert_module()
    monkeypatch.setattr(
        module,
        "resolve_source_context",
        lambda model_id, **kwargs: SourceContext(model_id, model_id, revision="a" * 40),
    )
    fixture_path = _write_fixture(tmp_path / "fixtures.json")
    report_path = tmp_path / "reports" / "recall_delta.json"
    report_calls: list[dict[str, Any]] = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"onnx")
        return path

    def fake_export_android_fp16(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"fp16")
        return path

    def fake_quantize_dynamic_int8(onnx_path, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"int8")
        return path

    def fake_validate_android_profile(model_path, **kwargs):
        return types.SimpleNamespace(
            to_metadata=lambda: {
                "profile": "android",
                "opset": module.ANDROID_ONNX_OPSET,
                "inputs": [],
                "outputs": [],
                "unsupported_ops": [],
                "warnings": [],
            }
        )

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        output_dir = Path(output_dir)
        config = {
            "model_type": "bert",
            "id2label": {"0": "O", "1": "B-NAME"},
            "max_sequence_length": kwargs["max_seq_length"],
        }
        (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (output_dir / "id2label.json").write_text(
            json.dumps(config["id2label"]),
            encoding="utf-8",
        )
        return config, []

    def fake_write_int8_recall_delta_report(**kwargs):
        report_calls.append(kwargs)
        return {
            "format": "onnx-int8",
            "limit": 0.005,
            "certified": True,
            "quant_recall_delta": 0.0,
            "metric": "character_recall",
            "report_path": "recall_delta.json",
        }

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "export_android_fp16", fake_export_android_fp16)
    monkeypatch.setattr(module, "quantize_dynamic_int8", fake_quantize_dynamic_int8)
    monkeypatch.setattr(
        module, "validate_android_profile", fake_validate_android_profile
    )
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(
        module,
        "convert_android_onnx_to_ort",
        lambda *args, **kwargs: types.SimpleNamespace(
            skipped=True,
            ort_path=None,
            skip_reason="tooling unavailable",
        ),
    )
    monkeypatch.setattr(
        module,
        "write_int8_recall_delta_report",
        fake_write_int8_recall_delta_report,
    )

    result = module.convert(
        "OpenMed/test-model",
        tmp_path / "artifact",
        profile="android",
        eval_suite_path=fixture_path,
        recall_delta_report_path=report_path,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.formats == ["onnx-android", "onnx-int8"]
    assert [item["path"] for item in manifest["artifacts"]] == [
        "model.onnx",
        "model_fp16.onnx",
        "model_int8.onnx",
    ]
    assert manifest["artifacts"][2]["format"] == "onnx-int8"
    assert manifest["artifacts"][2]["precision"] == "int8"
    assert manifest["artifacts"][2]["metadata"]["certified"] is True
    assert report_calls[0]["eval_suite_path"] == fixture_path
    assert report_calls[0]["output_path"] == report_path
    assert report_calls[0]["int8_model_path"] == result.output_dir / "model_int8.onnx"


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


class _FakeValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeSparseInitializer:
    def __init__(self, name: str) -> None:
        self.values = _FakeValue(name)


class _FakeAttribute:
    def __init__(
        self,
        attribute_type: int,
        graph: "_FakeGraph | None" = None,
        graphs: list["_FakeGraph"] | None = None,
    ) -> None:
        self.type = attribute_type
        self.g = graph
        self.graphs = graphs or []


class _FakeNode:
    def __init__(
        self,
        name: str,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        attributes: list[_FakeAttribute] | None = None,
    ) -> None:
        self.name = name
        self.input = inputs or []
        self.output = outputs or []
        self.attribute = attributes or []

    def SerializeToString(self) -> bytes:
        return repr((self.name, self.input, self.output)).encode()


class _FakeGraph:
    def __init__(
        self,
        nodes: list[_FakeNode],
        *,
        inputs: list[str] | None = None,
        initializers: list[str] | None = None,
        sparse_initializers: list[str] | None = None,
    ) -> None:
        self.node = nodes
        self.input = [_FakeValue(name) for name in inputs or []]
        self.initializer = [_FakeValue(name) for name in initializers or []]
        self.sparse_initializer = [
            _FakeSparseInitializer(name) for name in sparse_initializers or []
        ]


def _write_stub_onnx_artifact(path: Path) -> Path:
    path.mkdir()
    (path / "model.onnx").write_bytes(b"onnx")
    (path / "model_int8.onnx").write_bytes(b"int8")
    config = {
        "model_type": "bert",
        "id2label": {"0": "O", "1": "PERSON"},
        "max_sequence_length": 128,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "id2label.json").write_text(
        json.dumps(config["id2label"]),
        encoding="utf-8",
    )
    (path / "openmed-onnx.json").write_text(
        json.dumps(
            {
                "format": "openmed-onnx",
                "format_version": 1,
                "formats": ["onnx-android"],
                "artifacts": [
                    {
                        "format": "onnx-android",
                        "path": "model.onnx",
                        "precision": "float32",
                    }
                ],
            }
        ),
        encoding="utf-8",
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


def _assert_no_record_text_fields(value: Any) -> None:
    forbidden = {
        "text",
        "note_text",
        "record_text",
        "source_text",
        "span_text",
        "word",
        "predicted_spans",
        "gold_spans",
        "fixture_ids",
        "record_id",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for item in value.values():
            _assert_no_record_text_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_record_text_fields(item)
