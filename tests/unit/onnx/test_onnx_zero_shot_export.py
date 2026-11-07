"""GLiNER span and GLiClass ONNX export, qualification, and runtime tests.

Refusal tests run everywhere. Export tests build tiny GLiNER-shaped and
GLiClass-shaped checkpoints in-test and need PyTorch, Transformers, ONNX,
ONNX Runtime, tokenizers, and safetensors. The MLX parity test also needs
MLX (Apple silicon).
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from openmed.eval import export_qualification as eq
from openmed.eval.export_qualification import ExportQualificationError, ExportReason
from openmed.ner.families import zero_shot_export as zs
from tests.fixtures.modernbert_export_stub import (
    NetworkGuard,
    files_containing_sentinels,
    text_contains_sentinels,
)
from tests.fixtures.zero_shot_export_stub import (
    GLICLASS_LABELS,
    GLINER_LABELS,
    SENTINELS,
    gliclass_config,
    gliclass_fixture_rows,
    gliner_config,
    gliner_fixture_rows,
    write_jsonl,
)


def _write_config(directory: Path, config: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _export_module():
    import importlib

    return importlib.import_module("openmed.onnx.zero_shot_export")


# ---------------------------------------------------------------------------
# Refusals that never load weights (run everywhere)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            gliner_config(_mlx_family="gliner-uni-encoder-token-relex"),
            ExportReason.UNSUPPORTED_TASK,
        ),
        (
            gliner_config(_mlx_task="zero-shot-relation-extraction"),
            ExportReason.UNSUPPORTED_TASK,
        ),
        (
            gliner_config(auto_map={"AutoModel": "modeling.GLiNER"}),
            ExportReason.UNSUPPORTED_REMOTE_CODE,
        ),
        (
            gliner_config(_mlx_quantization={"bits": 4, "group_size": 64}),
            ExportReason.UNSUPPORTED_QUANTIZATION,
        ),
        (
            gliner_config(span_mode="token_level"),
            ExportReason.UNSUPPORTED_MODEL_VARIANT,
        ),
        (
            gliclass_config(class_token_pooling="first"),
            ExportReason.UNSUPPORTED_MODEL_VARIANT,
        ),
        (gliclass_config(scorer_type="dot"), ExportReason.UNSUPPORTED_MODEL_VARIANT),
        ({"model_type": "bert"}, ExportReason.UNSUPPORTED_FAMILY),
    ],
)
def test_unsupported_sources_fail_before_weights_load(
    tmp_path: Path, config: dict[str, Any], reason: ExportReason
) -> None:
    source = _write_config(tmp_path / "source", config)
    # No weights or tokenizer exist: a refusal must come first.
    with pytest.raises(ExportQualificationError) as error:
        _export_module().export_zero_shot_onnx(source, tmp_path / "out")
    assert error.value.reason is reason
    assert not (tmp_path / "out").exists()


def test_sources_must_be_local_artifacts(tmp_path: Path) -> None:
    module = _export_module()
    for source in ("OpenMed/gliner-remote", tmp_path / "missing"):
        with pytest.raises(ExportQualificationError) as error:
            module.export_zero_shot_onnx(source, tmp_path / "out")
        assert error.value.reason is ExportReason.ARTIFACT_MISSING
    only_config = _write_config(tmp_path / "config-only", gliner_config())
    with pytest.raises(ExportQualificationError) as error:
        module.export_zero_shot_onnx(only_config, tmp_path / "out")
    assert error.value.reason is ExportReason.ARTIFACT_MISSING


def test_cli_reports_only_the_reason_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write_config(
        tmp_path / "relex", gliner_config(_mlx_family="gliner-uni-encoder-token-relex")
    )
    code = _export_module().main(["--source", str(source), "--output", "out"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out.strip() == "export refused: unsupported_task"
    assert str(tmp_path) not in captured.out + captured.err


def test_token_classification_onnx_converter_refuses_gliner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    module = importlib.import_module("openmed.onnx.convert")
    calls: list[str] = []

    def fake_export_onnx(model_id, output_path, **kwargs):
        path = Path(output_path)
        path.write_bytes(b"graph")
        return path

    def fake_save_source_assets(model_id, output_dir, **kwargs):
        return dict(gliner_config()), []

    def forbidden(*args, **kwargs):
        calls.append("forbidden")
        raise AssertionError("a zero-shot checkpoint must be refused first")

    monkeypatch.setattr(module, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(module, "save_source_assets", fake_save_source_assets)
    monkeypatch.setattr(module, "export_webgpu", forbidden)
    monkeypatch.setattr(module, "publish_artifact", forbidden)
    with pytest.raises(ExportQualificationError) as error:
        module.convert(
            "OpenMed/stub",
            tmp_path / "out",
            optimize_onnx=False,
            publish_to_hub=True,
        )
    assert error.value.reason is ExportReason.UNSUPPORTED_TASK
    assert calls == []
    assert not (tmp_path / "out" / "openmed-onnx.json").exists()
    # export_onnx applies the same guard to the loaded model configuration.
    with pytest.raises(ExportQualificationError) as error:
        module._reject_zero_shot_checkpoint(SimpleNamespace(**gliclass_config()))
    assert error.value.reason is ExportReason.UNSUPPORTED_TASK


def test_portable_config_drops_path_bearing_fields() -> None:
    config = {
        "_name_or_path": "/home/someone/checkpoints/gliner",
        "_commit_hash": "abc",
        "tokenizer_dir": "C:\\models\\gliner",
        "model_type": "deberta-v2",
        "max_width": 12,
    }
    assert _export_module().portable_config(config) == {
        "model_type": "deberta-v2",
        "max_width": 12,
    }


def test_runtime_refuses_bundles_without_evidence(tmp_path: Path) -> None:
    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel

    with pytest.raises(ExportQualificationError) as error:
        ZeroShotOnnxModel(tmp_path)
    assert error.value.reason is ExportReason.MISSING_EVIDENCE
    (tmp_path / zs.ZERO_SHOT_MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(ExportQualificationError) as error:
        ZeroShotOnnxModel(tmp_path)
    assert error.value.reason is ExportReason.REPORT_INVALID


# ---------------------------------------------------------------------------
# Real exports on tiny in-test checkpoints
# ---------------------------------------------------------------------------


def _export_ready() -> bool:
    names = ("torch", "transformers", "onnx", "onnxruntime", "tokenizers")
    return all(importlib.util.find_spec(name) is not None for name in names) and (
        importlib.util.find_spec("safetensors") is not None
    )


requires_export = pytest.mark.skipif(
    not _export_ready(),
    reason="zero-shot ONNX export needs torch, transformers, onnx, onnxruntime, "
    "tokenizers, and safetensors",
)


@pytest.fixture(scope="module")
def gliner_bundle(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    if not _export_ready():
        pytest.skip("export dependencies are not installed")
    from tests.fixtures.zero_shot_export_stub import build_gliner_artifact

    root = tmp_path_factory.mktemp("gliner")
    source = build_gliner_artifact(root / "source")
    fixtures = write_jsonl(root / "suite.jsonl", gliner_fixture_rows())
    result = _export_module().export_zero_shot_onnx(
        source, root / "bundle", qualification_fixtures_path=fixtures
    )
    return SimpleNamespace(root=root, source=source, fixtures=fixtures, result=result)


@pytest.fixture(scope="module")
def gliclass_bundle(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    if not _export_ready():
        pytest.skip("export dependencies are not installed")
    from tests.fixtures.zero_shot_export_stub import build_gliclass_artifact

    root = tmp_path_factory.mktemp("gliclass")
    source = build_gliclass_artifact(root / "source")
    fixtures = write_jsonl(root / "suite.jsonl", gliclass_fixture_rows())
    result = _export_module().export_zero_shot_onnx(
        source, root / "bundle", qualification_fixtures_path=fixtures
    )
    return SimpleNamespace(root=root, source=source, fixtures=fixtures, result=result)


def _reference(source: Path) -> SimpleNamespace:
    import torch

    from openmed.torch.zero_shot_export import (
        build_zero_shot_module,
        load_reference_weights,
        read_mlx_weights,
        zero_shot_reference_runner,
    )

    loaded = _export_module().load_zero_shot_source(source)
    module = build_zero_shot_module(torch, loaded.config, loaded.spec)
    load_reference_weights(
        torch, module, read_mlx_weights(source / "weights.safetensors")
    )
    return SimpleNamespace(
        loaded=loaded, runner=zero_shot_reference_runner(torch, module)
    )


@requires_export
def test_gliner_span_export_qualifies_offline_without_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capfd: pytest.CaptureFixture[str],
) -> None:
    import onnxruntime as ort

    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel
    from tests.fixtures.zero_shot_export_stub import build_gliner_artifact

    caplog.set_level(logging.DEBUG)
    source = build_gliner_artifact(tmp_path / "source")
    fixtures = write_jsonl(tmp_path / "suite.jsonl", gliner_fixture_rows())
    guard = NetworkGuard().install(monkeypatch)
    result = _export_module().export_zero_shot_onnx(
        source, tmp_path / "bundle", qualification_fixtures_path=fixtures
    )
    out = result.output_dir
    assert result.passed

    report = eq.verify_qualification_report(
        out / eq.EXPORT_QUALIFICATION_FILENAME, artifact_root=out
    )
    assert report["task"] == "zero-shot-ner"
    assert report["qualified_variants"] == ["onnx-fp32"]
    variant = report["variants"][0]
    assert variant["claim"] == "qualified"
    assert variant["logit_parity"]["within_tolerance"] is True
    assert variant["logit_parity"]["label_orders"] == ["reversed", "suite"]
    assert variant["span_parity"]["identical"] is True
    assert variant["direct_identifier_recall"]["candidate"] == {
        "ID_NUM": 1.0,
        "PERSON": 1.0,
    }
    assert variant["critical_leakage"]["new_leaks"] == 0
    assert "ai.onnx:LSTM" in variant["evidence"]["operator_types"]
    assert variant["evidence"]["structural_validation"]["onnx_checker"] == "passed"
    assert {"model.onnx", "tokenizer.json", "config.json"} <= set(report["artifacts"])

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["profile"] == "gliner-span-uni-encoder-v1"
    assert manifest["qualification"]["status"] == "passed"
    assert manifest["bounds"]["max_span_width"] == 4
    assert manifest["prompt"]["word_splitter"]["pattern"] == zs.GLINER_WORD_PATTERN
    assert manifest["offset_contract"]["unit"] == "unicode_code_point"
    assert manifest["unsupported"]["coreml"] == "unsupported_runtime"
    assert manifest["intended_use"]["autonomous_clinical_decisions"] is False
    assert manifest["source"]["revision"].startswith("sha256:")
    assert manifest["source"]["model_id"] is None
    assert set(manifest["files"]) == {"model.onnx", "tokenizer.json", "config.json"}

    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == list(
        zs.GLINER_SPAN_PROFILE.input_names
    )
    assert [item.name for item in session.get_outputs()] == ["logits"]
    assert all(isinstance(dim, str) for dim in session.get_inputs()[0].shape)

    model = ZeroShotOnnxModel(out)
    assert model.qualification["claim"] == "qualified"
    text = "café visit with ZOË HALVARDSEN, mrn 44172290 🙂."
    spans = model.predict_entities(text, list(GLINER_LABELS))
    assert spans.ok
    assert [(item.label, text[item.start : item.end]) for item in spans.spans] == [
        ("person", "ZOË HALVARDSEN"),
        ("identifier", "44172290"),
    ]
    assert spans.spans[0].start == text.index("ZOË")  # code-point offsets

    assert guard.attempts == []
    assert files_containing_sentinels(out, SENTINELS) == []
    captured = capfd.readouterr()
    assert not text_contains_sentinels(caplog.text, SENTINELS)
    assert not text_contains_sentinels(captured.out + captured.err, SENTINELS)


@requires_export
def test_gliner_runtime_matches_the_reference_on_every_fixture(
    gliner_bundle: SimpleNamespace,
) -> None:
    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel

    reference = _reference(gliner_bundle.source)
    model = ZeroShotOnnxModel(gliner_bundle.result.output_dir)
    loaded = reference.loaded
    # The graph was traced with two labels; one and three must work too.
    label_sets = (
        GLINER_LABELS,
        GLINER_LABELS[::-1],
        GLINER_LABELS[:1],
        ("date", *GLINER_LABELS),
    )
    for row in gliner_fixture_rows():
        for labels in label_sets:
            encoding = zs.encode_gliner_span(
                loaded.tokenizer,
                row["text"],
                labels,
                prompt=loaded.prompt,
                bounds=model.bounds,
                graph_span_width=4,
            )
            expected = zs.decode_gliner_spans(
                reference.runner(encoding.feed),
                word_offsets=encoding.word_offsets,
                labels=labels,
            )
            observed = model.predict_entities(row["text"], list(labels))
            assert [(s.label, s.start, s.end) for s in observed.spans] == [
                (s.label, s.start, s.end) for s in expected.spans
            ]
            for left, right in zip(observed.spans, expected.spans):
                assert left.score == pytest.approx(right.score, abs=1e-5)
            prompts = {"PERSON": "person", "ID_NUM": "identifier"}
            gold = {
                (prompts[span["label"]], span["start"], span["end"])
                for span in row["gold_spans"]
                if prompts[span["label"]] in labels
            }
            assert {(s.label, s.start, s.end) for s in observed.spans} == gold


@requires_export
def test_gliner_runtime_enforces_bounds_with_typed_results(
    gliner_bundle: SimpleNamespace,
) -> None:
    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel

    model = ZeroShotOnnxModel(gliner_bundle.result.output_dir)
    cases = [
        (("hello", ["x"] * 33), {}, ExportReason.UNSUPPORTED_LABEL_COUNT, 32),
        (("hello", []), {}, ExportReason.UNSUPPORTED_LABEL_COUNT, 32),
        (("hello", ["person", "person"]), {}, ExportReason.UNSUPPORTED_LABEL, None),
        (
            ("hello", ["<<ENT>>"]),
            {},
            ExportReason.UNSUPPORTED_CONTROL_MARKER,
            None,
        ),
        (("word " * 97, ["person"]), {}, ExportReason.UNSUPPORTED_WORD_COUNT, 96),
        (("x" * 200, ["person"]), {}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH, 128),
        (
            ("hello there", ["person"]),
            {"span_width": 5},
            ExportReason.UNSUPPORTED_SPAN_WIDTH,
            4,
        ),
    ]
    for args, kwargs, reason, limit in cases:
        result = model.predict_entities(*args, **kwargs)
        assert isinstance(result, zs.ZeroShotUnsupported), reason
        assert result.reason is reason
        assert result.limit == limit
    assert model.predict_entities("", ["person"]).spans == ()
    assert model.classify("text", ["a"]).reason is ExportReason.UNSUPPORTED_TASK
    with pytest.raises(ValueError):
        model.predict_entities("text", ["person"], threshold=1.5)

    budget = ZeroShotOnnxModel(gliner_bundle.result.output_dir, memory_budget_bytes=1)
    refused = budget.predict_entities("ZORVAK QUILLAN", ["person"])
    assert refused.reason is ExportReason.UNSUPPORTED_MEMORY_BUDGET
    assert refused.limit == 1


@requires_export
def test_runtime_refuses_unqualified_providers_and_tampered_bundles(
    gliner_bundle: SimpleNamespace, tmp_path: Path
) -> None:
    import shutil

    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel

    out = gliner_bundle.result.output_dir
    with pytest.raises(ExportQualificationError) as error:
        ZeroShotOnnxModel(out, providers=("CoreMLExecutionProvider",))
    assert error.value.reason is ExportReason.UNSUPPORTED_PROVIDER

    tampered = tmp_path / "tampered"
    shutil.copytree(out, tampered)
    tokenizer_json = json.loads((tampered / "tokenizer.json").read_text())
    tokenizer_json["added_tokens"] = tokenizer_json["added_tokens"][:-1]
    (tampered / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    with pytest.raises(ExportQualificationError) as error:
        ZeroShotOnnxModel(tampered)
    assert error.value.reason is ExportReason.ARTIFACT_DIGEST_MISMATCH

    unqualified = _export_module().export_zero_shot_onnx(
        gliner_bundle.source, tmp_path / "unqualified", qualify=False
    )
    manifest = json.loads(unqualified.manifest_path.read_text())
    assert manifest["qualification"]["status"] == "not_run"
    assert manifest["qualification"]["qualified_variants"] == []
    with pytest.raises(ExportQualificationError) as error:
        ZeroShotOnnxModel(unqualified.output_dir)
    assert error.value.reason is ExportReason.MISSING_EVIDENCE


@requires_export
@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"opset": 17}, ExportReason.UNSUPPORTED_OPERATOR),
        ({"precision": "float16"}, ExportReason.UNSUPPORTED_PRECISION),
        ({"quantization": "int8"}, ExportReason.UNSUPPORTED_QUANTIZATION),
        (
            {"providers": ("CPUExecutionProvider", "WebGpuExecutionProvider")},
            ExportReason.UNSUPPORTED_PROVIDER,
        ),
        ({"max_labels": 65}, ExportReason.UNSUPPORTED_LABEL_COUNT),
        ({"max_sequence_length": 4096}, ExportReason.UNSUPPORTED_SEQUENCE_LENGTH),
        ({"max_span_width": 8}, ExportReason.UNSUPPORTED_SPAN_WIDTH),
        ({"threshold": 1.0}, ExportReason.IO_CONTRACT_MISMATCH),
        (
            {"expected_source_revision": "sha256:0"},
            ExportReason.SOURCE_REVISION_MISMATCH,
        ),
    ],
)
def test_unsupported_export_requests_fail_before_tracing(
    gliner_bundle: SimpleNamespace,
    tmp_path: Path,
    kwargs: dict[str, Any],
    reason: ExportReason,
) -> None:
    with pytest.raises(ExportQualificationError) as error:
        _export_module().export_zero_shot_onnx(
            gliner_bundle.source, tmp_path / "out", **kwargs
        )
    assert error.value.reason is reason
    assert not (tmp_path / "out" / "model.onnx").exists()


@requires_export
def test_gliclass_export_qualifies_and_matches_the_reference_ranking(
    gliclass_bundle: SimpleNamespace,
) -> None:
    import onnxruntime as ort

    from openmed.onnx.zero_shot_inference import ZeroShotOnnxModel

    result = gliclass_bundle.result
    assert result.passed
    out = result.output_dir
    report = eq.verify_qualification_report(
        out / eq.EXPORT_QUALIFICATION_FILENAME, artifact_root=out
    )
    variant = report["variants"][0]
    assert report["task"] == "zero-shot-sequence-classification"
    assert variant["classification_parity"]["ranking_identical"] is True
    assert variant["classification_parity"]["decisions_identical"] is True
    assert variant["span_parity"] == {"applicable": False}
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["prompt"]["few_shot_examples"] == "unsupported"
    assert manifest["tensors"]["outputs"][0]["shape"] == ["batch", "labels"]
    assert files_containing_sentinels(out, SENTINELS) == []
    text = json.dumps(report) + result.manifest_path.read_text()
    assert all(json.dumps(label) not in text for label in GLICLASS_LABELS)

    session = ort.InferenceSession(
        str(out / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    assert [item.name for item in session.get_inputs()] == [
        "input_ids",
        "attention_mask",
    ]

    reference = _reference(gliclass_bundle.source)
    loaded = reference.loaded
    model = ZeroShotOnnxModel(out)
    for row in gliclass_fixture_rows():
        labels = row["candidate_labels"]
        for order in (labels, labels[::-1], labels[:2]):
            encoding = zs.encode_gliclass(
                loaded.tokenizer,
                row["text"],
                order,
                prompt=loaded.prompt,
                bounds=model.bounds,
            )
            expected = zs.decode_gliclass(reference.runner(encoding.feed), labels=order)
            observed = model.classify(row["text"], order)
            assert [item.label for item in observed.scores] == [
                item.label for item in expected.scores
            ]
            for left, right in zip(observed.scores, expected.scores):
                assert left.score == pytest.approx(right.score, abs=1e-5)
                assert left.accepted is right.accepted

    marker = model.classify("text <<LABEL>> injected", ["billing"])
    assert marker.reason is ExportReason.UNSUPPORTED_CONTROL_MARKER
    assert model.classify("text", ["x"] * 33).reason is (
        ExportReason.UNSUPPORTED_LABEL_COUNT
    )
    assert model.predict_entities("text", ["a"]).reason is ExportReason.UNSUPPORTED_TASK

    # The existing GLiClass decision backend consumes the same graph contract.
    from openmed.structured.decision_gliclass import GLiClassOnnxRunner

    runner = GLiClassOnnxRunner(session, loaded.tokenizer, max_length=128)
    row = gliclass_fixture_rows()[0]
    encoding = zs.encode_gliclass(
        loaded.tokenizer,
        row["text"],
        row["candidate_labels"],
        prompt=loaded.prompt,
        bounds=model.bounds,
    )
    expected_logits = reference.runner(encoding.feed)[0]
    observed_logits = runner.classify(row["text"], row["candidate_labels"]).logits
    assert np.allclose(observed_logits, expected_logits, atol=1e-5)


def _mlx_ready() -> bool:
    return _export_ready() and importlib.util.find_spec("mlx") is not None


@pytest.mark.skipif(not _mlx_ready(), reason="MLX parity needs MLX and export deps")
@pytest.mark.parametrize(
    ("family", "overrides"),
    [
        ("gliner", {}),
        ("gliner", {"embed_ent_token": True, "wired": False}),
        ("gliner", {"num_rnn_layers": 0, "wired": False}),
        ("gliclass", {}),
        ("gliclass", {"embed_class_token": False, "pooling_strategy": "mean"}),
        ("gliclass", {"normalize_features": True, "use_segment_embeddings": False}),
    ],
)
def test_mlx_runtime_pytorch_reference_and_onnx_agree(
    tmp_path: Path, family: str, overrides: dict[str, Any]
) -> None:
    import mlx.core as mx

    from openmed.mlx.models import load_model
    from openmed.onnx.zero_shot_export import onnx_logits_runner
    from tests.fixtures import zero_shot_export_stub as stub

    if family == "gliner":
        source = stub.build_gliner_artifact(tmp_path / "source", **overrides)
        rows = gliner_fixture_rows()
    else:
        source = stub.build_gliclass_artifact(tmp_path / "source", **overrides)
        rows = gliclass_fixture_rows()
    fixtures = write_jsonl(tmp_path / "suite.jsonl", rows)
    result = _export_module().export_zero_shot_onnx(
        source, tmp_path / "bundle", qualification_fixtures_path=fixtures
    )
    assert result.passed
    import onnxruntime as ort

    reference = _reference(source)
    loaded = reference.loaded
    onnx_runner = onnx_logits_runner(ort, result.model_path, loaded.profile)
    mlx_model = load_model(source)
    bounds = zs.default_bounds(loaded.spec)
    for row in rows:
        if family == "gliner":
            encoding = zs.encode_gliner_span(
                loaded.tokenizer,
                row["text"],
                GLINER_LABELS,
                prompt=loaded.prompt,
                bounds=bounds,
                graph_span_width=4,
            )
            feed = encoding.feed
            output = mlx_model(
                input_ids=mx.array(feed["input_ids"].astype(np.int32)),
                attention_mask=mx.array(feed["attention_mask"].astype(np.float32)),
                words_mask=mx.array(feed["words_mask"].astype(np.int32)),
                span_idx=mx.array(feed["span_idx"].astype(np.int32)),
                span_mask=mx.array(feed["span_mask"]),
            )
        else:
            encoding = zs.encode_gliclass(
                loaded.tokenizer,
                row["text"],
                row["candidate_labels"],
                prompt=loaded.prompt,
                bounds=bounds,
            )
            feed = encoding.feed
            output = mlx_model(
                input_ids=mx.array(feed["input_ids"].astype(np.int32)),
                attention_mask=mx.array(feed["attention_mask"].astype(np.float32)),
            )
        torch_logits = reference.runner(feed)
        mlx_logits = np.array(output["logits"]).reshape(torch_logits.shape)
        onnx_logits = onnx_runner(feed)
        scale = max(1.0, float(np.abs(torch_logits[torch_logits > -1e8]).max()))
        assert np.abs(mlx_logits - torch_logits).max() <= 1e-4 * scale
        assert np.abs(onnx_logits - torch_logits).max() <= 1e-4 * scale
