"""End-to-end preparation of the ONNX Runtime Web browser lane fixture.

Builds the tiny randomly initialised BERT classifier offline, exports it with
the repository exporter, and checks the fixture the Playwright lane consumes.
Needs the ``onnx`` extra (Torch, Transformers, ONNX and ONNX Runtime); nothing
is downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("torch")
pytest.importorskip("transformers")

from openmed.onnx import browser_lane as lane  # noqa: E402


@pytest.fixture(scope="module")
def prepared(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    output_dir = tmp_path_factory.mktemp("ort-lane")
    fixture_path = lane.prepare_browser_lane(output_dir)
    return output_dir, json.loads(fixture_path.read_text(encoding="utf-8"))


def _session_logits(path: Path, fixture: dict) -> np.ndarray:
    feeds = fixture["feeds"]
    shape = (feeds["batch_size"], feeds["sequence_length"])
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {
        name: np.asarray(feeds[name], dtype=np.int64).reshape(shape)
        for name in feeds["input_names"]
    }
    return session.run(["logits"], inputs)[0]


def test_fixture_contract_and_exported_graphs(prepared: tuple[Path, dict]) -> None:
    output_dir, fixture = prepared
    lane.validate_lane_fixture(fixture)

    assert fixture["synthetic"] is True
    assert fixture["model"]["family"] == "bert"
    assert [case["case_id"] for case in fixture["cases"]] == [
        case.case_id for case in lane.DEFAULT_CASES
    ]
    fp32 = fixture["graphs"]["fp32"]
    assert fp32["file"] == "graphs/model.onnx"
    assert fixture["graphs"]["fp16"]["file"] == "graphs/model.webgpu.onnx"
    assert fixture["graphs"]["fp16"]["precision"] == "float16"
    for entry in fixture["graphs"].values():
        path = output_dir / entry["file"]
        assert lane._file_sha256(path) == entry["sha256"]
        for item in entry["external_data"]:
            assert lane._file_sha256(output_dir / item["file"]) == item["sha256"]
    control = onnx.load(str(output_dir / fixture["graphs"]["fallback-control"]["file"]))
    assert control.graph.node[-1].op_type == lane.FALLBACK_CONTROL_OP_TYPE
    manifest = json.loads(
        (output_dir / lane.GRAPHS_DIRNAME / "openmed-onnx.json").read_text("utf-8")
    )
    assert {item["path"] for item in manifest["artifacts"]} == {
        "model.onnx",
        "model.webgpu.onnx",
    }


def test_exported_fp32_graph_loads_from_disk_with_its_sidecar(
    prepared: tuple[Path, dict],
) -> None:
    output_dir, fixture = prepared
    fp32 = fixture["graphs"]["fp32"]
    model = onnx.load(str(output_dir / fp32["file"]), load_external_data=False)
    external = [
        tensor
        for tensor in model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL
    ]
    assert external, "the tiny export still exercises an external-data sidecar"
    assert len(external) < len(model.graph.initializer)

    logits = _session_logits(output_dir / fp32["file"], fixture)

    reference = np.asarray(fixture["reference"]["logits"], dtype=np.float32)
    assert logits.shape == tuple(fixture["reference"]["dims"])
    assert np.array_equal(logits.reshape(-1), reference)
    assert lane.logits_digest(logits) == fixture["reference"]["sha256"]


def test_reference_spans_margins_and_variant_parity(
    prepared: tuple[Path, dict],
) -> None:
    output_dir, fixture = prepared
    reference = np.asarray(fixture["reference"]["logits"], dtype=np.float32).reshape(
        fixture["reference"]["dims"]
    )
    mask = np.asarray(fixture["feeds"]["attention_mask"]).reshape(reference.shape[:2])

    assert fixture["reference"]["spans"]
    assert fixture["reference"]["spans"] == lane.decode_token_spans(
        reference, fixture["id2label"], mask
    )
    assert lane.minimum_top2_margin(reference, mask) > lane.MIN_REFERENCE_MARGIN
    fp16 = _session_logits(output_dir / fixture["graphs"]["fp16"]["file"], fixture)
    assert np.abs(fp16 - reference).max() <= (
        lane.WEBGPU_FLOAT16_TOLERANCE.max_abs_logit_delta
    )
    control = _session_logits(
        output_dir / fixture["graphs"]["fallback-control"]["file"], fixture
    )
    assert np.array_equal(control, reference)
    serialized = json.dumps(fixture)
    assert str(output_dir) not in serialized


def test_preparation_is_deterministic(
    prepared: tuple[Path, dict], tmp_path: Path
) -> None:
    _, fixture = prepared

    again = json.loads(lane.prepare_browser_lane(tmp_path).read_text("utf-8"))

    assert again["graphs"] == fixture["graphs"]
    assert again["feeds"] == fixture["feeds"]
    assert again["reference"] == fixture["reference"]
