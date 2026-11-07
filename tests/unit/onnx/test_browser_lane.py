"""Offline contract tests for the ONNX Runtime Web browser lane.

The browser run itself lives in ``tests/browser/ort-lane``; these tests cover
the synthetic feeds, the span decoder shared with the browser runtime, the
value-free record contract and the release gate without Torch or a browser.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from openmed.onnx import browser_lane as lane

ROOT = Path(__file__).resolve().parents[3]
GOLDEN = ROOT / "tests" / "web" / "fixtures" / "webgpu_token_classification_golden.json"
HARNESS = ROOT / "tests" / "browser" / "ort-lane" / "harness" / "lane.mjs"
SPEC = ROOT / "tests" / "browser" / "ort-lane" / "ort-token-classification.spec.ts"
BACKENDS = {
    "webgpu": "webgpu",
    "wasm-threaded": "wasm-simd-threads",
    "wasm-single": "wasm-basic",
}


def _fixture() -> dict[str, Any]:
    feeds = lane.build_synthetic_feeds(
        batch_size=1, sequence_length=4, padding_tokens=0
    )
    names = ("input_ids", "attention_mask", "token_type_ids")
    logits = np.asarray(
        [
            [
                [4.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    id2label = {str(index): label for index, label in enumerate(lane.SYNTHETIC_LABELS)}
    return {
        "format": lane.FIXTURE_FORMAT,
        "format_version": lane.FIXTURE_VERSION,
        "lane": lane.LANE_ID,
        "synthetic": True,
        "seed": lane.DEFAULT_SEED,
        "id2label": id2label,
        "graphs": {
            "fp32": {
                "file": "graphs/model.onnx",
                "precision": "float32",
                "sha256": "1" * 64,
                "bytes": 10,
                "external_data": [
                    {
                        "location": "model.onnx.data",
                        "file": "graphs/model.onnx.data",
                        "sha256": "2" * 64,
                        "bytes": 4,
                    }
                ],
                "op_types": ["Cast", "MatMul", "Shape"],
            },
            "fp16": {
                "file": "graphs/model.webgpu.onnx",
                "precision": "float16",
                "sha256": "3" * 64,
                "bytes": 10,
                "external_data": [],
                "op_types": ["Cast", "MatMul", "Shape"],
            },
            "fallback-control": {
                "file": "graphs/model.fallback-control.onnx",
                "precision": "float32",
                "sha256": "4" * 64,
                "bytes": 10,
                "external_data": [],
                "op_types": ["Cast", "MatMul", "Mod", "Shape"],
                "control_op_type": "Mod",
            },
        },
        "feeds": {
            "input_names": list(names),
            "batch_size": 1,
            "sequence_length": 4,
            **{name: feeds[name].reshape(-1).tolist() for name in names},
            "sha256": lane.feeds_digest(feeds, names),
        },
        "reference": {
            "runtime": "python-onnxruntime-cpu",
            "graph": "fp32",
            "dims": [1, 4, 5],
            "logits": logits.reshape(-1).tolist(),
            "sha256": lane.logits_digest(logits),
            "spans": lane.decode_token_spans(logits, id2label),
            "min_top2_margin": lane.minimum_top2_margin(logits),
        },
        "cases": [case.to_dict() for case in lane.DEFAULT_CASES],
    }


def _record(fixture: dict[str, Any], case_id: str) -> dict[str, Any]:
    case = next(item for item in fixture["cases"] if item["case_id"] == case_id)
    graph = fixture["graphs"][case["graph"]]
    provider = case["provider"]
    webgpu = provider == "webgpu"
    single = provider == "wasm-single"
    ops = ["Cast", "Shape"]
    if graph.get("control_op_type"):
        ops = sorted([*ops, graph["control_op_type"]])
    return {
        "format": lane.RECORD_FORMAT,
        "format_version": lane.RECORD_VERSION,
        "lane": lane.LANE_ID,
        "case_id": case_id,
        "graph": case["graph"],
        "precision": graph["precision"],
        "provider": {
            "id": provider,
            "backend": BACKENDS[provider],
            "execution_providers": ["webgpu", "wasm"] if webgpu else ["wasm"],
            "runtime": "onnxruntime-web",
            "runtime_version": "1.31.1",
            "runtime_bundle": "webgpu" if webgpu else "wasm",
            "cross_origin_isolated": case["isolation"] == "cross-origin-isolated",
            "wasm_threads": 1 if single else 4,
            "worker_count": 0 if single else 3,
        },
        "browser": {"name": "chromium", "version": "151.0.7922.34"},
        "digests": {
            "graph_sha256": graph["sha256"],
            "external_data_sha256": lane._external_data_digest(graph),
            "feeds_sha256": fixture["feeds"]["sha256"],
            "reference_sha256": fixture["reference"]["sha256"],
        },
        "tolerances": dict(case["tolerances"]),
        "outcomes": {
            "status": "passed",
            "reason_code": "ok",
            "webgpu_support": "partial" if webgpu else "not_applicable",
            "fallback_op_types": ops if webgpu else [],
            "fallback_node_count": len(ops) if webgpu else 0,
            "node_placement": {"webgpu": 90, "cpu": 38} if webgpu else {"cpu": 117},
            "max_abs_logit_delta": 2e-6,
            "logits_within_tolerance": True,
            "reference_span_count": 1,
            "candidate_span_count": 1,
            "matched_span_count": 1,
            "recall": 1.0,
            "span_parity": True,
            "same_origin_request_count": 12,
            "cross_origin_request_count": 0,
        },
    }


def _records(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    return [_record(fixture, case["case_id"]) for case in fixture["cases"]]


def _unavailable(record: dict[str, Any], reason: str) -> dict[str, Any]:
    outcomes = record["outcomes"]
    outcomes.update(
        {
            "status": "unavailable",
            "reason_code": reason,
            "webgpu_support": "unknown",
            "fallback_op_types": [],
            "fallback_node_count": 0,
            "node_placement": {},
        }
    )
    for key in lane._MEASURED_OUTCOMES:
        outcomes[key] = None
    return record


def test_decoder_matches_the_browser_runtime_golden_spans() -> None:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    tokens = golden["tokens"]
    logits = np.asarray(golden["reference_logits"], dtype=np.float32).reshape(
        tokens["batch_size"],
        tokens["sequence_length"],
        golden["classification_head"]["label_count"],
    )

    spans = lane.decode_token_spans(
        logits, golden["id2label"], tokens["attention_mask"]
    )

    assert spans == golden["reference_token_spans"]


def test_decoder_handles_bilou_masks_and_orphan_inside_labels() -> None:
    labels = {0: "O", 1: "B-ID", 2: "I-ID", 3: "L-ID", 4: "U-NAME"}
    picks = [[2, 2, 0, 4, 1, 3, 1, 2]]
    logits = np.zeros((1, 8, 5), dtype=np.float32)
    for token, label in enumerate(picks[0]):
        logits[0, token, label] = 1.0
    mask = [[1, 1, 1, 1, 1, 1, 1, 0]]

    spans = lane.decode_token_spans(logits, labels, mask)

    assert spans == [
        {"batch_index": 0, "label": "ID", "start_token": 0, "end_token": 2},
        {"batch_index": 0, "label": "NAME", "start_token": 3, "end_token": 4},
        {"batch_index": 0, "label": "ID", "start_token": 4, "end_token": 6},
        {"batch_index": 0, "label": "ID", "start_token": 6, "end_token": 7},
    ]


def test_decoder_rejects_incomplete_label_maps_and_bad_shapes() -> None:
    with pytest.raises(lane.BrowserLaneError, match="cover every"):
        lane.decode_token_spans(np.zeros((1, 2, 3)), {0: "O", 1: "B-ID"})
    with pytest.raises(lane.BrowserLaneError, match="shape"):
        lane.decode_token_spans(np.zeros((2, 3)), {0: "O"})


def test_top2_margin_ignores_masked_tokens() -> None:
    logits = np.asarray([[[3.0, 1.0], [0.5, 0.45]]], dtype=np.float32)

    assert lane.minimum_top2_margin(logits) == pytest.approx(0.05, abs=1e-6)
    assert lane.minimum_top2_margin(logits, [[1, 0]]) == pytest.approx(2.0)
    assert lane.minimum_top2_margin(logits, [[0, 0]]) == math.inf


def test_synthetic_feeds_are_deterministic_and_padded() -> None:
    first = lane.build_synthetic_feeds()
    second = lane.build_synthetic_feeds()
    redraw = lane.build_synthetic_feeds(attempt=1)

    for name in ("input_ids", "attention_mask", "token_type_ids"):
        assert np.array_equal(first[name], second[name])
        assert first[name].dtype == np.int64
    assert not np.array_equal(first["input_ids"], redraw["input_ids"])
    ids = first["input_ids"]
    mask = first["attention_mask"]
    assert ids.shape == (lane.DEFAULT_BATCH_SIZE, lane.DEFAULT_SEQUENCE_LENGTH)
    assert (ids[:, 0] == lane.CLS_TOKEN_ID).all()
    assert ids[0, -1] == lane.SEP_TOKEN_ID
    pad = lane.DEFAULT_PADDING_TOKENS
    assert ids[-1, -pad - 1] == lane.SEP_TOKEN_ID
    assert (ids[-1, -pad:] == lane.PAD_TOKEN_ID).all()
    assert mask[-1].tolist() == [1] * (lane.DEFAULT_SEQUENCE_LENGTH - pad) + [0] * pad
    assert mask[0].all()
    for row in range(ids.shape[0]):
        active = int(mask[row].sum())
        assert ids[row, 1 : active - 1].min() >= lane.FIRST_REGULAR_TOKEN_ID
    assert ids.max() < len(lane.synthetic_vocabulary())
    assert not first["token_type_ids"].any()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "one row"),
        ({"sequence_length": 2}, "three tokens"),
        ({"padding_tokens": 10}, "room"),
        ({"vocab_size": lane.FIRST_REGULAR_TOKEN_ID}, "regular tokens"),
    ],
)
def test_synthetic_feeds_reject_unusable_shapes(
    kwargs: dict[str, int], message: str
) -> None:
    with pytest.raises(lane.BrowserLaneError, match=message):
        lane.build_synthetic_feeds(**kwargs)


def test_digests_cover_little_endian_tensor_bytes_in_input_order() -> None:
    feeds = {
        "input_ids": np.asarray([[2, 9, 3]], dtype=np.int64),
        "attention_mask": np.asarray([[1, 1, 1]], dtype=np.int64),
    }
    expected = hashlib.sha256(
        np.asarray([2, 9, 3, 1, 1, 1], dtype="<i8").tobytes()
    ).hexdigest()

    assert lane.feeds_digest(feeds, ("input_ids", "attention_mask")) == expected
    assert lane.feeds_digest(feeds, ("attention_mask", "input_ids")) != expected
    logits = np.asarray([0.5, -1.25], dtype=np.float64)
    assert (
        lane.logits_digest(logits)
        == hashlib.sha256(np.asarray(logits, dtype="<f4").tobytes()).hexdigest()
    )


def test_default_cases_cover_every_provider_and_the_negative_control() -> None:
    cases = {case.case_id: case for case in lane.DEFAULT_CASES}

    assert {case.provider for case in cases.values()} == set(lane.PROVIDERS)
    assert {case.graph for case in cases.values()} == set(lane.GRAPH_IDS)
    assert cases["wasm-threaded-fp32"].isolation == "cross-origin-isolated"
    assert cases["wasm-single-fp32"].isolation == "none"
    assert cases["webgpu-fallback-control"].expected_webgpu_support == "partial"
    assert cases["webgpu-fp16"].tolerance == lane.WEBGPU_FLOAT16_TOLERANCE
    loosest = max(case.tolerance.max_abs_logit_delta for case in cases.values())
    assert lane.MIN_REFERENCE_MARGIN >= 2 * loosest
    assert all(case.tolerance.span_tolerance_tokens == 0 for case in cases.values())
    assert all(case.tolerance.max_recall_delta == 0 for case in cases.values())


def test_valid_records_pass_the_gate() -> None:
    fixture = _fixture()
    records = _records(fixture)
    for record in records:
        lane.validate_lane_record(record)

    summary = lane.evaluate_lane_records(records, fixture)

    assert summary["passed"] is True
    assert summary["failures"] == []
    assert set(summary["statuses"].values()) == {"passed"}
    assert summary["webgpu_support"]["webgpu-fallback-control"] == "partial"
    assert summary["webgpu_support"]["wasm-single-fp32"] == "not_applicable"


def _mutate(path: str, value: Any):
    def apply(record: dict[str, Any]) -> None:
        *parents, leaf = path.split(".")
        target = record
        for key in parents:
            target = target[key]
        target[leaf] = value

    return apply


@pytest.mark.parametrize(
    ("mutation", "case_id"),
    [
        (_mutate("outcomes.logits", [0.1, 0.2]), "wasm-single-fp32"),
        (_mutate("note", "Patient Avery Morgan"), "wasm-single-fp32"),
        (_mutate("browser.version", "http://127.0.0.1/fixture"), "wasm-single-fp32"),
        (_mutate("case_id", "../records/x"), "wasm-single-fp32"),
        (_mutate("outcomes.fallback_op_types", ["Avery Morgan"]), "webgpu-fp32"),
        (_mutate("outcomes.fallback_op_types", ["Shape", "Cast"]), "webgpu-fp32"),
        (_mutate("outcomes.reason_code", "span_mismatch"), "webgpu-fp32"),
        (_mutate("outcomes.reason_code", "free text"), "webgpu-fp32"),
        (_mutate("outcomes.max_abs_logit_delta", 0.5), "wasm-threaded-fp32"),
        (_mutate("outcomes.span_parity", False), "wasm-threaded-fp32"),
        (_mutate("outcomes.cross_origin_request_count", 1), "wasm-threaded-fp32"),
        (_mutate("outcomes.webgpu_support", "full"), "webgpu-fp32"),
        (_mutate("outcomes.webgpu_support", "full"), "wasm-single-fp32"),
        (_mutate("outcomes.webgpu_support", "unknown"), "webgpu-fp32"),
        (_mutate("outcomes.fallback_node_count", 0), "webgpu-fp32"),
        (_mutate("outcomes.fallback_node_count", True), "webgpu-fp32"),
        (_mutate("outcomes.same_origin_request_count", -1), "webgpu-fp32"),
        (_mutate("outcomes.node_placement", {"gpu0": 3}), "webgpu-fp32"),
        (_mutate("outcomes.recall", None), "webgpu-fp32"),
        (_mutate("digests.graph_sha256", "not-a-digest"), "webgpu-fp32"),
        (_mutate("tolerances.max_abs_logit_delta", -1.0), "webgpu-fp32"),
        (_mutate("tolerances.span_tolerance_tokens", 0.5), "webgpu-fp32"),
        (_mutate("provider.execution_providers", ["cuda"]), "webgpu-fp32"),
        (_mutate("provider.runtime_version", "latest from cdn"), "webgpu-fp32"),
        (_mutate("provider.cross_origin_isolated", "yes"), "webgpu-fp32"),
        (_mutate("format_version", 2), "webgpu-fp32"),
    ],
)
def test_record_validation_rejects_values_and_contradictions(
    mutation: Any, case_id: str
) -> None:
    record = _record(_fixture(), case_id)
    mutation(record)

    with pytest.raises(lane.BrowserLaneError):
        lane.validate_lane_record(record)


def test_record_validation_errors_never_echo_values() -> None:
    record = _record(_fixture(), "webgpu-fp32")
    record["outcomes"]["fallback_op_types"] = ["SYNTHETIC-PHI-3940"]
    record["browser"]["version"] = "SYNTHETIC-PHI-3940"

    with pytest.raises(lane.BrowserLaneError) as error:
        lane.validate_lane_record(record)

    assert "SYNTHETIC-PHI-3940" not in str(error.value)


def test_unavailable_records_carry_no_measurements() -> None:
    record = _unavailable(
        _record(_fixture(), "webgpu-fp16"), "webgpu_adapter_unavailable"
    )
    lane.validate_lane_record(record)

    record["outcomes"]["max_abs_logit_delta"] = 0.0
    with pytest.raises(lane.BrowserLaneError, match="measurements"):
        lane.validate_lane_record(record)

    failed = _unavailable(_record(_fixture(), "webgpu-fp16"), "span_mismatch")
    with pytest.raises(lane.BrowserLaneError, match="contradicts"):
        lane.validate_lane_record(failed)


def test_gate_never_accepts_a_failed_webgpu_record() -> None:
    fixture = _fixture()
    records = _records(fixture)
    fp16 = next(item for item in records if item["case_id"] == "webgpu-fp16")
    fp16["outcomes"].update(
        {"status": "failed", "reason_code": "span_mismatch", "span_parity": False}
    )

    summary = lane.evaluate_lane_records(
        records, fixture, allow_unavailable=sorted(lane.UNAVAILABLE_REASONS)
    )

    assert summary["passed"] is False
    assert summary["failures"] == ["webgpu-fp16:span_mismatch"]
    assert summary["statuses"]["webgpu-fp16"] == "failed"


def test_gate_accepts_only_listed_unavailable_reasons() -> None:
    fixture = _fixture()
    records = _records(fixture)
    for record in records:
        if record["case_id"] == "webgpu-fp16":
            _unavailable(record, "webgpu_shader_f16_unavailable")

    strict = lane.evaluate_lane_records(records, fixture)
    accepted = lane.evaluate_lane_records(
        records, fixture, allow_unavailable=["webgpu_shader_f16_unavailable"]
    )
    other = lane.evaluate_lane_records(
        records, fixture, allow_unavailable=["webgpu_adapter_unavailable"]
    )

    assert strict["failures"] == ["webgpu-fp16:webgpu_shader_f16_unavailable"]
    assert accepted["passed"] is True
    assert accepted["statuses"]["webgpu-fp16"] == "unavailable"
    assert other["passed"] is False
    with pytest.raises(lane.BrowserLaneError, match="non-unavailable"):
        lane.evaluate_lane_records(records, fixture, allow_unavailable=["ok"])


def test_gate_requires_exactly_one_record_per_case() -> None:
    fixture = _fixture()
    records = _records(fixture)
    duplicate = copy.deepcopy(records[0])
    unknown = copy.deepcopy(records[1])
    unknown["case_id"] = "webgpu-extra"
    invalid = copy.deepcopy(records[2])
    invalid["outcomes"]["text"] = "synthetic"

    summary = lane.evaluate_lane_records(
        [*records[1:], duplicate, duplicate, unknown, invalid], fixture
    )

    assert summary["passed"] is False
    assert f"{records[0]['case_id']}:record_duplicated" in summary["failures"]
    assert "webgpu-extra:record_unknown_case" in summary["failures"]
    assert "record_invalid" in summary["failures"]
    missing = lane.evaluate_lane_records(records[1:], fixture)
    assert missing["failures"] == [f"{records[0]['case_id']}:record_missing"]


def test_gate_rejects_digest_tolerance_and_case_drift() -> None:
    fixture = _fixture()
    records = _records(fixture)
    by_case = {record["case_id"]: record for record in records}
    by_case["wasm-single-fp32"]["digests"]["graph_sha256"] = "9" * 64
    by_case["wasm-threaded-fp32"]["tolerances"]["max_abs_logit_delta"] = 0.5
    by_case["wasm-threaded-fp32"]["outcomes"]["max_abs_logit_delta"] = 0.1
    by_case["webgpu-fp32"]["graph"] = "fp16"

    summary = lane.evaluate_lane_records(records, fixture)

    assert set(summary["failures"]) == {
        "wasm-single-fp32:digest_mismatch",
        "wasm-threaded-fp32:tolerance_mismatch",
        "webgpu-fp32:record_case_mismatch",
    }


def test_negative_control_must_report_partial_fallback_with_its_operator() -> None:
    fixture = _fixture()
    records = _records(fixture)
    control = next(r for r in records if r["case_id"] == "webgpu-fallback-control")
    control["outcomes"].update(
        {
            "webgpu_support": "full",
            "fallback_op_types": [],
            "fallback_node_count": 0,
        }
    )

    summary = lane.evaluate_lane_records(records, fixture)

    assert set(summary["failures"]) == {
        "webgpu-fallback-control:unexpected_webgpu_support",
        "webgpu-fallback-control:control_op_not_reported",
    }
    control["outcomes"].update(
        {
            "webgpu_support": "partial",
            "fallback_op_types": ["Cast"],
            "fallback_node_count": 1,
        }
    )
    summary = lane.evaluate_lane_records(records, fixture)
    assert summary["failures"] == ["webgpu-fallback-control:control_op_not_reported"]


def _drop_graph(fixture: dict[str, Any]) -> None:
    del fixture["graphs"]["fp16"]


def _bad_digest(fixture: dict[str, Any]) -> None:
    fixture["graphs"]["fp32"]["sha256"] = "zz"


def _short_feed(fixture: dict[str, Any]) -> None:
    fixture["feeds"]["input_ids"] = fixture["feeds"]["input_ids"][:-1]


def _bad_dims(fixture: dict[str, Any]) -> None:
    fixture["reference"]["dims"] = [1, 4, 4]


def _non_finite(fixture: dict[str, Any]) -> None:
    fixture["reference"]["logits"][0] = float("nan")


def _duplicate_case(fixture: dict[str, Any]) -> None:
    fixture["cases"].append(dict(fixture["cases"][0]))


def _control_in_base(fixture: dict[str, Any]) -> None:
    fixture["graphs"]["fp32"]["op_types"].append("Mod")


def _no_cases(fixture: dict[str, Any]) -> None:
    fixture["cases"] = []


def _unknown_provider(fixture: dict[str, Any]) -> None:
    fixture["cases"][0]["provider"] = "webgl"


@pytest.mark.parametrize(
    "mutation",
    [
        _drop_graph,
        _bad_digest,
        _short_feed,
        _bad_dims,
        _non_finite,
        _duplicate_case,
        _control_in_base,
        _no_cases,
        _unknown_provider,
    ],
)
def test_fixture_validation_rejects_unusable_fixtures(mutation: Any) -> None:
    fixture = _fixture()
    mutation(fixture)

    with pytest.raises(lane.BrowserLaneError):
        lane.validate_lane_fixture(fixture)


def _write_lane(tmp_path: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    fixture_path = tmp_path / "fixture" / lane.FIXTURE_FILENAME
    fixture_path.parent.mkdir()
    fixture_path.write_text(json.dumps(_fixture()), encoding="utf-8")
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    for record in records:
        (records_dir / f"{record['case_id']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return fixture_path, records_dir


def test_check_cli_prints_a_value_free_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_path, records_dir = _write_lane(tmp_path, _records(_fixture()))

    code = lane.main(
        ["check", "--fixture", str(fixture_path), "--records-dir", str(records_dir)]
    )

    output = capsys.readouterr().out
    assert code == 0
    assert json.loads(output)["passed"] is True
    assert str(tmp_path) not in output


def test_check_cli_fails_closed_on_unaccepted_unavailable_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    records = _records(_fixture())
    _unavailable(records[2], "webgpu_adapter_unavailable")
    fixture_path, records_dir = _write_lane(tmp_path, records)
    base = ["check", "--fixture", str(fixture_path), "--records-dir", str(records_dir)]

    assert lane.main(base) == 1
    assert json.loads(capsys.readouterr().out)["failures"] == [
        "webgpu-fp32:webgpu_adapter_unavailable"
    ]
    assert lane.main([*base, "--allow-unavailable", "webgpu_adapter_unavailable"]) == 0


def test_browser_harness_emits_the_python_record_contract() -> None:
    harness = HARNESS.read_text(encoding="utf-8")
    spec = SPEC.read_text(encoding="utf-8")

    for key in (
        lane._RECORD_KEYS | lane._PROVIDER_KEYS | lane._DIGEST_KEYS | lane._OUTCOME_KEYS
    ):
        assert re.search(rf"\b{key}:", harness), key
    codes = set(re.findall(r'(?:fail|unavailable)\(record, "([a-z0-9_]+)"\)', harness))
    codes |= set(re.findall(r'reason_code = "([a-z0-9_]+)"', harness + spec))
    assert codes
    assert codes <= lane.REASON_CODES
    assert f'"{lane.RECORD_FORMAT}"' in harness
    assert f'"{lane.LANE_ID}"' in harness


def test_fallback_control_graph_appends_a_value_preserving_mod(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["features"], ["logits"], name="head")],
        "tiny",
        [helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 3])],
    )
    source = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        ),
        str(source),
    )

    target = lane.build_fallback_control_graph(source, tmp_path / "control.onnx")

    control = onnx.load(str(target))
    assert [node.op_type for node in control.graph.node] == ["Identity", "Mod"]
    assert [output.name for output in control.graph.output] == ["logits"]
    assert control.graph.node[0].output[0] == "logits_before_fallback_control"
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    features = np.asarray([[-7.25, 0.0, 12.5]], dtype=np.float32)
    (logits,) = session.run(["logits"], {"features": features})
    assert logits.tolist() == features.tolist()


def test_fallback_control_requires_a_single_logits_output(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    graph = helper.make_graph(
        [helper.make_node("Identity", ["features"], ["scores"])],
        "tiny",
        [helper.make_tensor_value_info("features", onnx.TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("scores", onnx.TensorProto.FLOAT, [1])],
    )
    source = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), str(source))

    with pytest.raises(lane.BrowserLaneError, match="logits"):
        lane.build_fallback_control_graph(source, tmp_path / "control.onnx")
