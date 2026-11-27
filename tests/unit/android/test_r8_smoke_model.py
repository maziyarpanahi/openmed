"""Tests for the synthetic model bundled by the R8-minified consumer smoke app."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts.android.generate_r8_smoke_model import (
    BUNDLE_FILES,
    LABEL_LOGIT,
    LABELS,
    OPSET_VERSION,
    SMOKE_TEXT,
    VOCAB,
    build_id2label,
    build_model,
    build_smoke_case,
    build_tokenizer,
    bundle_contents,
    logits_table,
    main,
    reference_entities,
    tokenize,
)


def _fields(payload: bytes) -> list[tuple[int, int | bytes]]:
    """Decode one protobuf message into ``(field, value)`` pairs."""
    fields: list[tuple[int, int | bytes]] = []
    offset = 0

    def varint() -> int:
        nonlocal offset
        value = shift = 0
        while True:
            byte = payload[offset]
            offset += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                return value
            shift += 7

    while offset < len(payload):
        key = varint()
        number, wire_type = key >> 3, key & 0x7
        if wire_type == 0:
            fields.append((number, varint()))
        elif wire_type == 2:
            length = varint()
            fields.append((number, payload[offset : offset + length]))
            offset += length
        else:  # pragma: no cover - the encoder writes no other wire types
            raise AssertionError(f"unexpected wire type {wire_type}")
    return fields


def _all(payload: bytes, number: int) -> list[int | bytes]:
    return [value for field, value in _fields(payload) if field == number]


def _one(payload: bytes, number: int) -> int | bytes:
    (value,) = _all(payload, number)
    return value


def _text(value: int | bytes) -> str:
    assert isinstance(value, bytes)
    return value.decode("utf-8")


def test_model_is_a_lookup_classifier_graph() -> None:
    model = build_model()
    graph = _one(model, 7)
    opset = _one(model, 8)
    assert isinstance(graph, bytes) and isinstance(opset, bytes)

    nodes = [_fields(node) for node in _all(graph, 1)]  # type: ignore[arg-type]
    op_types = [_text(value) for node in nodes for field, value in node if field == 4]
    inputs = [_text(_one(item, 1)) for item in _all(graph, 11)]  # type: ignore[arg-type]
    outputs = [_text(_one(item, 1)) for item in _all(graph, 12)]  # type: ignore[arg-type]
    initializers = {
        _text(_one(tensor, 8)): tensor  # type: ignore[arg-type]
        for tensor in _all(graph, 5)
    }

    assert _one(model, 1) == 7
    assert _one(opset, 2) == OPSET_VERSION
    assert op_types == ["Gather", "Cast", "Unsqueeze", "Mul"]
    assert inputs == ["input_ids", "attention_mask"]
    assert outputs == ["logits"]
    table = initializers["logit_table"]
    assert _all(table, 1) == [len(VOCAB), len(LABELS)]  # type: ignore[arg-type]
    raw = _one(table, 9)  # type: ignore[arg-type]
    assert isinstance(raw, bytes)
    values = struct.unpack(f"<{len(VOCAB) * len(LABELS)}f", raw)
    assert list(values) == [value for row in logits_table() for value in row]
    axes = _one(initializers["mask_axes"], 9)  # type: ignore[arg-type]
    assert axes == struct.pack("<q", -1)


def test_lookup_table_labels_only_the_synthetic_name() -> None:
    table = logits_table()
    predicted = {token: LABELS[row.index(max(row))] for token, row in zip(VOCAB, table)}

    assert predicted["avery"] == "B-NAME"
    assert predicted["quill"] == "I-NAME"
    assert {label for token, label in predicted.items()} == set(LABELS)
    assert all(max(row) == LABEL_LOGIT for row in table)


def test_tokenizer_covers_the_smoke_text_without_unknown_tokens() -> None:
    tokenizer = build_tokenizer()
    vocab = tokenizer["model"]["vocab"]  # type: ignore[index]

    assert tokenizer["model"]["type"] == "WordLevel"  # type: ignore[index]
    assert [token for token, _, _ in tokenize(SMOKE_TEXT)] == list(VOCAB[4:])
    assert all(token in vocab for token, _, _ in tokenize(SMOKE_TEXT))
    assert build_id2label() == {"0": "O", "1": "B-NAME", "2": "I-NAME"}


def test_smoke_case_spans_are_the_reference_predictions() -> None:
    case = build_smoke_case()

    assert case["entities"] == [{"label": "NAME", "start": 8, "end": 19}]
    assert case["entities"] == reference_entities(SMOKE_TEXT)
    assert SMOKE_TEXT[8:19] == "Avery Quill"
    assert case["deidentify"] == {
        "policy": "hipaa_safe_harbor",
        "canonical_labels": ["PERSON"],
    }
    chunked = case["chunked"]
    assert isinstance(chunked, dict)
    # The name must sit wholly inside the first chunk, so chunking is
    # exercised without depending on cross-chunk merging.
    assert chunked["chunk_token_limit"] > 2
    assert chunked["token_overlap"] < chunked["chunk_token_limit"]


def test_reference_entities_follow_bio_aggregation() -> None:
    assert reference_entities("quill avery quill the avery") == [
        {"label": "NAME", "start": 0, "end": 5},
        {"label": "NAME", "start": 6, "end": 17},
        {"label": "NAME", "start": 22, "end": 27},
    ]


def test_bundle_is_deterministic_and_written_by_the_cli(tmp_path: Path) -> None:
    first, second = bundle_contents(), bundle_contents()

    assert first == second
    assert tuple(first) == BUNDLE_FILES
    assert main(["--output-dir", str(tmp_path / "bundle")]) == 0
    for name, payload in first.items():
        assert (tmp_path / "bundle" / name).read_bytes() == payload
    json.loads(first["smoke_case.json"])


def test_onnxruntime_predicts_the_expected_labels(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    np = pytest.importorskip("numpy")
    path = tmp_path / "model.onnx"
    path.write_bytes(build_model())
    onnx.checker.check_model(onnx.load(str(path)), full_check=True)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ids = np.array([[VOCAB.index(token) for token in ("[CLS]", "avery", "quill")]])
    mask = np.ones_like(ids)
    mask[0, 2] = 0
    (logits,) = session.run(["logits"], {"input_ids": ids, "attention_mask": mask})

    assert logits.shape == (1, 3, len(LABELS))
    assert [LABELS[index] for index in logits[0].argmax(-1)] == ["O", "B-NAME", "O"]


def test_tokenizers_encodes_the_smoke_text_with_expected_offsets(
    tmp_path: Path,
) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(build_tokenizer()), encoding="utf-8")

    encoding = tokenizers.Tokenizer.from_file(str(path)).encode(SMOKE_TEXT)

    assert encoding.ids == [VOCAB.index("[CLS]"), *range(4, len(VOCAB)), 3]
    assert encoding.offsets[1:-1] == [
        (start, end) for _, start, end in tokenize(SMOKE_TEXT)
    ]
