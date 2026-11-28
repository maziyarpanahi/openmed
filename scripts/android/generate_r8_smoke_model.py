#!/usr/bin/env python3
"""Generate the tiny synthetic model bundled by the R8-minified consumer smoke app.

The ``android/consumer-r8-smoke`` app runs OpenMedKit inside an R8-minified
release APK. It needs a real ONNX token classifier and a Hugging Face
``tokenizer.json`` so that tokenizer construction, ONNX Runtime session
creation and decoding all execute, without downloading a checkpoint.

The model is a hand-built lookup table, not a trained network: logits are
``Gather(table, input_ids) * attention_mask``, so every vocabulary token maps to
one fixed label. It is encoded with a minimal protobuf writer, so the Gradle
build needs no ONNX tooling and the output is byte-for-byte deterministic. The
bundle is generated into the build directory at build time and never committed.

Every string in the bundle is synthetic. The smoke case records the expected
labels and offsets; the app reports only those, never the text.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path
from typing import Sequence

MODEL_FILE = "model.onnx"
TOKENIZER_FILE = "tokenizer.json"
ID2LABEL_FILE = "id2label.json"
SMOKE_CASE_FILE = "smoke_case.json"
BUNDLE_FILES = (MODEL_FILE, TOKENIZER_FILE, ID2LABEL_FILE, SMOKE_CASE_FILE)

SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
WORDS = ("patient", "avery", "quill", "visited", "the", "clinic", ".")
VOCAB = SPECIAL_TOKENS + WORDS
LABELS = ("O", "B-NAME", "I-NAME")
TOKEN_LABELS = {"avery": "B-NAME", "quill": "I-NAME"}
LABEL_LOGIT = 4.0

SMOKE_TEXT = "Patient Avery Quill visited the clinic."
SMOKE_POLICY = "hipaa_safe_harbor"
CHUNK_TOKEN_LIMIT = 4
CHUNK_TOKEN_OVERLAP = 1

IR_VERSION = 7
OPSET_VERSION = 13
PRODUCER_NAME = "openmed-r8-smoke"

# ONNX TensorProto.DataType and AttributeProto.AttributeType values.
ONNX_FLOAT = 1
ONNX_INT64 = 7
ATTRIBUTE_INT = 2

WORD_PATTERN = re.compile(r"\w+|[^\w\s]+")


def _varint(value: int) -> bytes:
    """Encode a non-negative integer as a protobuf varint."""
    if value < 0:
        raise ValueError("negative varints are not used by this encoder")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _key(field: int, wire_type: int) -> bytes:
    return _varint((field << 3) | wire_type)


def _int_field(field: int, value: int) -> bytes:
    return _key(field, 0) + _varint(value)


def _bytes_field(field: int, value: bytes) -> bytes:
    return _key(field, 2) + _varint(len(value)) + value


def _str_field(field: int, value: str) -> bytes:
    return _bytes_field(field, value.encode("utf-8"))


def _tensor(name: str, data_type: int, dims: Sequence[int], raw: bytes) -> bytes:
    """Encode a ``TensorProto`` with little-endian ``raw_data``."""
    body = b"".join(_int_field(1, dim) for dim in dims)
    body += _int_field(2, data_type) + _str_field(8, name) + _bytes_field(9, raw)
    return body


def _value_info(name: str, elem_type: int, dims: Sequence[int | str]) -> bytes:
    """Encode a ``ValueInfoProto`` for a tensor with symbolic or fixed dims."""
    shape = b""
    for dim in dims:
        dimension = _str_field(2, dim) if isinstance(dim, str) else _int_field(1, dim)
        shape += _bytes_field(1, dimension)
    tensor_type = _int_field(1, elem_type) + _bytes_field(2, shape)
    return _str_field(1, name) + _bytes_field(2, _bytes_field(1, tensor_type))


def _node(
    op_type: str,
    inputs: Sequence[str],
    outputs: Sequence[str],
    name: str,
    int_attributes: dict[str, int] | None = None,
) -> bytes:
    """Encode a ``NodeProto`` in the default ONNX domain."""
    body = b"".join(_str_field(1, value) for value in inputs)
    body += b"".join(_str_field(2, value) for value in outputs)
    body += _str_field(3, name) + _str_field(4, op_type)
    for attribute, value in sorted((int_attributes or {}).items()):
        encoded = _str_field(1, attribute) + _int_field(3, value)
        encoded += _int_field(20, ATTRIBUTE_INT)
        body += _bytes_field(5, encoded)
    return body


def logits_table() -> list[list[float]]:
    """Return the ``[vocab, labels]`` logits table used by the lookup model."""
    table = []
    for token in VOCAB:
        row = [0.0] * len(LABELS)
        row[LABELS.index(TOKEN_LABELS.get(token, "O"))] = LABEL_LOGIT
        table.append(row)
    return table


def build_model() -> bytes:
    """Return the serialized ONNX ``ModelProto`` of the lookup classifier."""
    table = logits_table()
    raw_table = b"".join(struct.pack("<f", value) for row in table for value in row)
    nodes = [
        _node("Gather", ["logit_table", "input_ids"], ["token_logits"], "lookup"),
        _node("Cast", ["attention_mask"], ["mask_float"], "mask_cast", {"to": 1}),
        _node("Unsqueeze", ["mask_float", "mask_axes"], ["mask_3d"], "mask_axes"),
        _node("Mul", ["token_logits", "mask_3d"], ["logits"], "apply_mask"),
    ]
    initializers = [
        _tensor("logit_table", ONNX_FLOAT, [len(VOCAB), len(LABELS)], raw_table),
        _tensor("mask_axes", ONNX_INT64, [1], struct.pack("<q", -1)),
    ]
    graph = b"".join(_bytes_field(1, node) for node in nodes)
    graph += _str_field(2, "openmed_r8_smoke_lookup")
    graph += b"".join(_bytes_field(5, tensor) for tensor in initializers)
    graph += _bytes_field(11, _value_info("input_ids", ONNX_INT64, ["batch", "seq"]))
    graph += _bytes_field(
        11, _value_info("attention_mask", ONNX_INT64, ["batch", "seq"])
    )
    graph += _bytes_field(
        12, _value_info("logits", ONNX_FLOAT, ["batch", "seq", len(LABELS)])
    )
    opset = _str_field(1, "") + _int_field(2, OPSET_VERSION)
    return (
        _int_field(1, IR_VERSION)
        + _str_field(2, PRODUCER_NAME)
        + _bytes_field(7, graph)
        + _bytes_field(8, opset)
    )


def build_tokenizer() -> dict[str, object]:
    """Return a lowercase WordLevel ``tokenizer.json`` with BERT-style specials."""
    added_tokens = [
        {
            "id": index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for index, token in enumerate(SPECIAL_TOKENS)
    ]
    cls_id, sep_id = VOCAB.index("[CLS]"), VOCAB.index("[SEP]")
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added_tokens,
        "normalizer": {"type": "Lowercase"},
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            ],
            "pair": [
                {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                {"Sequence": {"id": "A", "type_id": 0}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                {"Sequence": {"id": "B", "type_id": 1}},
                {"SpecialToken": {"id": "[SEP]", "type_id": 1}},
            ],
            "special_tokens": {
                "[CLS]": {"id": "[CLS]", "ids": [cls_id], "tokens": ["[CLS]"]},
                "[SEP]": {"id": "[SEP]", "ids": [sep_id], "tokens": ["[SEP]"]},
            },
        },
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": {token: index for index, token in enumerate(VOCAB)},
            "unk_token": "[UNK]",
        },
    }


def build_id2label() -> dict[str, str]:
    """Return the ``id2label.json`` mapping for the lookup classifier."""
    return {str(index): label for index, label in enumerate(LABELS)}


def tokenize(text: str) -> list[tuple[str, int, int]]:
    """Split *text* like the Lowercase + Whitespace tokenizer, with offsets."""
    return [
        (match.group().lower(), match.start(), match.end())
        for match in WORD_PATTERN.finditer(text)
    ]


def reference_entities(text: str) -> list[dict[str, object]]:
    """Return the BIO entities the lookup model predicts for *text*.

    This mirrors the OpenMedKit aggregation for the label set used here and is
    the single source of the expected spans written to the smoke case.
    """
    entities: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for token, start, end in tokenize(text):
        label = TOKEN_LABELS.get(token, "O")
        if label == "O":
            current = None
            continue
        prefix, entity_label = label.split("-", maxsplit=1)
        if prefix == "I" and current is not None and current["label"] == entity_label:
            current["end"] = end
        else:
            current = {"label": entity_label, "start": start, "end": end}
            entities.append(current)
    return entities


def build_smoke_case() -> dict[str, object]:
    """Return the expected smoke-run outcome consumed by the Android app."""
    return {
        "schema_version": 1,
        "text": SMOKE_TEXT,
        "entities": reference_entities(SMOKE_TEXT),
        "chunked": {
            "chunk_token_limit": CHUNK_TOKEN_LIMIT,
            "token_overlap": CHUNK_TOKEN_OVERLAP,
        },
        "deidentify": {"policy": SMOKE_POLICY, "canonical_labels": ["PERSON"]},
    }


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def bundle_contents() -> dict[str, bytes]:
    """Return every bundle file name mapped to its deterministic bytes."""
    return {
        MODEL_FILE: build_model(),
        TOKENIZER_FILE: _json_bytes(build_tokenizer()),
        ID2LABEL_FILE: _json_bytes(build_id2label()),
        SMOKE_CASE_FILE: _json_bytes(build_smoke_case()),
    }


def write_bundle(output_dir: Path) -> list[Path]:
    """Write the bundle into *output_dir* and return the written paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, payload in bundle_contents().items():
        path = output_dir / name
        path.write_bytes(payload)
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory that receives model.onnx, tokenizer.json, id2label.json "
        "and smoke_case.json",
    )
    args = parser.parse_args(argv)
    written = write_bundle(args.output_dir)
    print(f"wrote {len(written)} synthetic R8 smoke model files", file=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
