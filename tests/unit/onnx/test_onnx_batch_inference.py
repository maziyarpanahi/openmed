"""Offline regressions for real tensor batches and complete source coverage."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openmed.onnx.inference import OnnxModel, _TokenizersTokenizerAdapter

VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "Anna": 4,
    "Maria": 5,
    "Müller": 6,
    "Befund": 7,
    "😀": 8,
    "Meier": 9,
    "Müller": 10,
}


class TensorSession:
    """Classify known synthetic tokens while recording actual runtime batches."""

    def __init__(self, *, fixed_batch=None, fixed_length=None):
        self.calls = []
        self.shape = [fixed_batch or "batch", fixed_length or "sequence"]

    def get_inputs(self):
        return [
            SimpleNamespace(name=name, type="tensor(int64)", shape=self.shape)
            for name in ("input_ids", "attention_mask", "token_type_ids")
        ]

    def get_outputs(self):
        return [SimpleNamespace(name="logits")]

    def run(self, output_names, feed):
        self.calls.append({key: value.copy() for key, value in feed.items()})
        ids = feed["input_ids"]
        labels = np.zeros_like(ids)
        labels[np.isin(ids, [4, 9])] = 1
        labels[np.isin(ids, [5, 6, 10])] = 2
        logits = np.full((*ids.shape, 3), -8.0, dtype=np.float32)
        np.put_along_axis(logits, labels[..., None], 8.0, axis=-1)
        return [logits]


def artifact(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(VOCAB, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tmp_path.mkdir(exist_ok=True)
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "id2label": {"0": "O", "1": "B-PERSON", "2": "I-PERSON"},
                "max_position_embeddings": 512,
                "pad_token_id": 0,
            }
        )
    )
    (tmp_path / "model.onnx").write_bytes(b"session injected in unit tests")
    return tmp_path, tokenizer


def model_fixture(tmp_path, session=None):
    pytest.importorskip("onnxruntime")
    path, tokenizer = artifact(tmp_path)
    session = session or TensorSession()
    model = OnnxModel(
        path,
        variant="fp32",
        session=session,
        tokenizer=_TokenizersTokenizerAdapter(tokenizer),
    )
    return model, session


def spans(entities):
    return [
        (entity.label, entity.start, entity.end, entity.text) for entity in entities
    ]


def test_execution_control_is_shared_across_all_token_windows(tmp_path):
    from openmed.onnx.execution import OnnxExecutionCancelled, OnnxExecutionControl

    control = OnnxExecutionControl()

    class CancelAfterFirstTensor(TensorSession):
        def run(self, names, feed, options=None):
            result = super().run(names, feed)
            control.cancel()
            return result

    model, session = model_fixture(tmp_path, CancelAfterFirstTensor())
    with pytest.raises(OnnxExecutionCancelled):
        model.predict_batch_detailed(
            ["Befund " * 50 + "Anna Müller"],
            max_length=8,
            stride=2,
            batch_size=1,
            execution_control=control,
        )
    assert len(session.calls) == 1


def test_tensor_batch_padding_order_and_single_parity(tmp_path):
    model, session = model_fixture(tmp_path)
    texts = ["Befund Anna Maria Müller", "Meier", "😀 Anna Müller", "Befund"]
    batch = model.predict_batch(texts, batch_size=4)
    assert len(session.calls) == 1
    feed = session.calls[0]
    assert feed["input_ids"].shape == (4, 6)
    assert feed["attention_mask"].sum(axis=1).tolist() == [3, 3, 5, 6]
    assert np.all(feed["input_ids"][feed["attention_mask"] == 0] == 0)
    assert np.all(feed["token_type_ids"] == 0)
    assert [spans(items) for items in batch] == [
        [("PERSON", 7, 24, "Anna Maria Müller")],
        [("PERSON", 0, 5, "Meier")],
        [("PERSON", 2, 13, "Anna Müller")],
        [],
    ]
    assert [spans(model.predict(text)) for text in texts] == [
        spans(items) for items in batch
    ]


@pytest.mark.parametrize("stride", [0, 1, 3])
def test_identifier_across_windows_and_tail_are_complete(tmp_path, stride):
    model, session = model_fixture(tmp_path)
    text = "Befund " * 4 + "Anna Maria Müller " + "Befund " * 20 + "Meier"
    result = model.predict_batch_detailed(
        [text],
        max_length=8,
        stride=stride,
        batch_size=4,
        max_batch_tokens=24,
    )[0]
    assert spans(result.entities) == [
        ("PERSON", 28, 45, "Anna Maria Müller"),
        ("PERSON", len(text) - 5, len(text), "Meier"),
    ]
    assert result.complete
    assert result.token_count == result.processed_tokens == 28
    assert result.window_count > 1
    assert all(feed["input_ids"].size <= 24 for feed in session.calls)
    assert all(feed["input_ids"].shape[1] <= 8 for feed in session.calls)


def test_unicode_offsets_and_concurrent_window_settings(tmp_path):
    model, _ = model_fixture(tmp_path)
    texts = ["😀\r\nAnna Müller", "Befund " * 12 + "Anna Müller"]

    def predict(index):
        text = texts[index % 2]
        return spans(model.predict(text, max_length=8 if index % 2 else 16, stride=2))

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(predict, range(20)))
    assert results[0] == [("PERSON", 3, 15, "Anna Müller")]
    assert results[1] == [("PERSON", 84, 95, "Anna Müller")]
    assert all(result == results[index % 2] for index, result in enumerate(results))
    assert model.tokenizer._tokenizer.truncation is None


def test_static_axes_use_safe_single_batches_and_padding(tmp_path):
    model, session = model_fixture(
        tmp_path, TensorSession(fixed_batch=1, fixed_length=8)
    )
    assert [len(items) for items in model.predict_batch(["Meier", "Befund"])] == [1, 0]
    assert [feed["input_ids"].shape for feed in session.calls] == [(1, 8), (1, 8)]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"max_length": 513}, "model token limit"),
        ({"batch_size": True}, "positive integer"),
        ({"stride": 6, "max_length": 8}, "content window"),
        ({"max_batch_tokens": 2}, "max_batch_tokens"),
        ({"max_windows": 1, "max_length": 4, "stride": 0}, "max_windows"),
    ],
)
def test_invalid_budgets_cannot_produce_partial_success(tmp_path, options, message):
    model, _ = model_fixture(tmp_path)
    with pytest.raises(ValueError, match=message):
        model.predict_batch_detailed(["Befund Anna Maria Müller"], **options)


@pytest.mark.parametrize(
    "bad_output",
    [
        np.zeros((1, 3, 4)),
        np.full((1, 3, 3), np.nan),
        np.zeros((1, 2, 3)),
    ],
)
def test_bad_runtime_output_fails_closed(tmp_path, monkeypatch, bad_output):
    model, session = model_fixture(tmp_path)
    monkeypatch.setattr(session, "run", lambda *args: [bad_output])
    with pytest.raises(RuntimeError, match="invalid shape or non-finite"):
        model.predict("Meier")


def test_backend_passes_batch_limits_to_runtime_once(tmp_path):
    from openmed.core.backends import OnnxTokenClassificationPipeline

    model, session = model_fixture(tmp_path)
    pipeline = OnnxTokenClassificationPipeline(model)
    result = pipeline(["Meier", "Anna Müller"], batch_size=4, max_batch_tokens=32)
    assert len(session.calls) == 1
    assert result[0][0]["word"] == "Meier"
    assert result[1][0]["word"] == "Anna Müller"


def test_real_runtime_nested_graph_external_weights_and_symlink(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    path, _ = artifact(tmp_path / "snapshot")
    (path / "model.onnx").unlink()
    graph_dir = path / "onnx"
    graph_dir.mkdir()
    weights = np.full((len(VOCAB), 3), -8, dtype=np.float32)
    weights[:, 0] = 8
    for token_id, label in [(4, 1), (5, 2), (6, 2), (9, 1), (10, 2)]:
        weights[token_id] = -8
        weights[token_id, label] = 8
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["weights", "input_ids"], ["logits"])],
        "synthetic_token_classifier",
        [
            onnx.helper.make_tensor_value_info(
                "input_ids", onnx.TensorProto.INT64, ["b", "s"]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, ["b", "s", 3]
            )
        ],
        [onnx.numpy_helper.from_array(weights, name="weights")],
    )
    proto = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    proto.ir_version = 10
    onnx.save_model(
        proto,
        str(graph_dir / "model.onnx"),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    # Mimic a Hub snapshot, where graph and external data are separate symlinks.
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    for index, filename in enumerate(["model.onnx", "model.onnx.data"]):
        blob = blobs / f"blob-{index}"
        (graph_dir / filename).rename(blob)
        (graph_dir / filename).symlink_to(blob)
    for source in [path, graph_dir / "model.onnx"]:
        model = OnnxModel.from_pretrained(source, variant="fp32")
        outputs = model.predict_batch(
            ["Meier", "Befund " * 12 + "Anna Müller"], max_length=8
        )
        assert outputs[0][0].text == "Meier"
        assert outputs[1][0].text == "Anna Müller"
        assert model.variant == "fp32"


def test_invalid_label_index_map_rejected_before_runtime(tmp_path):
    path, _ = artifact(tmp_path)
    (path / "id2label.json").write_text(json.dumps({"0": "O", "2": "B-PERSON"}))
    with pytest.raises(ValueError, match="contiguous"):
        OnnxModel(path, session=TensorSession())
