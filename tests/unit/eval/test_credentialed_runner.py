"""Exercise the public credentialed command using synthetic records."""

import json
import socket

from openmed.cli.main import main
from openmed.eval import credentialed


def _run(output):
    return main(
        [
            "eval",
            "credentialed",
            "--suite",
            "i2b2",
            "--year",
            "2014",
            "--model",
            "example/test",
            "--output",
            str(output),
        ]
    )


def test_offline_aggregate_only(tmp_path, monkeypatch):
    corpus = tmp_path / "input"
    corpus.mkdir()
    for i in range(6):
        (corpus / f"private-{i}.xml").write_text(
            '<ROOT><TEXT>Example Name</TEXT><TAGS><NAME TYPE="PATIENT" '
            'start="0" end="12"/></TAGS></ROOT>'
        )
    monkeypatch.setenv("OPENMED_I2B2_PATH", str(corpus))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: 1 / 0)
    monkeypatch.setattr(
        credentialed,
        "_load_cached_model",
        lambda *args: (
            "a" * 64,
            lambda *args: [{"start": 0, "end": 12, "label": "PERSON"}],
        ),
    )
    before = set(tmp_path.rglob("*"))
    output = tmp_path / "result.json"
    assert _run(output) == 0
    assert set(tmp_path.rglob("*")) - before == {output}
    assert output.stat().st_mode & 0o077 == 0
    raw = output.read_text()
    assert not any(
        v in raw
        for v in ("Example Name", "private-", str(corpus), "fixture_id", "start")
    )
    report = json.loads(raw)
    assert report["by_label"]["PERSON"]["tp"] == 6
    assert report["revision_digest"] == "a" * 64
    assert report["dua"] == "i2b2/DBMI DUA"
    assert report["model_id"] == "example/test"
    assert _run(output) == 2
    assert output.read_text() == raw


def test_worktree_and_symlink_refused_before_model(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        credentialed, "_load_cached_model", lambda *args: called.append(args)
    )
    repo = tmp_path / "linked"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /example")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    for target in ("inside-checkout.json", repo / "out", alias / "out"):
        assert _run(target) == 2
    assert called == []


def test_missing_model_does_not_download(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(socket.socket, "connect", lambda *a: 1 / 0)
    assert _run(tmp_path / "out") == 2
    assert "model_not_cached" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_inference_error_contains_no_source(tmp_path, monkeypatch, capsys):
    (tmp_path / "private.xml").write_text("<ROOT><TEXT>sample</TEXT><TAGS/></ROOT>")
    monkeypatch.setenv("OPENMED_I2B2_PATH", str(tmp_path))

    def fail(*args):
        raise ValueError("private payload")

    monkeypatch.setattr(
        credentialed, "_load_cached_model", lambda *args: ("a" * 64, fail)
    )
    assert _run(tmp_path / "out") == 2
    diagnostic = capsys.readouterr().err
    assert "credentialed_evaluation_failed" in diagnostic
    assert "private" not in diagnostic
    assert not (tmp_path / "out").exists()


def test_format_comparison_scores_one_corpus_and_suppresses_small_cells(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "private"
    corpus.mkdir()
    for i in range(6):
        age = '<AGE TYPE="AGE" start="13" end="15"/>' if i == 0 else ""
        (corpus / f"phi-{i}.xml").write_text(
            '<ROOT><TEXT>Example Name 99</TEXT><TAGS><NAME TYPE="PATIENT" '
            f'start="0" end="12"/>{age}</TAGS></ROOT>'
        )
    monkeypatch.setenv("OPENMED_I2B2_PATH", str(corpus))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: 1 / 0)
    monkeypatch.setattr(
        credentialed,
        "_load_cached_model",
        lambda *_: ("a" * 64, lambda *_: [{"label": "PERSON", "start": 0, "end": 12}]),
    )

    def load_variant(name, *_args):
        if name == "onnx-fp32":
            raise credentialed.CredentialedEvalError("model_not_cached")
        seen = 0

        def predict(*_):
            nonlocal seen
            seen += 1
            return [{"label": "PERSON", "start": 0, "end": 12}] if seen <= 3 else []

        return "b" * 64, predict

    monkeypatch.setattr(credentialed, "_load_format", load_variant)
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "eval",
                "credentialed",
                "--suite",
                "i2b2",
                "--year",
                "2014",
                "--model",
                "org/example",
                "--format",
                "onnx-int8",
                "--format",
                "onnx-fp32",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = json.loads(output.read_bytes())
    assert report["reference_format"] == "safetensors"
    candidate = report["formats"]["onnx-int8"]
    assert candidate["revision_digest"] == "b" * 64
    assert candidate["recall_delta_by_label"] == {"PERSON": -0.5}
    assert candidate["by_label"]["AGE"] == {"status": "suppressed"}
    assert candidate["by_label"]["PERSON"]["tp"] == 3
    assert report["formats"]["onnx-fp32"] == {
        "status": "skipped",
        "reason": "model_not_cached",
    }
    assert (
        "Example Name" not in output.read_text()
        and str(corpus) not in output.read_text()
    )


def test_native_onnx_selection_verifies_the_artifact_parent(tmp_path, monkeypatch):
    import hashlib

    import pytest

    from openmed.onnx import inference

    root = tmp_path / ("b" * 40)
    root.mkdir()
    source_pin = "a" * 40
    reference = hashlib.sha256(source_pin.encode()).hexdigest()
    manifest = root / "openmed-onnx.json"
    monkeypatch.setattr(credentialed, "_cached_snapshot", lambda *_: root)
    with pytest.raises(
        credentialed.CredentialedEvalError, match="source_revision_unverified"
    ):
        credentialed._load_format(
            "onnx-int8", "org/export", root.name, "org/source", reference
        )
    manifest.write_text(
        json.dumps(
            {
                "provenance": {
                    "source": {"model_id": "org/source", "revision": source_pin}
                }
            }
        )
    )
    with pytest.raises(credentialed.CredentialedEvalError, match="model_not_cached"):
        credentialed._load_format(
            "onnx-int8", "org/export", root.name, "org/source", reference
        )
    (root / "model_int8.onnx").touch()
    selected = []

    class Engine:
        def __init__(self, path, *, variant):
            selected.append((path, variant))

    monkeypatch.setattr(inference, "OnnxModel", Engine)
    digest, runner = credentialed._load_format(
        "onnx-int8", "org/export", root.name, "org/source", reference
    )
    assert selected == [(root, "int8")] and callable(runner)
    assert digest == hashlib.sha256(root.name.encode()).hexdigest()


def test_mlx_overlap_is_decoded_once_in_document_coordinates(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    import numpy as np

    parent = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.array, core.float32 = np.array, np.float32
    core.softmax = lambda arr, axis: arr
    parent.core = core
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    class Tokenizer:
        def num_special_tokens_to_add(self):
            return 2

        def __call__(self, text, *, stride, **kwargs):
            assert stride == 1 and kwargs["return_overflowing_tokens"]
            return {
                "input_ids": [[0, 1, 2, 0], [0, 2, 3, 0]],
                "attention_mask": [[1] * 4] * 2,
                "special_tokens_mask": [[1, 0, 0, 1]] * 2,
                "offset_mapping": [
                    [(0, 0), (0, 1), (1, 2), (0, 0)],
                    [(0, 0), (1, 2), (2, 3), (0, 0)],
                ],
            }

    def model(ids, attention_mask):
        return np.stack((ids == 0, ids > 0), axis=-1)

    engine = SimpleNamespace(
        tokenizer=Tokenizer(),
        max_length=4,
        model=model,
        _decode_grouped=lambda labels, probs, offsets, text: (labels, offsets, text),
    )
    assert credentialed._mlx_spans(engine, "abc") == (
        [1, 1, 1],
        [(0, 1), (1, 2), (2, 3)],
        "abc",
    )
