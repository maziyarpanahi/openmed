"""Weight-free CTC contract checks and optional real CPU inference on a toy graph."""

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openmed.multimodal.streaming.speech.providers import (
    AudioChunk,
    SpeechProviderError,
    StreamState,
)
from openmed.onnx import streaming_asr as asr


def profile(**kwargs):
    labels = ("", "X", "Y", " ")
    return replace(
        asr.OnnxSpeechProfile(
            model_sha256="a" * 64,
            labels=labels,
            vocabulary_sha256=asr.vocabulary_digest(labels),
            frames=8,
            language="en",
        ),
        **kwargs,
    )


def chunk(seq, ms):
    payload = (
        np.round(12000 * np.sin(2 * np.pi * 440 * np.arange(16 * ms) / 16000))
        .astype("<i2")
        .tobytes()
    )
    return AudioChunk(seq, payload, 16000, 1, ms)


class Fake:
    def options(self):
        return SimpleNamespace(terminate=False)

    def run(self, x, option):
        assert x.shape == (1, 8, 40) and x.dtype == np.float32
        scores = np.full((1, 8, 4), -3, dtype=np.float32)
        for i, label in enumerate([1, 1, 0, 1, 2, 2, 3, 3]):
            scores[0, i, label] = 3
        return scores


def test_synthetic_tone_matches_independent_scalar_spectrum():
    # Independent scalar DFT and triangle accumulation, no FFT or matrix product.
    import cmath
    import math

    wav = chunk(0, 30).payload
    pcm = np.frombuffer(wav, dtype="<i2")[:400].tolist()
    spectrum = []
    for bin_ in range(257):
        z = sum(
            value
            / 32768
            * (0.5 - 0.5 * math.cos(2 * math.pi * j / 400))
            * cmath.exp(-2j * math.pi * bin_ * j / 512)
            for j, value in enumerate(pcm)
        )
        spectrum.append(abs(z) ** 2 / 512)
    edges = [700 * (10 ** (i / 41 * math.log10(1 + 8000 / 700)) - 1) for i in range(42)]
    reference = []
    for lo, mid, hi in zip(edges, edges[1:], edges[2:]):
        power = sum(
            p
            * max(
                0,
                min(
                    (k * 16000 / 512 - lo) / (mid - lo),
                    (hi - k * 16000 / 512) / (hi - mid),
                ),
            )
            for k, p in enumerate(spectrum)
        )
        reference.append(math.log(max(power, 1e-10)))
    actual = asr.log_mel_features(wav, 8)
    np.testing.assert_allclose(actual[0, 0], reference, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(actual[0, 3:], np.log(1e-10), rtol=1e-6)


def test_partial_final_repeats_blank_timing_and_no_padded_hallucinations():
    s = asr._OnnxSpeechStream(profile(), Fake())
    first = s.feed(chunk(0, 24))
    assert first.hypothesis.text == "X" and first.hypothesis.token_times[0].end_ms == 20
    second = s.feed(chunk(1, 31), end_segment=True)
    assert second.hypothesis.text == "XXY"
    assert [(t.start_ms, t.end_ms) for t in second.hypothesis.token_times] == [
        (0, 20),
        (30, 40),
        (40, 55),
    ]
    assert all(
        990000 < t.confidence_ppm < 1000000 for t in second.hypothesis.token_times
    )
    assert second.hypothesis.start_ms == 0 and second.hypothesis.end_ms == 55
    s.feed(chunk(2, 10))
    last, report = s.finish()
    assert last.hypothesis.text == "X" and last.hypothesis.start_ms == 55
    assert report.state == StreamState.FINALIZED.value and report.buffered_chunks == 0
    assert "XXY" not in repr(second) and "XXY" not in second.envelope.to_json()
    assert (
        json.loads(second.envelope.to_json())["count_metadata"]["sample_count"] == 880
    )
    assert (
        profile().capability()["tier"] == "experimental"
        and not profile().capability()["qualified"]
    )


def test_limits_before_dispatch_and_value_free_failure():
    class F(Fake):
        def run(self, x, option):
            raise ValueError("PRIVATE SYNTHETIC TRANSCRIPT")

    s = asr._OnnxSpeechStream(profile(), F())
    with pytest.raises(SpeechProviderError):
        s.feed(chunk(0, 81))
    with pytest.raises(asr.OnnxSpeechError, match="pcm_profile_mismatch"):
        s.feed(AudioChunk(0, b"1234", 16000, 1, 20))
    with pytest.raises(asr.OnnxSpeechError, match="decode_failed") as err:
        s.feed(chunk(0, 50))
    assert err.value.__context__ is None and s.report().buffered_byte_count == 0
    with pytest.raises(asr.OnnxSpeechError, match="closed"):
        s.finish()


def test_cancel_fences_late_success_and_requests_native_termination():
    started, finish = threading.Event(), threading.Event()

    class Blocking(Fake):
        def run(self, x, option):
            started.set()
            assert finish.wait(5) and option.terminate
            return super().run(x, option)

    s = asr._OnnxSpeechStream(profile(), Blocking())
    errors = []

    def worker():
        try:
            s.feed(chunk(0, 40))
        except asr.OnnxSpeechError as e:
            errors.append(str(e))

    th = threading.Thread(target=worker)
    th.start()
    try:
        assert started.wait(5)
        with pytest.raises(asr.OnnxSpeechError, match="busy"):
            s.feed(chunk(1, 5))
        assert s.cancel().buffered_byte_count == 0
    finally:
        finish.set()
        th.join(5)
    assert errors == ["stream_cancelled"]


def test_missing_digest_and_vocabulary_fail_before_parser_or_native_session(
    tmp_path, monkeypatch
):
    def forbidden(*args):
        pytest.fail("unexpected parser/runtime access")

    monkeypatch.setattr(asr, "_CpuEngine", forbidden)
    monkeypatch.setattr(asr, "_inspect_graph", forbidden)
    for bad in ("https://private.invalid/x", Path("relative.onnx")):
        with pytest.raises(asr.OnnxSpeechError, match="local_graph_required"):
            asr.OnnxSpeechModel.load(bad, profile())
    path = tmp_path / "private-identifier.onnx"
    with pytest.raises(asr.OnnxSpeechError, match="graph_unavailable") as err:
        asr.OnnxSpeechModel.load(path, profile())
    assert err.value.__context__ is None
    path.write_bytes(b"synthetic model")
    with pytest.raises(asr.OnnxSpeechError, match="digest_mismatch"):
        asr.OnnxSpeechModel.load(path, profile())
    with pytest.raises(asr.OnnxSpeechError, match="vocabulary_digest"):
        profile(vocabulary_sha256="0" * 64)


def graph_blob(*, wrong=False, external=False):
    onnx = pytest.importorskip("onnx")
    h = onnx.helper
    inp = h.make_tensor_value_info(
        "features", onnx.TensorProto.FLOAT, [1, 8, 41 if wrong else 40]
    )
    out = h.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [1, 8, 4])
    # Deterministic small graph actually consumes the synthetic feature tensor.
    w = onnx.numpy_helper.from_array(np.zeros((40, 4), dtype=np.float32), name="W")
    b = onnx.numpy_helper.from_array(
        np.array([0, 5, 0, 0], dtype=np.float32), name="bias"
    )
    if external:
        onnx.external_data_helper.set_external_data(w, location="outside.bin")
        w.ClearField("raw_data")
    g = h.make_graph(
        [
            h.make_node("MatMul", ["features", "W"], ["p"]),
            h.make_node("Add", ["p", "bias"], ["logits"]),
        ],
        "synthetic",
        [inp],
        [out],
        [w, b],
    )
    return h.make_model(
        g, ir_version=9, opset_imports=[h.make_opsetid("", 17)]
    ).SerializeToString()


@pytest.mark.parametrize("variant", ["wrong", "external"])
def test_invalid_signature_and_external_data_never_create_session(
    tmp_path, monkeypatch, variant
):
    blob = graph_blob(**{variant: True})
    p = tmp_path / "bad.onnx"
    p.write_bytes(blob)
    monkeypatch.setattr(
        asr, "_CpuEngine", lambda b: pytest.fail("must fail before runtime creation")
    )
    with pytest.raises(
        asr.OnnxSpeechError, match="graph_signature_mismatch|external_graph_data"
    ):
        asr.OnnxSpeechModel.load(
            p, profile(model_sha256=hashlib.sha256(blob).hexdigest())
        )


@pytest.mark.integration
def test_real_cpu_session_from_verified_bytes_offline(tmp_path):
    pytest.importorskip("onnxruntime")
    blob = graph_blob()
    p = tmp_path / "synthetic.onnx"
    p.write_bytes(blob)
    model = asr.OnnxSpeechModel.load(
        p, profile(model_sha256=hashlib.sha256(blob).hexdigest())
    )
    s = model.open_stream()
    result = s.feed(chunk(0, 55), end_segment=True)
    assert (
        result.hypothesis.text == "X" and result.hypothesis.token_times[0].end_ms == 55
    )
    assert s.finish()[1].finalized_segments == 1
