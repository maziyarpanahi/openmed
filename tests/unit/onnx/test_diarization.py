"""Worker lifecycle, source binding and fail-closed local artifacts."""

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from openmed.multimodal.streaming.speech.diarization import (
    DiarizationError,
    DiarizationRequest,
)
from openmed.onnx import diarization as backend


def artifact():
    return backend.LocalDiarizationArtifact(
        path=Path(__file__).resolve().with_name("private-model.onnx"),
        sha256="a" * 64,
        revision_digest="b" * 64,
        terms_digest="c" * 64,
    )


def request(revision=0):
    return DiarizationRequest(
        pcm=b"\0\0" * 16000, revision=revision, offset_samples=0, request_ref="1" * 32
    )


def provider():
    return backend.SherpaDiarizationProvider(
        segmentation=artifact(), embedding=artifact()
    )


def test_adapter_scopes_clusters_and_provenance_without_roles(monkeypatch):
    observed = []

    def exchange(command, payload, limits, cancelled):
        observed.append((command, payload))
        return b'{"turns":[[0,8000,0,false],[4000,16000,1,true]],"runtime":"sherpa-onnx:2.0.1"}'

    monkeypatch.setattr(backend, "_exchange", exchange)
    monkeypatch.setattr(sys, "platform", "linux")
    a, b = provider().observe(request()), provider().observe(request(1))
    assert {t.cluster_ref for t in a.turns}.isdisjoint(t.cluster_ref for t in b.turns)
    assert a.provenance_digest == b.provenance_digest
    assert a.lineage == ()
    model_name = artifact().path.name
    assert model_name not in repr(a) and model_name not in repr(artifact())
    command, payload = observed[0]
    assert command[0] == sys.executable and command[1] == "-I"
    size = int.from_bytes(payload[:4], "big")
    assert json.loads(payload[4 : 4 + size])["pcm_bytes"] == len(request().pcm)
    assert payload[4 + size :] == request().pcm


@pytest.mark.parametrize(
    "response",
    [
        b"private diagnostic",
        b"null",
        b'{"turns":[],"runtime":"unreviewed"}',
        b'{"turns":[[0,16001,0,false]],"runtime":"sherpa-onnx:2.0.1"}',
        b'{"turns":[[0,8000,32,false]],"runtime":"sherpa-onnx:2.0.1"}',
        b'{"turns":[[false,8000,0,false]],"runtime":"sherpa-onnx:2.0.1"}',
    ],
)
def test_adapter_rejects_malformed_or_unbounded_native_output(monkeypatch, response):
    monkeypatch.setattr(backend, "_exchange", lambda *args: response)
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(DiarizationError, match="^invalid_diarization_response$"):
        provider().observe(request())


@pytest.mark.parametrize(
    "changes",
    [
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": True},
        {"max_rss_bytes": 0},
    ],
)
def test_limits_are_validated(changes):
    with pytest.raises(DiarizationError):
        backend.DiarizationLimits(**changes)


def test_artifact_provenance_is_required_but_not_a_weight_license_gate():
    assert artifact().provenance()["terms_digest"] == "c" * 64
    with pytest.raises(DiarizationError):
        replace(artifact(), terms_digest="unknown")
    with pytest.raises(DiarizationError):
        replace(artifact(), path=Path("relative.onnx"))


@pytest.mark.skipif(os.name != "posix", reason="native adapter supports POSIX pipes")
def test_real_pipe_backpressure_and_suppressed_stderr(monkeypatch):
    pytest.importorskip("psutil")
    data = b"x" * 100000
    # Input exceeds pipe capacity. Parent must drain stdout while writing stdin.
    command = [
        sys.executable,
        "-I",
        "-c",
        "import sys;sys.stderr.write('private');sys.stdout.buffer.write(b'ack');sys.stdout.flush();a=sys.stdin.buffer.read();sys.stdout.buffer.write(str(len(a)).encode())",
    ]
    assert (
        backend._exchange(command, data, backend.DiarizationLimits(), Event())
        == b"ack100000"
    )


@pytest.mark.skipif(os.name != "posix", reason="native adapter supports POSIX pipes")
@pytest.mark.parametrize(
    "condition,code",
    [
        ("cancel", "cancelled"),
        ("timeout", "timeout"),
        ("rss", "memory_limit"),
        ("output", "output_limit"),
        ("exit", "worker_failed"),
    ],
)
def test_worker_terminated_and_reaped_on_failure(monkeypatch, condition, code):
    psutil = pytest.importorskip("psutil")
    real_popen = backend.subprocess.Popen
    workers = []

    def popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        workers.append(process)
        return process

    monkeypatch.setattr(backend.subprocess, "Popen", popen)
    event = Event()
    limits = backend.DiarizationLimits()
    program = "import sys,time;sys.stdin.buffer.read();time.sleep(60)"
    if condition == "timeout":
        times = iter([0, 1, 100])
        monkeypatch.setattr(backend.time, "monotonic", lambda: next(times, 100))
    if condition in {"rss", "cancel"}:
        real_info = psutil.Process.memory_info

        def info(self):
            result = real_info(self)
            if condition == "cancel":
                event.set()
            return result

        monkeypatch.setattr(psutil.Process, "memory_info", info)
        if condition == "rss":
            limits = backend.DiarizationLimits(max_rss_bytes=1)
    if condition == "output":
        program = (
            "import sys;sys.stdin.buffer.read();sys.stdout.buffer.write(b'x'*300000)"
        )
    if condition == "exit":
        program = "import sys;sys.exit(1)"
    with pytest.raises(DiarizationError, match="^diarization_" + code + "$"):
        backend._exchange([sys.executable, "-I", "-c", program], b"x", limits, event)
    assert workers and all(p.poll() is not None for p in workers)


def test_precancel_does_not_launch_worker(monkeypatch):
    pytest.importorskip("psutil")
    event = Event()
    event.set()
    monkeypatch.setattr(
        backend.subprocess, "Popen", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(DiarizationError, match="cancelled"):
        backend._exchange([], b"", backend.DiarizationLimits(), event)


def test_native_segmentation_frame_intersects_input_and_marks_boundary():
    from openmed.onnx._diarization_worker import _bounded_turn

    assert _bounded_turn(0.5, 1.025, 1, 16000) == [8000, 16000, 1, True]
    assert _bounded_turn(0.5, 0.75, 1, 16000) == [8000, 12000, 1, False]
    for start, end in [(1.1, 1.2), (-0.1, 0.5), (0.5, 0.4), (0.5, float("nan"))]:
        with pytest.raises(ValueError):
            _bounded_turn(start, end, 1, 16000)


def test_clipped_native_turn_is_explicitly_unstable(monkeypatch):
    monkeypatch.setattr(
        backend,
        "_exchange",
        lambda *args: b'{"turns":[[0,16000,0,true]],"runtime":"sherpa-onnx:2.0.1"}',
    )
    monkeypatch.setattr(sys, "platform", "linux")
    turn = provider().observe(request()).turns[0]
    assert turn.boundary_clipped
    assert turn.evidence.value == "unstable"
