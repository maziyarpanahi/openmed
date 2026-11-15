"""Actual worker crashes, limits, parser probes and value-free recovery controls."""

import hashlib
import json
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.multimodal import decoder_guard as guard
from tests.fixtures.multimodal.hostile_media import controls, dicom, pdf, tiff, video

pytestmark = pytest.mark.integration


def require_linux():
    if not guard.isolation_supported():
        pytest.skip("Hard worker isolation is supported on native 64-bit Linux only")


def test_unsupported_isolation_refuses_without_starting_worker(monkeypatch):
    monkeypatch.setattr(guard, "isolation_supported", lambda: False)
    monkeypatch.setattr(
        guard.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned")
    )
    report = guard.probe_media(b"synthetic", guard.DecoderKind.PDF)
    assert report.status is guard.DecodeStatus.ISOLATION
    assert not report.controls_applied and not report.accepted


def test_invalid_and_oversized_input_never_starts_worker(monkeypatch):
    monkeypatch.setattr(
        guard.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned")
    )
    report = guard.probe_media(
        b"1234", guard.DecoderKind.PDF, limits=guard.DecoderLimits(input_bytes=3)
    )
    assert report.status is guard.DecodeStatus.LIMIT and report.source_digest is None
    for payload in ("/private/patient.pdf", bytearray(b"synthetic")):
        with pytest.raises(
            guard.DecoderGuardError, match="^invalid_decoder_guard_input$"
        ):
            guard.probe_media(payload, guard.DecoderKind.PDF)


def test_precancel_and_failed_cancel_callback_never_start_worker(monkeypatch):
    monkeypatch.setattr(
        guard.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned")
    )

    def fail():
        raise RuntimeError("SYNTHETIC-PRIVATE")

    for cancel in (lambda: True, fail, lambda: "not bool"):
        report = guard.probe_media(
            b"synthetic", guard.DecoderKind.PDF, cancelled=cancel
        )
        assert report.status is guard.DecodeStatus.CANCELLED


@pytest.mark.parametrize("kind", list(guard.DecoderKind))
def test_actual_positive_controls_decode_or_explicitly_report_missing_backend(kind):
    require_linux()
    payload = controls()[kind.value]
    report = guard.probe_media(payload, kind)
    assert report.worker_reaped and report.controls_applied
    if report.status is guard.DecodeStatus.UNAVAILABLE:
        assert os.getenv("OPENMED_REQUIRE_DECODER_BACKENDS") != "1", report
        return
    assert report.accepted, report
    assert report.source_digest == hashlib.sha256(payload).hexdigest()
    assert not report.to_dict()["clinical_readiness"]


@pytest.mark.parametrize("kind", list(guard.DecoderKind))
def test_truncated_media_have_stable_fail_closed_categories(kind):
    require_linux()
    payload = controls()[kind.value][:12]
    first = guard.probe_media(payload, kind)
    second = guard.probe_media(payload, kind)
    assert first.status == second.status
    assert first.status in (
        guard.DecodeStatus.MALFORMED,
        guard.DecodeStatus.UNAVAILABLE,
    )
    assert first.worker_reaped and second.worker_reaped
    assert not first.accepted


@pytest.mark.parametrize(
    "kind,payload",
    [
        (guard.DecoderKind.TIFF, tiff(width=100000, height=100000)),
        (guard.DecoderKind.WSI, tiff(width=100000, height=100000)),
        (guard.DecoderKind.DICOM, dicom(oversized=True)),
        (guard.DecoderKind.VIDEO, video(oversized=True)),
        (guard.DecoderKind.PDF, pdf(recursive=True)),
        (guard.DecoderKind.TIFF, tiff(recursive=True)),
        (guard.DecoderKind.WSI, tiff(recursive=True)),
    ],
)
def test_oversized_geometry_and_recursive_containers_cannot_pass(kind, payload):
    require_linux()
    report = guard.probe_media(payload, kind)
    assert not report.accepted, report
    assert report.worker_reaped and report.controls_applied
    assert report.status in (
        guard.DecodeStatus.LIMIT,
        guard.DecodeStatus.MALFORMED,
        guard.DecodeStatus.UNAVAILABLE,
        guard.DecodeStatus.TIMEOUT,
        guard.DecodeStatus.MEMORY,
    )


def test_compressed_pdf_stream_enforces_decoded_byte_budget():
    require_linux()
    report = guard.probe_media(
        pdf(compressed_bytes=100000),
        guard.DecoderKind.PDF,
        limits=guard.DecoderLimits(decoded_bytes=1000),
    )
    assert report.status in (guard.DecodeStatus.LIMIT, guard.DecodeStatus.UNAVAILABLE)
    assert not report.accepted


@pytest.mark.parametrize(
    "control,status",
    [
        ("crash", guard.DecodeStatus.CRASH),
        ("timeout", guard.DecodeStatus.TIMEOUT),
        ("memory", guard.DecodeStatus.MEMORY),
        ("cpu", guard.DecodeStatus.CPU),
    ],
)
def test_real_worker_fault_is_reaped_and_parent_can_decode_again(control, status):
    require_linux()
    limits = guard.DecoderLimits(
        wall_seconds=0.5 if control == "timeout" else 5, cpu_seconds=1
    )
    parent = {"pid": os.getpid(), "sentinel": object()}
    before = parent.copy()
    report = guard._execute(
        b"synthetic", guard.DecoderKind.AUDIO, limits, control=control
    )
    assert report.status is status, report
    assert report.controls_applied and report.worker_reaped
    assert parent == before and parent["pid"] == os.getpid()
    assert guard.probe_media(controls()["wav"], guard.DecoderKind.AUDIO).accepted


@pytest.mark.parametrize("control", ["write", "network", "environment", "log"])
def test_worker_blocks_writes_network_and_credentials_and_discards_logs(
    control, monkeypatch, capfd
):
    require_linux()
    private = b"SYNTHETIC-PRIVATE-IDENTIFIER"
    monkeypatch.setenv("OPENMED_HOSTILE_SENTINEL", private.decode())
    report = guard._execute(
        private, guard.DecoderKind.AUDIO, guard.DecoderLimits(), control=control
    )
    assert report.accepted, report
    output = capfd.readouterr()
    assert private.decode() not in output.out + output.err + json.dumps(
        report.to_dict()
    )
    assert not Path("/tmp/openmed-decoder-control").exists()


def test_deterministic_byte_mutations_always_terminate_with_fixed_categories():
    require_linux()
    rng = random.Random(3206)
    for kind in guard.DecoderKind:
        payload = bytearray(controls()[kind.value])
        # Bounded metadata/header corruption, not executable exploit payloads.
        for _ in range(4):
            index = rng.randrange(len(payload))
            payload[index] ^= rng.randrange(1, 256)
        result = guard.probe_media(bytes(payload), kind)
        assert result.worker_reaped
        assert type(result.status) is guard.DecodeStatus
        assert result.status is not guard.DecodeStatus.ISOLATION


def test_cancellation_during_native_worker_reaps_owned_process():
    require_linux()
    calls = 0

    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 5

    report = guard._execute(
        b"synthetic",
        guard.DecoderKind.AUDIO,
        guard.DecoderLimits(),
        control="timeout",
        cancelled=cancelled,
    )
    assert report.status is guard.DecodeStatus.CANCELLED
    assert report.worker_reaped


@pytest.mark.parametrize("interruption", [BlockingIOError, InterruptedError])
def test_transient_pipe_backpressure_retries_complete_original_input(
    monkeypatch, interruption
):
    require_linux()
    original_write = os.write
    interrupted = False

    def write(fd, data):
        nonlocal interrupted
        if not interrupted and data.startswith(b"RIFF"):
            interrupted = True
            raise interruption()
        return original_write(fd, data)

    monkeypatch.setattr(guard.os, "write", write)
    payload = controls()["wav"]
    report = guard.probe_media(payload, guard.DecoderKind.AUDIO)
    assert interrupted and report.accepted
    assert report.source_digest == hashlib.sha256(payload).hexdigest()
