"""Opt-in actual native checks using caller-owned local artifacts, never downloads."""

import json
import os
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from openmed.multimodal.streaming.speech.diarization import (
    DiarizationError,
    DiarizationRequest,
    DiarizationSession,
)
from openmed.onnx.diarization import (
    DiarizationLimits,
    LocalDiarizationArtifact,
    SherpaDiarizationProvider,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def artifacts():
    manifest = os.environ.get("OPENMED_DIARIZATION_MANIFEST")
    if not manifest:
        pytest.skip("requires caller-provided local diarization artifacts")
    data = json.loads(Path(manifest).read_text())
    return tuple(
        LocalDiarizationArtifact(**{**data[key], "path": Path(data[key]["path"])})
        for key in ("segmentation", "embedding")
    )


def silence():
    return DiarizationRequest(
        pcm=b"\x00\x00" * 16000 * 4,
        revision=0,
        offset_samples=0,
        request_ref="1" * 32,
    )


def test_actual_native_silence(artifacts):
    segmentation, embedding = artifacts
    provider = SherpaDiarizationProvider(segmentation=segmentation, embedding=embedding)
    req = silence()
    batch = provider.observe(req)
    assert batch.turns == ()
    assert (
        DiarizationSession("2" * 32).apply(req, batch)["status"]
        == "insufficient_evidence"
    )


@pytest.mark.parametrize(
    "failure", ["digest", "corrupt", "cancel", "deadline", "memory"]
)
def test_actual_native_failures_release_no_observations(artifacts, tmp_path, failure):
    segmentation, embedding = artifacts
    limits = DiarizationLimits()
    event = Event()
    if failure == "digest":
        segmentation = replace(segmentation, sha256="0" * 64)
    if failure == "corrupt":
        import hashlib

        path = tmp_path / "synthetic-invalid.onnx"
        path.write_bytes(b"not a model")
        segmentation = replace(
            segmentation,
            path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    if failure == "cancel":
        event.set()
    if failure == "deadline":
        limits = DiarizationLimits(timeout_seconds=0.000001)
    if failure == "memory":
        limits = DiarizationLimits(max_rss_bytes=1)
    provider = SherpaDiarizationProvider(
        segmentation=segmentation, embedding=embedding, limits=limits
    )
    with pytest.raises(DiarizationError) as error:
        provider.observe(silence(), cancelled=event)
    assert str(error.value).startswith("diarization_")
    assert str(tmp_path) not in str(error.value)
