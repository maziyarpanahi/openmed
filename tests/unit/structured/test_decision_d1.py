"""Synthetic local audio decisions, revision provenance and optional CPU test."""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.structured.decision import (
    DecisionError,
    DecisionMode,
    DecisionRequest,
    DecisionState,
    DecisionUnsupportedError,
    decide,
)
from openmed.structured.decision_d1 import (
    D1_LICENSE_SHA256,
    D1_OMNI,
    D1OmniDecisionBackend,
)
from openmed.structured.decision_interop import to_system_one_request
from openmed.structured.decision_media import DecisionMedia, DecisionMediaError


def audio(frames=1600, rate=16000, channels=1, extra=b""):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as w:
        w.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        w.writeframes(struct.pack("<h", 512) * frames * channels)
    b = stream.getvalue()
    if extra:
        b = b[:4] + struct.pack("<I", len(b) - 8 + len(extra)) + b[8:] + extra
    return DecisionMedia(
        b,
        byte_size=len(b),
        sha256=hashlib.sha256(b).hexdigest(),
        media_type="audio/wav",
    )


def request(*items):
    return DecisionRequest(
        DecisionMode.BOOLEAN_CHOICE,
        "Synthetic intake routing",
        options=("true", "false"),
        instructions="Does this request ask for a callback?",
        media=items,
        timeout_ms=30000,
    )


def test_wav_preflight_downmix_resampling_and_wire_refusal():
    pytest.importorskip("scipy.signal")
    tag = b"JUNK" + struct.pack("<I", 8) + b"PRIVATE!"
    asset = audio(8000, 8000, 2, tag)
    assert b"PRIVATE!" not in asset.protected_bytes()
    with wave.open(io.BytesIO(asset.protected_bytes()), "rb") as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 16000, 16000)
        raw = w.readframes(16000)
        assert abs(struct.unpack_from("<h", raw, 2000)[0] - 512) <= 1
    with pytest.raises(DecisionUnsupportedError):
        to_system_one_request(request(asset))
    assert not {"data", "text", "transcript"}.intersection(asset.safe_metadata())
    with pytest.raises(DecisionMediaError, match="duration"):
        audio(480001)
    assert audio(480000).byte_size == 960044
    with pytest.raises(DecisionMediaError):
        audio(extra=b"data" + struct.pack("<I", 2) + b"xx")


def test_clip_scored_once_without_bytes_in_logs_and_no_mixed_inputs(caplog):
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    backend = object.__new__(D1OmniDecisionBackend)
    backend._snapshot = SimpleNamespace(check=lambda: None)
    backend._lock = threading.Lock()

    def answer(**kw):
        assert kw["images"] is None and isinstance(kw["audio"], np.ndarray)
        assert kw["audio"].dtype == np.int16 and kw["audio"].shape == (1600,)
        return {"answers": {"decision": {"type": "noul", "noul": 0.5}}}

    backend._model = SimpleNamespace(system_one=Mock(side_effect=answer))
    clip = audio()
    result = decide(request(clip), backend=backend)
    assert (
        result.state is DecisionState.ABSTAINED
        and "audio_english_only" in result.warnings
    )
    assert (
        result.backend["license_sha256"] == D1_LICENSE_SHA256
        and result.backend["revision"] == D1_OMNI.revision
    )
    assert result.backend["license_id"] == "LFM Open License 1.0"
    assert (
        decide(request(clip, clip), backend=backend).state is DecisionState.UNSUPPORTED
    )
    assert backend._model.system_one.call_count == 1
    assert (
        clip.to_dict()["data"] not in caplog.text
        and "Synthetic intake routing" not in caplog.text
    )
    with pytest.raises(DecisionError, match="immutable"):
        replace(backend.identity, revision="main")
    with pytest.raises(DecisionUnsupportedError, match="revision_mismatch"):
        D1OmniDecisionBackend(Path("/unread"), revision="1" * 40)


@pytest.mark.slow
@pytest.mark.integration
def test_pinned_d1_cpu_text_image_audio():
    from PIL import Image

    root = os.getenv("OPENMED_D1_OMNI_SNAPSHOT")
    if not root:
        pytest.skip("operator-supplied pinned d1-omni snapshot")
    backend = D1OmniDecisionBackend(Path(root))
    assert (
        hashlib.sha256((Path(root) / "LICENSE").read_bytes()).hexdigest()
        == backend.identity.license_sha256
    )
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "red").save(buf, format="PNG")
    b = buf.getvalue()
    img = DecisionMedia(
        b,
        media_type="image/png",
        byte_size=len(b),
        sha256=hashlib.sha256(b).hexdigest(),
    )
    for media in [(), (img,), (audio(16000),)]:
        result = decide(request(*media), backend=backend)
        assert (
            result.state in {DecisionState.SUCCESS, DecisionState.ABSTAINED}
            and len(result.option_scores) == 2
        )
        assert ("audio_english_only" in result.warnings) == bool(
            media and media[0].media_type == "audio/wav"
        )
    assert (
        decide(request(img, audio()), backend=backend).state
        is DecisionState.UNSUPPORTED
    )
