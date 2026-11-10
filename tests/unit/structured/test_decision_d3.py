"""Pinned local code load controls and optional CPU end-to-end integration."""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from openmed.structured import decision_d3 as module
from openmed.structured import decision_snapshot as snapshots
from openmed.structured.decision import (
    DecisionMode,
    DecisionRequest,
    DecisionState,
    decide,
)
from openmed.structured.decision_media import DecisionMedia


def pinned(root):
    rows = []
    for name in ("modeling.py", "config.json"):
        data = (root / name).read_bytes()
        rows.append([[name, len(data), hashlib.sha256(data).hexdigest()]])
    codes, assets = (
        hashlib.sha256(json.dumps(r, separators=(",", ":")).encode()).hexdigest()
        for r in rows
    )
    return snapshots.SnapshotPin(
        "example/synthetic",
        "1" * 40,
        ("modeling.py",),
        ("config.json",),
        codes,
        assets,
        frozenset(),
    )


def test_snapshot_fails_closed_before_any_custom_load(tmp_path, monkeypatch):
    (tmp_path / "modeling.py").write_text("# synthetic fixture\n")
    (tmp_path / "config.json").write_text("{}")
    pin = pinned(tmp_path)
    monkeypatch.delenv("OPENMED_TRUSTED_REMOTE_CODE_MODELS", raising=False)
    with pytest.raises(snapshots.DecisionSnapshotError, match="not_allowed"):
        snapshots.verify_snapshot(tmp_path, pin.revision, pin)
    monkeypatch.setenv("OPENMED_TRUSTED_REMOTE_CODE_MODELS", pin.repository)
    with pytest.raises(snapshots.DecisionSnapshotError, match="revision"):
        snapshots.verify_snapshot(tmp_path, "2" * 40, pin)
    snapshot = snapshots.verify_snapshot(tmp_path, pin.revision, pin)
    (tmp_path / "extra.py").write_text("raise RuntimeError('PRIVATE')\n")
    with pytest.raises(snapshots.DecisionSnapshotError, match="inventory"):
        snapshots.verify_snapshot(tmp_path, pin.revision, pin)
    (tmp_path / "extra.py").unlink()
    (tmp_path / "modeling.py").write_text("# changed\n")
    with pytest.raises(snapshots.DecisionSnapshotError, match="code_mismatch"):
        snapshots.verify_snapshot(tmp_path, pin.revision, pin)
    with pytest.raises(snapshots.DecisionSnapshotError, match="changed"):
        snapshot.check()
    (tmp_path / "modeling.py").write_text("# synthetic fixture\n")
    (tmp_path / "config.json").write_text('{"altered":true}')
    with pytest.raises(snapshots.DecisionSnapshotError, match="assets_mismatch"):
        snapshots.verify_snapshot(tmp_path, pin.revision, pin)


def test_runtime_mismatch_never_imports_custom_model(monkeypatch):
    monkeypatch.setattr(snapshots, "version", lambda name: "5.0.0")
    with pytest.raises(
        snapshots.DecisionSnapshotError, match="transformers_version_mismatch"
    ):
        snapshots.require_runtime(module.D3_TRANSFORMERS)


def image():
    from PIL import Image

    f = io.BytesIO()
    Image.new("RGBA", (64, 64), (200, 10, 30, 0)).save(f, format="PNG")
    b = f.getvalue()
    return DecisionMedia(
        b,
        media_type="image/png",
        byte_size=len(b),
        sha256=hashlib.sha256(b).hexdigest(),
    )


def test_admitted_image_is_pixels_and_uncertain_answer_abstains(monkeypatch):
    pytest.importorskip("torch")
    import threading
    from types import SimpleNamespace
    from unittest.mock import Mock

    from PIL import Image

    backend = object.__new__(module.D3EdgeDecisionBackend)
    backend._snapshot = SimpleNamespace(check=lambda: None)
    backend._lock = threading.Lock()

    def score(**kw):
        assert list(kw["questions"]) == ["decision"] and kw["videos"] is None
        (img,) = kw["images"]
        assert isinstance(img, Image.Image) and img.getpixel((1, 1)) == (255, 255, 255)
        return {"answers": {"decision": {"type": "noul", "noul": 0.5}}}

    backend._model = SimpleNamespace(system_one=Mock(side_effect=score))
    request = DecisionRequest(
        DecisionMode.BOOLEAN_CHOICE,
        "synthetic",
        options=("true", "false"),
        media=(image(),),
    )
    result = decide(request, backend=backend)
    assert result.state is DecisionState.ABSTAINED and result.choice is None
    assert "synthetic" not in repr(result.backend)
    assert backend._model.system_one.call_count == 1


@pytest.mark.integration
@pytest.mark.slow
def test_pinned_d3_cpu_answers_text_image_video():
    import shutil
    import subprocess

    model = os.getenv("OPENMED_D3_EDGE_SNAPSHOT")
    if not model:
        pytest.skip("operator-supplied pinned d3-edge snapshot")
    backend = module.D3EdgeDecisionBackend(Path(model))
    request = DecisionRequest(
        DecisionMode.BOOLEAN_CHOICE,
        "Synthetic color test",
        options=("true", "false"),
        instructions="Does the sample show a solid color?",
        timeout_ms=30000,
    )
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    video = subprocess.check_output(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=64x64:r=4:d=1",
            "-c:v",
            "libx264rgb",
            "-bf",
            "0",
            "-threads",
            "1",
            "-movflags",
            "empty_moov+frag_keyframe+default_base_moof",
            "-f",
            "mp4",
            "pipe:1",
        ]
    )
    clip = DecisionMedia(
        video,
        media_type="video/mp4",
        byte_size=len(video),
        sha256=hashlib.sha256(video).hexdigest(),
    )
    for media in [(), (image(),), (clip,)]:
        result = decide(replace(request, media=media), backend=backend)
        assert result.state in {DecisionState.SUCCESS, DecisionState.ABSTAINED}
        assert (
            len(result.option_scores) == 2
            and result.backend["revision"] == module.D3_EDGE.revision
        )
