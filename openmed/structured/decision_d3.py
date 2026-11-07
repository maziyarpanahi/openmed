"""Opt-in in-process d3-edge decision adapter, CPU only and offline."""

from __future__ import annotations

import io
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

from .decision import (
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionMode,
    DecisionRequest,
    DecisionUnsupportedError,
)
from .decision_interop import from_system_one_response, to_system_one_request
from .decision_media import DecisionMedia, _rgb, _video_info
from .decision_snapshot import SnapshotPin, require_runtime, verify_snapshot

D3_EDGE = SnapshotPin(
    "vllm-sr/d3-edge",
    "bee5e872f75b9c5d1405767bf244a6948b95c0af",
    (
        "d3_engine.py",
        "d3_fast.py",
        "d3_format.py",
        "d3_kernels.py",
        "d3_runtime.py",
        "d3_server.py",
        "modeling_d3.py",
        "pipeline_d3.py",
    ),
    (
        "MODEL_MANIFEST.json",
        "chat_template.jinja",
        "config.json",
        "decision_config.json",
        "merges.txt",
        "model-00001-of-00001.safetensors",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "readout.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "video_preprocessor_config.json",
        "vocab.json",
    ),
    "f0be22a5bebcc9de31abf1e6447cd8a5753bd93d35fde840f15601618d1d6fc6",
    "646ac9bf0d64dcacaccb2b4791fda47b61d26a5c317e2cd40c7a48fbef24d281",
    frozenset(
        {
            ".eval_results",
            ".gitattributes",
            ".cache",
            "assets",
            "LICENSE",
            "NOTICE",
            "README.md",
            "requirements.txt",
        }
    ),
)
D3_TRANSFORMERS = "5.17.0"


def _local_image(item: DecisionMedia) -> Any:
    from PIL import Image

    with Image.open(io.BytesIO(item.protected_bytes()), formats=["PNG"]) as img:
        if img.mode == "RGBA":
            canvas = Image.new("RGB", img.size, "white")
            canvas.paste(img, mask=img.getchannel("A"))
            return canvas
        return img.convert("RGB")


def _local_video(item: DecisionMedia, model: Any) -> Any:
    import numpy as np

    blob = item.protected_bytes()
    info = _video_info(blob)
    pixels = np.frombuffer(_rgb(blob, info), dtype=np.uint8).reshape(
        info.frames, info.height, info.width, 3
    )
    runtime = sys.modules[type(model._require()).__module__]
    indices = runtime.video_sample_indices(info.frames, float(info.fps))
    return runtime._video(
        [pixels[i] for i in indices], indices, float(info.fps), info.frames
    )


class D3EdgeDecisionBackend:
    """Local specialist for text, verified images and timed RGB video frames."""

    supported_media_types = frozenset({"image/png", "video/mp4"})
    identity = DecisionBackendIdentity(
        backend_id="openmed.d3_edge",
        kind=DecisionBackendKind.SPECIALIST,
        runtime="transformers_cpu",
        model_id=D3_EDGE.repository,
        revision=D3_EDGE.revision,
        license_id="Apache-2.0",
    )

    def __init__(self, model_directory: Path, *, revision: str = D3_EDGE.revision):
        self._snapshot = verify_snapshot(model_directory, revision, D3_EDGE)
        require_runtime(D3_TRANSFORMERS)
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                str(self._snapshot.root),
                trust_remote_code=True,
                local_files_only=True,
                use_safetensors=True,
                device="cpu",
            ).eval()
        except Exception:
            pass
        else:
            self._snapshot.check()
            self._model = model
            self._lock = threading.Lock()
            return
        raise DecisionUnsupportedError("decision_d3_load_failed")

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        """Return model probabilities for one supported question."""
        if request.mode is DecisionMode.QUESTION_SET:
            raise DecisionUnsupportedError("decision_question_set_required")
        return self.score_questions(request)[request.as_questions()[0].question_id]

    def score_questions(
        self, request: DecisionRequest
    ) -> dict[str, DecisionBackendOutput]:
        """Score a question set in one forward call; no URL or path is passed as media."""
        if any(x.media_type not in self.supported_media_types for x in request.media):
            raise DecisionUnsupportedError("decision_d3_media_unsupported")
        wire = to_system_one_request(replace(request, media=()))
        with self._lock:
            self._snapshot.check()
            try:
                import torch

                images = [
                    _local_image(i)
                    for i in request.media
                    if i.media_type == "image/png"
                ]
                videos = [
                    _local_video(i, self._model)
                    for i in request.media
                    if i.media_type == "video/mp4"
                ]
                with torch.inference_mode():
                    data = self._model.system_one(
                        state=wire["state"],
                        questions=wire["questions"],
                        images=images or None,
                        videos=videos or None,
                    )
            except Exception:
                pass
            else:
                self._snapshot.check()
                return from_system_one_response(request, data)
        raise RuntimeError("decision_d3_inference_failed")
