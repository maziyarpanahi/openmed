"""Optional local CPU adapter for user-supplied d1-omni-600M decision weights."""

from __future__ import annotations

import io
import threading
import wave
from dataclasses import replace
from pathlib import Path

from .decision import (
    DecisionBackendIdentity,
    DecisionBackendKind,
    DecisionBackendOutput,
    DecisionMode,
    DecisionRequest,
    DecisionUnsupportedError,
)
from .decision_d3 import _local_image
from .decision_interop import from_system_one_response, to_system_one_request
from .decision_snapshot import SnapshotPin, require_runtime, verify_snapshot

D1_OMNI = SnapshotPin(
    "LiquidAI/d1-omni-600M",
    "02b55d7076f15129e59ab3f94783f32c4b088674",
    ("audio.py", "encoder.py", "modeling_d1.py", "prompt.py", "vision.py"),
    (
        "LICENSE",
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ),
    "c5eb224e336a03e076679fa3106dd1fff56c55a4eb697cd84f705618cd900769",
    "688546f374567bf380f72092eb1717ef53710604a662b115527452e5afed35e6",
    frozenset({".gitattributes", ".cache", "README.md"}),
)
D1_LICENSE_SHA256 = "9953509cbd8c1c6ce7b06ddf74fcc204a892692316201dd8f7400a84790d4120"


class D1OmniDecisionBackend:
    """Answer assistive questions about text, local images or one English audio clip."""

    supported_media_types = frozenset({"image/png", "audio/wav"})
    identity = DecisionBackendIdentity(
        backend_id="openmed.d1_omni",
        kind=DecisionBackendKind.SPECIALIST,
        runtime="transformers_cpu",
        model_id=D1_OMNI.repository,
        revision=D1_OMNI.revision,
        license_id="LFM Open License 1.0",
        license_sha256=D1_LICENSE_SHA256,
    )

    def __init__(self, model_directory: Path, *, revision: str = D1_OMNI.revision):
        self._snapshot = verify_snapshot(model_directory, revision, D1_OMNI)
        require_runtime("5.17.0")
        try:
            import torch
            from transformers import AutoModel

            model = (
                AutoModel.from_pretrained(
                    str(self._snapshot.root),
                    trust_remote_code=True,
                    local_files_only=True,
                    use_safetensors=True,
                    dtype=torch.float32,
                )
                .to("cpu")
                .eval()
            )
        except Exception:
            pass
        else:
            self._snapshot.check()
            self._model = model
            self._lock = threading.Lock()
            return
        raise DecisionUnsupportedError("decision_d1_load_failed")

    def score(self, request: DecisionRequest) -> DecisionBackendOutput:
        """Score the options of a single question."""
        if request.mode is DecisionMode.QUESTION_SET:
            raise DecisionUnsupportedError("decision_question_set_required")
        return self.score_questions(request)[request.as_questions()[0].question_id]

    def score_questions(
        self, request: DecisionRequest
    ) -> dict[str, DecisionBackendOutput]:
        """Score named questions locally; refuse mixed audio/image or multiple clips."""
        clips = [i for i in request.media if i.media_type == "audio/wav"]
        if any(
            i.media_type not in self.supported_media_types for i in request.media
        ) or (clips and len(request.media) != 1):
            raise DecisionUnsupportedError("decision_d1_media_unsupported")
        wire = to_system_one_request(replace(request, media=()))
        with self._lock:
            self._snapshot.check()
            try:
                import numpy as np
                import torch

                audio = None
                images = [
                    _local_image(i)
                    for i in request.media
                    if i.media_type == "image/png"
                ]
                if clips:
                    with wave.open(io.BytesIO(clips[0].protected_bytes()), "rb") as f:
                        audio = np.frombuffer(
                            f.readframes(f.getnframes()), dtype="<i2"
                        ).copy()
                with torch.inference_mode():
                    body = self._model.system_one(
                        state=wire["state"],
                        questions=wire["questions"],
                        images=images or None,
                        audio=audio,
                    )
            except Exception:
                pass
            else:
                self._snapshot.check()
                outputs = from_system_one_response(request, body)
                if clips:
                    outputs = {
                        key: replace(
                            out, warnings=(*out.warnings, "audio_english_only")
                        )
                        for key, out in outputs.items()
                    }
                return outputs
        raise RuntimeError("decision_d1_inference_failed")
