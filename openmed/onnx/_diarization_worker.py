"""Private, disposable diarization worker. No CLI inputs, downloads or audio files."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import stat
import sys
from typing import Any


def _read(size: int) -> bytes:
    data = sys.stdin.buffer.read(size)
    if len(data) != size:
        raise ValueError
    return data


def _verify(artifact: dict[str, Any]) -> str:
    path = artifact["path"]
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 2 * 1024**3:
            raise ValueError
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != artifact["sha256"]:
        raise ValueError
    return path


def _bounded_turn(
    start_seconds: float, end_seconds: float, speaker: int, samples: int
) -> list:
    if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
        raise ValueError
    start, end = round(start_seconds * 16000), round(end_seconds * 16000)
    if not 0 <= start < samples or not start < end or not 0 <= speaker < 32:
        raise ValueError
    # Native segmentation frames can extend past the final input sample. Retain
    # the intersection and mark it unstable; never invent audio outside the input.
    clipped = end > samples
    return [start, min(end, samples), int(speaker), clipped]


def _run() -> dict:
    header_size = int.from_bytes(_read(4), "big")
    if not 0 < header_size <= 16384:
        raise ValueError
    header = json.loads(_read(header_size))
    size = header["pcm_bytes"]
    if type(size) is not int or not 0 < size <= 60 * 16000 * 2 or size % 2:
        raise ValueError
    pcm = _read(size)
    if sys.stdin.buffer.read(1):
        raise ValueError
    for name in ("sherpa-onnx", "sherpa-onnx-core"):
        if importlib.metadata.version(name) != "2.0.1":
            raise ValueError
    segmentation = _verify(header["segmentation"])
    embedding = _verify(header["embedding"])
    import numpy as np
    import sherpa_onnx as sherpa

    config = sherpa.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=segmentation
            ),
            num_threads=1,
            debug=False,
            provider="cpu",
        ),
        embedding=sherpa.SpeakerEmbeddingExtractorConfig(
            model=embedding, num_threads=1, debug=False, provider="cpu"
        ),
        clustering=sherpa.FastClusteringConfig(num_clusters=-1, threshold=0.5),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError
    engine = sherpa.OfflineSpeakerDiarization(config)
    if engine.sample_rate != 16000:
        raise ValueError
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    result = engine.process(samples).sort_by_start_time()
    if len(result) > 1024:
        raise ValueError
    turns = []
    for turn in result:
        turns.append(_bounded_turn(turn.start, turn.end, turn.speaker, len(samples)))
    # Detect ordinary artifact replacement during loading/inference. Caller owns
    # stable trusted assets; this does not defend against hostile native code.
    _verify(header["segmentation"])
    _verify(header["embedding"])
    return {"turns": turns, "runtime": "sherpa-onnx:2.0.1"}


def main() -> None:
    """Suppress native diagnostics before loading dependencies or model files."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = os.dup(1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        response = json.dumps(_run(), separators=(",", ":")).encode("ascii")
        if len(response) > 256 * 1024:
            raise ValueError
        with os.fdopen(output, "wb") as stream:
            stream.write(response)
    except Exception:
        os._exit(1)


if __name__ == "__main__":
    main()
