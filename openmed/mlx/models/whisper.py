"""Local MLX Whisper assets and an interruptible, isolated inference process.

No model identifier is accepted in place of a local directory. Audio crosses a
private process pipe only; the worker never writes it, prints it, or downloads
assets. Process termination releases native allocations, not a secure-erasure
guarantee for the operating system, swap, or caller-owned input.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import platform
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_AUDIO_BYTES = 15 * 16_000 * 2
_MAX_REPLY_BYTES = 131_072
_DIMENSIONS = {
    "n_mels",
    "n_audio_ctx",
    "n_audio_state",
    "n_audio_head",
    "n_audio_layer",
    "n_vocab",
    "n_text_ctx",
    "n_text_state",
    "n_text_head",
    "n_text_layer",
}


class WhisperError(ValueError):
    """Controlled failure without paths, decoded content, or exception chaining."""


def _require(condition: bool) -> None:
    if not condition:
        raise WhisperError("invalid_local_whisper_assets")


def _file_digest(path: Path, maximum: int) -> str:
    _require(not path.is_symlink() and path.is_file())
    _require(0 < path.stat().st_size <= maximum)
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            size += len(block)
            _require(size <= maximum)
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class WhisperAssets:
    """Pin caller-managed local weights and a snapshot of their license metadata.

    Args:
        directory: Immutable, owner-controlled directory containing config.json
            and weights.safetensors. Symlink assets are rejected.
        revision: Exact upstream source revision, never a moving branch or tag.
        license_snapshot: Caller-reviewed metadata bytes, separately fingerprinted
            from this Apache-2.0 adapter. No license interpretation is performed.
    """

    directory: Path = field(repr=False)
    revision: str
    license_snapshot: bytes = field(repr=False)
    fingerprint: str = field(init=False)
    license_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _require(isinstance(self.directory, Path))
        _require(not self.directory.is_symlink() and self.directory.is_dir())
        _require(
            type(self.revision) is str
            and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", self.revision)
            is not None
        )
        _require(
            type(self.license_snapshot) is bytes
            and 0 < len(self.license_snapshot) <= 1_048_576
        )
        object.__setattr__(self, "directory", self.directory.resolve())
        object.__setattr__(
            self, "license_digest", hashlib.sha256(self.license_snapshot).hexdigest()
        )
        object.__setattr__(self, "fingerprint", self._fingerprint())

    def _fingerprint(self) -> str:
        config_digest = _file_digest(self.directory / "config.json", 16_384)
        failed = False
        try:
            config = json.loads((self.directory / "config.json").read_bytes())
        except Exception:
            failed = True
        _require(not failed and type(config) is dict)
        _require(set(config) <= _DIMENSIONS | {"model_type"})
        _require(
            _DIMENSIONS <= set(config)
            and config.get("model_type", "whisper") == "whisper"
        )
        _require(
            all(
                type(config[key]) is int and 1 <= config[key] <= 60_000
                for key in _DIMENSIONS
            )
        )
        _require(config["n_audio_ctx"] == 1500 and config["n_mels"] in (80, 128))
        _require(config["n_text_ctx"] == 448 and 51865 <= config["n_vocab"] <= 51866)
        _require(config["n_audio_state"] <= 1280 and config["n_text_state"] <= 1280)
        _require(config["n_audio_layer"] <= 32 and config["n_text_layer"] <= 32)
        weights = _file_digest(self.directory / "weights.safetensors", 4 * 1024**3)
        return hashlib.sha256(
            json.dumps(
                [self.revision, self.license_digest, config_digest, weights]
            ).encode()
        ).hexdigest()

    def verify(self) -> None:
        """Reject changed local assets before starting native inference."""
        _require(self._fingerprint() == self.fingerprint)


def _decode(model: Any, pcm: bytearray) -> dict[str, Any]:
    # Imports and native allocations remain inside the disposable worker.
    import mlx.core as mx
    import numpy as np
    from mlx_whisper.audio import N_FRAMES, log_mel_spectrogram, pad_or_trim
    from mlx_whisper.decoding import DecodingOptions
    from mlx_whisper.timing import add_word_timestamps
    from mlx_whisper.tokenizer import get_tokenizer

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    try:
        mel = pad_or_trim(
            log_mel_spectrogram(samples, n_mels=model.dims.n_mels), N_FRAMES, axis=-2
        ).astype(mx.float16)
        _, probabilities = model.detect_language(mel)
        language = max(probabilities, key=probabilities.get)
        result = model.decode(
            mel,
            DecodingOptions(
                language=language,
                task="transcribe",
                temperature=0.0,
                sample_len=192,
                without_timestamps=True,
            ),
        )
        tokenizer = get_tokenizer(
            True,
            num_languages=model.num_languages,
            language=language,
            task="transcribe",
        )
        segments = [
            {
                "seek": 0,
                "start": 0.0,
                "end": len(samples) / 16_000,
                "tokens": result.tokens,
            }
        ]
        add_word_timestamps(
            segments=segments,
            model=model,
            tokenizer=tokenizer,
            mel=mel,
            num_frames=len(samples) // 160,
            last_speech_timestamp=0.0,
        )
        return {
            "text": result.text,
            "language": language,
            "language_probability": probabilities[language],
            "words": segments[0].get("words", []),
            "no_speech_probability": result.no_speech_prob,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "token_count": len(result.tokens),
        }
    finally:
        samples.fill(0)


def _worker(connection: Any, assets: WhisperAssets) -> None:
    # The worker has no clinical output streams, network client, or file input
    # decoder. It receives only bounded PCM from its parent and local weights.
    import gc
    import resource
    import socket

    network_attempts = 0

    def offline(*_: Any, **__: Any) -> Any:
        nonlocal network_attempts
        network_attempts += 1
        raise WhisperError("network_disabled")

    socket.create_connection = offline
    socket.getaddrinfo = offline
    socket.socket.connect = offline  # type: ignore[method-assign]
    socket.socket.connect_ex = offline  # type: ignore[method-assign]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Hub offline mode also disables its telemetry; no opt-out switch is exposed.
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        assets.verify()
        from huggingface_hub import constants as hub_constants

        # A spawned main module may have imported Hub before these environment
        # flags were set. Enforce its cached mode inside this worker as well.
        hub_constants.HF_HUB_OFFLINE = True
        import mlx.core as mx
        from mlx_whisper.load_models import load_model

        model = load_model(str(assets.directory), dtype=mx.float16)
        if network_attempts:
            raise WhisperError("network_attempted")
        versions = {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-whisper", "numpy", "numba", "scipy", "tiktoken")
        }
        connection.send_bytes(
            json.dumps({"ready": True, "versions": versions}).encode()
        )
        while True:
            pcm = bytearray(connection.recv_bytes(_MAX_AUDIO_BYTES))
            try:
                if not pcm or len(pcm) % 2:
                    raise WhisperError("invalid_pcm")
                reply = _decode(model, pcm)
            finally:
                pcm[:] = b"\0" * len(pcm)
                del pcm
                gc.collect()
                mx.clear_cache()
            reply["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            reply["peak_device_bytes"] = mx.get_peak_memory()
            reply["network_attempts"] = network_attempts
            if network_attempts:
                raise WhisperError("network_attempted")
            payload = json.dumps(reply, allow_nan=False).encode()
            if len(payload) > _MAX_REPLY_BYTES:
                raise WhisperError("output_limit")
            connection.send_bytes(payload)
            del reply, payload
    except BaseException:
        try:
            connection.send_bytes(b'{"failed":true}')
        except BaseException:
            pass
    finally:
        connection.close()


class WhisperProcess:
    """One model process with bounded IPC, timeouts and concurrent cancellation.

    Args:
        assets: Verified local model files; never a Hub identifier.
        startup_timeout: Maximum model-load time before termination.
    """

    def __init__(self, assets: WhisperAssets, *, startup_timeout: float = 90.0) -> None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            raise WhisperError("whisper_platform_unavailable")
        _require(type(assets) is WhisperAssets and 0 < startup_timeout <= 120)
        self._lock = threading.RLock()
        self._closed = False
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker, args=(child, assets), daemon=True
        )
        try:
            self._process.start()
            child.close()
            ready = self._receive(startup_timeout)
            if ready.get("ready") is not True:
                raise WhisperError("whisper_load_failed")
            self.versions = ready["versions"]
        except BaseException:
            self.close()
            raise WhisperError("whisper_load_failed") from None
        finally:
            child.close()

    def _receive(self, timeout: float) -> dict[str, Any]:
        if not self._connection.poll(timeout):
            raise WhisperError("whisper_deadline")
        result = json.loads(self._connection.recv_bytes(_MAX_REPLY_BYTES))
        if type(result) is not dict or result.get("failed"):
            raise WhisperError("whisper_inference_failed")
        return result

    def decode(self, pcm: memoryview, *, timeout: float) -> dict[str, Any]:
        """Decode one bounded mono signed-16-bit little-endian 16 kHz window."""
        failed = False
        try:
            if self._closed or not 0 < pcm.nbytes <= _MAX_AUDIO_BYTES:
                raise WhisperError("whisper_closed_or_invalid_audio")
            self._connection.send_bytes(pcm)
            result = self._receive(timeout)
        except Exception:
            failed = True
        if failed or self._closed:
            self.close()
            raise WhisperError("whisper_inference_failed")
        return result

    def close(self) -> None:
        """Terminate and join the worker, discarding every pending response."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.0)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(1.0)
            self._connection.close()
            if self._process.is_alive():
                raise WhisperError("whisper_worker_cleanup_failed")
            self._process.close()
