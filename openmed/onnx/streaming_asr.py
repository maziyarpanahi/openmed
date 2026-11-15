"""Experimental, offline character-CTC speech inference using a pinned ONNX graph."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openmed.multimodal.provider_result import (
    ProviderResultEnvelope,
    ProviderResultOutcome,
)
from openmed.multimodal.streaming.speech.providers import (
    AudioChunk,
    FinalizedSegment,
    PartialHypothesis,
    ProviderProvenance,
    SpeechProviderDeclaration,
    SpeechStreamLimits,
    StreamingAsrSession,
    StreamingAsrSessionReport,
    StreamState,
    TokenTime,
)

_MAX_MODEL = 512 * 1024 * 1024
_RATE = 16000
_HOP = 160
_WINDOW = 400
_FFT = 512
_MELS = 40


class OnnxSpeechError(ValueError):
    """Fixed-code refusal that never includes samples, paths or transcript text."""


def _check(value: bool, reason: str) -> None:
    if not value:
        raise OnnxSpeechError(reason)


def vocabulary_digest(labels: tuple[str, ...]) -> str:
    """Hash an ordered character vocabulary, with CTC blank at index zero."""
    _check(
        type(labels) is tuple
        and 2 <= len(labels) <= 1024
        and labels[0] == ""
        and all(type(s) is str and len(s) == 1 and s.isprintable() for s in labels[1:])
        and len(set(labels)) == len(labels),
        "invalid_vocabulary",
    )
    return hashlib.sha256(
        json.dumps(labels, ensure_ascii=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True, kw_only=True)
class OnnxSpeechProfile:
    """Pins for a single static float32 log-mel to character-logits graph.

    The export must explicitly align one output row to each 10 ms input hop.
    The supported preprocessing is described in the streaming speech guide.
    """

    model_sha256: str
    labels: tuple[str, ...] = field(repr=False)
    vocabulary_sha256: str
    frames: int
    language: str

    def __post_init__(self) -> None:
        _check(
            type(self.model_sha256) is str
            and re.fullmatch(r"[a-f0-9]{64}", self.model_sha256) is not None,
            "invalid_model_pin",
        )
        _check(
            self.vocabulary_sha256 == vocabulary_digest(self.labels),
            "vocabulary_digest_mismatch",
        )
        _check(type(self.frames) is int and 1 <= self.frames <= 3000, "invalid_window")
        _check(
            type(self.language) is str
            and re.fullmatch(r"[a-z]{2,3}", self.language) is not None,
            "invalid_language",
        )

    @property
    def declaration(self) -> SpeechProviderDeclaration:
        """Return the existing speech contract for this bounded, local stream."""
        return SpeechProviderDeclaration(
            provenance=ProviderProvenance(
                provider_id="onnx-ctc-asr",
                provider_revision="v1",
                model_fingerprint="sha256:" + self.model_sha256,
            ),
            languages=(self.language,),
            sample_rates_hz=(_RATE,),
            limits=SpeechStreamLimits(
                max_buffered_chunks=min(self.frames * 10, 4096),
                max_buffered_byte_count=self.frames * _HOP * 2,
                max_buffered_duration_ms=self.frames * 10,
                max_stream_duration_ms=3600000,
                max_partial_hypotheses=4096,
                max_segments=4096,
            ),
        )

    def capability(self) -> dict[str, object]:
        """Report experimental capability, not model or platform qualification."""
        return {
            "tier": "experimental",
            "qualified": False,
            "requires_network": False,
            "model_sha256": self.model_sha256,
            "vocabulary_sha256": self.vocabulary_sha256,
            "sample_rate_hz": _RATE,
            "window_frames": self.frames,
            "frontend": "ctc-logmel-v1",
        }


def log_mel_features(pcm: bytes | bytearray, frames: int) -> Any:
    """Right-pad mono S16LE and return [1,frames,40] natural-log power features."""
    import numpy as np

    _check(
        type(frames) is int
        and 1 <= frames <= 3000
        and type(pcm) in (bytes, bytearray)
        and 0 < len(pcm) <= frames * _HOP * 2
        and len(pcm) % 2 == 0,
        "invalid_pcm_window",
    )
    audio = np.zeros((frames - 1) * _HOP + _WINDOW, dtype=np.float64)
    samples = np.frombuffer(pcm, dtype="<i2")
    audio[: samples.size] = samples.astype(np.float64) / 32768.0
    blocks = np.lib.stride_tricks.sliding_window_view(audio, _WINDOW)[::_HOP]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(_WINDOW) / _WINDOW)
    power = np.abs(np.fft.rfft(blocks * window, n=_FFT, axis=1)) ** 2 / _FFT
    mel_edges = np.linspace(0, 2595 * np.log10(1 + (_RATE / 2) / 700), _MELS + 2)
    hz = 700 * (10 ** (mel_edges / 2595) - 1)
    bins = np.fft.rfftfreq(_FFT, 1 / _RATE)[:, None]
    filters = np.maximum(
        0,
        np.minimum(
            (bins - hz[:-2]) / (hz[1:-1] - hz[:-2]),
            (hz[2:] - bins) / (hz[2:] - hz[1:-1]),
        ),
    )
    return np.log(np.maximum(power @ filters, 1e-10)).astype(np.float32)[None, :, :]


def _read_graph(path: Path, digest: str) -> bytes:
    _check(
        isinstance(path, Path)
        and path.is_absolute()
        and not str(path).startswith(("//", "\\\\")),
        "local_graph_required",
    )
    try:
        with os.fdopen(
            os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            ),
            "rb",
        ) as f:
            st = os.fstat(f.fileno())
            _check(
                stat.S_ISREG(st.st_mode) and 0 < st.st_size <= _MAX_MODEL,
                "invalid_graph_size",
            )
            blob = f.read(st.st_size + 1)
            _check(len(blob) == st.st_size, "invalid_graph_size")
    except OSError:
        pass
    else:
        _check(hashlib.sha256(blob).hexdigest() == digest, "graph_digest_mismatch")
        return blob
    raise OnnxSpeechError("graph_unavailable")


def _inspect_graph(blob: bytes, profile: OnnxSpeechProfile) -> None:
    """Inspect serialized bytes before the native runtime receives them."""
    try:
        import onnx
    except ImportError:
        raise OnnxSpeechError("onnx_extra_required") from None
    try:
        m = onnx.load_model_from_string(blob)
        graph = m.graph
        _check(
            not m.functions
            and not m.training_info
            and not graph.sparse_initializer
            and all(x.domain in ("", "ai.onnx") for x in m.opset_import),
            "unsupported_graph",
        )
        # Reject external assets anywhere (including tensor-valued attributes).
        # The small profile excludes custom operators, nested graphs and control flow.
        for tensor in graph.initializer:
            _check(
                not tensor.external_data
                and tensor.data_location != onnx.TensorProto.EXTERNAL,
                "external_graph_data",
            )
        for node in graph.node:
            _check(
                node.domain in ("", "ai.onnx")
                and node.op_type not in {"If", "Loop", "Scan"},
                "unsupported_graph",
            )
            for a in node.attribute:
                _check(
                    a.type
                    not in {a.GRAPH, a.GRAPHS, a.SPARSE_TENSOR, a.SPARSE_TENSORS},
                    "unsupported_graph",
                )
                for t in ([a.t] if a.HasField("t") else []) + list(a.tensors):
                    _check(
                        not t.external_data
                        and t.data_location != onnx.TensorProto.EXTERNAL,
                        "external_graph_data",
                    )
        for values, name, shape in (
            (graph.input, "features", [1, profile.frames, _MELS]),
            (graph.output, "logits", [1, profile.frames, len(profile.labels)]),
        ):
            _check(
                len(values) == 1 and values[0].name == name, "graph_signature_mismatch"
            )
            t = values[0].type.tensor_type
            _check(
                t.elem_type == onnx.TensorProto.FLOAT
                and [d.dim_value for d in t.shape.dim] == shape,
                "graph_signature_mismatch",
            )
        onnx.checker.check_model(m)
    except OnnxSpeechError:
        raise
    except Exception:
        pass
    else:
        return
    raise OnnxSpeechError("invalid_onnx_graph")


class _CpuEngine:
    def __init__(self, blob: bytes):
        import onnxruntime as ort

        self._ort = ort
        opts = ort.SessionOptions()
        opts.enable_profiling = False
        opts.log_severity_level = 4
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            blob, sess_options=opts, providers=["CPUExecutionProvider"]
        )

    def options(self) -> Any:
        option = self._ort.RunOptions()
        option.log_severity_level = 4
        return option

    def run(self, features: Any, options: Any) -> Any:
        return self._session.run(["logits"], {"features": features}, options)[0]


class OnnxSpeechModel:
    """An already verified, CPU-local model. Open independent bounded streams."""

    def __init__(self, path: Path, profile: OnnxSpeechProfile):
        _check(type(profile) is OnnxSpeechProfile, "invalid_profile")
        blob = _read_graph(path, profile.model_sha256)
        _inspect_graph(blob, profile)
        try:
            engine = _CpuEngine(blob)
        except Exception:
            pass
        else:
            self._profile = profile
            self._engine = engine
            return
        raise OnnxSpeechError("onnx_runtime_unavailable")

    @classmethod
    def load(cls, path: Path, profile: OnnxSpeechProfile) -> OnnxSpeechModel:
        """Load a caller-supplied pinned model; no repository or URL resolution."""
        return cls(path, profile)

    def open_stream(self) -> _OnnxSpeechStream:
        """Create a new stream; sharing model weights does not share audio state."""
        return _OnnxSpeechStream(self._profile, self._engine)


@dataclass(frozen=True, slots=True)
class SpeechWindowResult:
    """Protected hypothesis plus a separately serializable content-free result."""

    hypothesis: PartialHypothesis | FinalizedSegment = field(repr=False)
    envelope: ProviderResultEnvelope


def _decode(
    logits: Any, profile: OnnxSpeechProfile, start: int, duration: int
) -> tuple[TokenTime, ...]:
    import numpy as np

    _check(
        type(logits) is np.ndarray
        and logits.shape == (1, profile.frames, len(profile.labels))
        and logits.dtype == np.float32
        and bool(np.isfinite(logits).all()),
        "invalid_runtime_output",
    )
    valid = (duration + 9) // 10
    rows = logits[0, :valid].astype(np.float64)
    winners = rows.argmax(axis=1)
    # Max-class softmax probability. A run's confidence is its weakest frame.
    confidence = 1 / np.exp(rows - rows.max(axis=1, keepdims=True)).sum(axis=1)
    tokens: list[TokenTime] = []
    i = 0
    while i < valid:
        label = int(winners[i])
        end = i + 1
        while end < valid and winners[end] == label:
            end += 1
        if label != 0:
            tokens.append(
                TokenTime(
                    len(tokens),
                    start + i * 10,
                    start + min(end * 10, duration),
                    round(float(confidence[i:end].min()) * 1000000),
                    profile.labels[label],
                )
            )
        i = end
    return tuple(tokens)


class _OnnxSpeechStream:
    """One bounded, cancellable stream. One inference may be active at a time."""

    def __init__(self, profile: OnnxSpeechProfile, engine: Any):
        self._profile = profile
        self._engine = engine
        self._state = StreamingAsrSession(profile.declaration)
        self._pcm = bytearray()
        self._lock = threading.Lock()
        self._active: Any = None
        self._revision = 0
        self._index = 0

    def report(self) -> StreamingAsrSessionReport:
        """Read aggregate state without exposing hypotheses or PCM."""
        with self._lock:
            return self._state.report()

    def _clear(self) -> None:
        self._pcm[:] = bytes(len(self._pcm))
        self._pcm.clear()
        self._state.drain()

    def cancel(self) -> StreamingAsrSessionReport:
        """Terminate the active native run and discard all unpublished output."""
        with self._lock:
            if self._state.state is StreamState.OPEN:
                self._state.cancel()
                self._clear()
                if self._active is not None:
                    self._active.terminate = True
            return self._state.report()

    def feed(
        self, chunk: AudioChunk, *, end_segment: bool = False
    ) -> SpeechWindowResult:
        """Decode the bounded utterance prefix or finalize and drain this segment."""
        _check(type(chunk) is AudioChunk and type(end_segment) is bool, "invalid_chunk")
        result, _ = self._step(chunk, end_segment, False)
        assert result is not None
        return result

    def finish(self) -> tuple[SpeechWindowResult | None, StreamingAsrSessionReport]:
        """Flush any pending partial as a final segment, then seal the stream."""
        return self._step(None, True, True)

    def _step(
        self, chunk: AudioChunk | None, final: bool, close: bool
    ) -> tuple[SpeechWindowResult | None, StreamingAsrSessionReport]:
        started = time.monotonic_ns()
        with self._lock:
            _check(self._state.state is StreamState.OPEN, "stream_closed")
            _check(self._active is None, "stream_busy")
            if chunk is not None:
                _check(
                    chunk.sample_rate_hz == _RATE
                    and chunk.channel_count == 1
                    and len(chunk.payload) == chunk.duration_ms * 32,
                    "pcm_profile_mismatch",
                )
                # Validates sequence, cumulative lifetime and per-utterance bounds first.
                self._state.push_chunk(chunk)
                self._pcm.extend(chunk.payload)
            if not self._pcm:
                return None, self._state.finalize()
            duration = len(self._pcm) // 32
            start = self._state.stream_duration_ms - duration
            digest = hashlib.sha256(self._pcm).hexdigest()
            # Complete the bounded frontend under the lock: cancellation never
            # races an array view of audio being overwritten in another thread.
            failed = False
            try:
                opts = self._engine.options()
                features = log_mel_features(self._pcm, self._profile.frames)
            except Exception:
                failed = True
            if failed:
                self._state.cancel()
                self._clear()
                raise OnnxSpeechError("frontend_unavailable")
            self._active = opts
        try:
            logits = self._engine.run(features, opts)
            tokens = _decode(logits, self._profile, start, duration)
        except Exception:
            failed = True
        finally:
            features.fill(0)
        with self._lock:
            self._active = None
            _check(self._state.state is StreamState.OPEN, "stream_cancelled")
            if failed:
                self._state.cancel()
                self._clear()
                raise OnnxSpeechError("decode_failed")
            text = "".join(t.token for t in tokens)
            try:
                hypothesis: PartialHypothesis | FinalizedSegment
                if final:
                    hypothesis = FinalizedSegment(
                        segment_index=self._index,
                        text=text,
                        start_ms=start,
                        end_ms=start + duration,
                        token_times=tokens,
                    )
                    self._state.submit_segment(hypothesis)
                    self._index += 1
                    self._revision = 0
                    self._clear()
                else:
                    hypothesis = PartialHypothesis(
                        self._revision, text, token_times=tokens
                    )
                    self._state.submit_partial(hypothesis)
                    self._revision += 1
                envelope = ProviderResultEnvelope(
                    provider_id="onnx-ctc-asr",
                    model_id="local-ctc",
                    input_digest=digest,
                    outcome=ProviderResultOutcome.SUCCESS,
                    duration_ms=(time.monotonic_ns() - started) / 1e6,
                    count_metadata={
                        "sample_count": duration * 16,
                        "token_count": len(tokens),
                        "segment_count": int(final),
                    },
                    output_digest=hashlib.sha256(text.encode()).hexdigest(),
                )
                report = self._state.finalize() if close else self._state.report()
            except Exception:
                if self._state.state is StreamState.OPEN:
                    self._state.cancel()
                self._clear()
            else:
                return SpeechWindowResult(hypothesis, envelope), report
        raise OnnxSpeechError("speech_result_withheld")
