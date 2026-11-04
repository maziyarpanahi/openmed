"""Optional CPU diarization from caller-supplied local ONNX artifacts."""

from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

from openmed.multimodal.streaming.speech.diarization import (
    MAX_TURNS,
    AcousticTurn,
    ClusterEvidence,
    DiarizationBatch,
    DiarizationError,
    DiarizationRequest,
    digest,
    require,
)

MAX_RESPONSE_BYTES = 256 * 1024


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalDiarizationArtifact:
    """Caller-supplied file and pinned provenance, with no license allowlist.

    Args:
        path: Local regular ONNX file; hidden from diagnostics and reports.
        sha256: Expected artifact bytes.
        revision_digest: Hash of the caller's immutable source/revision record.
        terms_digest: Hash of the terms snapshot reviewed by the caller.

    Digests record provenance, not legal approval. No weight is downloaded or
    bundled, and this adapter never accepts terms or rechecks a remote license.
    """

    path: Path = field(repr=False)
    sha256: str
    revision_digest: str
    terms_digest: str

    def __post_init__(self) -> None:
        require(isinstance(self.path, Path) and self.path.is_absolute())
        for value in (self.sha256, self.revision_digest, self.terms_digest):
            require(
                type(value) is str
                and len(value) == 64
                and all(char in "0123456789abcdef" for char in value)
            )

    def provenance(self) -> dict[str, str]:
        """Return only artifact and source/terms digests, never the file path."""
        return {
            "sha256": self.sha256,
            "revision_digest": self.revision_digest,
            "terms_digest": self.terms_digest,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class DiarizationLimits:
    """Per-call watchdog limits, including loading and artifact verification.

    RSS is sampled every 20 ms; it is not an OS allocation quota. Deadline and
    cancellation terminate/reap the native worker, including blocked native calls.
    """

    timeout_seconds: float = 30.0
    max_rss_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        require(
            type(self.timeout_seconds) in (int, float)
            and math.isfinite(self.timeout_seconds)
            and 0 < self.timeout_seconds <= 300
        )
        require(
            type(self.max_rss_bytes) is int
            and 1 <= self.max_rss_bytes <= 8 * 1024 * 1024 * 1024
        )


def _exchange(
    command: list[str], payload: bytes, limits: DiarizationLimits, cancelled: Event
) -> bytes:
    # Imports are optional and kept out of the base SDK import path.
    try:
        import psutil
    except ImportError:
        raise DiarizationError("diarization_extra_required") from None
    deadline = time.monotonic() + limits.timeout_seconds
    if cancelled.is_set():
        raise DiarizationError("diarization_cancelled")
    environment = {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    process = None
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and worker path
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        probe = psutil.Process(process.pid)
        output = bytearray()
        offset = 0
        with selectors.DefaultSelector() as selector:
            for stream, event in (
                (process.stdin, selectors.EVENT_WRITE),
                (process.stdout, selectors.EVENT_READ),
            ):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event)
            while selector.get_map() or process.poll() is None:
                if cancelled.is_set():
                    raise DiarizationError("diarization_cancelled")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DiarizationError("diarization_timeout")
                if process.poll() is None:
                    try:
                        rss = probe.memory_info().rss
                    except psutil.NoSuchProcess:
                        rss = 0
                    except psutil.Error:
                        raise DiarizationError(
                            "diarization_memory_unavailable"
                        ) from None
                    if rss > limits.max_rss_bytes:
                        raise DiarizationError("diarization_memory_limit")
                for key, event in selector.select(min(0.02, remaining)):
                    try:
                        if event == selectors.EVENT_WRITE:
                            offset += os.write(key.fd, payload[offset : offset + 8192])
                            if offset == len(payload):
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                        else:
                            chunk = os.read(key.fd, 8192)
                            output.extend(chunk)
                            if len(output) > MAX_RESPONSE_BYTES:
                                raise DiarizationError("diarization_output_limit")
                            if not chunk:
                                selector.unregister(key.fileobj)
                    except (BlockingIOError, InterruptedError):
                        continue
        if process.returncode != 0 or offset != len(payload):
            raise DiarizationError("diarization_worker_failed")
        return bytes(output)
    except psutil.Error:
        raise DiarizationError("diarization_memory_unavailable") from None
    except (OSError, ValueError):
        # DiarizationError is a ValueError; preserve only our controlled codes.
        error = sys.exception()
        if type(error) is DiarizationError:
            raise
        raise DiarizationError("diarization_worker_failed") from None
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for pipe in (process.stdin, process.stdout):
                if pipe is not None:
                    pipe.close()


class SherpaDiarizationProvider:
    """Disposable local Sherpa-ONNX CPU worker for Linux and macOS.

    Args:
        segmentation: Caller-supplied compatible segmentation ONNX artifact.
        embedding: Caller-supplied compatible embedding ONNX artifact.
        limits: Deadline and sampled RSS limits for each call.

    The worker receives bounded audio through pipes and writes no audio files.
    Models must be trusted, stable local files; this is process lifecycle
    isolation, not a sandbox for hostile native code or hostile model files.
    Cluster numbers are batch-local, so new requests receive fresh cluster refs.
    No cross-window voice identity matching or clinical role inference is done.
    """

    def __init__(
        self,
        *,
        segmentation: LocalDiarizationArtifact,
        embedding: LocalDiarizationArtifact,
        limits: DiarizationLimits = DiarizationLimits(),
    ) -> None:
        require(type(segmentation) is LocalDiarizationArtifact)
        require(type(embedding) is LocalDiarizationArtifact)
        require(type(limits) is DiarizationLimits)
        self._segmentation, self._embedding, self._limits = (
            segmentation,
            embedding,
            limits,
        )

    def observe(
        self, request: DiarizationRequest, *, cancelled: Event | None = None
    ) -> DiarizationBatch:
        """Run one bounded local inference; failures never return partial turns."""
        require(type(request) is DiarizationRequest)
        if sys.platform not in {"darwin", "linux"}:
            raise DiarizationError("diarization_platform_unsupported")
        metadata: dict[str, Any] = {
            "segmentation": {
                **asdict(self._segmentation),
                "path": str(self._segmentation.path),
            },
            "embedding": {**asdict(self._embedding), "path": str(self._embedding.path)},
            "pcm_bytes": len(request.pcm),
        }
        header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
        require(len(header) <= 16384)
        payload = len(header).to_bytes(4, "big") + header + request.pcm
        raw = _exchange(
            [
                sys.executable,
                "-I",
                str(Path(__file__).with_name("_diarization_worker.py")),
            ],
            payload,
            self._limits,
            cancelled if cancelled is not None else Event(),
        )
        try:
            response = json.loads(raw)
            require(type(response) is dict and set(response) == {"turns", "runtime"})
            require(response["runtime"] == "sherpa-onnx:2.0.1")
            require(
                type(response["turns"]) is list and len(response["turns"]) <= MAX_TURNS
            )
            turns = []
            for row in response["turns"]:
                require(type(row) is list and len(row) == 4)
                start, end, cluster, clipped = row
                require(type(clipped) is bool)
                require(type(cluster) is int and 0 <= cluster < 32)
                turn = AcousticTurn(
                    start_sample=start,
                    end_sample=end,
                    cluster_ref=digest([request.binding, cluster])[:32],
                    boundary_clipped=clipped,
                    evidence=ClusterEvidence.UNSTABLE
                    if clipped
                    else ClusterEvidence.UNREVIEWED,
                )
                require(turn.end_sample <= request.sample_count)
                turns.append(turn)
            return DiarizationBatch(
                request_binding=request.binding,
                provenance_digest=digest(
                    {
                        "runtime": response["runtime"],
                        "segmentation": self._segmentation.provenance(),
                        "embedding": self._embedding.provenance(),
                        "config": "cpu-1thread-auto-clusters-threshold0.5-on0.3-off0.5-window-intersection",
                    }
                ),
                turns=tuple(turns),
            )
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            raise DiarizationError("invalid_diarization_response") from None
