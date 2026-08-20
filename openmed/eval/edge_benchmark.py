"""Offline, aggregate-only benchmarks for aarch64 edge installations."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import platform
import sys
import sysconfig
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.core.offline import network_blocked_if_offline
from openmed.eval.metrics import LatencyMetrics, compute_latency_summary
from openmed.eval.perf import (
    DEFAULT_PERF_WORKLOAD_PATH,
    PerfDocument,
    load_perf_documents,
)

DEFAULT_EDGE_WORKLOAD = DEFAULT_PERF_WORKLOAD_PATH
DEFAULT_EDGE_IDENTITY_MODEL = (
    Path(__file__).with_name("fixtures") / "edge_identity.onnx.b64"
)
EDGE_PROFILES = ("jetson-nano", "raspberry-pi-5")
SUPPORTED_AARCH64_NAMES = frozenset({"aarch64", "arm64"})

EdgeInference = Callable[[str], Any]
EdgeRuntimeLoader = Callable[[], "EdgeRuntime"]
Clock = Callable[[], float]
RssSampler = Callable[[], int | None]


@dataclass(frozen=True)
class EdgeRuntime:
    """Loaded local runtime plus PHI-safe provenance for a benchmark run."""

    name: str
    backend: str
    backend_version: str | None
    execution_provider: str
    artifact_sha256: str
    inference: EdgeInference = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate runtime metadata without model paths or input text."""

        return {
            "artifact_sha256": self.artifact_sha256,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "execution_provider": self.execution_provider,
            "name": self.name,
        }


@dataclass(frozen=True)
class EdgeBenchmarkReport:
    """Cold-start, throughput, memory, install, and machine measurements."""

    profile: str
    runtime: EdgeRuntime
    document_count: int
    sample_count: int
    token_count: int
    total_seconds: float
    tokens_per_second: float
    cold_start_ms: float
    steady_state_latency: LatencyMetrics
    peak_rss_bytes: int | None
    install_size_bytes: int
    repeat: int
    workload_name: str
    workload_sha256: str
    machine: Mapping[str, Any]
    generated_at: str

    @property
    def peak_rss_mib(self) -> float | None:
        """Return peak resident memory in mebibytes when RSS is available."""

        if self.peak_rss_bytes is None:
            return None
        return self.peak_rss_bytes / (1024.0 * 1024.0)

    @property
    def install_size_mib(self) -> float:
        """Return the measured installed footprint in mebibytes."""

        return self.install_size_bytes / (1024.0 * 1024.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable result record that never includes benchmark text."""

        return {
            "benchmark": "edge_sbc",
            "cold_start_ms": round(self.cold_start_ms, 6),
            "generated_at": self.generated_at,
            "install_size_bytes": self.install_size_bytes,
            "install_size_mib": round(self.install_size_mib, 6),
            "machine": dict(self.machine),
            "network_guard": "socket-blocked",
            "offline": True,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_mib": (
                None if self.peak_rss_mib is None else round(self.peak_rss_mib, 6)
            ),
            "profile": self.profile,
            "repeat": self.repeat,
            "runtime": self.runtime.to_dict(),
            "sample_count": self.sample_count,
            "schema_version": 1,
            "steady_state_latency_ms": {
                "p50": round(self.steady_state_latency.p50_ms, 6),
                "p95": round(self.steady_state_latency.p95_ms, 6),
                "p99": round(self.steady_state_latency.p99_ms, 6),
            },
            "token_count": self.token_count,
            "tokens_per_second": round(self.tokens_per_second, 6),
            "total_seconds": round(self.total_seconds, 6),
            "workload": {
                "document_count": self.document_count,
                "name": self.workload_name,
                "sha256": self.workload_sha256,
                "synthetic": True,
                "token_definition": "unicode-whitespace-fields",
            },
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the result record as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the deterministic result record to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path


def load_edge_documents(
    path: str | Path = DEFAULT_EDGE_WORKLOAD,
) -> list[PerfDocument]:
    """Load and fail closed on the committed explicitly synthetic workload."""

    documents = load_perf_documents(path)
    _validate_synthetic_documents(documents)
    return documents


def load_synthetic_ort_runtime(
    model_path: str | Path = DEFAULT_EDGE_IDENTITY_MODEL,
) -> EdgeRuntime:
    """Load the tiny committed ONNX identity graph on the CPU provider.

    The graph is a runtime/install smoke fixture, not a clinical model. It lets
    CI exercise the same aarch64 ONNX Runtime wheel used by local model
    deployments without downloading weights or making a network request.
    """

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "The edge benchmark requires the minimal edge profile. Install "
            "with: pip install 'openmed[edge-sbc]'"
        ) from exc

    encoded = Path(model_path).read_text(encoding="ascii").strip()
    try:
        model_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("edge identity fixture is not valid base64") from exc
    if not model_bytes:
        raise ValueError("edge identity fixture is empty")

    available_providers = set(ort.get_available_providers())
    if "CPUExecutionProvider" not in available_providers:
        raise RuntimeError("ONNX Runtime does not expose CPUExecutionProvider")
    session = ort.InferenceSession(
        model_bytes,
        providers=["CPUExecutionProvider"],
    )

    def infer(text: str) -> Any:
        token_lengths = [float(len(token)) for token in text.split()]
        if not token_lengths:
            token_lengths = [0.0]
        inputs = np.asarray([token_lengths], dtype=np.float32)
        return session.run(["output"], {"tokens": inputs})[0]

    return EdgeRuntime(
        name="synthetic-onnx-identity",
        backend="onnxruntime",
        backend_version=str(getattr(ort, "__version__", "unknown")),
        execution_provider="CPUExecutionProvider",
        artifact_sha256=hashlib.sha256(model_bytes).hexdigest(),
        inference=infer,
    )


def load_local_onnx_runtime(
    model_path: str | Path,
    *,
    variant: str = "int8",
) -> EdgeRuntime:
    """Load a caller-supplied local ONNX artifact without a Hub fallback."""

    local_path = Path(model_path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(
            "edge benchmarking requires an existing local model path"
        )

    from openmed.onnx.inference import OnnxModel

    model = OnnxModel.from_pretrained(
        local_path,
        variant=variant,
        local_files_only=True,
        providers=("CPUExecutionProvider",),
    )
    artifact_sha256 = _sha256_file(model.model_path)
    providers = tuple(getattr(model.session, "get_providers", lambda: ())())
    if providers and providers[0] != "CPUExecutionProvider":
        raise RuntimeError("edge benchmark requires CPUExecutionProvider")
    return EdgeRuntime(
        name=f"local-onnx-{model.variant}",
        backend="onnxruntime",
        backend_version=_package_version("onnxruntime"),
        execution_provider="CPUExecutionProvider",
        artifact_sha256=artifact_sha256,
        inference=model.predict,
    )


def measure_install_size(path: str | Path) -> int:
    """Return stable installed bytes, excluding symlinks and bytecode caches."""

    install_path = Path(path)
    if not install_path.exists():
        raise FileNotFoundError("install footprint path does not exist")
    if install_path.is_file():
        return install_path.stat().st_size
    return sum(
        item.stat().st_size
        for item in install_path.rglob("*")
        if item.is_file()
        and not item.is_symlink()
        and "__pycache__" not in item.parts
        and item.suffix not in {".pyc", ".pyo"}
    )


def default_install_path() -> Path:
    """Return the active interpreter's pure-Python installation directory."""

    return Path(sysconfig.get_paths()["purelib"])


def run_edge_benchmark(
    *,
    profile: str,
    documents: Sequence[PerfDocument] | None = None,
    corpus_path: str | Path = DEFAULT_EDGE_WORKLOAD,
    runtime_loader: EdgeRuntimeLoader | None = None,
    repeat: int = 10,
    install_path: str | Path | None = None,
    install_size_bytes: int | None = None,
    require_aarch64: bool = False,
    clock: Clock | None = None,
    rss_sampler: RssSampler | None = None,
    generated_at: str | None = None,
) -> EdgeBenchmarkReport:
    """Benchmark an offline local runtime over explicitly synthetic notes.

    Cold start includes runtime/model loading plus the first inference. Steady
    state then runs ``repeat`` complete corpus passes and reports whitespace
    tokens per second, per-note p50/p95/p99 latency, and process high-water RSS.
    The entire load and inference window runs under OpenMed's outbound socket
    guard. Reports contain aggregate counts and corpus/model hashes, never note
    text, document identifiers, detected spans, or local paths.

    Args:
        profile: ``raspberry-pi-5`` or ``jetson-nano`` budget profile name.
        documents: Optional explicitly synthetic documents. The committed
            synthetic workload is used when omitted.
        corpus_path: JSON/JSONL workload path used when ``documents`` is omitted.
        runtime_loader: Local runtime loader. Defaults to the committed tiny ONNX
            Runtime CPU smoke graph.
        repeat: Number of measured steady-state corpus passes.
        install_path: Isolated install directory to size. Defaults to the active
            interpreter's site-packages directory; generated bytecode caches and
            symlinks are excluded.
        install_size_bytes: Explicit measured install size, primarily for tests.
        require_aarch64: Fail unless the machine reports ``aarch64`` or ``arm64``.
        clock: Optional monotonic clock for deterministic tests.
        rss_sampler: Optional peak-RSS sampler returning bytes.
        generated_at: Optional ISO timestamp override.

    Returns:
        Aggregate :class:`EdgeBenchmarkReport` suitable for the footprint gate.
    """

    if profile not in EDGE_PROFILES:
        choices = ", ".join(EDGE_PROFILES)
        raise ValueError(f"unknown edge profile; expected one of: {choices}")
    if repeat < 1:
        raise ValueError("repeat must be a positive integer")

    machine = _machine_metadata()
    if require_aarch64 and machine["architecture"] not in SUPPORTED_AARCH64_NAMES:
        raise RuntimeError("edge benchmark requires an aarch64/arm64 runner")

    selected_documents = list(
        documents if documents is not None else load_edge_documents(corpus_path)
    )
    _validate_synthetic_documents(selected_documents)
    measured_install_size = (
        install_size_bytes
        if install_size_bytes is not None
        else measure_install_size(install_path or default_install_path())
    )
    if (
        isinstance(measured_install_size, bool)
        or not isinstance(measured_install_size, int)
        or measured_install_size < 0
    ):
        raise ValueError("install size must be a non-negative integer")

    load_runtime = runtime_loader or load_synthetic_ort_runtime
    now = clock or time.perf_counter
    sample_rss = rss_sampler or _peak_rss_bytes
    rss_values: list[int] = []
    initial_rss = sample_rss()
    if initial_rss is not None:
        rss_values.append(initial_rss)

    with network_blocked_if_offline(local_only=True):
        cold_started = now()
        runtime = load_runtime()
        if not isinstance(runtime, EdgeRuntime):
            raise TypeError("runtime_loader must return an EdgeRuntime")
        runtime.inference(selected_documents[0].text)
        cold_start_ms = max(now() - cold_started, 0.0) * 1000.0
        cold_rss = sample_rss()
        if cold_rss is not None:
            rss_values.append(cold_rss)

        latencies_ms: list[float] = []
        total_started = now()
        for _ in range(repeat):
            for document in selected_documents:
                started = now()
                runtime.inference(document.text)
                latencies_ms.append(max(now() - started, 0.0) * 1000.0)
                current_rss = sample_rss()
                if current_rss is not None:
                    rss_values.append(current_rss)
        total_seconds = max(now() - total_started, 0.0)

    tokens_per_pass = sum(_count_tokens(item.text) for item in selected_documents)
    token_count = tokens_per_pass * repeat
    tokens_per_second = token_count / total_seconds if total_seconds > 0.0 else 0.0
    workload_name = (
        DEFAULT_EDGE_WORKLOAD.name
        if documents is None
        and Path(corpus_path).resolve() == DEFAULT_EDGE_WORKLOAD.resolve()
        else "caller-supplied-synthetic"
    )
    return EdgeBenchmarkReport(
        profile=profile,
        runtime=runtime,
        document_count=len(selected_documents),
        sample_count=len(latencies_ms),
        token_count=token_count,
        total_seconds=total_seconds,
        tokens_per_second=tokens_per_second,
        cold_start_ms=cold_start_ms,
        steady_state_latency=compute_latency_summary(latencies_ms),
        peak_rss_bytes=max(rss_values) if rss_values else None,
        install_size_bytes=measured_install_size,
        repeat=repeat,
        workload_name=workload_name,
        workload_sha256=_workload_sha256(selected_documents),
        machine=machine,
        generated_at=generated_at or _utc_now(),
    )


def _validate_synthetic_documents(documents: Sequence[PerfDocument]) -> None:
    if not documents:
        raise ValueError("edge workload must contain at least one document")
    seen: set[str] = set()
    for index, document in enumerate(documents, start=1):
        if not document.text.strip():
            raise ValueError(f"edge workload document {index} is empty")
        if document.document_id in seen:
            raise ValueError("edge workload contains duplicate document ids")
        seen.add(document.document_id)
        source = str(document.metadata.get("source", "")).strip().lower()
        if source != "synthetic":
            raise ValueError(
                f"edge workload document {index} must declare source=synthetic"
            )


def _count_tokens(text: str) -> int:
    return len(text.split())


def _workload_sha256(documents: Sequence[PerfDocument]) -> str:
    canonical = [
        {
            "language": document.language,
            "source": "synthetic",
            "text": document.text,
        }
        for document in documents
    ]
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _machine_metadata() -> dict[str, Any]:
    return {
        "architecture": platform.machine().strip().lower(),
        "cpu_count": os.cpu_count(),
        "operating_system": platform.system(),
        "python": platform.python_version(),
    }


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline edge benchmark and always write one result record."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=EDGE_PROFILES, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional existing local ONNX artifact directory or file.",
    )
    parser.add_argument(
        "--variant",
        choices=("int8", "fp32", "fp16", "auto"),
        default="int8",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_EDGE_WORKLOAD)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument(
        "--install-path",
        type=Path,
        default=None,
        help="Isolated site-packages/target directory whose files are measured.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edge-benchmark-report.json"),
    )
    parser.add_argument("--require-aarch64", action="store_true")
    parser.add_argument("--generated-at", default=None)
    args = parser.parse_args(argv)

    loader: EdgeRuntimeLoader | None = None
    if args.model is not None:
        loader = lambda: load_local_onnx_runtime(args.model, variant=args.variant)

    try:
        report = run_edge_benchmark(
            profile=args.profile,
            corpus_path=args.corpus,
            runtime_loader=loader,
            repeat=args.repeat,
            install_path=args.install_path,
            require_aarch64=args.require_aarch64,
            generated_at=args.generated_at,
        )
        report.write_json(args.output)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Edge benchmark failed: {exc}", file=sys.stderr)
        return 2

    print(report.to_json())
    return 0


__all__ = [
    "DEFAULT_EDGE_IDENTITY_MODEL",
    "DEFAULT_EDGE_WORKLOAD",
    "EDGE_PROFILES",
    "EdgeBenchmarkReport",
    "EdgeRuntime",
    "default_install_path",
    "load_edge_documents",
    "load_local_onnx_runtime",
    "load_synthetic_ort_runtime",
    "measure_install_size",
    "run_edge_benchmark",
]


if __name__ == "__main__":
    raise SystemExit(main())
