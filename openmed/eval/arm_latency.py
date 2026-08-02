"""Offline SMS-scale latency benchmarking for ARM CPU deployments."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from openmed.eval.metrics import compute_latency_summary

DEFAULT_ARM_LATENCY_CORPUS = Path(__file__).with_name("fixtures") / "sms_clinical.jsonl"
DEFAULT_ARM_LATENCY_BUDGET = Path(__file__).with_name("budgets") / "arm_latency.json"
DEFAULT_ARM_LATENCY_MODEL = "OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-onnx-android"
DEFAULT_ARM_LATENCY_MODEL_REVISION = "79f7db205869b1be4be23ac4f42aa95bdedc5aee"
MAX_SMS_CHARACTERS = 280

LatencyRunner = Callable[[Any, "LatencyDocument"], Any]
Clock = Callable[[], float]
RssSampler = Callable[[], int | None]


@dataclass(frozen=True)
class LatencyDocument:
    """One explicitly synthetic SMS-scale clinical benchmark document."""

    document_id: str
    text: str
    language: str = "en"
    synthetic: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "LatencyDocument":
        """Build and validate a document from a JSON-compatible mapping."""

        return cls(
            document_id=str(data.get("id") or data.get("document_id") or ""),
            text=str(data.get("text") or ""),
            language=str(data.get("language") or "en"),
            synthetic=data.get("synthetic") is True,
        )


@dataclass(frozen=True)
class ArmLatencyBudget:
    """Committed p95 reference and permitted regression tolerance."""

    name: str
    reference_device: str
    reference_cpu: str
    reference_p95_ms: float
    regression_tolerance: float
    model_id: str
    model_revision: str
    artifact_sha256: str
    quantization: str

    @property
    def maximum_p95_ms(self) -> float:
        """Return the inclusive p95 gate after applying the tolerance."""

        return self.reference_p95_ms * (1.0 + self.regression_tolerance)

    def evaluate(self, measured_p95_ms: float) -> "ArmLatencyVerdict":
        """Compare a measured p95 against the inclusive regression gate."""

        return ArmLatencyVerdict(
            measured_p95_ms=measured_p95_ms,
            maximum_p95_ms=self.maximum_p95_ms,
            passed=measured_p95_ms <= self.maximum_p95_ms,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible budget metadata."""

        return {
            "maximum_p95_ms": round(self.maximum_p95_ms, 6),
            "artifact_sha256": self.artifact_sha256,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "name": self.name,
            "quantization": self.quantization,
            "reference_cpu": self.reference_cpu,
            "reference_device": self.reference_device,
            "reference_p95_ms": self.reference_p95_ms,
            "regression_tolerance": self.regression_tolerance,
        }


@dataclass(frozen=True)
class ArmLatencyVerdict:
    """Result of applying the committed p95 regression gate."""

    measured_p95_ms: float
    maximum_p95_ms: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON-compatible gate output."""

        return {
            "maximum_p95_ms": round(self.maximum_p95_ms, 6),
            "measured_p95_ms": round(self.measured_p95_ms, 6),
            "passed": self.passed,
        }


@dataclass(frozen=True)
class ArmLatencyReport:
    """Aggregate latency, throughput, memory, and provenance report."""

    document_count: int
    sample_count: int
    total_seconds: float
    throughput_texts_per_second: float
    p50_ms: float
    p95_ms: float
    peak_rss_mib: float | None
    model: Mapping[str, Any]
    machine: Mapping[str, Any]
    corpus: Mapping[str, Any]
    budget: ArmLatencyBudget
    verdict: ArmLatencyVerdict
    warmup_runs: int
    repeat: int
    generated_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return whether the p95 latency stayed inside the gated envelope."""

        return self.verdict.passed

    def to_dict(self) -> dict[str, Any]:
        """Return a stable aggregate report without benchmark source text."""

        return {
            "benchmark": "sms_arm_latency",
            "budget": self.budget.to_dict(),
            "corpus": dict(self.corpus),
            "document_count": self.document_count,
            "generated_at": self.generated_at,
            "latency_ms": {
                "p50": round(self.p50_ms, 6),
                "p95": round(self.p95_ms, 6),
            },
            "machine": dict(self.machine),
            "metadata": dict(self.metadata),
            "model": dict(self.model),
            "offline": True,
            "passed": self.passed,
            "peak_rss_mib": (
                None if self.peak_rss_mib is None else round(self.peak_rss_mib, 6)
            ),
            "repeat": self.repeat,
            "sample_count": self.sample_count,
            "schema_version": 1,
            "throughput_texts_per_second": round(
                self.throughput_texts_per_second,
                6,
            ),
            "total_seconds": round(self.total_seconds, 6),
            "verdict": self.verdict.to_dict(),
            "warmup_runs": self.warmup_runs,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path) -> Path:
        """Write the deterministic report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json() + "\n", encoding="utf-8")
        return output_path


def load_latency_documents(
    path: str | Path = DEFAULT_ARM_LATENCY_CORPUS,
) -> list[LatencyDocument]:
    """Load the committed synthetic SMS corpus from JSONL."""

    corpus_path = Path(path)
    documents = [
        LatencyDocument.from_mapping(json.loads(line))
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_documents(documents)
    return documents


def load_arm_latency_budget(
    path: str | Path = DEFAULT_ARM_LATENCY_BUDGET,
) -> ArmLatencyBudget:
    """Load and validate the committed ARM latency budget."""

    budget_path = Path(path)
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("ARM latency budget must contain a JSON object")
    reference = payload.get("reference")
    if not isinstance(reference, Mapping):
        raise ValueError("ARM latency budget must contain a reference object")
    budget = ArmLatencyBudget(
        name=str(payload.get("name") or ""),
        reference_device=str(reference.get("device") or ""),
        reference_cpu=str(reference.get("cpu") or ""),
        reference_p95_ms=float(reference.get("p95_latency_ms", 0.0)),
        regression_tolerance=float(payload.get("regression_tolerance", -1.0)),
        model_id=str(reference.get("model_id") or ""),
        model_revision=str(reference.get("model_revision") or ""),
        artifact_sha256=str(reference.get("artifact_sha256") or ""),
        quantization=str(reference.get("quantization") or ""),
    )
    _validate_budget(budget)
    return budget


def run_arm_latency_benchmark(
    model: Any,
    *,
    model_id: str,
    model_revision: str,
    documents: Sequence[LatencyDocument] | None = None,
    budget: ArmLatencyBudget | None = None,
    corpus_path: str | Path = DEFAULT_ARM_LATENCY_CORPUS,
    runner: LatencyRunner | None = None,
    clock: Clock | None = None,
    rss_sampler: RssSampler | None = None,
    warmup_runs: int = 1,
    repeat: int = 3,
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> ArmLatencyReport:
    """Benchmark one cached INT8 ONNX model over synthetic SMS-scale texts.

    Warm-up calls are excluded from p50, p95, and throughput. The caller is
    responsible for activating OpenMed's offline socket guard around model
    loading and this function.
    """

    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if repeat < 1:
        raise ValueError("repeat must be positive")

    selected_documents = list(
        documents if documents is not None else load_latency_documents(corpus_path)
    )
    _validate_documents(selected_documents)
    selected_budget = budget or load_arm_latency_budget()
    _validate_budget(selected_budget)
    if model_id != selected_budget.model_id:
        raise ValueError("model id does not match the committed ARM latency budget")
    if model_revision != selected_budget.model_revision:
        raise ValueError(
            "model revision does not match the committed ARM latency budget"
        )
    artifact_sha256 = _validate_int8_model(
        model,
        expected_sha256=selected_budget.artifact_sha256,
    )

    run_document = runner or _default_latency_runner
    now = clock or time.perf_counter
    sample_rss = rss_sampler or _peak_rss_bytes

    for index in range(warmup_runs):
        run_document(model, selected_documents[index % len(selected_documents)])

    rss_values: list[int] = []
    initial_rss = sample_rss()
    if initial_rss is not None:
        rss_values.append(initial_rss)

    latencies_ms: list[float] = []
    total_started = now()
    for _ in range(repeat):
        for document in selected_documents:
            started = now()
            run_document(model, document)
            latencies_ms.append(max(now() - started, 0.0) * 1000.0)
            current_rss = sample_rss()
            if current_rss is not None:
                rss_values.append(current_rss)
    total_seconds = max(now() - total_started, 0.0)

    latency = compute_latency_summary(latencies_ms)
    sample_count = len(latencies_ms)
    throughput = sample_count / total_seconds if total_seconds > 0.0 else 0.0
    peak_rss_mib = max(rss_values) / (1024.0 * 1024.0) if rss_values else None
    verdict = selected_budget.evaluate(latency.p95_ms)
    resolved_corpus_path = Path(corpus_path)

    return ArmLatencyReport(
        document_count=len(selected_documents),
        sample_count=sample_count,
        total_seconds=total_seconds,
        throughput_texts_per_second=throughput,
        p50_ms=latency.p50_ms,
        p95_ms=latency.p95_ms,
        peak_rss_mib=peak_rss_mib,
        model=_model_metadata(
            model,
            model_id=model_id,
            revision=model_revision,
            artifact_sha256=artifact_sha256,
        ),
        machine=_machine_metadata(),
        corpus={
            "max_characters": max(len(item.text) for item in selected_documents),
            "name": resolved_corpus_path.name,
            "synthetic": True,
        },
        budget=selected_budget,
        verdict=verdict,
        warmup_runs=warmup_runs,
        repeat=repeat,
        generated_at=generated_at or _utc_now(),
        metadata=dict(metadata or {}),
    )


def _default_latency_runner(model: Any, document: LatencyDocument) -> Any:
    return model.predict(document.text)


def _validate_documents(documents: Sequence[LatencyDocument]) -> None:
    if not documents:
        raise ValueError("latency benchmark corpus must contain at least one document")
    seen: set[str] = set()
    for document in documents:
        if not document.document_id:
            raise ValueError("latency benchmark documents require a non-empty id")
        if document.document_id in seen:
            raise ValueError(f"duplicate latency document id: {document.document_id}")
        seen.add(document.document_id)
        if not document.synthetic:
            raise ValueError(
                f"latency benchmark document {document.document_id!r} is not synthetic"
            )
        if not document.text:
            raise ValueError(
                f"latency benchmark document {document.document_id!r} is empty"
            )
        if len(document.text) > MAX_SMS_CHARACTERS:
            raise ValueError(
                f"latency benchmark document {document.document_id!r} exceeds "
                f"{MAX_SMS_CHARACTERS} characters"
            )


def _validate_budget(budget: ArmLatencyBudget) -> None:
    if not budget.name:
        raise ValueError("ARM latency budget requires a name")
    if budget.reference_p95_ms <= 0.0:
        raise ValueError("ARM latency reference p95 must be positive")
    if not 0.0 <= budget.regression_tolerance <= 1.0:
        raise ValueError("ARM latency regression_tolerance must be between 0 and 1")
    if budget.quantization != "int8":
        raise ValueError("ARM latency budget must target int8 quantization")
    if not budget.model_id:
        raise ValueError("ARM latency budget requires pinned model provenance")
    if len(budget.model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in budget.model_revision
    ):
        raise ValueError(
            "ARM latency budget requires a lowercase 40-character revision"
        )
    if len(budget.artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in budget.artifact_sha256
    ):
        raise ValueError("ARM latency budget requires a lowercase SHA-256 digest")


def _validate_int8_model(model: Any, *, expected_sha256: str) -> str:
    variant = str(getattr(model, "variant", "")).lower()
    model_path = Path(str(getattr(model, "model_path", "")))
    if variant != "int8" or model_path.name != "model_int8.onnx":
        raise ValueError("ARM latency benchmark requires model_int8.onnx")
    if not model_path.is_file():
        raise ValueError("ARM latency benchmark model_int8.onnx does not exist")
    artifact_sha256 = _sha256_file(model_path)
    if artifact_sha256 != expected_sha256:
        raise ValueError(
            "ARM latency benchmark model_int8.onnx does not match the committed "
            "SHA-256 digest"
        )
    return artifact_sha256


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_metadata(
    model: Any,
    *,
    model_id: str,
    revision: str,
    artifact_sha256: str,
) -> dict[str, Any]:
    model_path = Path(str(getattr(model, "model_path", "")))
    return {
        "artifact": model_path.name,
        "artifact_sha256": artifact_sha256,
        "id": model_id,
        "onnxruntime_version": _package_version("onnxruntime"),
        "quantization": "int8",
        "revision": revision,
    }


def _machine_metadata() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "architecture": uname.machine,
        "cpu_count": os.cpu_count(),
        "operating_system": uname.system,
        "processor": uname.processor or platform.processor(),
        "release": uname.release,
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:
        return None
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


__all__ = [
    "DEFAULT_ARM_LATENCY_BUDGET",
    "DEFAULT_ARM_LATENCY_CORPUS",
    "DEFAULT_ARM_LATENCY_MODEL",
    "DEFAULT_ARM_LATENCY_MODEL_REVISION",
    "MAX_SMS_CHARACTERS",
    "ArmLatencyBudget",
    "ArmLatencyReport",
    "ArmLatencyVerdict",
    "LatencyDocument",
    "load_arm_latency_budget",
    "load_latency_documents",
    "run_arm_latency_benchmark",
]
