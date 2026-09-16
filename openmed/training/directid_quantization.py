"""Quantized runtime export and G4/G5 evidence for DirectID Tiny candidates.

The module consumes an existing floating-point candidate and delegates runtime
measurement to a caller-supplied local evaluator. It writes unsigned,
PHI-safe evidence for the later certification task; it never publishes a model
or makes a final release decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from openmed.core.repro_hash import compute_canonical_payload_hash
from openmed.eval.quant_delta import evaluate_quant_recall_delta
from openmed.eval.tiers import TIERS
from openmed.training.directid import (
    DIRECTID_TINY_HEAD_CONTRACT,
    DirectIDHeadContract,
    validate_directid_contract,
)

DIRECTID_QUANTIZATION_SCHEMA_VERSION = "openmed.training.directid_quantization.v1"
DIRECTID_QUANTIZATION_MANIFEST_SCHEMA_VERSION = (
    "openmed.training.directid_quantization_manifest.v1"
)
DIRECTID_CANDIDATE_SCHEMA_VERSION = "openmed.training.directid_candidate.v1"
DIRECTID_CANDIDATE_FORMAT = "pytorch-fp32"
DIRECTID_QUANTIZATION_EVIDENCE_FILENAME = "directid_quantization_evidence.json"
DIRECTID_QUANTIZATION_MANIFEST_FILENAME = "directid_quantization_manifest.json"

DIRECTID_INT8_EXPORT_FORMATS = (
    "mlx-8bit",
    "coreml-int8",
    "onnx-int8",
)
DIRECTID_INT4_EXPORT_FORMATS = (
    "mlx-4bit",
    "coreml-int4",
)

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PHI_SHAPED_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b\d{10,}\b"),
)


class DirectIDQuantizationError(ValueError):
    """Raised when DirectID quantization cannot emit safe, complete evidence."""


@dataclass(frozen=True)
class DirectIDExportRequest:
    """One local runtime export requested by the DirectID quantization run."""

    checkpoint_path: Path
    output_dir: Path
    format: str
    runtime: str
    precision: str
    bits: int


@dataclass(frozen=True)
class DirectIDEvaluationRequest:
    """One exported artifact awaiting local recall and device measurement."""

    artifact_path: Path
    format: str
    runtime: str
    precision: str
    bits: int
    fp_parent_per_label_recall: Mapping[str, float]
    eval_set_hash: str


@dataclass(frozen=True)
class DirectIDArtifactMeasurement:
    """Aggregate, PHI-free recall and Tiny-tier measurements for an artifact."""

    per_label_recall: Mapping[str, float]
    eval_set_hash: str
    performance_fixture_hash: str
    device: str
    sample_count: int
    p50_ms: float
    p95_ms: float
    ram_mb: float


@dataclass(frozen=True)
class DirectIDQuantizationArtifacts:
    """Files and parsed evidence produced by a DirectID quantization run."""

    evidence: Mapping[str, Any]
    manifest: Mapping[str, Any]
    artifact_paths: Mapping[str, Path]
    evidence_path: Path
    manifest_path: Path


DirectIDArtifactExporter = Callable[[DirectIDExportRequest], str | Path]
DirectIDArtifactEvaluator = Callable[
    [DirectIDEvaluationRequest], DirectIDArtifactMeasurement
]


@dataclass(frozen=True)
class _FormatSpec:
    runtime: str
    precision: str
    bits: int


_FORMAT_SPECS: Mapping[str, _FormatSpec] = {
    "mlx-8bit": _FormatSpec(runtime="mlx", precision="int8", bits=8),
    "mlx-4bit": _FormatSpec(runtime="mlx", precision="int4", bits=4),
    "coreml-int8": _FormatSpec(runtime="coreml", precision="int8", bits=8),
    "coreml-int4": _FormatSpec(runtime="coreml", precision="int4", bits=4),
    "onnx-int8": _FormatSpec(runtime="onnx", precision="int8", bits=8),
}


def run_directid_tiny_quantization(
    *,
    candidate: Mapping[str, Any] | Any,
    checkpoint_path: str | Path,
    fp_parent_per_label_recall: Mapping[str, Any],
    eval_set_hash: str,
    param_count: int,
    evaluator: DirectIDArtifactEvaluator,
    output_dir: str | Path,
    exporter: DirectIDArtifactExporter | None = None,
    include_int4: bool = True,
    contract: DirectIDHeadContract = DIRECTID_TINY_HEAD_CONTRACT,
) -> DirectIDQuantizationArtifacts:
    """Export and evaluate DirectID Tiny runtime artifacts locally.

    Args:
        candidate: Path-free candidate mapping emitted by the DirectID Mode-A
            run. Objects exposing ``to_dict()`` are accepted as well.
        checkpoint_path: Local floating-point candidate file or directory.
        fp_parent_per_label_recall: Measured floating-point recall for every
            DirectID label.
        eval_set_hash: SHA-256 identity of the synthetic or user-supplied
            evaluation set used for both parent and quantized recall.
        param_count: Measured parameter count for G5 evidence.
        evaluator: Local callback that measures aggregate recall, latency, and
            RAM for each exported artifact.
        output_dir: Destination for runtime artifacts and evidence JSON.
        exporter: Optional export callback. The built-in MLX/CoreML/ONNX
            dispatcher is used by default.
        include_int4: Whether to evaluate conditional MLX and CoreML INT4
            artifacts. A failing INT4 artifact is quarantined independently.
        contract: DirectID head contract to enforce.

    Returns:
        Paths and parsed unsigned quantization evidence.

    Raises:
        DirectIDQuantizationError: If candidate, artifact, metric, or output
            evidence is incomplete, unsafe, or inconsistent.
    """

    active_contract = validate_directid_contract(contract)
    candidate_payload = _candidate_payload(candidate, contract=active_contract)
    source_path = _validated_artifact_path(
        checkpoint_path,
        field="candidate checkpoint",
    )
    if hash_directid_artifact(source_path) != candidate_payload["artifact_hash"]:
        raise DirectIDQuantizationError(
            "candidate checkpoint hash does not match candidate evidence"
        )

    destination = Path(output_dir).resolve()
    _validate_separate_trees(source_path, destination)
    parent_recall = _validated_recall_map(
        fp_parent_per_label_recall,
        field="fp_parent_per_label_recall",
        contract=active_contract,
    )
    expected_eval_set_hash = _require_digest(eval_set_hash, "eval_set_hash")
    if isinstance(param_count, bool) or not isinstance(param_count, int):
        raise DirectIDQuantizationError("param_count must be a positive integer")
    if param_count <= 0:
        raise DirectIDQuantizationError("param_count must be a positive integer")
    if not callable(evaluator):
        raise DirectIDQuantizationError("evaluator must be callable")

    active_exporter = exporter or export_directid_runtime_artifact
    if not callable(active_exporter):
        raise DirectIDQuantizationError("exporter must be callable")

    formats = list(DIRECTID_INT8_EXPORT_FORMATS)
    if include_int4:
        formats.extend(DIRECTID_INT4_EXPORT_FORMATS)

    artifact_paths: dict[str, Path] = {}
    records: list[dict[str, Any]] = []
    for format_name in formats:
        spec = _FORMAT_SPECS[format_name]
        request = DirectIDExportRequest(
            checkpoint_path=source_path,
            output_dir=destination / "artifacts" / format_name,
            format=format_name,
            runtime=spec.runtime,
            precision=spec.precision,
            bits=spec.bits,
        )
        artifact_path = _validated_export_result(
            active_exporter(request),
            request=request,
        )
        measurement = evaluator(
            DirectIDEvaluationRequest(
                artifact_path=artifact_path,
                format=format_name,
                runtime=spec.runtime,
                precision=spec.precision,
                bits=spec.bits,
                fp_parent_per_label_recall=parent_recall,
                eval_set_hash=expected_eval_set_hash,
            )
        )
        validated_measurement = _validated_measurement(
            measurement,
            format_name=format_name,
            contract=active_contract,
            expected_eval_set_hash=expected_eval_set_hash,
        )
        record = _artifact_evidence(
            artifact_path=artifact_path,
            destination=destination,
            format_name=format_name,
            spec=spec,
            measurement=validated_measurement,
            parent_recall=parent_recall,
            param_count=param_count,
            contract=active_contract,
        )
        artifact_paths[format_name] = artifact_path
        records.append(record)

    if hash_directid_artifact(source_path) != candidate_payload["artifact_hash"]:
        raise DirectIDQuantizationError(
            "candidate checkpoint changed during runtime export"
        )

    int8_records = [record for record in records if record["bits"] == 8]
    int4_records = [record for record in records if record["bits"] == 4]
    int8_g4_passed = all(bool(record["g4"]["passed"]) for record in int8_records)
    int8_tiny_fit_passed = all(
        bool(record["tiny_tier_fit"]["passed"]) for record in int8_records
    )
    ready_for_certification = int8_g4_passed and int8_tiny_fit_passed

    evidence: dict[str, Any] = {
        "schema_version": DIRECTID_QUANTIZATION_SCHEMA_VERSION,
        "contract_ref": active_contract.contract_ref,
        "family": active_contract.family,
        "tier": active_contract.tier,
        "candidate": {
            "artifact_hash": candidate_payload["artifact_hash"],
            "checkpoint_ref": candidate_payload["checkpoint_ref"],
            "reproducibility_hash": candidate_payload["reproducibility_hash"],
            "run_id": candidate_payload["run_id"],
        },
        "param_count": param_count,
        "fp_parent_format": DIRECTID_CANDIDATE_FORMAT,
        "fp_parent_per_label_recall": dict(parent_recall),
        "eval_set_hash": expected_eval_set_hash,
        "default_int8_formats": list(DIRECTID_INT8_EXPORT_FORMATS),
        "optional_int4_formats": list(DIRECTID_INT4_EXPORT_FORMATS),
        "int4_evaluated": include_int4,
        "artifacts": records,
        "accepted_formats": [
            str(record["format"])
            for record in records
            if record["disposition"] == "accepted"
        ],
        "quarantined_formats": [
            str(record["format"])
            for record in records
            if record["disposition"] == "quarantined"
        ],
        "int8_g4_passed": int8_g4_passed,
        "int8_tiny_fit_passed": int8_tiny_fit_passed,
        "int8_ready_for_certification": ready_for_certification,
        "int4_status": _int4_status(int4_records, evaluated=include_int4),
        "final_certification_performed": False,
        "publishing_performed": False,
        "raw_phi_persisted": False,
        "restricted_dataset_payloads_persisted": False,
    }
    evidence_hash = compute_canonical_payload_hash(evidence)
    manifest: dict[str, Any] = {
        "schema_version": DIRECTID_QUANTIZATION_MANIFEST_SCHEMA_VERSION,
        "contract_ref": active_contract.contract_ref,
        "family": active_contract.family,
        "tier": active_contract.tier,
        "candidate_artifact_hash": candidate_payload["artifact_hash"],
        "evidence_file": DIRECTID_QUANTIZATION_EVIDENCE_FILENAME,
        "evidence_hash": evidence_hash,
        "artifacts": [
            {
                "artifact_hash": record["artifact_hash"],
                "artifact_size_bytes": record["artifact_size_bytes"],
                "disposition": record["disposition"],
                "format": record["format"],
                "path": record["path"],
                "quant_recall_delta": record["quant_recall_delta"],
            }
            for record in records
        ],
        "int8_ready_for_certification": ready_for_certification,
        "final_certification_performed": False,
        "publishing_performed": False,
    }

    evidence_path = _write_json(
        destination / DIRECTID_QUANTIZATION_EVIDENCE_FILENAME,
        evidence,
    )
    manifest_path = _write_json(
        destination / DIRECTID_QUANTIZATION_MANIFEST_FILENAME,
        manifest,
    )
    return DirectIDQuantizationArtifacts(
        evidence=evidence,
        manifest=manifest,
        artifact_paths=artifact_paths,
        evidence_path=evidence_path,
        manifest_path=manifest_path,
    )


def export_directid_runtime_artifact(request: DirectIDExportRequest) -> Path:
    """Export one DirectID candidate with an existing local runtime converter.

    Optional backend dependencies are imported only for the requested format.
    The checkpoint path is passed directly to each converter, so no network
    lookup is required.
    """

    if request.format not in _FORMAT_SPECS:
        raise DirectIDQuantizationError("unsupported DirectID export format")
    expected = _FORMAT_SPECS[request.format]
    if (
        request.runtime != expected.runtime
        or request.precision != expected.precision
        or request.bits != expected.bits
    ):
        raise DirectIDQuantizationError("DirectID export request is inconsistent")
    if not request.checkpoint_path.is_dir():
        raise DirectIDQuantizationError(
            "built-in runtime export requires a checkpoint directory"
        )

    request.output_dir.mkdir(parents=True, exist_ok=True)
    model_id = str(request.checkpoint_path)
    if request.runtime == "mlx":
        try:
            import mlx.core  # noqa: F401
        except ImportError as exc:
            raise DirectIDQuantizationError(
                "MLX dependencies are required for the requested export"
            ) from exc
        from openmed.mlx.convert import convert as convert_mlx

        return Path(
            convert_mlx(
                model_id,
                request.output_dir,
                quantize_bits=request.bits,
            )
        )

    if request.runtime == "coreml":
        from openmed.coreml.convert import convert as convert_coreml

        base_path = request.output_dir / "model.mlpackage"
        quantized_path = request.output_dir / f"model_{request.precision}.mlpackage"
        convert_coreml(
            model_id,
            base_path,
            quantize=request.precision,
            quantized_output_path=quantized_path,
        )
        return quantized_path

    if request.runtime == "onnx":
        from openmed.onnx.convert import convert as convert_onnx

        result = convert_onnx(
            model_id,
            request.output_dir,
            include_webgpu=False,
            include_transformersjs=False,
            include_int8=True,
            profile="android",
        )
        for artifact in result.artifacts:
            if artifact.format == request.format:
                return artifact.path
        raise DirectIDQuantizationError(
            "ONNX converter did not emit the requested INT8 artifact"
        )

    raise DirectIDQuantizationError("unsupported DirectID export runtime")


def hash_directid_artifact(path: str | Path) -> str:
    """Return a stable hash for one artifact file or directory tree."""

    artifact_path = _validated_artifact_path(path, field="artifact")
    root = artifact_path if artifact_path.is_dir() else artifact_path.parent
    files = _artifact_files(artifact_path)
    entries = [
        {
            "path": file_path.relative_to(root).as_posix(),
            "sha256": _file_hash(file_path),
        }
        for file_path in files
    ]
    return compute_canonical_payload_hash(entries)


def _candidate_payload(
    candidate: Mapping[str, Any] | Any,
    *,
    contract: DirectIDHeadContract,
) -> dict[str, str]:
    if isinstance(candidate, Mapping):
        payload = candidate
    else:
        to_dict = getattr(candidate, "to_dict", None)
        if not callable(to_dict):
            raise DirectIDQuantizationError("candidate must be a mapping")
        payload = to_dict()
    if not isinstance(payload, Mapping):
        raise DirectIDQuantizationError("candidate must serialize to a mapping")
    if payload.get("schema_version") != DIRECTID_CANDIDATE_SCHEMA_VERSION:
        raise DirectIDQuantizationError("unsupported DirectID candidate schema")
    if payload.get("family") != contract.family or payload.get("tier") != contract.tier:
        raise DirectIDQuantizationError("candidate family or tier mismatch")
    if payload.get("format") != DIRECTID_CANDIDATE_FORMAT:
        raise DirectIDQuantizationError("candidate must be the FP32 parent format")
    if payload.get("ready_for_quantization") is not True:
        raise DirectIDQuantizationError("candidate is not ready for quantization")
    if payload.get("certified") is not False or payload.get("published") is not False:
        raise DirectIDQuantizationError(
            "candidate must be uncertified and unpublished before quantization"
        )
    artifact_hash = _require_digest(payload.get("artifact_hash"), "artifact_hash")
    reproducibility_hash = _require_digest(
        payload.get("reproducibility_hash"),
        "reproducibility_hash",
    )
    return {
        "artifact_hash": artifact_hash,
        "checkpoint_ref": _safe_identifier(
            payload.get("checkpoint_ref"),
            "checkpoint_ref",
        ),
        "reproducibility_hash": reproducibility_hash,
        "run_id": _safe_identifier(payload.get("run_id"), "run_id"),
    }


def _validated_recall_map(
    values: Mapping[str, Any],
    *,
    field: str,
    contract: DirectIDHeadContract,
) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise DirectIDQuantizationError(f"{field} must be a mapping")
    if set(values) != set(contract.labels):
        raise DirectIDQuantizationError(
            f"{field} must cover exactly the DirectID contract labels"
        )
    result: dict[str, float] = {}
    for label in contract.labels:
        value = values[label]
        if isinstance(value, bool):
            raise DirectIDQuantizationError(f"{field}.{label} must be a recall rate")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise DirectIDQuantizationError(
                f"{field}.{label} must be a recall rate"
            ) from exc
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            raise DirectIDQuantizationError(
                f"{field}.{label} must be between zero and one"
            )
        result[label] = parsed
    return result


def _validated_measurement(
    measurement: DirectIDArtifactMeasurement,
    *,
    format_name: str,
    contract: DirectIDHeadContract,
    expected_eval_set_hash: str,
) -> DirectIDArtifactMeasurement:
    if not isinstance(measurement, DirectIDArtifactMeasurement):
        raise DirectIDQuantizationError(
            "evaluator must return DirectIDArtifactMeasurement"
        )
    recall = _validated_recall_map(
        measurement.per_label_recall,
        field=f"{format_name}.per_label_recall",
        contract=contract,
    )
    eval_set_hash = _require_digest(
        measurement.eval_set_hash,
        f"{format_name}.eval_set_hash",
    )
    if eval_set_hash != expected_eval_set_hash:
        raise DirectIDQuantizationError(
            f"{format_name}.eval_set_hash does not match the FP parent evaluation"
        )
    performance_fixture_hash = _require_digest(
        measurement.performance_fixture_hash,
        f"{format_name}.performance_fixture_hash",
    )
    device = _safe_identifier(measurement.device, f"{format_name}.device")
    if (
        isinstance(measurement.sample_count, bool)
        or not isinstance(measurement.sample_count, int)
        or measurement.sample_count <= 0
    ):
        raise DirectIDQuantizationError(
            f"{format_name}.sample_count must be a positive integer"
        )
    p50_ms = _positive_float(measurement.p50_ms, f"{format_name}.p50_ms")
    p95_ms = _positive_float(measurement.p95_ms, f"{format_name}.p95_ms")
    ram_mb = _positive_float(measurement.ram_mb, f"{format_name}.ram_mb")
    if p95_ms < p50_ms:
        raise DirectIDQuantizationError(
            f"{format_name}.p95_ms must be greater than or equal to p50_ms"
        )
    return DirectIDArtifactMeasurement(
        per_label_recall=recall,
        eval_set_hash=eval_set_hash,
        performance_fixture_hash=performance_fixture_hash,
        device=device,
        sample_count=measurement.sample_count,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        ram_mb=ram_mb,
    )


def _artifact_evidence(
    *,
    artifact_path: Path,
    destination: Path,
    format_name: str,
    spec: _FormatSpec,
    measurement: DirectIDArtifactMeasurement,
    parent_recall: Mapping[str, float],
    param_count: int,
    contract: DirectIDHeadContract,
) -> dict[str, Any]:
    delta = evaluate_quant_recall_delta(
        format_name=format_name,
        candidate_recall=measurement.per_label_recall,
        parent_recall=parent_recall,
        labels=contract.critical_labels,
    )
    budget = TIERS["Tiny"]
    observed = {
        "p50_ms": measurement.p50_ms,
        "p95_ms": measurement.p95_ms,
        "ram_mb": measurement.ram_mb,
    }
    limits = {
        "p50_ms": float(cast(int, budget["p50_ms_max"])),
        "p95_ms": float(cast(int, budget["p95_ms_max"])),
        "ram_mb": float(cast(int, budget["ram_mb_max"])),
    }
    violations = {
        key: {"limit": limits[key], "observed": observed[key]}
        for key in observed
        if observed[key] > limits[key]
    }
    tiny_fit_passed = not violations
    quarantine_reasons: list[str] = []
    if not delta.passed:
        quarantine_reasons.append("G4_RECALL_DELTA")
    if not tiny_fit_passed:
        quarantine_reasons.append("G5_TINY_TIER_FIT")

    per_label = {
        label: {
            "critical": label in set(contract.critical_labels),
            "fp_parent_recall": parent_recall[label],
            "quantized_recall": measurement.per_label_recall[label],
            "recall_delta": delta.per_label_delta[label],
        }
        for label in contract.labels
    }
    return {
        "format": format_name,
        "runtime": spec.runtime,
        "precision": spec.precision,
        "bits": spec.bits,
        "path": artifact_path.relative_to(destination).as_posix(),
        "artifact_hash": hash_directid_artifact(artifact_path),
        "artifact_size_bytes": _artifact_size_bytes(artifact_path),
        "eval_set_hash": measurement.eval_set_hash,
        "fp_parent_per_label_recall": dict(parent_recall),
        "per_label_recall": dict(measurement.per_label_recall),
        "param_count": param_count,
        "performance_fixture_hash": measurement.performance_fixture_hash,
        "device": measurement.device,
        "sample_count": measurement.sample_count,
        "p50_ms": measurement.p50_ms,
        "p95_ms": measurement.p95_ms,
        "latency": {
            "p50_ms": measurement.p50_ms,
            "p95_ms": measurement.p95_ms,
        },
        "ram_mb": measurement.ram_mb,
        "per_label_recall_delta": per_label,
        "critical_label_deltas": {
            label: delta.per_label_delta[label] for label in contract.critical_labels
        },
        "quant_recall_delta": delta.max_delta,
        "g4": {
            **delta.to_dict(),
            "blocking_format": delta.blocking_format,
            "critical_offending_labels": sorted(delta.offending_labels),
        },
        "tiny_tier_fit": {
            "passed": tiny_fit_passed,
            "budget": limits,
            "observed": observed,
            "violations": violations,
        },
        "disposition": "accepted"
        if delta.passed and tiny_fit_passed
        else "quarantined",
        "quarantine_reasons": quarantine_reasons,
    }


def _validated_export_result(
    value: str | Path,
    *,
    request: DirectIDExportRequest,
) -> Path:
    path = _validated_artifact_path(value, field=f"{request.format} artifact")
    output_root = request.output_dir.resolve()
    if path != output_root and output_root not in path.parents:
        raise DirectIDQuantizationError(
            "exporter returned an artifact outside its assigned output directory"
        )
    return path


def _validated_artifact_path(value: str | Path, *, field: str) -> Path:
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise DirectIDQuantizationError(f"{field} must not be a symlink")
    path = unresolved.resolve()
    if not path.exists():
        raise DirectIDQuantizationError(f"{field} does not exist")
    descendants = list(path.rglob("*")) if path.is_dir() else []
    if any(item.is_symlink() for item in descendants):
        raise DirectIDQuantizationError(f"{field} must not contain symlinks")
    if not _artifact_files(path):
        raise DirectIDQuantizationError(f"{field} contains no files")
    return path


def _validate_separate_trees(checkpoint_path: Path, output_dir: Path) -> None:
    if (
        checkpoint_path == output_dir
        or checkpoint_path in output_dir.parents
        or output_dir in checkpoint_path.parents
    ):
        raise DirectIDQuantizationError(
            "output_dir must be separate from the candidate checkpoint tree"
        )


def _artifact_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(item for item in path.rglob("*") if item.is_file())


def _artifact_size_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in _artifact_files(path))


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _positive_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise DirectIDQuantizationError(f"{field} must be a positive number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DirectIDQuantizationError(f"{field} must be a positive number") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise DirectIDQuantizationError(f"{field} must be a positive number")
    return result


def _safe_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DirectIDQuantizationError(f"{field} must be a non-empty string")
    result = value.strip()
    if len(result) > 256 or "\n" in result or "\r" in result:
        raise DirectIDQuantizationError(f"{field} has an invalid format")
    if any(pattern.search(result) for pattern in _PHI_SHAPED_PATTERNS):
        raise DirectIDQuantizationError(f"{field} contains a PHI-shaped value")
    return result


def _require_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DirectIDQuantizationError(f"{field} must be a sha256 digest")
    return value


def _int4_status(records: Sequence[Mapping[str, Any]], *, evaluated: bool) -> str:
    if not evaluated:
        return "not_requested"
    accepted = sum(record.get("disposition") == "accepted" for record in records)
    if accepted == len(records):
        return "accepted"
    if accepted == 0:
        return "quarantined"
    return "partially_quarantined"


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    serialized = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != serialized:
            raise DirectIDQuantizationError(
                f"refusing to overwrite non-matching evidence file {path.name}"
            )
        return path
    path.write_text(serialized, encoding="utf-8")
    return path


__all__ = [
    "DIRECTID_CANDIDATE_FORMAT",
    "DIRECTID_CANDIDATE_SCHEMA_VERSION",
    "DIRECTID_INT4_EXPORT_FORMATS",
    "DIRECTID_INT8_EXPORT_FORMATS",
    "DIRECTID_QUANTIZATION_EVIDENCE_FILENAME",
    "DIRECTID_QUANTIZATION_MANIFEST_FILENAME",
    "DIRECTID_QUANTIZATION_MANIFEST_SCHEMA_VERSION",
    "DIRECTID_QUANTIZATION_SCHEMA_VERSION",
    "DirectIDArtifactEvaluator",
    "DirectIDArtifactExporter",
    "DirectIDArtifactMeasurement",
    "DirectIDEvaluationRequest",
    "DirectIDExportRequest",
    "DirectIDQuantizationArtifacts",
    "DirectIDQuantizationError",
    "export_directid_runtime_artifact",
    "hash_directid_artifact",
    "run_directid_tiny_quantization",
]
