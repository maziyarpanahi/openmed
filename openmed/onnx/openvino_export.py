"""OpenVINO IR export, certification, and INT8 quantization helpers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

from openmed.core.decoding import build_label_info, labels_to_token_spans
from openmed.eval.quant_delta import (
    QuantRecallDeltaResult,
    evaluate_quant_recall_delta,
)
from openmed.eval.report import BenchmarkReport
from openmed.onnx.openvino_session import (
    OPENVINO_DEVICE_FALLBACK_ORDER,
    OpenVinoTokenClassificationSession,
)

OPENVINO_PROFILE_NAME = "openvino"
OPENVINO_FORMAT = "openvino-ir"
OPENVINO_INT8_FORMAT = "openvino-ir-int8"
OPENVINO_IR_DIRNAME = "openvino"
OPENVINO_INT8_DIRNAME = "openvino_int8"
OPENVINO_MODEL_XML = "model.xml"
OPENVINO_BENCHMARK_REPORT = "openvino-benchmark.report.json"
SYNTHETIC_NOTE = "Jane Doe visited Boston Clinic on 2024-01-15."
DEFAULT_LOGIT_TOLERANCE = 1e-3


class OpenVinoVerificationError(ValueError):
    """Raised when an OpenVINO export fails the ONNX reference check."""


class OpenVinoQuantizationRejected(ValueError):
    """Raised when INT8 quantization lacks passing recall-delta evidence."""

    def __init__(self, message: str, gate: QuantRecallDeltaResult) -> None:
        super().__init__(message)
        self.gate = gate


@dataclass(frozen=True)
class OpenVinoExportVerification:
    """Synthetic ONNX/OpenVINO parity evidence."""

    sample_text: str
    tolerance: float
    max_abs_logit_delta: float
    reference_token_spans: tuple[dict[str, Any], ...]
    openvino_token_spans: tuple[dict[str, Any], ...]
    passed: bool = True

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable verification metadata."""

        return {
            "sample_text": self.sample_text,
            "tolerance": self.tolerance,
            "max_abs_logit_delta": self.max_abs_logit_delta,
            "reference_token_spans": [
                dict(span) for span in self.reference_token_spans
            ],
            "openvino_token_spans": [dict(span) for span in self.openvino_token_spans],
            "passed": self.passed,
        }


@dataclass(frozen=True)
class OpenVinoExportResult:
    """OpenVINO IR artifact paths and validation metadata."""

    output_dir: Path
    model_xml_path: Path
    model_bin_path: Path
    verification: OpenVinoExportVerification | None = None

    def to_metadata(self, root: str | Path | None = None) -> dict[str, Any]:
        """Return manifest metadata for the exported IR artifact."""

        base = Path(root) if root is not None else self.output_dir
        metadata: dict[str, Any] = {
            "profile": OPENVINO_PROFILE_NAME,
            "format": OPENVINO_FORMAT,
            "model_xml_path": _relative_or_absolute(self.model_xml_path, base),
            "model_bin_path": _relative_or_absolute(self.model_bin_path, base),
        }
        if self.verification is not None:
            metadata["synthetic_verification"] = self.verification.to_metadata()
        return metadata


@dataclass(frozen=True)
class OpenVinoQuantizationResult:
    """INT8 OpenVINO artifact plus fail-closed recall-delta evidence."""

    output_dir: Path
    model_xml_path: Path
    model_bin_path: Path
    family: str
    calibration_sample_count: int
    recall_delta_gate: QuantRecallDeltaResult

    def to_metadata(self, root: str | Path | None = None) -> dict[str, Any]:
        """Return manifest metadata for the INT8 artifact."""

        base = Path(root) if root is not None else self.output_dir
        return {
            "profile": OPENVINO_PROFILE_NAME,
            "format": OPENVINO_INT8_FORMAT,
            "family": self.family,
            "gate": "G4",
            "calibration_sample_count": self.calibration_sample_count,
            "model_xml_path": _relative_or_absolute(self.model_xml_path, base),
            "model_bin_path": _relative_or_absolute(self.model_bin_path, base),
            "recall_delta_gate": self.recall_delta_gate.to_dict(),
        }


@dataclass(frozen=True)
class OpenVinoBenchmarkRecord:
    """One OpenVINO device throughput and latency record."""

    device: str
    precision: str
    latency_ms: float
    throughput_items_per_second: float
    sample_count: int
    batch_size: int = 1
    sequence_length: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, Any]:
        """Return the BenchmarkReport metrics block for this device."""

        payload: dict[str, Any] = {
            "latency": {
                "p50_ms": self.latency_ms,
                "p95_ms": self.latency_ms,
                "count": self.sample_count,
            },
            "throughput": {
                "items_per_second": self.throughput_items_per_second,
            },
            "batch_size": self.batch_size,
            "precision": self.precision,
        }
        if self.sequence_length is not None:
            payload["sequence_length"] = self.sequence_length
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def export_openvino_ir(
    onnx_path: str | Path,
    output_dir: str | Path,
    *,
    sample_inputs: Mapping[str, Any] | None = None,
    reference_logits: Any | None = None,
    id2label: Mapping[str | int, str] | None = None,
    sample_text: str = SYNTHETIC_NOTE,
    tolerance: float = DEFAULT_LOGIT_TOLERANCE,
    device: str = "CPU",
    fallback_order: Sequence[str] = OPENVINO_DEVICE_FALLBACK_ORDER,
) -> OpenVinoExportResult:
    """Convert an ONNX token-classification graph to OpenVINO IR."""

    convert_model, save_model = _openvino_conversion_api()
    onnx_path = Path(onnx_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_xml = output_path / OPENVINO_MODEL_XML

    model = convert_model(str(onnx_path))
    _save_openvino_model(save_model, model, model_xml, compress_to_fp16=False)
    model_bin = model_xml.with_suffix(".bin")

    verification = None
    if (
        sample_inputs is not None
        or reference_logits is not None
        or id2label is not None
    ):
        if sample_inputs is None or reference_logits is None or id2label is None:
            raise ValueError(
                "sample_inputs, reference_logits, and id2label are all required "
                "for OpenVINO synthetic verification"
            )
        session = OpenVinoTokenClassificationSession(
            model_xml,
            device=device,
            fallback_order=fallback_order,
        )
        openvino_logits = session.run(**dict(sample_inputs))
        verification = certify_openvino_reference(
            reference_logits=reference_logits,
            openvino_logits=openvino_logits,
            id2label=id2label,
            attention_mask=sample_inputs.get("attention_mask"),
            sample_text=sample_text,
            tolerance=tolerance,
        )

    return OpenVinoExportResult(
        output_dir=output_path,
        model_xml_path=model_xml,
        model_bin_path=model_bin,
        verification=verification,
    )


def certify_openvino_reference(
    *,
    reference_logits: Any,
    openvino_logits: Any,
    id2label: Mapping[str | int, str],
    attention_mask: Any | None = None,
    sample_text: str = SYNTHETIC_NOTE,
    tolerance: float = DEFAULT_LOGIT_TOLERANCE,
) -> OpenVinoExportVerification:
    """Check OpenVINO logits and decoded token spans against the ONNX reference."""

    import numpy as np

    reference = np.asarray(reference_logits)
    candidate = np.asarray(openvino_logits)
    if reference.shape != candidate.shape:
        raise OpenVinoVerificationError(
            "OpenVINO logits shape does not match ONNX reference: "
            f"{candidate.shape} != {reference.shape}"
        )

    max_abs_delta = (
        float(np.max(np.abs(reference - candidate))) if reference.size else 0.0
    )
    reference_spans = token_spans_from_logits(
        reference,
        id2label,
        attention_mask=attention_mask,
    )
    candidate_spans = token_spans_from_logits(
        candidate,
        id2label,
        attention_mask=attention_mask,
    )

    if max_abs_delta > tolerance:
        raise OpenVinoVerificationError(
            "OpenVINO logits exceeded tolerance "
            f"{tolerance}: max abs delta {max_abs_delta}"
        )
    if candidate_spans != reference_spans:
        raise OpenVinoVerificationError(
            "OpenVINO decoded token spans do not match ONNX reference"
        )

    return OpenVinoExportVerification(
        sample_text=sample_text,
        tolerance=tolerance,
        max_abs_logit_delta=max_abs_delta,
        reference_token_spans=reference_spans,
        openvino_token_spans=candidate_spans,
    )


def token_spans_from_logits(
    logits: Any,
    id2label: Mapping[str | int, str],
    *,
    attention_mask: Any | None = None,
) -> tuple[dict[str, Any], ...]:
    """Decode argmax token labels into token-span records."""

    import numpy as np

    label_map = {int(key): str(value) for key, value in id2label.items()}
    label_info = build_label_info(label_map)
    values = np.asarray(logits)
    if values.ndim == 3:
        token_logits = values[0]
    elif values.ndim == 2:
        token_logits = values
    else:
        raise ValueError("token-classification logits must have rank 2 or 3")

    predicted = np.argmax(token_logits, axis=-1).tolist()
    mask = _first_batch_mask(attention_mask)
    labels_by_index = {
        index: int(label_id)
        for index, label_id in enumerate(predicted)
        if mask is None or (index < len(mask) and bool(mask[index]))
    }
    spans = labels_to_token_spans(labels_by_index, label_info)
    return tuple(
        {
            "label": label_info.span_class_names[span_label],
            "start_token": start,
            "end_token": end,
        }
        for span_label, start, end in spans
    )


def build_synthetic_token_inputs(
    tokenizer: Any,
    *,
    text: str = SYNTHETIC_NOTE,
    max_seq_length: int = 512,
) -> dict[str, Any]:
    """Tokenize the synthetic OpenVINO certification note."""

    encoded = tokenizer(
        [text],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )
    return {key: value for key, value in encoded.items()}


def run_onnx_reference_logits(
    onnx_path: str | Path,
    sample_inputs: Mapping[str, Any],
) -> Any:
    """Run ONNX Runtime and return the logits output for the sample inputs."""

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for OpenVINO export verification. "
            "Install with: pip install openmed[openvino]"
        ) from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    inputs = {
        name: value for name, value in sample_inputs.items() if name in input_names
    }
    output_names = [item.name for item in session.get_outputs()]
    preferred_outputs = ["logits"] if "logits" in output_names else [output_names[0]]
    return session.run(preferred_outputs, inputs)[0]


def quantize_openvino_int8(
    model_xml_path: str | Path,
    output_dir: str | Path,
    *,
    calibration_data: Iterable[Mapping[str, Any]],
    family: str,
    candidate_recall: Mapping[str, Any] | None = None,
    parent_recall: Mapping[str, Any] | None = None,
    precomputed_delta: Any = None,
    labels: Sequence[str] | None = None,
) -> OpenVinoQuantizationResult:
    """Create an INT8 OpenVINO IR artifact when G4 recall evidence passes."""

    rows = [dict(row) for row in calibration_data]
    gate = evaluate_quant_recall_delta(
        format_name=OPENVINO_INT8_FORMAT,
        candidate_recall=candidate_recall or {},
        parent_recall=parent_recall,
        precomputed_delta=precomputed_delta,
        labels=labels,
    )
    if not rows:
        raise OpenVinoQuantizationRejected(
            "OpenVINO INT8 export requires calibration samples", gate
        )
    if not gate.passed:
        raise OpenVinoQuantizationRejected(
            "OpenVINO INT8 export rejected by G4 recall-delta gate", gate
        )

    core, save_model = _openvino_runtime_api()
    nncf = _nncf_api()
    source_model = core.read_model(str(model_xml_path))
    dataset = nncf.Dataset(rows)
    quantized_model = nncf.quantize(source_model, dataset)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_xml = output_path / OPENVINO_MODEL_XML
    _save_openvino_model(save_model, quantized_model, model_xml, compress_to_fp16=False)

    return OpenVinoQuantizationResult(
        output_dir=output_path,
        model_xml_path=model_xml,
        model_bin_path=model_xml.with_suffix(".bin"),
        family=family,
        calibration_sample_count=len(rows),
        recall_delta_gate=gate,
    )


def measure_openvino_latency(
    session: OpenVinoTokenClassificationSession,
    sample_inputs: Mapping[str, Any],
    *,
    iterations: int = 5,
) -> OpenVinoBenchmarkRecord:
    """Measure simple per-device latency and throughput for one session."""

    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    latencies = []
    for _ in range(iterations):
        started = perf_counter()
        session.run(**dict(sample_inputs))
        latencies.append((perf_counter() - started) * 1000.0)

    latency_ms = sorted(latencies)[len(latencies) // 2]
    return OpenVinoBenchmarkRecord(
        device=session.selected_device,
        precision="float32",
        latency_ms=latency_ms,
        throughput_items_per_second=1000.0 / latency_ms if latency_ms else 0.0,
        sample_count=iterations,
        batch_size=_batch_size(sample_inputs),
        sequence_length=_sequence_length(sample_inputs),
        metadata={"profile": OPENVINO_PROFILE_NAME},
    )


def build_openvino_benchmark_report(
    *,
    model_name: str,
    records: Sequence[OpenVinoBenchmarkRecord],
    suite: str = "openvino-runtime",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> BenchmarkReport:
    """Build a BenchmarkReport containing per-device latency and throughput."""

    if not records:
        raise ValueError("at least one OpenVINO benchmark record is required")

    generated = generated_at or datetime.now(timezone.utc).isoformat()
    device_names = [record.device for record in records]
    return BenchmarkReport(
        suite=suite,
        model_name=model_name,
        device="openvino:" + ",".join(device_names),
        fixture_count=sum(record.sample_count for record in records),
        generated_at=generated,
        metadata={"profile": OPENVINO_PROFILE_NAME, **dict(metadata or {})},
        metrics={
            "devices": {record.device: record.to_metrics() for record in records},
        },
    )


def write_openvino_benchmark_report(
    output_path: str | Path,
    *,
    model_name: str,
    records: Sequence[OpenVinoBenchmarkRecord],
    suite: str = "openvino-runtime",
    generated_at: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write an OpenVINO BenchmarkReport JSON artifact."""

    report = build_openvino_benchmark_report(
        model_name=model_name,
        records=records,
        suite=suite,
        generated_at=generated_at,
        metadata=metadata,
    )
    return report.write_json(output_path)


def _openvino_conversion_api() -> tuple[Any, Any]:
    try:
        from openvino import convert_model, save_model
    except ImportError:
        try:
            from openvino.runtime import serialize as save_model
            from openvino.tools.mo import convert_model
        except ImportError as exc:
            raise ImportError(
                "OpenVINO is required for IR export. "
                "Install with: pip install openmed[openvino]"
            ) from exc
    return convert_model, save_model


def _openvino_runtime_api() -> tuple[Any, Any]:
    try:
        from openvino import Core, save_model
    except ImportError:
        try:
            from openvino.runtime import Core
            from openvino.runtime import serialize as save_model
        except ImportError as exc:
            raise ImportError(
                "OpenVINO runtime is required for INT8 export. "
                "Install with: pip install openmed[openvino]"
            ) from exc
    return Core(), save_model


def _nncf_api() -> Any:
    try:
        import nncf
    except ImportError as exc:
        raise ImportError(
            "NNCF is required for OpenVINO INT8 export. "
            "Install with: pip install openmed[openvino]"
        ) from exc
    return nncf


def _save_openvino_model(
    save_model: Any,
    model: Any,
    path: Path,
    *,
    compress_to_fp16: bool,
) -> None:
    try:
        save_model(model, str(path), compress_to_fp16=compress_to_fp16)
    except TypeError:
        save_model(model, str(path))


def _first_batch_mask(attention_mask: Any | None) -> list[Any] | None:
    if attention_mask is None:
        return None

    try:
        import numpy as np

        mask = np.asarray(attention_mask)
        if mask.ndim == 2:
            return mask[0].tolist()
        return mask.tolist()
    except Exception:
        mask = (
            attention_mask[0]
            if attention_mask and isinstance(attention_mask, list)
            else attention_mask
        )
        return list(mask)


def _batch_size(inputs: Mapping[str, Any]) -> int:
    value = inputs.get("input_ids")
    try:
        return int(value.shape[0])
    except AttributeError:
        try:
            return len(value)
        except TypeError:
            return 1


def _sequence_length(inputs: Mapping[str, Any]) -> int | None:
    value = inputs.get("input_ids")
    try:
        return int(value.shape[1])
    except (AttributeError, IndexError):
        try:
            return len(value[0])
        except (TypeError, IndexError):
            return None


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an ONNX token-classification graph to OpenVINO IR",
    )
    parser.add_argument("--onnx", required=True, help="Source ONNX model path")
    parser.add_argument("--output", required=True, help="Output OpenVINO IR directory")
    args = parser.parse_args()
    result = export_openvino_ir(args.onnx, args.output)
    print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_LOGIT_TOLERANCE",
    "OPENVINO_BENCHMARK_REPORT",
    "OPENVINO_FORMAT",
    "OPENVINO_INT8_DIRNAME",
    "OPENVINO_INT8_FORMAT",
    "OPENVINO_IR_DIRNAME",
    "OPENVINO_MODEL_XML",
    "OPENVINO_PROFILE_NAME",
    "SYNTHETIC_NOTE",
    "OpenVinoBenchmarkRecord",
    "OpenVinoExportResult",
    "OpenVinoExportVerification",
    "OpenVinoQuantizationRejected",
    "OpenVinoQuantizationResult",
    "OpenVinoVerificationError",
    "build_openvino_benchmark_report",
    "build_synthetic_token_inputs",
    "certify_openvino_reference",
    "export_openvino_ir",
    "measure_openvino_latency",
    "quantize_openvino_int8",
    "run_onnx_reference_logits",
    "token_spans_from_logits",
    "write_openvino_benchmark_report",
]
