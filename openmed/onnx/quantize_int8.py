"""Dynamic INT8 quantization and recall certification for Android ONNX."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.processing.tokenizer_cache import get_tokenizer_with_loader

INT8_ONNX_FILENAME = "model_int8.onnx"
ONNX_INT8_FORMAT = "onnx-int8"
RECALL_DELTA_REPORT_FILENAME = "recall_delta.json"
RECALL_DELTA_REPORT_VERSION = 1
DEFAULT_DYNAMIC_OP_TYPES = ("MatMul", "Gemm")

QuantEvalRunner = Callable[[Any, str, str], Iterable[Any]]


@dataclass(frozen=True)
class Int8QuantizationResult:
    """Paths and metadata produced by Android ONNX INT8 quantization."""

    model_path: Path
    metadata: Mapping[str, Any]


def quantize_dynamic_int8(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    *,
    op_types_to_quantize: Sequence[str] = DEFAULT_DYNAMIC_OP_TYPES,
) -> Path:
    """Create a dynamic QInt8 ONNX artifact from a fp32 Android ONNX model."""

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to create model_int8.onnx. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    onnx_path = Path(onnx_path)
    output_path = (
        Path(output_path)
        if output_path is not None
        else onnx_path.with_name(INT8_ONNX_FILENAME)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=list(op_types_to_quantize),
    )
    if not output_path.exists():
        raise RuntimeError(f"INT8 model was not written: {output_path}")
    _check_onnx_model(output_path)
    return output_path


def quantize_android_int8(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
    *,
    op_types_to_quantize: Sequence[str] = DEFAULT_DYNAMIC_OP_TYPES,
) -> Int8QuantizationResult:
    """Quantize an Android-profile ONNX graph and return artifact metadata."""

    model_path = quantize_dynamic_int8(
        onnx_path,
        output_path,
        op_types_to_quantize=op_types_to_quantize,
    )
    return Int8QuantizationResult(
        model_path=model_path,
        metadata=int8_artifact_metadata(),
    )


def int8_artifact_metadata(
    *,
    validation_metadata: Mapping[str, Any] | None = None,
    recall_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return manifest metadata for an ``onnx-int8`` artifact."""

    metadata: dict[str, Any] = {
        "format": ONNX_INT8_FORMAT,
        "quantization": _base_quantization_metadata(),
    }
    if validation_metadata:
        metadata.update(dict(validation_metadata))
    if recall_report:
        metadata.update(
            {
                "certification_limit": recall_report.get("limit"),
                "certified": bool(recall_report.get("certified")),
                "quant_recall_delta": recall_report.get("quant_recall_delta"),
                "recall_delta_path": recall_report.get("report_path"),
            }
        )
    return metadata


def write_int8_recall_delta_report(
    *,
    source_model_id: str,
    artifact_dir: str | Path,
    eval_suite_path: str | Path,
    fp_model_path: str | Path | None = None,
    int8_model_path: str | Path | None = None,
    output_path: str | Path | None = None,
    cache_dir: str | None = None,
    parent_runner: QuantEvalRunner | None = None,
    candidate_runner: QuantEvalRunner | None = None,
) -> dict[str, Any]:
    """Run FP-parent versus INT8 ONNX recall and write PHI-safe evidence JSON."""

    from openmed.eval.harness import load_fixtures, run_benchmark
    from openmed.eval.quant_delta import (
        evaluate_quant_recall_delta,
        limit_for_format,
    )

    artifact_dir = Path(artifact_dir)
    eval_suite_path = Path(eval_suite_path)
    fp_model_path = (
        Path(fp_model_path) if fp_model_path else artifact_dir / "model.onnx"
    )
    int8_model_path = (
        Path(int8_model_path) if int8_model_path else artifact_dir / INT8_ONNX_FILENAME
    )
    report_path = (
        Path(output_path)
        if output_path is not None
        else artifact_dir / RECALL_DELTA_REPORT_FILENAME
    )

    generated_at = _utc_now()
    fixtures = load_fixtures(eval_suite_path)
    parent_runner = parent_runner or _hf_token_classification_runner(
        source_model_id,
        cache_dir=cache_dir,
    )
    candidate_runner = candidate_runner or _onnx_token_classification_runner(
        artifact_dir=artifact_dir,
        model_path=int8_model_path,
    )

    suite_name = eval_suite_path.stem or "eval-suite"
    parent_report = run_benchmark(
        fixtures,
        suite=suite_name,
        model_name=source_model_id,
        device="fp-parent",
        runner=parent_runner,
        generated_at=generated_at,
        metadata={"format": "onnx-fp32", "source_model_id": source_model_id},
    )
    candidate_report = run_benchmark(
        fixtures,
        suite=suite_name,
        model_name=str(int8_model_path),
        device=ONNX_INT8_FORMAT,
        runner=candidate_runner,
        generated_at=generated_at,
        metadata={
            "format": ONNX_INT8_FORMAT,
            "source_model_id": source_model_id,
            "quantization": _base_quantization_metadata(),
        },
    )

    parent_recall = _per_label_recall(parent_report.metrics)
    candidate_recall = _per_label_recall(candidate_report.metrics)
    delta = evaluate_quant_recall_delta(
        format_name=ONNX_INT8_FORMAT,
        candidate_recall=candidate_recall,
        parent_recall=parent_recall,
    )
    delta_payload = delta.to_dict()
    delta_payload["blocking_format"] = delta.blocking_format
    report_relpath = _artifact_relative_path(report_path, artifact_dir)
    span_counts = _gold_span_counts_by_label(fixtures)
    char_counts = _gold_char_counts_by_label(fixtures)

    payload: dict[str, Any] = {
        "schema_version": RECALL_DELTA_REPORT_VERSION,
        "generated_at": generated_at,
        "source_model_id": source_model_id,
        "format": ONNX_INT8_FORMAT,
        "artifact": {
            "fp_parent_model_path": _artifact_relative_path(
                fp_model_path, artifact_dir
            ),
            "int8_model_path": _artifact_relative_path(int8_model_path, artifact_dir),
        },
        "quantization": _base_quantization_metadata(),
        "fixture_count": len(fixtures),
        "gold_span_count_by_label": span_counts,
        "gold_char_count_by_label": char_counts,
        "metric": "character_recall",
        "limit": limit_for_format(ONNX_INT8_FORMAT),
        "certified": bool(delta.passed),
        "quant_recall_delta": delta.max_delta,
        "per_label": _per_label_recall_comparison(
            parent_recall,
            candidate_recall,
            delta.per_label_delta,
            span_counts=span_counts,
            char_counts=char_counts,
        ),
        "fp_parent_per_label_recall": parent_recall,
        "candidate_per_label_recall": candidate_recall,
        "delta": delta_payload,
        "report_path": report_relpath,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    apply_int8_recall_certification(
        artifact_dir,
        payload,
        report_relpath=report_relpath,
    )
    return payload


def apply_int8_recall_certification(
    artifact_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    report_relpath: str,
) -> None:
    """Mirror INT8 recall certification into config and manifest metadata."""

    artifact_dir = Path(artifact_dir)
    quantization = {
        **_base_quantization_metadata(),
        "certification_limit": payload["limit"],
        "certified": payload["certified"],
        "format": payload["format"],
        "quant_recall_delta": payload["quant_recall_delta"],
        "recall_delta_path": report_relpath,
    }

    config_path = artifact_dir / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config["_onnx_quantization"] = quantization
        config["quant_recall_delta"] = payload["quant_recall_delta"]
        config["certified"] = payload["certified"]
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")

    _update_manifest_int8_certification(
        artifact_dir,
        quantization=quantization,
        payload=payload,
        report_relpath=report_relpath,
    )


def _hf_token_classification_runner(
    model_id: str,
    *,
    cache_dir: str | None = None,
) -> QuantEvalRunner:
    pipeline_instance: Any | None = None

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        nonlocal pipeline_instance
        if pipeline_instance is None:
            try:
                from transformers import (
                    AutoModelForTokenClassification,
                    AutoTokenizer,
                    pipeline,
                )
            except ImportError as exc:
                raise ImportError(
                    "transformers is required to certify ONNX INT8 quantization. "
                    "Install with: pip install transformers"
                ) from exc

            tokenizer = get_tokenizer_with_loader(
                model_id,
                AutoTokenizer.from_pretrained,
                cache_dir=cache_dir,
            )
            model = AutoModelForTokenClassification.from_pretrained(
                model_id,
                cache_dir=cache_dir,
            )
            pipeline_instance = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )
        return pipeline_instance(fixture.text)

    return run_fixture


def _onnx_token_classification_runner(
    *,
    artifact_dir: str | Path,
    model_path: str | Path,
) -> QuantEvalRunner:
    artifact_dir = Path(artifact_dir)
    model_path = Path(model_path)
    state: dict[str, Any] = {}

    def run_fixture(fixture: Any, model_name: str, device: str) -> Iterable[Any]:
        del model_name, device
        if not state:
            state.update(_load_onnx_runtime_state(artifact_dir, model_path))

        tokenizer = state["tokenizer"]
        session = state["session"]
        input_names = state["input_names"]
        config = state["config"]
        max_length = int(
            config.get("max_sequence_length")
            or config.get("max_position_embeddings")
            or 512
        )
        encoding = tokenizer(
            fixture.text,
            max_length=max_length,
            padding=False,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="np",
        )
        offsets = encoding.pop("offset_mapping")
        inputs = {
            name: value for name, value in encoding.items() if name in input_names
        }
        logits = session.run(None, inputs)[0]
        return _decode_token_classification(
            logits[0],
            offsets[0],
            state["id2label"],
            fixture.text,
        )

    return run_fixture


def _load_onnx_runtime_state(artifact_dir: Path, model_path: Path) -> dict[str, Any]:
    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "onnxruntime and transformers are required to evaluate ONNX INT8. "
            "Install with: pip install openmed[onnx]"
        ) from exc

    config = _read_json(artifact_dir / "config.json")
    tokenizer = get_tokenizer_with_loader(
        str(artifact_dir),
        AutoTokenizer.from_pretrained,
    )
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    return {
        "config": config,
        "id2label": _id2label(config),
        "input_names": {item.name for item in session.get_inputs()},
        "session": session,
        "tokenizer": tokenizer,
    }


def _decode_token_classification(
    logits: Any,
    offsets: Any,
    id2label: Mapping[int, str],
    text: str,
) -> list[dict[str, Any]]:
    probabilities = _softmax_rows(logits)
    token_ids = probabilities.argmax(axis=-1)
    token_scores = probabilities.max(axis=-1)
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for label_id, score, raw_offset in zip(token_ids, token_scores, offsets):
        start, end = _offset_tuple(raw_offset)
        raw_label = id2label.get(int(label_id), f"LABEL_{int(label_id)}")
        prefix, label = _split_token_label(raw_label)
        if start == end or label == "O":
            current = _flush_current(entities, current, text)
            continue

        starts_new = (
            current is None
            or current["entity_group"] != label
            or prefix in {"B", "S"}
            or start > int(current["end"])
        )
        if starts_new:
            current = _flush_current(entities, current, text)
            current = {
                "entity_group": label,
                "start": start,
                "end": end,
                "scores": [float(score)],
            }
        else:
            current["end"] = max(int(current["end"]), end)
            current["scores"].append(float(score))

        if prefix in {"E", "S"}:
            current = _flush_current(entities, current, text)

    _flush_current(entities, current, text)
    return entities


def _flush_current(
    entities: list[dict[str, Any]],
    current: dict[str, Any] | None,
    text: str,
) -> dict[str, Any] | None:
    if current is None:
        return None
    start = int(current["start"])
    end = int(current["end"])
    scores = current.get("scores") or [0.0]
    entities.append(
        {
            "entity_group": current["entity_group"],
            "score": sum(float(score) for score in scores) / len(scores),
            "start": start,
            "end": end,
            "word": text[start:end],
        }
    )
    return None


def _softmax_rows(logits: Any) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is required to evaluate ONNX INT8 outputs. "
            "Install with: pip install numpy"
        ) from exc

    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _offset_tuple(value: Any) -> tuple[int, int]:
    start, end = value
    return int(start), int(end)


def _split_token_label(raw_label: str) -> tuple[str, str]:
    normalized = raw_label.strip()
    if not normalized or normalized.upper() == "O":
        return "", "O"
    if len(normalized) > 2 and normalized[1] in {"-", "_"}:
        prefix = normalized[0].upper()
        if prefix in {"B", "I", "E", "S"}:
            return prefix, normalized[2:]
    return "", normalized


def _per_label_recall(metrics: Mapping[str, Any]) -> dict[str, float]:
    recall_slices = metrics.get("recall_slices")
    if isinstance(recall_slices, Mapping):
        by_label = recall_slices.get("by_label")
        if isinstance(by_label, Mapping):
            return _float_map(by_label)

    per_label = metrics.get("per_label_recall")
    if isinstance(per_label, Mapping):
        return _float_map(per_label)
    return {}


def _per_label_recall_comparison(
    parent_recall: Mapping[str, float],
    candidate_recall: Mapping[str, float],
    per_label_delta: Mapping[str, float],
    *,
    span_counts: Mapping[str, int],
    char_counts: Mapping[str, int],
) -> dict[str, dict[str, float | int | None]]:
    labels = sorted(
        set(parent_recall)
        | set(candidate_recall)
        | set(per_label_delta)
        | set(span_counts)
        | set(char_counts)
    )
    return {
        label: {
            "fp_recall": parent_recall.get(label),
            "int8_recall": candidate_recall.get(label),
            "delta": per_label_delta.get(label),
            "gold_span_count": int(span_counts.get(label, 0)),
            "gold_char_count": int(char_counts.get(label, 0)),
        }
        for label in labels
    }


def _gold_span_counts_by_label(fixtures: Sequence[Any]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for fixture in fixtures:
        for span in fixture.gold_spans:
            counts[str(span.label)] += 1
    return dict(sorted(counts.items()))


def _gold_char_counts_by_label(fixtures: Sequence[Any]) -> dict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    for fixture in fixtures:
        for span in fixture.gold_spans:
            counts[str(span.label)] += int(span.length)
    return dict(sorted(counts.items()))


def _update_manifest_int8_certification(
    artifact_dir: Path,
    *,
    quantization: Mapping[str, Any],
    payload: Mapping[str, Any],
    report_relpath: str,
) -> None:
    manifest_path = artifact_dir / "openmed-onnx.json"
    if not manifest_path.exists():
        return

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    manifest["formats"] = _dedupe_keep_order(
        [*[str(item) for item in manifest.get("formats", [])], ONNX_INT8_FORMAT]
    )
    manifest["quantization"] = dict(quantization)
    manifest["quant_recall_delta"] = payload["quant_recall_delta"]
    manifest["certified"] = payload["certified"]
    manifest["recall_delta_path"] = report_relpath
    manifest["certification"] = {
        "gate": "G4",
        "limit": payload["limit"],
        "metric": payload["metric"],
        "report_path": report_relpath,
    }
    manifest["artifacts"] = _upsert_int8_artifact(
        manifest.get("artifacts") or [],
        metadata=int8_artifact_metadata(recall_report=payload),
    )

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def _upsert_int8_artifact(
    artifacts: Sequence[Any],
    *,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    replaced = False
    for item in artifacts:
        artifact = dict(item) if isinstance(item, Mapping) else {}
        if artifact.get("path") == INT8_ONNX_FILENAME:
            artifact.update(
                {
                    "format": ONNX_INT8_FORMAT,
                    "path": INT8_ONNX_FILENAME,
                    "precision": "int8",
                    "metadata": dict(metadata),
                }
            )
            replaced = True
        result.append(artifact)
    if not replaced:
        result.append(
            {
                "format": ONNX_INT8_FORMAT,
                "path": INT8_ONNX_FILENAME,
                "precision": "int8",
                "metadata": dict(metadata),
            }
        )
    return result


def _base_quantization_metadata() -> dict[str, Any]:
    return {
        "scheme": "dynamic",
        "weight_type": "qint8",
        "op_types": list(DEFAULT_DYNAMIC_OP_TYPES),
        "calibration": "none",
        "target": "arm64-cpu",
        "output": INT8_ONNX_FILENAME,
    }


def _check_onnx_model(path: Path) -> None:
    try:
        import onnx
    except ImportError as exc:
        raise ImportError(
            "onnx is required to validate INT8 ONNX artifacts. "
            "Install with: pip install openmed[onnx]"
        ) from exc
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def _id2label(config: Mapping[str, Any]) -> dict[int, str]:
    id2label = config.get("id2label")
    if not isinstance(id2label, Mapping) or not id2label:
        raise ValueError("config.json must contain a non-empty id2label mapping")
    return {int(key): str(value) for key, value in id2label.items()}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def _float_map(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            result[str(key)] = parsed
    return result


def _artifact_relative_path(path: Path, artifact_dir: Path) -> str:
    try:
        return path.relative_to(artifact_dir).as_posix()
    except ValueError:
        return path.name


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "INT8_ONNX_FILENAME",
    "ONNX_INT8_FORMAT",
    "RECALL_DELTA_REPORT_FILENAME",
    "Int8QuantizationResult",
    "apply_int8_recall_certification",
    "int8_artifact_metadata",
    "quantize_android_int8",
    "quantize_dynamic_int8",
    "write_int8_recall_delta_report",
]
