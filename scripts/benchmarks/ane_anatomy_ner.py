#!/usr/bin/env python3
"""Benchmark AnatomyDetect-BioClinical-108M: PyTorch vs CoreML ANE.

Runs a real in-repo anatomy task (the synthetic ED note plus radiology
reports), then compares Hugging Face PyTorch latency against a CoreML
Neural Engine package.

Example::

    .venv/bin/python scripts/benchmarks/ane_anatomy_ner.py
    .venv/bin/python scripts/benchmarks/ane_anatomy_ner.py --skip-convert
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "OpenMed/OpenMed-NER-AnatomyDetect-BioClinical-108M"
DEFAULT_NOTE = ROOT / "tests" / "fixtures" / "clinical_note.txt"
DEFAULT_RADIOLOGY = ROOT / "tests" / "fixtures" / "clinical" / "radiology_report.jsonl"
COREML_FIXTURE_DIR = ROOT / "tests" / "fixtures" / "coreml"
DEFAULT_ANE_PACKAGE = (
    COREML_FIXTURE_DIR / "OpenMed-NER-AnatomyDetect-BioClinical-108M-ane.mlpackage"
)
DEFAULT_MLX_EXAMPLE = (
    "Synthetic note: the biopsy sampled liver tissue near the portal vein."
)


def _load_task_texts() -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = [
        ("mlx_anatomy_example", DEFAULT_MLX_EXAMPLE),
        ("clinical_note", DEFAULT_NOTE.read_text(encoding="utf-8")),
    ]
    with DEFAULT_RADIOLOGY.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            payload = json.loads(line)
            texts.append((f"radiology_{index}", str(payload["text"])))
    return texts


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return float(ordered[index])


def _compact_residency(report) -> dict[str, Any]:
    return {
        "ane_residency_percentage": report.ane_residency_percentage,
        "ane_resident_ops": report.ane_resident_ops,
        "total_ops": report.total_ops,
        "passed": report.passed,
        "blocking_cpu_ops": [
            {
                "name": layer.name,
                "op_type": layer.op_type,
                "compute_unit": layer.compute_unit,
            }
            for layer in report.blocking_cpu_fallback_layers
        ],
        "cpu_fallback_count": len(report.cpu_fallback_layers),
        "source": report.source,
    }


def _summarize(
    values: Sequence[float], *, seq_len: int | None = None
) -> dict[str, float]:
    payload: dict[str, float] = {
        "mean_ms": float(statistics.fmean(values)) if values else float("nan"),
        "p50_ms": float(statistics.median(values)) if values else float("nan"),
        "p95_ms": _percentile(values, 0.95),
    }
    p50 = payload["p50_ms"]
    if p50 and p50 > 0:
        payload["notes_per_s"] = 1000.0 / p50
        if seq_len:
            payload["tokens_per_s"] = (seq_len * 1000.0) / p50
    return payload


def _decode_entities(
    logits,
    offsets: Sequence[Sequence[int]],
    id2label: Mapping[int, str],
    text: str,
) -> list[dict[str, Any]]:
    from openmed.core.decoding import build_label_info, labels_to_token_spans

    label_ids = logits[0].argmax(axis=-1).tolist()
    label_info = build_label_info(id2label)
    labels_by_index = {
        index: int(label_id)
        for index, label_id in enumerate(label_ids)
        if index < len(offsets) and tuple(offsets[index]) != (0, 0)
    }
    entities = []
    for span_label, token_start, token_end in labels_to_token_spans(
        labels_by_index,
        label_info,
    ):
        if token_start >= len(offsets) or token_end - 1 >= len(offsets):
            continue
        start = int(offsets[token_start][0])
        end = int(offsets[token_end - 1][1])
        entities.append(
            {
                "label": label_info.span_class_names[span_label],
                "start": start,
                "end": end,
                "text": text[start:end],
            }
        )
    return entities


def _time_calls(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(iters):
        started = time.perf_counter()
        fn()
        values.append((time.perf_counter() - started) * 1000.0)
    return values


def _pytorch_backend(model_id: str, max_seq_length: int, device: str):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    model.eval()
    if device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
    elif device == "cpu":
        model = model.to("cpu")
    else:
        device = "cpu"
        model = model.to("cpu")
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}

    def run(text: str):
        encoded = tokenizer(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = (
                model(**encoded).logits.detach().to(dtype=torch.float32).cpu().numpy()
            )
        return _decode_entities(logits, offsets, id2label, text)

    return run, device


def _coreml_backend(
    package: Path, model_id: str, max_seq_length: int, compute_units: str
):
    import coremltools as ct
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    unit_attr = {
        "all": "ALL",
        "cpuAndNeuralEngine": "CPU_AND_NE",
        "cpuOnly": "CPU_ONLY",
    }[compute_units]
    mlmodel = ct.models.MLModel(
        str(package),
        compute_units=getattr(ct.ComputeUnit, unit_attr),
    )
    id2label_path = package.parent / f"{package.stem}_id2label.json"
    raw = json.loads(id2label_path.read_text(encoding="utf-8"))
    id2label = {int(key): str(value) for key, value in raw.items()}

    def run(text: str):
        encoded = tokenizer(
            text,
            max_length=max_seq_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
        )
        payload = {
            "input_ids": np.asarray([encoded["input_ids"]], dtype=np.int32),
            "attention_mask": np.asarray([encoded["attention_mask"]], dtype=np.int32),
        }
        logits = np.asarray(mlmodel.predict(payload)["logits"])
        return _decode_entities(logits, encoded["offset_mapping"], id2label, text)

    return run


def _convert(
    model_id: str,
    output: Path,
    *,
    max_seq_length: int,
    ane_conv_layout: bool,
    compute_units: str,
) -> Path:
    from openmed.coreml.convert import convert

    return convert(
        model_id,
        output,
        max_seq_length=max_seq_length,
        compute_precision="float16",
        compute_units=compute_units,
        optimize_for_ane=True,
        ane_conv_layout=ane_conv_layout,
        latency_iterations=0,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--work-dir",
        default=str(COREML_FIXTURE_DIR),
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument(
        "--compute-units",
        default="cpuAndNeuralEngine",
        choices=("all", "cpuAndNeuralEngine", "cpuOnly"),
    )
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    naive_pkg = work_dir / "naive.mlpackage"
    ane_pkg = (
        DEFAULT_ANE_PACKAGE
        if work_dir == COREML_FIXTURE_DIR
        else work_dir / DEFAULT_ANE_PACKAGE.name
    )
    report_path = work_dir / "bench.json"

    texts = _load_task_texts()
    long_name, long_text = texts[1]
    print(f"machine task: {long_name} ({len(long_text)} chars, {len(texts)} docs)")

    print("loading PyTorch baseline ...")
    torch_run, torch_device = _pytorch_backend(args.model, args.max_seq_length, "mps")
    torch_entities = torch_run(long_text)
    torch_times = _time_calls(
        lambda: torch_run(long_text),
        warmup=args.warmup,
        iters=args.iters,
    )
    torch_latency = _summarize(torch_times, seq_len=args.max_seq_length)
    print(
        f"  pytorch/{torch_device}: "
        f"{torch_latency['p50_ms']:.1f} ms p50, "
        f"{torch_latency.get('tokens_per_s', float('nan')):.0f} tok/s, "
        f"{len(torch_entities)} anatomy spans"
    )

    naive_error = None
    ane_error = None
    if not args.skip_convert:
        print("converting naive CoreML (HF graph, static shapes) ...")
        try:
            _convert(
                args.model,
                naive_pkg,
                max_seq_length=args.max_seq_length,
                ane_conv_layout=False,
                compute_units=args.compute_units,
            )
        except Exception as exc:
            naive_error = str(exc)
            print(f"  naive convert failed: {exc}")
        print("converting ANE CoreML (BC1S 1x1 conv) ...")
        try:
            _convert(
                args.model,
                ane_pkg,
                max_seq_length=args.max_seq_length,
                ane_conv_layout=True,
                compute_units=args.compute_units,
            )
        except Exception as exc:
            ane_error = str(exc)
            print(f"  ane convert failed: {exc}")
            if naive_error:
                raise
    elif not ane_pkg.exists():
        ane_error = "ane package missing; rerun without --skip-convert"

    from openmed.coreml.convert import analyze_ane_residency

    def _bench_package(package: Path, error: str | None) -> dict[str, Any]:
        if error or not package.exists():
            return {"path": str(package), "error": error or "missing package"}
        run = _coreml_backend(
            package, args.model, args.max_seq_length, args.compute_units
        )
        entities = run(long_text)
        times = _time_calls(
            lambda: run(long_text),
            warmup=args.warmup,
            iters=args.iters,
        )
        latency = _summarize(times, seq_len=args.max_seq_length)
        print(
            f"  {package.stem}: {latency['p50_ms']:.1f} ms p50, "
            f"{latency.get('tokens_per_s', float('nan')):.0f} tok/s, "
            f"{len(entities)} spans"
        )
        return {
            "path": str(package),
            "latency": latency,
            "entities": len(entities),
            "preview": entities[:12],
            "residency": _compact_residency(analyze_ane_residency(package)),
        }

    naive_report = _bench_package(naive_pkg, naive_error)
    ane_report = _bench_package(ane_pkg, ane_error)

    def _speedup(base: Sequence[float], fast: Sequence[float] | None) -> float | None:
        if not fast:
            return None
        return float(statistics.median(base) / statistics.median(fast))

    naive_times = (naive_report.get("latency") or {}).get("p50_ms")
    ane_times = (ane_report.get("latency") or {}).get("p50_ms")
    report = {
        "model": args.model,
        "task": long_name,
        "chars": len(long_text),
        "max_seq_length": args.max_seq_length,
        "compute_units": args.compute_units,
        "pytorch": {
            "device": torch_device,
            "latency": torch_latency,
            "entities": len(torch_entities),
        },
        "coreml_naive": naive_report,
        "coreml_ane": ane_report,
        "speedup_vs_pytorch": {
            "naive": (
                None if naive_times is None else torch_latency["p50_ms"] / naive_times
            ),
            "ane": (None if ane_times is None else torch_latency["p50_ms"] / ane_times),
        },
        "speedup_ane_vs_naive": (
            None
            if naive_times is None or ane_times is None
            else naive_times / ane_times
        ),
        "entity_preview": ane_report.get("preview") or torch_entities[:12],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
