#!/usr/bin/env python3
"""Tight CoreML ANE predict loop for xctrace (Time Profiler / Core ML / ANE).

Tokenize once, then call ``MLModel.predict`` in a loop so the trace is the
compiled graph rather than Hugging Face tokenization.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = (
    ROOT
    / "tests"
    / "fixtures"
    / "coreml"
    / "OpenMed-NER-AnatomyDetect-BioClinical-108M-ane.mlpackage"
)
DEFAULT_NOTE = ROOT / "tests" / "fixtures" / "clinical_note.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default=str(DEFAULT_PACKAGE))
    parser.add_argument("--text-file", default=str(DEFAULT_NOTE))
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=80)
    args = parser.parse_args()

    import coremltools as ct
    import numpy as np
    from transformers import AutoTokenizer

    package = Path(args.package)
    text = Path(args.text_file).read_text(encoding="utf-8")
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenMed/OpenMed-NER-AnatomyDetect-BioClinical-108M"
    )
    encoded = tokenizer(
        text,
        max_length=args.max_seq_length,
        padding="max_length",
        truncation=True,
    )
    payload = {
        "input_ids": np.asarray([encoded["input_ids"]], dtype=np.int32),
        "attention_mask": np.asarray([encoded["attention_mask"]], dtype=np.int32),
    }
    model = ct.models.MLModel(
        str(package),
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    for _ in range(args.warmup):
        model.predict(payload)
    print("READY", flush=True)
    started = time.perf_counter()
    for _ in range(args.iters):
        model.predict(payload)
    elapsed = time.perf_counter() - started
    print(
        f"predict-only {args.iters} iters in {elapsed * 1000:.1f} ms "
        f"({elapsed * 1000 / args.iters:.2f} ms/iter)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
