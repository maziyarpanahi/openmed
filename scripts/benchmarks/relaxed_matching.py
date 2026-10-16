"""Measure real relaxed scorers with bounded, synthetic offline workloads.

Run this same script through each checkout's frozen environment. Timing samples
exclude input construction, imports and tracing. Peak traced allocations include
one scoring call; process peak RSS also includes imports and constructed inputs.
These measurements do not establish a clinical workload distribution or a gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

from openmed.eval.harness import (
    BenchmarkFixture,
    FixtureResult,
    _multilingual_group_rows,
)
from openmed.eval.metrics import EvalSpan, compute_relaxed_span_f1
from openmed.eval.relation_metrics import EvalRelation, compute_relation_f1

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None


def spans(size: int, shape: str) -> tuple[list[EvalSpan], list[EvalSpan]]:
    """Build equal-sized inputs; all strings and labels are synthetic."""
    if shape in {"sparse", "permuted", "chain"}:
        gold = [EvalSpan(i * 4, i * 4 + 2, "NAME") for i in range(size)]
    else:
        gold = [EvalSpan(0, size * 4, "NAME") for _ in range(size)]
    if shape == "chain":
        predicted = [EvalSpan(i * 4, (i + 1) * 4 + 2, "NAME") for i in range(size - 1)]
        predicted.append(EvalSpan(0, 1, "NAME"))
    elif shape == "blocked":
        predicted = [
            replace(row, label="LOCATION" if i >= size // 2 else "NAME")
            for i, row in enumerate(gold)
        ]
    else:
        predicted = list(gold)
    if shape == "permuted":
        rng = random.Random(4208)
        rng.shuffle(gold)
        rng.shuffle(predicted)
    return gold, predicted


def workload(kind: str, shape: str, size: int, documents: int):
    """Return a callable using the native public scorer or multilingual group."""
    gold, predicted = spans(size, shape)
    if kind == "span":
        if documents != 1:
            raise ValueError("Use multilingual for aggregate span workloads")
        return lambda: compute_relaxed_span_f1(gold, predicted).to_dict()
    if kind == "relation":
        tail = EvalSpan(size * 4 + 1, size * 4 + 3, "LOCATION")
        gold_relations = [
            EvalRelation("RELATED_TO", head, tail, fixture_id=str(doc))
            for doc in range(documents)
            for head in gold
        ]
        predicted_relations = [
            EvalRelation(
                "OTHER" if shape == "blocked" and i >= size // 2 else "RELATED_TO",
                head,
                tail,
                fixture_id=str(doc),
            )
            for doc in range(documents)
            for i, head in enumerate(predicted)
        ]
        return lambda: compute_relation_f1(
            gold_relations, predicted_relations, match="relaxed"
        ).to_dict()
    fixtures = [
        BenchmarkFixture(
            str(doc),
            "x" * (size * 4 + 4),
            tuple(gold),
            metadata={"benchmark": "synthetic"},
        )
        for doc in range(documents)
    ]
    results = [
        FixtureResult(str(doc), tuple(predicted), 0.0) for doc in range(documents)
    ]
    return lambda: _multilingual_group_rows(
        fixtures, results, group_keys=("benchmark", "language")
    )[0]["relaxed_span_f1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", choices=["span", "relation", "multilingual"], required=True
    )
    parser.add_argument(
        "--shape",
        choices=["sparse", "dense", "permuted", "chain", "blocked"],
        required=True,
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if (
        not 1 <= args.size * args.documents <= 10000
        or args.size < 1
        or args.documents < 1
        or not 1 <= args.repeat <= 10
    ):
        parser.error(
            "Bound synthetic runs to 1..10000 records per side and 1..10 repetitions"
        )
    evaluate = workload(args.workload, args.shape, args.size, args.documents)
    reference = evaluate()
    timings = []
    for _ in range(args.repeat):
        gc.collect()
        start = time.perf_counter()
        result = evaluate()
        timings.append(time.perf_counter() - start)
        assert result == reference
    gc.collect()
    tracemalloc.start()
    try:
        result = evaluate()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == reference
    root = Path.cwd()
    source_hashes = {}
    for name in ["metrics.py", "relation_metrics.py", "_matching.py", "cache.py"]:
        path = root / "openmed/eval" / name
        if path.exists():
            source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "workload": args.workload,
                "shape": args.shape,
                "per_document_count": args.size,
                "documents": args.documents,
                "gold_count": args.size * args.documents,
                "prediction_count": args.size * args.documents,
                "seconds": timings,
                "median_seconds": statistics.median(timings),
                "peak_traced_bytes": peak,
                "process_peak_rss_bytes": (
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                    if resource is not None
                    else None
                ),
                "metrics": reference,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "module_root": str(root),
                "head_sha": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "uv_lock_sha256": hashlib.sha256(
                    (root / "uv.lock").read_bytes()
                ).hexdigest(),
                "source_sha256": source_hashes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
