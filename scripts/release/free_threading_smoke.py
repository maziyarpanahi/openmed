#!/usr/bin/env python3
"""Smoke-test the core install on free-threaded CPython.

The script imports ``openmed``, reports whether the GIL is still disabled and
which extension module re-enabled it, then runs the deterministic
``safety_sweep()`` on synthetic notes from several threads at once and
compares every result with a single-threaded baseline.

The report names modules and states only. Results are compared through a
SHA-256 digest of offsets, labels and confidences; no input or matched text is
written anywhere.

Exit status:
    0: every threaded result matched the baseline. A GIL re-enabled at import
       is reported as a warning, not a failure.
    1: a threaded result differed from the baseline, or a sweep raised.
    2: ``--require-free-threaded`` was given on a build with the GIL.

Usage::

    python scripts/release/free_threading_smoke.py --require-free-threaded \\
        --report free-threading-smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import sysconfig
import threading
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA_VERSION = 1
DEFAULT_THREADS = 4
DEFAULT_ITERATIONS = 25

# Synthetic notes with structured identifiers only. Values are fabricated test
# patterns (reserved card, documentation phone range, example.test domain).
SYNTHETIC_NOTES: tuple[tuple[str, str], ...] = (
    (
        "en",
        "Synthetic note: SSN 219-09-9999, MRN 00123456, card 4111 1111 1111 "
        "1111, phone (555) 010-0199, email synthetic.patient@example.test.",
    ),
    ("en", "Synthetic follow-up for MRN: 7654321 at ZIP code 10001."),
    ("en", "Synthetic split identifiers: SSN 2 1 9-0 9-9 9 9 9 and MRN #00-12-34-56."),
    ("en", "Synthetic wide digits: card ４１１１ １１１１ １１１１ １１１１."),
    ("it", "Nota sintetica: CAP 00100, telefono +39 06 0000 0000."),
    ("de", "Synthetische Notiz: Steuer-ID 12345678912 verified."),
)

_GIL_WARNING = re.compile(r"enabled to load module '([^']+)'")

SweepFunction = Callable[..., Sequence[Any]]


def free_threaded_build() -> bool:
    """Return whether this interpreter was built with ``Py_GIL_DISABLED``."""

    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def gil_enabled() -> bool | None:
    """Return whether the GIL is currently enabled, or ``None`` if unknown."""

    probe = getattr(sys, "_is_gil_enabled", None)
    return probe() if callable(probe) else None


def import_openmed() -> tuple[SweepFunction, list[str]]:
    """Import the core package and record modules that re-enabled the GIL.

    Returns:
        The ``safety_sweep`` function and the module names named by the
        interpreter's GIL re-enablement warnings, in import order.
    """

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        import openmed  # noqa: F401  - the import under test
        from openmed.core.safety_sweep import safety_sweep
    return safety_sweep, gil_reenabling_modules(caught)


def gil_reenabling_modules(caught: Sequence[warnings.WarningMessage]) -> list[str]:
    """Extract module names from CPython's GIL re-enablement warnings.

    Args:
        caught: Warnings recorded while importing.

    Returns:
        Module names, in the order the interpreter reported them.
    """

    return [
        match
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        for match in _GIL_WARNING.findall(str(warning.message))
    ]


def result_signature(spans: Sequence[Any]) -> list[tuple[Any, ...]]:
    """Return a text-free, comparable signature of sweep output.

    Args:
        spans: Spans returned by ``safety_sweep()``.

    Returns:
        ``(start, end, label, confidence)`` tuples in output order.
    """

    signature = []
    for span in spans:
        if isinstance(span, dict):
            values = (span.get("start"), span.get("end"), span.get("label"))
            confidence = span.get("confidence")
        else:
            values = (span.start, span.end, span.label)
            confidence = getattr(span, "confidence", None)
        rounded = round(float(confidence), 9) if confidence is not None else None
        signature.append((*values, rounded))
    return signature


def _sweep_all(sweep: SweepFunction) -> list[list[tuple[Any, ...]]]:
    return [
        result_signature(sweep(text, [], lang=lang)) for lang, text in SYNTHETIC_NOTES
    ]


def digest(signatures: Any) -> str:
    """Return the SHA-256 hex digest of JSON-encoded signatures."""

    encoded = json.dumps(signatures, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_threads(
    sweep: SweepFunction,
    *,
    threads: int = DEFAULT_THREADS,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    """Compare threaded sweeps with a single-threaded baseline.

    Args:
        sweep: The ``safety_sweep`` function, injectable for tests.
        threads: Number of concurrent worker threads.
        iterations: Passes over the synthetic notes per thread.

    Returns:
        A text-free report section with digests, counts and error types.
    """

    baseline = _sweep_all(sweep)
    baseline_digest = digest(baseline)
    barrier = threading.Barrier(threads)
    mismatches = [0] * threads
    errors: list[str] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        try:
            barrier.wait()
            for _ in range(iterations):
                if _sweep_all(sweep) != baseline:
                    mismatches[index] += 1
        except Exception as error:  # noqa: BLE001 - recorded by type only
            with lock:
                errors.append(type(error).__name__)

    workers = [
        threading.Thread(target=worker, args=(index,), name=f"sweep-{index}")
        for index in range(threads)
    ]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    if errors:
        status = "crash"
    elif any(mismatches):
        status = "mismatch"
    else:
        status = "match"
    return {
        "status": status,
        "threads": threads,
        "iterations": iterations,
        "notes": len(SYNTHETIC_NOTES),
        "baseline_spans": sum(len(signature) for signature in baseline),
        "baseline_digest": baseline_digest,
        "mismatched_passes": sum(mismatches),
        "error_types": sorted(set(errors)),
    }


def build_report(
    *,
    sweep: SweepFunction | None = None,
    threads: int = DEFAULT_THREADS,
    iterations: int = DEFAULT_ITERATIONS,
) -> dict[str, Any]:
    """Run the import probe and the threaded comparison.

    Args:
        sweep: Optional replacement for ``safety_sweep``; the real package is
            still imported so the GIL probe stays meaningful.
        threads: Number of concurrent worker threads.
        iterations: Passes over the synthetic notes per thread.

    Returns:
        The JSON-ready report.
    """

    gil_before = gil_enabled()
    imported_sweep, reenabled_by = import_openmed()
    gil_after = gil_enabled()
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "free_threaded_build": free_threaded_build(),
        "gil_enabled_before_import": gil_before,
        "gil_enabled_after_import": gil_after,
        "gil_reenabled_by": reenabled_by,
        "sweep": run_threads(
            sweep or imported_sweep, threads=threads, iterations=iterations
        ),
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--report", type=Path, help="Write the JSON report here.")
    parser.add_argument(
        "--require-free-threaded",
        action="store_true",
        help="Exit 2 unless the interpreter is a free-threaded build.",
    )
    args = parser.parse_args(argv)
    if args.threads < 2 or args.iterations < 1:
        parser.error("--threads must be at least 2 and --iterations at least 1")
    return args


def main(
    argv: Sequence[str] | None = None, *, sweep: SweepFunction | None = None
) -> int:
    """Run the smoke test from the command line.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.
        sweep: Optional ``safety_sweep`` replacement, used by tests to inject
            a thread-dependent mismatch.

    Returns:
        Process exit status as documented in the module docstring.
    """

    args = _parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.require_free_threaded and not free_threaded_build():
        print("free-threading smoke: this interpreter is not a free-threaded build")
        return 2

    report = build_report(sweep=sweep, threads=args.threads, iterations=args.iterations)
    if args.report is not None:
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    sweep_report = report["sweep"]
    print(
        "free-threading smoke: "
        f"python={report['python']} "
        f"free_threaded={str(report['free_threaded_build']).lower()} "
        f"gil_after_import={report['gil_enabled_after_import']} "
        f"sweep={sweep_report['status']} "
        f"threads={sweep_report['threads']} "
        f"mismatched_passes={sweep_report['mismatched_passes']}"
    )
    for module in report["gil_reenabled_by"]:
        # GitHub Actions renders this line as a warning annotation.
        print(
            "::warning title=GIL re-enabled::"
            f"importing openmed re-enabled the GIL via module {module}"
        )
    return 0 if sweep_report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
