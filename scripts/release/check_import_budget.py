#!/usr/bin/env python3
"""Enforce OpenMed core import-time and optional-module budgets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_FILE = ROOT / "gates" / "release_budgets.json"
IMPORT_GATE_KEY = "import::openmed::ubuntu-latest"
MODULE_SENTINEL = "OPENMED_IMPORT_MODULES="
IMPORT_TIME_RE = re.compile(
    r"^import time:\s+\d+\s+\|\s+(\d+)\s+\|\s+openmed\s*$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ImportBudget:
    """Committed core import constraints."""

    maximum_cumulative_microseconds: int
    forbidden_modules: tuple[str, ...]


@dataclass(frozen=True)
class ImportMeasurement:
    """Observed core import time and loaded optional modules."""

    cumulative_microseconds: int
    loaded_forbidden_modules: tuple[str, ...]


def load_import_budget(path: Path = DEFAULT_GATE_FILE) -> ImportBudget:
    """Load and validate the committed import budget."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["entries"][IMPORT_GATE_KEY]["metrics"]
    budget = ImportBudget(
        maximum_cumulative_microseconds=int(metrics["maximum_cumulative_microseconds"]),
        forbidden_modules=tuple(str(name) for name in metrics["forbidden_modules"]),
    )
    if budget.maximum_cumulative_microseconds <= 0:
        raise ValueError("import-time budget must be positive")
    if not budget.forbidden_modules or any(
        not name or "." in name for name in budget.forbidden_modules
    ):
        raise ValueError("forbidden_modules must contain top-level module names")
    if len(set(budget.forbidden_modules)) != len(budget.forbidden_modules):
        raise ValueError("forbidden_modules must not contain duplicates")
    return budget


def parse_cumulative_import_time(stderr: str) -> int:
    """Extract the cumulative ``openmed`` import time in microseconds."""

    matches = IMPORT_TIME_RE.findall(stderr)
    if len(matches) != 1:
        raise ValueError(
            "expected one cumulative import-time row for the openmed package"
        )
    return int(matches[0])


def parse_loaded_modules(stdout: str) -> tuple[str, ...]:
    """Extract the optional modules loaded by the measurement subprocess."""

    lines = [line for line in stdout.splitlines() if line.startswith(MODULE_SENTINEL)]
    if len(lines) != 1:
        raise ValueError("import subprocess did not emit its module sentinel")
    modules = json.loads(lines[0][len(MODULE_SENTINEL) :])
    if not isinstance(modules, list) or any(
        not isinstance(name, str) for name in modules
    ):
        raise ValueError("import subprocess emitted an invalid module list")
    return tuple(modules)


def measure_import(
    budget: ImportBudget,
    python: str = sys.executable,
) -> ImportMeasurement:
    """Measure a fresh ``import openmed`` outside the source checkout."""

    forbidden = repr(budget.forbidden_modules)
    code = "\n".join(
        [
            "import json",
            "import sys",
            "import openmed",
            f"forbidden = {forbidden}",
            "loaded = sorted(",
            "    name for name in sys.modules",
            "    if any(",
            "        name == module or name.startswith(module + '.')",
            "        for module in forbidden",
            "    )",
            ")",
            f"print({MODULE_SENTINEL!r} + json.dumps(loaded))",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="openmed-import-budget-") as workdir:
        result = subprocess.run(
            [python, "-X", "importtime", "-c", code],
            cwd=workdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"import subprocess failed: {detail}")

    return ImportMeasurement(
        cumulative_microseconds=parse_cumulative_import_time(result.stderr),
        loaded_forbidden_modules=parse_loaded_modules(result.stdout),
    )


def import_budget_failures(
    measurement: ImportMeasurement,
    budget: ImportBudget,
) -> list[str]:
    """Return every import-budget violation."""

    failures: list[str] = []
    if measurement.cumulative_microseconds > budget.maximum_cumulative_microseconds:
        failures.append(
            "cumulative import time exceeds the committed maximum "
            f"({measurement.cumulative_microseconds} > "
            f"{budget.maximum_cumulative_microseconds} microseconds)"
        )
    if measurement.loaded_forbidden_modules:
        failures.append(
            "core import loaded optional language modules: "
            + ", ".join(measurement.loaded_forbidden_modules)
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    """Measure a fresh core import and enforce committed constraints."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-file",
        type=Path,
        default=DEFAULT_GATE_FILE,
        help="Committed JSON gate file.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the fresh import.",
    )
    args = parser.parse_args(argv)

    try:
        budget = load_import_budget(args.gate_file)
        measurement = measure_import(budget, args.python)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Import budget check could not run: {exc}", file=sys.stderr)
        return 2

    failures = import_budget_failures(measurement, budget)
    if failures:
        print("Import budget failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("Import budget passed")
    print(
        f"- cumulative import: {measurement.cumulative_microseconds} / "
        f"{budget.maximum_cumulative_microseconds} microseconds"
    )
    print("- optional language modules loaded: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
