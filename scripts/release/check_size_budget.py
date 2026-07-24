#!/usr/bin/env python3
"""Build and enforce the OpenMed wheel-size budget."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_FILE = ROOT / "gates" / "release_budgets.json"
DEFAULT_WHEEL_DIR = ROOT / "dist"
DEFAULT_REPORT = ROOT / "size-budget-report.json"
WHEEL_GATE_KEY = "package::openmed::wheel"
INSTALL_PROFILES = (
    ("openmed", None),
    ("openmed[zh]", "zh"),
    ("openmed[indic]", "indic"),
)


@dataclass(frozen=True)
class WheelBudget:
    """Committed wheel-size baseline and maximum."""

    baseline_bytes: int
    maximum_bytes: int
    headroom_percent: int


def load_wheel_budget(path: Path = DEFAULT_GATE_FILE) -> WheelBudget:
    """Load and validate the committed wheel-size budget."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["entries"][WHEEL_GATE_KEY]["metrics"]
    budget = WheelBudget(
        baseline_bytes=int(metrics["baseline_bytes"]),
        maximum_bytes=int(metrics["maximum_bytes"]),
        headroom_percent=int(metrics["headroom_percent"]),
    )
    if budget.baseline_bytes <= 0 or budget.maximum_bytes <= 0:
        raise ValueError("wheel-size budget values must be positive")
    if budget.headroom_percent < 0:
        raise ValueError("wheel-size headroom must not be negative")

    expected_maximum = math.ceil(
        budget.baseline_bytes * (100 + budget.headroom_percent) / 100
    )
    if budget.maximum_bytes != expected_maximum:
        raise ValueError(
            "maximum_bytes must equal baseline_bytes plus the committed headroom"
        )
    return budget


def find_wheel(wheel_dir: Path) -> Path:
    """Return the single OpenMed wheel in ``wheel_dir``."""

    wheels = sorted(wheel_dir.glob("openmed-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            f"expected one OpenMed wheel in {wheel_dir}, found {len(wheels)}"
        )
    return wheels[0]


def build_wheel(wheel_dir: Path) -> Path:
    """Build an OpenMed wheel and return its path."""

    wheel_dir.mkdir(parents=True, exist_ok=True)
    existing = list(wheel_dir.glob("openmed-*.whl"))
    if existing:
        names = ", ".join(path.name for path in sorted(existing))
        raise RuntimeError(
            f"refusing to mix a new wheel with existing artifacts in {wheel_dir}: "
            f"{names}"
        )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"wheel build failed: {detail}")
    return find_wheel(wheel_dir)


def directory_size(path: Path) -> int:
    """Return the total size of regular files below ``path``."""

    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def install_command(target: str, requirement: str) -> list[str]:
    """Return an available isolated-target installer command."""

    if importlib.util.find_spec("pip") is not None:
        return [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--no-input",
            "--quiet",
            "--target",
            target,
            requirement,
        ]

    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "neither the selected Python's pip module nor the uv executable is "
            "available"
        )
    return [
        uv,
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-compile",
        "--quiet",
        "--target",
        target,
        requirement,
    ]


def install_requirement_size(wheel: Path, extra: str | None) -> int:
    """Install one wheel profile into an isolated target and return its size."""

    requirement = str(wheel.resolve())
    if extra is not None:
        requirement += f"[{extra}]"

    with tempfile.TemporaryDirectory(prefix=f"openmed-{extra or 'core'}-") as target:
        result = subprocess.run(
            install_command(target, requirement),
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"could not install {requirement}: {detail}")
        return directory_size(Path(target))


def installed_profile_report(wheel: Path) -> list[dict[str, Any]]:
    """Measure core and language-extra installed footprints."""

    measurements: list[tuple[str, str | None, int]] = []
    for name, extra in INSTALL_PROFILES:
        measurements.append((name, extra, install_requirement_size(wheel, extra)))

    core_bytes = measurements[0][2]
    return [
        {
            "delta_from_core_bytes": installed_bytes - core_bytes,
            "extra": extra,
            "installed_bytes": installed_bytes,
            "name": name,
        }
        for name, extra, installed_bytes in measurements
    ]


def wheel_budget_failure(
    wheel_size_bytes: int,
    budget: WheelBudget,
) -> str | None:
    """Return a failure message when ``wheel_size_bytes`` exceeds the budget."""

    if wheel_size_bytes <= budget.maximum_bytes:
        return None
    overage = wheel_size_bytes - budget.maximum_bytes
    return (
        f"wheel is {overage} bytes over the committed maximum "
        f"({wheel_size_bytes} > {budget.maximum_bytes})"
    )


def create_report(
    wheel: Path,
    budget: WheelBudget,
    gate_file: Path,
) -> dict[str, Any]:
    """Create the JSON-serializable size-budget report."""

    wheel_size = wheel.stat().st_size
    try:
        gate_name = str(gate_file.resolve().relative_to(ROOT))
    except ValueError:
        gate_name = str(gate_file)
    return {
        "gate_file": gate_name,
        "installations": installed_profile_report(wheel),
        "schema_version": 1,
        "wheel": {
            "baseline_bytes": budget.baseline_bytes,
            "filename": wheel.name,
            "headroom_percent": budget.headroom_percent,
            "maximum_bytes": budget.maximum_bytes,
            "size_bytes": wheel_size,
            "within_budget": wheel_size <= budget.maximum_bytes,
        },
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write ``report`` as deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    """Build or inspect a wheel, record footprints, and enforce its budget."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-file",
        type=Path,
        default=DEFAULT_GATE_FILE,
        help="Committed JSON gate file.",
    )
    parser.add_argument(
        "--wheel-dir",
        type=Path,
        default=DEFAULT_WHEEL_DIR,
        help="Directory to build into or inspect.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help="JSON report destination.",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Inspect the existing wheel instead of building one.",
    )
    args = parser.parse_args(argv)

    try:
        budget = load_wheel_budget(args.gate_file)
        wheel = (
            find_wheel(args.wheel_dir)
            if args.skip_build
            else build_wheel(args.wheel_dir)
        )
        report = create_report(wheel, budget, args.gate_file)
        write_report(report, args.report)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Wheel-size budget check could not run: {exc}", file=sys.stderr)
        return 2

    failure = wheel_budget_failure(report["wheel"]["size_bytes"], budget)
    if failure is not None:
        print(f"Wheel-size budget failed: {failure}", file=sys.stderr)
        print(f"Size report: {args.report}", file=sys.stderr)
        return 1

    print("Wheel-size budget passed")
    print(f"- wheel: {report['wheel']['size_bytes']} / {budget.maximum_bytes} bytes")
    for installation in report["installations"]:
        print(
            f"- {installation['name']}: {installation['installed_bytes']} bytes "
            f"(delta {installation['delta_from_core_bytes']:+d})"
        )
    print(f"- report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
