"""Wheel-size budget gate tests."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_size_budget.py"

spec = importlib.util.spec_from_file_location("check_size_budget", SCRIPT)
assert spec is not None
size_budget = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = size_budget
spec.loader.exec_module(size_budget)


def test_committed_wheel_budget_has_ten_percent_headroom():
    budget = size_budget.load_wheel_budget()

    assert budget.headroom_percent == 10
    assert budget.maximum_bytes == math.ceil(budget.baseline_bytes * 1.10)


def test_six_megabyte_wheel_growth_exceeds_committed_budget(tmp_path):
    budget = size_budget.load_wheel_budget()
    oversized_wheel = tmp_path / "openmed-test-py3-none-any.whl"
    with oversized_wheel.open("wb") as handle:
        handle.seek(budget.baseline_bytes + (6 * 1024 * 1024) - 1)
        handle.write(b"\0")

    failure = size_budget.wheel_budget_failure(
        oversized_wheel.stat().st_size,
        budget,
    )

    assert failure is not None
    assert "over the committed maximum" in failure


def test_install_command_falls_back_to_uv_without_pip(monkeypatch):
    monkeypatch.setattr(size_budget.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(size_budget.shutil, "which", lambda name: "/usr/bin/uv")

    command = size_budget.install_command("/tmp/target", "/tmp/openmed.whl[zh]")

    assert command == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
        "--no-compile",
        "--quiet",
        "--target",
        "/tmp/target",
        "/tmp/openmed.whl[zh]",
    ]


def test_size_report_lists_core_zh_and_indic_deltas(tmp_path, monkeypatch):
    wheel = tmp_path / "openmed-test-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    budget = size_budget.WheelBudget(
        baseline_bytes=5,
        maximum_bytes=6,
        headroom_percent=20,
    )
    sizes = {None: 1000, "zh": 5100, "indic": 1700}
    monkeypatch.setattr(
        size_budget,
        "install_requirement_size",
        lambda wheel_path, extra: sizes[extra],
    )

    report = size_budget.create_report(
        wheel,
        budget,
        size_budget.DEFAULT_GATE_FILE,
    )

    assert report["wheel"]["within_budget"] is True
    assert report["installations"] == [
        {
            "delta_from_core_bytes": 0,
            "extra": None,
            "installed_bytes": 1000,
            "name": "openmed",
        },
        {
            "delta_from_core_bytes": 4100,
            "extra": "zh",
            "installed_bytes": 5100,
            "name": "openmed[zh]",
        },
        {
            "delta_from_core_bytes": 700,
            "extra": "indic",
            "installed_bytes": 1700,
            "name": "openmed[indic]",
        },
    ]
