"""Core import-budget gate tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_import_budget.py"

spec = importlib.util.spec_from_file_location("check_import_budget", SCRIPT)
assert spec is not None
import_budget = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = import_budget
spec.loader.exec_module(import_budget)


def test_committed_import_budget_matches_release_contract():
    budget = import_budget.load_import_budget()

    assert budget.maximum_cumulative_microseconds == 300_000
    assert set(budget.forbidden_modules) == {
        "indicnlp",
        "jieba",
        "opencc",
        "pypinyin",
    }


def test_parse_cumulative_import_time_uses_top_level_package_row():
    stderr = "\n".join(
        [
            "import time:       100 |        100 | openmed.core",
            "import time:       200 |     245678 | openmed",
        ]
    )

    assert import_budget.parse_cumulative_import_time(stderr) == 245_678


def test_import_budget_reports_time_and_optional_module_violations():
    budget = import_budget.ImportBudget(
        maximum_cumulative_microseconds=300_000,
        forbidden_modules=("jieba", "indicnlp"),
    )
    measurement = import_budget.ImportMeasurement(
        cumulative_microseconds=300_001,
        loaded_forbidden_modules=("indicnlp", "jieba"),
    )

    failures = import_budget.import_budget_failures(measurement, budget)

    assert len(failures) == 2
    assert "300001 > 300000" in failures[0]
    assert "indicnlp, jieba" in failures[1]


def test_import_budget_accepts_fast_isolated_core_import():
    budget = import_budget.ImportBudget(
        maximum_cumulative_microseconds=300_000,
        forbidden_modules=("jieba", "indicnlp"),
    )
    measurement = import_budget.ImportMeasurement(
        cumulative_microseconds=299_999,
        loaded_forbidden_modules=(),
    )

    assert import_budget.import_budget_failures(measurement, budget) == []
