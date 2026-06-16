from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts/security/pip_audit_gate.py"
SPEC = importlib.util.spec_from_file_location("pip_audit_gate", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
pip_audit_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pip_audit_gate)


def test_fixable_vulnerabilities_only_returns_advisories_with_fixes() -> None:
    report = {
        "dependencies": [
            {
                "name": "fixed-package",
                "vulns": [
                    {"id": "PYSEC-2026-1", "fix_versions": ["1.2.3"]},
                    {"id": "PYSEC-2026-2", "fix_versions": []},
                ],
            },
            {"name": "clean-package", "vulns": []},
        ],
    }

    assert pip_audit_gate.fixable_vulnerabilities(report) == [
        ("fixed-package", "PYSEC-2026-1", ["1.2.3"]),
    ]


def test_load_ignores_requires_non_expired_review_date(tmp_path: Path) -> None:
    ignore_file = tmp_path / "pip-audit-ignore.toml"
    ignore_file.write_text(
        """
[[ignore]]
id = "PYSEC-2026-3"
reason = "No fixed version is available yet."
review_by = "2026-08-01"
""".strip()
    )

    assert pip_audit_gate.load_ignores(
        ignore_file,
        today=dt.date(2026, 6, 16),
    ) == {"PYSEC-2026-3"}


def test_load_ignores_rejects_expired_entries(tmp_path: Path) -> None:
    ignore_file = tmp_path / "pip-audit-ignore.toml"
    ignore_file.write_text(
        """
[[ignore]]
id = "PYSEC-2026-4"
reason = "No fixed version is available yet."
review_by = "2026-01-01"
""".strip()
    )

    with pytest.raises(ValueError, match="expired"):
        pip_audit_gate.load_ignores(ignore_file, today=dt.date(2026, 6, 16))
