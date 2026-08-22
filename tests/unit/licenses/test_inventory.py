"""Tests for the offline dependency license inventory gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "licenses" / "inventory.py"
SPEC = importlib.util.spec_from_file_location("license_inventory", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
inventory = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inventory
SPEC.loader.exec_module(inventory)


def write_markdown_inventory(path: Path, rows: str) -> Path:
    target = path / "license-inventory.md"
    target.write_text(
        f"| Dependency | Scope | License expression |\n| --- | --- | --- |\n{rows}",
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("MIT", inventory.LicenseClass.PERMISSIVE),
        ("Apache-2.0 OR BSD-3-Clause", inventory.LicenseClass.PERMISSIVE),
        ("GPL-3.0-only", inventory.LicenseClass.RESTRICTED),
        ("MIT OR GPL-3.0-only", inventory.LicenseClass.RESTRICTED),
        ("MIT OR", inventory.LicenseClass.UNKNOWN),
        ("LicenseRef-local-review", inventory.LicenseClass.UNKNOWN),
        ("", inventory.LicenseClass.UNKNOWN),
    ],
)
def test_classify_license_fails_closed(expression: str, expected: str) -> None:
    assert inventory.classify_license(expression) == expected


def test_checked_in_inventory_covers_non_dev_project_dependencies() -> None:
    entries = inventory.parse_inventory(inventory.DEFAULT_INVENTORY)
    records = inventory.audit_project()

    assert len(entries) == 90
    assert len(records) == len(entries)
    assert {record.classification for record in records} == {
        inventory.LicenseClass.PERMISSIVE
    }
    assert {record.name for record in records} >= {
        "faker",
        "jieba",
        "pysbd",
        "pyyaml",
    }


def test_missing_dependency_is_unknown_and_fails_closed() -> None:
    records = inventory.audit_inventory(
        [inventory.InventoryEntry("known-package", "MIT")],
        required_dependencies=("known-package", "unreviewed-package"),
    )

    missing = next(record for record in records if record.name == "unreviewed-package")
    assert missing.classification == inventory.LicenseClass.UNKNOWN
    assert missing.entry.scope == "missing"


def test_inventory_order_is_deterministic_and_names_are_normalized(
    tmp_path: Path,
) -> None:
    path = write_markdown_inventory(
        tmp_path,
        "| `Zed_Package` | optional | MIT |\n"
        "| `alpha.package` | base | BSD-3-Clause |\n",
    )

    assert [entry.name for entry in inventory.parse_inventory(path)] == [
        "alpha-package",
        "zed-package",
    ]


def test_json_inventory_is_supported_without_package_metadata(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "dependencies": [
                    {"name": "synthetic-b", "license": "MIT"},
                    {"name": "synthetic-a", "license": "ISC"},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = inventory.audit_inventory(inventory.parse_inventory(path))

    assert [record.name for record in records] == ["synthetic-a", "synthetic-b"]
    assert all(
        record.classification == inventory.LicenseClass.PERMISSIVE for record in records
    )


def test_report_and_cli_do_not_emit_raw_license_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_marker = "LicenseRef-private-example-001"
    path = write_markdown_inventory(
        tmp_path,
        f"| `synthetic-package` | test | {private_marker} |\n",
    )
    report_path = tmp_path / "report.json"

    assert (
        inventory.main(
            [
                "--inventory",
                str(path),
                "--inventory-only",
                "--report",
                str(report_path),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    report_text = report_path.read_text(encoding="utf-8")

    assert private_marker not in output.out
    assert private_marker not in output.err
    assert private_marker not in report_text
    assert "license_expression" not in report_text
    assert json.loads(report_text)["summary"][inventory.LicenseClass.UNKNOWN] == 1


def test_malformed_inventory_error_does_not_echo_source_values(tmp_path: Path) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        '{"dependencies": [{"license": "synthetic-private-value"}]}',
        encoding="utf-8",
    )

    with pytest.raises(inventory.InventoryError, match="no dependency name") as error:
        inventory.parse_inventory(path)

    assert "synthetic-private-value" not in str(error.value)
