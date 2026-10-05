from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

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


def test_fixable_vulnerabilities_are_not_suppressed_by_ignores() -> None:
    report = {
        "dependencies": [
            {
                "name": "fixed-package",
                "vulns": [{"id": "PYSEC-2026-1", "fix_versions": ["1.2.3"]}],
            },
        ],
    }

    assert pip_audit_gate.fixable_vulnerabilities(report) == [
        ("fixed-package", "PYSEC-2026-1", ["1.2.3"]),
    ]


def test_unfixable_vulnerabilities_honor_active_ignores() -> None:
    report = {
        "dependencies": [
            {
                "name": "reviewed-package",
                "vulns": [{"id": "PYSEC-2026-2", "fix_versions": []}],
            },
            {
                "name": "unreviewed-package",
                "vulns": [{"id": "PYSEC-2026-3", "fix_versions": []}],
            },
        ],
    }

    assert pip_audit_gate.unignored_unfixable_vulnerabilities(
        report,
        ignored_ids={"PYSEC-2026-2"},
    ) == [("unreviewed-package", "PYSEC-2026-3")]


def test_gate_fails_fixable_vulnerability_even_when_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "dependencies": [
            {
                "name": "fixed-package",
                "vulns": [{"id": "PYSEC-2026-1", "fix_versions": ["1.2.3"]}],
            },
        ],
    }
    monkeypatch.setattr(
        pip_audit_gate,
        "parse_args",
        lambda: argparse.Namespace(
            ignore_file=tmp_path / "ignore.toml",
            report=tmp_path / "report.json",
        ),
    )
    monkeypatch.setattr(pip_audit_gate, "load_ignores", lambda *_: {"PYSEC-2026-1"})
    monkeypatch.setattr(pip_audit_gate, "run_pip_audit", lambda *_: report)

    assert pip_audit_gate.main() == 1


def test_gate_passes_reviewed_unfixable_vulnerability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = {
        "dependencies": [
            {
                "name": "unfixed-package",
                "vulns": [{"id": "PYSEC-2026-2", "fix_versions": []}],
            },
        ],
    }
    monkeypatch.setattr(
        pip_audit_gate,
        "parse_args",
        lambda: argparse.Namespace(
            ignore_file=tmp_path / "ignore.toml",
            report=tmp_path / "report.json",
        ),
    )
    monkeypatch.setattr(pip_audit_gate, "load_ignores", lambda *_: {"PYSEC-2026-2"})
    monkeypatch.setattr(pip_audit_gate, "run_pip_audit", lambda *_: report)

    assert pip_audit_gate.main() == 0


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


def _model_dependency_metadata() -> tuple[dict, dict]:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)
    with (ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    return project, lock


def _torch_extras(lock: dict) -> set[str]:
    """Follow selected extra edges, including every locked platform variant."""
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    root = packages["openmed"][0]
    torch_extras = set()
    for extra, requirements in root["optional-dependencies"].items():
        pending = list(requirements)
        seen: set[tuple[str, tuple[str, ...]]] = set()
        while pending:
            requirement = pending.pop()
            name = requirement["name"]
            selected_extras = tuple(sorted(requirement.get("extra", [])))
            key = (name, selected_extras)
            if key in seen:
                continue
            seen.add(key)
            if name == "torch":
                torch_extras.add(extra)
                break
            for package in packages.get(name, []):
                pending.extend(package.get("dependencies", []))
                optional = package.get("optional-dependencies", {})
                for selected in selected_extras:
                    pending.extend(optional.get(selected, []))
    return torch_extras


def test_optional_torch_dependency_closure_has_safe_floor() -> None:
    project, lock = _model_dependency_metadata()
    extras = _torch_extras(lock)
    assert {
        "awq",
        "coreml",
        "gliner",
        "gptq",
        "hf",
        "multimodal",
        "onnx",
        "zh-hanlp",
    } <= extras
    for extra in extras:
        requirements = [
            Requirement(value)
            for value in project["project"]["optional-dependencies"][extra]
        ]
        torch = [item for item in requirements if item.name == "torch"]
        assert len(torch) == 1, extra
        for unsafe_version in ("2.0.0", "2.9.1", "2.10.0", "2.12.1"):
            assert unsafe_version not in torch[0].specifier, (extra, unsafe_version)
        assert "2.13.0" in torch[0].specifier, extra


def test_frozen_torch_resolution_and_core_boundaries() -> None:
    project, lock = _model_dependency_metadata()
    resolved = [item for item in lock["package"] if item["name"] == "torch"]
    assert resolved
    assert all(Version(item["version"]) >= Version("2.13.0") for item in resolved)
    assert project["project"]["requires-python"] == ">=3.10"
    assert all(
        Requirement(value).name != "torch"
        for value in project["project"]["dependencies"]
    )
    assert not {"dev", "cli", "mcp", "edge-sbc", "onnx-runtime"} & _torch_extras(lock)
    constraints = [
        Requirement(value) for value in project["tool"]["uv"]["constraint-dependencies"]
    ]
    torch = next(item for item in constraints if item.name == "torch")
    assert "2.12.1" not in torch.specifier
    assert "2.13.0" in torch.specifier


@pytest.mark.parametrize(
    "path",
    [
        "deploy/docker/Dockerfile",
        "examples/custom_tokenizer/requirements.txt",
        "openmed/interop/capabilities.py",
        "packaging/standalone_manifest.py",
        "scripts/release/smoke_test.py",
    ],
)
def test_current_optional_runtime_recipes_use_safe_torch_floor(path: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    assert "torch>=2.13.0" in source


def test_awq_torch_floor_keeps_the_existing_linux_only_boundary() -> None:
    project, _ = _model_dependency_metadata()
    requirements = project["project"]["optional-dependencies"]["awq"]
    torch = next(
        Requirement(value) for value in requirements if value.startswith("torch")
    )
    assert torch.marker is not None
    assert torch.marker.evaluate({"sys_platform": "linux"})
    assert not torch.marker.evaluate({"sys_platform": "darwin"})
    assert not torch.marker.evaluate({"sys_platform": "win32"})
