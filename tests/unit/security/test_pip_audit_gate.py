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


def _dependency_extras(lock: dict, dependency: str) -> set[str]:
    """Follow selected extra edges, including every locked platform variant."""
    packages: dict[str, list[dict]] = {}
    for package in lock["package"]:
        packages.setdefault(package["name"], []).append(package)
    root = packages["openmed"][0]
    dependency_extras = set()
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
            if name == dependency:
                dependency_extras.add(extra)
                break
            for package in packages.get(name, []):
                pending.extend(package.get("dependencies", []))
                optional = package.get("optional-dependencies", {})
                for selected in selected_extras:
                    pending.extend(optional.get(selected, []))
    return dependency_extras


def _torch_extras(lock: dict) -> set[str]:
    return _dependency_extras(lock, "torch")


def test_optional_fsspec_dependency_closure_has_safe_floor() -> None:
    project, lock = _model_dependency_metadata()
    extras = _dependency_extras(lock, "fsspec")
    assert {"dev", "cloud", "journey", "edge-sbc", "onnx-runtime"} <= extras
    for extra in extras:
        floors = [
            Requirement(value)
            for value in project["project"]["optional-dependencies"][extra]
            if Requirement(value).name == "fsspec"
        ]
        assert len(floors) == 1, extra
        assert "2026.4.0" not in floors[0].specifier, extra
        assert "2026.6.0" in floors[0].specifier, extra
        if extra == "awq":
            assert floors[0].marker is not None
            assert floors[0].marker.evaluate({"sys_platform": "linux"})
            assert not floors[0].marker.evaluate({"sys_platform": "darwin"})
            assert not floors[0].marker.evaluate({"sys_platform": "win32"})


def test_fsspec_profiles_require_patched_jinja_sandbox() -> None:
    project, lock = _model_dependency_metadata()
    for extra in _dependency_extras(lock, "fsspec"):
        floors = {
            requirement.name: requirement
            for value in project["project"]["optional-dependencies"][extra]
            for requirement in [Requirement(value)]
        }
        jinja = floors["jinja2"]
        assert "3.1.5" not in jinja.specifier, extra
        assert "3.1.6" in jinja.specifier, extra
        assert jinja.marker == floors["fsspec"].marker, extra
    resolved = [item for item in lock["package"] if item["name"] == "jinja2"]
    assert resolved
    assert all(Version(item["version"]) >= Version("3.1.6") for item in resolved)
    constraints = {
        requirement.name: requirement
        for value in project["tool"]["uv"]["constraint-dependencies"]
        for requirement in [Requirement(value)]
    }
    assert "3.1.5" not in constraints["jinja2"].specifier
    assert "3.1.6" in constraints["jinja2"].specifier
    assert not {"fsspec", "jinja2"} & {
        Requirement(value).name for value in project["project"]["dependencies"]
    }


@pytest.mark.parametrize("sink", ["refs", "templates", "gen"])
@pytest.mark.parametrize(
    "expression",
    [
        "range.__class__.__mro__",
        "''.__class__.__mro__",
        '("{0.__class__.__mro__}"|attr("format"))("")',
    ],
)
def test_reference_templates_reject_private_attribute_access(
    sink: str, expression: str
) -> None:
    import fsspec
    from jinja2.exceptions import SecurityError

    template = "{{ " + expression + " }}"
    spec: dict = {"version": 1, "refs": {}}
    if sink == "refs":
        spec["templates"] = {"root": "memory://"}
        spec["refs"] = {"chunk": [template]}
    elif sink == "templates":
        spec["templates"] = {"target": template}
        spec["refs"] = {"chunk": ["{{ target }}"]}
    else:
        spec["gen"] = [
            {
                "key": template,
                "url": "memory://chunk/{{ i }}",
                "offset": "0",
                "length": "1",
                "dimensions": {"i": [0]},
            }
        ]
    with pytest.raises(SecurityError):
        fsspec.filesystem("reference", fo=spec, simple_templates=False)


def test_reference_generator_retains_safe_variable_interpolation() -> None:
    import fsspec

    spec = {
        "version": 1,
        "refs": {},
        "gen": [
            {
                "key": "chunk/{{ i }}",
                "url": "memory://chunk/{{ i }}",
                "offset": "0",
                "length": "1",
                "dimensions": {"i": [0, 1]},
            }
        ],
    }
    fs = fsspec.filesystem("reference", fo=spec, simple_templates=False)
    assert fs.references == {
        "chunk/0": ["memory://chunk/0", 0, 1],
        "chunk/1": ["memory://chunk/1", 0, 1],
    }


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


@pytest.mark.parametrize(
    ("package", "unsafe", "fixed", "extras"),
    [
        ("banks", "2.4.4", "2.4.5", ("agents", "llamaindex")),
        ("datasets", "5.0.0", "5.0.1", ("awq", "gptq")),
        ("fsspec", "2026.4.0", "2026.6.0", ("dev", "cloud", "journey")),
        ("h2", "4.3.0", "4.4.1", ("beam", "prefect")),
        ("oauthlib", "3.3.1", "4.0.0", ("cloud", "prefect")),
    ],
)
def test_fixable_optional_dependency_floors_are_published_and_locked(
    package: str, unsafe: str, fixed: str, extras: tuple[str, ...]
) -> None:
    project, lock = _model_dependency_metadata()
    resolved = [item for item in lock["package"] if item["name"] == package]
    assert resolved
    assert all(Version(item["version"]) >= Version(fixed) for item in resolved)
    for extra in extras:
        requirements = [
            Requirement(value)
            for value in project["project"]["optional-dependencies"][extra]
        ]
        floor = next(item for item in requirements if item.name == package)
        assert unsafe not in floor.specifier, (package, extra)
        assert fixed in floor.specifier, (package, extra)
    constraints = [
        Requirement(value) for value in project["tool"]["uv"]["constraint-dependencies"]
    ]
    floor = next(item for item in constraints if item.name == package)
    assert unsafe not in floor.specifier
    assert fixed in floor.specifier
    assert package not in {
        Requirement(value).name for value in project["project"]["dependencies"]
    }


def test_awq_dataset_floor_retains_linux_only_installation() -> None:
    project, _ = _model_dependency_metadata()
    floor = next(
        Requirement(value)
        for value in project["project"]["optional-dependencies"]["awq"]
        if value.startswith("datasets")
    )
    assert floor.marker is not None
    assert floor.marker.evaluate({"sys_platform": "linux"})
    assert not floor.marker.evaluate({"sys_platform": "darwin"})
    assert not floor.marker.evaluate({"sys_platform": "win32"})
