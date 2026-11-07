"""Production configuration can never load or arm the fault harness."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from openmed.core.config import PROFILE_ENV_VAR
from tests.faults.guard import (
    ARM_FORBIDDEN_CODE,
    IMPORT_FORBIDDEN_CODE,
    FaultInjectionForbiddenError,
    refuse_production_import,
    require_arm_permitted,
    resolved_profile,
)

ROOT = Path(__file__).resolve().parents[2]
HARNESS = Path(__file__).resolve().parent
ACTIVATION_TOKENS = re.compile(
    r"(?i)tests\.faults|OPENMED_FAULT|service_fault_(?:plan|manifest)|fault_plan"
)


def _shipped_profiles() -> dict[str, str]:
    profiles: dict[str, str] = {}
    for name in (
        "Dockerfile",
        "Dockerfile.mcp",
        "deploy/docker/Dockerfile",
        "deploy/docker/Dockerfile.distroless",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        match = re.search(rf"{PROFILE_ENV_VAR}=([A-Za-z0-9_]+)", text)
        assert match is not None, name
        profiles[name] = match.group(1)
    for name in ("docker-compose.yml", "deploy/openmed-compose.yaml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        defaults = re.findall(rf"\$\{{{PROFILE_ENV_VAR}:-([A-Za-z0-9_]+)\}}", text)
        assert defaults, name
        for index, value in enumerate(defaults):
            profiles[f"{name}#{index}"] = value
    values = yaml.safe_load(
        (ROOT / "deploy/helm/openmed-service/values.yaml").read_text(encoding="utf-8")
    )
    profiles["helm/values.yaml"] = values["config"]["profile"]
    return profiles


def test_arming_requires_the_explicit_test_profile(subtests) -> None:
    for value in (None, "", "  ", "prod", " PROD ", "dev", "fast", "low_resource"):
        with subtests.test(profile=value):
            environ = {} if value is None else {PROFILE_ENV_VAR: value}
            with pytest.raises(FaultInjectionForbiddenError) as error:
                require_arm_permitted(environ)
            assert error.value.code == ARM_FORBIDDEN_CODE
            assert str(error.value) == ARM_FORBIDDEN_CODE
    for value in ("test", " Test "):
        with subtests.test(profile=value):
            require_arm_permitted({PROFILE_ENV_VAR: value})
    assert resolved_profile({}) == "prod"


def test_import_guard_refuses_an_explicit_production_profile(subtests) -> None:
    for value in ("prod", " Prod "):
        with subtests.test(profile=value):
            with pytest.raises(FaultInjectionForbiddenError) as error:
                refuse_production_import({PROFILE_ENV_VAR: value})
            assert error.value.code == IMPORT_FORBIDDEN_CODE
    for environ in ({}, {PROFILE_ENV_VAR: "test"}):
        with subtests.test(environ=sorted(environ)):
            refuse_production_import(environ)


def test_shipped_deployment_profiles_cannot_arm_or_load_faults(subtests) -> None:
    profiles = _shipped_profiles()
    assert len(profiles) >= 7
    for source, profile in profiles.items():
        with subtests.test(source=source):
            environ = {PROFILE_ENV_VAR: profile}
            with pytest.raises(FaultInjectionForbiddenError):
                require_arm_permitted(environ)
            with pytest.raises(FaultInjectionForbiddenError):
                refuse_production_import(environ)


def test_built_distributions_never_include_the_harness() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    hatch = config["tool"]["hatch"]["build"]
    includes = hatch["include"]
    force_include = hatch["targets"]["wheel"].get("force-include", {})

    assert includes and all(
        not entry.lstrip("/").startswith("tests") for entry in includes
    )
    assert all("tests" not in Path(source).parts for source in force_include)
    assert all("tests" not in Path(target).parts for target in force_include.values())
    assert HARNESS.relative_to(ROOT).parts[0] == "tests"


def test_service_surfaces_and_config_expose_no_fault_switch() -> None:
    surfaces = [
        "openmed/service/app.py",
        "openmed/service/graphql_app.py",
        "openmed/service/grpc_server.py",
        "openmed/service/runtime.py",
        "openmed/service/jobs.py",
        "openmed/core/config.py",
        "openmed/core/config.schema.json",
        "openmed/cli/main.py",
    ]
    for name in surfaces:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert ACTIVATION_TOKENS.search(text) is None, name


def test_harness_reads_no_environment_switch_except_the_profile() -> None:
    names: set[str] = set()
    for path in sorted(HARNESS.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args:
                function = node.func
                argument = node.args[0]
                is_getenv = isinstance(function, ast.Attribute) and (
                    function.attr == "getenv"
                    or (
                        function.attr in {"get", "pop", "setdefault"}
                        and isinstance(function.value, ast.Attribute)
                        and function.value.attr == "environ"
                    )
                )
                if is_getenv and isinstance(argument, ast.Constant):
                    names.add(str(argument.value))
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
            ):
                names.add(str(node.slice.value))
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not node.value.startswith("OPENMED_FAULT"), path.name
    assert names <= {PROFILE_ENV_VAR}


@pytest.mark.slow
def test_openmed_package_never_imports_or_names_the_harness() -> None:
    offenders: list[str] = []
    for path in sorted((ROOT / "openmed").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if ACTIVATION_TOKENS.search(node.value):
                    offenders.append(str(path.relative_to(ROOT)))
            if any(
                module == "tests" or module.startswith("tests.") for module in modules
            ):
                offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


@pytest.mark.slow
def test_production_process_cannot_import_the_harness(subtests) -> None:
    for profile, allowed in (("prod", False), ("test", True)):
        with subtests.test(profile=profile):
            environment = {**os.environ, PROFILE_ENV_VAR: profile}
            completed = subprocess.run(
                [sys.executable, "-c", "import tests.faults"],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if allowed:
                assert completed.returncode == 0, completed.stderr[-400:]
            else:
                assert completed.returncode != 0
                assert IMPORT_FORBIDDEN_CODE in completed.stderr
