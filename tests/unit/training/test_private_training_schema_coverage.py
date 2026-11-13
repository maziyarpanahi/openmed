"""Every federated payload version needs an offline catalog schema."""

import importlib
import pkgutil

import pytest

import openmed.training
from openmed.training.private_training_schemas import build_private_training_schemas

# Keys are fully qualified constant names; values explain intentional exclusions.
SCHEMA_EXEMPTIONS: dict[str, str] = {}


def _federated_versions() -> dict[str, str]:
    modules = []
    for info in pkgutil.iter_modules(
        openmed.training.__path__, prefix="openmed.training."
    ):
        if not info.name.startswith("openmed.training.federated"):
            continue
        module = importlib.import_module(info.name)
        modules.append(module)
        if info.ispkg:
            modules.extend(
                importlib.import_module(child.name)
                for child in pkgutil.walk_packages(
                    module.__path__, prefix=module.__name__ + "."
                )
            )
    return {
        f"{module.__name__}.{name}": value
        for module in modules
        for name, value in vars(module).items()
        if name.endswith("_SCHEMA_VERSION")
        and isinstance(value, str)
        and value.startswith("openmed.training.federated")
    }


def _check_schema_coverage() -> None:
    versions = _federated_versions()
    covered = {
        schema["properties"]["schema_version"]["const"]
        for schema in build_private_training_schemas().values()
    }
    missing = {name for name, value in versions.items() if value not in covered}
    stale = set(SCHEMA_EXEMPTIONS) - missing
    assert not stale, f"Stale schema exemptions: {sorted(stale)}"
    assert all(reason.strip() for reason in SCHEMA_EXEMPTIONS.values())
    uncovered = missing - SCHEMA_EXEMPTIONS.keys()
    assert not uncovered, f"Missing catalog schemas: {sorted(uncovered)}"


def test_every_federated_payload_version_has_a_catalog_schema():
    _check_schema_coverage()


def test_new_module_without_schema_is_detected(tmp_path, monkeypatch):
    (tmp_path / "federated_synthetic.py").write_text(
        'SYNTHETIC_SCHEMA_VERSION = "openmed.training.federated_synthetic.v1"\n'
    )
    monkeypatch.setattr(
        openmed.training, "__path__", [*openmed.training.__path__, str(tmp_path)]
    )
    importlib.invalidate_caches()
    with pytest.raises(
        AssertionError, match="federated_synthetic.SYNTHETIC_SCHEMA_VERSION"
    ):
        _check_schema_coverage()


def test_stale_exemption_fails(monkeypatch):
    monkeypatch.setitem(
        SCHEMA_EXEMPTIONS,
        "openmed.training.federated_round.FEDERATED_ROUND_SCHEMA_VERSION",
        "Synthetic stale exemption for an already covered version.",
    )
    with pytest.raises(AssertionError, match="Stale schema exemptions"):
        _check_schema_coverage()
