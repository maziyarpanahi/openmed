"""Canonical capability descriptor: shape, digest and value-free content."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from openmed.core import capabilities
from openmed.core.capabilities import (
    CAPABILITY_DESCRIPTOR_SCHEMA,
    CAPABILITY_DESCRIPTOR_SCHEMA_VERSION,
    CAPABILITY_SURFACES,
    capability_descriptor,
    capability_descriptor_digest,
    load_capability_descriptor_schema,
    registered_backends,
    validate_capability_descriptor,
)
from openmed.core.policy import list_policies
from openmed.core.schemas.span import SCHEMA_NAMES, load_schema

ROOT = Path(__file__).resolve().parents[3]


def test_schema_is_draft_2020_12_and_closes_the_vocabulary() -> None:
    schema = load_capability_descriptor_schema()
    Draft202012Validator.check_schema(schema)
    properties = schema["properties"]

    assert properties["schema"]["const"] == CAPABILITY_DESCRIPTOR_SCHEMA
    assert properties["schema_version"]["const"] == (
        CAPABILITY_DESCRIPTOR_SCHEMA_VERSION
    )
    assert schema["schema_version"] == CAPABILITY_DESCRIPTOR_SCHEMA_VERSION
    backend_names = properties["backends"]["items"]["properties"]["name"]["enum"]
    surface_names = properties["surfaces"]["items"]["properties"]["name"]["enum"]
    assert tuple(backend_names) == registered_backends()
    assert tuple(surface_names) == CAPABILITY_SURFACES
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == sorted(properties)


def test_descriptor_validates_and_lists_every_registered_capability() -> None:
    descriptor = capability_descriptor()

    validate_capability_descriptor(descriptor)
    assert [item["name"] for item in descriptor["backends"]] == list(
        registered_backends()
    )
    assert [item["name"] for item in descriptor["surfaces"]] == list(
        CAPABILITY_SURFACES
    )
    surfaces = {item["name"]: item["available"] for item in descriptor["surfaces"]}
    assert surfaces["python"] is True
    assert surfaces["cli"] is True
    assert descriptor["served_models"] == []
    assert descriptor["schema_versions"] == [
        {"name": name, "version": load_schema(name)["schema_version"]}
        for name in sorted(SCHEMA_NAMES)
    ]
    assert descriptor["policy_profiles"] == sorted(list_policies())


def test_backend_availability_mirrors_the_importless_probe() -> None:
    descriptor = capability_descriptor()
    statuses = capabilities.available_backends()

    assert descriptor["backends"] == [
        {"name": name, "extra": status.extra, "available": status.available}
        for name, status in statuses.items()
    ]


def test_surface_availability_follows_installed_packages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = capabilities._unimportable

    def without_grpc(modules):
        return tuple(name for name in modules if name == "grpc") or real(modules)

    monkeypatch.setattr(capabilities, "_unimportable", without_grpc)
    surfaces = {
        item["name"]: item["available"] for item in capability_descriptor()["surfaces"]
    }

    assert surfaces["grpc"] is False
    assert surfaces["python"] is True


def test_digest_is_deterministic_and_covers_every_field() -> None:
    first = capability_descriptor(served_models=["b-model", "OpenMed/a-model"])
    second = capability_descriptor(served_models=["OpenMed/a-model", "b-model"])

    assert first == second
    assert first["digest"] == capability_descriptor_digest(first)
    assert first["digest"].startswith("sha256:") and len(first["digest"]) == 71
    body = {key: value for key, value in first.items() if key != "digest"}
    assert capability_descriptor_digest(body) == first["digest"]
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    assert first["digest"] == (
        "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )
    assert capability_descriptor()["digest"] != first["digest"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["served_models"].append("synthetic-extra-model"),
        lambda doc: doc["backends"][0].update(
            available=not doc["backends"][0]["available"]
        ),
        lambda doc: doc["policy_profiles"].pop(),
        lambda doc: doc["schema_versions"][0].update(version=99),
    ],
)
def test_tampered_descriptor_fails_the_digest_check(mutate) -> None:
    descriptor = capability_descriptor()
    tampered = copy.deepcopy(descriptor)
    mutate(tampered)

    with pytest.raises(ValueError, match="digest") as exc_info:
        validate_capability_descriptor(tampered)
    assert "synthetic-extra-model" not in str(exc_info.value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.update(hostname="synthetic-host.invalid"),
        lambda doc: doc["served_models"].append("/synthetic/private/model"),
        lambda doc: doc["surfaces"].append({"name": "soap", "available": True}),
        lambda doc: doc.update(schema="openmed.other"),
        lambda doc: doc.update(digest="sha256:short"),
    ],
)
def test_schema_rejects_values_outside_the_contract(mutate) -> None:
    descriptor = capability_descriptor()
    mutate(descriptor)

    with pytest.raises(ValueError, match="schema") as exc_info:
        validate_capability_descriptor(descriptor)
    assert "synthetic" not in str(exc_info.value)


def test_served_models_keep_public_ids_and_drop_paths() -> None:
    served = [
        "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        "disease_detection_superclinical",
        "disease_detection_superclinical",
        "/synthetic/private/checkpoint",
        "~/synthetic-home/model",
        "./synthetic-relative",
        "../synthetic-parent/model",
        "synthetic-org/../escape",
        "a/b/c",
        "C:\\synthetic\\model",
        "file:///synthetic/model",
        "https://synthetic.invalid/model",
        "",
        "x" * 129,
    ]
    descriptor = capability_descriptor(served_models=served)

    assert descriptor["served_models"] == [
        "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
        "disease_detection_superclinical",
    ]
    text = json.dumps(descriptor)
    assert "synthetic" not in text
    validate_capability_descriptor(descriptor)


def test_planted_environment_values_never_reach_the_descriptor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel = "synthetic-capability-environment-sentinel"
    monkeypatch.setenv("HF_TOKEN", sentinel)
    monkeypatch.setenv("HF_HOME", str(tmp_path / sentinel))
    monkeypatch.setenv("OPENMED_CONFIG", str(tmp_path / sentinel / "config.toml"))

    text = json.dumps(capability_descriptor())

    assert sentinel not in text
    assert str(tmp_path) not in text
    assert str(Path.home()) not in text


def test_building_the_descriptor_imports_no_optional_package() -> None:
    probe = (
        "import sys\n"
        "from openmed.core.capabilities import capability_descriptor\n"
        "capability_descriptor()\n"
        "names = ('fastapi', 'uvicorn', 'strawberry', 'grpc', 'mcp',"
        " 'transformers', 'torch', 'mlx', 'google.protobuf')\n"
        "print(','.join(name for name in names if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == ""
