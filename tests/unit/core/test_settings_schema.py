"""Tests for the sectioned settings schema and its value-free registry."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from openmed.core import settings_registry
from openmed.core.config import OpenMedConfig, config_schema_path
from openmed.core.schemas import (
    SCHEMA_NAMES,
    build_schema_snapshot,
    load_schema,
    load_schema_snapshot,
)
from openmed.core.settings_registry import (
    ENV_FORMATS,
    SECTION_CORE,
    SECTION_SERVICE,
    SettingsSchemaError,
    declared_setting_names,
    excluded_identifiers,
    get_setting_declaration,
    load_settings_schema,
    parse_excluded_identifiers,
    parse_settings_schema,
    secret_setting_names,
    setting_declarations,
    settings_schema_path,
    settings_sections,
)
from openmed.service.secret_files import CREDENTIAL_SETTINGS
from tests.fixtures.settings_env_scan import (
    PACKAGE_ROOT,
    REPO_ROOT,
    literal_env_names,
    scan_env_reads,
)

SERVICE_ROOT = PACKAGE_ROOT / "service"
SERVICE_PREFIX = "OPENMED_SERVICE_"
SENTINEL = "synthetic-settings-sentinel-7f3a"

# Names that look like credentials but select a mode, with the reason.
NOT_SECRET_DESPITE_NAME = {
    "OPENMED_SERVICE_AUTH_FAILURE_RATE_LIMIT_KEY": (
        "selects how failed-authentication buckets are keyed"
    ),
    "OPENMED_SERVICE_THROTTLE_KEY": "selects how throttle buckets are keyed",
    "OPENMED_SERVICE_DIGEST_KEY_ID": "public identifier recorded with digests",
    "OPENMED_SERVICE_MLX_PAGED_KV_CACHE_BYTES_PER_TOKEN": "a byte count per token",
}
_CREDENTIAL_NAME_RE = re.compile(r"(TOKEN|PASSWORD|SECRET|API_KEYS|_KEY|_DSN)$")


@pytest.fixture(scope="module")
def service_scan():
    return scan_env_reads([SERVICE_ROOT])


def _walk_entries() -> Iterator[tuple[str, str, Mapping[str, Any]]]:
    schema = load_settings_schema()
    for section, body in schema["properties"].items():
        if section == "schema_version":
            continue
        for name, entry in body["properties"].items():
            yield section, name, entry


# -- schema document ----------------------------------------------------------


def test_schema_is_valid_draft_2020_12_and_versioned() -> None:
    schema = load_settings_schema()
    Draft202012Validator.check_schema(schema)

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["schema_version"] == settings_registry.SETTINGS_SCHEMA_VERSION
    assert schema["properties"]["schema_version"]["const"] == 1
    assert [section.name for section in settings_sections()][:2] == [
        SECTION_CORE,
        SECTION_SERVICE,
    ]


def test_schema_is_covered_by_the_fingerprint_snapshot() -> None:
    assert "settings" in SCHEMA_NAMES
    assert load_schema("settings") == load_settings_schema()
    snapshot = load_schema_snapshot()
    assert (
        snapshot["settings"]
        == build_schema_snapshot({"settings": load_schema("settings")})["settings"]
    )


def test_schema_file_repeats_no_key() -> None:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys)), keys
        return dict(pairs)

    json.loads(settings_schema_path().read_text("utf-8"), object_pairs_hook=reject)


def test_registry_rejects_a_document_that_repeats_a_key(tmp_path, monkeypatch):
    duplicated = (
        settings_schema_path()
        .read_text("utf-8")
        .replace(
            '"OPENMED_SERVICE_AUTH_ENABLED": {',
            '"OPENMED_SERVICE_AUTH_ENABLED": {"type": "boolean"},\n'
            '      "OPENMED_SERVICE_AUTH_ENABLED": {',
            1,
        )
    )
    path = tmp_path / "settings.schema.json"
    path.write_text(duplicated, encoding="utf-8")
    with pytest.raises(SettingsSchemaError, match="OPENMED_SERVICE_AUTH_ENABLED"):
        settings_registry._read_json(path)


def test_instance_with_every_default_validates_against_the_schema() -> None:
    schema = load_settings_schema()
    config = json.loads(config_schema_path().read_text("utf-8"))
    registry = Registry().with_resources(
        [
            (config["$id"], Resource.from_contents(config)),
            (schema["$id"], Resource.from_contents(schema)),
        ]
    )
    instance: dict[str, Any] = {"schema_version": 1}
    for declaration in setting_declarations():
        if declaration.has_default:
            section = instance.setdefault(declaration.section, {})
            section[declaration.name] = declaration.to_dict()["default"]
    validator = Draft202012Validator(schema, registry=registry)
    assert list(validator.iter_errors(instance)) == []

    instance["service"]["OPENMED_SERVICE_BATCH_MAX_SIZE"] = 0
    errors = list(validator.iter_errors(instance))
    assert errors and errors[0].path[-1] == "OPENMED_SERVICE_BATCH_MAX_SIZE"


# -- service coverage -----------------------------------------------------------


def test_every_service_read_is_declared_exactly_once(service_scan) -> None:
    read = service_scan.names(SERVICE_PREFIX)
    declared = declared_setting_names(SECTION_SERVICE)

    undeclared = {name: service_scan.locations(name) for name in read - declared}
    assert undeclared == {}
    assert declared - read == set(), "declared service settings that are never read"
    assert all(name.startswith(SERVICE_PREFIX) for name in declared)
    counts: dict[str, int] = {}
    for section in settings_sections():
        for name in section.names():
            counts[name] = counts.get(name, 0) + 1
    assert {name for name, count in counts.items() if count != 1} == set()


def test_every_service_literal_is_declared() -> None:
    # Broad cross-check: a read the AST resolver missed still shows up here.
    literals = literal_env_names([SERVICE_ROOT], SERVICE_PREFIX)
    declared = declared_setting_names(SECTION_SERVICE)
    assert {
        name: where for name, where in literals.items() if name not in declared
    } == {}


def test_service_scan_has_no_dynamic_names(service_scan) -> None:
    assert service_scan.dynamic == []


def test_core_section_covers_every_config_environment_read() -> None:
    core = declared_setting_names(SECTION_CORE)
    scan = scan_env_reads([PACKAGE_ROOT / "core" / "config.py"])
    assert scan.names() - core == set()
    for declaration in setting_declarations(SECTION_CORE):
        if declaration.name.startswith("OPENMED_"):
            owner = REPO_ROOT / Path(*declaration.module.split(".")).with_suffix(".py")
            reads = scan_env_reads([owner]).names()
            assert declaration.name in reads, declaration.name


def test_core_declarations_reference_config_schema_and_match_defaults() -> None:
    config = json.loads(config_schema_path().read_text("utf-8"))
    defaults = _config_defaults()
    for _section, name, entry in _walk_entries():
        if _section != SECTION_CORE or "$ref" not in entry:
            continue
        key = entry["x-config-key"]
        assert entry["$ref"] == f"config.schema.json#/properties/{key}"
        assert key in config["properties"]
        declaration = get_setting_declaration(name)
        if declaration.has_default:
            assert declaration.to_dict()["default"] == defaults[key], name


def _config_defaults() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config_field in fields(OpenMedConfig):
        result[config_field.name] = copy.deepcopy(config_field.default)
    return result


# -- declaration contract ---------------------------------------------------------


def test_every_declaration_names_type_module_and_secret_flag() -> None:
    for declaration in setting_declarations():
        assert declaration.description
        assert declaration.types
        assert isinstance(declaration.secret, bool)
        module = REPO_ROOT / Path(*declaration.module.split("."))
        assert module.with_suffix(".py").is_file() or (module / "__init__.py").is_file()
        assert declaration.env_format is None or declaration.env_format in ENV_FORMATS


def test_every_default_satisfies_its_own_type_and_bounds() -> None:
    for declaration in setting_declarations():
        if not declaration.has_default:
            continue
        validator = Draft202012Validator(declaration.json_schema())
        errors = list(validator.iter_errors(declaration.to_dict()["default"]))
        assert errors == [], declaration.name


def test_secrets_are_marked_and_declare_no_default() -> None:
    secrets = secret_setting_names()
    declared = declared_setting_names()
    for name in CREDENTIAL_SETTINGS:
        if name in declared:
            assert name in secrets, name
    for name in declared:
        if _CREDENTIAL_NAME_RE.search(name) and name not in NOT_SECRET_DESPITE_NAME:
            assert name in secrets, name
    for name in NOT_SECRET_DESPITE_NAME:
        assert name not in secrets
    assert "HF_TOKEN" in secrets
    for _section, name, entry in _walk_entries():
        if entry.get("x-secret"):
            assert "default" not in entry, name
            assert not get_setting_declaration(name).has_default


def test_secret_file_companions_point_at_declared_secrets() -> None:
    secrets = secret_setting_names()
    companions = {
        declaration.name: declaration.secret_file_for
        for declaration in setting_declarations()
        if declaration.secret_file_for is not None
    }
    assert companions
    for name, target in companions.items():
        assert name == f"{target}_FILE"
        assert target in secrets
        assert name not in secrets
        assert get_setting_declaration(name).env_format == "path"


def test_registry_rejects_a_secret_with_a_default() -> None:
    document = load_settings_schema()
    entry = document["properties"]["service"]["properties"][
        "OPENMED_SERVICE_DIGEST_KEY"
    ]
    entry["default"] = SENTINEL
    with pytest.raises(SettingsSchemaError) as exc_info:
        parse_settings_schema(document)
    assert "OPENMED_SERVICE_DIGEST_KEY" in str(exc_info.value)
    assert SENTINEL not in str(exc_info.value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda entry: entry.pop("x-module"), "owning module"),
        (lambda entry: entry.pop("x-secret"), "x-secret"),
        (lambda entry: entry.update(type="text"), "JSON type"),
        (lambda entry: entry.update({"x-env-format": "yaml"}), "x-env-format"),
        (lambda entry: entry.update(description=""), "description"),
    ],
)
def test_registry_rejects_malformed_declarations(mutation, message) -> None:
    document = load_settings_schema()
    mutation(
        document["properties"]["service"]["properties"]["OPENMED_SERVICE_LOG_LEVEL"]
    )
    with pytest.raises(SettingsSchemaError, match=message):
        parse_settings_schema(document)


def test_registry_rejects_a_name_declared_in_two_sections() -> None:
    document = load_settings_schema()
    service = document["properties"]["service"]["properties"]
    document["properties"]["core"]["properties"]["OPENMED_SERVICE_LOG_LEVEL"] = (
        copy.deepcopy(service["OPENMED_SERVICE_LOG_LEVEL"])
    )
    with pytest.raises(SettingsSchemaError, match="declared in both"):
        parse_settings_schema(document)


def test_registry_rejects_an_unknown_config_reference() -> None:
    document = load_settings_schema()
    core = document["properties"]["core"]["properties"]
    core["OPENMED_USE_MEDICAL_TOKENIZER"]["$ref"] = (
        "config.schema.json#/properties/not_a_field"
    )
    with pytest.raises(SettingsSchemaError, match="unknown config property"):
        parse_settings_schema(document)


# -- value-free registry -----------------------------------------------------------


class _ExplodingEnviron(dict):
    """Environment stand-in that fails on any read."""

    def _explode(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the settings registry read the environment")

    __getitem__ = get = __contains__ = __iter__ = keys = items = _explode


def _clear_registry_caches() -> None:
    for name in ("_cached_document", "_cached_sections", "_cached_index"):
        getattr(settings_registry, name).cache_clear()


def test_registry_never_reads_the_environment() -> None:
    _clear_registry_caches()
    # Restore the real environment before fixture teardown needs it.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "environ", _ExplodingEnviron())
        patch.setattr(os, "getenv", _ExplodingEnviron()._explode)
        count = len(setting_declarations())
        declared = declared_setting_names()
        secret = get_setting_declaration("OPENMED_SERVICE_AUTH_API_KEYS").secret

    assert count == len(declared)
    assert secret is True
    assert scan_env_reads([Path(settings_registry.__file__)]).reads == []


def test_declarations_carry_no_configured_values() -> None:
    with pytest.MonkeyPatch.context() as patch:
        for name in sorted(declared_setting_names()):
            patch.setenv(name, SENTINEL)
        _clear_registry_caches()
        rendered = json.dumps(
            [declaration.to_dict() for declaration in setting_declarations()]
        )
        sections = repr(settings_sections())

    assert SENTINEL not in rendered
    assert SENTINEL not in sections


def test_declarations_are_immutable() -> None:
    declaration = get_setting_declaration("OPENMED_SERVICE_TRUSTED_HOSTS")
    assert declaration.default == ("localhost", "127.0.0.1", "[::1]")
    with pytest.raises(TypeError):
        declaration.bounds["minimum"] = 1  # type: ignore[index]
    exported = declaration.to_dict()
    exported["default"].append("mutated")
    assert get_setting_declaration("OPENMED_SERVICE_TRUSTED_HOSTS").to_dict()[
        "default"
    ] == ["localhost", "127.0.0.1", "[::1]"]


def test_unknown_names_and_sections_raise_key_error() -> None:
    with pytest.raises(KeyError):
        get_setting_declaration("OPENMED_SERVICE_NOT_DECLARED")
    with pytest.raises(KeyError):
        setting_declarations("not-a-section")


def test_registry_import_does_not_load_service_frameworks() -> None:
    code = (
        "import sys\n"
        "import openmed.core.settings_registry as registry\n"
        "registry.setting_declarations()\n"
        "heavy = ('fastapi', 'starlette', 'strawberry', 'grpc')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))
            ),
        },
    )
    assert result.stdout.strip() == ""


# -- the scanner itself -------------------------------------------------------------


def test_scanner_finds_direct_constant_wrapper_and_table_reads(tmp_path) -> None:
    package = tmp_path / "openmed"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "names.py").write_text(
        'IMPORTED_ENV_VAR = "OPENMED_SERVICE_SYNTHETIC_IMPORTED"\n',
        encoding="utf-8",
    )
    (package / "synthetic.py").write_text(
        "import os\n"
        "from openmed.names import IMPORTED_ENV_VAR\n"
        'LOCAL_ENV_VAR = "OPENMED_SERVICE_SYNTHETIC_CONSTANT"\n'
        "TABLE = {'a': 'OPENMED_SERVICE_SYNTHETIC_TABLE'}\n"
        "def _env_int(environ, name, default):\n"
        "    raw = environ.get(name)\n"
        "    return default if raw is None else int(raw)\n"
        "def load(environ=os.environ):\n"
        '    direct = os.getenv("OPENMED_SERVICE_SYNTHETIC_DIRECT")\n'
        "    constant = os.environ.get(LOCAL_ENV_VAR)\n"
        "    imported = environ.get(IMPORTED_ENV_VAR)\n"
        '    wrapped = _env_int(environ, "OPENMED_SERVICE_SYNTHETIC_WRAPPED", 1)\n'
        "    tables = [environ.get(env) for _, env in TABLE.items()]\n"
        '    dynamic = os.getenv(f"OPENMED_SERVICE_{direct}")\n'
        "    return direct, constant, imported, wrapped, tables, dynamic\n",
        encoding="utf-8",
    )

    scan = scan_env_reads([package], root=tmp_path)

    assert scan.names() == {
        "OPENMED_SERVICE_SYNTHETIC_CONSTANT",
        "OPENMED_SERVICE_SYNTHETIC_DIRECT",
        "OPENMED_SERVICE_SYNTHETIC_IMPORTED",
        "OPENMED_SERVICE_SYNTHETIC_TABLE",
        "OPENMED_SERVICE_SYNTHETIC_WRAPPED",
    }
    assert scan.locations("OPENMED_SERVICE_SYNTHETIC_DIRECT") == [
        "openmed/synthetic.py:9"
    ]
    assert [item.location for item in scan.dynamic] == ["openmed/synthetic.py:14"]


# -- non-service sections ---------------------------------------------------------

NON_SERVICE_SECTIONS = (
    "mcp",
    "openhim",
    "journey",
    "eval_datasets",
    "integrations",
    "runtime",
    "release",
)
# Paths whose environment reads each owned section must declare, keyed by the
# section they belong to and the name prefix that marks them.
SECTION_SOURCES = {
    "mcp": (
        ("OPENMED_MCP_",),
        (PACKAGE_ROOT / "mcp", SERVICE_ROOT / "security.py"),
    ),
    "openhim": (("OPENMED_OPENHIM_",), (SERVICE_ROOT / "openhim_mediator.py",)),
    "journey": (
        ("OPENMED_JOURNEY_",),
        (
            PACKAGE_ROOT / "deploy" / "journey.py",
            PACKAGE_ROOT / "structured" / "store",
            SERVICE_ROOT / "security_profile.py",
        ),
    ),
    "eval_datasets": (
        ("OPENMED_",),
        (PACKAGE_ROOT / "eval" / "datasets",),
    ),
}


def test_sections_are_declared_in_order() -> None:
    assert [section.name for section in settings_sections()] == [
        SECTION_CORE,
        SECTION_SERVICE,
        *NON_SERVICE_SECTIONS,
    ]


@pytest.mark.parametrize("section", sorted(SECTION_SOURCES))
def test_mcp_openhim_journey_and_dataset_reads_are_declared(section) -> None:
    prefixes, paths = SECTION_SOURCES[section]
    scan = scan_env_reads(list(paths))
    literals = literal_env_names(list(paths))
    names = {
        name
        for name in scan.names() | set(literals)
        if name.startswith(prefixes) and name not in excluded_identifiers()
    }

    undeclared = {
        name: scan.locations(name) or literals[name]
        for name in names - declared_setting_names()
    }
    assert undeclared == {}, section
    # Each subsystem's own names live in its section; shared ones such as the
    # cache root or the Journey release key are declared elsewhere.
    own = {
        name
        for name in names
        if name.endswith(("_PATH", "_DIR", "_YEAR")) or section != "eval_datasets"
    }
    own -= {"OPENMED_CACHE_DIR", "OPENMED_JOURNEY_RELEASE_KEY"}
    assert own and own <= declared_setting_names(section), section


def test_each_named_prefix_lives_in_its_section() -> None:
    for declaration in setting_declarations():
        name = declaration.name
        if name.startswith("OPENMED_MCP_"):
            assert declaration.section == "mcp", name
        elif name.startswith("OPENMED_OPENHIM_"):
            assert declaration.section == "openhim", name
        elif name.startswith("OPENMED_JOURNEY_") and not name.endswith("_KEY"):
            assert declaration.section == "journey", name
        elif name.startswith("OPENMED_SERVICE_") and not name.endswith("_KEY"):
            assert declaration.section == SECTION_SERVICE, name


def test_dataset_paths_are_user_supplied_without_defaults() -> None:
    section = next(item for item in settings_sections() if item.name == "eval_datasets")
    assert "never" in section.description and "bundles" in section.description
    paths = [item for item in section.settings if item.name.endswith(("_PATH", "_DIR"))]
    assert len(paths) >= 15
    for declaration in paths:
        assert declaration.user_supplied, declaration.name
        assert not declaration.has_default, declaration.name
        assert declaration.env_format == "path", declaration.name
        assert declaration.to_dict()["user_supplied"] is True
    for declaration in setting_declarations():
        if declaration.user_supplied:
            assert not declaration.has_default, declaration.name


def test_excluded_identifiers_have_reasons_and_are_not_settings() -> None:
    excluded = excluded_identifiers()
    assert "OPENMED_CANONICAL_HASH_SYSTEM" in excluded
    assert not set(excluded) & declared_setting_names()
    for name, reason in excluded.items():
        assert name.startswith("OPENMED_")
        assert len(reason) >= 10, name
    with pytest.raises(TypeError):
        excluded["OPENMED_SYNTHETIC"] = "mutated"  # type: ignore[index]


def test_registry_rejects_malformed_exclusions() -> None:
    document = load_settings_schema()
    document["x-excluded-identifiers"]["OPENMED_SERVICE_LOG_LEVEL"] = "a constant"
    with pytest.raises(SettingsSchemaError, match="both declared and excluded"):
        parse_settings_schema(document)

    document = load_settings_schema()
    document["x-excluded-identifiers"]["OPENMED_SYNTHETIC_EMPTY_REASON"] = " "
    with pytest.raises(SettingsSchemaError, match="without a reason"):
        parse_excluded_identifiers(document)

    document = load_settings_schema()
    document["x-excluded-identifiers"]["lower_case_name"] = "a constant"
    with pytest.raises(SettingsSchemaError, match="malformed identifier"):
        parse_excluded_identifiers(document)


def test_registry_rejects_a_user_supplied_default() -> None:
    document = load_settings_schema()
    entry = document["properties"]["eval_datasets"]["properties"]["OPENMED_I2B2_PATH"]
    entry["default"] = "/synthetic/i2b2"
    with pytest.raises(SettingsSchemaError) as exc_info:
        parse_settings_schema(document)
    assert "OPENMED_I2B2_PATH" in str(exc_info.value)
    assert "/synthetic/i2b2" not in str(exc_info.value)
