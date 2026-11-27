"""Compatibility root for legacy exceptions and the unrooted-exception gate.

``LegacyRootedError`` lets an exception that predates the structured taxonomy
join ``OpenMedError`` without changing anything callers observe. The tests
below build each synthetic sample twice from one source, once with its
original builtin bases and once rerooted, and require identical behavior.
The gate tests scan the real package in a subprocess, so importing every
public module never leaks into other tests, and exercise the comparison on a
synthetic package as a negative control.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Iterator

import pytest

from openmed.core.errors import (
    ERROR_CODES,
    InputError,
    LegacyRootedError,
    OpenMedError,
    legacy_error_code,
)

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "release" / "check_error_roots.py"
ALLOWLIST = ROOT / "tests" / "fixtures" / "error_roots" / "unrooted_allowlist.txt"

# The allowlist was seeded with this many entries. It may only shrink, so the
# ceiling never moves up; rerooting changes simply delete lines.
SEEDED_ALLOWLIST_SIZE = 1063

_spec = importlib.util.spec_from_file_location("check_error_roots", SCRIPT)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = gate
_spec.loader.exec_module(gate)

# One source, built twice: ``{root}`` is empty for the legacy twin and adds
# LegacyRootedError for the rerooted twin; ``{bare}`` replaces an ``Exception``
# base. Synthetic tokens only.
_SAMPLE_SOURCE = textwrap.dedent(
    '''
    from openmed.core.errors import LegacyRootedError


    class SampleValueError(ValueError{root}):
        """Synthetic value failure."""


    class SampleRuntimeError(RuntimeError{root}):
        """Synthetic runtime failure."""


    class SampleKeyError(KeyError{root}):
        """Synthetic lookup failure with KeyError formatting."""


    class SampleImportError(ImportError{root}):
        """Synthetic import failure with name and path attributes."""


    class SamplePermissionError(PermissionError{root}):
        """Synthetic OSError subclass with errno formatting."""


    class SampleTimeoutError(TimeoutError{root}):
        """Synthetic timeout."""


    class SampleBareError({bare}):
        """Synthetic failure that used to derive from Exception directly."""


    class SampleChildError(SampleValueError):
        """Synthetic subclass that inherits its root."""


    class SampleDiamondError(SampleValueError, KeyError{diamond_root}):
        """Synthetic diamond; listing KeyError's chain keeps it ahead of the roots."""


    class SampleCodedError(ValueError{root}):
        """Synthetic failure with an explicit stable code."""

        code = "sample_explicit_code"


    class SampleCodedChildError(SampleCodedError):
        """Synthetic subclass that inherits an explicit code."""


    class SampleCustomInitError(ValueError{root}):
        """Synthetic failure with a keyword-only initializer argument."""

        def __init__(self, reason, *, field_name=None):
            super().__init__(reason)
            self.reason = reason
            self.field_name = field_name


    class SampleInstanceCodeError(RuntimeError{root}):
        """Synthetic failure that stores its code on the instance."""

        def __init__(self, code):
            super().__init__(code)
            self.code = code


    class SampleMessageError(ValueError{root}):
        """Synthetic failure that already owns message and details."""

        def __init__(self, message):
            super().__init__(message)
            self.message = message
            self.details = {{"field": "synthetic_field"}}
    '''
)

_LEGACY_MODULE = "openmed_root_samples_legacy"
_ROOTED_MODULE = "openmed_root_samples_rooted"
_SAMPLE_NAMES = (
    "SampleValueError",
    "SampleRuntimeError",
    "SampleKeyError",
    "SampleImportError",
    "SamplePermissionError",
    "SampleTimeoutError",
    "SampleBareError",
    "SampleChildError",
    "SampleDiamondError",
    "SampleCodedError",
    "SampleCodedChildError",
    "SampleCustomInitError",
    "SampleInstanceCodeError",
    "SampleMessageError",
)


def _build_module(name: str, *, rooted: bool) -> types.ModuleType:
    source = _SAMPLE_SOURCE.format(
        root=", LegacyRootedError" if rooted else "",
        bare="LegacyRootedError" if rooted else "Exception",
        diamond_root=", LookupError, LegacyRootedError" if rooted else "",
    )
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f"<{name}>", "exec"), module.__dict__)
    return module


@pytest.fixture(scope="module")
def twins() -> Iterator[tuple[types.ModuleType, types.ModuleType]]:
    legacy = _build_module(_LEGACY_MODULE, rooted=False)
    rooted = _build_module(_ROOTED_MODULE, rooted=True)
    yield legacy, rooted
    sys.modules.pop(_LEGACY_MODULE, None)
    sys.modules.pop(_ROOTED_MODULE, None)


def _normalized(snapshot: dict, module: str) -> dict:
    return json.loads(json.dumps(snapshot).replace(module, "<sample>"))


@pytest.mark.parametrize("name", _SAMPLE_NAMES)
def test_rerooted_sample_keeps_mro_args_str_and_pickling(twins, name) -> None:
    legacy, rooted = twins
    legacy_class = getattr(legacy, name)
    rooted_class = getattr(rooted, name)

    assert not issubclass(legacy_class, OpenMedError)
    assert issubclass(rooted_class, OpenMedError)
    assert issubclass(rooted_class, LegacyRootedError)
    assert _normalized(gate.behavior_snapshot(rooted_class), _ROOTED_MODULE) == (
        _normalized(gate.behavior_snapshot(legacy_class), _LEGACY_MODULE)
    )
    assert gate.legacy_root_violations(rooted_class) == []
    for base in legacy_class.__mro__:
        if base in vars(builtins).values():
            assert issubclass(rooted_class, base)


def test_rerooted_sample_keeps_builtin_isinstance_and_except_clauses(twins) -> None:
    _, rooted = twins

    with pytest.raises(ValueError) as caught_value:
        raise rooted.SampleValueError("synthetic_reason")
    assert isinstance(caught_value.value, OpenMedError)

    try:
        raise rooted.SampleTimeoutError("synthetic_reason")
    except OpenMedError as error:
        assert isinstance(error, TimeoutError)
        assert isinstance(error, OSError)

    with pytest.raises(KeyError):
        raise rooted.SampleDiamondError("synthetic_key")


def test_rerooted_class_raises_with_zero_and_many_arguments(twins) -> None:
    legacy, rooted = twins

    with pytest.raises(OpenMedError) as bare_raise:
        raise rooted.SampleValueError
    assert bare_raise.value.args == ()
    assert str(bare_raise.value) == ""

    for args in [(), ("synthetic_reason",), ("synthetic_a", "synthetic_b", 3)]:
        rooted_error = rooted.SampleBareError(*args)
        legacy_error = legacy.SampleBareError(*args)
        assert rooted_error.args == legacy_error.args == args
        assert str(rooted_error) == str(legacy_error) == str(Exception(*args))


def test_rerooted_class_keeps_builtin_formatting_and_attributes(twins) -> None:
    _, rooted = twins

    assert str(rooted.SampleKeyError("synthetic_key")) == "'synthetic_key'"
    assert str(rooted.SampleDiamondError("synthetic_key")) == "'synthetic_key'"
    assert repr(rooted.SampleValueError("synthetic_reason")) == (
        "SampleValueError('synthetic_reason')"
    )

    import_error = rooted.SampleImportError("synthetic_reason", name="synthetic_mod")
    assert import_error.name == "synthetic_mod"
    assert import_error.msg == "synthetic_reason"
    assert str(import_error) == "synthetic_reason"

    permission = rooted.SamplePermissionError(13, "synthetic denial")
    assert permission.errno == 13
    assert permission.strerror == "synthetic denial"
    assert str(permission) == "[Errno 13] synthetic denial"

    custom = rooted.SampleCustomInitError("synthetic_reason", field_name="field")
    assert custom.args == ("synthetic_reason",)
    assert custom.field_name == "field"
    restored = pickle.loads(pickle.dumps(rooted.SampleValueError("synthetic_reason")))
    assert type(restored) is rooted.SampleValueError
    assert restored.args == ("synthetic_reason",)


def test_rerooting_adds_no_instance_state(twins) -> None:
    legacy, rooted = twins

    for name in _SAMPLE_NAMES:
        legacy_class = getattr(legacy, name)
        rooted_class = getattr(rooted, name)
        args = ("synthetic_reason",)
        assert vars(rooted_class(*args)) == vars(legacy_class(*args)), name


def test_roots_sit_directly_before_exception(twins) -> None:
    _, rooted = twins

    for name in _SAMPLE_NAMES:
        mro = getattr(rooted, name).__mro__
        position = mro.index(LegacyRootedError)
        assert mro[position + 1 : position + 3] == (OpenMedError, Exception), name
        assert mro[position + 3 :] == (BaseException, object), name


def test_rerooted_classes_gain_stable_codes(twins) -> None:
    legacy, rooted = twins

    assert rooted.SampleValueError.code == "sample_value_error"
    assert rooted.SampleValueError("synthetic_reason").code == "sample_value_error"
    assert rooted.SampleChildError.code == "sample_child_error"
    assert rooted.SampleCodedError.code == "sample_explicit_code"
    assert rooted.SampleCodedChildError.code == "sample_explicit_code"
    assert rooted.SampleInstanceCodeError("synthetic_code").code == "synthetic_code"
    assert rooted.SampleInstanceCodeError.code == "sample_instance_code_error"
    assert LegacyRootedError.code == "legacy_rooted_error"

    for name in _SAMPLE_NAMES:
        # The code a class will report is predictable before it is rerooted.
        assert legacy_error_code(getattr(legacy, name)) == getattr(rooted, name).code
        assert legacy_error_code(getattr(rooted, name)) == getattr(rooted, name).code


@pytest.mark.parametrize(
    ("class_name", "code"),
    [
        ("AecgError", "aecg_error"),
        ("HTTPError", "http_error"),
        ("FHIRBundleError", "fhir_bundle_error"),
        ("ICD10CMError", "icd10_cm_error"),
        ("Snake_CaseError", "snake_case_error"),
        ("Ünicode", "openmed_error"),
    ],
)
def test_derived_codes_are_snake_case(class_name: str, code: str) -> None:
    assert legacy_error_code(type(class_name, (ValueError,), {})) == code


def test_legacy_error_code_rejects_non_exceptions_and_non_string_codes() -> None:
    with pytest.raises(TypeError):
        legacy_error_code(object)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        legacy_error_code("SampleError")  # type: ignore[arg-type]

    property_code = type("PropertyCodeError", (ValueError,), {"code": property(str)})
    with pytest.raises(TypeError):
        legacy_error_code(property_code)


def test_rerooted_classes_gain_to_dict_without_new_details(twins) -> None:
    _, rooted = twins

    error = rooted.SampleValueError("synthetic_reason")
    assert error.to_dict() == {
        "code": "sample_value_error",
        "message": "synthetic_reason",
        "details": {},
    }
    assert error.to_dict(include_details=False) == {
        "code": "sample_value_error",
        "message": "synthetic_reason",
    }
    assert rooted.SampleValueError().to_dict()["message"] == ""
    assert rooted.SampleKeyError("synthetic_key").to_dict()["message"] == (
        "'synthetic_key'"
    )

    # Legacy attributes are never promoted into the structured payload: they
    # were not written against the PHI-safe details contract.
    owned = rooted.SampleMessageError("synthetic_message")
    assert owned.details == {"field": "synthetic_field"}
    assert owned.to_dict() == {
        "code": "sample_message_error",
        "message": "synthetic_message",
        "details": {},
    }


def test_rerooting_adds_only_code_and_to_dict(twins) -> None:
    legacy, rooted = twins

    for name in _SAMPLE_NAMES:
        legacy_error = getattr(legacy, name)("synthetic_reason")
        rooted_error = getattr(rooted, name)("synthetic_reason")
        for attribute in ("message", "details", "envelope", "payload"):
            assert hasattr(rooted_error, attribute) == (
                hasattr(legacy_error, attribute)
            ), (name, attribute)
        added = {
            attribute
            for attribute in set(dir(rooted_error)) - set(dir(legacy_error))
            if not attribute.startswith("__")
        }
        assert added <= {"code", "to_dict"}, (name, added)


def test_legacy_root_used_directly_behaves_like_exception() -> None:
    for args in [(), ("synthetic_reason",), ("synthetic_a", 2)]:
        error = LegacyRootedError(*args)
        assert error.args == args
        assert str(error) == str(Exception(*args))
    with pytest.raises(TypeError) as rejected:
        LegacyRootedError(unexpected="synthetic")
    assert "keyword" in str(rejected.value)


def test_taxonomy_registry_is_unchanged_by_the_legacy_root() -> None:
    assert "LegacyRootedError" not in ERROR_CODES
    assert not issubclass(InputError, LegacyRootedError)
    error = InputError("Correct the synthetic field and retry.")
    assert str(error) == error.message


def test_wrong_root_order_is_a_violation() -> None:
    wrong_order = type("WrongOrderError", (LegacyRootedError, ValueError), {})
    assert gate.legacy_root_violations(wrong_order) == [
        "roots_not_directly_before_exception"
    ]

    parent = type("ParentError", (ValueError, LegacyRootedError), {})
    unrooted_diamond = type("DiamondError", (parent, KeyError), {})
    assert gate.legacy_root_violations(unrooted_diamond) == [
        "roots_not_directly_before_exception"
    ]

    bad_code = type("BadCodeError", (ValueError, LegacyRootedError), {"code": "Bad"})
    assert gate.legacy_root_violations(bad_code) == ["invalid_code"]


# --- transports keep their legacy mapping ---------------------------------


class _NoModelLoader:
    """Model loader double so the service tests never load a model."""

    def __init__(self, config: object) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def create_pipeline(self, model_name: str, **_: object) -> object:
        del model_name
        return object()


@pytest.fixture
def service_env(monkeypatch: pytest.MonkeyPatch):
    from openmed.service import auth as service_auth
    from openmed.service import runtime as service_runtime

    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in list(os.environ):
        if name.startswith("OPENMED_SERVICE_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", _NoModelLoader)

    def declare(path: str) -> None:
        monkeypatch.setitem(
            service_auth.ROUTE_POLICIES,
            ("GET", path),
            service_auth.RoutePolicy((), "control"),
        )

    return declare


def _envelope_without_request_id(response) -> dict:
    body = response.json()
    body["error"].pop("request_id", None)
    return body


def test_rest_service_keeps_legacy_handling_for_rerooted_errors(
    twins, service_env
) -> None:
    from fastapi.testclient import TestClient

    from openmed.service.app import create_app

    legacy, rooted = twins
    names = ("SampleValueError", "SampleRuntimeError", "SampleKeyError")
    routes = {}
    for module in (legacy, rooted):
        for name in names:
            path = f"/synthetic-error-root/{module.__name__}/{name}"
            routes[(module.__name__, name)] = path
            service_env(path)
    app = create_app()
    for (module_name, name), path in routes.items():
        error_class = getattr(sys.modules[module_name], name)

        async def _fail(error_class=error_class) -> None:
            raise error_class("synthetic_reason")

        app.add_api_route(path, _fail, methods=["GET"], include_in_schema=False)

    statuses = {}
    with TestClient(
        app, base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        for name in names:
            legacy_response = client.get(routes[(_LEGACY_MODULE, name)])
            rooted_response = client.get(routes[(_ROOTED_MODULE, name)])
            statuses[name] = rooted_response.status_code
            assert rooted_response.status_code == legacy_response.status_code, name
            assert _envelope_without_request_id(rooted_response) == (
                _envelope_without_request_id(legacy_response)
            ), name
            assert "synthetic_reason" not in rooted_response.text
    assert statuses == {
        "SampleValueError": 400,
        "SampleRuntimeError": 500,
        "SampleKeyError": 500,
    }

    with TestClient(app, base_url="http://127.0.0.1") as client:
        with pytest.raises(rooted.SampleRuntimeError):
            client.get(routes[(_ROOTED_MODULE, "SampleRuntimeError")])


def test_rest_service_still_maps_taxonomy_errors(service_env) -> None:
    from fastapi.testclient import TestClient

    from openmed.service.app import create_app

    path = "/synthetic-error-root/taxonomy"
    service_env(path)
    app = create_app()

    async def _fail() -> None:
        raise InputError("Correct the synthetic field and retry.")

    app.add_api_route(path, _fail, methods=["GET"], include_in_schema=False)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get(path)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "input_error"


def test_mcp_keeps_legacy_mapping_for_rerooted_errors(twins) -> None:
    from openmed.mcp.server import _error_envelope

    legacy, rooted = twins
    for name in _SAMPLE_NAMES:
        try:
            legacy_error = getattr(legacy, name)("synthetic_reason")
            rooted_error = getattr(rooted, name)("synthetic_reason")
        except TypeError:
            continue
        assert _error_envelope(rooted_error) == _error_envelope(legacy_error), name

    taxonomy = _error_envelope(InputError("Correct the synthetic field and retry."))
    assert taxonomy["error"]["code"] == "input_error"


def _outcome(call) -> tuple[str, object]:
    try:
        return ("returns", call())
    except Exception as error:  # noqa: BLE001 - compared by type only
        return ("raises", type(error).__name__)


def test_agent_envelopes_keep_legacy_mapping_for_rerooted_errors(twins) -> None:
    from openmed.agent.errors import ErrorStage, envelope_from_exception

    legacy, rooted = twins
    for name in _SAMPLE_NAMES:
        legacy_error = _outcome(lambda: getattr(legacy, name)("synthetic_reason"))
        rooted_error = _outcome(lambda: getattr(rooted, name)("synthetic_reason"))
        if legacy_error[0] == "raises":
            assert rooted_error == legacy_error, name
            continue
        for stage in ErrorStage:
            assert _outcome(
                lambda: envelope_from_exception(rooted_error[1], stage=stage)
            ) == _outcome(
                lambda: envelope_from_exception(legacy_error[1], stage=stage)
            ), (name, stage)


# --- the unrooted-exception gate ----------------------------------------------


@pytest.fixture(scope="module")
def package_report(tmp_path_factory) -> dict:
    output = tmp_path_factory.mktemp("error_roots") / "report.json"
    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT), *filter(None, [env.get("PYTHONPATH")])]
            ),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OPENMED_OFFLINE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--json-output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert output.is_file(), completed.stderr[-2000:]
    report = json.loads(output.read_text(encoding="utf-8"))
    report["exit_code"] = completed.returncode
    report["stdout"] = completed.stdout
    return report


# The package-wide scan imports every public module in a subprocess, so it
# follows the repository's ``slow`` tier (pushes to master and ``full-ci``).
# The synthetic negative controls below run on every pull request.
@pytest.mark.slow
def test_every_public_exception_is_rooted_or_allowlisted(package_report) -> None:
    report = gate.ScanReport.from_dict(package_report)
    result = gate.evaluate(report, gate.read_allowlist(ALLOWLIST))

    assert result.passed, result.describe()
    assert package_report["exit_code"] == 0, package_report["stdout"][-2000:]
    assert report.modules_scanned > 1000
    assert len(report.unrooted) > 0


def test_allowlist_only_shrinks_and_names_classes_only() -> None:
    entries = gate.read_allowlist(ALLOWLIST)

    assert len(entries) <= SEEDED_ALLOWLIST_SIZE
    assert entries == sorted(set(entries))
    for entry in entries:
        module, separator, qualname = entry.partition(":")
        assert separator == ":" and module.startswith("openmed"), entry
        assert qualname and " " not in entry, entry


@pytest.mark.slow
def test_scan_reports_only_qualified_names_and_skips_by_type(package_report) -> None:
    assert set(package_report["unrooted"]) <= set(gate.read_allowlist(ALLOWLIST))
    for module, failure in package_report["import_failures"].items():
        assert module.startswith("openmed"), module
        assert failure.isidentifier(), failure


_FIXTURE_FILES = {
    "__init__.py": "",
    "bad.py": """
        class BadError(ValueError):
            pass


        class PublicWarning(UserWarning):
            pass


        class _PrivateError(ValueError):
            pass


        Alias = ValueError
    """,
    "good.py": """
        from openmed.core.errors import InputError, LegacyRootedError


        class GoodError(ValueError, LegacyRootedError):
            pass


        class TaxonomyError(InputError):
            pass
    """,
    "wrong.py": """
        from openmed.core.errors import LegacyRootedError


        class WrongOrderError(LegacyRootedError, ValueError):
            pass
    """,
    "optional.py": """
        raise ImportError("synthetic optional dependency is not installed")
    """,
    "_hidden.py": """
        class HiddenError(ValueError):
            pass
    """,
    "sub/__init__.py": """
        from .._hidden import HiddenError as ExportedHiddenError
    """,
    "sub/nested.py": """
        class NestedError(RuntimeError):
            pass
    """,
    "data/not_a_package.py": """
        class NotScannedError(ValueError):
            pass
    """,
}


@pytest.fixture
def fixture_package(tmp_path, monkeypatch) -> Iterator[str]:
    name = "openmed_error_root_gate_fixture"
    package_dir = tmp_path / name
    for relative, source in _FIXTURE_FILES.items():
        path = package_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    for module in [key for key in sys.modules if key.split(".")[0] == name]:
        del sys.modules[module]


def test_scan_finds_unrooted_public_exceptions(fixture_package) -> None:
    report = gate.scan(fixture_package)
    name = fixture_package

    assert report.unrooted == (
        f"{name}._hidden:HiddenError",
        f"{name}.bad:BadError",
        f"{name}.sub.nested:NestedError",
    )
    assert report.import_failures == {f"{name}.optional": "ImportError"}
    assert set(report.legacy_rooted) == {
        f"{name}.good:GoodError",
        f"{name}.wrong:WrongOrderError",
    }
    assert report.legacy_rooted[f"{name}.good:GoodError"] == {
        "code": "good_error",
        "violations": [],
    }


def test_gate_fails_for_a_new_unrooted_exception(fixture_package) -> None:
    report = gate.scan(fixture_package)
    name = fixture_package
    allowlist = sorted(report.unrooted)

    result = gate.evaluate(report, allowlist[1:])
    assert not result.passed
    assert result.new == (allowlist[0],)
    assert allowlist[0] in result.describe()
    assert result.violations == (
        f"{name}.wrong:WrongOrderError: roots_not_directly_before_exception",
    )


def test_gate_passes_with_a_complete_allowlist_and_ignores_unverifiable(
    fixture_package,
) -> None:
    report = gate.scan(fixture_package)
    name = fixture_package
    compliant = gate.ScanReport(
        package=report.package,
        modules_scanned=report.modules_scanned,
        import_failures=report.import_failures,
        imported_modules=report.imported_modules,
        unrooted=report.unrooted,
        legacy_rooted={
            key: value
            for key, value in report.legacy_rooted.items()
            if not value["violations"]
        },
    )
    allowlist = sorted([*report.unrooted, f"{name}.optional:MissingError"])

    result = gate.evaluate(compliant, allowlist)
    assert result.passed, result.describe()
    assert result.unverified == (f"{name}.optional:MissingError",)


def test_gate_flags_stale_duplicate_and_unsorted_entries(fixture_package) -> None:
    report = gate.scan(fixture_package)
    name = fixture_package
    entries = list(report.unrooted)

    stale = gate.evaluate(report, sorted([*entries, f"{name}.good:GoodError"]))
    assert stale.stale == (f"{name}.good:GoodError",)
    assert not stale.passed

    duplicated = gate.evaluate(report, sorted([*entries, entries[0]]))
    assert duplicated.duplicates == (entries[0],)
    assert not gate.evaluate(report, list(reversed(entries))).passed


def test_prune_removes_only_stale_entries(fixture_package, tmp_path) -> None:
    report = gate.scan(fixture_package)
    name = fixture_package
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text(
        "# synthetic header\n"
        + "\n".join(sorted([*report.unrooted, f"{name}.good:GoodError"]))
        + "\n",
        encoding="utf-8",
    )

    result = gate.evaluate(report, gate.read_allowlist(allowlist))
    assert gate.prune_allowlist(allowlist, result.stale) == 1
    assert allowlist.read_text(encoding="utf-8").startswith("# synthetic header\n")
    assert gate.read_allowlist(allowlist) == list(report.unrooted)


def test_cli_exit_status_follows_the_gate(fixture_package, tmp_path, capsys) -> None:
    report = gate.scan(fixture_package)
    allowlist = tmp_path / "allowlist.txt"
    allowlist.write_text("\n".join(report.unrooted) + "\n", encoding="utf-8")
    argv = ["--package", fixture_package, "--allowlist", str(allowlist)]

    # The synthetic package carries a wrong-order class, so the gate fails.
    assert gate.main(argv) == 1
    assert "WrongOrderError" in capsys.readouterr().out

    allowlist.write_text("", encoding="utf-8")
    assert gate.main(argv) == 1
    output = capsys.readouterr().out
    assert "BadError" in output
    assert "synthetic optional dependency" not in output


def test_recorded_snapshot_detects_a_behavior_change(fixture_package, tmp_path) -> None:
    name = fixture_package
    document = gate.record_snapshots([f"{name}.bad"], package=name)

    assert document["schema_version"] == gate.SNAPSHOT_SCHEMA_VERSION
    assert list(document["classes"]) == [f"{name}.bad:BadError"]
    recorded = document["classes"][f"{name}.bad:BadError"]
    assert recorded["mro"] == [
        f"{name}.bad:BadError",
        "builtins:ValueError",
        "builtins:Exception",
        "builtins:BaseException",
    ]
    assert recorded["code"] == "bad_error"
    assert recorded["samples"][0] == {
        "call": 0,
        "args": "()",
        "str": "",
        "pickle_type": f"{name}.bad:BadError",
        "pickle_args": "()",
        "pickle_str": "",
    }

    bad = sys.modules[f"{name}.bad"]
    rerooted = type("BadError", (ValueError, LegacyRootedError), {})
    rerooted.__module__ = bad.__name__
    monkeypatch_target = bad.BadError
    bad.BadError = rerooted
    try:
        assert gate.replay_snapshot(f"{name}.bad:BadError", recorded) == []
        changed = type("BadError", (RuntimeError, LegacyRootedError), {})
        changed.__module__ = bad.__name__
        bad.BadError = changed
        assert gate.replay_snapshot(f"{name}.bad:BadError", recorded) == ["mro_changed"]
    finally:
        bad.BadError = monkeypatch_target
    assert gate.replay_snapshot(f"{name}.bad:BadError", recorded) == ["not_rooted"]
