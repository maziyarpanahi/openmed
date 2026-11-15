"""Tests for reading service credentials from mounted secret files."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.deploy import journey
from openmed.service import auth, openhim_mediator, sidecar, tracing
from openmed.service import secret_files as sf
from openmed.service.secret_files import (
    CREDENTIAL_SETTINGS,
    SECRET_SOURCE_ERROR_CODES,
    SecretSourceError,
    is_production_profile,
    read_secret_file,
    read_secret_setting,
)

ROOT = Path(__file__).resolve().parents[3]
SETTING = "OPENMED_TEST_CREDENTIAL"
FILE_SETTING = "OPENMED_TEST_CREDENTIAL_FILE"
SYNTHETIC_VALUE = "synthetic-credential-value"
POSIX_ONLY = pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")


@pytest.fixture(autouse=True)
def clean_credential_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "test")
    for name in CREDENTIAL_SETTINGS:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(f"{name}_FILE", raising=False)


def _secret(tmp_path: Path, content: bytes | str, *, mode: int = 0o600) -> Path:
    path = tmp_path / "credential"
    if isinstance(content, str):
        content = content.encode("utf-8")
    path.write_bytes(content)
    path.chmod(mode)
    return path


def _assert_value_free(error: BaseException, *fragments: object) -> None:
    rendered = f"{error!s} {error!r}"
    for fragment in fragments:
        assert str(fragment) not in rendered
    # No OSError (which carries the path) is chained or kept as context.
    assert error.__cause__ is None
    assert error.__context__ is None


# Resolver contract -----------------------------------------------------------


def test_inline_value_is_returned_unchanged_and_unset_is_none() -> None:
    assert read_secret_setting(SETTING, environ={SETTING: " raw "}) == " raw "
    assert read_secret_setting(SETTING, environ={}) is None


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (b"synthetic", "synthetic"),
        (b"synthetic\n", "synthetic"),
        (b"synthetic\r\n", "synthetic"),
        (b"synthetic\n\n", "synthetic\n"),
        (b"  synthetic  \n", "  synthetic  "),
        ("pässwörd\n".encode(), "pässwörd"),
    ],
)
def test_file_value_loses_exactly_one_trailing_line_ending(
    tmp_path: Path, content: bytes, expected: str
) -> None:
    path = _secret(tmp_path, content)

    value = read_secret_setting(SETTING, environ={FILE_SETTING: str(path)})

    assert value == expected


def test_blank_values_count_as_unset(tmp_path: Path) -> None:
    path = _secret(tmp_path, SYNTHETIC_VALUE)

    from_file = read_secret_setting(
        SETTING, environ={SETTING: "", FILE_SETTING: str(path)}
    )
    from_inline = read_secret_setting(
        SETTING, environ={SETTING: SYNTHETIC_VALUE, FILE_SETTING: "  "}
    )

    assert from_file == SYNTHETIC_VALUE
    assert from_inline == SYNTHETIC_VALUE


def test_inline_and_file_forms_are_mutually_exclusive(tmp_path: Path) -> None:
    path = _secret(tmp_path, SYNTHETIC_VALUE)

    with pytest.raises(SecretSourceError) as raised:
        read_secret_setting(
            SETTING, environ={SETTING: "inline-value", FILE_SETTING: str(path)}
        )

    assert raised.value.code == "secret_source_conflict"
    assert raised.value.setting == FILE_SETTING
    assert FILE_SETTING in str(raised.value)
    _assert_value_free(raised.value, path, tmp_path, "inline-value", SYNTHETIC_VALUE)


def _fifo(tmp_path: Path) -> Path:
    path = tmp_path / "fifo"
    os.mkfifo(path, 0o600)
    return path


@pytest.mark.parametrize(
    ("make_path", "code"),
    [
        (lambda tmp: tmp / "missing", "secret_file_unreadable"),
        (lambda tmp: tmp, "secret_file_not_regular"),
        pytest.param(
            _fifo,
            "secret_file_not_regular",
            marks=pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="no FIFOs"),
        ),
        (lambda tmp: _secret(tmp, b"\xff\xfe\x00"), "secret_file_invalid_encoding"),
        (lambda tmp: _secret(tmp, b""), "secret_file_empty"),
        (lambda tmp: _secret(tmp, b"\n"), "secret_file_empty"),
        (lambda tmp: _secret(tmp, b"   \r\n"), "secret_file_empty"),
    ],
)
def test_unusable_files_fail_closed_with_value_free_codes(
    tmp_path: Path, make_path: Any, code: str
) -> None:
    path = make_path(tmp_path)

    with pytest.raises(SecretSourceError) as raised:
        read_secret_setting(SETTING, environ={FILE_SETTING: str(path)})

    assert raised.value.code == code
    assert code in SECRET_SOURCE_ERROR_CODES
    assert str(raised.value).startswith(FILE_SETTING)
    _assert_value_free(raised.value, path, tmp_path)


def test_size_bound_is_inclusive(tmp_path: Path) -> None:
    path = _secret(tmp_path, "x" * 16)

    assert read_secret_file(str(path), setting=FILE_SETTING, max_bytes=16) == "x" * 16
    with pytest.raises(SecretSourceError) as raised:
        read_secret_file(str(path), setting=FILE_SETTING, max_bytes=15)

    assert raised.value.code == "secret_file_too_large"
    _assert_value_free(raised.value, path, "x" * 16)


def test_default_size_bound_rejects_oversized_files(tmp_path: Path) -> None:
    path = _secret(tmp_path, b"x" * (sf.DEFAULT_MAX_SECRET_FILE_BYTES + 1))

    with pytest.raises(SecretSourceError) as raised:
        read_secret_setting(SETTING, environ={FILE_SETTING: str(path)})

    assert raised.value.code == "secret_file_too_large"


def test_symlinked_secret_files_are_followed(tmp_path: Path) -> None:
    target = _secret(tmp_path, SYNTHETIC_VALUE)
    link = tmp_path / "link"
    link.symlink_to(target)

    assert read_secret_setting(SETTING, environ={FILE_SETTING: str(link)}) == (
        SYNTHETIC_VALUE
    )


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        (None, True),
        ("prod", True),
        (" PROD ", True),
        ("test", False),
        ("dev", False),
        ("", False),
    ],
)
def test_production_profile_matches_service_default(
    profile: str | None, expected: bool
) -> None:
    environ = {} if profile is None else {"OPENMED_PROFILE": profile}

    assert is_production_profile(environ) is expected


@POSIX_ONLY
@pytest.mark.parametrize("mode", [0o644, 0o604, 0o602, 0o640, 0o620, 0o660, 0o610])
def test_production_rejects_group_or_world_accessible_files(
    tmp_path: Path, mode: int
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root-owned files may be group-readable by design")
    path = _secret(tmp_path, SYNTHETIC_VALUE, mode=mode)
    environ = {FILE_SETTING: str(path), "OPENMED_PROFILE": "prod"}

    with pytest.raises(SecretSourceError) as raised:
        read_secret_setting(SETTING, environ=environ)

    assert raised.value.code == "secret_file_permissions_too_open"
    _assert_value_free(raised.value, path, tmp_path, SYNTHETIC_VALUE)


@POSIX_ONLY
@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_production_accepts_owner_only_files(tmp_path: Path, mode: int) -> None:
    path = _secret(tmp_path, SYNTHETIC_VALUE, mode=mode)

    assert (
        read_secret_setting(
            SETTING, environ={FILE_SETTING: str(path), "OPENMED_PROFILE": "prod"}
        )
        == SYNTHETIC_VALUE
    )


@POSIX_ONLY
def test_non_production_profiles_skip_permission_checks(tmp_path: Path) -> None:
    path = _secret(tmp_path, SYNTHETIC_VALUE, mode=0o644)

    dev = read_secret_setting(
        SETTING, environ={FILE_SETTING: str(path), "OPENMED_PROFILE": "dev"}
    )

    assert dev == SYNTHETIC_VALUE
    if os.geteuid() != 0:
        # An unset profile is the production default.
        with pytest.raises(SecretSourceError):
            read_secret_setting(SETTING, environ={FILE_SETTING: str(path)})


@POSIX_ONLY
@pytest.mark.parametrize(
    ("mode", "uid", "accepted"),
    [
        (0o440, 0, True),
        (0o640, 0, True),
        (0o440, 4242, False),
        (0o444, 0, False),
        (0o460, 0, False),
    ],
)
def test_group_read_is_accepted_only_for_root_owned_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    uid: int,
    accepted: bool,
) -> None:
    """Kubernetes Secret volumes with an fsGroup appear as root:fsGroup 0440."""
    path = _secret(tmp_path, SYNTHETIC_VALUE)
    real_fstat = os.fstat

    def fake_fstat(descriptor: int) -> os.stat_result:
        info = list(real_fstat(descriptor))
        info[stat.ST_MODE] = stat.S_IFREG | mode
        info[stat.ST_UID] = uid
        return os.stat_result(info)

    monkeypatch.setattr(sf.os, "fstat", fake_fstat)

    if accepted:
        assert read_secret_file(str(path), setting=FILE_SETTING) == SYNTHETIC_VALUE
    else:
        with pytest.raises(SecretSourceError) as raised:
            read_secret_file(str(path), setting=FILE_SETTING)
        assert raised.value.code == "secret_file_permissions_too_open"


def test_error_codes_are_controlled() -> None:
    error = SecretSourceError(FILE_SETTING, "unexpected /etc/secret detail")

    assert error.code == "secret_file_unreadable"
    assert "/etc/secret" not in str(error)


# Consumers -------------------------------------------------------------------


def test_credential_settings_match_their_owning_modules() -> None:
    assert set(CREDENTIAL_SETTINGS) == {
        auth.SERVICE_AUTH_API_KEYS_ENV_VAR,
        auth.SERVICE_AUTH_JWKS_ENV_VAR,
        tracing.OTLP_HEADERS_ENV_VAR,
        openhim_mediator.OPENHIM_PASSWORD_ENV_VAR,
        sidecar.SIDECAR_HASH_SECRET_ENV_VAR,
        "OPENMED_JOURNEY_POSTGRES_DSN",
    }
    assert auth.SERVICE_AUTH_API_KEYS_FILE_ENV_VAR == (
        f"{auth.SERVICE_AUTH_API_KEYS_ENV_VAR}_FILE"
    )
    assert (
        auth.SERVICE_AUTH_JWKS_FILE_ENV_VAR == f"{auth.SERVICE_AUTH_JWKS_ENV_VAR}_FILE"
    )
    assert tracing.OTLP_HEADERS_FILE_ENV_VAR == f"{tracing.OTLP_HEADERS_ENV_VAR}_FILE"
    assert sidecar.SIDECAR_HASH_SECRET_FILE_ENV_VAR == (
        f"{sidecar.SIDECAR_HASH_SECRET_ENV_VAR}_FILE"
    )


def test_every_file_setting_and_code_is_documented() -> None:
    docs = (ROOT / "docs" / "security" / "key-management.md").read_text("utf-8")

    for name in CREDENTIAL_SETTINGS:
        assert f"`{name}_FILE`" in docs
    for code in SECRET_SOURCE_ERROR_CODES:
        assert f"`{code}`" in docs


def _api_keys_json(raw_key: str) -> str:
    return json.dumps(
        [
            {
                "id": "file-key",
                "key_hash": auth.hash_api_key(raw_key),
                "principal": "file-service",
                "scopes": ["models:read"],
            }
        ]
    )


def test_api_keys_load_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _secret(tmp_path, _api_keys_json("synthetic-raw-key") + "\n")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS_FILE", str(path))

    config = auth.parse_service_auth_config()

    assert [item.key_id for item in config.api_keys] == ["file-key"]
    assert config.api_keys[0].key_hash == auth.hash_api_key("synthetic-raw-key")


def test_api_keys_from_file_authenticate_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from openmed.service import runtime as service_runtime
    from openmed.service.app import create_app

    class _Loader:
        def __init__(self, config: Any) -> None:
            self.config = config

        def loaded_models(self) -> dict[str, Any]:
            return {}

    monkeypatch.setattr(service_runtime, "ModelLoader", _Loader)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS_FILE",
        str(_secret(tmp_path, _api_keys_json("synthetic-raw-key"))),
    )

    with TestClient(
        create_app(), base_url="http://127.0.0.1", raise_server_exceptions=False
    ) as client:
        missing = client.get("/models/loaded")
        accepted = client.get(
            "/models/loaded", headers={"X-API-Key": "synthetic-raw-key"}
        )

    assert missing.status_code == 401
    assert accepted.status_code == 200


def test_api_keys_inline_and_file_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _secret(tmp_path, _api_keys_json("synthetic-raw-key"))
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS", _api_keys_json("other"))
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_API_KEYS_FILE", str(path))

    with pytest.raises(SecretSourceError) as raised:
        auth.parse_service_auth_config()

    assert raised.value.code == "secret_source_conflict"
    _assert_value_free(raised.value, path, auth.hash_api_key("other"))


def _jwks_json() -> str:
    return json.dumps(
        {"keys": [{"kty": "oct", "kid": "file-kid", "alg": "HS256", "k": "c3ludGg"}]}
    )


def test_jwks_file_keeps_working(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_JWKS_FILE", str(_secret(tmp_path, _jwks_json()))
    )

    config = auth.parse_service_auth_config()

    assert [key["kid"] for key in config.jwks] == ["file-kid"]


def test_jwks_inline_and_file_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWKS", _jwks_json())
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_JWKS_FILE", str(_secret(tmp_path, _jwks_json()))
    )

    with pytest.raises(SecretSourceError) as raised:
        auth.parse_service_auth_config()

    assert raised.value.setting == "OPENMED_SERVICE_AUTH_JWKS_FILE"
    assert raised.value.code == "secret_source_conflict"


@POSIX_ONLY
def test_jwks_file_permissions_are_checked_in_production(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.geteuid() == 0:
        pytest.skip("root-owned files may be group-readable by design")
    path = _secret(tmp_path, _jwks_json(), mode=0o644)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_JWKS_FILE", str(path))

    with pytest.raises(SecretSourceError) as raised:
        auth.parse_service_auth_config()

    assert raised.value.code == "secret_file_permissions_too_open"
    _assert_value_free(raised.value, path, "c3ludGg")


def _openhim_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMED_OPENHIM_MEDIATOR_ENABLED", "true")
    monkeypatch.setenv("OPENMED_OPENHIM_CORE_URL", "https://openhim-core.example:8080")
    monkeypatch.setenv("OPENMED_OPENHIM_USERNAME", "openhim@example.org")


def test_openhim_password_loads_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _openhim_env(monkeypatch)
    monkeypatch.setenv(
        "OPENMED_OPENHIM_PASSWORD_FILE", str(_secret(tmp_path, "synthetic-pass\n"))
    )

    settings = openhim_mediator.OpenHIMMediatorSettings.from_env()

    assert settings.password == "synthetic-pass"
    assert "synthetic-pass" not in repr(settings)


@pytest.mark.parametrize("inline", [True, False])
def test_openhim_password_file_errors_are_value_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inline: bool
) -> None:
    _openhim_env(monkeypatch)
    if inline:
        path = _secret(tmp_path, "synthetic-pass")
        monkeypatch.setenv("OPENMED_OPENHIM_PASSWORD", "synthetic-inline")
        code = "secret_source_conflict"
    else:
        path = tmp_path / "missing-password"
        code = "secret_file_unreadable"
    monkeypatch.setenv("OPENMED_OPENHIM_PASSWORD_FILE", str(path))

    with pytest.raises(openhim_mediator.OpenHIMMediatorConfigurationError) as raised:
        openhim_mediator.OpenHIMMediatorSettings.from_env()

    message = str(raised.value)
    assert "OPENMED_OPENHIM_PASSWORD_FILE" in message
    assert code in message
    for fragment in (str(path), "synthetic-pass", "synthetic-inline"):
        assert fragment not in message
    assert raised.value.__cause__ is None


def test_otlp_headers_load_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _secret(tmp_path, "Authorization=Bearer-synthetic,X-Scope=clinic-a\n")
    monkeypatch.setenv("OPENMED_SERVICE_OTLP_HEADERS_FILE", str(path))

    config = tracing.service_trace_config_from_env()

    assert dict(config.otlp_headers) == {
        "Authorization": "Bearer-synthetic",
        "X-Scope": "clinic-a",
    }


def test_otlp_headers_inline_and_file_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _secret(tmp_path, "Authorization=Bearer-synthetic")
    monkeypatch.setenv("OPENMED_SERVICE_OTLP_HEADERS", "X-Other=1")
    monkeypatch.setenv("OPENMED_SERVICE_OTLP_HEADERS_FILE", str(path))

    with pytest.raises(SecretSourceError) as raised:
        tracing.service_trace_config_from_env()

    assert raised.value.code == "secret_source_conflict"
    _assert_value_free(raised.value, path, "Bearer-synthetic")


def test_sidecar_hash_secret_loads_from_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OPENMED_SIDECAR_HASH_SECRET_FILE", str(_secret(tmp_path, "synthetic-hmac\n"))
    )

    assert sidecar._sidecar_hash_secret() == b"synthetic-hmac"


def test_sidecar_reports_secret_failure_without_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "missing-hmac"
    monkeypatch.setenv("OPENMED_SIDECAR_HASH_SECRET_FILE", str(path))
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    assert sidecar.main() == 1

    payload = json.loads(stderr.getvalue())
    assert payload == {
        "error_code": "SIDECAR_SECRET_INVALID",
        "event": "sidecar_failed",
    }
    assert str(tmp_path) not in stderr.getvalue()


class _ReadyReport:
    ready = True

    def to_dict(self) -> dict[str, Any]:
        return {"state": "ready"}


def test_journey_dsn_loads_from_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dsn = "postgresql://openmed@db.invalid:5432/openmed"
    monkeypatch.setenv(
        "OPENMED_JOURNEY_POSTGRES_DSN_FILE", str(_secret(tmp_path, dsn + "\n"))
    )
    seen: list[str] = []

    def fake_migrations(value: str, *, schema: str) -> _ReadyReport:
        del schema
        seen.append(value)
        return _ReadyReport()

    monkeypatch.setattr(journey, "run_migrations", fake_migrations)

    assert journey.main(["migrate"]) == 0
    assert seen == [dsn]
    assert dsn not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("mode", "inline", "code"),
    [
        (0o600, True, "secret_source_conflict"),
        (0o644, False, "secret_file_permissions_too_open"),
    ],
)
def test_journey_secret_failures_print_only_a_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: int,
    inline: bool,
    code: str,
) -> None:
    if code == "secret_file_permissions_too_open" and (
        os.name != "posix" or os.geteuid() == 0
    ):
        pytest.skip("needs POSIX permission bits and a non-root user")
    dsn = "postgresql://openmed@db.invalid:5432/openmed"
    path = _secret(tmp_path, dsn, mode=mode)
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_DSN_FILE", str(path))
    if inline:
        monkeypatch.setenv("OPENMED_JOURNEY_POSTGRES_DSN", dsn)

    assert journey.main(["migrate"]) == 2

    output = capsys.readouterr().out
    assert json.loads(output) == {
        "schema_version": "openmed.journey.deployment-health.v1",
        "state": "not_ready",
        "code": code,
    }
    assert str(tmp_path) not in output
    assert dsn not in output
