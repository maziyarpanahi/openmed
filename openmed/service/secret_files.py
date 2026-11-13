"""Read service credentials from inline settings or mounted secret files.

Every credential setting ``NAME`` also accepts ``NAME_FILE``: the path of a
file, typically a Docker/Compose secret or a Kubernetes Secret volume, whose
contents are the value.  The two forms are mutually exclusive.  File values are
size bounded, must be UTF-8 text, and lose exactly one trailing line ending.
Under the production profile (``OPENMED_PROFILE`` unset or ``prod``) a file
that other users can access, or that its group can write, is refused; group
read is accepted only for root-owned files, which is how Kubernetes Secret
volumes appear when a pod sets ``fsGroup``.

Failures raise :class:`SecretSourceError`.  Its message names only the setting
and a fixed reason code: never the configured path, the file contents, or the
underlying operating-system error, and the original exception is not chained.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from typing import Optional

SECRET_FILE_SUFFIX = "_FILE"
DEFAULT_MAX_SECRET_FILE_BYTES = 64 * 1024
PROFILE_ENV_VAR = "OPENMED_PROFILE"
PRODUCTION_PROFILE = "prod"

# Settings that accept a ``<NAME>_FILE`` companion.  Each owning module keeps
# its own constant; tests assert the two stay in sync.
CREDENTIAL_SETTINGS = (
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_OTLP_HEADERS",
    "OPENMED_OPENHIM_PASSWORD",
    "OPENMED_SIDECAR_HASH_SECRET",
    "OPENMED_JOURNEY_POSTGRES_DSN",
)

SECRET_SOURCE_CONFLICT = "secret_source_conflict"
SECRET_FILE_UNREADABLE = "secret_file_unreadable"
SECRET_FILE_NOT_REGULAR = "secret_file_not_regular"
SECRET_FILE_TOO_LARGE = "secret_file_too_large"
SECRET_FILE_PERMISSIONS_TOO_OPEN = "secret_file_permissions_too_open"
SECRET_FILE_INVALID_ENCODING = "secret_file_invalid_encoding"
SECRET_FILE_EMPTY = "secret_file_empty"

_REASONS = {
    SECRET_SOURCE_CONFLICT: "is set together with its inline value",
    SECRET_FILE_UNREADABLE: "could not be read",
    SECRET_FILE_NOT_REGULAR: "must name a regular file",
    SECRET_FILE_TOO_LARGE: "exceeds the secret file size limit",
    SECRET_FILE_PERMISSIONS_TOO_OPEN: (
        "must not be accessible to other users or writable by its group"
    ),
    SECRET_FILE_INVALID_ENCODING: "must contain UTF-8 text",
    SECRET_FILE_EMPTY: "is empty",
}
SECRET_SOURCE_ERROR_CODES = frozenset(_REASONS)

_READ_CHUNK_BYTES = 64 * 1024
_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOCTTY", 0)
    # A FIFO would otherwise block the open until a writer appears.
    | getattr(os, "O_NONBLOCK", 0)
)


class SecretSourceError(ValueError):
    """Credential source failure that names only the setting and a fixed code.

    Attributes:
        setting: Environment variable that holds the faulty source, such as
            ``OPENMED_OPENHIM_PASSWORD_FILE``.
        code: One of :data:`SECRET_SOURCE_ERROR_CODES`.
    """

    def __init__(self, setting: str, code: str) -> None:
        if code not in _REASONS:
            code = SECRET_FILE_UNREADABLE
        self.setting = setting
        self.code = code
        super().__init__(f"{setting} {_REASONS[code]} ({code})")


def secret_file_setting(name: str) -> str:
    """Return the ``<NAME>_FILE`` companion of a credential setting.

    Args:
        name: Inline credential setting, such as ``OPENMED_OPENHIM_PASSWORD``.

    Returns:
        The environment variable that names the credential file.
    """
    return f"{name}{SECRET_FILE_SUFFIX}"


def is_production_profile(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the service runs under the production profile.

    The service treats an unset ``OPENMED_PROFILE`` as ``prod``, so file
    permission checks are on unless a non-production profile is selected.

    Args:
        environ: Environment mapping; defaults to :data:`os.environ`.

    Returns:
        ``True`` when the active profile is ``prod``.
    """
    env = os.environ if environ is None else environ
    profile = env.get(PROFILE_ENV_VAR, PRODUCTION_PROFILE)
    return profile.strip().lower() == PRODUCTION_PROFILE


def read_secret_setting(
    name: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
    max_bytes: int = DEFAULT_MAX_SECRET_FILE_BYTES,
    enforce_permissions: Optional[bool] = None,
) -> Optional[str]:
    """Read a credential from ``NAME`` or from the file named by ``NAME_FILE``.

    Args:
        name: Inline credential setting, such as ``OPENMED_OPENHIM_PASSWORD``.
        environ: Environment mapping; defaults to :data:`os.environ`.
        max_bytes: Largest accepted secret file size in bytes.
        enforce_permissions: Refuse files that other users can access. Defaults
            to the production profile on POSIX hosts.

    Returns:
        The inline value unchanged, the file contents without one trailing line
        ending, or ``None`` when neither form is set.

    Raises:
        SecretSourceError: Both forms are set or the file cannot be used.
    """
    env = os.environ if environ is None else environ
    if enforce_permissions is None:
        enforce_permissions = is_production_profile(env)
    return resolve_secret(
        name,
        inline=env.get(name),
        file_path=env.get(secret_file_setting(name)),
        max_bytes=max_bytes,
        enforce_permissions=enforce_permissions,
    )


def resolve_secret(
    name: str,
    *,
    inline: Optional[str],
    file_path: Optional[str],
    max_bytes: int = DEFAULT_MAX_SECRET_FILE_BYTES,
    enforce_permissions: bool = True,
) -> Optional[str]:
    """Resolve one credential from already-read inline and file settings.

    Blank values count as unset, so a deployment can keep an empty inline
    default while pointing ``NAME_FILE`` at a mounted secret.

    Args:
        name: Inline credential setting used in error messages.
        inline: Raw inline value, or ``None``.
        file_path: Raw ``NAME_FILE`` value, or ``None``.
        max_bytes: Largest accepted secret file size in bytes.
        enforce_permissions: Refuse files that other users can access.

    Returns:
        The inline value unchanged, the file contents, or ``None``.

    Raises:
        SecretSourceError: Both forms are set or the file cannot be used.
    """
    file_setting = secret_file_setting(name)
    if file_path is not None and _is_set(file_path):
        if _is_set(inline):
            raise SecretSourceError(file_setting, SECRET_SOURCE_CONFLICT)
        return read_secret_file(
            file_path,
            setting=file_setting,
            max_bytes=max_bytes,
            enforce_permissions=enforce_permissions,
        )
    return inline


def read_secret_file(
    path: str,
    *,
    setting: str,
    max_bytes: int = DEFAULT_MAX_SECRET_FILE_BYTES,
    enforce_permissions: bool = True,
) -> str:
    """Read one bounded UTF-8 secret file.

    Symbolic links are followed, as Kubernetes Secret volumes require; the
    checks apply to the file they resolve to.

    Args:
        path: File to read.
        setting: Setting name reported by errors in place of the path.
        max_bytes: Largest accepted file size in bytes.
        enforce_permissions: Refuse files that other users can access, or that
            the group can write; group read is allowed for root-owned files.
            Ignored on platforms without POSIX permission bits.

    Returns:
        The file contents without one trailing ``\\n`` or ``\\r\\n``.

    Raises:
        SecretSourceError: The file is unreadable, not a regular file, too
            large, too permissive, not UTF-8, or empty.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    # Errors are raised outside ``except`` blocks so that no OSError (which
    # carries the path) is chained to, or kept as context of, the failure.
    descriptor: Optional[int]
    try:
        descriptor = os.open(path, _OPEN_FLAGS)
    except (OSError, TypeError, ValueError):
        descriptor = None
    if descriptor is None:
        raise SecretSourceError(setting, SECRET_FILE_UNREADABLE)
    try:
        data, code = _read_descriptor(
            descriptor,
            max_bytes=max_bytes,
            enforce_permissions=enforce_permissions,
        )
    finally:
        os.close(descriptor)
    if code is not None:
        raise SecretSourceError(setting, code)
    text: Optional[str]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = None
    if text is None:
        raise SecretSourceError(setting, SECRET_FILE_INVALID_ENCODING)
    text = _strip_one_line_ending(text)
    if not text.strip():
        raise SecretSourceError(setting, SECRET_FILE_EMPTY)
    return text


def _read_descriptor(
    descriptor: int,
    *,
    max_bytes: int,
    enforce_permissions: bool,
) -> tuple[bytes, Optional[str]]:
    """Return the file bytes, or a failure code instead of raising."""
    try:
        info = os.fstat(descriptor)
    except OSError:
        return b"", SECRET_FILE_UNREADABLE
    if not stat.S_ISREG(info.st_mode):
        return b"", SECRET_FILE_NOT_REGULAR
    if enforce_permissions and os.name == "posix" and _too_permissive(info):
        return b"", SECRET_FILE_PERMISSIONS_TOO_OPEN
    if info.st_size > max_bytes:
        return b"", SECRET_FILE_TOO_LARGE
    chunks: list[bytes] = []
    remaining = max_bytes + 1
    try:
        while remaining > 0:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    except OSError:
        return b"", SECRET_FILE_UNREADABLE
    if remaining <= 0:
        # The file grew past the limit after it was measured.
        return b"", SECRET_FILE_TOO_LARGE
    return b"".join(chunks), None


def _too_permissive(info: os.stat_result) -> bool:
    mode = stat.S_IMODE(info.st_mode)
    if mode & stat.S_IRWXO:
        return True
    if mode & (stat.S_IWGRP | stat.S_IXGRP):
        return True
    # Root-owned group-readable files are how orchestrators share a mounted
    # secret with the pod's fsGroup; libpq applies the same rule to keys.
    return bool(mode & stat.S_IRGRP) and info.st_uid != 0


def _strip_one_line_ending(text: str) -> str:
    if text.endswith("\r\n"):
        return text[:-2]
    if text.endswith("\n"):
        return text[:-1]
    return text


def _is_set(value: Optional[str]) -> bool:
    return value is not None and bool(value.strip())


__all__ = [
    "CREDENTIAL_SETTINGS",
    "DEFAULT_MAX_SECRET_FILE_BYTES",
    "SECRET_FILE_EMPTY",
    "SECRET_FILE_INVALID_ENCODING",
    "SECRET_FILE_NOT_REGULAR",
    "SECRET_FILE_PERMISSIONS_TOO_OPEN",
    "SECRET_FILE_SUFFIX",
    "SECRET_FILE_TOO_LARGE",
    "SECRET_FILE_UNREADABLE",
    "SECRET_SOURCE_CONFLICT",
    "SECRET_SOURCE_ERROR_CODES",
    "SecretSourceError",
    "is_production_profile",
    "read_secret_file",
    "read_secret_setting",
    "resolve_secret",
    "secret_file_setting",
]
