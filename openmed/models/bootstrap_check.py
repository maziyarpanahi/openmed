"""Deterministic, offline diagnostics for local model bootstrap readiness.

The bootstrap check is deliberately narrower than a general environment
doctor.  It answers whether a local model command has the local prerequisites
it was asked to require, without echoing paths, model identifiers, hashes,
environment values, or model contents.

The check never downloads, imports optional runtime packages, or opens model
payloads unless a local integrity manifest asks the checksum verifier to read
those files.  The verifier remains local-only; only aggregate pass/fail state
is retained in the report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.model_integrity import (
    ARTIFACT_MANIFEST_FILENAME,
    ModelIntegrityError,
    verify_artifact_manifest,
)
from ..core.offline import HF_OFFLINE_ENV_VARS, OFFLINE_ENV_VAR, env_flag_enabled

SCHEMA_VERSION = "openmed.bootstrap_diagnostics.v1"

EXIT_READY = 0
EXIT_NOT_READY = 1
EXIT_USAGE = 2

CATEGORY_ORDER = (
    "cache",
    "checksum",
    "optional_extras",
    "offline_policy",
)

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"

# These are dependency names, not package versions.  They are inspected with
# ``find_spec`` so an optional import cannot run arbitrary package startup code.
OPTIONAL_EXTRAS: Mapping[str, tuple[str, ...]] = {
    "coreml": ("coremltools",),
    "hf": ("huggingface_hub", "transformers"),
    "mlx": ("mlx",),
    "multimodal": ("PIL",),
    "onnx": ("onnxruntime",),
}

_MANIFEST_SUFFIX = ".json"


@dataclass(frozen=True)
class DiagnosticCategory:
    """One value-free bootstrap diagnostic category.

    ``facts`` contains only stable booleans, counts, and enumerated strings.
    It intentionally cannot carry the inspected path, model id, digest, or
    environment value.
    """

    status: str
    reason: str
    facts: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the category as a JSON-compatible mapping."""

        return {
            "status": self.status,
            "reason": self.reason,
            **dict(self.facts),
        }


@dataclass(frozen=True)
class BootstrapReport(Mapping[str, Any]):
    """Structured, value-free result of :func:`run_bootstrap_check`.

    The report implements :class:`collections.abc.Mapping` for callers that
    prefer ``report["ready"]`` while retaining typed helpers for Python APIs.
    """

    ready: bool
    categories: Mapping[str, DiagnosticCategory]

    @property
    def exit_code(self) -> int:
        """Return the stable process exit code for this report."""

        return EXIT_READY if self.ready else EXIT_NOT_READY

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible report shape."""

        return {
            "schema_version": SCHEMA_VERSION,
            "ready": self.ready,
            "exit_code": self.exit_code,
            "categories": {
                name: self.categories[name].to_dict() for name in CATEGORY_ORDER
            },
        }

    as_dict = to_dict

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def run_bootstrap_check(
    *,
    cache_dir: str | Path | None = None,
    model_id: str | None = None,
    model_path: str | Path | None = None,
    required_extras: Iterable[str] | None = None,
    require_checksum: bool = False,
    require_offline: bool = False,
    offline: bool | None = None,
    config: Any | None = None,
) -> BootstrapReport:
    """Inspect local prerequisites without making a network request.

    Args:
        cache_dir: Optional local Hugging Face cache directory.  When omitted,
            ``HF_HUB_CACHE``, ``HF_HOME``, ``XDG_CACHE_HOME``, and the standard
            local cache locations are considered in that order.
        model_id: Optional ``org/name`` model id to find in the cache.  The id
            is used only for local matching and is never included in output.
        model_path: Optional local model directory to check instead of a Hub
            cache.  It is mutually exclusive with ``model_id``.
        required_extras: Optional names from :data:`OPTIONAL_EXTRAS` that are
            required by the caller.  Other extras are reported as optional.
        require_checksum: Make an absent local integrity manifest a readiness
            failure instead of a warning.
        require_offline: Make an unrequested or unenforced offline policy a
            readiness failure.
        offline: Explicitly request or clear offline policy for this check.
            When omitted, the environment and optional config are inspected.
        config: Optional object with a boolean ``local_only`` attribute.

    Returns:
        A deterministic :class:`BootstrapReport`.  Warnings are informative;
        only failed categories change the report exit code to ``1``.

    Raises:
        ValueError: If mutually exclusive or unsupported diagnostic inputs are
            supplied.  The exception message is intentionally value-free.
    """

    if model_id is not None and model_path is not None:
        raise ValueError("model_id and model_path are mutually exclusive")
    if (
        isinstance(require_checksum, bool) is False
        or isinstance(require_offline, bool) is False
    ):
        raise ValueError("diagnostic requirements must be boolean")
    if offline is not None and not isinstance(offline, bool):
        raise ValueError("offline must be boolean when provided")

    normalized_model_id = _normalize_model_id(model_id)
    normalized_extras = _normalize_required_extras(required_extras)
    cache_category, cache_root = _check_cache(
        cache_dir=cache_dir,
        model_id=normalized_model_id,
        model_path=model_path,
    )
    checksum_category = _check_checksum(
        cache_root=cache_root,
        model_path=model_path,
        model_id=normalized_model_id,
        require_checksum=require_checksum,
        cache_ready=cache_category.status != STATUS_FAIL,
    )
    extras_category = _check_optional_extras(normalized_extras)
    offline_category = _check_offline_policy(
        config=config,
        offline=offline,
        require_offline=require_offline,
    )

    categories = {
        "cache": cache_category,
        "checksum": checksum_category,
        "optional_extras": extras_category,
        "offline_policy": offline_category,
    }
    ready = all(category.status != STATUS_FAIL for category in categories.values())
    return BootstrapReport(ready=ready, categories=categories)


check_bootstrap = run_bootstrap_check


def format_human(report: BootstrapReport) -> str:
    """Render a stable human-readable report without sensitive values."""

    state = "READY" if report.ready else "NOT READY"
    lines = [f"Bootstrap readiness: {state} (exit code {report.exit_code})"]
    for name in CATEGORY_ORDER:
        category = report.categories[name]
        label = category.status.upper()
        description = _reason_text(category.reason)
        lines.append(f"{name}: {label} - {description}")
    return "\n".join(lines)


def render_json(report: BootstrapReport) -> str:
    """Render a stable, value-free JSON report."""

    return json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone ``python -m`` argument parser."""

    parser = argparse.ArgumentParser(
        prog="python -m openmed.models.bootstrap_check",
        description="Check local model bootstrap readiness without network access.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Local model cache to inspect; the value is never printed.",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model-id",
        "--model",
        dest="model_id",
        help="Require one cached model repository; the value is never printed.",
    )
    model_group.add_argument(
        "--model-path",
        type=Path,
        help="Require one local model directory; the value is never printed.",
    )
    parser.add_argument(
        "--extra",
        dest="required_extras",
        action="append",
        metavar="NAME",
        help="Require an optional extra; repeat for multiple extras.",
    )
    parser.add_argument(
        "--require-checksum",
        action="store_true",
        help="Fail when no local integrity manifest is available.",
    )
    parser.add_argument(
        "--require-offline",
        "--offline",
        dest="require_offline",
        action="store_true",
        help="Require an enforced local-only policy.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the structured JSON report instead of human output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the standalone check and return a stable process exit code."""

    args = build_parser().parse_args(argv)
    try:
        report = run_bootstrap_check(
            cache_dir=args.cache_dir,
            model_id=args.model_id,
            model_path=args.model_path,
            required_extras=args.required_extras,
            require_checksum=args.require_checksum,
            require_offline=args.require_offline,
        )
    except ValueError:
        return _emit_invalid_input(args.json)

    output = render_json(report) if args.json else format_human(report)
    print(output)
    return report.exit_code


def _emit_invalid_input(json_output: bool) -> int:
    """Emit a value-free usage failure for invalid programmatic inputs."""

    if json_output:
        print(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "ready": False,
                    "exit_code": EXIT_USAGE,
                    "error": {
                        "code": "invalid_input",
                        "message": "Invalid bootstrap diagnostic input.",
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("Invalid bootstrap diagnostic input.", file=sys.stderr)
    return EXIT_USAGE


def _normalize_model_id(model_id: str | None) -> str | None:
    if model_id is None:
        return None
    if not isinstance(model_id, str):
        raise ValueError("model_id must be a non-empty safe string")
    value = model_id.strip()
    if (
        not value
        or any(not character.isprintable() for character in value)
        or value.startswith("/")
        or value.endswith("/")
        or "//" in value
        or ".." in value.split("/")
    ):
        raise ValueError("model_id must be a non-empty safe string")
    return value


def _normalize_required_extras(values: Iterable[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values,)

    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("required extras must use known names")
        name = value.strip().lower().replace("-", "_")
        if name not in OPTIONAL_EXTRAS:
            raise ValueError("required extras must use known names")
        normalized.add(name)
    return tuple(sorted(normalized))


def _check_cache(
    *,
    cache_dir: str | Path | None,
    model_id: str | None,
    model_path: str | Path | None,
) -> tuple[DiagnosticCategory, Path | None]:
    if model_path is not None:
        path = _coerce_path(model_path)
        if path is None or not _is_directory(path):
            return (
                DiagnosticCategory(
                    STATUS_FAIL,
                    "local_model_missing",
                    {"local_model_present": False},
                ),
                None,
            )
        if not _has_file(path, max_depth=4):
            return (
                DiagnosticCategory(
                    STATUS_FAIL,
                    "local_model_empty",
                    {"local_model_present": True},
                ),
                path,
            )
        return (
            DiagnosticCategory(
                STATUS_PASS,
                "local_model_available",
                {"local_model_present": True},
            ),
            path,
        )

    root = _cache_location(cache_dir)
    if root is None or not _is_directory(root):
        return (
            DiagnosticCategory(
                STATUS_FAIL,
                "cache_missing",
                {"cache_present": False, "snapshot_count": 0},
            ),
            root,
        )

    if model_id is not None:
        repo = root / _cache_repo_name(model_id)
        snapshots = _snapshot_directories(repo / "snapshots")
        usable = sum(1 for snapshot in snapshots if _has_file(snapshot, max_depth=1))
        if usable:
            return (
                DiagnosticCategory(
                    STATUS_PASS,
                    "snapshot_available",
                    {"cache_present": True, "snapshot_count": usable},
                ),
                root,
            )
        return (
            DiagnosticCategory(
                STATUS_FAIL,
                "model_not_cached",
                {"cache_present": True, "snapshot_count": 0},
            ),
            root,
        )

    repositories = _repository_directories(root)
    snapshots = [
        snapshot
        for repository in repositories
        for snapshot in _snapshot_directories(repository / "snapshots")
        if _has_file(snapshot, max_depth=1)
    ]
    if snapshots:
        return (
            DiagnosticCategory(
                STATUS_PASS,
                "snapshot_available",
                {
                    "cache_present": True,
                    "repository_count": len(repositories),
                    "snapshot_count": len(snapshots),
                },
            ),
            root,
        )
    return (
        DiagnosticCategory(
            STATUS_FAIL,
            "cache_empty",
            {
                "cache_present": True,
                "repository_count": len(repositories),
                "snapshot_count": 0,
            },
        ),
        root,
    )


def _check_checksum(
    *,
    cache_root: Path | None,
    model_path: str | Path | None,
    model_id: str | None,
    require_checksum: bool,
    cache_ready: bool,
) -> DiagnosticCategory:
    manifests = _integrity_manifests(
        cache_root=cache_root,
        model_path=model_path,
        model_id=model_id,
    )
    if not manifests:
        if not cache_ready:
            return DiagnosticCategory(
                STATUS_WARN,
                "checksum_not_checked",
                {"manifests_checked": 0, "verified": 0},
            )
        return DiagnosticCategory(
            STATUS_FAIL if require_checksum else STATUS_WARN,
            "checksum_required" if require_checksum else "checksum_unavailable",
            {"manifests_checked": 0, "verified": 0},
        )

    verified = 0
    failed = 0
    for manifest in manifests:
        try:
            verify_artifact_manifest(manifest, expected_model_id=model_id)
        except (
            ModelIntegrityError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            failed += 1
        else:
            verified += 1

    if failed:
        return DiagnosticCategory(
            STATUS_FAIL,
            "checksum_mismatch",
            {
                "manifests_checked": len(manifests),
                "verified": verified,
                "failed": failed,
            },
        )
    return DiagnosticCategory(
        STATUS_PASS,
        "checksums_verified",
        {"manifests_checked": len(manifests), "verified": verified},
    )


def _check_optional_extras(required: tuple[str, ...]) -> DiagnosticCategory:
    availability = {name: _extra_available(name) for name in sorted(OPTIONAL_EXTRAS)}
    missing = [name for name in required if not availability[name]]
    if missing:
        return DiagnosticCategory(
            STATUS_FAIL,
            "required_extras_missing",
            {
                "required": list(required),
                "missing_required": missing,
                "available_optional": [
                    name for name, present in availability.items() if present
                ],
            },
        )
    return DiagnosticCategory(
        STATUS_PASS,
        "required_extras_available" if required else "no_required_extras",
        {
            "required": list(required),
            "missing_required": [],
            "available_optional": [
                name for name, present in availability.items() if present
            ],
        },
    )


def _check_offline_policy(
    *,
    config: Any | None,
    offline: bool | None,
    require_offline: bool,
) -> DiagnosticCategory:
    config_requested = bool(getattr(config, "local_only", False))
    environment_requested = env_flag_enabled(os.getenv(OFFLINE_ENV_VAR))
    dependency_flags_enabled = all(
        env_flag_enabled(os.getenv(name)) for name in HF_OFFLINE_ENV_VARS
    )

    if offline is not None:
        requested = offline
        source = "argument"
    else:
        requested = config_requested or environment_requested
        if config_requested and environment_requested:
            source = "config+environment"
        elif config_requested:
            source = "config"
        elif environment_requested:
            source = "environment"
        else:
            source = "none"

    enforced = requested and dependency_flags_enabled
    facts = {
        "requested": requested,
        "enforced": enforced,
        "network": "blocked" if enforced else "allowed",
        "source": source,
    }
    if enforced:
        return DiagnosticCategory(STATUS_PASS, "offline_enforced", facts)
    if requested:
        return DiagnosticCategory(STATUS_FAIL, "offline_not_enforced", facts)
    if require_offline:
        return DiagnosticCategory(STATUS_FAIL, "offline_required", facts)
    return DiagnosticCategory(STATUS_PASS, "offline_not_requested", facts)


def _extra_available(name: str) -> bool:
    try:
        return all(
            importlib.util.find_spec(module) is not None
            for module in OPTIONAL_EXTRAS[name]
        )
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return False


def _cache_location(cache_dir: str | Path | None) -> Path | None:
    if cache_dir is not None:
        return _coerce_path(cache_dir)

    explicit = os.getenv("HF_HUB_CACHE")
    if explicit:
        return _coerce_path(explicit)
    hf_home = os.getenv("HF_HOME")
    if hf_home:
        return _coerce_path(Path(hf_home) / "hub")
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    if xdg_cache:
        return _coerce_path(Path(xdg_cache) / "huggingface" / "hub")
    try:
        return Path.home() / ".cache" / "huggingface" / "hub"
    except (OSError, RuntimeError):
        return None


def _coerce_path(value: str | Path) -> Path | None:
    try:
        return Path(value).expanduser()
    except (OSError, TypeError, ValueError):
        return None


def _normalize_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return _coerce_path(value)


def _cache_repo_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _repository_directories(root: Path) -> list[Path]:
    try:
        return sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.is_dir()
                and not entry.is_symlink()
                and entry.name.startswith("models--")
            ),
            key=lambda path: path.name,
        )
    except (OSError, ValueError):
        return []


def _snapshot_directories(root: Path) -> list[Path]:
    try:
        return sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.is_dir() and not entry.is_symlink()
            ),
            key=lambda path: path.name,
        )
    except (OSError, ValueError):
        return []


def _is_directory(path: Path) -> bool:
    try:
        return path.is_dir() and os.access(path, os.R_OK | os.X_OK)
    except (OSError, ValueError):
        return False


def _has_file(root: Path, *, max_depth: int) -> bool:
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        current, depth = pending.pop()
        try:
            entries = list(current.iterdir())
        except (OSError, ValueError):
            continue
        for entry in entries:
            try:
                if entry.is_file():
                    return True
                if depth < max_depth and entry.is_dir() and not entry.is_symlink():
                    pending.append((entry, depth + 1))
            except (OSError, ValueError):
                continue
    return False


def _integrity_manifests(
    *,
    cache_root: Path | None,
    model_path: str | Path | None,
    model_id: str | None,
) -> list[Path]:
    paths: list[Path] = []
    local_path = _normalize_path(model_path)
    if local_path is not None:
        paths.append(local_path / ARTIFACT_MANIFEST_FILENAME)
    elif cache_root is not None:
        paths.append(cache_root / ARTIFACT_MANIFEST_FILENAME)
        integrity_root = cache_root / "integrity"
        try:
            for owner in sorted(integrity_root.iterdir(), key=lambda path: path.name):
                if not owner.is_dir() or owner.is_symlink():
                    continue
                paths.extend(
                    sorted(
                        (
                            path
                            for path in owner.iterdir()
                            if path.is_file()
                            and not path.is_symlink()
                            and path.suffix == _MANIFEST_SUFFIX
                        ),
                        key=lambda path: path.name,
                    )
                )
        except (OSError, ValueError):
            pass

        repositories = _repository_directories(cache_root)
        if model_id is not None:
            repositories = [
                repository
                for repository in repositories
                if repository.name == _cache_repo_name(model_id)
            ]
        for repository in repositories:
            for snapshot in _snapshot_directories(repository / "snapshots"):
                paths.append(snapshot / ARTIFACT_MANIFEST_FILENAME)

    unique_paths = sorted(set(paths), key=lambda path: str(path))
    existing = [path for path in unique_paths if _is_regular_file(path)]
    if model_id is None:
        return existing
    return [path for path in existing if _manifest_matches_model(path, model_id)]


def _manifest_matches_model(path: Path, model_id: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return True
    if not isinstance(payload, Mapping):
        return True
    recorded_model_id = payload.get("model_id")
    return recorded_model_id == model_id


def _is_regular_file(path: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink()
    except (OSError, ValueError):
        return False


def _reason_text(reason: str) -> str:
    return {
        "cache_missing": "cache is unavailable",
        "cache_empty": "no local model snapshot is available",
        "model_not_cached": "requested model snapshot is unavailable",
        "local_model_missing": "local model directory is unavailable",
        "local_model_empty": "local model directory has no files",
        "local_model_available": "local model files are available",
        "snapshot_available": "local model snapshot is available",
        "checksum_not_checked": "checksum was not checked because cache is unavailable",
        "checksum_unavailable": "no local checksum manifest is recorded",
        "checksum_required": "a local checksum manifest is required",
        "checksum_mismatch": "one or more local checksum manifests failed",
        "checksums_verified": "local checksum manifests verified",
        "no_required_extras": "no optional extras are required",
        "required_extras_available": "required optional extras are available",
        "required_extras_missing": "one or more required optional extras are missing",
        "offline_enforced": "offline policy is enforced",
        "offline_not_enforced": "offline policy was requested but is not enforced",
        "offline_required": "offline policy is required",
        "offline_not_requested": "offline policy was not requested",
    }.get(reason, "diagnostic completed")


__all__ = [
    "BootstrapReport",
    "CATEGORY_ORDER",
    "DiagnosticCategory",
    "EXIT_NOT_READY",
    "EXIT_READY",
    "EXIT_USAGE",
    "OPTIONAL_EXTRAS",
    "SCHEMA_VERSION",
    "STATUS_FAIL",
    "STATUS_PASS",
    "STATUS_WARN",
    "build_parser",
    "check_bootstrap",
    "format_human",
    "main",
    "render_json",
    "run_bootstrap_check",
]


if __name__ == "__main__":  # pragma: no cover - exercised by the interpreter
    raise SystemExit(main())
