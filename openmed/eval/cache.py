"""Filesystem cache helpers for eval benchmark reports."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.eval.report import BenchmarkReport

CACHE_SCHEMA_VERSION = 1
DEFAULT_CODE_HASH_MODULES = ("openmed.eval.harness", "openmed.eval.metrics")


@dataclass(frozen=True)
class EvalCacheKey:
    """Content-addressed inputs that determine a benchmark report."""

    model_name: str
    suite: str
    fixture_set_hash: str
    code_hash: str
    device: str | None = None
    schema_version: int = CACHE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""
        payload: dict[str, Any] = {
            "code_hash": self.code_hash,
            "fixture_set_hash": self.fixture_set_hash,
            "model_name": self.model_name,
            "schema_version": self.schema_version,
            "suite": self.suite,
        }
        if self.device is not None:
            payload["device"] = self.device
        return payload

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest for this key."""
        return _hash_json(self.to_dict())


ReportKey = EvalCacheKey | Mapping[str, Any] | str


def default_cache_dir() -> Path:
    """Return the local default cache directory."""
    configured = os.environ.get("OPENMED_EVAL_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser() / "openmed" / "eval"
    return Path.home() / ".cache" / "openmed" / "eval"


def build_report_key(
    *,
    model_name: str,
    suite: str,
    fixture_set_hash: str,
    code_hash: str | None = None,
    device: str | None = None,
) -> EvalCacheKey:
    """Build a cache key for one benchmark report."""
    return EvalCacheKey(
        model_name=model_name,
        suite=suite,
        fixture_set_hash=fixture_set_hash,
        code_hash=code_hash or eval_code_hash(),
        device=device,
    )


def hash_fixture_set(fixtures: Iterable[Any]) -> str:
    """Hash normalized fixture contents without storing fixture text."""
    return _hash_json([_plain(fixture) for fixture in fixtures])


def eval_code_hash(
    module_names: Sequence[str] = DEFAULT_CODE_HASH_MODULES,
) -> str:
    """Hash eval harness and metric code so logic changes invalidate reports."""
    digest = hashlib.sha256()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        digest.update(module_name.encode("utf-8"))
        digest.update(b"\0")
        version = getattr(module, "EVAL_CACHE_VERSION", None)
        if version is not None:
            digest.update(f"version:{version}".encode("utf-8"))
        else:
            digest.update(_module_source(module))
        digest.update(b"\0")
    return digest.hexdigest()


def load(
    report_key: ReportKey, *, cache_dir: str | Path | None = None
) -> BenchmarkReport | None:
    """Load a cached report, returning ``None`` on a cache miss."""
    path = cache_path(report_key, cache_dir=cache_dir)
    if not path.exists():
        return None
    return BenchmarkReport.read_json(path)


def store(
    report_key: ReportKey,
    report: BenchmarkReport,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Store a report JSON payload under its content-addressed key."""
    path = cache_path(report_key, cache_dir=cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_json(indent=2) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        delete=False,
        dir=path.parent,
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as handle:
        handle.write(payload)
        tmp_path = Path(handle.name)
    tmp_path.replace(path)
    return path


def load_or_compute(
    report_key: ReportKey,
    compute_fn: Callable[[], BenchmarkReport],
    *,
    cache_dir: str | Path | None = None,
) -> BenchmarkReport:
    """Return a cached report or compute, store, and return a new one."""
    cached = load(report_key, cache_dir=cache_dir)
    if cached is not None:
        return cached

    report = compute_fn()
    if not isinstance(report, BenchmarkReport):
        raise TypeError("compute_fn must return a BenchmarkReport")
    store(report_key, report, cache_dir=cache_dir)
    return report


def invalidate(
    report_key: ReportKey,
    *,
    cache_dir: str | Path | None = None,
) -> bool:
    """Remove one cached report entry if it exists."""
    path = cache_path(report_key, cache_dir=cache_dir)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def clear(*, cache_dir: str | Path | None = None) -> int:
    """Remove all cached report JSON entries and return the removal count."""
    root = _cache_root(cache_dir)
    if not root.exists():
        return 0
    removed = 0
    for path in root.glob("*.json"):
        path.unlink()
        removed += 1
    return removed


def cache_path(report_key: ReportKey, *, cache_dir: str | Path | None = None) -> Path:
    """Return the local JSON path for a report key."""
    return _cache_root(cache_dir) / f"{_report_key_digest(report_key)}.json"


def _cache_root(cache_dir: str | Path | None) -> Path:
    return (
        Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
    )


def _report_key_digest(report_key: ReportKey) -> str:
    if isinstance(report_key, EvalCacheKey):
        return report_key.digest
    if isinstance(report_key, str):
        return hashlib.sha256(report_key.encode("utf-8")).hexdigest()
    if isinstance(report_key, Mapping):
        return _hash_json(report_key)
    raise TypeError("report_key must be an EvalCacheKey, mapping, or string")


def _module_source(module: Any) -> bytes:
    source_path = inspect.getsourcefile(module)
    if source_path is not None:
        return Path(source_path).read_bytes()
    try:
        return inspect.getsource(module).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise ValueError(f"cannot derive source hash for {module!r}") from exc


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(
        _plain(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_plain(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CODE_HASH_MODULES",
    "EvalCacheKey",
    "ReportKey",
    "build_report_key",
    "cache_path",
    "clear",
    "default_cache_dir",
    "eval_code_hash",
    "hash_fixture_set",
    "invalidate",
    "load",
    "load_or_compute",
    "store",
]
