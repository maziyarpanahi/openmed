"""Deterministic release reproducibility hashes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping


def compute_reproducibility_hash(
    *,
    recipe: Any,
    data_manifest: Any,
    base_model: Any,
    git_sha: str | None = None,
) -> str:
    """Return ``sha256(recipe + data manifest + base model + git SHA)``."""

    payload = {
        "base_model": _normalise_component(base_model),
        "data_manifest": _normalise_component(data_manifest),
        "git_sha": git_sha or resolve_git_sha(),
        "recipe": _normalise_component(recipe),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def resolve_git_sha(*, cwd: str | Path | None = None) -> str:
    """Resolve the current git SHA from CI environment or local checkout."""

    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _normalise_component(value: Any) -> Any:
    if isinstance(value, Path):
        return _path_component(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, Mapping):
        return {
            str(key): _normalise_component(value[key]) for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_normalise_component(item) for item in value]
    if isinstance(value, set):
        return [_normalise_component(item) for item in sorted(value, key=repr)]
    return value


def _path_component(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {
            "path": path.as_posix(),
            "sha256": _file_sha256(path),
        }
    if path.is_dir():
        return {
            "path": path.as_posix(),
            "files": [
                {
                    "path": file_path.relative_to(path).as_posix(),
                    "sha256": _file_sha256(file_path),
                }
                for file_path in sorted(path.rglob("*"))
                if file_path.is_file()
            ],
        }
    return {"path": path.as_posix(), "missing": True}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["compute_reproducibility_hash", "resolve_git_sha"]
