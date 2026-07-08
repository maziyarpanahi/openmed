"""Deterministic release reproducibility hashes."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

TRAINING_PROVENANCE_FILENAME = "training_provenance.json"
TRAINING_PROVENANCE_SCHEMA_VERSION = "openmed.training_provenance.v1"

_SHA256_DIGEST_RE = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_MODEL_CARD_REPRO_HASH_RE = re.compile(
    r"^\|\s*Reproducibility hash\s*\|\s*`(?P<hash>sha256:[0-9a-f]{64})`\s*\|$",
    re.MULTILINE,
)
_REQUIRED_TRAINING_PROVENANCE_FIELDS = (
    "schema_version",
    "rng_seeds",
    "data_manifest_hash",
    "recipe_config_hash",
    "env_lock_digest",
    "base_model",
    "base_model_revision",
    "git_sha",
    "reproducibility_hash",
)


class ReproducibilityVerificationError(ValueError):
    """Raised when a training provenance record cannot be verified."""


def compute_reproducibility_hash(
    *,
    recipe: Any,
    data_manifest: Any,
    base_model: Any,
    git_sha: str | None = None,
    rng_seeds: Mapping[str, int] | None = None,
    recipe_config_hash: str | None = None,
    env_lock_digest: str | None = None,
) -> str:
    """Return a deterministic ``sha256:`` hash for reproducible artifacts.

    Existing release hashes cover recipe, data manifest, base model, and git SHA.
    Training checkpoints can additionally fold in explicit RNG seeds, the recipe
    config file digest, and the resolved environment lock digest.
    """

    payload = {
        "base_model": _normalise_component(base_model),
        "data_manifest": _normalise_component(data_manifest),
        "git_sha": git_sha or resolve_git_sha(),
        "recipe": _normalise_component(recipe),
    }
    if rng_seeds is not None:
        payload["rng_seeds"] = _normalise_seed_set(rng_seeds)
    if recipe_config_hash is not None:
        payload["recipe_config_hash"] = _normalise_digest(
            recipe_config_hash,
            "recipe_config_hash",
        )
    if env_lock_digest is not None:
        payload["env_lock_digest"] = _normalise_digest(
            env_lock_digest,
            "env_lock_digest",
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def compute_file_digest(path: str | Path) -> str:
    """Return ``sha256:<digest>`` for one file."""

    return f"sha256:{_file_sha256(Path(path))}"


def compute_environment_lock_digest(lock_path: str | Path = "uv.lock") -> str:
    """Return the uv.lock digest used by the reproducible-lock gate."""

    return compute_file_digest(lock_path)


def compute_training_reproducibility_hash(
    *,
    rng_seeds: Mapping[str, int],
    data_manifest_hash: str,
    recipe_config_hash: str,
    env_lock_digest: str,
    base_model: str,
    base_model_revision: str,
    git_sha: str,
) -> str:
    """Return the checkpoint hash from pinned training provenance inputs."""

    recipe_hash = _normalise_digest(recipe_config_hash, "recipe_config_hash")
    data_hash = _normalise_digest(data_manifest_hash, "data_manifest_hash")
    lock_digest = _normalise_digest(env_lock_digest, "env_lock_digest")
    model = {
        "id": _required_string(base_model, "base_model"),
        "revision": _required_string(base_model_revision, "base_model_revision"),
    }
    return compute_reproducibility_hash(
        recipe={"config_hash": recipe_hash},
        data_manifest={"hash": data_hash},
        base_model=model,
        git_sha=_required_string(git_sha, "git_sha"),
        rng_seeds=_normalise_seed_set(rng_seeds),
        recipe_config_hash=recipe_hash,
        env_lock_digest=lock_digest,
    )


def build_training_provenance(
    *,
    rng_seeds: Mapping[str, int],
    data_manifest_hash: str,
    recipe_config_hash: str,
    env_lock_digest: str,
    base_model: str,
    base_model_revision: str,
    git_sha: str | None = None,
    repo_id: str | None = None,
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Build a complete ``training_provenance.json`` payload."""

    resolved_git_sha = git_sha or resolve_git_sha()
    seeds = _normalise_seed_set(rng_seeds)
    data_hash = _normalise_digest(data_manifest_hash, "data_manifest_hash")
    recipe_hash = _normalise_digest(recipe_config_hash, "recipe_config_hash")
    lock_digest = _normalise_digest(env_lock_digest, "env_lock_digest")
    payload: dict[str, Any] = {
        "schema_version": TRAINING_PROVENANCE_SCHEMA_VERSION,
        "base_model": _required_string(base_model, "base_model"),
        "base_model_revision": _required_string(
            base_model_revision,
            "base_model_revision",
        ),
        "data_manifest_hash": data_hash,
        "env_lock_digest": lock_digest,
        "git_sha": _required_string(resolved_git_sha, "git_sha"),
        "rng_seeds": seeds,
        "recipe_config_hash": recipe_hash,
    }
    if repo_id is not None:
        payload["repo_id"] = _required_string(repo_id, "repo_id")
    if checkpoint_id is not None:
        payload["checkpoint_id"] = _required_string(checkpoint_id, "checkpoint_id")
    payload["reproducibility_hash"] = compute_training_reproducibility_hash(
        rng_seeds=seeds,
        data_manifest_hash=data_hash,
        recipe_config_hash=recipe_hash,
        env_lock_digest=lock_digest,
        base_model=payload["base_model"],
        base_model_revision=payload["base_model_revision"],
        git_sha=payload["git_sha"],
    )
    return payload


def write_training_provenance(
    checkpoint_dir: str | Path,
    provenance: Mapping[str, Any],
    *,
    filename: str = TRAINING_PROVENANCE_FILENAME,
) -> Path:
    """Write ``training_provenance.json`` into a checkpoint directory."""

    verified = _normalise_training_provenance(provenance)
    verify_reproducibility(verified)
    output_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(
        json.dumps(
            verified,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def load_training_provenance(path: str | Path) -> dict[str, Any]:
    """Load and validate a training provenance JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ReproducibilityVerificationError("training provenance must be an object")
    return _normalise_training_provenance(payload)


def verify_reproducibility(
    provenance: Mapping[str, Any] | str | Path,
    *,
    manifest_row: Mapping[str, Any] | None = None,
    model_card_text: str | None = None,
) -> str:
    """Re-derive and verify the recorded training reproducibility hash."""

    payload = (
        load_training_provenance(provenance)
        if isinstance(provenance, (str, Path))
        else _normalise_training_provenance(provenance)
    )
    recorded = payload["reproducibility_hash"]
    expected = compute_training_reproducibility_hash(
        rng_seeds=payload["rng_seeds"],
        data_manifest_hash=payload["data_manifest_hash"],
        recipe_config_hash=payload["recipe_config_hash"],
        env_lock_digest=payload["env_lock_digest"],
        base_model=payload["base_model"],
        base_model_revision=payload["base_model_revision"],
        git_sha=payload["git_sha"],
    )
    if recorded != expected:
        raise ReproducibilityVerificationError(
            "recorded reproducibility_hash does not match pinned training inputs"
        )
    if manifest_row is not None:
        _verify_manifest_row_hash(manifest_row, recorded)
    if model_card_text is not None:
        _verify_model_card_hash(model_card_text, recorded)
    return recorded


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
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=30,
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


def _normalise_training_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(provenance)
    missing = [
        field
        for field in _REQUIRED_TRAINING_PROVENANCE_FIELDS
        if field not in payload or payload[field] in (None, "")
    ]
    if missing:
        fields = ", ".join(missing)
        raise ReproducibilityVerificationError(
            f"missing required training provenance fields: {fields}"
        )

    if payload["schema_version"] != TRAINING_PROVENANCE_SCHEMA_VERSION:
        raise ReproducibilityVerificationError(
            "unsupported training provenance schema_version: "
            f"{payload['schema_version']!r}"
        )

    normalised = dict(payload)
    normalised["rng_seeds"] = _normalise_seed_set(payload["rng_seeds"])
    for field in (
        "data_manifest_hash",
        "recipe_config_hash",
        "env_lock_digest",
        "reproducibility_hash",
    ):
        normalised[field] = _normalise_digest(str(payload[field]), field)
    for field in ("base_model", "base_model_revision", "git_sha"):
        normalised[field] = _required_string(payload[field], field)
    if "repo_id" in normalised:
        normalised["repo_id"] = _required_string(normalised["repo_id"], "repo_id")
    if "checkpoint_id" in normalised:
        normalised["checkpoint_id"] = _required_string(
            normalised["checkpoint_id"],
            "checkpoint_id",
        )
    return normalised


def _normalise_seed_set(seeds: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(seeds, Mapping) or not seeds:
        raise ReproducibilityVerificationError("rng_seeds must be a non-empty object")

    normalised: dict[str, int] = {}
    for key, value in sorted(seeds.items(), key=lambda item: str(item[0])):
        seed_name = _required_string(key, "rng_seeds key")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ReproducibilityVerificationError(
                f"rng_seeds.{seed_name} must be an integer"
            )
        normalised[seed_name] = int(value)
    return normalised


def _normalise_digest(value: str, field: str) -> str:
    digest = value.strip()
    if not _SHA256_DIGEST_RE.fullmatch(digest):
        raise ReproducibilityVerificationError(
            f"{field} must match sha256:<64 lowercase hex characters>"
        )
    if digest.startswith("sha256:"):
        return digest
    return f"sha256:{digest}"


def _required_string(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ReproducibilityVerificationError(f"{field} must be a non-empty string")
    return text


def _verify_manifest_row_hash(row: Mapping[str, Any], recorded_hash: str) -> None:
    manifest_hash = row.get("reproducibility_hash")
    if manifest_hash != recorded_hash:
        raise ReproducibilityVerificationError(
            "manifest reproducibility_hash does not match training provenance"
        )

    training_provenance = row.get("training_provenance")
    if isinstance(training_provenance, Mapping):
        embedded_hash = training_provenance.get("reproducibility_hash")
        if embedded_hash is not None and embedded_hash != recorded_hash:
            raise ReproducibilityVerificationError(
                "manifest training_provenance.reproducibility_hash does not "
                "match training provenance"
            )


def _verify_model_card_hash(model_card_text: str, recorded_hash: str) -> None:
    match = _MODEL_CARD_REPRO_HASH_RE.search(model_card_text)
    if match is None:
        raise ReproducibilityVerificationError(
            "model card is missing a reproducibility hash row"
        )
    if match.group("hash") != recorded_hash:
        raise ReproducibilityVerificationError(
            "model card reproducibility hash does not match training provenance"
        )


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


__all__ = [
    "TRAINING_PROVENANCE_FILENAME",
    "TRAINING_PROVENANCE_SCHEMA_VERSION",
    "ReproducibilityVerificationError",
    "build_training_provenance",
    "compute_environment_lock_digest",
    "compute_file_digest",
    "compute_reproducibility_hash",
    "compute_training_reproducibility_hash",
    "load_training_provenance",
    "resolve_git_sha",
    "verify_reproducibility",
    "write_training_provenance",
]
