"""Last-green baseline store for release gates.

The baseline JSON schema is intentionally small and versioned:

```
{
  "schema_version": 1,
  "entries": {
    "<family>::<tier>::<format>": {
      "key": "<family>::<tier>::<format>",
      "family": "PII",
      "tier": "Small",
      "format": "mlx-fp",
      "metrics": {"micro_f1": 0.98},
      "reproducibility_hash": "sha256:...",
      "repo_id": "OpenMed/...",
      "released": "2026-06-14"
    }
  }
}
```
"""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


BASELINE_SCHEMA_VERSION = 1
BASELINE_PATH = Path(__file__).resolve().parents[2] / "gates" / "baseline.json"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BaselineError(ValueError):
    """Raised when the baseline store is malformed."""


class BaselineMiss(LookupError):
    """Raised when a requested baseline key is not present."""


def baseline_key(family: str, tier: str | None, format_name: str) -> str:
    """Return the stable store key for ``(family, tier, format)``."""

    return "::".join(
        (
            _normalise_key_part(family),
            _normalise_key_part(tier or "none"),
            _normalise_key_part(format_name),
        )
    )


def load_baseline_store(path: str | Path = BASELINE_PATH) -> dict[str, Any]:
    """Load and validate a baseline store JSON document."""

    baseline_path = Path(path)
    with baseline_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_baseline_store(payload)
    return payload


def write_baseline_store(
    store: Mapping[str, Any],
    path: str | Path = BASELINE_PATH,
) -> Path:
    """Validate and write a baseline store with deterministic formatting."""

    payload = copy.deepcopy(dict(store))
    validate_baseline_store(payload)
    baseline_path = Path(path)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return baseline_path


def get_baseline(
    family: str,
    tier: str | None,
    format_name: str,
    *,
    path: str | Path = BASELINE_PATH,
    store: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return a baseline entry for ``(family, tier, format)`` or ``None``."""

    payload = copy.deepcopy(dict(store)) if store is not None else load_baseline_store(path)
    validate_baseline_store(payload)
    entry = payload["entries"].get(baseline_key(family, tier, format_name))
    return copy.deepcopy(entry) if entry is not None else None


def require_baseline(
    family: str,
    tier: str | None,
    format_name: str,
    *,
    path: str | Path = BASELINE_PATH,
    store: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a baseline entry or raise a clear miss for unknown keys."""

    entry = get_baseline(
        family,
        tier,
        format_name,
        path=path,
        store=store,
    )
    if entry is None:
        key = baseline_key(family, tier, format_name)
        raise BaselineMiss(f"No last-green baseline for key: {key}")
    return entry


def update_baseline_entry(
    path: str | Path,
    *,
    family: str,
    tier: str | None,
    format_name: str,
    metrics: Mapping[str, Any],
    reproducibility_hash: str,
    repo_id: str | None = None,
    source_model_id: str | None = None,
    released: str | None = None,
    git_sha: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Update one last-green baseline entry without changing other keys."""

    baseline_path = Path(path)
    if baseline_path.exists():
        store = load_baseline_store(baseline_path)
    else:
        store = {"schema_version": BASELINE_SCHEMA_VERSION, "entries": {}}

    key = baseline_key(family, tier, format_name)
    entry: dict[str, Any] = {
        "key": key,
        "family": family,
        "tier": tier,
        "format": format_name,
        "metrics": _jsonable(metrics),
        "reproducibility_hash": reproducibility_hash,
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if repo_id is not None:
        entry["repo_id"] = repo_id
    if source_model_id is not None:
        entry["source_model_id"] = source_model_id
    if released is not None:
        entry["released"] = released
    if git_sha is not None:
        entry["git_sha"] = git_sha
    if metadata is not None:
        entry["metadata"] = _jsonable(metadata)

    store["entries"][key] = entry
    write_baseline_store(store, baseline_path)
    return copy.deepcopy(entry)


def validate_baseline_store(store: Mapping[str, Any]) -> None:
    """Validate the documented baseline store schema."""

    if not isinstance(store, Mapping):
        raise BaselineError("baseline store must be a JSON object")
    if store.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise BaselineError(
            f"baseline schema_version must be {BASELINE_SCHEMA_VERSION}"
        )
    entries = store.get("entries")
    if not isinstance(entries, Mapping):
        raise BaselineError("baseline entries must be a JSON object")

    for key, entry in entries.items():
        _validate_entry(str(key), entry)


def _validate_entry(key: str, entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise BaselineError(f"baseline entry {key!r} must be a JSON object")

    required = {
        "key",
        "family",
        "tier",
        "format",
        "metrics",
        "reproducibility_hash",
    }
    missing = sorted(required - set(entry))
    if missing:
        raise BaselineError(f"baseline entry {key!r} missing fields: {missing}")

    if entry["key"] != key:
        raise BaselineError(f"baseline entry {key!r} has mismatched key")
    expected_key = baseline_key(entry["family"], entry["tier"], entry["format"])
    if expected_key != key:
        raise BaselineError(f"baseline entry {key!r} does not match its dimensions")
    if not isinstance(entry["family"], str) or not entry["family"]:
        raise BaselineError(f"baseline entry {key!r} family must be non-empty")
    if entry["tier"] is not None and not isinstance(entry["tier"], str):
        raise BaselineError(f"baseline entry {key!r} tier must be string or null")
    if not isinstance(entry["format"], str) or not entry["format"]:
        raise BaselineError(f"baseline entry {key!r} format must be non-empty")
    if not isinstance(entry["metrics"], Mapping):
        raise BaselineError(f"baseline entry {key!r} metrics must be an object")
    if not _SHA256_RE.fullmatch(str(entry["reproducibility_hash"])):
        raise BaselineError(f"baseline entry {key!r} has invalid reproducibility_hash")
    if "released" in entry and entry["released"] is not None:
        if not isinstance(entry["released"], str) or not _DATE_RE.fullmatch(entry["released"]):
            raise BaselineError(f"baseline entry {key!r} released must be YYYY-MM-DD")


def _normalise_key_part(value: str) -> str:
    return str(value).strip().lower().replace("_", "-") or "none"


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


__all__ = [
    "BASELINE_PATH",
    "BASELINE_SCHEMA_VERSION",
    "BaselineError",
    "BaselineMiss",
    "baseline_key",
    "get_baseline",
    "load_baseline_store",
    "require_baseline",
    "update_baseline_entry",
    "validate_baseline_store",
    "write_baseline_store",
]
