"""Shared validation for frozen inputs, licenses, claims and exceptions in local evidence packs."""

from __future__ import annotations

import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence, Set
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openmed.core.audit import stable_hash


class JourneyReleaseError(ValueError):
    """Raised when release evidence violates the strict input contract."""


ASSET_TYPES = frozenset({"dataset", "model", "software", "vocabulary"})


def resolve_beneath(root: Path, relative: str) -> Path:
    """Resolve a relative evidence path beneath root while rejecting symlink traversal."""
    unresolved = root / relative
    cursor = root
    for part in Path(relative).parts:
        cursor /= part
        if cursor.is_symlink():
            raise JourneyReleaseError("input path must not traverse a symlink")
    candidate = unresolved.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise JourneyReleaseError("input path escapes repo_root") from exc
    return candidate


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def parse_timestamp(value: Any, name: str) -> datetime:
    """Parse release evidence timestamps as whole UTC seconds."""
    if not isinstance(value, str) or not value.endswith("Z"):
        raise JourneyReleaseError(f"{name} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise JourneyReleaseError(f"{name} must be an RFC 3339 UTC timestamp") from exc
    if parsed.tzinfo != timezone.utc or parsed.microsecond != 0:
        raise JourneyReleaseError(f"{name} must use UTC seconds without fractions")
    return parsed


def require_portable_path(value: Any, name: str) -> str:
    """Validate and normalize a portable repository-relative input path."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise JourneyReleaseError(f"{name} must be a portable relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise JourneyReleaseError(f"{name} must be a portable relative path")
    return path.as_posix()


def verify_frozen_inputs(
    root: Path,
    inputs: Sequence[Mapping[str, str]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    """Verify declared local files beneath root and return summaries and mismatch codes."""
    results: list[Mapping[str, Any]] = []
    blockers: list[str] = []
    for item in inputs:
        input_id = item["id"]
        try:
            path = resolve_beneath(
                root.resolve(), require_portable_path(item["path"], "input.path")
            )
            if path.is_symlink() or not path.is_file():
                raise JourneyReleaseError("input is not a regular non-symlink file")
            digest = _file_digest(path)
            size = path.stat().st_size
            verified = hmac.compare_digest(digest, item["sha256"])
        except (OSError, JourneyReleaseError):
            digest = "sha256:" + ("0" * 64)
            size = 0
            verified = False
        if not verified:
            blockers.append(f"frozen_input_mismatch:{input_id}")
        results.append(
            {
                "byte_size": size,
                "digest": digest,
                "id": input_id,
                "origin": item["origin"],
                "verified": verified,
            }
        )
    return tuple(results), tuple(sorted(blockers))


def evaluate_license_catalog(
    licenses: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Summarize normalized assets and flag restricted distribution or training use."""
    blockers: list[str] = []
    counts = {asset_type: 0 for asset_type in sorted(ASSET_TYPES)}
    for item in licenses:
        counts[str(item["asset_type"])] += 1
        if item["redistributable"] is False and item["distribution"] == "bundled":
            blockers.append(f"restricted_asset_bundled:{item['asset_id']}")
        if item["use"] == "train" and item["redistributable"] is False:
            blockers.append(f"restricted_training_asset:{item['asset_id']}")
    if counts["model"] == 0:
        blockers.append("model_license_inventory_empty")
    if counts["dataset"] == 0:
        blockers.append("dataset_license_inventory_empty")
    summary = {
        "asset_count": len(licenses),
        "asset_type_counts": counts,
        "catalog_digest": stable_hash(list(licenses)),
        "restricted_asset_count": sum(
            1 for item in licenses if item["redistributable"] is False
        ),
    }
    return summary, tuple(sorted(blockers))


def claim_and_limitation_blockers(
    claims: Sequence[str],
    limitations: Sequence[str],
    *,
    required_claims: Set[str],
    required_limitations: Set[str],
) -> tuple[str, ...]:
    """List the required codes absent from the supplied policy declarations."""
    blockers = [
        *(f"missing_claim:{value}" for value in sorted(required_claims - set(claims))),
        *(
            f"missing_limitation:{value}"
            for value in sorted(required_limitations - set(limitations))
        ),
    ]
    return tuple(blockers)


def evaluate_exceptions(
    exceptions: Sequence[Mapping[str, str]],
    *,
    evaluated_at: datetime,
) -> tuple[tuple[Mapping[str, str], ...], tuple[str, ...]]:
    """Report unresolved, expired and nonwaivable exceptions at the evaluation time."""
    blockers: list[str] = []
    results: list[Mapping[str, str]] = []
    for item in exceptions:
        code = item["code"]
        expires_at = parse_timestamp(item["expires_at"], "exception.expires_at")
        if item["severity"] in {"high", "critical"}:
            blockers.append(f"nonwaivable_exception:{code}")
        if item["disposition"] != "accepted":
            blockers.append(f"unaccepted_exception:{code}")
        if expires_at <= evaluated_at:
            blockers.append(f"expired_exception:{code}")
        results.append(dict(item))
    return tuple(results), tuple(sorted(set(blockers)))


def canonical_json(payload: Any) -> str:
    """Serialize a value with sorted keys, ASCII escaping and finite numbers."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object and reject repeated keys at any parsing depth."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JourneyReleaseError("duplicate JSON key")
        result[key] = value
    return result
