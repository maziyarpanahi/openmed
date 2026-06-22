"""Manifest rollback helpers for release operations."""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from openmed.core.baseline import (
    BASELINE_PATH,
    baseline_key,
    load_baseline_store,
)
from openmed.core.model_card import render_model_card
from openmed.core.model_registry import MANIFEST_PATH

DEFAULT_CARD_DIR = Path(__file__).resolve().parents[2] / "docs" / "model-cards"
DEFAULT_ROLLBACK_LOG_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "rollback-log.jsonl"
)
DEFAULT_STATUS_PATH = DEFAULT_CARD_DIR / "release-status.json"

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_FORMAT_SUFFIX_RE = re.compile(r"-(?:mlx|coreml|onnx|webgpu|pytorch)$", re.IGNORECASE)
_VERSION_SUFFIX_RE = re.compile(r"-v\d+(?:\.\d+)?$", re.IGNORECASE)


class ManifestRollbackError(RuntimeError):
    """Raised when a manifest rollback cannot be resolved safely."""


@dataclass(frozen=True)
class RollbackResult:
    """Result of a manifest pointer rollback."""

    family: str
    tier: str | None
    format_name: str
    baseline_key: str
    previous_repo_id: str
    active_repo_id: str
    manifest_path: Path
    card_path: Path | None
    status_path: Path | None
    tracking_log_path: Path | None
    changed: bool


def load_manifest_rows(path: str | Path = MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load a JSONL model manifest."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ManifestRollbackError(f"manifest does not exist: {manifest_path}")

    rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ManifestRollbackError(
                    f"invalid JSON in {manifest_path} line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ManifestRollbackError(
                    f"manifest row {line_number} must be a JSON object"
                )
            rows.append(row)

    if not rows:
        raise ManifestRollbackError(f"manifest is empty: {manifest_path}")
    return rows


def write_manifest_rows(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path = MANIFEST_PATH,
) -> Path:
    """Write JSONL manifest rows with stable formatting."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=manifest_path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            for row in rows:
                handle.write(
                    json.dumps(dict(row), sort_keys=False, separators=(",", ":"))
                )
                handle.write("\n")
        os.replace(tmp_path, manifest_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return manifest_path


def _write_text_atomic(path: str | Path, text: str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=output_path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(text)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
    return output_path


def rollback_manifest_pointer(
    *,
    family: str,
    tier: str | None = None,
    format_name: str | None = None,
    manifest_path: str | Path = MANIFEST_PATH,
    baseline_path: str | Path = BASELINE_PATH,
    card_dir: str | Path | None = DEFAULT_CARD_DIR,
    status_path: str | Path | None = DEFAULT_STATUS_PATH,
    tracking_log_path: str | Path | None = DEFAULT_ROLLBACK_LOG_PATH,
    reason: str | None = None,
    dry_run: bool = False,
) -> RollbackResult:
    """Flip the active manifest row to the last-green baseline repo."""

    rows = load_manifest_rows(manifest_path)
    baseline = _resolve_baseline(
        baseline_path=baseline_path,
        family=family,
        tier=tier,
        format_name=format_name,
    )
    resolved_family = str(baseline["family"])
    resolved_tier = baseline.get("tier")
    resolved_format = str(baseline["format"])
    resolved_key = baseline_key(resolved_family, resolved_tier, resolved_format)
    baseline_repo_id = _require_baseline_repo_id(baseline)

    candidates = [
        index
        for index, row in enumerate(rows)
        if _row_matches(row, resolved_family, resolved_tier, resolved_format)
    ]
    if not candidates:
        raise ManifestRollbackError(
            "no manifest row matches "
            f"{resolved_family}/{resolved_tier or 'none'}/{resolved_format}"
        )

    candidates = _lineage_candidates(rows, candidates, baseline)
    current_index = max(candidates, key=lambda index: _active_rank(rows[index], index))
    current_row = copy.deepcopy(rows[current_index])
    baseline_index = next(
        (
            index
            for index, row in enumerate(rows)
            if row.get("repo_id") == baseline_repo_id
        ),
        None,
    )
    target_row = _target_row(
        current_row=current_row,
        baseline_row=rows[baseline_index] if baseline_index is not None else None,
        baseline=baseline,
        format_name=resolved_format,
    )
    previous_repo_id = str(current_row.get("repo_id") or "")
    active_repo_id = str(target_row["repo_id"])
    changed = previous_repo_id != active_repo_id

    new_rows = _flipped_rows(
        rows=rows,
        current_index=current_index,
        baseline_index=baseline_index,
        target_row=target_row,
        previous_row=current_row,
    )
    resolved_card_dir = _resolve_default_output_dir(card_dir, manifest_path)
    card_path = _card_path(
        resolved_card_dir,
        resolved_family,
        resolved_tier,
        resolved_format,
    )
    resolved_status_path = _resolve_default_status_path(
        status_path=status_path,
        card_dir=resolved_card_dir,
    )
    resolved_tracking_path = (
        _resolve_default_tracking_path(tracking_log_path, manifest_path)
        if tracking_log_path is not None
        else None
    )

    if not dry_run:
        if card_path is not None:
            write_model_card(
                target_row,
                card_path=card_path,
                previous_repo_id=previous_repo_id,
                baseline_key_value=resolved_key,
            )
        if resolved_status_path is not None:
            write_release_status_snapshot(
                new_rows,
                active_row=target_row,
                path=resolved_status_path,
                manifest_path=Path(manifest_path),
            )
        write_manifest_rows(new_rows, manifest_path)
        if resolved_tracking_path is not None:
            append_rollback_record(
                path=resolved_tracking_path,
                family=resolved_family,
                tier=resolved_tier,
                format_name=resolved_format,
                baseline_key_value=resolved_key,
                from_repo_id=previous_repo_id,
                to_repo_id=active_repo_id,
                manifest_path=Path(manifest_path),
                card_path=card_path,
                status_path=resolved_status_path,
                reason=reason,
                changed=changed,
            )

    return RollbackResult(
        family=resolved_family,
        tier=resolved_tier,
        format_name=resolved_format,
        baseline_key=resolved_key,
        previous_repo_id=previous_repo_id,
        active_repo_id=active_repo_id,
        manifest_path=Path(manifest_path),
        card_path=card_path,
        status_path=resolved_status_path,
        tracking_log_path=resolved_tracking_path,
        changed=changed,
    )


def write_model_card(
    row: Mapping[str, Any],
    *,
    card_path: str | Path,
    previous_repo_id: str | None = None,
    baseline_key_value: str | None = None,
) -> Path:
    """Write the canonical model card for the active manifest row."""

    del previous_repo_id, baseline_key_value
    return _write_text_atomic(card_path, render_model_card(dict(row)))


def write_release_status_snapshot(
    rows: Iterable[Mapping[str, Any]],
    *,
    active_row: Mapping[str, Any],
    path: str | Path,
    manifest_path: str | Path,
) -> Path:
    """Write a manifest-derived status and leaderboard snapshot."""

    leaderboard = [_leaderboard_entry(row) for row in rows]
    leaderboard.sort(
        key=lambda item: (
            item["micro_f1"] is None,
            -(item["micro_f1"] or 0),
            item["repo_id"],
        )
    )
    payload = {
        "schema_version": 1,
        "generated_from": str(Path(manifest_path)),
        "active": _leaderboard_entry(active_row),
        "leaderboard": leaderboard,
    }
    snapshot_path = Path(path)
    return _write_text_atomic(
        snapshot_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def append_rollback_record(
    *,
    path: str | Path,
    family: str,
    tier: str | None,
    format_name: str,
    baseline_key_value: str,
    from_repo_id: str,
    to_repo_id: str,
    manifest_path: Path,
    card_path: Path | None,
    status_path: Path | None,
    reason: str | None,
    changed: bool,
) -> Path:
    """Append a rollback tracking record as JSONL."""

    record = {
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "family": family,
        "tier": tier,
        "format": format_name,
        "baseline_key": baseline_key_value,
        "from_repo_id": from_repo_id,
        "to_repo_id": to_repo_id,
        "manifest_path": str(manifest_path),
        "card_path": str(card_path) if card_path is not None else None,
        "status_path": str(status_path) if status_path is not None else None,
        "reason": reason,
        "changed": changed,
        "operation": "manifest-pointer-flip",
    }
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
    return log_path


def _resolve_baseline(
    *,
    baseline_path: str | Path,
    family: str,
    tier: str | None,
    format_name: str | None,
) -> dict[str, Any]:
    store = load_baseline_store(baseline_path)
    entries = [
        copy.deepcopy(entry)
        for entry in store["entries"].values()
        if _baseline_matches(entry, family=family, tier=tier, format_name=format_name)
    ]
    if not entries:
        target = "/".join(
            part for part in (family, tier or None, format_name or None) if part
        )
        raise ManifestRollbackError(f"no last-green baseline matches {target}")
    if len(entries) > 1:
        options = ", ".join(sorted(str(entry["key"]) for entry in entries))
        raise ManifestRollbackError(
            "multiple last-green baselines match; pass --tier and --format "
            f"to choose one of: {options}"
        )
    return entries[0]


def _baseline_matches(
    entry: Mapping[str, Any],
    *,
    family: str,
    tier: str | None,
    format_name: str | None,
) -> bool:
    if _norm(entry.get("family")) != _norm(family):
        return False
    if tier is not None and _norm(entry.get("tier")) != _norm(tier):
        return False
    if format_name is not None and _norm(entry.get("format")) != _norm(format_name):
        return False
    return True


def _row_matches(
    row: Mapping[str, Any],
    family: str,
    tier: str | None,
    format_name: str,
) -> bool:
    if _norm(row.get("family")) != _norm(family):
        return False
    if _norm(row.get("tier")) != _norm(tier):
        return False
    return _norm(format_name) in {_norm(item) for item in row.get("formats", [])}


def _lineage_candidates(
    rows: list[dict[str, Any]],
    candidates: list[int],
    baseline: Mapping[str, Any],
) -> list[int]:
    baseline_lineage = {
        _lineage_key(value)
        for value in (baseline.get("repo_id"), baseline.get("source_model_id"))
        if isinstance(value, str) and value
    }
    baseline_lineage.discard("")
    if not baseline_lineage:
        return candidates

    narrowed = [
        index
        for index in candidates
        if _lineage_key(str(rows[index].get("repo_id") or "")) in baseline_lineage
        or _lineage_key(str(rows[index].get("base_model") or "")) in baseline_lineage
    ]
    return narrowed or candidates


def _target_row(
    *,
    current_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any] | None,
    baseline: Mapping[str, Any],
    format_name: str,
) -> dict[str, Any]:
    target = copy.deepcopy(dict(baseline_row or current_row))
    target["repo_id"] = _require_baseline_repo_id(baseline)
    target["family"] = baseline["family"]
    target["tier"] = baseline.get("tier")
    target["reproducibility_hash"] = baseline["reproducibility_hash"]
    if "source_model_id" in baseline:
        target["base_model"] = baseline.get("source_model_id")
    if baseline.get("released") is not None:
        target["released"] = baseline.get("released")
    if isinstance(baseline.get("metrics"), Mapping):
        benchmark = dict(target.get("benchmark") or {})
        benchmark.update(baseline["metrics"])
        target["benchmark"] = benchmark
    formats = [str(item) for item in target.get("formats", [])]
    if _norm(format_name) not in {_norm(item) for item in formats}:
        target["formats"] = [format_name, *formats]
    return target


def _flipped_rows(
    *,
    rows: list[dict[str, Any]],
    current_index: int,
    baseline_index: int | None,
    target_row: dict[str, Any],
    previous_row: dict[str, Any],
) -> list[dict[str, Any]]:
    previous_repo_id = previous_row.get("repo_id")
    target_repo_id = target_row.get("repo_id")
    flipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index == current_index:
            flipped.append(copy.deepcopy(target_row))
        elif baseline_index is not None and index == baseline_index:
            continue
        else:
            flipped.append(copy.deepcopy(row))

    if previous_repo_id != target_repo_id and not any(
        row.get("repo_id") == previous_repo_id for row in flipped
    ):
        flipped.append(copy.deepcopy(previous_row))
    return flipped


def _require_baseline_repo_id(baseline: Mapping[str, Any]) -> str:
    repo_id = baseline.get("repo_id")
    if not isinstance(repo_id, str) or not repo_id:
        raise ManifestRollbackError(
            f"baseline {baseline.get('key')} does not define repo_id"
        )
    return repo_id


def _active_rank(row: Mapping[str, Any], index: int) -> tuple[str, int]:
    released = row.get("released")
    return (str(released) if released is not None else "", index)


def _leaderboard_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = (
        row.get("benchmark") if isinstance(row.get("benchmark"), Mapping) else {}
    )
    micro_f1 = benchmark.get("micro_f1")
    recall = benchmark.get("recall")
    return {
        "repo_id": row.get("repo_id"),
        "family": row.get("family"),
        "tier": row.get("tier"),
        "formats": list(row.get("formats") or []),
        "micro_f1": micro_f1 if isinstance(micro_f1, (int, float)) else None,
        "recall": recall if isinstance(recall, (int, float)) else None,
        "released": row.get("released"),
        "reproducibility_hash": row.get("reproducibility_hash"),
    }


def _card_path(
    card_dir: Path | None,
    family: str,
    tier: str | None,
    format_name: str,
) -> Path | None:
    if card_dir is None:
        return None
    slug = "-".join(
        _slug(part)
        for part in (family, tier or "none", format_name)
        if part is not None
    )
    return Path(card_dir) / f"{slug}.md"


def _resolve_default_output_dir(
    card_dir: str | Path | None,
    manifest_path: str | Path,
) -> Path | None:
    if card_dir is None:
        return None
    if _same_path(card_dir, DEFAULT_CARD_DIR) and not _same_path(
        manifest_path, MANIFEST_PATH
    ):
        return Path(manifest_path).parent / DEFAULT_CARD_DIR.name
    return Path(card_dir)


def _resolve_default_status_path(
    *,
    status_path: str | Path | None,
    card_dir: Path | None,
) -> Path | None:
    if status_path is None:
        return card_dir / DEFAULT_STATUS_PATH.name if card_dir is not None else None
    if (
        card_dir is not None
        and _same_path(status_path, DEFAULT_STATUS_PATH)
        and not _same_path(card_dir, DEFAULT_CARD_DIR)
    ):
        return card_dir / DEFAULT_STATUS_PATH.name
    return Path(status_path)


def _resolve_default_tracking_path(
    tracking_log_path: str | Path,
    manifest_path: str | Path,
) -> Path:
    if _same_path(tracking_log_path, DEFAULT_ROLLBACK_LOG_PATH) and not _same_path(
        manifest_path, MANIFEST_PATH
    ):
        return Path(manifest_path).parent / DEFAULT_ROLLBACK_LOG_PATH.name
    return Path(tracking_log_path)


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def _slug(value: str) -> str:
    slug = _SLUG_RE.sub("-", str(value).lower()).strip("-")
    return slug or "model"


def _lineage_key(repo_id: str) -> str:
    name = repo_id.rsplit("/", 1)[-1].strip().lower()
    while True:
        stripped = _FORMAT_SUFFIX_RE.sub("", name)
        stripped = _VERSION_SUFFIX_RE.sub("", stripped)
        if stripped == name:
            return stripped
        name = stripped


def _norm(value: Any) -> str:
    if value is None:
        return "none"
    return str(value).strip().lower().replace("_", "-")


__all__ = [
    "DEFAULT_CARD_DIR",
    "DEFAULT_ROLLBACK_LOG_PATH",
    "DEFAULT_STATUS_PATH",
    "ManifestRollbackError",
    "RollbackResult",
    "append_rollback_record",
    "load_manifest_rows",
    "rollback_manifest_pointer",
    "write_manifest_rows",
    "write_model_card",
    "write_release_status_snapshot",
]
