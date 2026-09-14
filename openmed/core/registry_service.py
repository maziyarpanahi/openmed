"""Offline model-registry state with slot-keyed SemVer, lineage, and pointers.

Registry slots are sparse release channels keyed ``family::tier::format`` (the
:func:`openmed.core.baseline.baseline_key` convention shared by
``gates/baseline.json``, ``gates/rollout_state.json``, and the release ledger).
A slot exists only after an explicit, coordinate-matched promotion; the full
``repo_id`` remains the checkpoint identity, and ``latest`` names the selected
shipping target for that channel rather than the only manifest candidate.

SemVer values are assigned committed state: the first target promoted into a
new slot receives ``1.0.0`` and each new target increments the minor version.
Stored values are validated as SemVer but never recomputed from a repo id —
version-looking tokens such as ``-v1`` are upstream model names, not an
engine-maintained sequence.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import stable_hash
from .baseline import BASELINE_PATH, baseline_key

REGISTRY_STATE_SCHEMA_VERSION = 2
REGISTRY_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "registry_state.json"
)
MODEL_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models.jsonl"
REGISTRY_POINTER_NAMES = ("latest", "canary", "last_green")
REGISTRY_LINEAGE_RELATIONS = frozenset({"supersedes", "rolled-back-from"})

_STORED_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_REPO_SEMVER_RE = re.compile(
    r"(?:^|-)v(?P<major>\d+)"
    r"(?:\.(?P<minor>\d+)(?:\.(?P<patch>\d+))?)?(?=-|$)",
    re.IGNORECASE,
)


class RegistryError(RuntimeError):
    """Base error for offline registry operations."""


class RegistryStateError(RegistryError):
    """Raised when committed registry state is invalid or incoherent."""


class RegistryGateError(RegistryError):
    """Raised when a pointer target lacks matching releasable gate evidence."""


class RegistryMigrationError(RegistryError):
    """Raised when a v1 registry document cannot migrate deterministically."""


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def semantic_version(repo_id: str) -> str:
    """Return the display SemVer suggested by a ``-vN`` repo-id component.

    This is presentation metadata for manifest rows only. Version tokens in
    OpenMed repo ids are part of the upstream model name, not a registry
    sequence, so registry state never derives or checks assigned versions
    against this value. Unversioned names map to ``0.0.0``.
    """

    matches = list(_REPO_SEMVER_RE.finditer(repo_id.rsplit("/", 1)[-1]))
    if not matches:
        return "0.0.0"
    match = matches[-1]
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    patch = int(match.group("patch") or 0)
    return f"{major}.{minor}.{patch}"


def registry_slot_key(family: str, tier: str | None, format_name: str) -> str:
    """Return the normalized ``family::tier::format`` registry slot key."""

    return baseline_key(family, tier, format_name)


def empty_registry_state() -> dict[str, Any]:
    """Return an empty registry-state document using the current schema."""

    return {"schema_version": REGISTRY_STATE_SCHEMA_VERSION, "slots": {}}


def load_registry_state(
    path: str | Path = REGISTRY_STATE_PATH,
    *,
    missing_ok: bool = False,
) -> dict[str, Any]:
    """Load and structurally validate a committed registry-state document."""

    state_path = Path(path)
    if not state_path.is_file():
        if missing_ok:
            return empty_registry_state()
        raise RegistryStateError(f"registry state does not exist: {state_path}")
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryStateError(f"could not load registry state: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise RegistryStateError("registry state must be a JSON object")
    state = copy.deepcopy(dict(payload))
    _validate_state_shape(state)
    return state


def registry_state_errors(
    manifest_rows: Sequence[Mapping[str, Any]],
    state: Mapping[str, Any],
) -> list[str]:
    """Return pointer, checkpoint, and lineage coherence errors without mutating."""

    try:
        _validate_state_shape(state)
    except RegistryStateError as exc:
        return [str(exc)]

    try:
        rows_by_repo = _rows_by_repo(manifest_rows)
    except RegistryStateError as exc:
        return [str(exc)]
    errors: list[str] = []
    slots = state.get("slots", {})
    assert isinstance(slots, Mapping)  # validated above
    for slot, raw_entry in sorted(slots.items(), key=lambda item: str(item[0])):
        entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        checkpoints = entry.get("checkpoints", {})
        pointers = entry.get("pointers", {})
        lineage = entry.get("lineage", [])

        for pointer_name in REGISTRY_POINTER_NAMES:
            target = pointers.get(pointer_name)
            if target is None:
                continue
            location = f"slots.{slot}.pointers.{pointer_name}"
            if target not in checkpoints:
                errors.append(f"{location} target lacks a slot checkpoint entry")
            _append_slot_target_errors(
                errors,
                rows_by_repo,
                slot=str(slot),
                target=str(target),
                location=location,
            )

        seen_versions: dict[str, str] = {}
        for repo_id, version in checkpoints.items():
            location = f"slots.{slot}.checkpoints.{repo_id}"
            _append_slot_target_errors(
                errors,
                rows_by_repo,
                slot=str(slot),
                target=str(repo_id),
                location=location,
            )
            duplicate = seen_versions.get(str(version))
            if duplicate is not None:
                errors.append(
                    f"{location} repeats version {version!r} already assigned "
                    f"to {duplicate!r}"
                )
            else:
                seen_versions[str(version)] = str(repo_id)

        for index, edge in enumerate(lineage):
            assert isinstance(edge, Mapping)  # validated above
            for endpoint in ("from", "to"):
                target = str(edge[endpoint])
                location = f"slots.{slot}.lineage[{index}].{endpoint}"
                if target not in checkpoints:
                    errors.append(f"{location} target lacks a slot checkpoint entry")
                _append_slot_target_errors(
                    errors,
                    rows_by_repo,
                    slot=str(slot),
                    target=target,
                    location=location,
                )
    return errors


def pointer_targets(state: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    """Return a plain slot-to-pointer mapping from registry state."""

    _validate_state_shape(state)
    slots = state.get("slots", {})
    assert isinstance(slots, Mapping)
    return {
        str(slot): {
            name: (
                None
                if entry["pointers"].get(name) is None
                else str(entry["pointers"][name])
            )
            for name in REGISTRY_POINTER_NAMES
        }
        for slot, entry in sorted(slots.items(), key=lambda item: str(item[0]))
    }


def manifest_pii_languages(
    manifest_path: str | Path = MODEL_MANIFEST_PATH,
) -> set[str]:
    """Return PII language codes derived from the canonical local manifest."""

    languages: set[str] = set()
    for row in _load_manifest_rows(Path(manifest_path)):
        repo_id = str(row.get("repo_id") or "").casefold()
        family = str(row.get("family") or "").casefold()
        if family != "pii" and "pii" not in repo_id and "privacy" not in repo_id:
            continue
        raw_languages = row.get("languages")
        if isinstance(raw_languages, Sequence) and not isinstance(
            raw_languages, (str, bytes)
        ):
            languages.update(
                str(language) for language in raw_languages if str(language)
            )
    return languages


def migrate_registry_state(
    state: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    baseline_entries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Migrate a v1 family-keyed document to the v2 slot-keyed schema.

    A committed baseline entry supplies migration coordinate evidence only when
    its ``repo_id`` exactly matches a v1 pointer target and its key matches the
    coordinates of that target's manifest row. All non-null pointers in one v1
    family entry must resolve to the same single slot; zero matches, multiple
    matches, or pointers resolving to different slots raise
    :class:`RegistryMigrationError`. Assigned versions carry over unchanged and
    the migration never invents gate evidence.
    """

    _validate_v1_shape(state)
    rows_by_repo = _rows_by_repo(manifest_rows)
    migrated = empty_registry_state()
    for family, entry in sorted(
        state["families"].items(), key=lambda item: str(item[0])
    ):
        pointers = {
            name: str(entry["pointers"][name])
            for name in REGISTRY_POINTER_NAMES
            if entry["pointers"].get(name) is not None
        }
        if not pointers:
            raise RegistryMigrationError(
                f"families.{family} has checkpoints but no pointer targets to "
                "supply migration coordinates"
            )
        slots_by_target = {
            target: _resolve_migration_slot(
                family=str(family),
                target=target,
                rows_by_repo=rows_by_repo,
                baseline_entries=baseline_entries,
            )
            for target in sorted(set(pointers.values()))
        }
        distinct_slots = sorted(set(slots_by_target.values()))
        if len(distinct_slots) != 1:
            raise RegistryMigrationError(
                f"families.{family} pointers resolve to different slots: "
                f"{', '.join(distinct_slots)}"
            )
        slot = distinct_slots[0]
        if slot in migrated["slots"]:
            raise RegistryMigrationError(
                f"families.{family} resolves to slot {slot!r}, which another "
                "family already claimed"
            )

        versions = entry["versions"]
        stray = sorted(set(versions) - set(pointers.values()))
        if stray:
            raise RegistryMigrationError(
                f"families.{family} carries versions without pointer-target "
                f"coordinate evidence: {', '.join(stray)}"
            )
        checkpoints: dict[str, str] = {}
        for target in sorted(set(pointers.values())):
            version = versions.get(target)
            if version is None:
                raise RegistryMigrationError(
                    f"families.{family} pointer target {target!r} has no "
                    "committed version to carry over"
                )
            if not _STORED_SEMVER_RE.match(str(version)):
                raise RegistryMigrationError(
                    f"families.{family} version {version!r} for {target!r} "
                    "is not valid SemVer"
                )
            checkpoints[target] = str(version)

        lineage = copy.deepcopy(list(entry["lineage"]))
        for index, edge in enumerate(lineage):
            for endpoint in ("from", "to"):
                if str(edge[endpoint]) not in checkpoints:
                    raise RegistryMigrationError(
                        f"families.{family} lineage[{index}].{endpoint} names "
                        f"{edge[endpoint]!r}, which has no slot coordinates"
                    )

        migrated["slots"][slot] = {
            "checkpoints": checkpoints,
            "pointers": {name: pointers.get(name) for name in REGISTRY_POINTER_NAMES},
            "lineage": lineage,
        }
    return migrated


def migrate_registry_state_file(
    *,
    state_path: str | Path = REGISTRY_STATE_PATH,
    manifest_path: str | Path = MODEL_MANIFEST_PATH,
    baseline_path: str | Path = BASELINE_PATH,
) -> dict[str, Any]:
    """Migrate a committed v1 registry file to v2 in place, fail-closed.

    The migrated document is coherence-checked against the manifest before any
    write; every failure mode raises and leaves the committed file unchanged.
    """

    from .baseline import load_baseline_store

    resolved = Path(state_path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryMigrationError(
            f"could not load registry state for migration: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RegistryMigrationError("registry state must be a JSON object")
    if payload.get("schema_version") == REGISTRY_STATE_SCHEMA_VERSION:
        raise RegistryMigrationError(
            "registry state already uses schema_version "
            f"{REGISTRY_STATE_SCHEMA_VERSION}; nothing to migrate"
        )

    rows = _load_manifest_rows(Path(manifest_path))
    store = load_baseline_store(baseline_path)
    entries = store.get("entries", {})
    if not isinstance(entries, Mapping):
        raise RegistryMigrationError("baseline store 'entries' must be an object")
    migrated = migrate_registry_state(payload, rows, entries)
    errors = registry_state_errors(rows, migrated)
    if errors:
        raise RegistryMigrationError(
            "migrated registry state is incoherent: " + "; ".join(errors)
        )
    _write_state_atomic(resolved, migrated)
    return migrated


class RegistryService:
    """Mutate a committed registry state using only local manifest and gate data."""

    def __init__(
        self,
        *,
        manifest_path: str | Path,
        state_path: str | Path = REGISTRY_STATE_PATH,
        gate_reports: Mapping[str, Any] | None = None,
        clock: Clock = _utcnow,
    ) -> None:
        """Load local registry inputs and reject incoherent committed state."""

        self.manifest_path = Path(manifest_path)
        self.state_path = Path(state_path)
        self._rows = _load_manifest_rows(self.manifest_path)
        self._rows_by_repo = _rows_by_repo(self._rows)
        self._gate_reports = dict(gate_reports or {})
        self._clock = clock
        self._state = load_registry_state(self.state_path, missing_ok=True)
        self._validate_coherence(self._state)

    @property
    def state(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible copy of current registry state."""

        return copy.deepcopy(self._state)

    def pointers(self, slot: str | None = None) -> dict[str, Any]:
        """Return all pointer sets, or one requested slot pointer set."""

        pointers = pointer_targets(self._state)
        if slot is None:
            return pointers
        return dict(pointers.get(_canonical_slot(slot), _empty_pointers()))

    def lineage(self, slot: str) -> list[dict[str, Any]]:
        """Return an isolated lineage history for one registry slot."""

        entry = self._state["slots"].get(_canonical_slot(slot))
        if entry is None:
            return []
        return copy.deepcopy(list(entry["lineage"]))

    def promote(
        self, repo_id: str, *, gate_report: Any | None = None
    ) -> dict[str, Any]:
        """Promote a releasable checkpoint to ``latest`` in its gate's slot.

        The slot comes from the gate report's ``(family, tier, format)``
        coordinates, which must match the checkpoint's manifest row exactly —
        this is the only operation that creates a slot.
        """

        self._require_manifest_row(repo_id)
        report, slot = self._require_releasable_gate(repo_id, gate_report)
        candidate = copy.deepcopy(self._state)
        entry = candidate["slots"].setdefault(slot, _empty_slot_entry())
        previous = entry["pointers"].get("latest")

        if repo_id not in entry["checkpoints"]:
            entry["checkpoints"][repo_id] = _next_assigned_version(entry["checkpoints"])
        if previous is None:
            entry["pointers"]["last_green"] = repo_id
        elif previous != repo_id:
            entry["pointers"]["last_green"] = previous
            entry["lineage"].append(
                self._lineage_edge(
                    relation="supersedes",
                    from_repo=previous,
                    to_repo=repo_id,
                    reason="promotion",
                    gate_report=report,
                )
            )
        entry["pointers"]["latest"] = repo_id
        entry["pointers"]["canary"] = None
        self._commit(candidate)
        return self.state

    def flip_pointer(
        self,
        slot: str,
        name: str,
        target: str,
        *,
        gate_report: Any | None = None,
    ) -> dict[str, Any]:
        """Move one named pointer of an existing slot to a releasable checkpoint."""

        if name not in REGISTRY_POINTER_NAMES:
            raise RegistryStateError(
                f"unknown pointer {name!r}; expected one of {REGISTRY_POINTER_NAMES}"
            )
        canonical = self._require_slot(slot)
        self._require_manifest_row(target)
        report, report_slot = self._require_releasable_gate(target, gate_report)
        if report_slot != canonical:
            raise RegistryGateError(
                f"gate report coordinates resolve to slot {report_slot!r}, "
                f"not {canonical!r}"
            )

        candidate = copy.deepcopy(self._state)
        entry = candidate["slots"][canonical]
        previous = entry["pointers"].get(name)
        if target not in entry["checkpoints"]:
            entry["checkpoints"][target] = _next_assigned_version(entry["checkpoints"])
        entry["pointers"][name] = target
        if name == "latest" and previous not in {None, target}:
            entry["lineage"].append(
                self._lineage_edge(
                    relation="supersedes",
                    from_repo=previous,
                    to_repo=target,
                    reason="pointer-flip",
                    gate_report=report,
                )
            )
        self._commit(candidate)
        return self.state

    def rollback(
        self,
        slot: str,
        *,
        gate_report: Any | None = None,
    ) -> dict[str, Any]:
        """Repoint ``latest`` to ``last_green`` and record rollback lineage."""

        canonical = _canonical_slot(slot)
        current_entry = self._state["slots"].get(canonical)
        if current_entry is None:
            raise RegistryStateError(f"slot has no registry state: {slot!r}")
        previous = current_entry["pointers"].get("latest")
        target = current_entry["pointers"].get("last_green")
        if not previous:
            raise RegistryStateError(f"slot {canonical!r} has no latest pointer")
        if not target:
            raise RegistryStateError(
                f"slot {canonical!r} has no last_green rollback pointer"
            )
        report, report_slot = self._require_releasable_gate(target, gate_report)
        if report_slot != canonical:
            raise RegistryGateError(
                f"gate report coordinates resolve to slot {report_slot!r}, "
                f"not {canonical!r}"
            )

        candidate = copy.deepcopy(self._state)
        entry = candidate["slots"][canonical]
        entry["pointers"]["latest"] = target
        entry["pointers"]["canary"] = None
        if previous != target:
            entry["lineage"].append(
                self._lineage_edge(
                    relation="rolled-back-from",
                    from_repo=previous,
                    to_repo=target,
                    reason="rollback",
                    gate_report=report,
                )
            )
        self._commit(candidate)
        return self.state

    def save(self) -> Path:
        """Persist current validated state with canonical stable formatting."""

        self._validate_coherence(self._state)
        return _write_state_atomic(self.state_path, self._state)

    def _require_slot(self, slot: str) -> str:
        canonical = _canonical_slot(slot)
        if canonical not in self._state["slots"]:
            raise RegistryStateError(f"unknown registry slot: {slot!r}")
        return canonical

    def _require_manifest_row(self, repo_id: str) -> dict[str, Any]:
        row = self._rows_by_repo.get(repo_id)
        if row is None:
            raise RegistryStateError(f"checkpoint is absent from manifest: {repo_id}")
        family = row.get("family")
        if not isinstance(family, str) or not family:
            raise RegistryStateError(f"checkpoint has no manifest family: {repo_id}")
        return row

    def _require_releasable_gate(
        self,
        repo_id: str,
        supplied: Any | None,
    ) -> tuple[Any, str]:
        report = supplied if supplied is not None else self._gate_reports.get(repo_id)
        if report is None:
            raise RegistryGateError(
                f"checkpoint lacks a RELEASABLE gate report: {repo_id}"
            )
        payload = _gate_payload(report)
        if payload.get("decision") != "RELEASABLE":
            raise RegistryGateError(
                f"checkpoint gate report is not RELEASABLE: {repo_id}"
            )
        if str(payload.get("repo_id") or "") != repo_id:
            raise RegistryGateError("gate report repo_id does not match pointer target")
        family = payload.get("family")
        if not isinstance(family, str) or not family:
            raise RegistryGateError("gate report names no family coordinate")
        report_format = payload.get("format")
        if not isinstance(report_format, str) or not report_format:
            raise RegistryGateError("gate report names no format coordinate")
        raw_tier = payload.get("tier")
        tier = str(raw_tier) if raw_tier is not None else None

        slot = registry_slot_key(family, tier, report_format)
        if slot not in _row_slot_keys(self._rows_by_repo[repo_id]):
            raise RegistryGateError(
                "gate report coordinates do not match the manifest target: "
                f"{slot!r} is not a slot of {repo_id}"
            )
        return report, slot

    def _lineage_edge(
        self,
        *,
        relation: str,
        from_repo: str,
        to_repo: str,
        reason: str,
        gate_report: Any,
    ) -> dict[str, Any]:
        moment = self._clock()
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        else:
            moment = moment.astimezone(timezone.utc)
        payload = _gate_payload(gate_report)
        report_hash = str(payload.get("repro_hash") or "")
        if not report_hash:
            report_hash = stable_hash(payload)
        return {
            "relation": relation,
            "from": from_repo,
            "to": to_repo,
            "reason": reason,
            "recorded_at": moment.isoformat(),
            "gate_report_hash": report_hash,
        }

    def _validate_coherence(self, state: Mapping[str, Any]) -> None:
        errors = registry_state_errors(self._rows, state)
        if errors:
            raise RegistryStateError("; ".join(errors))

    def _commit(self, candidate: dict[str, Any]) -> None:
        self._validate_coherence(candidate)
        _write_state_atomic(self.state_path, candidate)
        self._state = candidate


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RegistryStateError(f"manifest does not exist: {path}")
    from .model_integrity import verify_manifest_signature_if_present

    verify_manifest_signature_if_present(path)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if not isinstance(row, Mapping):
                    raise RegistryStateError(
                        f"manifest row {line_number} must be a JSON object"
                    )
                rows.append(dict(row))
    except json.JSONDecodeError as exc:
        raise RegistryStateError(f"invalid manifest JSON: {exc}") from exc
    except OSError as exc:
        raise RegistryStateError(f"could not read manifest: {exc}") from exc
    if not rows:
        raise RegistryStateError(f"manifest is empty: {path}")
    return rows


def _rows_by_repo(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            continue
        if repo_id in result:
            raise RegistryStateError(f"duplicate manifest repo_id: {repo_id}")
        result[repo_id] = dict(row)
    return result


def _canonical_slot(slot: str) -> str:
    parts = [part.strip() for part in str(slot).split("::")]
    if len(parts) != 3 or not all(parts):
        raise RegistryStateError(
            f"registry slot must be family::tier::format, got: {slot!r}"
        )
    return registry_slot_key(parts[0], parts[1], parts[2])


def _row_slot_keys(row: Mapping[str, Any]) -> set[str]:
    family = row.get("family")
    if not isinstance(family, str) or not family:
        return set()
    raw_tier = row.get("tier")
    tier = str(raw_tier) if raw_tier is not None else None
    formats = row.get("formats")
    if not isinstance(formats, Sequence) or isinstance(formats, (str, bytes)):
        return set()
    return {
        registry_slot_key(family, tier, str(format_name))
        for format_name in formats
        if str(format_name)
    }


def _resolve_migration_slot(
    *,
    family: str,
    target: str,
    rows_by_repo: Mapping[str, Mapping[str, Any]],
    baseline_entries: Mapping[str, Mapping[str, Any]],
) -> str:
    row = rows_by_repo.get(target)
    if row is None:
        raise RegistryMigrationError(
            f"families.{family} pointer target is absent from the manifest: {target}"
        )
    row_slots = _row_slot_keys(row)
    candidates = sorted(
        str(key)
        for key, entry in baseline_entries.items()
        if isinstance(entry, Mapping)
        and str(entry.get("repo_id") or "") == target
        and str(key) in row_slots
    )
    if not candidates:
        raise RegistryMigrationError(
            f"families.{family} pointer target {target!r} has no committed "
            "baseline entry matching its manifest coordinates"
        )
    if len(candidates) > 1:
        raise RegistryMigrationError(
            f"families.{family} pointer target {target!r} matches multiple "
            f"baseline slots: {', '.join(candidates)}"
        )
    return candidates[0]


def _next_assigned_version(checkpoints: Mapping[str, str]) -> str:
    if not checkpoints:
        return "1.0.0"
    latest = max(
        tuple(int(part) for part in str(version).split("."))
        for version in checkpoints.values()
    )
    return f"{latest[0]}.{latest[1] + 1}.0"


def _validate_state_shape(state: Mapping[str, Any]) -> None:
    version = state.get("schema_version")
    if version == 1:
        raise RegistryStateError(
            "registry state uses retired schema_version 1; run "
            "'python scripts/release/registry_ctl.py migrate' to upgrade it"
        )
    if version != REGISTRY_STATE_SCHEMA_VERSION:
        raise RegistryStateError(f"unsupported registry schema_version: {version!r}")
    slots = state.get("slots")
    if not isinstance(slots, Mapping):
        raise RegistryStateError("registry state 'slots' must be an object")
    for slot, entry in slots.items():
        if not isinstance(slot, str) or not slot:
            raise RegistryStateError("registry slot keys must be non-empty strings")
        try:
            canonical = _canonical_slot(slot)
        except RegistryStateError as exc:
            raise RegistryStateError(str(exc)) from None
        if canonical != slot:
            raise RegistryStateError(
                f"registry slot {slot!r} is not in normalized form {canonical!r}"
            )
        if not isinstance(entry, Mapping):
            raise RegistryStateError(f"registry slot {slot!r} must be an object")
        checkpoints = entry.get("checkpoints")
        pointers = entry.get("pointers")
        lineage = entry.get("lineage")
        if not isinstance(pointers, Mapping) or set(pointers) != set(
            REGISTRY_POINTER_NAMES
        ):
            raise RegistryStateError(
                f"registry slot {slot!r} must define exactly "
                f"{REGISTRY_POINTER_NAMES} pointers"
            )
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in pointers.values()
        ):
            raise RegistryStateError(
                f"registry slot {slot!r} pointer targets must be strings or null"
            )
        if not isinstance(checkpoints, Mapping):
            raise RegistryStateError(
                f"registry slot {slot!r} checkpoints must be a string mapping"
            )
        for repo_id, semver in checkpoints.items():
            if not isinstance(repo_id, str) or not repo_id:
                raise RegistryStateError(
                    f"registry slot {slot!r} checkpoint ids must be non-empty strings"
                )
            if not isinstance(semver, str) or not _STORED_SEMVER_RE.match(semver):
                raise RegistryStateError(
                    f"registry slot {slot!r} checkpoint {repo_id!r} version "
                    f"{semver!r} is not MAJOR.MINOR.PATCH SemVer"
                )
        if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
            raise RegistryStateError(f"registry slot {slot!r} lineage must be an array")
        for edge in lineage:
            if not isinstance(edge, Mapping):
                raise RegistryStateError("registry lineage edges must be objects")
            if edge.get("relation") not in REGISTRY_LINEAGE_RELATIONS:
                raise RegistryStateError(
                    f"unknown registry lineage relation: {edge.get('relation')!r}"
                )
            for field in (
                "from",
                "to",
                "reason",
                "recorded_at",
                "gate_report_hash",
            ):
                if not isinstance(edge.get(field), str) or not edge[field]:
                    raise RegistryStateError(
                        f"registry lineage field {field!r} must be non-empty"
                    )


def _validate_v1_shape(state: Mapping[str, Any]) -> None:
    if state.get("schema_version") != 1:
        raise RegistryMigrationError(
            "migration input must use schema_version 1, got: "
            f"{state.get('schema_version')!r}"
        )
    families = state.get("families")
    if not isinstance(families, Mapping):
        raise RegistryMigrationError("v1 registry state 'families' must be an object")
    for family, entry in families.items():
        if not isinstance(family, str) or not family:
            raise RegistryMigrationError(
                "v1 registry family names must be non-empty strings"
            )
        if not isinstance(entry, Mapping):
            raise RegistryMigrationError(
                f"v1 registry family {family!r} must be an object"
            )
        pointers = entry.get("pointers")
        versions = entry.get("versions")
        lineage = entry.get("lineage")
        if not isinstance(pointers, Mapping) or set(pointers) != set(
            REGISTRY_POINTER_NAMES
        ):
            raise RegistryMigrationError(
                f"v1 registry family {family!r} must define exactly "
                f"{REGISTRY_POINTER_NAMES} pointers"
            )
        if not isinstance(versions, Mapping):
            raise RegistryMigrationError(
                f"v1 registry family {family!r} versions must be a mapping"
            )
        if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
            raise RegistryMigrationError(
                f"v1 registry family {family!r} lineage must be an array"
            )
        for edge in lineage:
            if not isinstance(edge, Mapping) or not all(
                isinstance(edge.get(field), str) and edge.get(field)
                for field in ("from", "to")
            ):
                raise RegistryMigrationError(
                    f"v1 registry family {family!r} lineage edges must name "
                    "'from' and 'to' checkpoints"
                )


def _append_slot_target_errors(
    errors: list[str],
    rows_by_repo: Mapping[str, Mapping[str, Any]],
    *,
    slot: str,
    target: str,
    location: str,
) -> None:
    row = rows_by_repo.get(target)
    if row is None:
        errors.append(f"{location} references missing manifest row {target!r}")
        return
    if slot not in _row_slot_keys(row):
        errors.append(
            f"{location} references {target!r}, whose manifest coordinates do "
            f"not include slot {slot!r}"
        )


def _empty_pointers() -> dict[str, None]:
    return {name: None for name in REGISTRY_POINTER_NAMES}


def _empty_slot_entry() -> dict[str, Any]:
    return {"checkpoints": {}, "pointers": _empty_pointers(), "lineage": []}


def _gate_payload(report: Any) -> dict[str, Any]:
    if isinstance(report, Mapping):
        return copy.deepcopy(dict(report))
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, Mapping):
            return copy.deepcopy(dict(payload))
    return {
        name: getattr(report, name, None)
        for name in ("decision", "repo_id", "family", "tier", "format", "repro_hash")
    }


def _write_state_atomic(path: Path, state: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(state, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path
