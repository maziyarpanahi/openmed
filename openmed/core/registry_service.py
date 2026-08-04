"""Offline model-registry state with SemVer, lineage, and named pointers."""

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

REGISTRY_STATE_SCHEMA_VERSION = 1
REGISTRY_STATE_PATH = (
    Path(__file__).resolve().parents[2] / "gates" / "registry_state.json"
)
MODEL_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "models.jsonl"
REGISTRY_POINTER_NAMES = ("latest", "canary", "last_green")
REGISTRY_LINEAGE_RELATIONS = frozenset({"supersedes", "rolled-back-from"})

_SEMVER_RE = re.compile(
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


Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def semantic_version(repo_id: str) -> str:
    """Return the SemVer represented by the last ``-vN`` repo-id component.

    Existing OpenMed checkpoints commonly use ``-v1`` or ``-v1.0`` before an
    artifact suffix such as ``-mlx``. Unversioned legacy checkpoints map to
    ``0.0.0`` so every manifest row still has a deterministic SemVer value.
    """

    matches = list(_SEMVER_RE.finditer(repo_id.rsplit("/", 1)[-1]))
    if not matches:
        return "0.0.0"
    match = matches[-1]
    major = int(match.group("major"))
    minor = int(match.group("minor") or 0)
    patch = int(match.group("patch") or 0)
    return f"{major}.{minor}.{patch}"


def empty_registry_state() -> dict[str, Any]:
    """Return an empty registry-state document using the current schema."""

    return {"schema_version": REGISTRY_STATE_SCHEMA_VERSION, "families": {}}


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
    """Return pointer, version, and lineage coherence errors without mutating state."""

    try:
        _validate_state_shape(state)
    except RegistryStateError as exc:
        return [str(exc)]

    try:
        rows_by_repo = _rows_by_repo(manifest_rows)
    except RegistryStateError as exc:
        return [str(exc)]
    errors: list[str] = []
    families = state.get("families", {})
    assert isinstance(families, Mapping)  # validated above
    for family, raw_entry in sorted(families.items(), key=lambda item: str(item[0])):
        entry = raw_entry if isinstance(raw_entry, Mapping) else {}
        pointers = entry.get("pointers", {})
        versions = entry.get("versions", {})
        lineage = entry.get("lineage", [])

        for pointer_name in REGISTRY_POINTER_NAMES:
            target = pointers.get(pointer_name)
            if target is None:
                continue
            location = f"families.{family}.pointers.{pointer_name}"
            _append_target_errors(
                errors,
                rows_by_repo,
                family=str(family),
                target=str(target),
                location=location,
            )
            if target not in versions:
                errors.append(f"{location} target lacks a family SemVer entry")

        for repo_id, version in versions.items():
            location = f"families.{family}.versions.{repo_id}"
            _append_target_errors(
                errors,
                rows_by_repo,
                family=str(family),
                target=str(repo_id),
                location=location,
            )
            expected = semantic_version(str(repo_id))
            if str(version) != expected:
                errors.append(
                    f"{location} is {version!r}; expected derived SemVer {expected!r}"
                )

        for index, edge in enumerate(lineage):
            assert isinstance(edge, Mapping)  # validated above
            for endpoint in ("from", "to"):
                target = str(edge[endpoint])
                location = f"families.{family}.lineage[{index}].{endpoint}"
                _append_target_errors(
                    errors,
                    rows_by_repo,
                    family=str(family),
                    target=target,
                    location=location,
                )
                if target not in versions:
                    errors.append(f"{location} target lacks a family SemVer entry")
    return errors


def pointer_targets(state: Mapping[str, Any]) -> dict[str, dict[str, str | None]]:
    """Return a plain family-to-pointer mapping from registry state."""

    _validate_state_shape(state)
    families = state.get("families", {})
    assert isinstance(families, Mapping)
    return {
        str(family): {
            name: (
                None
                if entry["pointers"].get(name) is None
                else str(entry["pointers"][name])
            )
            for name in REGISTRY_POINTER_NAMES
        }
        for family, entry in sorted(families.items(), key=lambda item: str(item[0]))
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

    def pointers(self, family: str | None = None) -> dict[str, Any]:
        """Return all pointer sets, or one requested family pointer set."""

        pointers = pointer_targets(self._state)
        if family is None:
            return pointers
        canonical = self._canonical_family(family, required=False)
        return dict(pointers.get(canonical, _empty_pointers()))

    def lineage(self, family: str) -> list[dict[str, Any]]:
        """Return an isolated lineage history for one model family."""

        canonical = self._canonical_family(family, required=False)
        entry = self._state["families"].get(canonical)
        if entry is None:
            return []
        return copy.deepcopy(list(entry["lineage"]))

    def promote(
        self, repo_id: str, *, gate_report: Any | None = None
    ) -> dict[str, Any]:
        """Promote a releasable checkpoint to ``latest`` and record lineage."""

        row = self._require_manifest_row(repo_id)
        family = str(row["family"])
        report = self._require_releasable_gate(repo_id, family, gate_report)
        candidate = copy.deepcopy(self._state)
        entry = _ensure_family_entry(candidate, family)
        previous = entry["pointers"].get("latest")

        entry["versions"][repo_id] = semantic_version(repo_id)
        if previous is None:
            entry["pointers"]["last_green"] = repo_id
        elif previous != repo_id:
            entry["versions"].setdefault(previous, semantic_version(previous))
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
        family: str,
        name: str,
        target: str,
        *,
        gate_report: Any | None = None,
    ) -> dict[str, Any]:
        """Move one named pointer to a manifest checkpoint with a releasable gate."""

        if name not in REGISTRY_POINTER_NAMES:
            raise RegistryStateError(
                f"unknown pointer {name!r}; expected one of {REGISTRY_POINTER_NAMES}"
            )
        row = self._require_manifest_row(target)
        canonical = self._canonical_family(family)
        if str(row["family"]).casefold() != canonical.casefold():
            raise RegistryStateError(
                f"checkpoint {target!r} belongs to family {row['family']!r}, "
                f"not {canonical!r}"
            )
        report = self._require_releasable_gate(target, canonical, gate_report)

        candidate = copy.deepcopy(self._state)
        entry = _ensure_family_entry(candidate, canonical)
        previous = entry["pointers"].get(name)
        entry["versions"][target] = semantic_version(target)
        entry["pointers"][name] = target
        if name == "latest" and previous not in {None, target}:
            entry["versions"].setdefault(previous, semantic_version(previous))
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
        family: str,
        *,
        gate_report: Any | None = None,
    ) -> dict[str, Any]:
        """Repoint ``latest`` to ``last_green`` and record rollback lineage."""

        canonical = self._canonical_family(family, required=False)
        current_entry = self._state["families"].get(canonical)
        if current_entry is None:
            raise RegistryStateError(f"family has no registry state: {family!r}")
        previous = current_entry["pointers"].get("latest")
        target = current_entry["pointers"].get("last_green")
        if not previous:
            raise RegistryStateError(f"family {canonical!r} has no latest pointer")
        if not target:
            raise RegistryStateError(
                f"family {canonical!r} has no last_green rollback pointer"
            )
        report = self._require_releasable_gate(target, canonical, gate_report)

        candidate = copy.deepcopy(self._state)
        entry = candidate["families"][canonical]
        entry["versions"].setdefault(target, semantic_version(target))
        entry["pointers"]["latest"] = target
        entry["pointers"]["canary"] = None
        if previous != target:
            entry["versions"].setdefault(previous, semantic_version(previous))
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

    def _canonical_family(self, family: str, *, required: bool = True) -> str:
        candidates = {
            str(row.get("family"))
            for row in self._rows
            if isinstance(row.get("family"), str) and row.get("family")
        }
        candidates.update(str(item) for item in self._state["families"])
        for candidate in candidates:
            if candidate.casefold() == family.casefold():
                return candidate
        if required:
            raise RegistryStateError(f"unknown model family: {family!r}")
        return family

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
        family: str,
        supplied: Any | None,
    ) -> Any:
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
        if str(payload.get("family") or "").casefold() != family.casefold():
            raise RegistryGateError("gate report family does not match pointer family")

        row = self._rows_by_repo[repo_id]
        report_tier = payload.get("tier")
        if row.get("tier") is not None and report_tier is not None:
            if str(row["tier"]).casefold() != str(report_tier).casefold():
                raise RegistryGateError(
                    "gate report tier does not match manifest target"
                )
        report_format = str(payload.get("format") or "")
        formats = {str(item).casefold() for item in row.get("formats") or []}
        if report_format and report_format.casefold() not in formats:
            raise RegistryGateError("gate report format does not match manifest target")
        return report

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


def _validate_state_shape(state: Mapping[str, Any]) -> None:
    version = state.get("schema_version")
    if version != REGISTRY_STATE_SCHEMA_VERSION:
        raise RegistryStateError(f"unsupported registry schema_version: {version!r}")
    families = state.get("families")
    if not isinstance(families, Mapping):
        raise RegistryStateError("registry state 'families' must be an object")
    for family, entry in families.items():
        if not isinstance(family, str) or not family:
            raise RegistryStateError("registry family names must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise RegistryStateError(f"registry family {family!r} must be an object")
        pointers = entry.get("pointers")
        versions = entry.get("versions")
        lineage = entry.get("lineage")
        if not isinstance(pointers, Mapping) or set(pointers) != set(
            REGISTRY_POINTER_NAMES
        ):
            raise RegistryStateError(
                f"registry family {family!r} must define exactly "
                f"{REGISTRY_POINTER_NAMES} pointers"
            )
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in pointers.values()
        ):
            raise RegistryStateError(
                f"registry family {family!r} pointer targets must be strings or null"
            )
        if not isinstance(versions, Mapping) or any(
            not isinstance(repo_id, str)
            or not repo_id
            or not isinstance(semver, str)
            or not semver
            for repo_id, semver in versions.items()
        ):
            raise RegistryStateError(
                f"registry family {family!r} versions must be a string mapping"
            )
        if not isinstance(lineage, Sequence) or isinstance(lineage, (str, bytes)):
            raise RegistryStateError(
                f"registry family {family!r} lineage must be an array"
            )
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


def _append_target_errors(
    errors: list[str],
    rows_by_repo: Mapping[str, Mapping[str, Any]],
    *,
    family: str,
    target: str,
    location: str,
) -> None:
    row = rows_by_repo.get(target)
    if row is None:
        errors.append(f"{location} references missing manifest row {target!r}")
        return
    row_family = str(row.get("family") or "")
    if row_family.casefold() != family.casefold():
        errors.append(
            f"{location} references family {row_family!r}, expected {family!r}"
        )


def _empty_pointers() -> dict[str, None]:
    return {name: None for name in REGISTRY_POINTER_NAMES}


def _ensure_family_entry(state: dict[str, Any], family: str) -> dict[str, Any]:
    return state["families"].setdefault(
        family,
        {"versions": {}, "pointers": _empty_pointers(), "lineage": []},
    )


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
