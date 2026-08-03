"""Diff helpers for canonical OpenMed model manifests."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .language_pack_catalog import (
    DEFAULT_MODEL_PLACEHOLDER_LANGUAGES,
)
from .language_pack_catalog import (
    SUPPORTED_LANGUAGES as REGISTERED_LANGUAGE_PACKS,
)
from .model_card import render_model_card
from .model_registry import MANIFEST_PATH, build_registry, load_manifest_rows
from .registry_service import (
    REGISTRY_STATE_PATH,
    RegistryError,
    load_registry_state,
    pointer_targets,
    registry_state_errors,
)

DIFF_FIELDS: tuple[str, ...] = (
    "tier",
    "param_count",
    "formats",
    "license",
    "benchmark",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_README_PATH = _REPO_ROOT / "README.md"
DEFAULT_REGISTRY_CARD_DIR = _REPO_ROOT / "docs" / "model-cards" / "registry"


@dataclass(frozen=True)
class ManifestFieldChange:
    """Before/after values for one changed manifest field."""

    before: Any
    after: Any

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the change."""
        return {"before": self.before, "after": self.after}


@dataclass(frozen=True)
class ManifestRepoChange:
    """Per-field changes for one repo present in both manifests."""

    repo_id: str
    changes: Mapping[str, ManifestFieldChange]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the repo change."""
        return {
            "repo_id": self.repo_id,
            "changes": {
                field: change.to_dict() for field, change in self.changes.items()
            },
        }


@dataclass(frozen=True)
class ManifestDiff:
    """Structured diff between two canonical model manifests."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[ManifestRepoChange, ...]

    @property
    def has_removed(self) -> bool:
        """Return whether any repo disappeared from the new manifest."""
        return bool(self.removed)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest diff."""
        return {
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": [change.to_dict() for change in self.changed],
        }


@dataclass(frozen=True)
class RegistrySurfaces:
    """Deterministic README, model catalog, language, and card derivations."""

    readme: str
    cards: Mapping[str, str]
    registry_keys: tuple[str, ...]
    supported_languages: tuple[str, ...]


def diff_manifests(old_path: str | Path, new_path: str | Path) -> ManifestDiff:
    """Return a structured diff between two local manifest JSONL files.

    Rows are keyed by ``repo_id``. The diff tracks the release-review fields
    ``tier``, ``param_count``, ``formats``, ``license``, and ``benchmark``.
    ``formats`` and benchmark structures are compared order-insensitively so
    equivalent reordering does not produce a changed repo.
    """

    old_manifest = Path(old_path)
    new_manifest = Path(new_path)
    for manifest in (old_manifest, new_manifest):
        if not manifest.is_file():
            raise FileNotFoundError(manifest)

    old_rows = _rows_by_repo(load_manifest_rows(old_manifest), old_manifest)
    new_rows = _rows_by_repo(load_manifest_rows(new_manifest), new_manifest)

    old_repo_ids = set(old_rows)
    new_repo_ids = set(new_rows)
    added = tuple(sorted(new_repo_ids - old_repo_ids))
    removed = tuple(sorted(old_repo_ids - new_repo_ids))

    changed: list[ManifestRepoChange] = []
    for repo_id in sorted(old_repo_ids & new_repo_ids):
        field_changes: dict[str, ManifestFieldChange] = {}
        old_row = old_rows[repo_id]
        new_row = new_rows[repo_id]
        for field in DIFF_FIELDS:
            old_value = old_row.get(field)
            new_value = new_row.get(field)
            if _normalized_field(field, old_value) == _normalized_field(
                field, new_value
            ):
                continue
            field_changes[field] = ManifestFieldChange(
                before=_display_field(field, old_value),
                after=_display_field(field, new_value),
            )

        if field_changes:
            changed.append(ManifestRepoChange(repo_id=repo_id, changes=field_changes))

    return ManifestDiff(added=added, removed=removed, changed=tuple(changed))


def build_registry_surfaces(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    state_path: str | Path = REGISTRY_STATE_PATH,
    readme_path: str | Path = DEFAULT_README_PATH,
) -> RegistrySurfaces:
    """Render all committed registry-derived surfaces without writing files."""

    rows = load_manifest_rows(Path(manifest_path))
    state = load_registry_state(state_path)
    errors = registry_state_errors(rows, state)
    if errors:
        raise ValueError("registry state is incoherent: " + "; ".join(errors))

    readme_source = Path(readme_path).read_text(encoding="utf-8")
    model_languages = _manifest_pii_languages(rows)
    supported_languages = model_languages | set(DEFAULT_MODEL_PLACEHOLDER_LANGUAGES)
    if supported_languages != set(REGISTERED_LANGUAGE_PACKS):
        raise ValueError(
            "manifest-derived PII languages drift from registered language packs"
        )
    readme = _render_readme_counts(
        readme_source,
        supported_routes=len(supported_languages),
        model_backed=len(model_languages),
    )
    registry = build_registry(rows, state)
    cards = _render_registry_cards(rows, state)
    return RegistrySurfaces(
        readme=readme,
        cards=cards,
        registry_keys=tuple(sorted(registry)),
        supported_languages=tuple(sorted(supported_languages)),
    )


def regenerate_registry_surfaces(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    state_path: str | Path = REGISTRY_STATE_PATH,
    readme_path: str | Path = DEFAULT_README_PATH,
    card_dir: str | Path = DEFAULT_REGISTRY_CARD_DIR,
) -> RegistrySurfaces:
    """Write registry-derived README counts and pointer-selected model cards."""

    snapshot = build_registry_surfaces(
        manifest_path=manifest_path,
        state_path=state_path,
        readme_path=readme_path,
    )
    resolved_readme = Path(readme_path)
    _write_text_atomic(resolved_readme, snapshot.readme)

    resolved_card_dir = Path(card_dir)
    resolved_card_dir.mkdir(parents=True, exist_ok=True)
    expected_names = set(snapshot.cards)
    for stale_path in resolved_card_dir.glob("*.md"):
        if stale_path.name not in expected_names:
            stale_path.unlink()
    for filename, content in snapshot.cards.items():
        _write_text_atomic(resolved_card_dir / filename, content)
    return snapshot


def registry_surface_errors(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    state_path: str | Path = REGISTRY_STATE_PATH,
    readme_path: str | Path = DEFAULT_README_PATH,
    card_dir: str | Path = DEFAULT_REGISTRY_CARD_DIR,
) -> list[str]:
    """Return drift errors for committed registry-derived surfaces."""

    try:
        snapshot = build_registry_surfaces(
            manifest_path=manifest_path,
            state_path=state_path,
            readme_path=readme_path,
        )
    except (OSError, ValueError, RegistryError) as exc:
        return [str(exc)]

    errors: list[str] = []
    readme = Path(readme_path)
    if readme.read_text(encoding="utf-8") != snapshot.readme:
        errors.append(f"README registry counts are stale: {readme}")
    cards = Path(card_dir)
    existing_names = (
        {path.name for path in cards.glob("*.md")} if cards.is_dir() else set()
    )
    if existing_names != set(snapshot.cards):
        errors.append("registry model-card file set is stale")
    for filename, content in snapshot.cards.items():
        path = cards / filename
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            errors.append(f"registry model card is stale: {path}")
    return errors


def _rows_by_repo(
    rows: list[dict[str, Any]], manifest_path: Path
) -> dict[str, dict[str, Any]]:
    by_repo: dict[str, dict[str, Any]] = {}
    for line_number, row in enumerate(rows, start=1):
        repo_id = row.get("repo_id")
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError(
                f"Manifest row in {manifest_path} line {line_number} has no repo_id"
            )
        if repo_id in by_repo:
            raise ValueError(f"Duplicate repo_id in {manifest_path}: {repo_id}")
        by_repo[repo_id] = row
    return by_repo


def _manifest_pii_languages(rows: list[dict[str, Any]]) -> set[str]:
    languages: set[str] = set()
    for row in rows:
        repo_id = str(row.get("repo_id") or "").casefold()
        family = str(row.get("family") or "").casefold()
        if family != "pii" and "pii" not in repo_id and "privacy" not in repo_id:
            continue
        raw_languages = row.get("languages")
        if isinstance(raw_languages, (list, tuple)):
            languages.update(str(language) for language in raw_languages if language)
    return languages


def _render_readme_counts(
    source: str,
    *,
    supported_routes: int,
    model_backed: int,
) -> str:
    rendered, badge_count = re.subn(
        r"\d+ model-backed PII languages",
        f"{model_backed} model-backed PII languages",
        source,
        count=1,
    )
    rendered, heading_count = re.subn(
        r"## Multilingual PII \(\d+ supported routes; \d+ model-backed\)",
        (
            "## Multilingual PII "
            f"({supported_routes} supported routes; {model_backed} model-backed)"
        ),
        rendered,
        count=1,
    )
    if badge_count != 1 or heading_count != 1:
        raise ValueError("README registry-count anchors are missing or ambiguous")
    return rendered


def _render_registry_cards(
    rows: list[dict[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, str]:
    rows_by_repo = _rows_by_repo(rows, Path("manifest"))
    cards: dict[str, str] = {}
    for family, pointers in pointer_targets(state).items():
        family_slug = re.sub(r"[^a-z0-9]+", "-", family.casefold()).strip("-")
        for pointer_name, repo_id in pointers.items():
            if repo_id is None:
                continue
            row = rows_by_repo[repo_id]
            marker = (
                f"<!-- Registry pointer: {family}/{pointer_name} -> {repo_id} -->\n"
            )
            generated_notice = (
                "<!-- Generated from models.jsonl. "
                "Do not edit this file directly. -->\n"
            )
            card = render_model_card(row)
            if generated_notice not in card:
                raise ValueError("model-card generator notice is missing")
            cards[f"{family_slug}-{pointer_name.replace('_', '-')}.md"] = card.replace(
                generated_notice,
                generated_notice + marker,
                1,
            )
    return dict(sorted(cards.items()))


def _write_text_atomic(path: Path, content: str) -> None:
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
            handle.write(content)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _normalized_field(field: str, value: Any) -> Any:
    if field == "formats":
        return _normalized_formats(value)
    if field == "benchmark":
        return _normalized_structured(value)
    return value


def _display_field(field: str, value: Any) -> Any:
    if field == "formats":
        return list(_normalized_formats(value))
    return value


def _normalized_formats(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        return (str(value),)
    return tuple(sorted({str(item) for item in value}))


def _normalized_structured(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_structured(value[key])
            for key in sorted(value, key=str)
        }
    if isinstance(value, (list, tuple)):
        encoded_items = {
            json.dumps(
                _normalized_structured(item),
                sort_keys=True,
                separators=(",", ":"),
            )
            for item in value
        }
        return [json.loads(item) for item in sorted(encoded_items)]
    return value
