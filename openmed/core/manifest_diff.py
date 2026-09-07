"""Diff helpers for canonical OpenMed model manifests."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
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
DEFAULT_CATALOG_DOC_PATH = _REPO_ROOT / "docs" / "model-registry.md"

_CATALOG_SECTION_BEGIN = "<!-- BEGIN MANIFEST MODEL TABLE -->"
_CATALOG_SECTION_END = "<!-- END MANIFEST MODEL TABLE -->"
_BENCHMARK_SECTION_BEGIN = "<!-- BEGIN MANIFEST BENCHMARK TABLE -->"
_BENCHMARK_SECTION_END = "<!-- END MANIFEST BENCHMARK TABLE -->"
_CATALOG_SURFACES_BEGIN = "<!-- BEGIN MANIFEST CATALOG SURFACES -->"
_CATALOG_SURFACES_END = "<!-- END MANIFEST CATALOG SURFACES -->"


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
    catalog_doc: str
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
    catalog_doc_path: str | Path = DEFAULT_CATALOG_DOC_PATH,
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
        manifest_entries=len(rows),
        supported_routes=len(supported_languages),
        model_backed=len(model_languages),
    )
    registry = build_registry(rows, state)
    cards = _render_registry_cards(rows, state)
    catalog_doc = _render_catalog_doc(
        Path(catalog_doc_path).read_text(encoding="utf-8"),
        rows,
        model_backed_languages=len(model_languages),
    )
    return RegistrySurfaces(
        readme=readme,
        catalog_doc=catalog_doc,
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
    catalog_doc_path: str | Path = DEFAULT_CATALOG_DOC_PATH,
) -> RegistrySurfaces:
    """Write all committed surfaces derived from the local model manifest."""

    snapshot = build_registry_surfaces(
        manifest_path=manifest_path,
        state_path=state_path,
        readme_path=readme_path,
        catalog_doc_path=catalog_doc_path,
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
    _write_text_atomic(Path(catalog_doc_path), snapshot.catalog_doc)
    return snapshot


def registry_surface_errors(
    *,
    manifest_path: str | Path = MANIFEST_PATH,
    state_path: str | Path = REGISTRY_STATE_PATH,
    readme_path: str | Path = DEFAULT_README_PATH,
    card_dir: str | Path = DEFAULT_REGISTRY_CARD_DIR,
    catalog_doc_path: str | Path = DEFAULT_CATALOG_DOC_PATH,
) -> list[str]:
    """Return drift errors for committed registry-derived surfaces."""

    try:
        snapshot = build_registry_surfaces(
            manifest_path=manifest_path,
            state_path=state_path,
            readme_path=readme_path,
            catalog_doc_path=catalog_doc_path,
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
    catalog_doc = Path(catalog_doc_path)
    if catalog_doc.read_text(encoding="utf-8") != snapshot.catalog_doc:
        errors.append(f"manifest catalog tables are stale: {catalog_doc}")
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
    manifest_entries: int,
    supported_routes: int,
    model_backed: int,
) -> str:
    rendered, entry_count = re.subn(
        r"(?:Local-first runtime|\d[\d,]* manifest entries)",
        f"{manifest_entries:,} manifest entries",
        source,
        count=1,
    )
    rendered, badge_count = re.subn(
        r"\d+ model-backed PII languages",
        f"{model_backed} model-backed PII languages",
        rendered,
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
    if entry_count != 1 or badge_count != 1 or heading_count != 1:
        raise ValueError("README registry-count anchors are missing or ambiguous")
    return rendered


def _render_catalog_doc(
    source: str,
    rows: list[dict[str, Any]],
    *,
    model_backed_languages: int,
) -> str:
    family_counts = Counter(str(row.get("family") or "Unknown") for row in rows)
    family_summary = ", ".join(
        f"{family}={count:,}" for family, count in sorted(family_counts.items())
    )
    benchmark_rows = _manifest_benchmark_rows(rows)
    generated = "\n".join(
        (
            _CATALOG_SURFACES_BEGIN,
            "## Manifest-backed catalog",
            "",
            (
                f"The committed manifest contains {len(rows):,} entries across "
                f"{model_backed_languages} model-backed PII languages. "
                f"Family counts: {family_summary}."
            ),
            "",
            _CATALOG_SECTION_BEGIN,
            _render_model_table(rows),
            _CATALOG_SECTION_END,
            "",
            "## Manifest benchmark evidence",
            "",
            (
                f"The committed manifest contains {len(benchmark_rows):,} model "
                "rows with named benchmark evidence. Missing metrics remain "
                "explicit rather than being inferred."
            ),
            "",
            _BENCHMARK_SECTION_BEGIN,
            _render_benchmark_table(benchmark_rows),
            _BENCHMARK_SECTION_END,
            _CATALOG_SURFACES_END,
        )
    )
    pattern = re.compile(
        rf"{re.escape(_CATALOG_SURFACES_BEGIN)}.*?"
        rf"{re.escape(_CATALOG_SURFACES_END)}",
        flags=re.DOTALL,
    )
    if pattern.search(source):
        rendered, count = pattern.subn(generated, source)
        if count != 1:
            raise ValueError("manifest catalog surface markers are ambiguous")
        return rendered
    if any(
        marker in source
        for marker in (
            _CATALOG_SURFACES_END,
            _CATALOG_SECTION_BEGIN,
            _CATALOG_SECTION_END,
            _BENCHMARK_SECTION_BEGIN,
            _BENCHMARK_SECTION_END,
        )
    ):
        raise ValueError("manifest catalog surface markers are incomplete")
    return source.rstrip() + "\n\n" + generated + "\n"


def _render_model_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | Family | Task | Languages | Tier | Formats |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item.get("family") or ""),
            str(item.get("task") or ""),
            str(item.get("repo_id") or ""),
        ),
    ):
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_value(row.get('repo_id'))}`",
                    _markdown_value(row.get("family")),
                    _markdown_value(row.get("task")),
                    _markdown_list(row.get("languages")),
                    _markdown_value(row.get("tier")),
                    _markdown_list(row.get("formats")),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _manifest_benchmark_rows(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for row in rows:
        raw = row.get("benchmark")
        candidates = raw if isinstance(raw, list) else [raw]
        for benchmark in candidates:
            if not isinstance(benchmark, Mapping):
                continue
            if not any(
                benchmark.get(field) is not None
                for field in ("suite", "dataset", "micro_f1", "recall", "leakage")
            ):
                continue
            result.append((row, dict(benchmark)))
    return sorted(
        result,
        key=lambda item: (
            str(item[0].get("repo_id") or ""),
            str(item[1].get("suite") or ""),
            str(item[1].get("dataset") or ""),
        ),
    )


def _render_benchmark_table(
    rows: list[tuple[dict[str, Any], dict[str, Any]]],
) -> str:
    lines = [
        "| Model | Suite | Dataset | Micro F1 | Recall | Leakage |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model, benchmark in rows:
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{_markdown_value(model.get('repo_id'))}`",
                    _markdown_value(benchmark.get("suite")),
                    _markdown_value(benchmark.get("dataset")),
                    _markdown_metric(benchmark.get("micro_f1")),
                    _markdown_metric(benchmark.get("recall")),
                    _markdown_metric(benchmark.get("leakage")),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _markdown_value(value: Any) -> str:
    if value is None or value == "":
        return "-"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_list(value: Any) -> str:
    if not isinstance(value, (list, tuple)):
        return _markdown_value(value)
    return ", ".join(_markdown_value(item) for item in value) or "-"


def _markdown_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return _markdown_value(value)


def _render_registry_cards(
    rows: list[dict[str, Any]],
    state: Mapping[str, Any],
) -> dict[str, str]:
    rows_by_repo = _rows_by_repo(rows, Path("manifest"))
    cards: dict[str, str] = {}
    for slot, pointers in pointer_targets(state).items():
        slot_slug = re.sub(r"[^a-z0-9]+", "-", slot.casefold()).strip("-")
        for pointer_name, repo_id in pointers.items():
            if repo_id is None:
                continue
            row = rows_by_repo[repo_id]
            marker = f"<!-- Registry pointer: {slot}/{pointer_name} -> {repo_id} -->\n"
            generated_notice = (
                "<!-- Generated from models.jsonl. "
                "Do not edit this file directly. -->\n"
            )
            card = render_model_card(row)
            if generated_notice not in card:
                raise ValueError("model-card generator notice is missing")
            title = f"{slot} {pointer_name.replace('_', '-')} registry checkpoint"
            description = (
                f"Manifest-backed model metadata for the OpenMed {slot} "
                f"{pointer_name.replace('_', '-')} registry pointer targeting "
                f"{repo_id}."
            )
            frontmatter_end = "\n---\n"
            if frontmatter_end not in card:
                raise ValueError("model-card front matter is missing")
            card = card.replace(
                frontmatter_end,
                (
                    f"\ntitle: {json.dumps(title)}"
                    f"\ndescription: {json.dumps(description)}"
                    f"{frontmatter_end}"
                ),
                1,
            )
            cards[f"{slot_slug}-{pointer_name.replace('_', '-')}.md"] = card.replace(
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
