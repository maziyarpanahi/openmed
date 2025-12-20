"""Model discovery and indexing utilities for zero-shot NER models.

The indexer walks a directory hierarchy that contains model artefacts (GLiNER
checkpoints, Hugging Face exports, etc.) and generates a consolidated
``index.json`` file. Each entry records the model identifier, detected family,
domain tags, supported languages, on-disk path, and optional notes. The resulting
metadata powers downstream domain default selection, smoke tests, and CLI UX.

The module intentionally avoids making strong assumptions about the storage
layout; instead it uses lightweight heuristics that can be refined as the model
collection grows. Paths and identifiers stay stable across runs so that other
components can reference models deterministically.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from .families.base import ModelFamily

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PACKAGE_ROOT / "models" / "index.json"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRecord:
    """Normalized metadata for a single model artefact."""

    id: str
    family: str
    domains: Tuple[str, ...] = field(default_factory=tuple)
    languages: Tuple[str, ...] = field(default_factory=tuple)
    path: str = ""
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Convert to a JSON-serialisable dictionary."""
        payload: Dict[str, object] = {
            "id": self.id,
            "family": self.family,
            "domains": list(self.domains),
            "languages": list(self.languages),
            "path": self.path,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(frozen=True)
class ModelIndex:
    """Container for a discovered collection of models."""

    models: Tuple[ModelRecord, ...]
    generated_at: datetime
    source_dir: Path

    def to_dict(self) -> Dict[str, object]:
        timestamp = (
            self.generated_at.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return {
            "meta": {
                "generated_at": timestamp,
                "source_dir": str(self.source_dir),
                "model_count": len(self.models),
                "domain_count": len(self.unique_domains),
            },
            "models": [record.to_dict() for record in self.models],
        }

    @property
    def unique_domains(self) -> Set[str]:
        domains: Set[str] = set()
        for record in self.models:
            domains.update(record.domains)
        return domains


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_index(models_dir: Path) -> ModelIndex:
    """Discover models located under ``models_dir`` and return a structured index.

    Args:
        models_dir: Root directory that houses the model artefacts.

    Returns:
        ``ModelIndex`` instance containing model metadata and generation info.

    Raises:
        FileNotFoundError: If ``models_dir`` does not exist.
        NotADirectoryError: If ``models_dir`` is not a directory.
    """

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    if not models_dir.is_dir():
        raise NotADirectoryError(f"Models path is not a directory: {models_dir}")

    records = tuple(_deduplicate(discover_models(models_dir)))
    return ModelIndex(
        models=records,
        generated_at=datetime.now(timezone.utc),
        source_dir=models_dir,
    )


def discover_models(models_dir: Path) -> Iterator[ModelRecord]:
    """Yield ``ModelRecord`` instances discovered under ``models_dir``."""

    for candidate in _iter_candidate_dirs(models_dir):
        yield _build_record(candidate, models_dir)


def write_index(index: ModelIndex, path: Path, *, pretty: bool = True) -> None:
    """Persist ``index`` to ``path``.

    Args:
        index: The ``ModelIndex`` to serialise.
        path: Destination file path. Parent directories will be created.
        pretty: When True, write indented JSON. Otherwise compact.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = index.to_dict()
    if pretty:
        content = json.dumps(payload, indent=2, sort_keys=True)
    else:
        content = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Discovery heuristics
# ---------------------------------------------------------------------------


MODEL_CORE_FILES = {
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "model.safetensors",
    "tf_model.h5",
    "rust_model.ot",
}

FAMILY_HINTS = {
    "gliner": ModelFamily.GLINER,
    "zero-shot": ModelFamily.GLINER,
    "zeroshot": ModelFamily.GLINER,
    "biomed": ModelFamily.GLINER,
    "gliner2": ModelFamily.GLINER2,
    "fastino": ModelFamily.GLINER2,
    "fastgliner": ModelFamily.GLINER2,
    "v2": ModelFamily.GLINER2,
}

LANGUAGE_HINTS = {
    "multilingual": ("multi",),
    "en": ("en", "english"),
    "es": ("es", "spanish"),
    "de": ("de", "german"),
    "fr": ("fr", "french"),
    "it": ("it", "italian"),
    "pt": ("pt", "portuguese"),
    "zh": ("zh", "chinese"),
    "ja": ("ja", "japanese"),
}

DOMAIN_TOKENS = {
    "biomed": "biomedical",
    "biomedical": "biomedical",
    "clinical": "clinical",
    "genomic": "genomic",
    "genomics": "genomic",
    "finance": "finance",
    "legal": "legal",
    "news": "news",
    "ecom": "ecommerce",
    "ecommerce": "ecommerce",
    "commerce": "ecommerce",
    "cyber": "cybersecurity",
    "cybersec": "cybersecurity",
    "cybersecurity": "cybersecurity",
    "security": "cybersecurity",
    "chem": "chemistry",
    "chemistry": "chemistry",
    "organism": "organism",
    "microbio": "organism",
    "education": "education",
    "acad": "education",
    "social": "social",
    "public": "public_health",
    "public_health": "public_health",
    "health": "public_health",
    "population": "public_health",
    "generic": "generic",
}


def _iter_candidate_dirs(root: Path) -> Iterator[Path]:
    """Traverse ``root`` and yield directories likely to contain models."""

    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if _looks_like_model_dir(path):
            yield path


def _looks_like_model_dir(path: Path) -> bool:
    """Heuristic to decide whether ``path`` represents a model artefact."""

    file_names = {child.name.lower() for child in path.iterdir() if child.is_file()}
    if file_names.intersection(MODEL_CORE_FILES):
        return True

    # Some exports may only contain safetensors with sharded suffixes.
    if any(name.endswith(".safetensors") for name in file_names):
        return True

    # Last resort: directories with metadata.json or config.yaml hints.
    if "metadata.json" in file_names or "config.yaml" in file_names:
        return True

    return False


def _build_record(model_path: Path, root: Path) -> ModelRecord:
    """Create a ``ModelRecord`` for ``model_path``."""

    rel = model_path.relative_to(root)
    identifier = _format_model_id(rel)
    family = _guess_family(rel)
    domains = tuple(sorted(_guess_domains(rel)))
    languages = tuple(sorted(_guess_languages(rel)))
    notes = _extract_notes(model_path)

    return ModelRecord(
        id=identifier,
        family=family,
        domains=domains,
        languages=languages if languages else ("en",),
        path=str(model_path),
        notes=notes,
    )


def _format_model_id(relative_path: Path) -> str:
    """Turn a relative path into a stable identifier."""

    parts = [segment for segment in relative_path.parts if segment]
    normalized = "-".join(part.replace(" ", "_") for part in parts)
    return normalized.lower()


def _guess_family(relative_path: Path) -> str:
    tokens = _tokenise_path(relative_path)
    for token in tokens:
        if token in FAMILY_HINTS:
            return FAMILY_HINTS[token].value
    return ModelFamily.OTHER.value


def _guess_domains(relative_path: Path) -> Set[str]:
    domains: Set[str] = set()
    tokens = _tokenise_path(relative_path)
    for token in tokens:
        match = DOMAIN_TOKENS.get(token)
        if match:
            domains.add(match)
    if not domains:
        domains.add("generic")
    return domains


def _guess_languages(relative_path: Path) -> Set[str]:
    langs: Set[str] = set()
    tokens = _tokenise_path(relative_path)
    for lang, patterns in LANGUAGE_HINTS.items():
        for pattern in patterns:
            if pattern in tokens:
                if lang == "multilingual":
                    langs.add("multi")
                else:
                    langs.add(lang)
                break
    return langs


def _extract_notes(model_path: Path) -> Optional[str]:
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "Metadata file unreadable"
        note = metadata.get("notes") or metadata.get("description")
        if isinstance(note, str):
            return note.strip()
    return None


def _tokenise_path(path: Path) -> Set[str]:
    tokens: Set[str] = set()
    pattern = re.compile(r"[^\w]+")
    for part in path.parts:
        lowered = part.lower()
        split = pattern.split(lowered)
        for token in split:
            if token:
                tokens.add(token)
    return tokens


def _deduplicate(records: Iterable[ModelRecord]) -> Iterator[ModelRecord]:
    unique: Dict[str, ModelRecord] = {}
    for record in records:
        unique.setdefault(record.id, record)
    for record in sorted(unique.values(), key=lambda item: item.id):
        yield record


def load_index(path: Optional[Path] = None) -> ModelIndex:
    """Load a previously generated index from ``path``."""

    index_path = Path(path) if path is not None else DEFAULT_INDEX_PATH
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    payload = json.loads(index_path.read_text(encoding="utf-8"))

    models_payload = payload.get("models", [])
    meta = payload.get("meta", {})

    models: List[ModelRecord] = []
    for entry in models_payload:
        models.append(
            ModelRecord(
                id=entry["id"],
                family=entry.get("family", ModelFamily.OTHER.value),
                domains=tuple(entry.get("domains", [])),
                languages=tuple(entry.get("languages", [])),
                path=entry.get("path", ""),
                notes=entry.get("notes"),
            )
        )

    generated_raw = meta.get("generated_at")
    if isinstance(generated_raw, str):
        generated_str = generated_raw.strip().replace("Z", "+00:00")
        try:
            generated_at = datetime.fromisoformat(generated_str)
        except ValueError:
            generated_at = datetime.now(timezone.utc)
    else:
        generated_at = datetime.now(timezone.utc)

    source_dir_raw = meta.get("source_dir")
    source_dir = (
        Path(source_dir_raw)
        if isinstance(source_dir_raw, str)
        else index_path.parent
    )

    return ModelIndex(
        models=tuple(models),
        generated_at=generated_at.astimezone(timezone.utc),
        source_dir=source_dir,
    )


__all__ = [
    "ModelRecord",
    "ModelIndex",
    "build_index",
    "discover_models",
    "write_index",
    "load_index",
    "DEFAULT_INDEX_PATH",
]
