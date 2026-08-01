"""Recall evaluation for the offline alias embedding index.

This suite measures how faithfully the alias embedding index reproduces the
exact brute-force nearest neighbors over a synthetic free vocabulary. It reuses
the committed synthetic grounding fixture (no real terminology), builds the
index with a caller-supplied :class:`~openmed.clinical.grounding.embeddings.AliasEncoder`,
and reports Recall@k of the served candidates against the brute-force reference.

The suite is fully offline and deterministic: the fixture is bundled test data,
the default encoder is the dependency-free
:class:`~openmed.clinical.grounding.embeddings.HashingAliasEncoder`, and no
network access or model download occurs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from openmed.clinical.grounding.embeddings import AliasEncoder, HashingAliasEncoder
from openmed.clinical.grounding.index import brute_force_neighbors, build_index
from openmed.clinical.grounding.vocab import VocabLoader, VocabSource, normalize_alias

GROUNDING_INDEX_RECALL = "grounding_index_recall"

DEFAULT_GROUNDING_VOCAB_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "fixtures"
    / "grounding_vocab_synthetic.jsonl"
)
_FIXTURE_SYSTEMS = ("icd10cm", "rxnorm", "loinc", "hpo", "mesh")


def load_grounding_index_fixtures(
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load the synthetic grounding vocabulary rows used by this suite."""

    fixture_path = Path(path) if path is not None else DEFAULT_GROUNDING_VOCAB_FIXTURE
    return [
        json.loads(line)
        for line in fixture_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grounding_index_recall_metadata(
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Return provenance metadata for the grounding index recall suite."""

    fixture_path = Path(path) if path is not None else DEFAULT_GROUNDING_VOCAB_FIXTURE
    return {
        "suite": GROUNDING_INDEX_RECALL,
        "fixture": fixture_path.name,
        "source": "synthetic committed fixture",
        "redistribution": "safe; no DUA or production terminology",
        "systems": list(_FIXTURE_SYSTEMS),
    }


def _synthetic_loader(
    fixture_path: Path,
    *,
    cache_dir: Path,
) -> VocabLoader:
    sha256 = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    registry = {
        system: VocabSource(system=system, path=fixture_path, sha256=sha256)
        for system in _FIXTURE_SYSTEMS
    }
    return VocabLoader(cache_dir=cache_dir, registry=registry)


def evaluate_grounding_index_recall(
    encoder: AliasEncoder | None = None,
    *,
    cache_dir: str | Path,
    fixture_path: str | Path | None = None,
    systems: Sequence[str] | None = None,
    k: int = 10,
    backend: str = "auto",
) -> dict[str, Any]:
    """Score index Recall@k against the brute-force reference.

    For every alias in the fixture, the alias surface is used as a query; the
    index's top-k concept codes are compared to the exact top-k concept codes
    computed by brute force over the same embedding vectors. Recall is the mean
    fraction of exact top-k concepts recovered by the index.
    """

    resolved_encoder = encoder if encoder is not None else HashingAliasEncoder()
    resolved_systems = tuple(systems) if systems is not None else _FIXTURE_SYSTEMS
    fixture = (
        Path(fixture_path)
        if fixture_path is not None
        else DEFAULT_GROUNDING_VOCAB_FIXTURE
    )

    loader = _synthetic_loader(fixture, cache_dir=Path(cache_dir))
    index = build_index(
        loader, resolved_encoder, systems=resolved_systems, backend=backend
    )
    if index is None:  # pragma: no cover - encoder is never None here
        raise RuntimeError("index build returned no-op with a configured encoder")

    # Rebuild the exact reference over the same (concept, alias) rows.
    reference_vectors: list[tuple[float, ...]] = []
    reference_codes: list[str] = []
    for row in load_grounding_index_fixtures(fixture):
        code = str(row["concept_id"])
        for alias in row["aliases"]:
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            (vector,) = resolved_encoder.encode([normalized])
            reference_vectors.append(vector)
            reference_codes.append(code)

    queries: list[tuple[tuple[float, ...], str]] = []
    for row in load_grounding_index_fixtures(fixture):
        code = str(row["concept_id"])
        for alias in row["aliases"]:
            normalized = normalize_alias(alias)
            if not normalized:
                continue
            (vector,) = resolved_encoder.encode([normalized])
            queries.append((vector, code))

    hits = 0.0
    query_count = 0
    for vector, _code in queries:
        exact_rows = brute_force_neighbors(reference_vectors, vector, k)
        exact_codes = {reference_codes[row] for row, _ in exact_rows}
        if not exact_codes:
            continue
        served_codes = {candidate.code for candidate in index.query(vector, k)}
        hits += len(exact_codes & served_codes) / len(exact_codes)
        query_count += 1

    recall_at_k = hits / query_count if query_count else 0.0
    return {
        "suite": GROUNDING_INDEX_RECALL,
        "k": k,
        "backend": index.backend,
        "query_count": query_count,
        "record_count": index.record_count,
        "recall_at_k": recall_at_k,
        "metadata": {
            **grounding_index_recall_metadata(fixture),
            "encoder_id": resolved_encoder.encoder_id,
            "index_key": index.index_key,
            "vocab_versions": dict(index.vocab_versions),
        },
    }


def run_grounding_index_recall(
    *,
    cache_dir: str | Path,
    encoder: AliasEncoder | None = None,
    fixture_path: str | Path | None = None,
    systems: Sequence[str] | None = None,
    k: int = 10,
    backend: str = "auto",
) -> dict[str, Any]:
    """Run the grounding index recall suite and return its report."""

    return evaluate_grounding_index_recall(
        encoder,
        cache_dir=cache_dir,
        fixture_path=fixture_path,
        systems=systems,
        k=k,
        backend=backend,
    )
