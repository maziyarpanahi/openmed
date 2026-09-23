"""Offline OpenMed synthetic clinical-PHI evaluation suite."""

from __future__ import annotations

from typing import Any, Sequence

from openmed.eval.golden import GoldenFixture
from openmed.eval.harness import BenchmarkFixture, ModelRunner, run_benchmark
from openmed.eval.report import BenchmarkReport
from openmed.eval.synth_corpus import (
    CORPUS_ID,
    CORPUS_LICENSE,
    CORPUS_VERSION,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_SEED,
    corpus_content_hash,
    generate_corpus,
    label_distribution,
)

OPENMED_SYNTH = CORPUS_ID
OPENMED_SYNTH_REFERENCE_MODEL = "openmed-synth-reference"
OPENMED_SYNTH_DATASET_CARD = "docs/datasets/openmed-synth-phi.md"
OPENMED_SYNTH_DEFAULT_SEED = DEFAULT_SEED
OPENMED_SYNTH_DEFAULT_CORPUS_SIZE = DEFAULT_CORPUS_SIZE


def load_openmed_synth_fixtures(
    *,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    size: int | None = None,
) -> list[BenchmarkFixture]:
    """Generate and validate the suite through the golden-fixture loader."""

    effective_size = corpus_size if size is None else size
    return [
        GoldenFixture.from_mapping(row).to_benchmark_fixture()
        for row in generate_corpus(seed=seed, size=effective_size)
    ]


def openmed_synth_suite_metadata(
    *,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    size: int | None = None,
) -> dict[str, Any]:
    """Return raw-text-free provenance and distribution metadata."""

    effective_size = corpus_size if size is None else size
    rows = generate_corpus(seed=seed, size=effective_size)
    labels = label_distribution(rows)
    languages = sorted({str(row["language"]) for row in rows})
    return {
        "suite": OPENMED_SYNTH,
        "dataset": CORPUS_ID,
        "version": CORPUS_VERSION,
        "seed": seed,
        "corpus_size": len(rows),
        "record_count": len(rows),
        "content_hash": corpus_content_hash(rows),
        "content_hash_algorithm": "sha256",
        "labels": list(labels),
        "label_distribution": labels,
        "languages": languages,
        "license": CORPUS_LICENSE,
        "redistribution": "public synthetic corpus; no credentialing required",
        "provenance": "Faker and OpenMed clinical-ID providers",
        "synthetic": True,
        "contains_real_phi": False,
        "requires_credentials": False,
        "dataset_card": OPENMED_SYNTH_DATASET_CARD,
    }


def openmed_synth_reference_runner(
    fixture: BenchmarkFixture,
    _model_name: str,
    _device: str,
) -> Sequence[Any]:
    """Return the gold spans for the no-model local smoke benchmark."""

    return fixture.gold_spans


def run_openmed_synth_benchmark(
    *,
    model_name: str = OPENMED_SYNTH_REFERENCE_MODEL,
    device: str = "cpu",
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    runner: ModelRunner | None = None,
    generated_at: str | None = None,
) -> BenchmarkReport:
    """Run the synthetic suite locally without model credentials."""

    fixtures = load_openmed_synth_fixtures(seed=seed, corpus_size=corpus_size)
    return run_benchmark(
        fixtures,
        suite=OPENMED_SYNTH,
        model_name=model_name,
        device=device,
        runner=runner or openmed_synth_reference_runner,
        generated_at=generated_at,
        metadata=openmed_synth_suite_metadata(seed=seed, corpus_size=corpus_size),
    )


__all__ = [
    "OPENMED_SYNTH",
    "OPENMED_SYNTH_DATASET_CARD",
    "OPENMED_SYNTH_DEFAULT_CORPUS_SIZE",
    "OPENMED_SYNTH_DEFAULT_SEED",
    "OPENMED_SYNTH_REFERENCE_MODEL",
    "load_openmed_synth_fixtures",
    "openmed_synth_reference_runner",
    "openmed_synth_suite_metadata",
    "run_openmed_synth_benchmark",
]
