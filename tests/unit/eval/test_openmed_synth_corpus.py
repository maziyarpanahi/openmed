"""Tests for the offline OpenMed synthetic clinical-PHI corpus."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.cli.main import main
from openmed.core.labels import CANONICAL_LABELS
from openmed.eval.suites import (
    OPENMED_SYNTH,
    load_suite_fixtures,
    openmed_synth_suite_metadata,
)
from scripts.eval.build_openmed_synth_corpus import (
    CORPUS_VERSION,
    DEFAULT_CORPUS_SIZE,
    DEFAULT_SEED,
    corpus_content_hash,
    generate_corpus,
)


def test_generation_is_deterministic_and_seed_sensitive() -> None:
    first = generate_corpus(seed=DEFAULT_SEED, size=4)
    second = generate_corpus(seed=DEFAULT_SEED, size=4)
    different = generate_corpus(seed=DEFAULT_SEED + 1, size=4)

    assert first == second
    assert first != different
    assert corpus_content_hash(first) == corpus_content_hash(second)
    assert corpus_content_hash(first) != corpus_content_hash(different)


def test_generated_rows_have_valid_canonical_spans_and_expected_output() -> None:
    rows = generate_corpus(seed=DEFAULT_SEED, size=3)

    for row in rows:
        text = row["text"]
        assert row["metadata"]["synthetic"] is True
        assert row["metadata"]["contains_real_phi"] is False
        assert row["metadata"]["dataset_version"] == CORPUS_VERSION
        for span in row["gold_spans"]:
            assert span["label"] in CANONICAL_LABELS
            assert text[span["start"] : span["end"]] == span["text"]
        expected = row["metadata"]["expected_output"]["text"]
        assert all(f"[{span['label']}]" in expected for span in row["gold_spans"])


def test_suite_registry_is_local_and_hashes_the_default_corpus() -> None:
    fixtures = load_suite_fixtures(OPENMED_SYNTH)
    metadata = openmed_synth_suite_metadata()

    assert len(fixtures) == DEFAULT_CORPUS_SIZE
    assert metadata["suite"] == OPENMED_SYNTH
    assert metadata["version"] == CORPUS_VERSION
    assert metadata["requires_credentials"] is False
    assert metadata["content_hash"] == corpus_content_hash(
        generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)
    )
    assert metadata["labels"]
    assert metadata["languages"]


def test_dataset_card_version_and_hash_match_generated_corpus() -> None:
    card = Path("docs/datasets/openmed-synth-phi.md").read_text(encoding="utf-8")
    rows = generate_corpus(seed=DEFAULT_SEED, size=DEFAULT_CORPUS_SIZE)

    assert f"| Version | {CORPUS_VERSION} |" in card
    assert f"| Content hash | {corpus_content_hash(rows)} |" in card
    assert f"| Record count | {DEFAULT_CORPUS_SIZE} |" in card
    assert "no real phi" in card.lower()


def test_cli_runs_the_suite_without_model_credentials(capsys) -> None:
    result = main(["benchmark", "pii", "--suite", OPENMED_SYNTH])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["suite"] == OPENMED_SYNTH
    assert payload["model_name"] == "openmed-synth-reference"
    assert payload["fixture_count"] == DEFAULT_CORPUS_SIZE
    assert payload["metadata"]["requires_credentials"] is False
