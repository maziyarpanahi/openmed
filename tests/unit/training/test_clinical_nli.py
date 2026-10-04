"""Offline split, partial-label and synthetic training-provenance tests."""

from __future__ import annotations

import csv

import pytest
import yaml

from openmed.training.clinical_nli import (
    LABELS,
    PARTIAL_LABEL,
    NLIPair,
    corpus_fingerprint,
    grouped_split,
    load_bionli_csv,
    load_training_recipe,
    remove_conflicting_public_groups,
    synthetic_clinical_pairs,
)


def test_all_source_groups_and_pairs_are_disjoint_and_deterministic():
    pairs = synthetic_clinical_pairs()
    first = grouped_split(pairs)
    second = grouped_split(reversed(pairs))
    assert len({pair.pair_digest for pair in pairs}) == len(pairs)
    seen_groups = set()
    seen_pairs = set()
    for name, rows in first.items():
        assert rows
        assert corpus_fingerprint(rows) == corpus_fingerprint(second[name])
        assert {pair.label for pair in rows} == set(LABELS)
        assert {p for pair in rows for p in pair.phenomena} == {
            "negation",
            "temporality",
            "experiencer",
            "numbers",
            "medication_status",
        }
        groups = {pair.source_group for pair in rows}
        fingerprints = {pair.pair_digest for pair in rows}
        assert not groups & seen_groups
        assert not fingerprints & seen_pairs
        seen_groups |= groups
        seen_pairs |= fingerprints
        assert all(pair.synthetic for pair in rows)


def test_duplicates_are_removed_before_splitting():
    pair = NLIPair("synthetic fact", "synthetic claim", "neutral", "group")
    assert sum(map(len, grouped_split([pair, pair]).values())) == 1


@pytest.mark.parametrize("label,group", [("entailment", "group"), ("neutral", "other")])
def test_conflicting_duplicate_cannot_hide_in_a_different_split(label, group):
    pairs = [
        NLIPair("synthetic fact", "synthetic claim", "neutral", "group"),
        NLIPair(" synthetic  fact ", "synthetic claim", label, group),
    ]
    with pytest.raises(ValueError, match="conflicting provenance"):
        grouped_split(pairs)


def test_biomedical_negative_labels_are_partial_not_invented_contradictions(tmp_path):
    path = tmp_path / "public-schema.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["pmid", "supp_set", "conclusion", "label_cat"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "pmid": "1",
                "supp_set": "Synthetic premise.",
                "conclusion": "Synthetic supported claim.",
                "label_cat": "pos",
            }
        )
        writer.writerow(
            {
                "pmid": "1",
                "supp_set": "Synthetic premise.",
                "conclusion": "Synthetic perturbed claim.",
                "label_cat": "SEP",
            }
        )
    rows = load_bionli_csv(path)
    assert [pair.label for pair in rows] == ["entailment", PARTIAL_LABEL]
    assert rows[0].source_group == rows[1].source_group
    assert not rows[0].synthetic
    assert "Synthetic premise" not in corpus_fingerprint(rows)


def test_unsupported_biomedical_schema_fails_without_echoing_input(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("patient,identifier\nprivate,sensitive\n")
    with pytest.raises(ValueError, match="unsupported public biomedical schema") as exc:
        load_bionli_csv(path)
    assert "private" not in str(exc.value)


def test_public_conflicts_remove_entire_connected_source_groups_before_split():
    pairs = [
        NLIPair("same", "claim", "entailment", "paper-a"),
        NLIPair("same", "claim", PARTIAL_LABEL, "paper-a"),
        NLIPair("other premise", "other claim", "entailment", "paper-a"),
        NLIPair("clean", "supported", "entailment", "paper-b"),
    ]
    rows, counts = remove_conflicting_public_groups(pairs)
    assert rows == [pairs[-1]]
    assert counts == {"excluded_groups": 1, "excluded_pairs": 3}
    assert sum(map(len, grouped_split(rows).values())) == 1


@pytest.mark.parametrize("label", ["unsupported", "abstention"])
def test_training_has_three_states_plus_an_explicit_partial_label(label):
    with pytest.raises(ValueError, match="unsupported training label"):
        NLIPair("Synthetic fact", "Synthetic claim", label, "group")


def test_checked_in_recipe_matches_the_audited_local_training_parameters():
    recipe = load_training_recipe()
    assert recipe["training"]["optimizer"] == "AdamW"
    assert recipe["training"]["learning_rate"] == 5e-5
    assert recipe["training"]["batch_size"] == 16
    assert recipe["training"]["sequence_length"] == 512
    assert recipe["training"]["bitwise_mps_determinism_claimed"] is False
    assert recipe["candidate_quality"]["negation_false_entailment_count_maximum"] == 0


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("base_model", "revision", "main"),
        ("data", "patient_data_allowed", True),
        ("data", "mednli_download_allowed", True),
        ("data", "mednli", "training"),
        ("publication", "enabled", True),
        ("training", "paid_compute_authorized", True),
        ("training", "gradient_accumulation_steps", 2),
    ],
)
def test_recipe_cannot_silently_expand_privacy_or_compute_scope(
    tmp_path, section, field, value
):
    recipe = load_training_recipe()
    recipe[section][field] = value
    path = tmp_path / "invalid-recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    with pytest.raises(ValueError, match="audited local-only provenance"):
        load_training_recipe(path)
