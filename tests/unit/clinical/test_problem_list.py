"""Tests for problem-list deduplication and status reconciliation."""

from __future__ import annotations

import pytest

from openmed.clinical import (
    PROBLEM_LIST_RECONCILIATION_ADVISORY,
    ClinicalAssertion,
    ProblemMention,
    clinical_status_from_assertion,
    deduplicate_problem_list,
)


def test_coded_affirmed_recent_outweighs_historical_and_preserves_offsets():
    deduplicated = deduplicate_problem_list(
        [
            ProblemMention(
                text="diabetes mellitus",
                system="http://snomed.info/sct",
                code="44054006",
                offset=(12, 29),
                negation="affirmed",
                temporality="recent",
            ),
            ProblemMention(
                text="history of diabetes",
                system="http://snomed.info/sct",
                code="44054006",
                offset=(130, 149),
                negation="affirmed",
                temporality="historical",
            ),
        ]
    )

    assert len(deduplicated) == 1
    problem = deduplicated[0]
    assert problem.clinical_status == "active"
    assert problem.mention_count == 2
    assert problem.source_offsets == ((12, 29), (130, 149))
    assert problem.system == "http://snomed.info/sct"
    assert problem.code == "44054006"


def test_all_negated_problem_reconciles_to_refuted_without_being_dropped():
    deduplicated = deduplicate_problem_list(
        [
            ProblemMention(
                text="pneumonia",
                offset=(5, 14),
                negation="negated",
            ),
            ProblemMention(
                text="Pneumonia",
                offset=(50, 59),
                negation="negated",
                temporality="historical",
            ),
        ]
    )

    assert len(deduplicated) == 1
    assert deduplicated[0].clinical_status == "refuted"
    assert deduplicated[0].mention_count == 2
    assert deduplicated[0].source_offsets == ((5, 14), (50, 59))


def test_text_fallback_merges_normalized_text_and_keeps_distinct_problems():
    deduplicated = deduplicate_problem_list(
        [
            ProblemMention(text="stroke", offset=(70, 76)),
            ProblemMention(text=" Diabetes   Mellitus ", offset=(10, 28)),
            ProblemMention(text="diabetes mellitus", offset=(90, 107)),
        ]
    )

    assert [problem.text for problem in deduplicated] == [
        "stroke",
        "Diabetes Mellitus",
    ]
    assert [problem.normalized_text for problem in deduplicated] == [
        "stroke",
        "diabetes mellitus",
    ]
    assert [problem.mention_count for problem in deduplicated] == [1, 2]
    assert deduplicated[0].source_offsets == ((70, 76),)
    assert deduplicated[1].source_offsets == ((10, 28), (90, 107))


@pytest.mark.parametrize(
    ("mentions", "expected_status"),
    [
        (
            [
                ProblemMention(
                    text="myocardial infarction",
                    negation="negated",
                ),
                ProblemMention(
                    text="myocardial infarction",
                    negation="affirmed",
                    temporality="historical",
                ),
            ],
            "inactive",
        ),
        (
            [
                ProblemMention(
                    text="bleeding",
                    negation="affirmed",
                    temporality="hypothetical",
                )
            ],
            "unconfirmed",
        ),
        (
            [
                ProblemMention(
                    text="renal failure",
                    negation="negated",
                    temporality="recent",
                )
            ],
            "refuted",
        ),
    ],
)
def test_status_precedence_rules(mentions, expected_status):
    assert deduplicate_problem_list(mentions)[0].clinical_status == expected_status


def test_mappings_are_supported_for_pipeline_entries():
    deduplicated = deduplicate_problem_list(
        [
            {
                "text": "COPD",
                "system": "http://snomed.info/sct",
                "code": "13645005",
                "start": 3,
                "end": 7,
            },
            {
                "text": "copd",
                "system": " HTTP://SNOMED.INFO/SCT ",
                "code": "13645005",
                "offset": (40, 44),
            },
        ]
    )

    assert len(deduplicated) == 1
    assert deduplicated[0].clinical_status == "active"
    assert deduplicated[0].mention_count == 2
    assert deduplicated[0].source_offsets == ((3, 7), (40, 44))


def test_advisory_constant_documents_heuristic_scope():
    assert "heuristic" in PROBLEM_LIST_RECONCILIATION_ADVISORY
    assert "not an automated clinical decision" in PROBLEM_LIST_RECONCILIATION_ADVISORY


@pytest.mark.parametrize(
    ("assertion", "expected_status"),
    [
        (
            ClinicalAssertion(
                negation="affirmed",
                temporality="recent",
                certainty="uncertain",
            ),
            "unconfirmed",
        ),
        (
            ClinicalAssertion(
                negation="affirmed",
                temporality="historical",
                certainty="certain",
            ),
            "inactive",
        ),
        (
            {
                "negation": "affirmed",
                "temporality": "recent",
                "certainty": "certain",
                "experiencer": "family",
            },
            "unconfirmed",
        ),
    ],
)
def test_clinical_status_derives_from_reconciled_assertion_axes(
    assertion,
    expected_status,
):
    assert clinical_status_from_assertion(assertion) == expected_status


def test_uncertain_problem_mentions_reconcile_to_unconfirmed():
    deduplicated = deduplicate_problem_list(
        [
            ProblemMention(
                text="sepsis",
                negation="affirmed",
                temporality="recent",
                certainty="uncertain",
            )
        ]
    )

    assert deduplicated[0].clinical_status == "unconfirmed"


@pytest.mark.parametrize(
    ("mention", "error"),
    [
        (ProblemMention(text=""), ValueError),
        (ProblemMention(text="sepsis", offset=(8, 4)), ValueError),
        (ProblemMention(text="sepsis", negation="unknown"), ValueError),
        (ProblemMention(text="sepsis", temporality="remote"), ValueError),
        (ProblemMention(text="sepsis", certainty="maybe"), ValueError),
    ],
)
def test_invalid_mentions_raise_clear_errors(mention, error):
    with pytest.raises(error):
        deduplicate_problem_list([mention])
