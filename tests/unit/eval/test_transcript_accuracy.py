"""Hand-computed synthetic transcript accuracy and privacy controls."""

import json
import os
import subprocess
import sys
from dataclasses import asdict

import pytest

from openmed.eval.transcript_accuracy import (
    ClinicalTermClass,
    ReferenceTermSpan,
    normalize_transcript,
    score_transcript,
    transcript_accuracy_report,
)


@pytest.mark.parametrize(
    "reference,hypothesis,word_counts,char_counts",
    [
        ("a b", "a x b", (2, 3, 0, 0, 1), (2, 3, 0, 0, 1)),
        ("a b", "a", (2, 1, 0, 1, 0), (2, 1, 0, 1, 0)),
        ("a b", "a c", (2, 2, 1, 0, 0), (2, 2, 1, 0, 0)),
        ("a", "x y z", (1, 3, 1, 0, 2), (1, 3, 1, 0, 2)),
        ("", "", (0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
        ("  \n", "x y", (0, 2, 0, 0, 2), (0, 2, 0, 0, 2)),
        ("Été", "E\u0301TE\u0301", (1, 1, 0, 0, 0), (3, 3, 0, 0, 0)),
        ("5 mg", "50 mg", (2, 2, 1, 0, 0), (3, 4, 0, 0, 1)),
        ("a b", "b a", (2, 2, 2, 0, 0), (2, 2, 2, 0, 0)),
        ("💊", "💉", (1, 1, 1, 0, 0), (1, 1, 1, 0, 0)),
    ],
)
def test_hand_computed_edits(reference, hypothesis, word_counts, char_counts):
    score = score_transcript(reference, hypothesis)
    assert tuple(asdict(score.words).values()) == word_counts
    assert tuple(asdict(score.characters).values()) == char_counts
    for counts in (score.words, score.characters):
        assert counts.rate == (
            counts.errors / counts.reference if counts.reference else None
        )


@pytest.mark.parametrize("language", ["en", "es", "fr", "de"])
def test_normalization_is_conservative_and_versioned(language):
    assert (
        normalize_transcript(" \tÀDA\u00a0 5.0 MG\n", language=language) == "àda 5.0 mg"
    )
    assert normalize_transcript("Straße STRASSE", language=language) == "straße strasse"
    assert score_transcript(".5 mg", "5 mg", language=language).words.errors == 1
    assert score_transcript("no pain", "pain", language=language).words.deletions == 1
    assert score_transcript("5 mg", "5 µg", language=language).words.errors == 1
    assert score_transcript("5", "five", language=language).words.errors == 1
    assert score_transcript("AB-123", "AB123", language=language).words.errors == 1
    assert score_transcript("café", "cafe", language=language).characters.errors == 1


def _term(score, term_class):
    return next(t for t in score.terms if t.term_class == term_class)


def test_low_wer_dose_change_is_visible_in_separate_slice():
    reference = "today the patient will take syntheticdrug 5 mg daily without pain"
    spans = [
        ReferenceTermSpan(
            reference.index("syntheticdrug"),
            reference.index(" 5"),
            ClinicalTermClass.MEDICATION,
        ),
        ReferenceTermSpan(
            reference.index("5"), reference.index("5") + 1, ClinicalTermClass.DOSE
        ),
        ReferenceTermSpan(
            reference.index("mg"), reference.index("mg") + 2, ClinicalTermClass.UNIT
        ),
        ReferenceTermSpan(
            reference.index("without"),
            reference.index("without") + 7,
            ClinicalTermClass.NEGATION,
        ),
    ]
    score = score_transcript(reference, reference.replace("5", "50"), spans=spans)
    assert score.words.rate == 1 / 11
    assert _term(score, ClinicalTermClass.DOSE).errors == 1
    for term_class in (
        ClinicalTermClass.UNIT,
        ClinicalTermClass.MEDICATION,
        ClinicalTermClass.NEGATION,
    ):
        assert _term(score, term_class).errors == 0
    report = transcript_accuracy_report([score] * 5, bootstrap_resamples=100)
    assert report["wer"]["rate"] == 1 / 11
    assert report["clinical_terms"]["dose"]["rate"] == 1
    assert report["clinical_terms"]["unit"]["rate"] == 0


@pytest.mark.parametrize("hypothesis", ["5 extra mg", "50 mg", "mg", "5"])
def test_multitoken_term_counts_once_and_interior_insertion_is_an_error(hypothesis):
    score = score_transcript(
        "5 mg", hypothesis, spans=[ReferenceTermSpan(0, 4, ClinicalTermClass.DOSE)]
    )
    assert _term(score, ClinicalTermClass.DOSE).size == 1
    assert _term(score, ClinicalTermClass.DOSE).errors == 1


def test_boundary_insertions_are_not_assigned_to_neighboring_terms():
    score = score_transcript(
        "5 mg",
        "extra 5 mg extra",
        spans=[ReferenceTermSpan(0, 4, ClinicalTermClass.DOSE)],
    )
    assert score.words.insertions == 2
    assert _term(score, ClinicalTermClass.DOSE).errors == 0


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        ("Élodie", "E\u0301lodie"),
        ("用户-123", "用户-124"),
        ("Ada-007", ""),
        ("ID-456", "ID-456"),
    ],
)
def test_identifier_recall_and_original_unicode_offsets(reference, hypothesis):
    score = score_transcript(
        reference,
        hypothesis,
        spans=[ReferenceTermSpan(0, len(reference), ClinicalTermClass.IDENTIFIER)],
    )
    expected = 0 if reference in ("用户-123", "Ada-007") else 1
    report = transcript_accuracy_report([score] * 5, bootstrap_resamples=100)
    cell = report["clinical_terms"]["identifier"]
    assert cell["size"] == 5
    assert cell["recall"] == expected
    assert cell["recall_ci95"] == [expected, expected]


def test_cross_class_overlap_allowed_but_duplicate_or_partial_spans_rejected():
    score = score_transcript(
        "Ada",
        "Ada",
        spans=[
            ReferenceTermSpan(0, 3, c)
            for c in (ClinicalTermClass.IDENTIFIER, ClinicalTermClass.MEDICATION)
        ],
    )
    assert sum(t.size for t in score.terms) == 2
    for spans in (
        [ReferenceTermSpan(0, 2, ClinicalTermClass.IDENTIFIER)],
        [ReferenceTermSpan(0, 4, ClinicalTermClass.IDENTIFIER)],
        [ReferenceTermSpan(0, 3, ClinicalTermClass.IDENTIFIER)] * 2,
    ):
        with pytest.raises(ValueError):
            score_transcript("Ada", "Ada", spans=spans)
    with pytest.raises(ValueError):
        score_transcript(
            "  ", "", spans=[ReferenceTermSpan(0, 1, ClinicalTermClass.IDENTIFIER)]
        )


def test_streaming_append_only_and_last_partial_revision():
    unchanged = score_transcript("a b c", "a b c", partials=["a", "a b"])
    assert unchanged.revision_tokens == 0
    assert unchanged.revision_opportunities == 3
    changed = score_transcript("a b c", "a b c", partials=["a b", "a x", "a x c"])
    assert changed.revision_tokens == 3  # 1 partial revision + 2 at final
    assert changed.revision_opportunities == 7
    assert changed.final_revision_tokens == 2
    assert changed.final_revision_opportunities == 3
    report = transcript_accuracy_report([changed] * 5, bootstrap_resamples=100)
    assert report["revision_churn"]["rate"] == 3 / 7
    assert report["final_revision_churn"]["rate"] == 2 / 3
    assert score_transcript("", "", partials=["a b"]).final_revision_tokens == 2
    assert score_transcript("a", "a").revision_opportunities == 0


def test_small_cells_and_complements_hide_counts_and_intervals():
    good = score_transcript(
        "Ada", "Ada", spans=[ReferenceTermSpan(0, 3, ClinicalTermClass.IDENTIFIER)]
    )
    bad = score_transcript(
        "Ada", "Eva", spans=[ReferenceTermSpan(0, 3, ClinicalTermClass.IDENTIFIER)]
    )
    for scores in ([good], [good] * 5 + [bad], [bad] * 5 + [good]):
        report = transcript_accuracy_report(scores, bootstrap_resamples=100)
        for key in ("wer", "cer"):
            assert report[key]["suppressed"]
        cell = report["clinical_terms"]["identifier"]
        assert cell["suppressed"]
        assert all(value is None for key, value in cell.items() if key != "suppressed")
    report = transcript_accuracy_report([good] * 20, bootstrap_resamples=100)
    assert report["wer"]["ci95"] == [0, 0]
    # Many annotations in one pair still form a small contributing-pair cell.
    text = " ".join(["a"] * 10)
    score = score_transcript(
        text,
        text,
        spans=[
            ReferenceTermSpan(i, i + 1, ClinicalTermClass.IDENTIFIER)
            for i in range(0, len(text), 2)
        ],
    )
    assert transcript_accuracy_report([score], bootstrap_resamples=100)[
        "clinical_terms"
    ]["identifier"]["suppressed"]


def test_report_determinism_non_degenerate_intervals_and_no_text():
    good = score_transcript("SyntheticAda", "SyntheticAda")
    bad = score_transcript("SyntheticAda", "SyntheticEva")
    scores = [good] * 5 + [bad] * 5
    report = transcript_accuracy_report(scores, bootstrap_resamples=200, seed=7)
    assert report == transcript_accuracy_report(
        reversed(scores), bootstrap_resamples=200, seed=7
    )
    assert report["wer"]["rate"] == 0.5
    assert report["wer"]["ci95"] == [0.2, 0.8]
    assert "Synthetic" not in json.dumps(report) + repr(good)
    assert report["reviewer_confirmation_required"] is True


def test_hash_seed_does_not_change_report_bytes():
    code = """
import json
from openmed.eval.transcript_accuracy import score_transcript, transcript_accuracy_report
scores = [score_transcript('a b', 'a b')] * 5 + [score_transcript('a b', 'a x')] * 5
print(json.dumps(transcript_accuracy_report(scores, bootstrap_resamples=100), sort_keys=True))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            text=True,
        )
        for seed in (1, 999)
    ]
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "policy",
    [
        {"minimum_cell_size": 1},
        {"minimum_cell_size": True},
        {"bootstrap_resamples": 0},
        {"seed": -1},
        {"seed": "SyntheticSecret"},
    ],
)
def test_invalid_report_policy_never_reflects_values(policy):
    with pytest.raises(ValueError) as error:
        transcript_accuracy_report([score_transcript("a", "a")], **policy)
    assert "SyntheticSecret" not in str(error.value)


def test_invalid_inputs_and_bounded_alignment():
    with pytest.raises(ValueError, match="unsupported"):
        score_transcript("secret", "secret", language="secret")
    with pytest.raises(TypeError):
        score_transcript(None, "")
    with pytest.raises(ValueError, match="budget"):
        score_transcript("a" * 1500, "a" * 1500)
    with pytest.raises(ValueError):
        transcript_accuracy_report([])
    with pytest.raises(TypeError):
        transcript_accuracy_report(["SyntheticSecret"])
