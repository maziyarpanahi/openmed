from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "generative" / "unanswerable_v1.jsonl"
)

_NOTE_KEYS = {
    "note_id",
    "synthetic",
    "provenance",
    "note",
    "questions",
}

_QUESTION_KEYS = {
    "question_id",
    "question",
    "expected",
    "supporting_quote",
}


def _load_fixture() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in _FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _validate_supporting_quote(note: str, quote: str) -> None:
    if not quote or note.count(quote) != 1:
        raise ValueError("Supporting quote must occur exactly once in the note")


def test_fixture_has_at_least_thirty_notes_and_sixty_questions() -> None:
    records = _load_fixture()

    assert len(records) >= 30
    assert sum(len(record["questions"]) for record in records) >= 60


def test_note_records_have_declared_shape_and_unique_ids() -> None:
    records = _load_fixture()
    note_ids = [record["note_id"] for record in records]

    assert len(note_ids) == len(set(note_ids))

    for record in records:
        assert set(record) == _NOTE_KEYS
        assert isinstance(record["note_id"], str)
        assert record["note_id"]
        assert record["synthetic"] is True
        assert isinstance(record["provenance"], str)
        assert record["provenance"]
        assert isinstance(record["note"], str)
        assert record["note"]
        assert isinstance(record["questions"], list)


def test_questions_have_declared_shape_and_unique_ids() -> None:
    records = _load_fixture()
    question_ids = []

    for record in records:
        assert len(record["questions"]) == 2

        for question in record["questions"]:
            assert set(question) == _QUESTION_KEYS
            assert isinstance(question["question_id"], str)
            assert question["question_id"]
            assert isinstance(question["question"], str)
            assert question["question"]
            assert question["expected"] in {"answer", "abstain"}
            question_ids.append(question["question_id"])

    assert len(question_ids) == len(set(question_ids))


def test_each_note_has_one_answer_and_one_abstention_question() -> None:
    for record in _load_fixture():
        expected_results = [question["expected"] for question in record["questions"]]

        assert expected_results.count("answer") == 1
        assert expected_results.count("abstain") == 1


def test_supporting_quotes_occur_exactly_once_in_their_notes() -> None:
    for record in _load_fixture():
        for question in record["questions"]:
            quote = question["supporting_quote"]

            if question["expected"] == "answer":
                assert isinstance(quote, str)
                _validate_supporting_quote(record["note"], quote)
            else:
                assert quote is None


@pytest.mark.parametrize(
    ("note", "quote"),
    [
        ("Patient has diabetes.", "Patient has hypertension."),
        (
            "Patient has diabetes. Patient has diabetes.",
            "Patient has diabetes.",
        ),
    ],
    ids=["missing-supporting-quote", "duplicate-supporting-quote"],
)
def test_invalid_supporting_quotes_are_rejected(note: str, quote: str) -> None:
    with pytest.raises(ValueError, match="must occur exactly once"):
        _validate_supporting_quote(note, quote)
