"""Offline tests for the i2b2/VA 2010 concept, assertion and relation loader."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from openmed.eval.datasets import (
    DUA_GATED_CORPORA,
    I2B2_2010_ASSERTION_CLASSES,
    I2B2_2010_PATH_ENV,
    DUACredentialRequired,
    I2B22010FormatError,
    license_for,
    load_dua_corpus,
    load_i2b2_2010,
    load_i2b2_deid,
)
from openmed.eval.datasets._dua import source_path_hash
from openmed.eval.relation_metrics import compute_relation_f1
from tests.fixtures.eval.i2b2_2010_synthetic import (
    CONCEPTS,
    EXPECTED_CONCEPT_COUNTS,
    NOTES,
    write_i2b2_2010_corpus,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SENTINEL = "zqxsentinelphi"


@pytest.fixture(autouse=True)
def _no_ambient_path(monkeypatch):
    monkeypatch.delenv(I2B2_2010_PATH_ENV, raising=False)


@pytest.fixture
def corpus(tmp_path):
    return write_i2b2_2010_corpus(tmp_path / "i2b2-2010")


def _by_text(fixtures):
    return {fixture.text: fixture for fixture in fixtures}


def _note_text(stem: str) -> str:
    return "\n".join(NOTES[stem]) + "\n"


def test_synthetic_corpus_loads_counts_assertions_and_relations(corpus):
    fixtures = load_i2b2_2010(corpus)

    assert len(fixtures) == len(NOTES)
    labels: dict[str, int] = {}
    for fixture in fixtures:
        for span in fixture.entities.values():
            labels[span.label] = labels.get(span.label, 0) + 1
    assert labels == EXPECTED_CONCEPT_COUNTS

    note = _by_text(fixtures)[_note_text("note-001")]
    assertions = {
        span.metadata["assertion"]
        for span in note.entities.values()
        if "assertion" in span.metadata
    }
    assert assertions == set(I2B2_2010_ASSERTION_CLASSES)
    assert {r.relation_type for r in note.gold_relations} == {
        "PIP",
        "TERP",
        "TRAP",
        "TECP",
    }
    treatment = next(r for r in note.gold_relations if r.relation_type == "TRAP")
    assert treatment.metadata["source_relation_type"] == "TrAP"
    assert (treatment.head.text, treatment.head.label) == ("aspirin", "OTHER")
    assert treatment.head.metadata["source_label"] == "treatment"
    assert treatment.tail.text == "chest pain"
    assert treatment.tail.metadata["assertion"] == "present"
    assert note.metadata["dataset"] == "i2b2-2010"
    assert note.metadata["eval_only"] is True


def test_gold_relations_score_against_predictions_in_the_file_spelling(corpus):
    note = _by_text(load_i2b2_2010(corpus))[_note_text("note-001")]
    predicted = [
        {
            "type": relation.metadata["source_relation_type"],
            "head": relation.head,
            "tail": relation.tail,
            "scope": relation.scope,
            "fixture_id": note.fixture_id,
        }
        for relation in note.gold_relations
    ]

    assert compute_relation_f1(note.gold_relations, predicted).f1 == 1.0


def test_every_concept_span_covers_exactly_its_tokens(corpus):
    fixtures = _by_text(load_i2b2_2010(corpus))

    for stem, text, start, end, kind, _ in CONCEPTS:
        lines = NOTES[stem]
        (start_line, start_token), (end_line, end_token) = (
            tuple(int(part) for part in start.split(":")),
            tuple(int(part) for part in end.split(":")),
        )
        assert start_line == end_line
        line = lines[start_line - 1]
        line_offset = sum(len(previous) + 1 for previous in lines[: start_line - 1])
        positions, cursor = [], 0
        for token in line.split():
            cursor = line.index(token, cursor)
            positions.append((line_offset + cursor, line_offset + cursor + len(token)))
            cursor += len(token)
        expected = (positions[start_token][0], positions[end_token][1])

        fixture = fixtures[_note_text(stem)]
        spans = [
            span
            for span in fixture.entities.values()
            if (span.start, span.end) == expected
        ]
        assert len(spans) == 1, (stem, text)
        assert spans[0].text.casefold() == text
        assert spans[0].metadata["source_label"] == kind


def test_sentence_offsets_are_the_line_bounds(corpus):
    fixture = _by_text(load_i2b2_2010(corpus))[_note_text("note-001")]

    bounds = fixture.metadata["sentence_offsets"]
    assert [fixture.text[start:end] for start, end in bounds] == NOTES["note-001"]


@pytest.mark.parametrize("ending", ["\r\n", "\r"])
def test_crlf_and_cr_line_endings_keep_line_numbers(corpus, ending):
    text_path = corpus / "txt" / "note-001.txt"
    text_path.write_bytes((ending.join(NOTES["note-001"]) + ending).encode("utf-8"))

    fixture = next(
        f for f in load_i2b2_2010(corpus) if f.text.startswith("The patient")
    )

    bounds = fixture.metadata["sentence_offsets"]
    assert [fixture.text[start:end] for start, end in bounds] == NOTES["note-001"]
    assert sorted(span.text.casefold() for span in fixture.entities.values()) == (
        sorted(concept[1] for concept in CONCEPTS if concept[0] == "note-001")
    )


def test_unreadable_source_keeps_its_path_out(tmp_path):
    corpus = write_i2b2_2010_corpus(tmp_path / SENTINEL)
    (corpus / "concept" / "note-001.con").write_bytes(b"\xff\xfe not utf-8")

    with pytest.raises(ValueError, match="could not be read") as error:
        load_i2b2_2010(corpus)
    assert SENTINEL not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__


def test_environment_variable_supplies_the_path(corpus, monkeypatch):
    monkeypatch.setenv(I2B2_2010_PATH_ENV, str(corpus))

    assert len(load_i2b2_2010()) == len(NOTES)


def _refuse_reads(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a corpus file was read")

    for name in ("open", "read_text", "read_bytes"):
        monkeypatch.setattr(Path, name, refuse)


def test_missing_path_fails_before_reading(monkeypatch):
    _refuse_reads(monkeypatch)

    with pytest.raises(DUACredentialRequired, match=I2B2_2010_PATH_ENV):
        load_i2b2_2010()


def test_path_inside_repository_fails_before_reading(monkeypatch):
    _refuse_reads(monkeypatch)

    with pytest.raises(DUACredentialRequired, match=I2B2_2010_PATH_ENV) as error:
        load_i2b2_2010(REPO_ROOT / "tests" / "fixtures")
    assert "outside the repository" in str(error.value)


@pytest.mark.parametrize(
    ("folder", "suffix", "line", "problem"),
    [
        (
            "concept",
            ".con",
            f'c="{SENTINEL}" 1:3 1:40||t="problem"',
            "token index out of range",
        ),
        (
            "concept",
            ".con",
            f'c="{SENTINEL} pain" 1:3 1:4||t="problem"',
            "concept text does not match its tokens",
        ),
        (
            "ast",
            ".ast",
            f'c="chest pain" 1:3 1:4||t="problem"||a="{SENTINEL}"',
            "unknown assertion class",
        ),
        (
            "rel",
            ".rel",
            f'c="aspirin" 2:6 2:6||r="{SENTINEL}"||c="chest pain" 2:10 2:11',
            "unknown relation type",
        ),
        (
            "ast",
            ".ast",
            'c="aspirin" 2:6 2:6||t="treatment"||a="present"',
            "assertion on a non-problem concept",
        ),
    ],
)
def test_format_errors_carry_hash_and_line_but_no_note_text(
    corpus, folder, suffix, line, problem
):
    note = corpus / "txt" / "note-001.txt"
    note_line = f"Sentinel {SENTINEL} closes the note ."
    note.write_text(_note_text("note-001") + note_line + "\n", encoding="utf-8")
    path = corpus / folder / f"note-001{suffix}"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([*lines, line]) + "\n", encoding="utf-8")

    with pytest.raises(I2B22010FormatError) as error:
        load_i2b2_2010(corpus)

    assert error.value.problem == problem
    assert error.value.line_number == len(lines) + 1
    assert error.value.source_path_hash == source_path_hash(path, corpus)
    assert re.fullmatch(r"[0-9a-f]{64}", error.value.source_path_hash)
    message = str(error.value)
    assert SENTINEL not in message.casefold()
    assert not any(line in message for line in [*NOTES["note-001"], note_line])
    assert "chest pain" not in message
    assert "aspirin" not in message


def test_duplicate_stems_are_refused(corpus):
    duplicate = corpus / "extra" / "note-001.con"
    duplicate.parent.mkdir()
    duplicate.write_text(
        (corpus / "concept" / "note-001.con").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate i2b2 2010 .con") as error:
        load_i2b2_2010(corpus)
    assert "note-001" not in str(error.value)


def test_unannotated_text_files_may_share_a_stem(corpus):
    for folder in ("txt", "concept"):
        (corpus / folder / "README.txt").write_text("notes\n", encoding="utf-8")

    assert len(load_i2b2_2010(corpus)) == len(NOTES)


def test_duplicate_annotated_text_is_refused(corpus):
    duplicate = corpus / "copy" / "note-001.txt"
    duplicate.parent.mkdir()
    duplicate.write_text(_note_text("note-001"), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate i2b2 2010 .txt") as error:
        load_i2b2_2010(corpus)
    assert "note-001" not in str(error.value)


def test_i2b2_deid_still_rejects_2010():
    with pytest.raises(ValueError, match="unsupported i2b2 de-identification year"):
        load_i2b2_deid(year=2010)


def test_registry_marks_i2b2_2010_dua_gated_and_eval_only():
    assert "i2b2-2010" in DUA_GATED_CORPORA
    assert "never redistributed" in license_for("i2b2-2010").redistribution
    with pytest.raises(DUACredentialRequired, match=I2B2_2010_PATH_ENV):
        load_dua_corpus("i2b2-2010")
