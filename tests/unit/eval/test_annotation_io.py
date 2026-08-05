"""Tests for synthetic annotation tasks and BRAT/CoNLL interchange."""

from __future__ import annotations

import pytest

from openmed.core.schemas import OpenMedSpan, hmac_text_hash
from openmed.eval.annotation import (
    AnnotationValidationError,
    format_brat,
    format_conll,
    generate_synthetic_annotation_task,
    parse_brat,
    parse_conll,
    read_brat,
    read_conll,
    write_brat,
    write_conll,
)

_SECRET = "synthetic-test-key"
_TEXT = "Patient Jane Roe visited on 2025-03-02."


def _span(start: int, end: int, label: str) -> OpenMedSpan:
    return OpenMedSpan(
        doc_id="synthetic-note-1",
        start=start,
        end=end,
        text_hash=hmac_text_hash(_TEXT[start:end], _SECRET),
        entity_type=label.lower(),
        canonical_label=label,
        detector="synthetic_fixture",
    )


def _spans() -> tuple[OpenMedSpan, ...]:
    return (_span(8, 16, "PERSON"), _span(28, 38, "DATE"))


def _offset_labels(spans: tuple[OpenMedSpan, ...]) -> list[tuple[int, int, str]]:
    return [(span.start, span.end, span.canonical_label) for span in spans]


def test_brat_round_trip_preserves_offsets_and_labels(tmp_path) -> None:
    text_path = tmp_path / "synthetic-note-1.txt"

    written_text, written_ann = write_brat(text_path, _TEXT, _spans())
    task = read_brat(written_text, written_ann, hash_secret=_SECRET, synthetic=True)

    assert task.text == _TEXT
    assert task.synthetic is True
    assert _offset_labels(tuple(task.spans)) == _offset_labels(_spans())
    assert format_brat(task.text, task.spans).startswith("T1\tPERSON 8 16\tJane Roe")


@pytest.mark.parametrize(
    ("annotations", "message"),
    [
        ("T1 PERSON 0 7 Patient", "three tab-separated fields"),
        ("T1\tPERSON zero 7\tPatient", "offsets must be integers"),
        ("T1\tPERSON 0 70\tPatient", "0 <= start < end"),
        ("T1\tPERSON 8 16\tJohn Doe", "does not match source offsets"),
        ("T1\tUNMAPPED 8 16\tJane Roe", "unknown label"),
        ("R1\tRelated Arg1:T1 Arg2:T2\tunused", "only text-bound IDs"),
        ("T1\tPERSON 8 12;13 16\tJane Roe", "discontinuous spans"),
    ],
)
def test_malformed_brat_has_actionable_line_errors(
    annotations: str,
    message: str,
) -> None:
    with pytest.raises(AnnotationValidationError) as exc_info:
        parse_brat(
            _TEXT,
            annotations,
            doc_id="synthetic-note-1",
            hash_secret=_SECRET,
        )

    rendered = str(exc_info.value)
    assert "Invalid BRAT standoff data" in rendered
    assert "line 1" in rendered
    assert message in rendered


def test_brat_collects_multiple_validation_issues() -> None:
    annotations = "T1\tPERSON bad 16\tJane Roe\nT2\tDATE 28 99\t2025-03-02\n"

    with pytest.raises(AnnotationValidationError) as exc_info:
        parse_brat(
            _TEXT,
            annotations,
            doc_id="synthetic-note-1",
            hash_secret=_SECRET,
        )

    assert len(exc_info.value.issues) == 2
    assert "line 1" in str(exc_info.value)
    assert "line 2" in str(exc_info.value)


@pytest.mark.parametrize("hash_secret", ["", b""])
def test_annotation_import_rejects_empty_hash_secret(
    hash_secret: str | bytes,
) -> None:
    with pytest.raises(
        AnnotationValidationError, match="hash_secret must be non-empty"
    ):
        parse_brat(
            _TEXT,
            "T1\tPERSON 8 16\tJane Roe\n",
            doc_id="synthetic-note-1",
            hash_secret=hash_secret,
        )


def test_conll_round_trip_preserves_offsets_and_labels(tmp_path) -> None:
    conll_path = write_conll(tmp_path / "synthetic-note-1.conll", _TEXT, _spans())
    task = read_conll(
        conll_path,
        text=_TEXT,
        doc_id="synthetic-note-1",
        hash_secret=_SECRET,
        synthetic=True,
    )

    assert task.synthetic is True
    assert _offset_labels(tuple(task.spans)) == _offset_labels(_spans())
    assert "Jane\tB-PERSON\nRoe\tI-PERSON" in format_conll(_TEXT, _spans())


def test_conll_round_trip_preserves_hash_tokens() -> None:
    text = "Synthetic case #42."

    columns = format_conll(text, ())
    spans = parse_conll(
        text,
        columns,
        doc_id="synthetic-case-42",
        hash_secret=_SECRET,
    )

    assert "#\tO" in columns
    assert spans == ()


def test_conll_accepts_traditional_extra_columns_and_bilou() -> None:
    columns = "\n".join(
        [
            "Patient NN B-NP O",
            "Jane NNP B-NP B-PERSON",
            "Roe NNP I-NP L-PERSON",
            "visited VBD B-VP O",
            "on IN B-PP O",
            "2025-03-02 CD B-NP U-DATE",
            ". . O O",
        ]
    )

    spans = parse_conll(
        _TEXT,
        columns,
        doc_id="synthetic-note-1",
        hash_secret=_SECRET,
    )

    assert _offset_labels(spans) == _offset_labels(_spans())


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ("Patient\n", "at least TOKEN and TAG"),
        (
            "Patient O\nJane I-PERSON\nRoe O\nvisited O\non O\n"
            "2025 O\n- O\n03 O\n- O\n02 O\n. O\n",
            "cannot start an entity",
        ),
        ("Patient O\nMissing O\n", "cannot be aligned"),
        (
            "Patient X-PERSON\nJane O\nRoe O\nvisited O\non O\n"
            "2025 O\n- O\n03 O\n- O\n02 O\n. O\n",
            "BIO/BIOES/BILOU",
        ),
    ],
)
def test_malformed_conll_has_actionable_errors(columns: str, message: str) -> None:
    with pytest.raises(AnnotationValidationError, match=message):
        parse_conll(
            _TEXT,
            columns,
            doc_id="synthetic-note-1",
            hash_secret=_SECRET,
        )


def test_conll_rejects_overlapping_spans() -> None:
    overlap = _span(8, 12, "FIRST_NAME")

    with pytest.raises(AnnotationValidationError, match="one label per token"):
        format_conll(_TEXT, (_spans()[0], overlap))


def test_writers_reject_spans_from_different_documents() -> None:
    other_document = OpenMedSpan(
        doc_id="other-note",
        start=28,
        end=38,
        text_hash=hmac_text_hash("2025-03-02", _SECRET),
        entity_type="date",
        canonical_label="DATE",
    )

    with pytest.raises(AnnotationValidationError, match="different document"):
        format_brat(_TEXT, (_spans()[0], other_document))
    with pytest.raises(AnnotationValidationError, match="different document"):
        format_conll(_TEXT, (_spans()[0], other_document))


def test_synthetic_task_generator_prelabels_for_human_review() -> None:
    def prelabeler(text: str):
        assert text == _TEXT
        return [
            {"start": 8, "end": 16, "label": "PERSON", "score": 0.91},
            {"start": 28, "end": 38, "entity_type": "date"},
        ]

    task = generate_synthetic_annotation_task(
        "synthetic-note-1",
        _TEXT,
        hash_secret=_SECRET,
        prelabeler=prelabeler,
        metadata={"scenario": "fictional outpatient note"},
    )

    assert task.synthetic is True
    assert task.metadata["synthetic"] is True
    assert task.metadata["prelabeled"] is True
    assert task.metadata["review_status"] == "needs_human_review"
    assert _offset_labels(tuple(task.spans)) == _offset_labels(_spans())


def test_synthetic_task_generator_rejects_misaligned_existing_span() -> None:
    wrong_document = OpenMedSpan(
        doc_id="other-note",
        start=8,
        end=16,
        text_hash=hmac_text_hash("Jane Roe", _SECRET),
        entity_type="person",
        canonical_label="PERSON",
    )

    with pytest.raises(AnnotationValidationError, match="different document"):
        generate_synthetic_annotation_task(
            "synthetic-note-1",
            _TEXT,
            hash_secret=_SECRET,
            spans=[wrong_document],
        )
