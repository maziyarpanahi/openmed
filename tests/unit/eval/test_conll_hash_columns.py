"""Literal hash tokens in whitespace-delimited synthetic CoNLL data."""

import pytest

from openmed.core.schemas import hmac_text_hash
from openmed.eval.annotation import (
    AnnotationValidationError,
    format_conll,
    parse_conll,
    read_conll,
    write_conll,
)

HMAC_SEED = "x"


@pytest.mark.parametrize("separator", [" ", "  ", "\t", " \t "])
@pytest.mark.parametrize("extra_columns", [False, True])
def test_literal_hash_rows_accept_all_column_separators(separator, extra_columns):
    text = "Synthetic #42."
    rows = [("Synthetic", "O"), ("#", "B-ID_NUM"), ("42", "I-ID_NUM"), (".", "O")]
    columns = "\n".join(
        separator.join((token, "SYM", "B-NP", tag) if extra_columns else (token, tag))
        for token, tag in rows
    )
    spans = parse_conll(text, columns, doc_id="synthetic", hash_secret=HMAC_SEED)
    assert [(s.start, s.end, s.canonical_label) for s in spans] == [(10, 13, "ID_NUM")]
    assert spans[0].text_hash == hmac_text_hash("#42", HMAC_SEED)


@pytest.mark.parametrize("tag", ["O", "B-ID_NUM", "S-ID_NUM", "U-ID_NUM"])
def test_space_delimited_hash_alone_is_a_token(tag):
    spans = parse_conll("#", f"# {tag}\n", doc_id="synthetic", hash_secret=HMAC_SEED)
    assert len(spans) == (0 if tag == "O" else 1)
    if spans:
        assert (spans[0].start, spans[0].end) == (0, 1)


def test_hash_comments_and_sentence_breaks_still_work():
    text = "alpha beta"
    columns = (
        "# synthetic comment\n#\nalpha B-PERSON\n\n# next sentence\nbeta B-PERSON\n"
    )
    spans = parse_conll(text, columns, doc_id="synthetic", hash_secret=HMAC_SEED)
    assert [(s.start, s.end) for s in spans] == [(0, 5), (6, 10)]


def test_hash_token_with_invalid_transition_is_not_hidden_as_comment():
    with pytest.raises(AnnotationValidationError, match="cannot start an entity"):
        parse_conll("#", "# I-ID_NUM\n", doc_id="synthetic", hash_secret=HMAC_SEED)


def test_space_converted_writer_output_round_trips_through_files(tmp_path):
    text = "Synthetic case #42."
    path = write_conll(tmp_path / "synthetic.conll", text, ())
    path.write_text(
        path.read_text(encoding="utf-8").replace("\t", " "), encoding="utf-8"
    )
    task = read_conll(
        path,
        text=text,
        doc_id="synthetic",
        hash_secret=HMAC_SEED,
        synthetic=True,
    )
    assert task.text == text
    assert task.spans == ()
    assert task.synthetic is True
    assert "#\tO\n" in format_conll(text, ())


def test_ordinary_extra_columns_are_unchanged():
    spans = parse_conll(
        "alpha", "alpha NN B-NP U-PERSON", doc_id="synthetic", hash_secret=HMAC_SEED
    )
    assert [(s.start, s.end) for s in spans] == [(0, 5)]


def test_comment_ending_in_tag_does_not_displace_next_source_token():
    text = "alpha #"
    columns = "# synthetic comment O\nalpha O\n# O\n"
    assert parse_conll(text, columns, doc_id="synthetic", hash_secret=HMAC_SEED) == ()


def test_comment_ending_in_tag_without_hash_source_remains_a_comment():
    assert (
        parse_conll(
            "alpha",
            "# synthetic comment O\nalpha O",
            doc_id="synthetic",
            hash_secret=HMAC_SEED,
        )
        == ()
    )
