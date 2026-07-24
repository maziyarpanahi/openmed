"""Tests for the deterministic HGVS variant-mention extractor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.clinical import GENOMICS_ADVISORY, parse_hgvs

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "clinical" / "hgvs_variant.jsonl"
)

MENTION_FIELDS = (
    "reference_sequence",
    "coordinate_type",
    "position",
    "edit",
    "raw_text",
    "status",
    "reason",
)


def _load_fixture() -> list[dict]:
    with FIXTURE.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _one(text: str):
    mentions = parse_hgvs(text)
    assert len(mentions) == 1, text
    return mentions[0]


# ---------------------------------------------------------------------------
# Fixture-driven gold rows
# ---------------------------------------------------------------------------


def test_fixture_rows_are_synthetic():
    rows = _load_fixture()
    assert rows
    assert all(row["metadata"]["synthetic"] is True for row in rows)


@pytest.mark.parametrize("row", _load_fixture())
def test_parser_matches_gold_row(row):
    produced = parse_hgvs(row["text"])
    assert len(produced) == len(row["gold"])
    for mention, gold in zip(produced, row["gold"]):
        for field in MENTION_FIELDS:
            assert mention[field] == gold[field], field
        assert list(mention["span"]) == gold["span"]
        # Offset stability: the span reproduces raw_text byte-for-byte.
        assert (
            row["text"][mention["span"][0] : mention["span"][1]] == mention["raw_text"]
        )


# ---------------------------------------------------------------------------
# Component decomposition
# ---------------------------------------------------------------------------


def test_reference_sequence_and_type_split():
    mention = _one("NM_000059.3:c.1521_1523delCTT")
    assert mention["reference_sequence"] == "NM_000059.3"
    assert mention["coordinate_type"] == "c"
    assert mention["position"] == "1521_1523"
    assert mention["edit"] == "delCTT"
    assert mention["status"] == "valid"


def test_bare_descriptor_has_no_reference_sequence():
    assert _one("c.76A>T")["reference_sequence"] is None


def test_substitution_position_and_edit():
    mention = _one("c.76A>T")
    assert (mention["position"], mention["edit"]) == ("76", "A>T")


@pytest.mark.parametrize(
    ("text", "position"),
    [
        ("c.88+1G>T", "88+1"),  # intron offset
        ("c.-14G>C", "-14"),  # 5' UTR
        ("c.*32A>T", "*32"),  # 3' UTR
    ],
)
def test_intron_and_utr_coordinates(text, position):
    assert _one(text)["position"] == position


@pytest.mark.parametrize(
    ("text", "edit"),
    [
        ("c.76dup", "dup"),
        ("c.76_77insACG", "insACG"),
        ("c.76_78del", "del"),
        ("c.112_117delinsTG", "delinsTG"),
        ("g.123_456inv", "inv"),
    ],
)
def test_edit_operations(text, edit):
    mention = _one(text)
    assert mention["edit"] == edit
    assert mention["status"] == "valid"


@pytest.mark.parametrize("ctype", list("cpgmnr"))
def test_every_coordinate_type_is_recognized(ctype):
    variant = "Arg97Gly" if ctype == "p" else "76A>T"
    mention = _one(f"{ctype}.{variant}")
    assert mention["coordinate_type"] == ctype
    assert mention["status"] == "valid"


# ---------------------------------------------------------------------------
# Protein
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "position", "edit"),
    [
        ("p.Arg97Gly", "Arg97", "Gly"),  # missense
        ("p.Phe508del", "Phe508", "del"),  # deletion
        ("p.Arg97Ter", "Arg97", "Ter"),  # nonsense (Ter)
        ("p.Arg97*", "Arg97", "*"),  # nonsense (*)
        ("p.Gly12fs", "Gly12", "fs"),  # frameshift
        ("p.Arg97_Gly99del", "Arg97_Gly99", "del"),  # range
    ],
)
def test_protein_changes(text, position, edit):
    mention = _one(text)
    assert (mention["position"], mention["edit"]) == (position, edit)
    assert mention["status"] == "valid"


def test_predicted_protein_parentheses_are_unwrapped():
    mention = _one("p.(Arg97Gly)")
    assert (mention["position"], mention["edit"]) == ("Arg97", "Gly")
    assert mention["status"] == "valid"


@pytest.mark.parametrize(
    "text",
    [
        "p.Cys28delinsTrpVal",  # protein delins
        "p.Lys23_Leu24insArgSer",  # protein insertion
        "p.Ter110GlnextTer17",  # stop-loss extension
    ],
)
def test_protein_delins_insertion_and_extension_are_valid(text):
    assert _one(text)["status"] == "valid"


@pytest.mark.parametrize("text", ["p.=", "p.?", "p.0", "p.(=)"])
def test_protein_synonymous_and_unknown_are_valid(text):
    assert _one(text)["status"] == "valid"


def test_protein_position_without_edit_is_malformed():
    mention = _one("p.Arg97")
    assert mention["status"] == "malformed"
    assert mention["reason"] == "missing_edit"


# ---------------------------------------------------------------------------
# Never-coerce: malformed / unparseable are flagged with reasons
# ---------------------------------------------------------------------------


def test_invalid_nucleotide_is_malformed_not_coerced():
    mention = _one("c.76X>Y")
    assert mention["position"] == "76"
    assert mention["status"] == "malformed"
    assert mention["reason"] == "invalid_nucleotide"


def test_missing_edit_is_flagged():
    mention = _one("c.76")
    assert mention["status"] == "malformed"
    assert mention["reason"] == "missing_edit"


def test_unknown_edit_operation_is_flagged():
    assert _one("c.76foo")["reason"] == "unknown_edit_operation"


def test_invalid_amino_acid_is_flagged():
    mention = _one("p.Arg97Zzz")
    assert mention["status"] == "malformed"
    assert mention["reason"] == "invalid_amino_acid"


def test_no_position_is_unparseable():
    mention = _one("c.splicedonor")
    assert mention["status"] == "unparseable"
    assert mention["reason"] == "no_position_found"
    assert mention["position"] is None and mention["edit"] is None


@pytest.mark.parametrize("text", ["p.Met1?", "p.(Met1?)", "c.76?"])
def test_unknown_consequence_question_mark_is_valid(text):
    # e.g. p.Met1? (start-loss, effect unknown) is a valid HGVS descriptor.
    mention = _one(text)
    assert mention["edit"] == "?"
    assert mention["status"] == "valid"


@pytest.mark.parametrize(
    "text", ["c.[76A>T]", "NM_000059.3:c.[76A>T]", "c.[76A>T];[88+1G>T]"]
)
def test_allele_bracket_notation_is_surfaced_not_dropped(text):
    # Bracket/allele notation is beyond the single-variant subset, but must be
    # captured and surfaced (not silently dropped) with a stable span.
    mentions = parse_hgvs(text)
    assert mentions, text  # at least one mention -- never zero (no silent loss)
    first = mentions[0]
    assert first["status"] == "unparseable"
    assert text[first["span"][0] : first["span"][1]] == first["raw_text"]


def test_insertion_without_sequence_is_flagged():
    mention = _one("c.76_77ins")
    assert mention["status"] == "malformed"
    assert mention["reason"] == "missing_inserted_sequence"


def test_deletion_with_non_nucleotide_payload_is_flagged():
    assert _one("c.76delXY")["reason"] == "invalid_nucleotide"


def test_insertion_with_non_nucleotide_payload_is_flagged():
    assert _one("c.76_77insZZ")["reason"] == "invalid_nucleotide"


@pytest.mark.parametrize("text", ["c.=", "c.?"])
def test_dna_no_change_and_unknown_are_valid(text):
    mention = _one(text)
    assert mention["status"] == "valid"
    assert mention["position"] is None


# ---------------------------------------------------------------------------
# Extraction over free text
# ---------------------------------------------------------------------------


def test_multiple_descriptors_extracted_with_stable_spans():
    text = "Somatic BRCA1 NM_000059.3:c.1521_1523delCTT (p.Phe508del), heterozygous."
    mentions = parse_hgvs(text)
    assert [m["coordinate_type"] for m in mentions] == ["c", "p"]
    # The parenthesized protein must not capture the trailing ")".
    assert mentions[1]["raw_text"] == "p.Phe508del"
    for mention in mentions:
        s, e = mention["span"]
        assert text[s:e] == mention["raw_text"]
        assert mention["status"] == "valid"


def test_prose_does_not_produce_false_descriptors():
    assert parse_hgvs("This applies to the gene, etc. See the report.") == []


def test_empty_input_returns_no_mentions():
    assert parse_hgvs("") == []


# ---------------------------------------------------------------------------
# Advisory
# ---------------------------------------------------------------------------


def test_advisory_attached_to_every_mention_and_disclaims_scope():
    for mention in parse_hgvs("c.76A>T and p.Arg97Gly"):
        assert mention["advisory"] == GENOMICS_ADVISORY
    lowered = GENOMICS_ADVISORY.lower()
    assert "syntactic" in lowered
    assert "does not validate" in lowered
    assert "not a clinical or molecular-diagnostic determination" in lowered
