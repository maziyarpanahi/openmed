"""Deterministic HGVS variant-mention extractor and syntactic normalizer.

Molecular-pathology notes record variants as HGVS descriptors such as
``NM_000059.3:c.1521_1523delCTT`` or ``p.Phe508del``. This module scans free
text for those descriptors and decomposes each into a structured
:class:`HgvsMention` -- reference sequence, coordinate type (``c``/``p``/``g``/
``m``/``n``/``r``), position, and edit -- while preserving exact character
offsets into the source text.

The parsing is *syntactic only*: each descriptor is classified ``valid``,
``malformed`` (with a reason code), or ``unparseable``. Malformed descriptors
are never coerced into a plausible-looking variant. There is no validation
against a reference sequence or transcript, no biological or pathogenicity
interpretation, and no database lookup -- see :data:`GENOMICS_ADVISORY`.

The grammar covers the common single-variant forms. Complex constructs beyond
that subset -- allele/phasing brackets (``c.[a];[b]``), parenthesized
coordinate uncertainty (``c.(76_78)del``), unknown-length insertions
(``ins(20)``), and insertions specified by another accession -- are surfaced as
``malformed``/``unparseable`` rather than decomposed, so they are never coerced
or silently dropped. Offsets are code-point (character) offsets, consistent with
the rest of the pipeline; for the ASCII HGVS descriptor they equal byte offsets.
"""

from __future__ import annotations

import re
from typing import Optional, TypedDict

#: Coordinate types recognized after the ``:`` (or at the start of a bare
#: descriptor): coding DNA, protein, genomic, mitochondrial, non-coding, RNA.
COORDINATE_TYPES: frozenset[str] = frozenset({"c", "p", "g", "m", "n", "r"})

GENOMICS_ADVISORY = (
    "HGVS variant extraction is deterministic, syntactic parsing and offset "
    "preservation only. It does not validate a variant against a reference "
    "sequence or transcript, perform biological or pathogenicity "
    "interpretation, or query any variant database (ClinVar, dbSNP, COSMIC, "
    "HGMD). Output assists review and is not a clinical or molecular-diagnostic "
    "determination."
)


class HgvsMention(TypedDict):
    """One HGVS descriptor found in text, decomposed into syntactic fields.

    ``reference_sequence`` is the accession before the ``:`` (e.g.
    ``"NM_000059.3"``) or ``None`` for a bare descriptor. ``coordinate_type`` is
    the single letter (``c``/``p``/``g``/``m``/``n``/``r``). ``position`` and
    ``edit`` split the variant (e.g. ``"1521_1523"`` and ``"delCTT"``, or
    ``"Phe508"`` and ``"del"``). ``raw_text`` is the exact matched descriptor and
    ``span`` its ``(start, end)`` character offsets, so ``text[start:end] ==
    raw_text``. ``status`` is ``"valid"``, ``"malformed"`` (with a ``reason``
    code), or ``"unparseable"`` (``reason`` names the failure); it is never
    coerced.
    """

    reference_sequence: Optional[str]
    coordinate_type: Optional[str]
    position: Optional[str]
    edit: Optional[str]
    raw_text: str
    span: tuple[int, int]
    status: str
    reason: Optional[str]
    advisory: str


# --- descriptor grammar ------------------------------------------------------

# A descriptor may carry a reference sequence (RefSeq/Ensembl/LRG accession or a
# gene symbol, optionally versioned and gene-qualified) followed by ``:``, then a
# coordinate-type letter, a dot, and a maximally-captured variant run. The whole
# thing must not start inside a word (so ``etc.`` never yields a ``c.`` mention).
_REFERENCE = r"[A-Za-z][A-Za-z0-9_]*(?:\.\d+)?(?:\([A-Za-z0-9_]+\))?"
_DESCRIPTOR_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    rf"(?:(?P<ref>{_REFERENCE}):)?"
    r"(?P<type>[cpgmnr])\."
    r"(?P<variant>[A-Za-z0-9_>()\[\]+*?=\-]+)"
)

# A single DNA coordinate: an optional 5'UTR ``-`` / 3'UTR ``*`` marker, digits,
# and an optional intron offset. A position may be one coordinate or a range.
_DNA_COORD = r"[*-]?\d+(?:[+-]\d+)?"
_DNA_POSITION_RE = re.compile(rf"(?P<pos>{_DNA_COORD}(?:_{_DNA_COORD})?)")
_DNA_SUBSTITUTION_RE = re.compile(r"([ACGTUacgtu])>([ACGTUacgtu])")
_DNA_OPERATION_RE = re.compile(
    r"(?P<op>delins|del|dup|ins|inv)(?P<payload>[A-Za-z0-9_]*)"
)
_NUCLEOTIDES_RE = re.compile(r"[ACGTUacgtu]+")

# Amino-acid residues: three-letter (incl. Ter) or one-letter (incl. ``*``).
_AA3 = (
    "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp"
    "|Tyr|Val|Ter|Sec|Pyl"
)
_AA1 = r"[ARNDCEQGHILKMFPSTWYVUOX*]"
_RESIDUE = rf"(?:{_AA3}|{_AA1})"
_PROTEIN_POSITION_RE = re.compile(rf"(?P<pos>{_RESIDUE}\d+(?:_{_RESIDUE}\d+)?)")
_RESIDUES_RE = re.compile(rf"(?:{_RESIDUE})+")


def _validate_dna_payload(op: str, payload: str) -> Optional[str]:
    """Return a reason code when a DNA edit payload is invalid, else ``None``."""
    if op in ("del", "dup", "inv"):
        # Deleted/duplicated sequence is optional; when written it is nucleotides
        # or a length.
        if payload == "" or _NUCLEOTIDES_RE.fullmatch(payload) or payload.isdigit():
            return None
        return "invalid_nucleotide"
    # ins / delins require an inserted sequence (nucleotides, a length, or a
    # coordinate range).
    if payload == "":
        return "missing_inserted_sequence"
    if (
        _NUCLEOTIDES_RE.fullmatch(payload)
        or payload.isdigit()
        or re.fullmatch(r"\d+_\d+", payload)
    ):
        return None
    return "invalid_nucleotide"


def _parse_dna(variant: str) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Decompose a DNA/RNA variant into ``(position, edit, status, reason)``."""
    if variant in ("=", "?"):
        return None, variant, "valid", None
    match = _DNA_POSITION_RE.match(variant)
    if not match:
        return None, None, "unparseable", "no_position_found"
    position = match.group("pos")
    edit = variant[match.end() :]
    if edit == "":
        return position, None, "malformed", "missing_edit"
    if edit == "?":  # consequence unknown at this position
        return position, edit, "valid", None
    if _DNA_SUBSTITUTION_RE.fullmatch(edit):
        return position, edit, "valid", None
    if ">" in edit:
        return position, edit, "malformed", "invalid_nucleotide"
    operation = _DNA_OPERATION_RE.fullmatch(edit)
    if operation is None:
        return position, edit, "malformed", "unknown_edit_operation"
    reason = _validate_dna_payload(operation.group("op"), operation.group("payload"))
    if reason:
        return position, edit, "malformed", reason
    return position, edit, "valid", None


def _valid_protein_edit(edit: str) -> bool:
    """True when a protein edit is a recognized HGVS operation."""
    if edit in ("del", "dup", "=", "?"):  # incl. unknown consequence (e.g. p.Met1?)
        return True
    if "fs" in edit:  # frameshift, incl. Ter/``*`` count variants
        return True
    if "ext" in edit:  # stop-loss / N-terminal extension (e.g. GlnextTer17)
        return True
    if edit.startswith("delins"):
        return bool(_RESIDUES_RE.fullmatch(edit[len("delins") :]))
    if edit.startswith("ins"):
        return bool(_RESIDUES_RE.fullmatch(edit[len("ins") :]))
    # Otherwise a variant residue (missense) or Ter / ``*`` (nonsense).
    return bool(_RESIDUES_RE.fullmatch(edit))


def _parse_protein(
    variant: str,
) -> tuple[Optional[str], Optional[str], str, Optional[str]]:
    """Decompose a protein variant into ``(position, edit, status, reason)``."""
    inner = variant
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]  # predicted consequence, e.g. p.(Arg97Gly)
    if inner in ("=", "?", "0"):
        return None, inner, "valid", None
    match = _PROTEIN_POSITION_RE.match(inner)
    if not match:
        return None, None, "unparseable", "no_position_found"
    position = match.group("pos")
    edit = inner[match.end() :]
    if edit == "":
        return position, None, "malformed", "missing_edit"
    if _valid_protein_edit(edit):
        return position, edit, "valid", None
    return position, edit, "malformed", "invalid_amino_acid"


def parse_hgvs(text: str) -> list[HgvsMention]:
    """Extract and syntactically parse every HGVS descriptor in ``text``.

    Args:
        text: Free-text molecular-pathology content that may contain one or more
            HGVS descriptors, e.g. ``"NM_000059.3:c.1521_1523delCTT, p.Phe508del"``.

    Returns:
        One :class:`HgvsMention` per descriptor, in order of appearance, each
        carrying its character ``span`` (so ``text[start:end] == raw_text``) and
        a ``status`` of ``valid`` / ``malformed`` / ``unparseable``. Malformed
        descriptors are surfaced with a reason code and never coerced, and every
        mention carries :data:`GENOMICS_ADVISORY`.
    """
    source = text or ""
    mentions: list[HgvsMention] = []
    for match in _DESCRIPTOR_RE.finditer(source):
        coordinate_type = match.group("type")
        variant = match.group("variant")
        raw_text = match.group(0)
        start, end = match.start(), match.end()
        # A descriptor written inside brackets, e.g. "(p.Phe508del)" or
        # "[g.123A>T]", captures a trailing ")"/"]" that belongs to the
        # surrounding text. Trim any unbalanced trailing bracket and shrink the
        # span so offsets stay exact against source. Balanced brackets that are
        # part of the descriptor (allele "[...]", predicted "(...)") are kept.
        while (variant.endswith(")") and variant.count(")") > variant.count("(")) or (
            variant.endswith("]") and variant.count("]") > variant.count("[")
        ):
            variant = variant[:-1]
            raw_text = raw_text[:-1]
            end -= 1
        if coordinate_type == "p":
            position, edit, status, reason = _parse_protein(variant)
        else:
            position, edit, status, reason = _parse_dna(variant)
        mentions.append(
            HgvsMention(
                reference_sequence=match.group("ref"),
                coordinate_type=coordinate_type,
                position=position,
                edit=edit,
                raw_text=raw_text,
                span=(start, end),
                status=status,
                reason=reason,
                advisory=GENOMICS_ADVISORY,
            )
        )
    return mentions


__all__ = [
    "COORDINATE_TYPES",
    "GENOMICS_ADVISORY",
    "HgvsMention",
    "parse_hgvs",
]
