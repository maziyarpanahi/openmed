"""Composable synthetic clinical note grammars with exact fact and identifier truth."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from itertools import islice

from openmed.core.labels import CANONICAL_LABELS, DATE, DISEASE, DOSAGE, DRUG, PROCEDURE
from openmed.training.synthetic.locale_phi import LocalePhiGenerator, SyntheticPhiSpan

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SOURCE = "openmed.synthetic.note_grammar"
GRAMMARS = {
    "encounter": (
        ("demographics", ("person", "identifier", "date")),
        ("assessment", ("diagnosis",)),
        ("medications", ("medication", "dose")),
        ("plan", ("procedure", "distractor")),
    ),
    "discharge": (
        ("demographics", ("person", "identifier", "date")),
        ("admission", ("diagnosis",)),
        ("course", ("procedure", "diagnosis")),
        ("medications", ("medication", "dose")),
        ("followup", ("distractor",)),
    ),
    "referral": (
        ("demographics", ("person", "identifier", "date")),
        ("reason", ("diagnosis",)),
        ("history", ("procedure",)),
        ("medications", ("medication", "dose")),
        ("request", ("distractor",)),
    ),
    "operative": (
        ("demographics", ("person", "identifier", "date")),
        ("preoperative", ("diagnosis",)),
        ("procedure", ("procedure",)),
        ("postoperative", ("diagnosis", "distractor")),
    ),
    "nursing": (
        ("demographics", ("person", "identifier", "date")),
        ("observations", ("diagnosis",)),
        ("administrations", ("medication", "dose")),
        ("handoff", ("procedure", "distractor")),
    ),
}


@dataclass(frozen=True, slots=True)
class SyntheticNoteFact:
    """One explicitly synthetic fact reused consistently across note sections."""

    key: str
    label: str
    value: str = field(repr=False)
    role: str = "clinical_fact"
    source: str = _SOURCE

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not _CODE.fullmatch(self.key):
            raise ValueError("invalid synthetic note fact code")
        if not isinstance(self.label, str) or self.label not in CANONICAL_LABELS:
            raise ValueError("invalid synthetic note canonical label")
        if (
            not isinstance(self.value, str)
            or not 1 <= len(self.value) <= 512
            or "\n" in self.value
            or "\r" in self.value
        ):
            raise ValueError("invalid synthetic note fact value")
        if not isinstance(self.role, str) or self.role not in {
            "identifier",
            "clinical_fact",
            "distractor",
        }:
            raise ValueError("invalid synthetic note fact role")
        if self.source not in {_SOURCE, "openmed.synthetic.locale_phi"}:
            raise ValueError("invalid synthetic note provenance")

    @property
    def fact_id(self) -> str:
        """Return a deterministic synthetic fact reference shared across occurrences."""
        return (
            "fact_"
            + hashlib.sha256(
                f"{self.key}:{self.label}:{self.value}:{self.role}:{self.source}".encode()
            ).hexdigest()[:24]
        )


@dataclass(frozen=True, slots=True)
class SyntheticGrammarNote:
    """One rendered synthetic note with reusable fact identities and exact gold spans."""

    document_type: str
    text: str = field(repr=False)
    facts: tuple[SyntheticNoteFact, ...]
    gold_spans: tuple[SyntheticPhiSpan, ...]
    sections: tuple[dict[str, object], ...]
    synthetic: bool = True
    source: str = _SOURCE


def synthetic_note_facts(*, seed: int = 0) -> tuple[SyntheticNoteFact, ...]:
    """Reuse the locale surrogate generator and append fictional clinical placeholders."""
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("invalid synthetic note seed")
    example = LocalePhiGenerator(seed=seed).generate("en")
    by_field = {span.metadata.get("field"): span for span in example.gold_spans}
    facts = []
    for key in ("person", "identifier"):
        span = by_field[key]
        facts.append(
            SyntheticNoteFact(
                key, span.label, span.text, "identifier", "openmed.synthetic.locale_phi"
            )
        )
    facts.extend(
        (
            SyntheticNoteFact("date", DATE, "2000-01-15", "identifier"),
            SyntheticNoteFact("diagnosis", DISEASE, "synthetic condition"),
            SyntheticNoteFact("medication", DRUG, "synthetic medication"),
            SyntheticNoteFact("dose", DOSAGE, "synthetic dose"),
            SyntheticNoteFact("procedure", PROCEDURE, "synthetic procedure"),
            SyntheticNoteFact(
                "distractor", DISEASE, "example-only distractor", "distractor"
            ),
        )
    )
    return tuple(facts)


def render_synthetic_note(
    document_type: str,
    facts: tuple[SyntheticNoteFact, ...],
    *,
    omitted_sections: tuple[str, ...] = (),
) -> SyntheticGrammarNote:
    """Render one declared grammar with original exact offsets and no fabricated facts."""
    if not isinstance(document_type, str) or document_type not in GRAMMARS:
        raise ValueError("unknown synthetic note grammar")
    facts = tuple(islice(facts, 129))
    if len(facts) > 128 or any(not isinstance(f, SyntheticNoteFact) for f in facts):
        raise ValueError("invalid synthetic note fact collection")
    by_key = {f.key: f for f in facts}
    if len(by_key) != len(facts):
        raise ValueError("duplicate synthetic note fact code")
    all_keys = {key for _section, keys in GRAMMARS[document_type] for key in keys}
    if set(by_key) - all_keys:
        raise ValueError("synthetic note fact not owned by grammar")
    if "dose" in by_key and "medication" not in by_key:
        raise ValueError("synthetic dose requires medication fact")
    omitted = tuple(islice(omitted_sections, 129))
    names = {section for section, _keys in GRAMMARS[document_type]}
    if (
        len(omitted) > 128
        or any(not isinstance(v, str) or v not in names for v in omitted)
        or len(set(omitted)) != len(omitted)
    ):
        raise ValueError("invalid synthetic note omission configuration")
    text = f"SYNTHETIC {document_type.upper()} NOTE\n"
    spans = []
    sections = []
    used = set()
    for section, keys in GRAMMARS[document_type]:
        present = [by_key[key] for key in keys if key in by_key]
        if section in omitted or not present:
            continue
        section_start = len(text)
        text += section.upper() + ":\n"
        for fact in present:
            text += fact.key + ": "
            start = len(text)
            text += fact.value + "\n"
            spans.append(
                SyntheticPhiSpan(
                    start,
                    start + len(fact.value),
                    fact.label,
                    fact.value,
                    {
                        "fact_id": fact.fact_id,
                        "role": fact.role,
                        "source": fact.source,
                        "section": section,
                    },
                )
            )
            used.add(fact.key)
        sections.append({"name": section, "start": section_start, "end": len(text)})
    return SyntheticGrammarNote(
        document_type,
        text,
        tuple(by_key[key] for key in sorted(used)),
        tuple(spans),
        tuple(sections),
    )


def validate_synthetic_note(note: SyntheticGrammarNote) -> tuple[str, ...]:
    """Independently check exact spans, fact provenance, role labels and section bounds."""
    if (
        not isinstance(note, SyntheticGrammarNote)
        or len(note.text) > 100000
        or len(note.gold_spans) > 1024
    ):
        raise ValueError("invalid synthetic grammar note")
    errors = set()
    if note.synthetic is not True or note.source != _SOURCE:
        errors.add("synthetic_provenance_missing")
    facts = {fact.fact_id: fact for fact in note.facts}
    expected = set()
    for fact in note.facts:
        prefix = fact.key + ": "
        pattern = re.escape(prefix + fact.value) + r"(?=\n|$)"
        for match in re.finditer(pattern, note.text):
            start = match.start() + len(prefix)
            expected.add((start, start + len(fact.value), fact.label, fact.fact_id))
    actual = {
        (span.start, span.end, span.label, span.metadata.get("fact_id"))
        for span in note.gold_spans
    }
    if actual != expected or len(actual) != len(note.gold_spans):
        errors.add("incomplete_or_duplicate_fact_truth")
    for span in note.gold_spans:
        if (
            type(span.start) is not int
            or type(span.end) is not int
            or not 0 <= span.start < span.end <= len(note.text)
            or note.text[span.start : span.end] != span.text
        ):
            errors.add("span_mismatch")
        fact = facts.get(span.metadata.get("fact_id"))
        if fact is None:
            errors.add("unknown_fact_reference")
        elif (
            fact.value != span.text
            or fact.label != span.label
            or fact.role != span.metadata.get("role")
            or fact.source != span.metadata.get("source")
        ):
            errors.add("fact_truth_mismatch")
        section = next(
            (s for s in note.sections if s["name"] == span.metadata.get("section")),
            None,
        )
        if (
            section is None
            or not section["start"] <= span.start < span.end <= section["end"]
        ):
            errors.add("section_bounds_mismatch")
    return tuple(sorted(errors))
