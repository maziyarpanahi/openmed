"""Structured synthetic multilingual dialogue with source-bound gold truth."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType

from openmed.clinical.journey_contracts import (
    ClinicalArtifact,
    ClinicalFact,
    EvidenceLocator,
    canonical_digest,
    derived_opaque_id,
    sha256_digest,
)
from openmed.core.decoding.spans import iter_grapheme_clusters
from openmed.training.synthetic.locale_phi import LocalePhiGenerator, SyntheticPhiSpan

_ROLES = frozenset({"patient", "caregiver", "interpreter", "clinician"})
_LANGUAGES = frozenset({"en", "es", "hi"})
_CONCEPTS = {
    "symptom_cough": (
        "condition",
        {"en": "a fictional cough", "es": "tos ficticia", "hi": "काल्पनिक खाँसी"},
    ),
    "medication_example": (
        "medication",
        {
            "en": "demonstration medicine",
            "es": "medicina de demostración",
            "hi": "प्रदर्शन की दवा",
        },
    ),
    "procedure_example": (
        "procedure",
        {
            "en": "a fictional procedure",
            "es": "un procedimiento ficticio",
            "hi": "काल्पनिक प्रक्रिया",
        },
    ),
}
_STATUS = {
    "affirmed": {"en": "reports", "es": "informa", "hi": "बताता है"},
    "negated": {"en": "denies", "es": "niega", "hi": "इनकार करता है"},
    "uncertain": {
        "en": "is uncertain about",
        "es": "no está seguro de",
        "hi": "अनिश्चित है",
    },
}
_SUBJECT = {
    "patient": {
        "en": "Synthetic patient",
        "es": "Paciente ficticio",
        "hi": "काल्पनिक रोगी",
    },
    "caregiver": {
        "en": "Synthetic caregiver",
        "es": "Cuidador ficticio",
        "hi": "काल्पनिक देखभालकर्ता",
    },
    "interpreter": {
        "en": "Synthetic interpreter",
        "es": "Intérprete ficticio",
        "hi": "काल्पनिक दुभाषिया",
    },
    "clinician": {
        "en": "Synthetic clinician",
        "es": "Profesional ficticio",
        "hi": "काल्पनिक चिकित्सक",
    },
}
_CORRECTION = {"en": "Correction", "es": "Corrección", "hi": "सुधार"}


@dataclass(frozen=True)
class DialogueTurnSpec:
    """One authored semantic turn and its explicit within-turn language switches."""

    speaker_role: str
    languages: tuple[str, ...]
    concept: str = "symptom_cough"
    status: str = "affirmed"
    experiencer_role: str = "patient"
    correction_of: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.speaker_role, str)
            or self.speaker_role not in _ROLES
            or not isinstance(self.experiencer_role, str)
            or self.experiencer_role not in _ROLES
            or not isinstance(self.concept, str)
            or self.concept not in _CONCEPTS
            or not isinstance(self.status, str)
            or self.status not in _STATUS
        ):
            raise ValueError("invalid controlled dialogue semantic declaration")
        if (
            not isinstance(self.languages, Sequence)
            or isinstance(self.languages, str)
            or not 1 <= len(self.languages) <= 4
            or any(not isinstance(code, str) for code in self.languages)
            or not set(self.languages) <= _LANGUAGES
        ):
            raise ValueError("dialogue language segments must use supported templates")
        if any(
            left == right for left, right in zip(self.languages, self.languages[1:])
        ):
            raise ValueError("adjacent dialogue segments must declare an actual switch")
        if self.correction_of is not None and (
            isinstance(self.correction_of, bool)
            or not isinstance(self.correction_of, int)
            or self.correction_of < 0
        ):
            raise ValueError("invalid dialogue correction reference")
        object.__setattr__(self, "languages", tuple(self.languages))


@dataclass(frozen=True)
class SyntheticDialogueTurn:
    """Generated turn with grapheme-safe spans and existing Journey fact evidence."""

    index: int
    speaker_role: str
    text: str
    language_segments: tuple[tuple[str, int, int], ...]
    phi_spans: tuple[SyntheticPhiSpan, ...]
    artifact: ClinicalArtifact
    facts: tuple[ClinicalFact, ...]
    evidence: tuple[EvidenceLocator, ...]
    correction_of: int | None
    needs_adjudication: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or not 0 <= self.index < 128
            or not isinstance(self.speaker_role, str)
            or self.speaker_role not in _ROLES
            or not isinstance(self.text, str)
            or not 1 <= len(self.text) <= 20_000
        ):
            raise ValueError("invalid generated dialogue turn")
        if not isinstance(self.needs_adjudication, bool) or (
            self.correction_of is not None
            and (
                isinstance(self.correction_of, bool)
                or not isinstance(self.correction_of, int)
                or not 0 <= self.correction_of < self.index
            )
        ):
            raise ValueError("invalid dialogue adjudication declaration")
        if (
            not isinstance(self.artifact, ClinicalArtifact)
            or self.artifact.attributes.get("synthetic") is not True
            or self.artifact.content_hash != sha256_digest(self.text)
            or self.artifact.byte_size != len(self.text.encode())
        ):
            raise ValueError("dialogue artifact does not bind synthetic turn content")
        for values, limit in (
            (self.language_segments, 4),
            (self.phi_spans, 128),
            (self.facts, 4),
            (self.evidence, 128),
        ):
            if not isinstance(values, Sequence) or len(values) > limit:
                raise ValueError("provide bounded dialogue record sequences")
        if any(
            not isinstance(segment, Sequence) or len(segment) != 3
            for segment in self.language_segments
        ):
            raise ValueError("invalid language segment shape")
        segments = tuple(tuple(segment) for segment in self.language_segments)
        if not 1 <= len(segments) <= 4:
            raise ValueError("dialogue turn requires bounded language segments")
        boundaries = {0, len(self.text)} | {
            point for pair in iter_grapheme_clusters(self.text) for point in pair
        }
        previous_end = 0
        for language, start, end in segments:
            if (
                not isinstance(language, str)
                or language not in _LANGUAGES
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (start, end)
                )
                or not previous_end <= start < end <= len(self.text)
                or start not in boundaries
                or end not in boundaries
            ):
                raise ValueError("invalid grapheme-safe dialogue language segment")
            previous_end = end
        if segments[0][1] != 0 or segments[-1][2] != len(self.text):
            raise ValueError("language segments do not span the generated turn")
        spans = tuple(self.phi_spans)
        for span in spans:
            if (
                not isinstance(span, SyntheticPhiSpan)
                or span.start not in boundaries
                or span.end not in boundaries
                or self.text[span.start : span.end] != span.text
            ):
                raise ValueError(
                    "dialogue PHI span does not match grapheme-safe content"
                )
            if not any(
                language == span.metadata.get("language")
                and start <= span.start < span.end <= end
                for language, start, end in segments
            ):
                raise ValueError(
                    "dialogue PHI span lacks its matching language segment"
                )
        evidence = tuple(self.evidence)
        facts = tuple(self.facts)
        if not all(
            isinstance(locator, EvidenceLocator)
            and locator.artifact_id == self.artifact.artifact_id
            and locator.location_type == "text_span"
            and isinstance(locator.location.get("start"), int)
            and isinstance(locator.location.get("end"), int)
            and 0
            <= locator.location["start"]
            < locator.location["end"]
            <= len(self.text)
            and locator.location["start"] in boundaries
            and locator.location["end"] in boundaries
            for locator in evidence
        ) or len({locator.locator_id for locator in evidence}) != len(evidence):
            raise ValueError("dialogue evidence does not bind its artifact")
        locators = {locator.locator_id for locator in evidence}
        if (
            len(facts) != len(segments)
            or len({fact.fact_id for fact in facts if isinstance(fact, ClinicalFact)})
            != len(facts)
            or not all(
                isinstance(fact, ClinicalFact)
                and fact.attributes.get("synthetic") is True
                and bool(fact.evidence_ids)
                and set(fact.evidence_ids) <= locators
                and any(
                    language == fact.attributes.get("language")
                    and all(
                        start
                        <= locator.location["start"]
                        < locator.location["end"]
                        <= end
                        for locator in evidence
                        if locator.locator_id in fact.evidence_ids
                    )
                    for language, start, end in segments
                )
                and fact.attributes.get("needs_adjudication") is self.needs_adjudication
                for fact in facts
            )
        ):
            raise ValueError(
                "dialogue facts do not preserve declared evidence and adjudication"
            )
        for field, value in (
            ("language_segments", segments),
            ("phi_spans", spans),
            ("facts", facts),
            ("evidence", evidence),
        ):
            object.__setattr__(self, field, value)

    def to_training_item(self) -> dict[str, object]:
        """Export generated synthetic text, language tags and declared gold truth."""
        return {
            "synthetic": True,
            "source": "openmed.synthetic.code_switch_dialogue",
            "turn_index": self.index,
            "speaker_role": self.speaker_role,
            "text": self.text,
            "language_segments": [
                {"language": language, "start": start, "end": end}
                for language, start, end in self.language_segments
            ],
            "phi_spans": [span.to_dict() for span in self.phi_spans],
            "clinical_facts": [fact.to_dict() for fact in self.facts],
            "evidence": [locator.to_dict() for locator in self.evidence],
            "correction_of": self.correction_of,
            "needs_adjudication": self.needs_adjudication,
        }


@dataclass(frozen=True)
class CodeSwitchedDialogue:
    """Whole conversation with preserved contradictions, never automatic adjudication."""

    turns: tuple[SyntheticDialogueTurn, ...]
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 2**53
        ):
            raise ValueError("invalid dialogue seed")
        turns = tuple(self.turns)
        if not 1 <= len(turns) <= 128 or not all(
            isinstance(turn, SyntheticDialogueTurn) and turn.index == index
            for index, turn in enumerate(turns)
        ):
            raise ValueError("dialogue must contain ordered validated turns")
        object.__setattr__(self, "turns", turns)

    def report(self) -> dict[str, object]:
        """Return controlled roles/languages and counts without generated identities."""
        languages = Counter(
            language
            for turn in self.turns
            for language, _start, _end in turn.language_segments
        )
        roles = Counter(turn.speaker_role for turn in self.turns)
        return {
            "synthetic": True,
            "seed": self.seed,
            "turn_count": len(self.turns),
            "switch_count": sum(len(turn.language_segments) - 1 for turn in self.turns),
            "role_counts": dict(sorted(roles.items())),
            "language_segment_counts": dict(sorted(languages.items())),
            "adjudication_turn_count": sum(
                turn.needs_adjudication for turn in self.turns
            ),
            "artifact_digests": [turn.artifact.content_hash for turn in self.turns],
        }


def generate_code_switch_dialogue(
    specifications: Sequence[DialogueTurnSpec], *, seed: int = 0
) -> CodeSwitchedDialogue:
    """Generate semantically controlled multi-turn and within-turn switches offline.

    Existing locale generators supply synthetic PERSON values. Existing Journey
    facts/locators bind every clinical statement to its exact generated segment.
    Conflicting status declarations and uncertainty remain adjudication-required;
    explicit correction edges never erase the earlier utterance or its evidence.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**53:
        raise ValueError("dialogue seed must be a bounded non-negative integer")
    if not 1 <= len(specifications) <= 128 or not all(
        isinstance(spec, DialogueTurnSpec) for spec in specifications
    ):
        raise ValueError("provide 1 to 128 declared semantic turns")
    conflicts = set()
    observed = {}
    for index, spec in enumerate(specifications):
        key = spec.concept, spec.experiencer_role
        for previous in observed.get(key, []):
            if specifications[previous].status != spec.status:
                conflicts.update((previous, index))
        observed.setdefault(key, []).append(index)
        if spec.correction_of is not None:
            if spec.correction_of >= index:
                raise ValueError("correction must reference an earlier turn")
            previous = specifications[spec.correction_of]
            if (previous.concept, previous.experiencer_role) != key:
                raise ValueError("correction must preserve its original fact identity")
    turns = []
    for index, spec in enumerate(specifications):
        parts = []
        segments = []
        spans = []
        statement_spans = []
        cursor = 0
        for language in spec.languages:
            if parts:
                parts.append(" / ")
                cursor += 3
            role_seed = int.from_bytes(
                hashlib.sha256(
                    f"{seed}:{spec.experiencer_role}:{language}".encode()
                ).digest()[:6],
                "big",
            )
            name = next(
                span.text
                for span in LocalePhiGenerator(seed=role_seed)
                .generate(language)
                .gold_spans
                if span.label == "PERSON"
            )
            prefix = _SUBJECT[spec.experiencer_role][language] + " "
            correction = (
                _CORRECTION[language] + ": " if spec.correction_of is not None else ""
            )
            phrase = (
                correction
                + _STATUS[spec.status][language]
                + " "
                + _CONCEPTS[spec.concept][1][language]
            )
            body = prefix + name + ". " + phrase + "."
            start = cursor
            name_start = cursor + len(prefix)
            spans.append(
                SyntheticPhiSpan(
                    name_start,
                    name_start + len(name),
                    "PERSON",
                    name,
                    MappingProxyType(
                        {
                            "synthetic": True,
                            "language": language,
                            "entity_role": spec.experiencer_role,
                        }
                    ),
                )
            )
            statement_start = cursor + len(prefix) + len(name) + 2
            statement_spans.append(
                (language, statement_start, statement_start + len(phrase))
            )
            parts.append(body)
            cursor += len(body)
            segments.append((language, start, cursor))
        text = "".join(parts)
        boundaries = {0, len(text)} | {
            point for pair in iter_grapheme_clusters(text) for point in pair
        }
        if any(
            span.start not in boundaries
            or span.end not in boundaries
            or text[span.start : span.end] != span.text
            for span in spans
        ):
            raise ValueError("synthetic dialogue PHI spans are not grapheme aligned")
        artifact_id = derived_opaque_id("artifact", "synthetic-dialogue", seed, index)
        subject_id = derived_opaque_id(
            "subject", "synthetic-dialogue", seed, spec.experiencer_role
        )
        artifact = ClinicalArtifact(
            artifact_id,
            "synthetic_dialogue_turn",
            "text/plain",
            sha256_digest(text),
            len(text.encode()),
            derived_opaque_id("source", "synthetic-dialogue"),
            "2000-01-01T00:00:00Z",
            subject_id=subject_id,
            attributes={"synthetic": True, "speaker_role": spec.speaker_role},
        )
        facts = []
        evidence = []
        review = index in conflicts or spec.status == "uncertain"
        parents = (
            tuple(fact.fact_id for fact in turns[spec.correction_of].facts)
            if spec.correction_of is not None
            else ()
        )
        for segment_index, (language, start, end) in enumerate(statement_spans):
            locator_id = derived_opaque_id("evidence", artifact_id, segment_index)
            evidence.append(
                EvidenceLocator(
                    locator_id, artifact_id, "text_span", {"start": start, "end": end}
                )
            )
            facts.append(
                ClinicalFact(
                    derived_opaque_id("fact", artifact_id, segment_index),
                    subject_id,
                    _CONCEPTS[spec.concept][0],
                    {"synthetic_code": spec.concept},
                    spec.status,
                    (locator_id,),
                    canonical_digest(
                        {
                            "seed": seed,
                            "turn": index,
                            "language": language,
                            "concept": spec.concept,
                            "status": spec.status,
                        }
                    ),
                    parent_fact_ids=parents,
                    effective_time={"turn_index": index},
                    attributes={
                        "synthetic": True,
                        "speaker_role": spec.speaker_role,
                        "experiencer_role": spec.experiencer_role,
                        "language": language,
                        "needs_adjudication": review,
                    },
                )
            )
        turns.append(
            SyntheticDialogueTurn(
                index,
                spec.speaker_role,
                text,
                tuple(segments),
                tuple(spans),
                artifact,
                tuple(facts),
                tuple(evidence),
                spec.correction_of,
                review,
            )
        )
    return CodeSwitchedDialogue(tuple(turns), seed)
