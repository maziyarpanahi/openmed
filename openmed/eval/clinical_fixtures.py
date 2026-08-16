"""Deterministic, synthetic clinical documents for offline evaluation.

The generator in this module is intentionally model-free and local-only.  It
creates small clinical documents with character offsets, section boundaries,
assertion axes, and coded concepts so extraction profiles can be exercised
without downloading a corpus or putting a real identifier in a fixture.

``ClinicalFixture.to_dict()`` and ``ClinicalFixture.to_json()`` are
privacy-safe by default: they include offsets, codes, counts, and a document
hash, but not the document or span text.  A caller that needs to pass the
document to a local model can use the in-memory ``text`` attribute or opt in
to ``include_text=True`` for a local round trip.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

__all__ = [
    "ASSERTION_VALUES",
    "CLINICAL_FIXTURE_DISCLAIMER",
    "CLINICAL_FIXTURE_SCHEMA_VERSION",
    "DEFAULT_PROFILES",
    "DEFAULT_SEED",
    "SYNTHETIC_CODE_SYSTEM",
    "CERTAINTY_VALUES",
    "TEMPORALITY_VALUES",
    "CodedValue",
    "ClinicalFixture",
    "ClinicalSection",
    "ExpectedField",
    "GoldSpan",
    "available_profiles",
    "generate_clinical_fixture",
    "generate_clinical_fixtures",
    "generate_fixture",
    "generate_fixtures",
    "normalize_profile",
    "validate_fixture",
]

CLINICAL_FIXTURE_SCHEMA_VERSION = "openmed.eval.clinical_fixtures.v1"
CLINICAL_FIXTURE_DISCLAIMER = (
    "Synthetic evaluation data only; not clinical ground truth, a medical "
    "device, or a substitute for qualified clinical judgment."
)
DEFAULT_SEED = 0
SYNTHETIC_CODE_SYSTEM = "openmed.synthetic"
DEFAULT_PROFILES: tuple[str, ...] = (
    "progress_note",
    "radiology_report",
    "lab_report",
    "discharge_summary",
)

ASSERTION_VALUES: tuple[str, ...] = (
    "present",
    "absent",
    "uncertain",
    "historical",
    "hypothetical",
)
TEMPORALITY_VALUES: tuple[str, ...] = ("current", "historical", "future")
CERTAINTY_VALUES: tuple[str, ...] = ("certain", "uncertain")

_PROFILE_ALIASES = {
    "discharge": "discharge_summary",
    "discharge_note": "discharge_summary",
    "discharge_summary": "discharge_summary",
    "generic": "progress_note",
    "generic_note": "progress_note",
    "clinical_note": "progress_note",
    "lab": "lab_report",
    "lab_note": "lab_report",
    "lab_report": "lab_report",
    "labs": "lab_report",
    "pathology": "pathology_report",
    "pathology_report": "pathology_report",
    "progress": "progress_note",
    "progress_note": "progress_note",
    "progress_note_report": "progress_note",
    "radiology": "radiology_report",
    "radiology_note": "radiology_report",
    "radiology_report": "radiology_report",
}

_SAFE_METADATA_KEYS = frozenset(
    {
        "generator_version",
        "medical_device_disclaimer",
        "phi",
        "profile",
        "source",
        "synthetic",
    }
)


def _non_empty(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _seed_value(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return seed


def normalize_profile(profile: str) -> str:
    """Return the canonical name for a supported clinical document profile.

    Profile aliases are accepted to make the generator convenient at call
    sites that already use the shorter ``radiology`` or ``lab`` names.
    """

    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("profile must be a non-empty string")
    key = profile.strip().casefold().replace("-", "_").replace(" ", "_")
    canonical = _PROFILE_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(available_profiles())
        raise ValueError(
            f"unknown clinical fixture profile; expected one of: {supported}"
        )
    return canonical


def available_profiles() -> tuple[str, ...]:
    """Return the canonical profiles supported by the generator."""

    return tuple(sorted(set(_PROFILE_ALIASES.values())))


@dataclass(frozen=True, repr=False)
class CodedValue:
    """A local terminology reference attached to a gold span or field."""

    system: str
    code: str
    display: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", _non_empty(self.system, "code system"))
        object.__setattr__(self, "code", _non_empty(self.code, "code"))
        object.__setattr__(self, "display", _non_empty(self.display, "code display"))

    def __repr__(self) -> str:
        return f"CodedValue(system={self.system!r}, code={self.code!r})"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "CodedValue":
        """Build a code from a JSON-ready mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("coded values must be mappings")
        return cls(
            system=str(data.get("system") or data.get("code_system") or ""),
            code=str(data.get("code") or ""),
            display=str(data.get("display") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        """Return the code without any source-document text."""

        return {
            "code": self.code,
            "display": self.display,
            "system": self.system,
        }


@dataclass(frozen=True, repr=False)
class ClinicalSection:
    """A character range for one section of a generated document."""

    name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty(self.name, "section name"))
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("section offsets must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("section offsets must be integers")
        if self.start < 0 or self.end < self.start:
            raise ValueError("section offsets are inconsistent")

    def __repr__(self) -> str:
        return (
            f"ClinicalSection(name={self.name!r}, start={self.start}, end={self.end})"
        )

    @property
    def length(self) -> int:
        """Return the section length in Python character offsets."""

        return self.end - self.start

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ClinicalSection":
        """Build a section from a JSON-ready mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("clinical sections must be mappings")
        return cls(
            name=str(data.get("name") or data.get("label") or ""),
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
        )

    def to_dict(self) -> dict[str, int | str]:
        """Return section identity and offsets only."""

        return {"end": self.end, "name": self.name, "start": self.start}


@dataclass(frozen=True, repr=False)
class GoldSpan:
    """One offset-based clinical annotation in a synthetic document.

    The optional ``text`` is retained in memory for convenient local model
    tests, but is deliberately excluded from :meth:`to_dict` unless the
    caller explicitly opts into text serialization.
    """

    span_id: str
    label: str
    start: int
    end: int
    section: str
    assertion: str = "present"
    temporality: str = "current"
    certainty: str = "certain"
    experiencer: str = "patient"
    code: CodedValue | None = None
    text: str = ""

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.span_id, "span id"),
            (self.label, "span label"),
            (self.section, "span section"),
            (self.assertion, "span assertion"),
            (self.temporality, "span temporality"),
            (self.certainty, "span certainty"),
            (self.experiencer, "span experiencer"),
        ):
            _non_empty(value, field_name)
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise TypeError("span offsets must be integers")
        if not isinstance(self.start, int) or not isinstance(self.end, int):
            raise TypeError("span offsets must be integers")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span offsets are inconsistent")
        if self.assertion not in ASSERTION_VALUES:
            raise ValueError("span assertion is not supported")
        if self.temporality not in TEMPORALITY_VALUES:
            raise ValueError("span temporality is not supported")
        if self.certainty not in CERTAINTY_VALUES:
            raise ValueError("span certainty is not supported")
        if not isinstance(self.text, str):
            raise TypeError("span text must be a string")
        if self.text and len(self.text) != self.end - self.start:
            raise ValueError("span text length does not match offsets")
        if self.code is not None and not isinstance(self.code, CodedValue):
            raise TypeError("span code must be a CodedValue")
        object.__setattr__(self, "span_id", self.span_id.strip())
        object.__setattr__(self, "label", self.label.strip())
        object.__setattr__(self, "section", self.section.strip())
        object.__setattr__(self, "assertion", self.assertion.strip())
        object.__setattr__(self, "temporality", self.temporality.strip())
        object.__setattr__(self, "certainty", self.certainty.strip())
        object.__setattr__(self, "experiencer", self.experiencer.strip())

    def __repr__(self) -> str:
        return (
            f"GoldSpan(id={self.span_id!r}, label={self.label!r}, "
            f"start={self.start}, end={self.end})"
        )

    @property
    def is_negated(self) -> bool:
        """Whether the gold assertion explicitly marks the concept absent."""

        return self.assertion == "absent"

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_text: str = "",
    ) -> "GoldSpan":
        """Build a gold span from a JSON-ready mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("gold spans must be mappings")
        start = int(data.get("start", 0))
        end = int(data.get("end", start))
        raw_text = data.get("text")
        text = str(raw_text) if raw_text is not None else source_text[start:end]
        raw_code = data.get("code")
        code = (
            CodedValue.from_mapping(raw_code) if isinstance(raw_code, Mapping) else None
        )
        return cls(
            span_id=str(data.get("id") or data.get("span_id") or ""),
            label=str(data.get("label") or ""),
            start=start,
            end=end,
            section=str(data.get("section") or ""),
            assertion=str(data.get("assertion") or "present"),
            temporality=str(data.get("temporality") or "current"),
            certainty=str(data.get("certainty") or "certain"),
            experiencer=str(data.get("experiencer") or "patient"),
            code=code,
            text=text,
        )

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Return a raw-text-free annotation unless text is explicitly requested."""

        payload: dict[str, Any] = {
            "assertion": self.assertion,
            "certainty": self.certainty,
            "end": self.end,
            "experiencer": self.experiencer,
            "id": self.span_id,
            "label": self.label,
            "section": self.section,
            "start": self.start,
            "temporality": self.temporality,
        }
        if self.code is not None:
            payload["code"] = self.code.to_dict()
        if include_text:
            payload["text"] = self.text
        return payload


@dataclass(frozen=True, repr=False)
class ExpectedField:
    """One expected structured output field for a generated fixture."""

    field_id: str
    name: str
    value_type: str
    span_id: str | None = None
    value: str | int | float | bool | None = None
    code: CodedValue | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.field_id, "field id"),
            (self.name, "field name"),
            (self.value_type, "field value type"),
        ):
            _non_empty(value, field_name)
        if self.span_id is not None:
            _non_empty(self.span_id, "field span id")
        if self.code is not None and not isinstance(self.code, CodedValue):
            raise TypeError("field code must be a CodedValue")
        object.__setattr__(self, "field_id", self.field_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "value_type", self.value_type.strip())
        if self.span_id is not None:
            object.__setattr__(self, "span_id", self.span_id.strip())

    def __repr__(self) -> str:
        return f"ExpectedField(id={self.field_id!r}, name={self.name!r})"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ExpectedField":
        """Build an expected field from a JSON-ready mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("expected fields must be mappings")
        raw_code = data.get("code")
        code = (
            CodedValue.from_mapping(raw_code) if isinstance(raw_code, Mapping) else None
        )
        value = data.get("value")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError("expected field values must be scalar")
        return cls(
            field_id=str(data.get("id") or data.get("field_id") or ""),
            name=str(data.get("name") or ""),
            value_type=str(data.get("value_type") or ""),
            span_id=(str(data["span_id"]) if data.get("span_id") is not None else None),
            value=value,
            code=code,
        )

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        """Return a structured-field record without values when requested."""

        payload: dict[str, Any] = {
            "id": self.field_id,
            "name": self.name,
            "span_id": self.span_id,
            "value_type": self.value_type,
        }
        if include_value and self.value is not None:
            payload["value"] = self.value
        if self.code is not None:
            payload["code"] = self.code.to_dict()
        return payload


@dataclass(frozen=True, repr=False)
class ClinicalFixture:
    """A seeded synthetic document and its offset-based extraction gold."""

    fixture_id: str
    profile: str
    seed: int
    text: str
    sections: tuple[ClinicalSection, ...]
    gold_spans: tuple[GoldSpan, ...]
    expected_fields: tuple[ExpectedField, ...]
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fixture_id", _non_empty(self.fixture_id, "fixture id")
        )
        object.__setattr__(self, "profile", normalize_profile(self.profile))
        object.__setattr__(self, "seed", _seed_value(self.seed))
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("fixture text must be non-empty")
        object.__setattr__(self, "language", _non_empty(self.language, "language"))
        sections = tuple(self.sections)
        spans = tuple(self.gold_spans)
        fields = tuple(self.expected_fields)
        if not all(isinstance(section, ClinicalSection) for section in sections):
            raise TypeError("fixture sections must be ClinicalSection instances")
        if not all(isinstance(span, GoldSpan) for span in spans):
            raise TypeError("fixture gold spans must be GoldSpan instances")
        if not all(isinstance(item, ExpectedField) for item in fields):
            raise TypeError("fixture expected fields must be ExpectedField instances")
        object.__setattr__(self, "sections", sections)
        object.__setattr__(self, "gold_spans", spans)
        object.__setattr__(self, "expected_fields", fields)
        metadata = dict(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        self.validate()

    def __repr__(self) -> str:
        return (
            f"ClinicalFixture(id={self.fixture_id!r}, profile={self.profile!r}, "
            f"characters={len(self.text)}, spans={len(self.gold_spans)}, "
            f"fields={len(self.expected_fields)})"
        )

    @property
    def document_text(self) -> str:
        """Alias for ``text`` when passing a fixture to an extraction runner."""

        return self.text

    @property
    def synthetic(self) -> bool:
        """Return the explicit synthetic-data marker."""

        return True

    @property
    def phi(self) -> bool:
        """Return the explicit no-PHI marker."""

        return False

    @property
    def text_hash(self) -> str:
        """Return a stable, non-reversible document fingerprint."""

        return _sha256(self.text)

    @property
    def structured_fields(self) -> Mapping[str, dict[str, Any]]:
        """Return expected fields keyed by their unique public field name."""

        return MappingProxyType(
            {item.name: item.to_dict() for item in self.expected_fields}
        )

    @property
    def expected_structured_fields(self) -> tuple[ExpectedField, ...]:
        """Alias emphasizing that fields are evaluation expectations."""

        return self.expected_fields

    def span_text(self, span: GoldSpan | str) -> str:
        """Return one span's source text for local, in-memory evaluation."""

        target = span
        if isinstance(span, str):
            matches = [item for item in self.gold_spans if item.span_id == span]
            if len(matches) != 1:
                raise KeyError("unknown or ambiguous fixture span id")
            target = matches[0]
        if not isinstance(target, GoldSpan):
            raise TypeError("span must be a GoldSpan or span id")
        return self.text[target.start : target.end]

    def validate(self) -> None:
        """Validate offsets and cross-references without exposing source text."""

        section_ids = [section.name for section in self.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("fixture section names must be unique")
        previous_end = 0
        for section in self.sections:
            if section.end > len(self.text) or section.start < previous_end:
                raise ValueError("fixture section offsets are inconsistent")
            previous_end = section.end

        span_ids = [span.span_id for span in self.gold_spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("fixture span ids must be unique")
        known_sections = set(section_ids)
        for span in self.gold_spans:
            if span.end > len(self.text) or span.start >= span.end:
                raise ValueError("fixture span offsets are inconsistent")
            if span.text and self.text[span.start : span.end] != span.text:
                raise ValueError("fixture span text does not match its offsets")
            if span.section not in known_sections:
                raise ValueError("fixture span references an unknown section")
            section = next(item for item in self.sections if item.name == span.section)
            if span.start < section.start or span.end > section.end:
                raise ValueError("fixture span is outside its section")

        field_ids = [item.field_id for item in self.expected_fields]
        field_names = [item.name for item in self.expected_fields]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("fixture field ids must be unique")
        if len(field_names) != len(set(field_names)):
            raise ValueError("fixture field names must be unique")
        known_spans = set(span_ids)
        for item in self.expected_fields:
            if item.span_id is not None and item.span_id not in known_spans:
                raise ValueError("fixture field references an unknown span")

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        """Serialize the fixture, omitting document text by default."""

        payload: dict[str, Any] = {
            "expected_fields": [
                item.to_dict(include_value=include_text)
                for item in self.expected_fields
            ],
            "fixture_id": self.fixture_id,
            "gold_spans": [
                item.to_dict(include_text=include_text) for item in self.gold_spans
            ],
            "language": self.language,
            "metadata": _safe_metadata(self.metadata),
            "profile": self.profile,
            "schema_version": CLINICAL_FIXTURE_SCHEMA_VERSION,
            "sections": [item.to_dict() for item in self.sections],
            "seed": self.seed,
            "synthetic": self.synthetic,
            "text_sha256": self.text_hash,
            "phi": self.phi,
        }
        if include_text:
            payload["text"] = self.text
        return payload

    def to_json(self, *, include_text: bool = False, indent: int = 2) -> str:
        """Return deterministic JSON with raw document text excluded by default."""

        return (
            json.dumps(
                self.to_dict(include_text=include_text),
                ensure_ascii=False,
                indent=indent,
                sort_keys=True,
            )
            + "\n"
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ClinicalFixture":
        """Build a fixture from a text-inclusive JSON-ready mapping."""

        if not isinstance(data, Mapping):
            raise TypeError("clinical fixtures must be mappings")
        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("clinical fixture mapping requires text")
        sections = tuple(
            ClinicalSection.from_mapping(item) for item in data.get("sections", ())
        )
        spans = tuple(
            GoldSpan.from_mapping(item, source_text=text)
            for item in data.get("gold_spans", ())
        )
        fields = tuple(
            ExpectedField.from_mapping(item) for item in data.get("expected_fields", ())
        )
        return cls(
            fixture_id=str(data.get("fixture_id") or ""),
            profile=str(data.get("profile") or ""),
            seed=_seed_value(data.get("seed", DEFAULT_SEED)),
            text=text,
            sections=sections,
            gold_spans=spans,
            expected_fields=fields,
            language=str(data.get("language") or "en"),
            metadata=data.get("metadata") or {},
        )


def validate_fixture(fixture: ClinicalFixture) -> None:
    """Validate one generated fixture and return ``None`` on success."""

    if not isinstance(fixture, ClinicalFixture):
        raise TypeError("fixture must be a ClinicalFixture")
    fixture.validate()


@dataclass(frozen=True, repr=False)
class _Segment:
    value: str
    span_id: str | None = None
    label: str = ""
    assertion: str = "present"
    temporality: str = "current"
    certainty: str = "certain"
    experiencer: str = "patient"
    code: CodedValue | None = None


def _mention(
    span_id: str,
    value: str,
    label: str,
    *,
    assertion: str = "present",
    temporality: str = "current",
    certainty: str = "certain",
    experiencer: str = "patient",
    code: CodedValue | None = None,
) -> _Segment:
    return _Segment(
        value=value,
        span_id=span_id,
        label=label,
        assertion=assertion,
        temporality=temporality,
        certainty=certainty,
        experiencer=experiencer,
        code=code,
    )


@dataclass
class _DocumentBuilder:
    parts: list[str] = field(default_factory=list)
    sections: list[ClinicalSection] = field(default_factory=list)
    spans: list[GoldSpan] = field(default_factory=list)
    expected_fields: list[ExpectedField] = field(default_factory=list)
    _active_section: str | None = None

    @property
    def position(self) -> int:
        return sum(len(part) for part in self.parts)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def add_section(
        self,
        name: str,
        header: str,
        lines: Sequence[Sequence[str | _Segment]],
    ) -> None:
        section_name = _non_empty(name, "section name")
        start = self.position
        self.parts.append(f"{header}\n")
        self._active_section = section_name
        for line in lines:
            self._append_line(line)
        self.sections.append(
            ClinicalSection(name=section_name, start=start, end=self.position)
        )
        self._active_section = None

    def _append_line(self, segments: Sequence[str | _Segment]) -> None:
        if self._active_section is None:
            raise RuntimeError("fixture line must be inside a section")
        for segment in segments:
            if isinstance(segment, str):
                self.parts.append(segment)
                continue
            start = self.position
            self.parts.append(segment.value)
            if segment.span_id is not None:
                self.spans.append(
                    GoldSpan(
                        span_id=segment.span_id,
                        label=segment.label,
                        start=start,
                        end=self.position,
                        section=self._active_section,
                        assertion=segment.assertion,
                        temporality=segment.temporality,
                        certainty=segment.certainty,
                        experiencer=segment.experiencer,
                        code=segment.code,
                        text=segment.value,
                    )
                )
        self.parts.append("\n")

    def add_field(
        self,
        field_id: str,
        name: str,
        value_type: str,
        *,
        span_id: str | None = None,
        value: str | int | float | bool | None = None,
        code: CodedValue | None = None,
    ) -> None:
        if code is None and span_id is not None:
            code = next(
                (
                    span.code
                    for span in self.spans
                    if span.span_id == span_id and span.code is not None
                ),
                None,
            )
        self.expected_fields.append(
            ExpectedField(
                field_id=field_id,
                name=name,
                value_type=value_type,
                span_id=span_id,
                value=value,
                code=code,
            )
        )


def _code(code: str, display: str) -> CodedValue:
    return CodedValue(system=SYNTHETIC_CODE_SYSTEM, code=code, display=display)


def _identity(rng: random.Random) -> tuple[str, str]:
    subject = rng.choice(
        (
            "SYNTH-SUBJECT-ALPHA",
            "SYNTH-SUBJECT-BRAVO",
            "SYNTH-SUBJECT-CHARLIE",
            "SYNTH-SUBJECT-DELTA",
        )
    )
    date = rng.choice(
        (
            "2024-02-14",
            "2024-05-09",
            "2024-08-21",
            "2025-01-17",
        )
    )
    return subject, date


def _build_progress_note(rng: random.Random) -> _DocumentBuilder:
    builder = _DocumentBuilder()
    subject, encounter_date = _identity(rng)
    symptom = rng.choice(("cough", "fatigue", "nausea"))
    absent = rng.choice(("fever", "wheeze", "rash"))
    history = rng.choice(("asthma", "migraine", "eczema"))
    assessment = rng.choice(
        (
            "viral upper respiratory infection",
            "self-limited gastritis",
            "tension headache",
        )
    )
    plan = rng.choice(("supportive care", "oral hydration", "routine follow-up"))

    builder.add_section(
        "administrative",
        "ADMINISTRATIVE",
        (
            ("Document type: progress note",),
            ("Subject reference: ", _mention("subject_id", subject, "ID_NUM")),
            ("Encounter date: ", _mention("encounter_date", encounter_date, "DATE")),
        ),
    )
    builder.add_section(
        "history_of_present_illness",
        "HISTORY OF PRESENT ILLNESS",
        (
            (
                "The synthetic subject reports ",
                _mention(
                    "reported_condition",
                    symptom,
                    "CONDITION",
                    code=_code("condition-symptom", "other general symptoms"),
                ),
                " today.",
            ),
            (
                "The synthetic subject denies ",
                _mention(
                    "denied_condition",
                    absent,
                    "CONDITION",
                    assertion="absent",
                    certainty="certain",
                    code=_code("condition-absent", "symptom absent"),
                ),
                ".",
            ),
        ),
    )
    builder.add_section(
        "past_medical_history",
        "PAST MEDICAL HISTORY",
        (
            (
                "A history of ",
                _mention(
                    "historical_condition",
                    history,
                    "CONDITION",
                    assertion="historical",
                    temporality="historical",
                    code=_code("condition-history", "personal history of disease"),
                ),
                " is recorded.",
            ),
        ),
    )
    builder.add_section(
        "assessment",
        "ASSESSMENT",
        (
            (
                "Assessment: ",
                _mention(
                    "assessment_condition",
                    assessment,
                    "CONDITION",
                    certainty="uncertain",
                    assertion="uncertain",
                    code=_code("condition-uncertain", "illness, unspecified"),
                ),
                ".",
            ),
        ),
    )
    builder.add_section(
        "plan",
        "PLAN",
        (("Plan: ", _mention("care_plan", plan, "CARE_PLAN"), "."),),
    )
    builder.add_field("document_type", "document_type", "string", value="progress_note")
    builder.add_field(
        "subject_reference", "subject_reference", "identifier", span_id="subject_id"
    )
    builder.add_field(
        "encounter_date", "encounter_date", "date", span_id="encounter_date"
    )
    builder.add_field(
        "reported_condition",
        "reported_condition",
        "clinical_concept",
        span_id="reported_condition",
    )
    builder.add_field(
        "denied_condition",
        "denied_condition",
        "clinical_concept",
        span_id="denied_condition",
    )
    builder.add_field(
        "historical_condition",
        "historical_condition",
        "clinical_concept",
        span_id="historical_condition",
    )
    builder.add_field(
        "assessment_condition",
        "assessment_condition",
        "clinical_concept",
        span_id="assessment_condition",
    )
    builder.add_field("care_plan", "care_plan", "intervention", span_id="care_plan")
    return builder


def _build_radiology_report(rng: random.Random) -> _DocumentBuilder:
    builder = _DocumentBuilder()
    subject, study_date = _identity(rng)
    modality = rng.choice(("CT", "MRI", "ultrasound"))
    anatomy = rng.choice(("left lower lobe", "right upper lobe", "abdominal aorta"))
    finding = rng.choice(("small consolidation", "simple cyst", "mild thickening"))
    absent_finding = rng.choice(("pleural effusion", "acute fracture", "free fluid"))
    uncertain = rng.choice(("inflammatory change", "early infection", "benign nodule"))

    builder.add_section(
        "administrative",
        "ADMINISTRATIVE",
        (
            ("Document type: radiology report",),
            ("Subject reference: ", _mention("subject_id", subject, "ID_NUM")),
            ("Study date: ", _mention("study_date", study_date, "DATE")),
        ),
    )
    builder.add_section(
        "indication",
        "INDICATION",
        (
            (
                "Study requested for evaluation of ",
                _mention(
                    "indication_condition",
                    rng.choice(("cough", "abdominal pain", "headache")),
                    "CONDITION",
                    code=_code("indication-symptom", "symptom evaluation"),
                ),
                ".",
            ),
        ),
    )
    builder.add_section(
        "findings",
        "FINDINGS",
        (
            (
                "Modality: ",
                _mention(
                    "modality",
                    modality,
                    "IMAGING_MODALITY",
                    code=_code("imaging-modality", "radiographic imaging procedure"),
                ),
                ".",
            ),
            (
                "The ",
                _mention(
                    "anatomy",
                    anatomy,
                    "BODY_SITE",
                    code=_code("anatomy-body-site", "body structure"),
                ),
                " demonstrates ",
                _mention(
                    "finding",
                    finding,
                    "FINDING",
                    code=_code("finding-morphology", "morphologic finding"),
                ),
                ".",
            ),
            (
                "There is no ",
                _mention(
                    "absent_finding",
                    absent_finding,
                    "FINDING",
                    assertion="absent",
                    code=_code("finding-absent", "finding absent"),
                ),
                ".",
            ),
        ),
    )
    builder.add_section(
        "impression",
        "IMPRESSION",
        (
            (
                "The pattern may represent ",
                _mention(
                    "uncertain_finding",
                    uncertain,
                    "FINDING",
                    assertion="uncertain",
                    certainty="uncertain",
                    code=_code("finding-uncertain", "possible finding"),
                ),
                ".",
            ),
        ),
    )
    builder.add_field(
        "document_type", "document_type", "string", value="radiology_report"
    )
    builder.add_field(
        "subject_reference", "subject_reference", "identifier", span_id="subject_id"
    )
    builder.add_field("study_date", "study_date", "date", span_id="study_date")
    builder.add_field("modality", "modality", "coded_concept", span_id="modality")
    builder.add_field("anatomy", "anatomy", "coded_concept", span_id="anatomy")
    builder.add_field("finding", "finding", "coded_concept", span_id="finding")
    builder.add_field(
        "absent_finding", "absent_finding", "coded_concept", span_id="absent_finding"
    )
    builder.add_field(
        "uncertain_finding",
        "uncertain_finding",
        "coded_concept",
        span_id="uncertain_finding",
    )
    return builder


def _build_lab_report(rng: random.Random) -> _DocumentBuilder:
    builder = _DocumentBuilder()
    subject, specimen_date = _identity(rng)
    test_name, synthetic_code, unit, result = rng.choice(
        (
            ("serum sodium", "lab-serum-sodium", "mmol/L", 139),
            ("hemoglobin", "lab-hemoglobin", "g/dL", 13.4),
            ("serum creatinine", "lab-serum-creatinine", "mg/dL", 0.9),
        )
    )
    interpretation = rng.choice(
        ("stable renal function", "expected variation", "no critical change")
    )

    builder.add_section(
        "administrative",
        "ADMINISTRATIVE",
        (
            ("Document type: laboratory report",),
            ("Subject reference: ", _mention("subject_id", subject, "ID_NUM")),
            ("Specimen date: ", _mention("specimen_date", specimen_date, "DATE")),
        ),
    )
    builder.add_section(
        "results",
        "RESULTS",
        (
            (
                "Test: ",
                _mention(
                    "lab_test",
                    test_name,
                    "LAB_TEST",
                    code=_code(synthetic_code, "synthetic laboratory measurement"),
                ),
                ".",
            ),
            (
                "Result: ",
                _mention("lab_value", str(result), "LAB_VALUE"),
                " ",
                _mention("lab_unit", unit, "UNIT"),
                ".",
            ),
        ),
    )
    builder.add_section(
        "interpretation",
        "INTERPRETATION",
        (
            (
                "Interpretation: ",
                _mention(
                    "lab_interpretation",
                    interpretation,
                    "LAB_INTERPRETATION",
                    code=_code("lab-interpretation", "clinical interpretation"),
                ),
                ".",
            ),
            ("No critical abnormality is asserted.",),
        ),
    )
    builder.add_field("document_type", "document_type", "string", value="lab_report")
    builder.add_field(
        "subject_reference", "subject_reference", "identifier", span_id="subject_id"
    )
    builder.add_field("specimen_date", "specimen_date", "date", span_id="specimen_date")
    builder.add_field("lab_test", "lab_test", "coded_concept", span_id="lab_test")
    builder.add_field(
        "lab_result",
        "lab_result",
        "quantity",
        span_id="lab_value",
        value=result,
    )
    builder.add_field("lab_unit", "lab_unit", "unit", span_id="lab_unit", value=unit)
    builder.add_field(
        "lab_interpretation",
        "lab_interpretation",
        "coded_concept",
        span_id="lab_interpretation",
    )
    return builder


def _build_discharge_summary(rng: random.Random) -> _DocumentBuilder:
    builder = _DocumentBuilder()
    subject, discharge_date = _identity(rng)
    diagnosis = rng.choice(
        ("viral bronchitis", "uncomplicated migraine", "mild dermatitis")
    )
    medication = rng.choice(
        ("synthetic inhaler", "oral rehydration", "topical emollient")
    )
    follow_up = rng.choice(
        ("primary care review", "routine clinic review", "symptom diary")
    )
    absent = rng.choice(("new neurologic deficit", "respiratory distress", "fever"))

    builder.add_section(
        "administrative",
        "ADMINISTRATIVE",
        (
            ("Document type: discharge summary",),
            ("Subject reference: ", _mention("subject_id", subject, "ID_NUM")),
            ("Discharge date: ", _mention("discharge_date", discharge_date, "DATE")),
        ),
    )
    builder.add_section(
        "diagnosis",
        "DIAGNOSIS",
        (
            (
                "Final diagnosis: ",
                _mention(
                    "diagnosis",
                    diagnosis,
                    "CONDITION",
                    code=_code("diagnosis-unspecified", "illness, unspecified"),
                ),
                ".",
            ),
            (
                "No ",
                _mention(
                    "absent_diagnosis",
                    absent,
                    "CONDITION",
                    assertion="absent",
                    code=_code("diagnosis-absent", "finding absent"),
                ),
                " was documented.",
            ),
        ),
    )
    builder.add_section(
        "medications",
        "MEDICATIONS",
        (
            (
                "Continue ",
                _mention(
                    "medication",
                    medication,
                    "MEDICATION",
                    code=_code("medication-synthetic", "synthetic medication"),
                ),
                " as directed.",
            ),
        ),
    )
    builder.add_section(
        "follow_up",
        "FOLLOW-UP",
        (("Follow-up: ", _mention("follow_up", follow_up, "CARE_PLAN"), "."),),
    )
    builder.add_field(
        "document_type", "document_type", "string", value="discharge_summary"
    )
    builder.add_field(
        "subject_reference", "subject_reference", "identifier", span_id="subject_id"
    )
    builder.add_field(
        "discharge_date", "discharge_date", "date", span_id="discharge_date"
    )
    builder.add_field("diagnosis", "diagnosis", "coded_concept", span_id="diagnosis")
    builder.add_field(
        "absent_diagnosis",
        "absent_diagnosis",
        "coded_concept",
        span_id="absent_diagnosis",
    )
    builder.add_field("medication", "medication", "coded_concept", span_id="medication")
    builder.add_field("follow_up", "follow_up", "intervention", span_id="follow_up")
    return builder


def _build_pathology_report(rng: random.Random) -> _DocumentBuilder:
    builder = _DocumentBuilder()
    subject, specimen_date = _identity(rng)
    site = rng.choice(("skin biopsy", "colonic tissue", "lymph node"))
    finding = rng.choice(("benign change", "low-grade dysplasia", "reactive change"))
    diagnosis = rng.choice(
        ("no malignancy identified", "indeterminate atypia", "benign process")
    )

    builder.add_section(
        "administrative",
        "ADMINISTRATIVE",
        (
            ("Document type: pathology report",),
            ("Subject reference: ", _mention("subject_id", subject, "ID_NUM")),
            ("Specimen date: ", _mention("specimen_date", specimen_date, "DATE")),
        ),
    )
    builder.add_section(
        "specimen",
        "SPECIMEN",
        (("Specimen: ", _mention("specimen_site", site, "SPECIMEN"), "."),),
    )
    builder.add_section(
        "microscopy",
        "MICROSCOPY",
        (
            (
                "Microscopy shows ",
                _mention(
                    "microscopic_finding",
                    finding,
                    "FINDING",
                    code=_code("pathology-morphology", "morphologic finding"),
                ),
                ".",
            ),
        ),
    )
    builder.add_section(
        "diagnosis",
        "DIAGNOSIS",
        (
            (
                "Diagnosis: ",
                _mention(
                    "pathology_diagnosis",
                    diagnosis,
                    "DIAGNOSIS",
                    assertion="uncertain"
                    if "indeterminate" in diagnosis
                    else "present",
                    certainty="uncertain"
                    if "indeterminate" in diagnosis
                    else "certain",
                    code=_code("pathology-diagnosis", "patient diagnosis"),
                ),
                ".",
            ),
        ),
    )
    builder.add_field(
        "document_type", "document_type", "string", value="pathology_report"
    )
    builder.add_field(
        "subject_reference", "subject_reference", "identifier", span_id="subject_id"
    )
    builder.add_field("specimen_date", "specimen_date", "date", span_id="specimen_date")
    builder.add_field(
        "specimen_site", "specimen_site", "coded_concept", span_id="specimen_site"
    )
    builder.add_field(
        "microscopic_finding",
        "microscopic_finding",
        "coded_concept",
        span_id="microscopic_finding",
    )
    builder.add_field(
        "pathology_diagnosis",
        "pathology_diagnosis",
        "coded_concept",
        span_id="pathology_diagnosis",
    )
    return builder


_BUILDERS = {
    "discharge_summary": _build_discharge_summary,
    "lab_report": _build_lab_report,
    "pathology_report": _build_pathology_report,
    "progress_note": _build_progress_note,
    "radiology_report": _build_radiology_report,
}


def _sha256(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _profile_seed(seed: int, profile: str) -> int:
    material = f"{seed}:{profile}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _fixture_id(profile: str, seed: int) -> str:
    material = f"{profile}:{seed}".encode("utf-8")
    suffix = hashlib.sha256(material).hexdigest()[:12]
    return f"clinical-{profile}-{suffix}"


def _safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    safe_values = {
        "generator_version": CLINICAL_FIXTURE_SCHEMA_VERSION,
        "medical_device_disclaimer": CLINICAL_FIXTURE_DISCLAIMER,
        "phi": False,
        "source": "seeded local synthetic generator",
        "synthetic": True,
    }
    safe: dict[str, Any] = {
        key: safe_values[key]
        for key in sorted(_SAFE_METADATA_KEYS)
        if key in safe_values and metadata.get(key) == safe_values[key]
    }
    profile = metadata.get("profile")
    if isinstance(profile, str) and profile in available_profiles():
        safe["profile"] = profile
    return safe


def generate_fixture(
    profile: str = DEFAULT_PROFILES[0], seed: int = DEFAULT_SEED
) -> ClinicalFixture:
    """Generate one deterministic synthetic clinical fixture.

    Args:
        profile: Canonical profile or a supported alias such as ``radiology``.
        seed: Integer seed.  No global random state is read or modified.

    Returns:
        A validated fixture containing the document, sections, gold spans, and
        expected structured fields.
    """

    canonical = normalize_profile(profile)
    root_seed = _seed_value(seed)
    builder = _BUILDERS[canonical](random.Random(_profile_seed(root_seed, canonical)))
    fixture = ClinicalFixture(
        fixture_id=_fixture_id(canonical, root_seed),
        profile=canonical,
        seed=root_seed,
        text=builder.text,
        sections=tuple(builder.sections),
        gold_spans=tuple(builder.spans),
        expected_fields=tuple(builder.expected_fields),
        metadata={
            "generator_version": CLINICAL_FIXTURE_SCHEMA_VERSION,
            "medical_device_disclaimer": CLINICAL_FIXTURE_DISCLAIMER,
            "phi": False,
            "profile": canonical,
            "source": "seeded local synthetic generator",
            "synthetic": True,
        },
    )
    validate_fixture(fixture)
    return fixture


def generate_fixtures(
    profiles: str | Sequence[str] | None = None,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[ClinicalFixture, ...]:
    """Generate a deterministic tuple for selected profiles.

    ``None`` selects all default profiles.  Profile-specific seed derivation
    means changing the requested order does not change any document's content.
    """

    root_seed = _seed_value(seed)
    if profiles is None:
        requested: Sequence[str] = DEFAULT_PROFILES
    elif isinstance(profiles, str):
        requested = (profiles,)
    else:
        requested = profiles
    canonical_profiles = tuple(normalize_profile(profile) for profile in requested)
    if len(canonical_profiles) != len(set(canonical_profiles)):
        raise ValueError("profiles must not contain duplicates")
    return tuple(
        generate_fixture(profile, seed=root_seed) for profile in canonical_profiles
    )


generate_clinical_fixture = generate_fixture
generate_clinical_fixtures = generate_fixtures
