"""Seeded synthetic HL7 v2 messages with field-level PHI truth.

The generator renders ADT, ORU and ORM messages from synthetic identities and
records every PHI value it places in a truth manifest. Each truth value is
addressed by zero-based segment index plus one-based field, repetition,
component and subcomponent positions, matching :mod:`openmed.interop.hl7v2`
(``HL7Segment.get_field`` and the redaction ``coverage_report``).

Offsets are half-open ``[start, end)`` indexes into the generated message
string, including the ``\\r`` segment separators. They address the raw,
escaped text: ``message.text[start:end] == truth.raw`` always holds, and
``truth.value`` is the decoded value after HL7 escape sequences (``F``, ``S``,
``R``, ``E`` and ``T``) are resolved. Values without delimiter characters have
``raw == value``.

PHI values come from OpenMed surrogate providers (the ``Anonymizer`` and the
``clinical_ids`` generators). Timestamp dates come from the date surrogate
providers; their time, fraction and offset suffixes come from the seeded
generator. The leakage harness runs :func:`openmed.interop.hl7v2.redact_hl7v2`
and reports residual truth values by field position and label as counts only.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from numbers import Real
from typing import Any, Final, Union

from openmed.core import labels as L
from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers import clinical_ids
from openmed.interop.hl7v2 import (
    DEFAULT_FIELD_MAP,
    FieldKey,
    HL7Message,
    HL7V2Encoding,
    TextDeidentifier,
    parse_hl7v2,
    redact_hl7v2,
)

HL7V2_SYNTHETIC_SCHEMA_VERSION: Final = 1
SYNTHETIC_HL7V2_SOURCE: Final = "openmed.synthetic.hl7v2_messages"
HL7V2_SYNTHETIC_MESSAGE_TYPES: Final[tuple[str, ...]] = ("ADT", "ORU", "ORM")
HL7V2_ADT_EVENTS: Final[tuple[str, ...]] = ("A01", "A04", "A08", "A40")
HL7V2_TIMESTAMP_PRECISIONS: Final[tuple[str, ...]] = (
    "year",
    "month",
    "day",
    "minute",
    "second",
    "fraction",
)
DEFAULT_HL7V2_TIMEZONE_OFFSETS: Final[tuple[str, ...]] = ("", "+0000", "-0500", "+0530")
MAX_HL7V2_MESSAGES: Final = 1000
MAX_HL7V2_REPETITIONS: Final = 5
MAX_HL7V2_OBX_SEGMENTS: Final = 20
MAX_HL7V2_NTE_SEGMENTS: Final = 10
HL7V2_SURROGATE_PROVIDERS: Final[frozenset[str]] = frozenset(
    {
        "anonymizer.DATE",
        "anonymizer.DATE_OF_BIRTH",
        "anonymizer.EMAIL",
        "anonymizer.FIRST_NAME",
        "anonymizer.ID_NUM",
        "anonymizer.LAST_NAME",
        "anonymizer.LOCATION",
        "anonymizer.MIDDLE_NAME",
        "anonymizer.ORGANIZATION",
        "anonymizer.PERSON",
        "anonymizer.PHONE",
        "anonymizer.STREET_ADDRESS",
        "anonymizer.ZIPCODE",
        "clinical_ids.generate_luhn_identifier",
        "clinical_ids.generate_npi",
        "clinical_ids.generate_ssn",
    }
)

_SEGMENT_SEPARATOR: Final = "\r"
_TIMEZONE_RE: Final = re.compile(r"^[+-](?:0\d|1[0-4])[0-5]\d$")
_ISO_DATE_RE: Final = re.compile(r"(\d{4})\D?(\d{2})\D?(\d{2})")
_STATES: Final[tuple[str, ...]] = ("CA", "MA", "NY", "TX", "WA")
_LAB_TESTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("GLU", "Glucose", "mg/dL"),
    ("NA", "Sodium", "mmol/L"),
    ("K", "Potassium", "mmol/L"),
    ("HGB", "Hemoglobin", "g/dL"),
)
_OBX_VALUE_TYPES: Final[tuple[str, ...]] = ("NM", "ST", "TX", "FT")
_NOTE_TEMPLATES: Final[tuple[tuple[str, ...], ...]] = (
    ("Results reviewed with ", "{person}", " by phone at ", "{phone}", "."),
    ("Specimen recollected", "{delim}", "patient ", "{person}", " notified."),
    ("Follow-up visit", "{delim}", "booked for ", "{date}", "."),
)
_SHORT_TEMPLATES: Final[tuple[tuple[str, ...], ...]] = (
    ("Collected by ", "{person}"),
    ("Courier contact ", "{phone}"),
)


class HL7V2SynthesisError(ValueError):
    """Raised for invalid or unbounded synthetic HL7 v2 settings.

    Attributes:
        code: Stable, controlled failure code. Messages never echo the
            rejected input.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class HL7V2LeakageError(RuntimeError):
    """Raised when the leakage harness cannot evaluate a message.

    Attributes:
        code: Stable, controlled failure code.
        message_index: Zero-based index of the failing message, when known.
    """

    def __init__(self, code: str, message_index: int | None = None) -> None:
        self.code = code
        self.message_index = message_index
        where = "" if message_index is None else f" at message index {message_index}"
        super().__init__(f"{code}: HL7 v2 leakage evaluation failed{where}")


@dataclass(frozen=True)
class HL7V2SynthesisConfig:
    """Bounded settings for synthetic HL7 v2 message generation.

    Args:
        message_types: Message types to rotate through: ``ADT``, ``ORU`` or
            ``ORM``.
        adt_events: ADT trigger events to sample. ``A40`` adds an MRG segment.
        encoding: Field separator and MSH-2 encoding characters. All five
            characters must be distinct, non-alphanumeric and non-whitespace.
        max_repetitions: Upper bound for repeating PHI fields, from 1 to
            :data:`MAX_HL7V2_REPETITIONS`.
        max_obx_segments: Upper bound for OBX segments in ORU messages.
        max_nte_segments: Upper bound for NTE segments in ORU and ORM messages.
        z_segments: Whether to append a ``ZPD`` segment carrying PHI outside
            the default redaction map.
        escape_probability: Probability, from 0 to 1, that an organization
            value or free-text connector contains an encoding character that
            must be escaped.
        timestamp_precisions: Timestamp precisions to sample from
            :data:`HL7V2_TIMESTAMP_PRECISIONS`.
        timezone_offsets: Timestamp offset suffixes to sample. Empty strings
            omit the offset; other values must look like ``+HHMM``.

    Raises:
        HL7V2SynthesisError: If any setting is unknown, malformed or outside
            its bound.
    """

    message_types: tuple[str, ...] = HL7V2_SYNTHETIC_MESSAGE_TYPES
    adt_events: tuple[str, ...] = HL7V2_ADT_EVENTS
    encoding: HL7V2Encoding = HL7V2Encoding()
    max_repetitions: int = 2
    max_obx_segments: int = 3
    max_nte_segments: int = 2
    z_segments: bool = True
    escape_probability: float = 0.25
    timestamp_precisions: tuple[str, ...] = HL7V2_TIMESTAMP_PRECISIONS
    timezone_offsets: tuple[str, ...] = DEFAULT_HL7V2_TIMEZONE_OFFSETS

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_types",
            _choices(
                self.message_types,
                HL7V2_SYNTHETIC_MESSAGE_TYPES,
                "unknown_message_type",
                "message types must be drawn from ADT, ORM and ORU",
            ),
        )
        object.__setattr__(
            self,
            "adt_events",
            _choices(
                self.adt_events,
                HL7V2_ADT_EVENTS,
                "unknown_trigger_event",
                "ADT events must be drawn from A01, A04, A08 and A40",
            ),
        )
        _validate_encoding(self.encoding)
        _bounded_int(
            self.max_repetitions,
            1,
            MAX_HL7V2_REPETITIONS,
            "repetitions_out_of_bounds",
        )
        _bounded_int(
            self.max_obx_segments, 0, MAX_HL7V2_OBX_SEGMENTS, "segments_out_of_bounds"
        )
        _bounded_int(
            self.max_nte_segments, 0, MAX_HL7V2_NTE_SEGMENTS, "segments_out_of_bounds"
        )
        if not isinstance(self.z_segments, bool):
            raise HL7V2SynthesisError("invalid_flag", "z_segments must be a bool")
        probability = self.escape_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise HL7V2SynthesisError(
                "invalid_probability", "escape_probability must be within [0, 1]"
            )
        object.__setattr__(self, "escape_probability", float(probability))
        object.__setattr__(
            self,
            "timestamp_precisions",
            _choices(
                self.timestamp_precisions,
                HL7V2_TIMESTAMP_PRECISIONS,
                "invalid_timestamp_precision",
                "timestamp precisions must be drawn from "
                + ", ".join(HL7V2_TIMESTAMP_PRECISIONS),
                upper=False,
            ),
        )
        offsets = _string_tuple(
            self.timezone_offsets, "invalid_timezone_offset", "offsets"
        )
        if not offsets or any(
            offset and not _TIMEZONE_RE.match(offset) for offset in offsets
        ):
            raise HL7V2SynthesisError(
                "invalid_timezone_offset",
                "timezone offsets must be empty or +HHMM/-HHMM within 14 hours",
            )
        object.__setattr__(self, "timezone_offsets", offsets)


@dataclass(frozen=True)
class HL7V2TruthValue:
    """One PHI value placed in a synthetic HL7 v2 message.

    Attributes:
        segment_index: Zero-based index into ``HL7Message.segments``.
        segment: Three-character segment name.
        field: One-based HL7 field position (``MSH-1`` is the separator).
        repetition: One-based field repetition.
        component: One-based component within the repetition.
        subcomponent: One-based subcomponent within the component.
        start: Inclusive offset into the raw message text.
        end: Exclusive offset into the raw message text.
        label: Canonical OpenMed PHI label.
        kind: Controlled code describing the HL7 role of the value.
        provider: Surrogate provider that produced the value.
        value: Decoded value with HL7 escape sequences resolved.
        raw: Escaped value exactly as written, equal to ``text[start:end]``.
        whole_leaf: Whether the value fills its subcomponent, rather than
            being embedded in free text.
        in_default_map: Whether a ``DEFAULT_FIELD_MAP`` rule applies to the
            field, including OBX value-type gating.
    """

    segment_index: int
    segment: str
    field: int
    repetition: int
    component: int
    subcomponent: int
    start: int
    end: int
    label: str
    kind: str
    provider: str
    value: str
    raw: str
    whole_leaf: bool
    in_default_map: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible truth record with a fixed key order."""

        return {
            "segment_index": self.segment_index,
            "segment": self.segment,
            "field": self.field,
            "repetition": self.repetition,
            "component": self.component,
            "subcomponent": self.subcomponent,
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "kind": self.kind,
            "provider": self.provider,
            "value": self.value,
            "raw": self.raw,
            "whole_leaf": self.whole_leaf,
            "in_default_map": self.in_default_map,
        }


@dataclass(frozen=True)
class SyntheticHL7V2Message:
    """One synthetic HL7 v2 message and its field-level PHI truth.

    Attributes:
        text: Message text with ``\\r`` segment separators and a trailing
            separator.
        message_type: ``ADT``, ``ORU`` or ``ORM``.
        trigger_event: HL7 trigger event, such as ``A01`` or ``R01``.
        control_id: Synthetic MSH-10 control identifier.
        encoding: Delimiters used to render the message.
        truth: PHI truth values in message offset order.
        seed: Generator seed.
        index: Zero-based message index within the generator.
    """

    text: str
    message_type: str
    trigger_event: str
    control_id: str
    encoding: HL7V2Encoding
    truth: tuple[HL7V2TruthValue, ...]
    seed: int
    index: int

    def to_bytes(self) -> bytes:
        """Return the UTF-8 encoded message bytes."""

        return self.text.encode("utf-8")

    @property
    def text_sha256(self) -> str:
        """SHA-256 hex digest of the UTF-8 message bytes."""

        return hashlib.sha256(self.to_bytes()).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Return the JSON-compatible truth manifest for this message."""

        return {
            "schema_version": HL7V2_SYNTHETIC_SCHEMA_VERSION,
            "source": SYNTHETIC_HL7V2_SOURCE,
            "synthetic": True,
            "contains_real_phi": False,
            "seed": self.seed,
            "index": self.index,
            "message_type": self.message_type,
            "trigger_event": self.trigger_event,
            "control_id": self.control_id,
            "encoding": self.encoding.field + self.encoding.as_msh_2(),
            "segment_separator": _SEGMENT_SEPARATOR,
            "offset_basis": "message_text_code_points",
            "text_sha256": self.text_sha256,
            "truth": [truth.to_dict() for truth in self.truth],
        }

    @property
    def manifest_sha256(self) -> str:
        """SHA-256 hex digest of the canonical JSON manifest."""

        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _Phi:
    value: str
    label: str
    kind: str
    provider: str


@dataclass(frozen=True)
class _Text:
    """A subcomponent made of literal and PHI pieces, such as free text."""

    pieces: tuple[Union[str, _Phi], ...]


_Leaf = Union[str, _Phi, _Text]
_Component = tuple[_Leaf, ...]
_Repetition = tuple[_Component, ...]
_Field = tuple[_Repetition, ...]
_Segment = tuple[str, Mapping[int, _Field]]


def _rep(*components: _Leaf | _Component) -> _Repetition:
    return tuple(item if isinstance(item, tuple) else (item,) for item in components)


def _fld(*repetitions: _Repetition) -> _Field:
    return tuple(repetitions)


def _one(value: _Leaf) -> _Field:
    return (((value,),),)


class HL7V2MessageGenerator:
    """Generate deterministic synthetic HL7 v2 messages with PHI truth.

    Args:
        config: Generation settings. Defaults to :class:`HL7V2SynthesisConfig`.
        seed: Integer seed. Identical seeds and settings produce identical
            message bytes and manifests.

    Raises:
        HL7V2SynthesisError: If ``seed`` is not an integer.
    """

    def __init__(
        self,
        config: HL7V2SynthesisConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise HL7V2SynthesisError("invalid_seed", "seed must be an integer")
        if config is not None and not isinstance(config, HL7V2SynthesisConfig):
            raise HL7V2SynthesisError(
                "invalid_config", "config must be an HL7V2SynthesisConfig"
            )
        self.config = config or HL7V2SynthesisConfig()
        self.seed = seed
        self._index = 0
        # One consistent anonymizer shares its Faker instance across values.
        # Each value is seeded by its own synthetic original, derived from the
        # seed, message index and value slot.
        self._anonymizer = Anonymizer(lang="en", consistent=True, seed=seed)

    def generate(self, message_type: str | None = None) -> SyntheticHL7V2Message:
        """Generate the next message.

        Args:
            message_type: Optional ``ADT``, ``ORU`` or ``ORM`` override. By
                default the configured message types rotate in order.

        Returns:
            The rendered message with its truth manifest.

        Raises:
            HL7V2SynthesisError: If ``message_type`` is not supported or the
                generator already produced :data:`MAX_HL7V2_MESSAGES` messages.
        """

        index = self._index
        if index >= MAX_HL7V2_MESSAGES:
            raise HL7V2SynthesisError(
                "count_out_of_bounds",
                f"a generator produces at most {MAX_HL7V2_MESSAGES} messages",
            )
        if message_type is None:
            types = self.config.message_types
            message_type = types[index % len(types)]
        elif (
            not isinstance(message_type, str)
            or message_type.upper() not in HL7V2_SYNTHETIC_MESSAGE_TYPES
        ):
            raise HL7V2SynthesisError(
                "unknown_message_type",
                "message types must be drawn from ADT, ORM and ORU",
            )
        self._index += 1
        builder = _MessageBuilder(self.config, self.seed, index, self._anonymizer)
        return builder.build(message_type.upper())

    def generate_batch(self, count: int) -> tuple[SyntheticHL7V2Message, ...]:
        """Generate ``count`` messages, bounded by :data:`MAX_HL7V2_MESSAGES`.

        Raises:
            HL7V2SynthesisError: If ``count`` is not an integer within bounds.
        """

        _bounded_int(count, 1, MAX_HL7V2_MESSAGES, "count_out_of_bounds")
        return tuple(self.generate() for _ in range(count))


def generate_hl7v2_messages(
    count: int = 10,
    *,
    seed: int = 0,
    config: HL7V2SynthesisConfig | None = None,
) -> tuple[SyntheticHL7V2Message, ...]:
    """Generate synthetic HL7 v2 messages with field-level PHI truth.

    Args:
        count: Number of messages, from 1 to :data:`MAX_HL7V2_MESSAGES`.
        seed: Integer seed for deterministic output.
        config: Optional generation settings.

    Returns:
        Messages in generation order.

    Raises:
        HL7V2SynthesisError: If ``count``, ``seed`` or ``config`` is invalid.
    """

    _bounded_int(count, 1, MAX_HL7V2_MESSAGES, "count_out_of_bounds")
    return HL7V2MessageGenerator(config, seed=seed).generate_batch(count)


class _MessageBuilder:
    def __init__(
        self,
        config: HL7V2SynthesisConfig,
        seed: int,
        index: int,
        anonymizer: Anonymizer,
    ) -> None:
        self.config = config
        self._anonymizer = anonymizer
        self.seed = seed
        self.index = index
        self.encoding = config.encoding
        self._rng = random.Random(self._derive("message"))
        self._slot = 0
        self._msh_time = ""

    def build(self, message_type: str) -> SyntheticHL7V2Message:
        rng = self._rng
        if message_type == "ADT":
            event = rng.choice(self.config.adt_events)
        else:
            event = {"ORU": "R01", "ORM": "O01"}[message_type]
        control_id = f"SYN{self._derive('control') % 10**10:010d}"
        segments: list[_Segment] = [self._msh(message_type, event, control_id)]
        if message_type == "ADT":
            segments.append(("EVN", {1: _one(event), 2: _one(self._msh_time)}))
            segments.append(self._pid())
            segments.append(("PD1", {3: _one(self._organization("primary_facility"))}))
            segments.append(self._nk1())
            segments.append(self._pv1())
            if event == "A40":
                segments.append(self._mrg())
        else:
            segments.append(self._pid())
            segments.append(self._pv1())
            placer = f"ORD{rng.randrange(10**6):06d}"
            segments.append(self._orc(message_type, placer))
            test = rng.choice(_LAB_TESTS)
            segments.append(self._obr(placer, test))
            if message_type == "ORU":
                segments.extend(self._obx_segments(test))
            segments.extend(self._nte_segments())
        if self.config.z_segments:
            segments.append(self._zpd())

        text, truth = _render(segments, self.encoding)
        return SyntheticHL7V2Message(
            text=text,
            message_type=message_type,
            trigger_event=event,
            control_id=control_id,
            encoding=self.encoding,
            truth=truth,
            seed=self.seed,
            index=self.index,
        )

    # Segments ---------------------------------------------------------

    def _msh(self, message_type: str, event: str, control_id: str) -> _Segment:
        rng = self._rng
        self._msh_time = (
            f"2026{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
            f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}"
        )
        structure = f"{message_type}_{event}"
        return (
            "MSH",
            {
                3: _one("OPENMED-SYN"),
                4: _one("SYNTHETIC-FACILITY"),
                5: _one("OPENMED"),
                6: _one("LOCAL"),
                7: _one(self._msh_time),
                9: _fld(_rep(message_type, event, structure)),
                10: _one(control_id),
                11: _one("P"),
                12: _one("2.5"),
            },
        )

    def _pid(self) -> _Segment:
        rng = self._rng
        identifiers = tuple(
            _rep(
                self._id("mrn" if rep == 1 else "patient_identifier"),
                "",
                "",
                ("OPENMED", "2.16.840.1.113883.19.5", "ISO"),
                "MR" if rep == 1 else "PI",
            )
            for rep in self._reps()
        )
        names = tuple(
            self._xpn("patient_name", "L" if rep == 1 else "A") for rep in self._reps()
        )
        addresses = tuple(
            self._xad("H" if rep == 1 else "M", subcomponents=rng.random() < 0.5)
            for rep in self._reps()
        )
        phones = tuple(_rep(self._phone("home_phone")) for _ in self._reps())
        return (
            "PID",
            {
                1: _one("1"),
                3: _fld(*identifiers),
                4: _one(self._luhn(L.ID_NUM, "alternate_patient_id", 13)),
                5: _fld(*names),
                6: _fld(_rep(self._name(L.LAST_NAME, "mother_maiden_name"))),
                7: _one(self._timestamp(L.DATE_OF_BIRTH, "birth_datetime")),
                8: _one(rng.choice(("F", "M", "U"))),
                9: _fld(self._xpn("alias_name", "A")),
                11: _fld(*addresses),
                13: _fld(*phones),
                14: _one(self._phone("business_phone")),
                18: _one(self._luhn(L.ACCOUNT_NUMBER, "account_number", 16)),
                19: _one(self._ssn()),
                20: _fld(
                    _rep(
                        self._id("drivers_license", prefix="DLN"),
                        rng.choice(_STATES),
                    )
                ),
                21: _fld(_rep(self._id("mother_identifier"), "", "", "OPENMED", "MR")),
            },
        )

    def _nk1(self) -> _Segment:
        return (
            "NK1",
            {
                1: _one("1"),
                2: _fld(
                    _rep(
                        self._name(L.LAST_NAME, "next_of_kin_name"),
                        self._name(L.FIRST_NAME, "next_of_kin_name"),
                    )
                ),
                3: _fld(_rep("SPO", "Spouse")),
                4: _fld(self._xad("H", subcomponents=False)),
                5: _one(self._phone("next_of_kin_phone")),
                6: _one(self._phone("next_of_kin_business_phone")),
            },
        )

    def _pv1(self) -> _Segment:
        rng = self._rng
        return (
            "PV1",
            {
                1: _one("1"),
                2: _one(rng.choice(("I", "O", "E"))),
                3: _fld(_rep("WARD", f"{rng.randint(100, 499)}", "A")),
                7: _fld(self._xcn("attending_provider")),
                19: _one(self._luhn(L.ID_NUM, "visit_number", 13)),
                44: _one(self._timestamp(L.DATE, "admit_datetime")),
            },
        )

    def _mrg(self) -> _Segment:
        return (
            "MRG",
            {
                1: _fld(
                    _rep(self._id("prior_patient_identifier"), "", "", "OPENMED", "MR")
                ),
                3: _one(self._luhn(L.ACCOUNT_NUMBER, "prior_account_number", 16)),
                7: _fld(self._xpn("prior_patient_name", "L")),
            },
        )

    def _orc(self, message_type: str, placer: str) -> _Segment:
        return (
            "ORC",
            {
                1: _one("NW" if message_type == "ORM" else "RE"),
                2: _one(placer),
                12: _fld(self._xcn("ordering_provider")),
            },
        )

    def _obr(self, placer: str, test: tuple[str, str, str]) -> _Segment:
        return (
            "OBR",
            {
                1: _one("1"),
                2: _one(placer),
                4: _fld(_rep(test[0], test[1], "L")),
                7: _one(self._timestamp(L.DATE, "observation_datetime")),
                16: _fld(self._xcn("ordering_provider")),
            },
        )

    def _obx_segments(self, test: tuple[str, str, str]) -> list[_Segment]:
        rng = self._rng
        maximum = self.config.max_obx_segments
        count = rng.randint(1, maximum) if maximum else 0
        offset = rng.randrange(len(_OBX_VALUE_TYPES))
        segments: list[_Segment] = []
        for position in range(1, count + 1):
            value_type = _OBX_VALUE_TYPES[(offset + position) % len(_OBX_VALUE_TYPES)]
            if value_type == "NM":
                code = _rep(test[0], test[1], "L")
                value: _Field = _one(f"{rng.randint(10, 250)}")
                units = test[2]
            else:
                code = _rep("NOTE", "Clinical note", "L")
                templates = _SHORT_TEMPLATES if value_type == "ST" else _NOTE_TEMPLATES
                value = _one(self._free_text(rng.choice(templates), "obx_free_text"))
                units = ""
            segments.append(
                (
                    "OBX",
                    {
                        1: _one(str(position)),
                        2: _one(value_type),
                        3: _fld(code),
                        5: value,
                        6: _one(units),
                        11: _one("F"),
                        14: _one(self._timestamp(L.DATE, "observation_datetime")),
                    },
                )
            )
        return segments

    def _nte_segments(self) -> list[_Segment]:
        rng = self._rng
        count = rng.randint(0, self.config.max_nte_segments)
        segments: list[_Segment] = []
        for position in range(1, count + 1):
            comments = tuple(
                _rep(self._free_text(rng.choice(_NOTE_TEMPLATES), "note_free_text"))
                for _ in self._reps()
            )
            segments.append(
                ("NTE", {1: _one(str(position)), 2: _one("L"), 3: _fld(*comments)})
            )
        return segments

    def _zpd(self) -> _Segment:
        return (
            "ZPD",
            {
                1: _one("1"),
                2: _one(self._surrogate(L.EMAIL, "patient_email")),
                3: _fld(
                    _rep(
                        self._name(L.LAST_NAME, "preferred_name"),
                        self._name(L.FIRST_NAME, "preferred_name"),
                    )
                ),
                4: _one(self._organization("employer")),
            },
        )

    # Composite data types --------------------------------------------

    def _xpn(self, kind: str, name_type: str) -> _Repetition:
        middle: _Leaf = ""
        if self._rng.random() < 0.5:
            middle = self._name(L.MIDDLE_NAME, kind)
        return _rep(
            self._name(L.LAST_NAME, kind),
            self._name(L.FIRST_NAME, kind),
            middle,
            "",
            "",
            "",
            name_type,
        )

    def _xcn(self, kind: str) -> _Repetition:
        npi = clinical_ids.generate_npi(rng=random.Random(self._next_seed()))
        return _rep(
            _Phi(npi, L.ID_NUM, f"{kind}_npi", "clinical_ids.generate_npi"),
            self._name(L.LAST_NAME, f"{kind}_name"),
            self._name(L.FIRST_NAME, f"{kind}_name"),
            "",
            "",
            "DR",
        )

    def _xad(self, address_type: str, *, subcomponents: bool) -> _Repetition:
        street = self._surrogate(L.STREET_ADDRESS, "street_address")
        number, _, name = street.value.partition(" ")
        street_component: _Leaf | _Component = street
        if subcomponents and number.isdigit() and name:
            provider = street.provider
            street_component = (
                street,
                _Phi(name, L.STREET_ADDRESS, "street_name", provider),
                _Phi(number, L.BUILDING_NUMBER, "dwelling_number", provider),
            )
        return _rep(
            street_component,
            "",
            self._surrogate(L.LOCATION, "city"),
            self._rng.choice(_STATES),
            self._surrogate(L.ZIPCODE, "postal_code"),
            "USA",
            address_type,
        )

    # Values -----------------------------------------------------------

    def _surrogate(self, label: str, kind: str, *, prefix: str = "MRN") -> _Phi:
        original = _synthetic_original(label, self._next_seed(), prefix)
        value = self._anonymizer.surrogate(original, label, lang="en")
        return _Phi(" ".join(str(value).split()), label, kind, f"anonymizer.{label}")

    def _name(self, label: str, kind: str) -> _Phi:
        return self._surrogate(label, kind)

    def _phone(self, kind: str) -> _Phi:
        return self._surrogate(L.PHONE, kind)

    def _id(self, kind: str, *, prefix: str = "MRN") -> _Phi:
        return self._surrogate(L.ID_NUM, kind, prefix=prefix)

    def _luhn(self, label: str, kind: str, length: int) -> _Phi:
        value = clinical_ids.generate_luhn_identifier(
            length=length, rng=random.Random(self._next_seed())
        )
        return _Phi(value, label, kind, "clinical_ids.generate_luhn_identifier")

    def _ssn(self) -> _Phi:
        value = clinical_ids.generate_ssn(rng=random.Random(self._next_seed()))
        return _Phi(value, L.SSN, "ssn", "clinical_ids.generate_ssn")

    def _organization(self, kind: str) -> _Phi:
        first = self._surrogate(L.ORGANIZATION, kind)
        if self._rng.random() >= self.config.escape_probability:
            return first
        second = self._surrogate(L.ORGANIZATION, kind)
        value = f"{first.value} {self._delimiter()} {second.value}"
        return _Phi(value, L.ORGANIZATION, kind, first.provider)

    def _timestamp(self, label: str, kind: str) -> _Phi:
        source = self._surrogate(label, kind)
        match = _ISO_DATE_RE.search(source.value)
        if match is None:
            raise HL7V2SynthesisError(
                "provider_format", "date surrogate provider returned no date"
            )
        rng = self._rng
        precision = rng.choice(self.config.timestamp_precisions)
        value = "".join(match.groups())
        value = {"year": value[:4], "month": value[:6]}.get(precision, value)
        if precision in {"minute", "second", "fraction"}:
            value += f"{rng.randint(0, 23):02d}{rng.randint(0, 59):02d}"
        if precision in {"second", "fraction"}:
            value += f"{rng.randint(0, 59):02d}"
        if precision == "fraction":
            value += f".{rng.randint(0, 9999):04d}"
        value += rng.choice(self.config.timezone_offsets)
        return _Phi(value, label, kind, source.provider)

    def _free_text(self, template: Sequence[str], kind: str) -> _Text:
        pieces: list[str | _Phi] = []
        for token in template:
            if token == "{person}":
                pieces.append(self._surrogate(L.PERSON, kind))
            elif token == "{phone}":
                pieces.append(self._phone(kind))
            elif token == "{date}":
                pieces.append(self._surrogate(L.DATE, kind))
            elif token == "{delim}":
                if self._rng.random() < self.config.escape_probability:
                    pieces.append(f" {self._delimiter()} ")
                else:
                    pieces.append(" and ")
            else:
                pieces.append(token)
        return _Text(tuple(pieces))

    def _delimiter(self) -> str:
        encoding = self.encoding
        return self._rng.choice(
            (
                encoding.field,
                encoding.component,
                encoding.repetition,
                encoding.escape,
                encoding.subcomponent,
            )
        )

    def _reps(self) -> range:
        return range(1, self._rng.randint(1, self.config.max_repetitions) + 1)

    def _next_seed(self) -> int:
        self._slot += 1
        return self._derive(f"slot-{self._slot}")

    def _derive(self, name: str) -> int:
        material = f"{self.seed}|{self.index}|{name}".encode("utf-8")
        digest = hashlib.blake2b(material, digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False)


def _synthetic_original(label: str, token: int, prefix: str) -> str:
    """Return a synthetic, format-bearing original used only to seed a provider."""

    if label == L.PHONE:
        return (
            f"({token % 900 + 100}){token // 10**3 % 900 + 100}"
            f"-{token // 10**6 % 10**4:04d}"
        )
    if label == L.ID_NUM:
        return f"{prefix}{token % 10**7:07d}"
    if label == L.ZIPCODE:
        return f"{token % 10**5:05d}"
    if label == L.STREET_ADDRESS:
        return f"{token % 9000 + 100} Main Street"
    if label == L.EMAIL:
        return f"patient{token % 10**8}@example.org"
    if label == L.DATE_OF_BIRTH:
        return (date(1940, 1, 1) + timedelta(days=token % 25000)).isoformat()
    if label == L.DATE:
        return (date(2000, 1, 1) + timedelta(days=token % 9000)).isoformat()
    return f"Example {token % 10**8}"


def _render(
    segments: Sequence[_Segment], encoding: HL7V2Encoding
) -> tuple[str, tuple[HL7V2TruthValue, ...]]:
    chunks: list[str] = []
    truth: list[HL7V2TruthValue] = []
    cursor = 0

    def write(raw: str) -> None:
        nonlocal cursor
        chunks.append(raw)
        cursor += len(raw)

    for segment_index, (name, fields) in enumerate(segments):
        if segment_index:
            write(_SEGMENT_SEPARATOR)
        write(name)
        first = 3 if name == "MSH" else 1
        if name == "MSH":
            write(encoding.field + encoding.as_msh_2())
        for position in range(first, max(fields) + 1):
            write(encoding.field)
            in_map = _default_map_applies(name, position, fields, encoding)
            for repetition, rep in enumerate(fields.get(position, ()), start=1):
                if repetition > 1:
                    write(encoding.repetition)
                for component, comp in enumerate(rep, start=1):
                    if component > 1:
                        write(encoding.component)
                    for subcomponent, leaf in enumerate(comp, start=1):
                        if subcomponent > 1:
                            write(encoding.subcomponent)
                        pieces = leaf.pieces if isinstance(leaf, _Text) else (leaf,)
                        for piece in pieces:
                            if isinstance(piece, str):
                                write(_escape(piece, encoding))
                                continue
                            raw = _escape(piece.value, encoding)
                            start = cursor
                            write(raw)
                            truth.append(
                                HL7V2TruthValue(
                                    segment_index=segment_index,
                                    segment=name,
                                    field=position,
                                    repetition=repetition,
                                    component=component,
                                    subcomponent=subcomponent,
                                    start=start,
                                    end=cursor,
                                    label=piece.label,
                                    kind=piece.kind,
                                    provider=piece.provider,
                                    value=piece.value,
                                    raw=raw,
                                    whole_leaf=len(pieces) == 1,
                                    in_default_map=in_map,
                                )
                            )
    write(_SEGMENT_SEPARATOR)
    return "".join(chunks), tuple(truth)


def _default_map_applies(
    segment: str,
    position: int,
    fields: Mapping[int, _Field],
    encoding: HL7V2Encoding,
) -> bool:
    rule = DEFAULT_FIELD_MAP.get((segment, position))
    if rule is None:
        return False
    if not rule.allowed_type_values or rule.type_field is None:
        return True
    type_field = fields.get(rule.type_field, ())
    value_type = ""
    if type_field and type_field[0] and type_field[0][0]:
        leaf = type_field[0][0][0]
        value_type = leaf if isinstance(leaf, str) else ""
    return value_type.upper() in rule.allowed_type_values


def _escape(text: str, encoding: HL7V2Encoding) -> str:
    codes = {
        encoding.field: "F",
        encoding.component: "S",
        encoding.repetition: "R",
        encoding.escape: "E",
        encoding.subcomponent: "T",
    }
    return "".join(
        f"{encoding.escape}{codes[char]}{encoding.escape}" if char in codes else char
        for char in text
    )


def _unescape(raw: str, encoding: HL7V2Encoding) -> str:
    chars = {
        "F": encoding.field,
        "S": encoding.component,
        "R": encoding.repetition,
        "E": encoding.escape,
        "T": encoding.subcomponent,
    }
    escape = encoding.escape
    decoded: list[str] = []
    index = 0
    while index < len(raw):
        if raw[index] == escape and raw[index + 2 : index + 3] == escape:
            code = raw[index + 1 : index + 2]
            if code in chars:
                decoded.append(chars[code])
                index += 3
                continue
        decoded.append(raw[index])
        index += 1
    return "".join(decoded)


# Leakage harness ---------------------------------------------------------


@dataclass(frozen=True)
class HL7V2ResidualPosition:
    """Residual counts for one field position and label.

    Attributes:
        segment: Segment name.
        field: One-based field position.
        repetition: One-based repetition.
        component: One-based component.
        subcomponent: One-based subcomponent.
        label: Canonical PHI label.
        in_default_map: Whether a ``DEFAULT_FIELD_MAP`` rule applies to the
            truth values at this position.
        truth_count: Truth values placed at this position.
        residual_count: Truth values still present after redaction.
    """

    segment: str
    field: int
    repetition: int
    component: int
    subcomponent: int
    label: str
    in_default_map: bool
    truth_count: int
    residual_count: int

    @property
    def field_key(self) -> str:
        """Field position such as ``PID-6``."""

        return f"{self.segment}-{self.field}"

    @property
    def position(self) -> str:
        """Full position such as ``PID-3[2].1.1``."""

        return (
            f"{self.segment}-{self.field}[{self.repetition}]"
            f".{self.component}.{self.subcomponent}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, value-free position record."""

        return {
            "segment": self.segment,
            "field": self.field,
            "repetition": self.repetition,
            "component": self.component,
            "subcomponent": self.subcomponent,
            "label": self.label,
            "in_default_map": self.in_default_map,
            "truth_count": self.truth_count,
            "residual_count": self.residual_count,
        }


@dataclass(frozen=True)
class HL7V2LeakageReport:
    """Value-free residual PHI counts after HL7 v2 redaction.

    The report holds only positions, canonical labels and counts. It never
    stores, renders or raises with truth values or message text.

    Attributes:
        message_count: Messages evaluated.
        truth_count: Truth values evaluated.
        residual_count: Truth values still present at their position.
        positions: Counts for every truth position, sorted by position.
    """

    message_count: int
    truth_count: int
    residual_count: int
    positions: tuple[HL7V2ResidualPosition, ...]

    @property
    def residual_positions(self) -> tuple[HL7V2ResidualPosition, ...]:
        """Positions with at least one residual truth value."""

        return tuple(item for item in self.positions if item.residual_count)

    @property
    def residual_by_field(self) -> dict[str, int]:
        """Residual counts keyed by field position, such as ``PID-6``."""

        totals: dict[str, int] = {}
        for item in self.residual_positions:
            totals[item.field_key] = totals.get(item.field_key, 0) + (
                item.residual_count
            )
        return totals

    @property
    def residual_fields(self) -> tuple[str, ...]:
        """Field positions with residual truth values, in position order."""

        return tuple(self.residual_by_field)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, value-free report."""

        return {
            "schema_version": HL7V2_SYNTHETIC_SCHEMA_VERSION,
            "message_count": self.message_count,
            "truth_count": self.truth_count,
            "residual_count": self.residual_count,
            "residual_by_field": self.residual_by_field,
            "positions": [item.to_dict() for item in self.positions],
        }

    def to_json(self) -> str:
        """Return the report as deterministic JSON."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


def evaluate_hl7v2_leakage(
    messages: Iterable[SyntheticHL7V2Message],
    *,
    field_map: Mapping[FieldKey | str, Any] | None = None,
    deidentifier: TextDeidentifier | None = None,
    deidentify_kwargs: Mapping[str, Any] | None = None,
    date_shift_days: int | None = None,
    seed: int | None = 0,
) -> HL7V2LeakageReport:
    """Redact synthetic messages and count residual truth values.

    Each message runs through :func:`openmed.interop.hl7v2.redact_hl7v2`. A
    truth value is residual when its raw or decoded value is still present,
    as a whole token not adjacent to other letters or digits, in the redacted
    subcomponent at the same segment index, field, repetition, component and
    subcomponent.

    Args:
        messages: Generated messages with truth manifests, bounded by
            :data:`MAX_HL7V2_MESSAGES`.
        field_map: Optional redaction map forwarded to ``redact_hl7v2``.
        deidentifier: Optional OBX/NTE free-text de-identifier. Pass one for
            offline runs; the default loads the OpenMed PII pipeline.
        deidentify_kwargs: Optional keyword arguments for the de-identifier.
        date_shift_days: Optional fixed date shift forwarded to redaction.
        seed: Surrogate seed forwarded to redaction.

    Returns:
        Value-free residual counts by field position and label.

    Raises:
        HL7V2LeakageError: If an input is not a generated message, the batch
            is too large, or redaction fails. The error carries a controlled
            code and message index only.
    """

    items = tuple(messages)
    if not 1 <= len(items) <= MAX_HL7V2_MESSAGES:
        raise HL7V2LeakageError("count_out_of_bounds")
    counts: dict[tuple[str, int, int, int, int, str, bool], list[int]] = {}
    residual_total = 0
    for index, message in enumerate(items):
        if not isinstance(message, SyntheticHL7V2Message):
            raise HL7V2LeakageError("invalid_message", index)
        try:
            redacted = parse_hl7v2(
                redact_hl7v2(
                    message.text,
                    field_map=field_map,
                    deidentifier=deidentifier,
                    deidentify_kwargs=deidentify_kwargs,
                    date_shift_days=date_shift_days,
                    seed=seed,
                )
            )
        except Exception:
            # The underlying error may quote message content; drop it.
            raise HL7V2LeakageError("redaction_failed", index) from None
        for truth in message.truth:
            key = (
                truth.segment,
                truth.field,
                truth.repetition,
                truth.component,
                truth.subcomponent,
                truth.label,
                truth.in_default_map,
            )
            bucket = counts.setdefault(key, [0, 0])
            bucket[0] += 1
            if _is_residual(redacted, truth):
                bucket[1] += 1
                residual_total += 1

    positions = tuple(
        HL7V2ResidualPosition(*key, truth_count=total, residual_count=residual)
        for key, (total, residual) in sorted(counts.items())
    )
    return HL7V2LeakageReport(
        message_count=len(items),
        truth_count=sum(position.truth_count for position in positions),
        residual_count=residual_total,
        positions=positions,
    )


def _is_residual(message: HL7Message, truth: HL7V2TruthValue) -> bool:
    if truth.segment_index >= len(message.segments):
        return False
    segment = message.segments[truth.segment_index]
    encoding = message.encoding
    value = segment.get_field(truth.field) or ""
    try:
        repetition = value.split(encoding.repetition)[truth.repetition - 1]
        component = repetition.split(encoding.component)[truth.component - 1]
        leaf = component.split(encoding.subcomponent)[truth.subcomponent - 1]
    except IndexError:
        return False
    return _contains_token(leaf, truth.raw) or _contains_token(
        _unescape(leaf, encoding), truth.value
    )


def _contains_token(haystack: str, needle: str) -> bool:
    # A value counts only as a whole token, so "Lake Maureenview" does not
    # count as a residual "Lake Maureen", while "Benjamin Burns" does count
    # as a residual "Benjamin".
    pattern = rf"(?<![^\W_]){re.escape(needle)}(?![^\W_])"
    return re.search(pattern, haystack) is not None


# Validation helpers ------------------------------------------------------


def _bounded_int(value: Any, minimum: int, maximum: int, code: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise HL7V2SynthesisError(
            code, f"value must be an integer from {minimum} to {maximum}"
        )
    return value


def _string_tuple(values: Any, code: str, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise HL7V2SynthesisError(code, f"{name} must be a sequence of strings")
    if not all(isinstance(value, str) for value in values):
        raise HL7V2SynthesisError(code, f"{name} must be a sequence of strings")
    return tuple(values)


def _choices(
    values: Any,
    allowed: Sequence[str],
    code: str,
    message: str,
    *,
    upper: bool = True,
) -> tuple[str, ...]:
    items = _string_tuple(values, code, "values")
    normalized = tuple(item.upper() if upper else item for item in items)
    if not normalized or any(item not in allowed for item in normalized):
        raise HL7V2SynthesisError(code, message)
    return tuple(dict.fromkeys(normalized))


def _validate_encoding(encoding: Any) -> None:
    if not isinstance(encoding, HL7V2Encoding):
        raise HL7V2SynthesisError(
            "invalid_encoding", "encoding must be an HL7V2Encoding"
        )
    chars = (
        encoding.field,
        encoding.component,
        encoding.repetition,
        encoding.escape,
        encoding.subcomponent,
    )
    if (
        not all(isinstance(char, str) and len(char) == 1 for char in chars)
        or len(set(chars)) != len(chars)
        or any(char.isalnum() or char.isspace() for char in chars)
    ):
        raise HL7V2SynthesisError(
            "invalid_encoding",
            "encoding characters must be five distinct, non-alphanumeric, "
            "non-whitespace characters",
        )


__all__ = [
    "DEFAULT_HL7V2_TIMEZONE_OFFSETS",
    "HL7V2_ADT_EVENTS",
    "HL7V2_SURROGATE_PROVIDERS",
    "HL7V2_SYNTHETIC_MESSAGE_TYPES",
    "HL7V2_SYNTHETIC_SCHEMA_VERSION",
    "HL7V2_TIMESTAMP_PRECISIONS",
    "MAX_HL7V2_MESSAGES",
    "MAX_HL7V2_NTE_SEGMENTS",
    "MAX_HL7V2_OBX_SEGMENTS",
    "MAX_HL7V2_REPETITIONS",
    "SYNTHETIC_HL7V2_SOURCE",
    "HL7V2LeakageError",
    "HL7V2LeakageReport",
    "HL7V2MessageGenerator",
    "HL7V2ResidualPosition",
    "HL7V2SynthesisConfig",
    "HL7V2SynthesisError",
    "HL7V2TruthValue",
    "SyntheticHL7V2Message",
    "evaluate_hl7v2_leakage",
    "generate_hl7v2_messages",
]
