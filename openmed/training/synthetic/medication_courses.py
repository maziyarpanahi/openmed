"""Coherent synthetic medication timelines with explicit regimen and relation truth."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from itertools import islice

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _code(value: str) -> None:
    if not isinstance(value, str) or not _CODE.fullmatch(value):
        raise ValueError("invalid synthetic medication code")


def _day(value: int) -> None:
    if type(value) is not int or not -100000 <= value <= 100000:
        raise ValueError("invalid synthetic medication day")


def _dose(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 < value <= 1000000
    ):
        raise ValueError("invalid synthetic medication dose")
    return float(value)


@dataclass(frozen=True, slots=True)
class DoseChange:
    """One synthetic regimen change, never a clinical dose recommendation."""

    day: int
    dose: float

    def __post_init__(self) -> None:
        _day(self.day)
        object.__setattr__(self, "dose", _dose(self.dose))


@dataclass(frozen=True, slots=True)
class AdherenceEvent:
    """A declared synthetic administration or missed-dose observation."""

    day: int
    state: str

    def __post_init__(self) -> None:
        _day(self.day)
        if not isinstance(self.state, str) or self.state not in {"taken", "missed"}:
            raise ValueError("invalid synthetic medication adherence state")


@dataclass(frozen=True, slots=True)
class MedicationOutcome:
    """A synthetic observed outcome with no inferred causal medication claim."""

    day: int
    outcome_code: str

    def __post_init__(self) -> None:
        _day(self.day)
        _code(self.outcome_code)


@dataclass(frozen=True, slots=True)
class MedicationCourseSpec:
    """A bounded explicitly configured prescription, titration and stop timeline."""

    medication_code: str
    indication_code: str
    dose_unit: str
    route: str
    start_day: int
    stop_day: int
    initial_dose: float
    changes: tuple[DoseChange, ...] = ()
    adherence: tuple[AdherenceEvent, ...] = ()
    outcomes: tuple[MedicationOutcome, ...] = ()
    followup_days: int = 0

    def __post_init__(self) -> None:
        for value in (
            self.medication_code,
            self.indication_code,
            self.dose_unit,
            self.route,
        ):
            _code(value)
        _day(self.start_day)
        _day(self.stop_day)
        if self.start_day >= self.stop_day:
            raise ValueError(
                "synthetic medication course must have a positive duration"
            )
        object.__setattr__(self, "initial_dose", _dose(self.initial_dose))
        if type(self.followup_days) is not int or not 0 <= self.followup_days <= 365:
            raise ValueError("invalid synthetic medication followup window")
        for name, cls in (
            ("changes", DoseChange),
            ("adherence", AdherenceEvent),
            ("outcomes", MedicationOutcome),
        ):
            values = tuple(islice(getattr(self, name), 1001))
            if len(values) > 1000 or any(
                not isinstance(value, cls) for value in values
            ):
                raise ValueError("invalid synthetic medication event collection")
            if len(set(values)) != len(values):
                raise ValueError("duplicate synthetic medication event")
            object.__setattr__(
                self,
                name,
                tuple(sorted(values, key=lambda value: (value.day, repr(value)))),
            )
        days = [change.day for change in self.changes]
        if len(set(days)) != len(days) or any(
            not self.start_day < day < self.stop_day for day in days
        ):
            raise ValueError("invalid synthetic medication titration timeline")
        adherence_days = [event.day for event in self.adherence]
        if len(set(adherence_days)) != len(adherence_days) or any(
            not self.start_day <= day < self.stop_day for day in adherence_days
        ):
            raise ValueError("invalid synthetic medication adherence timeline")
        if any(
            not self.start_day <= event.day <= self.stop_day + self.followup_days
            for event in self.outcomes
        ):
            raise ValueError("invalid synthetic medication outcome timeline")


@dataclass(frozen=True, slots=True)
class RegimenInterval:
    """Half-open gold regimen interval in synthetic relative days."""

    regimen_id: str
    start_day: int
    end_day: int
    dose: float
    unit: str
    route: str


@dataclass(frozen=True, slots=True)
class MedicationCourse:
    """Synthetic course facts and gold relation labels, ready for offline fixtures."""

    course_id: str
    medication_code: str
    indication_code: str
    regimens: tuple[RegimenInterval, ...]
    events: tuple[dict[str, object], ...]
    relations: tuple[dict[str, str], ...]
    followup_end_day: int = 0


def generate_medication_course(
    spec: MedicationCourseSpec, *, seed: int = 0
) -> MedicationCourse:
    """Materialize declared synthetic facts with stable seeded IDs and gold links.

    Seeds vary opaque identities; declared doses and clinical outcomes are never
    randomly invented. No drug terminology or clinical recommendation is bundled.
    """
    if (
        not isinstance(spec, MedicationCourseSpec)
        or type(seed) is not int
        or not 0 <= seed <= 2**63 - 1
    ):
        raise ValueError("invalid synthetic medication generation settings")
    digest = hashlib.sha256((str(seed) + ":" + repr(spec)).encode()).hexdigest()
    course_id = "course_" + digest[:24]

    def identity(kind: str, index: int) -> str:
        return (
            kind
            + "_"
            + hashlib.sha256(f"{course_id}:{kind}:{index}".encode()).hexdigest()[:24]
        )

    starts = [spec.start_day] + [c.day for c in spec.changes]
    ends = starts[1:] + [spec.stop_day]
    doses = [spec.initial_dose] + [c.dose for c in spec.changes]
    regimens = tuple(
        RegimenInterval(
            identity("regimen", i), start, end, dose, spec.dose_unit, spec.route
        )
        for i, (start, end, dose) in enumerate(zip(starts, ends, doses))
    )
    events = []
    relations = []
    indication_id = identity("indication", 0)
    events.append(
        {
            "event_id": indication_id,
            "kind": "indication",
            "day": spec.start_day,
            "code": spec.indication_code,
        }
    )
    for i, regimen in enumerate(regimens):
        kind = "prescription" if i == 0 else "titration"
        event_id = identity(kind, i)
        events.append(
            {
                "event_id": event_id,
                "kind": kind,
                "day": regimen.start_day,
                "regimen_id": regimen.regimen_id,
            }
        )
        relations.append(
            {"source": indication_id, "target": event_id, "label": "indication_for"}
        )
        if i:
            relations.append(
                {
                    "source": event_id,
                    "target": regimens[i - 1].regimen_id,
                    "label": "modifies_regimen",
                }
            )
    for i, event in enumerate(spec.adherence):
        regimen = next(r for r in regimens if r.start_day <= event.day < r.end_day)
        event_id = identity("adherence", i)
        events.append(
            {
                "event_id": event_id,
                "kind": "adherence",
                "day": event.day,
                "state": event.state,
                "regimen_id": regimen.regimen_id,
            }
        )
        relations.append(
            {"source": event_id, "target": regimen.regimen_id, "label": "adherence_to"}
        )
    stop_id = identity("stop", 0)
    events.append(
        {"event_id": stop_id, "kind": "discontinuation", "day": spec.stop_day}
    )
    relations.append(
        {"source": stop_id, "target": regimens[-1].regimen_id, "label": "discontinues"}
    )
    for i, event in enumerate(spec.outcomes):
        event_id = identity("outcome", i)
        events.append(
            {
                "event_id": event_id,
                "kind": "outcome",
                "day": event.day,
                "code": event.outcome_code,
            }
        )
        relations.append(
            {"source": event_id, "target": course_id, "label": "observed_after_start"}
        )
    priority = {
        "indication": 0,
        "prescription": 1,
        "titration": 2,
        "adherence": 3,
        "discontinuation": 4,
        "outcome": 5,
    }
    events.sort(key=lambda e: (e["day"], priority[e["kind"]], e["event_id"]))
    return MedicationCourse(
        course_id,
        spec.medication_code,
        spec.indication_code,
        regimens,
        tuple(events),
        tuple(sorted(relations, key=lambda r: (r["source"], r["target"], r["label"]))),
        followup_end_day=spec.stop_day + spec.followup_days,
    )


def validate_medication_course(course: MedicationCourse) -> tuple[str, ...]:
    """Independently validate regimen continuity, temporal facts and complete gold links."""
    if not isinstance(course, MedicationCourse):
        raise ValueError("invalid synthetic medication course")
    if (
        not isinstance(course.regimens, tuple)
        or not 1 <= len(course.regimens) <= 1001
        or not isinstance(course.events, tuple)
        or len(course.events) > 3003
        or not isinstance(course.relations, tuple)
        or len(course.relations) > 5000
    ):
        raise ValueError("invalid synthetic medication course collections")
    errors = set()
    identity = re.compile(r"[a-z][a-z0-9_]{0,31}_[0-9a-f]{24}")
    if not isinstance(course.course_id, str) or not identity.fullmatch(
        course.course_id
    ):
        errors.add("invalid_identity")
    for code in (course.medication_code, course.indication_code):
        try:
            _code(code)
        except ValueError:
            errors.add("invalid_metadata")
    if type(course.followup_end_day) is not int:
        errors.add("invalid_followup_window")
    regimens = course.regimens
    for i, regimen in enumerate(regimens):
        if not isinstance(regimen, RegimenInterval):
            errors.add("invalid_regimen")
            continue
        try:
            _day(regimen.start_day)
            _day(regimen.end_day)
            _dose(regimen.dose)
            _code(regimen.unit)
            _code(regimen.route)
        except ValueError:
            errors.add("invalid_regimen")
        if not isinstance(regimen.regimen_id, str) or not identity.fullmatch(
            regimen.regimen_id
        ):
            errors.add("invalid_identity")
        if (
            type(regimen.start_day) is int
            and type(regimen.end_day) is int
            and regimen.start_day >= regimen.end_day
        ):
            errors.add("invalid_regimen")
        if (
            i
            and isinstance(regimens[i - 1], RegimenInterval)
            and regimens[i - 1].end_day != regimen.start_day
        ):
            errors.add("regimen_discontinuity")
    kinds = {
        "indication",
        "prescription",
        "titration",
        "adherence",
        "discontinuation",
        "outcome",
    }
    for event in course.events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("event_id"), str)
            or not identity.fullmatch(event["event_id"])
            or type(event.get("day")) is not int
            or not isinstance(event.get("kind"), str)
            or event["kind"] not in kinds
            or (
                event["kind"] in {"prescription", "titration", "adherence"}
                and (
                    not isinstance(event.get("regimen_id"), str)
                    or not identity.fullmatch(event["regimen_id"])
                )
            )
        ):
            errors.add("invalid_event")
    if errors & {
        "invalid_regimen",
        "invalid_event",
        "invalid_followup_window",
        "invalid_identity",
    }:
        return tuple(sorted(errors))
    if (
        course.followup_end_day < regimens[-1].end_day
        or course.followup_end_day > regimens[-1].end_day + 365
    ):
        errors.add("invalid_followup_window")
    all_ids = (
        [course.course_id]
        + [r.regimen_id for r in regimens]
        + [e["event_id"] for e in course.events]
    )
    ids = set(all_ids)
    if len(ids) != len(all_ids):
        errors.add("duplicate_identity")
    actual_relations = set()
    labels = {
        "indication_for",
        "modifies_regimen",
        "adherence_to",
        "discontinues",
        "observed_after_start",
    }
    for relation in course.relations:
        if (
            not isinstance(relation, dict)
            or set(relation) != {"source", "target", "label"}
            or any(not isinstance(v, str) for v in relation.values())
            or relation["label"] not in labels
        ):
            errors.add("invalid_relation")
            continue
        if relation["source"] not in ids or relation["target"] not in ids:
            errors.add("dangling_relation")
        actual_relations.add(
            (relation["source"], relation["target"], relation["label"])
        )
    if len(actual_relations) != len(course.relations):
        errors.add("duplicate_or_invalid_relation")
    expected = set()
    indications = [e for e in course.events if e["kind"] == "indication"]
    if (
        len(indications) != 1
        or indications[0].get("code") != course.indication_code
        or indications[0]["day"] != regimens[0].start_day
    ):
        errors.add("indication_mismatch")
    for i, regimen in enumerate(regimens):
        kind = "prescription" if i == 0 else "titration"
        events = [
            e
            for e in course.events
            if e["kind"] == kind and e.get("regimen_id") == regimen.regimen_id
        ]
        if len(events) != 1 or events[0]["day"] != regimen.start_day:
            errors.add("prescription_titration_mismatch")
            continue
        if len(indications) == 1:
            expected.add(
                (indications[0]["event_id"], events[0]["event_id"], "indication_for")
            )
        if i:
            expected.add(
                (events[0]["event_id"], regimens[i - 1].regimen_id, "modifies_regimen")
            )
    if len(
        [e for e in course.events if e["kind"] in {"prescription", "titration"}]
    ) != len(regimens):
        errors.add("prescription_titration_mismatch")
    stops = [e for e in course.events if e["kind"] == "discontinuation"]
    if len(stops) != 1 or stops[0]["day"] != regimens[-1].end_day:
        errors.add("stop_mismatch")
    if len(stops) == 1:
        expected.add((stops[0]["event_id"], regimens[-1].regimen_id, "discontinues"))
    regimen_by_id = {r.regimen_id: r for r in regimens}
    adherence_days = []
    for event in course.events:
        if event["kind"] == "adherence":
            regimen = regimen_by_id.get(event.get("regimen_id"))
            if (
                regimen is None
                or not regimen.start_day <= event["day"] < regimen.end_day
            ):
                errors.add("adherence_outside_regimen")
            if not isinstance(event.get("state"), str) or event["state"] not in {
                "taken",
                "missed",
            }:
                errors.add("invalid_adherence_state")
            adherence_days.append(event["day"])
            if regimen is not None:
                expected.add((event["event_id"], regimen.regimen_id, "adherence_to"))
        elif event["kind"] == "outcome":
            if not regimens[0].start_day <= event["day"] <= course.followup_end_day:
                errors.add("outcome_outside_followup")
            try:
                _code(event.get("code"))
            except ValueError:
                errors.add("invalid_outcome_code")
            expected.add((event["event_id"], course.course_id, "observed_after_start"))
    if len(set(adherence_days)) != len(adherence_days):
        errors.add("conflicting_adherence")
    if expected != actual_relations:
        errors.add("gold_relation_mismatch")
    return tuple(sorted(errors))
