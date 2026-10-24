"""Deterministic rare-case privacy stress records with risk-contribution labels.

Average synthetic populations seldom contain the unusual attribute combinations
that make a record re-identifiable. This module composes repository-authored,
fictional vocabularies for geography, age, birth-date precision, diagnoses,
occupations, and family relationships into configurable rare combinations.
Every record carries truth labels for its direct identifiers and for the
quasi-identifier risk contribution of each dimension, so privacy attack suites
such as :func:`openmed.eval.attacks.linkage_attack` and
:func:`openmed.risk.reid.risk_report` can be scored against known ground truth.

Frequencies are synthetic priors of a plausible order of magnitude, not census,
registry, or prevalence statistics. Generation uses a caller-seeded local
``random.Random``, reads no files, logs nothing, and performs no network access.
Value-free reports carry opaque vocabulary codes, tiers, offsets, counts, and
digests; generated values appear only in the synthetic records themselves.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from itertools import combinations
from typing import Any, Final

from openmed.core.labels import (
    AGE,
    CONDITION,
    DATE,
    DATE_OF_BIRTH,
    DIRECT_IDENTIFIER,
    ID_NUM,
    LOCATION,
    OCCUPATION,
    PERSON,
    QUASI_IDENTIFIER,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    policy_label_for,
)

RARE_CASE_STRESS_SCHEMA_VERSION: Final = "openmed.training.rare_case_stress.v1"
SYNTHETIC_RARE_CASE_SOURCE: Final = "openmed.synthetic.rare_case_stress"

RARE_CASE_DIMENSIONS: Final[tuple[str, ...]] = (
    "geography",
    "age",
    "date",
    "diagnosis",
    "occupation",
    "family_relationship",
)
PREVALENCE_TIERS: Final[tuple[str, ...]] = ("common", "uncommon", "rare")
DIRECT_IDENTIFIER_DIMENSION: Final = "direct_identifier"

#: Columns that form the linkage key, in ``RARE_CASE_DIMENSIONS`` order.
RARE_CASE_QUASI_IDENTIFIER_FIELDS: Final[tuple[str, ...]] = (
    "region",
    "age",
    "birth_date",
    "diagnosis",
    "occupation",
    "family_relationship",
)
#: Synthetic direct-identifier columns emitted when direct identifiers are on.
RARE_CASE_DIRECT_IDENTIFIER_FIELDS: Final[tuple[str, ...]] = (
    "patient_name",
    "medical_record_number",
)

#: Size of the fictional reference population that geography shares and
#: expected-match estimates are expressed against.
SYNTHETIC_REFERENCE_POPULATION: Final = 10_000_000
DEFAULT_REFERENCE_DATE: Final = date(2025, 6, 30)

#: HIPAA Safe Harbor aggregates ages over 89 and geographic units of at most
#: 20,000 people; these thresholds only drive descriptive flags.
SAFE_HARBOR_MAX_AGE: Final = 89
SAFE_HARBOR_MAX_SMALL_AREA_POPULATION: Final = 20_000

SAFE_HARBOR_FLAGS: Final[tuple[str, ...]] = (
    "age_over_89",
    "date_indicates_age_over_89",
    "date_more_specific_than_year",
    "medical_record_number",
    "name",
    "small_geographic_unit",
)

_TIER_RISK_LEVELS: Final[Mapping[str, str]] = {
    "common": RISK_LOW,
    "uncommon": RISK_MEDIUM,
    "rare": RISK_HIGH,
}
_NOTE_HEADER: Final = "Synthetic rare-case stress note.\n"
_MIN_REFERENCE_YEAR: Final = 1900


@dataclass(frozen=True, slots=True)
class _DimensionSpec:
    field: str
    label: str | None
    reid_category: str
    prefix: str
    suffix: str = "."
    sensitive: bool = False


# ``reid_category`` reuses the openmed.risk.reid quasi-identifier vocabulary
# (age, date, geography, rare_condition) and extends it with occupation and
# family_relationship, which that profiler does not yet detect.
_DIMENSION_SPECS: Final[Mapping[str, _DimensionSpec]] = {
    "geography": _DimensionSpec(
        "region", LOCATION, "geography", "Residence: lives in "
    ),
    "age": _DimensionSpec("age", AGE, "age", "Age: ", " years old."),
    "date": _DimensionSpec("birth_date", DATE, "date", "Date of birth: "),
    "diagnosis": _DimensionSpec(
        "diagnosis", CONDITION, "rare_condition", "Diagnosis: ", sensitive=True
    ),
    "occupation": _DimensionSpec(
        "occupation", OCCUPATION, "occupation", "Occupation: "
    ),
    "family_relationship": _DimensionSpec(
        "family_relationship", None, "family_relationship", "Family: "
    ),
}
# (field, label, code suffix, Safe Harbor flag, note prefix)
_DIRECT_IDENTIFIER_SPECS: Final = (
    ("patient_name", PERSON, "person_name", "name", "Patient: "),
    (
        "medical_record_number",
        ID_NUM,
        "record_number",
        "medical_record_number",
        "MRN: ",
    ),
)


@dataclass(frozen=True, slots=True)
class _Entry:
    tier: str
    frequency: float
    surface: str = ""
    population: int = 0
    age_range: tuple[int, int] = (0, 0)
    precision: str = ""
    leap_day: bool = False
    relative_role: str | None = None


def _region(tier: str, surface: str, population: int) -> _Entry:
    return _Entry(
        tier,
        population / SYNTHETIC_REFERENCE_POPULATION,
        surface=surface,
        population=population,
    )


def _ages(tier: str, low: int, high: int, per_year_share: float) -> _Entry:
    return _Entry(tier, per_year_share, age_range=(low, high))


# Fictional places, repository-authored. Frequency is the region's share of the
# synthetic reference population.
_GEOGRAPHY: Final = (
    _region("common", "Quillmere", 2_400_000),
    _region("common", "Varnholt", 1_150_000),
    _region("common", "Brisenford", 640_000),
    _region("uncommon", "Ostermarsh", 48_000),
    _region("uncommon", "Lindqvale", 16_500),
    _region("rare", "Corrowen Isle", 1_800),
    _region("rare", "Thistlecairn", 640),
    _region("rare", "Fenwold Reach", 210),
)
# Frequency is the share of the population at each single year of age.
_AGE: Final = (
    _ages("common", 18, 29, 0.013),
    _ages("common", 30, 64, 0.014),
    _ages("uncommon", 85, 89, 0.0022),
    _ages("rare", 90, 99, 0.0006),
    _ages("rare", 100, 109, 0.00003),
)
# Birth dates are released at a precision. Frequency is the share of people of
# the same released age who share the released value, so the birth year adds
# no information beyond age and contributions are not double counted.
_DATE: Final = (
    _Entry("common", 1.0, precision="year"),
    _Entry("uncommon", 1 / 12, precision="month"),
    _Entry("rare", 1 / 365.25, precision="day"),
    _Entry("rare", 1 / 366, precision="day", leap_day=True),
)
_DIAGNOSIS: Final = (
    _Entry("common", 0.29, "essential hypertension"),
    _Entry("common", 0.11, "type 2 diabetes mellitus"),
    _Entry("common", 0.08, "asthma"),
    _Entry("uncommon", 0.006, "rheumatoid arthritis"),
    _Entry("uncommon", 0.005, "type 1 diabetes mellitus"),
    _Entry("uncommon", 0.003, "Crohn disease"),
    _Entry("rare", 0.0001, "cystic fibrosis"),
    _Entry("rare", 0.0001, "Huntington disease"),
    _Entry("rare", 0.000025, "Fabry disease"),
    _Entry("rare", 0.00002, "Gaucher disease"),
)
_OCCUPATION: Final = (
    _Entry("common", 0.025, "retail salesperson"),
    _Entry("common", 0.022, "schoolteacher"),
    _Entry("common", 0.02, "registered nurse"),
    _Entry("common", 0.018, "office clerk"),
    _Entry("uncommon", 0.005, "electrician"),
    _Entry("uncommon", 0.002, "pharmacist"),
    _Entry("uncommon", 0.0018, "firefighter"),
    _Entry("rare", 0.000015, "glassblower"),
    _Entry("rare", 0.000003, "falconer"),
    _Entry("rare", 0.000002, "harpsichord maker"),
    _Entry("rare", 0.000001, "lighthouse keeper"),
)
# Relative roles follow the openmed.clinical.family_history role vocabulary.
_FAMILY: Final = (
    _Entry("common", 0.45, "lives with spouse", relative_role="spouse"),
    _Entry("common", 0.3, "has one sibling", relative_role="sibling"),
    _Entry("common", 0.2, "parent of two children", relative_role="child"),
    _Entry("uncommon", 0.007, "identical twin", relative_role="sibling"),
    _Entry("uncommon", 0.002, "foster parent", relative_role="child"),
    _Entry("rare", 0.00002, "parent of eleven children", relative_role="child"),
    _Entry("rare", 0.00001, "one of quadruplets", relative_role="sibling"),
    _Entry("rare", 0.000003, "one of identical triplets", relative_role="sibling"),
)

_FIRST_NAMES: Final = tuple(
    "Avery Brennan Calla Dorian Elsa Florin Greer Hollis Isolde Jory"
    " Kestrel Linden Maren Niko Orla Perrin Quinn Rowan Saoirse Tamsin".split()
)
_LAST_NAMES: Final = tuple(
    "Ashgrove Bellweather Corriveau Dunmore Everholt Fairwick Gilmarsh"
    " Hawkridge Ingleby Jessamy Kettleby Larkspur Merrivale Northcote"
    " Oakhurst Pellow Quarrington Rookwood Stanwick Thornbury".split()
)


def _coded(
    dimension: str, entries: tuple[_Entry, ...]
) -> dict[str, tuple[tuple[str, _Entry], ...]]:
    by_tier: dict[str, tuple[tuple[str, _Entry], ...]] = {}
    for tier in PREVALENCE_TIERS:
        members = [entry for entry in entries if entry.tier == tier]
        by_tier[tier] = tuple(
            (f"{dimension}.{tier}.{ordinal:02d}", entry)
            for ordinal, entry in enumerate(members, 1)
        )
    return by_tier


_VOCABULARY: Final[Mapping[str, Mapping[str, tuple[tuple[str, _Entry], ...]]]] = {
    "geography": _coded("geography", _GEOGRAPHY),
    "age": _coded("age", _AGE),
    "date": _coded("date", _DATE),
    "diagnosis": _coded("diagnosis", _DIAGNOSIS),
    "occupation": _coded("occupation", _OCCUPATION),
    "family_relationship": _coded("family_relationship", _FAMILY),
}


@dataclass(frozen=True, slots=True)
class RareCaseProfile:
    """Requested prevalence tier per dimension for one stress combination.

    Dimensions named in neither ``rare`` nor ``uncommon`` are drawn from the
    common tier. Names are de-duplicated and stored in
    ``RARE_CASE_DIMENSIONS`` order, so equal requests yield equal profiles.

    Args:
        rare: Dimensions drawn from the rare tier.
        uncommon: Dimensions drawn from the uncommon tier.

    Raises:
        TypeError: If a tier list is a bare string or not iterable.
        ValueError: If a dimension is unknown or requested in both tiers.
    """

    rare: tuple[str, ...] = ()
    uncommon: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        rare = _validated_dimensions(self.rare, field_name="rare")
        uncommon = _validated_dimensions(self.uncommon, field_name="uncommon")
        if set(rare) & set(uncommon):
            raise ValueError("a dimension cannot be both rare and uncommon")
        object.__setattr__(self, "rare", rare)
        object.__setattr__(self, "uncommon", uncommon)

    @classmethod
    def from_tiers(cls, tiers: Mapping[str, str]) -> RareCaseProfile:
        """Build a profile from a ``{dimension: tier}`` mapping.

        Args:
            tiers: Mapping of dimension names to ``common``, ``uncommon``, or
                ``rare``. Omitted dimensions stay common.

        Returns:
            The equivalent profile.

        Raises:
            TypeError: If ``tiers`` is not a mapping.
            ValueError: If a tier is not one of ``PREVALENCE_TIERS``.
        """

        if not isinstance(tiers, Mapping):
            raise TypeError("tiers must be a mapping of dimension to tier")
        for tier in tiers.values():
            if not isinstance(tier, str) or tier not in PREVALENCE_TIERS:
                raise ValueError("tiers must be common, uncommon, or rare")
        return cls(
            rare=tuple(name for name, tier in tiers.items() if tier == "rare"),
            uncommon=tuple(name for name, tier in tiers.items() if tier == "uncommon"),
        )

    @property
    def profile_id(self) -> str:
        """Stable, value-free identifier such as ``rare:age+occupation``."""

        parts = []
        if self.rare:
            parts.append("rare:" + "+".join(self.rare))
        if self.uncommon:
            parts.append("uncommon:" + "+".join(self.uncommon))
        return ";".join(parts) or "common"

    def tier_for(self, dimension: str) -> str:
        """Return the requested tier for one dimension."""

        if dimension not in RARE_CASE_DIMENSIONS:
            raise ValueError("unknown rare-case dimension")
        if dimension in self.rare:
            return "rare"
        if dimension in self.uncommon:
            return "uncommon"
        return "common"


def default_rare_case_profiles(order: int = 2) -> tuple[RareCaseProfile, ...]:
    """Return every combination of ``order`` dimensions at the rare tier.

    Args:
        order: Number of simultaneously rare dimensions, from 1 to 6. The
            default of 2 yields the 15 pairwise rare combinations.

    Returns:
        Profiles in deterministic ``itertools.combinations`` order.

    Raises:
        ValueError: If ``order`` is outside 1..6.
    """

    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or not 1 <= order <= len(RARE_CASE_DIMENSIONS)
    ):
        raise ValueError("order must be an integer from 1 to 6")
    return tuple(
        RareCaseProfile(rare=combination)
        for combination in combinations(RARE_CASE_DIMENSIONS, order)
    )


@dataclass(frozen=True, slots=True)
class RareCaseStressConfig:
    """Configuration for one rare-case stress set.

    Args:
        profiles: Stress combinations, cycled in order across stress records.
            An empty value selects ``default_rare_case_profiles(2)``.
        background_classes: Number of all-common equivalence classes to mix in
            so stress records are rare relative to a crowd.
        background_class_size: Records per background class; each class shares
            one quasi-identifier tuple, so its members are at least k-anonymous
            with k equal to this size.
        include_direct_identifiers: Whether to generate a synthetic name and
            medical record number for every record.
        reference_date: Date at which ages are computed from birth dates.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a count is out of range.
    """

    profiles: tuple[RareCaseProfile, ...] = ()
    background_classes: int = 0
    background_class_size: int = 3
    include_direct_identifiers: bool = True
    reference_date: date = DEFAULT_REFERENCE_DATE

    def __post_init__(self) -> None:
        if isinstance(self.profiles, (str, bytes)) or not isinstance(
            self.profiles, Iterable
        ):
            raise TypeError("profiles must be a sequence of RareCaseProfile")
        profiles = tuple(self.profiles)
        if any(not isinstance(profile, RareCaseProfile) for profile in profiles):
            raise TypeError("profiles must be a sequence of RareCaseProfile")
        object.__setattr__(self, "profiles", profiles or default_rare_case_profiles())
        if (
            isinstance(self.background_classes, bool)
            or not isinstance(self.background_classes, int)
            or self.background_classes < 0
        ):
            raise ValueError("background_classes must be a non-negative integer")
        if (
            isinstance(self.background_class_size, bool)
            or not isinstance(self.background_class_size, int)
            or self.background_class_size < 2
        ):
            raise ValueError("background_class_size must be an integer >= 2")
        if not isinstance(self.include_direct_identifiers, bool):
            raise TypeError("include_direct_identifiers must be a boolean")
        if (
            not isinstance(self.reference_date, date)
            or isinstance(self.reference_date, datetime)
            or self.reference_date.year < _MIN_REFERENCE_YEAR
        ):
            raise ValueError("reference_date must be a date in or after 1900")

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free, JSON-compatible configuration summary."""

        return {
            "profiles": [profile.profile_id for profile in self.profiles],
            "background_classes": self.background_classes,
            "background_class_size": self.background_class_size,
            "include_direct_identifiers": self.include_direct_identifiers,
            "reference_date": self.reference_date.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RareCaseAttribute:
    """Truth label and risk contribution for one generated attribute.

    Args:
        dimension: One of ``RARE_CASE_DIMENSIONS`` or ``direct_identifier``.
        field: Column name in :meth:`RareCaseStressRecord.to_row`.
        label: Canonical OpenMed label, or ``None`` when no label exists.
        identifier_class: ``DIRECT_IDENTIFIER`` or ``QUASI_IDENTIFIER``.
        reid_category: Category in the re-identification risk vocabulary.
        tier: Prevalence tier, or ``None`` for direct identifiers.
        code: Opaque vocabulary code that never contains the value.
        value: Generated synthetic value, hidden from ``repr``.
        span: Half-open character offsets of the value in the note text.
        frequency: Synthetic population share of the released value.
        surprisal_bits: ``-log2(frequency)``, the attribute's information.
        contribution_share: Share of the record's combined surprisal.
        risk_level: ``low``, ``medium``, or ``high``.
        safe_harbor_flags: Descriptive Safe Harbor style flags.
        sensitive_attribute: Whether the value is a sensitive attribute.
        relative_role: Family-history relative role for family attributes.
    """

    dimension: str
    field: str
    label: str | None
    identifier_class: str
    reid_category: str
    tier: str | None
    code: str
    value: str | int = field(repr=False)
    span: tuple[int, int]
    frequency: float | None = None
    surprisal_bits: float | None = None
    contribution_share: float | None = None
    risk_level: str = RISK_HIGH
    safe_harbor_flags: tuple[str, ...] = ()
    sensitive_attribute: bool = False
    relative_role: str | None = None

    def to_dict(self, *, include_value: bool = False) -> dict[str, Any]:
        """Return a JSON-compatible truth label.

        Args:
            include_value: Whether to include the synthetic value. Reports
                keep the default of ``False``.

        Returns:
            Codes, tiers, offsets, and risk numbers, plus the value on request.
        """

        payload: dict[str, Any] = {
            "dimension": self.dimension,
            "field": self.field,
            "label": self.label,
            "identifier_class": self.identifier_class,
            "reid_category": self.reid_category,
            "tier": self.tier,
            "code": self.code,
            "span": list(self.span),
            "frequency": self.frequency,
            "surprisal_bits": self.surprisal_bits,
            "contribution_share": self.contribution_share,
            "risk_level": self.risk_level,
            "safe_harbor_flags": list(self.safe_harbor_flags),
            "sensitive_attribute": self.sensitive_attribute,
            "relative_role": self.relative_role,
        }
        if include_value:
            payload["value"] = self.value
        return payload


@dataclass(frozen=True, slots=True)
class RareCaseStressRecord:
    """One synthetic record, its note text, and its risk truth labels.

    Attributes:
        record_id: Stable synthetic identifier.
        case_kind: ``stress`` for requested combinations, ``background`` for
            all-common crowd records.
        profile_id: Value-free identifier of the requested tier profile.
        text: Synthetic note embedding every attribute, hidden from ``repr``.
        attributes: Direct identifiers first, then the six stress dimensions.
        equivalence_class_size: Records in the generated set that share this
            record's quasi-identifier tuple (the dataset k for this record).
        combined_surprisal_bits: Sum of stress-dimension surprisal, assuming
            independent attributes.
        expected_population_matches: Expected people in the synthetic
            reference population who share the tuple.
        population_unique_probability: Poisson estimate that no other person
            in the reference population shares the tuple.
        dominant_dimension: Stress dimension with the largest contribution.
    """

    record_id: str
    case_kind: str
    profile_id: str
    text: str = field(repr=False)
    attributes: tuple[RareCaseAttribute, ...]
    equivalence_class_size: int
    combined_surprisal_bits: float
    expected_population_matches: float
    population_unique_probability: float
    dominant_dimension: str
    synthetic: bool = True
    source: str = SYNTHETIC_RARE_CASE_SOURCE

    @property
    def stress_attributes(self) -> tuple[RareCaseAttribute, ...]:
        """The six linkage-key attributes in ``RARE_CASE_DIMENSIONS`` order."""

        return tuple(
            attribute
            for attribute in self.attributes
            if attribute.dimension != DIRECT_IDENTIFIER_DIMENSION
        )

    @property
    def direct_identifiers(self) -> tuple[RareCaseAttribute, ...]:
        """Attributes whose identifier class is ``DIRECT_IDENTIFIER``."""

        return tuple(
            attribute
            for attribute in self.attributes
            if attribute.identifier_class == DIRECT_IDENTIFIER
        )

    @property
    def quasi_identifiers(self) -> tuple[RareCaseAttribute, ...]:
        """Attributes whose identifier class is ``QUASI_IDENTIFIER``."""

        return tuple(
            attribute
            for attribute in self.attributes
            if attribute.identifier_class == QUASI_IDENTIFIER
        )

    @property
    def rare_dimensions(self) -> tuple[str, ...]:
        """Stress dimensions drawn from the rare tier."""

        return tuple(
            attribute.dimension
            for attribute in self.stress_attributes
            if attribute.tier == "rare"
        )

    @property
    def is_dataset_singleton(self) -> bool:
        """Whether no other generated record shares the quasi-identifier tuple."""

        return self.equivalence_class_size == 1

    def to_row(self, *, include_direct_identifiers: bool = True) -> dict[str, Any]:
        """Return a flat table row for tabular privacy attacks.

        Args:
            include_direct_identifiers: Whether to keep the synthetic name and
                medical record number columns. A full birth date stays in the
                row because it is part of the linkage key.

        Returns:
            ``record_id`` followed by attribute columns with typed values.
        """

        row: dict[str, Any] = {"record_id": self.record_id}
        for attribute in self.attributes:
            if (
                attribute.dimension == DIRECT_IDENTIFIER_DIMENSION
                and not include_direct_identifiers
            ):
                continue
            row[attribute.field] = attribute.value
        return row

    def risk_labels(self) -> dict[str, Any]:
        """Return value-free risk truth labels for privacy attack suites.

        Returns:
            Record aggregates and per-attribute codes, tiers, offsets, and
            contributions. Generated values and note text are never included.
        """

        return {
            "record_id": self.record_id,
            "case_kind": self.case_kind,
            "profile_id": self.profile_id,
            "equivalence_class_size": self.equivalence_class_size,
            "dataset_singleton": self.is_dataset_singleton,
            "direct_identifier_count": len(self.direct_identifiers),
            "quasi_identifier_count": len(self.quasi_identifiers),
            "rare_dimensions": list(self.rare_dimensions),
            "combined_surprisal_bits": self.combined_surprisal_bits,
            "expected_population_matches": self.expected_population_matches,
            "population_unique_probability": self.population_unique_probability,
            "dominant_dimension": self.dominant_dimension,
            "attributes": [attribute.to_dict() for attribute in self.attributes],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible synthetic training record with provenance.

        ``spans`` lists labelled attributes in the span shape OpenMed risk
        profilers read; ``risk`` holds the value-free truth labels.
        """

        row = self.to_row()
        row.pop("record_id")
        return {
            "record_id": self.record_id,
            "text": self.text,
            "spans": [
                {
                    "start": attribute.span[0],
                    "end": attribute.span[1],
                    "label": attribute.label,
                    "category": attribute.reid_category,
                    "identifier_class": attribute.identifier_class,
                    "field": attribute.field,
                }
                for attribute in self.attributes
                if attribute.label is not None
            ],
            "fields": row,
            "risk": self.risk_labels(),
            "metadata": {
                "synthetic": self.synthetic,
                "source": self.source,
                "restricted_data": False,
                "schema_version": RARE_CASE_STRESS_SCHEMA_VERSION,
            },
        }


@dataclass(frozen=True, slots=True)
class RareCaseStressSet:
    """A generated rare-case stress set with value-free summaries."""

    records: tuple[RareCaseStressRecord, ...]
    seed: int
    config: RareCaseStressConfig
    synthetic: bool = True
    source: str = SYNTHETIC_RARE_CASE_SOURCE

    @property
    def quasi_identifier_fields(self) -> tuple[str, ...]:
        """Linkage-key columns for ``linkage_attack`` and ``risk_report``."""

        return RARE_CASE_QUASI_IDENTIFIER_FIELDS

    @property
    def direct_identifier_fields(self) -> tuple[str, ...]:
        """Direct-identifier columns present in full rows."""

        if self.config.include_direct_identifiers:
            return RARE_CASE_DIRECT_IDENTIFIER_FIELDS
        return ()

    def rows(
        self, *, include_direct_identifiers: bool = True
    ) -> tuple[dict[str, Any], ...]:
        """Return every record as a flat table row.

        Args:
            include_direct_identifiers: Whether to keep name and record-number
                columns. Drop them to model a release where direct identifiers
                were removed but quasi-identifiers remain.

        Returns:
            Rows in record order.
        """

        return tuple(
            record.to_row(include_direct_identifiers=include_direct_identifiers)
            for record in self.records
        )

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        """Return every record as a synthetic training record."""

        return tuple(record.to_dict() for record in self.records)

    def dataset_hash(self) -> str:
        """Return a ``sha256:`` digest of the canonical record payloads."""

        payload = json.dumps(
            [record.to_dict() for record in self.records],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"

    def summary(self) -> dict[str, Any]:
        """Return a value-free manifest of counts, tiers, flags, and digests."""

        stress = [record for record in self.records if record.case_kind == "stress"]
        tier_counts = {
            dimension: {tier: 0 for tier in PREVALENCE_TIERS}
            for dimension in RARE_CASE_DIMENSIONS
        }
        flag_counts: Counter[str] = Counter()
        for record in self.records:
            for attribute in record.attributes:
                flag_counts.update(attribute.safe_harbor_flags)
                if attribute.tier is not None:
                    tier_counts[attribute.dimension][attribute.tier] += 1
        return {
            "schema_version": RARE_CASE_STRESS_SCHEMA_VERSION,
            "source": self.source,
            "synthetic": self.synthetic,
            "restricted_data": False,
            "contains_real_phi": False,
            "seed": self.seed,
            "config": self.config.to_dict(),
            "record_count": len(self.records),
            "stress_record_count": len(stress),
            "background_record_count": len(self.records) - len(stress),
            "dataset_singleton_count": sum(
                record.is_dataset_singleton for record in self.records
            ),
            "stress_singleton_count": sum(
                record.is_dataset_singleton for record in stress
            ),
            "min_equivalence_class_size": min(
                (record.equivalence_class_size for record in self.records),
                default=0,
            ),
            "profile_counts": dict(
                sorted(Counter(record.profile_id for record in stress).items())
            ),
            "tier_counts": tier_counts,
            "direct_identifier_count": sum(
                len(record.direct_identifiers) for record in self.records
            ),
            "safe_harbor_flag_counts": dict(sorted(flag_counts.items())),
            "max_combined_surprisal_bits": max(
                (record.combined_surprisal_bits for record in self.records),
                default=0.0,
            ),
            "quasi_identifier_fields": list(self.quasi_identifier_fields),
            "direct_identifier_fields": list(self.direct_identifier_fields),
            "dataset_hash": self.dataset_hash(),
        }


def rare_case_vocabulary() -> dict[str, list[dict[str, Any]]]:
    """Return the value-free synthetic priors behind every vocabulary code.

    Returns:
        For each dimension, its codes with tier and synthetic frequency. Surface
        values are not included.
    """

    return {
        dimension: [
            {"code": code, "tier": tier, "frequency": _significant(entry.frequency)}
            for tier in PREVALENCE_TIERS
            for code, entry in _VOCABULARY[dimension][tier]
        ]
        for dimension in RARE_CASE_DIMENSIONS
    }


def generate_rare_case_stress_set(
    count: int,
    *,
    seed: int = 0,
    config: RareCaseStressConfig | None = None,
) -> RareCaseStressSet:
    """Generate configurable rare-combination records with risk truth labels.

    Stress records cycle through ``config.profiles`` in order, so every
    configured combination is covered once ``count`` reaches the profile count.
    Background classes, when requested, add all-common records that share one
    quasi-identifier tuple per class. Records are shuffled with the same seed
    before identifiers are assigned, and equivalence-class sizes are computed
    over the final set. The same seed and configuration always yield identical
    records; no file, network, or global random state is touched.

    Args:
        count: Number of stress records to generate.
        seed: Deterministic seed for a local ``random.Random``.
        config: Optional generation settings.

    Returns:
        The generated stress set.

    Raises:
        TypeError: If ``config`` is not a ``RareCaseStressConfig``.
        ValueError: If ``count`` or ``seed`` is invalid, or a leap-day birth
            date cannot fit the drawn age band at the reference date.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if config is None:
        config = RareCaseStressConfig()
    elif not isinstance(config, RareCaseStressConfig):
        raise TypeError("config must be a RareCaseStressConfig")

    rng = random.Random(seed)
    used_record_numbers: set[str] = set()
    drafts: list[_Draft] = []
    background = RareCaseProfile()
    for _ in range(config.background_classes):
        sampled = _sample_stress_values(rng, background, config.reference_date)
        for _ in range(config.background_class_size):
            drafts.append(
                _Draft(
                    "background",
                    background.profile_id,
                    sampled,
                    _sample_direct_identifiers(rng, used_record_numbers, config),
                )
            )
    for index in range(count):
        profile = config.profiles[index % len(config.profiles)]
        drafts.append(
            _Draft(
                "stress",
                profile.profile_id,
                _sample_stress_values(rng, profile, config.reference_date),
                _sample_direct_identifiers(rng, used_record_numbers, config),
            )
        )
    rng.shuffle(drafts)

    class_sizes = Counter(draft.key for draft in drafts)
    records = tuple(
        _build_record(draft, f"synthetic-rare-case-{seed}-{index:05d}", class_sizes)
        for index, draft in enumerate(drafts)
    )
    return RareCaseStressSet(records=records, seed=seed, config=config)


@dataclass(frozen=True, slots=True)
class _Sampled:
    dimension: str
    code: str
    entry: _Entry
    value: str | int = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Draft:
    case_kind: str
    profile_id: str
    sampled: tuple[_Sampled, ...]
    direct: tuple[str, str] | None = field(repr=False)

    @property
    def key(self) -> tuple[str | int, ...]:
        return tuple(item.value for item in self.sampled)


def _validated_dimensions(values: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field_name} must be a sequence of dimension names")
    requested = set()
    for value in values:
        if not isinstance(value, str) or value not in RARE_CASE_DIMENSIONS:
            raise ValueError(
                f"{field_name} contains an unknown rare-case dimension; expected "
                "one of " + ", ".join(RARE_CASE_DIMENSIONS)
            )
        requested.add(value)
    return tuple(name for name in RARE_CASE_DIMENSIONS if name in requested)


def _sample_stress_values(
    rng: random.Random,
    profile: RareCaseProfile,
    reference_date: date,
) -> tuple[_Sampled, ...]:
    chosen: dict[str, tuple[str, _Entry]] = {}
    for dimension in RARE_CASE_DIMENSIONS:
        options = _VOCABULARY[dimension][profile.tier_for(dimension)]
        chosen[dimension] = options[rng.randrange(len(options))]

    date_entry = chosen["date"][1]
    low, high = chosen["age"][1].age_range
    ages = [
        age
        for age in range(low, high + 1)
        if not date_entry.leap_day
        or calendar.isleap(_birth_year(age, 2, 29, reference_date))
    ]
    if not ages:
        raise ValueError(
            "leap-day date stressor cannot fit the age band at the reference date"
        )
    age = ages[rng.randrange(len(ages))]
    if date_entry.leap_day:
        month, day = 2, 29
    else:
        month, day = rng.randint(1, 12), rng.randint(1, 28)
    year = _birth_year(age, month, day, reference_date)
    if date_entry.precision == "year":
        birth_date = f"{year:04d}"
    elif date_entry.precision == "month":
        birth_date = f"{year:04d}-{month:02d}"
    else:
        birth_date = f"{year:04d}-{month:02d}-{day:02d}"

    values: dict[str, str | int] = {"age": age, "date": birth_date}
    sampled = []
    for dimension in RARE_CASE_DIMENSIONS:
        code, entry = chosen[dimension]
        sampled.append(
            _Sampled(dimension, code, entry, values.get(dimension, entry.surface))
        )
    return tuple(sampled)


def _sample_direct_identifiers(
    rng: random.Random,
    used_record_numbers: set[str],
    config: RareCaseStressConfig,
) -> tuple[str, str] | None:
    if not config.include_direct_identifiers:
        return None
    name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    while True:
        record_number = f"SYN-{rng.randrange(10**8):08d}"
        if record_number not in used_record_numbers:
            used_record_numbers.add(record_number)
            return name, record_number


def _birth_year(age: int, month: int, day: int, reference_date: date) -> int:
    birthday_pending = (month, day) > (reference_date.month, reference_date.day)
    return reference_date.year - age - (1 if birthday_pending else 0)


def _build_record(
    draft: _Draft,
    record_id: str,
    class_sizes: Counter[tuple[str | int, ...]],
) -> RareCaseStressRecord:
    text = _NOTE_HEADER
    attributes: list[RareCaseAttribute] = []

    if draft.direct is not None:
        for (field_name, label, code, flag, prefix), value in zip(
            _DIRECT_IDENTIFIER_SPECS, draft.direct
        ):
            text += prefix
            start = len(text)
            text += value
            attributes.append(
                RareCaseAttribute(
                    dimension=DIRECT_IDENTIFIER_DIMENSION,
                    field=field_name,
                    label=label,
                    identifier_class=DIRECT_IDENTIFIER,
                    reid_category=DIRECT_IDENTIFIER_DIMENSION,
                    tier=None,
                    code=f"{DIRECT_IDENTIFIER_DIMENSION}.{code}",
                    value=value,
                    span=(start, len(text)),
                    safe_harbor_flags=(flag,),
                )
            )
            text += ".\n"

    surprisal = {
        item.dimension: (
            0.0 if item.entry.frequency >= 1.0 else -math.log2(item.entry.frequency)
        )
        for item in draft.sampled
    }
    total_bits = sum(surprisal.values())
    joint_frequency = math.prod(item.entry.frequency for item in draft.sampled)
    age = int(next(item.value for item in draft.sampled if item.dimension == "age"))
    for item in draft.sampled:
        spec = _DIMENSION_SPECS[item.dimension]
        label = spec.label
        if item.dimension == "date" and item.entry.precision == "day":
            label = DATE_OF_BIRTH
        # Stress dimensions are linkage keys, so a clinical concept such as a
        # diagnosis is scored as a quasi-identifier; a full birth date keeps
        # its core DIRECT_IDENTIFIER policy class.
        identifier_class = (
            DIRECT_IDENTIFIER
            if label is not None and policy_label_for(label) == DIRECT_IDENTIFIER
            else QUASI_IDENTIFIER
        )
        text += spec.prefix
        start = len(text)
        text += str(item.value)
        attributes.append(
            RareCaseAttribute(
                dimension=item.dimension,
                field=spec.field,
                label=label,
                identifier_class=identifier_class,
                reid_category=spec.reid_category,
                tier=item.entry.tier,
                code=item.code,
                value=item.value,
                span=(start, len(text)),
                frequency=_significant(item.entry.frequency),
                surprisal_bits=round(surprisal[item.dimension], 6),
                contribution_share=round(
                    surprisal[item.dimension] / total_bits if total_bits else 0.0, 6
                ),
                risk_level=_TIER_RISK_LEVELS[item.entry.tier],
                safe_harbor_flags=_safe_harbor_flags(item, age),
                sensitive_attribute=spec.sensitive,
                relative_role=item.entry.relative_role,
            )
        )
        text += spec.suffix + "\n"

    expected_matches = SYNTHETIC_REFERENCE_POPULATION * joint_frequency
    unique_probability = math.exp(
        -(SYNTHETIC_REFERENCE_POPULATION - 1) * joint_frequency
    )
    dominant = max(draft.sampled, key=lambda item: surprisal[item.dimension])
    return RareCaseStressRecord(
        record_id=record_id,
        case_kind=draft.case_kind,
        profile_id=draft.profile_id,
        text=text,
        attributes=tuple(attributes),
        equivalence_class_size=class_sizes[draft.key],
        combined_surprisal_bits=round(total_bits, 6),
        expected_population_matches=_significant(expected_matches),
        population_unique_probability=_significant(unique_probability),
        dominant_dimension=dominant.dimension,
    )


def _safe_harbor_flags(item: _Sampled, age: int) -> tuple[str, ...]:
    flags: list[str] = []
    entry = item.entry
    if item.dimension == "geography" and (
        entry.population <= SAFE_HARBOR_MAX_SMALL_AREA_POPULATION
    ):
        flags.append("small_geographic_unit")
    if item.dimension == "age" and age > SAFE_HARBOR_MAX_AGE:
        flags.append("age_over_89")
    if item.dimension == "date" and entry.precision in {"month", "day"}:
        flags.append("date_more_specific_than_year")
    if item.dimension == "date" and age > SAFE_HARBOR_MAX_AGE:
        flags.append("date_indicates_age_over_89")
    return tuple(flags)


def _significant(value: float) -> float:
    return float(f"{value:.6g}")


__all__ = [
    "DEFAULT_REFERENCE_DATE",
    "DIRECT_IDENTIFIER_DIMENSION",
    "PREVALENCE_TIERS",
    "RARE_CASE_DIMENSIONS",
    "RARE_CASE_DIRECT_IDENTIFIER_FIELDS",
    "RARE_CASE_QUASI_IDENTIFIER_FIELDS",
    "RARE_CASE_STRESS_SCHEMA_VERSION",
    "SAFE_HARBOR_FLAGS",
    "SAFE_HARBOR_MAX_AGE",
    "SAFE_HARBOR_MAX_SMALL_AREA_POPULATION",
    "SYNTHETIC_RARE_CASE_SOURCE",
    "SYNTHETIC_REFERENCE_POPULATION",
    "RareCaseAttribute",
    "RareCaseProfile",
    "RareCaseStressConfig",
    "RareCaseStressRecord",
    "RareCaseStressSet",
    "default_rare_case_profiles",
    "generate_rare_case_stress_set",
    "rare_case_vocabulary",
]
