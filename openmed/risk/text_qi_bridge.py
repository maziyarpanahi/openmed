"""Project locally bound text spans into generalized in-memory tabular QIs."""

from __future__ import annotations

import hmac
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType

from openmed.core.labels import CANONICAL_LABELS, normalize_label
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.processing.outputs import EntityPrediction

_COLUMNS = {
    "AGE": "age_band",
    "DATE": "year",
    "DATE_OF_BIRTH": "year",
    "ZIPCODE": "zip3",
    "LOCATION": "region",
    "OCCUPATION": "occupation_group",
}
_COLS = tuple(dict.fromkeys(_COLUMNS.values()))


@dataclass(frozen=True)
class TextQIPolicy:
    """Age-band width and local surface-to-group lookups for region and occupation."""

    age_band_width: int
    regions: Mapping[str, str] = field(default_factory=dict, repr=False)
    occupation_groups: Mapping[str, str] = field(default_factory=dict, repr=False)

    concept_frequencies: Mapping[tuple[str, str], float] | None = field(
        default=None, repr=False
    )
    rare_frequency_threshold: float | None = None

    def __post_init__(self):
        if type(self.age_band_width) is not int or not 2 <= self.age_band_width <= 100:
            raise ValueError("invalid age band width")
        for name in ("regions", "occupation_groups"):
            mapping = {}
            for surface, code in getattr(self, name).items():
                if (
                    not isinstance(surface, str)
                    or not isinstance(code, str)
                    or not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", code)
                    or not surface.strip()
                    or surface.casefold().strip() == code.casefold()
                ):
                    raise ValueError("invalid generalization mapping")
                key = surface.strip().casefold()
                if key in mapping and mapping[key] != code:
                    raise ValueError("ambiguous generalization mapping")
                mapping[key] = code
            object.__setattr__(self, name, MappingProxyType(mapping))

        table = self.concept_frequencies
        threshold = self.rare_frequency_threshold
        if table is None and threshold is None:
            return
        if (
            not isinstance(table, Mapping)
            or type(threshold) not in (int, float)
            or not 0 < threshold <= 1
        ):
            raise ValueError(
                "frequency table requires an explicit proportion threshold"
            )
        frequencies = {}
        for key, frequency in table.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError("invalid clinical concept key")
            label, surface = key
            if (
                not isinstance(label, str)
                or label not in CANONICAL_LABELS
                or not isinstance(surface, str)
                or not surface.strip()
            ):
                raise ValueError("invalid clinical concept key")
            normalized = (label, surface.strip().casefold())
            if (
                type(frequency) not in (int, float)
                or not 0 <= frequency <= 1
                or normalized in frequencies
            ):
                raise ValueError("invalid or duplicate concept frequency")
            frequencies[normalized] = frequency
        object.__setattr__(self, "concept_frequencies", MappingProxyType(frequencies))


@dataclass(frozen=True)
class TextQIDocument:
    """Local source text, its detections and an optional keyed unit fingerprint."""

    text: str = field(repr=False)
    spans: Sequence[OpenMedSpan | EntityPrediction] = field(repr=False)
    unit_digest: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class TextQIResult:
    """Generalized rows for local scorers and a separately shareable count profile."""

    rows: tuple[dict[str, str | None], ...] = field(repr=False)
    profile: dict


def _generalize(column: str, surface: str, policy: TextQIPolicy) -> str | None:
    value = surface.strip()
    if column == "age_band":
        match = re.fullmatch(
            r"(\d{1,3})(?:\s*[- ]?\s*(?:years?(?:[- ]old)?|y/?o))?",
            value,
            re.IGNORECASE,
        )
        if match and int(match[1]) <= 125:
            low = int(match[1]) // policy.age_band_width * policy.age_band_width
            return f"{low}-{low + policy.age_band_width - 1}"
    elif column == "year":
        for fmt in (
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%B %d, %Y",
            "%d %B %Y",
            "%b %d, %Y",
        ):
            try:
                return str(datetime.strptime(value, fmt).year)
            except ValueError:
                continue
    elif column == "zip3":
        if re.fullmatch(r"[0-9]{5}(?:-[0-9]{4})?", value):
            return value[:3]
    else:
        lookup = policy.regions if column == "region" else policy.occupation_groups
        return lookup.get(value.casefold())
    return None


def bridge_text_qis(
    documents: Iterable[TextQIDocument],
    policy: TextQIPolicy,
    *,
    by_unit: bool = False,
    span_hmac_key: str | bytes | None = None,
) -> TextQIResult:
    """Validate all half-open spans and project generalized values before grouping.

    Each input document normally contributes one row. With by_unit, documents
    sharing an HMAC unit digest form one row. Repeated generalized values are
    deduplicated and distinct values joined in sorted order within a cell.
    Supply the original HMAC key for OpenMedSpan inputs. Runtime entities bind
    offsets via their detected surface. Unknown generalizations are counted.
    """
    frequencies = policy.concept_frequencies
    columns = _COLS if frequencies is None else (*_COLS, "rare_attribute")
    concept_labels = (
        {label for label, _ in frequencies} if frequencies is not None else set()
    )
    groups = {}
    counts = Counter()
    unmapped = Counter()
    for index, doc in enumerate(documents):
        if by_unit and (
            not isinstance(doc.unit_digest, str)
            or not re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", doc.unit_digest)
        ):
            raise ValueError("missing keyed privacy unit")
        group = groups.setdefault(
            doc.unit_digest if by_unit else index, defaultdict(set)
        )
        verified = []
        for span in doc.spans:
            start, end = span.start, span.end
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(doc.text)
            ):
                raise ValueError("invalid text span bounds")
            surface = doc.text[start:end]
            if isinstance(span, OpenMedSpan):
                if not span_hmac_key or not hmac.compare_digest(
                    hmac_text_hash(surface, span_hmac_key), span.text_hash
                ):
                    raise ValueError("span does not match document")
                label = span.canonical_label
            else:
                if span.text != surface:
                    raise ValueError("span does not match document")
                label = getattr(span, "canonical_label", None) or normalize_label(
                    span.label
                )
            verified.append((start, end, label))
        previous_end = 0
        for start, end, label in sorted(verified):
            if start < previous_end:
                raise ValueError("overlapping text spans")
            previous_end = end
            if label in concept_labels:
                counts["rare_attribute"] += 1
                key = (label, doc.text[start:end].strip().casefold())
                if key not in frequencies:
                    unmapped["rare_attribute"] += 1
                elif frequencies[key] < policy.rare_frequency_threshold:
                    group["rare_attribute"].add("1")
            column = _COLUMNS.get(label)
            if column:
                counts[column] += 1
                value = _generalize(column, doc.text[start:end], policy)
                if value is None:
                    unmapped[column] += 1
                else:
                    group[column].add(value)
    if frequencies is not None:
        for group in groups.values():
            if not group["rare_attribute"]:
                group["rare_attribute"].add("0")
    rows = tuple(
        {column: "|".join(sorted(group[column])) or None for column in columns}
        for group in groups.values()
    )
    return TextQIResult(
        rows,
        {
            "row_count": len(rows),
            "columns": {
                col: {"span_count": counts[col], "unmapped_count": unmapped[col]}
                for col in columns
            },
        },
    )


def text_qi_risk(
    spans_by_unit: Iterable[TextQIDocument],
    policy: TextQIPolicy,
    reference_population: Iterable[TextQIDocument] | None = None,
    target_k: int | None = None,
    target_k_map: int | None = None,
    *,
    span_hmac_key: str | bytes | None = None,
) -> dict:
    """Measure k and optional exact k-map for generalized text QIs.

    Args:
        spans_by_unit: Documents with keyed privacy-unit digests.
        policy: Generalization rules shared by sample and reference.
        reference_population: Reference documents with the same unit semantics.
        target_k: Explicit minimum sample class size.
        target_k_map: Explicit minimum reference class size when supplied.
        span_hmac_key: Local key for verifying OpenMedSpan source binding.
    """
    from openmed.risk.kanon import kanon_report
    from openmed.risk.population import _analysis_unit_profiles

    if type(target_k) is not int or target_k < 1:
        raise ValueError("explicit positive target_k required")
    if reference_population is None and target_k_map is not None:
        raise ValueError("reference_population required for k-map")
    if reference_population is not None and (
        type(target_k_map) is not int or target_k_map < 1
    ):
        raise ValueError("explicit positive target_k_map required")
    sample = bridge_text_qis(
        spans_by_unit, policy, by_unit=True, span_hmac_key=span_hmac_key
    )
    if not sample.rows:
        raise ValueError("empty sample")
    columns = tuple(sample.profile["columns"])
    kanon = kanon_report(sample.rows, quasi_identifiers=columns)
    population = None
    if reference_population is not None:
        reference = bridge_text_qis(
            reference_population, policy, by_unit=True, span_hmac_key=span_hmac_key
        )
        if not reference.rows:
            raise ValueError("empty reference population")

        # Each bridge row already represents one distinct keyed unit. Use the
        # population scorer's typed joint profiles without exposing their keys.
        def frequencies(rows):
            return Counter(
                _analysis_unit_profiles(rows, columns, privacy_unit=None).values()
            )

        sample_counts, reference_counts = (
            frequencies(sample.rows),
            frequencies(reference.rows),
        )
        k_map = min(reference_counts[key] for key in sample_counts)
        consistent = all(
            reference_counts[key] >= count for key, count in sample_counts.items()
        )
        population = {
            "reference_unit_count": len(reference.rows),
            "achieved_k_map": k_map,
            "target_k_map": target_k_map,
            "reference_model_consistent": consistent,
            "meets_k_map": consistent and k_map >= target_k_map,
        }
    return {
        "schema_version": 1,
        "artifact": "text_qi_risk",
        "profile": sample.profile,
        "kanon": {
            "k": kanon["k"],
            "target_k": target_k,
            "class_count": kanon["class_count"],
            "meets_k": kanon["k"] >= target_k,
        },
        "population": population,
        "limitations": [
            "detected_spans_only",
            "caller_defined_generalization",
            "qualified_expert_review_required",
        ],
    }
