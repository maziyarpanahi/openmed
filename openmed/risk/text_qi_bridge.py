"""Project locally bound text spans into generalized in-memory tabular QIs."""

from __future__ import annotations

import hmac
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType

from openmed.core.labels import normalize_label
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
            column = _COLUMNS.get(label)
            if column:
                counts[column] += 1
                value = _generalize(column, doc.text[start:end], policy)
                if value is None:
                    unmapped[column] += 1
                else:
                    group[column].add(value)
    rows = tuple(
        {column: "|".join(sorted(group[column])) or None for column in _COLS}
        for group in groups.values()
    )
    return TextQIResult(
        rows,
        {
            "row_count": len(rows),
            "columns": {
                col: {"span_count": counts[col], "unmapped_count": unmapped[col]}
                for col in _COLS
            },
        },
    )
