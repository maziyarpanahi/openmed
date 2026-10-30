"""Live surrogate-quality scoring for every registered language pack.

:mod:`openmed.eval.surrogate_quality` grades committed fixture records. This
module grades what the anonymizer generates *now*. For every pack in the
language-pack registry and every national-ID-only language it builds
deterministic synthetic originals in the expected script, asks the live
provider stack for seeded surrogates for six labels (``name``, ``address``,
``city``, ``phone``, ``id`` and ``date``), and scores them on the existing
surrogate-quality dimensions:

* ``script``: letters fall in the scripts the pack declares.
* ``format``: dates follow the locale date order, addresses and cities do not
  reuse the US street or city template, and phone numbers do not take a North
  American (NANP) shape outside NANP regions.
* ``checksum``: national-ID surrogates pass the language's registered
  validators.
* ``consistency``: a fresh provider with the same seed reproduces the value,
  and the value neither equals the original nor retains an original token.
* ``naturalness``: no fallback placeholders, digits in names or cities,
  inserted honorifics, or implausible name shapes.

The result is a per-locale, per-label matrix. A cell that cannot be scored is
reported as ``unscored`` with a reason code instead of being skipped, and a
cell can be exempted only for a locale with a documented approximation and a
recorded reason. Reports carry codes, counts, rates and SHA-256 digests; they
never contain an original or a surrogate value. Generation is seeded and runs
offline.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import unicodedata
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Final, Protocol

from openmed.core.audit import stable_hash
from openmed.core.language_pack import (
    LANGUAGE_PACK_REGISTRY,
    LanguagePack,
    LanguagePackRegistry,
)
from openmed.core.language_pack_catalog import (
    NATIONAL_ID_ONLY_CAPABILITIES,
    NationalIdOnlyCapability,
)
from openmed.core.locale_formats import date_order_for_locale, parse_date
from openmed.core.script_detect import (
    CJK_SCRIPTS,
    INDIC_SCRIPTS,
    SUPPORTED_SCRIPTS,
    detect_script,
)
from openmed.eval.surrogate_quality import (
    _PLACEHOLDER_RE,
    DEFAULT_SURROGATE_QUALITY_PASS_RATE,
    SURROGATE_QUALITY_DIMENSIONS,
)

LIVE_SURROGATE_LABELS: Final = ("name", "address", "city", "phone", "id", "date")
"""Live labels scored for every locale, in matrix column order."""

LIVE_SURROGATE_SOURCE_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "name": "PERSON",
        "address": "STREET_ADDRESS",
        "city": "CITY",
        "phone": "PHONE",
        "id": "national_id",
        "date": "DATE_OF_BIRTH",
    }
)
"""Source label passed to the provider stack for each live label."""

LIVE_SURROGATE_LABEL_DIMENSIONS: Final[Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            "name": ("script", "consistency", "naturalness"),
            "address": ("script", "format", "consistency", "naturalness"),
            "city": ("script", "format", "consistency", "naturalness"),
            "phone": ("format", "consistency"),
            "id": ("checksum", "consistency"),
            "date": ("format", "consistency"),
        }
    )
)
"""Quality dimensions that apply to each live label."""

DEFAULT_LIVE_SURROGATE_SAMPLES: Final = 10
DEFAULT_LIVE_SURROGATE_SEED: Final = 0
DEFAULT_LIVE_SURROGATE_PASS_RATE: Final = DEFAULT_SURROGATE_QUALITY_PASS_RATE
DEFAULT_LIVE_SCRIPT_RATIO_FLOOR: Final = 0.80
LIVE_SCRIPT_RATIO_FLOORS: Final[Mapping[str, float]] = MappingProxyType(
    # Mirrors the fixture scorer, which accepts a 0.70 in-script letter ratio
    # for Devanagari and Han and 0.80 for Latin.
    {"Devanagari": 0.70, "Han": 0.70}
)

CELL_PASS: Final = "pass"
CELL_FAIL: Final = "fail"
CELL_UNSCORED: Final = "unscored"
CELL_EXEMPT: Final = "exempt"
CELL_STATUSES: Final = (CELL_PASS, CELL_FAIL, CELL_UNSCORED, CELL_EXEMPT)

TARGET_LANGUAGE_PACK: Final = "language_pack"
TARGET_NATIONAL_ID_ONLY: Final = "national_id_only"

SCRIPT_SOURCE_DECLARED: Final = "declared"
SCRIPT_SOURCE_FALLBACK: Final = "fallback"
SCRIPT_SOURCE_NONE: Final = "none"

LIVE_SURROGATE_SCRIPT_FALLBACKS: Final[Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            # Built-in packs routed without a script claim.
            "id": ("Latin",),
            "ro": ("Latin",),
            # National-ID-only languages declare no scripts.
            "af": ("Latin",),
            "bg": ("Cyrillic",),
            "et": ("Latin",),
            "fi": ("Latin",),
            "ha": ("Latin",),
            "hr": ("Latin",),
            "hu": ("Latin",),
            "ig": ("Latin",),
            "lv": ("Latin",),
            "ms": ("Latin",),
            "rw": ("Latin",),
            "sk": ("Latin",),
            "sr": ("Cyrillic", "Latin"),
            "tl": ("Latin",),
            "yo": ("Latin",),
        }
    )
)
"""Script expectations for languages whose declaration names no routable script.

A pack's declared scripts always win. This table only covers packs routed
without a script claim and national-ID-only languages. A language in neither
place is reported as ``unscored`` for the script-bearing labels.
"""

# Regions where the US street/city template and NANP phone shapes are native.
# Philippine street addresses follow the same number-name-suffix convention.
_US_ADDRESS_TEMPLATE_REGIONS: Final = frozenset({"US", "CA", "PH"})
_NANP_REGIONS: Final = frozenset({"US", "CA"})

_ALPHABET_SIZE: Final = 48
_NAME_TOKEN_LENGTHS: Final = (4, 5)
_UNSPACED_NAME_LENGTH: Final = 3
_LABEL_OFFSETS: Final = MappingProxyType(
    {"name": 0, "address": 11, "city": 23, "phone": 31, "id": 37, "date": 41}
)
_DIGEST_LENGTH: Final = 16

_FALLBACK_PLACEHOLDER_RE: Final = re.compile(r"^\s*\[[^\]]*\]")
_US_SECONDARY_UNIT_RE: Final = re.compile(r"(?<!\w)(?:Apt\.|Suite)\s+\d+")
_US_BUILDING_NUMBER_RE: Final = re.compile(r"^\d{3,5} ")
_US_CITY_SUFFIXES: Final = (
    # English-specific subset of Faker's en_US city suffixes. Suffixes that are
    # also native elsewhere (berg, furt, land, stad, ton) are left out.
    "borough",
    "burgh",
    "bury",
    "chester",
    "fort",
    "haven",
    "mouth",
    "port",
    "shire",
    "side",
    "town",
    "view",
    "ville",
)
_US_CITY_SUFFIX_RE: Final = re.compile(r"^\S{2,}?(?:%s)$" % "|".join(_US_CITY_SUFFIXES))
_US_PHONE_SHAPES: Final = (
    re.compile(
        r"^\s*(?:\+1|001)[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}"
        r"(?:\s*x\d{1,6})?\s*$"
    ),
    re.compile(r"^\s*\(\d{3}\)\s?\d{3}-\d{4}(?:\s*x\d{1,6})?\s*$"),
    re.compile(r"^\s*[2-9]\d{2}([.-])\d{3}\1\d{4}(?:\s*x\d{1,6})?\s*$"),
    re.compile(r"\d\s*x\d{1,6}\s*$"),
)
_FALLBACK_US_STREET_SUFFIXES: Final = frozenset(
    {
        "Avenue",
        "Boulevard",
        "Court",
        "Crossing",
        "Drive",
        "Fields",
        "Glens",
        "Knolls",
        "Lake",
        "Lakes",
        "Lane",
        "Road",
        "Street",
        "Suite",
    }
)
_FALLBACK_US_CITY_PREFIXES: Final = ("North", "East", "West", "South", "New", "Lake")


class LiveSurrogateSource(Protocol):
    """Provider object the live run asks for surrogates.

    :class:`openmed.core.anonymizer.Anonymizer` satisfies this protocol. Tests
    inject lightweight sources to model a provider without touching the
    process-wide generator registry.
    """

    def surrogate(
        self,
        original: str,
        label: str,
        *,
        locale: str | None = None,
    ) -> str:
        """Return a surrogate for ``original`` of source label ``label``."""


LiveSurrogateFactory = Callable[[str, int], LiveSurrogateSource]
"""Callable building a seeded :class:`LiveSurrogateSource` for a language."""


def default_live_surrogate_factory(language: str, seed: int) -> LiveSurrogateSource:
    """Build the production anonymizer used by the live run.

    Args:
        language: OpenMed language code.
        seed: Deterministic seed shared by every surrogate in the run.

    Returns:
        A deterministic :class:`~openmed.core.anonymizer.Anonymizer`.
    """

    from openmed.core.anonymizer import Anonymizer

    return Anonymizer(lang=language, consistent=True, seed=seed)


@dataclass(frozen=True)
class LiveSurrogateTarget:
    """One language scored by the live surrogate-quality run.

    Args:
        language: OpenMed language code.
        locale: Declared surrogate locale for the language.
        source: ``language_pack`` or ``national_id_only``.
        expected_scripts: Scripts surrogate letters must use. Empty when no
            script expectation can be derived.
        script_source: ``declared``, ``fallback`` or ``none``.
        national_id_locale: Locale of the registered national-ID provider.
        national_id_method: Provider method of the registered national-ID
            provider.
        approximation: Documented surrogate-locale approximation, if any.
    """

    language: str
    locale: str
    source: str
    expected_scripts: tuple[str, ...]
    script_source: str
    national_id_locale: str | None = None
    national_id_method: str | None = None
    approximation: str | None = None

    @property
    def region(self) -> str | None:
        """Return the upper-case region subtag of :attr:`locale`, if any."""

        parts = self.locale.replace("-", "_").split("_")
        if len(parts) < 2 or not parts[-1]:
            return None
        return parts[-1].upper()

    @property
    def primary_script(self) -> str | None:
        """Return the script synthetic originals are written in."""

        return self.expected_scripts[0] if self.expected_scripts else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "language": self.language,
            "locale": self.locale,
            "source": self.source,
            "expected_scripts": list(self.expected_scripts),
            "script_source": self.script_source,
            "national_id_locale": self.national_id_locale,
            "national_id_method": self.national_id_method,
            "approximation": self.approximation,
        }


@dataclass(frozen=True)
class LiveSurrogateExemption:
    """Recorded exemption for live cells of one documented approximation.

    Args:
        language: Language code whose cells are exempted.
        labels: Live labels covered by the exemption.
        reason: Non-empty human-readable reason recorded in the report.

    Raises:
        ValueError: If the language, labels or reason are empty or unknown.
    """

    language: str
    labels: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        """Validate and normalize the exemption declaration."""

        language = str(self.language or "").strip().lower()
        if not language:
            raise ValueError("live surrogate exemption requires a language")
        if isinstance(self.labels, str):
            raise TypeError("live surrogate exemption labels must be a sequence")
        labels = tuple(str(label).strip().lower() for label in self.labels)
        if not labels:
            raise ValueError("live surrogate exemption requires at least one label")
        unknown = sorted(set(labels) - set(LIVE_SURROGATE_LABELS))
        if unknown:
            raise ValueError(f"unknown live surrogate labels: {unknown}")
        if len(set(labels)) != len(labels):
            raise ValueError("live surrogate exemption labels must be unique")
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError(
                f"live surrogate exemption for {language!r} requires a recorded reason"
            )
        object.__setattr__(self, "language", language)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "reason", reason)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "LiveSurrogateExemption":
        """Build an exemption from a JSON-compatible mapping."""

        if not isinstance(payload, Mapping):
            raise TypeError("live surrogate exemption must be a mapping")
        labels = payload.get("labels")
        if labels is None and payload.get("label") is not None:
            labels = (payload["label"],)
        return cls(
            language=str(payload.get("language") or ""),
            labels=tuple(labels or ()),
            reason=str(payload.get("reason") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "language": self.language,
            "labels": list(self.labels),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class LiveSurrogateCell:
    """Aggregated live result for one ``(language, label)`` matrix cell."""

    language: str
    label: str
    status: str
    scored_status: str
    min_pass_rate: float
    samples: int = 0
    passed_samples: int = 0
    dimension_rates: Mapping[str, float] = field(default_factory=dict)
    reason_counts: Mapping[str, int] = field(default_factory=dict)
    value_digests: tuple[str, ...] = ()
    unscored_reason: str | None = None
    exemption: Mapping[str, str] | None = None

    @property
    def key(self) -> str:
        """Return the stable ``language/label`` cell key."""

        return f"{self.language}/{self.label}"

    @property
    def pass_rate(self) -> float:
        """Return the fully passing sample rate."""

        if not self.samples:
            return 0.0
        return self.passed_samples / self.samples

    @property
    def distinct_values(self) -> int:
        """Return the number of distinct surrogate digests."""

        return len(set(self.value_digests))

    @property
    def value_digest(self) -> str | None:
        """Return one SHA-256 digest over the ordered sample digests."""

        if not self.value_digests:
            return None
        material = "\n".join(self.value_digests).encode("utf-8")
        return "sha256:" + hashlib.sha256(material).hexdigest()

    def with_min_pass_rate(self, min_pass_rate: float) -> "LiveSurrogateCell":
        """Return the cell re-graded against ``min_pass_rate``."""

        if self.scored_status == CELL_UNSCORED:
            return replace(self, min_pass_rate=min_pass_rate)
        scored = CELL_PASS if self.pass_rate >= min_pass_rate else CELL_FAIL
        status = CELL_EXEMPT if self.exemption and scored != CELL_PASS else scored
        return replace(
            self,
            status=status,
            scored_status=scored,
            min_pass_rate=min_pass_rate,
        )

    def summary(self) -> dict[str, Any]:
        """Return a compact JSON-friendly summary for gate details."""

        payload: dict[str, Any] = {
            "status": self.status,
            "pass_rate": self.pass_rate,
            "samples": self.samples,
            "reason_counts": dict(sorted(self.reason_counts.items())),
        }
        if self.unscored_reason is not None:
            payload["unscored_reason"] = self.unscored_reason
        if self.exemption is not None:
            payload["exemption"] = dict(self.exemption)
        return payload

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation without surrogate values."""

        return {
            "language": self.language,
            "label": self.label,
            "status": self.status,
            "scored_status": self.scored_status,
            "min_pass_rate": self.min_pass_rate,
            "samples": self.samples,
            "passed_samples": self.passed_samples,
            "pass_rate": self.pass_rate,
            "dimension_rates": dict(sorted(self.dimension_rates.items())),
            "reason_counts": dict(sorted(self.reason_counts.items())),
            "distinct_values": self.distinct_values,
            "value_digest": self.value_digest,
            "value_digests": list(self.value_digests),
            "unscored_reason": self.unscored_reason,
            "exemption": dict(self.exemption) if self.exemption else None,
        }


@dataclass(frozen=True)
class LiveSurrogateLocaleReport:
    """Live cells for one target language."""

    target: LiveSurrogateTarget
    cells: Mapping[str, LiveSurrogateCell]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "target": self.target.to_dict(),
            "cells": {label: cell.to_dict() for label, cell in self.cells.items()},
        }


@dataclass(frozen=True)
class LiveSurrogateQualityReport:
    """Per-locale, per-label live surrogate-quality matrix.

    Args:
        locale_reports: Evaluated targets keyed by language code.
        labels: Live labels evaluated for every target.
        expected_languages: Every language the registry snapshot required.
        min_pass_rate: Per-cell fully passing sample-rate floor.
        samples: Seeded samples generated per cell.
        seed: Seed shared by every provider in the run.
        exemptions: Validated exemptions applied to the matrix.
    """

    locale_reports: Mapping[str, LiveSurrogateLocaleReport]
    labels: tuple[str, ...]
    expected_languages: tuple[str, ...]
    min_pass_rate: float = DEFAULT_LIVE_SURROGATE_PASS_RATE
    samples: int = DEFAULT_LIVE_SURROGATE_SAMPLES
    seed: int = DEFAULT_LIVE_SURROGATE_SEED
    exemptions: tuple[LiveSurrogateExemption, ...] = ()

    @property
    def languages(self) -> tuple[str, ...]:
        """Return evaluated language codes in sorted order."""

        return tuple(sorted(self.locale_reports))

    @property
    def missing_languages(self) -> tuple[str, ...]:
        """Return required languages the report did not evaluate."""

        evaluated = set(self.locale_reports)
        return tuple(
            language
            for language in self.expected_languages
            if language not in evaluated
        )

    @property
    def missing_labels(self) -> tuple[str, ...]:
        """Return live labels the report did not evaluate."""

        return tuple(
            label for label in LIVE_SURROGATE_LABELS if label not in self.labels
        )

    def cells(self) -> tuple[LiveSurrogateCell, ...]:
        """Return every cell in ``(language, label)`` order."""

        return tuple(
            self.locale_reports[language].cells[label]
            for language in self.languages
            for label in self.labels
            if label in self.locale_reports[language].cells
        )

    def _cell_keys(self, status: str) -> tuple[str, ...]:
        return tuple(cell.key for cell in self.cells() if cell.status == status)

    @property
    def failing_cells(self) -> tuple[str, ...]:
        """Return ``language/label`` keys below the pass-rate floor."""

        return self._cell_keys(CELL_FAIL)

    @property
    def unscored_cells(self) -> tuple[str, ...]:
        """Return ``language/label`` keys the live scorer could not grade."""

        return self._cell_keys(CELL_UNSCORED)

    @property
    def exempt_cells(self) -> tuple[str, ...]:
        """Return ``language/label`` keys covered by a recorded exemption."""

        return self._cell_keys(CELL_EXEMPT)

    @property
    def failing_languages(self) -> tuple[str, ...]:
        """Return languages with at least one failing or unscored cell."""

        return tuple(
            sorted(
                {
                    cell.language
                    for cell in self.cells()
                    if cell.status in {CELL_FAIL, CELL_UNSCORED}
                }
            )
        )

    @property
    def passed(self) -> bool:
        """Return whether the matrix is complete and every cell passes."""

        return not (
            self.missing_languages
            or self.missing_labels
            or self.failing_cells
            or self.unscored_cells
        )

    def matrix(self) -> dict[str, dict[str, str]]:
        """Return the ``{language: {label: status}}`` matrix."""

        return {
            language: {
                label: self.locale_reports[language].cells[label].status
                for label in self.labels
                if label in self.locale_reports[language].cells
            }
            for language in self.languages
        }

    def with_min_pass_rate(self, min_pass_rate: float) -> "LiveSurrogateQualityReport":
        """Return the report re-graded against ``min_pass_rate``."""

        rate = _probability(min_pass_rate, name="min_pass_rate")
        return replace(
            self,
            min_pass_rate=rate,
            locale_reports={
                language: replace(
                    locale_report,
                    cells={
                        label: cell.with_min_pass_rate(rate)
                        for label, cell in locale_report.cells.items()
                    },
                )
                for language, locale_report in self.locale_reports.items()
            },
        )

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "openmed.eval.live_surrogate_quality",
            "passed": self.passed,
            "min_pass_rate": self.min_pass_rate,
            "samples": self.samples,
            "seed": self.seed,
            "labels": list(self.labels),
            "dimensions": list(SURROGATE_QUALITY_DIMENSIONS),
            "expected_languages": list(self.expected_languages),
            "missing_languages": list(self.missing_languages),
            "missing_labels": list(self.missing_labels),
            "failing_cells": list(self.failing_cells),
            "unscored_cells": list(self.unscored_cells),
            "exempt_cells": list(self.exempt_cells),
            "exemptions": [exemption.to_dict() for exemption in self.exemptions],
            "matrix": self.matrix(),
            "locales": {
                language: self.locale_reports[language].to_dict()
                for language in self.languages
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the full JSON-friendly report with a body digest."""

        body = self._body()
        body["report_digest"] = stable_hash(body)
        return body

    def gate_details(self) -> dict[str, Any]:
        """Return the compact payload used by the release gate check."""

        cells = self.cells()
        body = self._body()
        return {
            "passed": self.passed,
            "min_pass_rate": self.min_pass_rate,
            "samples": self.samples,
            "seed": self.seed,
            "labels": list(self.labels),
            "expected_languages": list(self.expected_languages),
            "evaluated_languages": len(self.locale_reports),
            "missing_languages": list(self.missing_languages),
            "missing_labels": list(self.missing_labels),
            "failing_languages": list(self.failing_languages),
            "matrix": self.matrix(),
            "failing_cells": {
                cell.key: cell.summary() for cell in cells if cell.status == CELL_FAIL
            },
            "unscored_cells": {
                cell.key: cell.unscored_reason
                for cell in cells
                if cell.status == CELL_UNSCORED
            },
            "exempt_cells": {
                cell.key: cell.summary() for cell in cells if cell.status == CELL_EXEMPT
            },
            "cell_value_digests": {
                cell.key: cell.value_digest for cell in cells if cell.value_digest
            },
            "report_digest": stable_hash(body),
        }


@dataclass(frozen=True)
class _SampleOutcome:
    failures: tuple[tuple[str, str], ...]
    digest: str | None

    @property
    def passed(self) -> bool:
        return not self.failures


def live_surrogate_targets(
    *,
    registry: LanguagePackRegistry | None = None,
    national_id_only: Mapping[str, NationalIdOnlyCapability] | None = None,
    script_fallbacks: Mapping[str, Sequence[str]] | None = None,
) -> tuple[LiveSurrogateTarget, ...]:
    """Return one live target per registered pack and national-ID-only language.

    The registry is read at call time, so a pack registered later is picked up
    without any change to this module.

    Args:
        registry: Language-pack registry. Defaults to the process registry.
        national_id_only: National-ID-only declarations. Defaults to the
            built-in catalog.
        script_fallbacks: Script expectations for languages without a
            routable declared script. Defaults to
            :data:`LIVE_SURROGATE_SCRIPT_FALLBACKS`.

    Returns:
        Targets sorted by language code.
    """

    resolved_registry = registry if registry is not None else LANGUAGE_PACK_REGISTRY
    capabilities = (
        NATIONAL_ID_ONLY_CAPABILITIES if national_id_only is None else national_id_only
    )
    fallbacks = (
        LIVE_SURROGATE_SCRIPT_FALLBACKS
        if script_fallbacks is None
        else script_fallbacks
    )

    targets: dict[str, LiveSurrogateTarget] = {}
    for pack in resolved_registry.iter_packs():
        targets[pack.code] = _pack_target(pack, fallbacks)
    for code, capability in sorted(capabilities.items()):
        if code in targets:
            continue
        targets[code] = _national_id_only_target(code, capability, fallbacks)
    return tuple(targets[code] for code in sorted(targets))


def evaluate_live_surrogate_quality(
    *,
    registry: LanguagePackRegistry | None = None,
    national_id_only: Mapping[str, NationalIdOnlyCapability] | None = None,
    languages: Iterable[str] | None = None,
    labels: Sequence[str] = LIVE_SURROGATE_LABELS,
    samples: int = DEFAULT_LIVE_SURROGATE_SAMPLES,
    seed: int = DEFAULT_LIVE_SURROGATE_SEED,
    min_pass_rate: float = DEFAULT_LIVE_SURROGATE_PASS_RATE,
    exemptions: Iterable[LiveSurrogateExemption | Mapping[str, Any]] = (),
    surrogate_factory: LiveSurrogateFactory | None = None,
    script_fallbacks: Mapping[str, Sequence[str]] | None = None,
) -> LiveSurrogateQualityReport:
    """Score the surrogates the registered providers generate today.

    Args:
        registry: Language-pack registry. Defaults to the process registry.
        national_id_only: National-ID-only declarations. Defaults to the
            built-in catalog.
        languages: Optional subset to evaluate. Languages left out are listed
            in :attr:`LiveSurrogateQualityReport.missing_languages`, so a
            subset report never passes the gate.
        labels: Live labels to evaluate.
        samples: Seeded samples per cell.
        seed: Seed shared by every provider in the run.
        min_pass_rate: Per-cell fully passing sample-rate floor.
        exemptions: Exemptions for locales with a documented approximation.
        surrogate_factory: Builds a seeded provider per language. Defaults to
            :func:`default_live_surrogate_factory`.
        script_fallbacks: Overrides :data:`LIVE_SURROGATE_SCRIPT_FALLBACKS`.

    Returns:
        A deterministic report containing only codes, counts, rates and
        digests.

    Raises:
        TypeError: If ``labels``, ``languages`` or ``exemptions`` is not a
            sequence of declarations.
        ValueError: If an argument is out of range, a language or label is
            unknown, or an exemption does not name a documented approximation.
    """

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rate = _probability(min_pass_rate, name="min_pass_rate")
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of live labels")
    selected_labels = tuple(dict.fromkeys(str(label) for label in labels))
    unknown_labels = sorted(set(selected_labels) - set(LIVE_SURROGATE_LABELS))
    if unknown_labels or not selected_labels:
        raise ValueError(f"unknown or empty live surrogate labels: {unknown_labels}")
    selected_labels = tuple(
        label for label in LIVE_SURROGATE_LABELS if label in selected_labels
    )

    targets = live_surrogate_targets(
        registry=registry,
        national_id_only=national_id_only,
        script_fallbacks=script_fallbacks,
    )
    by_language = {target.language: target for target in targets}
    if languages is None:
        selected = tuple(by_language)
    else:
        if isinstance(languages, str):
            raise TypeError("languages must be a sequence of language codes")
        requested = tuple(dict.fromkeys(str(code).lower() for code in languages))
        unknown = sorted(set(requested) - set(by_language))
        if unknown:
            raise ValueError(f"languages are not registered: {unknown}")
        selected = tuple(code for code in by_language if code in requested)

    validated_exemptions = _validate_exemptions(exemptions, by_language)
    exemption_by_cell = {
        (exemption.language, label): exemption
        for exemption in validated_exemptions
        for label in exemption.labels
    }
    factory = surrogate_factory or default_live_surrogate_factory

    locale_reports = {
        code: _evaluate_target(
            by_language[code],
            labels=selected_labels,
            samples=samples,
            seed=seed,
            min_pass_rate=rate,
            factory=factory,
            exemption_by_cell=exemption_by_cell,
        )
        for code in selected
    }
    return LiveSurrogateQualityReport(
        locale_reports=locale_reports,
        labels=selected_labels,
        expected_languages=tuple(by_language),
        min_pass_rate=rate,
        samples=samples,
        seed=seed,
        exemptions=validated_exemptions,
    )


def documented_approximation(target: LiveSurrogateTarget) -> str | None:
    """Return the documented surrogate-locale approximation for ``target``.

    A pack's own ``surrogate_locale_approximation`` wins. Otherwise languages
    listed as documented approximations by the locale-coherence contract use a
    note naming the Faker backend that stands in for the locale.

    Args:
        target: Live target to inspect.

    Returns:
        The documented approximation text, or ``None``.
    """

    if target.approximation:
        return target.approximation
    from openmed.core.anonymizer.locales import (
        _APPROXIMATE_LOCALES,
        resolve_faker_backend_locale,
    )

    if target.language not in _APPROXIMATE_LOCALES:
        return None
    backend = resolve_faker_backend_locale(target.locale)
    if backend == target.locale:
        return f"no native Faker locale for {target.language!r}; uses {backend!r}"
    return (
        f"no native Faker locale for {target.language!r}; {target.locale!r} "
        f"is backed by {backend!r}"
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _pack_target(
    pack: LanguagePack,
    fallbacks: Mapping[str, Sequence[str]],
) -> LiveSurrogateTarget:
    scripts, script_source = _expected_scripts(pack.code, pack.scripts, fallbacks)
    id_locale, id_method = None, None
    if pack.national_id_providers:
        id_method, id_locale = next(iter(pack.national_id_providers.items()))
    return LiveSurrogateTarget(
        language=pack.code,
        locale=pack.surrogate_locale,
        source=TARGET_LANGUAGE_PACK,
        expected_scripts=scripts,
        script_source=script_source,
        national_id_locale=id_locale,
        national_id_method=id_method,
        approximation=pack.surrogate_locale_approximation,
    )


def _national_id_only_target(
    code: str,
    capability: NationalIdOnlyCapability,
    fallbacks: Mapping[str, Sequence[str]],
) -> LiveSurrogateTarget:
    scripts, script_source = _expected_scripts(code, (), fallbacks)
    id_locale, id_method = capability.provider
    return LiveSurrogateTarget(
        language=code,
        locale=capability.locale,
        source=TARGET_NATIONAL_ID_ONLY,
        expected_scripts=scripts,
        script_source=script_source,
        national_id_locale=id_locale,
        national_id_method=id_method,
    )


def _expected_scripts(
    code: str,
    declared: Sequence[str],
    fallbacks: Mapping[str, Sequence[str]],
) -> tuple[tuple[str, ...], str]:
    routable = tuple(script for script in declared if script in SUPPORTED_SCRIPTS)
    if routable:
        return routable, SCRIPT_SOURCE_DECLARED
    fallback = tuple(
        script for script in fallbacks.get(code, ()) if script in SUPPORTED_SCRIPTS
    )
    if fallback:
        return fallback, SCRIPT_SOURCE_FALLBACK
    return (), SCRIPT_SOURCE_NONE


def _validate_exemptions(
    exemptions: Iterable[LiveSurrogateExemption | Mapping[str, Any]],
    targets: Mapping[str, LiveSurrogateTarget],
) -> tuple[LiveSurrogateExemption, ...]:
    if isinstance(exemptions, (str, bytes, Mapping)):
        raise TypeError("exemptions must be a sequence of exemption declarations")
    validated: list[LiveSurrogateExemption] = []
    seen: set[tuple[str, str]] = set()
    for item in exemptions:
        exemption = (
            item
            if isinstance(item, LiveSurrogateExemption)
            else LiveSurrogateExemption.from_mapping(item)
        )
        target = targets.get(exemption.language)
        if target is None:
            raise ValueError(
                f"live surrogate exemption names an unregistered language "
                f"{exemption.language!r}"
            )
        if documented_approximation(target) is None:
            raise ValueError(
                f"live surrogate exemption for {exemption.language!r} requires a "
                "documented surrogate-locale approximation"
            )
        for label in exemption.labels:
            if (exemption.language, label) in seen:
                raise ValueError(
                    f"duplicate live surrogate exemption for "
                    f"{exemption.language}/{label}"
                )
            seen.add((exemption.language, label))
        validated.append(exemption)
    return tuple(sorted(validated, key=lambda item: (item.language, item.labels)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_target(
    target: LiveSurrogateTarget,
    *,
    labels: tuple[str, ...],
    samples: int,
    seed: int,
    min_pass_rate: float,
    factory: LiveSurrogateFactory,
    exemption_by_cell: Mapping[tuple[str, str], LiveSurrogateExemption],
) -> LiveSurrogateLocaleReport:
    with warnings.catch_warnings(), _preserved_locale_warning_state():
        # Locale-approximation and fallback warnings name locales and labels
        # only, but the live run keeps its output to the report itself and
        # leaves the one-time warning state as it found it.
        warnings.simplefilter("ignore")
        primary = _build_source(factory, target.language, seed)
        replay = _build_source(factory, target.language, seed)
        cells: dict[str, LiveSurrogateCell] = {}
        for label in labels:
            cell = _evaluate_cell(
                target,
                label,
                primary=primary,
                replay=replay,
                samples=samples,
                min_pass_rate=min_pass_rate,
            )
            exemption = exemption_by_cell.get((target.language, label))
            cells[label] = _apply_exemption(cell, target, exemption)
    return LiveSurrogateLocaleReport(target=target, cells=cells)


@contextmanager
def _preserved_locale_warning_state() -> Iterator[None]:
    """Restore the anonymizer's one-time locale-warning registry on exit."""

    from openmed.core.anonymizer import locales

    snapshot = set(locales._warned)
    try:
        yield
    finally:
        locales._warned.clear()
        locales._warned.update(snapshot)


def _build_source(
    factory: LiveSurrogateFactory,
    language: str,
    seed: int,
) -> LiveSurrogateSource | None:
    try:
        return factory(language, seed)
    except Exception:  # noqa: BLE001 - reported as a failed cell, never raised
        return None


def _evaluate_cell(
    target: LiveSurrogateTarget,
    label: str,
    *,
    primary: LiveSurrogateSource | None,
    replay: LiveSurrogateSource | None,
    samples: int,
    min_pass_rate: float,
) -> LiveSurrogateCell:
    unscored = _unscored_reason(target, label)
    if unscored is not None:
        return LiveSurrogateCell(
            language=target.language,
            label=label,
            status=CELL_UNSCORED,
            scored_status=CELL_UNSCORED,
            min_pass_rate=min_pass_rate,
            unscored_reason=unscored,
        )

    outcomes = tuple(
        _score_sample(target, label, index, primary=primary, replay=replay)
        for index in range(samples)
    )
    dimensions = LIVE_SURROGATE_LABEL_DIMENSIONS[label]
    dimension_rates = {
        dimension: sum(
            1
            for outcome in outcomes
            if all(failed != dimension for failed, _ in outcome.failures)
        )
        / samples
        for dimension in dimensions
    }
    reason_counts = Counter(
        reason for outcome in outcomes for _, reason in outcome.failures
    )
    passed_samples = sum(1 for outcome in outcomes if outcome.passed)
    pass_rate = passed_samples / samples
    status = CELL_PASS if pass_rate >= min_pass_rate else CELL_FAIL
    return LiveSurrogateCell(
        language=target.language,
        label=label,
        status=status,
        scored_status=status,
        min_pass_rate=min_pass_rate,
        samples=samples,
        passed_samples=passed_samples,
        dimension_rates=dimension_rates,
        reason_counts=dict(reason_counts),
        value_digests=tuple(
            outcome.digest for outcome in outcomes if outcome.digest is not None
        ),
    )


def _apply_exemption(
    cell: LiveSurrogateCell,
    target: LiveSurrogateTarget,
    exemption: LiveSurrogateExemption | None,
) -> LiveSurrogateCell:
    if exemption is None:
        return cell
    record = {
        "reason": exemption.reason,
        "approximation": documented_approximation(target) or "",
    }
    if cell.status == CELL_PASS:
        return replace(cell, exemption=record)
    return replace(cell, status=CELL_EXEMPT, exemption=record)


def _unscored_reason(target: LiveSurrogateTarget, label: str) -> str | None:
    if label in {"name", "address", "city"} and not target.expected_scripts:
        return "no_script_expectation"
    if label == "id":
        if not target.national_id_locale or not target.national_id_method:
            return "no_national_id_provider"
        if not _national_id_validators(target.language):
            return "no_national_id_validator"
    if label == "date" and date_order_for_locale(target.language) is None:
        return "no_locale_date_order"
    return None


def _score_sample(
    target: LiveSurrogateTarget,
    label: str,
    index: int,
    *,
    primary: LiveSurrogateSource | None,
    replay: LiveSurrogateSource | None,
) -> _SampleOutcome:
    original = _synthetic_original(target, label, index)
    if primary is None:
        return _SampleOutcome((("consistency", "source_unavailable"),), None)
    try:
        value = _generate(primary, target, label, original)
    except Exception:  # noqa: BLE001 - provider failures are scored, not raised
        return _SampleOutcome((("consistency", "generation_error"),), None)

    failures: list[tuple[str, str]] = []
    if label in {"name", "address", "city"}:
        reason = _script_failure(value, target)
        if reason:
            failures.append(("script", reason))
    for dimension, reason in _label_failures(target, label, value, original):
        failures.append((dimension, reason))
    reason = _consistency_failure(target, label, value, original, replay)
    if reason:
        failures.append(("consistency", reason))
    return _SampleOutcome(tuple(failures), _digest(value))


def _generate(
    source: LiveSurrogateSource,
    target: LiveSurrogateTarget,
    label: str,
    original: str,
) -> str:
    source_label = LIVE_SURROGATE_SOURCE_LABELS[label]
    if label == "id":
        value = source.surrogate(
            original,
            source_label,
            locale=target.national_id_locale,
        )
    else:
        value = source.surrogate(original, source_label)
    if not isinstance(value, str):
        raise TypeError("live surrogate providers must return strings")
    return value


def _label_failures(
    target: LiveSurrogateTarget,
    label: str,
    value: str,
    original: str,
) -> list[tuple[str, str]]:
    if label == "name":
        return [
            ("naturalness", reason)
            for reason in _name_failures(target, value, original)
        ]
    if label == "address":
        failures = [("naturalness", reason) for reason in _text_failures(value)]
        failures.extend(
            ("format", reason) for reason in _address_format_failures(target, value)
        )
        return failures
    if label == "city":
        failures = [("naturalness", reason) for reason in _text_failures(value)]
        if any(char.isdigit() for char in value):
            failures.append(("naturalness", "city_contains_digit"))
        reason = _city_format_failure(target, value)
        if reason:
            failures.append(("format", reason))
        return failures
    if label == "phone":
        reason = _phone_format_failure(target, value)
        return [("format", reason)] if reason else []
    if label == "id":
        reason = _checksum_failure(target, value)
        return [("checksum", reason)] if reason else []
    if label == "date":
        reason = _date_format_failure(target, value)
        return [("format", reason)] if reason else []
    raise ValueError(f"unknown live surrogate label: {label!r}")


# ---------------------------------------------------------------------------
# Synthetic originals
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _script_alphabet(script: str) -> tuple[str, ...]:
    """Return a stable inventory of lower-case or caseless letters for ``script``."""

    letters: list[str] = []
    for codepoint in range(0x0041, 0x10000):
        char = chr(codepoint)
        if unicodedata.category(char) not in {"Ll", "Lo"}:
            continue
        if unicodedata.decomposition(char).startswith("<"):
            continue
        if detect_script(char) != script:
            continue
        letters.append(char)
        if len(letters) >= _ALPHABET_SIZE:
            break
    return tuple(letters)


def _synthetic_word(script: str, length: int, *, index: int, salt: int) -> str:
    alphabet = _script_alphabet(script)
    word = "".join(
        alphabet[(index * 13 + position * 7 + salt) % len(alphabet)]
        for position in range(length)
    )
    return word.capitalize()


def _synthetic_original(target: LiveSurrogateTarget, label: str, index: int) -> str:
    salt = _LABEL_OFFSETS[label]
    if label == "phone":
        digits = f"{(index * 7919 + 1_234_567) % 10**9:09d}"
        return f"0{digits[:2]} {digits[2:5]} {digits[5:]}"
    if label == "id":
        return f"{(index * 104_729 + 123_456_789) % 10**9:09d}"
    if label == "date":
        year = 1950 + (index * 7) % 50
        month = 1 + (index * 5) % 12
        day = 13 + (index * 3) % 15
        order = date_order_for_locale(target.language)
        if order == "ymd":
            return f"{year:04d}-{month:02d}-{day:02d}"
        if order == "mdy":
            return f"{month:02d}/{day:02d}/{year:04d}"
        return f"{day:02d}/{month:02d}/{year:04d}"

    script = target.primary_script
    if script is None:  # pragma: no cover - unscored before generation
        raise ValueError("script-bearing labels require a script expectation")
    unspaced = script in CJK_SCRIPTS
    if label == "name":
        if unspaced:
            return _synthetic_word(
                script, _UNSPACED_NAME_LENGTH, index=index, salt=salt
            )
        return " ".join(
            _synthetic_word(script, length, index=index, salt=salt + offset)
            for offset, length in enumerate(_NAME_TOKEN_LENGTHS)
        )
    number = 10 + (index * 7) % 90
    if label == "address":
        first = _synthetic_word(script, 3 if unspaced else 5, index=index, salt=salt)
        second = _synthetic_word(
            script, 2 if unspaced else 4, index=index, salt=salt + 1
        )
        separator = "" if unspaced else " "
        return f"{first}{separator}{second}{separator}{number}"
    return _synthetic_word(script, 3 if unspaced else 6, index=index, salt=salt)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def _script_failure(value: str, target: LiveSurrogateTarget) -> str | None:
    ratio = _script_ratio(value, target.expected_scripts)
    primary = target.primary_script or ""
    floor = LIVE_SCRIPT_RATIO_FLOORS.get(primary, DEFAULT_LIVE_SCRIPT_RATIO_FLOOR)
    if ratio >= floor:
        return None
    return "script_ratio_below_floor"


def _script_ratio(value: str, scripts: Sequence[str]) -> float:
    letters = [char for char in value if unicodedata.category(char)[0] in {"L", "M"}]
    if not letters:
        return 0.0
    allowed = set(scripts)
    matched = sum(1 for char in letters if detect_script(char) in allowed)
    return matched / len(letters)


def _text_failures(value: str) -> list[str]:
    stripped = value.strip()
    if not stripped:
        return ["empty_surrogate"]
    if _FALLBACK_PLACEHOLDER_RE.match(stripped):
        return ["fallback_placeholder"]
    if _PLACEHOLDER_RE.search(_ascii_fold(stripped)):
        return ["placeholder_value"]
    return []


def _name_failures(target: LiveSurrogateTarget, value: str, original: str) -> list[str]:
    failures = _text_failures(value)
    if failures:
        return failures
    if any(char.isdigit() for char in value):
        failures.append("name_contains_digit")
    if _inserted_honorific(value, original, target.locale):
        failures.append("inserted_honorific")

    script = target.primary_script or ""
    if script in CJK_SCRIPTS:
        letters = sum(
            1
            for char in value
            if unicodedata.category(char)[0] == "L"
            and detect_script(char) in set(target.expected_scripts)
        )
        if not 2 <= letters <= 6:
            failures.append("name_shape_mismatch")
    else:
        tokens = value.split()
        expected_tokens = min(2, len(original.split()))
        if len(tokens) < expected_tokens or not all(
            any(unicodedata.category(char).startswith("L") for char in token)
            for token in tokens
        ):
            failures.append("name_shape_mismatch")
    if script in INDIC_SCRIPTS and _implausible_indic_token(value):
        failures.append("implausible_letter_sequence")
    return failures


def _implausible_indic_token(value: str) -> bool:
    """Return whether a token is a long run of bare letters without vowel signs.

    Brahmic names attach dependent vowel signs or viramas to most consonants,
    so four or more base letters with no combining mark is the signature of a
    random letter string rather than a name.
    """

    for token in value.split():
        base = sum(1 for char in token if unicodedata.category(char) == "Lo")
        marks = sum(1 for char in token if unicodedata.category(char) in {"Mn", "Mc"})
        if base >= 4 and marks == 0:
            return True
    return False


def _inserted_honorific(value: str, original: str, locale: str) -> bool:
    tokens = tuple(token.casefold() for token in value.split())
    original_tokens = {token.casefold() for token in original.split()}
    for entry in _honorific_vocabulary(locale):
        if set(entry) <= original_tokens:
            continue
        width = len(entry)
        if tokens[:width] == entry:
            return True
        if entry[-1].endswith(".") and any(
            tokens[start : start + width] == entry
            for start in range(1, len(tokens) - width + 1)
        ):
            return True
    return False


@lru_cache(maxsize=None)
def _honorific_vocabulary(locale: str) -> frozenset[tuple[str, ...]]:
    """Return title prefixes Faker's person provider can add for ``locale``.

    Gendered prefix lists are titles. Generic ``prefixes`` lists also carry
    name particles such as ``de`` or ``Le``, so only abbreviated (dotted)
    generic entries are kept.
    """

    from openmed.core.anonymizer.locales import resolve_faker_backend_locale

    provider = _faker_person_provider(resolve_faker_backend_locale(locale))
    entries: set[str] = set()
    for attribute in ("prefixes_male", "prefixes_female", "prefixes_nonbinary"):
        entries.update(_vocabulary_values(getattr(provider, attribute, ())))
    entries.update(
        value
        for value in _vocabulary_values(getattr(provider, "prefixes", ()))
        if value.endswith(".")
    )
    return frozenset(
        tuple(part.casefold() for part in entry.split())
        for entry in entries
        if entry.strip()
    )


def _faker_person_provider(locale: str) -> Any:
    try:
        module = importlib.import_module(f"faker.providers.person.{locale}")
    except ImportError:
        # Faker falls back to its default locale for missing person providers.
        module = importlib.import_module("faker.providers.person.en_US")
    return module.Provider


def _vocabulary_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        return tuple(str(item) for item in value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in value)
    return ()


def _address_format_failures(target: LiveSurrogateTarget, value: str) -> list[str]:
    if target.region in _US_ADDRESS_TEMPLATE_REGIONS:
        return []
    failures: list[str] = []
    street_suffixes, _ = _us_address_vocabulary()
    if _US_BUILDING_NUMBER_RE.match(value) and any(
        token.strip(",.") in street_suffixes for token in value.split()[1:]
    ):
        failures.append("us_street_template")
    if _US_SECONDARY_UNIT_RE.search(value):
        failures.append("us_secondary_unit")
    return failures


def _city_format_failure(target: LiveSurrogateTarget, value: str) -> str | None:
    if target.region in _US_ADDRESS_TEMPLATE_REGIONS:
        return None
    stripped = value.strip()
    _, city_prefixes = _us_address_vocabulary()
    tokens = stripped.split()
    if len(tokens) >= 2 and tokens[0] in city_prefixes:
        return "us_city_template"
    if len(tokens) == 1 and _US_CITY_SUFFIX_RE.match(stripped):
        return "us_city_template"
    return None


@lru_cache(maxsize=1)
def _us_address_vocabulary() -> tuple[frozenset[str], tuple[str, ...]]:
    """Return Faker's en_US street suffixes and city prefixes."""

    try:
        from faker.providers.address.en_US import Provider
    except ImportError:  # pragma: no cover - Faker is a hard dependency
        return _FALLBACK_US_STREET_SUFFIXES, _FALLBACK_US_CITY_PREFIXES
    street_suffixes = frozenset(
        _vocabulary_values(getattr(Provider, "street_suffixes", ()))
    )
    city_prefixes = tuple(_vocabulary_values(getattr(Provider, "city_prefixes", ())))
    return (
        street_suffixes or _FALLBACK_US_STREET_SUFFIXES,
        city_prefixes or _FALLBACK_US_CITY_PREFIXES,
    )


def _phone_format_failure(target: LiveSurrogateTarget, value: str) -> str | None:
    if not any(char.isdigit() for char in value):
        return "phone_without_digits"
    if target.region in _NANP_REGIONS:
        return None
    if any(pattern.search(value) for pattern in _US_PHONE_SHAPES):
        return "us_phone_shape"
    return None


def _checksum_failure(target: LiveSurrogateTarget, value: str) -> str | None:
    validators = _national_id_validators(target.language)
    for validator in validators:
        try:
            if validator(value):
                return None
        except Exception:  # noqa: BLE001 - a crashing validator is a failure
            continue
    return "national_id_checksum_failed"


def _national_id_validators(language: str) -> list[Callable[[str], bool]]:
    from openmed.core.language_pack_coherence import (
        _national_id_validators as validators_for_language,
    )

    return list(validators_for_language(language))


def _date_format_failure(target: LiveSurrogateTarget, value: str) -> str | None:
    expected_order = date_order_for_locale(target.language)
    parsed = parse_date(value, target.language)
    if parsed.normalized is None:
        return parsed.reason or "unparsed_date"
    if parsed.order != expected_order:
        return "locale_date_order_mismatch"
    return None


def _consistency_failure(
    target: LiveSurrogateTarget,
    label: str,
    value: str,
    original: str,
    replay: LiveSurrogateSource | None,
) -> str | None:
    try:
        replayed = (
            _generate(replay, target, label, original) if replay is not None else None
        )
    except Exception:  # noqa: BLE001 - a replay crash is non-determinism
        replayed = None
    if replayed != value:
        return "nondeterministic"
    if value.strip() == original.strip():
        return "surrogate_equals_original"
    if label in {"name", "address", "city"}:
        surrogate = value.casefold()
        for token in original.split():
            letters = sum(1 for char in token if unicodedata.category(char)[0] == "L")
            if letters >= 3 and token.casefold() in surrogate:
                return "original_token_retained"
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]


def _ascii_fold(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text)
    return "".join(char for char in folded if not unicodedata.combining(char))


def _probability(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number between zero and one")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number between zero and one") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a number between zero and one")
    return number


__all__ = [
    "CELL_EXEMPT",
    "CELL_FAIL",
    "CELL_PASS",
    "CELL_STATUSES",
    "CELL_UNSCORED",
    "DEFAULT_LIVE_SCRIPT_RATIO_FLOOR",
    "DEFAULT_LIVE_SURROGATE_PASS_RATE",
    "DEFAULT_LIVE_SURROGATE_SAMPLES",
    "DEFAULT_LIVE_SURROGATE_SEED",
    "LIVE_SCRIPT_RATIO_FLOORS",
    "LIVE_SURROGATE_LABELS",
    "LIVE_SURROGATE_LABEL_DIMENSIONS",
    "LIVE_SURROGATE_SCRIPT_FALLBACKS",
    "LIVE_SURROGATE_SOURCE_LABELS",
    "LiveSurrogateCell",
    "LiveSurrogateExemption",
    "LiveSurrogateFactory",
    "LiveSurrogateLocaleReport",
    "LiveSurrogateQualityReport",
    "LiveSurrogateSource",
    "LiveSurrogateTarget",
    "SCRIPT_SOURCE_DECLARED",
    "SCRIPT_SOURCE_FALLBACK",
    "SCRIPT_SOURCE_NONE",
    "TARGET_LANGUAGE_PACK",
    "TARGET_NATIONAL_ID_ONLY",
    "default_live_surrogate_factory",
    "documented_approximation",
    "evaluate_live_surrogate_quality",
    "live_surrogate_targets",
]
