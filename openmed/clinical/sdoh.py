"""SHAC-aligned SDOH finding schema and determinant dispatcher.

The employment and living-status extractors use a compact OpenMed-maintained
cue table containing only synthetic/public phrases. Food insecurity is an
OpenMed extension beyond the five core SHAC determinant categories. The real
Social History Annotated Corpus (SHAC) is DUA-gated, eval-only, and must never
be bundled with OpenMed or loaded by this runtime module.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .context import (
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    resolve_negation,
    resolve_temporality,
)
from .sdoh_completeness import (
    SDOHCategoryResult,
    SDOHCategoryState,
    audit_sdoh_completeness,
)
from .status_vocab import (
    normalize_employment_status,
    normalize_living_status,
    normalize_substance_status,
)

SOCIAL_HISTORY_SECTION = "social_history"

SDOH_SUBSTANCE_CUES_RESOURCE = "data/sdoh_substance_cues.yaml"
_SUBSTANCE_CUES_PACKAGE = "openmed.clinical"
_SUBSTANCE_CATEGORIES = (
    "tobacco",
    "alcohol",
    "drug",
)
_SUBSTANCE_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?<!\w)(?:but|however|although|whereas)(?!\w)",
    re.IGNORECASE,
)
_SUBSTANCE_COORDINATOR_RE = re.compile(
    r",|(?<!\w)(?:and|or)(?!\w)",
    re.IGNORECASE,
)
_SUBSTANCE_LOCAL_STATUS_RE = re.compile(
    r"(?<!\w)(?:"
    r"active|current(?:ly)?|former|ex[-\s]?smoker|quit|stopped|"
    r"past|remote|history\s+of|hx\s+of|in\s+remission|status\s+post|s/p|"
    r"den(?:y|ies|ied)|never|none|no|not|without|does\s+not|"
    r"abstain(?:s|ed|ing)?|abstinent|non[-\s]?smoker|"
    r"smoker|smoking|uses|drinks?|vapes?|vaping|"
    r"occasional(?:ly)?|daily|weekly|monthly|rarely"
    r")(?!\w)",
    re.IGNORECASE,
)

_SDOH_SUBSTANCE_STATUS = {
    "current": "current",
    "former": "past",
    "never": "none",
    "unknown": "unknown",
}

SHAC_DATA_POLICY = (
    "Real SHAC data is DUA-gated and eval-only; runtime extraction uses only "
    "synthetic or public data."
)
SDOH_SOCIAL_CUES_RESOURCE = "data/sdoh_social_cues.yaml"
FOOD_INSECURITY_EXTENSION_NOTE = (
    "Food insecurity is an OpenMed extension beyond the five core SHAC "
    "determinant categories."
)

_SOCIAL_CUES_PACKAGE = "openmed.clinical"
_CLAUSE_RE = re.compile(r"[^.;!?\n]+")
_NON_ASSERTIVE_CLAUSE_RE = re.compile(
    r"^\s*(?:ask\s+(?:about|whether)|screen(?:ing)?\s+(?:for|question\s*:)|"
    r"(?:patient\s+)?education\s*:|counsel(?:ing|ling)\s*:)",
    re.IGNORECASE,
)
_UNANSWERED_TEMPLATE_RE = re.compile(r"\[\s*\]|_{3,}")
_DOUBLE_NEGATED_UNEMPLOYMENT_RE = re.compile(
    r"(?<!\w)not\s+unemployed(?!\w)", re.IGNORECASE
)

SpanOffset = tuple[int, int]
_LANGUAGE_RE = re.compile(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*")


@dataclass(frozen=True)
class SDOHFinding:
    """One SHAC-style social-determinant trigger and its arguments.

    Args:
        category: Determinant trigger category, such as ``"tobacco"``.
        value: Trigger value or normalized determinant type.
        status: Optional SHAC Status argument.
        extent: Optional SHAC Extent argument.
        temporality: Optional SHAC Temporality argument.
        span: Half-open source character offsets for the trigger.
        score: Confidence between zero and one, inclusive.
    """

    category: str
    value: str
    status: str | None
    extent: str | None
    temporality: str | None
    span: SpanOffset
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _required_text(self.category, "category"))
        object.__setattr__(self, "value", _required_text(self.value, "value"))
        for field_name in ("status", "extent", "temporality"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "span", _span_offset(self.span, "finding span"))

        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be a number")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible finding mapping."""

        return {
            "category": self.category,
            "value": self.value,
            "status": self.status,
            "extent": self.extent,
            "temporality": self.temporality,
            "span": list(self.span),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SDOHFinding:
        """Build a finding from :meth:`to_dict` compatible data."""

        return cls(
            category=payload["category"],
            value=payload["value"],
            status=payload.get("status"),
            extent=payload.get("extent"),
            temporality=payload.get("temporality"),
            span=payload["span"],
            score=payload["score"],
        )


@dataclass(frozen=True)
class _CueMatch:
    start: int
    end: int
    value: str


@runtime_checkable
class DeterminantExtractor(Protocol):
    """Callable contract implemented by one determinant extractor."""

    def __call__(
        self,
        text: str,
        spans: Sequence[Any],
    ) -> Iterable[SDOHFinding]:
        """Extract findings from source text and candidate spans."""


class DeterminantExtractorRegistry:
    """Deterministic registry keyed by a determinant name."""

    def __init__(self) -> None:
        self._extractors: dict[str, DeterminantExtractor] = {}
        self._languages: dict[str, tuple[str, ...]] = {}

    def register(
        self,
        determinant: str,
        extractor: DeterminantExtractor,
        *,
        replace: bool = False,
        languages: Sequence[str] | None = None,
    ) -> None:
        """Register a callable for one determinant.

        Args:
            determinant: Stable non-empty registry key.
            extractor: Callable satisfying :class:`DeterminantExtractor`.
            replace: Replace an existing extractor when true.
            languages: Explicit supported language tags for the language-aware
                entry point. Omission leaves support undeclared; the legacy
                dispatcher still runs the extractor as before.

        Raises:
            TypeError: If ``extractor`` is not callable.
            ValueError: If the key is empty or already registered.
        """

        key = _required_text(determinant, "determinant")
        if not callable(extractor):
            raise TypeError("determinant extractor must be callable")
        if not replace and key in self._extractors:
            raise ValueError(f"determinant extractor already registered for {key!r}")
        declared = _declared_sdoh_languages(languages)
        self._extractors[key] = extractor
        self._languages[key] = declared

    def unregister(self, determinant: str) -> None:
        """Remove the extractor registered for ``determinant``."""

        key = _required_text(determinant, "determinant")
        del self._extractors[key]
        del self._languages[key]

    def available(self) -> tuple[str, ...]:
        """Return registered determinant keys in deterministic order."""

        return tuple(sorted(self._extractors))

    def items(self) -> tuple[tuple[str, DeterminantExtractor], ...]:
        """Return a stable snapshot of registered extractors."""

        return tuple((key, self._extractors[key]) for key in sorted(self._extractors))

    def language_items(
        self,
    ) -> tuple[tuple[str, DeterminantExtractor, tuple[str, ...]], ...]:
        """Return extractors and their explicit language declarations together."""
        return tuple(
            (key, self._extractors[key], self._languages[key])
            for key in sorted(self._extractors)
        )

    def __len__(self) -> int:
        return len(self._extractors)


_DETERMINANT_EXTRACTORS = DeterminantExtractorRegistry()


def register_determinant_extractor(
    determinant: str,
    extractor: DeterminantExtractor,
    *,
    replace: bool = False,
    languages: Sequence[str] | None = None,
) -> None:
    """Register a process-wide determinant extractor.

    Args:
        determinant: Stable registry category.
        extractor: Trusted local callback that emits findings.
        replace: Replace an existing extractor when true.
        languages: Explicit supported tags; omitted support is undeclared for
            :func:`extract_sdoh_with_language` only.
    """

    _DETERMINANT_EXTRACTORS.register(
        determinant, extractor, replace=replace, languages=languages
    )


def unregister_determinant_extractor(determinant: str) -> None:
    """Unregister a process-wide determinant extractor."""

    _DETERMINANT_EXTRACTORS.unregister(determinant)


def available_determinant_extractors() -> tuple[str, ...]:
    """Return process-wide determinant keys in deterministic order."""

    return _DETERMINANT_EXTRACTORS.available()


@dataclass(frozen=True, slots=True, repr=False)
class SDOHExtractionResult:
    """Protected findings plus value-free language-processing category results.

    Args:
        findings: Original typed findings for caller-owned downstream handling.
            Finding values and extents may be sensitive; do not log them.
        category_results: One processing result per configured determinant.
    """

    findings: tuple[SDOHFinding, ...]
    category_results: tuple[SDOHCategoryResult, ...]

    def __post_init__(self) -> None:
        if type(self.findings) is not tuple or any(
            type(item) is not SDOHFinding for item in self.findings
        ):
            raise ValueError("invalid_sdoh_findings")
        if type(self.category_results) is not tuple or any(
            type(item) is not SDOHCategoryResult for item in self.category_results
        ):
            raise ValueError("invalid_sdoh_category_results")
        audit = audit_sdoh_completeness(
            (item.category for item in self.category_results), self.category_results
        )
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.category] = counts.get(finding.category, 0) + 1
        if set(counts) - {item.category for item in audit.categories} or any(
            counts.get(item.category, 0) != item.finding_count
            for item in audit.categories
        ):
            raise ValueError("invalid_sdoh_finding_counts")
        object.__setattr__(self, "category_results", audit.categories)

    def __repr__(self) -> str:
        return (
            f"SDOHExtractionResult(findings={len(self.findings)}, "
            f"categories={len(self.category_results)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return categories, processing states, counts and source offsets only."""
        return {
            "schema_version": 1,
            "finding_count": len(self.findings),
            "categories": [item.to_dict() for item in self.category_results],
            "findings": [
                {"category": item.category, "span": list(item.span)}
                for item in self.findings
            ],
        }


def extract_sdoh_with_language(
    text: str,
    spans: Iterable[Any] = (),
    sections: Iterable[Mapping[str, Any] | object] | None = None,
    *,
    language: str | None,
) -> SDOHExtractionResult:
    """Extract only explicitly supported categories and report every outcome.

    Args:
        text: Caller-owned clinical text, never copied into the audit output.
        spans: Upstream candidate spans with character offsets.
        sections: Optional canonical Social History section boundaries.
        language: Explicit language tag or caller-provided ``None``. No language
            detection runs; missing support yields ``unsupported``. A declared
            base tag covers its regional tags, for example ``en``/``en-US``.

    Returns:
        Original findings plus category processing records suitable for
        :func:`audit_sdoh_completeness`. A processed zero is unmentioned, not
        negative. Failed categories never return their partial findings.

    Raises:
        ValueError: For malformed language tags or category metadata; submitted
            values are never included in the error.
        TypeError: If text is not a string. Invalid section/candidate inputs
            produce value-free ``failed`` category records.
    """
    _validate_extractor_text(text)
    requested = _sdoh_language(language)
    records = []
    selected = []
    for category, extractor, declared in _DETERMINANT_EXTRACTORS.language_items():
        # Validate category codes before any trusted callback is entered.
        SDOHCategoryResult(category, SDOHCategoryState.PROCESSED)
        reason = None
        failed = False
        if requested is None or not declared:
            reason = "language_undeclared"
        elif not _supports_sdoh_language(declared, requested):
            reason = "unsupported_language"
        else:
            try:
                cue_languages = _builtin_cue_languages(extractor)
            except Exception:
                failed = True
                cue_languages = None
            if failed:
                reason = "cue_language_invalid"
            elif cue_languages is not None and not cue_languages:
                reason = "cue_language_undeclared"
            elif cue_languages is not None and not _supports_sdoh_language(
                cue_languages, requested
            ):
                reason = "unsupported_language"
        if reason is not None:
            records.append(
                SDOHCategoryResult(
                    category,
                    SDOHCategoryState.FAILED
                    if failed
                    else SDOHCategoryState.UNSUPPORTED,
                    reason_code=reason,
                )
            )
        else:
            selected.append((category, extractor))
    if not selected:
        return SDOHExtractionResult((), tuple(records))

    scope_failed = False
    try:
        candidate_spans = tuple(spans)
        allowed_ranges = (
            None if sections is None else _social_history_ranges(text, sections)
        )
        if allowed_ranges is not None:
            candidate_spans = tuple(
                span
                for span in candidate_spans
                if _item_within_ranges(span, allowed_ranges)
            )
    except Exception:
        scope_failed = True
    if scope_failed:
        records.extend(
            SDOHCategoryResult(
                category, SDOHCategoryState.FAILED, reason_code="scope_invalid"
            )
            for category, _ in selected
        )
        return SDOHExtractionResult((), tuple(records))
    if allowed_ranges is not None:
        if not allowed_ranges:
            records.extend(
                SDOHCategoryResult(
                    category,
                    SDOHCategoryState.SKIPPED,
                    reason_code="section_not_selected",
                )
                for category, _ in selected
            )
            return SDOHExtractionResult((), tuple(records))
    findings = []
    for category, extractor in selected:
        category_findings = []
        failed = False
        try:
            emitted = extractor(text, candidate_spans)
            for finding in emitted:
                if (
                    type(finding) is not SDOHFinding
                    or finding.category != category
                    or not 0 <= finding.span[0] < finding.span[1] <= len(text)
                ):
                    raise ValueError("invalid_sdoh_finding")
                if _non_assertive_clause(text, finding.span):
                    continue
                if allowed_ranges is None or _offset_within_ranges(
                    finding.span, allowed_ranges
                ):
                    category_findings.append(finding)
        except Exception:
            failed = True
        if failed:
            records.append(
                SDOHCategoryResult(
                    category, SDOHCategoryState.FAILED, reason_code="extractor_failed"
                )
            )
        else:
            findings.extend(category_findings)
            records.append(
                SDOHCategoryResult(
                    category, SDOHCategoryState.PROCESSED, len(category_findings)
                )
            )
    return SDOHExtractionResult(tuple(findings), tuple(records))


def _sdoh_language(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) > 63:
        raise ValueError("invalid_sdoh_language")
    if not value:
        return None
    tag = value.lower().replace("_", "-")
    if _LANGUAGE_RE.fullmatch(tag) is None:
        raise ValueError("invalid_sdoh_language")
    return tag


def _declared_sdoh_languages(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or len(values) > 64
    ):
        raise ValueError("invalid_sdoh_language_declaration")
    declared = tuple(_sdoh_language(value) for value in values)
    if None in declared or len(set(declared)) != len(declared):
        raise ValueError("invalid_sdoh_language_declaration")
    return tuple(sorted(declared))


def _supports_sdoh_language(declared: tuple[str, ...], requested: str) -> bool:
    return requested in declared or requested.split("-", 1)[0] in declared


def _builtin_cue_languages(extractor: DeterminantExtractor) -> tuple[str, ...] | None:
    if any(
        extractor is builtin
        for builtin in (
            extract_employment_findings,
            extract_food_insecurity_findings,
            extract_living_status_findings,
        )
    ):
        return _declared_sdoh_languages(
            _load_default_sdoh_social_cues().get("languages")
        )
    if any(
        extractor is builtin
        for builtin in (_extract_tobacco, _extract_alcohol, _extract_drug)
    ):
        resource = resources.files(_SUBSTANCE_CUES_PACKAGE).joinpath(
            SDOH_SUBSTANCE_CUES_RESOURCE
        )
        payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
        return _declared_sdoh_languages(payload.get("languages"))
    return None


def extract_sdoh(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any] | object] | None = None,
) -> list[SDOHFinding]:
    """Dispatch registered determinant extractors over an optional section scope.

    Args:
        text: Original clinical document text.
        spans: Upstream candidate spans. Each scoped span must expose integer
            ``start`` and ``end`` mapping keys or attributes.
        sections: Optional section spans, normally returned by
            :func:`openmed.clinical.sections.detect_sections`. When supplied,
            only candidates and findings fully contained in canonical Social
            History sections are retained. When omitted, ``text`` and ``spans``
            are treated as an already selected caller-controlled window.

    Returns:
        Findings emitted by every registered determinant extractor.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    extractors = _DETERMINANT_EXTRACTORS.items()
    if not extractors:
        return []

    candidate_spans = tuple(spans)
    allowed_ranges: tuple[SpanOffset, ...] | None = None
    if sections is not None:
        allowed_ranges = _social_history_ranges(text, sections)
        if not allowed_ranges:
            return []
        candidate_spans = tuple(
            span
            for span in candidate_spans
            if _item_within_ranges(span, allowed_ranges)
        )

    findings: list[SDOHFinding] = []
    for _, extractor in extractors:
        for finding in extractor(text, candidate_spans):
            if not isinstance(finding, SDOHFinding):
                raise TypeError("determinant extractors must emit SDOHFinding values")
            if _non_assertive_clause(text, finding.span):
                continue
            if allowed_ranges is None or _offset_within_ranges(
                finding.span,
                allowed_ranges,
            ):
                findings.append(finding)
    return findings


def _non_assertive_clause(text: str, span: SpanOffset) -> bool:
    """Exclude questions, educational instructions, and empty templates."""

    for clause in _CLAUSE_RE.finditer(text):
        if clause.start() <= span[0] < clause.end():
            value = clause.group(0)
            return bool(
                text[clause.end() : clause.end() + 1] == "?"
                or _NON_ASSERTIVE_CLAUSE_RE.match(value)
                or _UNANSWERED_TEMPLATE_RE.search(value)
            )
    return False


def load_sdoh_social_cues(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the unrestricted social-determinant cue table.

    Args:
        path: Optional replacement YAML path, primarily for downstream
            validation. The packaged OpenMed cue table is used when omitted.

    Returns:
        A detached copy of the validated cue-table payload.
    """

    if path is None:
        return copy.deepcopy(_load_default_sdoh_social_cues())
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _validate_sdoh_social_cues(payload)


def extract_employment_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract deterministic employment status and occupation findings.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Employment findings anchored to their source cue spans.

    ``spans`` is accepted for registry compatibility. These compact cue-based
    extractors scan caller-selected text directly; the dispatcher still applies
    candidate and Social History section boundaries to its inputs and outputs.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("employment")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        clause = text[clause_start:clause_end]
        status_match = _status_match(clause, config)
        type_match = _typed_cue_match(clause, config["types"])
        if status_match is None and type_match is None:
            continue

        if status_match is None:
            assert type_match is not None
            status_match = _CueMatch(
                start=type_match.start,
                end=type_match.end,
                value="employed",
            )
        match_start, match_end = _combined_offset(status_match, type_match)
        absolute_start = clause_start + match_start
        absolute_end = clause_start + match_end
        temporality = _finding_temporality(text, absolute_start, absolute_end)
        status = normalize_employment_status(
            clause,
            temporality=temporality,
        )
        if status == "unknown":
            status = status_match.value
        if _DOUBLE_NEGATED_UNEMPLOYMENT_RE.search(clause):
            status = "unknown"
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=type_match.value if type_match else status_match.value,
                status=status,
                extent=None,
                temporality=temporality,
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


def extract_living_status_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract deterministic housing and living-situation findings.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Living-status findings anchored to their source cue spans.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("living_status")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        clause = text[clause_start:clause_end]
        status_match = _status_match(clause, config)
        if status_match is None:
            continue

        absolute_start = clause_start + status_match.start
        absolute_end = clause_start + status_match.end
        temporality = _finding_temporality(text, absolute_start, absolute_end)
        status = normalize_living_status(clause, temporality=temporality)
        if status == "unknown":
            status = status_match.value
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=status_match.value,
                status=status,
                extent=None,
                temporality=temporality,
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


def extract_food_insecurity_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract food-insecurity cues as an extension beyond core SHAC.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Food-insecurity findings anchored to their source cue spans.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("food_insecurity")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        cue_match = _cue_match(text[clause_start:clause_end], config["cues"])
        if cue_match is None:
            continue

        absolute_start = clause_start + cue_match.start
        absolute_end = clause_start + cue_match.end
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=config["value"],
                status=config["status"],
                extent=None,
                temporality=_finding_temporality(
                    text,
                    absolute_start,
                    absolute_end,
                ),
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


@lru_cache(maxsize=1)
def _load_default_sdoh_social_cues() -> dict[str, Any]:
    resource = resources.files(_SOCIAL_CUES_PACKAGE).joinpath(SDOH_SOCIAL_CUES_RESOURCE)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _validate_sdoh_social_cues(payload)


def _validate_sdoh_social_cues(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SDOH social cue table requires schema_version 1")

    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or not provenance.get("source")
        or provenance.get("restricted_data") is not False
    ):
        raise ValueError("SDOH social cues require unrestricted provenance")

    determinants = payload.get("determinants")
    if not isinstance(determinants, Mapping):
        raise ValueError("SDOH social cues require a determinants mapping")

    employment = _validate_status_determinant(determinants, "employment")
    types = employment.get("types")
    if not isinstance(types, Mapping) or not types:
        raise ValueError("employment social cues require occupation types")
    _validate_cue_mapping(types, "employment.types")

    _validate_status_determinant(determinants, "living_status")

    food = determinants.get("food_insecurity")
    if not isinstance(food, Mapping):
        raise ValueError("food_insecurity social cues must be a mapping")
    _validate_determinant_identity(food, "food_insecurity")
    _validate_cue_sequence(food.get("cues"), "food_insecurity.cues")
    if food.get("status") != "current" or food.get("value") != "food_insecure":
        raise ValueError("food_insecurity social cues require canonical values")
    if food.get("extension_beyond_core_shac") is not True:
        raise ValueError("food_insecurity must be marked as a SHAC extension")
    extension_note = food.get("extension_note")
    if not isinstance(extension_note, str) or "beyond the five core SHAC" not in (
        extension_note
    ):
        raise ValueError("food_insecurity requires its SHAC extension note")
    return payload


def _validate_status_determinant(
    determinants: Mapping[str, Any],
    determinant: str,
) -> Mapping[str, Any]:
    config = determinants.get(determinant)
    if not isinstance(config, Mapping):
        raise ValueError(f"{determinant} social cues must be a mapping")
    _validate_determinant_identity(config, determinant)

    priority = config.get("status_priority")
    status_cues = config.get("status_cues")
    _validate_cue_sequence(priority, f"{determinant}.status_priority")
    assert isinstance(priority, Sequence) and not isinstance(priority, str | bytes)
    if not isinstance(status_cues, Mapping) or not status_cues:
        raise ValueError(f"{determinant}.status_cues must be a mapping")
    _validate_cue_mapping(status_cues, f"{determinant}.status_cues")
    if set(priority) != set(status_cues):
        raise ValueError(f"{determinant} status priority must cover every status")
    return config


def _validate_determinant_identity(
    config: Mapping[str, Any],
    determinant: str,
) -> None:
    if config.get("category") != determinant:
        raise ValueError(f"{determinant} requires a matching category")
    score = config.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError(f"{determinant} requires a score between 0.0 and 1.0")


def _validate_cue_mapping(value: Mapping[Any, Any], field_name: str) -> None:
    for key, cues in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} requires non-empty string keys")
        _validate_cue_sequence(cues, f"{field_name}.{key}")


def _validate_cue_sequence(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field_name} requires non-empty string cues")


def _determinant_config(determinant: str) -> Mapping[str, Any]:
    return _load_default_sdoh_social_cues()["determinants"][determinant]


def _clause_offsets(text: str) -> Iterable[SpanOffset]:
    for match in _CLAUSE_RE.finditer(text):
        segment = match.group()
        leading_space = len(segment) - len(segment.lstrip())
        trailing_space = len(segment) - len(segment.rstrip())
        start = match.start() + leading_space
        end = match.end() - trailing_space
        if start < end:
            yield start, end


def _status_match(clause: str, config: Mapping[str, Any]) -> _CueMatch | None:
    status_cues = config["status_cues"]
    for status in config["status_priority"]:
        match = _cue_match(clause, status_cues[status])
        if match is not None:
            return _CueMatch(match.start, match.end, status)
    return None


def _typed_cue_match(
    clause: str,
    cue_mapping: Mapping[str, Sequence[str]],
) -> _CueMatch | None:
    matches: list[_CueMatch] = []
    for value, cues in cue_mapping.items():
        match = _cue_match(clause, cues)
        if match is not None:
            matches.append(_CueMatch(match.start, match.end, value))
    return (
        min(matches, key=lambda item: (item.start, -(item.end - item.start)))
        if matches
        else None
    )


def _cue_match(text: str, cues: Sequence[str]) -> _CueMatch | None:
    matches: list[_CueMatch] = []
    for cue in sorted(cues, key=len, reverse=True):
        match = _cue_pattern(cue).search(text)
        if match is not None:
            matches.append(_CueMatch(match.start(), match.end(), cue))
    return (
        min(matches, key=lambda item: (item.start, -(item.end - item.start)))
        if matches
        else None
    )


@lru_cache(maxsize=512)
def _cue_pattern(cue: str) -> re.Pattern[str]:
    escaped = re.escape(" ".join(cue.split())).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _combined_offset(
    required: _CueMatch,
    optional: _CueMatch | None,
) -> SpanOffset:
    if optional is None:
        return required.start, required.end
    return min(required.start, optional.start), max(required.end, optional.end)


def _finding_temporality(text: str, start: int, end: int) -> str | None:
    try:
        return resolve_temporality(
            {
                "text": text[start:end],
                "context": text,
                "start": start,
                "end": end,
            }
        )
    except (TypeError, ValueError):
        return None


def _validate_extractor_text(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")


def _extract_tobacco(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "tobacco",
    )


def _parse_tobacco_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?P<amount>\d+(?:\.\d+)?)\s*pack(?:-|\s+)years?\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    amount = match.group("amount")
    return f"{amount} pack-years"


def _extract_alcohol(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "alcohol",
    )


def _parse_alcohol_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?P<amount>\d+(?:\.\d+)?)\s+drinks?"
        r"\s*(?:/|per\s+|a\s+)week\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    amount = match.group("amount")
    return f"{amount} drinks/week"


def _extract_drug(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "drug",
    )


def _parse_drug_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?:occasional(?:ly)?|daily|weekly|monthly|rarely)\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    value = match.group(0).lower()

    if value == "occasionally":
        return "occasional"

    return value


def _parse_substance_extent(
    category: str,
    text: str,
) -> str | None:
    if category == "tobacco":
        return _parse_tobacco_extent(text)

    if category == "alcohol":
        return _parse_alcohol_extent(text)

    if category == "drug":
        return _parse_drug_extent(text)

    return None


@lru_cache(maxsize=1)
def _load_substance_cues() -> dict[str, tuple[str, ...]]:
    resource = resources.files(_SUBSTANCE_CUES_PACKAGE).joinpath(
        SDOH_SUBSTANCE_CUES_RESOURCE
    )

    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise ValueError("substance cue resource must be a mapping")

    if payload.get("schema_version") != 1:
        raise ValueError("substance cue resource requires schema_version 1")

    determinants = payload.get("determinants")

    if not isinstance(determinants, Mapping):
        raise ValueError("substance cue resource requires determinants")

    result: dict[str, tuple[str, ...]] = {}

    for category in _SUBSTANCE_CATEGORIES:
        entry = determinants.get(category)

        if not isinstance(entry, Mapping):
            raise ValueError(
                f"substance cue resource requires determinant {category!r}"
            )

        triggers = entry.get("triggers")

        if (
            not isinstance(triggers, Sequence)
            or isinstance(triggers, str | bytes)
            or not triggers
        ):
            raise ValueError(f"substance determinant {category!r} requires triggers")

        cleaned: list[str] = []

        for cue in triggers:
            if not isinstance(cue, str) or not cue.strip():
                raise ValueError(
                    f"substance determinant {category!r} contains an invalid trigger"
                )

            normalized = " ".join(cue.split())

            if normalized not in cleaned:
                cleaned.append(normalized)

        result[category] = tuple(cleaned)

    return result


def _substance_context_bounds(
    text: str,
    start: int,
    end: int,
) -> SpanOffset:
    boundaries = ".;\n!?"

    left = max(text.rfind(boundary, 0, start) for boundary in boundaries)

    right_positions = [text.find(boundary, end) for boundary in boundaries]

    right_positions = [position for position in right_positions if position != -1]

    right = min(right_positions) if right_positions else len(text)

    left += 1

    for boundary in _SUBSTANCE_CLAUSE_BOUNDARY_RE.finditer(text, left, right):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right = boundary.start()
            break

    return _coordinated_substance_context_bounds(
        text,
        start,
        end,
        left,
        right,
    )


def _coordinated_substance_context_bounds(
    text: str,
    start: int,
    end: int,
    left: int,
    right: int,
) -> SpanOffset:
    """Isolate explicit statuses while preserving shared coordinated cues."""

    coordinators = tuple(_SUBSTANCE_COORDINATOR_RE.finditer(text, left, right))
    if not coordinators:
        return left, right

    segments: list[SpanOffset] = []
    segment_start = left
    for coordinator in coordinators:
        segments.append((segment_start, coordinator.start()))
        segment_start = coordinator.end()
    segments.append((segment_start, right))

    target_index = next(
        (
            index
            for index, (segment_start, segment_end) in enumerate(segments)
            if segment_start <= start and end <= segment_end
        ),
        None,
    )
    if target_index is None:
        return left, right

    segment_categories = tuple(
        _substance_categories_in_text(text[segment_start:segment_end])
        for segment_start, segment_end in segments
    )
    categories = set().union(*segment_categories)
    if len(categories) < 2:
        return left, right

    local_status = tuple(
        bool(categories_in_segment)
        and _has_local_substance_status(
            text[segment_start:segment_end],
            categories_in_segment,
        )
        for (segment_start, segment_end), categories_in_segment in zip(
            segments,
            segment_categories,
            strict=True,
        )
    )
    if not any(local_status):
        return left, right

    if local_status[target_index]:
        left = segments[target_index][0]
    else:
        prior_local = [index for index in range(target_index) if local_status[index]]
        if prior_local:
            left = segments[prior_local[-1]][0]

    later_local = [
        index for index in range(target_index + 1, len(segments)) if local_status[index]
    ]
    if later_local:
        right = coordinators[later_local[0] - 1].start()

    return left, right


def _substance_categories_in_text(text: str) -> frozenset[str]:
    return frozenset(
        category
        for category in _SUBSTANCE_CATEGORIES
        if _substance_trigger_pattern(category).search(text) is not None
    )


def _has_local_substance_status(
    text: str,
    categories: Iterable[str],
) -> bool:
    if _SUBSTANCE_LOCAL_STATUS_RE.search(text) is not None:
        return True
    return any(
        _parse_substance_extent(category, text) is not None for category in categories
    )


def _extract_substance_category(
    text: str,
    category: str,
) -> list[SDOHFinding]:
    pattern = _substance_trigger_pattern(category)

    findings: list[SDOHFinding] = []
    seen_windows: set[SpanOffset] = set()

    for match in pattern.finditer(text):
        window = _substance_context_bounds(
            text,
            match.start(),
            match.end(),
        )
        if window in seen_windows:
            continue
        seen_windows.add(window)

        window_start, window_end = window
        context_text = text[window_start:window_end]
        target = {
            "text": match.group(0),
            "document_text": context_text,
            "start": match.start() - window_start,
            "end": match.end() - window_start,
        }

        negation = resolve_negation(target)
        temporality = resolve_temporality(target)

        status_text = context_text.strip()
        extent = _parse_substance_extent(
            category,
            status_text,
        )

        normalized_status = normalize_substance_status(
            status_text,
            negated=negation,
            temporality=temporality,
        )

        status = _SDOH_SUBSTANCE_STATUS[normalized_status]
        if status == "past":
            temporality = HISTORICAL

        if temporality == HYPOTHETICAL:
            status = "unknown"

        if (
            status == "unknown"
            and negation != NEGATED
            and temporality != HYPOTHETICAL
            and re.search(
                r"\b(?:occasional|occasionally)\b",
                status_text,
                re.IGNORECASE,
            )
        ):
            status = "current"
        if (
            status == "unknown"
            and extent is not None
            and negation != NEGATED
            and temporality != HYPOTHETICAL
        ):
            if temporality == HISTORICAL:
                status = "past"
            else:
                status = "current"

        if status == "none":
            extent = None

        findings.append(
            SDOHFinding(
                category=category,
                value=match.group(0),
                status=status,
                extent=extent,
                temporality=temporality,
                span=(match.start(), match.end()),
                score=1.0,
            )
        )
    return findings


@lru_cache(maxsize=None)
def _substance_trigger_pattern(category: str) -> re.Pattern[str]:
    cues = _load_substance_cues()[category]

    alternatives: list[str] = []

    for cue in sorted(cues, key=len, reverse=True):
        parts = cue.split()

        escaped = r"\s+".join(re.escape(part) for part in parts)

        prefix = r"(?<!\w)" if cue[0].isalnum() else ""
        suffix = r"(?!\w)" if cue[-1].isalnum() else ""

        alternatives.append(f"{prefix}(?:{escaped}){suffix}")

    return re.compile(
        "|".join(alternatives),
        re.IGNORECASE,
    )


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _span_offset(value: object, field_name: str) -> SpanOffset:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
    ):
        raise TypeError(f"{field_name} must be a two-item offset sequence")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{field_name} offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must satisfy 0 <= start < end")
    return start, end


def _social_history_ranges(
    text: str,
    sections: Iterable[Mapping[str, Any] | object],
) -> tuple[SpanOffset, ...]:
    ranges: list[SpanOffset] = []
    for section in sections:
        if _item_field(section, "label") != SOCIAL_HISTORY_SECTION:
            continue
        offset = _item_offset(section, "Social History section")
        if offset[1] > len(text):
            raise ValueError("Social History section is outside document bounds")
        ranges.append(offset)
    return tuple(sorted(ranges))


def _item_within_ranges(item: object, ranges: Sequence[SpanOffset]) -> bool:
    try:
        offset = _item_offset(item, "candidate span")
    except (TypeError, ValueError):
        return False
    return _offset_within_ranges(offset, ranges)


def _item_offset(item: object, field_name: str) -> SpanOffset:
    return _span_offset(
        (_item_field(item, "start"), _item_field(item, "end")),
        field_name,
    )


def _item_field(item: object, key: str) -> object:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _offset_within_ranges(
    offset: SpanOffset,
    ranges: Sequence[SpanOffset],
) -> bool:
    start, end = offset
    return any(
        range_start <= start and end <= range_end for range_start, range_end in ranges
    )


register_determinant_extractor(
    "employment", extract_employment_findings, languages=("en",)
)
register_determinant_extractor(
    "food_insecurity", extract_food_insecurity_findings, languages=("en",)
)
register_determinant_extractor(
    "living_status", extract_living_status_findings, languages=("en",)
)

register_determinant_extractor(
    "tobacco",
    _extract_tobacco,
    languages=("en",),
)

register_determinant_extractor(
    "alcohol",
    _extract_alcohol,
    languages=("en",),
)

register_determinant_extractor(
    "drug",
    _extract_drug,
    languages=("en",),
)


__all__ = [
    "FOOD_INSECURITY_EXTENSION_NOTE",
    "SHAC_DATA_POLICY",
    "SDOH_SOCIAL_CUES_RESOURCE",
    "SOCIAL_HISTORY_SECTION",
    "DeterminantExtractor",
    "DeterminantExtractorRegistry",
    "SDOHFinding",
    "SDOHExtractionResult",
    "available_determinant_extractors",
    "extract_employment_findings",
    "extract_food_insecurity_findings",
    "extract_living_status_findings",
    "extract_sdoh",
    "extract_sdoh_with_language",
    "load_sdoh_social_cues",
    "register_determinant_extractor",
    "unregister_determinant_extractor",
]
