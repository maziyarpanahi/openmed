"""Validated SNOMED CT post-coordination after pre-coordinated abstention.

The expression model implements the useful concept-reference subset of the
SNOMED CT compositional grammar: one focus concept and concept-valued,
optionally grouped refinements. Semantic validation is delegated to a
caller-supplied :class:`~openmed.clinical.grounding.ecl.ECLValidator`; no
SNOMED CT edition, terminology service, or credential is bundled.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol

from openmed.clinical.normalization.composite import (
    CompositeMention,
    CompositeNormalization,
    normalize_composite,
)

from .ecl import ECLValidationError, ECLValidator
from .types import Candidate, GroundedSpan
from .vocab import RestrictedVocabularyError

__all__ = [
    "LATERALITY_SITE_PATTERN",
    "LATERALITY_SITE_SLOTS",
    "POSTCOORDINATION_ATTRIBUTE_SLOTS",
    "POSTCOORDINATION_PROVENANCE_KEY",
    "ConceptReference",
    "LateralitySiteDecomposer",
    "MentionDecomposer",
    "PostCoordinationDecomposition",
    "PostCoordinationStage",
    "Refinement",
    "ResolvedRefinement",
    "RulesPostCoordinationDecomposer",
    "SnomedExpression",
    "build_expression",
    "decompose_laterality_site_mention",
    "decompose_mention",
    "is_postcoordinated_candidate",
]

POSTCOORDINATION_ATTRIBUTE_SLOTS = frozenset(
    {"laterality", "finding_site", "severity", "morphology", "causative_agent"}
)
POSTCOORDINATION_PROVENANCE_KEY = "snomed_postcoordination"
LATERALITY_SITE_SLOTS = frozenset({"laterality", "finding_site"})
LATERALITY_SITE_PATTERN = "laterality_finding_site"

_CONCEPT_ID_RE = re.compile(r"^[1-9][0-9]{5,17}$")
_DEFINITION_STATUSES = frozenset({"===", "<<<"})
_OF_CONNECTOR_RE = re.compile(r"\s+of\s+", flags=re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ConceptReference:
    """A SNOMED CT concept reference used in compositional grammar.

    ``term`` is optional because OpenMed must not supply terminology content.
    When present it came from the user's edition and is serialized between pipe
    delimiters as allowed by the grammar.
    """

    concept_id: str
    term: str | None = None

    def __post_init__(self) -> None:
        concept_id = str(self.concept_id).strip()
        if not _CONCEPT_ID_RE.fullmatch(concept_id):
            raise ValueError(
                "SNOMED concept_id must be a 6-18 digit identifier with no leading zero"
            )
        term = self.term
        if term is not None:
            if not isinstance(term, str) or not term.strip():
                raise ValueError("concept term must be non-empty when provided")
            term = " ".join(term.split())
            if "|" in term or any(ord(character) < 32 for character in term):
                raise ValueError("concept term contains a forbidden character")
        object.__setattr__(self, "concept_id", concept_id)
        object.__setattr__(self, "term", term)

    def to_scg(self) -> str:
        """Serialize this reference in SNOMED compositional grammar."""

        if self.term is None:
            return self.concept_id
        return f"{self.concept_id} |{self.term}|"

    def __str__(self) -> str:
        return self.to_scg()


@dataclass(frozen=True)
class Refinement:
    """One typed attribute-value refinement of a focus concept."""

    slot: str
    attribute: ConceptReference
    value: ConceptReference
    group: int | None = None

    def __post_init__(self) -> None:
        slot = _normalize_slot(self.slot)
        if slot not in POSTCOORDINATION_ATTRIBUTE_SLOTS:
            allowed = ", ".join(sorted(POSTCOORDINATION_ATTRIBUTE_SLOTS))
            raise ValueError(f"unsupported post-coordination slot; expected {allowed}")
        if not isinstance(self.attribute, ConceptReference):
            raise TypeError("refinement attribute must be a ConceptReference")
        if not isinstance(self.value, ConceptReference):
            raise TypeError("refinement value must be a ConceptReference")
        if self.group is not None and (type(self.group) is not int or self.group < 1):
            raise ValueError("refinement group must be a positive integer")
        object.__setattr__(self, "slot", slot)

    def to_scg(self) -> str:
        """Serialize the ungrouped attribute-value pair."""

        return f"{self.attribute.to_scg()} = {self.value.to_scg()}"


@dataclass(frozen=True)
class SnomedExpression:
    """One focus concept with deterministic SNOMED CT refinements."""

    focus: ConceptReference
    refinements: tuple[Refinement, ...]
    definition_status: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.focus, ConceptReference):
            raise TypeError("expression focus must be a ConceptReference")
        refinements = tuple(self.refinements)
        if not refinements:
            raise ValueError("post-coordinated expression requires a refinement")
        if any(not isinstance(item, Refinement) for item in refinements):
            raise TypeError("expression refinements must be Refinement objects")
        if self.definition_status not in {None, *_DEFINITION_STATUSES}:
            raise ValueError("definition_status must be '===', '<<<', or None")
        slots = [item.slot for item in refinements]
        if len(slots) != len(set(slots)):
            raise ValueError(
                "an expression may contain at most one refinement per slot"
            )
        ordered = tuple(sorted(refinements, key=_refinement_sort_key))
        object.__setattr__(self, "refinements", ordered)

    def to_scg(self) -> str:
        """Serialize to deterministic SNOMED CT compositional grammar."""

        ungrouped = [item for item in self.refinements if item.group is None]
        grouped: dict[int, list[Refinement]] = {}
        for refinement in self.refinements:
            if refinement.group is not None:
                grouped.setdefault(refinement.group, []).append(refinement)
        parts = [item.to_scg() for item in ungrouped]
        parts.extend(
            "{ " + ", ".join(item.to_scg() for item in grouped[group]) + " }"
            for group in sorted(grouped)
        )
        status = f"{self.definition_status} " if self.definition_status else ""
        return f"{status}{self.focus.to_scg()} : {', '.join(parts)}"

    def to_compositional_grammar(self) -> str:
        """Alias for :meth:`to_scg` using the specification's full name."""

        return self.to_scg()

    def __str__(self) -> str:
        return self.to_scg()


@dataclass(frozen=True)
class ResolvedRefinement:
    """A resolved modifier paired with its exact source fragment."""

    mention: CompositeMention
    refinement: Refinement


@dataclass(frozen=True)
class PostCoordinationDecomposition:
    """Focus and modifier spans recovered from a composite mention."""

    normalization: CompositeNormalization
    focus_mention: CompositeMention
    focus: ConceptReference
    refinements: tuple[ResolvedRefinement, ...]
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not self.refinements:
            raise ValueError("post-coordination decomposition requires a modifier")
        if not isinstance(self.focus, ConceptReference):
            raise TypeError("decomposition focus must be a ConceptReference")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("decomposition confidence must be between 0 and 1")

    @property
    def expression_refinements(self) -> tuple[Refinement, ...]:
        """Return resolved refinements without their source surfaces."""

        return tuple(item.refinement for item in self.refinements)


class MentionDecomposer(Protocol):
    """Resolve a mention into a focus and typed refinement concepts."""

    def decompose(
        self,
        mention: str,
        *,
        start: int = 0,
        byte_start: int | None = None,
    ) -> PostCoordinationDecomposition | None:
        """Return a decomposition, or ``None`` when the mention is unsupported."""


FocusResolver = Callable[[CompositeMention], ConceptReference | None]
RefinementResolver = Callable[[CompositeMention], Refinement | None]


@dataclass(frozen=True)
class RulesPostCoordinationDecomposer:
    """Adapt caller concept resolvers to the shared composite segmentation."""

    focus_resolver: FocusResolver
    refinement_resolver: RefinementResolver

    def decompose(
        self,
        mention: str,
        *,
        start: int = 0,
        byte_start: int | None = None,
    ) -> PostCoordinationDecomposition | None:
        """Decompose through :func:`normalize_composite` and resolve fragments."""

        return decompose_mention(
            mention,
            focus_resolver=self.focus_resolver,
            refinement_resolver=self.refinement_resolver,
            start=start,
            byte_start=byte_start,
        )


@dataclass(frozen=True)
class LateralitySiteDecomposer:
    """Resolve the focus + laterality + finding-site pattern only.

    The resolvers remain caller-owned so this class contains no SNOMED CT
    terminology. Mentions may use the existing composite segmentation (for
    example, three coordinated fragments) or the conservative
    ``"<focus> of <laterality> <site>"`` surface form.
    """

    focus_resolver: FocusResolver
    laterality_resolver: RefinementResolver
    finding_site_resolver: RefinementResolver

    def decompose(
        self,
        mention: str,
        *,
        start: int = 0,
        byte_start: int | None = None,
    ) -> PostCoordinationDecomposition | None:
        """Return an exact two-refinement decomposition, or ``None``."""

        return decompose_laterality_site_mention(
            mention,
            focus_resolver=self.focus_resolver,
            laterality_resolver=self.laterality_resolver,
            finding_site_resolver=self.finding_site_resolver,
            start=start,
            byte_start=byte_start,
        )


class PostCoordinationStage:
    """Compose only abstained or low-scoring pre-coordinated grounding.

    A non-empty user key is checked at construction and immediately discarded.
    The stage never reads environment credentials and never retains or emits the
    supplied key.
    """

    def __init__(
        self,
        *,
        license_key: str,
        validator: ECLValidator,
        decomposer: MentionDecomposer,
        precoordination_threshold: float = 0.75,
    ) -> None:
        if not isinstance(license_key, str) or not license_key.strip():
            raise RestrictedVocabularyError(
                "SNOMED post-coordination requires an explicit user-supplied "
                "license key and edition-backed ECL resolver."
            )
        if not isinstance(validator, ECLValidator):
            raise TypeError("validator must be an ECLValidator")
        if not callable(getattr(decomposer, "decompose", None)):
            raise TypeError("decomposer must implement decompose()")
        threshold = float(precoordination_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("precoordination_threshold must be between 0 and 1")
        self.validator = validator
        self.decomposer = decomposer
        self.precoordination_threshold = threshold

    def apply(self, grounded_span: GroundedSpan) -> GroundedSpan:
        """Prefer sufficient lookup results, otherwise attempt composition.

        Invalid or unsupported compositions are retained as explicit
        abstentions with structured rejection provenance; they never emit a
        SNOMED coding.
        """

        if not isinstance(grounded_span, GroundedSpan):
            raise TypeError("post-coordination expects a GroundedSpan")
        if self._prefer_precoordinated(grounded_span):
            return grounded_span
        byte_start = grounded_span.metadata.get("byte_start", grounded_span.start)
        try:
            decomposition = self.decomposer.decompose(
                grounded_span.text,
                start=grounded_span.start,
                byte_start=byte_start,
            )
        except Exception:
            return self._reject(grounded_span, ("decomposition_error",))
        if decomposition is None:
            return self._reject(grounded_span, ("unsupported_decomposition",))
        try:
            expression = build_expression(
                decomposition.focus,
                decomposition.expression_refinements,
                validator=self.validator,
            )
        except ECLValidationError as exc:
            return self._reject(
                grounded_span,
                tuple(issue.code for issue in exc.result.issues),
            )
        except (TypeError, ValueError):
            return self._reject(grounded_span, ("invalid_expression",))

        expression_text = expression.to_scg()
        candidate = Candidate(
            system="SNOMED",
            code=expression_text,
            display=grounded_span.text,
            score=float(decomposition.confidence),
            source_language=grounded_span.source_language,
            source="post-coordinated",
            match_kind="composed",
            vocab_version=self.validator.edition_uri,
        )
        provenance = dict(grounded_span.provenance)
        provenance[POSTCOORDINATION_PROVENANCE_KEY] = _composed_provenance(
            expression,
            edition_uri=self.validator.edition_uri,
        )
        return replace(
            grounded_span,
            candidates=(candidate,),
            calibrated_score=None,
            abstained=False,
            provenance=provenance,
        )

    def _prefer_precoordinated(self, grounded_span: GroundedSpan) -> bool:
        score = grounded_span.calibrated_score
        if score is None and grounded_span.candidates:
            score = max(float(item.score) for item in grounded_span.candidates)
        return bool(
            grounded_span.candidates
            and not grounded_span.abstained
            and score is not None
            and score >= self.precoordination_threshold
        )

    def _reject(
        self,
        grounded_span: GroundedSpan,
        reasons: tuple[str, ...],
    ) -> GroundedSpan:
        provenance = dict(grounded_span.provenance)
        provenance[POSTCOORDINATION_PROVENANCE_KEY] = {
            "status": "rejected",
            "edition_uri": self.validator.edition_uri,
            "reasons": list(dict.fromkeys(reasons)),
        }
        return replace(
            grounded_span,
            candidates=(),
            calibrated_score=None,
            abstained=True,
            provenance=provenance,
        )


def build_expression(
    focus: ConceptReference | str | Mapping[str, Any],
    refinements: Iterable[Refinement | Mapping[str, Any] | tuple[Any, ...]],
    *,
    validator: ECLValidator,
    definition_status: str | None = None,
) -> SnomedExpression:
    """Build and ECL-validate a post-coordinated expression.

    Validation is mandatory so a grammar-valid but semantically disallowed
    composition cannot accidentally be emitted.
    """

    if not isinstance(validator, ECLValidator):
        raise TypeError(
            "build_expression requires an edition-backed ECLValidator; "
            "SNOMED content is never bundled"
        )
    expression = SnomedExpression(
        focus=_coerce_concept(focus),
        refinements=tuple(_coerce_refinement(item) for item in refinements),
        definition_status=definition_status,
    )
    validator.require_valid(expression)
    return expression


def decompose_mention(
    mention: str,
    *,
    focus_resolver: FocusResolver,
    refinement_resolver: RefinementResolver,
    start: int = 0,
    byte_start: int | None = None,
) -> PostCoordinationDecomposition | None:
    """Reuse composite segmentation to resolve focus and modifier spans.

    Exactly one segmented fragment must resolve as the focus and every other
    fragment must resolve as a supported refinement. Unsupported shapes return
    ``None`` rather than guessing or dropping text.
    """

    normalization = normalize_composite(
        mention,
        start=start,
        byte_start=byte_start,
    )
    fragments = normalization.children
    if len(fragments) < 2:
        return None
    focus_matches = tuple(
        (fragment, concept)
        for fragment in fragments
        if (concept := focus_resolver(fragment)) is not None
    )
    if len(focus_matches) != 1:
        return None
    focus_mention, focus = focus_matches[0]
    resolved: list[ResolvedRefinement] = []
    for fragment in fragments:
        if fragment == focus_mention:
            continue
        refinement = refinement_resolver(fragment)
        if refinement is None:
            return None
        resolved.append(ResolvedRefinement(fragment, refinement))
    if not resolved:
        return None
    return PostCoordinationDecomposition(
        normalization=normalization,
        focus_mention=focus_mention,
        focus=focus,
        refinements=tuple(resolved),
    )


def decompose_laterality_site_mention(
    mention: str,
    *,
    focus_resolver: FocusResolver,
    laterality_resolver: RefinementResolver,
    finding_site_resolver: RefinementResolver,
    start: int = 0,
    byte_start: int | None = None,
) -> PostCoordinationDecomposition | None:
    """Resolve a focus finding with exactly laterality and finding site.

    Existing composite segmentation is tried first. If the mention is not a
    composite recognized by that shared normalizer, this function conservatively
    considers only ``"<focus> of <laterality> <site>"`` partitions. Caller
    resolvers must uniquely confirm all three spans; ambiguous or partial
    matches fail closed instead of dropping source text or guessing a concept.

    Args:
        mention: Exact source mention surface.
        focus_resolver: Caller-owned resolver for the focus finding span.
        laterality_resolver: Caller-owned resolver returning a ``laterality``
            refinement.
        finding_site_resolver: Caller-owned resolver returning a
            ``finding_site`` refinement.
        start: Character-coordinate base in the source document.
        byte_start: UTF-8 byte-coordinate base; defaults to ``start``.

    Returns:
        A unique exact-pattern decomposition, or ``None`` when unsupported or
        ambiguous.
    """

    normalization = normalize_composite(
        mention,
        start=start,
        byte_start=byte_start,
    )
    fragment_sets: list[tuple[CompositeMention, ...]] = []
    if len(normalization.children) == 3:
        fragment_sets.append(normalization.children)
    if not normalization.was_split:
        fragment_sets.extend(
            _laterality_site_phrase_partitions(
                mention,
                start=start,
                byte_start=start if byte_start is None else byte_start,
            )
        )

    matches = tuple(
        decomposition
        for fragments in fragment_sets
        if (
            decomposition := _resolve_laterality_site_fragments(
                normalization,
                fragments,
                focus_resolver=focus_resolver,
                laterality_resolver=laterality_resolver,
                finding_site_resolver=finding_site_resolver,
            )
        )
        is not None
    )
    if len(matches) != 1:
        return None
    return matches[0]


def is_postcoordinated_candidate(candidate: Candidate) -> bool:
    """Return whether a grounding candidate was composed rather than looked up."""

    return (
        candidate.system.upper() == "SNOMED"
        and candidate.source == "post-coordinated"
        and candidate.match_kind == "composed"
    )


def _normalize_slot(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("refinement slot must be text")
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _laterality_site_phrase_partitions(
    mention: str,
    *,
    start: int,
    byte_start: int,
) -> tuple[tuple[CompositeMention, ...], ...]:
    partitions: list[tuple[CompositeMention, ...]] = []
    for connector in _OF_CONNECTOR_RE.finditer(mention):
        focus_range = _trim_local_range(mention, 0, connector.start())
        tail_start, tail_end = _trim_local_range(mention, connector.end(), len(mention))
        if focus_range[0] == focus_range[1] or tail_start == tail_end:
            continue
        for separator in _WHITESPACE_RE.finditer(mention, tail_start, tail_end):
            laterality_range = _trim_local_range(
                mention,
                tail_start,
                separator.start(),
            )
            site_range = _trim_local_range(
                mention,
                separator.end(),
                tail_end,
            )
            if (
                laterality_range[0] == laterality_range[1]
                or site_range[0] == site_range[1]
            ):
                continue
            partitions.append(
                tuple(
                    _postcoordination_span(
                        mention,
                        local_start,
                        local_end,
                        char_base=start,
                        byte_base=byte_start,
                    )
                    for local_start, local_end in (
                        focus_range,
                        laterality_range,
                        site_range,
                    )
                )
            )
    return tuple(partitions)


def _resolve_laterality_site_fragments(
    normalization: CompositeNormalization,
    fragments: tuple[CompositeMention, ...],
    *,
    focus_resolver: FocusResolver,
    laterality_resolver: RefinementResolver,
    finding_site_resolver: RefinementResolver,
) -> PostCoordinationDecomposition | None:
    focus_matches = tuple(
        (fragment, concept)
        for fragment in fragments
        if (concept := focus_resolver(fragment)) is not None
    )
    laterality_matches = tuple(
        (fragment, refinement)
        for fragment in fragments
        if (refinement := laterality_resolver(fragment)) is not None
        and refinement.slot == "laterality"
    )
    site_matches = tuple(
        (fragment, refinement)
        for fragment in fragments
        if (refinement := finding_site_resolver(fragment)) is not None
        and refinement.slot == "finding_site"
    )
    if not all(
        len(matches) == 1
        for matches in (focus_matches, laterality_matches, site_matches)
    ):
        return None
    focus_mention, focus = focus_matches[0]
    laterality_mention, laterality = laterality_matches[0]
    site_mention, site = site_matches[0]
    if len({focus_mention, laterality_mention, site_mention}) != 3:
        return None
    return PostCoordinationDecomposition(
        normalization=normalization,
        focus_mention=focus_mention,
        focus=focus,
        refinements=(
            ResolvedRefinement(laterality_mention, laterality),
            ResolvedRefinement(site_mention, site),
        ),
    )


def _trim_local_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _postcoordination_span(
    mention: str,
    local_start: int,
    local_end: int,
    *,
    char_base: int,
    byte_base: int,
) -> CompositeMention:
    text = mention[local_start:local_end]
    local_byte_start = len(mention[:local_start].encode("utf-8"))
    local_byte_end = local_byte_start + len(text.encode("utf-8"))
    return CompositeMention(
        text=text,
        start=char_base + local_start,
        end=char_base + local_end,
        byte_start=byte_base + local_byte_start,
        byte_end=byte_base + local_byte_end,
    )


def _coerce_concept(
    value: ConceptReference | str | Mapping[str, Any],
) -> ConceptReference:
    if isinstance(value, ConceptReference):
        return value
    if isinstance(value, str):
        return ConceptReference(value)
    if isinstance(value, Mapping):
        concept_id = value.get("concept_id", value.get("code"))
        if concept_id is None:
            raise ValueError("concept mapping requires concept_id or code")
        return ConceptReference(
            str(concept_id), value.get("term", value.get("display"))
        )
    raise TypeError("concept must be a ConceptReference, identifier, or mapping")


def _coerce_refinement(
    value: Refinement | Mapping[str, Any] | tuple[Any, ...],
) -> Refinement:
    if isinstance(value, Refinement):
        return value
    if isinstance(value, Mapping):
        return Refinement(
            slot=str(value.get("slot", "")),
            attribute=_coerce_concept(value.get("attribute")),
            value=_coerce_concept(value.get("value")),
            group=value.get("group"),
        )
    if isinstance(value, tuple) and len(value) in {3, 4}:
        return Refinement(
            slot=str(value[0]),
            attribute=_coerce_concept(value[1]),
            value=_coerce_concept(value[2]),
            group=value[3] if len(value) == 4 else None,
        )
    raise TypeError("refinement must be a Refinement, mapping, or 3/4-tuple")


def _refinement_sort_key(refinement: Refinement) -> tuple[int, int, str, str, str]:
    return (
        1 if refinement.group is not None else 0,
        refinement.group or 0,
        refinement.slot,
        refinement.attribute.concept_id,
        refinement.value.concept_id,
    )


def _composed_provenance(
    expression: SnomedExpression,
    *,
    edition_uri: str,
) -> dict[str, Any]:
    expression_text = expression.to_scg()
    slots = frozenset(item.slot for item in expression.refinements)
    return {
        "status": "composed",
        "method": "post-coordinated",
        "pattern": (
            LATERALITY_SITE_PATTERN if slots == LATERALITY_SITE_SLOTS else "general"
        ),
        "edition_uri": edition_uri,
        "expression_sha256": hashlib.sha256(
            expression_text.encode("utf-8")
        ).hexdigest(),
        "focus_concept_id": expression.focus.concept_id,
        "refinements": [
            {
                "slot": item.slot,
                "attribute_id": item.attribute.concept_id,
                "value_id": item.value.concept_id,
                "group": item.group,
            }
            for item in expression.refinements
        ],
        "validated": True,
    }
