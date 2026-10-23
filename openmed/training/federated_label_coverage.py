"""Per-site label coverage declarations and loss masks for federated training.

Sites in a federated named-entity round often annotate different subsets of
the round's labels. When a site trains on its own BIO tags, an entity it never
annotates appears as an outside (``O``) token, so a naive token loss teaches
the shared model to suppress that entity. This module provides:

* a content-free coverage declaration listing the canonical labels a site
  annotates, its annotation-guideline digest and its text form;
* validation of declarations against a coordinator-owned round label schema;
* a framework-neutral loss-mask builder that removes uncovered labels from
  the negative class;
* a coordinator gate that blocks rounds where a critical direct-identifier
  label has no annotating site, and band-only coverage evidence.

Mask semantics: for every token the mask lists the *candidate* tag classes,
meaning the tags the token could truly carry given the site's coverage. A
token tagged ``B-X`` or ``I-X`` for a covered label ``X`` has exactly that
candidate. A token tagged ``O`` has ``O`` plus ``B-Y`` and ``I-Y`` for every
round label ``Y`` the site does not annotate, because an unannotated ``Y``
entity is indistinguishable from a true outside token at that site. Classes
outside the candidate set are definite negatives and stay in the loss; the
candidate classes of an ambiguous token are never trained as negatives.

No function here reads text, tensors or files. Errors carry a stable reason
code and a field name only.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Final, NoReturn, Protocol, Sequence

from openmed.core.labels import (
    CANONICAL_LABELS,
    DIRECT_IDENTIFIER,
    LABEL_METADATA,
    OTHER,
)

from .federated_status import DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE

FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION = "openmed.training.federated_label_coverage.v1"
FEDERATED_LABEL_COVERAGE_EVIDENCE_SCHEMA_VERSION = (
    "openmed.training.federated_label_coverage_evidence.v1"
)
OUTSIDE_TAG: Final = "O"

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_DECLARATION_FIELDS: Final = frozenset(
    {"schema_version", "labels", "guideline_digest", "text_form"}
)
_MAX_JSON_CHARS: Final = 64 * 1024
_SCHEMA_DIGEST_DOMAIN: Final = b"openmed.training.federated_label_schema.v1\x00"
_COVERABLE_LABELS: Final = frozenset(CANONICAL_LABELS - {OTHER})
_EVIDENCE_BUILDER_TOKEN: Final = object()


class FederatedTextForm(str, Enum):
    """Whether a site trains on raw text or on surrogate-substituted text."""

    RAW = "raw"
    SURROGATE = "surrogate"

    def __str__(self) -> str:
        return self.value


class FederatedLabelCoverageReasonCode(str, Enum):
    """Stable reason codes for label coverage failures."""

    INVALID_DECLARATION = "invalid_declaration"
    UNSUPPORTED_SCHEMA_VERSION = "unsupported_schema_version"
    INVALID_GUIDELINE_DIGEST = "invalid_guideline_digest"
    INVALID_TEXT_FORM = "invalid_text_form"
    UNKNOWN_LABEL = "unknown_label"
    DUPLICATE_LABEL = "duplicate_label"
    EMPTY_COVERAGE = "empty_coverage"
    INVALID_ROUND_SCHEMA = "invalid_round_schema"
    LABEL_NOT_IN_ROUND_SCHEMA = "label_not_in_round_schema"
    GUIDELINE_VERSION_MISMATCH = "guideline_version_mismatch"
    TEXT_FORM_NOT_ALLOWED = "text_form_not_allowed"
    INVALID_TAG = "invalid_tag"
    UNDECLARED_TAG_LABEL = "undeclared_tag_label"
    INVALID_MASK = "invalid_mask"
    NO_DECLARATIONS = "no_declarations"
    CRITICAL_LABEL_UNCOVERED = "critical_label_uncovered"
    INVALID_MINIMUM_GROUP_SIZE = "invalid_minimum_group_size"
    INVALID_EVIDENCE = "invalid_evidence"
    TRAINING_PORT_FAILED = "training_port_failed"

    def __str__(self) -> str:
        return self.value


class FederatedLabelCoverageError(ValueError):
    """Raised with a stable reason code and a field name, never input values.

    Args:
        reason_code: Stable machine-readable failure category.
        field: Name of the offending field or input.
    """

    def __init__(self, reason_code: FederatedLabelCoverageReasonCode, field: str):
        self.reason_code = reason_code
        self.field = field
        super().__init__(f"{reason_code.value}: {field}")


class FederatedLabelCoverageBand(str, Enum):
    """Coarse count of sites annotating one label, relative to the minimum."""

    NONE = "none"
    SUPPRESSED = "suppressed"
    MINIMUM_TO_UNDER_DOUBLE = "minimum_to_under_double"
    DOUBLE_TO_UNDER_FOURFOLD = "double_to_under_fourfold"
    FOURFOLD_OR_MORE = "fourfold_or_more"

    def __str__(self) -> str:
        return self.value


_R = FederatedLabelCoverageReasonCode


def _fail(reason: FederatedLabelCoverageReasonCode, field: str) -> NoReturn:
    raise FederatedLabelCoverageError(reason, field)


@dataclass(frozen=True, slots=True)
class FederatedRoundLabelSchema:
    """Coordinator-owned label schema that site declarations are checked against.

    Critical labels are every catalog direct identifier in ``labels`` plus any
    ``additional_critical_labels``. Configuration can add critical labels but
    can never downgrade a catalog direct identifier.

    Args:
        labels: Canonical labels trained in the round, as a non-empty tuple.
            Stored sorted. ``OTHER`` is not a coverable label.
        guideline_digest: ``sha256:`` reference to the annotation guideline
            version every site must follow.
        allowed_text_forms: Non-empty tuple of accepted text forms.
        additional_critical_labels: Extra labels from ``labels`` that must
            have at least one annotating site.
    """

    labels: tuple[str, ...]
    guideline_digest: str
    allowed_text_forms: tuple[FederatedTextForm, ...] = (
        FederatedTextForm.RAW,
        FederatedTextForm.SURROGATE,
    )
    additional_critical_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.labels) is not tuple:
            _fail(_R.INVALID_ROUND_SCHEMA, "labels")
        labels = _canonical_labels(self.labels, "labels")
        _require_digest(self.guideline_digest)
        forms = self.allowed_text_forms
        if (
            type(forms) is not tuple
            or not forms
            or any(type(form) is not FederatedTextForm for form in forms)
        ):
            _fail(_R.INVALID_TEXT_FORM, "allowed_text_forms")
        extra = self.additional_critical_labels
        if type(extra) is not tuple:
            _fail(_R.INVALID_ROUND_SCHEMA, "additional_critical_labels")
        extra_labels = (
            _canonical_labels(extra, "additional_critical_labels") if extra else ()
        )
        if not set(extra_labels) <= set(labels):
            _fail(_R.LABEL_NOT_IN_ROUND_SCHEMA, "additional_critical_labels")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(
            self,
            "allowed_text_forms",
            tuple(sorted(set(forms), key=lambda form: form.value)),
        )
        object.__setattr__(self, "additional_critical_labels", extra_labels)

    @property
    def critical_labels(self) -> tuple[str, ...]:
        """Return sorted labels that must have at least one annotating site."""
        catalog = {
            label
            for label in self.labels
            if LABEL_METADATA[label]["policy_label"] == DIRECT_IDENTIFIER
        }
        return tuple(sorted(catalog | set(self.additional_critical_labels)))

    @property
    def tag_classes(self) -> tuple[str, ...]:
        """Return the BIO class vocabulary: ``O``, then ``B-``/``I-`` per label."""
        classes = [OUTSIDE_TAG]
        for label in self.labels:
            classes.extend((f"B-{label}", f"I-{label}"))
        return tuple(classes)

    @property
    def digest(self) -> str:
        """Return a domain-separated SHA-256 reference to this schema."""
        canonical = json.dumps(
            {
                "allowed_text_forms": [form.value for form in self.allowed_text_forms],
                "critical_labels": list(self.critical_labels),
                "guideline_digest": self.guideline_digest,
                "labels": list(self.labels),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(_SCHEMA_DIGEST_DOMAIN + canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class FederatedLabelCoverageDeclaration:
    """Content-free statement of which canonical labels a site annotates.

    A declaration has no site identity, example text or counts. It lists the
    labels whose entities the site annotated completely under one guideline
    version; every other round label is treated as unannotated at that site.

    Args:
        labels: Non-empty tuple of exact canonical labels. Stored sorted.
        guideline_digest: ``sha256:`` reference to the guideline version.
        text_form: Whether the site trains on raw or surrogate text.
        schema_version: Exact supported declaration schema identifier.
    """

    labels: tuple[str, ...]
    guideline_digest: str
    text_form: FederatedTextForm
    schema_version: str = FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not str
            or self.schema_version != FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION
        ):
            _fail(_R.UNSUPPORTED_SCHEMA_VERSION, "schema_version")
        if type(self.labels) is not tuple:
            _fail(_R.INVALID_DECLARATION, "labels")
        object.__setattr__(self, "labels", _canonical_labels(self.labels, "labels"))
        _require_digest(self.guideline_digest)
        if type(self.text_form) is not FederatedTextForm:
            _fail(_R.INVALID_TEXT_FORM, "text_form")

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-style mapping with labels in canonical order."""
        return {
            "schema_version": self.schema_version,
            "labels": list(self.labels),
            "guideline_digest": self.guideline_digest,
            "text_form": self.text_form.value,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> FederatedLabelCoverageDeclaration:
        """Validate a JSON-style dictionary with exactly the declaration fields.

        Args:
            payload: Built-in dictionary with ``schema_version``, ``labels``
                (a list of strings), ``guideline_digest`` and ``text_form``.
                Any other field, such as a site name, is rejected.

        Returns:
            The validated immutable declaration.

        Raises:
            FederatedLabelCoverageError: With a stable reason code.
        """
        if (
            type(payload) is not dict
            or any(type(key) is not str for key in payload)
            or payload.keys() != _DECLARATION_FIELDS
        ):
            _fail(_R.INVALID_DECLARATION, "fields")
        labels = payload["labels"]
        if type(labels) is not list:
            _fail(_R.INVALID_DECLARATION, "labels")
        text_form = payload["text_form"]
        if type(text_form) is not str or text_form not in _TEXT_FORM_VALUES:
            _fail(_R.INVALID_TEXT_FORM, "text_form")
        return cls(
            labels=tuple(labels),
            guideline_digest=payload["guideline_digest"],
            text_form=FederatedTextForm(text_form),
            schema_version=payload["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: str) -> FederatedLabelCoverageDeclaration:
        """Parse bounded JSON, rejecting duplicate keys and non-string input.

        Args:
            payload: JSON text of at most 64 KiB.

        Returns:
            The validated immutable declaration.

        Raises:
            FederatedLabelCoverageError: With a stable reason code; parser
                excerpts are never included.
        """
        if type(payload) is not str or len(payload) > _MAX_JSON_CHARS:
            _fail(_R.INVALID_DECLARATION, "json")
        try:
            decoded = json.loads(payload, object_pairs_hook=_strict_object)
        except (ValueError, RecursionError):
            raise FederatedLabelCoverageError(_R.INVALID_DECLARATION, "json") from None
        return cls.from_dict(decoded)


_TEXT_FORM_VALUES: Final = frozenset(form.value for form in FederatedTextForm)


def validate_label_coverage_declaration(
    declaration: object, *, schema: FederatedRoundLabelSchema
) -> FederatedLabelCoverageDeclaration:
    """Check a site declaration against the coordinator's round label schema.

    Args:
        declaration: A :class:`FederatedLabelCoverageDeclaration`.
        schema: Trusted round label schema; never derive it from declarations.

    Returns:
        The same declaration when it is admissible for the round.

    Raises:
        FederatedLabelCoverageError: ``label_not_in_round_schema``,
            ``guideline_version_mismatch`` or ``text_form_not_allowed``.
    """
    if type(schema) is not FederatedRoundLabelSchema:
        _fail(_R.INVALID_ROUND_SCHEMA, "schema")
    if type(declaration) is not FederatedLabelCoverageDeclaration:
        _fail(_R.INVALID_DECLARATION, "declaration")
    if not set(declaration.labels) <= set(schema.labels):
        _fail(_R.LABEL_NOT_IN_ROUND_SCHEMA, "labels")
    if declaration.guideline_digest != schema.guideline_digest:
        _fail(_R.GUIDELINE_VERSION_MISMATCH, "guideline_digest")
    if declaration.text_form not in schema.allowed_text_forms:
        _fail(_R.TEXT_FORM_NOT_ALLOWED, "text_form")
    return declaration


@dataclass(frozen=True, slots=True)
class FederatedTokenLossMask:
    """Per-token candidate tag classes for one BIO sequence.

    Args:
        classes: Tag vocabulary, as returned by
            :attr:`FederatedRoundLabelSchema.tag_classes`.
        candidates: One sorted, non-empty tuple of class indices per token,
            listing the tags the token could truly carry.

    Apply the mask in either loss form:

    * Softmax cross-entropy: use ``-log(sum(p[c] for candidate c))``. For a
      single candidate this is ordinary cross-entropy.
    * Per-class binary (sigmoid) loss: train ``positive_targets()`` as
      positives and ``negative_matrix()`` entries as negatives; skip the
      remaining candidate classes of ambiguous tokens.
    """

    classes: tuple[str, ...]
    candidates: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        classes = self.classes
        if (
            type(classes) is not tuple
            or not classes
            or classes[0] != OUTSIDE_TAG
            or any(type(name) is not str for name in classes)
            or len(set(classes)) != len(classes)
        ):
            _fail(_R.INVALID_MASK, "classes")
        width = len(classes)
        if type(self.candidates) is not tuple:
            _fail(_R.INVALID_MASK, "candidates")
        for row in self.candidates:
            if (
                type(row) is not tuple
                or not row
                or any(type(index) is not int for index in row)
                or list(row) != sorted(set(row))
                or row[0] < 0
                or row[-1] >= width
            ):
                _fail(_R.INVALID_MASK, "candidates")

    def __len__(self) -> int:
        return len(self.candidates)

    def candidate_matrix(self) -> tuple[tuple[bool, ...], ...]:
        """Return per-token, per-class flags; True marks a possible true tag."""
        width = len(self.classes)
        return tuple(
            tuple(index in row for index in range(width)) for row in self.candidates
        )

    def negative_matrix(self) -> tuple[tuple[bool, ...], ...]:
        """Return per-token, per-class flags; True marks a definite negative."""
        width = len(self.classes)
        return tuple(
            tuple(index not in row for index in range(width)) for row in self.candidates
        )

    def positive_targets(self) -> tuple[int | None, ...]:
        """Return the definite class per token, or None when it is ambiguous."""
        return tuple(row[0] if len(row) == 1 else None for row in self.candidates)


def build_label_loss_mask(
    tags: Sequence[str],
    *,
    declaration: FederatedLabelCoverageDeclaration,
    schema: FederatedRoundLabelSchema,
) -> FederatedTokenLossMask:
    """Build the loss mask for one site-annotated BIO tag sequence.

    Args:
        tags: The site's observed tags: ``O``, ``B-<LABEL>`` or ``I-<LABEL>``
            with exact canonical labels the site declared.
        declaration: The site's coverage declaration.
        schema: Trusted round label schema.

    Returns:
        A mask whose ``O`` tokens keep the site's uncovered labels as
        candidates, so they are never trained as negatives for those labels.

    Raises:
        FederatedLabelCoverageError: For an inadmissible declaration
            or a tag that is malformed (``invalid_tag``), unknown
            (``unknown_label``), outside the round (``label_not_in_round_schema``)
            or not declared by the site (``undeclared_tag_label``).
    """
    return build_label_loss_masks((tags,), declaration=declaration, schema=schema)[0]


def build_label_loss_masks(
    tag_sequences: Sequence[Sequence[str]],
    *,
    declaration: FederatedLabelCoverageDeclaration,
    schema: FederatedRoundLabelSchema,
) -> tuple[FederatedTokenLossMask, ...]:
    """Build loss masks for a batch, refusing the batch on any invalid sequence.

    Args:
        tag_sequences: Sequence of BIO tag sequences from one site.
        declaration: The site's coverage declaration.
        schema: Trusted round label schema.

    Returns:
        One :class:`FederatedTokenLossMask` per input sequence, in order.

    Raises:
        FederatedLabelCoverageError: See :func:`build_label_loss_mask`.
    """
    validate_label_coverage_declaration(declaration, schema=schema)
    if isinstance(tag_sequences, (str, bytes)) or not isinstance(
        tag_sequences, Sequence
    ):
        _fail(_R.INVALID_TAG, "tag_sequences")
    classes = schema.tag_classes
    index = {name: position for position, name in enumerate(classes)}
    covered = set(declaration.labels)
    uncovered = tuple(label for label in schema.labels if label not in covered)
    outside = tuple(
        sorted(
            [index[OUTSIDE_TAG]]
            + [index[f"{prefix}-{label}"] for label in uncovered for prefix in "BI"]
        )
    )
    observed = {OUTSIDE_TAG: outside}
    for label in declaration.labels:
        for prefix in "BI":
            tag = f"{prefix}-{label}"
            observed[tag] = (index[tag],)

    rows: list[tuple[tuple[int, ...], ...]] = []
    for tags in tag_sequences:
        if isinstance(tags, (str, bytes)) or not isinstance(tags, Sequence):
            _fail(_R.INVALID_TAG, "tags")
        sequence = []
        for tag in tags:
            candidate = observed.get(tag) if type(tag) is str else None
            if candidate is None:
                _classify_tag_failure(tag, schema)
            sequence.append(candidate)
        rows.append(tuple(sequence))  # type: ignore[arg-type]
    return tuple(
        FederatedTokenLossMask(classes=classes, candidates=row) for row in rows
    )


def _classify_tag_failure(tag: object, schema: FederatedRoundLabelSchema) -> NoReturn:
    if type(tag) is not str or tag[:2] not in ("B-", "I-"):
        _fail(_R.INVALID_TAG, "tags")
    label = tag[2:]
    if label not in _COVERABLE_LABELS:
        _fail(_R.UNKNOWN_LABEL, "tags")
    if label not in schema.labels:
        _fail(_R.LABEL_NOT_IN_ROUND_SCHEMA, "tags")
    _fail(_R.UNDECLARED_TAG_LABEL, "tags")


class FederatedMaskedTrainingPort(Protocol):
    """Site training port that must apply coverage loss masks.

    Implementations receive one mask per local sequence, aligned by position
    with the port's own token batch, and must restrict their token loss as
    described on :class:`FederatedTokenLossMask`.
    """

    def train(self, masks: tuple[FederatedTokenLossMask, ...]) -> object:
        """Run one local update that applies ``masks`` to the token loss."""
        ...


def run_masked_site_training(
    port: FederatedMaskedTrainingPort,
    tag_sequences: Sequence[Sequence[str]],
    *,
    declaration: FederatedLabelCoverageDeclaration,
    schema: FederatedRoundLabelSchema,
) -> object:
    """Validate local coverage inputs, then call the port with loss masks.

    The declaration and every sequence are checked before the port is called.

    Args:
        port: Site training port that applies the masks.
        tag_sequences: The site's observed BIO tag sequences.
        declaration: The site's coverage declaration.
        schema: Trusted round label schema.

    Returns:
        Whatever the port returns.

    Raises:
        FederatedLabelCoverageError: For invalid inputs, or
            ``training_port_failed`` without the port's exception detail.
    """
    masks = build_label_loss_masks(
        tag_sequences, declaration=declaration, schema=schema
    )
    try:
        return port.train(masks)
    except Exception:
        raise FederatedLabelCoverageError(_R.TRAINING_PORT_FAILED, "port") from None


@dataclass(frozen=True, slots=True)
class FederatedLabelCoverageEntry:
    """Band-only coverage for one round label.

    Args:
        label: Canonical round label.
        critical: Whether the round requires an annotating site.
        band: Coarse count of annotating sites.
    """

    label: str
    critical: bool
    band: FederatedLabelCoverageBand

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping with the band value."""
        return {"band": self.band.value, "critical": self.critical, "label": self.label}


@dataclass(frozen=True)
class FederatedLabelCoverageEvidence:
    """Round evidence from :func:`build_label_coverage_evidence`.

    Holds per-label coverage bands and the gate decision only; exact site
    counts and declarations are discarded.
    """

    label_schema_digest: str
    guideline_digest: str
    minimum_group_size: int
    entries: tuple[FederatedLabelCoverageEntry, ...]
    admitted: bool
    reason_code: FederatedLabelCoverageReasonCode | None
    schema_version: str = FEDERATED_LABEL_COVERAGE_EVIDENCE_SCHEMA_VERSION
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _EVIDENCE_BUILDER_TOKEN:
            _fail(_R.INVALID_EVIDENCE, "builder")

    def require_admitted(self) -> FederatedLabelCoverageEvidence:
        """Return this evidence, or raise when the round is blocked.

        Raises:
            FederatedLabelCoverageError: ``critical_label_uncovered``.
        """
        if not self.admitted:
            _fail(_R.CRITICAL_LABEL_UNCOVERED, "labels")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a privacy-safe deterministic mapping."""
        return {
            "admitted": self.admitted,
            "entries": [entry.to_dict() for entry in self.entries],
            "guideline_digest": self.guideline_digest,
            "label_schema_digest": self.label_schema_digest,
            "minimum_group_size": self.minimum_group_size,
            "reason_code": (
                self.reason_code.value if self.reason_code is not None else None
            ),
            "schema_version": self.schema_version,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def build_label_coverage_evidence(
    declarations: Sequence[FederatedLabelCoverageDeclaration],
    *,
    schema: FederatedRoundLabelSchema,
    minimum_group_size: int = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE,
) -> FederatedLabelCoverageEvidence:
    """Summarize round coverage as bands and decide whether the round may run.

    Every declaration is validated first; one inadmissible declaration
    refuses the whole round. The round is blocked with
    ``critical_label_uncovered`` when a critical label has no annotating site.

    Args:
        declarations: All participating sites' declarations, in any order.
        schema: Trusted round label schema.
        minimum_group_size: Counts below this value are reported only as
            ``suppressed``; must be an integer of at least two.

    Returns:
        Order-independent evidence with one band entry per round label.

    Raises:
        FederatedLabelCoverageError: For invalid inputs.
    """
    if type(minimum_group_size) is not int or minimum_group_size < 2:
        _fail(_R.INVALID_MINIMUM_GROUP_SIZE, "minimum_group_size")
    if isinstance(declarations, (str, bytes)) or not isinstance(declarations, Sequence):
        _fail(_R.INVALID_DECLARATION, "declarations")
    if not declarations:
        _fail(_R.NO_DECLARATIONS, "declarations")
    for declaration in declarations:
        validate_label_coverage_declaration(declaration, schema=schema)
    critical = set(schema.critical_labels)
    entries = []
    uncovered_critical = False
    for label in schema.labels:
        count = sum(label in declaration.labels for declaration in declarations)
        uncovered_critical |= label in critical and count == 0
        entries.append(
            FederatedLabelCoverageEntry(
                label=label,
                critical=label in critical,
                band=_coverage_band(count, minimum_group_size),
            )
        )
    return FederatedLabelCoverageEvidence(
        label_schema_digest=schema.digest,
        guideline_digest=schema.guideline_digest,
        minimum_group_size=minimum_group_size,
        entries=tuple(entries),
        admitted=not uncovered_critical,
        reason_code=_R.CRITICAL_LABEL_UNCOVERED if uncovered_critical else None,
        _builder_token=_EVIDENCE_BUILDER_TOKEN,
    )


def check_round_label_coverage(
    declarations: Sequence[FederatedLabelCoverageDeclaration],
    *,
    schema: FederatedRoundLabelSchema,
    minimum_group_size: int = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE,
) -> FederatedLabelCoverageEvidence:
    """Build coverage evidence and block the round when it is not admitted.

    Args:
        declarations: All participating sites' declarations.
        schema: Trusted round label schema.
        minimum_group_size: Suppression threshold for coverage bands.

    Returns:
        Evidence for an admitted round.

    Raises:
        FederatedLabelCoverageError: ``critical_label_uncovered`` when a
            critical label has no annotating site, or any validation code.
    """
    return build_label_coverage_evidence(
        declarations, schema=schema, minimum_group_size=minimum_group_size
    ).require_admitted()


def _coverage_band(count: int, minimum: int) -> FederatedLabelCoverageBand:
    if count == 0:
        return FederatedLabelCoverageBand.NONE
    if count < minimum:
        return FederatedLabelCoverageBand.SUPPRESSED
    if count < 2 * minimum:
        return FederatedLabelCoverageBand.MINIMUM_TO_UNDER_DOUBLE
    if count < 4 * minimum:
        return FederatedLabelCoverageBand.DOUBLE_TO_UNDER_FOURFOLD
    return FederatedLabelCoverageBand.FOURFOLD_OR_MORE


def _canonical_labels(labels: tuple[object, ...], field: str) -> tuple[str, ...]:
    if not labels:
        _fail(_R.EMPTY_COVERAGE, field)
    if any(
        type(label) is not str or label not in _COVERABLE_LABELS for label in labels
    ):
        _fail(_R.UNKNOWN_LABEL, field)
    if len(set(labels)) != len(labels):
        _fail(_R.DUPLICATE_LABEL, field)
    return tuple(sorted(labels))  # type: ignore[arg-type]


def _require_digest(value: object) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        _fail(_R.INVALID_GUIDELINE_DIGEST, "guideline_digest")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(_R.INVALID_DECLARATION, "fields")
        result[key] = value
    return result


__all__ = [
    "FEDERATED_LABEL_COVERAGE_EVIDENCE_SCHEMA_VERSION",
    "FEDERATED_LABEL_COVERAGE_SCHEMA_VERSION",
    "OUTSIDE_TAG",
    "FederatedLabelCoverageBand",
    "FederatedLabelCoverageDeclaration",
    "FederatedLabelCoverageEntry",
    "FederatedLabelCoverageError",
    "FederatedLabelCoverageEvidence",
    "FederatedLabelCoverageReasonCode",
    "FederatedMaskedTrainingPort",
    "FederatedRoundLabelSchema",
    "FederatedTextForm",
    "FederatedTokenLossMask",
    "build_label_coverage_evidence",
    "build_label_loss_mask",
    "build_label_loss_masks",
    "check_round_label_coverage",
    "run_masked_site_training",
    "validate_label_coverage_declaration",
]
