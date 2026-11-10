"""Bounded modality validation packs and artifact-bound blinded review.

Review evidence is caller attested, not remote credential verification. Public
synthetic controls are not independent clinical validation. Protected packets
contain source material; aggregate reports must not contain that material.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from openmed.eval.governance.blinded_adjudication import (
    BlindedAdjudicationPacket,
    ComparisonCase,
    ConflictOfInterestMetadata,
    RubricCriterion,
    SealedIdentityMapping,
    SourceEvidence,
    render_blinded_adjudication_packets,
    verify_sealed_identity_mapping,
)
from openmed.eval.reviewer_disagreement import (
    DisagreementReason,
    ReviewerDecision,
    reviewer_disagreement_report,
)
from openmed.eval.synthetic_fidelity import FidelityCounts, wilson_interval

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_CRITERIA = ("quality", "utility", "privacy", "provenance")


class ClinicalValidationError(ValueError):
    """Fixed refusal without protected values or credential information."""


def _require(condition: bool) -> None:
    if not condition:
        raise ClinicalValidationError("invalid_multimodal_validation_evidence")


def _digest(value: str) -> None:
    _require(type(value) is str and _HEX.fullmatch(value) is not None)


def _text(value: str, limit: int) -> None:
    _require(type(value) is str and 0 < len(value) <= limit)
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ClinicalValidationError(
            "invalid_multimodal_validation_evidence"
        ) from None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object) -> str:
    return _sha(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    )


class ValidationTask(str, Enum):
    """Separate non-diagnostic tasks with qualified review requirements."""

    ECG_QUALITY = "ecg_quality"
    ECG_INTERVAL = "ecg_interval_candidates"
    FORM = "form_grounding"
    DICOM = "dicom_frame_coverage"
    PIXEL_PHI = "burned_in_phi"
    WSI = "wsi_tiling_provenance"


class ValidationSlice(str, Enum):
    """Disjoint, public development slices; never patient-defined labels."""

    NOMINAL = "nominal"
    ACQUISITION = "adverse_acquisition"
    AMBIGUITY = "ambiguity_or_conflict"


class ReviewOrigin(str, Enum):
    """Simulation is never presented as a clinician's completed review."""

    SIMULATED = "development_simulation"
    ATTESTED = "caller_attested_review"


_TASKS = {
    ValidationTask.ECG_QUALITY: (
        "Review acquisition integrity and quality holds before ECG processing.",
        "Clinician trained in ECG acquisition and signal-quality review.",
        "Does the quality decision match the visible acquisition evidence?",
        "Synthetic waveforms do not establish clinical signal-quality sensitivity.",
    ),
    ValidationTask.ECG_INTERVAL: (
        "Review original fiducials and interval candidates without diagnosis.",
        "Clinician qualified to review ECG fiducials and interval measurements.",
        "Are interval candidates or abstentions supported by the original fiducials?",
        "Candidates are not diagnostic interpretation or validated clinical measurements.",
    ),
    ValidationTask.FORM: (
        "Review label, value and source-box grounding on clinical forms.",
        "Clinical document reviewer trained in source-grounded abstraction.",
        "Is every proposed value associated with the correct visible label and source?",
        "Synthetic layouts do not establish clinical document extraction accuracy.",
    ),
    ValidationTask.DICOM: (
        "Review sampled-frame coverage and explicit unexamined regions.",
        "Radiology reviewer qualified for acquisition and frame-coverage assessment.",
        "Do sampled evidence and exclusions accurately represent acquisition coverage?",
        "Coverage is not diagnostic adequacy or lesion-detection performance.",
    ),
    ValidationTask.PIXEL_PHI: (
        "Review burned-in identifier coverage and fresh sanitized exports.",
        "Clinical privacy reviewer trained in pixel and metadata identifier leakage.",
        "Are all declared private carriers covered across source and export surfaces?",
        "Known synthetic carriers do not establish clinical identifier recall.",
    ),
    ValidationTask.WSI: (
        "Review tile selection, original coordinates and aggregation lineage.",
        "Digital pathology reviewer qualified to assess tiling and provenance.",
        "Are selected tiles, exclusions and aggregate provenance supported by the slide?",
        "Synthetic slides do not establish diagnostic pathology performance.",
    ),
}


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateLineage:
    """Immutable caller-supplied artifact and terms snapshots, not license gates."""

    model_digest: str
    preprocessing_digest: str
    runtime_digest: str
    hardware_digest: str
    harness_digest: str
    license_digest: str

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            _digest(value)

    @property
    def digest(self) -> str:
        """Bind this exact candidate's declared artifacts and configuration."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewAsset:
    """Protected in-memory source artifact, never copied into an aggregate."""

    media_type: str
    payload: bytes = field(repr=False)

    def __post_init__(self) -> None:
        _require(
            self.media_type
            in ("image/png", "application/dicom", "application/json", "image/tiff")
        )
        _require(type(self.payload) is bytes and 0 < len(self.payload) <= 4 * 1024**2)

    @property
    def digest(self) -> str:
        """Fingerprint actual artifact bytes, not a filename or mutable URL."""
        return _sha(self.payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateObservation:
    """Executed output plus non-compensable caller-observed safety evidence."""

    content: str = field(repr=False)
    abstained: bool
    critical_privacy_failure: bool
    provenance_failure: bool

    def __post_init__(self) -> None:
        _text(self.content, 8192)
        _require(
            all(
                type(value) is bool
                for value in (
                    self.abstained,
                    self.critical_privacy_failure,
                    self.provenance_failure,
                )
            )
        )

    @property
    def digest(self) -> str:
        """Bind output and observed safety state together."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationCase:
    """One unique source unit and two protected, executed candidate outputs."""

    assets: tuple[ReviewAsset, ...] = field(repr=False)
    context: str = field(repr=False)
    candidates: tuple[CandidateObservation, CandidateObservation] = field(repr=False)
    slice: ValidationSlice

    def __post_init__(self) -> None:
        _require(type(self.assets) is tuple and 1 <= len(self.assets) <= 8)
        _require(all(type(asset) is ReviewAsset for asset in self.assets))
        _text(self.context, 4096)
        _require(type(self.candidates) is tuple and len(self.candidates) == 2)
        _require(all(type(item) is CandidateObservation for item in self.candidates))
        _require(type(self.slice) is ValidationSlice)

    @property
    def source_digest(self) -> str:
        """Bind the source unit independent of its ratings or candidate output."""
        return _hash([(asset.media_type, asset.digest) for asset in self.assets])

    @property
    def digest(self) -> str:
        """Bind protected source, output, context and declared slice."""
        return _hash(
            {
                "source": self.source_digest,
                "context": _sha(self.context.encode()),
                "candidates": [item.digest for item in self.candidates],
                "slice": self.slice,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationPack:
    """Declared task, sampling and license lineage with bounded protected cases."""

    task: ValidationTask
    cases: tuple[ValidationCase, ...] = field(repr=False)
    candidates: tuple[CandidateLineage, CandidateLineage]
    sampling_manifest_digest: str
    dataset_license_digest: str
    source_kind: str = "synthetic_development"

    def __post_init__(self) -> None:
        _require(type(self.task) is ValidationTask)
        _require(type(self.cases) is tuple and 1 <= len(self.cases) <= 256)
        _require(all(type(case) is ValidationCase for case in self.cases))
        _require(len({case.source_digest for case in self.cases}) == len(self.cases))
        _require(
            sum(len(asset.payload) for case in self.cases for asset in case.assets)
            <= 64 * 1024**2
        )
        _require(type(self.candidates) is tuple and len(self.candidates) == 2)
        _require(all(type(item) is CandidateLineage for item in self.candidates))
        _require(self.candidates[0].digest != self.candidates[1].digest)
        _digest(self.sampling_manifest_digest)
        _digest(self.dataset_license_digest)
        _require(
            self.source_kind
            in ("synthetic_development", "caller_declared_redistributable")
        )

    @property
    def digest(self) -> str:
        """Bind every case, candidate, sampling declaration and terms snapshot."""
        return _hash(
            {
                "task": self.task,
                "cases": [case.digest for case in self.cases],
                "candidates": [candidate.digest for candidate in self.candidates],
                "sampling_manifest_digest": self.sampling_manifest_digest,
                "dataset_license_digest": self.dataset_license_digest,
                "source_kind": self.source_kind,
            }
        )

    def describe(self) -> dict[str, Any]:
        """Describe intended use and qualifications without cases or sample counts."""
        purpose, qualifications, _, limitation = _TASKS[self.task]
        return {
            "task": self.task.value,
            "intended_use": purpose,
            "reviewer_qualifications": qualifications,
            "limitations": limitation,
            "sample_construction": self.source_kind,
            "sampling_manifest_digest": self.sampling_manifest_digest,
            "dataset_license_digest": self.dataset_license_digest,
            "non_diagnostic": True,
            "independent_validation": False,
        }


def _rubric(task: ValidationTask) -> tuple[RubricCriterion, ...]:
    questions = (
        _TASKS[task][2],
        "Is the original task-relevant evidence retained without unsupported additions?",
        "Are all required privacy surfaces free of observed critical leakage?",
        "Is every output bound to the exact source, transformation and candidate artifacts?",
    )
    return tuple(
        RubricCriterion(name, question, 0, 1)
        for name, question in zip(_CRITERIA, questions)
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class ReviewerQualification:
    """Caller-attested qualification/conflict evidence; not credential verification."""

    reviewer_digest: str
    tasks: tuple[ValidationTask, ...]
    qualification_digest: str
    conflict_screening_digest: str
    origin: ReviewOrigin
    recused: bool = False

    def __post_init__(self) -> None:
        for value in (
            self.reviewer_digest,
            self.qualification_digest,
            self.conflict_screening_digest,
        ):
            _digest(value)
        _require(
            type(self.tasks) is tuple and 1 <= len(self.tasks) <= len(ValidationTask)
        )
        _require(all(type(task) is ValidationTask for task in self.tasks))
        _require(len(set(self.tasks)) == len(self.tasks))
        _require(type(self.origin) is ReviewOrigin and type(self.recused) is bool)

    @property
    def digest(self) -> str:
        """Bind qualification, conflict decision, scope and review origin."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class PreparedReview:
    """Protected reviewer packets and separate evaluator-only identity mapping."""

    pack_digest: str
    reviewer: ReviewerQualification = field(repr=False)
    packets: tuple[BlindedAdjudicationPacket, ...] = field(repr=False)
    mapping: SealedIdentityMapping = field(repr=False)

    def __post_init__(self) -> None:
        _digest(self.pack_digest)
        _require(type(self.reviewer) is ReviewerQualification)
        _require(type(self.packets) is tuple and 1 <= len(self.packets) <= 256)
        _require(
            all(type(packet) is BlindedAdjudicationPacket for packet in self.packets)
        )
        _require(type(self.mapping) is SealedIdentityMapping)


def prepare_review(
    pack: ValidationPack, reviewer: ReviewerQualification, *, key: bytes
) -> PreparedReview:
    """Create balanced blinded packets for one qualified, non-recused reviewer.

    Args:
        pack: Bounded protected source assets and executed candidate observations.
        reviewer: Caller-governed qualification and conflict evidence.
        key: Evaluator-held key; retain it separately from reviewer-visible packets.

    Returns:
        Protected review material. Packet source references resolve only against
        this pack's in-memory assets; the private mapping must not reach reviewers.
    """
    _require(type(pack) is ValidationPack and type(reviewer) is ReviewerQualification)
    _require(not reviewer.recused and pack.task in reviewer.tasks)
    _require(type(key) is bytes and len(key) >= 32)
    scoped_key = hmac.new(key, reviewer.digest.encode(), hashlib.sha256).digest()
    cases = tuple(
        ComparisonCase(
            case_ref=case.digest,
            source_evidence=(SourceEvidence("context", case.context),)
            + tuple(
                SourceEvidence(
                    f"asset-{index}",
                    json.dumps(
                        {"sha256": asset.digest, "media_type": asset.media_type},
                        sort_keys=True,
                    ),
                )
                for index, asset in enumerate(case.assets)
            ),
            candidate_outputs={
                lineage.digest: observation.content
                for lineage, observation in zip(pack.candidates, case.candidates)
            },
        )
        for case in pack.cases
    )
    try:
        packets, mapping = render_blinded_adjudication_packets(
            packet_set_ref=pack.digest,
            cases=cases,
            rubric=_rubric(pack.task),
            conflict_of_interest=ConflictOfInterestMetadata(
                reviewer.reviewer_digest,
                "sha256:" + reviewer.conflict_screening_digest,
            ),
            holdout_commitment_digest="sha256:" + pack.digest,
            submission_manifest_digests={
                item.digest: "sha256:" + item.digest for item in pack.candidates
            },
            randomization_key=scoped_key,
        )
    except Exception:
        raise ClinicalValidationError(
            "invalid_multimodal_validation_evidence"
        ) from None
    return PreparedReview(
        pack_digest=pack.digest, reviewer=reviewer, packets=packets, mapping=mapping
    )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PacketRating:
    """Protected binary rubric decision bound to an exact blinded packet."""

    packet_digest: str
    alias: str
    quality: bool
    utility: bool
    privacy: bool
    provenance: bool

    def __post_init__(self) -> None:
        _digest(self.packet_digest)
        _require(
            type(self.alias) is str and self.alias in ("candidate-a", "candidate-b")
        )
        _require(all(type(value) is bool for value in self.scores))

    @property
    def scores(self) -> tuple[bool, bool, bool, bool]:
        """Return the four rubric criteria in their declared order."""
        return self.quality, self.utility, self.privacy, self.provenance

    @property
    def digest(self) -> str:
        """Bind the exact decision, positional alias and packet."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CompletedReview:
    """Protected complete ratings for both candidates in every source unit."""

    prepared: PreparedReview
    ratings: tuple[PacketRating, ...]

    def __post_init__(self) -> None:
        _require(type(self.prepared) is PreparedReview)
        _require(type(self.ratings) is tuple and 2 <= len(self.ratings) <= 512)
        _require(all(type(rating) is PacketRating for rating in self.ratings))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Adjudication:
    """Independent qualified review of an exact set of conflicting decisions."""

    prepared: PreparedReview
    rating: PacketRating
    conflicting_rating_digests: tuple[str, ...]
    reason: DisagreementReason

    def __post_init__(self) -> None:
        _require(
            type(self.prepared) is PreparedReview and type(self.rating) is PacketRating
        )
        _require(
            type(self.conflicting_rating_digests) is tuple
            and 2 <= len(self.conflicting_rating_digests) <= 8
        )
        for value in self.conflicting_rating_digests:
            _digest(value)
        _require(
            len(set(self.conflicting_rating_digests))
            == len(self.conflicting_rating_digests)
        )
        _require(type(self.reason) is DisagreementReason)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ValidationAssessment:
    """Private evaluator decision and separately publishable suppressed report.

    The decision is intentionally excluded from public JSON: on a suppressed
    cohort even a pass/fail flag could disclose a small critical failure cell.
    """

    decision: str
    pack_digest: str
    candidate_digest: str
    _report_json: str

    def to_json(self) -> str:
        """Return only aggregate evidence; retain this object in protected storage."""
        return self._report_json


def packet_digest(packet: BlindedAdjudicationPacket) -> str:
    """Fingerprint the packet actually shown to a reviewer, including its rubric."""
    _require(type(packet) is BlindedAdjudicationPacket)
    return _sha(packet.to_json().encode())


def _verified_packets(
    pack: ValidationPack, prepared: PreparedReview, key: bytes
) -> dict[tuple[str, str], tuple[str, str]]:
    expected = prepare_review(pack, prepared.reviewer, key=key)
    _require(prepared == expected)
    scoped_key = hmac.new(
        key, prepared.reviewer.digest.encode(), hashlib.sha256
    ).digest()
    _require(
        verify_sealed_identity_mapping(
            prepared.packets, prepared.mapping, scoped_key
        ).valid
    )
    return {
        (packet_digest(packet), assignment.alias): (
            entry.case_ref,
            assignment.candidate_identity,
        )
        for packet, entry in zip(prepared.packets, prepared.mapping.entries)
        for assignment in entry.assignments
    }


def _public_cell(
    rows: list[dict[str, bool]], decisions: list[ReviewerDecision], minimum: int
) -> dict[str, Any]:
    suppressed = {
        "suppressed": True,
        "case_count": None,
        "metrics": None,
        "disagreement": None,
    }
    if len(rows) < minimum:
        return suppressed
    metrics: dict[str, Any] = {}
    for name in (*_CRITERIA, "emitted"):
        successes = sum(row[name] for row in rows)
        if any(0 < count < minimum for count in (successes, len(rows) - successes)):
            return suppressed
        metrics[name] = {
            "successes": successes,
            "trials": len(rows),
            "rate": successes / len(rows),
            "wilson_95": wilson_interval(FidelityCounts(successes, len(rows))),
        }
    disagreement = reviewer_disagreement_report(decisions, minimum_cell_size=minimum)
    if disagreement.has_suppressed_cells:
        return suppressed
    return {
        "suppressed": False,
        "case_count": len(rows),
        "metrics": metrics,
        "disagreement": disagreement.to_dict(),
    }


def assess_validation(
    pack: ValidationPack,
    reviews: tuple[CompletedReview, ...],
    *,
    candidate_index: int,
    key: bytes,
    adjudications: tuple[Adjudication, ...] = (),
    minimum_cell_size: int = 5,
) -> ValidationAssessment:
    """Assess complete blinded evidence without claiming clinical qualification.

    Args:
        pack: Protected exact source and candidate artifacts.
        reviews: Two to eight complete reviews by distinct eligible reviewers.
        candidate_index: Candidate to assess; both remain blinded during review.
        key: Evaluator key used to prepare these exact packets.
        adjudications: Independent resolutions bound to conflicting rating digests.
        minimum_cell_size: Complementary suppression threshold, at least five.

    Returns:
        Private rubric decision and a public, content-free aggregate report.
        Unresolved disagreements hold; any observed privacy or provenance failure
        blocks irrespective of majority, adjudication or quality scores. Confidence
        intervals describe case-level binary evidence, not clinical efficacy.

    Raises:
        ClinicalValidationError: Missing, stale, duplicate or ineligible evidence.
    """
    _require(type(pack) is ValidationPack)
    _require(type(candidate_index) is int and candidate_index in (0, 1))
    _require(type(minimum_cell_size) is int and 5 <= minimum_cell_size <= 256)
    _require(type(reviews) is tuple and 2 <= len(reviews) <= 8)
    _require(all(type(review) is CompletedReview for review in reviews))
    _require(type(adjudications) is tuple and len(adjudications) <= len(pack.cases))
    _require(all(type(item) is Adjudication for item in adjudications))
    identities = {review.prepared.reviewer.reviewer_digest for review in reviews}
    _require(len(identities) == len(reviews))
    target = pack.candidates[candidate_index].digest
    ratings: dict[str, list[tuple[ReviewerQualification, PacketRating]]] = {
        case.digest: [] for case in pack.cases
    }
    for review in reviews:
        bindings = _verified_packets(pack, review.prepared, key)
        seen: set[tuple[str, str]] = set()
        for rating in review.ratings:
            ref = (rating.packet_digest, rating.alias)
            _require(ref in bindings and ref not in seen)
            seen.add(ref)
            case_ref, candidate = bindings[ref]
            if candidate == target:
                ratings[case_ref].append((review.prepared.reviewer, rating))
        _require(seen == set(bindings))
    resolutions: dict[str, Adjudication] = {}
    for item in adjudications:
        _require(item.prepared.reviewer.reviewer_digest not in identities)
        bindings = _verified_packets(pack, item.prepared, key)
        ref = (item.rating.packet_digest, item.rating.alias)
        _require(ref in bindings)
        case_ref, candidate = bindings[ref]
        _require(candidate == target and case_ref not in resolutions)
        original = ratings[case_ref]
        _require(len({rating.scores for _, rating in original}) > 1)
        _require(
            set(item.conflicting_rating_digests)
            == {rating.digest for _, rating in original}
        )
        resolutions[case_ref] = item
    all_rows: list[dict[str, bool]] = []
    all_decisions: list[ReviewerDecision] = []
    by_slice: dict[
        ValidationSlice, tuple[list[dict[str, bool]], list[ReviewerDecision]]
    ] = {part: ([], []) for part in ValidationSlice}
    unresolved = False
    blocked = False
    origins = {review.prepared.reviewer.origin for review in reviews}
    for case in pack.cases:
        original = ratings[case.digest]
        scores = [rating.scores for _, rating in original]
        conflict = len(set(scores)) > 1
        resolution = resolutions.get(case.digest)
        unresolved |= conflict and resolution is None
        effective = (
            resolution.rating.scores
            if resolution
            else tuple(all(row[index] for row in scores) for index in range(4))
        )
        if resolution:
            origins.add(resolution.prepared.reviewer.origin)
            scores.append(resolution.rating.scores)
        observation = case.candidates[candidate_index]
        privacy = (
            all(row[2] for row in scores) and not observation.critical_privacy_failure
        )
        provenance = (
            all(row[3] for row in scores) and not observation.provenance_failure
        )
        blocked |= not privacy or not provenance
        row = dict(zip(_CRITERIA, effective))
        row.update(
            privacy=privacy, provenance=provenance, emitted=not observation.abstained
        )
        decisions = [
            ReviewerDecision(
                case.digest,
                reviewer.reviewer_digest,
                str(rating.scores),
                reason=(
                    resolution.reason
                    if resolution
                    else DisagreementReason.EVIDENCE_QUALITY
                )
                if conflict
                else None,
                adjudicated=resolution is not None,
            )
            for reviewer, rating in original
        ]
        all_rows.append(row)
        all_decisions.extend(decisions)
        by_slice[case.slice][0].append(row)
        by_slice[case.slice][1].extend(decisions)
    decision = (
        "blocked"
        if blocked
        else "hold"
        if (
            unresolved
            or not any(row["emitted"] for row in all_rows)
            or not all(row["quality"] and row["utility"] for row in all_rows)
        )
        else "rubric_satisfied"
    )
    slices = {
        part.value: _public_cell(rows, decisions, minimum_cell_size)
        for part, (rows, decisions) in by_slice.items()
    }
    # Empty declared slices have no observations, but are still indistinguishable
    # from suppressed small slices. Do not publish a reconstructable overall total.
    total = _public_cell(all_rows, all_decisions, minimum_cell_size)
    if any(value["suppressed"] for value in slices.values()):
        total = {
            "suppressed": True,
            "case_count": None,
            "metrics": None,
            "disagreement": None,
        }
    report = {
        "schema_version": "openmed.multimodal_clinical_validation.v1",
        "task": pack.task.value,
        "review_origin": (
            ReviewOrigin.ATTESTED.value
            if origins == {ReviewOrigin.ATTESTED}
            else ReviewOrigin.SIMULATED.value
        ),
        "non_diagnostic": True,
        "independent_clinical_validation": False,
        "credential_verification": "caller_governed",
        "interval_interpretation": "descriptive_case_level_wilson_not_population_inference",
        "minimum_cell_size": minimum_cell_size,
        "overall": total,
        "slices": slices,
    }
    return ValidationAssessment(
        decision=decision,
        pack_digest=pack.digest,
        candidate_digest=target,
        _report_json=json.dumps(report, sort_keys=True, allow_nan=False),
    )
