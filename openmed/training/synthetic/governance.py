"""Release controls for synthetic training artifacts.

Synthetic datasets, and the models trained on them, need downstream-use
controls before they leave a project. :func:`validate_synthetic_release`
checks that an artifact carries a machine-readable synthetic label (content
kind, intended and prohibited uses, generator and model disclosure), that its
lineage is complete and licensed, that neither it nor any ancestor has been
withdrawn, that it contains no benchmark canaries or benchmark records, and
that downstream tasks trained on it do not regress critically. Subgroup
quality and downstream utility are reported per slice with score-interval
uncertainty, and slices below a declared minimum cell size are suppressed
instead of exposed.

Every control consumes plain data: lineage nodes from
:mod:`openmed.compliance.artifact_lineage`, license records, generator and
model digests, benchmark canary and overlap counts, and subgroup counts.
Outputs carry digests, controlled identifiers, closed codes, counts, and
rates only. They never include record text, real records, license values, or
the names of suppressed groups. Validation fails closed: a missing control is
a failure, not a pass.

These controls do not establish that synthetic data is anonymous or that it is
clinically equivalent to any real population, and they are not legal advice.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
from statistics import NormalDist
from typing import TYPE_CHECKING, Any, Final, TypeVar

from openmed.compliance.artifact_lineage import ArtifactLineageManifest
from openmed.core.audit import hash_text, stable_hash
from openmed.training.corpus import normalize_passage_text

from .spdx_identifier import SpdxIdentifierStatus, normalize_spdx_identifier

if TYPE_CHECKING:
    from openmed.eval.release_gates import GateCheck

__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL",
    "DEFAULT_CRITICAL_REGRESSION_DROP",
    "DEFAULT_MAX_BENCHMARK_OVERLAP_RATE",
    "DEFAULT_MINIMUM_CELL_SIZE",
    "DEFAULT_PERMITTED_DATA_LICENSES",
    "DOWNSTREAM_REGRESSION_SCHEMA_VERSION",
    "MANDATORY_PROHIBITED_USES",
    "MIN_CANARY_LENGTH",
    "OVERALL_AXIS",
    "OVERALL_GROUP",
    "SUBGROUP_QUALITY_SCHEMA_VERSION",
    "SYNTHETIC_LABEL_DISCLOSURES",
    "SYNTHETIC_LABEL_SCHEMA_VERSION",
    "SYNTHETIC_RELEASE_DECISION_SCHEMA_VERSION",
    "SYNTHETIC_RELEASE_GATE",
    "SYNTHETIC_WITHDRAWAL_SCHEMA_VERSION",
    "BenchmarkContaminationEvidence",
    "CountBand",
    "DownstreamComparison",
    "DownstreamRegressionReport",
    "GenerationMethod",
    "GeneratorDisclosure",
    "InvalidatedArtifact",
    "LicenseRecord",
    "RegressionSlice",
    "ReleaseControl",
    "ReleaseControlFinding",
    "ReleaseFindingCode",
    "SubgroupDisparity",
    "SubgroupMetric",
    "SubgroupQualityReport",
    "SubgroupSlice",
    "SyntheticArtifactLabel",
    "SyntheticContentKind",
    "SyntheticGovernanceError",
    "SyntheticReleaseBlockedError",
    "SyntheticReleaseDecision",
    "SyntheticReleasePolicy",
    "SyntheticUse",
    "WithdrawalEvidence",
    "WithdrawalNotice",
    "WithdrawalReason",
    "assert_synthetic_release",
    "build_downstream_regression_report",
    "build_subgroup_quality_report",
    "fingerprint_record",
    "measure_benchmark_contamination",
    "propagate_withdrawal",
    "validate_synthetic_release",
]

SYNTHETIC_LABEL_SCHEMA_VERSION: Final = "openmed.training.synthetic.label.v1"
SYNTHETIC_RELEASE_DECISION_SCHEMA_VERSION: Final = (
    "openmed.training.synthetic.release_decision.v1"
)
SYNTHETIC_WITHDRAWAL_SCHEMA_VERSION: Final = "openmed.training.synthetic.withdrawal.v1"
SUBGROUP_QUALITY_SCHEMA_VERSION: Final = (
    "openmed.training.synthetic.subgroup_quality.v1"
)
DOWNSTREAM_REGRESSION_SCHEMA_VERSION: Final = (
    "openmed.training.synthetic.downstream_regression.v1"
)
SYNTHETIC_RELEASE_GATE: Final = "synthetic_release_controls"

# Cell-size suppression follows the common public-health convention of not
# reporting cells with 1 to 10 members. Callers may declare a stricter value.
DEFAULT_MINIMUM_CELL_SIZE: Final = 11
DEFAULT_CONFIDENCE_LEVEL: Final = 0.95
DEFAULT_CRITICAL_REGRESSION_DROP: Final = 0.05
DEFAULT_MAX_BENCHMARK_OVERLAP_RATE: Final = 0.0
# Data licenses accepted in addition to the permissive SPDX code allowlist.
# Share-alike, non-commercial, and no-derivatives variants are not accepted.
DEFAULT_PERMITTED_DATA_LICENSES: Final[frozenset[str]] = frozenset(
    {"CC-BY-4.0", "CC0-1.0"}
)
OVERALL_AXIS: Final = "overall"
OVERALL_GROUP: Final = "all"
MIN_CANARY_LENGTH: Final = 8
SYNTHETIC_LABEL_DISCLOSURES: Final[tuple[str, ...]] = (
    "anonymity_not_guaranteed",
    "not_clinically_equivalent_to_real_population",
    "not_real_patient_data",
)

_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE: Final = re.compile(r"^[a-z][a-z0-9_.:-]{0,63}$")
_VERSION_RE: Final = re.compile(r"^[0-9a-z][0-9a-z.+_-]{0,31}$")
_MAX_COUNT: Final = 10**12
_MAX_LICENSE_ID_LENGTH: Final = 128
_MIN_CELL_SIZE_FLOOR: Final = 2
_FLOAT_DIGITS: Final = 6
_FLOAT_TOLERANCE: Final = 1e-12

_T = TypeVar("_T")
_E = TypeVar("_E", bound=Enum)


class SyntheticGovernanceError(ValueError):
    """Raised when a synthetic release control input is malformed."""


class SyntheticUse(str, Enum):
    """Closed vocabulary for intended and prohibited uses of an artifact."""

    MODEL_TRAINING = "model_training"
    MODEL_EVALUATION = "model_evaluation"
    SOFTWARE_TESTING = "software_testing"
    EDUCATION = "education"
    METHODS_RESEARCH = "methods_research"
    PUBLIC_REDISTRIBUTION = "public_redistribution"
    CLINICAL_DECISION_MAKING = "clinical_decision_making"
    REIDENTIFICATION = "reidentification"
    REAL_DATA_SUBSTITUTION = "real_data_substitution"
    POPULATION_INFERENCE = "population_inference"


# Uses every synthetic label prohibits. Callers may add prohibitions but can
# never remove these.
MANDATORY_PROHIBITED_USES: Final[frozenset[SyntheticUse]] = frozenset(
    {
        SyntheticUse.CLINICAL_DECISION_MAKING,
        SyntheticUse.POPULATION_INFERENCE,
        SyntheticUse.REAL_DATA_SUBSTITUTION,
        SyntheticUse.REIDENTIFICATION,
    }
)


class SyntheticContentKind(str, Enum):
    """Whether an artifact is wholly generated or derived from real inputs."""

    FULLY_SYNTHETIC = "fully_synthetic"
    PARTIALLY_SYNTHETIC = "partially_synthetic"


class GenerationMethod(str, Enum):
    """Closed description of how a generator produces records."""

    RULE_BASED = "rule_based"
    TEMPLATE = "template"
    STATISTICAL_MODEL = "statistical_model"
    NEURAL_MODEL = "neural_model"
    HYBRID = "hybrid"


_MODEL_BACKED_METHODS: Final = frozenset(
    {
        GenerationMethod.HYBRID,
        GenerationMethod.NEURAL_MODEL,
        GenerationMethod.STATISTICAL_MODEL,
    }
)


class WithdrawalReason(str, Enum):
    """Closed reason codes recorded on a withdrawal notice."""

    SOURCE_LICENSE_REVOKED = "source_license_revoked"
    CONSENT_WITHDRAWN = "consent_withdrawn"
    PRIVACY_INCIDENT = "privacy_incident"
    MEMORIZATION_DETECTED = "memorization_detected"
    BENCHMARK_CONTAMINATION = "benchmark_contamination"
    GENERATOR_DEFECT = "generator_defect"
    QUALITY_DEFECT = "quality_defect"
    LEGAL_REQUEST = "legal_request"


class CountBand(str, Enum):
    """Coarse slice-size bands relative to the minimum cell size."""

    MINIMUM_TO_UNDER_DOUBLE = "minimum_to_under_double"
    DOUBLE_TO_UNDER_FOURFOLD = "double_to_under_fourfold"
    FOURFOLD_OR_MORE = "fourfold_or_more"


class ReleaseControl(str, Enum):
    """Release controls evaluated by :func:`validate_synthetic_release`."""

    LABEL = "label"
    LINEAGE = "lineage"
    LICENSE = "license"
    WITHDRAWAL = "withdrawal"
    CONTAMINATION = "contamination"
    REGRESSION = "regression"
    SUBGROUP = "subgroup"


class ReleaseFindingCode(str, Enum):
    """Closed reason codes for failed release controls."""

    LABEL_MISSING = "label_missing"
    INTENDED_USE_MISSING = "intended_use_missing"
    INTENDED_USE_PROHIBITED = "intended_use_prohibited"
    GENERATOR_DISCLOSURE_MISSING = "generator_disclosure_missing"
    MODEL_DIGEST_MISSING = "model_digest_missing"
    LINEAGE_MISSING = "lineage_missing"
    LINEAGE_INTEGRITY_FAILED = "lineage_integrity_failed"
    LINEAGE_ARTIFACT_ABSENT = "lineage_artifact_absent"
    LINEAGE_ANCESTOR_MISSING = "lineage_ancestor_missing"
    GENERATOR_NOT_IN_LINEAGE = "generator_not_in_lineage"
    LICENSE_MISSING = "license_missing"
    LICENSE_CONFLICT = "license_conflict"
    LICENSE_NOT_PERMITTED = "license_not_permitted"
    LICENSE_REDISTRIBUTION_DISALLOWED = "license_redistribution_disallowed"
    ARTIFACT_WITHDRAWN = "artifact_withdrawn"
    BENCHMARK_EVIDENCE_MISSING = "benchmark_evidence_missing"
    BENCHMARK_EVIDENCE_MISMATCH = "benchmark_evidence_mismatch"
    BENCHMARK_EVIDENCE_EMPTY = "benchmark_evidence_empty"
    REQUIRED_BENCHMARK_MISSING = "required_benchmark_missing"
    BENCHMARK_CANARY_DETECTED = "benchmark_canary_detected"
    BENCHMARK_OVERLAP_EXCEEDED = "benchmark_overlap_exceeded"
    REGRESSION_EVIDENCE_MISSING = "regression_evidence_missing"
    REGRESSION_EVIDENCE_INSUFFICIENT = "regression_evidence_insufficient"
    CRITICAL_REGRESSION = "critical_regression"
    SUBGROUP_REPORT_MISSING = "subgroup_report_missing"
    SUBGROUP_EVIDENCE_INSUFFICIENT = "subgroup_evidence_insufficient"
    SUBGROUP_DISPARITY_EXCEEDED = "subgroup_disparity_exceeded"


_FINDING_CONTROLS: Final[dict[ReleaseFindingCode, ReleaseControl]] = {
    ReleaseFindingCode.LABEL_MISSING: ReleaseControl.LABEL,
    ReleaseFindingCode.INTENDED_USE_MISSING: ReleaseControl.LABEL,
    ReleaseFindingCode.INTENDED_USE_PROHIBITED: ReleaseControl.LABEL,
    ReleaseFindingCode.GENERATOR_DISCLOSURE_MISSING: ReleaseControl.LABEL,
    ReleaseFindingCode.MODEL_DIGEST_MISSING: ReleaseControl.LABEL,
    ReleaseFindingCode.LINEAGE_MISSING: ReleaseControl.LINEAGE,
    ReleaseFindingCode.LINEAGE_INTEGRITY_FAILED: ReleaseControl.LINEAGE,
    ReleaseFindingCode.LINEAGE_ARTIFACT_ABSENT: ReleaseControl.LINEAGE,
    ReleaseFindingCode.LINEAGE_ANCESTOR_MISSING: ReleaseControl.LINEAGE,
    ReleaseFindingCode.GENERATOR_NOT_IN_LINEAGE: ReleaseControl.LINEAGE,
    ReleaseFindingCode.LICENSE_MISSING: ReleaseControl.LICENSE,
    ReleaseFindingCode.LICENSE_CONFLICT: ReleaseControl.LICENSE,
    ReleaseFindingCode.LICENSE_NOT_PERMITTED: ReleaseControl.LICENSE,
    ReleaseFindingCode.LICENSE_REDISTRIBUTION_DISALLOWED: ReleaseControl.LICENSE,
    ReleaseFindingCode.ARTIFACT_WITHDRAWN: ReleaseControl.WITHDRAWAL,
    ReleaseFindingCode.BENCHMARK_EVIDENCE_MISSING: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.BENCHMARK_EVIDENCE_MISMATCH: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.BENCHMARK_EVIDENCE_EMPTY: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.REQUIRED_BENCHMARK_MISSING: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.BENCHMARK_CANARY_DETECTED: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.BENCHMARK_OVERLAP_EXCEEDED: ReleaseControl.CONTAMINATION,
    ReleaseFindingCode.REGRESSION_EVIDENCE_MISSING: ReleaseControl.REGRESSION,
    ReleaseFindingCode.REGRESSION_EVIDENCE_INSUFFICIENT: ReleaseControl.REGRESSION,
    ReleaseFindingCode.CRITICAL_REGRESSION: ReleaseControl.REGRESSION,
    ReleaseFindingCode.SUBGROUP_REPORT_MISSING: ReleaseControl.SUBGROUP,
    ReleaseFindingCode.SUBGROUP_EVIDENCE_INSUFFICIENT: ReleaseControl.SUBGROUP,
    ReleaseFindingCode.SUBGROUP_DISPARITY_EXCEEDED: ReleaseControl.SUBGROUP,
}


# ---------------------------------------------------------------------------
# Value validation. Error messages name the field, never the offending value.
# ---------------------------------------------------------------------------


def _digest(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise SyntheticGovernanceError(f"{field_name} must be a SHA-256 digest")
    digest = value.lower()
    if _HEX_DIGEST_RE.fullmatch(digest) is not None:
        digest = f"sha256:{digest}"
    if _DIGEST_RE.fullmatch(digest) is None:
        raise SyntheticGovernanceError(f"{field_name} must be a SHA-256 digest")
    return digest


def _optional_digest(value: Any, field_name: str) -> str | None:
    return None if value is None else _digest(value, field_name)


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise SyntheticGovernanceError(
            f"{field_name} must be a lowercase controlled identifier"
        )
    return value


def _version(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise SyntheticGovernanceError(f"{field_name} must be a controlled version")
    return value


def _count(value: Any, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNT:
        raise SyntheticGovernanceError(f"{field_name} must be a non-negative integer")
    return value


def _flag(value: Any, field_name: str) -> bool:
    if type(value) is not bool:
        raise SyntheticGovernanceError(f"{field_name} must be a boolean")
    return value


def _rate(value: Any, field_name: str, *, allow_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SyntheticGovernanceError(f"{field_name} must be a finite rate")
    rate = float(value)
    lower_ok = rate >= 0.0 if allow_zero else rate > 0.0
    if not math.isfinite(rate) or not lower_ok or rate > 1.0:
        raise SyntheticGovernanceError(f"{field_name} must be a rate in [0, 1]")
    return rate


def _confidence_level(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SyntheticGovernanceError("confidence_level must be in (0, 1)")
    level = float(value)
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise SyntheticGovernanceError("confidence_level must be in (0, 1)")
    return level


def _minimum_cell_size(value: Any) -> int:
    if type(value) is not int or not _MIN_CELL_SIZE_FLOOR <= value <= _MAX_COUNT:
        raise SyntheticGovernanceError(
            f"minimum_cell_size must be an integer of at least {_MIN_CELL_SIZE_FLOOR}"
        )
    return value


def _enum(enum_cls: type[_E], value: Any, field_name: str) -> _E:
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except (TypeError, ValueError):
        raise SyntheticGovernanceError(
            f"{field_name} is not a supported value"
        ) from None


def _iso_date(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 10:
        raise SyntheticGovernanceError(f"{field_name} must be an ISO date")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise SyntheticGovernanceError(f"{field_name} must be an ISO date") from None


def _iterable(value: Any, field_name: str) -> Iterable[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Iterable):
        raise SyntheticGovernanceError(f"{field_name} must be a sequence")
    return value


def _sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    return tuple(_iterable(value, field_name))


def _strict_mapping(
    payload: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SyntheticGovernanceError(f"{name} must be an object")
    keys = set(payload)
    if not required <= keys or not keys <= required | optional:
        raise SyntheticGovernanceError(f"{name} has missing or unsupported fields")
    return payload


def _round(value: float) -> float:
    return round(float(value), _FLOAT_DIGITS)


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Labels and disclosures.
# ---------------------------------------------------------------------------


_GENERATOR_FIELDS: Final = frozenset(
    {
        "generator_digest",
        "generator_id",
        "generator_version",
        "lineage_node_hash",
        "method",
        "model_digests",
    }
)


@dataclass(frozen=True)
class GeneratorDisclosure:
    """Digest-only disclosure of one generator that produced an artifact.

    Attributes:
        generator_id: Controlled identifier of the generator.
        generator_version: Controlled version string of the generator.
        method: How the generator produces records.
        generator_digest: SHA-256 digest of the generator code and config.
        lineage_node_hash: Lineage node that represents this generator; it
            must be an ancestor of the labeled artifact.
        model_digests: SHA-256 digests of model weights the generator uses.
            Model-backed methods must disclose at least one.
    """

    generator_id: str
    generator_version: str
    method: GenerationMethod
    generator_digest: str
    lineage_node_hash: str
    model_digests: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "generator_id", _identifier(self.generator_id, "generator_id")
        )
        object.__setattr__(
            self,
            "generator_version",
            _version(self.generator_version, "generator_version"),
        )
        object.__setattr__(
            self, "method", _enum(GenerationMethod, self.method, "method")
        )
        object.__setattr__(
            self,
            "generator_digest",
            _digest(self.generator_digest, "generator_digest"),
        )
        object.__setattr__(
            self,
            "lineage_node_hash",
            _digest(self.lineage_node_hash, "lineage_node_hash"),
        )
        digests = tuple(
            sorted(
                {
                    _digest(item, "model_digests")
                    for item in _sequence(self.model_digests, "model_digests")
                }
            )
        )
        object.__setattr__(self, "model_digests", digests)

    @property
    def model_backed(self) -> bool:
        """Return whether the generation method relies on a trained model."""

        return self.method in _MODEL_BACKED_METHODS

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, digest-only disclosure."""

        return {
            "generator_digest": self.generator_digest,
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "lineage_node_hash": self.lineage_node_hash,
            "method": self.method.value,
            "model_digests": list(self.model_digests),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> GeneratorDisclosure:
        """Load a disclosure while rejecting unknown or missing fields."""

        data = _strict_mapping(
            payload,
            required=_GENERATOR_FIELDS - {"model_digests"},
            optional=frozenset({"model_digests"}),
            name="generator disclosure",
        )
        return cls(
            generator_id=data["generator_id"],
            generator_version=data["generator_version"],
            method=data["method"],
            generator_digest=data["generator_digest"],
            lineage_node_hash=data["lineage_node_hash"],
            model_digests=data.get("model_digests", ()),
        )


def _uses(value: Any, field_name: str) -> tuple[SyntheticUse, ...]:
    items = {
        _enum(SyntheticUse, item, field_name) for item in _sequence(value, field_name)
    }
    return tuple(sorted(items, key=lambda use: use.value))


_LABEL_FIELDS: Final = frozenset(
    {
        "artifact_hash",
        "content_digest",
        "content_kind",
        "generators",
        "intended_uses",
        "prohibited_uses",
        "schema_version",
        "synthetic",
    }
)


@dataclass(frozen=True)
class SyntheticArtifactLabel:
    """Machine-readable synthetic label for one released artifact.

    The label binds the artifact's lineage node and content digest to its
    content kind, intended and prohibited uses, and generator disclosure.
    :data:`MANDATORY_PROHIBITED_USES` are always merged into
    ``prohibited_uses`` and cannot be removed.

    Attributes:
        artifact_hash: Lineage node commitment of the artifact, as produced by
            :class:`openmed.compliance.artifact_lineage.ArtifactLineageNode`.
        content_digest: SHA-256 digest of the released artifact bytes.
        content_kind: Whether the artifact is fully or partially synthetic.
        generators: Disclosures for every generator that produced the
            artifact.
        intended_uses: Uses the artifact is released for.
        prohibited_uses: Uses the artifact must not be put to.
        schema_version: Label schema version.
    """

    artifact_hash: str
    content_digest: str
    content_kind: SyntheticContentKind
    generators: tuple[GeneratorDisclosure, ...]
    intended_uses: tuple[SyntheticUse, ...]
    prohibited_uses: tuple[SyntheticUse, ...] = ()
    schema_version: str = SYNTHETIC_LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SYNTHETIC_LABEL_SCHEMA_VERSION:
            raise SyntheticGovernanceError("label schema_version is not supported")
        object.__setattr__(
            self, "artifact_hash", _digest(self.artifact_hash, "artifact_hash")
        )
        object.__setattr__(
            self, "content_digest", _digest(self.content_digest, "content_digest")
        )
        object.__setattr__(
            self,
            "content_kind",
            _enum(SyntheticContentKind, self.content_kind, "content_kind"),
        )
        generators = tuple(
            item
            if isinstance(item, GeneratorDisclosure)
            else GeneratorDisclosure.from_mapping(item)
            for item in _sequence(self.generators, "generators")
        )
        object.__setattr__(
            self,
            "generators",
            tuple(
                sorted(
                    set(generators),
                    key=lambda item: (item.generator_id, item.generator_digest),
                )
            ),
        )
        object.__setattr__(
            self, "intended_uses", _uses(self.intended_uses, "intended_uses")
        )
        prohibited = set(_uses(self.prohibited_uses, "prohibited_uses"))
        prohibited.update(MANDATORY_PROHIBITED_USES)
        object.__setattr__(
            self,
            "prohibited_uses",
            tuple(sorted(prohibited, key=lambda use: use.value)),
        )

    @property
    def conflicting_uses(self) -> tuple[SyntheticUse, ...]:
        """Return intended uses that the label also prohibits."""

        prohibited = set(self.prohibited_uses)
        return tuple(use for use in self.intended_uses if use in prohibited)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic machine-readable label."""

        return {
            "artifact_hash": self.artifact_hash,
            "content_digest": self.content_digest,
            "content_kind": self.content_kind.value,
            "disclosures": list(SYNTHETIC_LABEL_DISCLOSURES),
            "generators": [item.to_dict() for item in self.generators],
            "intended_uses": [use.value for use in self.intended_uses],
            "prohibited_uses": [use.value for use in self.prohibited_uses],
            "schema_version": self.schema_version,
            "synthetic": True,
        }

    def to_json(self) -> str:
        """Serialize the label deterministically."""

        return _json(self.to_dict())

    @property
    def label_digest(self) -> str:
        """Return the SHA-256 commitment over the canonical label."""

        return stable_hash(self.to_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> SyntheticArtifactLabel:
        """Load a label, rejecting anything not explicitly marked synthetic.

        Raises:
            SyntheticGovernanceError: If the mapping is not a complete
                synthetic label, carries unsupported fields, or alters the
                fixed disclosures.
        """

        data = _strict_mapping(
            payload,
            required=_LABEL_FIELDS - {"prohibited_uses"},
            optional=frozenset({"prohibited_uses", "disclosures"}),
            name="synthetic label",
        )
        if data["synthetic"] is not True:
            raise SyntheticGovernanceError("label must declare synthetic content")
        disclosures = data.get("disclosures", SYNTHETIC_LABEL_DISCLOSURES)
        if tuple(_sequence(disclosures, "disclosures")) != SYNTHETIC_LABEL_DISCLOSURES:
            raise SyntheticGovernanceError("label disclosures must not be altered")
        return cls(
            artifact_hash=data["artifact_hash"],
            content_digest=data["content_digest"],
            content_kind=data["content_kind"],
            generators=tuple(_sequence(data["generators"], "generators")),
            intended_uses=data["intended_uses"],
            prohibited_uses=data.get("prohibited_uses", ()),
            schema_version=data["schema_version"],
        )

    @classmethod
    def from_json(cls, value: str) -> SyntheticArtifactLabel:
        """Parse a JSON label without echoing parser input in errors."""

        try:
            payload = json.loads(value)
        except (TypeError, ValueError):
            raise SyntheticGovernanceError("synthetic label JSON is invalid") from None
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class LicenseRecord:
    """License declaration for one lineage node.

    Attributes:
        node_hash: Lineage node the declaration applies to.
        license_id: SPDX short identifier of the node's license.
        redistribution_allowed: Whether the declared terms permit
            redistributing the node's content or derivatives.
        evidence_digest: Optional digest of supporting evidence, for example a
            license and lineage manifest.
    """

    node_hash: str
    license_id: str
    redistribution_allowed: bool
    evidence_digest: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "node_hash", _digest(self.node_hash, "node_hash"))
        if (
            not isinstance(self.license_id, str)
            or not 0 < len(self.license_id) <= _MAX_LICENSE_ID_LENGTH
        ):
            raise SyntheticGovernanceError("license_id must be a bounded string")
        _flag(self.redistribution_allowed, "redistribution_allowed")
        object.__setattr__(
            self,
            "evidence_digest",
            _optional_digest(self.evidence_digest, "evidence_digest"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the declaration in deterministic field order."""

        return {
            "evidence_digest": self.evidence_digest,
            "license_id": self.license_id,
            "node_hash": self.node_hash,
            "redistribution_allowed": self.redistribution_allowed,
        }


def _license_permitted(
    license_id: str, permitted_data_licenses: frozenset[str]
) -> bool:
    """Return whether a license is on the permissive or data allowlist."""

    result = normalize_spdx_identifier(license_id)
    if result.is_permissive:
        return True
    if result.status is not SpdxIdentifierStatus.UNKNOWN:
        return False
    folded = license_id.strip().casefold()
    return any(folded == item.casefold() for item in permitted_data_licenses)


# ---------------------------------------------------------------------------
# Benchmark contamination.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkContaminationEvidence:
    """Counts-only benchmark overlap and canary evidence for one artifact.

    Attributes:
        benchmark_id: Controlled identifier of the benchmark checked.
        benchmark_digest: SHA-256 digest of the benchmark version checked.
        artifact_content_digest: Content digest of the artifact checked; must
            match the label's ``content_digest``.
        checked_record_count: Artifact records compared with the benchmark.
        overlapping_record_count: Artifact records whose normalized
            fingerprint matches a benchmark record.
        canary_count: Distinct benchmark canaries scanned for.
        canary_hit_count: Distinct canaries found in the artifact.
    """

    benchmark_id: str
    benchmark_digest: str
    artifact_content_digest: str
    checked_record_count: int
    overlapping_record_count: int
    canary_count: int
    canary_hit_count: int

    def __post_init__(self) -> None:
        _identifier(self.benchmark_id, "benchmark_id")
        object.__setattr__(
            self,
            "benchmark_digest",
            _digest(self.benchmark_digest, "benchmark_digest"),
        )
        object.__setattr__(
            self,
            "artifact_content_digest",
            _digest(self.artifact_content_digest, "artifact_content_digest"),
        )
        checked = _count(self.checked_record_count, "checked_record_count")
        overlapping = _count(self.overlapping_record_count, "overlapping_record_count")
        canaries = _count(self.canary_count, "canary_count")
        hits = _count(self.canary_hit_count, "canary_hit_count")
        if overlapping > checked:
            raise SyntheticGovernanceError(
                "overlapping_record_count cannot exceed checked_record_count"
            )
        if hits > canaries:
            raise SyntheticGovernanceError(
                "canary_hit_count cannot exceed canary_count"
            )

    @property
    def overlap_rate(self) -> float:
        """Return the fraction of checked records that overlap the benchmark."""

        if not self.checked_record_count:
            return 0.0
        return self.overlapping_record_count / self.checked_record_count

    def to_dict(self) -> dict[str, Any]:
        """Return counts, rates, and digests only."""

        return {
            "artifact_content_digest": self.artifact_content_digest,
            "benchmark_digest": self.benchmark_digest,
            "benchmark_id": self.benchmark_id,
            "canary_count": self.canary_count,
            "canary_hit_count": self.canary_hit_count,
            "checked_record_count": self.checked_record_count,
            "overlap_rate": _round(self.overlap_rate),
            "overlapping_record_count": self.overlapping_record_count,
        }


def fingerprint_record(text: str) -> str:
    """Return the normalized SHA-256 fingerprint of one record's text.

    Text is NFKC-normalized, case-folded, and whitespace-collapsed with the
    same normalizer the corpus deduplicator uses, so benchmark owners can
    publish fingerprints instead of records.

    Args:
        text: Record text.

    Returns:
        A ``sha256:``-prefixed digest.

    Raises:
        SyntheticGovernanceError: If ``text`` is not a string.
    """

    if not isinstance(text, str):
        raise SyntheticGovernanceError("record text must be a string")
    return hash_text(normalize_passage_text(text))


def measure_benchmark_contamination(
    records: Iterable[str],
    *,
    benchmark_id: str,
    benchmark_digest: str,
    artifact_content_digest: str,
    benchmark_fingerprints: Iterable[str],
    canaries: Iterable[str] = (),
) -> BenchmarkContaminationEvidence:
    """Measure exact benchmark overlap and canary presence locally.

    Records and canaries are processed in memory and never returned. The
    evidence carries counts and digests only. Overlap is exact after
    normalization; near-duplicate and memorization audits are separate.

    Args:
        records: Text of every artifact record to check.
        benchmark_id: Controlled identifier of the benchmark.
        benchmark_digest: SHA-256 digest of the benchmark version.
        artifact_content_digest: Content digest of the artifact checked.
        benchmark_fingerprints: :func:`fingerprint_record` digests of the
            benchmark records.
        canaries: Canary strings planted in the benchmark. Each must have at
            least :data:`MIN_CANARY_LENGTH` characters after normalization.

    Returns:
        Counts-only contamination evidence.

    Raises:
        SyntheticGovernanceError: If a record, fingerprint, or canary is
            malformed. Messages carry positions, never values.
    """

    fingerprints = {
        _digest(item, "benchmark_fingerprints")
        for item in _sequence(benchmark_fingerprints, "benchmark_fingerprints")
    }
    normalized_canaries: set[str] = set()
    for item in _sequence(canaries, "canaries"):
        if not isinstance(item, str):
            raise SyntheticGovernanceError("canaries must be strings")
        normalized = normalize_passage_text(item)
        if len(normalized) < MIN_CANARY_LENGTH:
            raise SyntheticGovernanceError(
                f"canaries must have at least {MIN_CANARY_LENGTH} characters"
            )
        normalized_canaries.add(normalized)

    checked = 0
    overlapping = 0
    hits: set[str] = set()
    for index, text in enumerate(_iterable(records, "records")):
        if not isinstance(text, str):
            raise SyntheticGovernanceError(f"record {index} text must be a string")
        normalized = normalize_passage_text(text)
        checked += 1
        if hash_text(normalized) in fingerprints:
            overlapping += 1
        for canary in normalized_canaries - hits:
            if canary in normalized:
                hits.add(canary)
    return BenchmarkContaminationEvidence(
        benchmark_id=benchmark_id,
        benchmark_digest=benchmark_digest,
        artifact_content_digest=artifact_content_digest,
        checked_record_count=checked,
        overlapping_record_count=overlapping,
        canary_count=len(normalized_canaries),
        canary_hit_count=len(hits),
    )


# ---------------------------------------------------------------------------
# Slice statistics shared by the subgroup and regression reports.
# ---------------------------------------------------------------------------


def _z_score(confidence_level: float) -> float:
    return NormalDist().inv_cdf(0.5 + confidence_level / 2.0)


def _wilson_interval(successes: int, trials: int, z: float) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""

    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    centre = (proportion + z_squared / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def _difference_interval(
    candidate_successes: int,
    candidate_trials: int,
    baseline_successes: int,
    baseline_trials: int,
    z: float,
) -> tuple[float, float, float]:
    """Return Newcombe's hybrid score interval for candidate minus baseline."""

    candidate = candidate_successes / candidate_trials
    baseline = baseline_successes / baseline_trials
    candidate_lower, candidate_upper = _wilson_interval(
        candidate_successes, candidate_trials, z
    )
    baseline_lower, baseline_upper = _wilson_interval(
        baseline_successes, baseline_trials, z
    )
    delta = candidate - baseline
    lower = delta - math.sqrt(
        (candidate - candidate_lower) ** 2 + (baseline_upper - baseline) ** 2
    )
    upper = delta + math.sqrt(
        (candidate_upper - candidate) ** 2 + (baseline - baseline_lower) ** 2
    )
    return delta, max(-1.0, lower), min(1.0, upper)


def _count_band(size: int, minimum_cell_size: int) -> CountBand:
    if size < minimum_cell_size * 2:
        return CountBand.MINIMUM_TO_UNDER_DOUBLE
    if size < minimum_cell_size * 4:
        return CountBand.DOUBLE_TO_UNDER_FOURFOLD
    return CountBand.FOURFOLD_OR_MORE


def _suppressed_indices(
    items: Sequence[_T],
    *,
    family: Callable[[_T], tuple[str, ...]],
    axis: Callable[[_T], str],
    group: Callable[[_T], str],
    size: Callable[[_T], int],
    minimum_cell_size: int,
) -> set[int]:
    """Return indices suppressed by primary and complementary suppression.

    A slice below the minimum cell size is suppressed. When exactly one group
    on a non-overall axis is suppressed, the next smallest group on that axis
    is suppressed too, so the hidden group cannot be recovered by
    subtraction from the remaining groups and the overall slice.
    """

    suppressed = {
        index for index, item in enumerate(items) if size(item) < minimum_cell_size
    }
    buckets: dict[tuple[str, ...], list[int]] = {}
    for index, item in enumerate(items):
        if axis(item) != OVERALL_AXIS:
            buckets.setdefault((*family(item), axis(item)), []).append(index)
    for indices in buckets.values():
        primary = [index for index in indices if index in suppressed]
        released = [index for index in indices if index not in suppressed]
        if len(primary) == 1 and released:
            smallest = min(
                released,
                key=lambda index: (size(items[index]), group(items[index])),
            )
            suppressed.add(smallest)
    return suppressed


def _slice_identity(axis: str, group: str) -> tuple[str, str]:
    axis = _identifier(axis, "axis")
    group = _identifier(group, "group")
    if axis == OVERALL_AXIS and group != OVERALL_GROUP:
        raise SyntheticGovernanceError(
            f"the {OVERALL_AXIS} axis only has the {OVERALL_GROUP} group"
        )
    return axis, group


# ---------------------------------------------------------------------------
# Subgroup quality and bias.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgroupMetric:
    """Aggregate success counts for one metric on one subgroup slice.

    Attributes:
        metric_id: Controlled identifier of a proportion metric.
        axis: Slicing axis such as ``language`` or ``age_band``. Use
            :data:`OVERALL_AXIS` with :data:`OVERALL_GROUP` for the whole set.
        group: Controlled identifier of the subgroup on the axis.
        successes: Count of successful outcomes in the slice.
        trials: Count of evaluated outcomes in the slice.
    """

    metric_id: str
    axis: str
    group: str
    successes: int
    trials: int

    def __post_init__(self) -> None:
        _identifier(self.metric_id, "metric_id")
        _slice_identity(self.axis, self.group)
        if _count(self.successes, "successes") > _count(self.trials, "trials"):
            raise SyntheticGovernanceError("successes cannot exceed trials")


@dataclass(frozen=True)
class SubgroupSlice:
    """One released subgroup estimate with its score interval."""

    metric_id: str
    axis: str
    group: str
    count_band: CountBand
    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, Any]:
        """Return the slice without raw counts."""

        return {
            "axis": self.axis,
            "count_band": self.count_band.value,
            "estimate": _round(self.estimate),
            "group": self.group,
            "lower": _round(self.lower),
            "metric_id": self.metric_id,
            "upper": _round(self.upper),
        }


@dataclass(frozen=True)
class SubgroupDisparity:
    """Spread between the highest and lowest released groups on one axis.

    ``gap_supported`` is true only when the two groups' intervals do not
    overlap, so a gap that the data cannot distinguish from noise is never
    reported as a disparity.
    """

    metric_id: str
    axis: str
    released_group_count: int
    suppressed_group_count: int
    highest_group: str | None
    lowest_group: str | None
    gap: float | None
    gap_supported: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the disparity summary."""

        return {
            "axis": self.axis,
            "gap": None if self.gap is None else _round(self.gap),
            "gap_supported": self.gap_supported,
            "highest_group": self.highest_group,
            "lowest_group": self.lowest_group,
            "metric_id": self.metric_id,
            "released_group_count": self.released_group_count,
            "suppressed_group_count": self.suppressed_group_count,
        }


@dataclass(frozen=True)
class SubgroupQualityReport:
    """Slice-level quality and bias report with small-cell suppression.

    Released slices report a coarse count band, a point estimate, and a score
    interval. Suppressed slices are counted but never named.
    """

    minimum_cell_size: int
    confidence_level: float
    slices: tuple[SubgroupSlice, ...]
    disparities: tuple[SubgroupDisparity, ...]
    suppressed_slice_count: int
    interval_method: str = "wilson_score"
    schema_version: str = SUBGROUP_QUALITY_SCHEMA_VERSION

    @property
    def released_slice_count(self) -> int:
        """Return how many slices are released."""

        return len(self.slices)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, content-free report."""

        return {
            "confidence_level": self.confidence_level,
            "disparities": [item.to_dict() for item in self.disparities],
            "interval_method": self.interval_method,
            "minimum_cell_size": self.minimum_cell_size,
            "released_slice_count": self.released_slice_count,
            "schema_version": self.schema_version,
            "slices": [item.to_dict() for item in self.slices],
            "suppressed_slice_count": self.suppressed_slice_count,
        }

    def to_json(self) -> str:
        """Serialize the report deterministically."""

        return _json(self.to_dict())

    @property
    def report_digest(self) -> str:
        """Return the SHA-256 commitment over the report."""

        return stable_hash(self.to_dict())


def _unique_slices(keys: Iterable[tuple[str, ...]], name: str) -> None:
    seen: set[tuple[str, ...]] = set()
    for key in keys:
        if key in seen:
            raise SyntheticGovernanceError(f"{name} contain a duplicate slice")
        seen.add(key)


def build_subgroup_quality_report(
    metrics: Iterable[SubgroupMetric],
    *,
    minimum_cell_size: int = DEFAULT_MINIMUM_CELL_SIZE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> SubgroupQualityReport:
    """Build a subgroup quality and bias report from aggregate counts.

    Args:
        metrics: Per-slice success counts. Each (metric, axis, group) slice
            may appear once.
        minimum_cell_size: Declared minimum trials for a slice to be
            released. Smaller slices, and one complementary slice where
            needed, are suppressed.
        confidence_level: Two-sided level for Wilson score intervals.

    Returns:
        A report with released slices, per-axis disparities, and the count
        of suppressed slices.

    Raises:
        SyntheticGovernanceError: On malformed inputs or duplicate slices.
    """

    minimum = _minimum_cell_size(minimum_cell_size)
    level = _confidence_level(confidence_level)
    items = tuple(_sequence(metrics, "metrics"))
    if not all(isinstance(item, SubgroupMetric) for item in items):
        raise SyntheticGovernanceError("metrics must be SubgroupMetric values")
    _unique_slices(
        ((item.metric_id, item.axis, item.group) for item in items),
        "subgroup metrics",
    )
    suppressed = _suppressed_indices(
        items,
        family=lambda item: (item.metric_id,),
        axis=lambda item: item.axis,
        group=lambda item: item.group,
        size=lambda item: item.trials,
        minimum_cell_size=minimum,
    )
    z = _z_score(level)
    slices: list[SubgroupSlice] = []
    for index, item in enumerate(items):
        if index in suppressed:
            continue
        lower, upper = _wilson_interval(item.successes, item.trials, z)
        slices.append(
            SubgroupSlice(
                metric_id=item.metric_id,
                axis=item.axis,
                group=item.group,
                count_band=_count_band(item.trials, minimum),
                estimate=item.successes / item.trials,
                lower=lower,
                upper=upper,
            )
        )
    slices.sort(key=lambda item: (item.metric_id, item.axis, item.group))

    axes: dict[tuple[str, str], list[int]] = {}
    for index, item in enumerate(items):
        if item.axis != OVERALL_AXIS:
            axes.setdefault((item.metric_id, item.axis), []).append(index)
    disparities: list[SubgroupDisparity] = []
    for (metric_id, axis), indices in sorted(axes.items()):
        released = [
            entry
            for entry in slices
            if entry.metric_id == metric_id and entry.axis == axis
        ]
        hidden = sum(1 for index in indices if index in suppressed)
        if len(released) < 2:
            disparities.append(
                SubgroupDisparity(
                    metric_id=metric_id,
                    axis=axis,
                    released_group_count=len(released),
                    suppressed_group_count=hidden,
                    highest_group=None,
                    lowest_group=None,
                    gap=None,
                    gap_supported=False,
                )
            )
            continue
        highest = max(released, key=lambda entry: (entry.estimate, entry.group))
        lowest = min(released, key=lambda entry: (entry.estimate, entry.group))
        disparities.append(
            SubgroupDisparity(
                metric_id=metric_id,
                axis=axis,
                released_group_count=len(released),
                suppressed_group_count=hidden,
                highest_group=highest.group,
                lowest_group=lowest.group,
                gap=highest.estimate - lowest.estimate,
                gap_supported=lowest.upper < highest.lower,
            )
        )
    return SubgroupQualityReport(
        minimum_cell_size=minimum,
        confidence_level=level,
        slices=tuple(slices),
        disparities=tuple(disparities),
        suppressed_slice_count=len(suppressed),
    )


# ---------------------------------------------------------------------------
# Downstream task regression.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownstreamComparison:
    """Baseline and candidate counts for one downstream task slice.

    The candidate is the model or pipeline trained with the synthetic
    artifact; the baseline is the reference it must not regress against.

    Attributes:
        task_id: Controlled identifier of the downstream task.
        metric_id: Controlled identifier of a proportion metric.
        baseline_successes: Successful baseline outcomes in the slice.
        baseline_trials: Evaluated baseline outcomes in the slice.
        candidate_successes: Successful candidate outcomes in the slice.
        candidate_trials: Evaluated candidate outcomes in the slice.
        axis: Slicing axis; defaults to :data:`OVERALL_AXIS`.
        group: Subgroup on the axis; defaults to :data:`OVERALL_GROUP`.
    """

    task_id: str
    metric_id: str
    baseline_successes: int
    baseline_trials: int
    candidate_successes: int
    candidate_trials: int
    axis: str = OVERALL_AXIS
    group: str = OVERALL_GROUP

    def __post_init__(self) -> None:
        _identifier(self.task_id, "task_id")
        _identifier(self.metric_id, "metric_id")
        _slice_identity(self.axis, self.group)
        for successes, trials in (
            (self.baseline_successes, self.baseline_trials),
            (self.candidate_successes, self.candidate_trials),
        ):
            if _count(successes, "successes") > _count(trials, "trials"):
                raise SyntheticGovernanceError("successes cannot exceed trials")

    @property
    def smallest_trials(self) -> int:
        """Return the smaller of the baseline and candidate trial counts."""

        return min(self.baseline_trials, self.candidate_trials)


@dataclass(frozen=True)
class RegressionSlice:
    """One released candidate-minus-baseline comparison."""

    task_id: str
    metric_id: str
    axis: str
    group: str
    count_band: CountBand
    baseline_estimate: float
    candidate_estimate: float
    delta: float
    delta_lower: float
    delta_upper: float
    critical: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison without raw counts."""

        return {
            "axis": self.axis,
            "baseline_estimate": _round(self.baseline_estimate),
            "candidate_estimate": _round(self.candidate_estimate),
            "count_band": self.count_band.value,
            "critical": self.critical,
            "delta": _round(self.delta),
            "delta_lower": _round(self.delta_lower),
            "delta_upper": _round(self.delta_upper),
            "group": self.group,
            "metric_id": self.metric_id,
            "task_id": self.task_id,
        }


@dataclass(frozen=True)
class DownstreamRegressionReport:
    """Downstream utility regression report with small-cell suppression.

    A slice is critical when the candidate's point estimate falls at least
    ``critical_drop`` below the baseline. Every slice with enough trials
    counts toward ``critical_slice_count``, including slices hidden only by
    complementary suppression; slices below the minimum cell size cannot be
    assessed and are excluded. Each (task, metric) needs an overall slice at
    or above the minimum cell size, otherwise it counts as insufficient.
    """

    minimum_cell_size: int
    confidence_level: float
    critical_drop: float
    slices: tuple[RegressionSlice, ...]
    suppressed_slice_count: int
    critical_slice_count: int
    task_count: int
    insufficient_task_count: int
    interval_method: str = "newcombe_hybrid_score"
    schema_version: str = DOWNSTREAM_REGRESSION_SCHEMA_VERSION

    @property
    def has_critical_regression(self) -> bool:
        """Return whether any assessable slice regressed critically."""

        return self.critical_slice_count > 0

    @property
    def is_sufficient(self) -> bool:
        """Return whether every task has an assessable overall slice."""

        return self.task_count > 0 and self.insufficient_task_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, content-free report."""

        return {
            "confidence_level": self.confidence_level,
            "critical_drop": self.critical_drop,
            "critical_slice_count": self.critical_slice_count,
            "insufficient_task_count": self.insufficient_task_count,
            "interval_method": self.interval_method,
            "minimum_cell_size": self.minimum_cell_size,
            "schema_version": self.schema_version,
            "slices": [item.to_dict() for item in self.slices],
            "suppressed_slice_count": self.suppressed_slice_count,
            "task_count": self.task_count,
        }

    def to_json(self) -> str:
        """Serialize the report deterministically."""

        return _json(self.to_dict())

    @property
    def report_digest(self) -> str:
        """Return the SHA-256 commitment over the report."""

        return stable_hash(self.to_dict())


def build_downstream_regression_report(
    comparisons: Iterable[DownstreamComparison],
    *,
    minimum_cell_size: int = DEFAULT_MINIMUM_CELL_SIZE,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    critical_drop: float = DEFAULT_CRITICAL_REGRESSION_DROP,
) -> DownstreamRegressionReport:
    """Compare candidate and baseline downstream metrics slice by slice.

    Args:
        comparisons: Per-slice baseline and candidate counts. Each (task,
            metric, axis, group) slice may appear once.
        minimum_cell_size: Declared minimum trials, on both sides, for a
            slice to be assessed and released.
        confidence_level: Two-sided level for the difference interval.
        critical_drop: Absolute drop in the point estimate that counts as a
            critical regression.

    Returns:
        A regression report with released slices and critical counts.

    Raises:
        SyntheticGovernanceError: On malformed inputs or duplicate slices.
    """

    minimum = _minimum_cell_size(minimum_cell_size)
    level = _confidence_level(confidence_level)
    drop = _rate(critical_drop, "critical_drop", allow_zero=False)
    items = tuple(_sequence(comparisons, "comparisons"))
    if not all(isinstance(item, DownstreamComparison) for item in items):
        raise SyntheticGovernanceError(
            "comparisons must be DownstreamComparison values"
        )
    _unique_slices(
        ((item.task_id, item.metric_id, item.axis, item.group) for item in items),
        "downstream comparisons",
    )
    suppressed = _suppressed_indices(
        items,
        family=lambda item: (item.task_id, item.metric_id),
        axis=lambda item: item.axis,
        group=lambda item: item.group,
        size=lambda item: item.smallest_trials,
        minimum_cell_size=minimum,
    )
    z = _z_score(level)
    slices: list[RegressionSlice] = []
    critical_count = 0
    for index, item in enumerate(items):
        if item.smallest_trials < minimum:
            continue
        delta, lower, upper = _difference_interval(
            item.candidate_successes,
            item.candidate_trials,
            item.baseline_successes,
            item.baseline_trials,
            z,
        )
        critical = delta <= -drop + _FLOAT_TOLERANCE
        critical_count += critical
        if index in suppressed:
            continue
        slices.append(
            RegressionSlice(
                task_id=item.task_id,
                metric_id=item.metric_id,
                axis=item.axis,
                group=item.group,
                count_band=_count_band(item.smallest_trials, minimum),
                baseline_estimate=item.baseline_successes / item.baseline_trials,
                candidate_estimate=item.candidate_successes / item.candidate_trials,
                delta=delta,
                delta_lower=lower,
                delta_upper=upper,
                critical=critical,
            )
        )
    slices.sort(key=lambda item: (item.task_id, item.metric_id, item.axis, item.group))
    tasks = {(item.task_id, item.metric_id) for item in items}
    assessable = {
        (item.task_id, item.metric_id)
        for item in items
        if item.axis == OVERALL_AXIS and item.smallest_trials >= minimum
    }
    return DownstreamRegressionReport(
        minimum_cell_size=minimum,
        confidence_level=level,
        critical_drop=drop,
        slices=tuple(slices),
        suppressed_slice_count=len(suppressed),
        critical_slice_count=critical_count,
        task_count=len(tasks),
        insufficient_task_count=len(tasks - assessable),
    )


# ---------------------------------------------------------------------------
# Withdrawal propagation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WithdrawalNotice:
    """Content-free notice that withdraws one lineage node.

    Attributes:
        artifact_hash: Lineage node commitment of the withdrawn artifact.
        reason: Closed withdrawal reason code.
        effective_date: Optional ISO date the withdrawal takes effect.
    """

    artifact_hash: str
    reason: WithdrawalReason
    effective_date: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_hash", _digest(self.artifact_hash, "artifact_hash")
        )
        object.__setattr__(
            self, "reason", _enum(WithdrawalReason, self.reason, "reason")
        )
        object.__setattr__(
            self,
            "effective_date",
            _iso_date(self.effective_date, "effective_date"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the notice as digests and codes only."""

        return {
            "artifact_hash": self.artifact_hash,
            "effective_date": self.effective_date,
            "reason": self.reason.value,
        }

    @property
    def notice_digest(self) -> str:
        """Return the SHA-256 commitment over the notice."""

        return stable_hash(self.to_dict())


@dataclass(frozen=True)
class InvalidatedArtifact:
    """One lineage node invalidated by a withdrawal.

    Attributes:
        artifact_hash: Lineage node commitment of the invalidated node.
        artifact_type: Controlled lineage type of the node.
        distance: Shortest number of lineage edges from a withdrawn node;
            zero for a directly withdrawn node.
        withdrawn_by: Withdrawn node hashes that reach this node.
        reasons: Reason codes of the notices that reach this node.
    """

    artifact_hash: str
    artifact_type: str
    distance: int
    withdrawn_by: tuple[str, ...]
    reasons: tuple[WithdrawalReason, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the invalidation record as digests, codes, and counts."""

        return {
            "artifact_hash": self.artifact_hash,
            "artifact_type": self.artifact_type,
            "distance": self.distance,
            "reasons": [reason.value for reason in self.reasons],
            "withdrawn_by": list(self.withdrawn_by),
        }


@dataclass(frozen=True)
class WithdrawalEvidence:
    """Content-free evidence of how withdrawals propagate through lineage.

    Notices whose node is absent from the lineage graph cannot propagate and
    are counted in ``unresolved_notice_count``; they still invalidate their
    own hash.
    """

    lineage_manifest_hash: str | None
    notices: tuple[WithdrawalNotice, ...]
    invalidated: tuple[InvalidatedArtifact, ...]
    unresolved_notice_count: int
    schema_version: str = SYNTHETIC_WITHDRAWAL_SCHEMA_VERSION

    @property
    def invalidated_hashes(self) -> frozenset[str]:
        """Return every invalidated hash, including unresolved notice targets."""

        return frozenset(item.artifact_hash for item in self.invalidated) | {
            notice.artifact_hash for notice in self.notices
        }

    def invalidates(self, artifact_hash: str) -> bool:
        """Return whether ``artifact_hash`` is withdrawn or descends from one."""

        return _digest(artifact_hash, "artifact_hash") in self.invalidated_hashes

    @property
    def descendant_count(self) -> int:
        """Return how many invalidated nodes were not withdrawn directly."""

        return sum(1 for item in self.invalidated if item.distance > 0)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, content-free evidence."""

        return {
            "descendant_count": self.descendant_count,
            "invalidated": [item.to_dict() for item in self.invalidated],
            "invalidated_count": len(self.invalidated),
            "lineage_manifest_hash": self.lineage_manifest_hash,
            "notice_digests": [notice.notice_digest for notice in self.notices],
            "notices": [notice.to_dict() for notice in self.notices],
            "schema_version": self.schema_version,
            "unresolved_notice_count": self.unresolved_notice_count,
        }

    def to_json(self) -> str:
        """Serialize the evidence deterministically."""

        return _json(self.to_dict())

    @property
    def evidence_digest(self) -> str:
        """Return the SHA-256 commitment over the evidence."""

        return stable_hash(self.to_dict())


def _lineage(
    lineage: ArtifactLineageManifest | Mapping[str, Any] | None,
) -> ArtifactLineageManifest | None:
    if lineage is None or isinstance(lineage, ArtifactLineageManifest):
        return lineage
    return ArtifactLineageManifest.from_mapping(lineage)


def propagate_withdrawal(
    lineage: ArtifactLineageManifest | Mapping[str, Any] | None,
    notices: Iterable[WithdrawalNotice],
) -> WithdrawalEvidence:
    """Invalidate every descendant of the withdrawn lineage nodes.

    Propagation follows every parent reference in the lineage graph,
    including references whose type does not match, so a malformed edge can
    only widen a withdrawal, never narrow it.

    Args:
        lineage: Lineage manifest from
            :mod:`openmed.compliance.artifact_lineage`, its JSON mapping, or
            ``None`` when no graph is available.
        notices: Withdrawal notices.

    Returns:
        Content-free evidence listing every invalidated node.

    Raises:
        SyntheticGovernanceError: If a notice is malformed.
        ArtifactLineageError: If a lineage mapping is malformed.
    """

    manifest = _lineage(lineage)
    resolved_notices = tuple(_sequence(notices, "notices"))
    if not all(isinstance(item, WithdrawalNotice) for item in resolved_notices):
        raise SyntheticGovernanceError("notices must be WithdrawalNotice values")
    ordered = tuple(
        sorted(
            set(resolved_notices),
            key=lambda item: (
                item.artifact_hash,
                item.reason.value,
                item.notice_digest,
            ),
        )
    )
    types: dict[str, str] = {}
    children: dict[str, set[str]] = {}
    if manifest is not None:
        for node in manifest.nodes:
            types.setdefault(node.artifact_hash, node.artifact_type)
            for parent in node.parents:
                children.setdefault(parent.parent_hash, set()).add(node.artifact_hash)

    distance: dict[str, int] = {}
    sources: dict[str, set[str]] = {}
    reasons: dict[str, set[WithdrawalReason]] = {}
    unresolved = 0
    for notice in ordered:
        root = notice.artifact_hash
        if root not in types:
            unresolved += 1
            continue
        frontier = [root]
        depth = {root: 0}
        while frontier:
            next_frontier: list[str] = []
            for current in frontier:
                distance[current] = min(
                    distance.get(current, depth[current]), depth[current]
                )
                sources.setdefault(current, set()).add(root)
                reasons.setdefault(current, set()).add(notice.reason)
                for child in sorted(children.get(current, ())):
                    if child not in depth:
                        depth[child] = depth[current] + 1
                        next_frontier.append(child)
            frontier = next_frontier

    invalidated = tuple(
        InvalidatedArtifact(
            artifact_hash=artifact_hash,
            artifact_type=types[artifact_hash],
            distance=distance[artifact_hash],
            withdrawn_by=tuple(sorted(sources[artifact_hash])),
            reasons=tuple(sorted(reasons[artifact_hash], key=lambda item: item.value)),
        )
        for artifact_hash in sorted(distance)
    )
    return WithdrawalEvidence(
        lineage_manifest_hash=None if manifest is None else manifest.manifest_hash,
        notices=ordered,
        invalidated=invalidated,
        unresolved_notice_count=unresolved,
    )


# ---------------------------------------------------------------------------
# Release validation.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticReleasePolicy:
    """Thresholds applied by :func:`validate_synthetic_release`.

    Attributes:
        minimum_cell_size: Declared minimum trials for any reported slice.
        confidence_level: Two-sided level for reported intervals.
        critical_regression_drop: Point-estimate drop that blocks release.
        max_benchmark_overlap_rate: Highest tolerated benchmark overlap rate.
        required_benchmarks: Benchmarks that must have evidence.
        permitted_data_licenses: Data licenses accepted in addition to the
            permissive SPDX allowlist.
        max_subgroup_gap: When set, a supported subgroup gap above this value
            blocks release; when ``None`` disparities are reported only.
        require_subgroup_report: Whether subgroup metrics are mandatory.
    """

    minimum_cell_size: int = DEFAULT_MINIMUM_CELL_SIZE
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    critical_regression_drop: float = DEFAULT_CRITICAL_REGRESSION_DROP
    max_benchmark_overlap_rate: float = DEFAULT_MAX_BENCHMARK_OVERLAP_RATE
    required_benchmarks: tuple[str, ...] = ()
    permitted_data_licenses: frozenset[str] = DEFAULT_PERMITTED_DATA_LICENSES
    max_subgroup_gap: float | None = None
    require_subgroup_report: bool = True

    def __post_init__(self) -> None:
        _minimum_cell_size(self.minimum_cell_size)
        object.__setattr__(
            self, "confidence_level", _confidence_level(self.confidence_level)
        )
        object.__setattr__(
            self,
            "critical_regression_drop",
            _rate(
                self.critical_regression_drop,
                "critical_regression_drop",
                allow_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "max_benchmark_overlap_rate",
            _rate(
                self.max_benchmark_overlap_rate,
                "max_benchmark_overlap_rate",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "required_benchmarks",
            tuple(
                sorted(
                    {
                        _identifier(item, "required_benchmarks")
                        for item in _sequence(
                            self.required_benchmarks, "required_benchmarks"
                        )
                    }
                )
            ),
        )
        licenses = _sequence(self.permitted_data_licenses, "permitted_data_licenses")
        if not all(
            isinstance(item, str) and 0 < len(item) <= _MAX_LICENSE_ID_LENGTH
            for item in licenses
        ):
            raise SyntheticGovernanceError(
                "permitted_data_licenses must be bounded strings"
            )
        object.__setattr__(self, "permitted_data_licenses", frozenset(licenses))
        if self.max_subgroup_gap is not None:
            object.__setattr__(
                self,
                "max_subgroup_gap",
                _rate(self.max_subgroup_gap, "max_subgroup_gap", allow_zero=False),
            )
        _flag(self.require_subgroup_report, "require_subgroup_report")

    def to_dict(self) -> dict[str, Any]:
        """Return the policy in deterministic form."""

        return {
            "confidence_level": self.confidence_level,
            "critical_regression_drop": self.critical_regression_drop,
            "max_benchmark_overlap_rate": self.max_benchmark_overlap_rate,
            "max_subgroup_gap": self.max_subgroup_gap,
            "minimum_cell_size": self.minimum_cell_size,
            "permitted_data_licenses": sorted(self.permitted_data_licenses),
            "require_subgroup_report": self.require_subgroup_report,
            "required_benchmarks": list(self.required_benchmarks),
        }


@dataclass(frozen=True)
class ReleaseControlFinding:
    """One failed release control, as a closed code and a count."""

    code: ReleaseFindingCode
    count: int = 1

    @property
    def control(self) -> ReleaseControl:
        """Return the release control the finding belongs to."""

        return _FINDING_CONTROLS[self.code]

    def to_dict(self) -> dict[str, Any]:
        """Return the finding as codes and a count."""

        return {
            "code": self.code.value,
            "control": self.control.value,
            "count": self.count,
        }


@dataclass(frozen=True)
class SyntheticReleaseDecision:
    """Fail-closed release decision with content-free evidence.

    ``passed`` is true only when no control produced a finding. The decision
    commits to the label, lineage manifest, license declarations, withdrawal
    propagation, benchmark evidence, and both slice reports through
    :attr:`evidence_digest`.
    """

    findings: tuple[ReleaseControlFinding, ...]
    artifact_hash: str | None
    content_digest: str | None
    label_digest: str | None
    lineage_manifest_hash: str | None
    lineage_closure_count: int
    license_record_digest: str
    withdrawal: WithdrawalEvidence
    benchmark_evidence: tuple[BenchmarkContaminationEvidence, ...]
    regression_report: DownstreamRegressionReport | None
    subgroup_report: SubgroupQualityReport | None
    policy: SyntheticReleasePolicy
    schema_version: str = SYNTHETIC_RELEASE_DECISION_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """Return whether every release control passed."""

        return not self.findings

    @property
    def codes(self) -> tuple[str, ...]:
        """Return the finding codes in deterministic order."""

        return tuple(finding.code.value for finding in self.findings)

    @property
    def failed_controls(self) -> tuple[ReleaseControl, ...]:
        """Return the controls that produced at least one finding."""

        failed = {finding.control for finding in self.findings}
        return tuple(control for control in ReleaseControl if control in failed)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, content-free decision payload."""

        return {
            "artifact_hash": self.artifact_hash,
            "benchmark_evidence": [item.to_dict() for item in self.benchmark_evidence],
            "content_digest": self.content_digest,
            "controls": [control.value for control in ReleaseControl],
            "failed_controls": [control.value for control in self.failed_controls],
            "findings": [finding.to_dict() for finding in self.findings],
            "label_digest": self.label_digest,
            "license_record_digest": self.license_record_digest,
            "lineage_closure_count": self.lineage_closure_count,
            "lineage_manifest_hash": self.lineage_manifest_hash,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "regression_report": (
                None
                if self.regression_report is None
                else self.regression_report.to_dict()
            ),
            "schema_version": self.schema_version,
            "subgroup_report": (
                None if self.subgroup_report is None else self.subgroup_report.to_dict()
            ),
            "withdrawal": {
                "descendant_count": self.withdrawal.descendant_count,
                "evidence_digest": self.withdrawal.evidence_digest,
                "invalidated_count": len(self.withdrawal.invalidated),
                "notice_count": len(self.withdrawal.notices),
                "unresolved_notice_count": self.withdrawal.unresolved_notice_count,
            },
        }

    def to_json(self) -> str:
        """Serialize the decision deterministically."""

        return _json(self.to_dict())

    @property
    def evidence_digest(self) -> str:
        """Return the SHA-256 commitment over the decision payload."""

        return stable_hash(self.to_dict())

    def to_capsule_check(self) -> dict[str, Any]:
        """Return a release-capsule check outcome for this decision.

        The mapping uses the ``status``/``checked_count``/``evidence_digest``
        shape of a release capsule check, with a bare hexadecimal digest.
        """

        return {
            "checked_count": len(ReleaseControl),
            "evidence_digest": self.evidence_digest.removeprefix("sha256:"),
            "status": "pass" if self.passed else "fail",
        }

    def to_gate_check(self) -> GateCheck:
        """Return the decision as a release-gate check.

        Returns:
            A :class:`openmed.eval.release_gates.GateCheck` carrying the
            finding codes and evidence digest only.
        """

        from openmed.eval.release_gates import GateCheck

        return GateCheck(
            gate=SYNTHETIC_RELEASE_GATE,
            passed=self.passed,
            reason="ok" if self.passed else "synthetic_release_controls_failed",
            details={
                "evidence_digest": self.evidence_digest,
                "failed_controls": [control.value for control in self.failed_controls],
                "findings": [finding.to_dict() for finding in self.findings],
            },
        )


class SyntheticReleaseBlockedError(SyntheticGovernanceError):
    """Raised by :func:`assert_synthetic_release` when a control fails."""

    def __init__(self, decision: SyntheticReleaseDecision) -> None:
        self.decision = decision
        super().__init__("synthetic release blocked: " + ", ".join(decision.codes))


class _Findings:
    """Accumulate counts per finding code."""

    def __init__(self) -> None:
        self._counts: dict[ReleaseFindingCode, int] = {}

    def add(self, code: ReleaseFindingCode, count: int = 1) -> None:
        if count > 0:
            self._counts[code] = self._counts.get(code, 0) + count

    def freeze(self) -> tuple[ReleaseControlFinding, ...]:
        order = {code: index for index, code in enumerate(ReleaseFindingCode)}
        return tuple(
            ReleaseControlFinding(code=code, count=count)
            for code, count in sorted(
                self._counts.items(), key=lambda entry: order[entry[0]]
            )
        )


def _check_label(label: SyntheticArtifactLabel, findings: _Findings) -> None:
    if not label.intended_uses:
        findings.add(ReleaseFindingCode.INTENDED_USE_MISSING)
    findings.add(
        ReleaseFindingCode.INTENDED_USE_PROHIBITED, len(label.conflicting_uses)
    )
    if not label.generators:
        findings.add(ReleaseFindingCode.GENERATOR_DISCLOSURE_MISSING)
    findings.add(
        ReleaseFindingCode.MODEL_DIGEST_MISSING,
        sum(
            1
            for item in label.generators
            if item.model_backed and not item.model_digests
        ),
    )


def _ancestor_closure(
    manifest: ArtifactLineageManifest, artifact_hash: str
) -> tuple[set[str], int]:
    """Return the artifact plus every reachable ancestor, and missing parents."""

    by_hash = {}
    for node in manifest.nodes:
        by_hash.setdefault(node.artifact_hash, node)
    closure = {artifact_hash}
    missing: set[str] = set()
    stack = [artifact_hash]
    while stack:
        node = by_hash[stack.pop()]
        for parent in node.parents:
            if parent.parent_hash not in by_hash:
                missing.add(parent.parent_hash)
            elif parent.parent_hash not in closure:
                closure.add(parent.parent_hash)
                stack.append(parent.parent_hash)
    return closure, len(missing)


def _check_lineage(
    label: SyntheticArtifactLabel | None,
    manifest: ArtifactLineageManifest | None,
    findings: _Findings,
) -> set[str]:
    if manifest is None:
        findings.add(ReleaseFindingCode.LINEAGE_MISSING)
        return set()
    diagnostics = manifest.verify()
    if not diagnostics.valid:
        findings.add(
            ReleaseFindingCode.LINEAGE_INTEGRITY_FAILED,
            diagnostics.cycle_count
            + diagnostics.missing_parent_count
            + diagnostics.hash_mismatch_count
            + diagnostics.duplicate_hash_count,
        )
    if label is None:
        return set()
    if label.artifact_hash not in {node.artifact_hash for node in manifest.nodes}:
        findings.add(ReleaseFindingCode.LINEAGE_ARTIFACT_ABSENT)
        return set()
    closure, missing = _ancestor_closure(manifest, label.artifact_hash)
    findings.add(ReleaseFindingCode.LINEAGE_ANCESTOR_MISSING, missing)
    ancestors = closure - {label.artifact_hash}
    findings.add(
        ReleaseFindingCode.GENERATOR_NOT_IN_LINEAGE,
        sum(1 for item in label.generators if item.lineage_node_hash not in ancestors),
    )
    return closure


def _check_licenses(
    closure: set[str],
    records: tuple[LicenseRecord, ...],
    policy: SyntheticReleasePolicy,
    findings: _Findings,
) -> None:
    by_node: dict[str, LicenseRecord] = {}
    conflicts: set[str] = set()
    for record in records:
        existing = by_node.setdefault(record.node_hash, record)
        if existing != record:
            conflicts.add(record.node_hash)
    findings.add(ReleaseFindingCode.LICENSE_CONFLICT, len(conflicts & closure))
    missing = not_permitted = no_redistribution = 0
    for node_hash in closure:
        record = by_node.get(node_hash)
        if record is None:
            missing += 1
            continue
        if node_hash in conflicts:
            continue
        if not _license_permitted(record.license_id, policy.permitted_data_licenses):
            not_permitted += 1
        if not record.redistribution_allowed:
            no_redistribution += 1
    findings.add(ReleaseFindingCode.LICENSE_MISSING, missing)
    findings.add(ReleaseFindingCode.LICENSE_NOT_PERMITTED, not_permitted)
    findings.add(
        ReleaseFindingCode.LICENSE_REDISTRIBUTION_DISALLOWED, no_redistribution
    )


def _check_contamination(
    label: SyntheticArtifactLabel | None,
    evidence: tuple[BenchmarkContaminationEvidence, ...],
    policy: SyntheticReleasePolicy,
    findings: _Findings,
) -> None:
    if not evidence:
        findings.add(ReleaseFindingCode.BENCHMARK_EVIDENCE_MISSING)
    checked = {item.benchmark_id for item in evidence}
    findings.add(
        ReleaseFindingCode.REQUIRED_BENCHMARK_MISSING,
        len(set(policy.required_benchmarks) - checked),
    )
    for item in evidence:
        if label is not None and item.artifact_content_digest != label.content_digest:
            findings.add(ReleaseFindingCode.BENCHMARK_EVIDENCE_MISMATCH)
        if not item.checked_record_count:
            findings.add(ReleaseFindingCode.BENCHMARK_EVIDENCE_EMPTY)
        findings.add(
            ReleaseFindingCode.BENCHMARK_CANARY_DETECTED, item.canary_hit_count
        )
        if item.overlap_rate > policy.max_benchmark_overlap_rate + _FLOAT_TOLERANCE:
            findings.add(ReleaseFindingCode.BENCHMARK_OVERLAP_EXCEEDED)


def _license_record_digest(records: tuple[LicenseRecord, ...]) -> str:
    return stable_hash(
        sorted(
            (record.to_dict() for record in records),
            key=lambda item: (item["node_hash"], _json(item)),
        )
    )


def validate_synthetic_release(
    label: SyntheticArtifactLabel | None,
    lineage: ArtifactLineageManifest | Mapping[str, Any] | None,
    *,
    licenses: Iterable[LicenseRecord] = (),
    benchmark_evidence: Iterable[BenchmarkContaminationEvidence] = (),
    regression: Iterable[DownstreamComparison] | None = None,
    subgroup_metrics: Iterable[SubgroupMetric] | None = None,
    withdrawals: Iterable[WithdrawalNotice] = (),
    policy: SyntheticReleasePolicy | None = None,
) -> SyntheticReleaseDecision:
    """Validate a synthetic artifact against every release control.

    Validation fails closed. The decision does not pass when the label is
    missing or incomplete; the lineage graph is missing, corrupt, or does not
    reach the artifact, its generators, and every ancestor; any node in the
    artifact's lineage lacks a permitted, redistributable license; the
    artifact or any ancestor is withdrawn; benchmark evidence is missing,
    empty, for different content, or shows canaries or overlap; downstream
    regression evidence is missing, insufficient, or critical; or subgroup
    evidence is missing or, when a gap limit is set, shows a supported gap
    above it.

    Args:
        label: Machine-readable synthetic label, or ``None`` if unlabeled.
        lineage: Lineage manifest containing the artifact and its ancestors.
        licenses: License declarations for lineage nodes.
        benchmark_evidence: Benchmark canary and overlap evidence.
        regression: Downstream task comparisons for the artifact.
        subgroup_metrics: Subgroup quality counts for the artifact.
        withdrawals: Withdrawal notices to propagate through the lineage.
        policy: Release thresholds; defaults to :class:`SyntheticReleasePolicy`.

    Returns:
        A content-free decision with findings and slice reports.

    Raises:
        SyntheticGovernanceError: If an input is malformed rather than
            failing a control.
    """

    resolved_policy = policy if policy is not None else SyntheticReleasePolicy()
    if not isinstance(resolved_policy, SyntheticReleasePolicy):
        raise SyntheticGovernanceError("policy must be a SyntheticReleasePolicy")
    if label is not None and not isinstance(label, SyntheticArtifactLabel):
        raise SyntheticGovernanceError("label must be a SyntheticArtifactLabel")
    manifest = _lineage(lineage)
    license_records = tuple(_sequence(licenses, "licenses"))
    if not all(isinstance(item, LicenseRecord) for item in license_records):
        raise SyntheticGovernanceError("licenses must be LicenseRecord values")
    evidence = tuple(_sequence(benchmark_evidence, "benchmark_evidence"))
    if not all(isinstance(item, BenchmarkContaminationEvidence) for item in evidence):
        raise SyntheticGovernanceError(
            "benchmark_evidence must be BenchmarkContaminationEvidence values"
        )
    evidence = tuple(sorted(set(evidence), key=lambda item: _json(item.to_dict())))

    findings = _Findings()
    if label is None:
        findings.add(ReleaseFindingCode.LABEL_MISSING)
    else:
        _check_label(label, findings)

    closure = _check_lineage(label, manifest, findings)
    _check_licenses(closure, license_records, resolved_policy, findings)

    withdrawal = propagate_withdrawal(manifest, withdrawals)
    if label is not None and withdrawal.invalidates(label.artifact_hash):
        findings.add(ReleaseFindingCode.ARTIFACT_WITHDRAWN)

    _check_contamination(label, evidence, resolved_policy, findings)

    regression_report = None
    comparisons = (
        None if regression is None else tuple(_sequence(regression, "regression"))
    )
    if not comparisons:
        findings.add(ReleaseFindingCode.REGRESSION_EVIDENCE_MISSING)
    else:
        regression_report = build_downstream_regression_report(
            comparisons,
            minimum_cell_size=resolved_policy.minimum_cell_size,
            confidence_level=resolved_policy.confidence_level,
            critical_drop=resolved_policy.critical_regression_drop,
        )
        findings.add(
            ReleaseFindingCode.REGRESSION_EVIDENCE_INSUFFICIENT,
            regression_report.insufficient_task_count,
        )
        findings.add(
            ReleaseFindingCode.CRITICAL_REGRESSION,
            regression_report.critical_slice_count,
        )

    subgroup_report = None
    metrics = (
        None
        if subgroup_metrics is None
        else tuple(_sequence(subgroup_metrics, "subgroup_metrics"))
    )
    if not metrics:
        if resolved_policy.require_subgroup_report:
            findings.add(ReleaseFindingCode.SUBGROUP_REPORT_MISSING)
    else:
        subgroup_report = build_subgroup_quality_report(
            metrics,
            minimum_cell_size=resolved_policy.minimum_cell_size,
            confidence_level=resolved_policy.confidence_level,
        )
        if not subgroup_report.released_slice_count:
            findings.add(ReleaseFindingCode.SUBGROUP_EVIDENCE_INSUFFICIENT)
        if resolved_policy.max_subgroup_gap is not None:
            findings.add(
                ReleaseFindingCode.SUBGROUP_DISPARITY_EXCEEDED,
                sum(
                    1
                    for item in subgroup_report.disparities
                    if item.gap_supported
                    and item.gap is not None
                    and item.gap > resolved_policy.max_subgroup_gap + _FLOAT_TOLERANCE
                ),
            )

    return SyntheticReleaseDecision(
        findings=findings.freeze(),
        artifact_hash=None if label is None else label.artifact_hash,
        content_digest=None if label is None else label.content_digest,
        label_digest=None if label is None else label.label_digest,
        lineage_manifest_hash=None if manifest is None else manifest.manifest_hash,
        lineage_closure_count=len(closure),
        license_record_digest=_license_record_digest(license_records),
        withdrawal=withdrawal,
        benchmark_evidence=evidence,
        regression_report=regression_report,
        subgroup_report=subgroup_report,
        policy=resolved_policy,
    )


def assert_synthetic_release(
    label: SyntheticArtifactLabel | None,
    lineage: ArtifactLineageManifest | Mapping[str, Any] | None,
    **kwargs: Any,
) -> SyntheticReleaseDecision:
    """Validate a release and raise when any control fails.

    Accepts the same arguments as :func:`validate_synthetic_release`.

    Returns:
        The passing decision.

    Raises:
        SyntheticReleaseBlockedError: If any control fails. The message lists
            finding codes only and the decision is attached.
    """

    decision = validate_synthetic_release(label, lineage, **kwargs)
    if not decision.passed:
        raise SyntheticReleaseBlockedError(decision)
    return decision
