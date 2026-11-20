"""Patient-level site holdout splits committed before federated training.

Each site assigns opaque patient keys to a train or holdout partition with a
keyed hash under a site-held secret, then commits holdout membership with the
sealed holdout-commitment primitives in
:mod:`openmed.eval.governance.holdout_commitment`. Only the commitment digest
and coarse patient-count bands leave the site. Patient keys, the site secret,
membership tags and note identifiers stay local and never appear in exported
evidence, error messages or ``repr`` output.

A digest-only :class:`FederatedHoldoutLedger` refuses re-commitment within a
federation unless an explicit rotation is recorded, and a recorded rotation
invalidates evaluation evidence that cites the retired commitment. All
operations are local and perform no network requests.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar

from openmed.eval.governance.holdout_commitment import (
    HoldoutCommitment,
    commit_holdout_manifests,
)

from .federated_status import DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE

FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION = (
    "openmed.training.federated_holdout_split.evidence.v1"
)
FEDERATED_HOLDOUT_LEDGER_SCHEMA_VERSION = (
    "openmed.training.federated_holdout_split.ledger.v1"
)
FEDERATED_HOLDOUT_VERIFICATION_SCHEMA_VERSION = (
    "openmed.training.federated_holdout_split.verification.v1"
)
MINIMUM_SITE_SECRET_BYTES: Final = 16

_SCALE: Final = 10_000
_SHA256_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PUBLIC_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_ASSIGN_DOMAIN = b"openmed.training.federated_holdout_split.assign.v1\0"
_MEMBER_DOMAIN = b"openmed.training.federated_holdout_split.member.v1\0"
_POLICY_DOMAIN = b"openmed.training.federated_holdout_split.policy.v1\0"
_BINDING_DOMAIN = b"openmed.training.federated_holdout_split.binding.v1\0"
_SECRET_DOMAIN = b"openmed.training.federated_holdout_split.secret.v1\0"
_EVIDENCE_KEYS: Final = frozenset(
    {
        "commitment_digest",
        "holdout_patient_count_band",
        "schema_version",
        "train_patient_count_band",
    }
)
_LEDGER_KEYS: Final = frozenset({"federation_id", "schema_version", "sites"})
_SITE_KEYS: Final = frozenset({"active_commitment_digest", "rotations"})
_ROTATION_KEYS: Final = frozenset({"reason", "retired_commitment_digest"})
_CITATION_FIELD: Final = "holdout_commitment_digest"

_Example = TypeVar("_Example")


class HoldoutSplitReasonCode(str, Enum):
    """Stable, value-free reason codes for split, ledger and citation checks."""

    VALID = "valid"
    INVALID_SITE_SECRET = "invalid_site_secret"
    INVALID_PATIENT_KEY = "invalid_patient_key"
    INVALID_FEDERATION_ID = "invalid_federation_id"
    INVALID_HOLDOUT_FRACTION = "invalid_holdout_fraction"
    INVALID_ROTATION_EPOCH = "invalid_rotation_epoch"
    INVALID_MINIMUM_GROUP_SIZE = "invalid_minimum_group_size"
    INVALID_SITE_REF = "invalid_site_ref"
    INVALID_EVIDENCE = "invalid_evidence"
    INVALID_ROTATION_REASON = "invalid_rotation_reason"
    INVALID_LEDGER_STATE = "invalid_ledger_state"
    EMPTY_PARTITION = "empty_partition"
    HOLDOUT_MEMBER_IN_TRAINING = "holdout_member_in_training"
    UNASSIGNED_PATIENT = "unassigned_patient"
    RECOMMITMENT_WITHOUT_ROTATION = "recommitment_without_rotation"
    RETIRED_COMMITMENT_REUSED = "retired_commitment_reused"
    HOLDOUT_NOT_COMMITTED = "holdout_not_committed"
    EVALUATION_COMMITMENT_MISSING = "evaluation_commitment_missing"
    EVALUATION_COMMITMENT_MISMATCH = "evaluation_commitment_mismatch"
    HOLDOUT_COMMITMENT_ROTATED = "holdout_commitment_rotated"

    def __str__(self) -> str:
        return self.value


class HoldoutSplitPartition(str, Enum):
    """The fixed partition a patient belongs to for a whole federation."""

    TRAIN = "train"
    HOLDOUT = "holdout"

    def __str__(self) -> str:
        return self.value


class HoldoutPatientCountBand(str, Enum):
    """Coarse patient-count bands; exact partition sizes never leave a site."""

    SUPPRESSED = "suppressed"
    UNDER_100 = "under_100"
    HUNDREDS = "100_to_999"
    THOUSANDS = "1000_to_9999"
    TEN_THOUSANDS_OR_MORE = "10000_or_more"

    def __str__(self) -> str:
        return self.value


class HoldoutRotationReason(str, Enum):
    """Controlled reasons recorded when a site holdout split is rotated."""

    SECRET_COMPROMISED = "secret_compromised"
    COHORT_REFRESH = "cohort_refresh"
    POLICY_CHANGE = "policy_change"
    OPERATOR_REQUESTED = "operator_requested"

    def __str__(self) -> str:
        return self.value


class FederatedHoldoutSplitError(ValueError):
    """Raised with a stable reason code and no caller-supplied values.

    Attributes:
        code: The controlled reason code; ``str(error)`` equals its value.
    """

    def __init__(self, code: HoldoutSplitReasonCode) -> None:
        self.code = code
        super().__init__(code.value)


def _fail(code: HoldoutSplitReasonCode) -> FederatedHoldoutSplitError:
    return FederatedHoldoutSplitError(code)


def _is_sha256(value: object) -> bool:
    return type(value) is str and _SHA256_RE.fullmatch(value) is not None


def _is_public_id(value: object) -> bool:
    return type(value) is str and _PUBLIC_ID_RE.fullmatch(value) is not None


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _patient_key_bytes(patient_key: object) -> bytes:
    encoded: bytes | None = None
    if type(patient_key) is bytes:
        encoded = patient_key
    elif type(patient_key) is str:
        try:
            encoded = patient_key.encode("utf-8")
        except UnicodeEncodeError:
            encoded = None
    if not encoded:
        raise _fail(HoldoutSplitReasonCode.INVALID_PATIENT_KEY)
    return encoded


def _keyed(secret: bytes, domain: bytes, *parts: bytes) -> bytes:
    message = domain + b"\0".join(len(part).to_bytes(8, "big") + part for part in parts)
    return hmac.new(secret, message, hashlib.sha256).digest()


def _keyed_digest(secret: bytes, domain: bytes, *parts: bytes) -> str:
    return "sha256:" + _keyed(secret, domain, *parts).hex()


def _count_band(count: int, minimum_group_size: int) -> HoldoutPatientCountBand:
    if count < minimum_group_size:
        return HoldoutPatientCountBand.SUPPRESSED
    if count < 100:
        return HoldoutPatientCountBand.UNDER_100
    if count < 1_000:
        return HoldoutPatientCountBand.HUNDREDS
    if count < 10_000:
        return HoldoutPatientCountBand.THOUSANDS
    return HoldoutPatientCountBand.TEN_THOUSANDS_OR_MORE


@dataclass(frozen=True, slots=True)
class SiteHoldoutEvidence:
    """The only split artifact a site may export to its federation.

    Attributes:
        commitment_digest: Digest of the sealed holdout-membership commitment.
        holdout_patient_count_band: Coarse band for the holdout patient count.
        train_patient_count_band: Coarse band for the train patient count.
        schema_version: Evidence schema identifier.
    """

    commitment_digest: str
    holdout_patient_count_band: HoldoutPatientCountBand
    train_patient_count_band: HoldoutPatientCountBand
    schema_version: str = FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION
            or not _is_sha256(self.commitment_digest)
            or not isinstance(self.holdout_patient_count_band, HoldoutPatientCountBand)
            or not isinstance(self.train_patient_count_band, HoldoutPatientCountBand)
        ):
            raise _fail(HoldoutSplitReasonCode.INVALID_EVIDENCE)

    def to_dict(self) -> dict[str, str]:
        """Return the canonical JSON-compatible evidence document."""
        return {
            "commitment_digest": self.commitment_digest,
            "holdout_patient_count_band": self.holdout_patient_count_band.value,
            "schema_version": self.schema_version,
            "train_patient_count_band": self.train_patient_count_band.value,
        }

    def to_json(self) -> str:
        """Return stable compact JSON for transfer to the federation."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> SiteHoldoutEvidence:
        """Parse an evidence document, failing closed on any extra field.

        Args:
            document: A mapping produced by :meth:`to_dict`.

        Returns:
            The validated evidence.

        Raises:
            FederatedHoldoutSplitError: With ``invalid_evidence`` when the
                document has unexpected keys or values.
        """
        if not isinstance(document, Mapping) or set(document) != _EVIDENCE_KEYS:
            raise _fail(HoldoutSplitReasonCode.INVALID_EVIDENCE)
        bands = {item.value: item for item in HoldoutPatientCountBand}
        holdout_band = bands.get(_str_or_none(document["holdout_patient_count_band"]))
        train_band = bands.get(_str_or_none(document["train_patient_count_band"]))
        if holdout_band is None or train_band is None:
            raise _fail(HoldoutSplitReasonCode.INVALID_EVIDENCE)
        return cls(
            commitment_digest=document["commitment_digest"],
            holdout_patient_count_band=holdout_band,
            train_patient_count_band=train_band,
            schema_version=document["schema_version"],
        )


def _str_or_none(value: object) -> str | None:
    return value if type(value) is str else None


class SiteHoldoutSplit:
    """A site-local, fixed patient-level split and its sealed commitment.

    Instances hold the site secret and keyed membership tags, so they must
    never be exported. ``repr`` shows only the commitment digest and count
    bands, and pickling is refused. Build instances with
    :func:`build_site_holdout_split`.
    """

    __slots__ = (
        "_site_secret",
        "_federation_bytes",
        "_epoch_bytes",
        "_holdout_tags",
        "_train_tags",
        "_commitment",
        "_evidence",
    )

    def __init__(
        self,
        *,
        site_secret: bytes,
        federation_id: str,
        rotation_epoch: int,
        holdout_tags: frozenset[bytes],
        train_tags: frozenset[bytes],
        commitment: HoldoutCommitment,
        evidence: SiteHoldoutEvidence,
    ) -> None:
        self._site_secret = site_secret
        self._federation_bytes = federation_id.encode("ascii")
        self._epoch_bytes = str(rotation_epoch).encode("ascii")
        self._holdout_tags = holdout_tags
        self._train_tags = train_tags
        self._commitment = commitment
        self._evidence = evidence

    def __repr__(self) -> str:
        evidence = self._evidence
        holdout_band = evidence.holdout_patient_count_band.value
        train_band = evidence.train_patient_count_band.value
        return (
            "SiteHoldoutSplit("
            f"commitment_digest={evidence.commitment_digest!r}, "
            f"holdout_patient_count_band={holdout_band!r}, "
            f"train_patient_count_band={train_band!r})"
        )

    __str__ = __repr__

    def __reduce__(self) -> Any:
        raise TypeError("SiteHoldoutSplit holds site secrets and cannot be pickled")

    @property
    def commitment_digest(self) -> str:
        """The public digest of the sealed holdout-membership commitment."""
        return self._commitment.commitment_digest

    def export_evidence(self) -> SiteHoldoutEvidence:
        """Return the only artifact that may leave the site.

        Returns:
            Evidence containing the commitment digest and count bands only.
        """
        return self._evidence

    def _tag(self, patient_key: object) -> bytes:
        return _keyed(
            self._site_secret,
            _MEMBER_DOMAIN,
            self._federation_bytes,
            self._epoch_bytes,
            _patient_key_bytes(patient_key),
        )

    def partition_of(self, patient_key: str | bytes) -> HoldoutSplitPartition:
        """Return the committed partition for one opaque patient key.

        Args:
            patient_key: The opaque patient key used when the split was built.

        Returns:
            The fixed partition for that patient.

        Raises:
            FederatedHoldoutSplitError: With ``invalid_patient_key`` or
                ``unassigned_patient`` when the key was not committed.
        """
        tag = self._tag(patient_key)
        if tag in self._holdout_tags:
            return HoldoutSplitPartition.HOLDOUT
        if tag in self._train_tags:
            return HoldoutSplitPartition.TRAIN
        raise _fail(HoldoutSplitReasonCode.UNASSIGNED_PATIENT)

    def require_training_patient(self, patient_key: str | bytes) -> None:
        """Refuse a patient that is in the committed holdout or unassigned.

        Args:
            patient_key: The opaque patient key of a candidate training example.

        Raises:
            FederatedHoldoutSplitError: With ``holdout_member_in_training``,
                ``unassigned_patient`` or ``invalid_patient_key``.
        """
        if self.partition_of(patient_key) is not HoldoutSplitPartition.TRAIN:
            raise _fail(HoldoutSplitReasonCode.HOLDOUT_MEMBER_IN_TRAINING)

    def admit_training_examples(
        self,
        examples: Iterable[_Example],
        *,
        patient_key_of: Callable[[_Example], str | bytes],
    ) -> tuple[_Example, ...]:
        """Validate a whole training batch before any example is released.

        This is the training-side exclusion port: a batch that contains even
        one committed holdout member, or a patient outside the committed
        split, is refused in full rather than filtered.

        Args:
            examples: Site-local training examples, such as notes.
            patient_key_of: Returns the opaque patient key of one example.

        Returns:
            The admitted examples in input order.

        Raises:
            FederatedHoldoutSplitError: With ``holdout_member_in_training``,
                ``unassigned_patient`` or ``invalid_patient_key``.
        """
        admitted = tuple(examples)
        for example in admitted:
            self.require_training_patient(patient_key_of(example))
        return admitted

    def partition_examples(
        self,
        examples: Iterable[_Example],
        *,
        patient_key_of: Callable[[_Example], str | bytes],
    ) -> tuple[tuple[_Example, ...], tuple[_Example, ...]]:
        """Group site-local examples by their patient's committed partition.

        Args:
            examples: Site-local examples, such as notes.
            patient_key_of: Returns the opaque patient key of one example.

        Returns:
            ``(train, holdout)`` tuples in input order. Every example of one
            patient lands in the same tuple.

        Raises:
            FederatedHoldoutSplitError: With ``unassigned_patient`` or
                ``invalid_patient_key``.
        """
        train: list[_Example] = []
        holdout: list[_Example] = []
        for example in examples:
            partition = self.partition_of(patient_key_of(example))
            (holdout if partition is HoldoutSplitPartition.HOLDOUT else train).append(
                example
            )
        return tuple(train), tuple(holdout)


def _validate_fraction(holdout_fraction: object) -> int:
    if type(holdout_fraction) is not float or not math.isfinite(holdout_fraction):
        raise _fail(HoldoutSplitReasonCode.INVALID_HOLDOUT_FRACTION)
    scaled = round(holdout_fraction * _SCALE)
    if not 0 < scaled < _SCALE:
        raise _fail(HoldoutSplitReasonCode.INVALID_HOLDOUT_FRACTION)
    return scaled


def _unique_patient_keys(patient_keys: object) -> tuple[bytes, ...]:
    if isinstance(patient_keys, (str, bytes)) or not isinstance(patient_keys, Iterable):
        raise _fail(HoldoutSplitReasonCode.INVALID_PATIENT_KEY)
    unique: dict[bytes, None] = {}
    for patient_key in patient_keys:
        unique[_patient_key_bytes(patient_key)] = None
    return tuple(unique)


def build_site_holdout_split(
    patient_keys: Iterable[str | bytes],
    *,
    site_secret: bytes,
    federation_id: str,
    holdout_fraction: float = 0.2,
    rotation_epoch: int = 0,
    minimum_group_size: int = DEFAULT_FEDERATED_MINIMUM_GROUP_SIZE,
) -> SiteHoldoutSplit:
    """Assign patients to train or holdout and commit holdout membership.

    Assignment is a keyed HMAC-SHA256 of each opaque patient key under the
    site secret, bound to the federation and rotation epoch, so every note of
    a patient shares one partition and the split is reproducible only with
    the secret. Holdout membership is committed with
    :func:`~openmed.eval.governance.holdout_commitment.commit_holdout_manifests`
    over keyed membership tags; the split policy, federation binding and a
    keyed secret fingerprint fill the remaining manifests.

    Args:
        patient_keys: Opaque patient keys for every site example; repeated
            keys from multi-note patients are collapsed.
        site_secret: Site-held secret of at least
            :data:`MINIMUM_SITE_SECRET_BYTES` bytes. It never leaves the site.
        federation_id: Public, lowercase federation identifier.
        holdout_fraction: Target holdout share, strictly between 0 and 1.
        rotation_epoch: Non-negative epoch, incremented on each recorded
            rotation.
        minimum_group_size: Counts below this are reported as ``suppressed``.

    Returns:
        A site-local split whose :meth:`SiteHoldoutSplit.export_evidence`
        result is the only artifact that may leave the site.

    Raises:
        FederatedHoldoutSplitError: With a controlled reason code when an input
            is invalid or either partition would be empty.
    """
    if type(site_secret) is not bytes or len(site_secret) < MINIMUM_SITE_SECRET_BYTES:
        raise _fail(HoldoutSplitReasonCode.INVALID_SITE_SECRET)
    if not _is_public_id(federation_id):
        raise _fail(HoldoutSplitReasonCode.INVALID_FEDERATION_ID)
    fraction_scaled = _validate_fraction(holdout_fraction)
    if type(rotation_epoch) is not int or rotation_epoch < 0:
        raise _fail(HoldoutSplitReasonCode.INVALID_ROTATION_EPOCH)
    if type(minimum_group_size) is not int or minimum_group_size < 2:
        raise _fail(HoldoutSplitReasonCode.INVALID_MINIMUM_GROUP_SIZE)
    keys = _unique_patient_keys(patient_keys)

    federation_bytes = federation_id.encode("ascii")
    epoch_bytes = str(rotation_epoch).encode("ascii")
    holdout_tags: set[bytes] = set()
    train_tags: set[bytes] = set()
    for key in keys:
        draw = _keyed(site_secret, _ASSIGN_DOMAIN, federation_bytes, epoch_bytes, key)
        tag = _keyed(site_secret, _MEMBER_DOMAIN, federation_bytes, epoch_bytes, key)
        if int.from_bytes(draw[:8], "big") % _SCALE < fraction_scaled:
            holdout_tags.add(tag)
        else:
            train_tags.add(tag)
    if not holdout_tags or not train_tags:
        raise _fail(HoldoutSplitReasonCode.EMPTY_PARTITION)

    policy = _canonical_json(
        {
            "holdout_fraction_scaled": fraction_scaled,
            "holdout_patient_count": len(holdout_tags),
            "schema_version": FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION,
            "train_patient_count": len(train_tags),
        }
    ).encode("ascii")
    manifests = {
        "case": tuple(sorted("sha256:" + tag.hex() for tag in holdout_tags)),
        "label": (_keyed_digest(site_secret, _POLICY_DOMAIN, policy),),
        "template": (
            _keyed_digest(site_secret, _BINDING_DOMAIN, federation_bytes, epoch_bytes),
        ),
        "randomization": (
            _keyed_digest(site_secret, _SECRET_DOMAIN, federation_bytes, epoch_bytes),
        ),
    }
    commitment = commit_holdout_manifests(
        f"{federation_id}.rotation-{rotation_epoch}", manifests
    )
    evidence = SiteHoldoutEvidence(
        commitment_digest=commitment.commitment_digest,
        holdout_patient_count_band=_count_band(len(holdout_tags), minimum_group_size),
        train_patient_count_band=_count_band(len(train_tags), minimum_group_size),
    )
    return SiteHoldoutSplit(
        site_secret=site_secret,
        federation_id=federation_id,
        rotation_epoch=rotation_epoch,
        holdout_tags=frozenset(holdout_tags),
        train_tags=frozenset(train_tags),
        commitment=commitment,
        evidence=evidence,
    )


@dataclass(frozen=True, slots=True)
class HoldoutSplitVerification:
    """A value-free result for evaluation-evidence citation checks.

    Attributes:
        valid: Whether the evidence cites the site's active commitment.
        reason_codes: Controlled reason codes; never caller values.
        schema_version: Verification schema identifier.
    """

    valid: bool
    reason_codes: tuple[str, ...]
    schema_version: str = FEDERATED_HOLDOUT_VERIFICATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report containing no caller values."""
        return {
            "reason_codes": list(self.reason_codes),
            "schema_version": self.schema_version,
            "valid": self.valid,
        }


def _verification(code: HoldoutSplitReasonCode) -> HoldoutSplitVerification:
    return HoldoutSplitVerification(
        valid=code is HoldoutSplitReasonCode.VALID, reason_codes=(code.value,)
    )


@dataclass(slots=True)
class _SiteRecord:
    active: str | None
    rotations: list[tuple[str, HoldoutRotationReason]]

    def retired(self) -> frozenset[str]:
        return frozenset(digest for digest, _ in self.rotations)


class FederatedHoldoutLedger:
    """Digest-only record of each site's active and retired holdout commitments.

    The ledger never sees patient keys or secrets. A site may register one
    commitment; re-registering the same digest is idempotent, any different
    digest is refused until :meth:`record_rotation` retires the active one,
    and a retired digest can never become active again.

    Args:
        federation_id: Public, lowercase federation identifier.
    """

    __slots__ = ("_federation_id", "_sites", "_lock")

    def __init__(self, federation_id: str) -> None:
        if not _is_public_id(federation_id):
            raise _fail(HoldoutSplitReasonCode.INVALID_FEDERATION_ID)
        self._federation_id = federation_id
        self._sites: dict[str, _SiteRecord] = {}
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"FederatedHoldoutLedger(site_count={len(self._sites)})"

    @property
    def federation_id(self) -> str:
        """The public federation identifier this ledger governs."""
        return self._federation_id

    @staticmethod
    def _site_ref(site_ref: object) -> str:
        if not _is_public_id(site_ref):
            raise _fail(HoldoutSplitReasonCode.INVALID_SITE_REF)
        return site_ref  # type: ignore[return-value]

    def register(
        self, site_ref: str, evidence: SiteHoldoutEvidence | Mapping[str, Any]
    ) -> None:
        """Record a site's commitment before its first federated round.

        Args:
            site_ref: Coordinator-assigned opaque site reference.
            evidence: Exported site evidence or its parsed JSON mapping.

        Raises:
            FederatedHoldoutSplitError: With ``recommitment_without_rotation``
                when a different commitment is already active,
                ``retired_commitment_reused`` for a rotated-out digest, or an
                ``invalid_*`` code for malformed input.
        """
        site = self._site_ref(site_ref)
        parsed = (
            evidence
            if isinstance(evidence, SiteHoldoutEvidence)
            else SiteHoldoutEvidence.from_dict(evidence)
        )
        digest = parsed.commitment_digest
        with self._lock:
            record = self._sites.get(site)
            if record is None:
                self._sites[site] = _SiteRecord(active=digest, rotations=[])
                return
            if digest in record.retired():
                raise _fail(HoldoutSplitReasonCode.RETIRED_COMMITMENT_REUSED)
            if record.active is None:
                record.active = digest
                return
            if not hmac.compare_digest(record.active, digest):
                raise _fail(HoldoutSplitReasonCode.RECOMMITMENT_WITHOUT_ROTATION)

    def record_rotation(self, site_ref: str, reason: HoldoutRotationReason) -> None:
        """Retire a site's active commitment so a new split may be registered.

        Evaluation evidence citing the retired digest is invalid afterwards.

        Args:
            site_ref: Coordinator-assigned opaque site reference.
            reason: Controlled rotation reason.

        Raises:
            FederatedHoldoutSplitError: With ``holdout_not_committed`` when the
                site has no active commitment, or ``invalid_rotation_reason``.
        """
        site = self._site_ref(site_ref)
        if not isinstance(reason, HoldoutRotationReason):
            raise _fail(HoldoutSplitReasonCode.INVALID_ROTATION_REASON)
        with self._lock:
            record = self._sites.get(site)
            if record is None or record.active is None:
                raise _fail(HoldoutSplitReasonCode.HOLDOUT_NOT_COMMITTED)
            record.rotations.append((record.active, reason))
            record.active = None

    def require_active_commitment(self, site_ref: str) -> str:
        """Return the active digest or refuse to start training for the site.

        Args:
            site_ref: Coordinator-assigned opaque site reference.

        Returns:
            The site's active commitment digest.

        Raises:
            FederatedHoldoutSplitError: With ``holdout_not_committed`` before
                registration or while a rotation awaits a new commitment.
        """
        site = self._site_ref(site_ref)
        with self._lock:
            record = self._sites.get(site)
            if record is None or record.active is None:
                raise _fail(HoldoutSplitReasonCode.HOLDOUT_NOT_COMMITTED)
            return record.active

    def verify_evaluation_evidence(
        self, site_ref: str, evidence: object
    ) -> HoldoutSplitVerification:
        """Check that evaluation evidence cites the site's active commitment.

        Args:
            site_ref: Coordinator-assigned opaque site reference.
            evidence: A mapping with a ``holdout_commitment_digest`` field.

        Returns:
            A value-free verification; it fails closed with
            ``evaluation_commitment_missing``, ``holdout_not_committed``,
            ``holdout_commitment_rotated`` or
            ``evaluation_commitment_mismatch``.
        """
        site = self._site_ref(site_ref)
        cited = evidence.get(_CITATION_FIELD) if isinstance(evidence, Mapping) else None
        if not _is_sha256(cited):
            return _verification(HoldoutSplitReasonCode.EVALUATION_COMMITMENT_MISSING)
        with self._lock:
            record = self._sites.get(site)
            if record is None:
                return _verification(HoldoutSplitReasonCode.HOLDOUT_NOT_COMMITTED)
            if cited in record.retired():
                return _verification(HoldoutSplitReasonCode.HOLDOUT_COMMITMENT_ROTATED)
            if record.active is None:
                return _verification(HoldoutSplitReasonCode.HOLDOUT_NOT_COMMITTED)
            if not hmac.compare_digest(record.active, cited):  # type: ignore[arg-type]
                return _verification(
                    HoldoutSplitReasonCode.EVALUATION_COMMITMENT_MISMATCH
                )
        return _verification(HoldoutSplitReasonCode.VALID)

    def to_dict(self) -> dict[str, Any]:
        """Return digest-only durable ledger state."""
        with self._lock:
            sites = {
                site: {
                    "active_commitment_digest": record.active,
                    "rotations": [
                        {"reason": reason.value, "retired_commitment_digest": digest}
                        for digest, reason in record.rotations
                    ],
                }
                for site, record in sorted(self._sites.items())
            }
        return {
            "federation_id": self._federation_id,
            "schema_version": FEDERATED_HOLDOUT_LEDGER_SCHEMA_VERSION,
            "sites": sites,
        }

    def to_json(self) -> str:
        """Return stable compact JSON for durable local storage."""
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> FederatedHoldoutLedger:
        """Restore a ledger, failing closed on any inconsistent field.

        Args:
            state: A mapping produced by :meth:`to_dict`.

        Returns:
            The restored ledger.

        Raises:
            FederatedHoldoutSplitError: With ``invalid_ledger_state``.
        """
        sites = _parse_ledger_state(state)
        ledger = cls(state["federation_id"])
        ledger._sites = sites
        return ledger


def _parse_ledger_state(state: object) -> dict[str, _SiteRecord]:
    invalid = _fail(HoldoutSplitReasonCode.INVALID_LEDGER_STATE)
    if (
        not isinstance(state, Mapping)
        or set(state) != _LEDGER_KEYS
        or state["schema_version"] != FEDERATED_HOLDOUT_LEDGER_SCHEMA_VERSION
        or not _is_public_id(state["federation_id"])
        or not isinstance(state["sites"], Mapping)
    ):
        raise invalid
    reasons = {item.value: item for item in HoldoutRotationReason}
    sites: dict[str, _SiteRecord] = {}
    for site, entry in state["sites"].items():
        if (
            not _is_public_id(site)
            or not isinstance(entry, Mapping)
            or set(entry) != _SITE_KEYS
            or not isinstance(entry["rotations"], list)
        ):
            raise invalid
        active = entry["active_commitment_digest"]
        if active is not None and not _is_sha256(active):
            raise invalid
        rotations: list[tuple[str, HoldoutRotationReason]] = []
        for rotation in entry["rotations"]:
            if not isinstance(rotation, Mapping) or set(rotation) != _ROTATION_KEYS:
                raise invalid
            digest = rotation["retired_commitment_digest"]
            reason = reasons.get(_str_or_none(rotation["reason"]))
            if not _is_sha256(digest) or reason is None:
                raise invalid
            rotations.append((digest, reason))
        retired = [digest for digest, _ in rotations]
        if len(set(retired)) != len(retired) or active in retired:
            raise invalid
        if active is None and not rotations:
            raise invalid
        sites[site] = _SiteRecord(active=active, rotations=rotations)
    return sites


__all__ = [
    "FEDERATED_HOLDOUT_EVIDENCE_SCHEMA_VERSION",
    "FEDERATED_HOLDOUT_LEDGER_SCHEMA_VERSION",
    "FEDERATED_HOLDOUT_VERIFICATION_SCHEMA_VERSION",
    "MINIMUM_SITE_SECRET_BYTES",
    "FederatedHoldoutLedger",
    "FederatedHoldoutSplitError",
    "HoldoutPatientCountBand",
    "HoldoutRotationReason",
    "HoldoutSplitPartition",
    "HoldoutSplitReasonCode",
    "HoldoutSplitVerification",
    "SiteHoldoutEvidence",
    "SiteHoldoutSplit",
    "build_site_holdout_split",
]
