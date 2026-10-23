"""Value-free egress reports and approval-bound submission for federated updates.

A site renders a deterministic report of what one sealed update discloses:
parameter names, shapes and element counts, clipping and noise declarations,
privacy-spend bands, and digests. An injected approver signs a statement bound
to the report digest, the sealed update digest, the round digest and an expiry.
The submission guard refuses to hand bytes to an injected transport unless a
matching, unexpired approval verifies.

Nothing here reads tensors, seals updates or performs network calls. Errors
carry a stable reason code and a fixed message; they never echo submitted
values, field names, local metrics or paths.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import InitVar, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from fractions import Fraction
from typing import Any, Final, Protocol, TypeVar

from .federated_schema_fingerprint import fingerprint_update_schema
from .federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

FEDERATED_EGRESS_REPORT_SCHEMA_VERSION: Final = (
    "openmed.training.federated_egress_report.v1"
)
FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION: Final = (
    "openmed.training.federated_egress_approval.v1"
)
MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY: Final = timedelta(days=7)

FORBIDDEN_EGRESS_REPORT_FIELDS: Final = frozenset(
    {
        "accuracy",
        "address",
        "client",
        "client_id",
        "directory",
        "email",
        "endpoint",
        "example",
        "example_count",
        "examples",
        "file",
        "file_path",
        "filename",
        "gradient",
        "gradients",
        "host",
        "hostname",
        "hospital",
        "identifier",
        "identifiers",
        "institution",
        "institution_id",
        "ip",
        "local_metric",
        "local_metrics",
        "loss",
        "message",
        "messages",
        "metrics",
        "mrn",
        "note",
        "notes",
        "organization",
        "path",
        "paths",
        "patient",
        "patient_count",
        "patient_counts",
        "patient_id",
        "patient_ids",
        "record_id",
        "sample",
        "samples",
        "site",
        "site_id",
        "site_name",
        "tensor",
        "tensors",
        "text",
        "texts",
        "uri",
        "url",
        "user",
        "user_id",
        "username",
        "values",
        "weights",
    }
)
"""Field names that never appear at any level of a report or approval record."""

_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_VERSION: Final = re.compile(r"v[1-9][0-9]{0,3}\Z")
_SIGNATURE: Final = re.compile(r"[A-Za-z0-9._:+/=-]{16,512}\Z")
_TIMESTAMP: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z"
)
_TIMESTAMP_FORMAT: Final = "%Y-%m-%dT%H:%M:%SZ"
_MAX_JSON_CHARS: Final = 1024 * 1024
_MAX_SCAN_NODES: Final = 100_000
_MAX_PARAMETERS: Final = 1024
_APPROVAL_DOMAIN: Final = b"openmed.training.federated_egress_approval.v1\0"
_REPORT_BUILDER_TOKEN: Final = object()

_REPORT_FIELDS: Final = frozenset(
    {
        "schema_version",
        "round_digest",
        "sealed_update_digest",
        "update",
        "clipping",
        "mechanism",
        "privacy_spend",
    }
)
_UPDATE_FIELDS: Final = frozenset(
    {
        "metadata_schema_version",
        "model_digest",
        "update_digest",
        "adapter_format",
        "schema_fingerprint",
        "clipped",
        "parameter_count",
        "total_elements",
        "parameters",
    }
)
_PARAMETER_FIELDS: Final = frozenset({"name", "shape", "dtype", "element_count"})
_CLIPPING_FIELDS: Final = frozenset({"scope", "norm", "config_digest"})
_MECHANISM_FIELDS: Final = frozenset(
    {"mechanism", "version", "accountant", "config_digest"}
)
_SPEND_FIELDS: Final = frozenset({"epsilon_band", "delta_band", "ledger_digest"})
_APPROVAL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "report_digest",
        "sealed_update_digest",
        "round_digest",
        "approved_at",
        "expires_at",
        "signature",
    }
)

_T = TypeVar("_T")
_E = TypeVar("_E", bound=Enum)


class FederatedEgressReasonCode(str, Enum):
    """Stable reasons an egress report, approval or submission is refused."""

    INVALID_UPDATE_METADATA = "invalid_update_metadata"
    INVALID_SEALED_UPDATE = "invalid_sealed_update"
    INVALID_ROUND_DIGEST = "invalid_round_digest"
    INVALID_CLIPPING_DECLARATION = "invalid_clipping_declaration"
    INVALID_MECHANISM_DECLARATION = "invalid_mechanism_declaration"
    INVALID_PRIVACY_SPEND = "invalid_privacy_spend"
    PRIVACY_BUDGET_EXCEEDED = "privacy_budget_exceeded"
    CLIPPING_NOT_DECLARED = "clipping_not_declared"
    INVALID_REPORT = "invalid_report"
    FORBIDDEN_REPORT_FIELD = "forbidden_report_field"
    INVALID_APPROVAL = "invalid_approval"
    INVALID_APPROVAL_WINDOW = "invalid_approval_window"
    INVALID_CLOCK = "invalid_clock"
    APPROVAL_MISSING = "approval_missing"
    APPROVAL_DECLINED = "approval_declined"
    APPROVER_FAILED = "approver_failed"
    APPROVAL_REPORT_MISMATCH = "approval_report_mismatch"
    SEALED_UPDATE_MISMATCH = "sealed_update_mismatch"
    ROUND_MISMATCH = "round_mismatch"
    APPROVAL_SIGNATURE_INVALID = "approval_signature_invalid"
    APPROVAL_NOT_YET_VALID = "approval_not_yet_valid"
    APPROVAL_EXPIRED = "approval_expired"
    TRANSPORT_FAILED = "transport_failed"

    def __str__(self) -> str:
        return self.value


class FederatedEgressError(ValueError):
    """Raised when egress is refused; carries only a stable reason code.

    Args:
        reason_code: The controlled reason for the refusal.

    Attributes:
        reason_code: The controlled reason for the refusal.
    """

    def __init__(self, reason_code: FederatedEgressReasonCode) -> None:
        self.reason_code = reason_code
        super().__init__(f"federated egress refused: {reason_code.value}")


class FederatedClippingScope(str, Enum):
    """Where the declared norm clipping was applied."""

    PER_EXAMPLE = "per_example"
    PER_UPDATE = "per_update"


class FederatedClippingNorm(str, Enum):
    """The declared clipping norm."""

    L2 = "l2"


class FederatedUpdateNoiseMechanism(str, Enum):
    """The declared noise mechanism applied before an update leaves the site."""

    GAUSSIAN = "gaussian"


class FederatedPrivacyAccountant(str, Enum):
    """The declared privacy accountant used to track the site's spend."""

    RDP = "rdp"
    PRV = "prv"


class FederatedPrivacySpendBand(str, Enum):
    """Coarse fraction of a site's privacy budget already spent."""

    NONE = "none"
    UNDER_QUARTER = "under_quarter"
    QUARTER_TO_UNDER_HALF = "quarter_to_under_half"
    HALF_TO_UNDER_THREE_QUARTERS = "half_to_under_three_quarters"
    THREE_QUARTERS_TO_FULL = "three_quarters_to_full"


@dataclass(frozen=True)
class FederatedEgressClipping:
    """Declared clipping configuration, expressed as codes and a digest.

    Args:
        scope: Clipping scope code or member.
        norm: Clipping norm code or member.
        config_digest: SHA-256 reference to the committed clipping configuration.

    Raises:
        FederatedEgressError: ``invalid_clipping_declaration`` on any invalid
            code or digest.
    """

    scope: FederatedClippingScope
    norm: FederatedClippingNorm
    config_digest: str

    def __post_init__(self) -> None:
        reason = FederatedEgressReasonCode.INVALID_CLIPPING_DECLARATION
        object.__setattr__(
            self, "scope", _code(FederatedClippingScope, self.scope, reason)
        )
        object.__setattr__(
            self, "norm", _code(FederatedClippingNorm, self.norm, reason)
        )
        _require_digest(self.config_digest, reason)

    def to_dict(self) -> dict[str, str]:
        """Return a fresh mapping of clipping codes and the configuration digest."""
        return {
            "scope": self.scope.value,
            "norm": self.norm.value,
            "config_digest": self.config_digest,
        }


@dataclass(frozen=True)
class FederatedEgressMechanism:
    """Declared noise mechanism and accountant, expressed as codes and a digest.

    Args:
        mechanism: Noise mechanism code or member.
        version: Mechanism version code such as ``v1``.
        accountant: Privacy accountant code or member.
        config_digest: SHA-256 reference to the committed noise configuration.

    Raises:
        FederatedEgressError: ``invalid_mechanism_declaration`` on any invalid
            code, version or digest.
    """

    mechanism: FederatedUpdateNoiseMechanism
    version: str
    accountant: FederatedPrivacyAccountant
    config_digest: str

    def __post_init__(self) -> None:
        reason = FederatedEgressReasonCode.INVALID_MECHANISM_DECLARATION
        object.__setattr__(
            self,
            "mechanism",
            _code(FederatedUpdateNoiseMechanism, self.mechanism, reason),
        )
        if type(self.version) is not str or _VERSION.fullmatch(self.version) is None:
            raise FederatedEgressError(reason)
        object.__setattr__(
            self,
            "accountant",
            _code(FederatedPrivacyAccountant, self.accountant, reason),
        )
        _require_digest(self.config_digest, reason)

    def to_dict(self) -> dict[str, str]:
        """Return a fresh mapping of mechanism codes and the configuration digest."""
        return {
            "mechanism": self.mechanism.value,
            "version": self.version,
            "accountant": self.accountant.value,
            "config_digest": self.config_digest,
        }


@dataclass(frozen=True)
class FederatedEgressPrivacySpend:
    """The site's cumulative privacy spend, disclosed only as bands.

    Exact spend and budget values stay local: they are hidden from ``repr`` and
    only :attr:`epsilon_band`, :attr:`delta_band` and the ledger digest enter a
    report.

    Args:
        epsilon_spent: Cumulative epsilon spent, including this update.
        epsilon_budget: Positive epsilon budget for the site.
        delta_spent: Cumulative delta spent, including this update.
        delta_budget: Delta budget for the site, greater than zero and below one.
        ledger_digest: SHA-256 reference to the site's privacy ledger state.

    Raises:
        FederatedEgressError: ``invalid_privacy_spend`` for non-finite,
            negative, boolean or out-of-range values, and
            ``privacy_budget_exceeded`` when spend exceeds a budget.
    """

    epsilon_spent: float = field(repr=False)
    epsilon_budget: float = field(repr=False)
    delta_spent: float = field(repr=False)
    delta_budget: float = field(repr=False)
    ledger_digest: str

    def __post_init__(self) -> None:
        reason = FederatedEgressReasonCode.INVALID_PRIVACY_SPEND
        for value in (
            self.epsilon_spent,
            self.epsilon_budget,
            self.delta_spent,
            self.delta_budget,
        ):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise FederatedEgressError(reason)
        if self.epsilon_budget <= 0 or self.epsilon_spent < 0:
            raise FederatedEgressError(reason)
        if not 0 < self.delta_budget < 1 or self.delta_spent < 0:
            raise FederatedEgressError(reason)
        _require_digest(self.ledger_digest, reason)
        if (
            self.epsilon_spent > self.epsilon_budget
            or self.delta_spent > self.delta_budget
        ):
            raise FederatedEgressError(
                FederatedEgressReasonCode.PRIVACY_BUDGET_EXCEEDED
            )

    @property
    def epsilon_band(self) -> FederatedPrivacySpendBand:
        """Return the band of the epsilon budget already spent."""
        return _spend_band(self.epsilon_spent, self.epsilon_budget)

    @property
    def delta_band(self) -> FederatedPrivacySpendBand:
        """Return the band of the delta budget already spent."""
        return _spend_band(self.delta_spent, self.delta_budget)


@dataclass(frozen=True, repr=False)
class FederatedEgressReport:
    """An immutable, byte-stable, value-free egress report.

    Build reports with :func:`render_federated_egress_report` or parse them
    with :meth:`from_dict` or :meth:`from_json`; direct construction is
    refused.
    """

    _canonical_json: str
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _REPORT_BUILDER_TOKEN:
            raise FederatedEgressError(FederatedEgressReasonCode.INVALID_REPORT)

    def __repr__(self) -> str:
        return f"FederatedEgressReport(digest={self.digest!r})"

    @property
    def digest(self) -> str:
        """Return the SHA-256 reference of the canonical JSON bytes."""
        return _sha256(self._canonical_json.encode("ascii"))

    @property
    def sealed_update_digest(self) -> str:
        """Return the SHA-256 reference of the sealed update bytes."""
        return self.to_dict()["sealed_update_digest"]

    @property
    def round_digest(self) -> str:
        """Return the digest of the round this report was rendered for."""
        return self.to_dict()["round_digest"]

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh mapping of the canonical report."""
        return json.loads(self._canonical_json)

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return self._canonical_json

    def to_markdown(self) -> str:
        """Return a deterministic Markdown rendering for site reviewers."""
        report = self.to_dict()
        update = report["update"]
        clipping = report["clipping"]
        mechanism = report["mechanism"]
        spend = report["privacy_spend"]
        lines = [
            "# Federated update egress report",
            "",
            f"- Report digest: `{self.digest}`",
            f"- Schema version: `{report['schema_version']}`",
            f"- Round digest: `{report['round_digest']}`",
            f"- Sealed update digest: `{report['sealed_update_digest']}`",
            "",
            "## Update",
            "",
            f"- Metadata schema version: `{update['metadata_schema_version']}`",
            f"- Model digest: `{update['model_digest']}`",
            f"- Update digest: `{update['update_digest']}`",
            f"- Schema fingerprint: `{update['schema_fingerprint']}`",
            f"- Adapter format: `{update['adapter_format']}`",
            f"- Clipped: `{str(update['clipped']).lower()}`",
            f"- Parameter count: {update['parameter_count']}",
            f"- Total elements: {update['total_elements']}",
            "",
            "| Name | Shape | Dtype | Elements |",
            "| --- | --- | --- | --- |",
        ]
        for parameter in update["parameters"]:
            shape = "x".join(str(dimension) for dimension in parameter["shape"])
            lines.append(
                f"| `{parameter['name']}` | {shape} | `{parameter['dtype']}` "
                f"| {parameter['element_count']} |"
            )
        lines.extend(
            [
                "",
                "## Clipping",
                "",
                f"- Scope: `{clipping['scope']}`",
                f"- Norm: `{clipping['norm']}`",
                f"- Configuration digest: `{clipping['config_digest']}`",
                "",
                "## Noise mechanism",
                "",
                f"- Mechanism: `{mechanism['mechanism']}`",
                f"- Version: `{mechanism['version']}`",
                f"- Accountant: `{mechanism['accountant']}`",
                f"- Configuration digest: `{mechanism['config_digest']}`",
                "",
                "## Privacy spend",
                "",
                f"- Epsilon band: `{spend['epsilon_band']}`",
                f"- Delta band: `{spend['delta_band']}`",
                f"- Ledger digest: `{spend['ledger_digest']}`",
            ]
        )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_dict(cls, payload: object) -> FederatedEgressReport:
        """Validate a closed report mapping and return its canonical form.

        Args:
            payload: Built-in mapping with exactly the report fields.

        Returns:
            The validated report.

        Raises:
            FederatedEgressError: ``forbidden_report_field`` when any level
                carries a catalogued identifying or value-bearing field, and
                ``invalid_report`` for any other schema or consistency error.
        """
        _reject_forbidden_fields(payload)
        try:
            canonical = _rebuild_report(payload)
        except (
            FederatedEgressError,
            FederatedUpdateMetadataError,
            KeyError,
            TypeError,
            ValueError,
        ):
            raise FederatedEgressError(
                FederatedEgressReasonCode.INVALID_REPORT
            ) from None
        if canonical != payload:
            raise FederatedEgressError(FederatedEgressReasonCode.INVALID_REPORT)
        return cls(_canonical_dumps(canonical), _REPORT_BUILDER_TOKEN)

    @classmethod
    def from_json(cls, payload: object) -> FederatedEgressReport:
        """Parse bounded strict JSON into a validated report.

        Args:
            payload: JSON text of at most one MiB. Duplicate keys, floats and
                non-finite constants are rejected.

        Returns:
            The validated report.

        Raises:
            FederatedEgressError: ``invalid_report`` or
                ``forbidden_report_field``.
        """
        decoded = _strict_json(payload, FederatedEgressReasonCode.INVALID_REPORT)
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class FederatedEgressApproval:
    """A site approval bound to one report, sealed update, round and window.

    Args:
        report_digest: Digest of the approved report.
        sealed_update_digest: Digest of the sealed update bytes.
        round_digest: Digest of the round the update was approved for.
        approved_at: UTC approval time with whole-second precision.
        expires_at: UTC expiry, after ``approved_at`` and at most
            :data:`MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY` later.
        signature: Opaque token returned by the injected approver.
        schema_version: Exact supported approval schema identifier.

    Raises:
        FederatedEgressError: ``invalid_approval`` for any invalid field.
    """

    report_digest: str
    sealed_update_digest: str
    round_digest: str
    approved_at: datetime
    expires_at: datetime
    signature: str = field(repr=False)
    schema_version: str = FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        reason = FederatedEgressReasonCode.INVALID_APPROVAL
        if self.schema_version != FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION:
            raise FederatedEgressError(reason)
        for digest in (
            self.report_digest,
            self.sealed_update_digest,
            self.round_digest,
        ):
            _require_digest(digest, reason)
        for moment in (self.approved_at, self.expires_at):
            if (
                type(moment) is not datetime
                or moment.utcoffset() != timedelta(0)
                or moment.microsecond != 0
            ):
                raise FederatedEgressError(reason)
        window = self.expires_at - self.approved_at
        if not timedelta(0) < window <= MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY:
            raise FederatedEgressError(reason)
        _require_signature(self.signature)
        object.__setattr__(self, "approved_at", _as_utc(self.approved_at))
        object.__setattr__(self, "expires_at", _as_utc(self.expires_at))

    def statement(self) -> bytes:
        """Return the domain-separated bytes the approver signs and verifies."""
        return _approval_statement(
            report_digest=self.report_digest,
            sealed_update_digest=self.sealed_update_digest,
            round_digest=self.round_digest,
            approved_at=self.approved_at,
            expires_at=self.expires_at,
        )

    def to_dict(self) -> dict[str, str]:
        """Return a fresh mapping of digests, timestamps and the signature."""
        return {
            "schema_version": self.schema_version,
            "report_digest": self.report_digest,
            "sealed_update_digest": self.sealed_update_digest,
            "round_digest": self.round_digest,
            "approved_at": _format_time(self.approved_at),
            "expires_at": _format_time(self.expires_at),
            "signature": self.signature,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with sorted keys and a trailing newline."""
        return _canonical_dumps(self.to_dict())

    @classmethod
    def from_dict(cls, payload: object) -> FederatedEgressApproval:
        """Validate a closed approval mapping.

        Args:
            payload: Built-in mapping with exactly the approval fields.

        Returns:
            The validated approval record. Its signature is not verified here;
            :class:`FederatedEgressSubmissionGuard` verifies it.

        Raises:
            FederatedEgressError: ``forbidden_report_field`` or
                ``invalid_approval``.
        """
        reason = FederatedEgressReasonCode.INVALID_APPROVAL
        _reject_forbidden_fields(payload)
        payload = _require_fields(payload, _APPROVAL_FIELDS, reason)
        return cls(
            report_digest=payload["report_digest"],
            sealed_update_digest=payload["sealed_update_digest"],
            round_digest=payload["round_digest"],
            approved_at=_parse_time(payload["approved_at"]),
            expires_at=_parse_time(payload["expires_at"]),
            signature=payload["signature"],
            schema_version=payload["schema_version"],
        )

    @classmethod
    def from_json(cls, payload: object) -> FederatedEgressApproval:
        """Parse bounded strict JSON into an approval record.

        Args:
            payload: JSON text of at most one MiB.

        Returns:
            The validated approval record.

        Raises:
            FederatedEgressError: ``invalid_approval`` or
                ``forbidden_report_field``.
        """
        decoded = _strict_json(payload, FederatedEgressReasonCode.INVALID_APPROVAL)
        return cls.from_dict(decoded)


@dataclass(frozen=True)
class FederatedEgressClearance:
    """Digest-only evidence that a submission passed the approval guard.

    Args:
        report_digest: Digest of the approved report.
        sealed_update_digest: Digest of the sealed bytes that were cleared.
        round_digest: Digest of the round the bytes were cleared for.
        expires_at: Expiry of the approval that cleared them.
    """

    report_digest: str
    sealed_update_digest: str
    round_digest: str
    expires_at: datetime


class FederatedEgressApprover(Protocol):
    """Injected site approval authority, such as a governance signing service."""

    def approve(self, report: FederatedEgressReport, statement: bytes) -> str | None:
        """Present ``report`` for review and sign ``statement`` if approved.

        Args:
            report: The value-free report under review.
            statement: Domain-separated bytes binding the report digest,
                sealed update digest, round digest and approval window.

        Returns:
            An opaque signature token, or ``None`` when the site declines.
        """

    def verify(self, statement: bytes, signature: str) -> bool:
        """Return ``True`` only if ``signature`` is valid for ``statement``.

        Args:
            statement: The statement bytes recomputed from an approval record.
            signature: The signature token stored in that record.

        Returns:
            Whether the signature verifies.
        """


def render_federated_egress_report(
    metadata: FederatedUpdateMetadata,
    *,
    sealed_update: bytes,
    round_digest: str,
    clipping: FederatedEgressClipping,
    mechanism: FederatedEgressMechanism,
    privacy_spend: FederatedEgressPrivacySpend,
) -> FederatedEgressReport:
    """Render a deterministic, value-free report of what an update discloses.

    Args:
        metadata: Validated update metadata for the sealed update.
        sealed_update: The exact sealed bytes that would leave the site. Only
            their SHA-256 digest enters the report.
        round_digest: SHA-256 reference identifying the federated round.
        clipping: Declared clipping configuration.
        mechanism: Declared noise mechanism and accountant.
        privacy_spend: The site's cumulative privacy spend; only bands and the
            ledger digest are reported.

    Returns:
        The canonical report, validated through the same closed schema used for
        parsing so that no forbidden field can be emitted.

    Raises:
        FederatedEgressError: With a stable reason code for any invalid input,
            an update that does not declare clipping, or a failed self-check.
    """
    if type(metadata) is not FederatedUpdateMetadata:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_UPDATE_METADATA)
    if metadata.clipped is not True:
        raise FederatedEgressError(FederatedEgressReasonCode.CLIPPING_NOT_DECLARED)
    _require_sealed_update(sealed_update)
    _require_digest(round_digest, FederatedEgressReasonCode.INVALID_ROUND_DIGEST)
    if type(clipping) is not FederatedEgressClipping:
        raise FederatedEgressError(
            FederatedEgressReasonCode.INVALID_CLIPPING_DECLARATION
        )
    if type(mechanism) is not FederatedEgressMechanism:
        raise FederatedEgressError(
            FederatedEgressReasonCode.INVALID_MECHANISM_DECLARATION
        )
    if type(privacy_spend) is not FederatedEgressPrivacySpend:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_PRIVACY_SPEND)
    payload = _report_dict(
        metadata=metadata,
        sealed_update_digest=_sha256(sealed_update),
        round_digest=round_digest,
        clipping=clipping,
        mechanism=mechanism,
        epsilon_band=privacy_spend.epsilon_band,
        delta_band=privacy_spend.delta_band,
        ledger_digest=privacy_spend.ledger_digest,
    )
    return FederatedEgressReport.from_dict(payload)


def approve_federated_egress(
    report: FederatedEgressReport,
    *,
    approver: FederatedEgressApprover,
    clock: Callable[[], datetime],
    valid_for: timedelta,
) -> FederatedEgressApproval:
    """Ask the injected approver to approve one report for a bounded window.

    Args:
        report: The value-free report presented for review.
        approver: Injected approval authority that signs the statement.
        clock: Injected clock returning a timezone-aware ``datetime``.
        valid_for: Positive whole-second validity, at most
            :data:`MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY`.

    Returns:
        An approval record bound to the report digest, sealed update digest,
        round digest and expiry.

    Raises:
        FederatedEgressError: ``invalid_report``, ``invalid_approval_window``,
            ``invalid_clock``, ``approver_failed``, ``approval_declined`` or
            ``invalid_approval``.
    """
    if type(report) is not FederatedEgressReport:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_REPORT)
    if (
        type(valid_for) is not timedelta
        or valid_for <= timedelta(0)
        or valid_for.microseconds != 0
        or valid_for > MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY
    ):
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_APPROVAL_WINDOW)
    approved_at = _now(clock).replace(microsecond=0)
    try:
        expires_at = approved_at + valid_for
    except OverflowError:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_CLOCK) from None
    statement = _approval_statement(
        report_digest=report.digest,
        sealed_update_digest=report.sealed_update_digest,
        round_digest=report.round_digest,
        approved_at=approved_at,
        expires_at=expires_at,
    )
    signature = _call_port(lambda: approver.approve(report, statement))
    if signature is None:
        raise FederatedEgressError(FederatedEgressReasonCode.APPROVAL_DECLINED)
    _require_signature(signature)
    return FederatedEgressApproval(
        report_digest=report.digest,
        sealed_update_digest=report.sealed_update_digest,
        round_digest=report.round_digest,
        approved_at=approved_at,
        expires_at=expires_at,
        signature=signature,
    )


class FederatedEgressSubmissionGuard:
    """Refuses submission unless a matching, unexpired approval verifies.

    Args:
        approver: Injected approval authority used to verify signatures.
        clock: Injected clock returning a timezone-aware ``datetime``.
    """

    def __init__(
        self,
        *,
        approver: FederatedEgressApprover,
        clock: Callable[[], datetime],
    ) -> None:
        self._approver = approver
        self._clock = clock

    def check(
        self,
        sealed_update: bytes,
        *,
        report: FederatedEgressReport,
        approval: FederatedEgressApproval | None,
        round_digest: str,
    ) -> FederatedEgressClearance:
        """Verify that ``approval`` covers exactly these bytes, report and round.

        Args:
            sealed_update: The exact sealed bytes about to leave the site.
            report: The report the site reviewed.
            approval: The site's approval record, or ``None``.
            round_digest: Digest of the round the bytes are being submitted to.

        Returns:
            Digest-only clearance evidence.

        Raises:
            FederatedEgressError: With a stable reason code when the approval
                is missing, mismatched, unverifiable, not yet valid or expired.
        """
        _require_sealed_update(sealed_update)
        if type(report) is not FederatedEgressReport:
            raise FederatedEgressError(FederatedEgressReasonCode.INVALID_REPORT)
        _require_digest(round_digest, FederatedEgressReasonCode.INVALID_ROUND_DIGEST)
        if approval is None:
            raise FederatedEgressError(FederatedEgressReasonCode.APPROVAL_MISSING)
        if type(approval) is not FederatedEgressApproval:
            raise FederatedEgressError(FederatedEgressReasonCode.INVALID_APPROVAL)
        if not _same(approval.report_digest, report.digest):
            raise FederatedEgressError(
                FederatedEgressReasonCode.APPROVAL_REPORT_MISMATCH
            )
        sealed_digest = _sha256(sealed_update)
        if not (
            _same(sealed_digest, report.sealed_update_digest)
            and _same(sealed_digest, approval.sealed_update_digest)
        ):
            raise FederatedEgressError(FederatedEgressReasonCode.SEALED_UPDATE_MISMATCH)
        if not (
            _same(round_digest, report.round_digest)
            and _same(round_digest, approval.round_digest)
        ):
            raise FederatedEgressError(FederatedEgressReasonCode.ROUND_MISMATCH)
        statement = approval.statement()
        verified = _call_port(
            lambda: self._approver.verify(statement, approval.signature)
        )
        if verified is not True:
            raise FederatedEgressError(
                FederatedEgressReasonCode.APPROVAL_SIGNATURE_INVALID
            )
        now = _now(self._clock)
        if now < approval.approved_at:
            raise FederatedEgressError(FederatedEgressReasonCode.APPROVAL_NOT_YET_VALID)
        if now >= approval.expires_at:
            raise FederatedEgressError(FederatedEgressReasonCode.APPROVAL_EXPIRED)
        return FederatedEgressClearance(
            report_digest=report.digest,
            sealed_update_digest=sealed_digest,
            round_digest=round_digest,
            expires_at=approval.expires_at,
        )

    def submit(
        self,
        sealed_update: bytes,
        *,
        report: FederatedEgressReport,
        approval: FederatedEgressApproval | None,
        round_digest: str,
        transport: Callable[[bytes], _T],
    ) -> _T:
        """Check the approval, then hand the exact bytes to the injected sender.

        Args:
            sealed_update: The exact sealed bytes to submit.
            report: The report the site reviewed.
            approval: The site's approval record, or ``None``.
            round_digest: Digest of the round the bytes are being submitted to.
            transport: Injected sender; it is never called when the check fails.

        Returns:
            Whatever ``transport`` returns.

        Raises:
            FederatedEgressError: Any :meth:`check` refusal, or
                ``transport_failed`` when ``transport`` raises.
        """
        self.check(
            sealed_update, report=report, approval=approval, round_digest=round_digest
        )
        try:
            return transport(sealed_update)
        except Exception:
            raise FederatedEgressError(
                FederatedEgressReasonCode.TRANSPORT_FAILED
            ) from None


def _report_dict(
    *,
    metadata: FederatedUpdateMetadata,
    sealed_update_digest: str,
    round_digest: str,
    clipping: FederatedEgressClipping,
    mechanism: FederatedEgressMechanism,
    epsilon_band: FederatedPrivacySpendBand,
    delta_band: FederatedPrivacySpendBand,
    ledger_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": FEDERATED_EGRESS_REPORT_SCHEMA_VERSION,
        "round_digest": round_digest,
        "sealed_update_digest": sealed_update_digest,
        "update": {
            "metadata_schema_version": metadata.schema_version,
            "model_digest": metadata.model_digest,
            "update_digest": metadata.update_digest,
            "adapter_format": metadata.adapter_format,
            "schema_fingerprint": fingerprint_update_schema(metadata),
            "clipped": metadata.clipped,
            "parameter_count": len(metadata.parameters),
            "total_elements": metadata.total_elements,
            "parameters": [
                {
                    "name": parameter.name,
                    "shape": list(parameter.shape),
                    "dtype": parameter.dtype,
                    "element_count": parameter.element_count,
                }
                for parameter in metadata.parameters
            ],
        },
        "clipping": clipping.to_dict(),
        "mechanism": mechanism.to_dict(),
        "privacy_spend": {
            "epsilon_band": epsilon_band.value,
            "delta_band": delta_band.value,
            "ledger_digest": ledger_digest,
        },
    }


def _rebuild_report(payload: object) -> dict[str, Any]:
    reason = FederatedEgressReasonCode.INVALID_REPORT
    report = _require_fields(payload, _REPORT_FIELDS, reason)
    if report["schema_version"] != FEDERATED_EGRESS_REPORT_SCHEMA_VERSION:
        raise FederatedEgressError(reason)
    update = _require_fields(report["update"], _UPDATE_FIELDS, reason)
    if update["metadata_schema_version"] != FEDERATED_UPDATE_METADATA_SCHEMA_VERSION:
        raise FederatedEgressError(reason)
    if update["clipped"] is not True:
        raise FederatedEgressError(reason)
    for count in (update["parameter_count"], update["total_elements"]):
        if type(count) is not int:
            raise FederatedEgressError(reason)
    raw_parameters = update["parameters"]
    if (
        type(raw_parameters) is not list
        or not 1 <= len(raw_parameters) <= _MAX_PARAMETERS
    ):
        raise FederatedEgressError(reason)
    parameters = []
    for raw in raw_parameters:
        raw = _require_fields(raw, _PARAMETER_FIELDS, reason)
        if type(raw["shape"]) is not list or type(raw["element_count"]) is not int:
            raise FederatedEgressError(reason)
        parameters.append(
            FederatedParameterMetadata(raw["name"], tuple(raw["shape"]), raw["dtype"])
        )
    policy = FederatedUpdatePolicy(
        model_digest=update["model_digest"],
        parameters=tuple(parameters),
        adapter_format=update["adapter_format"],
        max_total_elements=update["total_elements"],
    )
    metadata = FederatedUpdateMetadata(
        model_digest=update["model_digest"],
        adapter_format=update["adapter_format"],
        parameters=tuple(parameters),
        total_elements=update["total_elements"],
        update_digest=update["update_digest"],
        clipped=True,
        policy=policy,
    )
    _require_digest(report["round_digest"], reason)
    _require_digest(report["sealed_update_digest"], reason)
    clipping = _require_fields(report["clipping"], _CLIPPING_FIELDS, reason)
    mechanism = _require_fields(report["mechanism"], _MECHANISM_FIELDS, reason)
    spend = _require_fields(report["privacy_spend"], _SPEND_FIELDS, reason)
    _require_digest(spend["ledger_digest"], reason)
    return _report_dict(
        metadata=metadata,
        sealed_update_digest=report["sealed_update_digest"],
        round_digest=report["round_digest"],
        clipping=FederatedEgressClipping(**clipping),
        mechanism=FederatedEgressMechanism(**mechanism),
        epsilon_band=_code(FederatedPrivacySpendBand, spend["epsilon_band"], reason),
        delta_band=_code(FederatedPrivacySpendBand, spend["delta_band"], reason),
        ledger_digest=spend["ledger_digest"],
    )


def _reject_forbidden_fields(payload: object) -> None:
    pending: list[object] = [payload]
    visited = 0
    while pending:
        node = pending.pop()
        visited += 1
        if visited > _MAX_SCAN_NODES:
            raise FederatedEgressError(FederatedEgressReasonCode.INVALID_REPORT)
        if isinstance(node, Mapping):
            for key, value in node.items():
                if type(key) is str and key.lower() in FORBIDDEN_EGRESS_REPORT_FIELDS:
                    raise FederatedEgressError(
                        FederatedEgressReasonCode.FORBIDDEN_REPORT_FIELD
                    )
                pending.append(value)
        elif isinstance(node, (list, tuple)):
            pending.extend(node)


def _require_fields(
    payload: object,
    expected: frozenset[str],
    reason: FederatedEgressReasonCode,
) -> dict[str, Any]:
    if (
        type(payload) is not dict
        or any(type(key) is not str for key in payload)
        or payload.keys() != expected
    ):
        raise FederatedEgressError(reason)
    return payload


def _require_digest(value: object, reason: FederatedEgressReasonCode) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise FederatedEgressError(reason)


def _require_signature(value: object) -> None:
    if type(value) is not str or _SIGNATURE.fullmatch(value) is None:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_APPROVAL)


def _require_sealed_update(value: object) -> None:
    if type(value) is not bytes or not value:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_SEALED_UPDATE)


def _code(enum: type[_E], value: object, reason: FederatedEgressReasonCode) -> _E:
    if isinstance(value, enum):
        return value
    if type(value) is not str:
        raise FederatedEgressError(reason)
    try:
        return enum(value)
    except ValueError:
        raise FederatedEgressError(reason) from None


def _spend_band(spent: float, budget: float) -> FederatedPrivacySpendBand:
    fraction = Fraction(spent) / Fraction(budget)
    if fraction == 0:
        return FederatedPrivacySpendBand.NONE
    if fraction < Fraction(1, 4):
        return FederatedPrivacySpendBand.UNDER_QUARTER
    if fraction < Fraction(1, 2):
        return FederatedPrivacySpendBand.QUARTER_TO_UNDER_HALF
    if fraction < Fraction(3, 4):
        return FederatedPrivacySpendBand.HALF_TO_UNDER_THREE_QUARTERS
    return FederatedPrivacySpendBand.THREE_QUARTERS_TO_FULL


def _approval_statement(
    *,
    report_digest: str,
    sealed_update_digest: str,
    round_digest: str,
    approved_at: datetime,
    expires_at: datetime,
) -> bytes:
    body = {
        "schema_version": FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION,
        "report_digest": report_digest,
        "sealed_update_digest": sealed_update_digest,
        "round_digest": round_digest,
        "approved_at": _format_time(approved_at),
        "expires_at": _format_time(expires_at),
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return _APPROVAL_DOMAIN + canonical.encode("ascii")


def _now(clock: Callable[[], datetime]) -> datetime:
    try:
        value = clock()
    except Exception:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_CLOCK) from None
    if type(value) is not datetime or value.utcoffset() is None:
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_CLOCK)
    try:
        return value.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        raise FederatedEgressError(FederatedEgressReasonCode.INVALID_CLOCK) from None


def _call_port(call: Callable[[], _T]) -> _T:
    try:
        return call()
    except Exception:
        raise FederatedEgressError(FederatedEgressReasonCode.APPROVER_FAILED) from None


def _as_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def _parse_time(value: object) -> datetime:
    reason = FederatedEgressReasonCode.INVALID_APPROVAL
    if type(value) is not str or _TIMESTAMP.fullmatch(value) is None:
        raise FederatedEgressError(reason)
    try:
        parsed = datetime.strptime(value, _TIMESTAMP_FORMAT)
    except ValueError:
        raise FederatedEgressError(reason) from None
    return parsed.replace(tzinfo=timezone.utc)


def _same(first: str, second: str) -> bool:
    return hmac.compare_digest(first.encode("ascii"), second.encode("ascii"))


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _canonical_dumps(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True
        )
        + "\n"
    )


def _strict_json(payload: object, reason: FederatedEgressReasonCode) -> object:
    if type(payload) is not str or len(payload) > _MAX_JSON_CHARS:
        raise FederatedEgressError(reason)
    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except (ValueError, RecursionError):
        raise FederatedEgressError(reason) from None


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def _reject_number(value: str) -> None:
    raise ValueError("non-integer number")


__all__ = [
    "FEDERATED_EGRESS_APPROVAL_SCHEMA_VERSION",
    "FEDERATED_EGRESS_REPORT_SCHEMA_VERSION",
    "FORBIDDEN_EGRESS_REPORT_FIELDS",
    "MAX_FEDERATED_EGRESS_APPROVAL_VALIDITY",
    "FederatedClippingNorm",
    "FederatedClippingScope",
    "FederatedEgressApproval",
    "FederatedEgressApprover",
    "FederatedEgressClearance",
    "FederatedEgressClipping",
    "FederatedEgressError",
    "FederatedEgressMechanism",
    "FederatedEgressPrivacySpend",
    "FederatedEgressReasonCode",
    "FederatedEgressReport",
    "FederatedEgressSubmissionGuard",
    "FederatedPrivacyAccountant",
    "FederatedPrivacySpendBand",
    "FederatedUpdateNoiseMechanism",
    "approve_federated_egress",
    "render_federated_egress_report",
]
