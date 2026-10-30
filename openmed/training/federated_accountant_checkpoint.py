"""Durable checkpoints for a composed federated privacy accountant.

A composed accountant must never forget privacy budget it already spent. This
module persists the accountant's state as a versioned, canonical-JSON
checkpoint: an append-only, hash-chained event log of every reserved and
reconciled round, the composition parameters, the committed-round boundary, a
monotone generation number and an HMAC integrity binding under an injected key.

The accountant itself is an injected port, so the checkpoint layer owns only
persistence and recovery. Writes are fenced by compare-and-swap on the
generation through an injected durable store. A round is charged when it is
reserved, before its mechanism runs, and the charge is retained after a crash
until the round is explicitly reconciled. Reconciliation never releases the
charge, so interrupted execution followed by a restart yields the same or a more
conservative bound than uninterrupted execution.

Every transition and every restore is checked with the existing
:mod:`openmed.risk.budget_migration` verifier, so spent budget can only grow.
Receipts carry only counts, aggregate bounds and digests: no site identity,
random seed, individual contribution or round reference.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

from openmed.risk import budget_migration as _budget_migration

__all__ = [
    "ACCOUNTANT_CHECKPOINT_REASON_CODES",
    "FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION",
    "FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION",
    "MAX_ACCOUNTANT_EVENTS",
    "RECONCILIATION_OUTCOMES",
    "AccountantCheckpointError",
    "AccountantCheckpointStore",
    "AccountantCompositionParameters",
    "AccountantMechanism",
    "AccountantReceipt",
    "AccountantStatePort",
    "CheckpointIntegrityKey",
    "CheckpointedAccountant",
    "InMemoryAccountantCheckpointStore",
    "StoredAccountantCheckpoint",
]

FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION: Final[str] = (
    "openmed.training.federated_accountant_checkpoint.v1"
)
FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION: Final[str] = (
    "openmed.training.federated_accountant_receipt.v1"
)
MAX_ACCOUNTANT_EVENTS: Final[int] = 20_000
RECONCILIATION_OUTCOMES: Final[tuple[str, ...]] = ("committed", "abandoned")

ACCOUNTANT_CHECKPOINT_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "accountant_failure",
        "accountant_mismatch",
        "boundary_mismatch",
        "budget_exceeded",
        "budget_regressed",
        "checkpoint_malformed",
        "checkpoint_missing",
        "generation_conflict",
        "generation_mismatch",
        "genesis_conflict",
        "history_chain_broken",
        "history_diverged",
        "history_limit_reached",
        "history_out_of_order",
        "history_truncated",
        "identifier_invalid",
        "integrity_key_invalid",
        "integrity_mismatch",
        "mechanism_invalid",
        "outcome_invalid",
        "parameters_invalid",
        "parameters_mismatch",
        "restore_required",
        "round_duplicate",
        "round_unknown",
        "schema_unsupported",
        "stale_checkpoint",
        "store_failure",
        "uncertain_spend_unreconciled",
        "witness_invalid",
    }
)

_MAC_ALGORITHM: Final[str] = "hmac-sha256"
_MIN_SECRET_BYTES: Final[int] = 32
_MAX_ORDERS: Final[int] = 64
_MAX_MECHANISM_PARAMETERS: Final[int] = 16
_DIGEST_RE: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
_VERSION_RE: Final = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,31}\Z")
_COMPOSITION_RE: Final = re.compile(r"[a-z][a-z_]{0,31}\Z")
_PARAMETER_NAME_RE: Final = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
# Name fragments that would let identities, seeds or contributions into state.
_FORBIDDEN_FRAGMENTS: Final[tuple[str, ...]] = (
    "client",
    "contribution",
    "gradient",
    "member",
    "nonce",
    "participant",
    "patient",
    "person",
    "record",
    "seed",
    "site",
    "update",
    "user",
)
_CHAIN_DOMAIN: Final[bytes] = (
    FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION.encode("ascii") + b"\0chain\0"
)
_MAC_DOMAIN: Final[bytes] = (
    FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION.encode("ascii") + b"\0mac\0"
)
_GENESIS_PREVIOUS_DIGEST: Final[str] = "sha256:" + "0" * 64
_LEDGER_CHARGED: Final[str] = "accountant.charged"
_LEDGER_COMMITTED: Final[str] = "accountant.committed"
_DOCUMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "schema_version",
        "accountant",
        "parameters",
        "parameters_fingerprint",
        "generation",
        "events",
        "committed_boundary",
        "uncertain_rounds",
        "bound",
        "head_digest",
        "integrity",
    }
)
_EVENT_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "genesis": frozenset(
            {
                "index",
                "kind",
                "genesis_id",
                "accountant_id",
                "accountant_version",
                "parameters_fingerprint",
                "chain_digest",
            }
        ),
        "reserve": frozenset(
            {"index", "kind", "round_digest", "mechanism", "parameters", "chain_digest"}
        ),
        "reconcile": frozenset(
            {"index", "kind", "round_digest", "outcome", "chain_digest"}
        ),
    }
)
_BOUNDARY_KEYS: Final[frozenset[str]] = frozenset(
    {"committed_rounds", "abandoned_rounds", "event_index", "chain_digest"}
)


class AccountantCheckpointError(ValueError):
    """Raised with a fixed reason code when a checkpoint operation is refused.

    The message is only ``"<code>: <field>"``. It never echoes input values,
    identifiers, payloads, store errors or accountant errors.

    Attributes:
        code: Stable reason code from :data:`ACCOUNTANT_CHECKPOINT_REASON_CODES`.
        field: Name of the field or component that failed.
    """

    def __init__(self, code: str, field_name: str) -> None:
        if code not in ACCOUNTANT_CHECKPOINT_REASON_CODES:
            code = "checkpoint_malformed"
        self.code = code
        self.field = field_name
        super().__init__(f"{code}: {field_name}")


def _fail(code: str, field_name: str) -> AccountantCheckpointError:
    return AccountantCheckpointError(code, field_name)


# --- value validation ---------------------------------------------------------


def _finite(value: Any, *, code: str, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(code, field_name)
    parsed = float(value)
    if not math.isfinite(parsed):
        raise _fail(code, field_name)
    return 0.0 if parsed == 0.0 else parsed


def _identifier(value: Any, *, code: str, field_name: str) -> str:
    if type(value) is not str or not _IDENTIFIER_RE.fullmatch(value):
        raise _fail(code, field_name)
    if any(fragment in value for fragment in _FORBIDDEN_FRAGMENTS):
        raise _fail(code, field_name)
    return value


def _round_reference(value: Any) -> str:
    if type(value) is not str or not _DIGEST_RE.fullmatch(value):
        raise _fail("identifier_invalid", "round_digest")
    return value


def _count(value: Any, *, code: str, field_name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _fail(code, field_name)
    return value


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


# --- public value types -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AccountantCompositionParameters:
    """Composition parameters that every checkpoint of one accountant shares.

    Args:
        composition: Composition rule name, such as ``"basic"`` or ``"rdp"``.
        delta: Target delta used to convert the composed state to epsilon.
        max_epsilon: Epsilon ceiling; a reservation that would exceed it is
            refused.
        orders: Optional strictly increasing integer Renyi orders, each at
            least two.
    """

    composition: str
    delta: float
    max_epsilon: float
    orders: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.composition) is not str or not _COMPOSITION_RE.fullmatch(
            self.composition
        ):
            raise _fail("parameters_invalid", "composition")
        delta = _finite(self.delta, code="parameters_invalid", field_name="delta")
        if not 0.0 < delta < 1.0:
            raise _fail("parameters_invalid", "delta")
        limit = _finite(
            self.max_epsilon, code="parameters_invalid", field_name="max_epsilon"
        )
        if limit <= 0.0:
            raise _fail("parameters_invalid", "max_epsilon")
        if isinstance(self.orders, (str, bytes)) or not isinstance(
            self.orders, Sequence
        ):
            raise _fail("parameters_invalid", "orders")
        orders = tuple(self.orders)
        if len(orders) > _MAX_ORDERS:
            raise _fail("parameters_invalid", "orders")
        for order in orders:
            if type(order) is not int or order < 2:
                raise _fail("parameters_invalid", "orders")
        if list(orders) != sorted(set(orders)):
            raise _fail("parameters_invalid", "orders")
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "max_epsilon", limit)
        object.__setattr__(self, "orders", orders)

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible parameter mapping."""

        return {
            "composition": self.composition,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
            "orders": list(self.orders),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AccountantCompositionParameters:
        """Parse a strict parameter mapping.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The validated parameters.

        Raises:
            AccountantCheckpointError: ``parameters_invalid`` when malformed.
        """

        if not isinstance(payload, Mapping) or set(payload) != {
            "composition",
            "delta",
            "max_epsilon",
            "orders",
        }:
            raise _fail("parameters_invalid", "parameters")
        orders = payload["orders"]
        if not isinstance(orders, list):
            raise _fail("parameters_invalid", "orders")
        return cls(
            composition=payload["composition"],
            delta=payload["delta"],
            max_epsilon=payload["max_epsilon"],
            orders=tuple(orders),
        )


@dataclass(frozen=True, slots=True)
class AccountantMechanism:
    """One reserved round mechanism as seen by the injected accountant.

    Args:
        round_digest: Opaque ``sha256:`` reference to the caller's round
            receipt. The receipt itself, with any site identity or seed, never
            enters the checkpoint.
        mechanism: Mechanism kind, such as ``"subsampled_gaussian"``.
        parameters: Finite numeric mechanism parameters, such as
            ``sampling_rate`` and ``noise_multiplier``. Names that suggest
            identities, seeds or contributions are refused.
    """

    round_digest: str
    mechanism: str
    parameters: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_digest", _round_reference(self.round_digest))
        object.__setattr__(
            self,
            "mechanism",
            _identifier(self.mechanism, code="mechanism_invalid", field_name="kind"),
        )
        raw = self.parameters
        if not isinstance(raw, Mapping) or not raw:
            raise _fail("mechanism_invalid", "parameters")
        if len(raw) > _MAX_MECHANISM_PARAMETERS:
            raise _fail("mechanism_invalid", "parameters")
        values: dict[str, float] = {}
        for name in sorted(raw, key=lambda item: str(item)):
            if type(name) is not str or not _PARAMETER_NAME_RE.fullmatch(name):
                raise _fail("mechanism_invalid", "parameter_name")
            if any(fragment in name for fragment in _FORBIDDEN_FRAGMENTS):
                raise _fail("mechanism_invalid", "parameter_name")
            values[name] = _finite(
                raw[name], code="mechanism_invalid", field_name="parameter_value"
            )
        object.__setattr__(self, "parameters", MappingProxyType(values))

    def __repr__(self) -> str:
        return f"AccountantMechanism(mechanism={self.mechanism!r})"


@dataclass(frozen=True, slots=True)
class CheckpointIntegrityKey:
    """Injected HMAC key that binds checkpoints to their legitimate writer.

    Args:
        key_id: Stable, non-secret key identifier recorded in checkpoints.
        secret: Secret key bytes, at least 32 bytes. Hidden from ``repr``.
    """

    key_id: str
    secret: bytes = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.key_id) is not str or not _IDENTIFIER_RE.fullmatch(self.key_id):
            raise _fail("integrity_key_invalid", "key_id")
        if type(self.secret) is not bytes or len(self.secret) < _MIN_SECRET_BYTES:
            raise _fail("integrity_key_invalid", "secret")


@dataclass(frozen=True, slots=True)
class StoredAccountantCheckpoint:
    """One durable store record: a generation and its canonical payload.

    Args:
        generation: Store generation; it must equal the checkpoint generation.
        payload: Canonical checkpoint JSON. Hidden from ``repr``.
    """

    generation: int
    payload: str = field(repr=False)


@runtime_checkable
class AccountantStatePort(Protocol):
    """Injected composed accountant, such as a round-aware Renyi accountant.

    ``compose`` must be deterministic and monotone: adding a mechanism never
    lowers the bound. The checkpoint layer recomputes the bound from history on
    every restore instead of trusting a stored number.
    """

    accountant_id: str
    accountant_version: str

    def compose(
        self,
        mechanisms: Sequence[AccountantMechanism],
        parameters: AccountantCompositionParameters,
    ) -> float:
        """Return the composed epsilon at ``parameters.delta``."""


@runtime_checkable
class AccountantCheckpointStore(Protocol):
    """Injected durable store with compare-and-swap on the generation."""

    def load(self) -> StoredAccountantCheckpoint | None:
        """Return the latest record, or ``None`` when no record exists."""

    def compare_and_swap(
        self, *, expected_generation: int, record: StoredAccountantCheckpoint
    ) -> bool:
        """Write ``record`` only if the current generation is expected.

        ``expected_generation`` is zero for an empty store, and
        ``record.generation`` must be ``expected_generation + 1``.
        """


class InMemoryAccountantCheckpointStore:
    """Thread-safe in-memory reference store for tests and examples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._record: StoredAccountantCheckpoint | None = None

    def load(self) -> StoredAccountantCheckpoint | None:
        """Return the latest record, or ``None`` when the store is empty."""

        with self._lock:
            return self._record

    def compare_and_swap(
        self, *, expected_generation: int, record: StoredAccountantCheckpoint
    ) -> bool:
        """Apply ``record`` when the generation matches and advances by one.

        Args:
            expected_generation: Generation the writer last observed.
            record: Record whose generation is ``expected_generation + 1``.

        Returns:
            ``True`` when the record was written, ``False`` for a stale writer.
        """

        with self._lock:
            current = 0 if self._record is None else self._record.generation
            if current != expected_generation:
                return False
            if record.generation != expected_generation + 1:
                return False
            self._record = record
            return True

    def __repr__(self) -> str:
        record = self._record
        generation = 0 if record is None else record.generation
        return f"InMemoryAccountantCheckpointStore(generation={generation})"


@dataclass(frozen=True, slots=True)
class AccountantReceipt:
    """Aggregate-only evidence of one checkpoint generation.

    Keep the latest receipt outside the store; passing it as a restore
    ``witness`` detects a rolled-back or re-created store. Receipts carry no
    site identity, random seed, individual contribution or round reference.
    """

    accountant_id: str
    accountant_version: str
    composition: str
    parameters_fingerprint: str
    generation: int
    transition: str
    head_digest: str
    checkpoint_digest: str
    committed_rounds: int
    abandoned_rounds: int
    uncertain_rounds: int
    charged_epsilon: float
    committed_epsilon: float
    remaining_epsilon: float
    delta: float
    max_epsilon: float
    schema_version: str = FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        code = "witness_invalid"
        if self.schema_version != FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION:
            raise _fail(code, "schema_version")
        _identifier(self.accountant_id, code=code, field_name="accountant_id")
        if type(self.accountant_version) is not str or not _VERSION_RE.fullmatch(
            self.accountant_version
        ):
            raise _fail(code, "accountant_version")
        if type(self.composition) is not str or not _COMPOSITION_RE.fullmatch(
            self.composition
        ):
            raise _fail(code, "composition")
        for name in ("parameters_fingerprint", "head_digest", "checkpoint_digest"):
            value = getattr(self, name)
            if type(value) is not str or not _DIGEST_RE.fullmatch(value):
                raise _fail(code, name)
        if self.transition not in ("genesis", "reserve", "reconcile"):
            raise _fail(code, "transition")
        _count(self.generation, code=code, field_name="generation", minimum=1)
        for name in ("committed_rounds", "abandoned_rounds", "uncertain_rounds"):
            _count(getattr(self, name), code=code, field_name=name)
        for name in (
            "charged_epsilon",
            "committed_epsilon",
            "remaining_epsilon",
            "delta",
            "max_epsilon",
        ):
            value = _finite(getattr(self, name), code=code, field_name=name)
            if value < 0.0:
                raise _fail(code, name)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed-shape, aggregate-only receipt mapping."""

        return {
            "schema_version": self.schema_version,
            "accountant_id": self.accountant_id,
            "accountant_version": self.accountant_version,
            "composition": self.composition,
            "parameters_fingerprint": self.parameters_fingerprint,
            "generation": self.generation,
            "transition": self.transition,
            "head_digest": self.head_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "committed_rounds": self.committed_rounds,
            "abandoned_rounds": self.abandoned_rounds,
            "uncertain_rounds": self.uncertain_rounds,
            "charged_epsilon": self.charged_epsilon,
            "committed_epsilon": self.committed_epsilon,
            "remaining_epsilon": self.remaining_epsilon,
            "delta": self.delta,
            "max_epsilon": self.max_epsilon,
        }

    def to_json(self) -> str:
        """Return canonical receipt JSON."""

        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AccountantReceipt:
        """Parse a strict receipt mapping, for example a stored witness.

        Args:
            payload: Mapping produced by :meth:`to_dict`.

        Returns:
            The validated receipt.

        Raises:
            AccountantCheckpointError: ``witness_invalid`` when malformed.
        """

        expected = set(cls.__dataclass_fields__)
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise _fail("witness_invalid", "receipt")
        return cls(**{name: payload[name] for name in expected})


# --- hashing and sealing ------------------------------------------------------


def _chain_digest(previous: str, event: Mapping[str, Any]) -> str:
    body = {key: value for key, value in event.items() if key != "chain_digest"}
    message = (
        _CHAIN_DOMAIN
        + previous.encode("ascii")
        + b"\0"
        + _canonical_json(body).encode("ascii")
    )
    return _sha256(message)


def _mac(body: Mapping[str, Any], key: CheckpointIntegrityKey) -> str:
    unsigned = {name: value for name, value in body.items() if name != "integrity"}
    message = (
        _MAC_DOMAIN
        + key.key_id.encode("ascii")
        + b"\0"
        + _canonical_json(unsigned).encode("ascii")
    )
    return hmac.new(key.secret, message, hashlib.sha256).hexdigest()


def _seal_document(
    body: Mapping[str, Any], key: CheckpointIntegrityKey
) -> dict[str, Any]:
    sealed = {name: value for name, value in body.items() if name != "integrity"}
    sealed["integrity"] = {
        "algorithm": _MAC_ALGORITHM,
        "key_id": key.key_id,
        "mac": _mac(sealed, key),
    }
    return sealed


def _parameters_fingerprint(
    accountant_id: str,
    accountant_version: str,
    parameters: AccountantCompositionParameters,
) -> str:
    payload = {
        "accountant_id": accountant_id,
        "accountant_version": accountant_version,
        "parameters": parameters.to_dict(),
        "schema_version": FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION,
    }
    return _sha256(_canonical_json(payload).encode("ascii"))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _reject_constant(_value: str) -> Any:
    raise ValueError("non-finite number")


def _parse_payload(payload: Any) -> dict[str, Any]:
    if type(payload) is not str or not payload.isascii():
        raise _fail("checkpoint_malformed", "payload")
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, TypeError, RecursionError):
        raise _fail("checkpoint_malformed", "payload") from None
    if not isinstance(document, dict):
        raise _fail("checkpoint_malformed", "payload")
    try:
        canonical = _canonical_json(document)
    except (ValueError, TypeError):
        raise _fail("checkpoint_malformed", "payload") from None
    if canonical != payload:
        raise _fail("checkpoint_malformed", "canonical_form")
    return document


# --- replayed state -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Replay:
    events: tuple[Mapping[str, Any], ...]
    charged: tuple[AccountantMechanism, ...]
    committed: tuple[AccountantMechanism, ...]
    uncertain: tuple[str, ...]
    used_rounds: frozenset[str]
    abandoned_count: int
    boundary_index: int
    boundary_digest: str


def _replay(
    events: Sequence[Any],
    *,
    accountant_id: str,
    accountant_version: str,
    fingerprint: str,
) -> _Replay:
    if not isinstance(events, list) or not events:
        raise _fail("checkpoint_malformed", "events")
    if len(events) > MAX_ACCOUNTANT_EVENTS:
        raise _fail("history_limit_reached", "events")
    previous = _GENESIS_PREVIOUS_DIGEST
    reserved: dict[str, AccountantMechanism] = {}
    charged: list[AccountantMechanism] = []
    committed: list[AccountantMechanism] = []
    pending: list[str] = []
    abandoned = 0
    boundary_index = 0
    boundary_digest = ""
    for position, event in enumerate(events):
        if not isinstance(event, dict):
            raise _fail("checkpoint_malformed", "event")
        kind = event.get("kind")
        if kind not in _EVENT_KEYS or set(event) != _EVENT_KEYS[kind]:
            raise _fail("checkpoint_malformed", "event")
        if event["index"] != position or type(event["index"]) is not int:
            raise _fail("history_out_of_order", "index")
        if (position == 0) != (kind == "genesis"):
            raise _fail("history_out_of_order", "genesis")
        if not hmac.compare_digest(
            str(event["chain_digest"]), _chain_digest(previous, event)
        ):
            raise _fail("history_chain_broken", "chain_digest")
        previous = event["chain_digest"]
        if kind == "genesis":
            _identifier(
                event["genesis_id"], code="checkpoint_malformed", field_name="genesis"
            )
            if (
                event["accountant_id"] != accountant_id
                or event["accountant_version"] != accountant_version
                or event["parameters_fingerprint"] != fingerprint
            ):
                raise _fail("accountant_mismatch", "genesis")
            boundary_digest = previous
            continue
        reference = event["round_digest"]
        if kind == "reserve":
            mechanism = _mechanism_from_event(event)
            if mechanism.round_digest in reserved:
                raise _fail("history_out_of_order", "round_digest")
            reserved[mechanism.round_digest] = mechanism
            charged.append(mechanism)
            pending.append(mechanism.round_digest)
            continue
        if reference not in pending:
            raise _fail("history_out_of_order", "reconcile")
        pending.remove(reference)
        if event["outcome"] == "committed":
            committed.append(reserved[reference])
        elif event["outcome"] == "abandoned":
            abandoned += 1
        else:
            raise _fail("checkpoint_malformed", "outcome")
        boundary_index = position
        boundary_digest = previous
    return _Replay(
        events=tuple(events),
        charged=tuple(charged),
        committed=tuple(committed),
        uncertain=tuple(pending),
        used_rounds=frozenset(reserved),
        abandoned_count=abandoned,
        boundary_index=boundary_index,
        boundary_digest=boundary_digest,
    )


def _mechanism_from_event(event: Mapping[str, Any]) -> AccountantMechanism:
    try:
        return AccountantMechanism(
            round_digest=event["round_digest"],
            mechanism=event["mechanism"],
            parameters=event["parameters"],
        )
    except AccountantCheckpointError:
        raise _fail("checkpoint_malformed", "mechanism") from None


def _compose(
    accountant: AccountantStatePort,
    mechanisms: Sequence[AccountantMechanism],
    parameters: AccountantCompositionParameters,
) -> float:
    if not mechanisms:
        return 0.0
    try:
        value = accountant.compose(tuple(mechanisms), parameters)
    except Exception:
        raise _fail("accountant_failure", "compose") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("accountant_failure", "compose")
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise _fail("accountant_failure", "compose")
    return value


def _ledger(
    *,
    charged_epsilon: float,
    committed_epsilon: float,
    charged_rounds: int,
    committed_rounds: int,
    parameters: AccountantCompositionParameters,
    fingerprint: str,
) -> _budget_migration.BudgetLedgerSnapshot:
    """Project accountant state onto the budget-migration ledger format."""

    def entry(release_id: str, spent: float, rounds: int):
        return _budget_migration.BudgetEntry(
            release_id=release_id,
            spent_epsilon=spent,
            spent_delta=parameters.delta if rounds else 0.0,
            max_epsilon=parameters.max_epsilon,
            max_delta=parameters.delta,
            composition=parameters.composition,
            policy_fingerprint=fingerprint,
        )

    try:
        return _budget_migration.BudgetLedgerSnapshot(
            entries=(
                entry(_LEDGER_CHARGED, charged_epsilon, charged_rounds),
                entry(_LEDGER_COMMITTED, committed_epsilon, committed_rounds),
            )
        )
    except _budget_migration.BudgetMigrationError:
        raise _fail("budget_exceeded", "max_epsilon") from None


def _enforce_monotone(
    before: _budget_migration.BudgetLedgerSnapshot,
    after: _budget_migration.BudgetLedgerSnapshot,
) -> None:
    try:
        _budget_migration.enforce_budget_migration(before, after)
    except _budget_migration.BudgetMigrationError:
        raise _fail("budget_regressed", "spent_budget") from None


@dataclass(frozen=True, slots=True)
class _State:
    document: Mapping[str, Any]
    replay: _Replay
    charged_epsilon: float
    committed_epsilon: float
    payload: str

    @property
    def generation(self) -> int:
        return len(self.replay.events)


class CheckpointedAccountant:
    """Durable, fenced and monotone composed-accountant session.

    Create the first state with :meth:`genesis` and recover it with
    :meth:`restore`; the constructor is private. Each transition appends one
    event, recomputes the composed bound with the injected accountant, verifies
    monotonicity with :func:`openmed.risk.budget_migration.enforce_budget_migration`
    and writes the next generation through compare-and-swap. A failed or
    ambiguous write poisons the session so it must be restored.
    """

    def __init__(
        self,
        *,
        _store: AccountantCheckpointStore,
        _accountant: AccountantStatePort,
        _parameters: AccountantCompositionParameters,
        _key: CheckpointIntegrityKey,
        _state: _State,
    ) -> None:
        self._store = _store
        self._accountant = _accountant
        self._parameters = _parameters
        self._key = _key
        self._state = _state
        self._lock = threading.Lock()
        self._poisoned = False

    # -- construction ---------------------------------------------------------

    @classmethod
    def genesis(
        cls,
        *,
        store: AccountantCheckpointStore,
        accountant: AccountantStatePort,
        parameters: AccountantCompositionParameters,
        integrity_key: CheckpointIntegrityKey,
        genesis_id: str,
    ) -> CheckpointedAccountant:
        """Record an explicit fresh accountant in an empty store.

        Args:
            store: Durable store; it must hold no record.
            accountant: Injected composed accountant.
            parameters: Composition parameters for the whole history.
            integrity_key: HMAC key that seals every checkpoint.
            genesis_id: Opaque operator identifier recorded in the genesis
                event.

        Returns:
            A session at generation one with no spent budget.

        Raises:
            AccountantCheckpointError: ``genesis_conflict`` when the store
                already holds state, or another fixed code when an input is
                invalid.
        """

        accountant_id, accountant_version = _accountant_identity(accountant)
        _require_types(parameters, integrity_key)
        genesis_id = _identifier(
            genesis_id, code="identifier_invalid", field_name="genesis_id"
        )
        if _load(store) is not None:
            raise _fail("genesis_conflict", "store")
        fingerprint = _parameters_fingerprint(
            accountant_id, accountant_version, parameters
        )
        event: dict[str, Any] = {
            "index": 0,
            "kind": "genesis",
            "genesis_id": genesis_id,
            "accountant_id": accountant_id,
            "accountant_version": accountant_version,
            "parameters_fingerprint": fingerprint,
        }
        event["chain_digest"] = _chain_digest(_GENESIS_PREVIOUS_DIGEST, event)
        replay = _replay(
            [event],
            accountant_id=accountant_id,
            accountant_version=accountant_version,
            fingerprint=fingerprint,
        )
        state = _build_state(
            replay,
            accountant_id=accountant_id,
            accountant_version=accountant_version,
            parameters=parameters,
            fingerprint=fingerprint,
            charged_epsilon=0.0,
            committed_epsilon=0.0,
            key=integrity_key,
        )
        if not _swap(store, 0, state):
            raise _fail("genesis_conflict", "store")
        return cls(
            _store=store,
            _accountant=accountant,
            _parameters=parameters,
            _key=integrity_key,
            _state=state,
        )

    @classmethod
    def restore(
        cls,
        *,
        store: AccountantCheckpointStore,
        accountant: AccountantStatePort,
        parameters: AccountantCompositionParameters,
        integrity_key: CheckpointIntegrityKey,
        witness: AccountantReceipt | Mapping[str, Any] | None = None,
    ) -> CheckpointedAccountant:
        """Restore the latest checkpoint, refusing incompatible or older state.

        The restore never starts a fresh accountant. It verifies the canonical
        form, schema, HMAC, store generation, accountant identity, parameters,
        hash chain and committed boundary, recomputes the bound from history
        and requires it to be no lower than the recorded bound. With a
        ``witness`` receipt it also refuses a store that is older than, or
        diverged from, the state the witness attests.

        Args:
            store: Durable store holding the latest checkpoint.
            accountant: Injected composed accountant with the recorded identity.
            parameters: Expected composition parameters.
            integrity_key: HMAC key that sealed the checkpoint.
            witness: Optional latest receipt kept outside the store.

        Returns:
            A session at the stored generation, with uncertain rounds charged.

        Raises:
            AccountantCheckpointError: With a fixed reason code on refusal.
        """

        accountant_id, accountant_version = _accountant_identity(accountant)
        _require_types(parameters, integrity_key)
        fingerprint = _parameters_fingerprint(
            accountant_id, accountant_version, parameters
        )
        receipt = _coerce_witness(witness)
        if receipt is not None and (
            receipt.accountant_id != accountant_id
            or receipt.accountant_version != accountant_version
            or receipt.parameters_fingerprint != fingerprint
        ):
            raise _fail("witness_invalid", "accountant")
        record = _load(store)
        if record is None:
            raise _fail("checkpoint_missing", "store")
        if not isinstance(record, StoredAccountantCheckpoint):
            raise _fail("store_failure", "record")
        document = _parse_payload(record.payload)
        if set(document) != _DOCUMENT_KEYS:
            if document.get("schema_version") not in (
                None,
                FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION,
            ):
                raise _fail("schema_unsupported", "schema_version")
            raise _fail("checkpoint_malformed", "document")
        _verify_integrity(document, integrity_key)
        if document["schema_version"] != FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION:
            raise _fail("schema_unsupported", "schema_version")
        generation = _count(
            document["generation"],
            code="checkpoint_malformed",
            field_name="generation",
            minimum=1,
        )
        if generation != record.generation:
            raise _fail("generation_mismatch", "generation")
        if document["accountant"] != {
            "accountant_id": accountant_id,
            "accountant_version": accountant_version,
        }:
            raise _fail("accountant_mismatch", "accountant")
        if document["parameters"] != parameters.to_dict():
            raise _fail("parameters_mismatch", "parameters")
        if document["parameters_fingerprint"] != fingerprint:
            raise _fail("parameters_mismatch", "parameters_fingerprint")
        replay = _replay(
            document["events"],
            accountant_id=accountant_id,
            accountant_version=accountant_version,
            fingerprint=fingerprint,
        )
        if len(replay.events) != generation:
            raise _fail("history_truncated", "generation")
        _verify_boundary(document, replay)
        recorded_charged, recorded_committed = _recorded_bound(document["bound"])
        charged = _compose(accountant, replay.charged, parameters)
        committed = _compose(accountant, replay.committed, parameters)
        recorded = _ledger(
            charged_epsilon=recorded_charged,
            committed_epsilon=recorded_committed,
            charged_rounds=len(replay.charged),
            committed_rounds=len(replay.committed),
            parameters=parameters,
            fingerprint=fingerprint,
        )
        recomputed = _ledger(
            charged_epsilon=charged,
            committed_epsilon=committed,
            charged_rounds=len(replay.charged),
            committed_rounds=len(replay.committed),
            parameters=parameters,
            fingerprint=fingerprint,
        )
        _enforce_monotone(recorded, recomputed)
        if committed > charged:
            raise _fail("accountant_failure", "compose")
        state = _State(
            document=document,
            replay=replay,
            charged_epsilon=charged,
            committed_epsilon=committed,
            payload=record.payload,
        )
        if receipt is not None:
            _verify_witness(receipt, replay, recomputed, parameters, fingerprint)
        return cls(
            _store=store,
            _accountant=accountant,
            _parameters=parameters,
            _key=integrity_key,
            _state=state,
        )

    # -- transitions ----------------------------------------------------------

    def begin_round(
        self,
        round_digest: str,
        *,
        mechanism: str,
        parameters: Mapping[str, float],
    ) -> AccountantReceipt:
        """Durably reserve and charge a round before its mechanism runs.

        Run the mechanism only after this call returns. If the process stops
        before :meth:`commit_round` or :meth:`reconcile_round`, the charge is
        retained as uncertain expenditure across restores.

        Args:
            round_digest: Opaque ``sha256:`` reference to the round receipt.
            mechanism: Mechanism kind.
            parameters: Finite numeric mechanism parameters.

        Returns:
            The receipt for the new generation.

        Raises:
            AccountantCheckpointError: ``uncertain_spend_unreconciled``,
                ``round_duplicate``, ``budget_exceeded``,
                ``generation_conflict`` or another fixed code.
        """

        record = AccountantMechanism(
            round_digest=round_digest, mechanism=mechanism, parameters=parameters
        )
        with self._lock:
            self._require_live()
            replay = self._state.replay
            if replay.uncertain:
                raise _fail("uncertain_spend_unreconciled", "round_digest")
            if record.round_digest in replay.used_rounds:
                raise _fail("round_duplicate", "round_digest")
            event = {
                "index": len(replay.events),
                "kind": "reserve",
                "round_digest": record.round_digest,
                "mechanism": record.mechanism,
                "parameters": dict(record.parameters),
            }
            return self._append(event)

    def commit_round(self, round_digest: str) -> AccountantReceipt:
        """Reconcile a reserved round as committed.

        Args:
            round_digest: Reference passed to :meth:`begin_round`.

        Returns:
            The receipt for the new generation.

        Raises:
            AccountantCheckpointError: ``round_unknown`` or another fixed code.
        """

        return self.reconcile_round(round_digest, outcome="committed")

    def reconcile_round(self, round_digest: str, *, outcome: str) -> AccountantReceipt:
        """Explicitly close an uncertain round without releasing its charge.

        ``"committed"`` records that the round's output was released.
        ``"abandoned"`` records that it was not; its charge is still retained
        because a partially executed mechanism cannot be proven unobserved. A
        retry needs a new round reference and is charged again.

        Args:
            round_digest: Reference passed to :meth:`begin_round`.
            outcome: One of :data:`RECONCILIATION_OUTCOMES`.

        Returns:
            The receipt for the new generation.

        Raises:
            AccountantCheckpointError: ``outcome_invalid``, ``round_unknown`` or
                another fixed code.
        """

        reference = _round_reference(round_digest)
        if type(outcome) is not str or outcome not in RECONCILIATION_OUTCOMES:
            raise _fail("outcome_invalid", "outcome")
        with self._lock:
            self._require_live()
            replay = self._state.replay
            if reference not in replay.uncertain:
                raise _fail("round_unknown", "round_digest")
            event = {
                "index": len(replay.events),
                "kind": "reconcile",
                "round_digest": reference,
                "outcome": outcome,
            }
            return self._append(event)

    # -- read-only views ------------------------------------------------------

    @property
    def generation(self) -> int:
        """Return the current checkpoint generation."""

        return self._state.generation

    @property
    def charged_epsilon(self) -> float:
        """Return the bound over every reserved round, including uncertain ones."""

        return self._state.charged_epsilon

    @property
    def committed_epsilon(self) -> float:
        """Return the bound over rounds reconciled as committed."""

        return self._state.committed_epsilon

    @property
    def uncertain_rounds(self) -> int:
        """Return how many reserved rounds still await reconciliation."""

        return len(self._state.replay.uncertain)

    @property
    def receipt(self) -> AccountantReceipt:
        """Return the receipt for the current generation."""

        return _receipt(self._state, self._parameters)

    def __repr__(self) -> str:
        state = self._state
        return (
            "CheckpointedAccountant("
            f"generation={state.generation}, "
            f"uncertain_rounds={len(state.replay.uncertain)}, "
            f"stale={self._poisoned})"
        )

    # -- internals ------------------------------------------------------------

    def _require_live(self) -> None:
        if self._poisoned:
            raise _fail("restore_required", "session")

    def _append(self, event: dict[str, Any]) -> AccountantReceipt:
        current = self._state
        if len(current.replay.events) >= MAX_ACCOUNTANT_EVENTS:
            raise _fail("history_limit_reached", "events")
        previous = current.replay.events[-1]["chain_digest"]
        event["chain_digest"] = _chain_digest(previous, event)
        document = current.document
        accountant_id = document["accountant"]["accountant_id"]
        accountant_version = document["accountant"]["accountant_version"]
        fingerprint = document["parameters_fingerprint"]
        replay = _replay(
            [*current.replay.events, event],
            accountant_id=accountant_id,
            accountant_version=accountant_version,
            fingerprint=fingerprint,
        )
        charged = _compose(self._accountant, replay.charged, self._parameters)
        committed = _compose(self._accountant, replay.committed, self._parameters)
        if charged > self._parameters.max_epsilon:
            raise _fail("budget_exceeded", "max_epsilon")
        before = _ledger(
            charged_epsilon=current.charged_epsilon,
            committed_epsilon=current.committed_epsilon,
            charged_rounds=len(current.replay.charged),
            committed_rounds=len(current.replay.committed),
            parameters=self._parameters,
            fingerprint=fingerprint,
        )
        after = _ledger(
            charged_epsilon=charged,
            committed_epsilon=committed,
            charged_rounds=len(replay.charged),
            committed_rounds=len(replay.committed),
            parameters=self._parameters,
            fingerprint=fingerprint,
        )
        _enforce_monotone(before, after)
        if committed > charged:
            raise _fail("accountant_failure", "compose")
        state = _build_state(
            replay,
            accountant_id=accountant_id,
            accountant_version=accountant_version,
            parameters=self._parameters,
            fingerprint=fingerprint,
            charged_epsilon=charged,
            committed_epsilon=committed,
            key=self._key,
        )
        try:
            written = _swap(self._store, current.generation, state)
        except AccountantCheckpointError:
            self._poisoned = True
            raise
        if not written:
            self._poisoned = True
            raise _fail("generation_conflict", "generation")
        self._state = state
        return _receipt(state, self._parameters)


# --- module helpers -----------------------------------------------------------


def _accountant_identity(accountant: Any) -> tuple[str, str]:
    try:
        accountant_id = accountant.accountant_id
        accountant_version = accountant.accountant_version
        compose = accountant.compose
    except Exception:
        raise _fail("accountant_failure", "accountant") from None
    if not callable(compose):
        raise _fail("accountant_failure", "accountant")
    accountant_id = _identifier(
        accountant_id, code="identifier_invalid", field_name="accountant_id"
    )
    if type(accountant_version) is not str or not _VERSION_RE.fullmatch(
        accountant_version
    ):
        raise _fail("identifier_invalid", "accountant_version")
    return accountant_id, accountant_version


def _require_types(parameters: Any, key: Any) -> None:
    if not isinstance(parameters, AccountantCompositionParameters):
        raise _fail("parameters_invalid", "parameters")
    if not isinstance(key, CheckpointIntegrityKey):
        raise _fail("integrity_key_invalid", "integrity_key")


def _load(store: AccountantCheckpointStore) -> StoredAccountantCheckpoint | None:
    try:
        return store.load()
    except Exception:
        raise _fail("store_failure", "load") from None


def _swap(store: AccountantCheckpointStore, expected: int, state: _State) -> bool:
    record = StoredAccountantCheckpoint(
        generation=state.generation, payload=state.payload
    )
    try:
        written = store.compare_and_swap(expected_generation=expected, record=record)
    except Exception:
        raise _fail("store_failure", "compare_and_swap") from None
    if type(written) is not bool:
        raise _fail("store_failure", "compare_and_swap")
    return written


def _build_state(
    replay: _Replay,
    *,
    accountant_id: str,
    accountant_version: str,
    parameters: AccountantCompositionParameters,
    fingerprint: str,
    charged_epsilon: float,
    committed_epsilon: float,
    key: CheckpointIntegrityKey,
) -> _State:
    body = {
        "schema_version": FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION,
        "accountant": {
            "accountant_id": accountant_id,
            "accountant_version": accountant_version,
        },
        "parameters": parameters.to_dict(),
        "parameters_fingerprint": fingerprint,
        "generation": len(replay.events),
        "events": [dict(event) for event in replay.events],
        "committed_boundary": _boundary(replay),
        "uncertain_rounds": list(replay.uncertain),
        "bound": {
            "charged_epsilon": charged_epsilon,
            "committed_epsilon": committed_epsilon,
        },
        "head_digest": replay.events[-1]["chain_digest"],
    }
    document = _seal_document(body, key)
    return _State(
        document=document,
        replay=replay,
        charged_epsilon=charged_epsilon,
        committed_epsilon=committed_epsilon,
        payload=_canonical_json(document),
    )


def _boundary(replay: _Replay) -> dict[str, Any]:
    return {
        "committed_rounds": len(replay.committed),
        "abandoned_rounds": replay.abandoned_count,
        "event_index": replay.boundary_index,
        "chain_digest": replay.boundary_digest,
    }


def _verify_integrity(document: Mapping[str, Any], key: CheckpointIntegrityKey) -> None:
    integrity = document["integrity"]
    if not isinstance(integrity, dict) or set(integrity) != {
        "algorithm",
        "key_id",
        "mac",
    }:
        raise _fail("integrity_mismatch", "integrity")
    if integrity["algorithm"] != _MAC_ALGORITHM or integrity["key_id"] != key.key_id:
        raise _fail("integrity_mismatch", "integrity")
    if type(integrity["mac"]) is not str or not hmac.compare_digest(
        integrity["mac"], _mac(document, key)
    ):
        raise _fail("integrity_mismatch", "mac")


def _verify_boundary(document: Mapping[str, Any], replay: _Replay) -> None:
    boundary = document["committed_boundary"]
    if not isinstance(boundary, dict) or set(boundary) != _BOUNDARY_KEYS:
        raise _fail("checkpoint_malformed", "committed_boundary")
    if boundary != _boundary(replay):
        raise _fail("boundary_mismatch", "committed_boundary")
    if document["uncertain_rounds"] != list(replay.uncertain):
        raise _fail("boundary_mismatch", "uncertain_rounds")
    if document["head_digest"] != replay.events[-1]["chain_digest"]:
        raise _fail("boundary_mismatch", "head_digest")


def _recorded_bound(bound: Any) -> tuple[float, float]:
    if not isinstance(bound, dict) or set(bound) != {
        "charged_epsilon",
        "committed_epsilon",
    }:
        raise _fail("checkpoint_malformed", "bound")
    charged = _finite(
        bound["charged_epsilon"], code="checkpoint_malformed", field_name="bound"
    )
    committed = _finite(
        bound["committed_epsilon"], code="checkpoint_malformed", field_name="bound"
    )
    if charged < 0.0 or committed < 0.0:
        raise _fail("checkpoint_malformed", "bound")
    return charged, committed


def _coerce_witness(witness: Any) -> AccountantReceipt | None:
    if witness is None or isinstance(witness, AccountantReceipt):
        return witness
    if isinstance(witness, Mapping):
        return AccountantReceipt.from_dict(witness)
    raise _fail("witness_invalid", "witness")


def _verify_witness(
    witness: AccountantReceipt,
    replay: _Replay,
    restored: _budget_migration.BudgetLedgerSnapshot,
    parameters: AccountantCompositionParameters,
    fingerprint: str,
) -> None:
    if (
        witness.composition != parameters.composition
        or witness.delta != parameters.delta
        or witness.max_epsilon != parameters.max_epsilon
    ):
        raise _fail("witness_invalid", "parameters")
    if len(replay.events) < witness.generation:
        raise _fail("stale_checkpoint", "generation")
    attested = replay.events[witness.generation - 1]["chain_digest"]
    if not hmac.compare_digest(attested, witness.head_digest):
        raise _fail("history_diverged", "head_digest")
    charged_rounds = (
        witness.committed_rounds + witness.abandoned_rounds + witness.uncertain_rounds
    )
    attested_ledger = _ledger(
        charged_epsilon=witness.charged_epsilon,
        committed_epsilon=witness.committed_epsilon,
        charged_rounds=charged_rounds,
        committed_rounds=witness.committed_rounds,
        parameters=parameters,
        fingerprint=fingerprint,
    )
    _enforce_monotone(attested_ledger, restored)


def _receipt(
    state: _State, parameters: AccountantCompositionParameters
) -> AccountantReceipt:
    document = state.document
    replay = state.replay
    return AccountantReceipt(
        accountant_id=document["accountant"]["accountant_id"],
        accountant_version=document["accountant"]["accountant_version"],
        composition=parameters.composition,
        parameters_fingerprint=document["parameters_fingerprint"],
        generation=state.generation,
        transition=replay.events[-1]["kind"],
        head_digest=replay.events[-1]["chain_digest"],
        checkpoint_digest=_sha256(state.payload.encode("ascii")),
        committed_rounds=len(replay.committed),
        abandoned_rounds=replay.abandoned_count,
        uncertain_rounds=len(replay.uncertain),
        charged_epsilon=state.charged_epsilon,
        committed_epsilon=state.committed_epsilon,
        remaining_epsilon=max(0.0, parameters.max_epsilon - state.charged_epsilon),
        delta=parameters.delta,
        max_epsilon=parameters.max_epsilon,
    )
