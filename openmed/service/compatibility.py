"""Mixed-version compatibility contract for zero-downtime service upgrades.

A rolling upgrade briefly runs two builds side by side. This module declares
which builds may share durable state and traffic, and gives every surface one
fail-closed decision to make *before* it processes or mutates anything:

* :data:`CURRENT_CONTRACT` publishes this build's contract generation, the
  oldest peer generation it shares state with, the schema range it reads and
  the schema version it writes for every durable component, and its worker
  capabilities.
* :func:`negotiate_peer` and :func:`peer_contract_decision` decide whether a
  peer build may run beside this one.
* :func:`check_work` decides whether this node may claim one unit of persisted
  work (a job record, checkpoint or webhook delivery) before mutating it.
* :func:`evaluate_persisted_state` gates readiness on the schema versions of
  the shared stores.
* :func:`rollback_decision` says whether returning to an older build is still
  inside the declared window.

Compatibility rule (N/N-1). Generation numbers increase monotonically and each
build declares a window of at most one prior generation. The newer of two
builds owns the decision: two builds are compatible when the newer build's
``min_peer_generation`` is at most the older build's generation and, for every
durable component both use, each build reads the schema version the other
writes. A release may therefore only start *writing* a schema version that its
predecessor already *reads* (expand first, contract in the next generation).

Every output is value-free: controlled lowercase tokens, integers and SHA-256
digests. Hostnames, paths, payloads, identifiers, configuration values and
exception text never enter a descriptor, decision, report or evidence record.
Partial or unknown compatibility is always a blocking result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

COMPATIBILITY_SCHEMA_VERSION = "openmed.service.compatibility.v1"
COMPATIBILITY_MATRIX_SCHEMA_VERSION = "openmed.service.compatibility-matrix.v1"
ROLLOUT_EVIDENCE_SCHEMA_VERSION = "openmed.service.rollout-evidence.v1"

#: Contract generation published by this build. Bump it whenever a wire
#: contract, durable schema or worker capability changes in a way an older
#: build cannot ignore, and declare the new window in :data:`CONTRACT_HISTORY`.
CONTRACT_GENERATION = 1
#: Oldest peer generation this build shares durable state and traffic with.
MIN_PEER_GENERATION = 1
#: Schema version of the async job metadata store written by this build. The
#: unversioned layout written before this contract existed is version 1.
JOB_STORE_SCHEMA_VERSION = 1

PEER_CONTRACT_HEADER = "X-OpenMed-Peer-Contract"
PEER_CONTRACT_METADATA_KEY = "x-openmed-peer-contract"
DEFAULT_REFRESH_INTERVAL_SECONDS = 5.0
MAX_GENERATION = 9999
MAX_SCHEMA_VERSION = 9999

JOB_STORE_COMPONENT = "jobs"
JOURNEY_LOCAL_COMPONENT = "journey_local"
JOURNEY_POSTGRES_COMPONENT = "journey_postgres"

JOB_DEIDENTIFY_CAPABILITY = "jobs.deidentify.v1"
JOB_WEBHOOK_CAPABILITY = "jobs.webhook.v1"
STREAM_NDJSON_CAPABILITY = "stream.ndjson.v1"
WORKER_CAPABILITIES = frozenset(
    {JOB_DEIDENTIFY_CAPABILITY, JOB_WEBHOOK_CAPABILITY, STREAM_NDJSON_CAPABILITY}
)

PEER_REASON_CODES = frozenset(
    {
        "peer_compatible",
        "peer_contract_malformed",
        "peer_descriptor_unsupported",
        "peer_generation_outside_window",
        "peer_schema_outside_window",
    }
)
WORK_REASON_CODES = frozenset(
    {
        "work_compatible",
        "work_envelope_invalid",
        "work_generation_outside_window",
        "work_component_unknown",
        "work_schema_outside_window",
        "capability_missing",
        "model_generation_unavailable",
    }
)
STATE_REASON_CODES = frozenset(
    {
        "schema_compatible",
        "schema_absent",
        "schema_migration_required",
        "schema_ahead_of_window",
        "schema_unknown",
    }
)
ROLLBACK_REASON_CODES = frozenset({"rollback_within_window"})
REASON_CODES = (
    PEER_REASON_CODES | WORK_REASON_CODES | STATE_REASON_CODES | ROLLBACK_REASON_CODES
)

_STATE_BY_CODE = {
    "schema_migration_required": "pending",
    "schema_ahead_of_window": "incompatible",
    "schema_unknown": "unknown",
}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{0,31}(?:\.[a-z0-9_]{1,32}){0,4}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PEER_CONTRACT_RE = re.compile(r"([1-9][0-9]{0,3})/([1-9][0-9]{0,3})\Z")
_DESCRIPTOR_KEYS = frozenset(
    {
        "schema_version",
        "contract_generation",
        "min_peer_generation",
        "durable_schemas",
        "capabilities",
        "model_generations",
    }
)
_ENVELOPE_KEYS = frozenset(
    {
        "contract_generation",
        "min_peer_generation",
        "schema_versions",
        "required_capabilities",
        "model_generation",
    }
)


class CompatibilityError(ValueError):
    """Raised when a contract, descriptor or envelope is not well formed.

    Attributes:
        code: Controlled reason code; the message never contains input values.
    """

    def __init__(self, code: str) -> None:
        super().__init__("Compatibility contract input is invalid")
        self.code = code


class IncompatibleStateError(RuntimeError):
    """Raised before a durable store is mutated outside the declared window.

    Attributes:
        component: Controlled durable component name.
        code: Controlled reason code from :data:`STATE_REASON_CODES`.
    """

    def __init__(self, component: str, code: str) -> None:
        super().__init__("Persisted state is outside the compatibility window")
        self.component = component
        self.code = code

    @property
    def state(self) -> str:
        """Return the blocking readiness state implied by :attr:`code`."""
        return _STATE_BY_CODE.get(self.code, "unknown")

    def to_details(self) -> dict[str, Any]:
        """Return value-free error details for service envelopes."""
        return {
            "components": {self.component: {"state": self.state, "reason": self.code}}
        }


def _is_version(value: Any, *, maximum: int) -> bool:
    return type(value) is int and 1 <= value <= maximum


def _require_version(value: Any, *, maximum: int, code: str) -> int:
    if not _is_version(value, maximum=maximum):
        raise CompatibilityError(code)
    return int(value)


def _require_token(value: Any, *, code: str) -> str:
    if type(value) is not str or _TOKEN_RE.fullmatch(value) is None:
        raise CompatibilityError(code)
    return value


def _require_digest(value: Any, *, code: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise CompatibilityError(code)
    return value


def _require_window(generation: int, min_peer_generation: int, *, code: str) -> None:
    if not generation - 1 <= min_peer_generation <= generation:
        raise CompatibilityError(code)


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def model_generation_ref(model_id: str) -> str:
    """Return the value-free reference used to pin work to a model generation.

    Args:
        model_id: Public served model identifier or rollout pointer.

    Returns:
        ``sha256:<hex>`` digest of the identifier.
    """
    if type(model_id) is not str or not model_id:
        raise CompatibilityError("model_generation_invalid")
    return f"sha256:{hashlib.sha256(model_id.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True, slots=True)
class SchemaRange:
    """Schema versions one build reads and writes for one durable component.

    Attributes:
        read_min: Oldest persisted version the build operates on without a
            migration.
        read_max: Newest persisted version the build operates on.
        write: Version the build writes. It must lie inside the read range.
    """

    read_min: int
    read_max: int
    write: int

    def __post_init__(self) -> None:
        for value in (self.read_min, self.read_max, self.write):
            _require_version(value, maximum=MAX_SCHEMA_VERSION, code="schema_invalid")
        if not self.read_min <= self.write <= self.read_max:
            raise CompatibilityError("schema_invalid")

    def reads(self, version: int) -> bool:
        """Return whether this build operates on persisted ``version``."""
        return self.read_min <= version <= self.read_max

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free wire representation."""
        return {"read": [self.read_min, self.read_max], "write": self.write}

    @classmethod
    def from_dict(cls, payload: Any) -> "SchemaRange":
        """Parse a strict wire representation.

        Raises:
            CompatibilityError: When the payload is not a valid range.
        """
        if not isinstance(payload, Mapping) or set(payload) != {"read", "write"}:
            raise CompatibilityError("schema_invalid")
        read = payload["read"]
        if not isinstance(read, list) or len(read) != 2:
            raise CompatibilityError("schema_invalid")
        return cls(read_min=read[0], read_max=read[1], write=payload["write"])


def _freeze_schemas(schemas: Mapping[str, SchemaRange]) -> Mapping[str, SchemaRange]:
    if not isinstance(schemas, Mapping):
        raise CompatibilityError("contract_invalid")
    frozen: dict[str, SchemaRange] = {}
    for name in sorted(schemas):
        _require_token(name, code="contract_invalid")
        value = schemas[name]
        if not isinstance(value, SchemaRange):
            raise CompatibilityError("contract_invalid")
        frozen[name] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class CompatibilityDescriptor:
    """Value-free compatibility contract one build publishes to its peers.

    Attributes:
        contract_generation: Monotonic contract generation of the build.
        min_peer_generation: Oldest peer generation the build shares state
            with; either the build's own generation or the one before it.
        durable_schemas: Read range and write version per durable component.
        capabilities: Worker capability tokens the build can execute.
        model_generations: Digests of the model generations the build serves.
    """

    contract_generation: int
    min_peer_generation: int
    durable_schemas: Mapping[str, SchemaRange]
    capabilities: frozenset[str]
    model_generations: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        generation = _require_version(
            self.contract_generation, maximum=MAX_GENERATION, code="contract_invalid"
        )
        minimum = _require_version(
            self.min_peer_generation, maximum=MAX_GENERATION, code="contract_invalid"
        )
        _require_window(generation, minimum, code="contract_invalid")
        object.__setattr__(
            self, "durable_schemas", _freeze_schemas(self.durable_schemas)
        )
        object.__setattr__(
            self,
            "capabilities",
            frozenset(
                _require_token(item, code="contract_invalid")
                for item in self.capabilities
            ),
        )
        object.__setattr__(
            self,
            "model_generations",
            frozenset(
                _require_digest(item, code="contract_invalid")
                for item in self.model_generations
            ),
        )

    @property
    def peer_contract(self) -> str:
        """Return the compact ``<generation>/<min_peer_generation>`` token."""
        return f"{self.contract_generation}/{self.min_peer_generation}"

    def with_model_generations(self, refs: Iterable[str]) -> "CompatibilityDescriptor":
        """Return a copy that serves exactly the given model generation digests."""
        return replace(self, model_generations=frozenset(refs))

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, value-free wire representation."""
        return {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "contract_generation": self.contract_generation,
            "min_peer_generation": self.min_peer_generation,
            "durable_schemas": {
                name: schema.to_dict() for name, schema in self.durable_schemas.items()
            },
            "capabilities": sorted(self.capabilities),
            "model_generations": sorted(self.model_generations),
        }

    def digest(self) -> str:
        """Return a canonical SHA-256 digest of :meth:`to_dict`."""
        return _canonical_digest(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Any) -> "CompatibilityDescriptor":
        """Parse a peer descriptor strictly.

        Args:
            payload: Decoded JSON object published by a peer.

        Returns:
            The parsed descriptor.

        Raises:
            CompatibilityError: With code ``peer_descriptor_unsupported`` when the
                payload is not a descriptor this build understands.
        """
        code = "peer_descriptor_unsupported"
        if not isinstance(payload, Mapping) or set(payload) != _DESCRIPTOR_KEYS:
            raise CompatibilityError(code)
        if payload["schema_version"] != COMPATIBILITY_SCHEMA_VERSION:
            raise CompatibilityError(code)
        schemas = payload["durable_schemas"]
        capabilities = payload["capabilities"]
        models = payload["model_generations"]
        if (
            not isinstance(schemas, Mapping)
            or not isinstance(capabilities, list)
            or not isinstance(models, list)
        ):
            raise CompatibilityError(code)
        try:
            return cls(
                contract_generation=payload["contract_generation"],
                min_peer_generation=payload["min_peer_generation"],
                durable_schemas={
                    _require_token(name, code=code): SchemaRange.from_dict(value)
                    for name, value in schemas.items()
                },
                capabilities=frozenset(capabilities),
                model_generations=frozenset(models),
            )
        except (CompatibilityError, TypeError):
            raise CompatibilityError(code) from None


#: Durable schema declarations of this build. The Journey ranges mirror the
#: append-only migration chains in ``openmed.structured.store``; both stores
#: migrate older schemas forward on open, so they operate only on their latest
#: version. A drift test fails when a migration is added without updating this
#: table, which is where the durable-schema registry plugs in.
DURABLE_SCHEMAS: Mapping[str, SchemaRange] = MappingProxyType(
    {
        JOB_STORE_COMPONENT: SchemaRange(
            read_min=1, read_max=1, write=JOB_STORE_SCHEMA_VERSION
        ),
        JOURNEY_LOCAL_COMPONENT: SchemaRange(read_min=3, read_max=3, write=3),
        JOURNEY_POSTGRES_COMPONENT: SchemaRange(read_min=3, read_max=3, write=3),
    }
)

CURRENT_CONTRACT = CompatibilityDescriptor(
    contract_generation=CONTRACT_GENERATION,
    min_peer_generation=MIN_PEER_GENERATION,
    durable_schemas=DURABLE_SCHEMAS,
    capabilities=WORKER_CAPABILITIES,
)

#: Every generation inside the supported window, oldest first. The last entry
#: is always :data:`CURRENT_CONTRACT`; generation N keeps N-1 here until the
#: window closes, and the published matrix is derived from this tuple.
CONTRACT_HISTORY: tuple[CompatibilityDescriptor, ...] = (CURRENT_CONTRACT,)


def local_descriptor(
    runtime: Any = None,
    *,
    contract: CompatibilityDescriptor = CURRENT_CONTRACT,
) -> CompatibilityDescriptor:
    """Return this node's descriptor, including its served model generations.

    Args:
        runtime: Optional service runtime exposing ``served_model_ids()``.
        contract: Static contract to extend; defaults to :data:`CURRENT_CONTRACT`.

    Returns:
        Descriptor whose model generations are digests of the served model ids.
    """
    served = getattr(runtime, "served_model_ids", None)
    if not callable(served):
        return contract
    return contract.with_model_generations(
        model_generation_ref(model_id) for model_id in served()
    )


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    """Typed, value-free outcome of one compatibility check.

    Attributes:
        compatible: Whether the checked peer, work item or rollback may proceed.
        code: Controlled reason code from :data:`REASON_CODES`.
        component: Durable component that caused a refusal, when applicable.
    """

    compatible: bool
    code: str
    component: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free representation."""
        payload: dict[str, Any] = {"compatible": self.compatible, "code": self.code}
        if self.component is not None:
            payload["component"] = self.component
        return payload


def _allow(code: str) -> CompatibilityDecision:
    return CompatibilityDecision(True, code)


def _deny(code: str, component: str | None = None) -> CompatibilityDecision:
    return CompatibilityDecision(False, code, component)


def generations_compatible(
    generation: int,
    min_peer_generation: int,
    other_generation: int,
    other_min_peer_generation: int,
) -> bool:
    """Apply the N/N-1 rule: the newer build must declare the older one.

    Args:
        generation: First build's contract generation.
        min_peer_generation: First build's oldest supported peer generation.
        other_generation: Second build's contract generation.
        other_min_peer_generation: Second build's oldest supported peer.

    Returns:
        Whether the two generations may share state and traffic.
    """
    if generation >= other_generation:
        return min_peer_generation <= other_generation
    return other_min_peer_generation <= generation


def negotiate_peer(
    local: CompatibilityDescriptor,
    peer: CompatibilityDescriptor | Mapping[str, Any],
) -> CompatibilityDecision:
    """Decide whether a peer build may run beside ``local``.

    Args:
        local: This node's descriptor.
        peer: The peer's descriptor, or its decoded wire representation.

    Returns:
        ``peer_compatible`` or a typed refusal. Unparseable peers are refused.
    """
    if not isinstance(peer, CompatibilityDescriptor):
        try:
            peer = CompatibilityDescriptor.from_dict(peer)
        except CompatibilityError:
            return _deny("peer_descriptor_unsupported")
    if not generations_compatible(
        local.contract_generation,
        local.min_peer_generation,
        peer.contract_generation,
        peer.min_peer_generation,
    ):
        return _deny("peer_generation_outside_window")
    shared = sorted(local.durable_schemas.keys() & peer.durable_schemas.keys())
    for component in shared:
        mine = local.durable_schemas[component]
        theirs = peer.durable_schemas[component]
        if not (mine.reads(theirs.write) and theirs.reads(mine.write)):
            return _deny("peer_schema_outside_window", component)
    return _allow("peer_compatible")


def parse_peer_contract(raw: Any) -> tuple[int, int]:
    """Parse a ``<generation>/<min_peer_generation>`` peer contract token.

    Raises:
        CompatibilityError: With code ``peer_contract_malformed``.
    """
    code = "peer_contract_malformed"
    if type(raw) is not str:
        raise CompatibilityError(code)
    match = _PEER_CONTRACT_RE.fullmatch(raw.strip())
    if match is None:
        raise CompatibilityError(code)
    generation, minimum = int(match.group(1)), int(match.group(2))
    _require_window(generation, minimum, code=code)
    return generation, minimum


def peer_contract_decision(
    local: CompatibilityDescriptor,
    values: Sequence[Any],
) -> CompatibilityDecision:
    """Decide on a peer that announced itself with one peer contract token.

    The token carries generation and window only; the durable schema check is
    implied because every generation publishes exactly one contract and
    :data:`CONTRACT_HISTORY` is verified to be self-consistent. Peers that need
    the per-component decision exchange full descriptors with
    :func:`negotiate_peer`.

    Args:
        local: This node's descriptor.
        values: Every value the peer sent for :data:`PEER_CONTRACT_HEADER`.

    Returns:
        ``peer_compatible`` or a typed refusal; repeated values are malformed.
    """
    if len(values) != 1:
        return _deny("peer_contract_malformed")
    try:
        generation, minimum = parse_peer_contract(values[0])
    except CompatibilityError:
        return _deny("peer_contract_malformed")
    if not generations_compatible(
        local.contract_generation, local.min_peer_generation, generation, minimum
    ):
        return _deny("peer_generation_outside_window")
    return _allow("peer_compatible")


@dataclass(frozen=True, slots=True)
class WorkEnvelope:
    """Contract stamp persisted with one unit of shared work.

    Accepted work keeps the contract and model generation it was admitted
    under until completion or a durable checkpoint, so every claimant can
    refuse it before mutation.

    Attributes:
        contract_generation: Generation of the build that admitted the work.
        min_peer_generation: That build's oldest supported peer generation.
        schema_versions: Durable schema version per component the work uses.
        required_capabilities: Capabilities a claimant must provide.
        model_generation: Digest of the pinned model generation, if any.
    """

    contract_generation: int
    min_peer_generation: int
    schema_versions: Mapping[str, int]
    required_capabilities: frozenset[str]
    model_generation: str | None = None

    def __post_init__(self) -> None:
        code = "work_envelope_invalid"
        generation = _require_version(
            self.contract_generation, maximum=MAX_GENERATION, code=code
        )
        minimum = _require_version(
            self.min_peer_generation, maximum=MAX_GENERATION, code=code
        )
        _require_window(generation, minimum, code=code)
        if not isinstance(self.schema_versions, Mapping):
            raise CompatibilityError(code)
        versions = {
            _require_token(name, code=code): _require_version(
                self.schema_versions[name], maximum=MAX_SCHEMA_VERSION, code=code
            )
            for name in sorted(self.schema_versions)
        }
        object.__setattr__(self, "schema_versions", MappingProxyType(versions))
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(
                _require_token(item, code=code) for item in self.required_capabilities
            ),
        )
        if self.model_generation is not None:
            _require_digest(self.model_generation, code=code)

    @classmethod
    def for_work(
        cls,
        descriptor: CompatibilityDescriptor,
        *,
        components: Iterable[str],
        required_capabilities: Iterable[str],
        model_generation: str | None = None,
    ) -> "WorkEnvelope":
        """Stamp new work with the admitting build's contract.

        Args:
            descriptor: Descriptor of the admitting build.
            components: Durable components the work will be persisted in.
            required_capabilities: Capabilities a claimant must provide.
            model_generation: Optional pinned model generation digest.

        Returns:
            Envelope recording the write versions of ``components``.
        """
        return cls(
            contract_generation=descriptor.contract_generation,
            min_peer_generation=descriptor.min_peer_generation,
            schema_versions={
                name: descriptor.durable_schemas[name].write for name in components
            },
            required_capabilities=frozenset(required_capabilities),
            model_generation=model_generation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic, value-free persisted representation."""
        return {
            "contract_generation": self.contract_generation,
            "min_peer_generation": self.min_peer_generation,
            "schema_versions": dict(self.schema_versions),
            "required_capabilities": sorted(self.required_capabilities),
            "model_generation": self.model_generation,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "WorkEnvelope":
        """Parse a persisted envelope strictly.

        Raises:
            CompatibilityError: With code ``work_envelope_invalid``.
        """
        code = "work_envelope_invalid"
        if not isinstance(payload, Mapping) or set(payload) != _ENVELOPE_KEYS:
            raise CompatibilityError(code)
        capabilities = payload["required_capabilities"]
        if not isinstance(capabilities, list):
            raise CompatibilityError(code)
        try:
            return cls(
                contract_generation=payload["contract_generation"],
                min_peer_generation=payload["min_peer_generation"],
                schema_versions=payload["schema_versions"],
                required_capabilities=frozenset(capabilities),
                model_generation=payload["model_generation"],
            )
        except TypeError:
            raise CompatibilityError(code) from None


def check_work(
    local: CompatibilityDescriptor,
    envelope: WorkEnvelope | Mapping[str, Any] | None,
) -> CompatibilityDecision:
    """Decide whether ``local`` may claim one unit of persisted work.

    Call this before taking a lease, marking a job running, replaying a
    checkpoint or delivering a webhook. A refusal leaves the work untouched
    for a compatible claimant.

    Args:
        local: This node's descriptor.
        envelope: The work's persisted envelope; ``None`` means it has none.

    Returns:
        ``work_compatible`` or a typed refusal.
    """
    if not isinstance(envelope, WorkEnvelope):
        try:
            envelope = WorkEnvelope.from_dict(envelope)
        except CompatibilityError:
            return _deny("work_envelope_invalid")
    if not generations_compatible(
        local.contract_generation,
        local.min_peer_generation,
        envelope.contract_generation,
        envelope.min_peer_generation,
    ):
        return _deny("work_generation_outside_window")
    for component, version in envelope.schema_versions.items():
        schema = local.durable_schemas.get(component)
        if schema is None:
            return _deny("work_component_unknown", component)
        if not schema.reads(version):
            return _deny("work_schema_outside_window", component)
    if not envelope.required_capabilities <= local.capabilities:
        return _deny("capability_missing")
    if (
        envelope.model_generation is not None
        and envelope.model_generation not in local.model_generations
    ):
        return _deny("model_generation_unavailable")
    return _allow("work_compatible")


class CompatibilityState(str, Enum):
    """Readiness state of one durable component against this build."""

    COMPATIBLE = "compatible"
    ABSENT = "absent"
    PENDING = "pending"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


_USABLE_STATES = frozenset({CompatibilityState.COMPATIBLE, CompatibilityState.ABSENT})


@dataclass(frozen=True, slots=True)
class SchemaObservation:
    """What a probe saw in one persisted store, without any stored values.

    Attributes:
        known: Whether the probe could determine the persisted state.
        version: Persisted schema version, or ``None`` when nothing is stored.
    """

    known: bool
    version: int | None = None

    @classmethod
    def absent(cls) -> "SchemaObservation":
        """Return the observation for a store with no persisted state yet."""
        return cls(known=True)

    @classmethod
    def unknown(cls) -> "SchemaObservation":
        """Return the blocking observation for unreadable or damaged state."""
        return cls(known=False)

    @classmethod
    def at(cls, version: Any) -> "SchemaObservation":
        """Return an observation for a persisted version, or unknown if invalid."""
        if not _is_version(version, maximum=MAX_SCHEMA_VERSION):
            return cls.unknown()
        return cls(known=True, version=int(version))


@dataclass(frozen=True, slots=True)
class ComponentCompatibility:
    """Value-free compatibility of one durable component.

    Attributes:
        state: Readiness state.
        code: Controlled reason code from :data:`STATE_REASON_CODES`.
        schema_version: Persisted version, when one was observed.
    """

    state: CompatibilityState
    code: str
    schema_version: int | None = None

    @property
    def usable(self) -> bool:
        """Whether this build may read and mutate the component."""
        return self.state in _USABLE_STATES

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free representation."""
        payload: dict[str, Any] = {"state": self.state.value, "code": self.code}
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        return payload


def evaluate_component(
    schema: SchemaRange | None,
    observation: SchemaObservation,
) -> ComponentCompatibility:
    """Compare one persisted schema observation with a declared range.

    Args:
        schema: This build's range for the component, or ``None`` if undeclared.
        observation: Probe result for the component.

    Returns:
        The component's state; unknown and undeclared state is blocking.
    """
    version = observation.version
    if not observation.known:
        return ComponentCompatibility(CompatibilityState.UNKNOWN, "schema_unknown")
    if version is None:
        return ComponentCompatibility(CompatibilityState.ABSENT, "schema_absent")
    if schema is None:
        return ComponentCompatibility(CompatibilityState.UNKNOWN, "schema_unknown")
    if version < schema.read_min:
        return ComponentCompatibility(
            CompatibilityState.PENDING, "schema_migration_required", version
        )
    if version > schema.read_max:
        return ComponentCompatibility(
            CompatibilityState.INCOMPATIBLE, "schema_ahead_of_window", version
        )
    return ComponentCompatibility(
        CompatibilityState.COMPATIBLE, "schema_compatible", version
    )


@dataclass(frozen=True, slots=True)
class PersistedStateReport:
    """Readiness view of every probed durable component.

    Attributes:
        components: Compatibility per component name.
    """

    components: Mapping[str, ComponentCompatibility]

    @property
    def ready(self) -> bool:
        """Whether every probed component is compatible or empty."""
        return all(item.usable for item in self.components.values())

    def blocking(self) -> dict[str, ComponentCompatibility]:
        """Return the components that block readiness."""
        return {name: item for name, item in self.components.items() if not item.usable}

    def to_dict(self) -> dict[str, Any]:
        """Return the value-free representation."""
        return {
            "state": "compatible" if self.ready else "blocked",
            "components": {
                name: self.components[name].to_dict()
                for name in sorted(self.components)
            },
        }


def evaluate_persisted_state(
    local: CompatibilityDescriptor,
    observations: Mapping[str, SchemaObservation],
) -> PersistedStateReport:
    """Gate readiness on the persisted schema of each shared store.

    Args:
        local: This node's descriptor.
        observations: Probe result per durable component in use.

    Returns:
        Report whose ``ready`` property is false for any blocking component.
    """
    return PersistedStateReport(
        components=MappingProxyType(
            {
                name: evaluate_component(
                    local.durable_schemas.get(name), observations[name]
                )
                for name in sorted(observations)
            }
        )
    )


def rollback_decision(
    target: CompatibilityDescriptor,
    *,
    running: Iterable[CompatibilityDescriptor],
    observations: Mapping[str, SchemaObservation],
) -> CompatibilityDecision:
    """Decide whether returning to ``target`` stays inside the declared window.

    A rollback is permitted only while ``target`` is compatible with every
    build still running and reads every persisted schema without migration.
    Otherwise restore a verified backup taken before the newer schema was
    written instead of starting the older build.

    Args:
        target: Descriptor of the build to return to.
        running: Descriptors of builds that will coexist during the rollback.
        observations: Probe result per durable component in use.

    Returns:
        ``rollback_within_window`` or the first blocking reason code.
    """
    for peer in running:
        decision = negotiate_peer(target, peer)
        if not decision.compatible:
            return decision
    report = evaluate_persisted_state(target, observations)
    for name, component in report.components.items():
        if not component.usable:
            return _deny(component.code, name)
    return _allow("rollback_within_window")


def compatibility_matrix(
    contracts: Sequence[CompatibilityDescriptor] = CONTRACT_HISTORY,
) -> dict[str, Any]:
    """Return the machine-readable pairwise compatibility matrix.

    Args:
        contracts: Generations inside the declared window, oldest first.

    Returns:
        Value-free matrix with one entry per unordered generation pair.
    """
    ordered = sorted(contracts, key=lambda item: item.contract_generation)
    combinations = []
    for index, older in enumerate(ordered):
        for newer in ordered[index:]:
            decision = negotiate_peer(newer, older)
            combinations.append(
                {
                    "older": older.contract_generation,
                    "newer": newer.contract_generation,
                    **decision.to_dict(),
                }
            )
    return {
        "schema_version": COMPATIBILITY_MATRIX_SCHEMA_VERSION,
        "generations": [item.contract_generation for item in ordered],
        "descriptor_digests": {
            str(item.contract_generation): item.digest() for item in ordered
        },
        "combinations": combinations,
    }


def validate_contract_history(
    contracts: Sequence[CompatibilityDescriptor] = CONTRACT_HISTORY,
) -> None:
    """Check that a declared window is consecutive and self-compatible.

    Raises:
        CompatibilityError: With code ``contract_history_invalid``.
    """
    code = "contract_history_invalid"
    generations = [item.contract_generation for item in contracts]
    if not generations or generations != list(
        range(generations[0], generations[0] + len(generations))
    ):
        raise CompatibilityError(code)
    if len(contracts) > 2:
        raise CompatibilityError(code)
    for older, newer in zip(contracts, contracts[1:]):
        if not negotiate_peer(newer, older).compatible:
            raise CompatibilityError(code)


ObservationProbe = Callable[[], SchemaObservation]


class CompatibilityGate:
    """Cached readiness gate over the persisted state of shared stores.

    Probes are re-run at most once per ``refresh_interval_seconds`` so that a
    rolling migration by a newer peer is noticed without probing on every
    request. A probe that raises or returns anything else is unknown, which
    blocks readiness.
    """

    def __init__(
        self,
        descriptor: CompatibilityDescriptor,
        probes: Mapping[str, ObservationProbe],
        *,
        refresh_interval_seconds: float = DEFAULT_REFRESH_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a gate.

        Args:
            descriptor: This node's descriptor.
            probes: One observation probe per durable component in use.
            refresh_interval_seconds: Minimum seconds between probe runs.
            clock: Monotonic clock in seconds.
        """
        self.descriptor = descriptor
        self._probes = dict(probes)
        for name in self._probes:
            _require_token(name, code="contract_invalid")
        self._interval = max(0.0, float(refresh_interval_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._report: PersistedStateReport | None = None
        self._checked_at = 0.0

    def report(self, *, force: bool = False) -> PersistedStateReport:
        """Return the current report, re-probing when it is stale."""
        with self._lock:
            now = self._clock()
            if (
                force
                or self._report is None
                or now - self._checked_at >= self._interval
            ):
                self._report = evaluate_persisted_state(
                    self.descriptor, self._observe()
                )
                self._checked_at = now
            return self._report

    def snapshot(self, *, force: bool = False) -> dict[str, Any]:
        """Return the descriptor, its digest and the current report."""
        report = self.report(force=force)
        return {
            **self.descriptor.to_dict(),
            "peer_contract": self.descriptor.peer_contract,
            "descriptor_digest": self.descriptor.digest(),
            **report.to_dict(),
        }

    def _observe(self) -> dict[str, SchemaObservation]:
        observations: dict[str, SchemaObservation] = {}
        for name, probe in self._probes.items():
            try:
                observation = probe()
            except Exception:
                observation = SchemaObservation.unknown()
            if not isinstance(observation, SchemaObservation):
                observation = SchemaObservation.unknown()
            observations[name] = observation
        return observations


def job_store_document_observation(document: Any) -> SchemaObservation:
    """Return the schema observation for a decoded job metadata document.

    The unversioned layout written before this contract existed is version 1.
    """
    if not isinstance(document, Mapping):
        return SchemaObservation.unknown()
    if "schema_version" in document:
        return SchemaObservation.at(document["schema_version"])
    if isinstance(document.get("jobs"), Mapping):
        return SchemaObservation.at(1)
    return SchemaObservation.unknown()


def job_store_observation(path: Path) -> SchemaObservation:
    """Probe the async job metadata store without exposing its contents.

    Args:
        path: Store file location. It never enters the observation.

    Returns:
        Absent when the file does not exist, unknown when it cannot be read.
    """
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return SchemaObservation.absent()
    except (OSError, UnicodeDecodeError):
        return SchemaObservation.unknown()
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError):
        return SchemaObservation.unknown()
    return job_store_document_observation(document)


def journey_store_observation(store: Any) -> SchemaObservation:
    """Probe an opened Journey store through its ``schema_version`` property.

    Args:
        store: ``SQLiteJourneyStore``, ``PostgresJourneyStore`` or a fake.

    Returns:
        The persisted version, absent for an empty chain, or unknown on error.
    """
    try:
        version = store.schema_version
    except Exception:
        return SchemaObservation.unknown()
    if version == 0:
        return SchemaObservation.absent()
    return SchemaObservation.at(version)


class RolloutEvidence:
    """Value-free, machine-readable record of one rollout rehearsal.

    Every field is a controlled token, an integer or a descriptor digest, so
    the evidence can be attached to a release without review for PHI.
    """

    def __init__(self, scenario: str) -> None:
        """Create an empty record for a named scenario token."""
        self.scenario = _require_token(scenario, code="evidence_invalid")
        self._steps: list[dict[str, Any]] = []
        self._digests: dict[str, str] = {}
        self._counts: dict[str, int] = {}

    def register(self, descriptor: CompatibilityDescriptor) -> None:
        """Record the digest of a participating generation."""
        self._digests[str(descriptor.contract_generation)] = descriptor.digest()

    def record(
        self,
        step: str,
        decision: CompatibilityDecision,
        *,
        generations: Sequence[int],
    ) -> None:
        """Record one decision taken during the rehearsal.

        Args:
            step: Controlled step token.
            decision: The decision taken.
            generations: Generations involved, in the order they acted.
        """
        _require_token(step, code="evidence_invalid")
        if decision.code not in REASON_CODES:
            raise CompatibilityError("evidence_invalid")
        entry: dict[str, Any] = {
            "step": step,
            "generations": [
                _require_version(item, maximum=MAX_GENERATION, code="evidence_invalid")
                for item in generations
            ],
            **decision.to_dict(),
        }
        self._steps.append(entry)
        self.count(decision.code)

    def count(self, name: str, amount: int = 1) -> None:
        """Increment a controlled counter."""
        _require_token(name, code="evidence_invalid")
        self._counts[name] = self._counts.get(name, 0) + int(amount)

    @property
    def steps(self) -> tuple[dict[str, Any], ...]:
        """Recorded steps, oldest first."""
        return tuple(dict(step) for step in self._steps)

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic evidence document."""
        return {
            "schema_version": ROLLOUT_EVIDENCE_SCHEMA_VERSION,
            "scenario": self.scenario,
            "descriptor_digests": dict(sorted(self._digests.items())),
            "steps": [dict(step) for step in self._steps],
            "counts": dict(sorted(self._counts.items())),
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Print this build's descriptor or compatibility matrix as JSON.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m openmed.service.compatibility",
        description="Print the value-free service compatibility contract.",
    )
    parser.add_argument(
        "view",
        choices=("descriptor", "matrix"),
        nargs="?",
        default="descriptor",
    )
    args = parser.parse_args(argv)
    if args.view == "matrix":
        payload = compatibility_matrix()
    else:
        payload = {
            **CURRENT_CONTRACT.to_dict(),
            "peer_contract": CURRENT_CONTRACT.peer_contract,
            "descriptor_digest": CURRENT_CONTRACT.digest(),
        }
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


__all__ = [
    "COMPATIBILITY_MATRIX_SCHEMA_VERSION",
    "COMPATIBILITY_SCHEMA_VERSION",
    "CONTRACT_GENERATION",
    "CONTRACT_HISTORY",
    "CURRENT_CONTRACT",
    "DURABLE_SCHEMAS",
    "JOB_DEIDENTIFY_CAPABILITY",
    "JOB_STORE_COMPONENT",
    "JOB_STORE_SCHEMA_VERSION",
    "JOB_WEBHOOK_CAPABILITY",
    "JOURNEY_LOCAL_COMPONENT",
    "JOURNEY_POSTGRES_COMPONENT",
    "MIN_PEER_GENERATION",
    "PEER_CONTRACT_HEADER",
    "PEER_CONTRACT_METADATA_KEY",
    "REASON_CODES",
    "ROLLOUT_EVIDENCE_SCHEMA_VERSION",
    "STREAM_NDJSON_CAPABILITY",
    "WORKER_CAPABILITIES",
    "CompatibilityDecision",
    "CompatibilityDescriptor",
    "CompatibilityError",
    "CompatibilityGate",
    "PersistedStateReport",
    "CompatibilityState",
    "ComponentCompatibility",
    "IncompatibleStateError",
    "RolloutEvidence",
    "SchemaObservation",
    "SchemaRange",
    "WorkEnvelope",
    "check_work",
    "compatibility_matrix",
    "evaluate_component",
    "evaluate_persisted_state",
    "generations_compatible",
    "job_store_document_observation",
    "job_store_observation",
    "journey_store_observation",
    "local_descriptor",
    "model_generation_ref",
    "negotiate_peer",
    "parse_peer_contract",
    "peer_contract_decision",
    "rollback_decision",
    "validate_contract_history",
]


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
