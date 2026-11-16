"""Replica-local service state inventory and the multi-replica readiness gate.

Several REST service components keep state inside one process or on one
node's disk: the async job store and work queue, the FHIR Bulk Data and SMART
Backend job managers, the rate and concurrency limiters, the authentication
failure limiter, and the request coalescer. One replica is the supported
topology for that state. Behind several replicas, a status read can reach a
replica that never saw the job, and every limit is enforced once per replica.

This module records each component's scope and multi-replica safety, parses the
operator's replica policy, and evaluates a value-free report. In ``multi``
mode, readiness stays false with ``replica_unsafe_state`` while an enabled
component that is not multi-replica safe is neither disabled nor explicitly
accepted. Reports contain controlled component categories, scopes, safety
classes, and states only. They never include paths, hosts, job or client
identifiers, request content, or configuration values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

SERVICE_REPLICA_MODE_ENV_VAR = "OPENMED_SERVICE_REPLICA_MODE"
SERVICE_REPLICA_ACCEPT_ENV_VAR = "OPENMED_SERVICE_REPLICA_ACCEPT_LOCAL_STATE"
SERVICE_REPLICA_DISABLE_ENV_VAR = "OPENMED_SERVICE_REPLICA_DISABLE_LOCAL_STATE"
REPLICA_STATE_REPORT_SCHEMA = "openmed.service.replica_state.v1"
REPLICA_UNSAFE_STATE = "replica_unsafe_state"
COMPONENT_DISABLED = "component_disabled"

ASYNC_JOBS = "async_jobs"
FHIR_BULK_JOBS = "fhir_bulk_jobs"
SMART_BACKEND_JOBS = "smart_backend_jobs"
RATE_LIMITER = "rate_limiter"
CONCURRENCY_LIMITER = "concurrency_limiter"
AUTH_FAILURE_LIMITER = "auth_failure_limiter"
REQUEST_COALESCER = "request_coalescer"


class ReplicaMode(str, Enum):
    """Replica topology declared by the deployment."""

    SINGLE = "single"
    MULTI = "multi"


class StateScope(str, Enum):
    """Where a component's state lives."""

    PROCESS = "process"
    NODE = "node"
    SHARED = "shared"


class ReplicaSafety(str, Enum):
    """How a component behaves when several replicas serve the same traffic.

    ``safe`` components stay correct at any replica count. ``per_replica``
    components stay correct inside each replica, but every replica enforces its
    own copy, so effective limits multiply by the replica count. ``unsafe``
    components hold state that later requests must read back, so a request
    routed to another replica cannot see it.
    """

    SAFE = "safe"
    PER_REPLICA = "per_replica"
    UNSAFE = "unsafe"


class ComponentState(str, Enum):
    """Evaluated state of one component under the current replica policy."""

    DISABLED = "disabled"
    SAFE = "safe"
    ACCEPTED = "accepted"
    UNSAFE = "unsafe"


@dataclass(frozen=True)
class ReplicaLocalComponent:
    """Static inventory entry for one stateful service component.

    Attributes:
        category: Stable component category used in reports and policy lists.
        scope: Where the component keeps its state.
        safety: Multi-replica safety class.
        disableable: Whether the replica policy can turn the component off.
            Only route-backed job components qualify; limiters and the
            coalescer are switched off through their own settings.
    """

    category: str
    scope: StateScope
    safety: ReplicaSafety
    disableable: bool = False


REPLICA_LOCAL_COMPONENTS: tuple[ReplicaLocalComponent, ...] = (
    ReplicaLocalComponent(
        ASYNC_JOBS, StateScope.NODE, ReplicaSafety.UNSAFE, disableable=True
    ),
    ReplicaLocalComponent(
        FHIR_BULK_JOBS, StateScope.PROCESS, ReplicaSafety.UNSAFE, disableable=True
    ),
    ReplicaLocalComponent(
        SMART_BACKEND_JOBS, StateScope.PROCESS, ReplicaSafety.UNSAFE, disableable=True
    ),
    ReplicaLocalComponent(RATE_LIMITER, StateScope.PROCESS, ReplicaSafety.PER_REPLICA),
    ReplicaLocalComponent(
        CONCURRENCY_LIMITER, StateScope.PROCESS, ReplicaSafety.PER_REPLICA
    ),
    ReplicaLocalComponent(
        AUTH_FAILURE_LIMITER, StateScope.PROCESS, ReplicaSafety.PER_REPLICA
    ),
    ReplicaLocalComponent(REQUEST_COALESCER, StateScope.PROCESS, ReplicaSafety.SAFE),
)
_COMPONENTS_BY_CATEGORY = {
    component.category: component for component in REPLICA_LOCAL_COMPONENTS
}
_KNOWN_CATEGORIES = tuple(component.category for component in REPLICA_LOCAL_COMPONENTS)
_DISABLEABLE_CATEGORIES = tuple(
    component.category
    for component in REPLICA_LOCAL_COMPONENTS
    if component.disableable
)


class ReplicaComponentUnavailableError(RuntimeError):
    """Raised when a route's replica-local component may not serve requests.

    Attributes:
        category: Component category that refused the request.
        code: ``component_disabled`` or ``replica_unsafe_state``.
    """

    def __init__(self, category: str, code: str) -> None:
        self.category = category
        self.code = code
        super().__init__("A replica-local service component is unavailable")


@dataclass(frozen=True)
class ReplicaComponentStatus:
    """Evaluated, value-free status of one inventory component.

    Attributes:
        category: Stable component category.
        scope: Where the component keeps its state.
        safety: Multi-replica safety class.
        state: Evaluated state under the active policy.
    """

    category: str
    scope: StateScope
    safety: ReplicaSafety
    state: ComponentState

    def to_dict(self) -> dict[str, str]:
        """Return the component status as controlled strings only."""
        return {
            "category": self.category,
            "scope": self.scope.value,
            "safety": self.safety.value,
            "state": self.state.value,
        }


@dataclass(frozen=True)
class ReplicaStateReport:
    """Value-free inventory report for replica-local service state.

    Attributes:
        mode: Replica mode the report was evaluated under.
        components: One status per inventory component, in inventory order.
    """

    mode: ReplicaMode
    components: tuple[ReplicaComponentStatus, ...]

    @property
    def unsafe_categories(self) -> tuple[str, ...]:
        """Return categories that keep readiness false, sorted."""
        return tuple(
            sorted(
                status.category
                for status in self.components
                if status.state is ComponentState.UNSAFE
            )
        )

    @property
    def ready(self) -> bool:
        """Return whether replica-local state allows readiness."""
        return not self.unsafe_categories

    @property
    def reason_code(self) -> Optional[str]:
        """Return ``replica_unsafe_state`` when readiness is blocked."""
        return None if self.ready else REPLICA_UNSAFE_STATE

    def readiness_details(self) -> dict[str, Any]:
        """Return the category-only details used by the readiness envelope."""
        return {
            "replica_mode": self.mode.value,
            "components": list(self.unsafe_categories),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-ready report."""
        return {
            "schema": REPLICA_STATE_REPORT_SCHEMA,
            "replica_mode": self.mode.value,
            "ready": self.ready,
            "reason_code": self.reason_code,
            "components": [status.to_dict() for status in self.components],
        }


@dataclass(frozen=True)
class ReplicaStatePolicy:
    """Operator replica policy for replica-local service state.

    Attributes:
        mode: Declared replica topology. ``single`` is the default.
        accepted: Categories whose per-replica semantics the operator accepts.
            An accepted component stays enabled and does not block readiness.
        disabled: Route-backed job categories the operator turned off. Their
            routes return ``503 component_disabled`` in every mode.

    Raises:
        ValueError: If a list names an unknown category, disables a component
            that cannot be disabled here, or both accepts and disables one.
    """

    mode: ReplicaMode = ReplicaMode.SINGLE
    accepted: frozenset[str] = frozenset()
    disabled: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ReplicaMode):
            raise ValueError("Replica mode must be 'single' or 'multi'")
        _require_known(self.accepted, env_var=SERVICE_REPLICA_ACCEPT_ENV_VAR)
        _require_known(self.disabled, env_var=SERVICE_REPLICA_DISABLE_ENV_VAR)
        if not self.disabled <= frozenset(_DISABLEABLE_CATEGORIES):
            raise ValueError(
                f"{SERVICE_REPLICA_DISABLE_ENV_VAR} may list only "
                f"{', '.join(_DISABLEABLE_CATEGORIES)}"
            )
        if self.accepted & self.disabled:
            raise ValueError(
                "A replica-local component cannot be both accepted and disabled"
            )

    @classmethod
    def from_env(
        cls, environ: Optional[Mapping[str, str]] = None
    ) -> "ReplicaStatePolicy":
        """Read the replica policy from service environment variables.

        Args:
            environ: Optional mapping used instead of ``os.environ``.

        Returns:
            Parsed policy. Unset values mean ``single`` with nothing accepted
            or disabled.

        Raises:
            ValueError: If a value is not a supported mode or category list.
        """
        source = os.environ if environ is None else environ
        return cls(
            mode=parse_replica_mode(source.get(SERVICE_REPLICA_MODE_ENV_VAR)),
            accepted=parse_component_categories(
                source.get(SERVICE_REPLICA_ACCEPT_ENV_VAR),
                env_var=SERVICE_REPLICA_ACCEPT_ENV_VAR,
            ),
            disabled=parse_component_categories(
                source.get(SERVICE_REPLICA_DISABLE_ENV_VAR),
                env_var=SERVICE_REPLICA_DISABLE_ENV_VAR,
            ),
        )

    def component_state(self, category: str, *, enabled: bool = True) -> ComponentState:
        """Evaluate one component under this policy.

        Args:
            category: Inventory category.
            enabled: Whether the component is active in this process.

        Returns:
            The component's evaluated state.

        Raises:
            ValueError: If ``category`` is not in the inventory.
        """
        component = _COMPONENTS_BY_CATEGORY.get(category)
        if component is None:
            raise ValueError("Unknown replica-local component category")
        if category in self.disabled or not enabled:
            return ComponentState.DISABLED
        if self.mode is ReplicaMode.SINGLE or component.safety is ReplicaSafety.SAFE:
            return ComponentState.SAFE
        if category in self.accepted:
            return ComponentState.ACCEPTED
        return ComponentState.UNSAFE

    def refusal_code(self, category: str) -> Optional[str]:
        """Return the controlled refusal code for a route-backed component.

        Args:
            category: Inventory category guarding the route.

        Returns:
            ``component_disabled``, ``replica_unsafe_state``, or ``None`` when
            the component may serve the request.
        """
        state = self.component_state(category)
        if state is ComponentState.DISABLED:
            return COMPONENT_DISABLED
        if state is ComponentState.UNSAFE:
            return REPLICA_UNSAFE_STATE
        return None

    def require_available(self, category: str) -> None:
        """Raise when a route-backed component may not serve requests.

        Args:
            category: Inventory category guarding the route.

        Raises:
            ReplicaComponentUnavailableError: If the component is disabled or
                unsafe under this policy.
        """
        code = self.refusal_code(category)
        if code is not None:
            raise ReplicaComponentUnavailableError(category, code)

    def evaluate(self, enabled: Mapping[str, bool]) -> ReplicaStateReport:
        """Build the value-free report for the current process.

        Args:
            enabled: Whether each category is active. A missing category is
                treated as enabled so an incomplete caller fails closed.

        Returns:
            Report with one status per inventory component.
        """
        return ReplicaStateReport(
            mode=self.mode,
            components=tuple(
                ReplicaComponentStatus(
                    category=component.category,
                    scope=component.scope,
                    safety=component.safety,
                    state=self.component_state(
                        component.category,
                        enabled=bool(enabled.get(component.category, True)),
                    ),
                )
                for component in REPLICA_LOCAL_COMPONENTS
            ),
        )


def parse_replica_mode(raw_value: Optional[str]) -> ReplicaMode:
    """Parse ``OPENMED_SERVICE_REPLICA_MODE``.

    Args:
        raw_value: Raw environment value. Unset or blank means ``single``.

    Returns:
        Parsed replica mode.

    Raises:
        ValueError: If the value is neither ``single`` nor ``multi``.
    """
    if raw_value is None or not raw_value.strip():
        return ReplicaMode.SINGLE
    normalized = raw_value.strip().lower()
    for mode in ReplicaMode:
        if mode.value == normalized:
            return mode
    raise ValueError(f"{SERVICE_REPLICA_MODE_ENV_VAR} must be 'single' or 'multi'")


def parse_component_categories(
    raw_value: Optional[str], *, env_var: str
) -> frozenset[str]:
    """Parse a comma-separated replica-local component category list.

    Args:
        raw_value: Raw environment value. Unset or blank means no categories.
        env_var: Environment variable name used in error messages.

    Returns:
        Normalized category set.

    Raises:
        ValueError: If any entry is not an inventory category. The message
            names the allowed categories, never the rejected value.
    """
    if raw_value is None or not raw_value.strip():
        return frozenset()
    categories = frozenset(
        item.strip().lower() for item in raw_value.split(",") if item.strip()
    )
    _require_known(categories, env_var=env_var)
    return categories


def _require_known(categories: frozenset[str], *, env_var: str) -> None:
    if not categories <= frozenset(_KNOWN_CATEGORIES):
        raise ValueError(f"{env_var} may list only {', '.join(_KNOWN_CATEGORIES)}")


__all__ = [
    "ASYNC_JOBS",
    "AUTH_FAILURE_LIMITER",
    "COMPONENT_DISABLED",
    "CONCURRENCY_LIMITER",
    "FHIR_BULK_JOBS",
    "RATE_LIMITER",
    "REPLICA_LOCAL_COMPONENTS",
    "REPLICA_STATE_REPORT_SCHEMA",
    "REPLICA_UNSAFE_STATE",
    "REQUEST_COALESCER",
    "SERVICE_REPLICA_ACCEPT_ENV_VAR",
    "SERVICE_REPLICA_DISABLE_ENV_VAR",
    "SERVICE_REPLICA_MODE_ENV_VAR",
    "SMART_BACKEND_JOBS",
    "ComponentState",
    "ReplicaComponentStatus",
    "ReplicaComponentUnavailableError",
    "ReplicaLocalComponent",
    "ReplicaMode",
    "ReplicaSafety",
    "ReplicaStatePolicy",
    "ReplicaStateReport",
    "StateScope",
    "parse_component_categories",
    "parse_replica_mode",
]
