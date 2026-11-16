"""Test-only service fault-injection harness.

The harness defines a versioned, seeded fault-plan schema
(``openmed/core/schemas/json/service_fault_plan.schema.json``), injection seams
at documented adapter boundaries (disk, clock, process, network, cancellation
and corrupted state), component adapters for the service code that exists
today and a deterministic matrix runner that emits a value-free manifest
(``service_fault_manifest.schema.json``).

It is never packaged into the ``openmed`` wheel, refuses to load into a process
explicitly configured with the production profile, and arms only under
``OPENMED_PROFILE=test``. Later features register a component in
:mod:`tests.faults.scenarios` and add cases to ``service_fault_plan.json``; they
plug into the same seams, runner and manifest.
"""

from __future__ import annotations

from .guard import refuse_production_import

refuse_production_import()

from .guard import FaultInjectionForbiddenError, require_arm_permitted  # noqa: E402
from .plan import FaultCase, FaultPlan, FaultPlanError, ResolvedCase  # noqa: E402
from .seams import (  # noqa: E402
    FaultClock,
    FaultInjector,
    FaultSession,
    FaultTransport,
    SimulatedProcessKill,
    corrupt_file,
    disk_faults,
    sqlite_commit_faults,
)

__all__ = [
    "FaultCase",
    "FaultClock",
    "FaultInjectionForbiddenError",
    "FaultInjector",
    "FaultPlan",
    "FaultPlanError",
    "FaultSession",
    "FaultTransport",
    "ResolvedCase",
    "SimulatedProcessKill",
    "corrupt_file",
    "disk_faults",
    "require_arm_permitted",
    "sqlite_commit_faults",
]
