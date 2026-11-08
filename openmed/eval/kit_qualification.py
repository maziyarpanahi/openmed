"""Real-device qualification collector for the frozen OpenMedKit support matrix.

The support matrix (:mod:`openmed.eval.kit_support_matrix`) only lets a cell be
``supported`` when it names an evidence record. This module produces the
``device-run`` kind of that evidence from qualification runs:

* A **qualification plan** pins, for chosen matrix cells, the model bundles,
  fixtures and policies to run by SHA-256 digest, plus the gates each run must
  meet. :func:`qualification_targets` expands it into one target per
  bundle x runtime/provider x platform x precision x fixture x policy.
* An **evidence record** is what one run on one device, simulator or emulator
  reports. :func:`validate_evidence_record` checks its value-free schema.
* :func:`collect_qualification` matches records to targets and reports missing
  targets, stale digests and every failed gate separately. Simulator, emulator
  and fake-runtime evidence is kept apart and can never qualify a target.
* :func:`propose_matrix_updates` turns qualified cells into a reviewable
  matrix change and :func:`apply_matrix_proposal` applies it to a copy. Nothing
  here edits the committed matrix.

The gates reuse the EdgeHealthBench rules in
:mod:`openmed.eval.suites.edge_health`: a ``not_run`` result never counts as a
pass, any critical or document PHI leak fails, missing required resource
readings fail as ``missing_resource_evidence``, energy proxies are labelled as
proxies, and accelerator tiers need separate allocator evidence.

Plans, records, reports and proposals hold controlled codes, coarse device
classes, version numbers, counts, timings and digests. Device models, serials,
user data, file paths and free text are refused.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from openmed.eval.kit_support_matrix import (
    ABIS,
    ANY,
    ARTIFACTS,
    CELL_PLATFORMS,
    DEVICE_CLASSES,
    EXPERIMENTAL,
    PENDING_EVIDENCE,
    PRECISIONS,
    PROVIDERS,
    SUPPORTED,
    identifier_shaped,
    tier_ram_ceiling_mb,
    validate_kit_support_matrix,
    version_key,
)
from openmed.eval.suites.edge_health import (
    DEVICE_MEMORY_TIERS,
    DeviceTier,
)
from openmed.eval.suites.edge_health import (
    FAILURE_CODES as EDGE_HEALTH_FAILURE_CODES,
)
from openmed.eval.tiers import TIERS

PLAN_SCHEMA: Final = "openmed.kit_qualification.plan.v1"
EVIDENCE_SCHEMA: Final = "openmed.kit_qualification.evidence.v1"
REPORT_SCHEMA: Final = "openmed.kit_qualification.report.v1"
PROPOSAL_SCHEMA: Final = "openmed.kit_qualification.proposal.v1"
EDGE_HEALTH_SCHEMA: Final = "openmed.edge_health.v1"

PHYSICAL_DEVICE: Final = "physical-device"
SIMULATOR: Final = "simulator"
EMULATOR: Final = "emulator"
FAKE_RUNTIME: Final = "fake-runtime"
EXECUTIONS: Final = (PHYSICAL_DEVICE, SIMULATOR, EMULATOR, FAKE_RUNTIME)

DEVICE_EVIDENCE: Final = "device"
SIMULATOR_EVIDENCE: Final = "simulator"
FAKE_RUNTIME_EVIDENCE: Final = "fake_runtime"
EVIDENCE_CLASSES: Final = (DEVICE_EVIDENCE, SIMULATOR_EVIDENCE, FAKE_RUNTIME_EVIDENCE)
_EVIDENCE_CLASS: Final = MappingProxyType(
    {
        PHYSICAL_DEVICE: DEVICE_EVIDENCE,
        SIMULATOR: SIMULATOR_EVIDENCE,
        EMULATOR: SIMULATOR_EVIDENCE,
        FAKE_RUNTIME: FAKE_RUNTIME_EVIDENCE,
    }
)

# Virtual matrix device classes and the physical class each one stands in for.
VIRTUAL_DEVICE_CLASSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "android-emulator": "android-device",
        "ios-simulator": "iphone-ipad",
        "visionos-simulator": "apple-vision",
        "watchos-simulator": "apple-watch",
    }
)
_VIRTUAL_EXECUTION: Final = MappingProxyType(
    {
        "android-emulator": EMULATOR,
        "ios-simulator": SIMULATOR,
        "visionos-simulator": SIMULATOR,
        "watchos-simulator": SIMULATOR,
    }
)
_ANDROID_CLASSES: Final = frozenset({"android-device", "android-emulator"})
_APPLE_MOBILE_CLASSES: Final = frozenset({"apple-vision", "apple-watch", "iphone-ipad"})
_BROWSER_PROVIDERS: Final = frozenset({"wasm-basic", "wasm-simd-threads", "webgpu"})
# Providers that can silently fall back to CPU and must report that they did not.
ACCELERATED_PROVIDERS: Final = frozenset({"coreml", "metal", "nnapi", "qnn", "webgpu"})

RUN_COMPLETED: Final = "completed"
RUN_FAILED: Final = "failed"
RUN_NOT_RUN: Final = "not_run"
RUN_STATUSES: Final = (RUN_COMPLETED, RUN_FAILED, RUN_NOT_RUN)
RUN_FAILURE_CODES: Final = EDGE_HEALTH_FAILURE_CODES | frozenset(
    {
        "app_terminated",
        "out_of_memory",
        "provider_unavailable",
        "runtime_fallback",
        "thermal_shutdown",
    }
)
EDGE_HEALTH_STATUSES: Final = ("failed", "not_run", "passed")

MEASURED: Final = "measured"
PROXY: Final = "proxy"
UNAVAILABLE: Final = "unavailable"
METRICS: Final = (
    "direct_identifier_recall",
    "leakage",
    "cold_latency_ms",
    "warm_latency_ms",
    "sustained_latency_ms",
    "cancel_latency_ms",
    "memory_peak_bytes",
    "device_memory_peak_bytes",
    "thermal",
    "energy",
    "provider_fallback",
)
ENERGY_PROXIES: Final = (
    "battery_charge_delta",
    "battery_percent_delta",
    "process_cpu_seconds",
)
# Same scopes as :class:`openmed.eval.suites.edge_health.ResourceSample`.
RSS_SCOPES: Final = ("process_current", "process_lifetime_highwater")
DEVICE_MEMORY_SCOPES: Final = ("allocator_current", "allocator_peak")
MAX_THERMAL_LEVEL: Final = 3

GATES: Final = (
    "run",
    "edge_health",
    "recall",
    "leakage",
    "cold_latency",
    "warm_latency",
    "sustained_duration",
    "sustained_latency",
    "cancel_latency",
    "memory",
    "device_memory",
    "thermal",
    "energy",
    "provider_binding",
)
GATE_PASSED: Final = "passed"
GATE_FAILED: Final = "failed"
GATE_MISSING: Final = "missing"
GATE_NOT_APPLICABLE: Final = "not_applicable"
GATE_INFORMATIONAL: Final = "informational"
_GATE_OK: Final = frozenset({GATE_PASSED, GATE_NOT_APPLICABLE, GATE_INFORMATIONAL})

REQUIRED_GATES: Final = (
    "minimum_direct_identifier_recall",
    "maximum_cold_latency_ms",
    "maximum_warm_p95_latency_ms",
    "maximum_sustained_p95_latency_ms",
    "minimum_sustained_seconds",
    "maximum_cancel_latency_ms",
    "maximum_thermal_level",
    "require_thermal",
    "require_measured_energy",
)
OPTIONAL_GATES: Final = ("maximum_memory_peak_mb", "maximum_device_memory_mb")
# Same floor as ``DeviceProfile.trials``: fewer warm runs are not a measurement.
MIN_WARM_RUNS: Final = 3

QUALIFIED: Final = "qualified"
FAILED: Final = "failed"
STALE: Final = "stale"
MISSING: Final = "missing"
NOT_RUN: Final = "not_run"
SIMULATOR_ONLY: Final = "simulator_only"
FAKE_RUNTIME_ONLY: Final = "fake_runtime_only"
INCOMPLETE: Final = "incomplete"
UNPLANNED: Final = "unplanned"
# Worst first. A cell reports the worst status among its targets.
TARGET_STATUSES: Final = (
    FAILED,
    STALE,
    MISSING,
    NOT_RUN,
    SIMULATOR_ONLY,
    FAKE_RUNTIME_ONLY,
    QUALIFIED,
)
CELL_STATUSES: Final = (*TARGET_STATUSES[:-1], INCOMPLETE, UNPLANNED, QUALIFIED)

RECORD_PASSED: Final = "passed"
RECORD_FAILED: Final = "failed"
RECORD_NOT_RUN: Final = "not_run"
RECORD_STALE: Final = "stale"
RECORD_OUTCOMES: Final = (RECORD_PASSED, RECORD_FAILED, RECORD_NOT_RUN, RECORD_STALE)

MAX_EVIDENCE_RECORDS: Final = 10_000
DEVICE_RUN_COVERS: Final = ("runtime-model", "memory-budget", "device-qualification")
DEVICE_RUN_PREFIX: Final = "device-run-"

_HEX: Final = re.compile(r"[0-9a-f]{64}\Z")
_ID: Final = re.compile(r"[a-z0-9][a-z0-9-]{0,63}\Z")
_VERSION: Final = re.compile(r"\d{1,6}(?:\.\d{1,6}){0,3}\Z")
_DATE: Final = re.compile(r"\d{4}-\d{2}-\d{2}\Z")
_MAX_COUNT: Final = 10_000_000
_MAX_LATENCY_MS: Final = 86_400_000.0
_MAX_DURATION_S: Final = 604_800.0
_MAX_BYTES: Final = 2**50
_MAX_ENERGY: Final = 1_000_000_000.0
_MIB: Final = 1024 * 1024
_CELL_IDENTITY_FIELDS: Final = (
    "id",
    "surface",
    "platform",
    "device_class",
    "abi",
    "runtime",
    "provider",
    "artifact",
    "families",
    "precisions",
    "tier",
    "condition",
)
_NOTICE: Final = (
    "Device classes, versions, counts, timings and digests only. Physical-device "
    "evidence alone can qualify a target; simulator, emulator and fake-runtime "
    "evidence is reported separately. Proposals are never applied automatically "
    "and carry no clinical qualification claim."
)


class KitQualificationError(ValueError):
    """Raised when a plan, record set, report or proposal cannot be used.

    Attributes:
        problems: Value-free problem codes. They name fields and controlled
            matrix or plan identifiers only, never submitted values.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("; ".join(self.problems) or "kit_qualification_invalid")


# --- digests and small checks ----------------------------------------------


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _try_hash(value: Any) -> str | None:
    try:
        return _hash(value)
    except (TypeError, ValueError):
        return None


def _source_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _is_int(value: Any, low: int, high: int) -> bool:
    return type(value) is int and low <= value <= high


def _is_number(value: Any, low: float, high: float, *, strict_low: bool) -> bool:
    if type(value) not in (int, float) or not math.isfinite(value):
        return False
    above = value > low if strict_low else value >= low
    return above and value <= high


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _HEX.fullmatch(value) is not None


def _is_version(value: Any) -> bool:
    return isinstance(value, str) and _VERSION.fullmatch(value) is not None


def _is_date(value: Any) -> bool:
    if not isinstance(value, str) or _DATE.fullmatch(value) is None:
        return False
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _is_id(value: Any) -> bool:
    """Controlled identifier that cannot carry a device or user identifier."""
    return (
        isinstance(value, str)
        and _ID.fullmatch(value) is not None
        and not identifier_shaped(value)
        and not identifier_shaped(value.upper())
    )


def _in(value: Any, options: frozenset[str] | Sequence[str]) -> bool:
    """Membership test that tolerates unhashable decoded JSON values."""
    return isinstance(value, str) and value in options


def _is_code_list(value: Any, allowed: frozenset[str]) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item in allowed for item in value)
        and len(set(value)) == len(value)
    )


def _fields(
    value: Mapping[str, Any], required: frozenset[str], optional: frozenset[str]
) -> bool:
    keys = set(value)
    return required <= keys and not keys - required - optional


# --- matrix access -----------------------------------------------------------


def matrix_digest(matrix: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of a matrix document's canonical JSON.

    Args:
        matrix: A decoded support matrix.

    Returns:
        Lower-case hex digest.
    """
    return _hash(matrix)


def _require_valid_matrix(matrix: Mapping[str, Any]) -> None:
    problems = validate_kit_support_matrix(matrix)
    if problems:
        raise KitQualificationError(tuple(f"matrix:{code}" for code in problems))


def _pins(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {pin["id"]: pin for pin in matrix["runtime_pins"]}


def cell_digest(cell: Mapping[str, Any], matrix: Mapping[str, Any]) -> str:
    """Return the digest of the fields that define what a cell runs.

    The digest covers the platform range, device class, ABI, runtime pin and
    its version, provider, artifact, families, precisions, tier and condition.
    Status, evidence, related issues and notes are excluded, so promoting a
    cell does not make an existing plan stale, but changing a runtime pin does.

    Args:
        cell: A matrix cell.
        matrix: The matrix that holds the cell, used to resolve its runtime pin.

    Returns:
        Lower-case hex digest.
    """
    identity: dict[str, Any] = {
        field: cell.get(field) for field in _CELL_IDENTITY_FIELDS
    }
    pin = _pins(matrix).get(cell.get("runtime", ""))
    identity["runtime_pin"] = (
        None
        if pin is None
        else {key: pin.get(key) for key in ("runtime", "kind", "version", "pinning")}
    )
    return _hash(identity)


def edge_health_tier(device_class: str, provider: str) -> DeviceTier:
    """Return the EdgeHealthBench tier a run of this class and provider uses.

    Android and iOS-family hardware map to the mobile device tiers. Apple
    simulators execute on the host Mac, so they map to the host tiers; the
    Android emulator still runs Android ONNX Runtime.

    Args:
        device_class: A matrix device class.
        provider: A matrix execution provider.

    Returns:
        The declared execution tier.
    """
    if provider in _BROWSER_PROVIDERS:
        return DeviceTier.BROWSER
    if device_class in _ANDROID_CLASSES:
        android = {
            "cpu": DeviceTier.ANDROID_ORT_CPU,
            "nnapi": DeviceTier.ANDROID_ORT_NNAPI,
            "qnn": DeviceTier.ANDROID_ORT_QNN,
        }
        return android.get(provider, DeviceTier.ACCELERATOR)
    if device_class in _APPLE_MOBILE_CLASSES:
        apple = {
            "cpu": DeviceTier.IOS_DEVICE_CPU,
            "metal": DeviceTier.IOS_DEVICE_MLX,
            "coreml": DeviceTier.IOS_DEVICE_CORE_ML,
        }
        return apple.get(provider, DeviceTier.ACCELERATOR)
    host = {
        "cpu": DeviceTier.CPU,
        "metal": DeviceTier.MLX,
        "coreml": DeviceTier.CORE_ML,
    }
    return host.get(provider, DeviceTier.ACCELERATOR)


def qualification_cells(matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate every supported and experimental inference cell.

    Key-store and OCR cells have no model runtime and are not qualification
    targets; :func:`collect_qualification` lists them as out of scope.

    Args:
        matrix: A valid support matrix.

    Returns:
        One row per cell in matrix order with its identity axes, tier RAM
        ceiling, EdgeHealthBench tier and current :func:`cell_digest`.

    Raises:
        KitQualificationError: If the matrix fails validation.
    """
    _require_valid_matrix(matrix)
    pins = _pins(matrix)
    rows: list[dict[str, Any]] = []
    for cell in matrix["cells"]:
        pin = pins[cell["runtime"]]
        if pin["kind"] != "inference":
            continue
        tier = edge_health_tier(cell["device_class"], cell["provider"])
        rows.append(
            {
                "cell": cell["id"],
                "surface": cell["surface"],
                "matrix_status": cell["status"],
                "platform": dict(cell["platform"]),
                "device_class": cell["device_class"],
                "abi": cell["abi"],
                "runtime": cell["runtime"],
                "runtime_version": pin["version"],
                "provider": cell["provider"],
                "artifact": cell["artifact"],
                "families": list(cell["families"]),
                "precisions": list(cell["precisions"]),
                "tier": cell["tier"],
                "ram_ceiling_mb": tier_ram_ceiling_mb(cell["tier"], TIERS),
                "edge_health_tier": tier.value,
                "device_memory_required": tier in DEVICE_MEMORY_TIERS,
                "virtual_device_class": cell["device_class"] in VIRTUAL_DEVICE_CLASSES,
                "cell_digest": cell_digest(cell, matrix),
            }
        )
    return rows


# --- qualification plan ------------------------------------------------------

_PLAN_REQUIRED: Final = frozenset(
    {"schema", "release", "bundles", "fixtures", "policies", "gates", "cells"}
)
_PLAN_OPTIONAL: Final = frozenset({"harness_digest"})
_BUNDLE_FIELDS: Final = frozenset({"id", "family", "artifact", "precision", "digest"})
_ARTIFACT_FIELDS: Final = frozenset({"id", "digest"})
_PLAN_CELL_REQUIRED: Final = frozenset({"cell", "cell_digest", "bundles"})
_PLAN_CELL_OPTIONAL: Final = frozenset(
    {"fixtures", "policies", "platform_versions", "gates"}
)


def _latency_gate(value: Any) -> bool:
    return _is_number(value, 0, _MAX_LATENCY_MS, strict_low=True)


_GATE_CHECKS: Final[Mapping[str, Callable[[Any], bool]]] = MappingProxyType(
    {
        "minimum_direct_identifier_recall": lambda value: _is_number(
            value, 0, 1, strict_low=False
        ),
        "maximum_cold_latency_ms": _latency_gate,
        "maximum_warm_p95_latency_ms": _latency_gate,
        "maximum_sustained_p95_latency_ms": _latency_gate,
        "minimum_sustained_seconds": lambda value: _is_number(
            value, 0, _MAX_DURATION_S, strict_low=True
        ),
        "maximum_cancel_latency_ms": _latency_gate,
        "maximum_thermal_level": lambda value: _is_int(value, 0, MAX_THERMAL_LEVEL),
        "require_thermal": lambda value: type(value) is bool,
        "require_measured_energy": lambda value: type(value) is bool,
        "maximum_memory_peak_mb": lambda value: _is_int(value, 1, _MAX_BYTES // _MIB),
        "maximum_device_memory_mb": lambda value: _is_int(value, 1, _MAX_BYTES // _MIB),
    }
)


def plan_skeleton(matrix: Mapping[str, Any]) -> dict[str, Any]:
    """Return an unfilled plan listing every qualification cell.

    The skeleton pins each cell's current digest. It deliberately fails
    :func:`validate_qualification_plan` until bundles, fixtures, policies and
    gate values are declared; no thresholds are invented.

    Args:
        matrix: A valid support matrix.

    Returns:
        A plan document with empty artifact lists and ``None`` gate values.
    """
    return {
        "schema": PLAN_SCHEMA,
        "release": matrix["release"],
        "bundles": [],
        "fixtures": [],
        "policies": [],
        "gates": {name: None for name in REQUIRED_GATES},
        "cells": [
            {"cell": row["cell"], "cell_digest": row["cell_digest"], "bundles": []}
            for row in qualification_cells(matrix)
        ],
    }


def _check_gates(
    problems: set[str], gates: Any, label: str, *, complete: bool
) -> dict[str, Any]:
    if not isinstance(gates, Mapping):
        problems.add(f"plan_gates_invalid:{label}")
        return {}
    known = set(REQUIRED_GATES) | set(OPTIONAL_GATES)
    if set(gates) - known:
        problems.add(f"plan_gates_field_unknown:{label}")
    valid: dict[str, Any] = {}
    for name in sorted(set(gates) & known):
        if _GATE_CHECKS[name](gates[name]):
            valid[name] = gates[name]
        else:
            problems.add(f"plan_gate_invalid:{label}:{name}")
    if complete:
        for name in REQUIRED_GATES:
            if name not in gates:
                problems.add(f"plan_gate_missing:{label}:{name}")
    return valid


def _check_artifacts(
    problems: set[str], rows: Any, kind: str, fields: frozenset[str]
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list) or not rows:
        problems.add(f"plan_{kind}_empty")
        return {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or not _fields(row, fields, frozenset()):
            problems.add(f"plan_{kind}_invalid:{index}")
            continue
        if not _is_id(row["id"]):
            problems.add(f"plan_{kind}_id_invalid:{index}")
            continue
        if row["id"] in by_id:
            problems.add(f"plan_{kind}_duplicate:{row['id']}")
            continue
        if not _is_digest(row["digest"]):
            problems.add(f"plan_{kind}_digest_invalid:{row['id']}")
            continue
        by_id[row["id"]] = row
    return by_id


def _id_subset(
    problems: set[str], values: Any, known: Mapping[str, Any], code: str
) -> list[str] | None:
    if (
        not isinstance(values, list)
        or not values
        or len(set(map(str, values))) != len(values)
        or not all(isinstance(value, str) and value in known for value in values)
    ):
        problems.add(code)
        return None
    return list(values)


def effective_gates(plan: Mapping[str, Any], cell: str) -> dict[str, Any]:
    """Return the plan's gates for one cell with its overrides applied.

    Args:
        plan: A valid qualification plan.
        cell: A planned matrix cell identifier.

    Returns:
        The merged gate values for that cell.

    Raises:
        KitQualificationError: If the cell is not planned.
    """
    for entry in plan["cells"]:
        if entry["cell"] == cell:
            return {**plan["gates"], **entry.get("gates", {})}
    raise KitQualificationError(("plan_cell_not_planned",))


def validate_qualification_plan(
    plan: Mapping[str, Any], matrix: Mapping[str, Any]
) -> tuple[str, ...]:
    """Validate a qualification plan against a support matrix.

    A plan must name existing inference cells, bundles that fit each cell's
    artifact, family and precision, digests for every bundle, fixture and
    policy, and every required gate. A memory budget may tighten the cell's
    tier RAM ceiling but never exceed it, and tiers that keep model memory in
    a separate allocator must declare ``maximum_device_memory_mb``. A stale
    ``cell_digest`` is not a validation problem: the report marks those
    targets ``stale``.

    Args:
        plan: A decoded plan document.
        matrix: A valid support matrix.

    Returns:
        Sorted, de-duplicated problem codes. Empty means valid. Unknown cell
        or artifact identifiers are reported by position, never echoed.

    Raises:
        KitQualificationError: If the matrix itself fails validation.
    """
    if not isinstance(plan, Mapping):
        return ("plan_not_object",)
    cells = {row["cell"]: row for row in qualification_cells(matrix)}
    matrix_cells = {cell["id"] for cell in matrix["cells"]}
    problems: set[str] = set()
    if not _fields(plan, _PLAN_REQUIRED, _PLAN_OPTIONAL):
        problems.add("plan_fields_invalid")
    if plan.get("schema") != PLAN_SCHEMA:
        problems.add("plan_schema_unsupported")
    if plan.get("release") != matrix["release"]:
        problems.add("plan_release_mismatch")
    if "harness_digest" in plan and not _is_digest(plan["harness_digest"]):
        problems.add("plan_harness_digest_invalid")

    bundles = _check_artifacts(problems, plan.get("bundles"), "bundle", _BUNDLE_FIELDS)
    families = frozenset(row["id"] for row in matrix["families"])
    for bundle_id, bundle in bundles.items():
        if not (
            _in(bundle["family"], families)
            and _in(bundle["artifact"], ARTIFACTS)
            and _in(bundle["precision"], PRECISIONS)
        ):
            problems.add(f"plan_bundle_invalid:{bundle_id}")
    fixtures = _check_artifacts(
        problems, plan.get("fixtures"), "fixture", _ARTIFACT_FIELDS
    )
    policies = _check_artifacts(
        problems, plan.get("policies"), "policy", _ARTIFACT_FIELDS
    )
    top_gates = _check_gates(problems, plan.get("gates"), "plan", complete=True)

    entries = plan.get("cells")
    if not isinstance(entries, list) or not entries:
        problems.add("plan_cells_empty")
        entries = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or not _fields(
            entry, _PLAN_CELL_REQUIRED, _PLAN_CELL_OPTIONAL
        ):
            problems.add(f"plan_cell_invalid:{index}")
            continue
        cell_id = entry["cell"]
        if not isinstance(cell_id, str) or cell_id not in matrix_cells:
            problems.add(f"plan_cell_unknown:{index}")
            continue
        if cell_id not in cells:
            problems.add(f"plan_cell_out_of_scope:{cell_id}")
            continue
        if cell_id in seen:
            problems.add(f"plan_cell_duplicate:{cell_id}")
            continue
        seen.add(cell_id)
        row = cells[cell_id]
        if not _is_digest(entry["cell_digest"]):
            problems.add(f"plan_cell_digest_invalid:{cell_id}")
        chosen = _id_subset(
            problems, entry["bundles"], bundles, f"plan_cell_bundles_invalid:{cell_id}"
        )
        for bundle_id in chosen or []:
            bundle = bundles[bundle_id]
            if (
                bundle["artifact"] != row["artifact"]
                or bundle["family"] not in row["families"]
                or bundle["precision"] not in row["precisions"]
            ):
                problems.add(f"plan_bundle_mismatch:{cell_id}:{bundle_id}")
        if "fixtures" in entry:
            _id_subset(
                problems,
                entry["fixtures"],
                fixtures,
                f"plan_cell_fixtures_invalid:{cell_id}",
            )
        if "policies" in entry:
            _id_subset(
                problems,
                entry["policies"],
                policies,
                f"plan_cell_policies_invalid:{cell_id}",
            )
        if "platform_versions" in entry:
            _check_platform_versions(problems, entry["platform_versions"], row)
        overrides = _check_gates(
            problems, entry.get("gates", {}), cell_id, complete=False
        )
        gates = {**top_gates, **overrides}
        ceiling = row["ram_ceiling_mb"]
        memory = gates.get("maximum_memory_peak_mb")
        if ceiling is not None and memory is not None and memory > ceiling:
            problems.add(f"plan_gate_looser_than_tier:{cell_id}:maximum_memory_peak_mb")
        if row["device_memory_required"] and "maximum_device_memory_mb" not in gates:
            problems.add(f"plan_gate_missing:{cell_id}:maximum_device_memory_mb")
    return tuple(sorted(problems))


def _check_platform_versions(
    problems: set[str], versions: Any, row: Mapping[str, Any]
) -> None:
    code = f"plan_cell_platform_versions_invalid:{row['cell']}"
    if (
        not isinstance(versions, list)
        or not versions
        or not all(_is_version(value) for value in versions)
        or len({version_key(value) for value in versions}) != len(versions)
    ):
        problems.add(code)
        return
    if not all(_version_in_range(value, row["platform"]) for value in versions):
        problems.add(code)


def _version_in_range(version: str, platform: Mapping[str, Any]) -> bool:
    key = version_key(version)
    low, high = platform.get("min"), platform.get("max")
    if low is not None and key < version_key(low):
        return False
    if high is not None and key[: len(version_key(high))] > version_key(high):
        return False
    return True


def _require_valid_plan(plan: Mapping[str, Any], matrix: Mapping[str, Any]) -> None:
    problems = validate_qualification_plan(plan, matrix)
    if problems:
        raise KitQualificationError(problems)


def qualification_targets(
    plan: Mapping[str, Any], matrix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Expand a plan into bundle x fixture x policy x platform-version targets.

    A plan cell without ``fixtures`` or ``policies`` uses every plan fixture
    or policy. Without ``platform_versions`` a target accepts any platform
    version inside the cell's range and reports ``"*"``.

    Args:
        plan: A valid qualification plan.
        matrix: The support matrix the plan was written for.

    Returns:
        Targets in plan cell order, then bundle, fixture, policy and version
        order. Each names its cell, bundle, family, precision, fixture, policy
        and platform version.

    Raises:
        KitQualificationError: If the plan or matrix fails validation.
    """
    _require_valid_plan(plan, matrix)
    bundles = {row["id"]: row for row in plan["bundles"]}
    fixture_ids = [row["id"] for row in plan["fixtures"]]
    policy_ids = [row["id"] for row in plan["policies"]]
    targets: list[dict[str, Any]] = []
    for entry in plan["cells"]:
        for bundle_id in entry["bundles"]:
            bundle = bundles[bundle_id]
            for fixture_id in entry.get("fixtures", fixture_ids):
                for policy_id in entry.get("policies", policy_ids):
                    for version in entry.get("platform_versions", [ANY]):
                        targets.append(
                            {
                                "cell": entry["cell"],
                                "bundle": bundle_id,
                                "family": bundle["family"],
                                "precision": bundle["precision"],
                                "fixture": fixture_id,
                                "policy": policy_id,
                                "platform_version": version,
                            }
                        )
    return targets


def _target_key(target: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        target["cell"],
        target["bundle"],
        target["fixture"],
        target["policy"],
        target["platform_version"],
    )


# --- evidence records --------------------------------------------------------

_RECORD_REQUIRED: Final = frozenset(
    {
        "schema",
        "recorded_on",
        "cell",
        "execution",
        "device_class",
        "abi",
        "platform",
        "runtime",
        "provider",
        "bundle",
        "precision",
        "fixture",
        "policy",
        "harness_digest",
        "run",
        "metrics",
    }
)
_RECORD_OPTIONAL: Final = frozenset({"edge_health"})
_EDGE_HEALTH_FIELDS: Final = frozenset(
    {"schema", "tier", "status", "failures", "report_digest"}
)
_DEVICE_TIER_VALUES: Final = frozenset(tier.value for tier in DeviceTier)


def _count(value: Any) -> bool:
    return _is_int(value, 0, _MAX_COUNT)


def _positive_count(value: Any) -> bool:
    return _is_int(value, 1, _MAX_COUNT)


def _latency(value: Any) -> bool:
    return _is_number(value, 0, _MAX_LATENCY_MS, strict_low=True)


def _latency_or_zero(value: Any) -> bool:
    return _is_number(value, 0, _MAX_LATENCY_MS, strict_low=False)


def _member(options: Sequence[str]) -> Callable[[Any], bool]:
    return lambda value: isinstance(value, str) and value in options


_Spec = Mapping[str, Callable[[Any], bool]]
_METRIC_SPECS: Final[Mapping[str, Mapping[str, _Spec]]] = MappingProxyType(
    {
        "direct_identifier_recall": {
            MEASURED: {"detected": _count, "expected": _positive_count}
        },
        "leakage": {
            MEASURED: {"assessed": _count, "critical": _count, "any_phi": _count}
        },
        "cold_latency_ms": {MEASURED: {"value": _latency}},
        "warm_latency_ms": {
            MEASURED: {"n": _positive_count, "p50": _latency, "p95": _latency}
        },
        "sustained_latency_ms": {
            MEASURED: {
                "n": _positive_count,
                "p95": _latency,
                "duration_s": lambda value: _is_number(
                    value, 0, _MAX_DURATION_S, strict_low=True
                ),
            }
        },
        "cancel_latency_ms": {
            MEASURED: {"n": _positive_count, "p95": _latency_or_zero}
        },
        "memory_peak_bytes": {
            MEASURED: {
                "value": lambda value: _is_int(value, 1, _MAX_BYTES),
                "scope": _member(RSS_SCOPES),
            }
        },
        "device_memory_peak_bytes": {
            MEASURED: {
                "value": lambda value: _is_int(value, 0, _MAX_BYTES),
                "scope": _member(DEVICE_MEMORY_SCOPES),
            }
        },
        "thermal": {
            MEASURED: {"max_level": lambda value: _is_int(value, 0, MAX_THERMAL_LEVEL)}
        },
        "energy": {
            MEASURED: {
                "joules": lambda value: _is_number(
                    value, 0, _MAX_ENERGY, strict_low=False
                )
            },
            PROXY: {
                "proxy": _member(ENERGY_PROXIES),
                "value": lambda value: _is_number(
                    value, 0, _MAX_ENERGY, strict_low=False
                ),
            },
        },
        "provider_fallback": {
            MEASURED: {"observed": lambda value: type(value) is bool}
        },
    }
)


def _metric_invariant(name: str, metric: Mapping[str, Any]) -> bool:
    if name == "direct_identifier_recall":
        return bool(metric["detected"] <= metric["expected"])
    if name == "leakage":
        return bool(metric["critical"] <= metric["any_phi"] <= metric["assessed"])
    if name == "warm_latency_ms":
        return bool(metric["p50"] <= metric["p95"])
    return True


def _metric_valid(name: str, metric: Any) -> bool:
    if not isinstance(metric, Mapping):
        return False
    status = metric.get("status")
    if status == UNAVAILABLE:
        return set(metric) == {"status"}
    spec = _METRIC_SPECS[name].get(status) if isinstance(status, str) else None
    if spec is None or set(metric) != {"status", *spec}:
        return False
    return all(check(metric[field]) for field, check in spec.items()) and (
        _metric_invariant(name, metric)
    )


def _pair_valid(value: Any, checks: Mapping[str, Callable[[Any], bool]]) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == set(checks)
        and all(check(value[field]) for field, check in checks.items())
    )


def _platform_name(value: Any) -> bool:
    return _in(value, CELL_PLATFORMS)


def validate_evidence_record(record: Any) -> tuple[str, ...]:
    """Validate one device-run evidence record without a plan or matrix.

    The schema is closed: an unknown field, such as a device name or serial,
    fails as ``evidence_field_unknown`` and its value is never repeated.
    Every metric is present with ``status`` ``measured``, ``proxy`` (energy
    only) or ``unavailable``, so a missing reading is explicit rather than
    omitted.

    Args:
        record: A decoded evidence record.

    Returns:
        Sorted, de-duplicated problem codes. Empty means structurally valid.
    """
    if not isinstance(record, Mapping):
        return ("evidence_not_object",)
    problems: set[str] = set()
    keys = set(record)
    if keys - _RECORD_REQUIRED - _RECORD_OPTIONAL:
        problems.add("evidence_field_unknown")
    for name in sorted(_RECORD_REQUIRED - keys):
        problems.add(f"evidence_field_missing:{name}")
    if record.get("schema") != EVIDENCE_SCHEMA:
        problems.add("evidence_schema_unsupported")
    if "recorded_on" in record and not _is_date(record["recorded_on"]):
        problems.add("evidence_recorded_on_invalid")
    if "cell" in record and not _is_id(record["cell"]):
        problems.add("evidence_cell_invalid")
    execution = record.get("execution")
    device_class = record.get("device_class")
    if "execution" in record and not _in(execution, EXECUTIONS):
        problems.add("evidence_execution_invalid")
    if "device_class" in record and not _in(device_class, DEVICE_CLASSES):
        problems.add("evidence_device_class_invalid")
    elif _in(execution, EXECUTIONS) and _in(device_class, DEVICE_CLASSES):
        virtual = _VIRTUAL_EXECUTION.get(str(device_class))
        if execution in (PHYSICAL_DEVICE, SIMULATOR, EMULATOR) and (
            (execution == PHYSICAL_DEVICE and virtual is not None)
            or (execution != PHYSICAL_DEVICE and virtual != execution)
        ):
            problems.add("evidence_execution_device_class_mismatch")
    if "abi" in record and not _in(record["abi"], ABIS):
        problems.add("evidence_abi_invalid")
    if "platform" in record and not _pair_valid(
        record["platform"], {"name": _platform_name, "version": _is_version}
    ):
        problems.add("evidence_platform_invalid")
    if "runtime" in record and not _pair_valid(
        record["runtime"], {"pin": _is_id, "version": _is_version}
    ):
        problems.add("evidence_runtime_invalid")
    if "provider" in record and not _in(record["provider"], PROVIDERS):
        problems.add("evidence_provider_invalid")
    for name in ("bundle", "fixture", "policy"):
        if name in record and not _pair_valid(
            record[name], {"id": _is_id, "digest": _is_digest}
        ):
            problems.add(f"evidence_{name}_invalid")
    if "precision" in record and not _in(record["precision"], PRECISIONS):
        problems.add("evidence_precision_invalid")
    if "harness_digest" in record and not _is_digest(record["harness_digest"]):
        problems.add("evidence_harness_digest_invalid")
    if "run" in record and not _run_valid(record["run"]):
        problems.add("evidence_run_invalid")
    if "edge_health" in record and not _edge_health_valid(record["edge_health"]):
        problems.add("evidence_edge_health_invalid")
    metrics = record.get("metrics")
    if "metrics" in record:
        if not isinstance(metrics, Mapping):
            problems.add("evidence_metrics_invalid")
        else:
            if set(metrics) - set(METRICS):
                problems.add("evidence_metrics_field_unknown")
            for name in METRICS:
                if name not in metrics:
                    problems.add(f"evidence_metric_missing:{name}")
                elif not _metric_valid(name, metrics[name]):
                    problems.add(f"evidence_metric_invalid:{name}")
    return tuple(sorted(problems))


def _run_valid(run: Any) -> bool:
    if not isinstance(run, Mapping) or set(run) != {"status", "failures"}:
        return False
    status, failures = run["status"], run["failures"]
    if not _in(status, RUN_STATUSES) or not _is_code_list(failures, RUN_FAILURE_CODES):
        return False
    if status == RUN_COMPLETED:
        return not failures
    if status == RUN_FAILED:
        return bool(failures)
    return True


def _edge_health_valid(block: Any) -> bool:
    if not isinstance(block, Mapping) or set(block) != _EDGE_HEALTH_FIELDS:
        return False
    status, failures = block["status"], block["failures"]
    if (
        block["schema"] != EDGE_HEALTH_SCHEMA
        or not _in(block["tier"], _DEVICE_TIER_VALUES)
        or not _in(status, EDGE_HEALTH_STATUSES)
        or not _is_code_list(failures, EDGE_HEALTH_FAILURE_CODES)
        or not _is_digest(block["report_digest"])
    ):
        return False
    if status == "passed":
        return not failures
    if status == "failed":
        return bool(failures)
    return True


def evidence_record_digest(record: Mapping[str, Any]) -> str:
    """Return the digest that identifies an evidence record in reports.

    Args:
        record: A decoded evidence record.

    Returns:
        Lower-case hex SHA-256 of the record's canonical JSON.

    Raises:
        KitQualificationError: If the record is not JSON-serialisable.
    """
    digest = _try_hash(record)
    if digest is None:
        raise KitQualificationError(("evidence_not_serialisable",))
    return digest


def edge_health_block(report: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise an EdgeHealthBench report for an evidence record.

    The report digest is recomputed, so an edited report is refused. Only the
    schema, tier, status, failure codes and digest are kept.

    Args:
        report: A report from
            :func:`openmed.eval.suites.edge_health.run_device_profile`, either
            in memory or decoded from JSON.

    Returns:
        The value for an evidence record's ``edge_health`` field.

    Raises:
        KitQualificationError: If the report is not a v1 report or its digest
            does not match its content.
    """
    if not isinstance(report, Mapping) or report.get("schema") != EDGE_HEALTH_SCHEMA:
        raise KitQualificationError(("edge_health_report_invalid",))
    body = {key: value for key, value in report.items() if key != "report_digest"}
    if _try_hash(body) != report.get("report_digest"):
        raise KitQualificationError(("edge_health_report_digest_mismatch",))
    failures = report.get("failures")
    block = {
        "schema": EDGE_HEALTH_SCHEMA,
        "tier": report.get("tier"),
        "status": report.get("status"),
        "failures": sorted(failures) if isinstance(failures, list) else failures,
        "report_digest": report["report_digest"],
    }
    if not _edge_health_valid(block):
        raise KitQualificationError(("edge_health_report_invalid",))
    return block


# --- collection --------------------------------------------------------------


def _version_matches(version: str, declared: str) -> bool:
    key, prefix = version_key(version), version_key(declared)
    return key[: len(prefix)] == prefix or key == prefix


def _runtime_version_ok(version: str, pin: Mapping[str, Any]) -> bool:
    pinning, declared = pin["pinning"], pin["version"]
    if pinning == "exact":
        return version_key(version) == version_key(declared)
    if pinning == "minimum":
        return version_key(version) >= version_key(declared)
    if pinning == "major":
        return version_key(version)[0] == version_key(declared)[0]
    return True  # unpinned and system runtimes only need a recorded version


def _consistency_reasons(
    record: Mapping[str, Any], row: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    execution, device_class = record["execution"], record["device_class"]
    cell_class = row["device_class"]
    if record["platform"]["name"] != row["platform"]["name"]:
        reasons.append("platform_mismatch")
    elif not _version_in_range(record["platform"]["version"], row["platform"]):
        reasons.append("platform_version_out_of_range")
    if record["runtime"]["pin"] != row["runtime"]:
        reasons.append("runtime_mismatch")
    if record["provider"] != row["provider"]:
        reasons.append("provider_mismatch")
    if execution == PHYSICAL_DEVICE:
        if cell_class in VIRTUAL_DEVICE_CLASSES:
            reasons.append("execution_device_class_mismatch")
        elif device_class != cell_class:
            reasons.append("device_class_mismatch")
        elif row["abi"] not in (ANY, "any") and record["abi"] != row["abi"]:
            reasons.append("abi_mismatch")
    elif device_class != cell_class and (
        VIRTUAL_DEVICE_CLASSES.get(device_class) != cell_class
    ):
        reasons.append("device_class_mismatch")
    return reasons


def _target_for(
    record: Mapping[str, Any],
    entry: Mapping[str, Any],
    plan: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, str, str, str, str] | None, list[str]]:
    reasons: list[str] = []
    bundle_id = record["bundle"]["id"]
    fixture_id = record["fixture"]["id"]
    policy_id = record["policy"]["id"]
    if bundle_id not in entry["bundles"]:
        reasons.append("bundle_not_planned")
    elif record["precision"] != bundles[bundle_id]["precision"]:
        reasons.append("precision_mismatch")
    fixtures = entry.get("fixtures", [row["id"] for row in plan["fixtures"]])
    if fixture_id not in fixtures:
        reasons.append("fixture_not_planned")
    policies = entry.get("policies", [row["id"] for row in plan["policies"]])
    if policy_id not in policies:
        reasons.append("policy_not_planned")
    version: str = ANY
    if "platform_versions" in entry:
        matches = [
            declared
            for declared in entry["platform_versions"]
            if _version_matches(record["platform"]["version"], declared)
        ]
        if not matches:
            reasons.append("platform_version_unplanned")
        else:
            version = max(matches, key=lambda value: len(version_key(value)))
    if reasons:
        return None, reasons
    return (entry["cell"], bundle_id, fixture_id, policy_id, version), []


def _stale_reasons(
    record: Mapping[str, Any],
    plan: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
    pin: Mapping[str, Any],
    plan_fresh: bool,
) -> list[str]:
    reasons: list[str] = []
    if not plan_fresh:
        reasons.append("cell_definition_changed")
    if record["bundle"]["digest"] != bundles[record["bundle"]["id"]]["digest"]:
        reasons.append("bundle_digest_mismatch")
    for name, rows in (("fixture", plan["fixtures"]), ("policy", plan["policies"])):
        pinned = {row["id"]: row["digest"] for row in rows}
        if record[name]["digest"] != pinned[record[name]["id"]]:
            reasons.append(f"{name}_digest_mismatch")
    if not _runtime_version_ok(record["runtime"]["version"], pin):
        reasons.append("runtime_version_mismatch")
    harness = plan.get("harness_digest")
    if harness is not None and record["harness_digest"] != harness:
        reasons.append("harness_digest_mismatch")
    return reasons


def _gate(
    name: str,
    status: str,
    reasons: Sequence[str] = (),
    *,
    measurement: str | None = None,
    observed: Any = None,
    limit: Any = None,
) -> dict[str, Any]:
    return {
        "gate": name,
        "status": status,
        "reasons": list(reasons),
        "measurement": measurement,
        "observed": observed,
        "limit": limit,
    }


def _threshold_gate(
    name: str,
    metric: Mapping[str, Any],
    field: str,
    limit: float,
    missing: str,
    failure: str,
    *,
    minimum: bool = False,
) -> dict[str, Any]:
    if metric["status"] != MEASURED:
        return _gate(
            name, GATE_MISSING, [missing], measurement=UNAVAILABLE, limit=limit
        )
    value = metric[field]
    ok = value >= limit if minimum else value <= limit
    return _gate(
        name,
        GATE_PASSED if ok else GATE_FAILED,
        [] if ok else [failure],
        measurement=MEASURED,
        observed=value,
        limit=limit,
    )


# gate, metric, field, plan gate, missing code, failure code, lower bound
_TIMING_GATES: Final = (
    (
        "cold_latency",
        "cold_latency_ms",
        "value",
        "maximum_cold_latency_ms",
        "missing_latency_evidence",
        "cold_latency",
        False,
    ),
    (
        "warm_latency",
        "warm_latency_ms",
        "p95",
        "maximum_warm_p95_latency_ms",
        "missing_latency_evidence",
        "latency",
        False,
    ),
    (
        "sustained_duration",
        "sustained_latency_ms",
        "duration_s",
        "minimum_sustained_seconds",
        "missing_latency_evidence",
        "sustained_duration",
        True,
    ),
    (
        "sustained_latency",
        "sustained_latency_ms",
        "p95",
        "maximum_sustained_p95_latency_ms",
        "missing_latency_evidence",
        "sustained_latency",
        False,
    ),
    (
        "cancel_latency",
        "cancel_latency_ms",
        "p95",
        "maximum_cancel_latency_ms",
        "missing_latency_evidence",
        "cancel_latency",
        False,
    ),
)


def _run_gates(
    record: Mapping[str, Any], row: Mapping[str, Any]
) -> list[dict[str, Any]]:
    run = record["run"]
    results = [
        _gate("run", GATE_PASSED)
        if run["status"] == RUN_COMPLETED
        else _gate("run", GATE_FAILED, sorted(run["failures"]))
    ]
    block = record.get("edge_health")
    expected_tier = edge_health_tier(record["device_class"], row["provider"]).value
    if block is None:
        results.append(_gate("edge_health", GATE_NOT_APPLICABLE))
    elif block["tier"] != expected_tier:
        results.append(_gate("edge_health", GATE_FAILED, ["edge_health_tier_mismatch"]))
    elif block["status"] == "failed":
        results.append(_gate("edge_health", GATE_FAILED, sorted(block["failures"])))
    else:
        results.append(_gate("edge_health", GATE_PASSED))
    return results


def _privacy_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> list[dict[str, Any]]:
    recall = metrics["direct_identifier_recall"]
    minimum = gates["minimum_direct_identifier_recall"]
    if recall["status"] == MEASURED:
        recall = {"status": MEASURED, "rate": recall["detected"] / recall["expected"]}
    recall_gate = _threshold_gate(
        "recall",
        recall,
        "rate",
        minimum,
        "missing_privacy_evidence",
        "recall",
        minimum=True,
    )
    leakage = metrics["leakage"]
    if leakage["status"] != MEASURED:
        return [
            recall_gate,
            _gate(
                "leakage",
                GATE_MISSING,
                ["missing_privacy_evidence"],
                measurement=UNAVAILABLE,
                limit=0,
            ),
        ]
    reasons = []
    if not leakage["assessed"]:
        reasons.append("unassessed_or_failed_units")
    if leakage["critical"] or leakage["any_phi"]:
        reasons.append("critical_or_document_phi_leakage")
    leakage_gate = _gate(
        "leakage",
        GATE_FAILED if reasons else GATE_PASSED,
        reasons,
        measurement=MEASURED,
        observed=leakage["any_phi"],
        limit=0,
    )
    return [recall_gate, leakage_gate]


def _timing_gates(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> list[dict[str, Any]]:
    results = []
    for name, metric, field, gate, missing, failure, minimum in _TIMING_GATES:
        value = metrics[metric]
        result = _threshold_gate(
            name, value, field, gates[gate], missing, failure, minimum=minimum
        )
        if (
            name == "warm_latency"
            and value["status"] == MEASURED
            and value["n"] < MIN_WARM_RUNS
        ):
            reasons = sorted({*result["reasons"], "incomplete_measurements"})
            result.update(status=GATE_FAILED, reasons=reasons)
        results.append(result)
    return results


def _resource_gates(
    metrics: Mapping[str, Any], row: Mapping[str, Any], gates: Mapping[str, Any]
) -> list[dict[str, Any]]:
    missing, budget = "missing_resource_evidence", "resource_budget"
    memory_mb = min(
        value
        for value in (row["ram_ceiling_mb"], gates.get("maximum_memory_peak_mb"))
        if value is not None
    )
    results = [
        _threshold_gate(
            "memory",
            metrics["memory_peak_bytes"],
            "value",
            memory_mb * _MIB,
            missing,
            budget,
        )
    ]
    device_memory = metrics["device_memory_peak_bytes"]
    device_budget = gates.get("maximum_device_memory_mb")
    if device_budget is not None and (
        row["device_memory_required"] or device_memory["status"] == MEASURED
    ):
        results.append(
            _threshold_gate(
                "device_memory",
                device_memory,
                "value",
                device_budget * _MIB,
                missing,
                budget,
            )
        )
    else:
        results.append(
            _gate(
                "device_memory",
                GATE_NOT_APPLICABLE,
                measurement=device_memory["status"],
            )
        )
    thermal = metrics["thermal"]
    if thermal["status"] == MEASURED or gates["require_thermal"]:
        results.append(
            _threshold_gate(
                "thermal",
                thermal,
                "max_level",
                gates["maximum_thermal_level"],
                missing,
                budget,
            )
        )
    else:
        results.append(_gate("thermal", GATE_INFORMATIONAL, measurement=UNAVAILABLE))
    energy = metrics["energy"]
    label = _energy_label(energy)
    observed = energy.get("joules", energy.get("value"))
    if energy["status"] == MEASURED:
        status, reasons = GATE_PASSED, []
    elif gates["require_measured_energy"]:
        status, reasons = GATE_MISSING, [missing]
    else:
        status, reasons = GATE_INFORMATIONAL, []
    results.append(
        _gate("energy", status, reasons, measurement=label, observed=observed)
    )
    return results


def _provider_gate(
    metrics: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    fallback = metrics["provider_fallback"]
    if row["provider"] not in ACCELERATED_PROVIDERS:
        return _gate(
            "provider_binding", GATE_NOT_APPLICABLE, measurement=fallback["status"]
        )
    if fallback["status"] != MEASURED:
        return _gate(
            "provider_binding",
            GATE_MISSING,
            ["missing_provider_evidence"],
            measurement=UNAVAILABLE,
        )
    fell_back = fallback["observed"]
    return _gate(
        "provider_binding",
        GATE_FAILED if fell_back else GATE_PASSED,
        ["runtime_fallback"] if fell_back else [],
        measurement=MEASURED,
        observed=fell_back,
    )


def _evaluate_gates(
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate every gate separately, in :data:`GATES` order."""
    metrics = record["metrics"]
    return [
        *_run_gates(record, row),
        *_privacy_gates(metrics, gates),
        *_timing_gates(metrics, gates),
        *_resource_gates(metrics, row, gates),
        _provider_gate(metrics, row),
    ]


def _energy_label(energy: Mapping[str, Any]) -> str:
    if energy["status"] == PROXY:
        return f"{PROXY}:{energy['proxy']}"
    return str(energy["status"])


def _record_entry(
    record: Mapping[str, Any],
    digest: str,
    row: Mapping[str, Any],
    gates: Mapping[str, Any],
    stale: Sequence[str],
) -> dict[str, Any]:
    metrics = record["metrics"]
    energy = metrics["energy"]
    entry: dict[str, Any] = {
        "record": digest,
        "execution": record["execution"],
        "evidence_class": _EVIDENCE_CLASS[record["execution"]],
        "device_class": record["device_class"],
        "recorded_on": record["recorded_on"],
        "platform_version": record["platform"]["version"],
        "runtime_version": record["runtime"]["version"],
        "outcome": RECORD_PASSED,
        "reasons": [],
        "thermal_measured": metrics["thermal"]["status"] == MEASURED,
        "energy_measured": energy["status"] == MEASURED,
        "energy_measurement": _energy_label(energy),
        "device_memory_measured": (
            metrics["device_memory_peak_bytes"]["status"] == MEASURED
        ),
        "gates": [],
    }
    block = record.get("edge_health")
    if stale:
        entry.update(outcome=RECORD_STALE, reasons=sorted(stale))
    elif record["run"]["status"] == RUN_NOT_RUN or (
        block is not None and block["status"] == "not_run"
    ):
        failures = sorted({"not_run", *record["run"]["failures"]})
        entry.update(outcome=RECORD_NOT_RUN, reasons=failures)
    else:
        results = _evaluate_gates(record, row, gates)
        failed = [result for result in results if result["status"] not in _GATE_OK]
        reasons = sorted({reason for result in failed for reason in result["reasons"]})
        entry.update(
            outcome=RECORD_FAILED if failed else RECORD_PASSED,
            reasons=reasons,
            gates=results,
        )
    return entry


def _outcome_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts = {name: dict.fromkeys(RECORD_OUTCOMES, 0) for name in EVIDENCE_CLASSES}
    for record in records:
        counts[record["evidence_class"]][record["outcome"]] += 1
    return counts


def _target_status(
    records: Sequence[Mapping[str, Any]], plan_fresh: bool
) -> tuple[str, list[str]]:
    if not plan_fresh:
        return STALE, ["cell_definition_changed"]
    device = [row for row in records if row["evidence_class"] == DEVICE_EVIDENCE]
    fresh = [row for row in device if row["outcome"] != RECORD_STALE]
    failed = [row for row in fresh if row["outcome"] == RECORD_FAILED]
    if failed:
        return FAILED, sorted({reason for row in failed for reason in row["reasons"]})
    if any(row["outcome"] == RECORD_PASSED for row in fresh):
        return QUALIFIED, []
    if fresh:
        return NOT_RUN, ["not_run"]
    if device:
        return STALE, sorted({reason for row in device for reason in row["reasons"]})
    classes = {row["evidence_class"] for row in records}
    if SIMULATOR_EVIDENCE in classes:
        return SIMULATOR_ONLY, ["device_evidence_missing"]
    if FAKE_RUNTIME_EVIDENCE in classes:
        return FAKE_RUNTIME_ONLY, ["device_evidence_missing"]
    return MISSING, ["no_evidence"]


def _coverage_gaps(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    bundles: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    planned = {
        (bundles[bundle_id]["family"], bundles[bundle_id]["precision"])
        for bundle_id in entry["bundles"]
    }
    return [
        f"family_precision_unplanned:{family}:{precision}"
        for family in row["families"]
        for precision in row["precisions"]
        if (family, precision) not in planned
    ]


def collect_qualification(
    matrix: Mapping[str, Any],
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Match evidence records to plan targets and build the qualification report.

    Each record is validated, matched to exactly one target, checked for
    staleness (bundle, fixture, policy or harness digest, runtime version, or
    a changed cell definition) and, when fresh, evaluated gate by gate. A
    target is ``qualified`` only with at least one passing physical-device
    record and no failing one. A cell is qualified only when every target is
    qualified and its plan covers every family and precision it declares.

    Args:
        matrix: A valid support matrix.
        plan: A valid qualification plan for that matrix.
        records: Decoded evidence records. ``None`` stands for an input that
            could not be decoded and is reported as ``evidence_unreadable``.

    Returns:
        A deterministic, digest-bound report. It names records by digest and
        never repeats rejected values.

    Raises:
        KitQualificationError: If the matrix or plan is invalid, or more than
            :data:`MAX_EVIDENCE_RECORDS` records are supplied.
    """
    _require_valid_matrix(matrix)
    _require_valid_plan(plan, matrix)
    if len(records) > MAX_EVIDENCE_RECORDS:
        raise KitQualificationError(("evidence_too_many",))
    rows = {row["cell"]: row for row in qualification_cells(matrix)}
    pins = _pins(matrix)
    entries = {entry["cell"]: entry for entry in plan["cells"]}
    bundles = {row["id"]: row for row in plan["bundles"]}
    fresh_plan = {
        cell: entry["cell_digest"] == rows[cell]["cell_digest"]
        for cell, entry in entries.items()
    }
    targets = {
        _target_key(target): target for target in qualification_targets(plan, matrix)
    }
    matched: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {
        key: [] for key in targets
    }
    known_cells = frozenset(cell["id"] for cell in matrix["cells"])
    unsupported = frozenset(entry["id"] for entry in matrix["unsupported"])
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    accepted = 0
    for record in records:
        if record is None:
            rejected.append({"record": None, "reasons": ["evidence_unreadable"]})
            continue
        digest = _try_hash(record)
        problems = validate_evidence_record(record)
        if problems or digest is None:
            rejected.append(
                {"record": digest, "reasons": list(problems or ["evidence_invalid"])}
            )
            continue
        if digest in seen:
            rejected.append({"record": digest, "reasons": ["evidence_duplicate"]})
            continue
        seen.add(digest)
        cell = record["cell"]
        if cell in unsupported:
            rejected.append({"record": digest, "reasons": ["cell_unsupported"]})
            continue
        if cell not in known_cells:
            rejected.append({"record": digest, "reasons": ["cell_unknown"]})
            continue
        if cell not in rows:
            rejected.append({"record": digest, "reasons": ["cell_out_of_scope"]})
            continue
        if cell not in entries:
            rejected.append({"record": digest, "reasons": ["cell_not_planned"]})
            continue
        row = rows[cell]
        reasons = _consistency_reasons(record, row)
        key, target_reasons = _target_for(record, entries[cell], plan, bundles)
        reasons += target_reasons
        if reasons or key is None:
            rejected.append({"record": digest, "reasons": sorted(reasons)})
            continue
        stale = _stale_reasons(
            record, plan, bundles, pins[row["runtime"]], fresh_plan[cell]
        )
        gates = effective_gates(plan, cell)
        matched[key].append(_record_entry(record, digest, row, gates, stale))
        accepted += 1

    target_rows: list[dict[str, Any]] = []
    by_cell: dict[str, list[dict[str, Any]]] = {cell: [] for cell in entries}
    for key, target in targets.items():
        found = sorted(matched[key], key=lambda row: row["record"])
        status, reasons = _target_status(found, fresh_plan[target["cell"]])
        target_row = {
            **target,
            "status": status,
            "reasons": reasons,
            "evidence": _outcome_counts(found),
            "records": found,
        }
        target_rows.append(target_row)
        by_cell[target["cell"]].append(target_row)

    cell_rows: list[dict[str, Any]] = []
    for cell, row in rows.items():
        base = {
            "cell": cell,
            "surface": row["surface"],
            "matrix_status": row["matrix_status"],
            "device_class": row["device_class"],
            "provider": row["provider"],
            "tier": row["tier"],
            "ram_ceiling_mb": row["ram_ceiling_mb"],
            "edge_health_tier": row["edge_health_tier"],
            "virtual_device_class": row["virtual_device_class"],
        }
        if cell not in entries:
            cell_rows.append(
                {
                    **base,
                    "planned": False,
                    "plan_fresh": None,
                    "qualified": False,
                    "status": UNPLANNED,
                    "targets": 0,
                    "targets_by_status": dict.fromkeys(TARGET_STATUSES, 0),
                    "coverage_gaps": [],
                    "gates": None,
                }
            )
            continue
        cell_targets = by_cell[cell]
        counts = dict.fromkeys(TARGET_STATUSES, 0)
        for target_row in cell_targets:
            counts[target_row["status"]] += 1
        gaps = _coverage_gaps(row, entries[cell], bundles)
        worst = next(
            (status for status in TARGET_STATUSES if counts[status]), QUALIFIED
        )
        qualified = worst == QUALIFIED and not gaps and bool(cell_targets)
        status = INCOMPLETE if worst == QUALIFIED and not qualified else worst
        cell_rows.append(
            {
                **base,
                "planned": True,
                "plan_fresh": fresh_plan[cell],
                "qualified": qualified,
                "status": status,
                "targets": len(cell_targets),
                "targets_by_status": counts,
                "coverage_gaps": gaps,
                "gates": effective_gates(plan, cell),
            }
        )

    pins_by_id = pins
    out_of_scope = [
        {"cell": cell["id"], "reason": "runtime_kind_not_inference"}
        for cell in matrix["cells"]
        if pins_by_id[cell["runtime"]]["kind"] != "inference"
    ]
    rejected.sort(key=lambda row: (row["record"] or "", row["reasons"]))
    target_counts = dict.fromkeys(TARGET_STATUSES, 0)
    for target_row in target_rows:
        target_counts[target_row["status"]] += 1
    class_counts = dict.fromkeys(EVIDENCE_CLASSES, 0)
    for target_row in target_rows:
        for record_row in target_row["records"]:
            class_counts[record_row["evidence_class"]] += 1
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "matrix": {
            "matrix": matrix["matrix"],
            "release": matrix["release"],
            "frozen_on": matrix["frozen_on"],
            "digest": matrix_digest(matrix),
        },
        "plan_digest": _hash(plan),
        "collector_digest": _source_digest(),
        "summary": {
            "cells_in_scope": len(rows),
            "cells_planned": len(entries),
            "cells_qualified": sum(1 for row in cell_rows if row["qualified"]),
            "cells_unplanned": sum(1 for row in cell_rows if not row["planned"]),
            "cells_out_of_scope": len(out_of_scope),
            "targets": len(target_rows),
            "targets_by_status": target_counts,
            "records": len(records),
            "records_accepted": accepted,
            "records_rejected": len(rejected),
            "records_by_class": class_counts,
        },
        "cells": cell_rows,
        "targets": target_rows,
        "out_of_scope_cells": out_of_scope,
        "supported_without_device_qualification": [
            row["cell"]
            for row in cell_rows
            if row["matrix_status"] == SUPPORTED and not row["qualified"]
        ],
        "rejected_records": rejected,
        "auto_promotion": False,
        "requires_review": True,
        "clinical_readiness": False,
        "independent_validation": False,
        "notice": _NOTICE,
    }
    report["report_digest"] = _hash(report)
    return report


# --- matrix proposals --------------------------------------------------------


def _verify_digest(document: Mapping[str, Any], field: str, code: str) -> None:
    body = {key: value for key, value in document.items() if key != field}
    if _try_hash(body) != document.get(field):
        raise KitQualificationError((code,))


def _with_evidence(refs: Any, evidence_id: str) -> list[str]:
    if refs == PENDING_EVIDENCE:
        return [evidence_id]
    current = list(refs)
    return current if evidence_id in current else [*current, evidence_id]


def propose_matrix_updates(
    matrix: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    """Turn qualified cells into a reviewable support-matrix change.

    Only cells the report marks ``qualified`` (every target qualified by
    physical-device evidence) are proposed. Each gets a ``device-run``
    evidence record; experimental cells are proposed for ``supported`` and
    their surface claim follows. Supported cells that are planned but not
    qualified are listed for review and never demoted. The proposal is
    checked by applying it to a copy and validating the result.

    Args:
        matrix: The support matrix the report was collected against.
        report: A report from :func:`collect_qualification`.
        evidence_refs: Repository paths where the report and records will be
            committed. They become the new evidence records' ``refs``.

    Returns:
        A digest-bound proposal with ``auto_applied`` set to false.

    Raises:
        KitQualificationError: If the report was edited or collected against a
            different matrix, no evidence reference is given, or the proposed
            matrix would fail validation.
    """
    _require_valid_matrix(matrix)
    if not isinstance(report, Mapping) or report.get("schema") != REPORT_SCHEMA:
        raise KitQualificationError(("report_invalid",))
    _verify_digest(report, "report_digest", "report_digest_mismatch")
    if report["matrix"]["digest"] != matrix_digest(matrix):
        raise KitQualificationError(("report_matrix_mismatch",))
    refs = list(evidence_refs)
    if not refs:
        raise KitQualificationError(("proposal_evidence_refs_missing",))
    cells = {cell["id"]: cell for cell in matrix["cells"]}
    surfaces = {row["id"]: row for row in matrix["surfaces"]}
    existing = {row["id"] for row in matrix["evidence"]}
    evidence: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    additions: list[dict[str, Any]] = []
    claims: dict[str, dict[str, Any]] = {}
    reviews: list[dict[str, Any]] = []
    for row in report["cells"]:
        cell = cells[row["cell"]]
        if not row["qualified"]:
            if row["planned"] and cell["status"] == SUPPORTED:
                reviews.append(
                    {"cell": row["cell"], "status": row["status"], "action": "review"}
                )
            continue
        qualified = row["targets_by_status"][QUALIFIED]
        if qualified < 1 or qualified != row["targets"]:
            raise KitQualificationError(("report_inconsistent",))
        evidence_id = f"{DEVICE_RUN_PREFIX}{row['cell']}"
        if evidence_id not in existing:
            evidence.append(
                {
                    "id": evidence_id,
                    "kind": "device-run",
                    "covers": list(DEVICE_RUN_COVERS),
                    "refs": refs,
                    "note": (
                        f"Physical-device qualification: {qualified} planned "
                        f"target{'s' if qualified != 1 else ''} passed every "
                        f"declared gate (report {report['report_digest'][:12]})."
                    ),
                }
            )
        if cell["status"] == EXPERIMENTAL:
            promotions.append(
                {
                    "cell": row["cell"],
                    "from": EXPERIMENTAL,
                    "to": SUPPORTED,
                    "evidence": evidence_id,
                    "qualified_targets": qualified,
                }
            )
            surface = surfaces[cell["surface"]]
            if surface["support_claim"] != "supported":
                claims[surface["id"]] = {
                    "surface": surface["id"],
                    "from": surface["support_claim"],
                    "to": "supported",
                }
        elif (
            cell["evidence"] == PENDING_EVIDENCE or evidence_id not in cell["evidence"]
        ):
            additions.append({"cell": row["cell"], "evidence": evidence_id})
    proposal: dict[str, Any] = {
        "schema": PROPOSAL_SCHEMA,
        "matrix": dict(report["matrix"]),
        "report_digest": report["report_digest"],
        "auto_applied": False,
        "evidence": evidence,
        "promotions": promotions,
        "evidence_additions": additions,
        "surface_claims": [claims[key] for key in sorted(claims)],
        "reviews": reviews,
        "notice": (
            "Review before applying. Apply to a copy with apply_matrix_proposal, "
            "then regenerate docs/openmedkit-support-matrix.md and run "
            "scripts/release/check_kit_support_matrix.py."
        ),
    }
    proposal["proposal_digest"] = _hash(proposal)
    apply_matrix_proposal(matrix, proposal)
    return proposal


def apply_matrix_proposal(
    matrix: Mapping[str, Any], proposal: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a copy of the matrix with a proposal applied and validated.

    The input matrix is not modified and nothing is written to disk.

    Args:
        matrix: The support matrix the proposal was made for.
        proposal: A proposal from :func:`propose_matrix_updates`.

    Returns:
        The updated matrix document.

    Raises:
        KitQualificationError: If the proposal was edited, targets a different
            matrix, names a cell that is no longer experimental, or the result
            fails :func:`openmed.eval.kit_support_matrix.validate_kit_support_matrix`.
    """
    _require_valid_matrix(matrix)
    if not isinstance(proposal, Mapping) or proposal.get("schema") != PROPOSAL_SCHEMA:
        raise KitQualificationError(("proposal_invalid",))
    _verify_digest(proposal, "proposal_digest", "proposal_digest_mismatch")
    try:
        return _apply(matrix, proposal)
    except (KeyError, TypeError, AttributeError):
        raise KitQualificationError(("proposal_invalid",)) from None


def _apply(matrix: Mapping[str, Any], proposal: Mapping[str, Any]) -> dict[str, Any]:
    if proposal["matrix"]["digest"] != matrix_digest(matrix):
        raise KitQualificationError(("proposal_matrix_mismatch",))
    updated = copy.deepcopy(dict(matrix))
    existing = {row["id"] for row in updated["evidence"]}
    for record in proposal["evidence"]:
        if record["id"] not in existing:
            updated["evidence"].append(dict(record))
            existing.add(record["id"])
    cells = {cell["id"]: cell for cell in updated["cells"]}
    for promotion in proposal["promotions"]:
        cell = cells.get(promotion["cell"])
        if cell is None or cell["status"] != EXPERIMENTAL:
            raise KitQualificationError(("proposal_cell_stale",))
        cell["status"] = SUPPORTED
        cell["evidence"] = _with_evidence(cell["evidence"], promotion["evidence"])
    for addition in proposal["evidence_additions"]:
        cell = cells.get(addition["cell"])
        if cell is None:
            raise KitQualificationError(("proposal_cell_stale",))
        cell["evidence"] = _with_evidence(cell["evidence"], addition["evidence"])
    surfaces = {row["id"]: row for row in updated["surfaces"]}
    for claim in proposal["surface_claims"]:
        surfaces[claim["surface"]]["support_claim"] = claim["to"]
    problems = validate_kit_support_matrix(updated)
    if problems:
        raise KitQualificationError(
            tuple(f"proposed_matrix:{code}" for code in problems)
        )
    return updated


__all__ = [
    "ACCELERATED_PROVIDERS",
    "CELL_STATUSES",
    "DEVICE_EVIDENCE",
    "DEVICE_MEMORY_SCOPES",
    "DEVICE_RUN_COVERS",
    "DEVICE_RUN_PREFIX",
    "EDGE_HEALTH_SCHEMA",
    "EDGE_HEALTH_STATUSES",
    "EMULATOR",
    "ENERGY_PROXIES",
    "EVIDENCE_CLASSES",
    "EVIDENCE_SCHEMA",
    "EXECUTIONS",
    "FAKE_RUNTIME",
    "FAKE_RUNTIME_EVIDENCE",
    "GATES",
    "MAX_EVIDENCE_RECORDS",
    "MAX_THERMAL_LEVEL",
    "METRICS",
    "MIN_WARM_RUNS",
    "OPTIONAL_GATES",
    "PHYSICAL_DEVICE",
    "PLAN_SCHEMA",
    "PROPOSAL_SCHEMA",
    "RECORD_OUTCOMES",
    "REPORT_SCHEMA",
    "REQUIRED_GATES",
    "RSS_SCOPES",
    "RUN_FAILURE_CODES",
    "RUN_STATUSES",
    "SIMULATOR",
    "SIMULATOR_EVIDENCE",
    "TARGET_STATUSES",
    "VIRTUAL_DEVICE_CLASSES",
    "KitQualificationError",
    "apply_matrix_proposal",
    "cell_digest",
    "collect_qualification",
    "edge_health_block",
    "edge_health_tier",
    "effective_gates",
    "evidence_record_digest",
    "matrix_digest",
    "plan_skeleton",
    "propose_matrix_updates",
    "qualification_cells",
    "qualification_targets",
    "validate_evidence_record",
    "validate_qualification_plan",
]
