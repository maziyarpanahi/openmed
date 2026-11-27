"""Measured local device profiles with independent privacy and quality gates.

Tier names and host declarations are not remote attestation. Unavailable tiers
remain unmeasured. CPU activity is an energy proxy, never joules or GPU energy.
Android ONNX Runtime and iOS-family device tiers name the runtime a host bridge
drives on a phone, tablet, watch or headset; the harness cannot tell a physical
device from a simulator, so the bridge declares which one it ran.
"""

from __future__ import annotations

import math
import platform
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from statistics import fmean
from typing import Any, Final, Protocol

from openmed.eval.suites.openphi_vision import (
    ReleaseProvenance,
    SyntheticVisionControl,
    VisionCase,
    VisionGatePolicy,
    _digest,
    _hash,
    _rate,
    _require,
    _sha,
    run_openphi_vision,
)


class DeviceTier(str, Enum):
    """Declared execution tiers, never inferred from a successful CPU fallback."""

    CPU = "cpu"
    BROWSER = "browser"
    MLX = "mlx"
    CORE_ML = "core_ml"
    ACCELERATOR = "optional_accelerator"
    ANDROID_ORT_CPU = "android_ort_cpu"
    ANDROID_ORT_NNAPI = "android_ort_nnapi"
    ANDROID_ORT_QNN = "android_ort_qnn"
    IOS_DEVICE_CPU = "ios_device_cpu"
    IOS_DEVICE_MLX = "ios_device_mlx"
    IOS_DEVICE_CORE_ML = "ios_device_core_ml"


# Tiers whose model memory lives outside process RSS (GPU, Neural Engine, NPU
# or DSP allocators). Their profiles need a separate allocator budget.
DEVICE_MEMORY_TIERS: Final = frozenset(
    {
        DeviceTier.MLX,
        DeviceTier.CORE_ML,
        DeviceTier.ACCELERATOR,
        DeviceTier.ANDROID_ORT_NNAPI,
        DeviceTier.ANDROID_ORT_QNN,
        DeviceTier.IOS_DEVICE_MLX,
        DeviceTier.IOS_DEVICE_CORE_ML,
    }
)
# Tiers measured on Android or iOS-family hardware through a host bridge.
MOBILE_DEVICE_TIERS: Final = frozenset(
    {
        DeviceTier.ANDROID_ORT_CPU,
        DeviceTier.ANDROID_ORT_NNAPI,
        DeviceTier.ANDROID_ORT_QNN,
        DeviceTier.IOS_DEVICE_CPU,
        DeviceTier.IOS_DEVICE_MLX,
        DeviceTier.IOS_DEVICE_CORE_ML,
    }
)
# Every failure code :func:`run_device_profile` can report.
FAILURE_CODES: Final = frozenset(
    {
        "abstention",
        "cancelled",
        "critical_or_document_phi_leakage",
        "incomplete_measurements",
        "latency",
        "missing_resource_evidence",
        "provider_or_measurement_failure",
        "quality",
        "reference_parity",
        "repeat_parity",
        "resource_budget",
        "throughput",
        "unassessed_or_failed_units",
        "utility",
    }
)


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceProfile:
    """Immutable checkpoint/workload binding and explicit per-axis budgets.

    The host qualifies these thresholds for its task. No clinical defaults are
    supplied. Disabling required thermal evidence leaves that metric unmeasured;
    it does not insert a nominal temperature or qualify sustained operation.
    """

    tier: DeviceTier
    provenance: ReleaseProvenance
    dataset_digest: str
    workload_digest: str
    reference_output_digest: str
    units_per_trial: int
    trials: int
    warmups: int
    maximum_latency_ms: float
    minimum_throughput: float
    maximum_rss_bytes: int
    maximum_thermal_level: int
    require_thermal: bool
    minimum_quality_rate: float
    minimum_utility_rate: float
    maximum_abstention_rate: float
    maximum_device_memory_bytes: int | None = None

    def __post_init__(self) -> None:
        _require(
            type(self.tier) is DeviceTier and type(self.provenance) is ReleaseProvenance
        )
        _digest(self.dataset_digest)
        _digest(self.workload_digest)
        _digest(self.reference_output_digest)
        for integer, low, high in (
            (self.units_per_trial, 1, 1000000),
            (self.trials, 3, 64),
            (self.warmups, 0, 8),
            (self.maximum_rss_bytes, 1, 2**50),
            (self.maximum_thermal_level, 0, 3),
        ):
            _require(type(integer) is int and low <= integer <= high)
        _require(type(self.require_thermal) is bool)
        _require(
            self.maximum_device_memory_bytes is None
            or (
                type(self.maximum_device_memory_bytes) is int
                and 0 < self.maximum_device_memory_bytes <= 2**50
            )
        )
        if self.tier in DEVICE_MEMORY_TIERS:
            _require(self.maximum_device_memory_bytes is not None)
        for value in (self.maximum_latency_ms, self.minimum_throughput):
            _require(type(value) in (int, float) and math.isfinite(value) and value > 0)
        for value in (
            self.minimum_quality_rate,
            self.minimum_utility_rate,
            self.maximum_abstention_rate,
        ):
            _require(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
            )

    @property
    def digest(self) -> str:
        """Bind all thresholds, immutable source declarations and execution tier."""
        return _hash(asdict(self))


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceTrial:
    """Value-free outcome from one actually executed local workload.

    Counts refer to this invocation's unique corpus units. Repeated invocations
    are not pooled as independent quality/privacy samples. The output digest
    supports exact repeat parity, not equivalence to another device's output.
    """

    provenance_digest: str
    dataset_digest: str
    output_digest: str
    units: int
    assessed: int
    critical_leaks: int
    any_phi_leaks: int
    quality_correct: int
    utility_correct: int
    abstentions: int
    failed: int = 0

    def __post_init__(self) -> None:
        for digest_value in (
            self.provenance_digest,
            self.dataset_digest,
            self.output_digest,
        ):
            _digest(digest_value)
        _require(type(self.units) is int and 1 <= self.units <= 1000000)
        for value in (
            self.assessed,
            self.critical_leaks,
            self.any_phi_leaks,
            self.quality_correct,
            self.utility_correct,
            self.abstentions,
            self.failed,
        ):
            _require(type(value) is int and 0 <= value <= self.units)
        _require(self.critical_leaks <= self.any_phi_leaks <= self.assessed)
        _require(self.assessed + self.abstentions + self.failed == self.units)


class DeviceWorkload(Protocol):
    """Trusted local adapter to a concrete workload and its immutable artifacts."""

    tier: DeviceTier
    local_only: bool
    provenance: ReleaseProvenance
    dataset_digest: str
    workload_digest: str

    def run(self) -> DeviceTrial:
        """Execute one bounded workload with no network, logs or implicit fallback."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ResourceSample:
    """Host readings with explicit missing values and measurement scope."""

    observed_at: float
    rss_bytes: int | None
    thermal_level: int | None
    source_digest: str
    rss_scope: str = "process_lifetime_highwater"
    device_memory_bytes: int | None = None
    device_memory_scope: str | None = None

    def __post_init__(self) -> None:
        _require(
            type(self.observed_at) in (float, int) and math.isfinite(self.observed_at)
        )
        _require(
            self.rss_bytes is None
            or (type(self.rss_bytes) is int and 0 < self.rss_bytes <= 2**50)
        )
        _require(
            self.thermal_level is None
            or (type(self.thermal_level) is int and 0 <= self.thermal_level <= 3)
        )
        _require(
            self.device_memory_bytes is None
            or (
                type(self.device_memory_bytes) is int
                and 0 <= self.device_memory_bytes <= 2**50
            )
        )
        _require(
            self.device_memory_scope in (None, "allocator_current", "allocator_peak")
        )
        _require(
            (self.device_memory_bytes is None) == (self.device_memory_scope is None)
        )
        _digest(self.source_digest)
        _require(self.rss_scope in ("process_lifetime_highwater", "process_current"))


def observe_cpu_resources() -> ResourceSample:
    """Read process-lifetime RSS high water; thermal state remains unavailable.

    RSS is not model-exclusive or device/GPU memory. No process-global tracing
    state is reset. Platforms with unknown resource units report missing RSS.
    """
    rss = None
    if platform.system() in ("Darwin", "Linux"):
        import resource

        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = int(raw * (1024 if platform.system() == "Linux" else 1))
        if rss <= 0:
            rss = None
    return ResourceSample(
        observed_at=time.perf_counter(),
        rss_bytes=rss,
        thermal_level=None,
        source_digest=_sha(b"getrusage RUSAGE_SELF ru_maxrss platform units v1"),
    )


def _continuous(values: list[float]) -> dict[str, Any]:
    """Descriptive seeded bootstrap CI for the mean, without normality claims."""
    if not values:
        return {"n": 0, "mean": None, "p95": None, "maximum": None, "mean_ci95": None}
    ordered = sorted(values)
    interval = None
    if len(values) >= 3:
        rng = random.Random(3205)
        means = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(512))
        interval = (means[12], means[499])
    return {
        "n": len(values),
        "mean": fmean(values),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "maximum": max(values),
        "mean_ci95": interval,
        "interval_method": "seeded_percentile_bootstrap_mean_512_descriptive",
    }


def run_device_profile(
    profile: DeviceProfile,
    workload: DeviceWorkload | None,
    *,
    observe: Callable[[], ResourceSample] = observe_cpu_resources,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = time.process_time,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Measure the actual callable while preserving independent failure gates.

    Cold/warmup runs participate in privacy, quality and repeat checks but not
    warm latency statistics. Resource samples are before/after observations,
    not proof of continuous peak coverage. Callbacks must be bounded and local;
    this evaluator cannot preempt or sandbox arbitrary provider code.
    """
    _require(type(profile) is DeviceProfile)
    binding = {
        **asdict(profile.provenance),
        "dataset_digest": profile.dataset_digest,
        "workload_digest": profile.workload_digest,
        "profile_digest": profile.digest,
        "harness_digest": _sha(Path(__file__).read_bytes()),
    }
    report: dict[str, Any] = {
        "schema": "openmed.edge_health.v1",
        "tier": profile.tier.value,
        "provenance": binding,
        "binding_digest": _hash(binding),
        "status": "not_run",
        "passed": False,
        "requires_review": True,
        "clinical_readiness": False,
        "independent_validation": False,
        "warmups_requested": profile.warmups,
        "trials_requested": profile.trials,
        "executed": 0,
        "failures": [],
        "quality_privacy_trials": [],
        "resource_scope": None,
        "measurements": {},
    }
    if workload is None:
        report["report_digest"] = _hash(report)
        return report
    measurements: dict[str, list[float]] = {
        name: []
        for name in (
            "latency_ms",
            "throughput_units_per_second",
            "process_rss_bytes",
            "device_allocator_bytes",
            "process_cpu_seconds_proxy",
            "observed_thermal_level",
        )
    }
    baseline = None
    failed: set[str] = set()
    sensor_identity = None
    measured = 0

    def validate_binding() -> None:
        _require(workload is not None and workload.tier is profile.tier)
        _require(workload.local_only is True)
        _require(workload.provenance == profile.provenance)
        _require(workload.dataset_digest == profile.dataset_digest)
        _require(workload.workload_digest == profile.workload_digest)

    def sample() -> ResourceSample:
        nonlocal sensor_identity
        value = observe()
        now = clock()
        _require(type(value) is ResourceSample and 0 <= now - value.observed_at <= 0.5)
        identity = (value.source_digest, value.rss_scope, value.device_memory_scope)
        if sensor_identity is None:
            sensor_identity = identity
        _require(identity == sensor_identity)
        return value

    def resource_failure(value: ResourceSample) -> str | None:
        if (
            value.rss_bytes is None
            or (profile.require_thermal and value.thermal_level is None)
            or (
                profile.maximum_device_memory_bytes is not None
                and value.device_memory_bytes is None
            )
        ):
            return "missing_resource_evidence"
        if (
            value.rss_bytes > profile.maximum_rss_bytes
            or (
                value.thermal_level is not None
                and value.thermal_level > profile.maximum_thermal_level
            )
            or (
                profile.maximum_device_memory_bytes is not None
                and value.device_memory_bytes is not None
                and value.device_memory_bytes > profile.maximum_device_memory_bytes
            )
        ):
            return "resource_budget"
        return None

    for index in range(profile.warmups + profile.trials):
        try:
            stop = cancelled()
            _require(type(stop) is bool)
            if stop:
                failed.add("cancelled")
                break
            validate_binding()
            before = sample()
            reason = resource_failure(before)
            if reason is not None:
                failed.add(reason)
                break
            started, cpu_started = clock(), cpu_clock()
            result = workload.run()
            cpu_ended, ended = cpu_clock(), clock()
            after = sample()
            validate_binding()
            _require(type(result) is DeviceTrial)
            _require(
                result.provenance_digest == profile.provenance.digest
                and result.dataset_digest == profile.dataset_digest
            )
            _require(result.units == profile.units_per_trial)
            elapsed, cpu = ended - started, cpu_ended - cpu_started
            _require(
                math.isfinite(elapsed)
                and elapsed > 0
                and math.isfinite(cpu)
                and cpu >= 0
            )
            _require(
                math.isfinite(elapsed * 1000) and math.isfinite(result.units / elapsed)
            )
            report["executed"] += 1
            if baseline is None:
                baseline = result.output_digest
            parity = baseline == result.output_digest
            reference_parity = profile.reference_output_digest == result.output_digest
            row = {
                "warmup": index < profile.warmups,
                "output_digest": result.output_digest,
                "quality": _rate(result.quality_correct, result.units),
                "utility": _rate(result.utility_correct, result.units),
                "abstention": _rate(result.abstentions, result.units),
                "critical_burned_in": _rate(result.critical_leaks, result.assessed),
                "document_any_phi": _rate(result.any_phi_leaks, result.assessed),
                "failed_units": result.failed,
                "repeat_parity": parity,
                "reference_parity": reference_parity,
            }
            report["quality_privacy_trials"].append(row)
            if result.failed or not result.assessed:
                failed.add("unassessed_or_failed_units")
            if result.critical_leaks or result.any_phi_leaks:
                failed.add("critical_or_document_phi_leakage")
            if result.quality_correct / result.units < profile.minimum_quality_rate:
                failed.add("quality")
            if result.utility_correct / result.units < profile.minimum_utility_rate:
                failed.add("utility")
            if result.abstentions / result.units > profile.maximum_abstention_rate:
                failed.add("abstention")
            if not parity:
                failed.add("repeat_parity")
            if not reference_parity:
                failed.add("reference_parity")
            reason = resource_failure(after)
            if reason is not None:
                failed.add(reason)
            if index >= profile.warmups:
                measured += 1
                measurements["latency_ms"].append(elapsed * 1000)
                measurements["throughput_units_per_second"].append(
                    result.units / elapsed
                )
                measurements["process_cpu_seconds_proxy"].append(cpu)
                if before.rss_bytes is not None and after.rss_bytes is not None:
                    measurements["process_rss_bytes"].append(
                        float(max(before.rss_bytes, after.rss_bytes))
                    )
                if (
                    before.device_memory_bytes is not None
                    and after.device_memory_bytes is not None
                ):
                    measurements["device_allocator_bytes"].append(
                        float(
                            max(before.device_memory_bytes, after.device_memory_bytes)
                        )
                    )
                thermals = [
                    value
                    for value in (before.thermal_level, after.thermal_level)
                    if value is not None
                ]
                if len(thermals) == 2:
                    measurements["observed_thermal_level"].append(float(max(thermals)))
        except Exception:
            failed.add("provider_or_measurement_failure")
            break
    report["measurements"] = {
        name: _continuous(values) for name, values in measurements.items()
    }
    if measured != profile.trials:
        failed.add("incomplete_measurements")
    if (
        measurements["latency_ms"]
        and max(measurements["latency_ms"]) > profile.maximum_latency_ms
    ):
        failed.add("latency")
    if (
        measurements["throughput_units_per_second"]
        and min(measurements["throughput_units_per_second"])
        < profile.minimum_throughput
    ):
        failed.add("throughput")
    report.update(
        status="passed" if not failed else "failed",
        passed=not failed,
        failures=sorted(failed),
        resource_scope=sensor_identity,
        measured_trials=measured,
        exact_repeat_baseline_digest=baseline,
        thermal_measured=len(measurements["observed_thermal_level"]) == profile.trials,
        device_memory_measured=len(measurements["device_allocator_bytes"])
        == profile.trials,
        energy_measured=False,
        energy_proxy="process_cpu_seconds_only_not_joules_or_gpu_energy",
        repeated_quality_samples_are_independent=False,
        notice="Host-declared tier and immutable artifacts; before/after resource observations, no continuous thermal peak or clinical qualification claim.",
    )
    report["report_digest"] = _hash(report)
    return report


class SyntheticCPUWorkload:
    """Actually execute the public vision controls for local CPU measurements."""

    tier = DeviceTier.CPU
    local_only = True

    def __init__(self, cases: tuple[VisionCase, ...], policy: VisionGatePolicy) -> None:
        self._candidate = SyntheticVisionControl()
        self._cases, self._policy = cases, policy
        self.provenance = self._candidate.provenance
        self.dataset_digest = _hash([case.digest for case in cases])
        self.workload_digest = _hash(
            {"harness": self.provenance.model_digest, "policy": asdict(policy)}
        )

    def run(self) -> DeviceTrial:
        """Measure real redaction/routing calls and retain only aggregate evidence."""
        _require(self.provenance == self._candidate.provenance)
        _require(self.dataset_digest == _hash([case.digest for case in self._cases]))
        _require(
            self.workload_digest
            == _hash(
                {
                    "harness": self.provenance.model_digest,
                    "policy": asdict(self._policy),
                }
            )
        )
        report = run_openphi_vision(self._candidate, self._cases, self._policy)
        _require(report["provenance"]["harness_digest"] == self.provenance.model_digest)
        tracks = tuple(report["tracks"].values())
        return DeviceTrial(
            provenance_digest=self.provenance.digest,
            dataset_digest=report["provenance"]["dataset_digest"],
            output_digest=report["output_digest"],
            units=report["expected_cases"],
            assessed=sum(track["assessed"] for track in tracks),
            critical_leaks=sum(
                track["critical_burned_in"]["events"] for track in tracks
            ),
            any_phi_leaks=sum(track["document_any_phi"]["events"] for track in tracks),
            quality_correct=sum(track["quality"]["events"] for track in tracks),
            utility_correct=sum(track["utility"]["events"] for track in tracks),
            abstentions=sum(track["abstention"]["events"] for track in tracks),
            failed=sum(track["failed"] for track in tracks),
        )
