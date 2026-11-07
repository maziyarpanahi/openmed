"""Executed leakage, provenance, abstention and independent device-gate controls."""

import hashlib
import io
import json
import socket
from dataclasses import replace

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pydicom")

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from openmed.eval.suites.openphi_vision import (
    SyntheticVisionControl,
    VisionGateError,
    VisionGatePolicy,
    VisionOutput,
    VisionTrack,
    run_openphi_vision,
    synthetic_vision_cases,
)


@pytest.fixture
def cases():
    return synthetic_vision_cases(cases_per_track=1)


@pytest.fixture
def policy():
    return VisionGatePolicy(
        minimum_cases_per_track=1,
        minimum_utility_rate=1,
        minimum_quality_rate=1,
        maximum_abstention_rate=0,
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket, "socket", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


class Candidate:
    def __init__(self, mutate):
        self.control = SyntheticVisionControl()
        self.mutate = mutate
        self.local_only = True
        self.provenance = self.control.provenance
        self.calls = 0

    def process(self, request):
        self.calls += 1
        return self.mutate(request, self.control.process(request))


def png(image, *, info=None):
    stream = io.BytesIO()
    image.save(stream, format="PNG", **({"pnginfo": info} if info else {}))
    return stream.getvalue()


def test_default_control_executes_every_track_with_cis_and_bound_provenance(
    cases, policy, capsys, caplog
):
    report = run_openphi_vision(SyntheticVisionControl(), cases, policy)
    assert report["passed"] and report["privacy_passed"]
    assert report["expected_cases"] == report["executed_cases"] == 7
    assert set(report["tracks"]) == {track.value for track in VisionTrack}
    assert report["requires_review"]
    assert not report["clinical_readiness"]
    assert (
        not report["independent_validation"] and not report["contamination_resistant"]
    )
    for track in report["tracks"].values():
        assert track["n"] == track["emitted"] == 1
        assert track["failed"] == 0
        assert track["critical_burned_in"]["events"] == 0
        assert track["critical_burned_in"]["confidence_interval_95"][1] > 0.5
        assert track["quality"]["rate"] == track["utility"]["rate"] == 1
    assert (
        report["language_slices"]["es"]["n"]
        == report["language_slices"]["fr"]["n"]
        == 1
    )
    assert report["language_slices"]["en"]["n"] == 7
    serialized = json.dumps(report) + caplog.text + capsys.readouterr().out
    for case in cases:
        for marker in case.private_markers:
            assert marker not in serialized
    for value in report["provenance"].values():
        assert len(value) == 64
    assert report == run_openphi_vision(SyntheticVisionControl(), cases, policy)


def test_one_moved_critical_marker_fails_despite_perfect_quality(cases, policy):
    def leak(request, output):
        if request.track is not VisionTrack.IMAGE:
            return output
        with Image.open(io.BytesIO(output.pages[0])) as image:
            image.paste("white", (250, 90, 254, 94))  # Outside original gold boxes.
            return replace(output, pages=(png(image),))

    report = run_openphi_vision(Candidate(leak), cases, policy)
    assert not report["privacy_passed"] and not report["passed"]
    track = report["tracks"][VisionTrack.IMAGE.value]
    assert (
        track["critical_burned_in"]["events"]
        == track["document_any_phi"]["events"]
        == 1
    )
    assert track["quality"]["rate"] == track["utility"]["rate"] == 1


def test_last_page_metadata_leak_is_non_compensable(cases, policy):
    def leak(request, output):
        if request.track is not VisionTrack.CROSS_DOCUMENT:
            return output
        info = PngInfo()
        info.add_text("CopiedIdentifier", request.observations[0])
        with Image.open(io.BytesIO(output.pages[-1])) as image:
            return replace(output, pages=output.pages[:-1] + (png(image, info=info),))

    report = run_openphi_vision(Candidate(leak), cases, policy)
    track = report["tracks"][VisionTrack.CROSS_DOCUMENT.value]
    assert track["document_any_phi"]["events"] == 1
    assert track["critical_burned_in"]["events"] == 0
    assert not report["privacy_passed"]


def test_blackout_can_pass_privacy_but_cannot_hide_utility_loss(cases, policy):
    def blackout(request, output):
        with Image.new("RGB", (256, 96)) as image:
            return replace(output, pages=tuple(png(image) for _ in output.pages))

    report = run_openphi_vision(Candidate(blackout), cases, policy)
    assert report["privacy_passed"] and not report["passed"]
    assert all(not track["utility_passed"] for track in report["tracks"].values())
    assert all(track["quality_passed"] for track in report["tracks"].values())


def test_all_abstentions_are_not_zero_leak_success(cases, policy):
    report = run_openphi_vision(
        Candidate(lambda *_: VisionOutput(abstained=True)),
        cases,
        replace(
            policy,
            maximum_abstention_rate=1,
            minimum_utility_rate=0,
            minimum_quality_rate=0,
        ),
    )
    assert not report["privacy_passed"] and not report["passed"]
    assert all(
        track["critical_burned_in"]["rate"] is None
        for track in report["tracks"].values()
    )
    assert all(track["abstention"]["rate"] == 1 for track in report["tracks"].values())


def test_omitted_tracks_and_duplicate_cases_cannot_change_denominators(cases, policy):
    report = run_openphi_vision(SyntheticVisionControl(), cases[:-1], policy)
    assert not report["passed"]
    assert report["tracks"][VisionTrack.CROSS_DOCUMENT.value]["n"] == 0
    with pytest.raises(VisionGateError, match="invalid_multimodal_gate_input"):
        run_openphi_vision(SyntheticVisionControl(), cases + cases[:1], policy)


def test_missing_page_and_malformed_output_count_as_failures(cases, policy):
    report = run_openphi_vision(
        Candidate(lambda request, output: replace(output, pages=(b"not an image",))),
        cases,
        policy,
    )
    assert not report["passed"]
    assert all(track["failed"] == 1 for track in report["tracks"].values())


def test_routing_quality_remains_independent_of_privacy(cases, policy):
    report = run_openphi_vision(
        Candidate(lambda request, output: replace(output, structural_hold=False)),
        cases,
        policy,
    )
    assert report["privacy_passed"] and not report["passed"]
    assert not report["tracks"][VisionTrack.CROSS_DOCUMENT.value]["quality_passed"]
    assert not report["tracks"][VisionTrack.OCR_CONFLICT.value]["quality_passed"]


def test_changed_license_snapshot_is_bound_without_an_allowlist(cases, policy):
    first = Candidate(lambda request, output: output)
    second = Candidate(lambda request, output: output)
    second.provenance = replace(
        second.provenance,
        model_license_digest=hashlib.sha256(
            b"different caller model terms"
        ).hexdigest(),
    )
    before = run_openphi_vision(first, cases, policy)
    after = run_openphi_vision(second, cases, policy)
    assert before["passed"] and after["passed"]
    assert before["binding_digest"] != after["binding_digest"]


def test_changed_checkpoint_mid_run_is_refused(cases, policy):
    candidate = Candidate(lambda request, output: output)
    original = candidate.process

    def change(request):
        result = original(request)
        candidate.provenance = replace(candidate.provenance, model_digest="e" * 64)
        return result

    candidate.process = change
    report = run_openphi_vision(candidate, cases, policy)
    assert not report["passed"]
    assert all(track["failed"] == 1 for track in report["tracks"].values())


def test_cancellation_never_completes_partial_evidence(cases, policy):
    candidate = Candidate(lambda request, output: output)
    report = run_openphi_vision(
        candidate, cases, policy, cancelled=lambda: candidate.calls == 2
    )
    assert report["interrupted"] and report["executed_cases"] == 2
    assert not report["passed"] and not report["privacy_passed"]


def test_provider_errors_and_bad_initial_declarations_are_value_free(cases, policy):
    private = cases[0].private_markers[0]

    def error(*args):
        raise RuntimeError(private)

    candidate = Candidate(error)
    report = run_openphi_vision(candidate, cases, policy)
    assert not report["passed"] and private not in json.dumps(report)
    candidate.local_only = False
    with pytest.raises(VisionGateError) as caught:
        run_openphi_vision(candidate, cases, policy)
    assert caught.value.__context__ is None
    assert private not in str(caught.value)


class Clock:
    def __init__(self):
        self.wall = 10.0
        self.cpu = 1.0

    def now(self):
        return self.wall

    def cpu_now(self):
        return self.cpu


class Workload:
    def __init__(self, clock):
        from openmed.eval.suites.edge_health import DeviceTier

        self.tier = DeviceTier.CPU
        self.local_only = True
        self.provenance = SyntheticVisionControl().provenance
        self.dataset_digest = "a" * 64
        self.workload_digest = "b" * 64
        self.clock = clock
        self.calls = 0
        self.mutate = lambda result: result

    def run(self):
        from openmed.eval.suites.edge_health import DeviceTrial

        self.calls += 1
        self.clock.wall += 0.01 * self.calls
        self.clock.cpu += 0.002 * self.calls
        return self.mutate(
            DeviceTrial(
                provenance_digest=self.provenance.digest,
                dataset_digest=self.dataset_digest,
                output_digest="c" * 64,
                units=2,
                assessed=2,
                critical_leaks=0,
                any_phi_leaks=0,
                quality_correct=2,
                utility_correct=2,
                abstentions=0,
            )
        )


def device_setup():
    from openmed.eval.suites.edge_health import (
        DeviceProfile,
        DeviceTier,
        ResourceSample,
    )

    clock = Clock()
    workload = Workload(clock)
    profile = DeviceProfile(
        tier=DeviceTier.CPU,
        provenance=workload.provenance,
        dataset_digest=workload.dataset_digest,
        workload_digest=workload.workload_digest,
        reference_output_digest="c" * 64,
        units_per_trial=2,
        trials=3,
        warmups=1,
        maximum_latency_ms=100,
        minimum_throughput=40,
        maximum_rss_bytes=2000,
        maximum_thermal_level=1,
        require_thermal=True,
        minimum_quality_rate=1,
        minimum_utility_rate=1,
        maximum_abstention_rate=0,
    )

    def observe():
        return ResourceSample(
            observed_at=clock.now(),
            rss_bytes=1000,
            thermal_level=0,
            source_digest="d" * 64,
        )

    return profile, workload, clock, observe


def device_run(profile, workload, clock, observe, **kwargs):
    from openmed.eval.suites.edge_health import run_device_profile

    return run_device_profile(
        profile,
        workload,
        observe=observe,
        clock=clock.now,
        cpu_clock=clock.cpu_now,
        **kwargs,
    )


def test_device_executes_workload_and_keeps_warmup_quality_out_of_timing_samples():
    profile, workload, clock, observe = device_setup()
    report = device_run(profile, workload, clock, observe)
    assert report["passed"], report
    assert workload.calls == report["executed"] == 4
    assert report["measured_trials"] == 3
    assert len(report["quality_privacy_trials"]) == 4
    assert report["quality_privacy_trials"][0]["warmup"]
    assert not report["quality_privacy_trials"][1]["warmup"]
    latency = report["measurements"]["latency_ms"]
    assert latency["n"] == 3 and latency["mean"] == pytest.approx(30)
    assert latency["p95"] == pytest.approx(40)
    assert latency["mean_ci95"][0] <= latency["mean"] <= latency["mean_ci95"][1]
    assert report["thermal_measured"]
    assert not report["energy_measured"]
    assert not report["repeated_quality_samples_are_independent"]
    assert report["measurements"]["process_cpu_seconds_proxy"]["mean"] == pytest.approx(
        0.006
    )
    assert report["resource_scope"] == ("d" * 64, "process_lifetime_highwater", None)


def test_leaking_warmup_fails_independently_of_all_warm_performance():
    profile, workload, clock, observe = device_setup()
    workload.mutate = lambda result: (
        replace(result, critical_leaks=1, any_phi_leaks=1)
        if workload.calls == 1
        else result
    )
    report = device_run(profile, workload, clock, observe)
    assert not report["passed"]
    assert report["failures"] == ["critical_or_document_phi_leakage"]
    assert report["measurements"]["latency_ms"]["n"] == 3


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"quality_correct": 1}, "quality"),
        ({"utility_correct": 1}, "utility"),
        ({"assessed": 1, "abstentions": 1}, "abstention"),
        ({"assessed": 0, "abstentions": 2}, "unassessed_or_failed_units"),
    ],
)
def test_device_quality_utility_and_abstention_cannot_be_compensated(change, reason):
    profile, workload, clock, observe = device_setup()
    workload.mutate = lambda result: replace(result, **change)
    report = device_run(profile, workload, clock, observe)
    assert not report["passed"] and reason in report["failures"]


def test_exact_reference_parity_is_distinct_from_repeat_stability():
    profile, workload, clock, observe = device_setup()
    profile = replace(profile, reference_output_digest="e" * 64)
    report = device_run(profile, workload, clock, observe)
    assert report["failures"] == ["reference_parity"]
    assert all(
        row["repeat_parity"] and not row["reference_parity"]
        for row in report["quality_privacy_trials"]
    )


def test_unknown_thermal_never_turns_into_a_nominal_measurement():
    profile, workload, clock, observe = device_setup()
    unknown = lambda: replace(observe(), thermal_level=None)
    refused = device_run(profile, workload, clock, unknown)
    assert not refused["passed"] and workload.calls == 0
    assert "missing_resource_evidence" in refused["failures"]
    optional = device_run(
        replace(profile, require_thermal=False), workload, clock, unknown
    )
    assert optional["passed"]
    assert not optional["thermal_measured"]
    assert optional["measurements"]["observed_thermal_level"]["mean"] is None


@pytest.mark.parametrize(
    "change",
    [
        {"rss_bytes": None},
        {"rss_bytes": 3000},
        {"thermal_level": 3},
        {"observed_at": 1.0},
    ],
)
def test_unsafe_missing_or_stale_pre_samples_do_not_execute(change):
    profile, workload, clock, observe = device_setup()
    report = device_run(profile, workload, clock, lambda: replace(observe(), **change))
    assert not report["passed"] and workload.calls == 0


def test_post_execution_pressure_blocks_success():
    profile, workload, clock, observe = device_setup()
    report = device_run(
        profile,
        workload,
        clock,
        lambda: replace(observe(), thermal_level=3) if workload.calls else observe(),
    )
    assert not report["passed"] and workload.calls == 1
    assert "resource_budget" in report["failures"]


def test_a_cpu_adapter_cannot_satisfy_an_accelerator_tier():
    from openmed.eval.suites.edge_health import DeviceTier

    profile, workload, clock, observe = device_setup()
    report = device_run(
        replace(profile, tier=DeviceTier.MLX, maximum_device_memory_bytes=2000),
        workload,
        clock,
        observe,
    )
    assert not report["passed"] and workload.calls == 0
    assert "provider_or_measurement_failure" in report["failures"]


def test_unavailable_tiers_are_never_presented_as_passes():
    from openmed.eval.suites.edge_health import DeviceTier

    profile, _, clock, observe = device_setup()
    for tier in DeviceTier:
        report = device_run(
            replace(profile, tier=tier, maximum_device_memory_bytes=2000),
            None,
            clock,
            observe,
        )
        assert report["status"] == "not_run" and not report["passed"]
        assert report["executed"] == 0 and report["measurements"] == {}


def test_partial_measurement_or_changed_artifact_refuses_success():
    profile, workload, clock, observe = device_setup()
    report = device_run(
        profile, workload, clock, observe, cancelled=lambda: workload.calls == 2
    )
    assert not report["passed"] and report["measured_trials"] == 1
    assert "cancelled" in report["failures"]
    profile, workload, clock, observe = device_setup()

    def change(result):
        workload.provenance = replace(
            workload.provenance, model_license_digest="f" * 64
        )
        return result

    workload.mutate = change
    report = device_run(profile, workload, clock, observe)
    assert not report["passed"]
    assert "provider_or_measurement_failure" in report["failures"]


def test_actual_cpu_workload_is_bound_to_real_control_outputs(cases, policy):
    from openmed.eval.suites.edge_health import SyntheticCPUWorkload

    workload = SyntheticCPUWorkload(cases, policy)
    trial = workload.run()
    report = run_openphi_vision(SyntheticVisionControl(), cases, policy)
    assert trial.units == trial.assessed == 7
    assert trial.critical_leaks == trial.any_phi_leaks == trial.failed == 0
    assert trial.output_digest == report["output_digest"]
    assert trial.dataset_digest == report["provenance"]["dataset_digest"]
    workload.dataset_digest = "e" * 64
    with pytest.raises(VisionGateError):
        workload.run()


def test_accelerator_profiles_require_separate_allocator_evidence():
    from openmed.eval.suites.edge_health import DeviceTier

    profile, workload, clock, observe = device_setup()
    with pytest.raises(VisionGateError):
        replace(profile, tier=DeviceTier.MLX)
    workload.tier = DeviceTier.MLX
    profile = replace(profile, tier=DeviceTier.MLX, maximum_device_memory_bytes=2000)
    missing = device_run(profile, workload, clock, observe)
    assert not missing["passed"] and workload.calls == 0
    assert "missing_resource_evidence" in missing["failures"]
    measured = lambda: replace(
        observe(), device_memory_bytes=1500, device_memory_scope="allocator_peak"
    )
    passed = device_run(profile, workload, clock, measured)
    assert passed["passed"] and passed["device_memory_measured"]
    assert passed["measurements"]["device_allocator_bytes"]["mean"] == 1500
    over_budget = device_run(
        profile,
        workload,
        clock,
        lambda: replace(measured(), device_memory_bytes=3000),
    )
    assert not over_budget["passed"]
    assert "resource_budget" in over_budget["failures"]


def test_unknown_metadata_is_unassessed_instead_of_zero_leak_evidence(cases, policy):
    def metadata(request, output):
        info = PngInfo()
        info.add_text("Unknown", "unclassified value")
        with Image.open(io.BytesIO(output.pages[0])) as image:
            return replace(output, pages=(png(image, info=info),) + output.pages[1:])

    report = run_openphi_vision(Candidate(metadata), cases, policy)
    assert not report["passed"] and not report["privacy_passed"]
    for track in report["tracks"].values():
        assert track["emitted"] == track["failed"] == 1
        assert track["assessed"] == 0
        assert track["document_any_phi"]["rate"] is None


def test_sensor_identity_changes_and_remote_workloads_fail_before_more_calls():
    profile, workload, clock, observe = device_setup()
    workload.local_only = False
    report = device_run(profile, workload, clock, observe)
    assert not report["passed"] and workload.calls == 0
    workload.local_only = True
    report = device_run(
        profile,
        workload,
        clock,
        lambda: (
            replace(observe(), source_digest="e" * 64) if workload.calls else observe()
        ),
    )
    assert not report["passed"] and workload.calls == 1
    assert "provider_or_measurement_failure" in report["failures"]
