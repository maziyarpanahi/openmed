"""Offline tests for the OS-enforced no-egress gate runner."""

from __future__ import annotations

import errno
import importlib.util
import ipaddress
import json
import socket
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.attestation import (
    NO_EGRESS_NEGATIVE_CONTROLS,
    NO_EGRESS_POSITIVE_CONTROLS,
    NO_EGRESS_WORKFLOW_CASES,
    validate_no_egress_attestation,
)

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "scripts" / "security" / "no_egress_gate.py"
SPEC = importlib.util.spec_from_file_location("no_egress_gate", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)

_FIXTURE = ROOT / "tests" / "fixtures" / "security" / "no_egress_service_workflow.json"
_GENERATED_AT = "2026-11-07T12:00:00Z"


def _inner_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kernel": "linux",
        "interfaces": {"total": 10, "loopback_up": 1, "non_loopback_up": 0},
        "routes": {"ipv4_non_loopback": 0, "ipv6_non_loopback": 0},
        "negative_controls": [
            {"id": case, "status": "blocked", "reason_code": "network_unreachable"}
            for case in NO_EGRESS_NEGATIVE_CONTROLS
        ],
        "positive_controls": [
            {"id": case, "status": "reachable", "reason_code": "loopback_reachable"}
            for case in NO_EGRESS_POSITIVE_CONTROLS
        ],
        "workflow": [
            {
                "id": case,
                "status": "passed",
                "reason_code": "ok",
                "result_digest": None,
            }
            for case in NO_EGRESS_WORKFLOW_CASES
        ],
    }
    result.update(overrides)
    return result


def _run_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    plan: Any,
    outcome: Any = None,
    argv: tuple[str, ...] = (),
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(gate, "select_boundary", lambda *args, **kwargs: plan)

    def fake_run_inner(*args: Any, **kwargs: Any) -> Any:
        if outcome is None:
            raise AssertionError("the runner must not start without a boundary")
        return outcome

    monkeypatch.setattr(gate, "run_inner", fake_run_inner)
    output = tmp_path / "no-egress.json"
    code = gate.main(
        [
            "--output",
            str(output),
            "--generated-at",
            _GENERATED_AT,
            "--commit",
            "0123456789abcdef0123456789abcdef01234567",
            *argv,
        ]
    )
    return code, json.loads(output.read_text(encoding="utf-8"))


# -- classification ---------------------------------------------------------


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (
            gate.ProbeOutcome("error", errno.ENETUNREACH),
            ("blocked", "network_unreachable"),
        ),
        (
            gate.ProbeOutcome("error", errno.EADDRNOTAVAIL),
            ("blocked", "address_unavailable"),
        ),
        (
            gate.ProbeOutcome("error", errno.EAFNOSUPPORT),
            ("blocked", "family_unsupported"),
        ),
        (gate.ProbeOutcome("error", errno.EPERM), ("blocked", "permission_denied")),
        (
            gate.ProbeOutcome("gai_error", socket.EAI_AGAIN),
            ("blocked", "resolver_unreachable"),
        ),
        (gate.ProbeOutcome("connected"), ("not_blocked", "connection_established")),
        (gate.ProbeOutcome("sent"), ("not_blocked", "datagram_sent")),
        (gate.ProbeOutcome("resolved"), ("not_blocked", "name_resolved")),
        (gate.ProbeOutcome("replied"), ("not_blocked", "resolver_answered")),
        (
            gate.ProbeOutcome("gai_error", socket.EAI_NONAME),
            ("not_blocked", "resolver_answered"),
        ),
        (
            gate.ProbeOutcome("error", errno.ECONNREFUSED),
            ("not_blocked", "peer_refused"),
        ),
        (gate.ProbeOutcome("error", errno.ECONNRESET), ("not_blocked", "peer_reset")),
        (
            gate.ProbeOutcome("error", errno.EHOSTUNREACH),
            ("inconclusive", "host_unreachable"),
        ),
        (gate.ProbeOutcome("timeout"), ("inconclusive", "timeout")),
        (gate.ProbeOutcome("error", errno.EINPROGRESS), ("inconclusive", "timeout")),
        (gate.ProbeOutcome("error", None), ("inconclusive", "probe_error")),
        (gate.ProbeOutcome("unavailable"), ("not_run", "executor_unavailable")),
    ],
)
def test_negative_classification_counts_only_no_route_errors_as_blocked(
    outcome, expected
) -> None:
    assert gate.classify_negative(outcome) == expected


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        (gate.ProbeOutcome("connected"), ("reachable", "loopback_reachable")),
        (gate.ProbeOutcome("replied"), ("reachable", "loopback_reachable")),
        (gate.ProbeOutcome("resolved"), ("reachable", "loopback_reachable")),
        (gate.ProbeOutcome("sent"), ("unreachable", "timeout")),
        (
            gate.ProbeOutcome("error", errno.ENETUNREACH),
            ("unreachable", "network_unreachable"),
        ),
        (
            gate.ProbeOutcome("gai_error", socket.EAI_AGAIN),
            ("unreachable", "probe_error"),
        ),
        (gate.ProbeOutcome("unavailable"), ("not_run", "executor_unavailable")),
    ],
)
def test_positive_classification(outcome, expected) -> None:
    assert gate.classify_positive(outcome) == expected


def test_crashing_probe_is_not_run_rather_than_blocked() -> None:
    def explode() -> Any:
        raise RuntimeError("SYNTHETIC-PROBE-FAILURE")

    rows = gate.run_probe_set({"ipv4_tcp_socket": explode}, gate.classify_negative)

    assert rows == [
        {
            "id": "ipv4_tcp_socket",
            "status": "not_run",
            "reason_code": "executor_unavailable",
        }
    ]


# -- probe wiring -----------------------------------------------------------


def test_probe_sets_match_the_attestation_vocabulary() -> None:
    responders = gate.LoopbackResponders()
    try:
        assert list(gate.negative_probes()) == list(NO_EGRESS_NEGATIVE_CONTROLS)
        assert list(gate.positive_probes(responders)) == list(
            NO_EGRESS_POSITIVE_CONTROLS
        )
    finally:
        responders.close()


def test_negative_probes_target_documentation_destinations_only() -> None:
    assert ipaddress.ip_address(gate.PROBE_IPV4) in ipaddress.ip_network("192.0.2.0/24")
    assert ipaddress.ip_address(gate.PROBE_IPV4_RESOLVER) in ipaddress.ip_network(
        "192.0.2.0/24"
    )
    assert ipaddress.ip_address(gate.PROBE_IPV6) in ipaddress.ip_network(
        "2001:db8::/32"
    )
    assert gate.PROBE_DNS_NAME.endswith(".invalid")


def test_dns_query_packet_is_a_single_a_question() -> None:
    packet = gate.dns_query_packet("probe.invalid", 0x1234)

    assert packet[:12] == bytes.fromhex("123401000001000000000000")
    assert packet[12:] == b"\x05probe\x07invalid\x00\x00\x01\x00\x01"


def test_linux_sockaddr_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")

    ipv4 = gate.sockaddr_bytes(socket.AF_INET, "192.0.2.10", 443)
    ipv6 = gate.sockaddr_bytes(socket.AF_INET6, "2001:db8::10", 443)

    assert len(ipv4) == 16
    assert int.from_bytes(ipv4[:2], sys.byteorder) == socket.AF_INET
    assert ipv4[2:4] == (443).to_bytes(2, "big")
    assert ipv4[4:8] == socket.inet_aton("192.0.2.10")
    assert len(ipv6) == 28
    assert ipv6[8:24] == socket.inet_pton(socket.AF_INET6, "2001:db8::10")


def test_bsd_sockaddr_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    ipv4 = gate.sockaddr_bytes(socket.AF_INET, "127.0.0.1", 9)

    assert ipv4[0] == 16
    assert ipv4[1] == socket.AF_INET
    assert len(ipv4) == 16


def test_loopback_probes_reach_local_responders() -> None:
    responders = gate.LoopbackResponders()
    try:
        tcp = gate.tcp_probe("ipv4", "127.0.0.1", responders.tcp_port)
        query = gate.dns_query_packet("probe.invalid", 7)
        udp = gate.udp_probe(
            "ipv4", "127.0.0.1", responders.udp_port, query, expect_reply=True
        )
        child = gate.subprocess_probe("tcp", "ipv4", "127.0.0.1", responders.tcp_port)
    finally:
        responders.close()

    assert gate.classify_positive(tcp) == ("reachable", "loopback_reachable")
    assert gate.classify_positive(udp) == ("reachable", "loopback_reachable")
    assert gate.classify_positive(child) == ("reachable", "loopback_reachable")


@pytest.mark.skipif(sys.platform == "win32", reason="the native probe calls libc")
def test_native_probe_reaches_loopback_through_libc() -> None:
    responders = gate.LoopbackResponders()
    try:
        outcome = gate.native_probe("tcp", "ipv4", "127.0.0.1", responders.tcp_port)
    finally:
        responders.close()

    assert outcome == gate.ProbeOutcome("connected")


def test_reachable_peer_is_never_counted_as_blocked() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        closed_port = sock.getsockname()[1]

    outcome = gate.tcp_probe("ipv4", "127.0.0.1", closed_port)

    assert gate.classify_negative(outcome)[0] != "blocked"


def test_route_counting_fails_closed(tmp_path: Path) -> None:
    ipv4 = tmp_path / "route"
    ipv4.write_text(
        "Iface\tDestination\tGateway\nlo\t0000007F\t00000000\n"
        "eth0\t00000000\t0100A8C0\n",
        encoding="ascii",
    )
    ipv6 = tmp_path / "ipv6_route"
    ipv6.write_text(
        "00000000000000000000000000000001 80 0 00 0 00 0 00000000 1 0 1 lo\n",
        encoding="ascii",
    )

    assert gate._count_routes(str(ipv4), 0, {"lo"}) == 1
    assert gate._count_routes(str(ipv6), -1, {"lo"}) == 0
    assert gate._count_routes(str(tmp_path / "missing"), 0, {"lo"}) == 0
    assert gate._count_routes(str(tmp_path), 0, {"lo"}) == 1


# -- fixture and synthetic loader -------------------------------------------


def test_fixture_is_synthetic_and_drives_the_synthetic_loader() -> None:
    fixture = gate.load_fixture(_FIXTURE)
    loader = gate.synthetic_loader_class(fixture.lexicon)(config=None)
    pipeline = loader.create_pipeline(fixture.model_name)

    rows = [
        [item["entity_group"], item["start"], item["end"]]
        for item in pipeline(fixture.document)
    ]
    assert rows == fixture.expected_entities
    assert set(fixture.phi_terms) == {"Jane Example", "555-0100", "MRN-000123"}
    assert loader.get_max_sequence_length(fixture.model_name) == 512
    assert pipeline([fixture.document, "nothing here"])[1] == []


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        json.dumps({"fixture_version": 2}),
        json.dumps({"fixture_version": 1, "lexicon": []}),
        json.dumps(
            {
                "fixture_version": 1,
                "model_name": "synthetic/model",
                "lexicon": [{"term": "absent", "label": "NAME", "phi": True}],
                "document": "no match",
                "stream_repeat": 4,
                "stream_chunk_size": 8,
                "journey_recorded_at": "2026-01-01T00:00:00Z",
            }
        ),
    ],
)
def test_malformed_fixtures_are_rejected(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "fixture.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(gate.FixtureError):
        gate.load_fixture(path)


def test_service_env_is_offline_and_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("OPENMED_EGRESS_RULES", "[]")
    monkeypatch.setenv("HF_TOKEN", "synthetic-token")

    env = gate._service_env(gate.load_fixture(_FIXTURE))

    assert "HTTPS_PROXY" not in env
    assert "OPENMED_EGRESS_RULES" not in env
    assert "HF_TOKEN" not in env
    assert env["OPENMED_OFFLINE"] == "1"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["OPENMED_PROFILE"] == "prod"
    assert env["OPENMED_SERVICE_SERVED_MODELS"] == "synthetic/no-egress-fixture"


# -- boundary selection -----------------------------------------------------


class _Runner:
    def __init__(self, results: dict[str, int]) -> None:
        self.results = results
        self.calls: list[list[str]] = []

    def __call__(self, command: Any, timeout: float) -> int:
        self.calls.append(list(command))
        joined = " ".join(command)
        for needle, code in self.results.items():
            if needle in joined:
                return code
        return 0


def _which(available: set[str]):
    return lambda name: f"/usr/bin/{name}" if name in available else None


@pytest.mark.parametrize("requested", ["auto", "netns"])
@pytest.mark.parametrize("host", ["darwin", "windows", "other"])
def test_non_linux_hosts_without_a_container_report_unsupported(
    monkeypatch: pytest.MonkeyPatch, requested: str, host: str
) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which({"unshare", "docker"}))
    runner = _Runner({})

    plan = gate.select_boundary(requested, host, None, runner=runner)

    assert plan == gate.BoundaryPlan("none", skip_reason="unsupported_host")
    assert not any("unshare" in " ".join(call) for call in runner.calls)


def test_linux_without_unshare_reports_boundary_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which(set()))

    plan = gate.select_boundary("netns", "linux", None, runner=_Runner({}))

    assert plan == gate.BoundaryPlan("none", skip_reason="boundary_unavailable")


def test_restricted_user_namespaces_report_boundary_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which({"unshare"}))
    runner = _Runner({"--map-root-user": 1})

    plan = gate.select_boundary("auto", "linux", None, runner=runner)

    assert plan.skip_reason == "boundary_unavailable"
    assert runner.calls[0][1:4] == ["--user", "--map-root-user", "--net"]


def test_linux_prefers_a_network_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which({"unshare", "docker"}))

    plan = gate.select_boundary("auto", "linux", "image:local", runner=_Runner({}))

    assert plan == gate.BoundaryPlan("netns", executable="/usr/bin/unshare")


def test_auto_falls_back_to_a_local_container_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which({"unshare", "docker"}))
    runner = _Runner({"--map-root-user": 1})

    plan = gate.select_boundary("auto", "linux", "image:local", runner=runner)

    assert plan == gate.BoundaryPlan(
        "container", executable="/usr/bin/docker", image="image:local"
    )


@pytest.mark.parametrize(
    ("available", "results", "image"),
    [
        (set(), {}, "image:local"),
        ({"docker"}, {}, None),
        ({"docker"}, {" info": 1}, "image:local"),
        ({"docker"}, {"image inspect": 1}, "image:missing"),
    ],
)
def test_container_mode_never_pulls_and_skips_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    available: set[str],
    results: dict[str, int],
    image: str | None,
) -> None:
    monkeypatch.setattr(gate.shutil, "which", _which(available))
    runner = _Runner(results)

    plan = gate.select_boundary("container", "darwin", image, runner=runner)

    assert plan == gate.BoundaryPlan("none", skip_reason="boundary_unavailable")
    assert not any("pull" in call for call in runner.calls)


def test_inner_commands_enforce_the_boundary() -> None:
    netns = gate.inner_command(
        gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        _FIXTURE,
        verbose=False,
        container_name="openmed-no-egress-test",
    )
    container = gate.inner_command(
        gate.BoundaryPlan("container", executable="/usr/bin/docker", image="img"),
        _FIXTURE,
        verbose=False,
        container_name="openmed-no-egress-test",
    )

    assert netns[:5] == ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--"]
    assert "--inside" in netns and netns[netns.index("--boundary-kind") + 1] == "netns"
    joined = " ".join(container)
    assert container[container.index("--network") + 1] == "none"
    assert container[container.index("--pull") + 1] == "never"
    assert container[container.index("--cap-drop") + 1] == "ALL"
    assert "--read-only" in container
    assert "readonly" in container[container.index("--mount") + 1]
    assert "host" not in container and "bridge" not in joined
    with pytest.raises(ValueError):
        gate.inner_command(
            gate.BoundaryPlan("none", skip_reason="unsupported_host"),
            _FIXTURE,
            verbose=False,
            container_name="x",
        )


def test_boundary_command_wraps_other_workloads_with_the_same_hardening(
    tmp_path: Path,
) -> None:
    script = ROOT / "scripts" / "airgap" / "rehearse_inside.py"
    bundle = tmp_path / "kit"
    container = gate.boundary_command(
        gate.BoundaryPlan("container", executable="/usr/bin/docker", image="img"),
        script,
        ["--bundle", "/airgap/bundle"],
        container_name="openmed-test",
        mounts=[(bundle, "/airgap/bundle")],
        tmpfs_size="2g",
    )
    netns = gate.boundary_command(
        gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        script,
        ["--bundle", str(bundle)],
        container_name="openmed-test",
    )

    mounts = [container[i + 1] for i, arg in enumerate(container) if arg == "--mount"]
    assert mounts == [
        f"type=bind,src={gate.ROOT},dst={gate.CONTAINER_WORKSPACE},readonly",
        f"type=bind,src={bundle},dst=/airgap/bundle,readonly",
    ]
    assert container[container.index("--network") + 1] == "none"
    assert container[container.index("--tmpfs") + 1] == "/tmp:rw,nosuid,nodev,size=2g"
    assert container[container.index("img") + 2] == (
        "/workspace/scripts/airgap/rehearse_inside.py"
    )
    assert netns[:5] == ["/usr/bin/unshare", "--user", "--map-root-user", "--net", "--"]
    assert netns[5:] == [sys.executable, str(script), "--bundle", str(bundle)]
    with pytest.raises(ValueError, match="boundary_script_invalid"):
        gate.boundary_command(
            gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
            tmp_path / "outside.py",
            [],
            container_name="x",
        )


def test_boundary_evidence_probes_before_any_workload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class _Responders:
        def close(self) -> None:
            calls.append("close")

    monkeypatch.setattr(gate, "bring_loopback_up", lambda: calls.append("lo") or True)
    monkeypatch.setattr(
        gate,
        "inspect_boundary",
        lambda: (calls.append("inspect") or {"total": 1}, {"ipv4_non_loopback": 0}),
    )
    monkeypatch.setattr(gate, "LoopbackResponders", _Responders)
    monkeypatch.setattr(gate, "positive_probes", lambda responders: {"p": None})
    monkeypatch.setattr(gate, "negative_probes", lambda: {"n": None})

    def _run(probes: dict[str, Any], classify: Any) -> list[dict[str, str]]:
        calls.append("positive" if "p" in probes else "negative")
        return [{"id": next(iter(probes))}]

    monkeypatch.setattr(gate, "run_probe_set", _run)

    netns = gate.collect_boundary_evidence("netns")
    calls_netns = list(calls)
    calls.clear()
    gate.collect_boundary_evidence("container")

    assert calls_netns == ["lo", "inspect", "positive", "close", "negative"]
    assert calls == ["inspect", "positive", "close", "negative"]
    assert set(netns) == {
        "kernel",
        "interfaces",
        "routes",
        "negative_controls",
        "positive_controls",
    }


def test_run_in_boundary_scrubs_proxies_and_cleans_up_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def _completed(command: Any, **kwargs: Any) -> Any:
        seen["env"] = kwargs["env"]
        return gate.subprocess.CompletedProcess(command, 0, stdout="out")

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setattr(gate.subprocess, "run", _completed)
    plan = gate.BoundaryPlan("container", executable="/usr/bin/docker", image="img")

    run = gate.run_in_boundary(
        plan, ["docker", "run"], container_name="c", verbose=False, timeout=5
    )

    assert run == gate.BoundaryRun("out", 0)
    assert "HTTPS_PROXY" not in seen["env"]

    def _timeout(command: Any, **kwargs: Any) -> Any:
        raise gate.subprocess.TimeoutExpired(command, 5)

    removed: list[list[str]] = []
    monkeypatch.setattr(gate.subprocess, "run", _timeout)
    monkeypatch.setattr(
        gate, "run_quiet", lambda command, timeout: removed.append(command)
    )

    run = gate.run_in_boundary(
        plan, ["docker", "run"], container_name="c", verbose=False, timeout=5
    )

    assert run.failure == "runner_timeout"
    assert removed == [["/usr/bin/docker", "rm", "--force", "c"]]


# -- inner output parsing ---------------------------------------------------


def test_inner_output_parsing_accepts_a_declared_result_shape() -> None:
    marker = gate.RESULT_MARKER
    keys = frozenset({"kernel", "rehearsal"})

    good = gate.parse_inner_output(
        marker + json.dumps({"kernel": "linux", "rehearsal": {}}), 0, required=keys
    )
    service_shape = gate.parse_inner_output(
        marker + json.dumps(_inner_result()), 0, required=keys
    )

    assert good.result == {"kernel": "linux", "rehearsal": {}}
    assert service_shape.failure == "runner_result_invalid"


def test_inner_output_parsing() -> None:
    marker = gate.RESULT_MARKER
    good = marker + json.dumps(_inner_result())

    assert gate.parse_inner_output("noise\n" + good + "\n", 0).result is not None
    assert gate.parse_inner_output("noise only", 0).failure == "runner_failed"
    assert gate.parse_inner_output(marker + "{", 0).failure == "runner_result_invalid"
    assert gate.parse_inner_output(good, 1).failure == "runner_result_invalid"
    extra = marker + json.dumps(_inner_result(destination="example.org"))
    assert gate.parse_inner_output(extra, 0).failure == "runner_result_invalid"


# -- end-to-end gate decisions (runner mocked) ------------------------------


@pytest.mark.parametrize(
    ("host", "reason"),
    [("darwin", "unsupported_host"), ("linux", "boundary_unavailable")],
)
def test_unsupported_hosts_exit_three_and_are_never_a_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, host: str, reason: str
) -> None:
    monkeypatch.setattr(gate, "host_kind", lambda: host)

    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("none", skip_reason=reason),
    )

    assert code == gate.EXIT_SKIPPED == 3
    assert payload["status"] == "skipped"
    assert payload["reason_code"] == reason
    assert payload["host"] == {"kind": host}
    assert payload["counts"]["negative_controls"]["total"] == 0
    assert validate_no_egress_attestation(payload).exit_code == 3


def test_blocked_controls_and_passing_workflow_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        outcome=gate.InnerOutcome(_inner_result(), None),
    )

    assert code == 0
    assert payload["status"] == "passed"
    assert payload["commit"] == "0123456789abcdef0123456789abcdef01234567"
    assert {item["id"] for item in payload["fixtures"]} == {
        "service_workflow_fixture",
        "no_egress_gate_runner",
    }
    assert validate_no_egress_attestation(payload).passed


def test_a_single_egressing_probe_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _inner_result()
    result["negative_controls"][-1] = {
        "id": "native_ipv4_udp",
        "status": "not_blocked",
        "reason_code": "datagram_sent",
    }

    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("container", executable="/usr/bin/docker", image="i"),
        outcome=gate.InnerOutcome(result, None),
    )

    assert code == 1
    assert payload["reason_code"] == "egress_not_blocked"


def test_visible_network_interface_fails_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        outcome=gate.InnerOutcome(
            _inner_result(
                interfaces={"total": 2, "loopback_up": 1, "non_loopback_up": 1}
            ),
            None,
        ),
    )

    assert code == 1
    assert payload["reason_code"] == "boundary_not_isolated"
    assert payload["boundary"]["isolated"] is False


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        (gate.InnerOutcome(None, "runner_timeout"), "runner_timeout"),
        (gate.InnerOutcome(None, "runner_failed"), "runner_failed"),
        (
            gate.InnerOutcome(
                _inner_result(
                    negative_controls=[
                        {
                            "id": "tcp_to_example_org",
                            "status": "blocked",
                            "reason_code": "ok",
                        }
                    ]
                ),
                None,
            ),
            "runner_result_invalid",
        ),
        (gate.InnerOutcome(_inner_result(kernel=7), None), "runner_result_invalid"),
    ],
)
def test_runner_problems_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: Any, reason: str
) -> None:
    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        outcome=outcome,
    )

    assert code == 1
    assert payload["status"] == "failed"
    assert payload["reason_code"] == reason
    assert "example_org" not in json.dumps(payload)


def test_markdown_output_and_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    markdown = tmp_path / "no-egress.md"
    code, payload = _run_gate(
        monkeypatch,
        tmp_path,
        plan=gate.BoundaryPlan("netns", executable="/usr/bin/unshare"),
        outcome=gate.InnerOutcome(_inner_result(), None),
        argv=("--markdown", str(markdown)),
    )
    saved = tmp_path / "no-egress.json"

    assert code == 0
    assert "`all_controls_blocked`" in markdown.read_text(encoding="utf-8")
    assert gate.main(["--verify", str(saved)]) == 0

    payload["status"] = "failed"
    saved.write_text(json.dumps(payload), encoding="utf-8")
    assert gate.main(["--verify", str(saved)]) == 1
    assert gate.main(["--verify", str(tmp_path / "missing.json")]) == 1


def test_unreadable_fixture_is_a_usage_error(tmp_path: Path) -> None:
    assert gate.main(["--fixture", str(tmp_path / "missing.json")]) == 2
