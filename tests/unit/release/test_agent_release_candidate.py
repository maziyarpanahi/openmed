"""Synthetic offline controls for exact-source v3.1 candidate decisions."""

from __future__ import annotations

import hashlib
import hmac
import json
import subprocess
from types import SimpleNamespace

import pytest

from openmed.eval.agent_release_candidate import (
    AgentCandidateError,
    AgentCandidatePacket,
    candidate_from_arguments,
    run_agent_release_candidate,
)
from tests.fixtures.eval.agent_release_candidate import (
    _KEY,
    _git,
    build_candidate_inputs,
)


@pytest.fixture
def candidate_inputs(tmp_path):
    return build_candidate_inputs(tmp_path)


def _rewrite(inputs):
    inputs.evidence.write_text(json.dumps(inputs.envelope))


def test_repeated_candidate_is_canonical_and_verifies_offline(candidate_inputs):
    args = candidate_inputs.params
    first = run_agent_release_candidate(**args)
    second = run_agent_release_candidate(**args)
    assert first.to_dict() == second.to_dict()
    assert first.manifest_digest == second.manifest_digest
    assert first.packet.packet_digest == second.packet.packet_digest
    assert first.packet.decision == "READY"
    assert first.packet.verify(_KEY)
    assert not first.packet.verify(b"different-synthetic-key-000000000")
    assert first.packet.to_dict()["manifest_digest"] == first.manifest_digest
    assert first.manifest["repository"]["git_sha"] == candidate_inputs.sha
    assert first.manifest["repository"]["tree_status"] == "clean"
    assert all(
        value.startswith("sha256:") for value in first.manifest["artifacts"].values()
    )
    for name, value in first.manifest["artifacts"].items():
        expected = "sha256:" + hashlib.sha256(args[name].read_bytes()).hexdigest()
        assert value == expected
    payload = first.packet.to_dict()
    signature = payload.pop("signature")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert (
        signature == "sha256:" + hmac.new(_KEY, canonical, hashlib.sha256).hexdigest()
    )
    assert first.manifest["evidence"][0]["metric_count"] == 9
    # Returned dictionaries cannot mutate signed state.
    first.manifest["artifacts"]["wheel"] = "tampered"
    first.packet.to_dict()["decision"] = "NOT_READY"
    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("kind", ["tracked", "untracked"])
def test_dirty_tree_fails_with_stable_reason(candidate_inputs, kind):
    name = "source.txt" if kind == "tracked" else "new-source.txt"
    (candidate_inputs.root / name).write_text("synthetic mutation")
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert result.packet.decision == "NOT_READY"
    assert result.packet.to_dict()["reason_codes"] == ["dirty_tree"]
    assert result.packet.verify(_KEY)


def test_mismatched_sha_fails_closed(candidate_inputs):
    result = run_agent_release_candidate(
        **{**candidate_inputs.params, "source_sha": "f" * 40}
    )
    assert result.packet.decision == "NOT_READY"
    assert "sha_mismatch" in result.packet.to_dict()["reason_codes"]
    assert "evidence_sha_mismatch" in result.packet.to_dict()["reason_codes"]


def test_existing_tag_must_point_to_exact_source(candidate_inputs):
    inputs = candidate_inputs
    tags = inputs.root / ".git" / "refs" / "tags"
    tags.mkdir(exist_ok=True)
    (tags / "v3.1.0-rc1").write_text(inputs.sha + "\n")
    matching = run_agent_release_candidate(**inputs.params, tag="v3.1.0-rc1")
    assert matching.packet.decision == "READY"
    (inputs.root / "source.txt").write_text("next synthetic source")
    _git(inputs.root, "add", "source.txt")
    _git(inputs.root, "commit", "--no-gpg-sign", "-m", "Update synthetic source")
    next_sha = _git(inputs.root, "rev-parse", "HEAD")
    inputs.envelope["source_sha"] = next_sha
    _rewrite(inputs)
    mismatched = run_agent_release_candidate(
        **{**inputs.params, "source_sha": next_sha}, tag="v3.1.0-rc1"
    )
    assert mismatched.packet.decision == "NOT_READY"
    assert mismatched.packet.to_dict()["reason_codes"] == ["tag_mismatch"]
    missing = run_agent_release_candidate(
        **{**inputs.params, "source_sha": next_sha}, tag="missing"
    )
    assert missing.packet.to_dict()["reason_codes"] == ["tag_unavailable"]


def test_wrong_source_evidence_cannot_clear_gates(candidate_inputs):
    candidate_inputs.envelope["source_sha"] = "f" * 40
    _rewrite(candidate_inputs)
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert result.packet.decision == "NOT_READY"
    assert "evidence_sha_mismatch" in result.packet.to_dict()["reason_codes"]
    assert all(
        row["reason_code"] == "missing_evidence"
        for row in result.packet.to_dict()["gates"]
    )
    assert result.packet.verify(_KEY)


@pytest.mark.parametrize("name", ["wheel", "sdist", "tool_catalog", "policy"])
def test_missing_artifact_blocks_passing_gates(candidate_inputs, name):
    candidate_inputs.params[name].unlink()
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert result.packet.decision == "NOT_READY"
    assert result.packet.to_dict()["reason_codes"] == [name + "_missing"]
    assert result.manifest["artifacts"][name] is None


def test_missing_evidence_and_missing_metric_fail_closed(candidate_inputs):
    empty = run_agent_release_candidate(
        **{**candidate_inputs.params, "evidence_files": []}
    )
    assert "evidence_missing" in empty.packet.to_dict()["reason_codes"]
    candidate_inputs.envelope["metrics"].pop()
    _rewrite(candidate_inputs)
    missing_metric = run_agent_release_candidate(**candidate_inputs.params)
    assert "gate_missing_evidence" in missing_metric.packet.to_dict()["reason_codes"]
    candidate_inputs.evidence.unlink()
    absent_file = run_agent_release_candidate(**candidate_inputs.params)
    assert "evidence_missing" in absent_file.packet.to_dict()["reason_codes"]


@pytest.mark.parametrize(
    "mutation",
    ["extra", "slice_extra", "unknown", "duplicate", "basis", "schema", "sha"],
)
def test_invalid_or_duplicate_evidence_never_passes(candidate_inputs, mutation):
    record = candidate_inputs.envelope["metrics"][0]
    if mutation == "extra":
        record["raw_patient"] = "synthetic secret"
    elif mutation == "slice_extra":
        record["slices"][0]["raw_patient"] = "synthetic secret"
    elif mutation == "unknown":
        record["metric"] = "unknown_metric"
    elif mutation == "duplicate":
        candidate_inputs.envelope["metrics"].append(record)
    elif mutation == "basis":
        record["event_count"] = None
    elif mutation == "schema":
        candidate_inputs.envelope["schema_version"] = "unknown"
    else:
        candidate_inputs.envelope["source_sha"] = "invalid"
    _rewrite(candidate_inputs)
    result = run_agent_release_candidate(**candidate_inputs.params)
    expected = (
        "evidence_duplicate_metric" if mutation == "duplicate" else "evidence_invalid"
    )
    assert expected in result.packet.to_dict()["reason_codes"]
    assert result.packet.decision == "NOT_READY"


@pytest.mark.parametrize(
    "raw",
    [b'{"schema_version": 1, "schema_version": 2}', b'{"raw": NaN}', b"\xff", b"[]"],
)
def test_malformed_json_produces_value_free_failure(candidate_inputs, raw):
    candidate_inputs.evidence.write_bytes(raw)
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert "evidence_invalid" in result.packet.to_dict()["reason_codes"]
    assert result.packet.verify(_KEY)


def test_threshold_failure_is_signed_and_non_compensable(candidate_inputs):
    record = candidate_inputs.envelope["metrics"][0]
    record["event_count"] = 1
    record["value"] = 1 / record["sample_size"]
    _rewrite(candidate_inputs)
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert result.packet.to_dict()["reason_codes"] == ["gate_failed"]
    assert result.packet.to_dict()["gates"][0]["event_count"] == 1
    assert result.packet.to_dict()["gates"][0]["reason_code"] == "threshold_failed"
    assert result.packet.verify(_KEY)


def test_evidence_file_order_is_irrelevant_and_duplicate_files_fail(candidate_inputs):
    inputs = candidate_inputs
    second = inputs.tmp / "second.json"
    second.write_text(
        json.dumps({**inputs.envelope, "metrics": inputs.envelope["metrics"][4:]})
    )
    inputs.envelope["metrics"] = inputs.envelope["metrics"][:4]
    _rewrite(inputs)
    a = run_agent_release_candidate(
        **{**inputs.params, "evidence_files": [inputs.evidence, second]}
    )
    b = run_agent_release_candidate(
        **{**inputs.params, "evidence_files": [second, inputs.evidence]}
    )
    assert a.to_dict() == b.to_dict()
    assert a.packet.decision == "READY"
    duplicate = run_agent_release_candidate(
        **{**inputs.params, "evidence_files": [inputs.evidence, second, second]}
    )
    assert "evidence_duplicate_metric" in duplicate.packet.to_dict()["reason_codes"]


def test_all_input_and_decision_fields_are_digest_bound(candidate_inputs):
    original = run_agent_release_candidate(**candidate_inputs.params)
    candidate_inputs.params["policy"].write_bytes(b"different policy")
    changed = run_agent_release_candidate(**candidate_inputs.params)
    assert original.manifest_digest != changed.manifest_digest
    assert original.packet.packet_digest != changed.packet.packet_digest
    for field in ("manifest_digest", "report_digest", "packet_digest", "signature"):
        payload = original.packet.to_dict()
        payload[field] = "sha256:" + "f" * 64
        assert not AgentCandidatePacket.from_dict(payload).verify(_KEY)
    payload = original.packet.to_dict()
    payload["gates"][0]["sample_size"] += 1
    assert not AgentCandidatePacket.from_dict(payload).verify(_KEY)
    payload["extra"] = "synthetic PHI"
    with pytest.raises(AgentCandidateError, match="packet_invalid"):
        AgentCandidatePacket.from_dict(payload)


def test_exports_contain_no_payloads_paths_keys_or_arbitrary_refs(candidate_inputs):
    inputs = candidate_inputs
    probes = ["JaneDoeSynthetic", "محمد_مصطنع", "सिंथेटिक_नाम", "private/path/credential"]
    # Catalog/policy content is never parsed or echoed; it is only hashed.
    inputs.params["tool_catalog"].write_text("\n".join(probes))
    for metric in inputs.envelope["metrics"]:
        metric["limitations"] = ["synthetic_patient_identifier"]
        for index, row in enumerate(metric["slices"]):
            row["slice_ref"] = f"synthetic_patient_identifier_{index}"
    _rewrite(inputs)
    result = run_agent_release_candidate(**inputs.params)
    encoded = json.dumps(result.to_dict())
    for secret in [
        *probes,
        "synthetic_patient_identifier",
        str(inputs.tmp),
        _KEY.decode(),
    ]:
        assert secret not in encoded
    for row in result.packet.to_dict()["gates"]:
        assert "value" not in row
        assert "ci_lower" not in row
        assert "limitations" not in row
        assert "slice_ref" not in row
    assert result.packet.verify(_KEY)


def test_repository_unavailable_and_concurrent_change_fail_closed(
    candidate_inputs, monkeypatch
):
    import openmed.eval.agent_release_candidate as module

    unavailable = run_agent_release_candidate(
        **{**candidate_inputs.params, "repo_root": candidate_inputs.tmp / "absent"}
    )
    assert "repository_unavailable" in unavailable.packet.to_dict()["reason_codes"]
    original = module._read_file

    def mutate(path):
        data = original(path)
        (candidate_inputs.root / "source.txt").write_text(
            "concurrent synthetic mutation"
        )
        return data

    monkeypatch.setattr(module, "_read_file", mutate)
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert "repository_changed" in result.packet.to_dict()["reason_codes"]
    assert "dirty_tree" in result.packet.to_dict()["reason_codes"]


def test_configuration_and_exclusive_output(candidate_inputs):
    inputs = candidate_inputs
    with pytest.raises(AgentCandidateError, match="signing_key_invalid"):
        run_agent_release_candidate(**{**inputs.params, "signing_key": b"short"})
    with pytest.raises(AgentCandidateError, match="source_sha_invalid"):
        run_agent_release_candidate(**{**inputs.params, "source_sha": "HEAD"})
    key_file = inputs.tmp / "key"
    key_file.write_bytes(_KEY)
    output = inputs.tmp / "bundle.json"
    args = SimpleNamespace(
        **{
            key: value
            for key, value in inputs.params.items()
            if key not in {"evidence_files", "signing_key"}
        },
        evidence=[inputs.evidence],
        signing_key_file=key_file,
        output=output,
        tag=None,
    )
    result = candidate_from_arguments(args)
    assert json.loads(output.read_text()) == result.to_dict()
    with pytest.raises(AgentCandidateError, match="output_unavailable"):
        candidate_from_arguments(args)
    output.unlink()
    output.symlink_to(inputs.params["policy"])
    with pytest.raises(AgentCandidateError, match="output_unavailable"):
        candidate_from_arguments(args)


def test_git_file_checkout_and_annotated_tag_are_supported(candidate_inputs):
    inputs = candidate_inputs
    git_dir = inputs.tmp / "detached-git-dir"
    (inputs.root / ".git").rename(git_dir)
    (inputs.root / ".git").write_text(f"gitdir: {git_dir}\n")
    assert run_agent_release_candidate(**inputs.params).packet.decision == "READY"
    # Create only a synthetic annotated-tag object and reference in the fixture.
    annotation = (
        f"object {inputs.sha}\ntype commit\ntag synthetic-rc\n"
        "tagger Release Fixture <release@example.invalid> 0 +0000\n\n"
        "synthetic tag\n"
    )
    completed = subprocess.run(
        ["git", "hash-object", "-t", "tag", "-w", "--stdin"],
        cwd=inputs.root,
        input=annotation,
        text=True,
        capture_output=True,
        check=True,
    )
    (git_dir / "refs" / "tags" / "synthetic-rc").write_text(completed.stdout)
    tagged = run_agent_release_candidate(**inputs.params, tag="synthetic-rc")
    assert tagged.packet.decision == "READY"
    assert tagged.manifest["repository"]["tag_sha"] == inputs.sha


def test_runner_never_calls_network_or_dispatch(candidate_inputs, monkeypatch):
    import socket

    import openmed.eval.agent_release_candidate as module

    original = module.subprocess.run
    commands = []

    def local_git_only(arguments, **kwargs):
        assert arguments[0] == "git"
        assert arguments[1] in {"rev-parse", "status"}
        commands.append(arguments)
        return original(arguments, **kwargs)

    def denied(*args, **kwargs):
        raise AssertionError("network operation")

    monkeypatch.setattr(module.subprocess, "run", local_git_only)
    monkeypatch.setattr(socket, "socket", denied)
    result = run_agent_release_candidate(**candidate_inputs.params)
    assert result.packet.decision == "READY"
    assert commands


def test_tag_argument_cannot_be_a_revision_expression(candidate_inputs):
    tags = candidate_inputs.root / ".git" / "refs" / "tags"
    (tags / "synthetic-rc").write_text(candidate_inputs.sha + "\n")
    result = run_agent_release_candidate(
        **candidate_inputs.params, tag="synthetic-rc~0"
    )
    assert result.packet.decision == "NOT_READY"
    assert result.packet.to_dict()["reason_codes"] == ["tag_unavailable"]
