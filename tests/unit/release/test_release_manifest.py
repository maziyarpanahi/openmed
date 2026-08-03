"""Release run-ledger builder regression tests.

Covers the acceptance criteria for the release-manifest builder: every ledger
row binds an artifact digest to a ``RELEASABLE`` gate report by hash, the gate
report hash is recomputed rather than trusted, a non-releasable family is
quarantined with no publish target, the run outcome is reconstructable offline,
provenance hashes are deterministic, and no raw PHI is written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from openmed.core.repro_hash import compute_canonical_payload_hash
from openmed.eval.release_gates import QUARANTINED, RELEASABLE, GateReport

ROOT = Path(__file__).resolve().parents[3]

# The builder lives under scripts/, which is not an importable package, so load
# it by path the same way the other scripts/release tests reach their targets.
_SPEC = importlib.util.spec_from_file_location(
    "release_orchestrate", ROOT / "scripts" / "release" / "orchestrate.py"
)
assert _SPEC and _SPEC.loader
orchestrate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(orchestrate)

FIXED_TS = "2026-07-24T00:00:00+00:00"
FIXED_SHA = "0123456789abcdef0123456789abcdef01234567"


def _digest(seed: str) -> str:
    """Return a well-formed candidate artifact digest for a fixture."""

    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _report(
    family: str,
    *,
    decision: str = RELEASABLE,
    repo_id: str | None = None,
    tier: str = "Tiny",
    fmt: str = "mlx-fp",
) -> GateReport:
    """Build a minimal valid GateReport for a family and decision."""

    return GateReport(
        repo_id=repo_id or f"OpenMed/{family.lower()}-tiny",
        family=family,
        tier=tier,
        param_count=44_000_000,
        format=fmt,
        per_label_recall={"PERSON": 0.995},
        per_label_precision={"PERSON": 0.98},
        critical_leakage_count=0 if decision == RELEASABLE else 3,
        residual_leakage_rate=0.0,
        quant_recall_delta=0.0,
        p50_ms=50.0,
        p95_ms=120.0,
        ram_mb=128.0,
        eval_set_hash="sha256:eval",
        leakage_fixture_hash="sha256:leakage",
        decision=decision,
    )


def _build(
    reports,
    tmp_path,
    *,
    run_id="run-1",
    git_sha=FIXED_SHA,
    pointer_targets=None,
    artifact_digests=None,
):
    # Each family gets a stable per-family digest unless a test pins its own.
    digests = (
        {r.family: _digest(f"artifact::{r.family}") for r in reports}
        if artifact_digests is None
        else artifact_digests
    )
    return orchestrate.build_release_manifest(
        reports,
        run_id=run_id,
        created_at=FIXED_TS,
        git_sha=git_sha,
        artifact_digests=digests,
        pointer_targets=pointer_targets,
        ledger_path=tmp_path / "release_runs.jsonl",
    )


# --- gate binding + fail-closed quarantine -------------------------------------


def test_releasable_family_binds_to_gate_report_by_hash(tmp_path: Path) -> None:
    report = _report("PII")
    (record,) = _build([report], tmp_path)

    assert record["decision"] == RELEASABLE
    assert record["quarantined"] is False
    # The row links to the gate report by hash, and to the artifact by digest.
    assert record["gate_report_hash"] == report.repro_hash
    assert record["artifact_digest"] == _digest("artifact::PII")
    # A releasable family gets a publish target (defaults to its repo id).
    assert record["pointer_target"] == report.repo_id
    assert orchestrate.verify_record(record)


# --- artifact digest binding ---------------------------------------------------


def test_provenance_hash_covers_the_artifact_digest(tmp_path: Path) -> None:
    # Same family, same gate report, different candidate artifact: the two rows
    # must not share a provenance hash, or a row could be re-pointed at another
    # artifact undetected.
    (first,) = _build(
        [_report("PII")], tmp_path / "a", artifact_digests={"PII": _digest("build-1")}
    )
    (second,) = _build(
        [_report("PII")], tmp_path / "b", artifact_digests={"PII": _digest("build-2")}
    )

    assert first["gate_report_hash"] == second["gate_report_hash"]
    assert first["provenance_hash"] != second["provenance_hash"]


def test_family_without_an_artifact_digest_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"

    try:
        _build([_report("PII")], tmp_path, artifact_digests={})
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("unbound family was not rejected")

    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_malformed_artifact_digest_is_refused(tmp_path: Path) -> None:
    try:
        _build([_report("PII")], tmp_path, artifact_digests={"PII": "not-a-digest"})
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("malformed artifact digest was not rejected")


# --- gate report hash is recomputed, not trusted --------------------------------


def test_gate_report_hash_is_recomputed_from_report_contents(tmp_path: Path) -> None:
    report = _report("PII")
    (record,) = _build([report], tmp_path)

    assert record["gate_report_hash"] == report.recompute_repro_hash()


def test_report_with_a_forged_stored_hash_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    report = _report("PII")
    # A report arriving from disk claiming a hash it does not actually have.
    report.repro_hash = _digest("forged")

    try:
        _build([report], tmp_path)
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("forged gate report hash was not detected")

    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_report_mutated_after_hashing_is_refused(tmp_path: Path) -> None:
    report = _report("Clinical", decision=QUARANTINED)
    # Flipping the decision after the report hashed itself must not slip
    # through by riding the stale stored hash.
    report.decision = RELEASABLE

    try:
        _build([report], tmp_path)
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("post-hash mutation was not detected")


def test_non_releasable_family_is_quarantined_with_no_target(tmp_path: Path) -> None:
    report = _report("Clinical", decision=QUARANTINED)
    # Even an explicit target must be stripped for a quarantined family.
    (record,) = _build(
        [report], tmp_path, pointer_targets={"Clinical": "OpenMed/should-not-publish"}
    )

    assert record["decision"] == QUARANTINED
    assert record["quarantined"] is True
    assert record["pointer_target"] is None
    assert record["gate_report_hash"] == report.repro_hash


def test_mixed_batch_publishes_only_releasable_families(tmp_path: Path) -> None:
    reports = [
        _report("PII", decision=RELEASABLE),
        _report("Clinical", decision=QUARANTINED),
        _report("Oncology", decision=RELEASABLE),
    ]
    records = _build(reports, tmp_path)
    by_family = {r["family"]: r for r in records}

    assert by_family["PII"]["pointer_target"] == "OpenMed/pii-tiny"
    assert by_family["Oncology"]["pointer_target"] == "OpenMed/oncology-tiny"
    assert by_family["Clinical"]["pointer_target"] is None
    assert by_family["Clinical"]["quarantined"] is True


# --- reconstruct offline -------------------------------------------------------


def test_full_run_outcome_reconstructable_from_ledger(tmp_path: Path) -> None:
    reports = [
        _report("PII", decision=RELEASABLE),
        _report("Clinical", decision=QUARANTINED),
    ]
    _build(reports, tmp_path, run_id="run-42")

    outcome = orchestrate.reconstruct_run(
        "run-42", ledger_path=tmp_path / "release_runs.jsonl"
    )

    assert outcome["PII"]["published"] is True
    assert outcome["PII"]["pointer_target"] == "OpenMed/pii-tiny"
    assert outcome["Clinical"]["published"] is False
    assert outcome["Clinical"]["pointer_target"] is None


def test_reconstruct_isolates_by_run_id_and_is_append_only(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    _build([_report("PII")], tmp_path, run_id="run-A")
    _build([_report("Oncology")], tmp_path, run_id="run-B")

    # Both runs coexist; neither reconstruction bleeds into the other.
    assert set(orchestrate.reconstruct_run("run-A", ledger_path=ledger)) == {"PII"}
    assert set(orchestrate.reconstruct_run("run-B", ledger_path=ledger)) == {"Oncology"}
    assert len(ledger.read_text(encoding="utf-8").strip().splitlines()) == 2


def test_reconstruct_rejects_a_tampered_row(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    _build([_report("Clinical", decision=QUARANTINED)], tmp_path, run_id="run-x")

    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    # Flip a quarantined family to look published without a matching hash.
    rows[0]["decision"] = RELEASABLE
    rows[0]["quarantined"] = False
    rows[0]["pointer_target"] = "OpenMed/forged"
    ledger.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )

    try:
        orchestrate.reconstruct_run("run-x", ledger_path=ledger)
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("tampered provenance hash was not detected")


def test_reconstruct_rejects_quarantine_only_tampering(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    _build([_report("PII")], tmp_path, run_id="run-x")

    (row,) = [json.loads(line) for line in ledger.read_text().splitlines()]
    row["quarantined"] = True
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(orchestrate.ReleaseManifestError):
        orchestrate.reconstruct_run("run-x", ledger_path=ledger)


# --- determinism ---------------------------------------------------------------


def test_provenance_hash_matches_recomputation(tmp_path: Path) -> None:
    records = _build(
        [_report("PII"), _report("Clinical", decision=QUARANTINED)], tmp_path
    )
    for record in records:
        assert orchestrate.verify_record(record)
        assert record["provenance_hash"].startswith("sha256:")


def test_provenance_hash_matches_the_public_canonical_helper(tmp_path: Path) -> None:
    # The ledger hash is reproducible by anyone holding the row and the public
    # helper -- no private canonicalization needed to audit it.
    (record,) = _build([_report("PII")], tmp_path)
    payload = {field: record[field] for field in orchestrate._BINDING_FIELDS}

    assert record["provenance_hash"] == compute_canonical_payload_hash(payload)


def test_same_inputs_produce_identical_provenance_hash(tmp_path: Path) -> None:
    (first,) = _build([_report("PII")], tmp_path / "a", run_id="run-1")
    (second,) = _build([_report("PII")], tmp_path / "b", run_id="run-1")
    assert first["provenance_hash"] == second["provenance_hash"]


def test_provenance_hash_covers_run_context_and_quarantine_state(
    tmp_path: Path,
) -> None:
    (record,) = _build([_report("PII")], tmp_path)

    mutations = {
        "run_id": "different-run",
        "created_at": "2026-07-25T00:00:00+00:00",
        "quarantined": True,
    }
    for field, value in mutations.items():
        tampered = dict(record)
        tampered[field] = value
        assert not orchestrate.verify_record(tampered), field


def test_invalid_git_sha_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"

    with pytest.raises(orchestrate.ReleaseManifestError):
        _build([_report("PII")], tmp_path, git_sha="patient-123-45-6789")

    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""


def test_unknown_gate_decision_is_refused(tmp_path: Path) -> None:
    with pytest.raises(orchestrate.ReleaseManifestError):
        _build([_report("PII", decision="PENDING")], tmp_path)


def test_duplicate_family_in_one_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(orchestrate.ReleaseManifestError):
        _build([_report("PII"), _report("PII")], tmp_path)


def test_empty_run_is_refused(tmp_path: Path) -> None:
    with pytest.raises(orchestrate.ReleaseManifestError):
        _build([], tmp_path)


def test_duplicate_run_family_append_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    _build([_report("PII")], tmp_path, run_id="run-duplicate")

    with pytest.raises(orchestrate.ReleaseManifestError):
        _build([_report("PII")], tmp_path, run_id="run-duplicate")

    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_non_object_ledger_row_is_refused(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    ledger.write_text("[]\n", encoding="utf-8")

    with pytest.raises(orchestrate.ReleaseManifestError):
        orchestrate.reconstruct_run("run-1", ledger_path=ledger)


# --- no raw PHI ----------------------------------------------------------------


def test_ledger_contains_no_raw_phi(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    _build(
        [_report("PII"), _report("Clinical", decision=QUARANTINED)],
        tmp_path,
        run_id="run-phi",
    )

    text = ledger.read_text(encoding="utf-8")
    for row in (json.loads(line) for line in text.splitlines()):
        for value in row.values():
            if isinstance(value, str):
                for pattern in orchestrate._PHI_PATTERNS:
                    assert not pattern.search(value), (
                        f"PHI-shaped value written to ledger: {value!r}"
                    )


def test_phi_shaped_value_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    ledger = tmp_path / "release_runs.jsonl"
    # A repo id carrying an SSN-shaped token must never reach the ledger.
    poisoned = _report("PII", repo_id="OpenMed/patient-123-45-6789")

    try:
        _build([poisoned], tmp_path)
    except orchestrate.ReleaseManifestError:
        pass
    else:  # pragma: no cover - failure path
        raise AssertionError("PHI-shaped value was not rejected")

    assert not ledger.exists() or ledger.read_text(encoding="utf-8") == ""
