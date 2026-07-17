"""Tests for the evaluation provenance and reproducibility ledger."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmed.eval.provenance_ledger import (
    EvalProvenanceLedger,
    LedgerVerificationError,
    UnsafeResultMetadataError,
    append_eval_result,
    compute_result_hash,
    load_ledger,
    main,
    verify_ledger,
)


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _append(
    path: Path,
    *,
    seed: int = 7,
    f1: float = 0.98,
    fixture_count: int = 20,
):
    return append_eval_result(
        path,
        model_digest=_digest("a"),
        suite_id="synthetic-pii",
        suite_version="2.1.0",
        code_hash="abc1234",
        seed=seed,
        config_digest=_digest("b"),
        env_fingerprint=_digest("c"),
        result_metadata={
            "fixture_count": fixture_count,
            "metrics": {"micro_f1": f1, "recall": 0.99},
            "passed": True,
        },
    )


def test_append_builds_a_valid_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"

    first = _append(path, seed=7)
    second = _append(path, seed=8)
    ledger = load_ledger(path)

    assert ledger.record_count == 2
    assert ledger.records == (first, second)
    assert first.sequence == 0
    assert second.sequence == 1
    assert second.previous_hash == first.record_hash
    assert ledger.head_hash == second.record_hash
    assert ledger.verify() is True
    assert verify_ledger(path) is True


def test_tampering_with_a_result_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["result_metadata"]["metrics"]["micro_f1"] = 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert verify_ledger(path) is False
    with pytest.raises(LedgerVerificationError, match="invalid result_hash"):
        verify_ledger(path, raise_on_error=True)


@pytest.mark.parametrize("mutation", ["delete", "reorder", "head"])
def test_deletion_reordering_and_head_tampering_fail_verification(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path, seed=7)
    _append(path, seed=8)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if mutation == "delete":
        payload["records"].pop()
    elif mutation == "reorder":
        payload["records"].reverse()
    else:
        payload["head_hash"] = _digest("f")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert verify_ledger(path) is False


def test_identical_eval_results_have_matching_result_hashes(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"

    first = _append(path)
    second = _append(path)

    assert first.result_hash == second.result_hash
    assert first.reproducibility_hash == second.reproducibility_hash
    assert first.record_hash != second.record_hash
    assert (
        compute_result_hash(
            {
                "passed": True,
                "metrics": {"recall": 0.99, "micro_f1": 0.98},
                "fixture_count": 20,
            }
        )
        == first.result_hash
    )


def test_input_change_preserves_result_hash_but_changes_reproducibility_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval-provenance.json"

    first = _append(path, seed=7)
    second = _append(path, seed=8)

    assert first.result_hash == second.result_hash
    assert first.reproducibility_hash != second.reproducibility_hash


def test_raw_or_identifying_result_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"

    with pytest.raises(UnsafeResultMetadataError):
        append_eval_result(
            path,
            model_digest=_digest("a"),
            suite_id="synthetic-pii",
            suite_version="2.1.0",
            code_hash="abc1234",
            seed=7,
            config_digest=_digest("b"),
            env_fingerprint=_digest("c"),
            result_metadata={"patient_name": "Alice Smith"},
        )

    assert not path.exists()


def test_ledger_persists_only_digests_and_aggregate_metadata(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path)

    serialized = path.read_text(encoding="utf-8")

    assert "Alice Smith" not in serialized
    assert "clinical note" not in serialized
    assert "micro_f1" in serialized
    assert serialized.count("sha256:") >= 8


def test_safe_aggregate_dimensions_can_describe_evaluation_coverage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval-provenance.json"

    record = append_eval_result(
        path,
        model_digest=_digest("a"),
        suite_id="synthetic-pii",
        suite_version="2.1.0",
        code_hash="abc1234",
        seed=7,
        config_digest=_digest("b"),
        env_fingerprint=_digest("c"),
        result_metadata={"document_count": 20, "text_length": 4000},
    )

    assert record.result_metadata == {"document_count": 20, "text_length": 4000}


def test_append_fails_closed_when_existing_ledger_was_tampered(tmp_path: Path) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["code_hash"] = "def5678"
    path.write_text(json.dumps(payload), encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(LedgerVerificationError):
        _append(path, seed=9)

    assert path.read_bytes() == before


def test_strict_schema_rejects_unknown_fields_that_could_carry_phi(
    tmp_path: Path,
) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["raw_note"] = "Patient Alice Smith"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert verify_ledger(path) is False


def test_empty_ledger_is_valid() -> None:
    ledger = EvalProvenanceLedger()

    assert ledger.record_count == 0
    assert ledger.verify() is True


def test_verify_command_reports_success_and_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "eval-provenance.json"
    _append(path)

    assert main(["verify", str(path)]) == 0
    assert "Verified 1 eval provenance record" in capsys.readouterr().out

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"][0]["result_hash"] = _digest("f")
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert main(["verify", str(path)]) == 1
    assert "verification failed" in capsys.readouterr().err
