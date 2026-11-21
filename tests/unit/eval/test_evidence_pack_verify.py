import json
import socket
from datetime import datetime, timedelta, timezone

import pytest

from openmed.cli.main import main
from openmed.core.intended_use import load_intended_use_registry
from openmed.eval.evidence_pack import ValidationEvidencePack
from openmed.eval.evidence_pack.lanes import required_lanes
from openmed.eval.evidence_pack.verify import sign_evidence_pack, verify_evidence_pack

KEY = b"synthetic-example-review-key-only"
NOW = datetime(2026, 1, 2, tzinfo=timezone.utc)


def unsigned():
    declaration = load_intended_use_registry().declarations["deidentify"]
    return {
        "schema_version": "1.0.0",
        "capability": "deidentify",
        "version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "intended_use_digest": declaration.digest(),
        "regulatory_tier": declaration.regulatory_tier.value,
        "lanes": [
            {
                "id": name,
                "state": "success",
                "mandatory": True,
                "report_digest": "sha256:" + "a" * 64,
                "required_metrics": list(metrics),
                "metrics": {m: 1 for m in metrics},
                "claims": [],
            }
            for name, metrics in required_lanes("deidentify").items()
        ],
        "frozen_inputs": [],
        "licenses": [],
        "exceptions": [],
        "claims": [],
        "limitations": ["not_clinically_validated", "not_a_medical_device"],
        "signature": None,
    }


def signed(data):
    return sign_evidence_pack(ValidationEvidencePack(data), key_id="review", key=KEY)


def verify(raw, **kwargs):
    options = dict(key=KEY, key_id="review", as_of=NOW, max_age=timedelta(days=1))
    return verify_evidence_pack(raw, **{**options, **kwargs})


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda p: p["lanes"].reverse(), "digest_mismatch"),
        (
            lambda p: p["lanes"][0]["metrics"].update(direct_identifier_recall=0),
            "digest_mismatch",
        ),
        (lambda p: p["signature"].update(key_id="other"), "signature_mismatch"),
    ],
)
def test_tampering(mutate, reason):
    data = signed(unsigned()).to_dict()
    mutate(data)
    result = verify(json.dumps(data))
    assert result.exit_code == 1 and reason in result.reasons


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (
            lambda p: p.update(intended_use_digest="sha256:" + "f" * 64),
            "intended_use_mismatch",
        ),
        (lambda p: p["lanes"].pop(), "missing_required_lane"),
        (lambda p: p["lanes"][0].update(mandatory=False), "missing_required_lane"),
        (
            lambda p: p["lanes"][0].update(metrics={}, required_metrics=[]),
            "missing_required_metric",
        ),
    ],
)
def test_signed_but_incomplete(mutate, reason):
    data = unsigned()
    mutate(data)
    result = verify(signed(data).to_json())
    assert result.exit_code == 1 and reason in result.reasons


def test_key_and_freshness():
    raw = signed(unsigned()).to_json()
    assert verify(raw).exit_code == 0
    assert "signature_mismatch" in verify(raw, key=b"x" * 32).reasons
    assert "expired_evidence" in verify(raw, as_of=NOW + timedelta(seconds=1)).reasons
    assert "future_timestamp" in verify(raw, as_of=NOW - timedelta(days=2)).reasons
    assert verify(raw, as_of=NOW.replace(tzinfo=None)).exit_code == 2


def test_offline_cli(tmp_path, capsys, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("unexpected network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    path = tmp_path / "synthetic.json"
    key = tmp_path / "review.key"
    key.write_bytes(KEY)
    path.write_text(signed(unsigned()).to_json())
    args = [
        "evidence",
        "verify",
        str(path),
        "--key-id",
        "review",
        "--key-file",
        str(key),
        "--as-of",
        NOW.isoformat(),
        "--max-age-days",
        "1",
    ]
    assert main(args) == 0
    key.write_bytes(b"x" * 32)
    assert main(args) == 1
    path.write_text("sensitive-fixture-marker")
    assert main(args) == 2
    output = capsys.readouterr()
    assert "sensitive-fixture-marker" not in output.out + output.err
    assert "invalid_pack_json" in output.out + output.err


def test_corrective_action_ledger_on_signed_pack(tmp_path, capsys):
    from openmed.compliance.corrective_action import (
        CorrectiveAction,
        CorrectiveActionLedger,
    )

    ledger = CorrectiveActionLedger()

    def add(number, severity="high", capability="deidentify", version="1.0.0"):
        return ledger.open(
            CorrectiveAction(
                record_id=f"CAPA-2026-{number:04d}",
                severity=severity,
                source="audit",
                capabilities=[capability],
                versions=[version],
                local_note="private-note-marker",
            )
        ).record_id

    closed = add(1)
    for state in [
        "investigating",
        "action_planned",
        "implemented",
        "verified",
        "closed",
    ]:
        ledger.transition(closed, state, after_evidence_digest="sha256:" + "a" * 64)
    add(2, capability="ecg")
    add(3, version="4.3.0")
    add(4, severity="medium")
    pack = signed(unsigned()).to_json()
    assert (
        verify(pack, corrective_action_ledger=json.dumps(ledger.to_dict())).exit_code
        == 0
    )
    first = add(5, severity="critical")
    second = add(6)
    ledger.transition(second, "investigating")
    snapshot = ledger.to_dict()
    result = verify(pack, corrective_action_ledger=json.dumps(snapshot))
    assert result.exit_code == 1
    assert result.reasons == ("open_corrective_action",)
    assert result.corrective_action_ids == (first, second)

    path, key, ledger_file = (tmp_path / name for name in ["pack", "key", "ledger"])
    path.write_text(pack)
    key.write_bytes(KEY)
    ledger_file.write_text(json.dumps(snapshot))
    args = [
        "evidence",
        "verify",
        str(path),
        "--key-id",
        "review",
        "--key-file",
        str(key),
        "--as-of",
        NOW.isoformat(),
        "--max-age-days",
        "1",
        "--corrective-action-ledger",
        str(ledger_file),
        "--json",
    ]
    assert main(args) == 1
    out = capsys.readouterr().out
    assert (
        first in out
        and second in out
        and closed not in out
        and "private-note-marker" not in out
    )
    assert "open_corrective_action" in out
    snapshot["records"][-1]["severity"] = "low"
    ledger_file.write_text(json.dumps(snapshot))
    assert main(args) == 2
    out = capsys.readouterr()
    assert "invalid_corrective_action_ledger" in out.out + out.err
