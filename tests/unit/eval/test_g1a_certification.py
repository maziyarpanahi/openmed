"""Synthetic G1a certification and offline verification tests."""

import hashlib
import json

import pytest

from openmed.cli.main import main
from openmed.eval.g1a_certification import _canonical, certify_g1a
from openmed.eval.release_gates import _G1A_LABELS

SEED = bytes(range(32))


def aggregate():
    return {
        "model_id": "org/test",
        "revision_digest": "a" * 64,
        "dataset_digest": "b" * 64,
        "year": 2014,
        "corpus": "synthetic",
        "minimum_gold_spans": 5,
        "by_label": {
            label: {"status": "reported", "tp": 996, "fn": 4, "recall": {"rate": 0.996}}
            for label in _G1A_LABELS
        },
        "private_record": "PRIVATE-RECORD-ID",
    }


def sign(data):
    return certify_g1a(data, model_id="org/test", private_key=SEED)


def test_observed_recall_and_signature():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    report = sign(aggregate())
    payload = report["certification"]
    assert payload["state"] == "CERTIFIED" and not payload["not_assessed"]
    assert payload["by_label"]["PERSON"]["wilson_lower_95"] < 0.995
    assert report["digest"] == hashlib.sha256(_canonical(payload)).hexdigest()
    Ed25519PrivateKey.from_private_bytes(SEED).public_key().verify(
        bytes.fromhex(report["signature"]), _canonical(payload)
    )
    assert "PRIVATE-RECORD-ID" not in json.dumps(report)


def test_failed_and_unassessed_labels():
    data = aggregate()
    data["by_label"]["PERSON"].update(tp=990, fn=10, recall={"rate": 0.99})
    data["by_label"]["SSN"] = {"status": "suppressed"}
    payload = sign(data)["certification"]
    assert payload["state"] == "NOT_CERTIFIED"
    assert payload["failing_labels"] == ["PERSON"]
    assert payload["not_assessed"] == ["SSN"]
    data["by_label"] = {}
    assert sign(data)["certification"]["state"] == "NOT_CERTIFIED"


def test_reject_inconsistent_inputs():
    data = aggregate()
    data["by_label"]["PERSON"]["recall"]["rate"] = 1
    with pytest.raises(ValueError):
        sign(data)
    data = aggregate()
    data["model_id"] = "other/model"
    with pytest.raises(ValueError):
        sign(data)


def test_certify_cli(tmp_path, capsys):
    agg, seed = tmp_path / "aggregate.json", tmp_path / "key"
    agg.write_text(json.dumps(aggregate()))
    seed.write_bytes(SEED)
    args = [
        "eval",
        "certify-g1a",
        "--aggregate",
        str(agg),
        "--model",
        "org/test",
        "--private-key-file",
        str(seed),
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["certification"]["state"] == "CERTIFIED"
    seed.write_bytes(b"wrong")
    assert main(args) == 2
    assert str(tmp_path) not in capsys.readouterr().err
