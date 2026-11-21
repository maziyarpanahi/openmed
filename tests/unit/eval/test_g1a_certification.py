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
            label: {
                "status": "reported",
                "tp": 996,
                "fn": 4,
                "fp": 0,
                "precision": {"rate": 1},
                "recall": {"rate": 0.996},
            }
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


def _public_key():
    from openmed.core.portable_bundle import ed25519_public_key

    return ed25519_public_key(SEED)


def test_verify_cli_requires_only_report_and_public_key(tmp_path, capsys):
    report, key = tmp_path / "cert", tmp_path / "key"
    report.write_text(json.dumps(sign(aggregate())))
    key.write_bytes(_public_key())
    args = ["eval", "verify-g1a", str(report), "--public-key-file", str(key)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == {
        "verified": True,
        "state": "CERTIFIED",
    }
    key.write_bytes(bytes(32))
    assert main(args) == 2


@pytest.mark.parametrize("field", ["recall", "floor", "wilson_lower_95", "fn"])
def test_signed_inconsistent_arithmetic_is_rejected(field):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from openmed.eval.g1a_certification import verify_g1a

    report = sign(aggregate())
    report["certification"]["by_label"]["PERSON"][field] = 0
    raw = _canonical(report["certification"])
    report["digest"] = hashlib.sha256(raw).hexdigest()
    report["signature"] = Ed25519PrivateKey.from_private_bytes(SEED).sign(raw).hex()
    with pytest.raises(ValueError):
        verify_g1a(report, public_key=_public_key())


def test_digest_signature_schema_and_decision():
    from openmed.eval.g1a_certification import verify_g1a

    data = aggregate()
    data["by_label"] = {}
    report = sign(data)
    assert verify_g1a(report, public_key=_public_key())["state"] == "NOT_CERTIFIED"
    for changes in (
        {"digest": "c" * 64},
        {"signature": "0" * 128},
        {"private_record": "record"},
    ):
        with pytest.raises(ValueError):
            verify_g1a({**report, **changes}, public_key=_public_key())


def test_precision_is_descriptive_and_respects_suppression():
    from openmed.eval.g1a_certification import verify_g1a

    data = aggregate()
    data["postprocessing_profile"] = "transformers-simple-v1"
    data["by_label"]["PERSON"].update(fp=996, precision={"rate": 0.5})
    data["by_label"]["SSN"] = {"status": "suppressed"}
    payload = verify_g1a(sign(data), public_key=_public_key())
    assert payload["state"] == "CERTIFIED"
    assert payload["postprocessing_profile"] == "transformers-simple-v1"
    cells = payload["precision_by_label"]
    assert cells["PERSON"] == {"fp": 996, "precision": 0.5}
    assert "SSN" not in cells
    data["by_label"]["PERSON"].update(
        tp=0, fn=1000, fp=0, recall={"rate": 0}, precision={"rate": None}
    )
    payload = verify_g1a(sign(data), public_key=_public_key())
    assert payload["precision_by_label"]["PERSON"] == {"fp": 0, "precision": None}
    assert payload["state"] == "NOT_CERTIFIED"


def test_v1_recall_only_remains_verifiable():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from openmed.eval.g1a_certification import verify_g1a

    report = sign(aggregate())
    payload = report["certification"]
    payload["schema_version"] = 1
    del payload["precision_by_label"], payload["postprocessing_profile"]
    encoded = _canonical(payload)
    report["digest"] = hashlib.sha256(encoded).hexdigest()
    report["signature"] = Ed25519PrivateKey.from_private_bytes(SEED).sign(encoded).hex()
    assert verify_g1a(report, public_key=_public_key()) == payload
