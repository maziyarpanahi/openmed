import json
from datetime import timedelta

import pytest

from openmed.cli.main import main
from openmed.core.portable_bundle import ed25519_public_key
from openmed.eval.evidence_pack import EvidencePackError, ValidationEvidencePack
from openmed.eval.evidence_pack.verify import sign_evidence_pack, verify_evidence_pack
from tests.unit.eval.test_evidence_pack_verify import NOW, unsigned

# RFC 8032 public test vector. Used only for deterministic offline tests.
SEED = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
PUBLIC = bytes.fromhex(
    "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
)


def verify(raw, **keys):
    return verify_evidence_pack(
        raw, key_id="review", as_of=NOW, max_age=timedelta(days=1), **keys
    )


def test_public_key_only_and_tampering():
    pack = ValidationEvidencePack(unsigned())
    signed = sign_evidence_pack(pack, key_id="review", key=SEED, algorithm="Ed25519")
    raw = signed.to_json()
    assert ed25519_public_key(SEED) == PUBLIC
    assert verify(raw, public_key=PUBLIC).exit_code == 0
    assert verify(raw, public_key=b"x" * 32).exit_code == 1
    assert verify(raw + " ", public_key=PUBLIC).exit_code == 1
    assert (
        verify(
            raw.replace('"version":"1.0.0"', '"version":"1.0.1"'), public_key=PUBLIC
        ).exit_code
        == 1
    )
    assert (
        verify(
            raw.replace('"key_id":"review"', '"key_id":"other"'), public_key=PUBLIC
        ).exit_code
        == 1
    )
    assert verify(raw, public_key=PUBLIC, key=SEED).exit_code == 2
    assert verify(raw, key=SEED).exit_code == 1
    hmac_pack = sign_evidence_pack(pack, key_id="review", key=SEED)
    assert verify(hmac_pack.to_json(), key=SEED).exit_code == 0
    assert verify(hmac_pack.to_json(), public_key=PUBLIC).exit_code == 1
    with pytest.raises(EvidencePackError):
        sign_evidence_pack(pack, key_id="x" * 65, key=SEED, algorithm="Ed25519")


@pytest.mark.parametrize("source", ["file", "env"])
def test_sign_and_verify_via_cli(tmp_path, capsys, monkeypatch, source):
    pack = tmp_path / "pack.json"
    pack.write_text(json.dumps(unsigned()))
    seed, public = tmp_path / "seed", tmp_path / "public"
    seed.write_bytes(SEED)
    public.write_bytes(PUBLIC)
    monkeypatch.setenv("TEST_RELEASE_SEED", SEED.hex())
    options = (
        ["--private-key-file", str(seed)]
        if source == "file"
        else ["--private-key-env", "TEST_RELEASE_SEED"]
    )
    assert main(["evidence", "sign", str(pack), "--key-id", "review", *options]) == 0
    signed = capsys.readouterr().out
    assert SEED.hex() not in signed and "signature" in signed
    pack.write_text(signed)
    assert (
        main(
            [
                "evidence",
                "verify",
                str(pack),
                "--public-key-file",
                str(public),
                "--key-id",
                "review",
                "--as-of",
                NOW.isoformat(),
                "--max-age-days",
                "1",
            ]
        )
        == 0
    )
