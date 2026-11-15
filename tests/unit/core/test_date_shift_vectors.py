"""Recompute the synthetic cross-runtime date-shift vectors."""

import hashlib
import hmac
import json
import re
from datetime import date, timedelta
from pathlib import Path

import pytest

from openmed.core.date_shift import stable_offset_for, stable_offset_from_seed

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2] / "fixtures/parity/date_shift_v1.json"
)
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SECRET = FIXTURE["secret"]["utf8"]
SEED_DOMAIN = bytes.fromhex(FIXTURE["algorithm"]["seed_domain_hex"])


def _offset_from_hex_digest(digest_hex: str, max_days: int) -> int:
    bucket = int(digest_hex, 16) % (2 * max_days)
    return bucket - max_days if bucket < max_days else bucket - max_days + 1


def test_fixture_is_synthetic_with_a_test_only_secret():
    assert FIXTURE["synthetic_only"] is True
    assert FIXTURE["version"] == 1
    assert FIXTURE["secret"]["test_only"] is True
    assert bytes.fromhex(FIXTURE["secret"]["hex"]) == SECRET.encode("utf-8")
    keys = {vector["patient_key"] for vector in FIXTURE["patient_key_offsets"]}
    assert keys and all(re.fullmatch(r"SYNTH-PATIENT-\d{4}", key) for key in keys)


@pytest.mark.parametrize("vector", FIXTURE["patient_key_offsets"])
def test_patient_key_offsets_recompute(vector):
    key, max_days = vector["patient_key"], vector["max_days"]
    digest = hmac.new(SECRET.encode("utf-8"), key.encode("utf-8"), hashlib.sha256)
    assert digest.hexdigest() == vector["digest_sha256_hex"]
    offset = vector["offset_days"]
    assert _offset_from_hex_digest(vector["digest_sha256_hex"], max_days) == offset
    assert stable_offset_for(key, max_days=max_days, secret=SECRET) == offset
    assert (
        stable_offset_for(key.encode(), max_days=max_days, secret=SECRET.encode())
        == offset
    )
    assert offset != 0 and -max_days <= offset <= max_days


@pytest.mark.parametrize("vector", FIXTURE["seed_offsets"])
def test_seed_offsets_recompute(vector):
    seed, max_days = vector["seed"], vector["max_days"]
    digest = hashlib.sha256(SEED_DOMAIN + str(seed).encode("ascii")).hexdigest()
    assert digest == vector["digest_sha256_hex"]
    offset = vector["offset_days"]
    assert _offset_from_hex_digest(digest, max_days) == offset
    assert stable_offset_from_seed(seed, max_days=max_days) == offset
    assert offset != 0 and -max_days <= offset <= max_days


@pytest.mark.parametrize("vector", FIXTURE["date_shifts"])
def test_date_shifts_recompute(vector):
    max_days = vector["max_days"]
    if "patient_key" in vector:
        offset = stable_offset_for(
            vector["patient_key"], max_days=max_days, secret=SECRET
        )
    else:
        offset = stable_offset_from_seed(vector["seed"], max_days=max_days)
    assert offset == vector["offset_days"]
    source = date.fromisoformat(vector["source_date"])
    shifted = source + timedelta(days=offset)
    assert shifted.isoformat() == vector["shifted_date"]
    assert shifted - timedelta(days=offset) == source


def test_vectors_cover_every_max_days_leap_days_and_year_boundaries():
    max_days_values = set(FIXTURE["max_days_values"])
    for section in ("patient_key_offsets", "seed_offsets"):
        assert {vector["max_days"] for vector in FIXTURE[section]} == max_days_values
    signs = {vector["offset_days"] > 0 for vector in FIXTURE["seed_offsets"]}
    assert signs == {True, False}

    pairs = [
        (date.fromisoformat(v["source_date"]), date.fromisoformat(v["shifted_date"]))
        for v in FIXTURE["date_shifts"]
    ]
    assert any(day.month == 2 and day.day == 29 for pair in pairs for day in pair)
    assert any(source.year != shifted.year for source, shifted in pairs)
    assert any(source.year % 100 == 0 and source.year % 400 != 0 for source, _ in pairs)


def test_wrong_secret_or_seed_text_does_not_reproduce_the_vectors():
    vectors = FIXTURE["patient_key_offsets"]
    assert any(
        stable_offset_for(
            vector["patient_key"],
            max_days=vector["max_days"],
            secret=SECRET + "-other",
        )
        != vector["offset_days"]
        for vector in vectors
        if vector["max_days"] > 1
    )
    undomained = [
        hashlib.sha256(str(vector["seed"]).encode("ascii")).hexdigest()
        for vector in FIXTURE["seed_offsets"]
    ]
    assert all(
        digest != vector["digest_sha256_hex"]
        for digest, vector in zip(undomained, FIXTURE["seed_offsets"])
    )
