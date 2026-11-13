"""Recompute the synthetic cross-runtime date-shift vectors.

Failures name the vector id only, never its patient key, secret or digest.
"""

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
SECRETS = {item["id"]: bytes.fromhex(item["hex"]) for item in FIXTURE["secrets"]}
SEED_DOMAIN = bytes.fromhex(FIXTURE["algorithm"]["seed_domain_hex"])
PATIENT_VECTORS = FIXTURE["patient_key_offsets"]
SEED_VECTORS = FIXTURE["seed_offsets"]
DATE_VECTORS = FIXTURE["date_shifts"]
SEEDED_VECTORS = SEED_VECTORS + [v for v in DATE_VECTORS if "seed_decimal" in v]
# Largest integer every JSON reader (including JavaScript) parses exactly.
MAX_SAFE_JSON_INTEGER = 2**53 - 1


def _ids(vectors):
    return [vector["id"] for vector in vectors]


def _check(matches: bool, vector_id: str, what: str) -> None:
    # pytrace=False keeps keys, secrets and digests out of the report.
    if not matches:
        pytest.fail(f"date-shift vector {vector_id}: {what} mismatch", pytrace=False)


def _offset_from_hex_digest(digest_hex: str, max_days: int) -> int:
    bucket = int(digest_hex, 16) % (2 * max_days)
    return bucket - max_days if bucket < max_days else bucket - max_days + 1


def _is_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def _seed(vector) -> int:
    return int(vector["seed_decimal"])


def _expected_offset(vector) -> int:
    if "patient_key" in vector:
        return stable_offset_for(
            vector["patient_key"],
            max_days=vector["max_days"],
            secret=SECRETS[vector["secret_id"]],
        )
    return stable_offset_from_seed(_seed(vector), max_days=vector["max_days"])


def test_fixture_is_synthetic_with_test_only_secrets_and_unique_ids():
    assert FIXTURE["synthetic_only"] is True
    assert FIXTURE["version"] == 1
    assert FIXTURE["secrets"]
    assert all(item["test_only"] is True for item in FIXTURE["secrets"])
    utf8 = [item for item in FIXTURE["secrets"] if "utf8" in item]
    assert all(SECRETS[item["id"]] == item["utf8"].encode("utf-8") for item in utf8)
    ids = _ids(PATIENT_VECTORS) + _ids(SEED_VECTORS) + _ids(DATE_VECTORS)
    assert len(ids) == len(set(ids))
    keys = {vector["patient_key"] for vector in PATIENT_VECTORS}
    assert all(re.fullmatch(r"SYNTH-PATIENT-\d{4}(-\S+)?", key) for key in keys)


@pytest.mark.parametrize("vector", PATIENT_VECTORS, ids=_ids(PATIENT_VECTORS))
def test_patient_key_offsets_recompute(vector):
    vector_id, max_days = vector["id"], vector["max_days"]
    secret = SECRETS[vector["secret_id"]]
    message = vector["patient_key"].encode("utf-8")
    digest = hmac.new(secret, message, hashlib.sha256).hexdigest()
    _check(digest == vector["digest_sha256_hex"], vector_id, "digest")
    offset = vector["offset_days"]
    _check(_offset_from_hex_digest(digest, max_days) == offset, vector_id, "formula")
    _check(_expected_offset(vector) == offset, vector_id, "offset")
    by_bytes = stable_offset_for(message, max_days=max_days, secret=secret)
    _check(by_bytes == offset, vector_id, "bytes input")
    _check(offset != 0 and -max_days <= offset <= max_days, vector_id, "bounds")


@pytest.mark.parametrize("vector", SEED_VECTORS, ids=_ids(SEED_VECTORS))
def test_seed_offsets_recompute(vector):
    vector_id, max_days = vector["id"], vector["max_days"]
    text = vector["seed_decimal"].encode("ascii")
    digest = hashlib.sha256(SEED_DOMAIN + text).hexdigest()
    _check(digest == vector["digest_sha256_hex"], vector_id, "digest")
    offset = vector["offset_days"]
    _check(_offset_from_hex_digest(digest, max_days) == offset, vector_id, "formula")
    _check(_expected_offset(vector) == offset, vector_id, "offset")
    _check(offset != 0 and -max_days <= offset <= max_days, vector_id, "bounds")


@pytest.mark.parametrize("vector", DATE_VECTORS, ids=_ids(DATE_VECTORS))
def test_date_shifts_recompute(vector):
    vector_id = vector["id"]
    offset = _expected_offset(vector)
    _check(offset == vector["offset_days"], vector_id, "offset")
    source = date.fromisoformat(vector["source_date"])
    shifted = source + timedelta(days=offset)
    _check(shifted.isoformat() == vector["shifted_date"], vector_id, "shifted date")


def test_vectors_cover_inputs_and_calendar_edges():
    max_days_values = set(FIXTURE["max_days_values"])
    for vectors in (PATIENT_VECTORS, SEED_VECTORS):
        assert {vector["max_days"] for vector in vectors} == max_days_values
    assert {vector["offset_days"] > 0 for vector in SEED_VECTORS} == {True, False}
    assert any(not vector["patient_key"].isascii() for vector in PATIENT_VECTORS)
    assert set(SECRETS) == {vector["secret_id"] for vector in PATIENT_VECTORS}
    assert any(not _is_utf8(secret) for secret in SECRETS.values())
    seeds = {_seed(vector) for vector in SEED_VECTORS}
    assert any(not -(2**63) <= seed < 2**63 for seed in seeds)
    assert any(abs(seed) > MAX_SAFE_JSON_INTEGER for seed in seeds)

    def pair(vector):
        return (
            date.fromisoformat(vector["source_date"]),
            date.fromisoformat(vector["shifted_date"]),
        )

    def crosses(vector, boundary):
        low, high = sorted(pair(vector))
        return low <= boundary < high

    def covering(name):
        return [vector for vector in DATE_VECTORS if vector["covers"] == name]

    for vector in covering("leap_day") + covering("leap_day_source"):
        assert any(day.month == 2 and day.day == 29 for day in pair(vector))
    for vector in covering("year_boundary"):
        low, high = sorted(pair(vector))
        assert low.year != high.year, vector["id"]
    for name in ("month_end", "non_leap_february", "non_leap_century"):
        cases = covering(name)
        assert cases, name
        for vector in cases:
            low, high = sorted(pair(vector))
            assert low.month != high.month, vector["id"]
    for vector in covering("non_leap_february") + covering("non_leap_century"):
        year = pair(vector)[0].year
        assert year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)
        assert crosses(vector, date(year, 2, 28)), vector["id"]
    assert covering("leap_day") and covering("leap_day_source")


def test_wrong_secret_or_seed_text_does_not_reproduce_the_vectors():
    assert any(
        stable_offset_for(
            vector["patient_key"],
            max_days=vector["max_days"],
            secret=SECRETS[vector["secret_id"]] + b"-other",
        )
        != vector["offset_days"]
        for vector in PATIENT_VECTORS
        if vector["max_days"] > 1
    )
    assert all(
        hashlib.sha256(vector["seed_decimal"].encode("ascii")).hexdigest()
        != vector["digest_sha256_hex"]
        for vector in SEED_VECTORS
    )


@pytest.mark.parametrize("vector", SEEDED_VECTORS, ids=_ids(SEEDED_VECTORS))
def test_seed_text_is_canonical_and_numbers_are_json_safe(vector):
    # Ports hash seed_decimal; a numeric seed is only kept where every JSON
    # reader gets it exactly, so large seeds never lose precision.
    vector_id, text = vector["id"], vector["seed_decimal"]
    _check(str(int(text)) == text, vector_id, "seed text")
    if "seed" in vector:
        exact = type(vector["seed"]) is int and vector["seed"] == int(text)
        _check(exact, vector_id, "numeric seed")
        safe = abs(vector["seed"]) <= MAX_SAFE_JSON_INTEGER
        _check(safe, vector_id, "numeric seed range")
    else:
        _check(abs(int(text)) > MAX_SAFE_JSON_INTEGER, vector_id, "missing seed")


def test_fixture_has_no_json_number_beyond_the_safe_range():
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)
        elif type(value) is int:
            yield value

    assert all(abs(number) <= MAX_SAFE_JSON_INTEGER for number in walk(FIXTURE))
