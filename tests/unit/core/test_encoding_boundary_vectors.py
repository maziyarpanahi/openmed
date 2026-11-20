"""Encoding-boundary vectors shared by every binding that crosses a string edge.

Each case in ``encoding_boundary_v1.json`` states the reason code that
``normalize_text`` and ``hmac_text_hash`` produce today for lone surrogates,
invalid UTF-8 bytes, NUL, BOM, CRLF and U+2028 inputs.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import pytest

from openmed.core.schemas.span import hmac_text_hash
from openmed.utils.gateway import InputValidationError, normalize_text

FIXTURE_PATH = (
    Path(__file__).parents[2] / "fixtures" / "parity" / "encoding_boundary_v1.json"
)
FIXTURE_BYTES = FIXTURE_PATH.read_bytes()
FIXTURE = json.loads(FIXTURE_BYTES.decode("utf-8"))
CASES: list[dict[str, Any]] = FIXTURE["cases"]
HMAC_KEY: str = FIXTURE["hmac_key"]


def _case_input(case: dict[str, Any]) -> str | bytes:
    if "bytes_hex" in case:
        return bytes.fromhex(case["bytes_hex"])
    return case["text"]


def _is_lone_surrogate(char: str) -> bool:
    return 0xD800 <= ord(char) <= 0xDFFF


def test_fixture_is_ascii_json_with_escaped_surrogates_and_hex_bytes() -> None:
    FIXTURE_BYTES.decode("ascii")
    assert FIXTURE["version"] == 1
    assert FIXTURE["synthetic_only"] is True
    assert len({case["id"] for case in CASES}) == len(CASES)
    for case in CASES:
        assert ("text" in case) != ("bytes_hex" in case), case["id"]
    assert b"\\ud800" in FIXTURE_BYTES
    assert b"\\udc00" in FIXTURE_BYTES


def test_fixture_covers_required_boundaries() -> None:
    texts = [case["text"] for case in CASES if "text" in case]
    byte_inputs = [
        bytes.fromhex(case["bytes_hex"]) for case in CASES if "bytes_hex" in case
    ]

    assert any(any(0xD800 <= ord(c) <= 0xDBFF for c in text) for text in texts)
    assert any(any(0xDC00 <= ord(c) <= 0xDFFF for c in text) for text in texts)
    assert any("\x00" in text for text in texts)
    assert any("﻿" in text for text in texts)
    assert any("\r\n" in text for text in texts)
    assert any(" " in text for text in texts)
    invalid = [raw for raw in byte_inputs if not _decodes_as_utf8(raw)]
    assert len(invalid) >= 3


def _decodes_as_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return True


def test_fixture_outcomes_use_closed_reason_code_sets() -> None:
    normalize_codes = set(FIXTURE["normalize_outcomes"])
    hmac_codes = set(FIXTURE["hmac_outcomes"])
    for case in CASES:
        assert case["normalize"]["outcome"] in normalize_codes, case["id"]
        assert case["hmac"]["outcome"] in hmac_codes, case["id"]
    assert {case["normalize"]["outcome"] for case in CASES} == normalize_codes
    assert {case["hmac"]["outcome"] for case in CASES} == hmac_codes


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["id"]))
def test_normalize_text_matches_encoding_boundary_vector(
    case: dict[str, Any],
) -> None:
    expected = case["normalize"]
    value = _case_input(case)

    if expected["outcome"] == "ok":
        assert normalize_text(value) == expected["text"]
        return

    with pytest.raises(InputValidationError) as exc_info:
        normalize_text(value)
    error = exc_info.value
    assert error.code == expected["outcome"]
    # Rejected input must never be echoed back in the error.
    message = str(error)
    if isinstance(value, str):
        assert not any(_is_lone_surrogate(char) for char in message)
        assert value not in message
    else:
        assert value.hex() not in message
        assert repr(value) not in message


@pytest.mark.parametrize("case", CASES, ids=lambda case: str(case["id"]))
def test_hmac_text_hash_matches_encoding_boundary_vector(
    case: dict[str, Any],
) -> None:
    expected = case["hmac"]
    value = _case_input(case)

    if expected["outcome"] == "unencodable_text":
        with pytest.raises(UnicodeEncodeError):
            hmac_text_hash(value, HMAC_KEY)
        return

    digest = hmac_text_hash(value, HMAC_KEY)
    assert digest == expected["digest"]
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    independent = hmac.new(HMAC_KEY.encode("utf-8"), payload, hashlib.sha256)
    assert digest == f"hmac-sha256:{independent.hexdigest()}"


def test_invalid_utf8_bytes_are_rejected_by_normalize_but_hashed_raw() -> None:
    """Document today's split: the gateway rejects, the HMAC helper hashes bytes."""
    invalid_byte_cases = [
        case
        for case in CASES
        if "bytes_hex" in case
        and not _decodes_as_utf8(bytes.fromhex(case["bytes_hex"]))
    ]
    assert invalid_byte_cases
    for case in invalid_byte_cases:
        assert case["normalize"]["outcome"] == "invalid_encoding"
        assert case["hmac"]["outcome"] == "hashed"
