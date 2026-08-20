"""Focused tests for the deterministic, raw-value-free secret detector."""

from __future__ import annotations

import hashlib
import json

import pytest

from openmed.core.secrets import (
    ACCESS_KEY,
    ACCESS_TOKEN,
    API_KEY,
    AUTHORIZATION_HEADER,
    ENVIRONMENT_SECRET,
    PRIVATE_KEY,
    SecretDetector,
    SecretFinding,
    detect_secrets,
)


def _token(prefix: str, size: int = 24) -> str:
    """Build deterministic synthetic material without committing a credential."""
    return prefix + "A" * size


def _fingerprint(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_detects_authorization_header_and_emits_only_safe_fields():
    token = "synthetic-bearer-" + "A" * 24
    text = "trace Authorization: Bearer " + token + ", next"

    (finding,) = detect_secrets(text)

    start = text.index(token)
    assert finding.category == AUTHORIZATION_HEADER
    assert finding.offset == (start, start + len(token))
    assert finding.start == start
    assert finding.end == start + len(token)
    assert finding.fingerprint == _fingerprint(token)
    assert set(finding.to_dict()) == {"category", "offset", "fingerprint"}
    assert token not in repr(finding)
    assert token not in json.dumps(finding.to_dict())


def test_detects_known_token_shapes_and_resolves_overlapping_assignments():
    github_token = _token("gh" + "p_", 24)
    api_key = _token("s" + "k-", 24)
    access_key = "A" + "KIA" + "B" * 16
    jwt = "eyJ" + "C" * 12 + "." + "D" * 12 + "." + "E" * 12
    text = f"github={github_token} api_key={api_key} access={access_key} bearer={jwt}"

    findings = detect_secrets(text)

    assert [finding.category for finding in findings] == [
        ACCESS_TOKEN,
        API_KEY,
        ACCESS_KEY,
        ACCESS_TOKEN,
    ]
    assert [text[finding.start : finding.end] for finding in findings] == [
        github_token,
        api_key,
        access_key,
        jwt,
    ]
    assert len({finding.fingerprint for finding in findings}) == 4


def test_detects_private_key_material_as_one_span():
    private_key = (
        "-----BEGIN PRIVATE KEY-----\nsynthetic-key-material\n-----END PRIVATE KEY-----"
    )
    text = "payload=" + private_key

    (finding,) = detect_secrets(text)

    start = text.index(private_key)
    assert finding.category == PRIVATE_KEY
    assert finding.offset == (start, start + len(private_key))
    assert finding.fingerprint == _fingerprint(private_key)
    assert private_key not in json.dumps(finding.to_dict())


def test_unmatched_private_key_headers_scale_without_backtracking():
    header = "-----BEGIN PRIVATE KEY-----\n"
    findings = detect_secrets(header * 2_000)

    assert len(findings) == 2_000
    assert all(finding.category == PRIVATE_KEY for finding in findings)


def test_detects_parameterized_authorization_value_as_one_span():
    access_key = "A" + "KIA" + "B" * 16
    signature = "C" * 64
    value = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/20260819/eu/test, "
        f"SignedHeaders=host, Signature={signature}"
    )
    text = f"Authorization: {value}\nnext: safe"

    (finding,) = detect_secrets(text)

    assert finding.category == AUTHORIZATION_HEADER
    assert finding.offset == (text.index(value), text.index(value) + len(value))
    assert access_key not in repr(finding)
    assert signature not in json.dumps(finding.to_dict())


def test_detects_environment_secret_without_exposing_assignment_name_or_value():
    value = "synthetic-config-value-" + "Q" * 16
    text = f'OPENMED_API_KEY = "{value}"'

    (finding,) = detect_secrets(text)

    start = text.index(value)
    assert finding.category == ENVIRONMENT_SECRET
    assert finding.offset == (start, start + len(value))
    assert value not in repr(finding.to_dict())


def test_bracket_wrapping_does_not_hide_real_environment_secrets():
    value = "[synthetic-config-value-" + "Q" * 16 + "]"
    text = f"PASSWORD={value}; FALLBACK_PASSWORD=[REDACTED]"

    (finding,) = detect_secrets(text)

    assert finding.offset == (text.index(value), text.index(value) + len(value))


def test_near_miss_placeholders_are_not_reported():
    text = (
        "Authorization: Bearer <TOKEN>; "
        'OPENMED_API_KEY="your-api-key"; '
        "SECRET=replace-me; TOKEN=${TOKEN}"
    )

    assert detect_secrets(text) == []


def test_scan_is_deterministic_and_repeated_values_share_fingerprints():
    value = "synthetic-repeat-value-" + "R" * 16
    text = f"TOKEN={value}; TOKEN={value}"
    detector = SecretDetector()

    first = detector.scan(text)
    second = detector.detect(text)

    assert first == second
    assert len(first) == 2
    assert first[0].fingerprint == first[1].fingerprint


def test_rejects_non_text_input_without_echoing_the_value():
    with pytest.raises(TypeError, match="text must be a string"):
        detect_secrets(object())


def test_safe_finding_constructor_rejects_untrusted_report_fields():
    with pytest.raises(ValueError, match="supported secret category"):
        SecretFinding("synthetic-secret-value", (0, 1), "sha256:" + "a" * 64)

    with pytest.raises(ValueError, match="sha256 format"):
        SecretFinding(API_KEY, (0, 1), "sha256:synthetic-secret-value")
