"""Shared AuditReport receipt verification vectors for the kit verifiers."""

from __future__ import annotations

import base64
import copy
import hashlib
import hmac
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from openmed.core.audit import AuditReport, verify_repro_hash

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "parity" / "generate_receipt_vectors.py"
FIXTURE = ROOT / "tests" / "fixtures" / "parity" / "receipt_verification_v1.json"

SPEC = importlib.util.spec_from_file_location("generate_receipt_vectors", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = generator
SPEC.loader.exec_module(generator)

TEXT = FIXTURE.read_text(encoding="utf-8")
VECTORS = json.loads(TEXT)
CASES = {case["id"]: case for case in VECTORS["cases"]}
MATERIAL = VECTORS["test_keys"][0]["material"].encode("utf-8")
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    """Canonical JSON as the fixture documents it, without OpenMed code."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _ordered_spans(report: dict[str, Any], order: str) -> list[dict[str, Any]]:
    spans = list(report["spans"])
    if order == "stored":
        return spans
    return sorted(
        spans,
        key=lambda span: (
            span["start"],
            span["end"],
            span["canonical_label"],
            span["action"],
            _canonical(span),
        ),
    )


def _independent_payloads(report: dict[str, Any], order: str) -> tuple[bytes, bytes]:
    body = {key: value for key, value in report.items() if key != "signature"}
    body["spans"] = _ordered_spans(report, order)
    repro = {key: value for key, value in body.items() if key != "repro_hash"}
    return _canonical(repro), _canonical(body)


def _independent_verify(report: dict[str, Any], material: bytes) -> str | None:
    signature = report.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != "HMAC-SHA256":
        return None
    for order in ("sorted", "stored"):
        repro, signing = _independent_payloads(report, order)
        if "sha256:" + hashlib.sha256(repro).hexdigest() != report["repro_hash"]:
            continue
        mac = hmac.new(material, signing, hashlib.sha256).hexdigest()
        if hmac.compare_digest(mac, signature["value"]):
            return order
    return None


def test_regeneration_is_byte_stable_and_matches_the_committed_fixture() -> None:
    first = generator.render()

    assert first == generator.render()
    assert first == TEXT
    assert generator.check() == []
    assert generator.main(["--check"]) == 0


def test_a_stale_fixture_fails_the_check_with_a_value_free_code(
    tmp_path: Path, capsys
) -> None:
    stale = tmp_path / FIXTURE.name
    document = copy.deepcopy(VECTORS)
    document["cases"][0]["candidates"][0]["expected_hmac"] = "0" * 64
    stale.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    assert generator.main(["--check", "--fixture", str(stale)]) == 1
    error = capsys.readouterr().err
    assert "fixture_stale path=receipt_verification_v1.json" in error
    assert str(tmp_path) not in error
    assert generator.main(["--check", "--fixture", str(tmp_path / "x.json")]) == 1


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_every_case_verifies_in_python_with_the_expected_result(case_id: str) -> None:
    case = CASES[case_id]
    expected = case["expected"]
    report = AuditReport.from_dict(copy.deepcopy(case["report"]))

    assert report.verify(MATERIAL) is expected["verifies"]
    assert (
        verify_repro_hash(copy.deepcopy(case["report"]))
        is (expected["repro_hash_matches"])
    )
    assert report.verify(MATERIAL + b"-wrong") is False
    # A JSON round trip keeps the signed bytes, so the verdict is unchanged.
    restored = AuditReport.from_json(report.to_json())
    assert restored.verify(MATERIAL) is expected["verifies"]


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_payload_bytes_hashes_and_hmacs_recompute_independently(
    case_id: str,
) -> None:
    case = CASES[case_id]
    report = case["report"]
    orders = [candidate["span_order"] for candidate in case["candidates"]]

    assert orders[0] == "sorted"
    assert orders in (["sorted"], ["sorted", "stored"])
    for candidate in case["candidates"]:
        repro = base64.b64decode(candidate["repro_payload_b64"], validate=True)
        signing = base64.b64decode(candidate["signing_payload_b64"], validate=True)

        assert (repro, signing) == _independent_payloads(
            report, candidate["span_order"]
        )
        assert candidate["expected_repro_hash"] == (
            "sha256:" + hashlib.sha256(repro).hexdigest()
        )
        assert candidate["expected_hmac"] == (
            hmac.new(MATERIAL, signing, hashlib.sha256).hexdigest()
        )
        assert repro.isascii() and signing.isascii()
    assert (
        _independent_verify(report, MATERIAL)
        == (case["expected"]["matched_span_order"])
    )
    assert (case["expected"]["matched_span_order"] is not None) is (
        case["expected"]["verifies"]
    )


def test_vectors_cover_tampering_algorithms_order_unicode_and_floats() -> None:
    assert set(CASES) == {
        "signed_baseline",
        "unsigned",
        "tampered_offset",
        "tampered_label",
        "tampered_action",
        "tampered_signature_byte",
        "unknown_algorithm",
        "legacy_span_order",
        "non_ascii_keys_and_values",
        "float_edge_cases",
    }
    verdicts = {
        case_id: case["expected"]["verifies"] for case_id, case in CASES.items()
    }
    assert {case_id for case_id, ok in verdicts.items() if ok} == {
        "signed_baseline",
        "legacy_span_order",
        "non_ascii_keys_and_values",
        "float_edge_cases",
    }
    # Content tampering breaks the repro hash; signature tampering only the HMAC.
    for case_id in ("tampered_offset", "tampered_label", "tampered_action"):
        assert CASES[case_id]["expected"]["repro_hash_matches"] is False
    for case_id in ("tampered_signature_byte", "unknown_algorithm", "unsigned"):
        assert CASES[case_id]["expected"]["repro_hash_matches"] is True
    assert (
        CASES["unknown_algorithm"]["report"]["signature"]["algorithm"]
        != (VECTORS["signature_algorithm"])
    )
    assert CASES["unsigned"]["report"]["signature"] is None
    legacy = CASES["legacy_span_order"]
    assert [item["span_order"] for item in legacy["candidates"]] == [
        "sorted",
        "stored",
    ]
    assert legacy["expected"]["matched_span_order"] == "stored"

    floats = base64.b64decode(
        CASES["float_edge_cases"]["candidates"][0]["repro_payload_b64"]
    ).decode("ascii")
    for literal in (
        ":0.1,",
        ":1e-07",
        ":0.30000000000000004",
        ":5e-324",
        ":1e+16",
        ":1.0,",
    ):
        assert literal in floats

    unicode_case = CASES["non_ascii_keys_and_values"]
    payload = base64.b64decode(unicode_case["candidates"][0]["repro_payload_b64"])
    assert b"\\u00c9" in payload
    assert b"\\ud83d\\ude00" in payload
    keys = list(unicode_case["report"]["thresholds"])
    by_code_point = sorted(keys)
    by_utf16 = sorted(keys, key=lambda key: key.encode("utf-16-be"))
    assert by_code_point != by_utf16
    thresholds = payload[payload.index(b'"thresholds":') :]
    assert thresholds.index(b'"\\uff21":') < thresholds.index(b'"\\ud83d\\ude00":')
    labels = [span["canonical_label"] for span in unicode_case["report"]["spans"]]
    assert labels == sorted(labels)
    assert labels != sorted(labels, key=lambda label: label.encode("utf-16-be"))


def test_fixture_marks_the_material_test_only_and_carries_no_source_text() -> None:
    keys = VECTORS["test_keys"]

    assert VECTORS["synthetic_only"] is True
    assert len(keys) == 1 and keys[0]["test_only"] is True
    assert keys[0]["key_id"] == generator.TEST_KEY_ID
    assert "test only" in keys[0]["material"]
    decoded = [TEXT]
    for case in VECTORS["cases"]:
        report = case["report"]
        assert case["key_id"] == keys[0]["key_id"]
        if report["signature"] is not None:
            assert report["signature"]["key_id"] == keys[0]["key_id"]
        for field in (
            "input_hash",
            "deidentified_text_hash",
            "manifest_hash",
            "repro_hash",
        ):
            assert SHA256_RE.match(report[field]), (case["id"], field)
        for span in report["spans"]:
            assert "text" not in span
            assert SHA256_RE.match(span["text_hash"])
            for descriptor in span["context"].values():
                assert set(descriptor) == {"start", "end", "length", "text_hash"}
        for candidate in case["candidates"]:
            for field in ("repro_payload_b64", "signing_payload_b64"):
                decoded.append(base64.b64decode(candidate[field]).decode("ascii"))
    corpus = "\n".join(decoded)
    # The placeholders behind every hash never appear, and nothing looks like
    # a contact detail or a long identifier.
    assert "synthetic-span-" not in corpus
    assert "synthetic receipt vector document" not in corpus
    assert "@" not in corpus
    assert re.search(r"\d{3}-\d{2}-\d{4}", corpus) is None
