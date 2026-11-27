#!/usr/bin/env python3
"""Generate or check the shared AuditReport receipt verification vectors.

``tests/fixtures/parity/receipt_verification_v1.json`` lets the Swift, Kotlin
and web verifiers check a signed de-identification ``AuditReport`` exactly as
:meth:`openmed.core.audit.AuditReport.verify` does. Every case is a synthetic
report signed with test-only HMAC-SHA256 material that is published in the
fixture. For each span order a verifier must try, a case records the canonical
payload bytes (base64), the expected ``repro_hash``, the canonical signing bytes
and the expected HMAC, plus the expected verification result. The cases cover
tampering, an unknown signature algorithm, the legacy stored span order,
non-ASCII keys and values, and float encodings that differ between languages.

The reports carry offsets, labels, actions and SHA-256 hashes of synthetic
placeholder strings only: no identifiers and no source text. The Python
reference produces every value, so regeneration is byte-stable.

``--check`` (the default) fails with value-free reason codes when the committed
fixture is stale. ``--write`` regenerates it; review the diff and rerun every
kit's verifier tests afterwards.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmed.core import audit  # noqa: E402
from openmed.core.audit import (  # noqa: E402
    AuditReport,
    AuditSpan,
    DetectorInfo,
    hash_text,
    verify_repro_hash,
)

FIXTURE = ROOT / "tests" / "fixtures" / "parity" / "receipt_verification_v1.json"
UPDATE_COMMAND = "python scripts/parity/generate_receipt_vectors.py --write"
SCHEMA = "openmed.receipt_verification_vectors"
VERSION = 1

# Published, synthetic HMAC material. It exists only so independent verifiers
# can reproduce the vectors; it must never sign a real report.
TEST_KEY_ID = "test-only"
TEST_MATERIAL = "OpenMed receipt vectors v1: test only, never production"

SORTED = "sorted"
STORED = "stored"


def _material() -> bytes:
    return TEST_MATERIAL.encode("utf-8")


def _span(
    start: int,
    end: int,
    label: str,
    canonical_label: str,
    action: str,
    *,
    placeholder: str,
    confidence: float = 0.97,
    threshold: float = 0.5,
    surrogate: str | None = None,
    sources: Sequence[str] = ("synthetic",),
    context: dict[str, Any] | None = None,
) -> AuditSpan:
    return AuditSpan(
        start=start,
        end=end,
        label=label,
        canonical_label=canonical_label,
        sources=list(sources),
        confidence=confidence,
        threshold=threshold,
        action=action,
        surrogate=surrogate,
        text_hash=hash_text(placeholder),
        evidence={"detector": "synthetic-detector"},
        context=context or {},
    )


def _report(
    spans: list[AuditSpan],
    *,
    thresholds: dict[str, float] | None = None,
    policy: str = "hipaa_safe_harbor",
    resolved_profile: dict[str, Any] | None = None,
    document: str = "01",
) -> AuditReport:
    return AuditReport(
        policy=policy,
        resolved_profile=resolved_profile or {"language": "en", "method": "mask"},
        detectors=[
            DetectorInfo(
                source="synthetic",
                model_id="synthetic-detector",
                model_format="none",
            )
        ],
        safety_sweep={"enabled": True, "spans_added": 0},
        spans=spans,
        thresholds=thresholds or {"DATE": 0.5, "PERSON": 0.5},
        residual_risk={"projected_leakage": 0.0},
        openmed_version="0.0.0+receipt-vectors",
        manifest_hash=hash_text("synthetic receipt vector manifest"),
        document_length=64,
        input_hash=hash_text(f"synthetic receipt vector document {document}"),
        deidentified_text_hash=hash_text(f"synthetic receipt vector output {document}"),
    )


def _baseline_spans() -> list[AuditSpan]:
    return [
        _span(
            8,
            16,
            "NAME",
            "PERSON",
            "mask",
            placeholder="synthetic-span-alpha",
            surrogate="[PERSON]",
            context={
                "before": {
                    "start": 0,
                    "end": 8,
                    "length": 8,
                    "text_hash": hash_text("synthetic-before-alpha"),
                }
            },
        ),
        _span(
            30,
            42,
            "DATE",
            "DATE",
            "redact",
            placeholder="synthetic-span-beta",
            confidence=0.88,
            sources=("rule", "synthetic"),
        ),
    ]


def _signed(report: AuditReport) -> dict[str, Any]:
    return report.sign(_material(), key_id=TEST_KEY_ID).to_dict()


def _baseline() -> dict[str, Any]:
    return _signed(_report(_baseline_spans()))


def _edit(mutate: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    document = _baseline()
    mutate(document)
    return document


def _flip_first_hex_digit(document: dict[str, Any]) -> None:
    value = document["signature"]["value"]
    first = "1" if value[0] == "0" else "0"
    document["signature"]["value"] = first + value[1:]


def _legacy_stored_order() -> dict[str, Any]:
    """Sign a report the way releases before deterministic span order did."""
    alpha, beta = _baseline_spans()
    report = _report([beta, alpha], document="02")
    stored = list(report.spans)
    report.repro_hash = audit.stable_hash(
        report._payload(include_repro_hash=False, include_signature=False, spans=stored)
    )
    message = audit._canonical_json(
        report._payload(include_repro_hash=True, include_signature=False, spans=stored)
    ).encode("utf-8")
    report.signature = audit.AuditSignature(
        key_id=TEST_KEY_ID,
        algorithm="HMAC-SHA256",
        value=hmac.new(_material(), message, hashlib.sha256).hexdigest(),
    )
    return report.to_dict()


def _non_ascii() -> dict[str, Any]:
    spans = [
        _span(
            4,
            10,
            "ÉTIQUETTE",
            "Ａ_LABEL",
            "mask",
            placeholder="synthetic-span-gamma",
            surrogate="[標籤]",
        ),
        # Same offsets: the order depends on comparing canonical labels by
        # code point (U+FF21 before U+1F600), not by UTF-16 code unit.
        _span(
            4,
            10,
            "\U0001f600",
            "\U0001f600_LABEL",
            "redact",
            placeholder="synthetic-span-delta",
        ),
    ]
    report = _report(
        spans,
        thresholds={
            "z": 0.5,
            "É": 0.5,
            "Ａ": 0.5,
            "\U0001f600": 0.5,
            "標籤": 0.5,
        },
        policy="hipaa_safe_harbor",
        resolved_profile={
            "language": "fr",
            "méthode": "masque",
            "注記": "合成",
            "\U0001f600": "\U0001f9ea",
        },
        document="03",
    )
    return _signed(report)


def _floats() -> dict[str, Any]:
    spans = [
        _span(
            8,
            16,
            "NAME",
            "PERSON",
            "mask",
            placeholder="synthetic-span-epsilon",
            confidence=0.1,
            threshold=1e-07,
        ),
        _span(
            20,
            28,
            "ID",
            "ID",
            "hash",
            placeholder="synthetic-span-zeta",
            confidence=0.30000000000000004,
            threshold=5e-324,
        ),
    ]
    report = _report(
        spans,
        thresholds={
            "HUGE": 1e16,
            "ID": 5e-324,
            "INTEGRAL": 1.0,
            "PERSON": 1e-07,
            "SUM": 0.30000000000000004,
            "TENTH": 0.1,
        },
        document="04",
    )
    return _signed(report)


def _unsigned() -> dict[str, Any]:
    return _report(_baseline_spans()).to_dict()


CASES: tuple[tuple[str, tuple[str, ...], Callable[[], dict[str, Any]]], ...] = (
    ("signed_baseline", ("baseline",), _baseline),
    ("unsigned", ("unsigned",), _unsigned),
    (
        "tampered_offset",
        ("tamper", "offset"),
        lambda: _edit(lambda doc: doc["spans"][1].update(start=31)),
    ),
    (
        "tampered_label",
        ("tamper", "label"),
        lambda: _edit(lambda doc: doc["spans"][0].update(canonical_label="LOCATION")),
    ),
    (
        "tampered_action",
        ("tamper", "action"),
        lambda: _edit(lambda doc: doc["spans"][0].update(action="keep")),
    ),
    (
        "tampered_signature_byte",
        ("tamper", "signature"),
        lambda: _edit(_flip_first_hex_digit),
    ),
    (
        "unknown_algorithm",
        ("algorithm",),
        lambda: _edit(lambda doc: doc["signature"].update(algorithm="HMAC-SHA512")),
    ),
    ("legacy_span_order", ("legacy_order",), _legacy_stored_order),
    ("non_ascii_keys_and_values", ("non_ascii", "key_order"), _non_ascii),
    ("float_edge_cases", ("floats",), _floats),
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _candidates(document: dict[str, Any]) -> list[dict[str, Any]]:
    report = AuditReport.from_dict(copy.deepcopy(document))
    orders = [(SORTED, report._sorted_spans())]
    stored = list(report.spans)
    if [span.to_dict() for span in stored] != [span.to_dict() for span in orders[0][1]]:
        orders.append((STORED, stored))
    candidates = []
    for name, spans in orders:
        repro = audit._canonical_json(
            report._payload(
                include_repro_hash=False, include_signature=False, spans=spans
            )
        ).encode("utf-8")
        signing = audit._canonical_json(
            report._payload(
                include_repro_hash=True, include_signature=False, spans=spans
            )
        ).encode("utf-8")
        candidates.append(
            {
                "span_order": name,
                "repro_payload_b64": _b64(repro),
                "expected_repro_hash": "sha256:" + hashlib.sha256(repro).hexdigest(),
                "signing_payload_b64": _b64(signing),
                "expected_hmac": hmac.new(
                    _material(), signing, hashlib.sha256
                ).hexdigest(),
            }
        )
    return candidates


def _matched_order(
    document: dict[str, Any], candidates: Sequence[dict[str, Any]]
) -> str | None:
    signature = document.get("signature")
    if not isinstance(signature, dict) or signature.get("algorithm") != ("HMAC-SHA256"):
        return None
    for candidate in candidates:
        if candidate["expected_repro_hash"] != document["repro_hash"]:
            continue
        if hmac.compare_digest(candidate["expected_hmac"], signature["value"]):
            return str(candidate["span_order"])
    return None


def build_vectors() -> dict[str, Any]:
    """Build the vector document from the Python reference.

    Returns:
        The fixture document. Building it twice yields equal documents.

    Raises:
        RuntimeError: ``reference_disagrees`` when the recorded candidates and
            :meth:`AuditReport.verify` disagree on a case.
    """
    cases = []
    for case_id, covers, build in CASES:
        document = build()
        candidates = _candidates(document)
        matched = _matched_order(document, candidates)
        verifies = AuditReport.from_dict(copy.deepcopy(document)).verify(_material())
        if verifies != (matched is not None):
            raise RuntimeError(f"reference_disagrees case={case_id}")
        cases.append(
            {
                "id": case_id,
                "covers": list(covers),
                "key_id": TEST_KEY_ID,
                "report": document,
                "candidates": candidates,
                "expected": {
                    "repro_hash_matches": verify_repro_hash(copy.deepcopy(document)),
                    "verifies": verifies,
                    "matched_span_order": matched,
                },
            }
        )
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "synthetic_only": True,
        "description": (
            "Synthetic AuditReport receipt verification vectors. Reports carry "
            "offsets, labels, actions and SHA-256 hashes of synthetic "
            "placeholders only. Regenerate with " + UPDATE_COMMAND + "."
        ),
        "reference": "openmed.core.audit.AuditReport.verify",
        "test_keys": [
            {
                "key_id": TEST_KEY_ID,
                "test_only": True,
                "encoding": "utf-8",
                "material": TEST_MATERIAL,
                "note": "Published test material. Never sign a real report with it.",
            }
        ],
        "signature_algorithm": "HMAC-SHA256",
        "canonical_json": {
            "object_member_order": "unicode_code_point",
            "separators": [",", ":"],
            "ascii_only": True,
            "escape": "lowercase \\uXXXX; astral characters as UTF-16 surrogate pairs",
            "non_finite_numbers": "rejected",
            "floats": "shortest round-trip repr as Python prints it (1e-07, 5e-324, "
            "1e+16, 1.0)",
        },
        "span_orders": {
            SORTED: "spans sorted by (start, end, canonical_label, action, canonical "
            "JSON of the span), strings compared by code point",
            STORED: "spans in the order the report stores them",
        },
        "payloads": {
            "repro": "the report without repro_hash and signature; its SHA-256 is "
            "the repro_hash, written sha256:<hex>",
            "signing": "the report without signature, keeping the stored "
            "repro_hash; HMAC-SHA256 of it is signature.value",
        },
        "verification": (
            "A report verifies when signature is an object whose algorithm is "
            "HMAC-SHA256 and some span order, tried sorted first and then stored, "
            "has a repro hash equal to the stored repro_hash and a signing-payload "
            "HMAC equal to signature.value. Anything else does not verify."
        ),
        "cases": cases,
    }


def render() -> str:
    """Return the fixture text exactly as it is committed.

    Returns:
        Two-space indented ASCII JSON with a trailing newline.
    """
    return json.dumps(build_vectors(), indent=2, ensure_ascii=True) + "\n"


def check(path: Path = FIXTURE) -> list[str]:
    """Return value-free problems with the committed fixture.

    Args:
        path: The fixture to compare with a fresh generation.

    Returns:
        ``fixture_missing`` or ``fixture_stale`` codes; empty when current.
    """
    if not path.is_file():
        return [f"fixture_missing path={path.name}"]
    if path.read_text(encoding="utf-8") != render():
        return [f"fixture_stale path={path.name}"]
    return []


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generator; ``--check`` is the default."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail when stale")
    mode.add_argument("--write", action="store_true", help="regenerate in place")
    parser.add_argument("--fixture", default=str(FIXTURE), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = Path(args.fixture)
    if args.write:
        path.write_text(render(), encoding="utf-8")
        return 0
    problems = check(path)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"refresh with: {UPDATE_COMMAND}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
