"""Signed G1a observed-recall floor decisions for aggregate-only reports."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from openmed.__about__ import __version__
from openmed.eval.release_gates import (
    _G1A_LABELS,
    G1A_V20_RECALL_FLOOR,
    _recall_floor_check,
)
from openmed.eval.synthetic_fidelity import FidelityCounts, wilson_interval

SCOPE = "Observed G1a recall on the named corpus edition and assessed labels only. Clinical validation is outside this scope."


def _canonical(value: dict) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _decision(counts: dict[str, tuple[int, int]]) -> dict:
    recalls = {}
    totals = {}
    labels = {}
    for label, (tp, fn) in sorted(counts.items()):
        if (
            label not in _G1A_LABELS
            or any(type(v) is not int or v < 0 for v in (tp, fn))
            or not tp + fn
        ):
            raise ValueError("invalid_label_counts")
        totals[label] = tp + fn
        recalls[label] = tp / totals[label]
        labels[label] = {
            "tp": tp,
            "fn": fn,
            "recall": recalls[label],
            "wilson_lower_95": wilson_interval(FidelityCounts(tp, tp + fn))[0],
            "floor": G1A_V20_RECALL_FLOOR,
        }
    gate = _recall_floor_check(
        "G1a", _G1A_LABELS, recalls, totals, G1A_V20_RECALL_FLOOR
    )
    return {
        "state": "CERTIFIED" if counts and gate.passed else "NOT_CERTIFIED",
        "floor": G1A_V20_RECALL_FLOOR,
        "by_label": labels,
        "failing_labels": sorted(gate.details["violations"]),
        "not_assessed": sorted(_G1A_LABELS - counts.keys()),
    }


def certify_g1a(aggregate: dict, *, model_id: str, private_key: bytes) -> dict:
    """Sign the per-label recall decision with a caller-supplied Ed25519 seed.

    Args:
        aggregate: Aggregate projection produced by the offline credentialed runner.
        model_id: Expected model identifier bound to the aggregate.
        private_key: Raw 32-byte signing seed.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if (
        not isinstance(aggregate, dict)
        or aggregate.get("model_id") != model_id
        or not re.fullmatch(r"[\w-][\w.-]*(?:/[\w-][\w.-]*)?", model_id, re.ASCII)
    ):
        raise ValueError("model_identity_mismatch")
    if (
        aggregate.get("corpus") not in {"i2b2", "n2c2", "cegs-n-grid", "synthetic"}
        or type(aggregate.get("year")) is not int
        or not 1000 <= aggregate["year"] <= 9999
    ):
        raise ValueError("invalid_corpus_edition")
    for field in ("revision_digest", "dataset_digest"):
        if not re.fullmatch(r"[a-f0-9]{64}", str(aggregate.get(field))):
            raise ValueError("invalid_provenance_digest")
    threshold = aggregate.get("minimum_gold_spans")
    cells = aggregate.get("by_label")
    if type(threshold) is not int or threshold < 1 or not isinstance(cells, dict):
        raise ValueError("invalid_aggregate")
    counts = {}
    for label in sorted(_G1A_LABELS.intersection(cells)):
        cell = cells[label]
        if cell == {"status": "suppressed"}:
            continue
        if not isinstance(cell, dict) or cell.get("status") != "reported":
            raise ValueError("invalid_aggregate_cell")
        tp, fn = cell.get("tp"), cell.get("fn")
        if any(type(v) is not int or v < 0 for v in (tp, fn)) or tp + fn < threshold:
            raise ValueError("invalid_label_counts")
        if cell.get("recall", {}).get("rate") != tp / (tp + fn):
            raise ValueError("inconsistent_aggregate_recall")
        counts[label] = (tp, fn)
    payload = {
        "schema_version": 1,
        "scope": SCOPE,
        "openmed_version": __version__,
        "date": datetime.now(timezone.utc).date().isoformat(),
        **{
            k: aggregate[k]
            for k in ("model_id", "revision_digest", "corpus", "year", "dataset_digest")
        },
        **_decision(counts),
    }
    encoded = _canonical(payload)
    return {
        "certification": payload,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "signature": Ed25519PrivateKey.from_private_bytes(private_key)
        .sign(encoded)
        .hex(),
    }


def verify_g1a(report: dict, *, public_key: bytes) -> dict:
    """Validate the envelope against the supplied trusted key and recompute G1a.

    Args:
        report: Signed certification envelope.
        public_key: Raw 32-byte trusted Ed25519 public key obtained separately.

    Returns:
        The verified certification payload, including its pass or fail decision.
    """
    from importlib.resources import files

    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from jsonschema import Draft202012Validator, FormatChecker

    schema = files("openmed.core.schemas.json").joinpath(
        "g1a_certification.schema.json"
    )
    validator = Draft202012Validator(
        json.loads(schema.read_bytes()), format_checker=FormatChecker()
    )
    if not validator.is_valid(report):
        raise ValueError("invalid_certification_schema")
    payload = report["certification"]
    encoded = _canonical(payload)
    if hashlib.sha256(encoded).hexdigest() != report["digest"]:
        raise ValueError("certification_digest_mismatch")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(
            bytes.fromhex(report["signature"]), encoded
        )
    except (ValueError, InvalidSignature):
        raise ValueError("untrusted_certification_signature") from None
    decision = _decision(
        {label: (cell["tp"], cell["fn"]) for label, cell in payload["by_label"].items()}
    )
    if any(payload[field] != value for field, value in decision.items()):
        raise ValueError("inconsistent_certification_decision")
    return payload
