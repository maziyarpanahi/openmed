"""Deterministic audit reports for de-identification runs."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_HASH_ALGORITHM = "sha256"
_SIGNATURE_ALGORITHM = "HMAC-SHA256"


def _canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_bytes(data: bytes) -> str:
    return f"{_HASH_ALGORITHM}:{hashlib.sha256(data).hexdigest()}"


def hash_text(text: str) -> str:
    """Return a stable SHA-256 hash for text content."""
    return _sha256_bytes(text.encode("utf-8"))


def stable_hash(data: Any) -> str:
    """Return a stable SHA-256 hash for canonical JSON data."""
    return _sha256_bytes(_canonical_json(data).encode("utf-8"))


def manifest_hash(path: Path | None = None) -> str:
    """Return the committed model manifest hash used by audit reports."""
    manifest_path = path or Path(__file__).resolve().parents[2] / "models.jsonl"
    try:
        return _sha256_bytes(manifest_path.read_bytes())
    except OSError:
        return _sha256_bytes(b"")


def _key_bytes(key: bytes | str) -> bytes:
    if isinstance(key, bytes):
        return key
    if isinstance(key, str):
        return key.encode("utf-8")
    raise TypeError("Signing key must be str or bytes")


@dataclass
class DetectorInfo:
    """Detector provenance recorded in an audit report."""

    source: str
    model_id: str
    model_format: str
    commit: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "model_id": self.model_id,
            "model_format": self.model_format,
            "commit": self.commit,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DetectorInfo":
        return cls(
            source=str(data.get("source", "")),
            model_id=str(data.get("model_id", "")),
            model_format=str(data.get("model_format", "")),
            commit=(str(data["commit"]) if data.get("commit") is not None else None),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class AuditSpan:
    """Per-span provenance and redaction action."""

    start: int
    end: int
    label: str
    canonical_label: str
    sources: list[str]
    confidence: float
    threshold: float
    action: str
    surrogate: str | None
    text_hash: str
    evidence: dict[str, Any] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "canonical_label": self.canonical_label,
            "sources": list(self.sources),
            "confidence": float(self.confidence),
            "threshold": float(self.threshold),
            "action": self.action,
            "surrogate": self.surrogate,
            "text_hash": self.text_hash,
            "evidence": copy.deepcopy(self.evidence),
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditSpan":
        return cls(
            start=int(data.get("start", 0)),
            end=int(data.get("end", 0)),
            label=str(data.get("label", "")),
            canonical_label=str(data.get("canonical_label", "")),
            sources=[str(source) for source in data.get("sources", [])],
            confidence=float(data.get("confidence", 0.0)),
            threshold=float(data.get("threshold", 0.0)),
            action=str(data.get("action", "")),
            surrogate=(
                str(data["surrogate"]) if data.get("surrogate") is not None else None
            ),
            text_hash=str(data.get("text_hash", "")),
            evidence=dict(data.get("evidence") or {}),
            context=dict(data.get("context") or {}),
        )


@dataclass
class AuditSignature:
    """Signature metadata for an audit report."""

    key_id: str
    algorithm: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key_id": self.key_id,
            "algorithm": self.algorithm,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditSignature":
        return cls(
            key_id=str(data.get("key_id", "")),
            algorithm=str(data.get("algorithm", "")),
            value=str(data.get("value", "")),
        )


@dataclass
class AuditReport:
    """Signed, reproducible de-identification audit report."""

    policy: str
    resolved_profile: dict[str, Any]
    detectors: list[DetectorInfo]
    safety_sweep: dict[str, Any]
    spans: list[AuditSpan]
    thresholds: dict[str, float]
    residual_risk: dict[str, Any]
    openmed_version: str
    manifest_hash: str
    document_length: int
    input_hash: str
    deidentified_text_hash: str
    repro_hash: str = ""
    signature: AuditSignature | None = None

    def __post_init__(self) -> None:
        if not self.repro_hash:
            self.repro_hash = self.recompute_repro_hash()

    def _payload(
        self,
        *,
        include_repro_hash: bool,
        include_signature: bool,
    ) -> dict[str, Any]:
        payload = {
            "policy": self.policy,
            "resolved_profile": copy.deepcopy(self.resolved_profile),
            "detectors": [detector.to_dict() for detector in self.detectors],
            "safety_sweep": copy.deepcopy(self.safety_sweep),
            "spans": [span.to_dict() for span in self.spans],
            "thresholds": {
                str(label): float(value)
                for label, value in sorted(self.thresholds.items())
            },
            "residual_risk": copy.deepcopy(self.residual_risk),
            "openmed_version": self.openmed_version,
            "manifest_hash": self.manifest_hash,
            "document_length": int(self.document_length),
            "input_hash": self.input_hash,
            "deidentified_text_hash": self.deidentified_text_hash,
        }
        if include_repro_hash:
            payload["repro_hash"] = self.repro_hash
        if include_signature:
            payload["signature"] = (
                self.signature.to_dict() if self.signature is not None else None
            )
        return payload

    def recompute_repro_hash(self) -> str:
        """Recompute the report hash without trusting the stored value."""
        return stable_hash(
            self._payload(include_repro_hash=False, include_signature=False)
        )

    def to_dict(self) -> dict[str, Any]:
        return self._payload(include_repro_hash=True, include_signature=True)

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AuditReport":
        signature_data = data.get("signature")
        return cls(
            policy=str(data.get("policy", "")),
            resolved_profile=dict(data.get("resolved_profile") or {}),
            detectors=[
                DetectorInfo.from_dict(item)
                for item in data.get("detectors", [])
                if isinstance(item, Mapping)
            ],
            safety_sweep=dict(data.get("safety_sweep") or {}),
            spans=[
                AuditSpan.from_dict(item)
                for item in data.get("spans", [])
                if isinstance(item, Mapping)
            ],
            thresholds={
                str(label): float(value)
                for label, value in (data.get("thresholds") or {}).items()
            },
            residual_risk=dict(data.get("residual_risk") or {}),
            openmed_version=str(data.get("openmed_version", "")),
            manifest_hash=str(data.get("manifest_hash", "")),
            document_length=int(data.get("document_length", 0)),
            input_hash=str(data.get("input_hash", "")),
            deidentified_text_hash=str(data.get("deidentified_text_hash", "")),
            repro_hash=str(data.get("repro_hash", "")),
            signature=(
                AuditSignature.from_dict(signature_data)
                if isinstance(signature_data, Mapping)
                else None
            ),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "AuditReport":
        return cls.from_dict(json.loads(data))

    def sign(self, key: bytes | str, *, key_id: str = "release") -> "AuditReport":
        """Sign the report with a release key and return ``self``."""
        self.repro_hash = self.recompute_repro_hash()
        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        signature = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        self.signature = AuditSignature(
            key_id=key_id,
            algorithm=_SIGNATURE_ALGORITHM,
            value=signature,
        )
        return self

    def verify(
        self,
        key: bytes | str,
        *,
        original_text: str | None = None,
        deidentified_text: str | None = None,
    ) -> bool:
        """Verify the signature and reproducibility hash."""
        if original_text is not None and hash_text(original_text) != self.input_hash:
            return False
        if (
            deidentified_text is not None
            and hash_text(deidentified_text) != self.deidentified_text_hash
        ):
            return False
        if self.recompute_repro_hash() != self.repro_hash:
            return False
        if self.signature is None or self.signature.algorithm != _SIGNATURE_ALGORITHM:
            return False

        message = _canonical_json(
            self._payload(include_repro_hash=True, include_signature=False)
        ).encode("utf-8")
        expected = hmac.new(_key_bytes(key), message, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature.value)

    def export_review_bundle(self) -> dict[str, Any]:
        """Export reviewable spans and context windows without full text."""
        return {
            "policy": self.policy,
            "document_length": self.document_length,
            "input_hash": self.input_hash,
            "deidentified_text_hash": self.deidentified_text_hash,
            "repro_hash": self.repro_hash,
            "spans": [
                {
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "canonical_label": span.canonical_label,
                    "sources": list(span.sources),
                    "confidence": float(span.confidence),
                    "threshold": float(span.threshold),
                    "action": span.action,
                    "surrogate": span.surrogate,
                    "text_hash": span.text_hash,
                    "context": dict(span.context),
                }
                for span in self.spans
            ],
        }

    def export_review_bundle_json(self) -> str:
        return _canonical_json(self.export_review_bundle())


def recompute_repro_hash(report: AuditReport | Mapping[str, Any]) -> str:
    """Offline helper to recompute a report hash from an object or mapping."""
    if isinstance(report, AuditReport):
        return report.recompute_repro_hash()
    return AuditReport.from_dict(report).recompute_repro_hash()


def verify_repro_hash(report: AuditReport | Mapping[str, Any]) -> bool:
    """Return whether a report's stored hash matches its canonical payload."""
    if isinstance(report, AuditReport):
        return report.recompute_repro_hash() == report.repro_hash
    parsed = AuditReport.from_dict(report)
    return parsed.recompute_repro_hash() == parsed.repro_hash


__all__ = [
    "AuditReport",
    "AuditSignature",
    "AuditSpan",
    "DetectorInfo",
    "hash_text",
    "manifest_hash",
    "recompute_repro_hash",
    "stable_hash",
    "verify_repro_hash",
]
