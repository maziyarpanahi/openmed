"""Strict offline representation of a capability's validation evidence."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from openmed.core.schemas import schema_fingerprint
from openmed.eval.journey_release import _canonical_json, _reject_duplicate_keys


class EvidencePackError(ValueError):
    """A malformed pack, with a controlled reason code."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    path = files("openmed.core.schemas.json").joinpath(
        "validation_evidence_pack.schema.json"
    )
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True, init=False)
class ValidationEvidencePack:
    """Immutable validated wire representation, retaining signed array order."""

    _canonical: str = field(repr=False)

    def __init__(self, payload: Mapping[str, Any]):
        try:
            canonical = _canonical_json(payload)
            data = json.loads(canonical)
            if not _validator().is_valid(data):
                raise EvidencePackError("invalid_pack_schema")
            for collection, key in (
                ("lanes", "id"),
                ("frozen_inputs", "id"),
                ("licenses", "asset_id"),
                ("exceptions", "code"),
            ):
                if len({item[key] for item in data[collection]}) != len(
                    data[collection]
                ):
                    raise EvidencePackError("duplicate_pack_identifier")
            lanes = {lane["id"] for lane in data["lanes"]}
            if any(item["lane"] not in lanes for item in data["exceptions"]):
                raise EvidencePackError("unknown_exception_lane")
            claims = set(data["claims"])
            linked = {claim for lane in data["lanes"] for claim in lane["claims"]}
            if linked != claims:
                raise EvidencePackError("unbound_pack_claim")
            for stamp in [
                data["generated_at"],
                *(item["expires_at"] for item in data["exceptions"]),
            ]:
                datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")
        except EvidencePackError:
            raise
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise EvidencePackError("invalid_pack_input") from None
        object.__setattr__(self, "_canonical", canonical)

    @classmethod
    def from_json(cls, raw: str | bytes) -> ValidationEvidencePack:
        """Parse JSON while refusing duplicate keys, non-finite numbers and bad types."""
        try:
            data = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        except (ValueError, TypeError, RecursionError):
            raise EvidencePackError("invalid_pack_json") from None
        return cls(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the complete pack, including its signature."""
        return json.loads(self._canonical)

    def to_json(self) -> str:
        """Return the canonical wire representation."""
        return self._canonical

    def signing_payload(self) -> dict[str, Any]:
        """Return the exact fields bound by the pack digest and signature."""
        payload = self.to_dict()
        del payload["signature"]
        return payload

    def digest(self) -> str:
        """Compute the SHA-256 fingerprint over the canonical unsigned payload."""
        return schema_fingerprint(self.signing_payload())

    def lane_blockers(self) -> tuple[str, ...]:
        """Report missing metrics or evidence and non-success states for mandatory lanes."""
        blockers = []
        for lane in self.to_dict()["lanes"]:
            if not lane["mandatory"]:
                continue
            code = lane["id"]
            if lane["state"] != "success":
                blockers.append(f"state:{code}:{lane['state']}")
            if lane["report_digest"] is None:
                blockers.append(f"missing:{code}:report_digest")
            blockers.extend(
                f"missing:{code}:{metric}"
                for metric in sorted(
                    set(lane["required_metrics"]) - lane["metrics"].keys()
                )
            )
        return tuple(blockers)
