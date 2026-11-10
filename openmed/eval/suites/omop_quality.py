"""Frozen synthetic-cohort evidence for OMOP projection reconciliation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from openmed.clinical.journey_contracts import canonical_digest
from openmed.interop.omop.quality import (
    OmopProjectionAggregate,
    OmopQualityConflictError,
    OmopQualityInput,
    OmopReconciliation,
    reconcile_omop_aggregates,
)


@dataclass(frozen=True, slots=True)
class FrozenOmopQualityFixture:
    """One digest-locked synthetic cohort and two aggregate ETL snapshots."""

    quality_input: OmopQualityInput
    openmed: OmopProjectionAggregate
    reference: OmopProjectionAggregate
    expected_table_deltas: Mapping[str, int]
    expected_mapped_coverage_delta_ppm: int
    semantic_difference_codes: tuple[str, ...]
    provenance: Mapping[str, str]
    fixture_digest: str

    def __post_init__(self) -> None:
        expected = MappingProxyType(
            {
                str(key): int(value)
                for key, value in sorted(self.expected_table_deltas.items())
            }
        )
        provenance = MappingProxyType(
            {str(key): str(value) for key, value in sorted(self.provenance.items())}
        )
        object.__setattr__(self, "expected_table_deltas", expected)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "semantic_difference_codes",
            tuple(sorted(set(self.semantic_difference_codes))),
        )
        if self.quality_input.projection_digest != self.openmed.snapshot_digest:
            raise OmopQualityConflictError("fixture projection digest differs")
        if (
            self.quality_input.reference_snapshot_digest
            != self.reference.snapshot_digest
        ):
            raise OmopQualityConflictError("fixture reference digest differs")
        if self.quality_input.reference_license != self.reference.license:
            raise OmopQualityConflictError("fixture reference license differs")
        expected_digest = canonical_digest(self._payload())
        if self.fixture_digest != expected_digest:
            raise OmopQualityConflictError("frozen fixture digest differs")

    def _payload(self) -> dict[str, Any]:
        return {
            "expected_mapped_coverage_delta_ppm": (
                self.expected_mapped_coverage_delta_ppm
            ),
            "expected_table_deltas": dict(self.expected_table_deltas),
            "openmed": self.openmed.to_dict(),
            "provenance": dict(self.provenance),
            "quality_input": self.quality_input.to_dict(),
            "reference": self.reference.to_dict(),
            "semantic_difference_codes": list(self.semantic_difference_codes),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the complete persisted fixture."""

        return self._payload() | {"fixture_digest": self.fixture_digest}

    def reconcile(self) -> OmopReconciliation:
        """Reproduce and verify every documented aggregate delta."""

        result = reconcile_omop_aggregates(
            self.openmed,
            self.reference,
            expected_table_deltas=self.expected_table_deltas,
            expected_mapped_coverage_delta_ppm=(
                self.expected_mapped_coverage_delta_ppm
            ),
            semantic_difference_codes=self.semantic_difference_codes,
        )
        if not result.expectations_met:
            raise OmopQualityConflictError("frozen fixture deltas drifted")
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenOmopQualityFixture":
        """Restore and verify one frozen synthetic fixture."""

        if not isinstance(value, Mapping):
            raise TypeError("frozen OMOP quality fixture must be an object")
        expected_fields = {
            "expected_mapped_coverage_delta_ppm",
            "expected_table_deltas",
            "fixture_digest",
            "openmed",
            "provenance",
            "quality_input",
            "reference",
            "semantic_difference_codes",
        }
        if set(value) != expected_fields:
            raise ValueError("frozen OMOP quality fixture fields are invalid")
        deltas = value["expected_table_deltas"]
        provenance = value["provenance"]
        differences = value["semantic_difference_codes"]
        if not isinstance(deltas, Mapping) or not isinstance(provenance, Mapping):
            raise TypeError("frozen OMOP quality maps are invalid")
        if isinstance(differences, (str, bytes)) or not isinstance(differences, list):
            raise TypeError("semantic difference codes must be an array")
        return cls(
            quality_input=OmopQualityInput.from_dict(
                _mapping(value["quality_input"], "quality_input")
            ),
            openmed=OmopProjectionAggregate.from_dict(
                _mapping(value["openmed"], "openmed")
            ),
            reference=OmopProjectionAggregate.from_dict(
                _mapping(value["reference"], "reference")
            ),
            expected_table_deltas={
                str(key): _integer(item, "expected table delta")
                for key, item in deltas.items()
            },
            expected_mapped_coverage_delta_ppm=_integer(
                value["expected_mapped_coverage_delta_ppm"],
                "expected_mapped_coverage_delta_ppm",
            ),
            semantic_difference_codes=tuple(str(item) for item in differences),
            provenance={str(key): str(item) for key, item in provenance.items()},
            fixture_digest=str(value["fixture_digest"]),
        )


def load_frozen_omop_quality_fixture(
    path: str | Path,
) -> FrozenOmopQualityFixture:
    """Load and verify a synthetic reconciliation fixture from disk."""

    fixture_path = Path(path)
    try:
        value = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("frozen OMOP quality fixture is unreadable") from exc
    return FrozenOmopQualityFixture.from_dict(
        _mapping(value, "frozen OMOP quality fixture")
    )


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return value


def _integer(value: Any, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer")
    return value


__all__ = [
    "FrozenOmopQualityFixture",
    "load_frozen_omop_quality_fixture",
]
