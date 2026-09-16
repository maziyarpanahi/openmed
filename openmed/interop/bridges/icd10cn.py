"""Exact local bridge between ICD-10-CN extensions and ICD-10 codes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openmed.clinical.grounding.crosswalk import (
    CrosswalkResource,
    load_crosswalk,
    load_default_crosswalks,
)

__all__ = [
    "ICD10CNBridge",
    "ICD10CNMapping",
    "load_icd10cn_crosswalk",
    "map_icd10_to_icd10cn",
    "map_icd10cn_code",
]

_DEFAULT_RESOURCE_NAME = "openmed-icd10cn-icd10-starter"


@dataclass(frozen=True)
class ICD10CNMapping:
    """One exact ICD-10-CN to international ICD-10 mapping."""

    source_code: str
    target_code: str
    target_display: str
    resource_name: str
    resource_version: str


class ICD10CNBridge:
    """Bidirectional exact-code view of a validated local crosswalk."""

    def __init__(self, resource: CrosswalkResource) -> None:
        if not isinstance(resource, CrosswalkResource):
            raise TypeError("resource must be a CrosswalkResource")
        invalid = [
            entry
            for entry in resource.entries
            if entry.source_system.casefold() != "icd-10-cn"
            or entry.target_system != "ICD10"
        ]
        if invalid:
            raise ValueError(
                "ICD10CNBridge resource entries must map ICD-10-CN to ICD10"
            )
        self.resource = resource

    def to_icd10(self, source_code: str) -> ICD10CNMapping | None:
        """Return the exact international mapping for ``source_code``."""

        matches = self.resource.entries_for_source_code(
            source_code, source_system="ICD-10-CN"
        )
        if not matches:
            return None
        entry = matches[0]
        return self._mapping(entry.source_code, entry.target_code, entry.target_display)

    def from_icd10(self, target_code: str) -> tuple[ICD10CNMapping, ...]:
        """Return every exact ICD-10-CN extension for an ICD-10 code."""

        return tuple(
            self._mapping(
                entry.source_code,
                entry.target_code,
                entry.target_display,
            )
            for entry in self.resource.entries_for_target_code(
                target_code, target_system="ICD10"
            )
        )

    def _mapping(
        self, source_code: str, target_code: str, target_display: str
    ) -> ICD10CNMapping:
        return ICD10CNMapping(
            source_code=source_code,
            target_code=target_code,
            target_display=target_display,
            resource_name=self.resource.name,
            resource_version=self.resource.resource_version,
        )


def load_icd10cn_crosswalk(
    path: str | Path | None = None,
) -> CrosswalkResource:
    """Load a caller crosswalk or the bundled permissive starter table."""

    if path is not None:
        return load_crosswalk(path)
    for resource in load_default_crosswalks():
        if resource.name == _DEFAULT_RESOURCE_NAME:
            return resource
    raise RuntimeError("bundled ICD-10-CN crosswalk is unavailable")


def map_icd10cn_code(
    source_code: str,
    *,
    resource: CrosswalkResource | None = None,
) -> ICD10CNMapping | None:
    """Map one ICD-10-CN code exactly using only local data."""

    return ICD10CNBridge(resource or load_icd10cn_crosswalk()).to_icd10(source_code)


def map_icd10_to_icd10cn(
    target_code: str,
    *,
    resource: CrosswalkResource | None = None,
) -> tuple[ICD10CNMapping, ...]:
    """Return exact reverse mappings for one international ICD-10 code."""

    return ICD10CNBridge(resource or load_icd10cn_crosswalk()).from_icd10(target_code)
