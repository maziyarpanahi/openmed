"""License metadata and source-tree gates for user-supplied terminology."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TERMINOLOGY_REDISTRIBUTION_PERMITTED = "permitted"
TERMINOLOGY_REDISTRIBUTION_RESTRICTED = "restricted"
TERMINOLOGY_REDISTRIBUTION_VALUES = frozenset(
    {
        TERMINOLOGY_REDISTRIBUTION_PERMITTED,
        TERMINOLOGY_REDISTRIBUTION_RESTRICTED,
    }
)


class RestrictedTerminologyLocationError(ValueError):
    """Raised when restricted terminology is placed inside the repository."""


@dataclass(frozen=True)
class TerminologyLicense:
    """Caller-declared license metadata for a local terminology dictionary.

    ``redistribution`` is intentionally a small gate rather than free-form
    prose. Restricted dictionaries must remain outside the source tree and are
    read directly from the caller-provided path; OpenMed never copies them into
    fixtures, package data, or its cache.
    """

    license_id: str
    redistribution: str
    notes: str = ""

    def __post_init__(self) -> None:
        license_id = self.license_id.strip()
        redistribution = self.redistribution.strip().casefold()
        if not license_id:
            raise ValueError("terminology license_id must not be blank")
        if redistribution not in TERMINOLOGY_REDISTRIBUTION_VALUES:
            raise ValueError(
                "terminology redistribution must be one of "
                f"{sorted(TERMINOLOGY_REDISTRIBUTION_VALUES)!r}"
            )
        object.__setattr__(self, "license_id", license_id)
        object.__setattr__(self, "redistribution", redistribution)

    @property
    def restricted(self) -> bool:
        """Return whether the dictionary must remain outside the repository."""

        return self.redistribution == TERMINOLOGY_REDISTRIBUTION_RESTRICTED

    def to_dict(self) -> dict[str, str | bool]:
        """Return provenance-safe license metadata without a local path."""

        return {
            "license_id": self.license_id,
            "redistribution": self.redistribution,
            "restricted": self.restricted,
            "notes": self.notes,
        }


def validate_terminology_source_path(
    path: str | Path,
    license_metadata: TerminologyLicense,
    *,
    repository_root: str | Path | None = None,
) -> Path:
    """Validate and return an external terminology source path.

    Restricted sources are rejected when they resolve anywhere below the
    repository root. The function performs no copy and intentionally returns
    only the original resolved path for direct, read-only loading.
    """

    resolved = Path(path).expanduser().resolve()
    root = (
        Path(repository_root).expanduser().resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if license_metadata.restricted and resolved.is_relative_to(root):
        raise RestrictedTerminologyLocationError(
            "restricted terminology dictionaries must be stored outside the "
            "OpenMed repository"
        )
    return resolved


__all__ = [
    "RestrictedTerminologyLocationError",
    "TERMINOLOGY_REDISTRIBUTION_PERMITTED",
    "TERMINOLOGY_REDISTRIBUTION_RESTRICTED",
    "TERMINOLOGY_REDISTRIBUTION_VALUES",
    "TerminologyLicense",
    "validate_terminology_source_path",
]
