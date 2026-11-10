"""Deterministic SPDX identifier normalization for synthetic lineage.

Synthetic-dataset lineage manifests need stable handling of common SPDX
short-identifier spelling and casing without guessing unknown licenses. A
small committed allowlist of permissive identifiers and the ``LicenseRef-``
production from the SPDX specification are the only values this module
normalizes; everything else stays unknown and is never promoted to
permissive. This is not an SPDX expression parser and it is not legal
advice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "PERMISSIVE_SPDX_IDENTIFIERS",
    "SPDX_IDENTIFIER_SCHEMA_VERSION",
    "SpdxIdentifier",
    "SpdxIdentifierStatus",
    "normalize_spdx_identifier",
]

SPDX_IDENTIFIER_SCHEMA_VERSION: Final = 1

# Committed, permissive-only allowlist. Deliberately small: extending it is a
# reviewed policy change, never an inference from input data.
PERMISSIVE_SPDX_IDENTIFIERS: Final[frozenset[str]] = frozenset(
    {
        "0BSD",
        "Apache-2.0",
        "BlueOak-1.0.0",
        "BSD-2-Clause",
        "BSD-2-Clause-Patent",
        "BSD-3-Clause",
        "BSL-1.0",
        "ISC",
        "MIT",
        "MPL-2.0",
        "Python-2.0",
        "Unlicense",
        "Zlib",
    }
)

# Deprecated SPDX identifiers with a documented, permissive successor. The
# NetBSD variant was deprecated in SPDX license list 3.9 and SPDX documents
# it as a match to BSD-2-Clause. No other alias is inferred.
_DEPRECATED_ALIASES: Final[dict[str, str]] = {
    "BSD-2-Clause-NetBSD": "BSD-2-Clause",
}

_LICENSEREF_PREFIX: Final = "LicenseRef-"
_IDENTIFIER_RE: Final = re.compile(r"^[A-Za-z0-9.-]+$")
_LICENSEREF_RE: Final = re.compile(r"^LicenseRef-[A-Za-z0-9.-]+$")
_EXPRESSION_TOKENS: Final = frozenset({"AND", "OR", "WITH"})


class SpdxIdentifierStatus(str, Enum):
    """Closed set of normalization outcomes for one candidate value."""

    CANONICAL = "canonical"
    NORMALIZED = "normalized"
    DEPRECATED_ALIAS = "deprecated_alias"
    UNKNOWN = "unknown"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class SpdxIdentifier:
    """One stable normalization outcome, with no free-form text added."""

    schema_version: int
    status: SpdxIdentifierStatus
    normalized: str | None
    is_permissive: bool
    category: str

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary in fixed field order."""
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "normalized": self.normalized,
            "is_permissive": self.is_permissive,
            "category": self.category,
        }

    def to_json(self) -> str:
        """Serialize with deterministic key order and no insignificant space."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=False,
            separators=(",", ":"),
        )


def _result(
    status: SpdxIdentifierStatus,
    normalized: str | None,
    is_permissive: bool,
    category: str,
) -> SpdxIdentifier:
    return SpdxIdentifier(
        schema_version=SPDX_IDENTIFIER_SCHEMA_VERSION,
        status=status,
        normalized=normalized,
        is_permissive=is_permissive,
        category=category,
    )


def normalize_spdx_identifier(value: Any) -> SpdxIdentifier:
    """Classify one candidate SPDX identifier deterministically.

    Only committed allowlist members, their case and whitespace variants,
    the committed deprecated-alias table, and well-formed ``LicenseRef-``
    values normalize to a canonical form. Syntactically valid but
    unallowlisted identifiers stay unknown with ``is_permissive=False``;
    values that are not SPDX short identifiers are malformed. No network
    lookup, expression parsing, or compatibility decision is performed.
    """
    if type(value) is not str:
        return _result(
            SpdxIdentifierStatus.MALFORMED,
            None,
            False,
            "spdx_identifier_malformed_type",
        )
    trimmed = value.strip()
    if not trimmed:
        return _result(
            SpdxIdentifierStatus.MALFORMED,
            None,
            False,
            "spdx_identifier_malformed_empty",
        )
    whitespace_changed = trimmed != value
    if any(
        token in _EXPRESSION_TOKENS for token in trimmed.split()
    ) or trimmed.endswith("+"):
        return _result(
            SpdxIdentifierStatus.MALFORMED,
            None,
            False,
            "spdx_identifier_malformed_expression",
        )
    if trimmed.startswith(_LICENSEREF_PREFIX):
        if not _LICENSEREF_RE.match(trimmed):
            return _result(
                SpdxIdentifierStatus.MALFORMED,
                None,
                False,
                "spdx_identifier_malformed_characters",
            )
        # LicenseRef identifiers are case sensitive per the SPDX grammar, so
        # only surrounding whitespace is normalized and the value is never
        # treated as permissive; the caller owns that decision.
        if whitespace_changed:
            return _result(
                SpdxIdentifierStatus.NORMALIZED,
                trimmed,
                False,
                "spdx_identifier_normalized_licenseref_whitespace",
            )
        return _result(
            SpdxIdentifierStatus.CANONICAL,
            trimmed,
            False,
            "spdx_identifier_canonical_licenseref",
        )
    if not _IDENTIFIER_RE.match(trimmed):
        return _result(
            SpdxIdentifierStatus.MALFORMED,
            None,
            False,
            "spdx_identifier_malformed_characters",
        )

    if trimmed in PERMISSIVE_SPDX_IDENTIFIERS:
        if whitespace_changed:
            return _result(
                SpdxIdentifierStatus.NORMALIZED,
                trimmed,
                True,
                "spdx_identifier_normalized_whitespace",
            )
        return _result(
            SpdxIdentifierStatus.CANONICAL,
            trimmed,
            True,
            "spdx_identifier_canonical",
        )
    case_folded = trimmed.casefold()
    alias = _DEPRECATED_ALIASES.get(trimmed)
    if alias is None:
        for deprecated, successor in _DEPRECATED_ALIASES.items():
            if deprecated.casefold() == case_folded:
                alias = successor
                break
    if alias is not None:
        return _result(
            SpdxIdentifierStatus.DEPRECATED_ALIAS,
            alias,
            alias in PERMISSIVE_SPDX_IDENTIFIERS,
            "spdx_identifier_deprecated_alias",
        )
    for allowed in PERMISSIVE_SPDX_IDENTIFIERS:
        if allowed.casefold() == case_folded:
            return _result(
                SpdxIdentifierStatus.NORMALIZED,
                allowed,
                True,
                "spdx_identifier_normalized_case",
            )
    return _result(
        SpdxIdentifierStatus.UNKNOWN,
        None,
        False,
        "spdx_identifier_unknown",
    )
