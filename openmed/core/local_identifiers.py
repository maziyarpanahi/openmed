"""Deterministic detection of usernames embedded in local filesystem paths.

Tracebacks and debug records often contain absolute paths such as
``/home/fixture_user/project`` or ``C:\\Users\\fixture_user\\project``.
The path itself is useful context, but the home-directory component can
identify the machine user.  This module returns the username component as a
normal :class:`~openmed.processing.outputs.EntityPrediction` so callers can
pass it through the existing anonymizer without a model or network request.

Only absolute, platform-shaped paths are considered.  Relative fixture paths
and generic system roots are deliberately left untouched.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Final

from ..processing.outputs import EntityPrediction
from .labels import USERNAME, normalize_label

LOCAL_IDENTIFIER_SOURCE: Final = "local_path_identifier"
LOCAL_IDENTIFIER_LABEL: Final = USERNAME

# Account names that identify a shared or provisioned account rather than a
# person.  These are common in paths emitted by installers and CI fixtures;
# suppressing them prevents a generic system path from becoming a false PII
# finding while preserving ordinary user names such as ``admin``.
_NON_PERSONAL_ACCOUNT_NAMES: Final = frozenset(
    {
        "all users",
        "default",
        "default user",
        "guest",
        "nobody",
        "public",
        "root",
        "shared",
    }
)

# Usernames are intentionally conservative: a path separator, whitespace, or
# a log delimiter ends the segment.  This covers the normal POSIX and Windows
# account-name shapes without consuming prose after a path at the end of a
# trace line.
_USERNAME_PATTERN = r"[^\s/\\\"'`<>|()[\]{};,!?]+"

# A POSIX path may contain an intermediate directory before a conventional
# home/profile marker (for example ``/srv/users/fixture_user``).  Requiring
# the absolute root and a non-path boundary prevents ``fixtures/Users/...``
# from being mistaken for a local absolute path.
_POSIX_PATH_PATTERN = re.compile(
    rf"""
    (?<![\w./\\:-])
    /
    (?:[^/\\\s\"'`<>|:]+/)*
    (?P<root>home|users|profiles?)
    /
    (?P<username>{_USERNAME_PATTERN})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Windows paths can use either separator and can place the home marker below
# another directory on the drive.  ``Documents and Settings`` is retained for
# older Windows traces that predate the ``Users`` directory.
_WINDOWS_PATH_PATTERN = re.compile(
    rf"""
    (?<![\w.-])
    [A-Za-z]:[\\/]
    (?:[^/\\\s\"'`<>|:]+[\\/])*
    (?P<root>users|profiles?|documents\ and\ settings)
    [\\/]
    (?P<username>{_USERNAME_PATTERN})
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A rooted Windows path may omit the drive letter when it is copied from a
# process-local trace.  This also avoids treating a relative ``foo\\Users``
# fixture path as absolute.
_ROOTED_WINDOWS_PATH_PATTERN = re.compile(
    rf"""
    (?<![\w./\\-])
    \\
    (?:[^/\\\s\"'`<>|:]+\\)*
    (?P<root>users|profiles?|documents\ and\ settings)
    \\
    (?P<username>{_USERNAME_PATTERN})
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PATH_PATTERNS: Final = (
    ("posix_home", _POSIX_PATH_PATTERN),
    ("windows_home", _WINDOWS_PATH_PATTERN),
    ("rooted_windows_home", _ROOTED_WINDOWS_PATH_PATTERN),
)


def _is_personal_account(username: str) -> bool:
    """Return whether *username* is worth treating as a direct identifier."""

    return username.casefold() not in _NON_PERSONAL_ACCOUNT_NAMES


def _iter_matches(text: str) -> Iterable[tuple[int, int, str]]:
    """Yield unique username offsets and path kinds in source order."""

    matches: dict[tuple[int, int], str] = {}
    for path_kind, pattern in _PATH_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span("username")
            username = match.group("username")
            if start >= end or not _is_personal_account(username):
                continue
            matches.setdefault((start, end), path_kind)

    for (start, end), path_kind in sorted(matches.items()):
        yield start, end, path_kind


def detect_local_identifiers(
    text: str,
    *,
    lang: str = "en",
) -> list[EntityPrediction]:
    """Return username entities found in absolute local paths.

    Args:
        text: Text containing logs, traces, or other path-bearing content.
        lang: Language code used by the shared label normalizer.  The detector
            emits the same canonical ``USERNAME`` label for every language,
            but accepting the argument keeps the result compatible with other
            local recognizers.

    Returns:
        Entity predictions whose offsets refer to *text*.  The entity surface
        is the username segment only; path context is not redacted by this
        detector.  Metadata contains detector provenance and no source text.

    Raises:
        TypeError: If *text* is not a string.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    canonical_label = normalize_label(LOCAL_IDENTIFIER_LABEL, lang=lang)
    entities: list[EntityPrediction] = []
    for start, end, path_kind in _iter_matches(text):
        entities.append(
            EntityPrediction(
                text=text[start:end],
                label=canonical_label,
                confidence=1.0,
                start=start,
                end=end,
                metadata={
                    "source": LOCAL_IDENTIFIER_SOURCE,
                    "detector": LOCAL_IDENTIFIER_SOURCE,
                    "canonical_label": canonical_label,
                    "normalized_label": canonical_label,
                    "path_kind": path_kind,
                },
            )
        )
    return entities


class LocalIdentifierDetector:
    """Callable adapter exposing the local-path detector as a recognizer."""

    def detect_entities(
        self,
        text: str,
        *,
        lang: str = "en",
    ) -> list[EntityPrediction]:
        """Return local-path username predictions for *text*."""

        return detect_local_identifiers(text, lang=lang)

    def detect(self, text: str, *, lang: str = "en") -> list[EntityPrediction]:
        """Alias for :meth:`detect_entities` used by recognizer callers."""

        return self.detect_entities(text, lang=lang)

    def __call__(self, text: str, *, lang: str = "en") -> list[EntityPrediction]:
        """Return local-path username predictions for *text*."""

        return self.detect_entities(text, lang=lang)


detect_local_path_identifiers = detect_local_identifiers


__all__ = [
    "LOCAL_IDENTIFIER_LABEL",
    "LOCAL_IDENTIFIER_SOURCE",
    "LocalIdentifierDetector",
    "detect_local_identifiers",
    "detect_local_path_identifiers",
]
