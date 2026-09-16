"""User-key-gated local adapters for restricted clinical vocabularies.

No terminology data, credentials, or download path is bundled here. Callers
must explicitly provide both a local normalized alias table and proof that they
hold the applicable license. The key is checked and immediately discarded; it
is never retained, logged, serialized, or read from process credentials.

SNOMED CT use remains subject to SNOMED International affiliate and member-
territory terms. UMLS use remains subject to the UMLS Metathesaurus license.
This adapter performs local assistive matching only and makes no licensing
determination for the caller.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .matcher import LexicalConcept, VocabularyTerms
from .vocab import RestrictedVocabularyError

__all__ = [
    "RESTRICTED_SYSTEM_URIS",
    "UserKeyVocabularyLoader",
]

RESTRICTED_SYSTEM_URIS: Mapping[str, str] = {
    "umls": "http://terminology.hl7.org/CodeSystem/umls",
    "snomed": "http://snomed.info/sct",
}

_SYSTEM_ALIASES = {
    "umls": "umls",
    "snomed": "snomed",
    "snomed-ct": "snomed",
    "snomedct": "snomed",
    "sct": "snomed",
}
_SUPPORTED_SUFFIXES = frozenset({".csv", ".jsonl", ".tsv"})


class UserKeyVocabularyLoader:
    """Load a caller-normalized UMLS or SNOMED alias table from local storage.

    The table must contain ``code`` (or ``concept_id``), ``preferred_term`` (or
    ``display``), and optional ``synonyms``/``aliases`` fields. Delimited files
    use ``|`` between aliases; JSONL rows may use either a string or a list.

    Args:
        system: ``"umls"`` or ``"snomed"`` (common SNOMED aliases accepted).
        path: Local CSV, TSV, or JSONL alias table. Direct vendor release dumps
            are intentionally not parsed; normalize them inside the caller's
            licensed environment.
        license_key: Explicit non-empty proof-of-license token. It is validated
            and discarded during construction.

    Raises:
        RestrictedVocabularyError: If the system, key, or local path is invalid.
    """

    redistributable = False
    restricted_license = True

    def __init__(
        self,
        system: str,
        path: str | Path,
        *,
        license_key: str,
    ) -> None:
        normalized = _normalize_restricted_system(system)
        if not isinstance(license_key, str) or not license_key.strip():
            raise RestrictedVocabularyError(
                f"{normalized.upper()} requires an explicit user-supplied license "
                "key and local normalized alias table; nothing is bundled or "
                "downloaded."
            )
        resolved_path = Path(path).expanduser()
        if not resolved_path.is_file():
            raise RestrictedVocabularyError(
                f"User-supplied {normalized.upper()} alias table does not exist: "
                f"{resolved_path}"
            )
        if resolved_path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
            raise RestrictedVocabularyError(
                "Restricted vocabulary alias tables must be CSV, TSV, or JSONL."
            )

        self.system = normalized
        self.system_uri = RESTRICTED_SYSTEM_URIS[normalized]
        self.path = resolved_path
        self.content_hash = _sha256(resolved_path)

    def load(self) -> VocabularyTerms:
        """Return local vocabulary terms without any network access."""

        terms: dict[str, list[LexicalConcept]] = defaultdict(list)
        for row_number, row in enumerate(_read_rows(self.path), start=1):
            code = _first_text(row, "code", "concept_id", "cui", "sctid")
            display = _first_text(
                row,
                "preferred_term",
                "display",
                "concept_name",
                "term",
            )
            if not code or not display:
                raise RestrictedVocabularyError(
                    f"{self.path}:{row_number} requires code and preferred_term."
                )
            concept = LexicalConcept(
                system_uri=self.system_uri,
                code=code,
                display=display,
                metadata={"source": "user-supplied-local"},
            )
            for alias in _aliases(row, display):
                terms[alias].append(concept)
        if not terms:
            raise RestrictedVocabularyError(
                f"User-supplied {self.system.upper()} alias table is empty."
            )
        return dict(terms)


def _normalize_restricted_system(system: str) -> str:
    if not isinstance(system, str):
        raise RestrictedVocabularyError("restricted vocabulary system must be text")
    normalized = system.strip().casefold().replace("_", "-")
    try:
        return _SYSTEM_ALIASES[normalized]
    except KeyError:
        raise RestrictedVocabularyError(
            "User-key vocabulary adapters support only UMLS and SNOMED CT. "
            "CPT remains outside the in-process grounding API."
        ) from None


def _read_rows(path: Path) -> Iterable[Mapping[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RestrictedVocabularyError(
                        f"{path}:{line_number} is not valid JSON."
                    ) from exc
                if not isinstance(row, Mapping):
                    raise RestrictedVocabularyError(
                        f"{path}:{line_number} must contain a JSON object."
                    )
                yield row
        return

    delimiter = "\t" if path.suffix.casefold() == ".tsv" else ","
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter=delimiter)


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _aliases(row: Mapping[str, Any], display: str) -> tuple[str, ...]:
    values: list[str] = [display]
    for key in ("synonyms", "aliases", "alias"):
        raw = row.get(key)
        if raw is None:
            continue
        if isinstance(raw, str):
            aliases = raw.split("|")
        elif isinstance(raw, Iterable):
            aliases = [str(value) for value in raw]
        else:
            aliases = [str(raw)]
        values.extend(alias.strip() for alias in aliases if alias.strip())
    return tuple(dict.fromkeys(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"
