"""Cross-modal subject-surrogate consistency for structured records.

The caller supplies the subject key column and, when note aliases are present,
the already-matched PII entities for each subject. No identity inference occurs
here. Raw identifiers exist only while resolving HMAC vault keys and rewriting
the local records; the manifest contains hashes and aggregate counts only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openmed.core.pii import PIIEntity
from openmed.core.surrogate_vault import SubjectResolutionError, SurrogateVault
from openmed.interop._pii import resolve_subject_surrogate


@dataclass(frozen=True)
class CrossModalConsistencyReport:
    """Raw-free result of comparing one note and table subject surrogate."""

    checked: int
    matched: int

    @property
    def passed(self) -> bool:
        """Return whether every compared surrogate matched."""

        return self.checked == self.matched

    def to_dict(self) -> dict[str, Any]:
        """Serialize aggregate consistency evidence without identifiers."""

        return {
            "checked": self.checked,
            "matched": self.matched,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SubjectConsistencyManifest:
    """HMAC-only audit metadata for a subject-column transformation."""

    key_id: str
    subject_column: str
    row_count: int
    subject_count: int
    source_hashes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the manifest without raw identifiers or mappings."""

        return {
            "key_id": self.key_id,
            "subject_column": self.subject_column,
            "row_count": self.row_count,
            "subject_count": self.subject_count,
            "source_hashes": list(self.source_hashes),
        }


@dataclass(frozen=True)
class SubjectTableDeidentificationResult:
    """Structured records with one consistently pseudonymized subject column."""

    records: tuple[dict[str, Any], ...]
    manifest: SubjectConsistencyManifest = field(repr=False)


def deidentify_subject_column(
    records: Sequence[Mapping[str, Any]],
    *,
    subject_column: str,
    vault: SurrogateVault,
    note_identifiers_by_subject: Mapping[Any, Sequence[PIIEntity | Mapping[str, Any]]]
    | None = None,
    lang: str = "en",
) -> SubjectTableDeidentificationResult:
    """Replace a structured subject key through the shared surrogate vault.

    ``note_identifiers_by_subject`` is an optional in-memory bridge from each
    structured key to PII entities that the caller has already matched to that
    subject. It is consumed only during resolution and is never returned.
    """

    if not isinstance(subject_column, str) or not subject_column:
        raise ValueError("subject_column must be a non-empty string")
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence of row mappings")

    surrogate_by_subject: dict[str, str] = {}
    subject_hash_by_surrogate: dict[str, str] = {}
    rewritten: list[dict[str, Any]] = []

    for row_index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise TypeError("every record must be a row mapping")
        if subject_column not in row:
            raise ValueError(f"subject column is missing at row {row_index}")
        raw_subject = row[subject_column]
        source_identifier = _subject_identifier(raw_subject, row_index=row_index)

        surrogate = surrogate_by_subject.get(source_identifier)
        if surrogate is None:
            note_identifiers = _note_identifiers_for(
                note_identifiers_by_subject,
                raw_subject,
                source_identifier,
            )
            if note_identifiers:
                surrogate = resolve_subject_surrogate(
                    note_identifiers,
                    structured_identifier=source_identifier,
                    vault=vault,
                    lang=lang,
                )
            else:
                surrogate = vault.resolve_subject(source_identifier)

            source_hash = vault.subject_key_for(source_identifier).text_hash
            owner = subject_hash_by_surrogate.get(surrogate)
            if owner is not None and owner != source_hash:
                raise SubjectResolutionError(
                    "distinct subjects resolved to the same surrogate"
                )
            subject_hash_by_surrogate[surrogate] = source_hash
            surrogate_by_subject[source_identifier] = surrogate

        released = dict(row)
        released[subject_column] = surrogate
        rewritten.append(released)

    manifest = SubjectConsistencyManifest(
        key_id=vault.current_key_id,
        subject_column=subject_column,
        row_count=len(rewritten),
        subject_count=len(surrogate_by_subject),
        source_hashes=tuple(sorted(subject_hash_by_surrogate.values())),
    )
    return SubjectTableDeidentificationResult(tuple(rewritten), manifest)


def verify_cross_modal_consistency(
    note_surrogate: str,
    table_surrogate: str,
) -> CrossModalConsistencyReport:
    """Compare released note and table surrogates without retaining them."""

    if not note_surrogate or not table_surrogate:
        raise ValueError("note and table surrogates must be non-empty")
    matched = int(note_surrogate == table_surrogate)
    return CrossModalConsistencyReport(checked=1, matched=matched)


def assert_cross_modal_consistency(
    note_surrogate: str,
    table_surrogate: str,
) -> CrossModalConsistencyReport:
    """Return consistency evidence or fail without echoing either value."""

    report = verify_cross_modal_consistency(note_surrogate, table_surrogate)
    if not report.passed:
        raise SubjectResolutionError("note and table subject surrogates do not match")
    return report


def _subject_identifier(value: Any, *, row_index: int) -> str:
    if isinstance(value, bool):
        raise ValueError(f"subject identifier is invalid at row {row_index}")
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(
                f"subject identifier is invalid at row {row_index}"
            ) from None
    elif isinstance(value, int):
        value = str(value)
    if not isinstance(value, str) or not value:
        raise ValueError(f"subject identifier is invalid at row {row_index}")
    return value


def _note_identifiers_for(
    note_identifiers_by_subject: Mapping[Any, Sequence[PIIEntity | Mapping[str, Any]]]
    | None,
    raw_subject: Any,
    source_identifier: str,
) -> Sequence[PIIEntity | Mapping[str, Any]]:
    if note_identifiers_by_subject is None:
        return ()
    try:
        matched = note_identifiers_by_subject.get(raw_subject)
    except TypeError:
        matched = None
    if matched is None and raw_subject != source_identifier:
        matched = note_identifiers_by_subject.get(source_identifier)
    return matched or ()


__all__ = [
    "CrossModalConsistencyReport",
    "SubjectConsistencyManifest",
    "SubjectTableDeidentificationResult",
    "assert_cross_modal_consistency",
    "deidentify_subject_column",
    "verify_cross_modal_consistency",
]
