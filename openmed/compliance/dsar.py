"""GDPR subject-access (Article 15) and erasure-companion (Article 17) export.

Under Article 15 a controller must produce all personal data held about a data
subject. When OpenMed maintains a cross-document surrogate vault, the data held
about a subject is the set of pseudonymous vault entries whose source surface
hashes to one of the subject's known identifier values.

The surrogate vault is keyed by ``(canonical_label, lang, text_hash)`` where
``text_hash`` is an HMAC of a single source surface; it has no subject grouping.
So this helper takes the *caller-provided* known identifier surfaces for the
subject (the values the controller already holds -- name, MRN, DOB, ...),
recomputes each ``text_hash`` through the existing vault configuration, and
selects only the vault entries whose hash matches. The package therefore
contains only data tied to the requested identifiers.

Export generation is recorded to an injected :class:`AuditSink`; the audit
payload carries counts, the subject reference, and content hashes only -- never
raw PHI. The Article 17 companion lists what *would* be erased and is
non-destructive by default.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from openmed.core.audit import stable_hash
from openmed.core.surrogate_vault import SurrogateVault

from .audit_chain import AuditRecord, AuditSink

DSAR_ADVISORY = (
    "This subject-access export is assembled from caller-provided identifiers "
    "and the local surrogate vault. It does not verify the requester's identity "
    "or discover data outside OpenMed; review before release."
)

EXPORT_EVENT = "dsar.export"
ERASURE_PREVIEW_EVENT = "dsar.erasure_preview"


@dataclass(frozen=True)
class SubjectIdentifier:
    """One known identifier value the controller holds for the data subject."""

    surface: str
    label: str
    lang: str = "en"


@dataclass(frozen=True)
class DsarEntry:
    """A pseudonymous vault holding tied to the subject.

    Only the vault's privacy-safe fields are exposed: the canonical label, the
    HMAC ``text_hash``, and the surrogate. The raw source surface is never
    stored in the vault and never appears here.
    """

    canonical_label: str
    lang: str
    text_hash: str
    surrogate: str


@dataclass(frozen=True)
class DsarPackage:
    """A structured Article 15 subject-access package."""

    subject_ref: str
    entries: tuple[DsarEntry, ...]
    categories: tuple[str, ...]
    audit_references: tuple[str, ...] = ()
    audit_record: AuditRecord | None = None


@dataclass(frozen=True)
class ErasurePlan:
    """A non-destructive Article 17 erasure companion."""

    subject_ref: str
    erasable: tuple[DsarEntry, ...]
    executed: bool = False
    audit_record: AuditRecord | None = None


def _subject_ref(text_hashes: Iterable[str]) -> str:
    """A stable, non-PHI reference derived from the subject's identifier hashes."""

    return stable_hash(sorted(set(text_hashes)))


def _matching_entries(
    identifiers: Sequence[SubjectIdentifier],
    vault: SurrogateVault,
) -> tuple[tuple[DsarEntry, ...], frozenset[str]]:
    subject_keys = {
        vault.key_for(ident.surface, label=ident.label, lang=ident.lang)
        for ident in identifiers
    }
    subject_hashes = {key.text_hash for key in subject_keys}
    entries = tuple(
        DsarEntry(
            canonical_label=entry.key.canonical_label,
            lang=entry.key.lang,
            text_hash=entry.key.text_hash,
            surrogate=entry.surrogate,
        )
        for entry in vault.entries()
        if entry.key in subject_keys
    )
    return entries, frozenset(subject_hashes)


def assemble_dsar_package(
    identifiers: Iterable[SubjectIdentifier],
    vault: SurrogateVault,
    *,
    audit_sink: AuditSink | None = None,
    audit_references: Iterable[str] = (),
) -> DsarPackage:
    """Assemble the Article 15 package for the subject's known identifiers.

    Only vault entries whose ``text_hash`` matches one of the recomputed
    identifier hashes are included. When ``audit_sink`` is provided, the export
    is recorded with counts / subject reference / hashes only (no raw PHI).
    """

    identifiers = list(identifiers)
    references = tuple(str(ref) for ref in audit_references)
    entries, subject_hashes = _matching_entries(identifiers, vault)
    subject_ref = _subject_ref(subject_hashes)
    categories = tuple(sorted({entry.canonical_label for entry in entries}))

    record: AuditRecord | None = None
    if audit_sink is not None:
        record = audit_sink.append(
            EXPORT_EVENT,
            {
                "subject_ref": subject_ref,
                "identifier_count": len(identifiers),
                "entry_count": len(entries),
                "categories": list(categories),
                "text_hashes": sorted(entry.text_hash for entry in entries),
                "audit_references": list(references),
            },
        )

    return DsarPackage(
        subject_ref=subject_ref,
        entries=entries,
        categories=categories,
        audit_references=references,
        audit_record=record,
    )


def render_dsar_summary(package: DsarPackage) -> str:
    """Render a deterministic human-readable summary of the package."""

    lines = [
        "GDPR Article 15 subject-access export",
        f"Subject reference: {package.subject_ref}",
        f"Records held: {len(package.entries)}",
        f"Categories: {', '.join(package.categories) or '(none)'}",
    ]
    for entry in package.entries:
        lines.append(
            f"  - {entry.canonical_label} [{entry.lang}] -> surrogate "
            f"{entry.surrogate!r} (hash {entry.text_hash})"
        )
    if package.audit_references:
        lines.append(f"Audit references: {', '.join(package.audit_references)}")
    lines.append(DSAR_ADVISORY)
    return "\n".join(lines)


def plan_erasure(
    package: DsarPackage,
    vault: SurrogateVault,
    *,
    audit_sink: AuditSink | None = None,
) -> ErasurePlan:
    """List what an Article 17 erasure would remove, without deleting anything.

    The companion is non-destructive by default: it returns the erasable
    holdings and leaves the vault untouched. ``vault`` is accepted so the plan
    reflects the live vault contents at preview time.
    """

    live_hashes = {entry.key.text_hash for entry in vault.entries()}
    erasable = tuple(
        entry for entry in package.entries if entry.text_hash in live_hashes
    )

    record: AuditRecord | None = None
    if audit_sink is not None:
        record = audit_sink.append(
            ERASURE_PREVIEW_EVENT,
            {
                "subject_ref": package.subject_ref,
                "erasable_count": len(erasable),
                "text_hashes": sorted(entry.text_hash for entry in erasable),
                "destructive": False,
            },
        )

    return ErasurePlan(
        subject_ref=package.subject_ref,
        erasable=erasable,
        executed=False,
        audit_record=record,
    )


__all__ = [
    "DSAR_ADVISORY",
    "EXPORT_EVENT",
    "ERASURE_PREVIEW_EVENT",
    "SubjectIdentifier",
    "DsarEntry",
    "DsarPackage",
    "ErasurePlan",
    "assemble_dsar_package",
    "render_dsar_summary",
    "plan_erasure",
]
