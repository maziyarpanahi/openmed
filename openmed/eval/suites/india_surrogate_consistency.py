"""Cross-document surrogate-consistency gate for the synthetic India corpus.

The gate answers one question that the policy-coverage and residual-leakage
suites deliberately cannot: when the same synthetic person or the same
synthetic identifier appears in more than one document, and in more than one
writing system, does de-identification produce **one** stable surrogate
identity for all of them?

"Consistent" is defined here as the conjunction of six checks:

* **One identity per declared group.** Every alias of a
  ``cross_document_identities`` group in the corpus manifest must resolve to a
  single :class:`~openmed.core.surrogate_vault.SurrogateKey` identity, across
  documents and across Latin, Devanagari, and Tamil.
* **One surrogate per identity.** The vault must hand back a single stored
  surrogate for the group rather than a fresh value per document.
* **Script-faithful rendering.** The surrogate rendered for a Devanagari alias
  must itself be Devanagari, and likewise for Tamil and Latin, so replacement
  does not silently switch scripts mid-document.
* **No alias survives.** No alias surface may remain in a document after its
  surrogate is substituted, and no surrogate may embed an alias surface.
* **Stable identifier linkage.** A structured identifier (ABHA, Aadhaar, PAN,
  phone) repeated across documents must map to one surrogate, distinct source
  values must map to distinct surrogates, and the generated surrogate must
  itself satisfy the corresponding shape or checksum validator. Surrogates are
  drawn from a per-run counter rather than derived from the source value, so
  cross-document reuse depends on the vault actually linking the occurrences.
* **No negative collision.** Independently generated synthetic *identifiers*
  must not collide with the corpus identities. This suite generates no negative
  name surfaces and therefore does not gate name collisions; those are covered
  by the ``indic-name-consistency`` suite.

Every run is executed twice with a fresh in-memory vault and the same secret;
a differing result fails the gate as non-deterministic. Within a run each
alias is also resolved twice, so a store that silently drops writes is caught
even when key equality still holds.

On the gated default path the vault renders the stored identity into each
source script itself; the ``Anonymizer.render_name_surrogate`` callback is only
consulted on the transliteration-aware path, where the vault key language is
``indic``.

Recorded, not gated: the opt-in ``transliteration_aware_name_matching`` path
(:mod:`openmed.core.indic_name_match`) does **not** collapse this corpus's
declared aliases into one identity. Its canonical keys differ across scripts
(the Devanagari inherent vowel and the Tamil rendering of the surname produce
different Latin folds), so the same person yields three identities instead of
one. That divergence is reported in every result as a
:class:`IndiaSurrogateDivergence` with ``gated=False`` so a passing verdict is
never read as evidence that transliteration-aware matching is cross-script
stable on clinical text. The gated verdict covers the default matching path,
which is the shipped default. Fixing the transliteration-aware fold is tracked
separately; it touches shared core matching used well beyond this corpus.

Recorded, not gated (second divergence): the vault normalizes the key language
for personal names to ``india``, but keeps the document language in the key for
structured identifiers. The same Aadhaar in a Hindi note and a Tamil note is
therefore two vault keys and two surrogates. Names link across languages;
identifiers do not. This is reported as a
:data:`LANGUAGE_SCOPED_IDENTIFIER_KEYS` divergence with ``gated=False``, and the
gated linkage check is scoped to ``(value, language)`` -- the granularity the
vault actually guarantees -- so it still fails closed on a genuine regression.

The corpus is committed synthetic data only: no real PHI, no production data,
and no DUA-restricted content. The gate is deterministic and fully offline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from openmed.core.anonymizer import Anonymizer
from openmed.core.anonymizer.providers.clinical_ids import (
    generate_aadhaar,
    generate_abha_number,
    generate_indian_phone,
    generate_pan,
    validate_abha_number,
    validate_indian_phone,
    validate_pan,
)
from openmed.core.indic_name_match import canonical_indic_name_key, detect_name_script
from openmed.core.pii_i18n import validate_aadhaar
from openmed.core.surrogate_vault import (
    SurrogateVault,
    _contains_indian_name,
    _contains_source_surface,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openmed.eval.datasets.clinical_phi import (
        IndiaClinicalPHICorpus,
        IndiaClinicalPHIIdentity,
    )
    from openmed.eval.harness import BenchmarkFixture
    from openmed.eval.india_dpdp_coverage import DpdpCoverageReport
    from openmed.eval.suites.india_clinical_leakage import IndiaClinicalLeakageResult

INDIA_SURROGATE_CONSISTENCY = "india_surrogate_consistency"

#: Identifier for the combined coverage + leakage + consistency report.
INDIA_CLINICAL_SUITE = "india_clinical"

#: Local evaluation secret. It protects nothing but this offline fixture set.
INDIA_SURROGATE_CONSISTENCY_SECRET = "openmed-synthetic-india-surrogate-consistency"

DEFAULT_NAME_MATCHING = "default"
TRANSLITERATION_AWARE_NAME_MATCHING = "transliteration_aware"

IDENTITY_SPLIT_ACROSS_SCRIPTS = "identity_split_across_scripts"
SURROGATE_NOT_REUSED = "surrogate_not_reused"
SURROGATE_NOT_STABLE = "surrogate_not_stable"
SURROGATE_SCRIPT_MISMATCH = "surrogate_script_mismatch"
ALIAS_SURFACE_SURVIVED = "alias_surface_survived"
IDENTIFIER_SURROGATE_UNSTABLE = "identifier_surrogate_unstable"
IDENTIFIER_SURROGATE_COLLIDED = "identifier_surrogate_collided"
IDENTIFIER_SURROGATE_INVALID = "identifier_surrogate_invalid"
NEGATIVE_IDENTIFIER_COLLISION = "negative_identifier_collision"
NEGATIVE_IDENTIFIER_INVALID = "negative_identifier_invalid"
IDENTIFIER_LINKAGE_LANGUAGE_SCOPED = "identifier_linkage_language_scoped"
LANGUAGE_SCOPED_IDENTIFIER_KEYS = "language_scoped_identifier_keys"
NONDETERMINISTIC_RUN = "nondeterministic_run"
UNKNOWN_MATCHING_MODE = "unknown_matching_mode"

#: The matching paths this suite knows how to evaluate.
SUPPORTED_NAME_MATCHING_MODES: tuple[str, ...] = (
    DEFAULT_NAME_MATCHING,
    TRANSLITERATION_AWARE_NAME_MATCHING,
)

#: Identifier types whose cross-document linkage and surrogate validity are
#: gated. Each entry maps to a generator and a validator from the shared India
#: identifier provider registry.
LINKED_IDENTIFIER_TYPES: tuple[str, ...] = ("aadhaar", "abha", "indian_phone", "pan")

_IDENTIFIER_GENERATORS: dict[str, Callable[..., str]] = {
    "aadhaar": generate_aadhaar,
    "abha": generate_abha_number,
    "indian_phone": generate_indian_phone,
    "pan": generate_pan,
}
_IDENTIFIER_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "aadhaar": validate_aadhaar,
    "abha": validate_abha_number,
    "indian_phone": validate_indian_phone,
    "pan": validate_pan,
}

#: Number of independently generated negative *identifiers* checked for
#: collision. These are structured identifiers only -- this suite does not
#: generate negative name surfaces, so it does not gate name collisions. Name
#: collisions are covered by the ``indic-name-consistency`` suite.
#: Edit-similarity at or above which a surrogate is treated as still
#: carrying its alias. Calibrated on the shipped corpus: real surrogates score
#: 0.64-0.78 against their aliases, near-transliterations score 0.94-1.00.
ALIAS_LEAK_SIMILARITY_THRESHOLD = 0.85

NEGATIVE_IDENTIFIER_COUNT = 8
_NEGATIVE_SEED = "openmed-india-surrogate-negatives"


@dataclass(frozen=True)
class IndiaSurrogateIdentityVerdict:
    """Per-identity-group consistency verdict without raw alias surfaces."""

    group_id: str
    alias_count: int
    document_count: int
    scripts: tuple[str, ...]
    identity_count: int
    surrogate_count: int
    missing_surrogate_count: int
    unstable_surrogate_count: int
    script_mismatch_count: int
    alias_leak_count: int

    @property
    def passed(self) -> bool:
        """Return whether this group met every gated consistency check.

        ``surrogate_count`` alone cannot carry the "one surrogate" claim: once
        the aliases share a key it can only mirror ``identity_count``. The
        missing and unstable counters are what make the claim falsifiable.
        """

        return (
            self.identity_count == 1
            and self.surrogate_count == 1
            and self.missing_surrogate_count == 0
            and self.unstable_surrogate_count == 0
            and self.script_mismatch_count == 0
            and self.alias_leak_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free per-group verdict."""

        return {
            "alias_count": self.alias_count,
            "alias_leak_count": self.alias_leak_count,
            "document_count": self.document_count,
            "group_id": self.group_id,
            "identity_count": self.identity_count,
            "missing_surrogate_count": self.missing_surrogate_count,
            "passed": self.passed,
            "script_mismatch_count": self.script_mismatch_count,
            "scripts": list(self.scripts),
            "surrogate_count": self.surrogate_count,
            "unstable_surrogate_count": self.unstable_surrogate_count,
        }


@dataclass(frozen=True)
class IndiaIdentifierLinkageVerdict:
    """Per-identifier-type cross-document linkage and validity verdict."""

    identifier_type: str
    canonical_label: str
    occurrence_count: int
    distinct_source_count: int
    distinct_surrogate_count: int
    repeated_source_count: int
    unstable_surrogate_count: int
    collision_count: int
    cross_language_split_count: int
    invalid_surrogate_count: int

    @property
    def passed(self) -> bool:
        """Return whether linkage holds for the guarantee the vault makes.

        ``cross_language_split_count`` is recorded but not gated: the vault
        keys structured identifiers by document language, so the same value in
        a Hindi and a Tamil note is two keys by construction. Gating it here
        would ship a red gate for a shared-core behaviour.
        """

        return (
            self.collision_count == 0
            and self.unstable_surrogate_count == 0
            and self.invalid_surrogate_count == 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free per-identifier verdict."""

        return {
            "canonical_label": self.canonical_label,
            "collision_count": self.collision_count,
            "cross_language_split_count": self.cross_language_split_count,
            "distinct_source_count": self.distinct_source_count,
            "distinct_surrogate_count": self.distinct_surrogate_count,
            "identifier_type": self.identifier_type,
            "invalid_surrogate_count": self.invalid_surrogate_count,
            "occurrence_count": self.occurrence_count,
            "passed": self.passed,
            "repeated_source_count": self.repeated_source_count,
            "unstable_surrogate_count": self.unstable_surrogate_count,
        }


@dataclass(frozen=True)
class IndiaSurrogateDivergence:
    """A recorded, deliberately un-gated divergence from an alternate path."""

    mode: str
    group_id: str
    expected_identity_count: int
    observed_identity_count: int
    reason: str
    gated: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, raw-text-free divergence record."""

        return {
            "expected_identity_count": self.expected_identity_count,
            "gated": self.gated,
            "group_id": self.group_id,
            "mode": self.mode,
            "observed_identity_count": self.observed_identity_count,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IndiaSurrogateConsistencyResult:
    """Aggregate surrogate-consistency verdict for the India clinical corpus."""

    passed: bool
    mode: str
    corpus_id: str
    group_count: int
    alias_count: int
    scripts: tuple[str, ...]
    deterministic: bool
    negative_identifier_count: int
    negative_collision_count: int
    negative_invalid_count: int
    identity_verdicts: tuple[IndiaSurrogateIdentityVerdict, ...] = ()
    identifier_verdicts: tuple[IndiaIdentifierLinkageVerdict, ...] = ()
    known_divergences: tuple[IndiaSurrogateDivergence, ...] = ()
    failures: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, raw-surface-free consistency report."""

        return {
            "alias_count": self.alias_count,
            "corpus_id": self.corpus_id,
            "deterministic": self.deterministic,
            "failures": list(self.failures),
            "group_count": self.group_count,
            "identifier_verdicts": [
                verdict.to_dict() for verdict in self.identifier_verdicts
            ],
            "identity_verdicts": [
                verdict.to_dict() for verdict in self.identity_verdicts
            ],
            "known_divergences": [
                divergence.to_dict() for divergence in self.known_divergences
            ],
            "known_divergence_note": (
                "Divergences are recorded, never gated. The gated verdict "
                "covers the default name-matching path; the opt-in "
                "transliteration-aware path does not collapse this corpus's "
                "cross-script aliases into one identity."
            ),
            "mode": self.mode,
            "negative_collision_count": self.negative_collision_count,
            "negative_identifier_count": self.negative_identifier_count,
            "negative_invalid_count": self.negative_invalid_count,
            "negative_scope": (
                "Structured identifiers only. This suite generates no negative "
                "name surfaces, so it does not gate name collisions; those are "
                "covered by the indic-name-consistency suite."
            ),
            "passed": self.passed,
            "scripts": list(self.scripts),
            "suite": INDIA_SURROGATE_CONSISTENCY,
        }


@dataclass(frozen=True)
class _RunEvidence:
    """Single-pass evidence used for the determinism comparison."""

    identity_hashes: tuple[str, ...]
    surrogates: tuple[str, ...]
    identifier_surrogates: tuple[str, ...]
    identity_verdicts: tuple[IndiaSurrogateIdentityVerdict, ...]
    identifier_verdicts: tuple[IndiaIdentifierLinkageVerdict, ...]
    negative_collision_count: int
    negative_invalid_count: int
    failures: tuple[str, ...]


def validate_name_matching_mode(mode: str) -> str:
    """Return *mode* if this suite knows how to evaluate it.

    An unvalidated mode string would let a caller typo
    ``transliteration-aware`` and receive a passing verdict stamped with the
    name of the path that is documented as not cross-script stable.
    """

    if mode not in SUPPORTED_NAME_MATCHING_MODES:
        allowed = ", ".join(SUPPORTED_NAME_MATCHING_MODES)
        raise ValueError(
            f"{UNKNOWN_MATCHING_MODE}: {mode!r}; expected one of: {allowed}"
        )
    return mode


def load_india_surrogate_consistency_fixtures(
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
) -> list["BenchmarkFixture"]:
    """Load the India corpus as standard eval-harness benchmark fixtures.

    The typed corpus loader validates the safety manifest, span offsets, and
    cross-document identity metadata first, so the harness never sees an
    unvalidated fixture.
    """

    from openmed.eval.datasets.clinical_phi import load_india_clinical_phi_corpus

    corpus = load_india_clinical_phi_corpus(manifest_path, fixture_path)
    return [record.to_benchmark_fixture() for record in corpus.records]


def india_surrogate_consistency_metadata(
    *,
    fixture_path: str | Path | None = None,
    mode: str = DEFAULT_NAME_MATCHING,
) -> dict[str, Any]:
    """Return discoverable, raw-text-free metadata for the consistency gate."""

    from openmed.eval.datasets.clinical_phi import INDIA_CLINICAL_PHI_CORPUS_ID

    return {
        "corpus_id": INDIA_CLINICAL_PHI_CORPUS_ID,
        "fixture_path": str(fixture_path) if fixture_path is not None else None,
        "gated_mode": mode,
        "linked_identifier_types": list(LINKED_IDENTIFIER_TYPES),
        "recorded_not_gated_modes": [TRANSLITERATION_AWARE_NAME_MATCHING],
        "required_identity_count_per_group": 1,
        "negative_scope": "structured_identifiers_only",
        "required_negative_collisions": 0,
        "supported_modes": list(SUPPORTED_NAME_MATCHING_MODES),
        "suite": INDIA_SURROGATE_CONSISTENCY,
        "synthetic": True,
    }


def evaluate_india_surrogate_consistency(
    corpus: "IndiaClinicalPHICorpus | None" = None,
    *,
    mode: str = DEFAULT_NAME_MATCHING,
    seed: int = 677,
    record_known_divergences: bool = True,
) -> IndiaSurrogateConsistencyResult:
    """Evaluate cross-document, cross-script surrogate consistency.

    Args:
        corpus: Optional preloaded India clinical corpus. Defaults to the
            committed synthetic OM-677 corpus.
        mode: Name-matching path to gate. ``default`` mirrors the shipped
            configuration; ``transliteration_aware`` exercises the opt-in path.
        seed: Deterministic anonymizer seed.
        record_known_divergences: When true and *mode* is the default path,
            also run the transliteration-aware path and record any identity
            split as an un-gated divergence.

    Returns:
        A privacy-safe aggregate result with per-group and per-identifier
        verdicts, recorded divergences, and failure reasons.
    """

    from openmed.eval.datasets.clinical_phi import load_india_clinical_phi_corpus

    resolved_mode = validate_name_matching_mode(mode)
    active = corpus if corpus is not None else load_india_clinical_phi_corpus()
    transliteration_aware = resolved_mode == TRANSLITERATION_AWARE_NAME_MATCHING

    first = _run_once(active, transliteration_aware=transliteration_aware, seed=seed)
    second = _run_once(active, transliteration_aware=transliteration_aware, seed=seed)
    deterministic = (
        first.identity_hashes == second.identity_hashes
        and first.surrogates == second.surrogates
        and first.identifier_surrogates == second.identifier_surrogates
    )

    failures = list(first.failures)
    if not deterministic:
        failures.append(NONDETERMINISTIC_RUN)

    divergences: list[IndiaSurrogateDivergence] = []
    if record_known_divergences and not transliteration_aware:
        divergences.extend(_known_divergences(active, seed=seed))
    # Structured identifiers are keyed by document language, so one value used
    # in two languages yields two surrogates. Recorded, never gated.
    if record_known_divergences:
        divergences.extend(
            IndiaSurrogateDivergence(
                mode=LANGUAGE_SCOPED_IDENTIFIER_KEYS,
                group_id=verdict.identifier_type,
                expected_identity_count=1,
                observed_identity_count=verdict.distinct_surrogate_count,
                reason=IDENTIFIER_LINKAGE_LANGUAGE_SCOPED,
            )
            for verdict in first.identifier_verdicts
            if verdict.cross_language_split_count
        )

    identities = active.manifest.cross_document_identities
    scripts = tuple(
        sorted({alias.script for identity in identities for alias in identity.aliases})
    )
    passed = (
        not failures
        and deterministic
        and all(verdict.passed for verdict in first.identity_verdicts)
        and all(verdict.passed for verdict in first.identifier_verdicts)
        and first.negative_collision_count == 0
        and first.negative_invalid_count == 0
    )
    return IndiaSurrogateConsistencyResult(
        passed=passed,
        mode=resolved_mode,
        corpus_id=active.manifest.corpus_id,
        group_count=len(identities),
        alias_count=sum(len(identity.aliases) for identity in identities),
        scripts=scripts,
        deterministic=deterministic,
        negative_identifier_count=NEGATIVE_IDENTIFIER_COUNT,
        negative_collision_count=first.negative_collision_count,
        negative_invalid_count=first.negative_invalid_count,
        identity_verdicts=first.identity_verdicts,
        identifier_verdicts=first.identifier_verdicts,
        known_divergences=tuple(divergences),
        failures=tuple(failures),
    )


def run_india_surrogate_consistency_gate(
    *,
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    mode: str = DEFAULT_NAME_MATCHING,
    seed: int = 677,
) -> IndiaSurrogateConsistencyResult:
    """Load the committed India corpus and run the consistency gate."""

    from openmed.eval.datasets.clinical_phi import load_india_clinical_phi_corpus

    corpus = load_india_clinical_phi_corpus(manifest_path, fixture_path)
    return evaluate_india_surrogate_consistency(corpus, mode=mode, seed=seed)


def assert_india_surrogate_consistency_gate(
    *,
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    mode: str = DEFAULT_NAME_MATCHING,
    seed: int = 677,
) -> IndiaSurrogateConsistencyResult:
    """Return a passing result or raise with raw-surface-free diagnostics."""

    result = run_india_surrogate_consistency_gate(
        manifest_path=manifest_path,
        fixture_path=fixture_path,
        mode=mode,
        seed=seed,
    )
    if not result.passed:
        reasons = ", ".join(sorted(set(result.failures))) or "verdict_failure"
        groups = ", ".join(
            sorted(
                verdict.group_id
                for verdict in result.identity_verdicts
                if not verdict.passed
            )
        )
        identifiers = ", ".join(
            sorted(
                verdict.identifier_type
                for verdict in result.identifier_verdicts
                if not verdict.passed
            )
        )
        raise AssertionError(
            "India surrogate-consistency gate failed "
            f"(mode={result.mode}); reasons [{reasons}]; "
            f"group(s) [{groups}]; identifier(s) [{identifiers}]"
        )
    return result


@dataclass(frozen=True)
class IndiaClinicalSuiteReport:
    """One report combining coverage, leakage, and consistency verdicts.

    The report is the single artifact required by the India clinical suite: it
    exposes DPDP per-label policy coverage, the residual-PHI zero-leak verdict,
    and the cross-document surrogate-consistency verdict, together with the
    corpus safety boundary. It contains counts, canonical labels, offsets, and
    HMAC hashes only -- never a raw identifier or alias surface.
    """

    passed: bool
    corpus_id: str
    policy: str
    coverage: "DpdpCoverageReport"
    leakage: "IndiaClinicalLeakageResult"
    consistency: IndiaSurrogateConsistencyResult
    safety_boundary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the combined, raw-text-free India clinical suite report."""

        return {
            "corpus_id": self.corpus_id,
            "passed": self.passed,
            "policy": self.policy,
            "policy_coverage": self.coverage.to_dict(),
            "residual_leakage": self.leakage.to_dict(),
            "safety_boundary": dict(self.safety_boundary),
            "suite": INDIA_CLINICAL_SUITE,
            "surrogate_consistency": self.consistency.to_dict(),
        }


def run_india_clinical_suite_report(
    *,
    manifest_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    seed: int = 677,
) -> IndiaClinicalSuiteReport:
    """Run policy-coverage, residual-leakage, and consistency in one report."""

    from openmed.eval.datasets.clinical_phi import load_india_clinical_phi_corpus
    from openmed.eval.india_dpdp_coverage import run_india_dpdp_coverage
    from openmed.eval.suites.india_clinical_leakage import (
        evaluate_india_clinical_leakage,
    )

    corpus = load_india_clinical_phi_corpus(manifest_path, fixture_path)
    coverage = run_india_dpdp_coverage(corpus=corpus)
    leakage = evaluate_india_clinical_leakage(corpus.records)
    consistency = evaluate_india_surrogate_consistency(corpus, seed=seed)
    manifest = corpus.manifest
    return IndiaClinicalSuiteReport(
        passed=coverage.passed and leakage.passed and consistency.passed,
        corpus_id=manifest.corpus_id,
        policy=coverage.policy,
        coverage=coverage,
        leakage=leakage,
        consistency=consistency,
        safety_boundary={
            "assist_only_non_decisional": True,
            "contains_dua_data": manifest.contains_dua_data,
            "contains_real_phi": manifest.contains_real_phi,
            "disclaimer": manifest.disclaimer,
            "execution": "local-offline-deterministic",
            "excludes_real_hospital_data": True,
            "license_id": manifest.license_id,
            "provenance": manifest.provenance,
            "synthetic_only": manifest.synthetic_only,
        },
    )


def india_clinical_suite_metadata() -> dict[str, Any]:
    """Return raw-text-free metadata for the combined India clinical suite."""

    from openmed.eval.datasets.clinical_phi import INDIA_CLINICAL_PHI_CORPUS_ID
    from openmed.eval.india_dpdp_coverage import INDIA_DPDP_COVERAGE
    from openmed.eval.suites.india_clinical_leakage import INDIA_CLINICAL_PHI_LEAKAGE

    return {
        "components": [
            INDIA_DPDP_COVERAGE,
            INDIA_CLINICAL_PHI_LEAKAGE,
            INDIA_SURROGATE_CONSISTENCY,
        ],
        "corpus_id": INDIA_CLINICAL_PHI_CORPUS_ID,
        "suite": INDIA_CLINICAL_SUITE,
        "synthetic": True,
    }


def _run_once(
    corpus: "IndiaClinicalPHICorpus",
    *,
    transliteration_aware: bool,
    seed: int,
) -> _RunEvidence:
    vault = SurrogateVault.in_memory(
        INDIA_SURROGATE_CONSISTENCY_SECRET,
        transliteration_aware_name_matching=transliteration_aware,
    )
    anonymizer = Anonymizer(
        lang="hi",
        consistent=True,
        seed=seed,
        transliteration_aware_name_matching=transliteration_aware,
        indic_name_normalizer=vault.indic_name_normalizer,
    )
    language_by_document = {
        record.document_id: record.language for record in corpus.records
    }
    text_by_document = {record.document_id: record.text for record in corpus.records}

    failures: list[str] = []
    identity_hashes: list[str] = []
    surrogates: list[str] = []
    identity_verdicts: list[IndiaSurrogateIdentityVerdict] = []

    for identity in corpus.manifest.cross_document_identities:
        verdict, evidence = _evaluate_identity(
            identity,
            vault=vault,
            anonymizer=anonymizer,
            language_by_document=language_by_document,
            text_by_document=text_by_document,
        )
        identity_verdicts.append(verdict)
        identity_hashes.extend(evidence[0])
        surrogates.extend(evidence[1])
        if verdict.identity_count != 1:
            failures.append(f"{identity.group_id}:{IDENTITY_SPLIT_ACROSS_SCRIPTS}")
        if verdict.surrogate_count != 1 or verdict.missing_surrogate_count:
            failures.append(f"{identity.group_id}:{SURROGATE_NOT_REUSED}")
        if verdict.unstable_surrogate_count:
            failures.append(f"{identity.group_id}:{SURROGATE_NOT_STABLE}")
        if verdict.script_mismatch_count:
            failures.append(f"{identity.group_id}:{SURROGATE_SCRIPT_MISMATCH}")
        if verdict.alias_leak_count:
            failures.append(f"{identity.group_id}:{ALIAS_SURFACE_SURVIVED}")

    identifier_verdicts, identifier_surrogates, identifier_failures = (
        _evaluate_identifier_linkage(corpus, vault=vault)
    )
    failures.extend(identifier_failures)

    negative_collisions, negative_invalid = _negative_identifier_checks(
        corpus,
        vault=vault,
    )
    if negative_collisions:
        failures.append(NEGATIVE_IDENTIFIER_COLLISION)
    if negative_invalid:
        failures.append(NEGATIVE_IDENTIFIER_INVALID)

    return _RunEvidence(
        identity_hashes=tuple(identity_hashes),
        surrogates=tuple(surrogates),
        identifier_surrogates=tuple(identifier_surrogates),
        identity_verdicts=tuple(identity_verdicts),
        identifier_verdicts=tuple(identifier_verdicts),
        negative_collision_count=negative_collisions,
        negative_invalid_count=negative_invalid,
        failures=tuple(failures),
    )


def _evaluate_identity(
    identity: "IndiaClinicalPHIIdentity",
    *,
    vault: SurrogateVault,
    anonymizer: Anonymizer,
    language_by_document: dict[str, str],
    text_by_document: dict[str, str],
) -> tuple[IndiaSurrogateIdentityVerdict, tuple[tuple[str, ...], tuple[str, ...]]]:
    hashes: list[str] = []
    rendered: list[str] = []
    stored: list[str] = []
    missing = 0
    unstable = 0
    script_mismatches = 0
    alias_leaks = 0

    for alias in identity.aliases:
        lang = language_by_document.get(alias.document_id, "hi")
        key = vault.key_for(alias.text, label="PERSON", lang=lang)
        hashes.append(key.text_hash)

        def create(attempt: int, source: str = alias.text) -> str:
            return anonymizer.surrogate_identity(
                source,
                "PERSON",
                lang=lang,
                attempt=attempt,
            )

        def render(candidate: str, source: str = alias.text) -> str:
            return anonymizer.render_name_surrogate(candidate, source_surface=source)

        surrogate = vault.get_or_create(
            alias.text,
            label="PERSON",
            lang=lang,
            create_surrogate=create,
            render_surrogate=render,
        )
        rendered.append(surrogate)

        # A store miss is not a surrogate. Collecting ``None`` into a set would
        # make "no surrogate at all" indistinguishable from "one surrogate".
        entry = vault.store.get(key)
        if entry is None:
            missing += 1
        else:
            stored.append(entry)

        # Independent of the identity check: ask the vault again. If linkage is
        # working the second call must return the byte-identical surface. A
        # store that silently drops writes mints a fresh unrelated name here,
        # which no amount of key-equality checking would reveal.
        repeat = vault.get_or_create(
            alias.text,
            label="PERSON",
            lang=lang,
            create_surrogate=create,
            render_surrogate=render,
        )
        if repeat != surrogate:
            unstable += 1

        expected_script = alias.script.casefold()
        if detect_name_script(surrogate).casefold() != expected_script:
            script_mismatches += 1

        # Canonical-fold comparison, matching the vault's own leak guard: an
        # exact-substring test accepts a surrogate that is merely the
        # transliterated fold of the real name.
        if alias.text and _leaks_alias(surrogate, alias.text):
            alias_leaks += 1

    verdict = IndiaSurrogateIdentityVerdict(
        group_id=identity.group_id,
        alias_count=len(identity.aliases),
        document_count=len({alias.document_id for alias in identity.aliases}),
        scripts=tuple(sorted({alias.script for alias in identity.aliases})),
        identity_count=len(set(hashes)),
        # One identity may legitimately *render* into several scripts, so the
        # gate compares the single stored vault value rather than the rendered
        # surfaces.
        surrogate_count=len(set(stored)),
        missing_surrogate_count=missing,
        unstable_surrogate_count=unstable,
        script_mismatch_count=script_mismatches,
        alias_leak_count=alias_leaks,
    )
    return verdict, (tuple(hashes), tuple(rendered))


def _leaks_alias(surrogate: str, alias_text: str) -> bool:
    """Return whether *surrogate* still carries the alias.

    Exact substring containment is not sufficient: a surrogate that is merely a
    near-transliteration of the real name (``Arav Sarma`` for ``Aarav Sharma``)
    is trivially re-identifiable but shares no exact substring, and the
    repository's canonical fold deliberately keeps those keys distinct. The
    comparison therefore falls back to edit similarity over the canonical
    folds. On the shipped corpus real surrogates score 0.64-0.78 against their
    aliases while near-misses score 0.94-1.00, so the threshold sits between
    with margin on both sides.
    """

    if _contains_source_surface(surrogate, alias_text):
        return True
    try:
        folded_alias = canonical_indic_name_key(alias_text)
        folded_surrogate = canonical_indic_name_key(surrogate)
    except ValueError:  # pragma: no cover - empty surfaces are filtered above
        return False
    if _contains_indian_name(folded_surrogate, folded_alias):
        return True
    similarity = SequenceMatcher(None, folded_surrogate, folded_alias).ratio()
    return similarity >= ALIAS_LEAK_SIMILARITY_THRESHOLD


def _evaluate_identifier_linkage(
    corpus: "IndiaClinicalPHICorpus",
    *,
    vault: SurrogateVault,
) -> tuple[tuple[IndiaIdentifierLinkageVerdict, ...], tuple[str, ...], list[str]]:
    occurrences: dict[str, list[tuple[str, str, str]]] = {}
    canonical_by_type: dict[str, str] = {}

    for record in corpus.records:
        for span in record.gold_spans:
            identifier_type = str(span.metadata.get("identifier_type") or "")
            if identifier_type not in LINKED_IDENTIFIER_TYPES:
                continue
            canonical_by_type.setdefault(identifier_type, str(span.label))
            occurrences.setdefault(identifier_type, []).append(
                (span.text, record.language, str(span.label))
            )

    verdicts: list[IndiaIdentifierLinkageVerdict] = []
    all_surrogates: list[str] = []
    failures: list[str] = []

    # One counter per run, consumed in deterministic iteration order. Seeding
    # the generator from the *source* instead would make every surrogate a pure
    # function of the source string, so a vault that lost its linkage entirely
    # would still hand back matching values and the linkage check could never
    # fail. Drawing from a run counter makes cross-document reuse depend on the
    # vault actually linking the two occurrences.
    draw = count()

    for identifier_type in sorted(occurrences):
        rows = occurrences[identifier_type]
        # Keyed by (source, lang): that is the granularity the vault itself
        # links at, so a difference here is a genuine linkage regression.
        surrogate_by_scope: dict[tuple[str, str], str] = {}
        surrogates_by_source: dict[str, set[str]] = {}
        invalid = 0
        unstable = 0
        validator = _IDENTIFIER_VALIDATORS[identifier_type]
        for source, lang, label in rows:

            def create(attempt: int, kind: str = identifier_type) -> str:
                generator = _IDENTIFIER_GENERATORS[kind]
                return generator(
                    rng=random.Random(f"{_NEGATIVE_SEED}|{kind}|{next(draw)}|{attempt}")
                )

            surrogate = vault.get_or_create(
                source,
                label=label,
                lang=lang,
                create_surrogate=create,
            )
            all_surrogates.append(surrogate)
            scope = (source, lang)
            previous = surrogate_by_scope.get(scope)
            if previous is None:
                surrogate_by_scope[scope] = surrogate
            elif previous != surrogate:
                unstable += 1
            surrogates_by_source.setdefault(source, set()).add(surrogate)
            if not validator(surrogate):
                invalid += 1

        # A collision is two *different* sources sharing one surrogate.
        owners: dict[str, set[str]] = {}
        for source, values in surrogates_by_source.items():
            for value in values:
                owners.setdefault(value, set()).add(source)
        collisions = sum(1 for holders in owners.values() if len(holders) > 1)
        cross_language = sum(
            1 for values in surrogates_by_source.values() if len(values) > 1
        )

        distinct_sources = len(surrogates_by_source)
        distinct_surrogates = len(owners)
        repeated = sum(1 for source in surrogates_by_source if _count(rows, source) > 1)
        verdict = IndiaIdentifierLinkageVerdict(
            identifier_type=identifier_type,
            canonical_label=canonical_by_type[identifier_type],
            occurrence_count=len(rows),
            distinct_source_count=distinct_sources,
            distinct_surrogate_count=distinct_surrogates,
            repeated_source_count=repeated,
            unstable_surrogate_count=unstable,
            collision_count=collisions,
            cross_language_split_count=cross_language,
            invalid_surrogate_count=invalid,
        )
        verdicts.append(verdict)
        if collisions:
            failures.append(f"{identifier_type}:{IDENTIFIER_SURROGATE_COLLIDED}")
        if unstable:
            failures.append(f"{identifier_type}:{IDENTIFIER_SURROGATE_UNSTABLE}")
        if invalid:
            failures.append(f"{identifier_type}:{IDENTIFIER_SURROGATE_INVALID}")

    return tuple(verdicts), tuple(all_surrogates), failures


def _count(rows: Sequence[tuple[str, str, str]], source: str) -> int:
    return sum(1 for row in rows if row[0] == source)


def _negative_identifier_checks(
    corpus: "IndiaClinicalPHICorpus",
    *,
    vault: SurrogateVault,
) -> tuple[int, int]:
    """Return (collisions, invalid) for generated synthetic negative identifiers.

    Negatives are structured identifiers generated algorithmically from the
    shared India providers and checked with the matching validators. A
    collision indicates a keying defect rather than a malformed fixture. A
    validator rejection is returned as a countable failure rather than raised,
    so a provider regression fails the gate instead of crashing the harness.

    This function generates no names, so it does not probe name collisions.
    """

    corpus_hashes = {
        vault.key_for(
            span.text,
            label=str(span.label),
            lang=record.language,
        ).text_hash
        for record in corpus.records
        for span in record.gold_spans
        if span.metadata.get("direct_identifier") is True
    }
    corpus_values = {
        span.text for record in corpus.records for span in record.gold_spans
    }

    collisions = 0
    invalid = 0
    for index in range(NEGATIVE_IDENTIFIER_COUNT):
        identifier_type = LINKED_IDENTIFIER_TYPES[index % len(LINKED_IDENTIFIER_TYPES)]
        rng = random.Random(f"{_NEGATIVE_SEED}|{identifier_type}|{index}")
        candidate = _IDENTIFIER_GENERATORS[identifier_type](rng=rng)
        if not _IDENTIFIER_VALIDATORS[identifier_type](candidate):
            invalid += 1
            continue
        if candidate in corpus_values:
            collisions += 1
            continue
        digest = vault.key_for(candidate, label="ID_NUM", lang="hi").text_hash
        if digest in corpus_hashes:
            collisions += 1
    return collisions, invalid


def _known_divergences(
    corpus: "IndiaClinicalPHICorpus",
    *,
    seed: int,
) -> tuple[IndiaSurrogateDivergence, ...]:
    """Record, without gating, how the opt-in matching path behaves."""

    alternate = evaluate_india_surrogate_consistency(
        corpus,
        mode=TRANSLITERATION_AWARE_NAME_MATCHING,
        seed=seed,
        record_known_divergences=False,
    )
    # Record any verdict the alternate path fails, not only an identity split,
    # so a divergence that shows up as a script mismatch or an alias leak is
    # still surfaced rather than silently dropped.
    return tuple(
        IndiaSurrogateDivergence(
            mode=TRANSLITERATION_AWARE_NAME_MATCHING,
            group_id=verdict.group_id,
            expected_identity_count=1,
            observed_identity_count=verdict.identity_count,
            reason=(
                IDENTITY_SPLIT_ACROSS_SCRIPTS
                if verdict.identity_count != 1
                else SURROGATE_SCRIPT_MISMATCH
                if verdict.script_mismatch_count
                else ALIAS_SURFACE_SURVIVED
                if verdict.alias_leak_count
                else SURROGATE_NOT_REUSED
            ),
        )
        for verdict in alternate.identity_verdicts
        if not verdict.passed
    )


__all__ = [
    "ALIAS_LEAK_SIMILARITY_THRESHOLD",
    "ALIAS_SURFACE_SURVIVED",
    "DEFAULT_NAME_MATCHING",
    "INDIA_CLINICAL_SUITE",
    "IDENTIFIER_SURROGATE_COLLIDED",
    "IDENTIFIER_SURROGATE_INVALID",
    "IDENTIFIER_LINKAGE_LANGUAGE_SCOPED",
    "IDENTIFIER_SURROGATE_UNSTABLE",
    "LANGUAGE_SCOPED_IDENTIFIER_KEYS",
    "IDENTITY_SPLIT_ACROSS_SCRIPTS",
    "INDIA_SURROGATE_CONSISTENCY",
    "INDIA_SURROGATE_CONSISTENCY_SECRET",
    "LINKED_IDENTIFIER_TYPES",
    "NEGATIVE_IDENTIFIER_COLLISION",
    "NEGATIVE_IDENTIFIER_COUNT",
    "NEGATIVE_IDENTIFIER_INVALID",
    "NONDETERMINISTIC_RUN",
    "SUPPORTED_NAME_MATCHING_MODES",
    "SURROGATE_NOT_REUSED",
    "SURROGATE_NOT_STABLE",
    "SURROGATE_SCRIPT_MISMATCH",
    "TRANSLITERATION_AWARE_NAME_MATCHING",
    "UNKNOWN_MATCHING_MODE",
    "IndiaClinicalSuiteReport",
    "IndiaIdentifierLinkageVerdict",
    "IndiaSurrogateConsistencyResult",
    "IndiaSurrogateDivergence",
    "IndiaSurrogateIdentityVerdict",
    "assert_india_surrogate_consistency_gate",
    "evaluate_india_surrogate_consistency",
    "india_clinical_suite_metadata",
    "india_surrogate_consistency_metadata",
    "load_india_surrogate_consistency_fixtures",
    "run_india_clinical_suite_report",
    "run_india_surrogate_consistency_gate",
    "validate_name_matching_mode",
]
