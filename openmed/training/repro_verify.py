"""Reproducibility hash recompute-and-verify harness.

This module provides offline-first verification that a training recipe, data
manifest reference, base model, and git SHA reproduce the claimed hash
recorded in model cards, manifests, and release ledgers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from openmed.core.repro_hash import (
    _MODEL_CARD_REPRO_HASH_RE,
    _SHA256_DIGEST_RE,
    ReproducibilityVerificationError,
    _normalise_component,
    compute_reproducibility_hash,
)

CORE_PROVENANCE_KEYS = ("recipe", "data_manifest", "base_model", "git_sha")
OPTIONAL_PROVENANCE_KEYS = ("rng_seeds", "recipe_config_hash", "env_lock_digest")
ALL_PROVENANCE_KEYS = CORE_PROVENANCE_KEYS + OPTIONAL_PROVENANCE_KEYS


@dataclass(frozen=True)
class ReproVerificationResult:
    """Outcome of a reproducibility hash recompute-and-verify run.

    Attributes:
        status: Verification status string ('MATCH', 'MISMATCH', or
            'UNVERIFIABLE').
        matched: True if the recomputed hash exactly matches the claimed hash.
        recomputed_hash: The newly computed sha256 digest from candidate inputs,
            or None if inputs were insufficient or invalid.
        claimed_hash: The reference sha256 digest extracted from manifests,
            model cards, or explicit arguments.
        diverging_inputs: Names of candidate inputs that differ from the
            reference provenance.
        details: Structural telemetry and diagnostic metadata (contains no
            raw PHI).
    """

    status: Literal["MATCH", "MISMATCH", "UNVERIFIABLE"]
    matched: bool
    recomputed_hash: str | None
    claimed_hash: str | None
    diverging_inputs: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "status": self.status,
            "matched": self.matched,
            "recomputed_hash": self.recomputed_hash,
            "claimed_hash": self.claimed_hash,
            "diverging_inputs": list(self.diverging_inputs),
            "details": dict(self.details),
        }


def verify_reproducibility_inputs(
    *,
    claimed_hash: str | None = None,
    recipe: Any = None,
    data_manifest: Any = None,
    base_model: Any = None,
    git_sha: str | None = None,
    reference_provenance: Mapping[str, Any] | None = None,
    rng_seeds: Mapping[str, int] | None = None,
    recipe_config_hash: str | None = None,
    env_lock_digest: str | None = None,
    manifest_row: Mapping[str, Any] | None = None,
    model_card_text: str | None = None,
) -> ReproVerificationResult:
    """Recompute the reproducibility hash from inputs and verify against a claim.

    Recomputes the hash using OM-023's canonical formula and compares it to the
    claimed hash extracted from manifest records, model cards, or explicit
    arguments. When a mismatch occurs, any diverging inputs are localized and
    named.

    Args:
        claimed_hash: Explicit claimed sha256 hash to verify against.
        recipe: Candidate recipe configuration dictionary, object, or path.
        data_manifest: Candidate data manifest reference, dictionary, or path.
        base_model: Candidate base model identifier or mapping.
        git_sha: Candidate git commit SHA.
        reference_provenance: Optional reference input dictionary to diff
            candidate inputs against for granular divergence attribution.
        rng_seeds: Optional random seed mapping.
        recipe_config_hash: Optional recipe config digest.
        env_lock_digest: Optional environment lock digest.
        manifest_row: Optional manifest row dictionary from models.jsonl.
        model_card_text: Optional raw markdown text of a model card.

    Returns:
        A ReproVerificationResult instance with MATCH, MISMATCH, or
        UNVERIFIABLE status and identified diverging inputs.
    """
    resolved_claim, claim_source = _resolve_claimed_hash(
        explicit_hash=claimed_hash,
        manifest_row=manifest_row,
        model_card_text=model_card_text,
        reference_provenance=reference_provenance,
    )

    if resolved_claim is None or not _is_valid_sha256(resolved_claim):
        return ReproVerificationResult(
            status="UNVERIFIABLE",
            matched=False,
            recomputed_hash=None,
            claimed_hash=resolved_claim,
            diverging_inputs=(),
            details={
                "reason": "Missing or malformed claimed reproducibility hash",
                "claim_source": claim_source,
            },
        )

    if not resolved_claim.startswith("sha256:"):
        resolved_claim = f"sha256:{resolved_claim}"

    # Resolve candidate inputs from explicit args, with reference fallback
    candidate_inputs: dict[str, Any] = {
        "recipe": recipe,
        "data_manifest": data_manifest,
        "base_model": base_model,
        "git_sha": git_sha,
        "rng_seeds": rng_seeds,
        "recipe_config_hash": recipe_config_hash,
        "env_lock_digest": env_lock_digest,
    }

    # If reference provenance is available, diff inputs to isolate divergences
    resolved_ref = _resolve_reference_provenance(
        reference_provenance=reference_provenance,
        manifest_row=manifest_row,
    )

    diverging_inputs: tuple[str, ...] = ()
    if resolved_ref is not None:
        diverging_inputs = _localize_divergences(
            candidate=candidate_inputs,
            reference=resolved_ref,
        )

    try:
        recomputed = compute_reproducibility_hash(
            recipe=recipe,
            data_manifest=data_manifest,
            base_model=base_model,
            git_sha=git_sha,
            rng_seeds=rng_seeds,
            recipe_config_hash=recipe_config_hash,
            env_lock_digest=env_lock_digest,
        )
    except (TypeError, ValueError, ReproducibilityVerificationError) as exc:
        return ReproVerificationResult(
            status="MISMATCH",
            matched=False,
            recomputed_hash=None,
            claimed_hash=resolved_claim,
            diverging_inputs=diverging_inputs or ("inputs_invalid",),
            details={
                "error_class": exc.__class__.__name__,
                "claim_source": claim_source,
            },
        )

    if recomputed == resolved_claim:
        return ReproVerificationResult(
            status="MATCH",
            matched=True,
            recomputed_hash=recomputed,
            claimed_hash=resolved_claim,
            diverging_inputs=(),
            details={"claim_source": claim_source},
        )

    details: dict[str, Any] = {
        "claim_source": claim_source,
        "has_reference_provenance": resolved_ref is not None,
    }
    if not diverging_inputs:
        if resolved_ref is not None:
            missing_ref = [k for k in CORE_PROVENANCE_KEYS if k not in resolved_ref]
            if missing_ref:
                details["unlocalized_reason"] = "incomplete_reference_provenance"
                details["missing_reference_keys"] = missing_ref
            else:
                details["unlocalized_reason"] = "systemic_integrity_fault"
        else:
            details["unlocalized_reason"] = "no_reference_provenance_to_diff"

    return ReproVerificationResult(
        status="MISMATCH",
        matched=False,
        recomputed_hash=recomputed,
        claimed_hash=resolved_claim,
        diverging_inputs=diverging_inputs,
        details=details,
    )


def _resolve_claimed_hash(
    *,
    explicit_hash: str | None,
    manifest_row: Mapping[str, Any] | None,
    model_card_text: str | None,
    reference_provenance: Mapping[str, Any] | None,
) -> tuple[str | None, str]:
    """Extract claimed reproducibility hash across precedence tiers."""
    if explicit_hash is not None:
        return explicit_hash.strip(), "explicit_argument"

    if manifest_row is not None:
        manifest_hash = manifest_row.get("reproducibility_hash")
        if isinstance(manifest_hash, str) and manifest_hash.strip():
            return manifest_hash.strip(), "manifest_row"
        training_prov = manifest_row.get("training_provenance")
        if isinstance(training_prov, Mapping):
            prov_hash = training_prov.get("reproducibility_hash")
            if isinstance(prov_hash, str) and prov_hash.strip():
                return prov_hash.strip(), "manifest_training_provenance"

    if model_card_text is not None:
        match = _MODEL_CARD_REPRO_HASH_RE.search(model_card_text)
        if match:
            return match.group("hash").strip(), "model_card"

    if reference_provenance is not None:
        ref_hash = reference_provenance.get("reproducibility_hash")
        if isinstance(ref_hash, str) and ref_hash.strip():
            return ref_hash.strip(), "reference_provenance"

    return None, "none"


def _resolve_reference_provenance(
    *,
    reference_provenance: Mapping[str, Any] | None,
    manifest_row: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Consolidate reference provenance from explicit input or manifest row."""
    if reference_provenance is not None:
        return dict(reference_provenance)

    if manifest_row is not None:
        ref: dict[str, Any] = {}
        if "base_model" in manifest_row:
            ref["base_model"] = manifest_row["base_model"]
        if "recipe" in manifest_row:
            ref["recipe"] = manifest_row["recipe"]
        if "data_manifest" in manifest_row:
            ref["data_manifest"] = manifest_row["data_manifest"]
        if "git_sha" in manifest_row:
            ref["git_sha"] = manifest_row["git_sha"]

        training_prov = manifest_row.get("training_provenance")
        if isinstance(training_prov, Mapping):
            for k in ALL_PROVENANCE_KEYS:
                if k in training_prov and k not in ref:
                    ref[k] = training_prov[k]
        if ref:
            return ref

    return None


def _localize_divergences(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[str, ...]:
    """Compare candidate components against reference provenance."""
    diverged: list[str] = []

    for key in ALL_PROVENANCE_KEYS:
        has_cand = key in candidate and candidate[key] is not None
        has_ref = key in reference and reference[key] is not None

        # If both are absent or None, no divergence
        if not has_cand and not has_ref:
            continue

        # Presence mismatch
        if has_cand != has_ref:
            diverged.append(key)
            continue

        cand_val = _normalise_component(candidate[key])
        ref_val = _normalise_component(reference[key])

        if cand_val != ref_val:
            diverged.append(key)

    return tuple(diverged)


def _is_valid_sha256(value: str) -> bool:
    """Return True if value matches the canonical sha256: digest format."""
    return bool(_SHA256_DIGEST_RE.fullmatch(value.strip()))


__all__ = [
    "ALL_PROVENANCE_KEYS",
    "CORE_PROVENANCE_KEYS",
    "OPTIONAL_PROVENANCE_KEYS",
    "ReproVerificationResult",
    "verify_reproducibility_inputs",
]
