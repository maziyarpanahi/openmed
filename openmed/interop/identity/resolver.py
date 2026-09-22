"""Deterministic exact and optional probabilistic identity resolvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from openmed.clinical.journey_contracts import canonical_digest
from openmed.compliance.projections import (
    ProjectionNamespace,
    ProjectionOperation,
    ProjectionPolicy,
    ProjectionPolicyOutcome,
    ProjectionPolicyRequest,
)
from openmed.structured.store import StoreResult, StoreState

from .contracts import (
    IdentityEvidence,
    IdentityLink,
    IdentityResolution,
    IdentityResolutionRequest,
)
from .store import IdentityResolutionStore


@runtime_checkable
class IdentityResolver(Protocol):
    """Backend-neutral patient and encounter resolution surface."""

    def resolve(
        self,
        request: IdentityResolutionRequest,
    ) -> StoreResult[IdentityResolution]:
        """Resolve opaque source-local keys without auto-merging uncertainty."""


@dataclass(frozen=True, slots=True, order=True)
class ProbabilisticIdentityCandidate:
    """Value-safe candidate emitted by an optional local plugin."""

    canonical_key: str
    score_basis_points: int
    evidence_digest: str
    plugin_id: str
    plugin_version: str

    def __post_init__(self) -> None:
        from .contracts import _controlled, _digest, _opaque_id, _version

        _opaque_id(self.canonical_key, "canonical_key")
        if (
            type(self.score_basis_points) is not int
            or not 0 <= self.score_basis_points <= 10_000
        ):
            raise ValueError("score_basis_points must be between 0 and 10000")
        _digest(self.evidence_digest, "evidence_digest")
        _controlled(self.plugin_id, "plugin_id")
        _version(self.plugin_version, "plugin_version")


@runtime_checkable
class IdentityCandidatePlugin(Protocol):
    """Optional local candidate generator; candidates always require review."""

    def candidates(
        self,
        request: IdentityResolutionRequest,
    ) -> StoreResult[tuple[ProbabilisticIdentityCandidate, ...]]:
        """Return candidates without selecting or merging an identity."""


class ExactIdentityResolver:
    """Resolve only persisted deterministic exact links."""

    resolver_id = "openmed.identity.exact"
    resolver_version = "1.0.0"

    def __init__(
        self,
        store: IdentityResolutionStore,
        *,
        access_policy: ProjectionPolicy | None = None,
    ) -> None:
        self.store = store
        self.access_policy = access_policy or ProjectionPolicy()

    def resolve(
        self,
        request: IdentityResolutionRequest,
    ) -> StoreResult[IdentityResolution]:
        """Resolve exact links and persist the deterministic outcome."""

        evaluated = self.evaluate(request)
        if not evaluated.ok or evaluated.value is None:
            return evaluated
        return self.store.save_resolution(evaluated.value)

    def evaluate(
        self,
        request: IdentityResolutionRequest,
    ) -> StoreResult[IdentityResolution]:
        """Evaluate exact links without persisting the outcome."""

        policy_result = _authorize(request, self.access_policy)
        if policy_result is not None:
            return policy_result
        links_result = self.store.active_links(request.source_keys)
        if not links_result.ok or links_result.value is None:
            return StoreResult.outcome(
                links_result.state,
                links_result.code or "identity_link_read_failed",
            )
        links = tuple(
            link
            for source_links in links_result.value.values()
            for link in source_links
        )
        candidates = tuple(sorted({link.canonical_key for link in links}))
        evidence = tuple(sorted(_exact_evidence(link) for link in links))
        if not candidates:
            state = "unmatched"
            canonical_key = None
            review_required = False
        elif len(candidates) == 1:
            state = "matched"
            canonical_key = candidates[0]
            review_required = False
        else:
            state = "conflict"
            canonical_key = None
            review_required = True
        resolution = _resolution(
            request,
            state=state,
            canonical_key=canonical_key,
            candidate_keys=candidates,
            evidence=evidence,
            resolver_id=self.resolver_id,
            resolver_version=self.resolver_version,
        )
        return StoreResult.success(resolution)


class CompositeIdentityResolver:
    """Run exact resolution first, then isolate plugin candidates for review."""

    resolver_id = "openmed.identity.composite"
    resolver_version = "1.0.0"

    def __init__(
        self,
        exact: ExactIdentityResolver,
        plugin: IdentityCandidatePlugin,
    ) -> None:
        if not isinstance(plugin, IdentityCandidatePlugin):
            raise TypeError("plugin does not satisfy the candidate protocol")
        self.exact = exact
        self.plugin = plugin

    def resolve(
        self,
        request: IdentityResolutionRequest,
    ) -> StoreResult[IdentityResolution]:
        """Return exact results or a review-required probabilistic result."""

        exact_result = self.exact.evaluate(request)
        if not exact_result.ok or exact_result.value is None:
            return exact_result
        if exact_result.value.state != "unmatched":
            return self.exact.store.save_resolution(exact_result.value)

        try:
            candidates_result = self.plugin.candidates(request)
        except Exception:
            return StoreResult.outcome(StoreState.FAILURE, "identity_plugin_failed")
        if not isinstance(candidates_result, StoreResult):
            return StoreResult.outcome(StoreState.FAILURE, "identity_plugin_invalid")
        if not candidates_result.ok or candidates_result.value is None:
            return StoreResult.outcome(
                candidates_result.state,
                candidates_result.code or "identity_plugin_failed",
            )
        if not isinstance(candidates_result.value, tuple) or not all(
            isinstance(item, ProbabilisticIdentityCandidate)
            for item in candidates_result.value
        ):
            return StoreResult.outcome(StoreState.FAILURE, "identity_plugin_invalid")
        candidates = tuple(sorted(set(candidates_result.value)))
        if not candidates:
            return self.exact.store.save_resolution(exact_result.value)
        candidate_keys = tuple(sorted({item.canonical_key for item in candidates}))
        evidence = tuple(_probabilistic_evidence(request, item) for item in candidates)
        resolution = _resolution(
            request,
            state="ambiguous",
            canonical_key=None,
            candidate_keys=candidate_keys,
            evidence=tuple(sorted(evidence)),
            resolver_id=self.resolver_id,
            resolver_version=self.resolver_version,
        )
        return self.exact.store.save_resolution(resolution)


def _authorize(
    request: IdentityResolutionRequest,
    policy: ProjectionPolicy,
) -> StoreResult[IdentityResolution] | None:
    policy_request = ProjectionPolicyRequest(
        operation=ProjectionOperation.READ,
        namespace=ProjectionNamespace.IDENTIFIED,
        purpose=request.purpose,
        role=request.role,
        attributes=request.attributes,
        consent_state="active",
    )
    decision = policy.evaluate(policy_request, decided_at=request.requested_at)
    if decision.outcome is ProjectionPolicyOutcome.ALLOW:
        return None
    state = (
        StoreState.PARTIAL
        if decision.outcome is ProjectionPolicyOutcome.REVIEW
        else StoreState.DENIED
    )
    return StoreResult.outcome(state, decision.reason_code)


def _exact_evidence(link: IdentityLink) -> IdentityEvidence:
    identity = canonical_digest(
        {"link_id": link.link_id, "source_key": link.source_key.fingerprint}
    )
    return IdentityEvidence(
        evidence_id=f"evidence_{identity.removeprefix('sha256:')[:32]}",
        evidence_type="exact_link",
        source_key_fingerprint=link.source_key.fingerprint,
        candidate_key=link.canonical_key,
        evidence_digest=link.evidence_digest,
        method="exact",
        policy_id=link.policy_id,
        policy_version=link.policy_version,
    )


def _probabilistic_evidence(
    request: IdentityResolutionRequest,
    candidate: ProbabilisticIdentityCandidate,
) -> IdentityEvidence:
    source_digest = canonical_digest(
        {"source_keys": [key.fingerprint for key in request.source_keys]}
    )
    identity = canonical_digest(
        {
            "candidate_key": candidate.canonical_key,
            "evidence_digest": candidate.evidence_digest,
            "plugin_id": candidate.plugin_id,
            "plugin_version": candidate.plugin_version,
            "score_basis_points": candidate.score_basis_points,
            "source_digest": source_digest,
        }
    )
    return IdentityEvidence(
        evidence_id=f"evidence_{identity.removeprefix('sha256:')[:32]}",
        evidence_type="candidate_score",
        source_key_fingerprint=source_digest,
        candidate_key=candidate.canonical_key,
        evidence_digest=identity,
        method="probabilistic",
        policy_id=request.policy_id,
        policy_version=request.policy_version,
    )


def _resolution(
    request: IdentityResolutionRequest,
    *,
    state: str,
    canonical_key: str | None,
    candidate_keys: tuple[str, ...],
    evidence: tuple[IdentityEvidence, ...],
    resolver_id: str,
    resolver_version: str,
) -> IdentityResolution:
    identity = canonical_digest(
        {
            "candidate_keys": list(candidate_keys),
            "evidence": [item.to_dict() for item in evidence],
            "request_digest": request.request_digest,
            "resolver_id": resolver_id,
            "resolver_version": resolver_version,
            "state": state,
        }
    )
    return IdentityResolution(
        resolution_id=f"resolution_{identity.removeprefix('sha256:')[:32]}",
        request_digest=request.request_digest,
        entity_type=request.entity_type,
        source_keys=request.source_keys,
        state=state,
        canonical_key=canonical_key,
        candidate_keys=candidate_keys,
        evidence=evidence,
        resolver_id=resolver_id,
        resolver_version=resolver_version,
        policy_id=request.policy_id,
        policy_version=request.policy_version,
        resolved_at=request.requested_at,
        review_required=state in {"ambiguous", "conflict"},
    )
