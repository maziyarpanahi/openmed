"""Offline declared license, consent and lineage gates for parallel corpus shards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import islice

from openmed.training.synthetic.spdx_identifier import normalize_spdx_identifier

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LICENSE = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,127}\Z")
_USES = frozenset({"training", "evaluation", "export"})
_TRANSLATIONS = frozenset({"none", "human", "local_model", "unknown"})
_CONSENTS = frozenset({"synthetic", "explicit", "not_required", "unknown", "denied"})
_DERIVATIVES = frozenset(
    {"redistributable", "user_supplied_only", "forbidden", "unknown"}
)


def _optional_digest(value: object) -> None:
    if value is not None and (
        not isinstance(value, str) or not _DIGEST.fullmatch(value)
    ):
        raise ValueError("invalid corpus evidence digest")


def _bounded(values, maximum):
    if isinstance(values, (str, bytes)):
        raise ValueError("invalid corpus metadata collection")
    try:
        result = tuple(islice(values, maximum + 1))
    except Exception:
        raise ValueError("invalid corpus metadata collection") from None
    if len(result) > maximum:
        raise ValueError("corpus metadata collection limit exceeded")
    return result


@dataclass(frozen=True, slots=True)
class ParallelCorpusShard:
    """Declared shard governance metadata, excluding corpus text and source URLs.

    Evidence digests and permission claims require independent owner review;
    this class validates metadata structure, not the truth of declared rights.
    Restricted licenses never become bundled or exportable through review alone.
    """

    shard_digest: str
    source_digest: str | None = None
    license_id: str | None = field(default=None, repr=False)
    translation_method: str = "unknown"
    consent_category: str = "unknown"
    permitted_uses: tuple[str, ...] = ()
    derivative_policy: str = "unknown"
    lineage_digests: tuple[str, ...] = ()
    translation_evidence_digest: str | None = None
    consent_evidence_digest: str | None = None
    license_review_digest: str | None = None
    user_supplied: bool = False

    def __post_init__(self) -> None:
        if self.shard_digest is None:
            raise ValueError("corpus shard digest is required")
        for value in (
            self.shard_digest,
            self.source_digest,
            self.translation_evidence_digest,
            self.consent_evidence_digest,
            self.license_review_digest,
        ):
            _optional_digest(value)
        if self.license_id is not None and (
            not isinstance(self.license_id, str)
            or not _LICENSE.fullmatch(self.license_id)
        ):
            raise ValueError("invalid corpus license identifier")
        for value, allowed in (
            (self.translation_method, _TRANSLATIONS),
            (self.consent_category, _CONSENTS),
            (self.derivative_policy, _DERIVATIVES),
        ):
            if not isinstance(value, str) or value not in allowed:
                raise ValueError("invalid corpus governance category")
        if type(self.user_supplied) is not bool:
            raise ValueError("invalid corpus input custody declaration")
        uses = _bounded(self.permitted_uses, 3)
        if any(
            not isinstance(value, str) or value not in _USES for value in uses
        ) or len(set(uses)) != len(uses):
            raise ValueError("invalid or duplicate permitted corpus use")
        lineage = _bounded(self.lineage_digests, 32)
        for value in lineage:
            _optional_digest(value)
            if value is None:
                raise ValueError("corpus lineage digest is required")
        if len(set(lineage)) != len(lineage):
            raise ValueError("duplicate corpus lineage reference")
        object.__setattr__(self, "permitted_uses", tuple(sorted(uses)))
        object.__setattr__(self, "lineage_digests", lineage)


def corpus_use_report(shard: ParallelCorpusShard, use: str) -> dict[str, object]:
    """Evaluate declared rights and provenance without reading or enrolling data.

    Args:
        shard: Immutable declared source, translation, rights and lineage metadata.
        use: Training, evaluation or export, explicitly requested by the caller.

    Returns:
        A deterministic report with shard digest, use, license class, outcome and
        controlled reasons only. No raw license labels or source metadata is echoed.
    """
    if (
        not isinstance(shard, ParallelCorpusShard)
        or not isinstance(use, str)
        or use not in _USES
    ):
        raise ValueError("invalid corpus permission request")
    reasons = []
    if shard.source_digest is None:
        reasons.append("source_missing")
    if not shard.lineage_digests:
        reasons.append("lineage_missing")
    else:
        if shard.lineage_digests[0] != shard.source_digest:
            reasons.append("lineage_source_unbound")
        if shard.lineage_digests[-1] != shard.shard_digest:
            reasons.append("lineage_shard_unbound")
    if shard.translation_method == "unknown":
        reasons.append("translation_unknown")
    elif shard.translation_method != "none":
        if shard.translation_evidence_digest is None:
            reasons.append("translation_evidence_missing")
        if len(shard.lineage_digests) < 2:
            reasons.append("translation_lineage_missing")
    if shard.consent_category in {"unknown", "denied"}:
        reasons.append("consent_" + shard.consent_category)
    elif shard.consent_category == "explicit" and shard.consent_evidence_digest is None:
        reasons.append("consent_evidence_missing")
    if use not in shard.permitted_uses:
        reasons.append("use_not_permitted")
    if shard.derivative_policy in {"unknown", "forbidden"}:
        reasons.append("derivative_" + shard.derivative_policy)
    if shard.license_id is None:
        license_class = "missing"
        reasons.append("license_missing")
    else:
        normalized = normalize_spdx_identifier(shard.license_id)
        license_class = (
            "permissive" if normalized.is_permissive else "restricted_or_unknown"
        )
        if not normalized.is_permissive:
            if shard.license_review_digest is None:
                reasons.append("restricted_license_unreviewed")
            if not shard.user_supplied:
                reasons.append("restricted_input_not_user_supplied")
            if shard.derivative_policy != "user_supplied_only":
                reasons.append("restricted_derivative_not_user_supplied_only")
            if use == "export":
                reasons.append("restricted_export_prohibited")
    if shard.derivative_policy == "user_supplied_only":
        if not shard.user_supplied:
            reasons.append("restricted_input_not_user_supplied")
        if use == "export":
            reasons.append("user_supplied_export_prohibited")
    return {
        "schema_version": 1,
        "shard_digest": shard.shard_digest,
        "use": use,
        "license_class": license_class,
        "outcome": "allow" if not reasons else "deny",
        "reasons": sorted(set(reasons)),
    }


def require_corpus_permission(shard: ParallelCorpusShard, use: str) -> None:
    """Fail before a caller's training/export action when declared governance fails.

    This explicit gate must be called at the controlled action entry point; it
    does not intercept unrelated training functions or infer legal permission.
    """
    if corpus_use_report(shard, use)["outcome"] != "allow":
        raise ValueError("corpus_permission_denied")


def render_corpus_use_report(shard: ParallelCorpusShard, use: str) -> str:
    """Render a fresh, metadata-only deterministic gate result as JSON."""
    return (
        json.dumps(corpus_use_report(shard, use), sort_keys=True, separators=(",", ":"))
        + "\n"
    )
