"""Non-compensable v3.3 private-learning release gate and signed packet.

The evaluator is offline and metadata-only.  A versioned release-evidence
manifest lists every definition-of-done item with its required evidence digest,
schema version and non-compensable status, plus a moved-work section with
recorded rationale.  Producers of the underlying evidence (protocol profile,
site lifecycle, chaos suite, DP utility study, language cohort, synthetic
controls, compatibility matrix and the release-assurance artifacts) are not
imported: each one is referenced only by digest and schema version and is
supplied as an HMAC-signed evidence entry.

Every item passes or fails separately with stable reason codes.  Items are
never collapsed into a score, and a ``READY`` packet is not release authority:
tagging and publishing still require separate explicit authorization.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final

from openmed.core.audit import stable_hash

PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION: Final = (
    "openmed.eval.private_learning_release_manifest.v1"
)
PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION: Final = (
    "openmed.eval.private_learning_release_evidence.v1"
)
PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION: Final = (
    "openmed.eval.private_learning_release_packet.v1"
)

READY: Final = "READY"
NOT_READY: Final = "NOT_READY"

P0: Final = "P0"
P1: Final = "P1"
P2: Final = "P2"
PRIORITIES: Final = (P0, P1, P2)

PASSED: Final = "passed"
FAILED: Final = "failed"
MOVED: Final = "moved"

RELEASE_AUTHORITY_NOT_GRANTED: Final = "not_granted"

MOVE_RATIONALE_CODES: Final = frozenset(
    {
        "blocked_by_external_dependency",
        "deferred_by_scope_policy",
        "split_into_later_milestone",
        "superseded_by_later_work",
    }
)

ITEM_REASON_CODES: Final = frozenset(
    {
        "evidence_commit_mismatch",
        "evidence_digest_mismatch",
        "evidence_duplicate",
        "evidence_failed",
        "evidence_from_future",
        "evidence_invalid",
        "evidence_missing",
        "evidence_schema_mismatch",
        "evidence_signature_invalid",
        "evidence_stale",
        "evidence_unsigned",
        "item_compensable",
        "item_contract_mismatch",
        "item_declared_twice",
        "item_move_refused",
        "item_undeclared",
    }
)
MOVED_REASON_CODES: Final = frozenset(
    {
        "moved_contract_mismatch",
        "moved_duplicate",
        "moved_item_still_declared",
        "moved_p0_refused",
        "moved_privacy_invariant_refused",
        "moved_rationale_invalid",
        "moved_rationale_missing",
        "moved_target_not_later",
    }
)
RELEASE_REASON_CODES: Final = frozenset(
    {
        "evidence_invalid",
        "evidence_unexpected_item",
        "manifest_item_duplicate",
        "manifest_item_order_invalid",
        "manifest_item_unknown",
        "moved_work_refused",
        "release_commit_mismatch",
    }
)

_SHA = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ITEM_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_WORK_REF = re.compile(r"gh-[1-9][0-9]{0,6}")
_MILESTONE = re.compile(r"v(0|[1-9][0-9]{0,3})\.(0|[1-9][0-9]{0,3})")
_SCHEMA = re.compile(r"openmed(?:\.[a-z0-9_]{1,64}){1,6}\.v[1-9][0-9]{0,3}")
_TIMESTAMP = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z")
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_MAX_AGE_SECONDS = 31_536_000
_MIN_KEY_BYTES = 32

_MANIFEST_FIELDS = frozenset(
    {"items", "moved_work", "policy", "release", "schema_version"}
)
_RELEASE_FIELDS = frozenset({"git_commit", "release_id"})
_POLICY_FIELDS = frozenset({"max_evidence_age_seconds"})
_ITEM_FIELDS = frozenset(
    {
        "item_id",
        "non_compensable",
        "priority",
        "privacy_invariant",
        "required_evidence_digest",
        "required_schema_version",
    }
)
_MOVED_FIELDS = frozenset(
    {
        "priority",
        "privacy_invariant",
        "rationale_code",
        "rationale_digest",
        "target_milestone",
        "work_ref",
    }
)
_UNSIGNED_EVIDENCE_FIELDS = frozenset(
    {
        "evidence_digest",
        "evidence_schema_version",
        "generated_at",
        "item_id",
        "schema_version",
        "source_commit",
        "status",
    }
)
_EVIDENCE_FIELDS = _UNSIGNED_EVIDENCE_FIELDS | {"signature"}
_EVIDENCE_STATUSES = frozenset({PASSED, FAILED})

_PACKET_FIELDS = frozenset(
    {
        "catalog_digest",
        "decision",
        "evidence_set_digest",
        "items",
        "manifest_digest",
        "moved_work",
        "packet_digest",
        "reason_codes",
        "release",
        "release_authority",
        "schema_version",
        "signature",
        "source_commit",
    }
)
_PACKET_RELEASE_FIELDS = frozenset({"evaluated_at", "git_commit", "release_id"})
_PACKET_ITEM_FIELDS = frozenset(
    {
        "entry_digest",
        "item_id",
        "priority",
        "privacy_invariant",
        "reason_codes",
        "required_evidence_digest",
        "required_schema_version",
        "status",
    }
)
_PACKET_MOVED_FIELDS = frozenset(
    {
        "accepted",
        "priority",
        "privacy_invariant",
        "rationale_code",
        "rationale_digest",
        "reason_codes",
        "target_milestone",
        "work_ref",
    }
)


class PrivateLearningReleaseError(ValueError):
    """Raised with a fixed, value-free code when configuration is invalid."""


@dataclass(frozen=True, slots=True)
class DoneItemSpec:
    """One closed definition-of-done item in a private-learning release.

    Attributes:
        item_id: Stable item identifier used by manifests, evidence and packets.
        priority: Scope-policy priority (``P0``, ``P1`` or ``P2``).
        privacy_invariant: Whether the item protects a privacy invariant and
            therefore can never be moved out of the release.
        source_ref: Public work reference that owns the evidence producer.
    """

    item_id: str
    priority: str
    privacy_invariant: bool
    source_ref: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.item_id) is not str
            or _ITEM_ID.fullmatch(self.item_id) is None
            or self.priority not in PRIORITIES
            or type(self.privacy_invariant) is not bool
            or (
                self.source_ref is not None
                and (
                    type(self.source_ref) is not str
                    or _WORK_REF.fullmatch(self.source_ref) is None
                )
            )
        ):
            raise PrivateLearningReleaseError("catalog_invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return the published item contract."""
        return {
            "item_id": self.item_id,
            "priority": self.priority,
            "privacy_invariant": self.privacy_invariant,
            "source_ref": self.source_ref,
        }


_V3_3_ITEMS: Final = (
    DoneItemSpec("mvp_scope_closed", P0, False),
    DoneItemSpec("protocol_profile", P0, True, "gh-3209"),
    DoneItemSpec("site_lifecycle", P0, True, "gh-3210"),
    DoneItemSpec("multi_node_chaos", P0, False, "gh-3211"),
    DoneItemSpec("dp_utility_study", P0, True, "gh-3212"),
    DoneItemSpec("language_cohort", P0, False, "gh-3213"),
    DoneItemSpec("synthetic_release_controls", P0, True, "gh-3214"),
    DoneItemSpec("compatibility_matrix", P1, False, "gh-3215"),
    DoneItemSpec("promotion_privacy_gate", P0, True),
    DoneItemSpec("promotion_leakage_gate", P0, True),
    DoneItemSpec("promotion_utility_gate", P0, False),
    DoneItemSpec("promotion_language_gate", P0, False),
    DoneItemSpec("promotion_drift_gate", P0, False),
    DoneItemSpec("promotion_lineage_gate", P0, False),
    DoneItemSpec("promotion_evidence_gate", P0, False),
    DoneItemSpec("release_notes", P0, False),
    DoneItemSpec("migration_guidance", P0, False),
    DoneItemSpec("api_protocol_diff", P0, False),
    DoneItemSpec("security_privacy_evidence", P0, True),
    DoneItemSpec("release_candidate_manifest", P0, False),
)

PRIVATE_LEARNING_RELEASE_CATALOGS: Final[Mapping[str, tuple[DoneItemSpec, ...]]] = {
    "v3.3": _V3_3_ITEMS,
}


def _canonical(payload: Any) -> str:
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _hmac(payload: Mapping[str, Any], key: bytes) -> str:
    value = hmac.new(key, _canonical(payload).encode("utf-8"), hashlib.sha256)
    return "sha256:" + value.hexdigest()


def _valid_key(key: Any) -> bool:
    return isinstance(key, bytes) and len(key) >= _MIN_KEY_BYTES


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _is_list(value: Any) -> bool:
    return isinstance(value, list)


def _exact(value: Any, fields: frozenset[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == fields


def _parse_timestamp(value: Any) -> datetime | None:
    if not _matches(_TIMESTAMP, value):
        return None
    parsed = None
    try:
        parsed = datetime.strptime(value, _TIMESTAMP_FORMAT)
    except ValueError:
        pass
    return parsed.replace(tzinfo=timezone.utc) if parsed is not None else None


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime(_TIMESTAMP_FORMAT)


def _milestone(value: str) -> tuple[int, int]:
    major, minor = value[1:].split(".")
    return int(major), int(minor)


def _catalog_digest(release_id: str) -> str:
    return stable_hash(
        {
            "items": [
                spec.to_dict() for spec in PRIVATE_LEARNING_RELEASE_CATALOGS[release_id]
            ],
            "release_id": release_id,
        }
    )


def _validate_unsigned_entry(entry: Any) -> bool:
    return (
        _exact(entry, _UNSIGNED_EVIDENCE_FIELDS)
        and entry["schema_version"] == PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION
        and _matches(_ITEM_ID, entry["item_id"])
        and _matches(_SCHEMA, entry["evidence_schema_version"])
        and _matches(_DIGEST, entry["evidence_digest"])
        and _matches(_SHA, entry["source_commit"])
        and _parse_timestamp(entry["generated_at"]) is not None
        and entry["status"] in _EVIDENCE_STATUSES
    )


def sign_private_learning_evidence(
    entry: Mapping[str, Any],
    evidence_key: bytes,
) -> dict[str, Any]:
    """Return a copy of an evidence entry with its HMAC-SHA256 signature.

    Producers call this with a caller-controlled key to attest one
    definition-of-done result.  The entry carries only codes, digests, the
    source commit and a UTC timestamp; any other field is refused.

    Args:
        entry: Unsigned evidence fields: ``schema_version``, ``item_id``,
            ``evidence_schema_version``, ``evidence_digest``,
            ``source_commit``, ``generated_at`` and ``status``.
        evidence_key: Injected HMAC key of at least 32 bytes.

    Returns:
        A new mapping containing the entry and its ``signature``.

    Raises:
        PrivateLearningReleaseError: The key or the entry schema is invalid.
    """
    if not _valid_key(evidence_key):
        raise PrivateLearningReleaseError("evidence_key_invalid")
    if not _validate_unsigned_entry(entry):
        raise PrivateLearningReleaseError("evidence_invalid")
    unsigned = {field: entry[field] for field in sorted(_UNSIGNED_EVIDENCE_FIELDS)}
    return {**unsigned, "signature": _hmac(unsigned, evidence_key)}


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not _exact(manifest, _MANIFEST_FIELDS) or (
        manifest["schema_version"] != PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION
    ):
        raise PrivateLearningReleaseError("manifest_invalid")
    release = manifest["release"]
    if not _exact(release, _RELEASE_FIELDS) or not _matches(
        _SHA, release["git_commit"]
    ):
        raise PrivateLearningReleaseError("manifest_invalid")
    if (
        not _matches(_MILESTONE, release["release_id"])
        or release["release_id"] not in PRIVATE_LEARNING_RELEASE_CATALOGS
    ):
        raise PrivateLearningReleaseError("release_unsupported")
    policy = manifest["policy"]
    age = (
        policy.get("max_evidence_age_seconds") if _exact(policy, _POLICY_FIELDS) else 0
    )
    if type(age) is not int or not 1 <= age <= _MAX_AGE_SECONDS:
        raise PrivateLearningReleaseError("manifest_invalid")
    catalog_ids = {
        spec.item_id
        for spec in PRIVATE_LEARNING_RELEASE_CATALOGS[release["release_id"]]
    }
    items = manifest["items"]
    moved = manifest["moved_work"]
    if not _is_list(items) or not _is_list(moved):
        raise PrivateLearningReleaseError("manifest_invalid")
    for item in items:
        if not (
            _exact(item, _ITEM_FIELDS)
            and _matches(_ITEM_ID, item["item_id"])
            and item["priority"] in PRIORITIES
            and type(item["privacy_invariant"]) is bool
            and type(item["non_compensable"]) is bool
            and _matches(_SCHEMA, item["required_schema_version"])
            and _matches(_DIGEST, item["required_evidence_digest"])
        ):
            raise PrivateLearningReleaseError("manifest_invalid")
    for row in moved:
        if not (
            _exact(row, _MOVED_FIELDS)
            and (_matches(_WORK_REF, row["work_ref"]) or row["work_ref"] in catalog_ids)
            and row["priority"] in PRIORITIES
            and type(row["privacy_invariant"]) is bool
            and _matches(_MILESTONE, row["target_milestone"])
            and (
                row["rationale_code"] is None
                or _matches(_ITEM_ID, row["rationale_code"])
            )
            and (
                row["rationale_digest"] is None
                or _matches(_DIGEST, row["rationale_digest"])
            )
        ):
            raise PrivateLearningReleaseError("manifest_invalid")
    return {
        "items": [
            {field: item[field] for field in sorted(_ITEM_FIELDS)} for item in items
        ],
        "moved_work": [
            {field: row[field] for field in sorted(_MOVED_FIELDS)} for row in moved
        ],
        "policy": {"max_evidence_age_seconds": age},
        "release": {
            "git_commit": release["git_commit"],
            "release_id": release["release_id"],
        },
        "schema_version": PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION,
    }


def _evaluate_moved(
    manifest: Mapping[str, Any],
    catalog: Mapping[str, DoneItemSpec],
    declared_ids: set[str],
) -> list[dict[str, Any]]:
    release_id = manifest["release"]["release_id"]
    rows = manifest["moved_work"]
    counts = Counter(row["work_ref"] for row in rows)
    results: list[dict[str, Any]] = []
    for row in rows:
        ref = row["work_ref"]
        reasons: set[str] = set()
        priorities = {row["priority"]}
        privacy = row["privacy_invariant"]
        spec = catalog.get(ref)
        if counts[ref] > 1:
            reasons.add("moved_duplicate")
        if spec is not None:
            if (
                spec.priority != row["priority"]
                or spec.privacy_invariant is not row["privacy_invariant"]
            ):
                reasons.add("moved_contract_mismatch")
            priorities.add(spec.priority)
            privacy = privacy or spec.privacy_invariant
            if ref in declared_ids:
                reasons.add("moved_item_still_declared")
        if P0 in priorities:
            reasons.add("moved_p0_refused")
        if privacy:
            reasons.add("moved_privacy_invariant_refused")
        rationale = row["rationale_code"]
        if rationale is None or row["rationale_digest"] is None:
            reasons.add("moved_rationale_missing")
        elif rationale not in MOVE_RATIONALE_CODES:
            reasons.add("moved_rationale_invalid")
        if _milestone(row["target_milestone"]) <= _milestone(release_id):
            reasons.add("moved_target_not_later")
        results.append(
            {
                "accepted": not reasons,
                "priority": row["priority"],
                "privacy_invariant": row["privacy_invariant"],
                "rationale_code": (
                    rationale if rationale in MOVE_RATIONALE_CODES else None
                ),
                "rationale_digest": row["rationale_digest"],
                "reason_codes": sorted(reasons),
                "target_milestone": row["target_milestone"],
                "work_ref": ref,
            }
        )
    return sorted(results, key=lambda item: (item["work_ref"], _canonical(item)))


def _classify_evidence(
    evidence: Sequence[Any],
    catalog: Mapping[str, DoneItemSpec],
) -> tuple[dict[str, list[tuple[Mapping[str, Any], bool]]], set[str]]:
    by_item: dict[str, list[tuple[Mapping[str, Any], bool]]] = {}
    release_reasons: set[str] = set()
    for raw in evidence:
        item_id = raw.get("item_id") if isinstance(raw, Mapping) else None
        if not _matches(_ITEM_ID, item_id):
            release_reasons.add("evidence_invalid")
            continue
        if item_id not in catalog:
            release_reasons.add("evidence_unexpected_item")
            continue
        fields = set(raw)
        unsigned = {
            field: raw[field] for field in _UNSIGNED_EVIDENCE_FIELDS if field in raw
        }
        signature = raw.get("signature")
        valid = (
            fields in (_EVIDENCE_FIELDS, _UNSIGNED_EVIDENCE_FIELDS)
            and _validate_unsigned_entry(unsigned)
            and (signature is None or _matches(_DIGEST, signature))
        )
        by_item.setdefault(item_id, []).append((raw, valid))
    return by_item, release_reasons


def _entry_reasons(
    entry: Mapping[str, Any],
    declared: Mapping[str, Any] | None,
    *,
    git_commit: str,
    evaluated_at: datetime,
    max_age_seconds: int,
    evidence_key: bytes,
) -> set[str]:
    reasons: set[str] = set()
    unsigned = {field: entry[field] for field in _UNSIGNED_EVIDENCE_FIELDS}
    signature = entry.get("signature")
    if signature is None:
        reasons.add("evidence_unsigned")
    elif not hmac.compare_digest(signature, _hmac(unsigned, evidence_key)):
        reasons.add("evidence_signature_invalid")
    if declared is not None:
        if not hmac.compare_digest(
            entry["evidence_digest"], declared["required_evidence_digest"]
        ):
            reasons.add("evidence_digest_mismatch")
        if entry["evidence_schema_version"] != declared["required_schema_version"]:
            reasons.add("evidence_schema_mismatch")
    if entry["source_commit"] != git_commit:
        reasons.add("evidence_commit_mismatch")
    generated_at = _parse_timestamp(entry["generated_at"])
    age = (evaluated_at - generated_at).total_seconds() if generated_at else -1
    if age < 0:
        reasons.add("evidence_from_future")
    elif age > max_age_seconds:
        reasons.add("evidence_stale")
    if entry["status"] != PASSED:
        reasons.add("evidence_failed")
    return reasons


def evaluate_private_learning_release(
    manifest: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    *,
    source_commit: str,
    evidence_key: bytes,
    signing_key: bytes,
    clock: Callable[[], datetime] | None = None,
) -> PrivateLearningReleasePacket:
    """Evaluate a private-learning release manifest into a signed packet.

    Every definition-of-done item is evaluated independently and in catalog
    order.  The packet is ``READY`` only when each required item has exactly
    one signed, fresh, passing entry for the manifest commit whose digest and
    schema version match the manifest, every moved item has an accepted move,
    and no release-level blocker exists.  Failures never short-circuit: all
    applicable reason codes are reported.

    Args:
        manifest: Versioned release-evidence manifest.
        evidence: Signed evidence entries from independent producers, in any
            order.  Malformed entries become reason codes, not exceptions.
        source_commit: Full lowercase commit SHA of the evaluated checkout.
        evidence_key: Caller-controlled HMAC key that verifies evidence
            entries (at least 32 bytes).
        signing_key: Caller-controlled HMAC key that signs the packet (at
            least 32 bytes).  It may equal ``evidence_key``.
        clock: Injected UTC clock.  Defaults to the system UTC clock.

    Returns:
        A signed, canonical packet, including ``NOT_READY`` decisions.

    Raises:
        PrivateLearningReleaseError: A key, commit, clock or the manifest
            structure is invalid.  Messages are fixed codes.
    """
    if not _valid_key(signing_key):
        raise PrivateLearningReleaseError("signing_key_invalid")
    if not _valid_key(evidence_key):
        raise PrivateLearningReleaseError("evidence_key_invalid")
    if not _matches(_SHA, source_commit):
        raise PrivateLearningReleaseError("source_commit_invalid")
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
        raise PrivateLearningReleaseError("evidence_invalid")
    now = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(now, datetime) or now.utcoffset() is None:
        raise PrivateLearningReleaseError("clock_invalid")
    evaluated_at = now.astimezone(timezone.utc).replace(microsecond=0)

    normalized = _validate_manifest(manifest)
    release = normalized["release"]
    specs = PRIVATE_LEARNING_RELEASE_CATALOGS[release["release_id"]]
    catalog = {spec.item_id: spec for spec in specs}
    order = {spec.item_id: index for index, spec in enumerate(specs)}
    release_reasons: set[str] = set()
    if release["git_commit"] != source_commit:
        release_reasons.add("release_commit_mismatch")

    declared: dict[str, Mapping[str, Any]] = {}
    declared_twice: set[str] = set()
    for item in normalized["items"]:
        item_id = item["item_id"]
        if item_id not in catalog:
            release_reasons.add("manifest_item_unknown")
        elif item_id in declared:
            declared_twice.add(item_id)
            release_reasons.add("manifest_item_duplicate")
        else:
            declared[item_id] = item
    declared_order = [order[item_id] for item_id in declared]
    if declared_order != sorted(declared_order):
        release_reasons.add("manifest_item_order_invalid")

    moved_rows = _evaluate_moved(normalized, catalog, set(declared))
    moved_items = {
        row["work_ref"]
        for row in moved_rows
        if row["accepted"] and row["work_ref"] in catalog
    }
    refused_items = {
        row["work_ref"]
        for row in moved_rows
        if not row["accepted"] and row["work_ref"] in catalog
    }
    if any(not row["accepted"] for row in moved_rows):
        release_reasons.add("moved_work_refused")

    by_item, evidence_reasons = _classify_evidence(evidence, catalog)
    release_reasons |= evidence_reasons
    if moved_items & set(by_item):
        release_reasons.add("evidence_unexpected_item")

    item_rows: list[dict[str, Any]] = []
    for spec in specs:
        item_id = spec.item_id
        row: dict[str, Any] = {
            "entry_digest": None,
            "item_id": item_id,
            "priority": spec.priority,
            "privacy_invariant": spec.privacy_invariant,
            "reason_codes": [],
            "required_evidence_digest": None,
            "required_schema_version": None,
            "status": MOVED,
        }
        if item_id in moved_items:
            item_rows.append(row)
            continue
        reasons: set[str] = set()
        item = declared.get(item_id)
        if item is None:
            # A refused move never removes the item; it stays required.
            reasons.add("item_undeclared")
            if item_id in refused_items:
                reasons.add("item_move_refused")
        else:
            row["required_evidence_digest"] = item["required_evidence_digest"]
            row["required_schema_version"] = item["required_schema_version"]
            if item_id in declared_twice:
                reasons.add("item_declared_twice")
            if (
                item["priority"] != spec.priority
                or item["privacy_invariant"] is not spec.privacy_invariant
            ):
                reasons.add("item_contract_mismatch")
            if item["non_compensable"] is not True:
                reasons.add("item_compensable")
        entries = by_item.get(item_id, [])
        if not entries:
            reasons.add("evidence_missing")
        elif len(entries) > 1:
            reasons.add("evidence_duplicate")
            if not all(valid for _, valid in entries):
                reasons.add("evidence_invalid")
        elif not entries[0][1]:
            reasons.add("evidence_invalid")
        else:
            entry = entries[0][0]
            reasons |= _entry_reasons(
                entry,
                item,
                git_commit=release["git_commit"],
                evaluated_at=evaluated_at,
                max_age_seconds=normalized["policy"]["max_evidence_age_seconds"],
                evidence_key=evidence_key,
            )
            row["entry_digest"] = stable_hash(dict(entry))
        row["reason_codes"] = sorted(reasons)
        row["status"] = FAILED if reasons else PASSED
        item_rows.append(row)

    entry_digests = sorted(
        stable_hash(dict(entry))
        for entries in by_item.values()
        for entry, valid in entries
        if valid
    )
    decision = (
        READY
        if not release_reasons
        and all(row["status"] in (PASSED, MOVED) for row in item_rows)
        else NOT_READY
    )
    unsigned = {
        "catalog_digest": _catalog_digest(release["release_id"]),
        "decision": decision,
        "evidence_set_digest": stable_hash(entry_digests),
        "items": item_rows,
        "manifest_digest": stable_hash(normalized),
        "moved_work": moved_rows,
        "reason_codes": sorted(release_reasons),
        "release": {
            "evaluated_at": _format_timestamp(evaluated_at),
            "git_commit": release["git_commit"],
            "release_id": release["release_id"],
        },
        "release_authority": RELEASE_AUTHORITY_NOT_GRANTED,
        "schema_version": PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION,
        "source_commit": source_commit,
    }
    signed = {**unsigned, "packet_digest": stable_hash(unsigned)}
    return PrivateLearningReleasePacket.from_dict(
        {**signed, "signature": _hmac(signed, signing_key)}
    )


def _valid_reason_list(value: Any, allowed: frozenset[str]) -> bool:
    return (
        _is_list(value)
        and all(type(code) is str and code in allowed for code in value)
        and value == sorted(set(value))
    )


def _valid_packet_item(row: Any, spec: DoneItemSpec) -> bool:
    if not (
        _exact(row, _PACKET_ITEM_FIELDS)
        and row["item_id"] == spec.item_id
        and row["priority"] == spec.priority
        and row["privacy_invariant"] is spec.privacy_invariant
        and _valid_reason_list(row["reason_codes"], ITEM_REASON_CODES)
    ):
        return False
    status = row["status"]
    digests = (row["required_evidence_digest"], row["entry_digest"])
    if status == MOVED:
        return (
            not row["reason_codes"]
            and spec.priority != P0
            and not spec.privacy_invariant
            and digests == (None, None)
            and row["required_schema_version"] is None
        )
    if any(value is not None and not _matches(_DIGEST, value) for value in digests):
        return False
    schema = row["required_schema_version"]
    if schema is not None and not _matches(_SCHEMA, schema):
        return False
    if status == PASSED:
        return (
            not row["reason_codes"]
            and all(value is not None for value in digests)
            and schema is not None
        )
    return status == FAILED and bool(row["reason_codes"])


def _valid_packet_moved(row: Any) -> bool:
    return (
        _exact(row, _PACKET_MOVED_FIELDS)
        and (
            _matches(_WORK_REF, row["work_ref"]) or _matches(_ITEM_ID, row["work_ref"])
        )
        and row["priority"] in PRIORITIES
        and type(row["privacy_invariant"]) is bool
        and _matches(_MILESTONE, row["target_milestone"])
        and (
            row["rationale_code"] is None
            or row["rationale_code"] in MOVE_RATIONALE_CODES
        )
        and (
            row["rationale_digest"] is None
            or _matches(_DIGEST, row["rationale_digest"])
        )
        and _valid_reason_list(row["reason_codes"], MOVED_REASON_CODES)
        and row["accepted"] is (not row["reason_codes"])
    )


def _validate_packet(payload: Any) -> None:
    if not _exact(payload, _PACKET_FIELDS):
        raise PrivateLearningReleaseError("packet_invalid")
    release = payload["release"]
    if not (
        payload["schema_version"] == PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION
        and payload["release_authority"] == RELEASE_AUTHORITY_NOT_GRANTED
        and _exact(release, _PACKET_RELEASE_FIELDS)
        and release["release_id"] in PRIVATE_LEARNING_RELEASE_CATALOGS
        and _matches(_SHA, release["git_commit"])
        and _parse_timestamp(release["evaluated_at"]) is not None
        and _matches(_SHA, payload["source_commit"])
        and all(
            _matches(_DIGEST, payload[field])
            for field in (
                "catalog_digest",
                "evidence_set_digest",
                "manifest_digest",
                "packet_digest",
                "signature",
            )
        )
        and _valid_reason_list(payload["reason_codes"], RELEASE_REASON_CODES)
        and _is_list(payload["items"])
        and _is_list(payload["moved_work"])
    ):
        raise PrivateLearningReleaseError("packet_invalid")
    specs = PRIVATE_LEARNING_RELEASE_CATALOGS[release["release_id"]]
    items = payload["items"]
    moved = payload["moved_work"]
    if (
        len(items) != len(specs)
        or not all(_valid_packet_item(row, spec) for row, spec in zip(items, specs))
        or not all(_valid_packet_moved(row) for row in moved)
        or moved != sorted(moved, key=lambda row: (row["work_ref"], _canonical(row)))
    ):
        raise PrivateLearningReleaseError("packet_invalid")
    accepted = {row["work_ref"] for row in moved if row["accepted"]}
    if any(row["status"] == MOVED and row["item_id"] not in accepted for row in items):
        raise PrivateLearningReleaseError("packet_invalid")
    expected = (
        READY
        if not payload["reason_codes"]
        and all(row["status"] != FAILED for row in items)
        and all(row["accepted"] for row in moved)
        else NOT_READY
    )
    if payload["decision"] != expected:
        raise PrivateLearningReleaseError("packet_invalid")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non_finite")


@dataclass(frozen=True, slots=True)
class PrivateLearningReleasePacket:
    """Immutable canonical release packet with offline HMAC verification."""

    _json: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrivateLearningReleasePacket:
        """Restore a packet after checking its closed schema.

        The schema check enforces catalog item order, closed reason codes and
        decision consistency.  Authenticity must separately pass ``verify``.

        Args:
            payload: Serialized packet fields.

        Returns:
            The canonical packet.

        Raises:
            PrivateLearningReleaseError: The packet schema is invalid.
        """
        canonical = None
        try:
            _validate_packet(payload)
            canonical = _canonical(payload)
        except (
            PrivateLearningReleaseError,
            TypeError,
            ValueError,
            KeyError,
            RecursionError,
        ):
            pass
        if canonical is None:
            raise PrivateLearningReleaseError("packet_invalid")
        return cls(canonical)

    @classmethod
    def from_json(cls, text: str) -> PrivateLearningReleasePacket:
        """Parse stored packet JSON, rejecting duplicate keys and NaN.

        Args:
            text: Packet JSON text.

        Returns:
            The canonical packet.

        Raises:
            PrivateLearningReleaseError: The text is not a valid packet.
        """
        payload = None
        try:
            payload = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_constant,
            )
        except (TypeError, ValueError, RecursionError):
            pass
        if not isinstance(payload, dict):
            raise PrivateLearningReleaseError("packet_invalid")
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return an independent copy of the value-free packet."""
        return json.loads(self._json)

    def to_json(self) -> str:
        """Return byte-stable canonical JSON."""
        return self._json

    @property
    def decision(self) -> str:
        """Return ``READY`` or ``NOT_READY``."""
        return self.to_dict()["decision"]

    @property
    def packet_digest(self) -> str:
        """Return the digest of every unsigned packet field."""
        return self.to_dict()["packet_digest"]

    def failing_items(self) -> tuple[dict[str, Any], ...]:
        """Return every failed item row in catalog order."""
        return tuple(row for row in self.to_dict()["items"] if row["status"] == FAILED)

    def verify(self, signing_key: bytes) -> bool:
        """Verify the packet digest and HMAC signature offline.

        Args:
            signing_key: The caller-controlled packet key (at least 32 bytes).

        Returns:
            Whether the closed schema, packet digest and signature are intact.
        """
        if not _valid_key(signing_key):
            return False
        try:
            payload = self.to_dict()
            _validate_packet(payload)
            signature = payload.pop("signature")
            expected_signature = _hmac(payload, signing_key)
            packet_digest = payload.pop("packet_digest")
            return hmac.compare_digest(
                packet_digest, stable_hash(payload)
            ) and hmac.compare_digest(signature, expected_signature)
        except (
            PrivateLearningReleaseError,
            TypeError,
            ValueError,
            KeyError,
            RecursionError,
        ):
            return False


__all__ = [
    "FAILED",
    "ITEM_REASON_CODES",
    "MOVED",
    "MOVED_REASON_CODES",
    "MOVE_RATIONALE_CODES",
    "NOT_READY",
    "P0",
    "P1",
    "P2",
    "PASSED",
    "PRIORITIES",
    "PRIVATE_LEARNING_RELEASE_CATALOGS",
    "PRIVATE_LEARNING_RELEASE_EVIDENCE_SCHEMA_VERSION",
    "PRIVATE_LEARNING_RELEASE_MANIFEST_SCHEMA_VERSION",
    "PRIVATE_LEARNING_RELEASE_PACKET_SCHEMA_VERSION",
    "READY",
    "RELEASE_AUTHORITY_NOT_GRANTED",
    "RELEASE_REASON_CODES",
    "DoneItemSpec",
    "PrivateLearningReleaseError",
    "PrivateLearningReleasePacket",
    "evaluate_private_learning_release",
    "sign_private_learning_evidence",
]
