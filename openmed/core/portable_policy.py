"""Portable OpenMedKit policy subset: canonical IR, digests, and decisions.

The Python policy runtime (:mod:`openmed.core.policy`) is richer than what the
on-device kits can execute. This module declares the bounded *portable subset*
that every kit (Python, Swift, Kotlin, and TypeScript) must execute
identically, and is the Python reference implementation of it.

* :data:`PORTABLE_PROFILES` names the profiles every kit ships. Every other
  Python policy file is listed in :data:`EXCLUDED_PROFILES` with a value-free
  reason, and loading one through the portable loader fails with
  ``profile_not_portable`` instead of being approximated.
* :func:`build_resources` lowers the Python profiles, label taxonomy, and
  threshold matrix into versioned canonical JSON documents
  (``openmed.portable_policy`` version 1). The generator script
  ``scripts/parity/generate_portable_policies.py`` writes these exact bytes into
  the Swift, Android, and TypeScript resource directories; its ``--check`` mode
  fails CI when a kit copy is stale.
* :func:`load_portable_policy` verifies a profile against the manifest digest
  (and an optional caller-pinned digest), the taxonomy digest, the IR version,
  and the capabilities this runtime supports, failing closed with a stable
  :class:`PortablePolicyError` reason code otherwise.
* :func:`decide` applies a profile to normalized detections: label
  normalization, semantic-unit merging, threshold decisions, action lookup
  with caller overrides and the strict no-leak upgrade, overlap resolution,
  and deterministic keyed surrogates. Its output is a kit-parity result
  (:mod:`openmed.core.kit_parity`) plus one decision record per candidate.

Profiles, the manifest, the taxonomy, decisions, and errors carry labels,
offsets, actions, digests, and reason codes only. Source text, span surface
text, and surrogate keys never enter them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any, Final

from . import kit_parity
from .kit_parity import KitParityResult, KitParitySpan

IR_SCHEMA: Final = "openmed.portable_policy"
IR_VERSION: Final = 1
MANIFEST_SCHEMA: Final = "openmed.portable_policy_manifest"
MANIFEST_VERSION: Final = 1
TAXONOMY_SCHEMA: Final = "openmed.portable_taxonomy"
TAXONOMY_VERSION: Final = 1

MANIFEST_FILE: Final = "portable_policy_manifest.json"
TAXONOMY_FILE: Final = "portable_policy_taxonomy.json"

MERGE_ALGORITHM: Final = "openmed.portable_merge"
MERGE_VERSION: Final = 1
SURROGATE_ALGORITHM: Final = "openmed.portable_surrogate.hmac_sha256"
SURROGATE_VERSION: Final = 1
SURROGATE_DOMAIN: Final = b"openmed.portable_surrogate.v1"
SURROGATE_KEY_MIN_BYTES: Final = 16
SURROGATE_KEY_MAX_BYTES: Final = 64
SURROGATE_TOKEN_BYTES: Final = 8
HASH_TOKEN_BYTES: Final = 12
CONFIDENCE_SCALE: Final = kit_parity.CONFIDENCE_SCALE

#: Profiles every kit ships and executes identically.
PORTABLE_PROFILES: Final[tuple[str, ...]] = (
    "australia_privacy_act",
    "canada_pipeda",
    "clinical_minimal_redaction",
    "gdpr_art9_health",
    "gdpr_pseudonymization",
    "hipaa_expert_review_assist",
    "hipaa_safe_harbor",
    "research_limited_dataset",
    "strict_no_leak",
)
DEFAULT_PROFILE: Final = "hipaa_safe_harbor"

#: Reason codes for Python policy files outside the portable subset.
EXCLUDED_PYTHON_ONLY: Final = "python_only_profile"
EXCLUDED_STRUCTURED_SCHEMA: Final = "structured_schema_policy"

#: Capabilities this runtime implements. A profile requiring anything else is
#: refused with ``capability_missing`` rather than run with a missing feature.
SUPPORTED_CAPABILITIES: Final[tuple[str, ...]] = (
    "actions.v1",
    "labels.portable_taxonomy.v1",
    "merge.cluster_union.v1",
    "merge.semantic_units.v1",
    "strict_no_leak.v1",
    "surrogate.hmac_sha256.v1",
    "thresholds.keep_floor.v1",
)
_BASE_CAPABILITIES: Final = (
    "actions.v1",
    "labels.portable_taxonomy.v1",
    "merge.cluster_union.v1",
    "merge.semantic_units.v1",
    "thresholds.keep_floor.v1",
)

#: Portable actions, identical to the kit-parity action vocabulary.
PORTABLE_ACTIONS: Final[tuple[str, ...]] = kit_parity.KIT_ACTIONS
#: Redaction strength used by ``high_recall_union`` arbitration.
ACTION_STRENGTH: Final[Mapping[str, int]] = {
    "keep": 0,
    "replace": 1,
    "hash": 2,
    "mask": 3,
    "redact": 4,
    "remove": 5,
}
ARBITRATION_MODES: Final[frozenset[str]] = frozenset({"balanced", "high_recall_union"})
DETECTION_SOURCES: Final[frozenset[str]] = frozenset({"model", "semantic"})

_BIOES_PREFIX: Final = re.compile(r"^[BIES]-")
_KEY_STRIP: Final = re.compile(r"[^a-z0-9]")
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")


class PortablePolicyError(ValueError):
    """A value-free portable policy failure.

    Attributes:
        code: Stable reason code such as ``resource_digest_mismatch``.
        field_path: JSON path of the offending field, for example
            ``$.detections[2].end``. Map keys are never echoed.
    """

    def __init__(self, code: str, field_path: str = "$") -> None:
        self.code = code
        self.field_path = field_path
        super().__init__(f"{code} at {field_path}")


@dataclass(frozen=True)
class PortableTaxonomy:
    """Label taxonomy shared by every portable profile.

    Attributes:
        labels: Canonical labels in sorted order.
        policy_labels: Policy class for each canonical label.
        specificity: Arbitration specificity rank for each canonical label.
        aliases: Normalized label key to canonical label.
        unknown_label: Canonical label for unrecognized detector labels.
        digest: ``sha256:`` digest of the taxonomy resource bytes.
    """

    labels: tuple[str, ...]
    policy_labels: Mapping[str, str]
    specificity: Mapping[str, int]
    aliases: Mapping[str, str]
    unknown_label: str
    digest: str

    def normalize(self, label: str) -> tuple[str, bool]:
        """Normalize a detector label to the portable taxonomy.

        Args:
            label: Detector label (ASCII, matching the kit-parity pattern).

        Returns:
            ``(canonical_label, recognized)``; unrecognized labels map to
            :attr:`unknown_label` with ``recognized`` false.
        """
        canonical = self.aliases.get(label_key(label))
        if canonical is None:
            return self.unknown_label, False
        return canonical, True


@dataclass(frozen=True)
class PortableThreshold:
    """Threshold entry in parts per million.

    Attributes:
        keep_floor_ppm: Minimum confidence for a candidate to be retained.
        escalate_below_ppm: Retained candidates below this are flagged for
            review (cascade metadata only; no cascade runs portably).
    """

    keep_floor_ppm: int
    escalate_below_ppm: int


@dataclass(frozen=True)
class PortablePolicy:
    """A verified portable policy profile.

    Attributes:
        name: Canonical profile name.
        digest: ``sha256:`` digest of the profile resource bytes.
        taxonomy: Shared label taxonomy.
        actions: Action for every taxonomy label.
        default_action: Profile default action (metadata; every label has an
            explicit action).
        arbitration_mode: ``balanced`` or ``high_recall_union``.
        strict_no_leak: Whether ``keep`` is upgraded to ``mask``.
        threshold_profile: Source threshold profile name.
        default_threshold: Threshold for labels without an entry.
        label_thresholds: Per-label thresholds.
        required_capabilities: Capabilities the profile requires.
        excluded_features: Python-only features of the source profile that
            the portable subset does not execute.
        forced_cascade_tiers: Cascade tiers (metadata only).
        document: The decoded profile document.
    """

    name: str
    digest: str
    taxonomy: PortableTaxonomy
    actions: Mapping[str, str]
    default_action: str
    arbitration_mode: str
    strict_no_leak: bool
    threshold_profile: str
    default_threshold: PortableThreshold
    label_thresholds: Mapping[str, PortableThreshold]
    required_capabilities: tuple[str, ...]
    excluded_features: tuple[str, ...]
    forced_cascade_tiers: tuple[str, ...]
    document: Mapping[str, Any]

    def threshold_for(self, canonical_label: str) -> PortableThreshold:
        """Return the threshold entry for a canonical label."""
        return self.label_thresholds.get(canonical_label, self.default_threshold)

    def action_for(self, label: str) -> str:
        """Return the resolved action for a detector or canonical label.

        Applies the strict no-leak upgrade (``keep`` becomes ``mask``).
        """
        canonical, _ = self.taxonomy.normalize(label)
        action = self.actions[canonical]
        if self.strict_no_leak and action == "keep":
            return "mask"
        return action


@dataclass(frozen=True)
class PortableDetection:
    """One normalized detector span offered to :func:`decide`.

    Attributes:
        label: Detector label (ASCII, matching the kit-parity pattern).
        start: Inclusive Unicode scalar offset.
        end: Exclusive Unicode scalar offset.
        confidence_ppm: Confidence in parts per million.
        source: ``model`` for model spans, ``semantic`` for pattern units.
        validated: For semantic units, whether the pattern validator passed.
    """

    label: str
    start: int
    end: int
    confidence_ppm: int
    source: str = "model"
    validated: bool = True


@dataclass(frozen=True)
class PortableDecision:
    """Decision record for one candidate span.

    Attributes:
        inputs: Indexes of the input detections the candidate came from.
        kind: ``model``, ``merged`` (semantic unit plus model spans), or
            ``semantic_only``.
        label: Detector label carried by the candidate.
        canonical_label: Portable canonical label.
        recognized: Whether the label resolved through the taxonomy.
        start: Candidate start offset.
        end: Candidate end offset.
        confidence_ppm: Candidate confidence in parts per million.
        keep_floor_ppm: Threshold the candidate was compared against.
        escalate: Whether a retained candidate is below ``escalate_below``.
        action: Resolved action for the candidate's label.
        strict_upgrade: Whether strict no-leak upgraded ``keep`` to ``mask``.
        outcome: ``applied``, ``overlap_loser``, ``duplicate``, or
            ``below_threshold``.
        span: Index of the output span the candidate contributed to.
    """

    inputs: tuple[int, ...]
    kind: str
    label: str
    canonical_label: str
    recognized: bool
    start: int
    end: int
    confidence_ppm: int
    keep_floor_ppm: int
    escalate: bool
    action: str
    strict_upgrade: bool
    outcome: str
    span: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical, value-free JSON form of the decision."""
        record: dict[str, Any] = {
            "action": self.action,
            "canonical_label": self.canonical_label,
            "confidence_ppm": self.confidence_ppm,
            "end": self.end,
            "escalate": self.escalate,
            "inputs": list(self.inputs),
            "keep_floor_ppm": self.keep_floor_ppm,
            "kind": self.kind,
            "label": self.label,
            "outcome": self.outcome,
            "recognized": self.recognized,
            "start": self.start,
            "strict_upgrade": self.strict_upgrade,
        }
        if self.span is not None:
            record["span"] = self.span
        return record


@dataclass(frozen=True)
class PortableDecisionResult:
    """Output of :func:`decide`.

    Attributes:
        policy_name: Canonical profile name.
        policy_digest: Profile digest the decisions were made under.
        result: Kit-parity projection of the de-identified output.
        decisions: One record per candidate, in candidate order.
    """

    policy_name: str
    policy_digest: str
    result: KitParityResult
    decisions: tuple[PortableDecision, ...]


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def canonical_bytes(document: Mapping[str, Any]) -> bytes:
    """Return the canonical resource bytes for a portable document.

    Resources are ASCII JSON with sorted keys, two-space indentation, and a
    trailing newline, so every kit ships byte-identical copies.

    Args:
        document: JSON-compatible mapping.

    Returns:
        The canonical UTF-8 (ASCII) bytes.
    """
    text = json.dumps(
        document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    )
    return (text + "\n").encode("ascii")


def digest_bytes(data: bytes) -> str:
    """Return the ``sha256:`` digest of resource bytes."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def label_key(label: str) -> str:
    """Return the normalization key of a detector label.

    Strips surrounding whitespace and one BIOES prefix (``B-``, ``I-``,
    ``E-``, ``S-``), lowercases, and drops everything but ``a-z`` and ``0-9``.
    This mirrors :func:`openmed.core.labels.normalize_label` for ASCII labels.
    """
    stripped = _BIOES_PREFIX.sub("", label.strip(), count=1)
    return _KEY_STRIP.sub("", stripped.lower())


def confidence_to_ppm(value: float) -> int:
    """Encode a probability as integer parts per million (half-up)."""
    return kit_parity.confidence_to_ppm(value)


# ---------------------------------------------------------------------------
# Building the IR from the Python profiles
# ---------------------------------------------------------------------------


def build_taxonomy_document() -> dict[str, Any]:
    """Lower the Python label taxonomy into the portable taxonomy document.

    The alias table holds every normalized key of the Python alias map plus
    the key of every canonical label that the alias map does not already
    claim, so portable normalization equals
    :func:`openmed.core.labels.normalize_label` for ASCII labels outside the
    Chinese CMeEE mapping (a language-conditional, Python-only feature).

    Returns:
        The taxonomy document.
    """
    from .arbitration import specificity_rank
    from .labels import _ALIAS_MAP, CANONICAL_LABELS, OTHER, policy_label_for

    aliases: dict[str, str] = {}
    for key, canonical in _ALIAS_MAP.items():
        if label_key(key) == key and key:
            aliases[key] = canonical
    for label in CANONICAL_LABELS:
        aliases.setdefault(label_key(label), label)
    labels = {
        label: {
            "policy_label": policy_label_for(label),
            "specificity": specificity_rank(label),
        }
        for label in sorted(CANONICAL_LABELS)
    }
    return {
        "aliases": dict(sorted(aliases.items())),
        "labels": labels,
        "name": TAXONOMY_SCHEMA,
        "normalization": {
            "bioes_prefix": "^[BIES]-",
            "key": "lowercase_ascii_alphanumeric",
            "unknown": "unknown_label",
        },
        "unknown_label": OTHER,
        "version": TAXONOMY_VERSION,
    }


def build_profile_document(name: str, taxonomy_digest: str) -> dict[str, Any]:
    """Lower one Python policy profile into its portable IR document.

    Args:
        name: A name from :data:`PORTABLE_PROFILES`.
        taxonomy_digest: Digest of the taxonomy resource the profile binds to.

    Returns:
        The profile IR document.

    Raises:
        PortablePolicyError: ``profile_not_portable`` for names outside the
            subset, or ``action_not_portable`` when the Python profile uses an
            action the kits cannot execute.
    """
    from .africa_context import profile_defaults_for
    from .labels import CANONICAL_LABELS
    from .policy import load_policy

    if name not in PORTABLE_PROFILES:
        raise PortablePolicyError("profile_not_portable", "$.name")
    profile = load_policy(name)
    source = _core_resource_bytes("policies", f"{name}.json")
    threshold_source = _core_resource_bytes("thresholds.json")
    matrix = json.loads(threshold_source.decode("utf-8"))
    thresholds = matrix["profiles"][profile.threshold_profile]

    actions = {label: profile.actions[label] for label in sorted(CANONICAL_LABELS)}
    if any(action not in PORTABLE_ACTIONS for action in actions.values()):
        raise PortablePolicyError("action_not_portable", "$.actions")

    label_thresholds: dict[str, dict[str, int]] = {}
    language_specific = False
    for label, entries in sorted((thresholds.get("labels") or {}).items()):
        if any(language != "*" for language in entries):
            language_specific = True
        if "*" in entries:
            label_thresholds[label] = _threshold_ppm(entries["*"])

    excluded: set[str] = {"script_recall_floors"}
    if "replace" in actions.values():
        excluded.add("realistic_surrogates")
    if profile.safety_sweep_mandatory:
        excluded.add("safety_sweep")
    if profile.forced_cascade_tiers:
        excluded.add("forced_cascade_execution")
    if profile.keep_mapping:
        excluded.add("keep_mapping")
    if profile.reversible_id:
        excluded.add("reversible_id")
    if language_specific:
        excluded.add("language_specific_thresholds")
    defaults = profile_defaults_for(name)
    if defaults.get("actions") or defaults.get("policy_label_actions"):
        excluded.add("context_defaults")

    required = set(_BASE_CAPABILITIES)
    if profile.strict_no_leak:
        required.add("strict_no_leak.v1")
    if "replace" in actions.values():
        required.add("surrogate.hmac_sha256.v1")

    return {
        "actions": actions,
        "arbitration_mode": profile.arbitration_mode,
        "default_action": profile.default_action,
        "default_action_bias": profile.default_action_bias,
        "excluded_features": sorted(excluded),
        "forced_cascade_tiers": list(profile.forced_cascade_tiers),
        "ir_schema": IR_SCHEMA,
        "ir_version": IR_VERSION,
        "keep_mapping": profile.keep_mapping,
        "merge": {"algorithm": MERGE_ALGORITHM, "version": MERGE_VERSION},
        "name": name,
        "policy_label_actions": dict(sorted(profile.policy_label_actions.items())),
        "posture": profile.posture,
        "required_capabilities": sorted(required),
        "reversible_id": profile.reversible_id,
        "safety_sweep_mandatory": profile.safety_sweep_mandatory,
        "schema_version": profile.schema_version,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "strict_no_leak": profile.strict_no_leak,
        "surrogate": {"algorithm": SURROGATE_ALGORITHM, "version": SURROGATE_VERSION},
        "taxonomy_digest": taxonomy_digest,
        "threshold_profile": profile.threshold_profile,
        "thresholds": {
            "default": _threshold_ppm(thresholds["default"]),
            "labels": label_thresholds,
            "source_sha256": hashlib.sha256(threshold_source).hexdigest(),
        },
    }


def excluded_profiles() -> dict[str, str]:
    """Return every Python policy outside the portable subset with its reason.

    Returns:
        Mapping of policy name to a value-free exclusion reason.

    Raises:
        PortablePolicyError: ``policy_unclassified`` when a bundled policy
            file is neither a profile nor a structured schema policy.
    """
    from .policy import CANONICAL_POLICY_NAMES

    excluded: dict[str, str] = {}
    for name in CANONICAL_POLICY_NAMES:
        if name not in PORTABLE_PROFILES:
            excluded[name] = EXCLUDED_PYTHON_ONLY
    directory = resources.files("openmed.core").joinpath("policies")
    for entry in sorted(directory.iterdir(), key=lambda item: item.name):
        if not entry.name.endswith(".json"):
            continue
        name = entry.name[: -len(".json")]
        if name in PORTABLE_PROFILES or name in excluded:
            continue
        payload = json.loads(entry.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "base_policy" in payload:
            excluded[name] = EXCLUDED_STRUCTURED_SCHEMA
        else:
            raise PortablePolicyError("policy_unclassified", "$.excluded_profiles")
    return dict(sorted(excluded.items()))


def build_resources() -> dict[str, bytes]:
    """Build every portable resource file from the Python sources.

    Returns:
        Mapping of resource file name to canonical bytes: one file per
        portable profile, the taxonomy, and the manifest.
    """
    from .policy import POLICY_ALIASES

    files: dict[str, bytes] = {}
    taxonomy = canonical_bytes(build_taxonomy_document())
    taxonomy_digest = digest_bytes(taxonomy)
    files[TAXONOMY_FILE] = taxonomy
    digests: dict[str, str] = {}
    for name in PORTABLE_PROFILES:
        data = canonical_bytes(build_profile_document(name, taxonomy_digest))
        files[f"{name}.json"] = data
        digests[name] = digest_bytes(data)
    manifest = {
        "default_profile": DEFAULT_PROFILE,
        "excluded_profiles": excluded_profiles(),
        "ir_schema": IR_SCHEMA,
        "ir_version": IR_VERSION,
        "profile_aliases": dict(sorted(POLICY_ALIASES.items())),
        "profiles": digests,
        "schema": MANIFEST_SCHEMA,
        "schema_version": MANIFEST_VERSION,
        "taxonomy_digest": taxonomy_digest,
    }
    files[MANIFEST_FILE] = canonical_bytes(manifest)
    return dict(sorted(files.items()))


def bundled_resources() -> dict[str, bytes]:
    """Return the Python reference resources (built from the sources)."""
    return _cached_resources()


# ---------------------------------------------------------------------------
# Loading and verification
# ---------------------------------------------------------------------------


def resolve_profile_name(name: object, manifest: Mapping[str, Any]) -> str:
    """Resolve a profile name or alias against a verified manifest.

    Args:
        name: Caller-supplied profile name.
        manifest: Decoded manifest document.

    Returns:
        The canonical portable profile name.

    Raises:
        PortablePolicyError: ``profile_name_invalid``, ``profile_not_portable``
            for Python-only and structured policies, or ``profile_unknown``.
    """
    if not isinstance(name, str) or not name.strip():
        raise PortablePolicyError("profile_name_invalid", "$.name")
    normalized = name.strip().lower().replace("-", "_")
    normalized = manifest["profile_aliases"].get(normalized, normalized)
    if normalized in manifest["profiles"]:
        return str(normalized)
    if normalized in manifest["excluded_profiles"]:
        raise PortablePolicyError("profile_not_portable", "$.name")
    raise PortablePolicyError("profile_unknown", "$.name")


def decode_manifest(data: bytes) -> dict[str, Any]:
    """Decode and validate manifest bytes.

    Args:
        data: Manifest resource bytes.

    Returns:
        The manifest document.

    Raises:
        PortablePolicyError: ``manifest_invalid``,
            ``manifest_version_unsupported``, or ``ir_version_unsupported``.
    """
    document = _json_object(data, "manifest_invalid")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise PortablePolicyError("manifest_invalid", "$.schema")
    if not _is_int(document.get("schema_version")) or (
        document["schema_version"] != MANIFEST_VERSION
    ):
        raise PortablePolicyError("manifest_version_unsupported", "$.schema_version")
    if document.get("ir_schema") != IR_SCHEMA:
        raise PortablePolicyError("manifest_invalid", "$.ir_schema")
    if not _is_int(document.get("ir_version")) or document["ir_version"] != IR_VERSION:
        raise PortablePolicyError("ir_version_unsupported", "$.ir_version")
    for field in ("profiles", "excluded_profiles", "profile_aliases"):
        value = document.get(field)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise PortablePolicyError("manifest_invalid", f"$.{field}")
    if not all(_DIGEST.match(item) for item in document["profiles"].values()):
        raise PortablePolicyError("manifest_invalid", "$.profiles")
    digest = document.get("taxonomy_digest")
    if not isinstance(digest, str) or not _DIGEST.match(digest):
        raise PortablePolicyError("manifest_invalid", "$.taxonomy_digest")
    if document.get("default_profile") not in document["profiles"]:
        raise PortablePolicyError("manifest_invalid", "$.default_profile")
    return document


def decode_taxonomy(data: bytes, digest: str) -> PortableTaxonomy:
    """Decode and validate taxonomy bytes.

    Args:
        data: Taxonomy resource bytes.
        digest: Digest of ``data`` (already verified by the caller).

    Returns:
        The taxonomy.

    Raises:
        PortablePolicyError: ``taxonomy_invalid``.
    """
    document = _json_object(data, "taxonomy_invalid")
    if document.get("name") != TAXONOMY_SCHEMA:
        raise PortablePolicyError("taxonomy_invalid", "$.name")
    if document.get("version") != TAXONOMY_VERSION:
        raise PortablePolicyError("taxonomy_invalid", "$.version")
    raw_labels = document.get("labels")
    if not isinstance(raw_labels, dict) or not raw_labels:
        raise PortablePolicyError("taxonomy_invalid", "$.labels")
    policy_labels: dict[str, str] = {}
    specificity: dict[str, int] = {}
    for label, entry in raw_labels.items():
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("policy_label"), str)
            or not _is_int(entry.get("specificity"))
        ):
            raise PortablePolicyError("taxonomy_invalid", "$.labels")
        policy_labels[label] = entry["policy_label"]
        specificity[label] = entry["specificity"]
    aliases = document.get("aliases")
    if not isinstance(aliases, dict) or not all(
        isinstance(key, str) and value in raw_labels for key, value in aliases.items()
    ):
        raise PortablePolicyError("taxonomy_invalid", "$.aliases")
    unknown = document.get("unknown_label")
    if unknown not in raw_labels:
        raise PortablePolicyError("taxonomy_invalid", "$.unknown_label")
    return PortableTaxonomy(
        labels=tuple(sorted(raw_labels)),
        policy_labels=policy_labels,
        specificity=specificity,
        aliases=dict(aliases),
        unknown_label=str(unknown),
        digest=digest,
    )


def decode_profile(
    document: Mapping[str, Any],
    *,
    name: str,
    digest: str,
    taxonomy: PortableTaxonomy,
    supported_capabilities: Sequence[str] = SUPPORTED_CAPABILITIES,
) -> PortablePolicy:
    """Validate a decoded profile document and return the policy.

    Checks run in a fixed order shared by every kit: IR schema, IR version,
    name, taxonomy digest, required capabilities, merge algorithm, surrogate
    algorithm, arbitration mode, strict no-leak flag, actions, default action,
    and thresholds.

    Args:
        document: Decoded profile document.
        name: Canonical profile name the document must declare.
        digest: Digest of the profile resource bytes.
        taxonomy: Verified taxonomy.
        supported_capabilities: Capabilities the runtime implements.

    Returns:
        The verified policy.

    Raises:
        PortablePolicyError: On the first violation.
    """
    if not isinstance(document, Mapping):
        raise PortablePolicyError("profile_invalid")
    if document.get("ir_schema") != IR_SCHEMA:
        raise PortablePolicyError("profile_invalid", "$.ir_schema")
    if not _is_int(document.get("ir_version")) or document["ir_version"] != IR_VERSION:
        raise PortablePolicyError("ir_version_unsupported", "$.ir_version")
    if document.get("name") != name:
        raise PortablePolicyError("profile_invalid", "$.name")
    if document.get("taxonomy_digest") != taxonomy.digest:
        raise PortablePolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest")
    required = document.get("required_capabilities")
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise PortablePolicyError("profile_invalid", "$.required_capabilities")
    supported = set(supported_capabilities)
    if any(item not in supported for item in required):
        raise PortablePolicyError("capability_missing", "$.required_capabilities")
    merge = document.get("merge")
    if not isinstance(merge, Mapping) or (
        merge.get("algorithm") != MERGE_ALGORITHM
        or not _is_int(merge.get("version"))
        or merge.get("version") != MERGE_VERSION
    ):
        raise PortablePolicyError("merge_algorithm_unsupported", "$.merge")
    surrogate = document.get("surrogate")
    if not isinstance(surrogate, Mapping) or (
        surrogate.get("algorithm") != SURROGATE_ALGORITHM
        or not _is_int(surrogate.get("version"))
        or surrogate.get("version") != SURROGATE_VERSION
    ):
        raise PortablePolicyError("surrogate_algorithm_unsupported", "$.surrogate")
    mode = document.get("arbitration_mode")
    if mode not in ARBITRATION_MODES:
        raise PortablePolicyError("profile_invalid", "$.arbitration_mode")
    strict = document.get("strict_no_leak")
    if not isinstance(strict, bool):
        raise PortablePolicyError("profile_invalid", "$.strict_no_leak")
    actions = document.get("actions")
    if (
        not isinstance(actions, Mapping)
        or set(actions) != set(taxonomy.labels)
        or any(action not in PORTABLE_ACTIONS for action in actions.values())
    ):
        raise PortablePolicyError("profile_invalid", "$.actions")
    default_action = document.get("default_action")
    if default_action not in PORTABLE_ACTIONS:
        raise PortablePolicyError("profile_invalid", "$.default_action")
    thresholds = document.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise PortablePolicyError("profile_invalid", "$.thresholds")
    default_threshold = _decode_threshold(thresholds.get("default"))
    raw_label_thresholds = thresholds.get("labels")
    if not isinstance(raw_label_thresholds, Mapping) or any(
        label not in taxonomy.policy_labels for label in raw_label_thresholds
    ):
        raise PortablePolicyError("profile_invalid", "$.thresholds")
    label_thresholds = {
        label: _decode_threshold(entry) for label, entry in raw_label_thresholds.items()
    }
    return PortablePolicy(
        name=name,
        digest=digest,
        taxonomy=taxonomy,
        actions=dict(actions),
        default_action=str(default_action),
        arbitration_mode=str(mode),
        strict_no_leak=strict,
        threshold_profile=str(document.get("threshold_profile", "")),
        default_threshold=default_threshold,
        label_thresholds=label_thresholds,
        required_capabilities=tuple(required),
        excluded_features=tuple(
            str(item) for item in document.get("excluded_features") or ()
        ),
        forced_cascade_tiers=tuple(
            str(item) for item in document.get("forced_cascade_tiers") or ()
        ),
        document=document,
    )


def load_portable_policy(
    name: str,
    *,
    expected_digest: str | None = None,
    resource_files: Mapping[str, bytes] | None = None,
    supported_capabilities: Sequence[str] = SUPPORTED_CAPABILITIES,
) -> PortablePolicy:
    """Load and verify a portable profile.

    Verification order: manifest, profile name, caller-pinned digest, profile
    resource digest, taxonomy resource digest, then the profile document.

    Args:
        name: Profile name or alias.
        expected_digest: Optional ``sha256:`` digest the caller pins; a
            different bundled profile fails with ``digest_mismatch``.
        resource_files: Resource bytes by file name. Defaults to the Python
            reference resources built from the source profiles.
        supported_capabilities: Capabilities the runtime implements.

    Returns:
        The verified policy.

    Raises:
        PortablePolicyError: With a stable, value-free reason code.
    """
    files = bundled_resources() if resource_files is None else resource_files
    manifest_bytes = files.get(MANIFEST_FILE)
    if manifest_bytes is None:
        raise PortablePolicyError("resource_missing", "$.manifest")
    manifest = decode_manifest(manifest_bytes)
    canonical = resolve_profile_name(name, manifest)
    digest = manifest["profiles"][canonical]
    if expected_digest is not None and expected_digest != digest:
        raise PortablePolicyError("digest_mismatch", "$.digest")
    data = files.get(f"{canonical}.json")
    if data is None:
        raise PortablePolicyError("resource_missing", "$.profiles")
    if digest_bytes(data) != digest:
        raise PortablePolicyError("resource_digest_mismatch", "$.profiles")
    taxonomy_bytes = files.get(TAXONOMY_FILE)
    if taxonomy_bytes is None:
        raise PortablePolicyError("resource_missing", "$.taxonomy_digest")
    taxonomy_digest = digest_bytes(taxonomy_bytes)
    if taxonomy_digest != manifest["taxonomy_digest"]:
        raise PortablePolicyError("taxonomy_digest_mismatch", "$.taxonomy_digest")
    taxonomy = decode_taxonomy(taxonomy_bytes, taxonomy_digest)
    document = _json_object(data, "profile_invalid")
    return decode_profile(
        document,
        name=canonical,
        digest=digest,
        taxonomy=taxonomy,
        supported_capabilities=supported_capabilities,
    )


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


@dataclass
class _Candidate:
    inputs: tuple[int, ...]
    kind: str
    label: str
    canonical_label: str
    recognized: bool
    start: int
    end: int
    confidence_ppm: int
    keep_floor_ppm: int = 0
    escalate: bool = False
    action: str = "keep"
    strict_upgrade: bool = False
    outcome: str = "below_threshold"
    span: int | None = None


def decide(
    policy: PortablePolicy,
    text: str,
    detections: Sequence[PortableDetection],
    *,
    overrides: Mapping[str, str] | None = None,
    surrogate_key: bytes | None = None,
) -> PortableDecisionResult:
    """Apply a portable policy to normalized detections.

    Stages, in order: validation; label normalization; semantic-unit merging
    (each semantic unit absorbs the model spans it overlaps); threshold
    decisions in integer parts per million (retained when
    ``confidence_ppm >= keep_floor_ppm``); action lookup with caller
    overrides and the strict no-leak upgrade; exact-duplicate collapse;
    overlap clustering, where each cluster becomes one span over the union of
    its members; and rendering right to left at Unicode scalar offsets.

    Args:
        policy: Verified portable policy.
        text: Source text.
        detections: Normalized detections with scalar offsets.
        overrides: Optional canonical label to action overrides.
        surrogate_key: Optional caller-supplied key (16 to 64 bytes) for
            keyed ``replace`` surrogates. Without it ``replace`` renders the
            value-free ``[LABEL_REPLACED]`` placeholder.

    Returns:
        The kit-parity result and per-candidate decisions.

    Raises:
        PortablePolicyError: For invalid detections, overrides, or keys.
    """
    length = len(text)
    _validate_detections(detections, length)
    resolved_overrides = _validate_overrides(overrides, policy.taxonomy)
    if surrogate_key is not None and not (
        isinstance(surrogate_key, (bytes, bytearray))
        and SURROGATE_KEY_MIN_BYTES <= len(surrogate_key) <= SURROGATE_KEY_MAX_BYTES
    ):
        raise PortablePolicyError("surrogate_key_invalid", "$.surrogate_key")

    candidates = _build_candidates(policy.taxonomy, detections)
    for candidate in candidates:
        threshold = policy.threshold_for(candidate.canonical_label)
        candidate.keep_floor_ppm = threshold.keep_floor_ppm
        configured = resolved_overrides.get(
            candidate.canonical_label, policy.actions[candidate.canonical_label]
        )
        if policy.strict_no_leak and configured == "keep":
            candidate.action = "mask"
            candidate.strict_upgrade = True
        else:
            candidate.action = configured
        if candidate.confidence_ppm >= threshold.keep_floor_ppm:
            candidate.outcome = "retained"
            candidate.escalate = candidate.confidence_ppm < threshold.escalate_below_ppm

    retained = [item for item in candidates if item.outcome == "retained"]
    unique = _collapse_duplicates(policy, retained)
    clusters = _clusters(unique)
    spans: list[KitParitySpan] = []
    for cluster in clusters:
        representative = _representative(policy, cluster)
        start = min(item.start for item in cluster)
        end = max(item.end for item in cluster)
        index = len(spans)
        for item in cluster:
            item.outcome = "applied" if item is representative else "overlap_loser"
            item.span = index
        surface = text[start:end]
        spans.append(
            KitParitySpan(
                label=representative.label,
                canonical_label=representative.canonical_label,
                action=representative.action,
                start=start,
                end=end,
                confidence_ppm=representative.confidence_ppm,
                replacement=render_replacement(
                    representative.action,
                    representative.canonical_label,
                    surface,
                    surrogate_key,
                ),
            )
        )
    for item in retained:
        if item.outcome == "duplicate":
            winner = _duplicate_winner(item, unique)
            item.span = winner.span

    output = kit_parity.apply_spans(text, spans)
    result = KitParityResult(deidentified_text=output, spans=tuple(spans))
    kit_parity.validate_result(result)
    decisions = tuple(
        PortableDecision(
            inputs=item.inputs,
            kind=item.kind,
            label=item.label,
            canonical_label=item.canonical_label,
            recognized=item.recognized,
            start=item.start,
            end=item.end,
            confidence_ppm=item.confidence_ppm,
            keep_floor_ppm=item.keep_floor_ppm,
            escalate=item.escalate,
            action=item.action,
            strict_upgrade=item.strict_upgrade,
            outcome=item.outcome,
            span=item.span,
        )
        for item in candidates
    )
    return PortableDecisionResult(
        policy_name=policy.name,
        policy_digest=policy.digest,
        result=result,
        decisions=decisions,
    )


def render_replacement(
    action: str,
    canonical_label: str,
    surface: str,
    surrogate_key: bytes | None = None,
) -> str | None:
    """Return the replacement text for one output span.

    Args:
        action: Portable action.
        canonical_label: Canonical label of the span.
        surface: Source text covered by the span.
        surrogate_key: Optional caller key for keyed ``replace`` surrogates.

    Returns:
        ``None`` for ``keep``; otherwise the replacement text.
    """
    if action == "keep":
        return None
    if action in ("mask", "redact"):
        return f"[{canonical_label}]"
    if action == "remove":
        return ""
    if action == "replace":
        if surrogate_key is None:
            return f"[{canonical_label}_REPLACED]"
        return f"[{canonical_label}_{surrogate_token(surrogate_key, canonical_label, surface)}]"
    if action == "hash":
        digest = hashlib.sha256(surface.encode("utf-8")).digest()
        return f"{canonical_label}_{digest[:HASH_TOKEN_BYTES].hex()}"
    raise PortablePolicyError("action_unknown", "$.action")


def surrogate_token(key: bytes, canonical_label: str, surface: str) -> str:
    """Return the deterministic keyed surrogate token for a span.

    ``HMAC-SHA256(key, "openmed.portable_surrogate.v1" 0x00 label 0x00
    surface)`` truncated to 8 bytes, as lowercase hex. The same key, label,
    and surface always yield the same token; the key never leaves the caller.

    Args:
        key: Caller-supplied surrogate key.
        canonical_label: Canonical label of the span.
        surface: Source text covered by the span.

    Returns:
        Sixteen lowercase hex characters.
    """
    message = b"\x00".join(
        (SURROGATE_DOMAIN, canonical_label.encode("ascii"), surface.encode("utf-8"))
    )
    mac = hmac.new(bytes(key), message, hashlib.sha256).digest()
    return mac[:SURROGATE_TOKEN_BYTES].hex()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _cached_resources() -> dict[str, bytes]:
    global _RESOURCE_CACHE
    if _RESOURCE_CACHE is None:
        _RESOURCE_CACHE = build_resources()
    return dict(_RESOURCE_CACHE)


_RESOURCE_CACHE: dict[str, bytes] | None = None


def _core_resource_bytes(*parts: str) -> bytes:
    return resources.files("openmed.core").joinpath(*parts).read_bytes()


def _threshold_ppm(entry: Mapping[str, Any]) -> dict[str, int]:
    return {
        "escalate_below_ppm": confidence_to_ppm(float(entry["escalate_below"])),
        "keep_floor_ppm": confidence_to_ppm(float(entry["keep_floor"])),
    }


def _decode_threshold(entry: object) -> PortableThreshold:
    if not isinstance(entry, Mapping):
        raise PortablePolicyError("profile_invalid", "$.thresholds")
    keep = entry.get("keep_floor_ppm")
    escalate = entry.get("escalate_below_ppm")
    if not (_is_ppm(keep) and _is_ppm(escalate)):
        raise PortablePolicyError("profile_invalid", "$.thresholds")
    return PortableThreshold(keep_floor_ppm=int(keep), escalate_below_ppm=int(escalate))


def _json_object(data: bytes, code: str) -> dict[str, Any]:
    try:
        document = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise PortablePolicyError(code) from error
    if not isinstance(document, dict):
        raise PortablePolicyError(code)
    return document


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_ppm(value: object) -> bool:
    return _is_int(value) and 0 <= int(value) <= CONFIDENCE_SCALE  # type: ignore[call-overload]


def _validate_detections(detections: Sequence[PortableDetection], length: int) -> None:
    for index, detection in enumerate(detections):
        path = f"$.detections[{index}]"
        if not isinstance(detection.label, str) or not kit_parity.LABEL_PATTERN.match(
            detection.label
        ):
            raise PortablePolicyError("detection_label_invalid", f"{path}.label")
        if not _is_int(detection.start) or detection.start < 0:
            raise PortablePolicyError("detection_offset_invalid", f"{path}.start")
        if not _is_int(detection.end) or detection.end <= detection.start:
            raise PortablePolicyError("detection_offset_invalid", f"{path}.end")
        if detection.end > length:
            raise PortablePolicyError("detection_offset_out_of_bounds", f"{path}.end")
        if not _is_ppm(detection.confidence_ppm):
            raise PortablePolicyError(
                "detection_confidence_invalid", f"{path}.confidence_ppm"
            )
        if detection.source not in DETECTION_SOURCES:
            raise PortablePolicyError("detection_source_invalid", f"{path}.source")
        if not isinstance(detection.validated, bool):
            raise PortablePolicyError("detection_source_invalid", f"{path}.validated")


def _validate_overrides(
    overrides: Mapping[str, str] | None, taxonomy: PortableTaxonomy
) -> dict[str, str]:
    if not overrides:
        return {}
    resolved: dict[str, str] = {}
    for label in sorted(overrides):
        if label not in taxonomy.policy_labels:
            raise PortablePolicyError("override_label_unknown", "$.overrides")
        action = overrides[label]
        if action not in PORTABLE_ACTIONS:
            raise PortablePolicyError("override_action_unknown", "$.overrides")
        resolved[label] = action
    return resolved


def _order_key(item: _Candidate) -> tuple[int, int, str, str, int]:
    return (item.start, item.end, item.canonical_label, item.label, item.inputs[0])


def _build_candidates(
    taxonomy: PortableTaxonomy, detections: Sequence[PortableDetection]
) -> list[_Candidate]:
    normalized = [
        (index, detection, *taxonomy.normalize(detection.label))
        for index, detection in enumerate(detections)
    ]

    def detection_key(
        entry: tuple[int, PortableDetection, str, bool],
    ) -> tuple[int, int, str, str, int]:
        index, detection, canonical, _ = entry
        return (detection.start, detection.end, canonical, detection.label, index)

    models = sorted(
        (entry for entry in normalized if entry[1].source == "model"),
        key=detection_key,
    )
    units = sorted(
        (entry for entry in normalized if entry[1].source == "semantic"),
        key=detection_key,
    )
    consumed: set[int] = set()
    candidates: list[_Candidate] = []
    for unit_index, unit, unit_canonical, unit_recognized in units:
        overlapping = [
            entry
            for entry in models
            if entry[1].start < unit.end and entry[1].end > unit.start
        ]
        if not overlapping:
            candidates.append(
                _Candidate(
                    inputs=(unit_index,),
                    kind="semantic_only",
                    label=unit.label,
                    canonical_label=unit_canonical,
                    recognized=unit_recognized,
                    start=unit.start,
                    end=unit.end,
                    confidence_ppm=unit.confidence_ppm,
                )
            )
            continue
        dominant = _dominant_label(overlapping)
        model_labels = {entry[2] for entry in overlapping}
        if unit_canonical in model_labels or (
            taxonomy.specificity[dominant] > taxonomy.specificity[unit_canonical]
        ):
            final = dominant
        else:
            final = unit_canonical
        carriers = [entry for entry in overlapping if entry[2] == final]
        if carriers:
            carrier = min(carriers, key=lambda entry: -entry[1].confidence_ppm)
            label, recognized = carrier[1].label, carrier[3]
        else:
            label, recognized = unit.label, unit_recognized
        average = sum(entry[1].confidence_ppm for entry in overlapping) // len(
            overlapping
        )
        model_weight, unit_weight = (6, 4) if unit.validated else (9, 1)
        confidence = (
            model_weight * average + unit_weight * unit.confidence_ppm + 5
        ) // 10
        consumed.update(entry[0] for entry in overlapping)
        candidates.append(
            _Candidate(
                inputs=tuple(
                    sorted([unit_index, *(entry[0] for entry in overlapping)])
                ),
                kind="merged",
                label=label,
                canonical_label=final,
                recognized=recognized,
                start=min(unit.start, *(entry[1].start for entry in overlapping)),
                end=max(unit.end, *(entry[1].end for entry in overlapping)),
                confidence_ppm=confidence,
            )
        )
    for index, detection, canonical, recognized in models:
        if index in consumed:
            continue
        candidates.append(
            _Candidate(
                inputs=(index,),
                kind="model",
                label=detection.label,
                canonical_label=canonical,
                recognized=recognized,
                start=detection.start,
                end=detection.end,
                confidence_ppm=detection.confidence_ppm,
            )
        )
    candidates.sort(key=_order_key)
    return candidates


def _dominant_label(entries: Sequence[tuple[int, PortableDetection, str, bool]]) -> str:
    counts: dict[str, int] = {}
    sums: dict[str, int] = {}
    for _, detection, canonical, _ in entries:
        counts[canonical] = counts.get(canonical, 0) + 1
        sums[canonical] = sums.get(canonical, 0) + detection.confidence_ppm
    best = ""
    for label in sorted(counts):
        if not best:
            best = label
            continue
        if counts[label] != counts[best]:
            if counts[label] > counts[best]:
                best = label
            continue
        # Equal counts: the higher average confidence wins (compared exactly
        # by cross-multiplication); remaining ties keep the smaller label.
        if sums[label] * counts[best] > sums[best] * counts[label]:
            best = label
    return best


def _winner_key(
    policy: PortablePolicy, item: _Candidate
) -> tuple[int, int, int, int, int, str, str, int]:
    return (
        0 if item.action != "keep" else 1,
        -policy.taxonomy.specificity[item.canonical_label],
        -(item.end - item.start),
        -item.confidence_ppm,
        item.start,
        item.canonical_label,
        item.label,
        item.inputs[0],
    )


def _collapse_duplicates(
    policy: PortablePolicy, retained: Sequence[_Candidate]
) -> list[_Candidate]:
    groups: dict[tuple[int, int, str], list[_Candidate]] = {}
    for item in retained:
        groups.setdefault((item.start, item.end, item.canonical_label), []).append(item)
    unique: list[_Candidate] = []
    for group in groups.values():
        winner = min(group, key=lambda item: _winner_key(policy, item))
        for item in group:
            if item is not winner:
                item.outcome = "duplicate"
        unique.append(winner)
    unique.sort(key=_order_key)
    return unique


def _duplicate_winner(item: _Candidate, unique: Sequence[_Candidate]) -> _Candidate:
    for candidate in unique:
        if (candidate.start, candidate.end, candidate.canonical_label) == (
            item.start,
            item.end,
            item.canonical_label,
        ):
            return candidate
    raise PortablePolicyError("internal_error")  # pragma: no cover - invariant


def _clusters(items: Sequence[_Candidate]) -> list[list[_Candidate]]:
    clusters: list[list[_Candidate]] = []
    current: list[_Candidate] = []
    current_end = -1
    for item in items:
        if current and item.start < current_end:
            current.append(item)
            current_end = max(current_end, item.end)
            continue
        if current:
            clusters.append(current)
        current = [item]
        current_end = item.end
    if current:
        clusters.append(current)
    return clusters


def _representative(
    policy: PortablePolicy, cluster: Sequence[_Candidate]
) -> _Candidate:
    members = list(cluster)
    if policy.arbitration_mode == "high_recall_union":
        strongest = max(ACTION_STRENGTH[item.action] for item in members)
        members = [
            item for item in members if ACTION_STRENGTH[item.action] == strongest
        ]
    return min(members, key=lambda item: _winner_key(policy, item))


__all__ = [
    "ACTION_STRENGTH",
    "DEFAULT_PROFILE",
    "EXCLUDED_PYTHON_ONLY",
    "EXCLUDED_STRUCTURED_SCHEMA",
    "IR_SCHEMA",
    "IR_VERSION",
    "MANIFEST_FILE",
    "MANIFEST_SCHEMA",
    "MANIFEST_VERSION",
    "MERGE_ALGORITHM",
    "MERGE_VERSION",
    "PORTABLE_ACTIONS",
    "PORTABLE_PROFILES",
    "SUPPORTED_CAPABILITIES",
    "SURROGATE_ALGORITHM",
    "SURROGATE_VERSION",
    "TAXONOMY_FILE",
    "TAXONOMY_SCHEMA",
    "TAXONOMY_VERSION",
    "PortableDecision",
    "PortableDecisionResult",
    "PortableDetection",
    "PortablePolicy",
    "PortablePolicyError",
    "PortableTaxonomy",
    "PortableThreshold",
    "build_profile_document",
    "build_resources",
    "build_taxonomy_document",
    "bundled_resources",
    "canonical_bytes",
    "confidence_to_ppm",
    "decide",
    "decode_manifest",
    "decode_profile",
    "decode_taxonomy",
    "digest_bytes",
    "excluded_profiles",
    "label_key",
    "load_portable_policy",
    "render_replacement",
    "resolve_profile_name",
    "surrogate_token",
]
