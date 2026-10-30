"""Atomic activation of verified offline language bundles.

An offline language bundle is installed by a separate packaging step. This
module owns only the last step: it takes the bundle manifest, readiness
evidence, and handles to model artifacts that are already on local disk, and
turns them into one generation of the process-local
:class:`~openmed.core.language_pack.LanguagePackRegistry`.

Activation is all-or-nothing. A manifest with an invalid signature, readiness
evidence that is stale or for a different language tag, an unknown critical
field, an incoherent pack, or a pack identity that collides with an existing
pack is refused before anything is committed, so the previous registry state is
unchanged. A successful activation can be rolled back explicitly.

The module performs no discovery, download, or network access. Readiness is
matched by exact language tag, never by prefix, and a pack whose surrogate
locale is an approximation never yields a ``supported`` clinical-language
claim. Errors carry controlled reason codes only; they never echo manifest
values, signatures, model paths, or evidence content.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import PurePosixPath, PureWindowsPath
from threading import RLock
from typing import Any, Protocol

from .language_pack import LANGUAGE_PACK_REGISTRY, LanguagePack, LanguagePackRegistry
from .language_pack_coherence import APPROXIMATED, pack_coherence_report
from .locale_tag import LocaleTagError, normalize_locale_tag

BUNDLE_SCHEMA_VERSION = "1.0"
"""Manifest schema version understood by this activator."""

ACTIVATION_SCHEMA_VERSION = "1.0"
"""Schema version of serialized :class:`LanguageBundleActivation` records."""

DEFAULT_MAX_READINESS_AGE = timedelta(days=30)
"""Default maximum age of readiness evidence at activation time."""

DEFAULT_CLOCK_SKEW = timedelta(minutes=5)
"""Default tolerance for readiness evidence issued slightly in the future."""

READINESS_STATES = ("experimental", "preview", "supported")
"""Closed readiness vocabulary, ordered from weakest to strongest."""

ACTIVATABLE_READINESS_STATES = frozenset({"preview", "supported"})
"""Readiness states that may activate a bundle; experimental evidence cannot."""

ARTIFACT_KINDS = frozenset({"evidence", "lexicon", "model", "policy", "self_test"})
"""Artifact kinds a bundle manifest may declare."""

UNDERSTOOD_EXTENSIONS: frozenset[str] = frozenset()
"""Manifest extensions this activator understands when marked critical."""

CLAIM_PREVIEW = "preview"
CLAIM_SUPPORTED = "supported"
LOCALE_NATIVE = "native"
LOCALE_APPROXIMATED = "approximated"

_MANIFEST_FIELDS = frozenset(
    {
        "artifacts",
        "bundle_id",
        "bundle_version",
        "critical",
        "extensions",
        "packs",
        "schema_version",
        "signature",
    }
)
_REQUIRED_MANIFEST_FIELDS = frozenset(
    {"artifacts", "bundle_id", "bundle_version", "packs", "schema_version"}
)
_PACK_DECLARATION_FIELDS = frozenset(
    {
        "candidate_priority",
        "context_scripts",
        "national_id_providers",
        "policy_overrides",
        "recall_floor_overrides",
        "recognizers",
        "routing_markers",
        "scripts",
        "segmenter_id",
        "surrogate_locale",
        "surrogate_locale_approximation",
    }
)
_PACK_FIELDS = _PACK_DECLARATION_FIELDS | {"language", "model", "tasks"}
_REQUIRED_PACK_FIELDS = frozenset(
    {
        "language",
        "model",
        "recognizers",
        "scripts",
        "segmenter_id",
        "surrogate_locale",
        "tasks",
    }
)
_ARTIFACT_FIELDS = frozenset({"kind", "sha256"})
_SIGNATURE_FIELDS = frozenset({"key_id", "scheme", "value"})
_READINESS_FIELDS = frozenset(
    {
        "evidence_digest",
        "issued_at",
        "language",
        "model_digest",
        "qualified_tasks",
        "state",
    }
)
_LANGUAGE_FIELDS = frozenset(
    {
        "claim",
        "code",
        "language",
        "locale_status",
        "model_digest",
        "readiness_digest",
        "readiness_state",
        "tasks",
        "withheld_tasks",
    }
)
_ACTIVATION_FIELDS = frozenset(
    {
        "activated_at",
        "activation_id",
        "bundle_id",
        "bundle_version",
        "languages",
        "manifest_digest",
        "previous_activation_id",
        "schema_version",
    }
)

_BUNDLE_ID = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_ALIAS = re.compile(r"[a-z][a-z0-9_.-]{0,127}")
_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_EXTENSION = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_SIGNATURE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_SIGNATURE_VALUE = re.compile(r"[A-Za-z0-9+/=_-]{1,8192}")
_VERSION = re.compile(r"(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})\.(0|[1-9][0-9]{0,8})")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PRIMARY_CODE = re.compile(r"[a-z]{2}")
_REMOTE_REFERENCE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:(?://|[^\\/])")
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_PACKS = 64
_MAX_TASKS = 64
_MAX_ARTIFACTS = 256
_MAX_READINESS = 256


class ActivationRefusal(str, Enum):
    """Controlled, value-free reasons an activation or rollback was refused."""

    MANIFEST_MALFORMED = "manifest_malformed"
    MANIFEST_SCHEMA_UNSUPPORTED = "manifest_schema_unsupported"
    UNKNOWN_CRITICAL_FIELD = "unknown_critical_field"
    SIGNATURE_MISSING = "signature_missing"
    SIGNATURE_INVALID = "signature_invalid"
    ARTIFACT_HANDLE_MISSING = "artifact_handle_missing"
    ARTIFACT_HANDLE_NOT_LOCAL = "artifact_handle_not_local"
    ARTIFACT_DIGEST_MISMATCH = "artifact_digest_mismatch"
    READINESS_MALFORMED = "readiness_malformed"
    READINESS_MISSING = "readiness_missing"
    READINESS_STALE = "readiness_stale"
    READINESS_NOT_YET_VALID = "readiness_not_yet_valid"
    READINESS_INSUFFICIENT = "readiness_insufficient"
    READINESS_SUBJECT_MISMATCH = "readiness_subject_mismatch"
    TASKS_NOT_QUALIFIED = "tasks_not_qualified"
    LOCALE_APPROXIMATION_UNDECLARED = "locale_approximation_undeclared"
    PACK_INCOHERENT = "pack_incoherent"
    PACK_IDENTITY_CONFLICT = "pack_identity_conflict"
    CAPABILITY_CONFLICT = "capability_conflict"
    BUNDLE_VERSION_NOT_NEWER = "bundle_version_not_newer"
    ACTIVATION_ID_MISMATCH = "activation_id_mismatch"
    ACTIVATION_RECORD_MALFORMED = "activation_record_malformed"
    REGISTRY_CHANGED = "registry_changed"
    REGISTRY_REFRESH_FAILED = "registry_refresh_failed"
    ROLLBACK_NOT_ACTIVE = "rollback_not_active"
    ROLLBACK_CONFLICT = "rollback_conflict"


class LanguageBundleActivationError(ValueError):
    """Raised when activation or rollback is refused.

    The registry is unchanged whenever this error is raised. The message is the
    controlled reason code only.

    Attributes:
        reason: The :class:`ActivationRefusal` that stopped the operation.
    """

    def __init__(self, reason: ActivationRefusal) -> None:
        """Store the refusal reason and use its code as the message."""

        self.reason = reason
        super().__init__(reason.value)


def _refuse(reason: ActivationRefusal) -> LanguageBundleActivationError:
    return LanguageBundleActivationError(reason)


class BundleSignatureVerifier(Protocol):
    """Caller-owned trust anchor that checks a bundle manifest signature.

    OpenMed does not ship or fetch signing keys. Applications inject a verifier
    bound to the keys they trust, for example an Ed25519 public key from their
    own release process.
    """

    def verify(
        self,
        payload: bytes,
        *,
        scheme: str,
        key_id: str,
        signature: str,
    ) -> bool:
        """Return ``True`` only when ``signature`` is valid for ``payload``.

        Args:
            payload: Canonical manifest bytes from
                :func:`language_bundle_signing_payload`.
            scheme: Signature scheme declared by the manifest.
            key_id: Key identifier declared by the manifest.
            signature: Encoded signature value declared by the manifest.

        Returns:
            ``True`` for a valid signature from a trusted key, else ``False``.
        """
        ...


@dataclass(frozen=True, slots=True)
class LocalModelHandle:
    """A model artifact that the installer already placed on local disk.

    Args:
        reference: Absolute local path that becomes the pack's
            ``default_model``. It is never copied into activation records.
        sha256: ``sha256:<hex>`` digest the installer measured for the
            artifact; it must equal the digest the signed manifest declares.
    """

    reference: str = field(repr=False)
    sha256: str

    def __post_init__(self) -> None:
        """Require string values; locality and digests are checked later."""

        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError("reference must be a non-empty string")
        if not isinstance(self.sha256, str):
            raise ValueError("sha256 must be a string")


def _is_local_reference(reference: str) -> bool:
    if _REMOTE_REFERENCE.match(reference) and not PureWindowsPath(reference).drive:
        return False
    return (
        PurePosixPath(reference).is_absolute()
        or PureWindowsPath(reference).is_absolute()
    )


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: object, reason: ActivationRefusal) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and 0 < len(value) <= 64:
        text = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            raise _refuse(reason) from None
    else:
        raise _refuse(reason)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _refuse(reason)
    return parsed.astimezone(timezone.utc)


def _tokens(
    values: object,
    pattern: re.Pattern[str],
    reason: ActivationRefusal,
    *,
    limit: int = _MAX_TASKS,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise _refuse(reason)
    items = []
    for item in values:
        if not isinstance(item, str) or pattern.fullmatch(item) is None:
            raise _refuse(reason)
        items.append(item)
        if len(items) > limit:
            raise _refuse(reason)
    if len(set(items)) != len(items) or (not items and not allow_empty):
        raise _refuse(reason)
    return tuple(sorted(items))


def _choice(
    value: object,
    allowed: Iterable[str],
    reason: ActivationRefusal,
) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _refuse(reason)
    return value


def _digest(value: object, reason: ActivationRefusal) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise _refuse(reason)
    return value


def _canonical_bytes(value: object, reason: ActivationRefusal) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return text.encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise _refuse(reason) from None


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def language_bundle_signing_payload(manifest: Mapping[str, Any]) -> bytes:
    """Return the canonical bytes a bundle signature must cover.

    The payload is the UTF-8 encoding of the manifest without its top-level
    ``signature`` member, serialized as JSON with sorted keys, compact
    separators, and no non-finite numbers. Packaging tools sign exactly these
    bytes; :class:`LanguageBundleActivator` verifies exactly these bytes.

    Args:
        manifest: Plain bundle manifest mapping, for example from ``json.load``.

    Returns:
        Canonical signing payload.

    Raises:
        LanguageBundleActivationError: If the manifest is not a JSON-compatible
            mapping or exceeds the 1 MiB manifest limit.
    """

    if not isinstance(manifest, Mapping) or not all(
        isinstance(key, str) for key in manifest
    ):
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED)
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    payload = _canonical_bytes(unsigned, ActivationRefusal.MANIFEST_MALFORMED)
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED)
    return payload


def language_bundle_digest(manifest: Mapping[str, Any]) -> str:
    """Return the ``sha256:<hex>`` digest of a manifest's signing payload.

    Args:
        manifest: Plain bundle manifest mapping.

    Returns:
        Digest that identifies the signed manifest content.
    """

    return _sha256(language_bundle_signing_payload(manifest))


@dataclass(frozen=True, slots=True)
class LanguageReadinessEvidence:
    """Readiness status for one exact language tag, as plain data.

    Args:
        language: BCP 47 tag the evidence covers. It must equal a pack's
            ``language`` exactly; ``pt`` evidence never qualifies ``pt-BR``
            and ``pt-BR`` evidence never qualifies ``pt``.
        state: ``experimental``, ``preview``, or ``supported``.
        qualified_tasks: Task codes whose evidence is complete.
        issued_at: Timezone-aware time the evidence was produced. Activation
            refuses evidence older than the activator's maximum age.
        evidence_digest: ``sha256:<hex>`` digest of the exported evidence.
        model_digest: Optional ``sha256:<hex>`` digest of the model artifact
            the evidence was measured on. When present it must equal the
            pack's model artifact digest.
    """

    language: str
    state: str
    qualified_tasks: tuple[str, ...]
    issued_at: datetime
    evidence_digest: str
    model_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the evidence fields."""

        malformed = ActivationRefusal.READINESS_MALFORMED
        try:
            language = normalize_locale_tag(self.language)
        except (LocaleTagError, TypeError):
            raise _refuse(malformed) from None
        _choice(self.state, READINESS_STATES, malformed)
        object.__setattr__(self, "language", language)
        object.__setattr__(
            self,
            "qualified_tasks",
            _tokens(self.qualified_tasks, _TOKEN, malformed),
        )
        object.__setattr__(self, "issued_at", _parse_time(self.issued_at, malformed))
        _digest(self.evidence_digest, malformed)
        if self.model_digest is not None:
            _digest(self.model_digest, malformed)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping that :meth:`from_dict` accepts."""

        return {
            "evidence_digest": self.evidence_digest,
            "issued_at": _format_time(self.issued_at),
            "language": self.language,
            "model_digest": self.model_digest,
            "qualified_tasks": list(self.qualified_tasks),
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LanguageReadinessEvidence:
        """Build evidence from plain data, rejecting unknown fields.

        Args:
            data: Mapping produced by :meth:`to_dict` or an equivalent exporter.

        Returns:
            Validated readiness evidence.

        Raises:
            LanguageBundleActivationError: If a field is unknown, missing, or
                invalid.
        """

        if not isinstance(data, Mapping) or not set(data) <= _READINESS_FIELDS:
            raise _refuse(ActivationRefusal.READINESS_MALFORMED)
        required = _READINESS_FIELDS - {"model_digest"}
        if not required <= set(data):
            raise _refuse(ActivationRefusal.READINESS_MALFORMED)
        return cls(
            language=data["language"],
            state=data["state"],
            qualified_tasks=data["qualified_tasks"],
            issued_at=data["issued_at"],
            evidence_digest=data["evidence_digest"],
            model_digest=data.get("model_digest"),
        )


@dataclass(frozen=True, slots=True)
class ActivatedLanguage:
    """The capabilities one activated pack registered, without private values.

    Attributes:
        language: Exact canonical BCP 47 tag the readiness evidence covers.
        code: Language-pack registry code the pack was registered under.
        tasks: Declared tasks that the readiness evidence qualifies.
        withheld_tasks: Declared tasks that were not qualified and therefore
            not registered.
        claim: ``supported`` or ``preview``. Approximated locales are always
            ``preview``.
        locale_status: ``native`` or ``approximated``.
        readiness_state: Readiness state reported by the evidence.
        readiness_digest: Digest of the evidence used for activation.
        model_digest: Digest of the bound model artifact.
    """

    language: str
    code: str
    tasks: tuple[str, ...]
    withheld_tasks: tuple[str, ...]
    claim: str
    locale_status: str
    readiness_state: str
    readiness_digest: str
    model_digest: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible mapping of the activated capabilities."""

        return {
            "claim": self.claim,
            "code": self.code,
            "language": self.language,
            "locale_status": self.locale_status,
            "model_digest": self.model_digest,
            "readiness_digest": self.readiness_digest,
            "readiness_state": self.readiness_state,
            "tasks": list(self.tasks),
            "withheld_tasks": list(self.withheld_tasks),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActivatedLanguage:
        """Rebuild an activated-language entry from :meth:`to_dict` output.

        Raises:
            LanguageBundleActivationError: If the entry is malformed.
        """

        malformed = ActivationRefusal.ACTIVATION_RECORD_MALFORMED
        if not isinstance(data, Mapping) or set(data) != _LANGUAGE_FIELDS:
            raise _refuse(malformed)
        try:
            language = normalize_locale_tag(data["language"])
        except (LocaleTagError, TypeError):
            raise _refuse(malformed) from None
        code = data["code"]
        if language != data["language"] or code != language.split("-")[0]:
            raise _refuse(malformed)
        claim = _choice(data["claim"], (CLAIM_PREVIEW, CLAIM_SUPPORTED), malformed)
        locale_status = _choice(
            data["locale_status"], (LOCALE_NATIVE, LOCALE_APPROXIMATED), malformed
        )
        readiness_state = _choice(
            data["readiness_state"], ACTIVATABLE_READINESS_STATES, malformed
        )
        # A serialized record can never upgrade an approximated locale or
        # preview evidence into a supported claim.
        if claim == CLAIM_SUPPORTED and (
            locale_status != LOCALE_NATIVE or readiness_state != CLAIM_SUPPORTED
        ):
            raise _refuse(malformed)
        return cls(
            language=language,
            code=code,
            tasks=_tokens(data["tasks"], _TOKEN, malformed, allow_empty=False),
            withheld_tasks=_tokens(data["withheld_tasks"], _TOKEN, malformed),
            claim=claim,
            locale_status=locale_status,
            readiness_state=readiness_state,
            readiness_digest=_digest(data["readiness_digest"], malformed),
            model_digest=_digest(data["model_digest"], malformed),
        )


def _activation_id(
    bundle_id: str,
    bundle_version: str,
    manifest_digest: str,
    languages: tuple[ActivatedLanguage, ...],
) -> str:
    body = {
        "bundle_id": bundle_id,
        "bundle_version": bundle_version,
        "languages": [language.to_dict() for language in languages],
        "manifest_digest": manifest_digest,
    }
    return _sha256(_canonical_bytes(body, ActivationRefusal.MANIFEST_MALFORMED))


@dataclass(frozen=True, slots=True)
class LanguageBundleActivation:
    """Serializable record of one committed bundle activation.

    The ``activation_id`` is a content digest over the bundle identity, the
    manifest digest, and the activated capabilities. Re-activating the same
    signed manifest with the same evidence in a fresh process yields the same
    identifier, which :meth:`LanguageBundleActivator.activate` can enforce via
    ``expected_activation_id``. Records hold only codes, versions, timestamps,
    and digests; they never include model paths or evidence content.

    Attributes:
        activation_id: ``sha256:<hex>`` content identifier.
        bundle_id: Signed bundle identifier.
        bundle_version: Signed ``MAJOR.MINOR.PATCH`` bundle version.
        manifest_digest: Digest of the signed manifest payload.
        activated_at: UTC time the activation was committed.
        languages: Activated packs ordered by language tag.
        previous_activation_id: Activation of the same bundle this one
            superseded, if any.
    """

    activation_id: str
    bundle_id: str
    bundle_version: str
    manifest_digest: str
    activated_at: datetime
    languages: tuple[ActivatedLanguage, ...]
    previous_activation_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible mapping of the record."""

        return {
            "activated_at": _format_time(self.activated_at),
            "activation_id": self.activation_id,
            "bundle_id": self.bundle_id,
            "bundle_version": self.bundle_version,
            "languages": [language.to_dict() for language in self.languages],
            "manifest_digest": self.manifest_digest,
            "previous_activation_id": self.previous_activation_id,
            "schema_version": ACTIVATION_SCHEMA_VERSION,
        }

    def to_json(self) -> str:
        """Serialize the record as canonical JSON with a trailing newline."""

        payload = _canonical_bytes(
            self.to_dict(), ActivationRefusal.ACTIVATION_RECORD_MALFORMED
        )
        return payload.decode("utf-8") + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LanguageBundleActivation:
        """Rebuild a record and verify that its identifier matches its content.

        Args:
            data: Mapping produced by :meth:`to_dict`.

        Returns:
            The validated activation record.

        Raises:
            LanguageBundleActivationError: If the record is malformed or its
                ``activation_id`` does not match the serialized content.
        """

        malformed = ActivationRefusal.ACTIVATION_RECORD_MALFORMED
        if not isinstance(data, Mapping) or set(data) != _ACTIVATION_FIELDS:
            raise _refuse(malformed)
        _choice(data["schema_version"], (ACTIVATION_SCHEMA_VERSION,), malformed)
        bundle_id = data["bundle_id"]
        bundle_version = data["bundle_version"]
        if not isinstance(bundle_id, str) or not _BUNDLE_ID.fullmatch(bundle_id):
            raise _refuse(malformed)
        if not isinstance(bundle_version, str) or not _VERSION.fullmatch(
            bundle_version
        ):
            raise _refuse(malformed)
        entries = data["languages"]
        if isinstance(entries, (str, bytes, Mapping)) or not isinstance(
            entries, Iterable
        ):
            raise _refuse(malformed)
        languages = tuple(ActivatedLanguage.from_dict(entry) for entry in entries)
        tags = [language.language for language in languages]
        if not languages or tags != sorted(set(tags)):
            raise _refuse(malformed)
        previous = data["previous_activation_id"]
        if previous is not None:
            _digest(previous, malformed)
        record = cls(
            activation_id=_digest(data["activation_id"], malformed),
            bundle_id=bundle_id,
            bundle_version=bundle_version,
            manifest_digest=_digest(data["manifest_digest"], malformed),
            activated_at=_parse_time(data["activated_at"], malformed),
            languages=languages,
            previous_activation_id=previous,
        )
        expected = _activation_id(
            record.bundle_id,
            record.bundle_version,
            record.manifest_digest,
            record.languages,
        )
        if record.activation_id != expected:
            raise _refuse(ActivationRefusal.ACTIVATION_ID_MISMATCH)
        return record


@dataclass(frozen=True, slots=True)
class _PackSpec:
    language: str
    code: str
    model: str
    tasks: tuple[str, ...]
    declaration: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _Bundle:
    bundle_id: str
    version: str
    digest: str
    packs: tuple[_PackSpec, ...]
    artifacts: Mapping[str, tuple[str, str]]


@dataclass(frozen=True, slots=True)
class _ActiveEntry:
    record: LanguageBundleActivation
    packs: Mapping[str, LanguagePack]
    previous: _ActiveEntry | None


def _require_fields(
    data: Mapping[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
) -> None:
    # Every manifest member is critical in schema 1.0: a field this activator
    # does not know could change meaning, so it is never silently ignored.
    if not all(isinstance(key, str) for key in data):
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED)
    if not set(data) <= allowed:
        raise _refuse(ActivationRefusal.UNKNOWN_CRITICAL_FIELD)
    if not required <= set(data):
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED)


def _parse_signature(manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    signature = manifest.get("signature")
    if signature is None:
        raise _refuse(ActivationRefusal.SIGNATURE_MISSING)
    if not isinstance(signature, Mapping):
        raise _refuse(ActivationRefusal.SIGNATURE_INVALID)
    if not all(isinstance(key, str) for key in signature):
        raise _refuse(ActivationRefusal.SIGNATURE_INVALID)
    if not set(signature) <= _SIGNATURE_FIELDS:
        raise _refuse(ActivationRefusal.UNKNOWN_CRITICAL_FIELD)
    scheme = signature.get("scheme")
    key_id = signature.get("key_id")
    value = signature.get("value")
    if not (
        isinstance(scheme, str)
        and _SIGNATURE_TOKEN.fullmatch(scheme)
        and isinstance(key_id, str)
        and _SIGNATURE_TOKEN.fullmatch(key_id)
        and isinstance(value, str)
        and _SIGNATURE_VALUE.fullmatch(value)
    ):
        raise _refuse(ActivationRefusal.SIGNATURE_INVALID)
    return scheme, key_id, value


def _parse_artifacts(value: object) -> dict[str, tuple[str, str]]:
    malformed = ActivationRefusal.MANIFEST_MALFORMED
    if not isinstance(value, Mapping) or not 0 < len(value) <= _MAX_ARTIFACTS:
        raise _refuse(malformed)
    artifacts: dict[str, tuple[str, str]] = {}
    for alias, entry in value.items():
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise _refuse(malformed)
        if not isinstance(entry, Mapping):
            raise _refuse(malformed)
        _require_fields(entry, _ARTIFACT_FIELDS, _ARTIFACT_FIELDS)
        kind = entry["kind"]
        if not isinstance(kind, str) or kind not in ARTIFACT_KINDS:
            raise _refuse(malformed)
        artifacts[alias] = (kind, _digest(entry["sha256"], malformed))
    return artifacts


def _parse_pack(
    value: object,
    artifacts: Mapping[str, tuple[str, str]],
) -> _PackSpec:
    malformed = ActivationRefusal.MANIFEST_MALFORMED
    if not isinstance(value, Mapping):
        raise _refuse(malformed)
    _require_fields(value, _PACK_FIELDS, _REQUIRED_PACK_FIELDS)
    tag = value["language"]
    try:
        language = normalize_locale_tag(tag)
    except (LocaleTagError, TypeError):
        raise _refuse(malformed) from None
    code = language.split("-")[0]
    if language != tag or _PRIMARY_CODE.fullmatch(code) is None:
        raise _refuse(malformed)
    model = value["model"]
    if not isinstance(model, str) or artifacts.get(model, ("", ""))[0] != "model":
        raise _refuse(malformed)
    tasks = _tokens(value["tasks"], _TOKEN, malformed, allow_empty=False)
    declaration = {
        key: item for key, item in value.items() if key in _PACK_DECLARATION_FIELDS
    }
    return _PackSpec(language, code, model, tasks, declaration)


def _signed_payload(manifest: Mapping[str, Any]) -> bytes:
    if not isinstance(manifest, Mapping):
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise _refuse(ActivationRefusal.MANIFEST_SCHEMA_UNSUPPORTED)
    return language_bundle_signing_payload(manifest)


def _parse_manifest(manifest: Mapping[str, Any], payload: bytes) -> _Bundle:
    # Called only after the signature over ``payload`` has been verified.
    malformed = ActivationRefusal.MANIFEST_MALFORMED
    _require_fields(manifest, _MANIFEST_FIELDS, _REQUIRED_MANIFEST_FIELDS)
    bundle_id = manifest["bundle_id"]
    version = manifest["bundle_version"]
    if not isinstance(bundle_id, str) or _BUNDLE_ID.fullmatch(bundle_id) is None:
        raise _refuse(malformed)
    if not isinstance(version, str) or _VERSION.fullmatch(version) is None:
        raise _refuse(malformed)

    extensions = manifest.get("extensions", {})
    if not isinstance(extensions, Mapping) or not all(
        isinstance(key, str) and _EXTENSION.fullmatch(key) for key in extensions
    ):
        raise _refuse(malformed)
    critical = _tokens(manifest.get("critical", ()), _EXTENSION, malformed)
    if not set(critical) <= set(extensions):
        raise _refuse(malformed)
    if not set(critical) <= UNDERSTOOD_EXTENSIONS:
        raise _refuse(ActivationRefusal.UNKNOWN_CRITICAL_FIELD)

    artifacts = _parse_artifacts(manifest["artifacts"])
    raw_packs = manifest["packs"]
    if isinstance(raw_packs, (str, bytes, Mapping)) or not isinstance(
        raw_packs, Iterable
    ):
        raise _refuse(malformed)
    packs: list[_PackSpec] = []
    for item in raw_packs:
        packs.append(_parse_pack(item, artifacts))
        if len(packs) > _MAX_PACKS:
            raise _refuse(malformed)
    if not packs:
        raise _refuse(malformed)
    if len({spec.code for spec in packs}) != len(packs):
        raise _refuse(ActivationRefusal.PACK_IDENTITY_CONFLICT)
    return _Bundle(
        bundle_id=bundle_id,
        version=version,
        digest=_sha256(payload),
        packs=tuple(sorted(packs, key=lambda spec: spec.language)),
        artifacts=artifacts,
    )


def _index_readiness(
    readiness: Iterable[LanguageReadinessEvidence | Mapping[str, Any]],
) -> dict[str, LanguageReadinessEvidence]:
    malformed = ActivationRefusal.READINESS_MALFORMED
    if isinstance(readiness, (str, bytes, Mapping)) or not isinstance(
        readiness, Iterable
    ):
        raise _refuse(malformed)
    indexed: dict[str, LanguageReadinessEvidence] = {}
    for item in readiness:
        if not isinstance(item, LanguageReadinessEvidence):
            item = LanguageReadinessEvidence.from_dict(item)
        if item.language in indexed or len(indexed) >= _MAX_READINESS:
            raise _refuse(malformed)
        indexed[item.language] = item
    return indexed


def _locale_language(locale: str) -> str:
    return re.split(r"[_-]", locale, maxsplit=1)[0].lower()


def _build_pack(spec: _PackSpec, reference: str) -> LanguagePack:
    try:
        pack = LanguagePack(
            code=spec.code,
            default_model=reference,
            **spec.declaration,
        )
    except (AttributeError, TypeError, ValueError):
        raise _refuse(ActivationRefusal.MANIFEST_MALFORMED) from None
    if (
        _locale_language(pack.surrogate_locale) != pack.code
        and pack.surrogate_locale_approximation is None
    ):
        raise _refuse(ActivationRefusal.LOCALE_APPROXIMATION_UNDECLARED)
    return pack


def _approximated_codes(packs: Mapping[str, LanguagePack]) -> frozenset[str]:
    staging = LanguagePackRegistry()
    for pack in packs.values():
        staging.register(pack)
    try:
        rows = pack_coherence_report(registry=staging)
    except Exception:
        raise _refuse(ActivationRefusal.PACK_INCOHERENT) from None
    if any(not row["coherent"] for row in rows):
        raise _refuse(ActivationRefusal.PACK_INCOHERENT)
    approximated = set()
    for row in rows:
        surrogate = row["surrogate_locale"]
        if isinstance(surrogate, Mapping) and surrogate.get("status") == APPROXIMATED:
            approximated.add(str(row["language"]))
    approximated.update(
        code
        for code, pack in packs.items()
        if pack.surrogate_locale_approximation is not None
        or _locale_language(pack.surrogate_locale) != code
    )
    return frozenset(approximated)


def _capability_conflict(
    candidates: Iterable[LanguagePack],
    foreign: Iterable[LanguagePack],
) -> bool:
    foreign_packs = tuple(foreign)
    for pack in candidates:
        markers = {marker.casefold() for marker in pack.routing_markers}
        for other in foreign_packs:
            shared = set(pack.scripts) & set(other.scripts)
            if not shared:
                continue
            # A new pack must not outrank an existing language on a shared
            # script, or claim the lexical cues that existing pack relies on.
            if any(pack.priority_for(script) > 0 for script in shared):
                return True
            if markers & {marker.casefold() for marker in other.routing_markers}:
                return True
    return False


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


class LanguageBundleActivator:
    """Activate verified offline language bundles as atomic registry generations.

    The activator keeps a process-local ledger of the bundles it activated.
    Packs registered by other code (built-in packs or user packs registered
    with :func:`~openmed.core.language_pack.register_language_pack`) are never
    replaced or removed: a bundle that declares one of their codes is refused.
    A bundle may supersede only its own earlier activation, and only with a
    strictly newer version; :meth:`rollback` restores the superseded state.

    Args:
        verifier: Trust anchor used to check every manifest signature.
        registry: Registry to activate into. Defaults to OpenMed's
            process-local language-pack registry.
        clock: Zero-argument callable returning the current timezone-aware
            time. Defaults to the system UTC clock.
        max_readiness_age: Maximum age of readiness evidence at activation.
        clock_skew: Tolerance for evidence issued slightly in the future.

    Raises:
        TypeError: If ``verifier`` has no callable ``verify`` method or the
            durations are not :class:`~datetime.timedelta` values.
        ValueError: If a duration is negative or ``max_readiness_age`` is zero.
    """

    def __init__(
        self,
        *,
        verifier: BundleSignatureVerifier,
        registry: LanguagePackRegistry | None = None,
        clock: Callable[[], datetime] | None = None,
        max_readiness_age: timedelta = DEFAULT_MAX_READINESS_AGE,
        clock_skew: timedelta = DEFAULT_CLOCK_SKEW,
    ) -> None:
        """Create an activator bound to one registry and one trust anchor."""

        if not callable(getattr(verifier, "verify", None)):
            raise TypeError("verifier must provide a verify() method")
        if registry is not None and not isinstance(registry, LanguagePackRegistry):
            raise TypeError("registry must be a LanguagePackRegistry")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        if not isinstance(max_readiness_age, timedelta) or not isinstance(
            clock_skew, timedelta
        ):
            raise TypeError("readiness durations must be timedelta values")
        if max_readiness_age <= timedelta(0) or clock_skew < timedelta(0):
            raise ValueError("readiness durations must not be negative")
        self._verifier = verifier
        self._registry = registry if registry is not None else LANGUAGE_PACK_REGISTRY
        self._clock = clock if clock is not None else _default_clock
        self._max_age = max_readiness_age
        self._skew = clock_skew
        self._active: dict[str, _ActiveEntry] = {}
        self._lock = RLock()

    @property
    def registry(self) -> LanguagePackRegistry:
        """Return the registry this activator commits generations to."""

        return self._registry

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.utcoffset() is None:
            raise TypeError("clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)

    def _verify_signature(
        self,
        payload: bytes,
        scheme: str,
        key_id: str,
        value: str,
    ) -> None:
        try:
            valid = self._verifier.verify(
                payload, scheme=scheme, key_id=key_id, signature=value
            )
        except Exception:
            valid = False
        # Only an explicit ``True`` passes; truthy placeholders fail closed.
        if valid is not True:
            raise _refuse(ActivationRefusal.SIGNATURE_INVALID)

    def _check_readiness(
        self,
        evidence: LanguageReadinessEvidence | None,
        model_digest: str,
        now: datetime,
    ) -> LanguageReadinessEvidence:
        if evidence is None:
            raise _refuse(ActivationRefusal.READINESS_MISSING)
        if evidence.issued_at > now + self._skew:
            raise _refuse(ActivationRefusal.READINESS_NOT_YET_VALID)
        if now - evidence.issued_at > self._max_age:
            raise _refuse(ActivationRefusal.READINESS_STALE)
        if evidence.state not in ACTIVATABLE_READINESS_STATES:
            raise _refuse(ActivationRefusal.READINESS_INSUFFICIENT)
        if evidence.model_digest is not None and evidence.model_digest != model_digest:
            raise _refuse(ActivationRefusal.READINESS_SUBJECT_MISMATCH)
        return evidence

    def _commit(self, table: Mapping[str, LanguagePack], generation: int) -> None:
        try:
            committed = self._registry._commit_generation(
                table, expected_generation=generation
            )
        except Exception:
            raise _refuse(ActivationRefusal.REGISTRY_REFRESH_FAILED) from None
        if committed is None:
            raise _refuse(ActivationRefusal.REGISTRY_CHANGED)

    def activate(
        self,
        manifest: Mapping[str, Any],
        readiness: Iterable[LanguageReadinessEvidence | Mapping[str, Any]],
        model_handles: Mapping[str, LocalModelHandle],
        *,
        expected_activation_id: str | None = None,
    ) -> LanguageBundleActivation:
        """Validate a local bundle and commit its packs as one generation.

        Validation runs completely before anything is committed: schema,
        signature, unknown critical fields, model-handle locality and digests,
        readiness language/freshness/state, task qualification, locale
        approximation, pack coherence, and pack identity or capability
        conflicts. Only declared tasks that the readiness evidence also
        qualifies are recorded as activated.

        Args:
            manifest: Plain bundle manifest mapping.
            readiness: Evidence objects or their :meth:`to_dict` mappings.
            model_handles: Installed model artifacts keyed by manifest alias.
            expected_activation_id: Optional identifier from an earlier
                serialized record. When given, activation is refused unless
                the recomputed identifier matches, which makes reloading a
                persisted activation fail closed on any drift.

        Returns:
            The committed activation record.

        Raises:
            LanguageBundleActivationError: With a controlled reason when the
                bundle is refused. The registry is unchanged in that case.
        """

        payload = _signed_payload(manifest)
        self._verify_signature(payload, *_parse_signature(manifest))
        # Parse the verified bytes rather than the caller's mutable mapping, so
        # activation uses exactly the content the signature covers.
        bundle = _parse_manifest(json.loads(payload), payload)
        evidence = _index_readiness(readiness)
        if not isinstance(model_handles, Mapping):
            raise _refuse(ActivationRefusal.ARTIFACT_HANDLE_MISSING)
        now = self._now()

        packs: dict[str, LanguagePack] = {}
        checked: list[tuple[_PackSpec, LanguageReadinessEvidence, str]] = []
        for spec in bundle.packs:
            model_digest = bundle.artifacts[spec.model][1]
            handle = model_handles.get(spec.model)
            if not isinstance(handle, LocalModelHandle):
                raise _refuse(ActivationRefusal.ARTIFACT_HANDLE_MISSING)
            if not _is_local_reference(handle.reference):
                raise _refuse(ActivationRefusal.ARTIFACT_HANDLE_NOT_LOCAL)
            if handle.sha256 != model_digest:
                raise _refuse(ActivationRefusal.ARTIFACT_DIGEST_MISMATCH)
            item = self._check_readiness(evidence.get(spec.language), model_digest, now)
            if not set(spec.tasks) & set(item.qualified_tasks):
                raise _refuse(ActivationRefusal.TASKS_NOT_QUALIFIED)
            packs[spec.code] = _build_pack(spec, handle.reference)
            checked.append((spec, item, model_digest))

        approximated = _approximated_codes(packs)
        languages = []
        for spec, item, model_digest in checked:
            qualified = set(item.qualified_tasks)
            locale_status = (
                LOCALE_APPROXIMATED if spec.code in approximated else LOCALE_NATIVE
            )
            claim = (
                CLAIM_SUPPORTED
                if item.state == CLAIM_SUPPORTED and locale_status == LOCALE_NATIVE
                else CLAIM_PREVIEW
            )
            languages.append(
                ActivatedLanguage(
                    language=spec.language,
                    code=spec.code,
                    tasks=tuple(task for task in spec.tasks if task in qualified),
                    withheld_tasks=tuple(
                        task for task in spec.tasks if task not in qualified
                    ),
                    claim=claim,
                    locale_status=locale_status,
                    readiness_state=item.state,
                    readiness_digest=item.evidence_digest,
                    model_digest=model_digest,
                )
            )
        activated = tuple(languages)
        activation_id = _activation_id(
            bundle.bundle_id, bundle.version, bundle.digest, activated
        )
        if (
            expected_activation_id is not None
            and expected_activation_id != activation_id
        ):
            raise _refuse(ActivationRefusal.ACTIVATION_ID_MISMATCH)

        with self._lock:
            generation, table = self._registry._snapshot()
            current = self._active.get(bundle.bundle_id)
            owned: Mapping[str, LanguagePack] = {}
            if current is not None:
                if _version_key(bundle.version) <= _version_key(
                    current.record.bundle_version
                ):
                    raise _refuse(ActivationRefusal.BUNDLE_VERSION_NOT_NEWER)
                owned = current.packs
                if any(table.get(code) is not pack for code, pack in owned.items()):
                    raise _refuse(ActivationRefusal.PACK_IDENTITY_CONFLICT)
            if any(code in table and code not in owned for code in packs):
                raise _refuse(ActivationRefusal.PACK_IDENTITY_CONFLICT)
            retained = {code: pack for code, pack in table.items() if code not in owned}
            if _capability_conflict(packs.values(), retained.values()):
                raise _refuse(ActivationRefusal.CAPABILITY_CONFLICT)
            record = LanguageBundleActivation(
                activation_id=activation_id,
                bundle_id=bundle.bundle_id,
                bundle_version=bundle.version,
                manifest_digest=bundle.digest,
                activated_at=now,
                languages=activated,
                previous_activation_id=(
                    current.record.activation_id if current is not None else None
                ),
            )
            self._commit({**retained, **packs}, generation)
            self._active[bundle.bundle_id] = _ActiveEntry(
                record=record,
                packs=dict(packs),
                previous=current,
            )
        return record

    def rollback(self, activation_id: str) -> LanguageBundleActivation | None:
        """Undo the active activation ``activation_id`` as one generation.

        The bundle's packs are removed and, when the activation superseded an
        earlier version of the same bundle, that earlier generation's packs are
        restored exactly as they were activated. Packs registered by other
        code are never touched. If a pack this activation installed was since
        replaced or removed by other code, or a restored code is now taken,
        the rollback is refused rather than guessing.

        Args:
            activation_id: Identifier of the currently active activation.

        Returns:
            The restored earlier activation record, or ``None`` when the bundle
            is no longer active.

        Raises:
            LanguageBundleActivationError: With ``rollback_not_active`` or
                ``rollback_conflict``; the registry is unchanged.
        """

        with self._lock:
            entry = next(
                (
                    item
                    for item in self._active.values()
                    if item.record.activation_id == activation_id
                ),
                None,
            )
            if entry is None:
                raise _refuse(ActivationRefusal.ROLLBACK_NOT_ACTIVE)
            generation, table = self._registry._snapshot()
            if any(table.get(code) is not pack for code, pack in entry.packs.items()):
                raise _refuse(ActivationRefusal.ROLLBACK_CONFLICT)
            retained = {
                code: pack for code, pack in table.items() if code not in entry.packs
            }
            previous = entry.previous
            if previous is not None:
                if any(code in retained for code in previous.packs) or (
                    _capability_conflict(previous.packs.values(), retained.values())
                ):
                    raise _refuse(ActivationRefusal.ROLLBACK_CONFLICT)
                retained.update(previous.packs)
            self._commit(retained, generation)
            if previous is None:
                del self._active[entry.record.bundle_id]
                return None
            self._active[entry.record.bundle_id] = previous
            return previous.record

    def active_activations(self) -> tuple[LanguageBundleActivation, ...]:
        """Return the active activation records ordered by bundle identifier."""

        with self._lock:
            return tuple(
                self._active[bundle_id].record for bundle_id in sorted(self._active)
            )

    def _activated_language(self, language: str) -> ActivatedLanguage | None:
        tag = normalize_locale_tag(language)
        with self._lock:
            for entry in self._active.values():
                for item in entry.record.languages:
                    if item.language == tag:
                        return item
        return None

    def qualified_tasks(self, language: str) -> tuple[str, ...]:
        """Return tasks activated for exactly ``language``.

        Lookup is by exact canonical tag: tasks activated for ``pt-BR`` are not
        returned for ``pt`` and vice versa.

        Args:
            language: BCP 47 language tag.

        Returns:
            Sorted activated task codes, or an empty tuple.
        """

        item = self._activated_language(language)
        return item.tasks if item is not None else ()

    def clinical_claim(self, language: str) -> str | None:
        """Return the activated clinical-language claim for exactly ``language``.

        Args:
            language: BCP 47 language tag.

        Returns:
            ``supported`` or ``preview`` for an activated tag, else ``None``.
            A pack with an approximated surrogate locale is never
            ``supported``.
        """

        item = self._activated_language(language)
        return item.claim if item is not None else None


__all__ = [
    "ACTIVATABLE_READINESS_STATES",
    "ACTIVATION_SCHEMA_VERSION",
    "ARTIFACT_KINDS",
    "BUNDLE_SCHEMA_VERSION",
    "DEFAULT_CLOCK_SKEW",
    "DEFAULT_MAX_READINESS_AGE",
    "READINESS_STATES",
    "UNDERSTOOD_EXTENSIONS",
    "ActivatedLanguage",
    "ActivationRefusal",
    "BundleSignatureVerifier",
    "LanguageBundleActivation",
    "LanguageBundleActivationError",
    "LanguageBundleActivator",
    "LanguageReadinessEvidence",
    "LocalModelHandle",
    "language_bundle_digest",
    "language_bundle_signing_payload",
]
