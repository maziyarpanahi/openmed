"""Decision calibration profiles signed inside portable model bundles.

A decision model and the thresholds qualified for it must not drift apart. The
``org.openmed.decision_calibration`` manifest extension carries one or more
calibration profiles inside the signed manifest, so the bundle signature covers
them. Each profile records:

* its identity (``calibration_id`` and ``version``) and calibration method with
  integer parameters;
* the thresholds that define the abstention bands, in parts per million;
* the option, question, or label set it was calibrated for, as a digest;
* the model it was calibrated on: model id, revision, and a digest of every
  runtime variant it qualifies;
* a compatible decision-schema range;
* provenance digests for the calibration data and an optional bundled
  evaluation report. Calibration data never ships in a bundle;
* a self digest that deployments can pin.

Readers in Python, Swift, Kotlin, and TypeScript run the same checks in the
same order and report the same value-free reason code for the golden vectors in
``tests/fixtures/portable_bundle_calibration/``. Verifiers that list the
extension in ``understood_extensions`` validate it during bundle verification
and fail with ``extension_invalid``; :func:`read_decision_calibration` and
:func:`resolve_decision_calibration` report the detailed reason. Nothing here
opens a network connection, runs a model, or reports a value other than
controlled codes, identifiers, digests, and indexes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from importlib import resources
from typing import Any

from openmed.core.portable_bundle import (
    SCHEMA_PACKAGE,
    PortableBundleError,
    PortableBundleManifest,
    PortableBundleVerification,
    canonical_json_bytes,
    content_id,
)

DECISION_CALIBRATION_EXTENSION = "org.openmed.decision_calibration"
"""Manifest extension name that carries decision calibration profiles."""
DECISION_CALIBRATION_EXTENSION_VERSION = 1
SCHEMA_NAME = "portable_bundle_decision_calibration"

PPM_SCALE = 1_000_000
"""Every probability in the extension is an integer in parts per million."""
MAX_PROFILES = 16
MAX_PROFILE_RUNTIMES = 32
MAX_ISOTONIC_POINTS = 64
MAX_TEMPERATURE_PPM = 100_000_000
MAX_PLATT_PPM = 100_000_000

CALIBRATION_METHODS = frozenset(
    {
        "deterministic_margin",
        "isotonic_regression",
        "platt_scaling",
        "temperature_scaling",
    }
)
"""Closed set of calibration methods; any other method fails closed."""
SCOPE_KINDS = frozenset({"labels", "options", "questions"})
"""What a profile was calibrated for: the bundle label map, an option list, or
an ordered question set."""


class DecisionCalibrationReason(str, Enum):
    """Stable, value-free reason codes shared by every calibration reader."""

    BUNDLE_UNVERIFIED = "calibration_bundle_unverified"
    EXTENSION_MISSING = "calibration_extension_missing"
    MALFORMED = "calibration_malformed"
    VERSION_UNSUPPORTED = "calibration_version_unsupported"
    DIGEST_MISMATCH = "calibration_digest_mismatch"
    MODEL_MISMATCH = "calibration_model_mismatch"
    RUNTIME_MISMATCH = "calibration_runtime_mismatch"
    SCOPE_MISMATCH = "calibration_scope_mismatch"
    EVIDENCE_MISMATCH = "calibration_evidence_mismatch"
    PROFILE_MISSING = "calibration_profile_missing"
    RUNTIME_UNBOUND = "calibration_runtime_unbound"
    SCHEMA_INCOMPATIBLE = "calibration_schema_incompatible"
    PIN_MISMATCH = "calibration_pin_mismatch"
    REQUEST_INVALID = "calibration_request_invalid"


REASON_CODES: tuple[str, ...] = tuple(item.value for item in DecisionCalibrationReason)
"""Every calibration reason code, grouped by reading stage."""


class DecisionCalibrationError(ValueError):
    """Raised with a controlled reason code only.

    Attributes:
        reason: The controlled :class:`DecisionCalibrationReason`.
    """

    def __init__(self, reason: DecisionCalibrationReason) -> None:
        """Store the controlled reason and use its value as the message."""

        self.reason = reason
        super().__init__(reason.value)


_R = DecisionCalibrationReason
_CONTROLLED = re.compile(r"[a-z][a-z0-9_.:/-]{0,127}")
_TOKEN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")
_SEMVER = re.compile(r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){2}")
# Any declared bundle path matches; whether it is declared is checked later.
_EVALUATION_PATH = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._+/-]{0,511}")

_EXTENSION_KEYS = frozenset({"profiles", "schema_version"})
_PROFILE_KEYS = frozenset(
    {
        "calibration_id",
        "decision_schema",
        "method",
        "model",
        "parameters",
        "profile_digest",
        "provenance",
        "scope",
        "thresholds",
        "version",
    }
)
_PARAMETER_KEYS: Mapping[str, frozenset[str]] = {
    "deterministic_margin": frozenset(),
    "isotonic_regression": frozenset({"points"}),
    "platt_scaling": frozenset({"intercept_ppm", "slope_ppm"}),
    "temperature_scaling": frozenset({"temperature_ppm"}),
}
_POINT_KEYS = frozenset({"input_ppm", "output_ppm"})
_THRESHOLD_KEYS = frozenset(
    {"minimum_confidence_ppm", "minimum_margin_ppm", "multi_label_threshold_ppm"}
)
_SCOPE_KEYS = frozenset({"digest", "kind"})
_MODEL_KEYS = frozenset({"id", "revision", "runtimes"})
_BINDING_KEYS = frozenset({"digest", "id"})
_SCHEMA_RANGE_KEYS = frozenset({"max_exclusive", "min"})
_PROVENANCE_KEYS = frozenset(
    {"dataset_digest", "evaluation", "in_domain_digest", "out_of_domain_digest"}
)
_EVALUATION_KEYS = frozenset({"path", "sha256"})


class _Malformed(Exception):
    pass


# --------------------------------------------------------------------------
# Digests
# --------------------------------------------------------------------------


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def calibration_profile_digest(profile: Mapping[str, Any]) -> str:
    """Return the self digest of one extension profile object.

    The digest is the SHA-256 of the RFC 8785 canonical encoding of the profile
    without its ``profile_digest`` member.

    Args:
        profile: One entry of the extension ``profiles`` array.

    Returns:
        ``sha256:<hex>`` digest.

    Raises:
        PortableBundleError: ``manifest_not_canonical`` when the profile holds
            values outside the canonical JSON subset.
    """

    body = {key: value for key, value in profile.items() if key != "profile_digest"}
    return _sha256(canonical_json_bytes(body))


def calibration_runtime_digest(manifest: Mapping[str, Any], runtime_id: str) -> str:
    """Return the digest that binds a profile to one runtime variant.

    The digest is the SHA-256 of the canonical encoding of
    ``{"entrypoint", "files": [{"path", "sha256"}], "format", "id",
    "precision"}`` with the runtime's files in declared order and each file's
    ``sha256`` taken from the manifest inventory. Re-exported weights, a new
    precision, or a different entrypoint all change it.

    Args:
        manifest: Structurally valid manifest mapping.
        runtime_id: Identifier of a runtime declared in ``manifest``.

    Returns:
        ``sha256:<hex>`` digest.

    Raises:
        KeyError: If the runtime or one of its files is not declared.
    """

    runtime = next(
        (item for item in manifest["runtimes"] if item["id"] == runtime_id), None
    )
    if runtime is None:
        raise KeyError("runtime is not declared")
    digests = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    body = {
        "entrypoint": runtime["entrypoint"],
        "files": [{"path": path, "sha256": digests[path]} for path in runtime["files"]],
        "format": runtime["format"],
        "id": runtime["id"],
        "precision": runtime["precision"],
    }
    return _sha256(canonical_json_bytes(body))


def options_scope_digest(options: Sequence[str]) -> str:
    """Return the ``options`` scope digest of an ordered option list.

    Args:
        options: Option labels in request order.

    Returns:
        SHA-256 of the canonical JSON array of the options.
    """

    return _sha256(canonical_json_bytes([str(item) for item in options]))


def questions_scope_digest(questions: Sequence[Mapping[str, Any]]) -> str:
    """Return the ``questions`` scope digest of an ordered question set.

    Args:
        questions: Canonical question mappings in request order, as produced by
            ``DecisionQuestion.to_dict()``.

    Returns:
        SHA-256 of the canonical JSON array of the question mappings.
    """

    return _sha256(canonical_json_bytes([dict(item) for item in questions]))


# --------------------------------------------------------------------------
# Profile model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRuntimeBinding:
    """One runtime variant a profile was calibrated on.

    Attributes:
        runtime_id: Runtime identifier declared in the manifest.
        digest: :func:`calibration_runtime_digest` of that runtime.
    """

    runtime_id: str
    digest: str


@dataclass(frozen=True)
class BundleCalibrationProfile:
    """A validated calibration profile read from a signed bundle.

    Probabilities are integers in parts per million; the float properties
    divide by :data:`PPM_SCALE`. ``to_dict`` rebuilds the exact extension
    profile object from these typed fields.
    """

    calibration_id: str
    version: str
    method: str
    minimum_confidence_ppm: int
    minimum_margin_ppm: int
    multi_label_threshold_ppm: int
    scope_kind: str
    scope_digest: str
    model_id: str
    model_revision: str | None
    runtimes: tuple[CalibrationRuntimeBinding, ...]
    decision_schema_min: str
    decision_schema_max_exclusive: str
    dataset_digest: str
    in_domain_digest: str
    out_of_domain_digest: str
    evaluation_path: str | None
    evaluation_sha256: str | None
    profile_digest: str
    temperature_ppm: int | None = None
    platt_slope_ppm: int | None = None
    platt_intercept_ppm: int | None = None
    isotonic_points: tuple[tuple[int, int], ...] = ()

    @property
    def minimum_confidence(self) -> float:
        """Return the minimum top-choice confidence as a probability."""

        return self.minimum_confidence_ppm / PPM_SCALE

    @property
    def minimum_margin(self) -> float:
        """Return the minimum top-two margin as a probability."""

        return self.minimum_margin_ppm / PPM_SCALE

    @property
    def multi_label_threshold(self) -> float:
        """Return the multi-label selection threshold as a probability."""

        return self.multi_label_threshold_ppm / PPM_SCALE

    @property
    def runtime_ids(self) -> tuple[str, ...]:
        """Return the bound runtime identifiers in declared order."""

        return tuple(item.runtime_id for item in self.runtimes)

    def runtime_digest(self, runtime_id: str) -> str | None:
        """Return the bound digest for ``runtime_id``, or ``None``."""

        for item in self.runtimes:
            if item.runtime_id == runtime_id:
                return item.digest
        return None

    @property
    def parameters(self) -> dict[str, Any]:
        """Return the method parameters as the extension stores them."""

        if self.method == "temperature_scaling":
            return {"temperature_ppm": self.temperature_ppm}
        if self.method == "platt_scaling":
            return {
                "intercept_ppm": self.platt_intercept_ppm,
                "slope_ppm": self.platt_slope_ppm,
            }
        if self.method == "isotonic_regression":
            return {
                "points": [
                    {"input_ppm": point[0], "output_ppm": point[1]}
                    for point in self.isotonic_points
                ]
            }
        return {}

    def to_dict(self) -> dict[str, Any]:
        """Return the extension profile object rebuilt from typed fields."""

        evaluation = (
            None
            if self.evaluation_path is None
            else {"path": self.evaluation_path, "sha256": self.evaluation_sha256}
        )
        return {
            "calibration_id": self.calibration_id,
            "decision_schema": {
                "max_exclusive": self.decision_schema_max_exclusive,
                "min": self.decision_schema_min,
            },
            "method": self.method,
            "model": {
                "id": self.model_id,
                "revision": self.model_revision,
                "runtimes": [
                    {"digest": item.digest, "id": item.runtime_id}
                    for item in self.runtimes
                ],
            },
            "parameters": self.parameters,
            "profile_digest": self.profile_digest,
            "provenance": {
                "dataset_digest": self.dataset_digest,
                "evaluation": evaluation,
                "in_domain_digest": self.in_domain_digest,
                "out_of_domain_digest": self.out_of_domain_digest,
            },
            "scope": {"digest": self.scope_digest, "kind": self.scope_kind},
            "thresholds": {
                "minimum_confidence_ppm": self.minimum_confidence_ppm,
                "minimum_margin_ppm": self.minimum_margin_ppm,
                "multi_label_threshold_ppm": self.multi_label_threshold_ppm,
            },
            "version": self.version,
        }

    def canonical_bytes(self) -> bytes:
        """Return the canonical JSON encoding of :meth:`to_dict`."""

        return canonical_json_bytes(self.to_dict())


# --------------------------------------------------------------------------
# Extension validation
# --------------------------------------------------------------------------


def _object(value: object, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise _Malformed()
    return value


def _text(value: object, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise _Malformed()
    return value


def _int(value: object, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _Malformed()
    return value


def _semver_key(value: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in value.split("."))
    return (major, minor, patch)


def _parse_parameters(method: str, value: object) -> dict[str, Any]:
    parameters = _object(value, _PARAMETER_KEYS[method])
    if method == "temperature_scaling":
        return {
            "temperature_ppm": _int(
                parameters["temperature_ppm"], 1, MAX_TEMPERATURE_PPM
            )
        }
    if method == "platt_scaling":
        slope = _int(parameters["slope_ppm"], -MAX_PLATT_PPM, MAX_PLATT_PPM)
        if slope == 0:
            raise _Malformed()
        return {
            "platt_slope_ppm": slope,
            "platt_intercept_ppm": _int(
                parameters["intercept_ppm"], -MAX_PLATT_PPM, MAX_PLATT_PPM
            ),
        }
    if method == "isotonic_regression":
        raw = parameters["points"]
        if not isinstance(raw, list) or not 2 <= len(raw) <= MAX_ISOTONIC_POINTS:
            raise _Malformed()
        points: list[tuple[int, int]] = []
        for item in raw:
            point = _object(item, _POINT_KEYS)
            pair = (
                _int(point["input_ppm"], 0, PPM_SCALE),
                _int(point["output_ppm"], 0, PPM_SCALE),
            )
            if points and (pair[0] <= points[-1][0] or pair[1] < points[-1][1]):
                raise _Malformed()
            points.append(pair)
        return {"isotonic_points": tuple(points)}
    return {}


def _parse_profile(value: object) -> BundleCalibrationProfile:
    profile = _object(value, _PROFILE_KEYS)
    calibration_id = _text(profile["calibration_id"], _CONTROLLED)
    version = _text(profile["version"], _SEMVER)
    method = profile["method"]
    if not isinstance(method, str) or method not in CALIBRATION_METHODS:
        raise _Malformed()
    parameters = _parse_parameters(method, profile["parameters"])

    thresholds = _object(profile["thresholds"], _THRESHOLD_KEYS)
    confidence = _int(thresholds["minimum_confidence_ppm"], 0, PPM_SCALE)
    margin = _int(thresholds["minimum_margin_ppm"], 0, PPM_SCALE)
    multi_label = _int(thresholds["multi_label_threshold_ppm"], 0, PPM_SCALE)
    if margin > confidence:
        raise _Malformed()

    scope = _object(profile["scope"], _SCOPE_KEYS)
    kind = scope["kind"]
    if not isinstance(kind, str) or kind not in SCOPE_KINDS:
        raise _Malformed()
    scope_digest = _text(scope["digest"], _SHA256)

    model = _object(profile["model"], _MODEL_KEYS)
    model_id = _text(model["id"], _MODEL_ID)
    revision = model["revision"]
    if revision is not None:
        _text(revision, _REVISION)
    raw_runtimes = model["runtimes"]
    if (
        not isinstance(raw_runtimes, list)
        or not 1 <= len(raw_runtimes) <= MAX_PROFILE_RUNTIMES
    ):
        raise _Malformed()
    bindings = []
    for item in raw_runtimes:
        binding = _object(item, _BINDING_KEYS)
        bindings.append(
            CalibrationRuntimeBinding(
                runtime_id=_text(binding["id"], _TOKEN),
                digest=_text(binding["digest"], _SHA256),
            )
        )
    if len({item.runtime_id for item in bindings}) != len(bindings):
        raise _Malformed()

    schema_range = _object(profile["decision_schema"], _SCHEMA_RANGE_KEYS)
    schema_min = _text(schema_range["min"], _SEMVER)
    schema_max = _text(schema_range["max_exclusive"], _SEMVER)
    if _semver_key(schema_min) >= _semver_key(schema_max):
        raise _Malformed()

    provenance = _object(profile["provenance"], _PROVENANCE_KEYS)
    dataset_digest = _text(provenance["dataset_digest"], _SHA256)
    in_domain = _text(provenance["in_domain_digest"], _SHA256)
    out_of_domain = _text(provenance["out_of_domain_digest"], _SHA256)
    evaluation_path: str | None = None
    evaluation_sha256: str | None = None
    if provenance["evaluation"] is not None:
        evaluation = _object(provenance["evaluation"], _EVALUATION_KEYS)
        evaluation_path = _text(evaluation["path"], _EVALUATION_PATH)
        evaluation_sha256 = _text(evaluation["sha256"], _SHA256)

    return BundleCalibrationProfile(
        calibration_id=calibration_id,
        version=version,
        method=method,
        minimum_confidence_ppm=confidence,
        minimum_margin_ppm=margin,
        multi_label_threshold_ppm=multi_label,
        scope_kind=kind,
        scope_digest=scope_digest,
        model_id=model_id,
        model_revision=revision,
        runtimes=tuple(bindings),
        decision_schema_min=schema_min,
        decision_schema_max_exclusive=schema_max,
        dataset_digest=dataset_digest,
        in_domain_digest=in_domain,
        out_of_domain_digest=out_of_domain,
        evaluation_path=evaluation_path,
        evaluation_sha256=evaluation_sha256,
        profile_digest=_text(profile["profile_digest"], _SHA256),
        **parameters,
    )


def validate_decision_calibration_extension(
    manifest: Mapping[str, Any],
) -> tuple[BundleCalibrationProfile, ...]:
    """Validate the calibration extension of a structurally valid manifest.

    Checks run in the normative order: the extension envelope, every profile's
    shape, unique ``(calibration_id, runtime)`` bindings, then for each profile
    its self digest, model identity, runtime digests, label-map scope, and
    evaluation evidence.

    Args:
        manifest: Manifest mapping that already passed portable-bundle
            structural validation and contains the extension.

    Returns:
        The validated profiles in declared order.

    Raises:
        DecisionCalibrationError: With the first failing reason.
    """

    raw = manifest["extensions"][DECISION_CALIBRATION_EXTENSION]
    try:
        if not isinstance(raw, dict) or type(raw.get("schema_version")) is not int:
            raise _Malformed()
        if raw["schema_version"] != DECISION_CALIBRATION_EXTENSION_VERSION:
            raise DecisionCalibrationError(_R.VERSION_UNSUPPORTED)
        if set(raw) != _EXTENSION_KEYS:
            raise _Malformed()
        items = raw["profiles"]
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_PROFILES:
            raise _Malformed()
        profiles = tuple(_parse_profile(item) for item in items)
        bindings = [
            (profile.calibration_id, runtime_id)
            for profile in profiles
            for runtime_id in profile.runtime_ids
        ]
        if len(set(bindings)) != len(bindings):
            raise _Malformed()
    except _Malformed:
        raise DecisionCalibrationError(_R.MALFORMED) from None

    model = manifest["model"]
    runtime_ids = {str(item["id"]) for item in manifest["runtimes"]}
    digests = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    labels = manifest["labels"]
    for profile, item in zip(profiles, items, strict=True):
        if calibration_profile_digest(item) != profile.profile_digest:
            raise DecisionCalibrationError(_R.DIGEST_MISMATCH)
        if (
            profile.model_id != model["id"]
            or profile.model_revision != model["revision"]
        ):
            raise DecisionCalibrationError(_R.MODEL_MISMATCH)
        for binding in profile.runtimes:
            if binding.runtime_id not in runtime_ids or (
                calibration_runtime_digest(manifest, binding.runtime_id)
                != binding.digest
            ):
                raise DecisionCalibrationError(_R.RUNTIME_MISMATCH)
        if profile.scope_kind == "labels" and (
            labels is None or digests.get(str(labels["path"])) != profile.scope_digest
        ):
            raise DecisionCalibrationError(_R.SCOPE_MISMATCH)
        if profile.evaluation_path is not None and (
            digests.get(profile.evaluation_path) != profile.evaluation_sha256
        ):
            raise DecisionCalibrationError(_R.EVIDENCE_MISMATCH)
    return profiles


# --------------------------------------------------------------------------
# Reading verified bundles
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionCalibrationRead:
    """Value-free result of reading calibration profiles from a bundle.

    Attributes:
        ok: Whether every profile validated.
        reason: First failing reason code, or ``None`` when ``ok``.
        content_id: Manifest content identity.
        profiles: Validated profiles in declared order; empty unless ``ok``.
    """

    ok: bool
    reason: str | None
    content_id: str
    profiles: tuple[BundleCalibrationProfile, ...] = ()


@dataclass(frozen=True)
class DecisionCalibrationRequest:
    """What the caller loaded and needs a calibration profile for.

    Args:
        calibration_id: Calibration identifier the decision request names.
        runtime_id: Runtime variant the caller loaded from the bundle.
        decision_schema_version: Decision schema version the caller speaks.
        model_id: Model id of the loaded model, when known independently.
        model_revision: Model revision of the loaded model, when known.
        runtime_digest: Runtime digest of the loaded model, when known.
        scope_kind: Scope the caller is about to decide over (with
            ``scope_digest``).
        scope_digest: Digest of that scope.
        profile_digest: Profile digest pinned by deployment qualification.

    Raises:
        DecisionCalibrationError: ``calibration_request_invalid`` when a field
            is malformed.
    """

    calibration_id: str
    runtime_id: str
    decision_schema_version: str
    model_id: str | None = None
    model_revision: str | None = None
    runtime_digest: str | None = None
    scope_kind: str | None = None
    scope_digest: str | None = None
    profile_digest: str | None = None

    def __post_init__(self) -> None:
        """Validate every caller field against the extension patterns."""

        def matches(value: object, pattern: re.Pattern[str]) -> bool:
            return isinstance(value, str) and pattern.fullmatch(value) is not None

        optional = (
            (self.model_id, _MODEL_ID),
            (self.model_revision, _REVISION),
            (self.runtime_digest, _SHA256),
            (self.profile_digest, _SHA256),
        )
        if (
            not matches(self.calibration_id, _CONTROLLED)
            or not matches(self.runtime_id, _TOKEN)
            or not matches(self.decision_schema_version, _SEMVER)
            or not all(value is None or matches(value, rx) for value, rx in optional)
            or (self.scope_kind is None) != (self.scope_digest is None)
            or (
                self.scope_kind is not None
                and (
                    self.scope_kind not in SCOPE_KINDS
                    or not matches(self.scope_digest, _SHA256)
                )
            )
        ):
            raise DecisionCalibrationError(_R.REQUEST_INVALID)


@dataclass(frozen=True)
class DecisionCalibrationSelection:
    """Value-free result of resolving the profile for a loaded runtime.

    Attributes:
        ok: Whether a profile was selected.
        reason: First failing reason code, or ``None`` when ``ok``.
        content_id: Manifest content identity.
        profile: The selected profile; ``None`` unless ``ok``.
    """

    ok: bool
    reason: str | None
    content_id: str
    profile: BundleCalibrationProfile | None = None


def _structural_manifest(manifest_bytes: bytes) -> dict[str, Any] | None:
    try:
        value = json.loads(manifest_bytes.decode("utf-8"))
        critical = value.get("critical_extensions") if isinstance(value, dict) else None
        understood = (
            [item for item in critical if isinstance(item, str)]
            if isinstance(critical, list)
            else []
        )
        return PortableBundleManifest.from_bytes(
            manifest_bytes, understood_extensions=understood
        ).to_dict()
    except (PortableBundleError, ValueError, UnicodeDecodeError, RecursionError):
        return None


def _read(
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
    *,
    require_files: bool,
) -> DecisionCalibrationRead:
    if not isinstance(verification, PortableBundleVerification):
        raise TypeError("verification must be a PortableBundleVerification")
    manifest_bytes = bytes(manifest_bytes)
    identity = content_id(manifest_bytes)

    def refused(reason: DecisionCalibrationReason) -> DecisionCalibrationRead:
        return DecisionCalibrationRead(
            ok=False, reason=reason.value, content_id=identity
        )

    if (
        not verification.ok
        or verification.content_id != identity
        or (require_files and not verification.files_verified)
    ):
        return refused(_R.BUNDLE_UNVERIFIED)
    manifest = _structural_manifest(manifest_bytes)
    if manifest is None:
        return refused(_R.BUNDLE_UNVERIFIED)
    if DECISION_CALIBRATION_EXTENSION not in manifest["extensions"]:
        return refused(_R.EXTENSION_MISSING)
    try:
        profiles = validate_decision_calibration_extension(manifest)
    except DecisionCalibrationError as error:
        return refused(error.reason)
    return DecisionCalibrationRead(
        ok=True, reason=None, content_id=identity, profiles=profiles
    )


def read_decision_calibration(
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
) -> DecisionCalibrationRead:
    """Read and validate every calibration profile of a verified bundle.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.
        verification: The verifier result for those bytes; it must be ``ok``
            and carry the same content identity.

    Returns:
        The profiles, or the first failing reason code.

    Raises:
        TypeError: If ``verification`` is not a verification result.
    """

    return _read(manifest_bytes, verification, require_files=False)


def resolve_decision_calibration(
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
    request: DecisionCalibrationRequest,
) -> DecisionCalibrationSelection:
    """Select the profile for the runtime a caller loaded, failing closed.

    The bundle must have verified with its file tree. After every profile
    validates, the profile with ``request.calibration_id`` that binds
    ``request.runtime_id`` is selected. Then any loaded model identity,
    decision schema version, scope, and pinned profile digest the caller
    supplies must match it.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.
        verification: The verifier result for those bytes, including files.
        request: What was loaded and what is about to be decided.

    Returns:
        The selected profile, or the first failing reason code.

    Raises:
        TypeError: If an argument has the wrong type.
    """

    if not isinstance(request, DecisionCalibrationRequest):
        raise TypeError("request must be a DecisionCalibrationRequest")
    read = _read(manifest_bytes, verification, require_files=True)

    def refused(reason: DecisionCalibrationReason) -> DecisionCalibrationSelection:
        return DecisionCalibrationSelection(
            ok=False, reason=reason.value, content_id=read.content_id
        )

    if not read.ok:
        return DecisionCalibrationSelection(
            ok=False, reason=read.reason, content_id=read.content_id
        )
    candidates = [
        item for item in read.profiles if item.calibration_id == request.calibration_id
    ]
    if not candidates:
        return refused(_R.PROFILE_MISSING)
    profile = next(
        (item for item in candidates if request.runtime_id in item.runtime_ids), None
    )
    if profile is None:
        return refused(_R.RUNTIME_UNBOUND)
    if (
        (request.model_id is not None and request.model_id != profile.model_id)
        or (
            request.model_revision is not None
            and request.model_revision != profile.model_revision
        )
        or (
            request.runtime_digest is not None
            and request.runtime_digest != profile.runtime_digest(request.runtime_id)
        )
    ):
        return refused(_R.MODEL_MISMATCH)
    version = _semver_key(request.decision_schema_version)
    if not (
        _semver_key(profile.decision_schema_min)
        <= version
        < _semver_key(profile.decision_schema_max_exclusive)
    ):
        return refused(_R.SCHEMA_INCOMPATIBLE)
    if request.scope_kind is not None and (
        request.scope_kind != profile.scope_kind
        or request.scope_digest != profile.scope_digest
    ):
        return refused(_R.SCOPE_MISMATCH)
    if (
        request.profile_digest is not None
        and request.profile_digest != profile.profile_digest
    ):
        return refused(_R.PIN_MISMATCH)
    return DecisionCalibrationSelection(
        ok=True, reason=None, content_id=read.content_id, profile=profile
    )


# --------------------------------------------------------------------------
# Producing profiles
# --------------------------------------------------------------------------


def build_calibration_profile(
    manifest: Mapping[str, Any],
    *,
    calibration_id: str,
    version: str,
    runtime_ids: Sequence[str],
    scope_kind: str,
    decision_schema_min: str,
    decision_schema_max_exclusive: str,
    dataset_digest: str,
    in_domain_digest: str,
    out_of_domain_digest: str,
    minimum_confidence_ppm: int,
    minimum_margin_ppm: int,
    multi_label_threshold_ppm: int,
    method: str = "deterministic_margin",
    parameters: Mapping[str, Any] | None = None,
    scope_digest: str | None = None,
    evaluation_path: str | None = None,
) -> dict[str, Any]:
    """Return one extension profile bound to the manifest's model and files.

    The model id and revision, every runtime digest, the evaluation report
    digest, and (for ``labels`` scope) the label-map digest are read from the
    manifest, and the self digest is computed last. The profile is not yet
    validated; :func:`with_decision_calibration` validates the whole extension.

    Args:
        manifest: Manifest mapping the profile will be embedded in.
        calibration_id: Controlled calibration identifier.
        version: ``MAJOR.MINOR.PATCH`` profile version.
        runtime_ids: Runtime variants the thresholds were qualified on.
        scope_kind: ``labels``, ``options``, or ``questions``.
        decision_schema_min: Lowest compatible decision schema version.
        decision_schema_max_exclusive: First incompatible decision schema
            version.
        dataset_digest: Digest of the frozen calibration data (never the data).
        in_domain_digest: Digest of the in-domain calibration slice.
        out_of_domain_digest: Digest of the out-of-domain calibration slice.
        minimum_confidence_ppm: Top-choice confidence floor, in ppm.
        minimum_margin_ppm: Top-two margin floor, in ppm.
        multi_label_threshold_ppm: Multi-label selection threshold, in ppm.
        method: One of :data:`CALIBRATION_METHODS`.
        parameters: Integer method parameters as the extension stores them.
        scope_digest: Scope digest; derived from the label map for ``labels``.
        evaluation_path: Bundled evaluation report path, if any.

    Returns:
        A profile mapping ready for :func:`with_decision_calibration`.

    Raises:
        KeyError: If a runtime, the label map, or the evaluation report is not
            declared in ``manifest``.
    """

    digests = {str(item["path"]): str(item["sha256"]) for item in manifest["files"]}
    if scope_kind == "labels" and scope_digest is None:
        labels = manifest["labels"]
        if labels is None:
            raise KeyError("labels are not declared")
        scope_digest = digests[str(labels["path"])]
    evaluation = None
    if evaluation_path is not None:
        evaluation = {"path": evaluation_path, "sha256": digests[evaluation_path]}
    profile: dict[str, Any] = {
        "calibration_id": calibration_id,
        "decision_schema": {
            "max_exclusive": decision_schema_max_exclusive,
            "min": decision_schema_min,
        },
        "method": method,
        "model": {
            "id": manifest["model"]["id"],
            "revision": manifest["model"]["revision"],
            "runtimes": [
                {
                    "digest": calibration_runtime_digest(manifest, runtime_id),
                    "id": runtime_id,
                }
                for runtime_id in runtime_ids
            ],
        },
        "parameters": copy.deepcopy(dict(parameters or {})),
        "provenance": {
            "dataset_digest": dataset_digest,
            "evaluation": evaluation,
            "in_domain_digest": in_domain_digest,
            "out_of_domain_digest": out_of_domain_digest,
        },
        "scope": {"digest": scope_digest, "kind": scope_kind},
        "thresholds": {
            "minimum_confidence_ppm": minimum_confidence_ppm,
            "minimum_margin_ppm": minimum_margin_ppm,
            "multi_label_threshold_ppm": multi_label_threshold_ppm,
        },
        "version": version,
    }
    profile["profile_digest"] = calibration_profile_digest(profile)
    return profile


def with_decision_calibration(
    manifest: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    *,
    critical: bool = True,
) -> dict[str, Any]:
    """Return a copy of ``manifest`` that carries the calibration extension.

    The extension is listed in ``critical_extensions`` by default, so readers
    that do not implement it refuse the bundle instead of running a decision
    model without its qualified thresholds. The result is validated as a whole
    before it is returned.

    Args:
        manifest: Manifest mapping following ``portable_bundle.schema.json``.
        profiles: Profiles from :func:`build_calibration_profile`.
        critical: Whether readers must implement the extension.

    Returns:
        The new manifest mapping, ready for ``encode_portable_manifest``.

    Raises:
        PortableBundleError: If the manifest itself is invalid.
        DecisionCalibrationError: If the extension is invalid.
    """

    value = copy.deepcopy(dict(manifest))
    extensions = dict(value.get("extensions") or {})
    extensions[DECISION_CALIBRATION_EXTENSION] = {
        "profiles": [copy.deepcopy(dict(item)) for item in profiles],
        "schema_version": DECISION_CALIBRATION_EXTENSION_VERSION,
    }
    value["extensions"] = extensions
    names = [
        item
        for item in value.get("critical_extensions") or []
        if item != DECISION_CALIBRATION_EXTENSION
    ]
    if critical:
        names.append(DECISION_CALIBRATION_EXTENSION)
    value["critical_extensions"] = sorted(names)
    checked = PortableBundleManifest.from_mapping(
        value, understood_extensions=value["critical_extensions"]
    ).to_dict()
    validate_decision_calibration_extension(checked)
    return checked


def load_decision_calibration_schema() -> dict[str, Any]:
    """Load the JSON Schema of the calibration extension value.

    Returns:
        The parsed ``portable_bundle_decision_calibration.schema.json``.
    """

    resource = resources.files(SCHEMA_PACKAGE).joinpath(f"{SCHEMA_NAME}.schema.json")
    with resource.open("r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "CALIBRATION_METHODS",
    "DECISION_CALIBRATION_EXTENSION",
    "DECISION_CALIBRATION_EXTENSION_VERSION",
    "MAX_ISOTONIC_POINTS",
    "MAX_PLATT_PPM",
    "MAX_PROFILES",
    "MAX_PROFILE_RUNTIMES",
    "MAX_TEMPERATURE_PPM",
    "PPM_SCALE",
    "REASON_CODES",
    "SCOPE_KINDS",
    "BundleCalibrationProfile",
    "CalibrationRuntimeBinding",
    "DecisionCalibrationError",
    "DecisionCalibrationRead",
    "DecisionCalibrationReason",
    "DecisionCalibrationRequest",
    "DecisionCalibrationSelection",
    "build_calibration_profile",
    "calibration_profile_digest",
    "calibration_runtime_digest",
    "load_decision_calibration_schema",
    "options_scope_digest",
    "questions_scope_digest",
    "read_decision_calibration",
    "resolve_decision_calibration",
    "validate_decision_calibration_extension",
    "with_decision_calibration",
]
