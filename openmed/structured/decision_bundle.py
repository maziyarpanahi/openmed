"""Use decision calibration profiles signed inside portable model bundles.

This module connects the ``org.openmed.decision_calibration`` bundle extension
(:mod:`openmed.core.portable_bundle_calibration`) to the decision runtime. A
caller that loaded a decision model from a verified bundle resolves the
profile for that runtime and request, and passes it to
:func:`openmed.structured.decision.decide`. Every mismatch between the profile
and the loaded model, the request's options or questions, the decision schema,
or a deployment pin fails closed with a controlled reason code.

Decision results stay assistive review artifacts. Nothing here authorizes a
clinical action, opens a network connection, or reads calibration data.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from openmed.core.portable_bundle import PortableBundleVerification
from openmed.core.portable_bundle_calibration import (
    PPM_SCALE,
    BundleCalibrationProfile,
    DecisionCalibrationError,
    DecisionCalibrationReason,
    DecisionCalibrationRequest,
    build_calibration_profile,
    options_scope_digest,
    questions_scope_digest,
    resolve_decision_calibration,
)

from .decision import (
    DECISION_BASE_SCHEMA_VERSION,
    DecisionBackendIdentity,
    DecisionCalibrationProfile,
    DecisionMode,
    DecisionRequest,
)

DEFAULT_DECISION_SCHEMA_MAX_EXCLUSIVE = "2.0.0"
"""Profiles built here stay compatible with the current decision major."""

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_REVISION = re.compile(r"[0-9a-f]{40}")


def decision_profile_from_bundle(
    profile: BundleCalibrationProfile,
) -> DecisionCalibrationProfile:
    """Return the decision-runtime profile for a validated bundle profile.

    Args:
        profile: A profile from ``read_decision_calibration`` or
            ``resolve_decision_calibration``.

    Returns:
        The equivalent :class:`DecisionCalibrationProfile`.
    """

    if not isinstance(profile, BundleCalibrationProfile):
        raise TypeError("profile must be a BundleCalibrationProfile")
    return DecisionCalibrationProfile(
        calibration_id=profile.calibration_id,
        version=profile.version,
        method=profile.method,
        minimum_confidence=profile.minimum_confidence,
        minimum_margin=profile.minimum_margin,
        multi_label_threshold=profile.multi_label_threshold,
        in_domain_digest=profile.in_domain_digest,
        out_of_domain_digest=profile.out_of_domain_digest,
    )


def _ppm(value: float) -> int:
    scaled = round(value * PPM_SCALE)
    if scaled / PPM_SCALE != value:
        raise DecisionCalibrationError(DecisionCalibrationReason.MALFORMED)
    return scaled


def bundle_profile_from_decision(
    profile: DecisionCalibrationProfile,
    manifest: Mapping[str, Any],
    *,
    runtime_ids: Sequence[str],
    scope_kind: str,
    dataset_digest: str,
    scope_digest: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    decision_schema_min: str = DECISION_BASE_SCHEMA_VERSION,
    decision_schema_max_exclusive: str = DEFAULT_DECISION_SCHEMA_MAX_EXCLUSIVE,
    evaluation_path: str | None = None,
) -> dict[str, Any]:
    """Return a bundle extension profile for a qualified decision profile.

    Thresholds must be exact in parts per million, so a qualified threshold
    never changes when it is signed into a bundle.

    Args:
        profile: The qualified decision-runtime profile.
        manifest: Manifest mapping the profile will be embedded in.
        runtime_ids: Runtime variants the thresholds were qualified on.
        scope_kind: ``labels``, ``options``, or ``questions``.
        dataset_digest: Digest of the frozen calibration data, for example the
            ``fixture_digest`` of ``evaluate_decision_calibration``.
        scope_digest: Scope digest; derived from the label map for ``labels``.
        parameters: Integer method parameters for ``profile.method``.
        decision_schema_min: Lowest compatible decision schema version.
        decision_schema_max_exclusive: First incompatible decision schema
            version.
        evaluation_path: Bundled evaluation report path, if any.

    Returns:
        A profile mapping for ``with_decision_calibration``.

    Raises:
        DecisionCalibrationError: ``calibration_malformed`` when a threshold is
            not exact in parts per million.
    """

    return build_calibration_profile(
        manifest,
        calibration_id=profile.calibration_id,
        version=profile.version,
        method=profile.method,
        parameters=parameters,
        runtime_ids=runtime_ids,
        scope_kind=scope_kind,
        scope_digest=scope_digest,
        decision_schema_min=decision_schema_min,
        decision_schema_max_exclusive=decision_schema_max_exclusive,
        dataset_digest=dataset_digest,
        in_domain_digest=profile.in_domain_digest,
        out_of_domain_digest=profile.out_of_domain_digest,
        minimum_confidence_ppm=_ppm(profile.minimum_confidence),
        minimum_margin_ppm=_ppm(profile.minimum_margin),
        multi_label_threshold_ppm=_ppm(profile.multi_label_threshold),
        evaluation_path=evaluation_path,
    )


def decision_request_scope(request: DecisionRequest) -> tuple[str, str]:
    """Return the scope kind and digest a request decides over.

    Question-set requests use the ``questions`` digest of their ordered
    question definitions; every other request uses the ``options`` digest of
    its ordered options.

    Args:
        request: A validated decision request.

    Returns:
        ``(scope_kind, scope_digest)``.
    """

    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest")
    if request.mode is DecisionMode.QUESTION_SET:
        return "questions", questions_scope_digest(
            [item.to_dict() for item in request.questions]
        )
    return "options", options_scope_digest(request.options)


def resolve_bundle_decision_profile(
    manifest_bytes: bytes,
    verification: PortableBundleVerification,
    *,
    request: DecisionRequest,
    runtime_id: str,
    backend: DecisionBackendIdentity | None = None,
    profile_digest: str | None = None,
) -> DecisionCalibrationProfile:
    """Return the signed profile for a loaded runtime and request.

    The request's calibration id, contract version, and options or questions,
    and the backend's model id and revision when given, must all match the
    profile bound to ``runtime_id``.

    Args:
        manifest_bytes: Exact bytes of ``openmed-bundle.json``.
        verification: The verifier result for that bundle, including files.
        request: The decision request about to be decided.
        runtime_id: Runtime variant the backend loaded from the bundle.
        backend: Identity of the loaded backend, if available.
        profile_digest: Profile digest pinned by deployment qualification.

    Returns:
        A :class:`DecisionCalibrationProfile` for ``decide``.

    Raises:
        DecisionCalibrationError: With the first failing reason code.
    """

    if not isinstance(request, DecisionRequest):
        raise TypeError("request must be a DecisionRequest")
    model_id: str | None = None
    model_revision: str | None = None
    if backend is not None:
        if not isinstance(backend, DecisionBackendIdentity):
            raise TypeError("backend must be a DecisionBackendIdentity")
        if (
            _MODEL_ID.fullmatch(backend.model_id) is None
            or _REVISION.fullmatch(backend.revision) is None
        ):
            raise DecisionCalibrationError(DecisionCalibrationReason.MODEL_MISMATCH)
        model_id, model_revision = backend.model_id, backend.revision
    scope_kind, scope_digest = decision_request_scope(request)
    selection = resolve_decision_calibration(
        manifest_bytes,
        verification,
        DecisionCalibrationRequest(
            calibration_id=request.calibration_id,
            runtime_id=runtime_id,
            decision_schema_version=request.contract_version,
            model_id=model_id,
            model_revision=model_revision,
            scope_kind=scope_kind,
            scope_digest=scope_digest,
            profile_digest=profile_digest,
        ),
    )
    if not selection.ok or selection.profile is None:
        raise DecisionCalibrationError(DecisionCalibrationReason(selection.reason))
    return decision_profile_from_bundle(selection.profile)


__all__ = [
    "DEFAULT_DECISION_SCHEMA_MAX_EXCLUSIVE",
    "bundle_profile_from_decision",
    "decision_profile_from_bundle",
    "decision_request_scope",
    "resolve_bundle_decision_profile",
]
