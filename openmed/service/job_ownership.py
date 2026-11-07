"""Object-level ownership for asynchronous service job records.

Async de-identification jobs, FHIR Bulk Data exports/imports and SMART backend
ingestions record a keyed digest of the submitting principal. Status, manifest,
report, summary and cancel routes then answer only that principal, or one that
holds an explicit administrative scope, and treat every other caller exactly
like a request for an unknown job id. The raw principal identity is never
stored; the digest cannot be mapped back to it without the deployment key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Optional

from openmed.core.config import PROFILE_ENV_VAR

from .auth import AuthPrincipal
from .keyed_digest import (
    DEVELOPMENT_PROFILES,
    KeyedDigest,
    ServiceDigestKey,
    encode_digest_fields,
    resolve_service_digest_key,
)

JOB_OWNER_DIGEST_DOMAIN: Final = "job-owner"
JOBS_ADMIN_SCOPE: Final = "jobs:admin"
BULK_ADMIN_SCOPE: Final = "bulk:admin"


def principal_digest_material(principal: AuthPrincipal) -> bytes:
    """Return the canonical identity bytes digested for job ownership.

    The material binds the credential type, the JWT issuer when present and
    the subject. API-key ids are excluded so rotating a key for the same
    configured principal keeps ownership.

    Args:
        principal: Authenticated principal attached by the auth or mTLS layer.

    Returns:
        Canonical digest material.

    Raises:
        ValueError: If the principal does not carry string identity fields.
    """
    if type(principal.subject) is not str or type(principal.credential_type) is not str:
        raise ValueError("principal identity is malformed")
    issuer: Optional[str] = None
    if principal.credential_type == "jwt" and isinstance(principal.claims, Mapping):
        raw_issuer = principal.claims.get("iss")
        issuer = raw_issuer if type(raw_issuer) is str else None
    return encode_digest_fields(principal.credential_type, issuer, principal.subject)


@dataclass(frozen=True, eq=False)
class JobOwnershipPolicy:
    """Bind service job records to the principal that submitted them.

    Attributes:
        key: Deployment digest key used for owner digests.
        enforced: Whether service authentication is active, so requests
            without a principal must not see owned records.
        allow_unowned: Whether records without an owner digest remain visible
            to any authenticated caller. Only the ``dev`` and ``test`` profiles
            allow this; elsewhere such legacy records are administrator-only.
    """

    key: ServiceDigestKey
    enforced: bool = False
    allow_unowned: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        authentication_enabled: bool,
        environ: Optional[Mapping[str, str]] = None,
    ) -> "JobOwnershipPolicy":
        """Build the policy from service environment settings.

        Args:
            authentication_enabled: Whether API-key/JWT or mTLS authentication
                is enabled for the service.
            environ: Environment mapping; defaults to the process environment.

        Returns:
            The configured ownership policy.

        Raises:
            DigestKeyError: If authentication is enabled outside the ``dev``
                and ``test`` profiles without ``OPENMED_SERVICE_DIGEST_KEY``,
                or if the configured key is invalid.
        """
        env = os.environ if environ is None else environ
        profile = (env.get(PROFILE_ENV_VAR) or "prod").strip()
        return cls(
            key=resolve_service_digest_key(
                profile=profile,
                required=authentication_enabled,
                environ=env,
            ),
            enforced=authentication_enabled,
            allow_unowned=profile in DEVELOPMENT_PROFILES,
        )

    def owner_digest(self, principal: Any) -> Optional[dict[str, str]]:
        """Return the stored owner digest for a submitting principal.

        Args:
            principal: Value of ``request.scope["openmed.auth"]``, if any.

        Returns:
            A JSON-safe keyed digest, or ``None`` when no principal is
            authenticated (authentication disabled).
        """
        if not isinstance(principal, AuthPrincipal):
            return None
        material = principal_digest_material(principal)
        return self.key.digest(JOB_OWNER_DIGEST_DOMAIN, material).to_dict()

    def permits(self, principal: Any, owner: Any, *, admin_scope: str) -> bool:
        """Return whether *principal* may read or cancel a record.

        Args:
            principal: Value of ``request.scope["openmed.auth"]``, if any.
            owner: The record's stored owner digest, or ``None`` if unowned.
            admin_scope: Literal scope that overrides ownership. Wildcard
                grants such as ``*`` or ``jobs:*`` do not count.

        Returns:
            ``True`` for the owner, an administrator, and every caller when
            authentication is disabled.
        """
        if not isinstance(principal, AuthPrincipal):
            return not self.enforced
        if admin_scope in principal.scopes:
            return True
        if owner is None:
            return self.allow_unowned
        try:
            candidate = KeyedDigest.from_dict(owner)
            material = principal_digest_material(principal)
        except ValueError:
            return False
        return self.key.verify(candidate, JOB_OWNER_DIGEST_DOMAIN, material)


__all__ = [
    "BULK_ADMIN_SCOPE",
    "JOBS_ADMIN_SCOPE",
    "JOB_OWNER_DIGEST_DOMAIN",
    "JobOwnershipPolicy",
    "principal_digest_material",
]
