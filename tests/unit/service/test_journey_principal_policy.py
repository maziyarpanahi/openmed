"""Principal-bound Journey authorization across REST and GraphQL."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Receive, Scope, Send

from openmed.service import runtime as service_runtime
from openmed.service.app import create_app
from openmed.service.auth import AuthPrincipal, hash_api_key, principal_digest
from openmed.service.journey_access import (
    SERVICE_JOURNEY_GRANTS_ENV_VAR,
    SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR,
    ConfiguredJourneyAccessContextProvider,
    JourneyAccessAuditLog,
    JourneyAccessContext,
    JourneyAccessGate,
    JourneyAccessGrant,
    JourneyAccessRequest,
    bind_journey_access,
    parse_journey_grants,
)
from openmed.service.journey_resources import (
    JourneyAccessPolicy,
    JourneyResourceCatalog,
    JourneyResourceKind,
    JourneyResourceRecord,
    JourneyResourceState,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests" / "fixtures" / "service" / "journey_resources.json"
LOOPBACK_BASE_URL = "http://127.0.0.1"
HS_SECRET = b"synthetic-journey-hs256-secret"
SYNTHETIC_URI_SAN = "uri:spiffe://openmed.test/journey-client"
MTLS_SUBJECT = "synthetic-journey-client"
SYNTHETIC_VALUE = "synthetic.condition.alpha"
API_KEYS = {
    "reader": ("synthetic-reader-key", "svc-reader"),
    "steward": ("synthetic-steward-key", "svc-steward"),
    "withdrawn": ("synthetic-withdrawn-key", "svc-withdrawn"),
    "unknown": ("synthetic-unknown-key", "svc-unknown"),
    "expired": ("synthetic-expired-key", "svc-expired"),
    "ungranted": ("synthetic-ungranted-key", "svc-ungranted"),
    "shadow": ("synthetic-shadow-key", MTLS_SUBJECT),
}
GRANT_DOCUMENT = {
    "version": "1.2.0",
    "jwt_claim": "openmed_journey",
    "grants": [
        {
            "credential_type": "api_key",
            "principal": "svc-reader",
            "roles": ["researcher"],
            "namespaces": ["default"],
            "purposes": ["analytics"],
            "export_policies": ["metadata_only"],
            "consent_state": "active",
        },
        {
            "credential_type": "api_key",
            "principal": "svc-steward",
            "roles": ["data_steward", "privacy_officer"],
            "attributes": ["approved_device"],
            "namespaces": ["default", "ward_a"],
            "purposes": ["care_review", "quality"],
            "export_policies": ["metadata_only", "full_record"],
            "consent_state": "active",
        },
        {
            "credential_type": "api_key",
            "principal": "svc-withdrawn",
            "roles": ["clinician"],
            "namespaces": ["default"],
            "purposes": ["care_review"],
            "export_policies": ["metadata_only"],
            "consent_state": "withdrawn",
        },
        {
            "credential_type": "api_key",
            "principal": "svc-unknown",
            "roles": ["clinician"],
            "namespaces": ["default"],
            "purposes": ["care_review"],
            "export_policies": ["metadata_only"],
        },
        {
            "credential_type": "api_key",
            "principal": "svc-expired",
            "roles": ["clinician"],
            "namespaces": ["default"],
            "purposes": ["care_review"],
            "export_policies": ["metadata_only"],
            "consent_state": "active",
            "expires_at": "2020-01-01T00:00:00Z",
        },
        {
            "credential_type": "api_key",
            "principal": MTLS_SUBJECT,
            "roles": ["privacy_officer"],
            "namespaces": ["default", "ward_a"],
            "purposes": ["care_review"],
            "export_policies": ["metadata_only", "full_record"],
            "consent_state": "active",
        },
        {
            "credential_type": "jwt",
            "principal": "jwt-analyst",
            "roles": ["researcher"],
            "namespaces": ["default"],
            "purposes": ["quality"],
            "export_policies": ["metadata_only"],
            "consent_state": "active",
        },
        {
            "credential_type": "mtls",
            "principal": MTLS_SUBJECT,
            "roles": ["clinician"],
            "namespaces": ["default"],
            "purposes": ["care_review"],
            "export_policies": ["metadata_only"],
            "consent_state": "active",
        },
    ],
}
POLICY = JourneyAccessPolicy(
    allowed_namespaces=frozenset({"default", "ward_a"}),
    allowed_export_policies=frozenset({"metadata_only", "full_record"}),
    required_attributes_by_resource={
        JourneyResourceKind.EVIDENCE: frozenset({"approved_device"})
    },
)
_SERVICE_ENV_VARS = (
    "OPENMED_SERVICE_AUTH_ENABLED",
    "OPENMED_SERVICE_AUTH_DENY_BY_DEFAULT",
    "OPENMED_SERVICE_AUTH_API_KEYS",
    "OPENMED_SERVICE_AUTH_JWKS",
    "OPENMED_SERVICE_AUTH_JWKS_FILE",
    "OPENMED_SERVICE_AUTH_JWT_ISSUER",
    "OPENMED_SERVICE_AUTH_JWT_AUDIENCE",
    "OPENMED_SERVICE_AUTH_JWT_MAX_LIFETIME_SECONDS",
    "OPENMED_SERVICE_AUTH_ROUTE_SCOPES",
    "OPENMED_SERVICE_MTLS_ENABLED",
    "OPENMED_SERVICE_MTLS_CA_BUNDLE",
    "OPENMED_SERVICE_MTLS_CLIENT_CERT_HEADER",
    "OPENMED_SERVICE_MTLS_TRUSTED_PROXIES",
    "OPENMED_SERVICE_MTLS_PRINCIPALS",
    "OPENMED_SERVICE_PRELOAD_MODELS",
    "OPENMED_SERVICE_RATE_LIMIT_RPS",
    "OPENMED_SERVICE_RATE_LIMIT_BURST",
    SERVICE_JOURNEY_GRANTS_ENV_VAR,
    SERVICE_JOURNEY_GRANTS_FILE_ENV_VAR,
)
_GRAPHQL_QUERY = """
query Journey(
  $resourceType: JourneyResourceKind!
  $namespace: String
  $purpose: String
  $role: String
  $attributes: [String!]
  $consentState: String
  $exportPolicy: String
  $first: Int!
  $after: String
  $fields: [String!]
) {
  journeyResources(
    resourceType: $resourceType
    namespace: $namespace
    purpose: $purpose
    role: $role
    attributes: $attributes
    consentState: $consentState
    exportPolicy: $exportPolicy
    first: $first
    after: $after
    fields: $fields
  ) {
    state code schemaVersion compatibilityPolicy
    resources {
      resourceType resourceId namespace data state version revision
      schemaVersion compatibilityPolicy extensions
    }
    pageInfo { hasNextPage endCursor pageSize snapshotDigest }
    policy {
      state namespace purpose role attributes consentState exportPolicy
      decisionId requestDigest allowedFields code policyVersion
    }
  }
}
"""


class FakeLoader:
    """Minimal offline loader for service construction."""

    def __init__(self, config: Any) -> None:
        self.config = config

    def resolve_model_name(self, model_name: str) -> str:
        return model_name

    def loaded_models(self) -> dict[str, Any]:
        return {}


class TLSClientScopeMiddleware:
    """Attach a synthetic ASGI TLS client chain to every HTTP request."""

    def __init__(self, app: ASGIApp, *, client_cert_chain: Sequence[str]) -> None:
        self.app = app
        self.client_cert_chain = tuple(client_cert_chain)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_scope = dict(scope)
        extensions = dict(request_scope.get("extensions", {}))
        extensions["tls"] = {
            "server_cert": None,
            "client_cert_chain": self.client_cert_chain,
            "client_cert_name": None,
            "client_cert_error": None,
        }
        request_scope["extensions"] = extensions
        await self.app(request_scope, receive, send)


@pytest.fixture(autouse=True)
def clean_service_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for env_var in (*_SERVICE_ENV_VARS, "OPENMED_PROFILE"):
        monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(service_runtime, "ModelLoader", FakeLoader)


def _catalog() -> JourneyResourceCatalog:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    records = [JourneyResourceRecord.from_dict(item) for item in payload]
    records.append(
        JourneyResourceRecord(
            JourneyResourceKind.FACT,
            "fact_cccccccccccccccc",
            "ward_a",
            {
                "subject_id": "subject_wardaaaaaaaa",
                "concept": "synthetic.condition.ward",
                "assertion": "present",
            },
        )
    )
    return JourneyResourceCatalog(records)


def _b64url(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _jwt(claims: dict[str, Any]) -> str:
    header = {"alg": "HS256", "kid": "journey-hs", "typ": "JWT"}
    segments = [
        _b64url(json.dumps(item, sort_keys=True).encode()) for item in (header, claims)
    ]
    signing_input = ".".join(segments)
    signature = hmac.new(HS_SECRET, signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _jwt_claims(subject: str, **extra: Any) -> dict[str, Any]:
    now = int(time.time())
    return {
        "sub": subject,
        "scope": "journey:read",
        "iat": now,
        "exp": now + 600,
        **extra,
    }


def _configure_auth(
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str = "prod",
    grants: Any = GRANT_DOCUMENT,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", profile)
    monkeypatch.setenv("OPENMED_SERVICE_AUTH_ENABLED", "true")
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_API_KEYS",
        json.dumps(
            [
                {
                    "id": f"key-{name}",
                    "key_hash": hash_api_key(raw),
                    "principal": subject,
                    "scopes": ["journey:read"],
                }
                for name, (raw, subject) in API_KEYS.items()
            ]
        ),
    )
    monkeypatch.setenv(
        "OPENMED_SERVICE_AUTH_JWKS",
        json.dumps(
            {
                "keys": [
                    {
                        "kty": "oct",
                        "kid": "journey-hs",
                        "alg": "HS256",
                        "k": _b64url(HS_SECRET),
                    }
                ]
            }
        ),
    )
    if grants is not None:
        monkeypatch.setenv(SERVICE_JOURNEY_GRANTS_ENV_VAR, json.dumps(grants))


def _client(app: Any) -> TestClient:
    return TestClient(app, base_url=LOOPBACK_BASE_URL, raise_server_exceptions=False)


@pytest.fixture
def auth_app(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    _configure_auth(monkeypatch)
    app = create_app()
    app.state.journey_resources = _catalog()
    app.state.journey_access_policy = POLICY
    with _client(app) as client:
        yield client


def _api_key(name: str) -> dict[str, str]:
    return {"X-API-Key": API_KEYS[name][0]}


def _bearer(claims: dict[str, Any]) -> dict[str, str]:
    return {"Authorization": f"Bearer {_jwt(claims)}"}


def _rest_page(
    client: TestClient,
    headers: dict[str, str],
    params: dict[str, Any],
) -> dict[str, Any]:
    response = client.get("/v1/journey/resources", params=params, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


def _graphql_page(
    client: TestClient,
    headers: dict[str, str],
    params: dict[str, Any],
) -> dict[str, Any]:
    variables = {
        "resourceType": str(params["resource_type"]).upper(),
        "namespace": params.get("namespace"),
        "purpose": params.get("purpose"),
        "role": params.get("role"),
        "attributes": (
            None
            if "attributes" not in params
            else [item for item in str(params["attributes"]).split(",") if item]
        ),
        "consentState": params.get("consent_state"),
        "exportPolicy": params.get("export_policy"),
        "first": int(params.get("first", 20)),
        "after": params.get("after"),
        "fields": (
            None if "fields" not in params else str(params["fields"]).split(",")
        ),
    }
    response = client.post(
        "/graphql",
        json={"query": _GRAPHQL_QUERY, "variables": variables},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "errors" not in body, body
    return _snake(body["data"]["journeyResources"])


def _snake(value: Any) -> Any:
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            name = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()
            converted[name] = item if name in {"data", "extensions"} else _snake(item)
        return converted
    if isinstance(value, list):
        return [_snake(item) for item in value]
    return value


def _assert_denied(page: dict[str, Any], code: str) -> None:
    assert page["state"] == "denied"
    assert page["code"] == code
    assert page["resources"] == []
    assert page["policy"]["state"] == "denied"
    assert page["policy"]["allowed_fields"] == []
    assert SYNTHETIC_VALUE not in json.dumps(page)


def test_api_key_principal_receives_only_its_authoritative_context(
    auth_app: TestClient,
) -> None:
    page = _rest_page(auth_app, _api_key("reader"), {"resource_type": "fact"})

    assert page["state"] == "success"
    assert len(page["resources"]) == 2
    assert {
        key: page["policy"][key]
        for key in ("namespace", "purpose", "role", "consent_state", "export_policy")
    } == {
        "namespace": "default",
        "purpose": "analytics",
        "role": "researcher",
        "consent_state": "active",
        "export_policy": "metadata_only",
    }


@pytest.mark.parametrize("transport", ["rest", "graphql"])
@pytest.mark.parametrize(
    ("principal", "params", "code"),
    [
        ("reader", {"role": "privacy_officer"}, "role_not_granted"),
        ("reader", {"role": "clinician"}, "role_not_granted"),
        ("reader", {"namespace": "ward_a"}, "namespace_not_granted"),
        ("reader", {"purpose": "care_review"}, "purpose_not_granted"),
        ("reader", {"export_policy": "full_record"}, "export_policy_not_granted"),
        ("reader", {"attributes": "approved_device"}, "attribute_not_granted"),
        ("withdrawn", {"consent_state": "active"}, "consent_withdrawn"),
        ("unknown", {"consent_state": "active"}, "consent_unknown"),
        ("ungranted", {"role": "clinician"}, "access_context_missing"),
        ("expired", {}, "access_context_expired"),
        ("steward", {}, "role_selection_required"),
    ],
)
def test_request_parameters_cannot_grant_authority(
    auth_app: TestClient,
    transport: str,
    principal: str,
    params: dict[str, str],
    code: str,
) -> None:
    query = {"resource_type": "fact", **params}
    reader = _rest_page if transport == "rest" else _graphql_page

    _assert_denied(reader(auth_app, _api_key(principal), query), code)


def test_narrower_requests_remain_supported(auth_app: TestClient) -> None:
    steward = _api_key("steward")

    narrowed = _rest_page(
        auth_app,
        steward,
        {
            "resource_type": "fact",
            "role": "data_steward",
            "namespace": "ward_a",
            "purpose": "quality",
            "export_policy": "metadata_only",
            "attributes": "",
            "fields": "subject_id",
        },
    )
    assert narrowed["state"] == "success"
    assert [item["data"] for item in narrowed["resources"]] == [
        {"subject_id": "subject_wardaaaaaaaa"}
    ]
    assert narrowed["policy"]["attributes"] == []

    evidence = _rest_page(
        auth_app, steward, {"resource_type": "evidence", "role": "data_steward"}
    )
    assert evidence["state"] == "success"
    assert evidence["policy"]["attributes"] == ["approved_device"]

    dropped = _rest_page(
        auth_app,
        steward,
        {"resource_type": "evidence", "role": "data_steward", "attributes": ""},
    )
    _assert_denied(dropped, "attribute_denied")

    withdrawn = _rest_page(
        auth_app,
        steward,
        {"resource_type": "fact", "role": "data_steward", "consent_state": "withdrawn"},
    )
    _assert_denied(withdrawn, "consent_withdrawn")


@pytest.mark.parametrize(
    ("headers", "params"),
    [
        (_api_key("reader"), {"resource_type": "fact", "first": 1}),
        (
            _api_key("steward"),
            {
                "resource_type": "fact",
                "role": "privacy_officer",
                "purpose": "quality",
                "fields": "concept,subject_id",
            },
        ),
        (_api_key("reader"), {"resource_type": "fact", "role": "clinician"}),
        (_api_key("withdrawn"), {"resource_type": "fact"}),
        (_api_key("ungranted"), {"resource_type": "fact"}),
    ],
)
def test_rest_and_graphql_reach_the_same_decision(
    auth_app: TestClient,
    headers: dict[str, str],
    params: dict[str, Any],
) -> None:
    rest = _rest_page(auth_app, headers, params)
    graph = _graphql_page(auth_app, headers, params)

    assert graph == rest


def test_jwt_principals_bind_to_grants_and_signed_claims(
    auth_app: TestClient,
) -> None:
    analyst = _bearer(_jwt_claims("jwt-analyst"))
    granted = _rest_page(auth_app, analyst, {"resource_type": "fact"})
    assert granted["state"] == "success"
    assert granted["policy"]["purpose"] == "quality"
    assert _graphql_page(auth_app, analyst, {"resource_type": "fact"}) == granted
    _assert_denied(
        _rest_page(auth_app, analyst, {"resource_type": "fact", "role": "clinician"}),
        "role_not_granted",
    )

    claimed = _bearer(
        _jwt_claims(
            "jwt-claimed",
            openmed_journey={
                "roles": ["clinician"],
                "namespaces": ["default"],
                "purposes": ["care_review"],
                "export_policies": ["metadata_only"],
                "consent_state": "active",
            },
        )
    )
    claim_page = _rest_page(auth_app, claimed, {"resource_type": "fact"})
    assert claim_page["state"] == "success"
    assert auth_app.app.state.journey_access.audit.records[-1].context_source == (
        "jwt_claim"
    )
    _assert_denied(
        _rest_page(auth_app, claimed, {"resource_type": "fact", "role": "researcher"}),
        "role_not_granted",
    )

    unclaimed = _bearer(_jwt_claims("jwt-unclaimed", role="privacy_officer"))
    _assert_denied(
        _rest_page(auth_app, unclaimed, {"resource_type": "fact"}),
        "access_context_missing",
    )


def test_jwt_claims_are_not_authority_unless_the_operator_opts_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {**GRANT_DOCUMENT, "jwt_claim": None}
    _configure_auth(monkeypatch, grants=document)
    app = create_app()
    app.state.journey_resources = _catalog()
    claimed = _bearer(
        _jwt_claims(
            "jwt-claimed",
            openmed_journey={
                "roles": ["clinician"],
                "namespaces": ["default"],
                "purposes": ["care_review"],
                "export_policies": ["metadata_only"],
                "consent_state": "active",
            },
        )
    )
    with _client(app) as client:
        _assert_denied(
            _rest_page(client, claimed, {"resource_type": "fact"}),
            "access_context_missing",
        )


def test_jwt_context_never_outlives_the_token() -> None:
    provider = ConfiguredJourneyAccessContextProvider(
        jwt_claim="openmed_journey",
    )
    principal = AuthPrincipal(
        subject="jwt-claimed",
        credential_type="jwt",
        claims={
            "exp": 1_000.0,
            "openmed_journey": {
                "roles": ["clinician"],
                "namespaces": ["default"],
                "purposes": ["care_review"],
                "export_policies": ["metadata_only"],
                "consent_state": "active",
                "expires_at": 5_000,
            },
        },
    )
    context = provider.resolve(principal, policy=POLICY)
    assert context is not None
    assert context.expires_at == 1_000.0
    request = JourneyAccessRequest(resource_type=JourneyResourceKind.FACT)
    assert bind_journey_access(request, context, now=999.0).code is None
    assert (
        bind_journey_access(request, context, now=1_000.0).code
        == "access_context_expired"
    )

    malformed = AuthPrincipal(
        subject="jwt-claimed",
        credential_type="jwt",
        claims={"openmed_journey": {"roles": "clinician", "admin": True}},
    )
    assert provider.resolve(malformed, policy=POLICY) is None


@pytest.fixture(scope="module")
def mtls_material(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    now = datetime.now(timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "OpenMed Synthetic Journey CA")]
    )
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=30))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    leaf_key = ec.generate_private_key(ec.SECP256R1())
    leaf = (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "journey-client")])
        )
        .issuer_name(ca.subject)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(hours=1))
        .not_valid_after(now + timedelta(days=7))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]), critical=False
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.UniformResourceIdentifier("spiffe://openmed.test/journey-client")]
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(leaf_key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    bundle = tmp_path_factory.mktemp("journey-mtls") / "ca.pem"
    bundle.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    return bundle, leaf.public_bytes(serialization.Encoding.PEM).decode("ascii")


def test_mtls_principal_cannot_borrow_a_same_named_api_key_grant(
    monkeypatch: pytest.MonkeyPatch,
    mtls_material: tuple[Path, str],
) -> None:
    bundle, leaf_pem = mtls_material
    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    monkeypatch.setenv("OPENMED_SERVICE_MTLS_ENABLED", "true")
    monkeypatch.setenv("OPENMED_SERVICE_MTLS_CA_BUNDLE", str(bundle))
    monkeypatch.setenv(
        "OPENMED_SERVICE_MTLS_PRINCIPALS",
        json.dumps(
            [
                {
                    "identities": [SYNTHETIC_URI_SAN],
                    "principal": MTLS_SUBJECT,
                    "scopes": ["journey:read"],
                }
            ]
        ),
    )
    monkeypatch.setenv(SERVICE_JOURNEY_GRANTS_ENV_VAR, json.dumps(GRANT_DOCUMENT))
    app = create_app()
    app.state.journey_resources = _catalog()
    app.state.journey_access_policy = POLICY
    wrapped = TLSClientScopeMiddleware(app, client_cert_chain=[leaf_pem])

    with TestClient(wrapped, base_url="https://127.0.0.1") as client:
        page = _rest_page(client, {}, {"resource_type": "fact"})
        assert page["state"] == "success"
        assert page["policy"]["role"] == "clinician"
        assert _graphql_page(client, {}, {"resource_type": "fact"}) == page
        for params, code in (
            ({"role": "privacy_officer"}, "role_not_granted"),
            ({"namespace": "ward_a"}, "namespace_not_granted"),
            ({"export_policy": "full_record"}, "export_policy_not_granted"),
        ):
            query = {"resource_type": "fact", **params}
            _assert_denied(_rest_page(client, {}, query), code)
            _assert_denied(_graphql_page(client, {}, query), code)

    record = app.state.journey_access.audit.records[-1]
    assert record.credential_type == "mtls"
    assert record.principal_digest is not None


@pytest.mark.parametrize("profile", [None, "prod", "fast", "low_resource"])
def test_anonymous_production_requests_receive_no_journey_records(
    monkeypatch: pytest.MonkeyPatch,
    profile: Optional[str],
) -> None:
    if profile is not None:
        monkeypatch.setenv("OPENMED_PROFILE", profile)
    app = create_app()
    app.state.journey_resources = _catalog()

    with _client(app) as client:
        rest = _rest_page(client, {}, {"resource_type": "fact", "role": "clinician"})
        graph = _graphql_page(client, {}, {"resource_type": "fact"})

    _assert_denied(rest, "anonymous_access_denied")
    _assert_denied(graph, "anonymous_access_denied")
    assert all(
        record.credential_type == "anonymous" and record.principal_digest is None
        for record in app.state.journey_access.audit.records
    )


def test_only_named_development_profiles_allow_anonymous_resolution() -> None:
    provider = ConfiguredJourneyAccessContextProvider(anonymous_development=True)
    for profile, allowed in (
        ("dev", True),
        ("test", True),
        ("prod", False),
        ("staging", False),
        ("Dev", False),
    ):
        gate = JourneyAccessGate(provider, profile=profile)
        assert gate.anonymous_allowed is allowed
        page = gate.list_resources(
            _catalog(),
            JourneyAccessRequest(resource_type=JourneyResourceKind.FACT),
            principal=None,
            transport="python",
        )
        expected = (
            JourneyResourceState.SUCCESS if allowed else (JourneyResourceState.DENIED)
        )
        assert page.state is expected
    assert JourneyAccessGate.from_env(environ={}).profile == "prod"


def test_anonymous_production_denial_never_consults_the_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    class PermissiveProvider:
        def resolve(self, principal: Any, *, policy: JourneyAccessPolicy) -> Any:
            calls.append(principal)
            return JourneyAccessContext(
                credential_type="anonymous",
                source="operator_hook",
                roles=policy.allowed_roles,
                namespaces=policy.allowed_namespaces,
                purposes=policy.allowed_purposes,
                export_policies=policy.allowed_export_policies,
                consent_state="active",
            )

    monkeypatch.setenv("OPENMED_PROFILE", "prod")
    app = create_app(journey_access_context_provider=PermissiveProvider())
    app.state.journey_resources = _catalog()
    with _client(app) as client:
        _assert_denied(
            _rest_page(client, {}, {"resource_type": "fact"}),
            "anonymous_access_denied",
        )
    assert calls == []


def test_development_profiles_keep_an_anonymous_policy_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    app = create_app()
    app.state.journey_resources = _catalog()
    with _client(app) as client:
        page = _rest_page(client, {}, {"resource_type": "fact"})
        wider = _rest_page(client, {}, {"resource_type": "fact", "role": "guest"})

    assert page["state"] == "success"
    assert page["policy"]["role"] == "clinician"
    _assert_denied(wider, "role_not_granted")
    assert app.state.journey_access.audit.records[0].context_source == (
        "anonymous_development"
    )


def test_operator_hook_provider_is_the_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StewardProvider:
        def resolve(self, principal: Any, *, policy: JourneyAccessPolicy) -> Any:
            if principal is None:
                return None
            return JourneyAccessContext(
                credential_type=principal.credential_type,
                source="operator_hook",
                roles=frozenset({"data_steward"}),
                namespaces=frozenset({"default"}),
                purposes=frozenset({"quality"}),
                export_policies=frozenset({"metadata_only"}),
                consent_state="active",
            )

    _configure_auth(monkeypatch, grants=None)
    app = create_app(journey_access_context_provider=StewardProvider())
    app.state.journey_resources = _catalog()
    with _client(app) as client:
        page = _rest_page(client, _api_key("ungranted"), {"resource_type": "fact"})
    assert page["state"] == "success"
    assert page["policy"]["role"] == "data_steward"


def test_cursor_from_one_principal_cannot_be_replayed_by_another(
    auth_app: TestClient,
) -> None:
    steward = _api_key("steward")
    first = _rest_page(
        auth_app,
        steward,
        {"resource_type": "fact", "role": "privacy_officer", "first": 1},
    )
    cursor = first["page_info"]["end_cursor"]
    assert cursor is not None
    second = _rest_page(
        auth_app,
        steward,
        {
            "resource_type": "fact",
            "role": "privacy_officer",
            "first": 1,
            "after": cursor,
        },
    )
    assert second["state"] == "success"

    replayed = _rest_page(
        auth_app,
        _api_key("reader"),
        {"resource_type": "fact", "first": 1, "after": cursor},
    )
    assert replayed["state"] == "failure"
    assert replayed["code"] == "cursor_query_mismatch"
    assert replayed["resources"] == []


def test_decisions_are_audited_with_opaque_principal_digests(
    auth_app: TestClient,
) -> None:
    gate = auth_app.app.state.journey_access
    reader = _rest_page(auth_app, _api_key("reader"), {"resource_type": "fact"})
    _graphql_page(auth_app, _api_key("reader"), {"resource_type": "fact"})
    denied = _rest_page(
        auth_app, _api_key("steward"), {"resource_type": "fact", "role": "guest"}
    )

    first, second, third = gate.audit.records[-3:]
    assert (first.transport, second.transport, third.transport) == (
        "rest",
        "graphql",
        "rest",
    )
    assert first.decision_id == second.decision_id == reader["policy"]["decision_id"]
    assert third.decision_id == denied["policy"]["decision_id"]
    assert first.principal_digest == second.principal_digest
    assert first.principal_digest != third.principal_digest
    assert re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", first.principal_digest or "")
    assert first.credential_type == "api_key"
    assert first.context_source == "operator_grant"
    assert first.context_version == "1.2.0"
    assert first.policy_version == POLICY.policy_version
    assert third.decision_state == "denied"
    assert third.code == "role_not_granted"

    serialized = json.dumps([record.to_dict() for record in gate.audit.records])
    for raw_key, subject in API_KEYS.values():
        assert raw_key not in serialized
        assert subject not in serialized
    assert "svc-reader" not in json.dumps(reader)


def test_audit_sink_failure_withholds_records() -> None:
    def failing_sink(record: Any) -> None:
        raise RuntimeError("sink offline")

    gate = JourneyAccessGate(
        ConfiguredJourneyAccessContextProvider(anonymous_development=True),
        profile="test",
        audit=JourneyAccessAuditLog(sink=failing_sink),
    )
    page = gate.list_resources(
        _catalog(),
        JourneyAccessRequest(resource_type=JourneyResourceKind.FACT),
        principal=None,
        transport="python",
    )

    assert page.state is JourneyResourceState.DENIED
    assert page.code == "audit_unavailable"
    assert page.resources == ()


def test_provider_failures_and_mismatched_contexts_fail_closed() -> None:
    principal = AuthPrincipal(subject="svc-reader", credential_type="api_key")
    request = JourneyAccessRequest(resource_type=JourneyResourceKind.FACT)

    class FailingProvider:
        def resolve(self, principal: Any, *, policy: JourneyAccessPolicy) -> Any:
            raise RuntimeError("registry offline")

    class MismatchedProvider:
        def resolve(self, principal: Any, *, policy: JourneyAccessPolicy) -> Any:
            return JourneyAccessContext(
                credential_type="mtls",
                source="operator_hook",
                roles=frozenset({"clinician"}),
                namespaces=frozenset({"default"}),
                purposes=frozenset({"care_review"}),
                export_policies=frozenset({"metadata_only"}),
                consent_state="active",
            )

    class UntypedProvider:
        def resolve(self, principal: Any, *, policy: JourneyAccessPolicy) -> Any:
            return {"roles": ["clinician"], "consent_state": "active"}

    for provider, code in (
        (FailingProvider(), "access_context_unavailable"),
        (MismatchedProvider(), "access_context_mismatch"),
        (UntypedProvider(), "access_context_missing"),
    ):
        gate = JourneyAccessGate(provider, profile="prod")
        page = gate.list_resources(
            _catalog(), request, principal=principal, transport="python"
        )
        assert page.state is JourneyResourceState.DENIED
        assert page.code == code
        assert page.resources == ()


def test_consent_hook_is_authoritative_and_fails_closed() -> None:
    grant = JourneyAccessGrant(
        credential_type="api_key",
        principal="svc-reader",
        roles=frozenset({"researcher"}),
        namespaces=frozenset({"default"}),
        purposes=frozenset({"analytics"}),
        export_policies=frozenset({"metadata_only"}),
        consent_state="active",
    )
    principal = AuthPrincipal(subject="svc-reader", credential_type="api_key")
    request = JourneyAccessRequest(resource_type=JourneyResourceKind.FACT)
    seen: list[Any] = []

    def recording(scope: Any) -> str:
        seen.append(scope)
        return "active"

    def raising(scope: Any) -> str:
        raise LookupError("registry offline")

    for resolver, state, code in (
        (recording, JourneyResourceState.SUCCESS, None),
        (lambda scope: "withdrawn", JourneyResourceState.DENIED, "consent_withdrawn"),
        (lambda scope: None, JourneyResourceState.DENIED, "consent_unknown"),
        (lambda scope: "revoked", JourneyResourceState.DENIED, "consent_unknown"),
        (raising, JourneyResourceState.DENIED, "consent_unknown"),
    ):
        provider = ConfiguredJourneyAccessContextProvider(
            (grant,), consent_resolver=resolver
        )
        page = JourneyAccessGate(provider, profile="prod").list_resources(
            _catalog(), request, principal=principal, transport="python"
        )
        assert page.state is state
        assert page.code == code
    assert seen[0].namespaces == frozenset({"default"})
    assert seen[0].purposes == frozenset({"analytics"})
    assert not hasattr(seen[0], "principal")


def test_context_expiry_uses_the_gate_clock() -> None:
    grant = JourneyAccessGrant(
        credential_type="api_key",
        principal="svc-reader",
        roles=frozenset({"researcher"}),
        namespaces=frozenset({"default"}),
        purposes=frozenset({"analytics"}),
        export_policies=frozenset({"metadata_only"}),
        consent_state="active",
        expires_at=2_000,
    )
    principal = AuthPrincipal(subject="svc-reader", credential_type="api_key")
    request = JourneyAccessRequest(resource_type=JourneyResourceKind.FACT)
    provider = ConfiguredJourneyAccessContextProvider((grant,))

    for now, state in (
        (1_999.0, JourneyResourceState.SUCCESS),
        (2_000.0, JourneyResourceState.DENIED),
    ):
        gate = JourneyAccessGate(provider, profile="prod", clock=lambda: now)
        page = gate.list_resources(
            _catalog(), request, principal=principal, transport="python"
        )
        assert page.state is state


def test_selection_rules_prefer_defaults_then_single_grants() -> None:
    context = JourneyAccessContext(
        credential_type="api_key",
        source="operator_grant",
        roles=frozenset({"clinician", "researcher"}),
        namespaces=frozenset({"ward_a"}),
        purposes=frozenset({"analytics", "quality"}),
        export_policies=frozenset({"metadata_only"}),
        consent_state="active",
    )
    request = JourneyAccessRequest(resource_type=JourneyResourceKind.FACT)
    binding = bind_journey_access(request, context, now=0.0)
    assert binding.code == "purpose_selection_required"
    assert binding.query.role == "clinician"
    assert binding.query.namespace == "ward_a"

    chosen = bind_journey_access(
        JourneyAccessRequest(resource_type=JourneyResourceKind.FACT, purpose="quality"),
        context,
        now=0.0,
    )
    assert chosen.code is None
    assert chosen.query.purpose == "quality"

    narrowed = bind_journey_access(
        JourneyAccessRequest(
            resource_type=JourneyResourceKind.FACT,
            purpose="quality",
            consent_state="unknown",
        ),
        context,
        now=0.0,
    )
    assert narrowed.query.consent_state == "unknown"
    assert bind_journey_access(request, None, now=0.0).code == (
        "access_context_missing"
    )


def test_principal_digest_is_keyed_and_identity_free() -> None:
    principal = AuthPrincipal(subject="svc-reader", credential_type="api_key")
    key_a = b"a" * 32
    key_b = b"b" * 32

    digest = principal_digest(principal, key=key_a)
    assert digest == principal_digest(principal, key=key_a)
    assert digest != principal_digest(principal, key=key_b)
    assert digest != principal_digest(
        AuthPrincipal(subject="svc-reader", credential_type="jwt"), key=key_a
    )
    assert "svc-reader" not in digest
    with pytest.raises(ValueError):
        principal_digest(principal, key=b"short")


def test_grant_document_parsing_is_strict_and_value_free(tmp_path: Path) -> None:
    secret_subject = "svc-subject-should-not-leak"
    base = {
        "credential_type": "api_key",
        "principal": secret_subject,
        "roles": ["clinician"],
        "consent_state": "active",
    }
    invalid_documents = [
        {"grants": [{**base, "credential_type": "password"}]},
        {"grants": [{**base, "roles": "clinician"}]},
        {"grants": [{**base, "roles": ["Clinician Admin"]}]},
        {"grants": [{**base, "consent_state": "granted"}]},
        {"grants": [{**base, "expires_at": "2027-01-01T00:00:00"}]},
        {"grants": [{**base, "superuser": True}]},
        {"grants": [base, dict(base)]},
        {"grants": [base], "trust_everyone": True},
        {"version": "v1", "grants": [base]},
        {"jwt_claim": "Bad Claim", "grants": [base]},
    ]
    for document in invalid_documents:
        with pytest.raises(ValueError) as excinfo:
            parse_journey_grants(json.dumps(document))
        assert secret_subject not in str(excinfo.value)
        assert excinfo.value.__cause__ is None
    for raw in ("not json", '{"grants": [{"expires_at": NaN}]}'):
        with pytest.raises(ValueError, match="valid JSON"):
            parse_journey_grants(raw)

    grant_file = tmp_path / "journey-grants.json"
    grant_file.write_text(json.dumps([base]), encoding="utf-8")
    config = parse_journey_grants(None, str(grant_file))
    assert len(config.grants) == 1
    assert config.grants[0].consent_state == "active"
    assert secret_subject not in repr(config)
    with pytest.raises(ValueError, match="only one"):
        parse_journey_grants(json.dumps([base]), str(grant_file))
    with pytest.raises(ValueError, match="readable"):
        parse_journey_grants(None, str(tmp_path / "missing.json"))
    assert parse_journey_grants(None, None).grants == ()


def test_invalid_grant_configuration_fails_service_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SERVICE_JOURNEY_GRANTS_ENV_VAR, '{"grants": [{"roles": 1}]}')
    with pytest.raises(ValueError, match=SERVICE_JOURNEY_GRANTS_ENV_VAR):
        create_app()


def test_omitted_consent_defaults_to_unknown() -> None:
    grant = JourneyAccessGrant(
        credential_type="mtls",
        principal="mtls:" + "0" * 64,
        roles=frozenset({"clinician"}),
        namespaces=frozenset({"default"}),
        purposes=frozenset({"care_review"}),
        export_policies=frozenset({"metadata_only"}),
    )
    assert grant.consent_state == "unknown"
    assert "mtls:" not in repr(grant)
