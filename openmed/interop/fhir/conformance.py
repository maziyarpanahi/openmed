"""Privacy-safe result model for opt-in FHIR reference-server probes.

The probe harness keeps URLs, credentials, and resources in memory. Only fixed
case names and reason codes may cross into a report or test failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from itertools import islice
from urllib.parse import urlsplit
from uuid import uuid4

from openmed.interop.fhir.concurrency_guard import (
    VersionEvidence,
    guard_update,
)
from openmed.interop.fhir_capability_preflight import (
    FHIRPreflightStatus,
    FHIRWriteInteraction,
    FHIRWritePlan,
    parse_capability_statement,
    preflight_write_plan,
)
from openmed.interop.fhir_server import FHIRServerClient, FHIRServerConfig

MATRIX_VERSION = "openmed.fhir.reference-server.v1"
CASES = (
    "read_search_pagination",
    "capability_preflight",
    "conditional_write",
    "etag_conflict",
    "transaction_atomicity",
    "batch_partial_failure",
    "subscription_duplicate",
    "token_refresh",
    "scope_narrowing",
    "token_revocation",
)
SERVERS = ("hapi", "medplum", "aidbox")


class Outcome(str, Enum):
    """One bounded compatibility observation."""

    PASS = "pass"
    EXPECTED_VARIANCE = "expected_variance"
    UNSUPPORTED = "unsupported"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class ReferenceProfile:
    """Declared software version and bounded test policy for one server."""

    server: str
    version: str
    base_url: str

    def __post_init__(self) -> None:
        if self.server not in SERVERS:
            raise ValueError("unknown reference server")
        if (
            not isinstance(self.version, str)
            or not self.version
            or len(self.version) > 64
            or not all(
                character.isalnum() or character in ".-_" for character in self.version
            )
        ):
            raise ValueError("invalid reference server version")
        parsed = urlsplit(self.base_url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "[::1]", "::1"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("reference server must use a local HTTP endpoint")

    def __repr__(self) -> str:
        return f"ReferenceProfile(server={self.server!r}, version={self.version!r})"


@dataclass(frozen=True, slots=True)
class ConformanceResult:
    """Result without endpoint, resource, identifier, or credential fields."""

    server: str
    version: str
    case: str
    outcome: Outcome
    reason_code: str

    def __post_init__(self) -> None:
        if self.server not in SERVERS or self.case not in CASES:
            raise ValueError("invalid conformance result")
        if (
            not isinstance(self.version, str)
            or not self.version
            or len(self.version) > 64
            or not all(
                character.isalnum() or character in ".-_" for character in self.version
            )
            or not isinstance(self.outcome, Outcome)
        ):
            raise ValueError("invalid conformance result")
        if self.reason_code not in {
            "observed",
            "server_variance",
            "capability_absent",
            "dependency_pending",
            "profile_disabled",
            "unexpected_response",
        }:
            raise ValueError("invalid conformance reason")

    def to_dict(self) -> dict[str, str]:
        """Return only allowlisted matrix fields."""

        return {
            "matrix_version": MATRIX_VERSION,
            "server": self.server,
            "version": self.version,
            "case": self.case,
            "outcome": self.outcome.value,
            "reason_code": self.reason_code,
        }


def classify(
    profile: ReferenceProfile,
    case: str,
    *,
    supported: bool,
    passed: bool,
    expected_variance: bool = False,
) -> ConformanceResult:
    """Classify a probe without copying its response or exception content."""

    if case not in CASES:
        raise ValueError("unknown conformance case")
    if not supported:
        return ConformanceResult(
            profile.server,
            profile.version,
            case,
            Outcome.UNSUPPORTED,
            "capability_absent",
        )
    if passed:
        return ConformanceResult(
            profile.server, profile.version, case, Outcome.PASS, "observed"
        )
    if not expected_variance:
        return ConformanceResult(
            profile.server, profile.version, case, Outcome.FAIL, "unexpected_response"
        )
    return ConformanceResult(
        profile.server,
        profile.version,
        case,
        Outcome.EXPECTED_VARIANCE,
        "server_variance",
    )


def run_reference_server(
    profile: ReferenceProfile, client: object
) -> tuple[ConformanceResult, ...]:
    """Probe an explicitly supplied local container with synthetic Patients.

    ``client`` is an HTTPX-compatible client configured by the caller. No
    response body, URL, token, or generated resource ID enters the result.
    Cleanup is best effort; discard the isolated container after a run.
    """

    if not isinstance(profile, ReferenceProfile):
        raise TypeError("profile must be ReferenceProfile")
    base = profile.base_url.rstrip("/")
    results: list[ConformanceResult] = []
    ids = [f"openmed-test-{uuid4().hex}" for _ in range(2)]
    created: list[str] = []

    def request(method: str, path: str, **kwargs: object) -> object:
        # The profile's URL is validated as loopback before any request.
        return client.request(method, f"{base}/{path}", **kwargs)  # type: ignore[attr-defined]

    try:
        metadata = request("GET", "metadata")
        statement = metadata.json()  # type: ignore[attr-defined]
        software = statement.get("software", {})
        if (
            metadata.status_code != 200  # type: ignore[attr-defined]
            or statement.get("resourceType") != "CapabilityStatement"
            or statement.get("fhirVersion") not in {"4.0", "4.0.0", "4.0.1"}
            or not isinstance(software, dict)
            or software.get("version") != profile.version
        ):
            return tuple(
                classify(profile, case, supported=True, passed=False) for case in CASES
            )
        preflight = preflight_write_plan(
            statement, FHIRWritePlan(FHIRWriteInteraction.UPDATE, "Patient")
        )
        write_supported = preflight.status is FHIRPreflightStatus.COMPATIBLE
        results.append(
            classify(
                profile,
                "capability_preflight",
                supported=preflight.status is not FHIRPreflightStatus.INCOMPATIBLE,
                passed=write_supported,
            )
        )
        if not write_supported:
            results.extend(
                ConformanceResult(
                    profile.server,
                    profile.version,
                    case,
                    Outcome.UNSUPPORTED,
                    "profile_disabled",
                )
                for case in (
                    "read_search_pagination",
                    "conditional_write",
                    "etag_conflict",
                )
            )
        else:
            for resource_id in ids:
                response = request(
                    "PUT",
                    f"Patient/{resource_id}",
                    json={"resourceType": "Patient", "id": resource_id, "active": True},
                )
                if response.status_code in {200, 201}:  # type: ignore[attr-defined]
                    created.append(resource_id)
            if len(created) == 2:
                read = request("GET", f"Patient/{ids[0]}")
                reader = FHIRServerClient(FHIRServerConfig(base), client=client)
                pages = list(
                    islice(
                        reader.iter_bundle_pages(
                            "Patient", params={"_id": ",".join(ids), "_count": 1}
                        ),
                        3,
                    )
                )
                found = {
                    entry.get("resource", {}).get("id")
                    for page in pages
                    for entry in page.get("entry", [])
                    if isinstance(entry, dict)
                }
                results.append(
                    classify(
                        profile,
                        "read_search_pagination",
                        supported=True,
                        passed=(
                            read.status_code == 200  # type: ignore[attr-defined]
                            and len(pages) == 2
                            and found == set(ids)
                        ),
                    )
                )
                resource = read.json()  # type: ignore[attr-defined]
                try:
                    evidence = VersionEvidence.from_resource(resource)
                    guard_update(evidence, evidence)
                    stale = request(
                        "PUT",
                        f"Patient/{ids[0]}",
                        headers={"If-Match": 'W/"openmed-stale"'},
                        json=resource,
                    )
                    etag_ok = stale.status_code in {409, 412, 428}  # type: ignore[attr-defined]
                except (TypeError, ValueError):
                    etag_ok = False
                results.append(
                    classify(profile, "etag_conflict", supported=True, passed=etag_ok)
                )
            else:
                results.extend(
                    classify(profile, case, supported=True, passed=False)
                    for case in ("read_search_pagination", "etag_conflict")
                )
            conditional = preflight_write_plan(
                statement,
                FHIRWritePlan(FHIRWriteInteraction.CREATE, "Patient", conditional=True),
            )
            if conditional.status is FHIRPreflightStatus.COMPATIBLE:
                conditional_id = f"openmed-test-{uuid4().hex}"
                patient = {
                    "resourceType": "Patient",
                    "active": True,
                    "identifier": [{"system": "urn:uuid", "value": conditional_id}],
                }
                headers = {
                    "If-None-Exist": f"identifier=urn:uuid|{conditional_id}",
                    "Prefer": "return=representation",
                }
                first = request("POST", "Patient", json=patient, headers=headers)
                if first.status_code in {200, 201}:  # type: ignore[attr-defined]
                    try:
                        returned = first.json()  # type: ignore[attr-defined]
                    except (TypeError, ValueError):
                        returned = {}
                    returned_id = (
                        returned.get("id") if isinstance(returned, dict) else None
                    )
                    if not returned_id:
                        location = first.headers.get("Location", "")  # type: ignore[attr-defined]
                        parts = urlsplit(location).path.split("/")
                        if "Patient" in parts:
                            index = parts.index("Patient")
                            if index + 1 < len(parts):
                                returned_id = parts[index + 1]
                    if isinstance(returned_id, str) and re.fullmatch(
                        r"[A-Za-z0-9.-]{1,64}", returned_id
                    ):
                        created.append(returned_id)
                second = request("POST", "Patient", json=patient, headers=headers)
                conditional_ok = (
                    first.status_code == 201  # type: ignore[attr-defined]
                    and second.status_code == 200  # type: ignore[attr-defined]
                )
                results.append(
                    classify(
                        profile,
                        "conditional_write",
                        supported=True,
                        passed=conditional_ok,
                    )
                )
            else:
                results.append(
                    classify(
                        profile, "conditional_write", supported=False, passed=False
                    )
                )
        interactions = parse_capability_statement(statement).system_interactions
        batch_supported = write_supported and "batch" in interactions
        transaction_supported = write_supported and "transaction" in interactions
        if transaction_supported:
            transaction_id = f"openmed-test-{uuid4().hex}"
            created.append(transaction_id)
            transaction_bundle = {
                "resourceType": "Bundle",
                "type": "transaction",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": transaction_id,
                            "active": True,
                        },
                        "request": {
                            "method": "PUT",
                            "url": f"Patient/{transaction_id}",
                        },
                    },
                    {
                        "resource": {"resourceType": "InvalidSyntheticResource"},
                        "request": {
                            "method": "POST",
                            "url": "InvalidSyntheticResource",
                        },
                    },
                ],
            }
            transaction_response = request("POST", "", json=transaction_bundle)
            after = request("GET", f"Patient/{transaction_id}")
            results.append(
                classify(
                    profile,
                    "transaction_atomicity",
                    supported=True,
                    passed=(
                        transaction_response.status_code >= 400  # type: ignore[attr-defined]
                        and after.status_code == 404
                    ),  # type: ignore[attr-defined]
                )
            )
        else:
            results.append(
                classify(
                    profile, "transaction_atomicity", supported=False, passed=False
                )
            )
        if batch_supported:
            # A batch is used here because transaction atomicity must not leave
            # a deliberately failed write committed. Responses are inspected
            # in memory and never included in the report.
            batch_id = f"openmed-test-{uuid4().hex}"
            created.append(batch_id)
            batch = {
                "resourceType": "Bundle",
                "type": "batch",
                "entry": [
                    {
                        "resource": {
                            "resourceType": "Patient",
                            "id": batch_id,
                            "active": True,
                        },
                        "request": {"method": "PUT", "url": f"Patient/{batch_id}"},
                    },
                    {
                        "resource": {"resourceType": "InvalidSyntheticResource"},
                        "request": {
                            "method": "POST",
                            "url": "InvalidSyntheticResource",
                        },
                    },
                ],
            }
            response = request("POST", "", json=batch)
            if response.status_code in {200, 201}:  # type: ignore[attr-defined]
                from openmed.interop.fhir.compensation_report import (
                    build_compensation_report,
                )

                packet = build_compensation_report(batch, response.json())  # type: ignore[attr-defined]
                batch_ok = packet.has_partial_failure
            else:
                batch_ok = False
            results.append(
                classify(
                    profile,
                    "batch_partial_failure",
                    supported=True,
                    passed=batch_ok,
                )
            )
        else:
            results.append(
                classify(
                    profile,
                    "batch_partial_failure",
                    supported=False,
                    passed=False,
                )
            )
    except Exception:
        # HTTP exceptions commonly contain request URLs and Authorization headers.
        return tuple(
            classify(profile, case, supported=True, passed=False) for case in CASES
        )
    finally:
        for resource_id in created:
            try:
                request("DELETE", f"Patient/{resource_id}")
            except Exception:
                pass

    for case in (
        "subscription_duplicate",
        "token_refresh",
        "scope_narrowing",
        "token_revocation",
    ):
        results.append(
            ConformanceResult(
                profile.server,
                profile.version,
                case,
                Outcome.UNSUPPORTED,
                "dependency_pending",
            )
        )
    return tuple(sorted(results, key=lambda item: CASES.index(item.case)))
