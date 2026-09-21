"""Explicit network boundary for public clinical-trial study synchronization."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from openmed.clinical.journey_contracts import canonical_digest

from .contracts import (
    TrialContractError,
    TrialIntervention,
    TrialLocation,
    TrialSchemaDriftError,
    TrialSourceUnavailableError,
    TrialStudyRecord,
)

OFFICIAL_TRIAL_API_URL: Final = "https://clinicaltrials.gov/api/v2/studies"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


@runtime_checkable
class TrialSourceTransport(Protocol):
    """Injected byte transport for the opt-in public registry boundary."""

    def fetch(self, url: str, *, timeout_seconds: float) -> bytes:
        """Fetch one public metadata page."""


class UrlLibTrialSourceTransport:
    """Standard-library transport used only after an explicit fetch call."""

    def fetch(self, url: str, *, timeout_seconds: float) -> bytes:
        """Fetch one public JSON page without sending caller data."""

        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return response.read()


@dataclass(frozen=True, slots=True)
class TrialSourcePage:
    """Parsed public studies plus opaque registry pagination custody."""

    studies: tuple[TrialStudyRecord, ...]
    retrieved_at: str
    response_digest: str
    next_page_token: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(item, TrialStudyRecord) for item in self.studies):
            raise TypeError("studies must contain TrialStudyRecord")
        if len({item.study_id for item in self.studies}) != len(self.studies):
            raise TrialContractError("source page contains duplicate study identifiers")
        if (
            not isinstance(self.response_digest, str)
            or _DIGEST_RE.fullmatch(self.response_digest) is None
        ):
            raise TrialContractError(
                "response_digest must be a normalized SHA-256 digest"
            )
        if self.studies and any(
            item.retrieved_at != self.retrieved_at for item in self.studies
        ):
            raise TrialContractError("source page retrieval custody is inconsistent")
        if not self.studies:
            _validate_empty_page_timestamp(self.retrieved_at)
        object.__setattr__(
            self, "studies", tuple(sorted(self.studies, key=lambda x: x.study_id))
        )


class ClinicalTrialSource:
    """Official public study source with no patient-shaped request fields."""

    def __init__(
        self,
        *,
        transport: TrialSourceTransport | None = None,
        base_url: str = OFFICIAL_TRIAL_API_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not base_url.startswith("https://"):
            raise TrialContractError("trial source base_url must use HTTPS")
        if timeout_seconds <= 0 or timeout_seconds > 120:
            raise TrialContractError("timeout_seconds must be in (0, 120]")
        self._transport = transport or UrlLibTrialSourceTransport()
        self._base_url = base_url
        self._timeout_seconds = float(timeout_seconds)

    def fetch_page(
        self,
        *,
        retrieved_at: str,
        page_token: str | None = None,
        page_size: int = 100,
    ) -> TrialSourcePage:
        """Explicitly fetch one metadata-only page from the public registry."""

        if type(page_size) is not int or not 1 <= page_size <= 1_000:
            raise TrialContractError("page_size must be between 1 and 1000")
        params = {"format": "json", "pageSize": str(page_size)}
        if page_token is not None:
            if not isinstance(page_token, str) or not page_token:
                raise TrialContractError("page_token must be non-empty text")
            params["pageToken"] = page_token
        url = f"{self._base_url}?{urlencode(params)}"
        try:
            payload = self._transport.fetch(url, timeout_seconds=self._timeout_seconds)
        except OSError as exc:
            raise TrialSourceUnavailableError(
                "public trial metadata fetch failed"
            ) from exc
        return parse_trial_source_page(payload, retrieved_at=retrieved_at)


def parse_trial_source_page(payload: bytes, *, retrieved_at: str) -> TrialSourcePage:
    """Parse a frozen official API response into deterministic study records."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrialSchemaDriftError("trial source response is not valid JSON") from exc
    root = _source_mapping(decoded, "trial source response")
    allowed = {"studies", "nextPageToken", "totalCount"}
    if "studies" not in root or any(key not in allowed for key in root):
        raise TrialSchemaDriftError("trial source response fields changed")
    studies_raw = _source_sequence(root["studies"], "studies")
    studies = tuple(
        _parse_study(_source_mapping(item, "study"), retrieved_at=retrieved_at)
        for item in studies_raw
    )
    token = root.get("nextPageToken")
    if token is not None and (not isinstance(token, str) or not token):
        raise TrialSchemaDriftError("nextPageToken changed type")
    return TrialSourcePage(
        studies=studies,
        retrieved_at=retrieved_at,
        response_digest=canonical_digest(decoded),
        next_page_token=token,
    )


def _parse_study(study: Mapping[str, Any], *, retrieved_at: str) -> TrialStudyRecord:
    protocol = _required_mapping(study, "protocolSection", "study")
    identification = _required_mapping(
        protocol, "identificationModule", "protocolSection"
    )
    status = _required_mapping(protocol, "statusModule", "protocolSection")
    study_id = _required_text(identification, "nctId", "identificationModule")
    brief_title = _required_text(identification, "briefTitle", "identificationModule")
    official_title = _optional_source_text(
        identification.get("officialTitle"), "officialTitle"
    )
    overall_status = _required_text(status, "overallStatus", "statusModule")
    update_struct = _required_mapping(
        status, "lastUpdatePostDateStruct", "statusModule"
    )
    last_update = _required_text(update_struct, "date", "lastUpdatePostDateStruct")

    conditions_module = _optional_mapping(
        protocol.get("conditionsModule"), "conditionsModule"
    )
    conditions = _optional_text_array(conditions_module.get("conditions"), "conditions")
    arms = _optional_mapping(
        protocol.get("armsInterventionsModule"), "armsInterventionsModule"
    )
    interventions = tuple(
        TrialIntervention(
            intervention_type=_required_text(item, "type", "intervention"),
            name=_required_text(item, "name", "intervention"),
        )
        for item in _mapping_array(arms.get("interventions"), "interventions")
    )
    contacts = _optional_mapping(
        protocol.get("contactsLocationsModule"), "contactsLocationsModule"
    )
    locations = tuple(
        _parse_location(item)
        for item in _mapping_array(contacts.get("locations"), "locations")
    )
    eligibility = _optional_mapping(
        protocol.get("eligibilityModule"), "eligibilityModule"
    )
    eligibility_text = (
        _optional_source_text(
            eligibility.get("eligibilityCriteria"),
            "eligibilityCriteria",
            allow_empty=True,
        )
        or ""
    )
    return TrialStudyRecord(
        study_id=study_id,
        overall_status=overall_status,
        brief_title=brief_title,
        official_title=official_title,
        last_update_date=last_update,
        conditions=conditions,
        interventions=interventions,
        locations=locations,
        eligibility_text=eligibility_text,
        retrieved_at=retrieved_at,
        source_digest=canonical_digest(study),
    )


def _parse_location(value: Mapping[str, Any]) -> TrialLocation:
    facility = value.get("facility")
    if isinstance(facility, Mapping):
        facility = facility.get("name")
    return TrialLocation(
        facility=_optional_source_text(facility, "facility"),
        city=_optional_source_text(value.get("city"), "city"),
        state=_optional_source_text(value.get("state"), "state"),
        country=_optional_source_text(value.get("country"), "country"),
    )


def _source_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TrialSchemaDriftError(f"{name} changed type")
    return value


def _source_sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TrialSchemaDriftError(f"{name} changed type")
    return value


def _required_mapping(
    parent: Mapping[str, Any], key: str, name: str
) -> Mapping[str, Any]:
    if key not in parent:
        raise TrialSchemaDriftError(f"{name}.{key} is missing")
    return _source_mapping(parent[key], f"{name}.{key}")


def _optional_mapping(value: Any, name: str) -> Mapping[str, Any]:
    return {} if value is None else _source_mapping(value, name)


def _required_text(parent: Mapping[str, Any], key: str, name: str) -> str:
    if key not in parent:
        raise TrialSchemaDriftError(f"{name}.{key} is missing")
    value = parent[key]
    if not isinstance(value, str) or not value:
        raise TrialSchemaDriftError(f"{name}.{key} changed type")
    return value


def _optional_source_text(
    value: Any, name: str, *, allow_empty: bool = False
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise TrialSchemaDriftError(f"{name} changed type")
    return value


def _optional_text_array(value: Any, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(
        _optional_source_text(item, name) or ""
        for item in _source_sequence(value, name)
    )


def _mapping_array(value: Any, name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    return tuple(_source_mapping(item, name) for item in _source_sequence(value, name))


def _validate_empty_page_timestamp(value: Any) -> None:
    TrialStudyRecord(
        study_id="NCT00000000",
        overall_status="UNKNOWN",
        brief_title="timestamp validation probe",
        official_title=None,
        last_update_date="1970-01-01",
        conditions=(),
        interventions=(),
        locations=(),
        eligibility_text="",
        retrieved_at=value,
        source_digest="sha256:" + "0" * 64,
    )
