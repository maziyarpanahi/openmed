"""Bounded property fuzzing for the private-learning metadata contracts.

Five versioned surfaces carry federated update metadata across a participant
trust boundary: ``FederatedRoundLifecycle``, ``FederatedRoundSchedule``,
``FederatedUpdateMetadata``, ``FederatedMetricEnvelope``, and the round-status
builder.  Every payload generated here is synthetic -- fabricated digests,
counts, and timestamps -- and no clinical record is involved.

Invariants asserted for any generated payload:

* **Typed rejection only** -- a payload is either accepted or rejected through
  the contract's own error.  Any other exception is a crash: that is how the
  two escapes fixed alongside this harness were found (``RecursionError`` for
  deeply nested JSON and the integer-string digit limit for oversized integer
  literals in ``from_json``).
* **Canonical round-trip** -- a payload accepted from JSON re-serializes to the
  exact text it came from, and a payload accepted from a mapping re-serializes
  to the same mapping.
* **No payload echo** -- rejection messages never repeat the injected sentinel
  value, so a malformed update cannot leak the offending text into logs.

Integer magnitude is deliberately not constrained: the metadata contracts
preserve participant counts and group sizes of any size, and an out-of-band
integer large enough to exceed the interpreter's digit limit is a caller
concern rather than a metadata-validation concern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openmed.training.federated_metrics import (
    FederatedMetricEnvelope,
    FederatedMetricError,
    FederatedMetricKind,
    FederatedPrivacyMechanism,
    build_federated_metric_envelope,
)
from openmed.training.federated_round import (
    FederatedRoundLifecycle,
    FederatedRoundStateError,
)
from openmed.training.federated_schedule import (
    FederatedRoundSchedule,
    FederatedScheduleError,
)
from openmed.training.federated_status import (
    FederatedRoundReasonCode,
    FederatedRoundState,
    FederatedRoundStatusError,
    build_federated_round_status,
)
from openmed.training.federated_update_metadata import (
    FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdateMetadataError,
    FederatedUpdatePolicy,
)

# Ensure the bounded/nightly Hypothesis profile is registered.
from . import conftest as _fuzz_conftest  # noqa: F401  (import for side effects)

pytestmark = pytest.mark.fuzz

_SENTINEL = "openmed-fuzz-sentinel-2f8c41d7"
_MODEL_DIGEST = "sha256:" + "a" * 64
_UPDATE_DIGEST = "sha256:" + "b" * 64
_START = datetime(2026, 9, 1, tzinfo=timezone.utc)

# Expected outcome for a generated input.  ``_REJECT`` inputs are invalid by
# construction, so the contract has to raise its own error for them;
# ``_TYPED_OR_CANONICAL`` inputs may legitimately be accepted only when they are
# already canonical.
_REJECT = "reject"
_TYPED_OR_CANONICAL = "typed-or-canonical"


@dataclass(frozen=True)
class _Contract:
    """One public metadata surface with a canonical payload for it."""

    name: str
    typed_error: type[Exception]
    scalar_field: str
    canonical_mapping: dict[str, Any]
    canonical_json: str
    parse_mapping: Callable[[Any], Any]
    parse_json: Callable[[str], Any] | None


def _policy() -> FederatedUpdatePolicy:
    """Return the coordinator policy matching the canonical update payload."""
    return FederatedUpdatePolicy(
        model_digest=_MODEL_DIGEST,
        parameters=(
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
        ),
        max_total_elements=14,
    )


def _update_payload() -> dict[str, Any]:
    """Return the canonical private-learning update payload."""
    return {
        "schema_version": FEDERATED_UPDATE_METADATA_SCHEMA_VERSION,
        "model_digest": _MODEL_DIGEST,
        "adapter_format": "dense",
        "parameters": [
            {"name": "adapter.lora_A.weight", "shape": [2, 3], "dtype": "float32"},
            {"name": "adapter.lora_B.weight", "shape": [4, 2], "dtype": "float32"},
        ],
        "total_elements": 14,
        "update_digest": _UPDATE_DIGEST,
        "clipped": True,
    }


def _contracts() -> tuple[_Contract, ...]:
    """Return every contract under test with one canonical payload each."""
    round_lifecycle = FederatedRoundLifecycle()
    schedule = FederatedRoundSchedule(
        enrollment_starts_at=_START,
        update_submission_starts_at=_START + timedelta(hours=1),
        aggregation_starts_at=_START + timedelta(hours=2),
        evaluation_starts_at=_START + timedelta(hours=3),
        finishes_at=_START + timedelta(hours=4),
    )
    policy = _policy()
    update = FederatedUpdateMetadata.from_dict(_update_payload(), policy=policy)
    metric = build_federated_metric_envelope(
        metric_id="metric-1",
        metric_kind=FederatedMetricKind.BOUNDED_MEAN,
        aggregate_value=12.5,
        clipping_lower_bound=0,
        clipping_upper_bound=100,
        privacy_mechanism=FederatedPrivacyMechanism.THRESHOLD_ONLY,
        privacy_mechanism_version="v1",
        participant_count=25,
        minimum_group_size=5,
    )
    return (
        _Contract(
            name="federated_round",
            typed_error=FederatedRoundStateError,
            scalar_field="state",
            canonical_mapping=round_lifecycle.to_dict(),
            canonical_json=round_lifecycle.to_json(),
            parse_mapping=FederatedRoundLifecycle.from_dict,
            parse_json=FederatedRoundLifecycle.from_json,
        ),
        _Contract(
            name="federated_schedule",
            typed_error=FederatedScheduleError,
            scalar_field="schema_version",
            canonical_mapping=schedule.to_dict(),
            canonical_json=schedule.to_json(),
            parse_mapping=FederatedRoundSchedule.from_dict,
            parse_json=FederatedRoundSchedule.from_json,
        ),
        _Contract(
            name="federated_update_metadata",
            typed_error=FederatedUpdateMetadataError,
            scalar_field="adapter_format",
            canonical_mapping=update.to_dict(),
            canonical_json=update.to_json(),
            parse_mapping=lambda payload: FederatedUpdateMetadata.from_dict(
                payload, policy=policy
            ),
            parse_json=lambda text: FederatedUpdateMetadata.from_json(
                text, policy=policy
            ),
        ),
        _Contract(
            name="federated_metric",
            typed_error=FederatedMetricError,
            scalar_field="aggregate_value",
            canonical_mapping=metric.to_dict(),
            canonical_json=metric.to_json(),
            parse_mapping=FederatedMetricEnvelope.from_dict,
            parse_json=None,
        ),
    )


_CONTRACTS = _contracts()
_JSON_CONTRACTS = tuple(contract for contract in _CONTRACTS if contract.parse_json)

# Textual mutations of a canonical JSON payload.
_JSON_MUTATIONS = (
    "duplicate-key",
    "unknown-field",
    "version-type",
    "version-sentinel",
    "scalar-non-finite",
    "scalar-lone-surrogate",
    "oversized-integer",
    "nested-array",
    "nested-object",
    "arbitrary-text",
)

# Structural mutations of a canonical mapping payload.
_MAPPING_MUTATIONS = (
    "unknown-field",
    "unknown-nested-field",
    "version-null",
    "version-int",
    "version-list",
    "version-object",
    "version-bool",
    "version-sentinel",
    "scalar-sentinel",
    "scalar-list",
    "non-mapping",
)

_VERSION_MUTATIONS: dict[str, Any] = {
    "version-null": None,
    "version-int": 17,
    "version-list": [1, 2, 3],
    "version-object": {"schema_version": 1},
    "version-bool": True,
    "version-sentinel": _SENTINEL,
}

_NON_FINITE_TOKENS = ("NaN", "Infinity", "-Infinity")


def _describe(payload: object) -> str:
    """Return a length and digest fingerprint without echoing payload text."""
    try:
        raw = (
            payload.encode("utf-8", "surrogatepass")
            if isinstance(payload, str)
            else repr(payload).encode("utf-8", "backslashreplace")
        )
    except Exception:  # pragma: no cover - defensive, e.g. an unprintable object
        raw = type(payload).__name__.encode("ascii", "replace")
    return f"len={len(raw)} sha256={hashlib.sha256(raw).hexdigest()[:16]}"


def _assert_no_echo(error: Exception) -> None:
    """A rejection message must not repeat injected payload values."""
    assert _SENTINEL not in str(error), "rejection message echoed the injected sentinel"
    assert _SENTINEL not in repr(error), "rejection repr echoed the injected sentinel"


def _parse(contract: _Contract, parse: Callable[[Any], Any], payload: Any) -> Any:
    """Run one parser, returning ``None`` when the contract rejected the input."""
    try:
        return parse(payload)
    except contract.typed_error as error:
        _assert_no_echo(error)
        return None
    except Exception as error:  # noqa: BLE001 - the harness reports every escape
        pytest.fail(
            f"{contract.name}: {type(error).__name__} escaped for {_describe(payload)}"
        )


def _check(
    contract: _Contract, parse: Callable[[Any], Any], payload: Any, outcome: str
) -> None:
    """Assert one generated input is rejected or canonically accepted."""
    parsed = _parse(contract, parse, payload)
    if parsed is None:
        return
    if outcome == _REJECT:
        pytest.fail(f"{contract.name}: accepted invalid input {_describe(payload)}")
    assert parsed.to_json() == payload, (
        f"{contract.name}: accepted non-canonical input {_describe(payload)}"
    )


def _first_key_marker(text: str) -> str:
    """Return the first object key plus its colon, for either indent style."""
    key = sorted(json.loads(text))[0]
    start = text.index(json.dumps(key))
    return text[start : text.index(":", start) + 1]


@st.composite
def _json_mutations(draw: st.DrawFn) -> tuple[_Contract, str, str]:
    """Draw one JSON contract plus a structured mutation of its payload."""
    contract = draw(st.sampled_from(_JSON_CONTRACTS))
    text = contract.canonical_json
    payload = json.loads(text)
    version = json.dumps(payload["schema_version"])
    scalar = json.dumps(payload[contract.scalar_field])
    kind = draw(st.sampled_from(_JSON_MUTATIONS))
    if kind == "duplicate-key":
        marker = _first_key_marker(text)
        return contract, text.replace(marker, marker + " 17, " + marker, 1), _REJECT
    if kind == "unknown-field":
        return contract, text[:-1] + f', "unknown_sentinel": "{_SENTINEL}"}}', _REJECT
    if kind == "version-type":
        replacement = draw(st.sampled_from(("null", "17", "1.5", "[]", "{}", "true")))
        return contract, text.replace(version, replacement, 1), _REJECT
    if kind == "version-sentinel":
        return contract, text.replace(version, json.dumps(_SENTINEL), 1), _REJECT
    if kind == "scalar-non-finite":
        return (
            contract,
            text.replace(scalar, draw(st.sampled_from(_NON_FINITE_TOKENS)), 1),
            _REJECT,
        )
    if kind == "scalar-lone-surrogate":
        return contract, text.replace(scalar, '"\\ud800"', 1), _REJECT
    if kind == "oversized-integer":
        digits = draw(st.integers(min_value=4301, max_value=6000))
        return contract, '{"' + sorted(payload)[0] + '": ' + "1" * digits + "}", _REJECT
    if kind == "nested-array":
        depth = draw(st.integers(min_value=1, max_value=1500))
        return contract, "[" * depth + "]" * depth, _REJECT
    if kind == "nested-object":
        depth = draw(st.integers(min_value=1, max_value=1500))
        return contract, '{"a": ' * depth + "1" + "}" * depth, _REJECT
    return contract, draw(st.text(max_size=64)), _TYPED_OR_CANONICAL


@st.composite
def _mapping_mutations(draw: st.DrawFn) -> tuple[_Contract, Any, str]:
    """Draw one contract plus a structured mutation of its mapping payload."""
    contract = draw(st.sampled_from(_CONTRACTS))
    payload = dict(contract.canonical_mapping)
    first_key = sorted(payload)[0]
    kind = draw(st.sampled_from(_MAPPING_MUTATIONS))
    if kind == "unknown-field":
        payload["unknown_sentinel"] = _SENTINEL
    elif kind == "unknown-nested-field":
        payload[first_key] = {"unknown_sentinel": _SENTINEL}
    elif kind in _VERSION_MUTATIONS:
        payload["schema_version"] = _VERSION_MUTATIONS[kind]
    elif kind == "scalar-sentinel":
        payload[contract.scalar_field] = _SENTINEL
    elif kind == "scalar-list":
        payload[contract.scalar_field] = [_SENTINEL]
    else:
        return (
            contract,
            draw(
                st.none()
                | st.booleans()
                | st.integers()
                | st.floats()
                | st.text(max_size=32)
                | st.lists(st.text(max_size=8), max_size=4)
            ),
            _REJECT,
        )
    return contract, payload, _REJECT


@pytest.mark.parametrize("contract", _CONTRACTS, ids=lambda contract: contract.name)
def test_canonical_payloads_round_trip(contract: _Contract) -> None:
    """Canonical payloads are accepted and re-serialized byte for byte."""
    parsed = _parse(contract, contract.parse_mapping, dict(contract.canonical_mapping))
    assert parsed is not None
    assert parsed.to_dict() == contract.canonical_mapping
    assert parsed.to_json() == contract.canonical_json
    if contract.parse_json is not None:
        reparsed = _parse(contract, contract.parse_json, contract.canonical_json)
        assert reparsed is not None
        assert reparsed.to_json() == contract.canonical_json


@given(case=_json_mutations())
@settings(deadline=1000)
def test_json_mutations_never_raise_untyped_errors(case) -> None:
    """Malformed JSON payloads only ever raise the contract's own error."""
    contract, payload, outcome = case
    assert contract.parse_json is not None
    _check(contract, contract.parse_json, payload, outcome)


@given(case=_mapping_mutations())
@settings(deadline=1000)
def test_mapping_mutations_never_raise_untyped_errors(case) -> None:
    """Malformed mapping payloads only ever raise the contract's own error."""
    contract, payload, outcome = case
    _check(contract, contract.parse_mapping, payload, outcome)


# Payloads that crashed the JSON parsers before the typed-error handling was
# widened: a deeply nested document and an integer literal past the
# interpreter's digit limit.  They stay as deterministic regressions.
_REGRESSION_PAYLOADS: tuple[tuple[str, str], ...] = (
    ("nested-array-3000", "[" * 3000 + "]" * 3000),
    ("nested-object-3000", '{"state": ' * 3000 + "1" + "}" * 3000),
    ("oversized-integer-4500", '{"state": ' + "1" * 4500 + "}"),
)


@pytest.mark.parametrize(
    "payload",
    [payload for _, payload in _REGRESSION_PAYLOADS],
    ids=[label for label, _ in _REGRESSION_PAYLOADS],
)
def test_json_regression_payloads_raise_typed_errors(payload: str) -> None:
    """Previously crashing payloads now surface as the contract's own error."""
    for contract in _JSON_CONTRACTS:
        _check(contract, contract.parse_json, payload, _REJECT)


_STATUS_REJECTIONS: tuple[dict[str, Any], ...] = (
    {"reason_code": _SENTINEL},
    {"aggregate_digest_refs": [_SENTINEL]},
    {"aggregate_digest_refs": [17]},
    {"aggregate_digest_refs": ["digest-1"]},
    {"participant_count": -1},
    {"completed_participant_count": 99},
    {"required_quorum": 0},
    {"minimum_group_size": 0},
    {"state": _SENTINEL},
    {"participant_count": 1.5},
    {"participant_count": "25"},
)


@pytest.mark.parametrize(
    "override",
    _STATUS_REJECTIONS,
    ids=[
        next(iter(override)) + "-" + str(next(iter(override.values())))
        for override in _STATUS_REJECTIONS
    ],
)
def test_round_status_rejections_are_typed(override: dict[str, Any]) -> None:
    """Malformed round-status arguments raise the status builder's error."""
    arguments: dict[str, Any] = {
        "state": FederatedRoundState.AGGREGATING,
        "participant_count": 25,
        "completed_participant_count": 20,
        "required_quorum": 10,
        **override,
    }
    with pytest.raises(FederatedRoundStatusError) as captured:
        build_federated_round_status(**arguments)
    _assert_no_echo(captured.value)


@st.composite
def _status_arguments(draw: st.DrawFn) -> dict[str, Any]:
    """Draw a round-status argument set, including out-of-range combinations."""
    participant_count = draw(st.integers(min_value=0, max_value=10_000))
    return {
        "state": draw(st.sampled_from(tuple(FederatedRoundState))),
        "participant_count": participant_count,
        "completed_participant_count": draw(
            st.integers(min_value=0, max_value=participant_count + 1)
        ),
        "required_quorum": draw(
            st.integers(min_value=0, max_value=participant_count + 1)
        ),
        "minimum_group_size": draw(st.integers(min_value=0, max_value=10_000)),
        "aggregate_digest_refs": draw(st.lists(st.text(max_size=16), max_size=3)),
        "reason_code": draw(
            st.none()
            | st.sampled_from(tuple(FederatedRoundReasonCode))
            | st.text(max_size=8)
        ),
    }


@given(arguments=_status_arguments())
@settings(deadline=1000)
def test_round_status_builder_only_raises_typed_errors(arguments) -> None:
    """The status builder either rejects with its own error or is canonical."""
    try:
        status = build_federated_round_status(**arguments)
    except FederatedRoundStatusError as error:
        _assert_no_echo(error)
        return
    except Exception as error:  # noqa: BLE001 - the harness reports every escape
        pytest.fail(f"round status: {type(error).__name__} escaped: {error}")
    text = status.to_json()
    assert _SENTINEL not in text
    assert json.loads(text) == status.to_dict()
