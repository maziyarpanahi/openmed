"""Synthetic, offline contracts for federated local contribution floors."""

from __future__ import annotations

import json
import pickle
import traceback
from dataclasses import FrozenInstanceError, fields

import pytest

from openmed.training import federated_contribution_floors as floors_module
from openmed.training.federated_contribution_floors import (
    DEFAULT_DP_CONTRIBUTION_FLOORS,
    DEFAULT_NON_DP_CONTRIBUTION_FLOORS,
    FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION,
    FEDERATED_CONTRIBUTION_SCHEMA_VERSION,
    MAX_FEDERATED_CONTRIBUTION_FLOOR,
    FederatedContributionBand,
    FederatedContributionDecision,
    FederatedContributionDeclaration,
    FederatedContributionError,
    FederatedContributionFloorPolicy,
    FederatedContributionReasonCode,
    FederatedContributionRefusedError,
    check_federated_contribution,
    seal_federated_contribution,
)
from openmed.training.federated_metrics import FederatedPrivacyMechanism

UPDATE = "sha256:" + "b" * 64
OTHER_UPDATE = "sha256:" + "c" * 64
R = FederatedContributionReasonCode
B = FederatedContributionBand
M = FederatedPrivacyMechanism
FLOOR_FIELDS = ("min_local_examples", "min_local_steps", "min_effective_batch_size")
FLOOR_REASONS = {
    "min_local_examples": R.LOCAL_EXAMPLES_BELOW_FLOOR,
    "min_local_steps": R.LOCAL_STEPS_BELOW_FLOOR,
    "min_effective_batch_size": R.EFFECTIVE_BATCH_SIZE_BELOW_FLOOR,
}
BAND_FIELDS = {
    "min_local_examples": "local_example_band",
    "min_local_steps": "local_step_band",
    "min_effective_batch_size": "effective_batch_size_band",
}
COUNT_ARGUMENTS = {
    "min_local_examples": "local_examples",
    "min_local_steps": "local_steps",
    "min_effective_batch_size": "effective_batch_size",
}


@pytest.fixture
def contract():
    return FederatedContributionFloorPolicy()


def _counts(**changes: int) -> dict[str, int]:
    counts = {"local_examples": 300, "local_steps": 10, "effective_batch_size": 40}
    counts.update(changes)
    return counts


def _seal(contract, **changes: int) -> FederatedContributionDeclaration:
    return seal_federated_contribution(
        contract, update_digest=UPDATE, **_counts(**changes)
    )


def _declaration_payload(contract, **bands: str) -> dict[str, str]:
    payload = {
        "schema_version": FEDERATED_CONTRIBUTION_SCHEMA_VERSION,
        "contract_digest": contract.contract_digest,
        "update_digest": UPDATE,
        "local_example_band": "256_to_under_512",
        "local_step_band": "8_to_under_16",
        "effective_batch_size_band": "32_to_under_64",
    }
    payload.update(bands)
    return payload


# Round contract defaults and explicit overrides.


def test_rounds_without_dp_mechanism_apply_stricter_defaults():
    contract = FederatedContributionFloorPolicy()

    assert contract.privacy_mechanism is M.THRESHOLD_ONLY
    assert contract.dp_mechanism_declared is False
    assert contract.min_local_examples == 256
    assert contract.min_local_steps == 8
    assert contract.min_effective_batch_size == 32
    assert contract.allow_floors_below_default is False
    assert dict(DEFAULT_NON_DP_CONTRIBUTION_FLOORS) == {
        "min_local_examples": 256,
        "min_local_steps": 8,
        "min_effective_batch_size": 32,
    }


@pytest.mark.parametrize("mechanism", [M.LAPLACE, M.GAUSSIAN])
def test_declared_dp_mechanism_applies_dp_defaults(mechanism):
    contract = FederatedContributionFloorPolicy(privacy_mechanism=mechanism)

    assert contract.dp_mechanism_declared is True
    assert contract.min_local_examples == 64
    assert contract.min_local_steps == 1
    assert contract.min_effective_batch_size == 8
    assert dict(DEFAULT_DP_CONTRIBUTION_FLOORS) == {
        "min_local_examples": 64,
        "min_local_steps": 1,
        "min_effective_batch_size": 8,
    }


def test_non_dp_defaults_are_strictly_higher_than_dp_defaults():
    for field in FLOOR_FIELDS:
        assert (
            DEFAULT_NON_DP_CONTRIBUTION_FLOORS[field]
            > DEFAULT_DP_CONTRIBUTION_FLOORS[field]
        )
    with pytest.raises(TypeError):
        DEFAULT_NON_DP_CONTRIBUTION_FLOORS["min_local_steps"] = 1  # type: ignore[index]


@pytest.mark.parametrize("field", FLOOR_FIELDS)
@pytest.mark.parametrize("mechanism", [M.THRESHOLD_ONLY, M.GAUSSIAN])
def test_raising_a_floor_needs_no_override(field, mechanism):
    raised = {"min_local_examples": MAX_FEDERATED_CONTRIBUTION_FLOOR}
    raised[field] = MAX_FEDERATED_CONTRIBUTION_FLOOR
    contract = FederatedContributionFloorPolicy(privacy_mechanism=mechanism, **raised)
    assert getattr(contract, field) == MAX_FEDERATED_CONTRIBUTION_FLOOR


@pytest.mark.parametrize("field", FLOOR_FIELDS)
def test_lowering_a_non_dp_floor_requires_explicit_policy_field(field):
    lowered = DEFAULT_DP_CONTRIBUTION_FLOORS[field]

    with pytest.raises(FederatedContributionError, match="explicit override"):
        FederatedContributionFloorPolicy(**{field: lowered})

    contract = FederatedContributionFloorPolicy(
        allow_floors_below_default=True, **{field: lowered}
    )
    assert getattr(contract, field) == lowered
    assert contract.allow_floors_below_default is True


@pytest.mark.parametrize(
    ("field", "lowered"), [("min_local_examples", 32), ("min_effective_batch_size", 4)]
)
def test_lowering_a_dp_floor_also_requires_explicit_policy_field(field, lowered):
    with pytest.raises(FederatedContributionError, match="explicit override"):
        FederatedContributionFloorPolicy(
            privacy_mechanism=M.LAPLACE, **{field: lowered}
        )
    contract = FederatedContributionFloorPolicy(
        privacy_mechanism=M.LAPLACE, allow_floors_below_default=True, **{field: lowered}
    )
    assert getattr(contract, field) == lowered


def test_dp_defaults_do_not_leak_into_non_dp_rounds():
    # The DP default is below the non-DP default, so it is an explicit lowering.
    with pytest.raises(FederatedContributionError, match="explicit override"):
        FederatedContributionFloorPolicy(
            privacy_mechanism=M.THRESHOLD_ONLY, **dict(DEFAULT_DP_CONTRIBUTION_FLOORS)
        )


@pytest.mark.parametrize("mechanism", [M.THRESHOLD_ONLY, M.GAUSSIAN])
def test_override_without_a_lowered_floor_is_rejected(mechanism):
    with pytest.raises(FederatedContributionError, match="override"):
        FederatedContributionFloorPolicy(
            privacy_mechanism=mechanism, allow_floors_below_default=True
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, -256, True, False, 256.0, "256", [], 3, 100, 1 << 17],
)
@pytest.mark.parametrize("field", FLOOR_FIELDS)
def test_bad_floor_values_fail_closed(field, value):
    with pytest.raises(FederatedContributionError, match="contribution floor"):
        FederatedContributionFloorPolicy(
            allow_floors_below_default=True, **{field: value}
        )


def test_excessive_floor_is_rejected():
    with pytest.raises(FederatedContributionError, match="contribution floor"):
        FederatedContributionFloorPolicy(
            min_local_examples=MAX_FEDERATED_CONTRIBUTION_FLOOR * 2
        )


@pytest.mark.parametrize("value", [1, "true", None, 0])
def test_override_flag_must_be_a_strict_boolean(value):
    with pytest.raises(FederatedContributionError, match="override"):
        FederatedContributionFloorPolicy(allow_floors_below_default=value)


@pytest.mark.parametrize("value", ["threshold_only", "gaussian", None, 1, "dp"])
def test_privacy_mechanism_must_be_the_existing_enum(value):
    with pytest.raises(FederatedContributionError, match="privacy mechanism"):
        FederatedContributionFloorPolicy(privacy_mechanism=value)


def test_batch_floor_cannot_exceed_example_floor():
    with pytest.raises(FederatedContributionError, match="batch"):
        FederatedContributionFloorPolicy(
            min_local_examples=256, min_effective_batch_size=512
        )


def test_contract_is_immutable_and_versioned(contract):
    with pytest.raises(FrozenInstanceError):
        contract.min_local_steps = 1  # type: ignore[misc]
    with pytest.raises(FederatedContributionError, match="schema"):
        FederatedContributionFloorPolicy(schema_version="openmed.training.other.v1")


# Contract serialization and binding.


def test_contract_round_trips_with_explicit_floors(contract):
    payload = contract.to_dict()

    assert payload == {
        "schema_version": FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION,
        "privacy_mechanism": "threshold_only",
        "min_local_examples": 256,
        "min_local_steps": 8,
        "min_effective_batch_size": 32,
        "allow_floors_below_default": False,
    }
    assert FederatedContributionFloorPolicy.from_dict(payload) == contract
    assert FederatedContributionFloorPolicy.from_json(contract.to_json()) == contract
    assert contract.to_json().endswith("\n")
    reordered = json.dumps(dict(reversed(list(payload.items()))))
    assert FederatedContributionFloorPolicy.from_json(reordered).to_json() == (
        contract.to_json()
    )


def test_contract_digest_binds_every_floor_and_the_mechanism(contract):
    digest = contract.contract_digest

    assert digest.startswith("sha256:") and len(digest) == 71
    assert digest == FederatedContributionFloorPolicy().contract_digest
    variants = [
        FederatedContributionFloorPolicy(privacy_mechanism=M.GAUSSIAN),
        FederatedContributionFloorPolicy(min_local_examples=512),
        FederatedContributionFloorPolicy(min_local_steps=16),
        FederatedContributionFloorPolicy(min_effective_batch_size=64),
        FederatedContributionFloorPolicy(
            min_local_steps=4, allow_floors_below_default=True
        ),
    ]
    digests = {variant.contract_digest for variant in variants}
    assert digest not in digests
    assert len(digests) == len(variants)


@pytest.mark.parametrize("field", FLOOR_FIELDS)
def test_contract_payload_cannot_lower_a_floor_without_the_policy_field(
    contract, field
):
    payload = contract.to_dict()
    payload[field] = DEFAULT_DP_CONTRIBUTION_FLOORS[field]
    with pytest.raises(FederatedContributionError, match="explicit override"):
        FederatedContributionFloorPolicy.from_dict(payload)
    payload["allow_floors_below_default"] = True
    assert FederatedContributionFloorPolicy.from_dict(
        payload
    ).allow_floors_below_default


@pytest.mark.parametrize(
    "change",
    [
        {"min_local_steps": None},
        {"privacy_mechanism": "none"},
        {"privacy_mechanism": None},
        {"schema_version": "openmed.training.federated_contribution_floor_policy.v2"},
    ],
)
def test_contract_payload_fields_are_explicit_and_closed(contract, change):
    payload = {**contract.to_dict(), **change}
    with pytest.raises(FederatedContributionError):
        FederatedContributionFloorPolicy.from_dict(payload)


def test_contract_payload_rejects_missing_and_unknown_fields(contract):
    payload = contract.to_dict()
    del payload["allow_floors_below_default"]
    with pytest.raises(FederatedContributionError, match="fields"):
        FederatedContributionFloorPolicy.from_dict(payload)
    with pytest.raises(FederatedContributionError, match="fields"):
        FederatedContributionFloorPolicy.from_dict(
            {**contract.to_dict(), "local_examples": 300}
        )
    with pytest.raises(FederatedContributionError, match="fields"):
        FederatedContributionFloorPolicy.from_dict([contract.to_dict()])


_CONTRACT_JSON = (
    '{"allow_floors_below_default": false, "min_effective_batch_size": 32, '
    '"min_local_examples": 256, "min_local_steps": %s, '
    '"privacy_mechanism": "threshold_only", '
    '"schema_version": "' + FEDERATED_CONTRIBUTION_FLOOR_POLICY_SCHEMA_VERSION + '"%s}'
)


def test_contract_json_fixture_is_valid(contract):
    assert FederatedContributionFloorPolicy.from_json(_CONTRACT_JSON % ("8", "")) == (
        contract
    )


@pytest.mark.parametrize(
    "raw",
    [
        _CONTRACT_JSON % ("8", ', "min_local_steps": 1'),
        _CONTRACT_JSON % ("8.0", ""),
        _CONTRACT_JSON % ("NaN", ""),
        _CONTRACT_JSON % ("8e0", ""),
        _CONTRACT_JSON % ("1" * 20, ""),
        "[",
        "x" * (1024 * 1024 + 1),
        b"{}",
    ],
)
def test_contract_json_is_strict(raw):
    with pytest.raises(FederatedContributionError):
        FederatedContributionFloorPolicy.from_json(raw)  # type: ignore[arg-type]


# Bands.


def test_bands_are_power_of_two_ranges_with_an_open_top():
    bands = list(FederatedContributionBand)

    assert bands[0].value == "1_to_under_2"
    assert bands[-1].value == "65536_or_more"
    assert [band.lower_bound for band in bands] == [1 << i for i in range(17)]
    assert MAX_FEDERATED_CONTRIBUTION_FLOOR == bands[-1].lower_bound
    for band in bands[:-1]:
        upper = band.lower_bound * 2
        assert band.value == f"{band.lower_bound}_to_under_{upper}"
    assert str(B.FROM_256) == "256_to_under_512"


@pytest.mark.parametrize(
    ("count", "band"),
    [
        (256, B.FROM_256),
        (257, B.FROM_256),
        (511, B.FROM_256),
        (512, B.FROM_512),
        (65535, B.FROM_32768),
        (65536, B.FROM_65536),
        ((1 << 63) - 1, B.FROM_65536),
    ],
)
def test_client_seals_examples_into_their_band(contract, count, band):
    declaration = _seal(contract, local_examples=count)
    assert declaration.local_example_band is band


# Client-side guard.


def test_client_seals_band_only_record_at_or_above_every_floor(contract):
    declaration = _seal(
        contract, local_examples=256, local_steps=8, effective_batch_size=32
    )

    assert declaration.to_dict() == {
        "schema_version": FEDERATED_CONTRIBUTION_SCHEMA_VERSION,
        "contract_digest": contract.contract_digest,
        "update_digest": UPDATE,
        "local_example_band": "256_to_under_512",
        "local_step_band": "8_to_under_16",
        "effective_batch_size_band": "32_to_under_64",
    }
    assert {field.name for field in fields(declaration)} == {
        "schema_version",
        "contract_digest",
        "update_digest",
        "local_example_band",
        "local_step_band",
        "effective_batch_size_band",
    }
    with pytest.raises(FrozenInstanceError):
        declaration.local_step_band = B.FROM_1  # type: ignore[misc]


@pytest.mark.parametrize("floor", FLOOR_FIELDS)
def test_client_refuses_to_seal_below_each_floor_independently(contract, floor):
    below = getattr(contract, floor) - 1
    with pytest.raises(FederatedContributionRefusedError) as refused:
        _seal(contract, **{COUNT_ARGUMENTS[floor]: below})

    assert refused.value.reason_codes == (FLOOR_REASONS[floor],)
    assert isinstance(refused.value, FederatedContributionError)


def test_refusal_pickles_with_reason_codes_only():
    refused = FederatedContributionRefusedError((R.LOCAL_STEPS_BELOW_FLOOR,))
    restored = pickle.loads(pickle.dumps(refused))

    assert type(restored) is FederatedContributionRefusedError
    assert restored.reason_codes == refused.reason_codes
    assert restored.args == refused.args


def test_client_reports_every_failed_floor_in_stable_order(contract):
    with pytest.raises(FederatedContributionRefusedError) as refused:
        _seal(contract, local_examples=0, local_steps=0, effective_batch_size=0)

    assert refused.value.reason_codes == (
        R.LOCAL_EXAMPLES_BELOW_FLOOR,
        R.LOCAL_STEPS_BELOW_FLOOR,
        R.EFFECTIVE_BATCH_SIZE_BELOW_FLOOR,
    )


def test_no_dp_round_refuses_counts_that_a_dp_round_accepts():
    counts = {"local_examples": 64, "local_steps": 1, "effective_batch_size": 8}
    dp = FederatedContributionFloorPolicy(privacy_mechanism=M.GAUSSIAN)
    no_dp = FederatedContributionFloorPolicy()

    assert seal_federated_contribution(dp, update_digest=UPDATE, **counts)
    with pytest.raises(FederatedContributionRefusedError) as refused:
        seal_federated_contribution(no_dp, update_digest=UPDATE, **counts)
    assert refused.value.reason_codes == tuple(FLOOR_REASONS.values())


def test_explicitly_lowered_floor_is_honored_by_the_client():
    contract = FederatedContributionFloorPolicy(
        min_local_steps=1, allow_floors_below_default=True
    )
    assert _seal(contract, local_steps=1).local_step_band is B.FROM_1


@pytest.mark.parametrize("argument", sorted(COUNT_ARGUMENTS.values()))
@pytest.mark.parametrize("value", [-1, True, False, 3.0, "300", None, 1 << 63])
def test_client_counts_must_be_bounded_non_negative_integers(contract, argument, value):
    with pytest.raises(FederatedContributionError) as error:
        _seal(contract, **{argument: value})
    assert not isinstance(error.value, FederatedContributionRefusedError)


def test_client_rejects_batch_larger_than_local_examples(contract):
    with pytest.raises(FederatedContributionError, match="batch") as error:
        _seal(contract, local_examples=300, effective_batch_size=301)
    assert not isinstance(error.value, FederatedContributionRefusedError)


@pytest.mark.parametrize(
    "digest", ["sha256:" + "B" * 64, "b" * 64, None, UPDATE + "\n"]
)
def test_client_requires_a_digest_reference_for_the_update(contract, digest):
    with pytest.raises(FederatedContributionError, match="digest"):
        seal_federated_contribution(contract, update_digest=digest, **_counts())


def test_client_requires_a_contract_object(contract):
    with pytest.raises(FederatedContributionError, match="contract"):
        seal_federated_contribution(
            contract.to_dict(), update_digest=UPDATE, **_counts()
        )


# Coordinator-side check over declared bands.


def test_coordinator_accepts_a_sealed_declaration(contract):
    decision = check_federated_contribution(
        _seal(contract), contract=contract, update_digest=UPDATE
    )

    assert decision.accepted is True
    assert decision.reason_codes == ()
    assert decision.to_dict() == {"accepted": True, "reason_codes": []}


@pytest.mark.parametrize("floor", FLOOR_FIELDS)
def test_coordinator_rejects_declared_bands_below_each_floor_independently(
    contract, floor
):
    # The payload never passed the client guard: it is a hand-built declaration.
    payload = _declaration_payload(contract, **{BAND_FIELDS[floor]: "1_to_under_2"})
    declaration = FederatedContributionDeclaration.from_dict(payload)

    decision = check_federated_contribution(
        declaration, contract=contract, update_digest=UPDATE
    )

    assert decision.accepted is False
    assert decision.reason_codes == (FLOOR_REASONS[floor],)
    assert decision.to_dict() == {
        "accepted": False,
        "reason_codes": [FLOOR_REASONS[floor].value],
    }


@pytest.mark.parametrize("floor", FLOOR_FIELDS)
def test_coordinator_rejects_the_band_just_below_each_floor(contract, floor):
    below = FederatedContributionBand.for_count(getattr(contract, floor) - 1)
    payload = _declaration_payload(contract, **{BAND_FIELDS[floor]: below.value})

    decision = check_federated_contribution(
        FederatedContributionDeclaration.from_json(json.dumps(payload)),
        contract=contract,
        update_digest=UPDATE,
    )

    assert decision.reason_codes == (FLOOR_REASONS[floor],)


def test_coordinator_applies_its_own_contract_not_the_client_contract():
    lenient = FederatedContributionFloorPolicy(
        min_local_examples=1,
        min_local_steps=1,
        min_effective_batch_size=1,
        allow_floors_below_default=True,
    )
    strict = FederatedContributionFloorPolicy()
    declaration = seal_federated_contribution(
        lenient,
        update_digest=UPDATE,
        local_examples=2,
        local_steps=1,
        effective_batch_size=1,
    )

    decision = check_federated_contribution(
        declaration, contract=strict, update_digest=UPDATE
    )

    assert decision.reason_codes == (
        R.CONTRACT_MISMATCH,
        R.LOCAL_EXAMPLES_BELOW_FLOOR,
        R.LOCAL_STEPS_BELOW_FLOOR,
        R.EFFECTIVE_BATCH_SIZE_BELOW_FLOOR,
    )


def test_coordinator_rejects_declarations_bound_to_another_contract_or_update(
    contract,
):
    declaration = _seal(contract)
    dp_contract = FederatedContributionFloorPolicy(privacy_mechanism=M.GAUSSIAN)

    assert check_federated_contribution(
        declaration, contract=dp_contract, update_digest=UPDATE
    ).reason_codes == (R.CONTRACT_MISMATCH,)
    assert check_federated_contribution(
        declaration, contract=contract, update_digest=OTHER_UPDATE
    ).reason_codes == (R.UPDATE_MISMATCH,)


def test_coordinator_requires_typed_inputs(contract):
    declaration = _seal(contract)
    with pytest.raises(FederatedContributionError, match="declaration"):
        check_federated_contribution(
            declaration.to_dict(), contract=contract, update_digest=UPDATE
        )
    with pytest.raises(FederatedContributionError, match="contract"):
        check_federated_contribution(
            declaration, contract=contract.to_dict(), update_digest=UPDATE
        )
    with pytest.raises(FederatedContributionError, match="digest"):
        check_federated_contribution(
            declaration, contract=contract, update_digest="sha256:short"
        )


def test_decision_reason_codes_are_canonical():
    with pytest.raises(FederatedContributionError, match="reason"):
        FederatedContributionDecision(
            reason_codes=(R.LOCAL_STEPS_BELOW_FLOOR, R.LOCAL_EXAMPLES_BELOW_FLOOR)
        )
    with pytest.raises(FederatedContributionError, match="reason"):
        FederatedContributionDecision(reason_codes=("local_steps_below_floor",))
    with pytest.raises(FederatedContributionError, match="reason"):
        FederatedContributionDecision(
            reason_codes=(R.UPDATE_MISMATCH, R.UPDATE_MISMATCH)
        )


# Declaration payload validation.


def test_declaration_round_trips_and_is_byte_stable(contract):
    declaration = _seal(contract)

    assert FederatedContributionDeclaration.from_json(declaration.to_json()) == (
        declaration
    )
    reordered = json.dumps(dict(reversed(list(declaration.to_dict().items()))))
    assert FederatedContributionDeclaration.from_json(reordered).to_json() == (
        declaration.to_json()
    )


@pytest.mark.parametrize(
    "field", ["local_examples", "local_steps", "effective_batch_size", "count"]
)
def test_declaration_rejects_exact_count_fields(contract, field):
    payload = {**_declaration_payload(contract), field: 7919}
    with pytest.raises(FederatedContributionError) as error:
        FederatedContributionDeclaration.from_dict(payload)
    assert str(error.value) == "invalid contribution fields"
    assert "7919" not in str(error.value)


@pytest.mark.parametrize(
    "change",
    [
        {"local_example_band": "300"},
        {"local_example_band": 300},
        {"local_step_band": "0_to_under_1"},
        {"effective_batch_size_band": None},
        {"contract_digest": "sha256:" + "A" * 64},
        {"update_digest": None},
        {"schema_version": "openmed.training.federated_contribution.v2"},
    ],
)
def test_declaration_fields_are_closed_and_typed(contract, change):
    with pytest.raises(FederatedContributionError):
        FederatedContributionDeclaration.from_dict(
            {**_declaration_payload(contract), **change}
        )


def test_direct_declaration_construction_requires_band_members(contract):
    with pytest.raises(FederatedContributionError, match="band"):
        FederatedContributionDeclaration(
            contract_digest=contract.contract_digest,
            update_digest=UPDATE,
            local_example_band="256_to_under_512",
            local_step_band=B.FROM_8,
            effective_batch_size_band=B.FROM_32,
        )


# Negative-control sentinel: exact local counts never leave the site.

SENTINEL_COUNTS = {
    "local_examples": 7654321,
    "local_steps": 40961,
    "effective_batch_size": 54321,
}
REFUSED_SENTINEL_COUNTS = {
    "local_examples": 197,
    "local_steps": 7,
    "effective_batch_size": 31,
}


def _rendered(*values: object) -> str:
    return "\n".join(f"{value!s}\n{value!r}" for value in values)


def _exception_text(error: BaseException) -> str:
    return "".join(
        [
            str(error),
            repr(error),
            repr(error.args),
            "".join(traceback.format_exception_only(type(error), error)),
            repr(getattr(error, "__notes__", None)),
            repr(vars(error)),
        ]
    )


def test_sentinel_detector_finds_counts_when_they_are_present():
    # Positive control: the substring checks below would catch a leaked count.
    leaked = json.dumps({"band": "256_to_under_512", "count": 7654321})
    assert str(SENTINEL_COUNTS["local_examples"]) in leaked
    refused = FederatedContributionRefusedError((R.LOCAL_STEPS_BELOW_FLOOR,))
    assert "local_steps_below_floor" in _exception_text(refused)


def test_exact_counts_never_appear_in_records_decisions_or_repr(
    contract, caplog, capsys
):
    declaration = seal_federated_contribution(
        contract, update_digest=UPDATE, **SENTINEL_COUNTS
    )
    decision = check_federated_contribution(
        declaration, contract=contract, update_digest=UPDATE
    )
    rejected = check_federated_contribution(
        declaration,
        contract=FederatedContributionFloorPolicy(privacy_mechanism=M.GAUSSIAN),
        update_digest=OTHER_UPDATE,
    )
    surfaces = _rendered(
        declaration,
        declaration.to_dict(),
        declaration.to_json(),
        decision,
        decision.to_dict(),
        rejected,
        rejected.to_dict(),
        contract,
        contract.to_json(),
    )
    for value in SENTINEL_COUNTS.values():
        assert str(value) not in surfaces
    slot_values = [getattr(declaration, field.name) for field in fields(declaration)]
    assert not any(type(value) is int for value in slot_values)
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


def test_exact_counts_never_appear_in_client_refusals(contract, caplog, capsys):
    with pytest.raises(FederatedContributionRefusedError) as refused:
        seal_federated_contribution(
            contract, update_digest=UPDATE, **REFUSED_SENTINEL_COUNTS
        )

    text = _exception_text(refused.value)
    for value in REFUSED_SENTINEL_COUNTS.values():
        assert str(value) not in text
    assert refused.value.__cause__ is None
    assert refused.value.__context__ is None
    assert caplog.text == ""
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "counts",
    [
        {"local_examples": -7654321},
        {"local_examples": 300, "effective_batch_size": 7654321},
        {"local_steps": 7654321.5},
    ],
)
def test_exact_counts_never_appear_in_validation_errors(contract, counts):
    with pytest.raises(FederatedContributionError) as error:
        _seal(contract, **counts)

    text = _exception_text(error.value)
    assert "7654321" not in text
    assert error.value.__cause__ is None


def test_lazy_training_exports_resolve():
    import openmed.training as training

    for name in floors_module.__all__:
        assert training.__all__.count(name) == 1
        assert getattr(training, name) is getattr(floors_module, name)
