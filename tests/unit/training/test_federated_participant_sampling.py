"""Tests for secret Poisson participant selection bound to DP accounting."""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import random
import secrets
from dataclasses import dataclass

import pytest

from openmed.training.federated_participant_sampling import (
    FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION,
    SAMPLING_RATE_DENOMINATOR,
    AmplificationReasonCode,
    EligiblePopulationBand,
    ParticipantRoster,
    ParticipantSamplingDeclaration,
    ParticipantSamplingError,
    ParticipantSamplingErrorCode,
    ParticipantSelection,
    ParticipantSelectionMode,
    ParticipantSelectionProcedure,
    ParticipationDisclosure,
    PoissonParticipantSelector,
    SeededTestRandom,
    declare_unsampled_round,
    require_accountant_rate,
    require_accountant_rates,
)
from openmed.training.federated_round import FederatedRoundState
from openmed.training.federated_status import build_federated_round_status

A = AmplificationReasonCode
E = ParticipantSamplingErrorCode
P = ParticipantSelectionProcedure
OPERATIONAL = ParticipantSelectionMode.OPERATIONAL
TEST = ParticipantSelectionMode.TEST
UNDISCLOSED = ParticipationDisclosure(
    ledger_discloses_participation=False,
    status_discloses_participation=False,
)
SENTINEL_SEED = 918_273_645
SENTINEL_PREFIX = "sentinelclient"


@dataclass(frozen=True)
class FakeAccountantRound:
    """Synthetic stand-in for the round-aware accountant input."""

    round_index: int
    sampling_rate: float


class FakeAccountant:
    """Records the rates it is charged after the binding check passes."""

    def __init__(self) -> None:
        self.charged_rates: list[float] = []

    def charge(
        self,
        declaration: ParticipantSamplingDeclaration,
        accountant_round: FakeAccountantRound,
    ) -> None:
        require_accountant_rate(declaration, accountant_round)
        self.charged_rates.append(accountant_round.sampling_rate)


def _pseudonyms(count: int, prefix: str = "client") -> list[str]:
    return [f"{prefix}-{index:05d}" for index in range(count)]


def _test_selector(rate: float, seed: int = 7) -> PoissonParticipantSelector:
    return PoissonParticipantSelector(rate, mode=TEST, rng=SeededTestRandom(seed))


def _operational_selection(
    rate: float = 0.25,
    disclosure: ParticipationDisclosure = UNDISCLOSED,
    round_index: int = 3,
):
    selector = PoissonParticipantSelector(rate, rng=random.SystemRandom())
    return selector.select(
        round_index=round_index,
        eligible_pseudonyms=_pseudonyms(40),
        disclosure=disclosure,
    )


def _raises(code: ParticipantSamplingErrorCode):
    return pytest.raises(ParticipantSamplingError, match=rf"^{code.value}: ")


# Amplification is only claimed for secret Poisson selection.


def test_operational_poisson_without_disclosure_permits_amplification() -> None:
    declaration = _operational_selection().declaration

    assert declaration.procedure is P.POISSON
    assert declaration.selection_mode is OPERATIONAL
    assert declaration.selection_rate == 0.25
    assert declaration.accountant_sampling_rate == 0.25
    assert declaration.amplification_permitted is True
    assert declaration.reason_codes == (A.POISSON_SELECTION_UNDISCLOSED,)


@pytest.mark.parametrize(
    ("procedure", "reason"),
    [
        (P.AVAILABILITY_DRIVEN, A.AVAILABILITY_DRIVEN_ROSTER),
        (P.OPERATOR_CHOSEN, A.OPERATOR_CHOSEN_ROSTER),
        (P.FIXED_SIZE, A.FIXED_SIZE_ROSTER),
    ],
)
def test_unsampled_rosters_never_claim_amplification(
    procedure: ParticipantSelectionProcedure, reason: AmplificationReasonCode
) -> None:
    eligible = _pseudonyms(12)
    selection = declare_unsampled_round(
        round_index=0,
        procedure=procedure,
        eligible_pseudonyms=eligible,
        roster_pseudonyms=eligible[:4],
        rng=random.SystemRandom(),
    )
    declaration = selection.declaration
    accountant = FakeAccountant()

    accountant.charge(declaration, FakeAccountantRound(0, 1.0))

    assert declaration.amplification_permitted is False
    assert declaration.selection_rate == 1.0
    assert declaration.accountant_sampling_rate == 1.0
    assert declaration.reason_codes == (reason,)
    assert accountant.charged_rates == [1.0]
    assert len(selection.roster) == 4


@pytest.mark.parametrize("claimed_rate", [0.25, 0.5, 0.999999])
def test_unsampled_roster_rejects_an_amplified_accountant_rate(
    claimed_rate: float,
) -> None:
    eligible = _pseudonyms(12)
    declaration = declare_unsampled_round(
        round_index=0,
        procedure=P.AVAILABILITY_DRIVEN,
        eligible_pseudonyms=eligible,
        roster_pseudonyms=eligible,
    ).declaration
    accountant = FakeAccountant()

    with _raises(E.ACCOUNTANT_RATE_MISMATCH):
        accountant.charge(declaration, FakeAccountantRound(0, claimed_rate))

    assert accountant.charged_rates == []


def test_unsampled_round_refuses_the_poisson_procedure() -> None:
    eligible = _pseudonyms(6)
    with _raises(E.INVALID_PROCEDURE):
        declare_unsampled_round(
            round_index=0,
            procedure=P.POISSON,
            eligible_pseudonyms=eligible,
            roster_pseudonyms=eligible,
        )


def test_unsampled_roster_must_be_drawn_from_the_eligible_population() -> None:
    with _raises(E.ROSTER_NOT_ELIGIBLE):
        declare_unsampled_round(
            round_index=0,
            procedure=P.OPERATOR_CHOSEN,
            eligible_pseudonyms=_pseudonyms(6),
            roster_pseudonyms=["outsider-00001"],
        )


def test_full_rate_poisson_is_accounted_without_amplification() -> None:
    declaration = _operational_selection(rate=1.0).declaration

    assert declaration.amplification_permitted is False
    assert declaration.accountant_sampling_rate == 1.0
    assert declaration.reason_codes == (A.FULL_PARTICIPATION_RATE,)


# Per-round participation disclosure downgrades amplification.


@pytest.mark.parametrize(
    ("disclosure", "reasons"),
    [
        (
            ParticipationDisclosure(
                ledger_discloses_participation=True,
                status_discloses_participation=False,
            ),
            (A.LEDGER_DISCLOSES_PARTICIPATION,),
        ),
        (
            ParticipationDisclosure(
                ledger_discloses_participation=False,
                status_discloses_participation=True,
            ),
            (A.STATUS_DISCLOSES_PARTICIPATION,),
        ),
        (
            ParticipationDisclosure(
                ledger_discloses_participation=True,
                status_discloses_participation=True,
            ),
            (A.LEDGER_DISCLOSES_PARTICIPATION, A.STATUS_DISCLOSES_PARTICIPATION),
        ),
    ],
)
def test_disclosed_participation_downgrades_amplification(
    disclosure: ParticipationDisclosure,
    reasons: tuple[AmplificationReasonCode, ...],
) -> None:
    declaration = _operational_selection(disclosure=disclosure).declaration

    assert declaration.amplification_permitted is False
    assert declaration.selection_rate == 0.25
    assert declaration.accountant_sampling_rate == 1.0
    assert declaration.reason_codes == reasons
    with _raises(E.ACCOUNTANT_RATE_MISMATCH):
        require_accountant_rate(declaration, FakeAccountantRound(3, 0.25))
    require_accountant_rate(declaration, FakeAccountantRound(3, 1.0))


def test_status_with_released_participant_count_discloses_participation() -> None:
    status = build_federated_round_status(
        state=FederatedRoundState.COLLECTING,
        participant_count=12,
        completed_participant_count=0,
        required_quorum=5,
    )
    disclosure = ParticipationDisclosure.for_round_status(
        status, ledger_discloses_participation=False
    )

    assert disclosure.status_discloses_participation is True
    declaration = _operational_selection(disclosure=disclosure).declaration
    assert declaration.reason_codes == (A.STATUS_DISCLOSES_PARTICIPATION,)
    assert declaration.accountant_sampling_rate == 1.0


def test_status_with_suppressed_counts_does_not_disclose_participation() -> None:
    status = build_federated_round_status(
        state=FederatedRoundState.COLLECTING,
        participant_count=3,
        completed_participant_count=1,
        required_quorum=2,
    )
    disclosure = ParticipationDisclosure.for_round_status(
        status, ledger_discloses_participation=False
    )

    assert disclosure.status_discloses_participation is False
    declaration = _operational_selection(disclosure=disclosure).declaration
    assert declaration.amplification_permitted is True


def test_status_disclosure_helper_requires_a_built_status() -> None:
    with _raises(E.INVALID_DISCLOSURE):
        ParticipationDisclosure.for_round_status(
            {"participant_count": 12}, ledger_discloses_participation=False
        )


@pytest.mark.parametrize("value", [None, 1, 0, "yes"])
def test_disclosure_flags_must_be_explicit_booleans(value: object) -> None:
    with _raises(E.INVALID_DISCLOSURE):
        ParticipationDisclosure(
            ledger_discloses_participation=value,  # type: ignore[arg-type]
            status_discloses_participation=False,
        )


def test_selection_requires_a_disclosure_declaration() -> None:
    selector = PoissonParticipantSelector(0.25)
    with _raises(E.INVALID_DISCLOSURE):
        selector.select(
            round_index=0,
            eligible_pseudonyms=_pseudonyms(5),
            disclosure=None,  # type: ignore[arg-type]
        )


# Seeded test selection matches expected inclusion statistics.


def test_seeded_selection_count_is_within_binomial_bounds() -> None:
    population = 4000
    rate = 0.25
    selection = _test_selector(rate, seed=20261022).select(
        round_index=0,
        eligible_pseudonyms=_pseudonyms(population),
        disclosure=UNDISCLOSED,
    )
    mean = population * rate
    spread = 5 * math.sqrt(population * rate * (1 - rate))

    assert mean - spread <= len(selection.roster) <= mean + spread


def test_seeded_per_client_inclusion_frequency_is_within_binomial_bounds() -> None:
    clients = _pseudonyms(30)
    rounds = 400
    rate = 0.1
    selector = _test_selector(rate, seed=11)
    counts = dict.fromkeys(clients, 0)
    for round_index in range(rounds):
        roster = selector.select(
            round_index=round_index,
            eligible_pseudonyms=clients,
            disclosure=UNDISCLOSED,
        ).roster
        for client in roster.members():
            counts[client] += 1
    mean = rounds * rate
    spread = 5 * math.sqrt(rounds * rate * (1 - rate))

    assert all(mean - spread <= count <= mean + spread for count in counts.values())
    total_mean = rounds * len(clients) * rate
    total_spread = 5 * math.sqrt(rounds * len(clients) * rate * (1 - rate))
    assert abs(sum(counts.values()) - total_mean) <= total_spread


def test_seeded_selection_is_deterministic_and_order_independent() -> None:
    clients = _pseudonyms(200)
    reversed_clients = list(reversed(clients))

    first = _test_selector(0.3, seed=5).select(
        round_index=2, eligible_pseudonyms=clients, disclosure=UNDISCLOSED
    )
    second = _test_selector(0.3, seed=5).select(
        round_index=2, eligible_pseudonyms=reversed_clients, disclosure=UNDISCLOSED
    )
    other = _test_selector(0.3, seed=6).select(
        round_index=2, eligible_pseudonyms=clients, disclosure=UNDISCLOSED
    )

    assert first.roster.members() == second.roster.members()
    assert first.declaration == second.declaration
    assert first.roster.members() != other.roster.members()
    assert first.declaration.roster_commitment != other.declaration.roster_commitment


class _ScriptedGenerator:
    """Replace the seeded generator with fixed draws for exactness checks."""

    def __init__(self, draws: list[int]) -> None:
        self._draws = list(draws)
        self.stops: list[int] = []

    def randrange(self, stop: int) -> int:
        self.stops.append(stop)
        return self._draws.pop(0)

    def randbytes(self, size: int) -> bytes:
        return b"\x00" * size


def test_inclusion_uses_an_exact_integer_threshold() -> None:
    rng = SeededTestRandom(1)
    threshold = SAMPLING_RATE_DENOMINATOR // 4
    scripted = _ScriptedGenerator([threshold - 1, threshold, 0, 999_999])
    rng._generator = scripted  # type: ignore[assignment]
    selector = PoissonParticipantSelector(0.25, mode=TEST, rng=rng)
    clients = ["a-1", "a-2", "a-3", "a-4"]

    roster = selector.select(
        round_index=0, eligible_pseudonyms=clients, disclosure=UNDISCLOSED
    ).roster

    assert roster.members() == ("a-1", "a-3")
    assert scripted.stops == [SAMPLING_RATE_DENOMINATOR] * 4


@pytest.mark.parametrize("rate", [0.1, 0.25, 0.000001, 0.999999, 1.0, 1])
def test_exactly_representable_rates_are_accepted(rate: float) -> None:
    assert PoissonParticipantSelector(rate).sampling_rate == float(rate)


@pytest.mark.parametrize("rate", [0.0000001, 0.1234567, 1 / 3])
def test_rates_off_the_fixed_denominator_are_rejected(rate: float) -> None:
    with _raises(E.INEXACT_RATE):
        PoissonParticipantSelector(rate)


@pytest.mark.parametrize(
    "rate", [0.0, -0.25, 1.5, math.nan, math.inf, True, "0.25", None]
)
def test_invalid_rates_are_rejected(rate: object) -> None:
    with _raises(E.INVALID_RATE):
        PoissonParticipantSelector(rate)  # type: ignore[arg-type]


# Operational mode refuses seeded or non-cryptographic randomness.


def test_operational_mode_defaults_to_system_randomness() -> None:
    selection = PoissonParticipantSelector(0.5).select(
        round_index=0, eligible_pseudonyms=_pseudonyms(5), disclosure=UNDISCLOSED
    )
    assert selection.declaration.selection_mode is OPERATIONAL


def test_operational_mode_accepts_secrets_system_random() -> None:
    PoissonParticipantSelector(0.5, rng=secrets.SystemRandom())


class _SubclassedSystemRandom(random.SystemRandom):
    def randrange(self, *args: object, **kwargs: object) -> int:  # type: ignore
        return 0


@pytest.mark.parametrize(
    ("rng", "code"),
    [
        (random.Random(SENTINEL_SEED), E.NON_CRYPTOGRAPHIC_RANDOMNESS),
        (_SubclassedSystemRandom(), E.NON_CRYPTOGRAPHIC_RANDOMNESS),
        (object(), E.NON_CRYPTOGRAPHIC_RANDOMNESS),
        (SeededTestRandom(SENTINEL_SEED), E.SEEDED_RANDOMNESS_REFUSED),
    ],
)
def test_operational_mode_refuses_seeded_or_non_cryptographic_randomness(
    rng: object, code: ParticipantSamplingErrorCode
) -> None:
    with _raises(code) as caught:
        PoissonParticipantSelector(0.25, rng=rng)  # type: ignore[arg-type]
    assert str(SENTINEL_SEED) not in str(caught.value)
    with _raises(code):
        declare_unsampled_round(
            round_index=0,
            procedure=P.OPERATOR_CHOSEN,
            eligible_pseudonyms=_pseudonyms(5),
            roster_pseudonyms=_pseudonyms(5),
            rng=rng,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "rng", [None, random.SystemRandom(), random.Random(3)], ids=str
)
def test_test_mode_requires_an_explicitly_seeded_test_source(rng: object) -> None:
    with _raises(E.TEST_RANDOMNESS_REQUIRED):
        PoissonParticipantSelector(0.25, mode=TEST, rng=rng)  # type: ignore[arg-type]


def test_seeded_test_selection_never_claims_amplification() -> None:
    declaration = (
        _test_selector(0.25)
        .select(
            round_index=0,
            eligible_pseudonyms=_pseudonyms(20),
            disclosure=UNDISCLOSED,
        )
        .declaration
    )

    assert declaration.selection_mode is TEST
    assert declaration.amplification_permitted is False
    assert declaration.accountant_sampling_rate == 1.0
    assert declaration.reason_codes == (A.SEEDED_TEST_RANDOMNESS,)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "7", 2**256])
def test_seeded_test_source_validates_its_seed(seed: object) -> None:
    with pytest.raises(ParticipantSamplingError, match="^invalid_seed: seed$"):
        SeededTestRandom(seed)  # type: ignore[arg-type]


def test_invalid_mode_is_rejected() -> None:
    with _raises(E.INVALID_MODE):
        PoissonParticipantSelector(0.25, mode="operational")  # type: ignore[arg-type]


def test_a_round_index_is_selected_at_most_once() -> None:
    selector = PoissonParticipantSelector(0.25)
    selector.select(
        round_index=4, eligible_pseudonyms=_pseudonyms(10), disclosure=UNDISCLOSED
    )
    with _raises(E.ROUND_ALREADY_SELECTED):
        selector.select(
            round_index=4,
            eligible_pseudonyms=_pseudonyms(10),
            disclosure=UNDISCLOSED,
        )


# Accountant input rates must match the selector's declaration.


def test_matching_accountant_rate_is_accepted() -> None:
    declaration = _operational_selection(round_index=9).declaration
    accountant = FakeAccountant()

    accountant.charge(declaration, FakeAccountantRound(9, 0.25))

    assert accountant.charged_rates == [0.25]


@pytest.mark.parametrize("rate", [0.2, 0.250001, 1.0, 0.125])
def test_accountant_rate_differing_from_declaration_is_rejected(rate: float) -> None:
    declaration = _operational_selection(round_index=9).declaration
    with _raises(E.ACCOUNTANT_RATE_MISMATCH):
        require_accountant_rate(declaration, FakeAccountantRound(9, rate))


def test_accountant_round_index_must_match_the_declaration() -> None:
    declaration = _operational_selection(round_index=9).declaration
    with _raises(E.ACCOUNTANT_ROUND_MISMATCH):
        require_accountant_rate(declaration, FakeAccountantRound(8, 0.25))


@pytest.mark.parametrize(
    "accountant_round",
    [
        object(),
        FakeAccountantRound(9, True),  # type: ignore[arg-type]
        FakeAccountantRound(9, math.nan),
        FakeAccountantRound(9, "0.25"),  # type: ignore[arg-type]
        FakeAccountantRound(True, 0.25),  # type: ignore[arg-type]
        FakeAccountantRound(-1, 0.25),
    ],
)
def test_malformed_accountant_input_is_rejected(accountant_round: object) -> None:
    declaration = _operational_selection(round_index=9).declaration
    with _raises(E.INVALID_ACCOUNTANT_INPUT):
        require_accountant_rate(declaration, accountant_round)  # type: ignore[arg-type]


def test_forged_declarations_are_refused() -> None:
    declaration = _operational_selection().declaration
    with _raises(E.UNVERIFIED_DECLARATION):
        dataclasses.replace(declaration, accountant_sampling_rate=0.01)
    with _raises(E.UNVERIFIED_DECLARATION):
        ParticipantSamplingDeclaration(
            **{  # type: ignore[arg-type]
                field.name: getattr(declaration, field.name)
                for field in dataclasses.fields(declaration)
            }
        )
    with _raises(E.UNVERIFIED_DECLARATION):
        require_accountant_rate(
            declaration.to_dict(),
            FakeAccountantRound(3, 0.25),  # type: ignore[arg-type]
        )


def test_batch_binding_accepts_matching_rounds() -> None:
    selector = PoissonParticipantSelector(0.5)
    declarations = [
        selector.select(
            round_index=index,
            eligible_pseudonyms=_pseudonyms(8),
            disclosure=UNDISCLOSED,
        ).declaration
        for index in range(3)
    ]
    rounds = [FakeAccountantRound(index, 0.5) for index in (2, 0, 1)]

    require_accountant_rates(declarations, rounds)


def test_batch_binding_rejects_the_whole_batch_on_one_mismatch() -> None:
    selector = PoissonParticipantSelector(0.5)
    declarations = [
        selector.select(
            round_index=index,
            eligible_pseudonyms=_pseudonyms(8),
            disclosure=UNDISCLOSED,
        ).declaration
        for index in range(3)
    ]
    rounds = [
        FakeAccountantRound(0, 0.5),
        FakeAccountantRound(1, 0.25),
        FakeAccountantRound(2, 0.5),
    ]
    with _raises(E.ACCOUNTANT_RATE_MISMATCH):
        require_accountant_rates(declarations, rounds)


def test_batch_binding_rejects_missing_or_duplicate_rounds() -> None:
    selector = PoissonParticipantSelector(0.5)
    first, second = (
        selector.select(
            round_index=index,
            eligible_pseudonyms=_pseudonyms(8),
            disclosure=UNDISCLOSED,
        ).declaration
        for index in range(2)
    )

    with _raises(E.MISSING_DECLARATION):
        require_accountant_rates([first], [FakeAccountantRound(1, 0.5)])
    with _raises(E.MISSING_ACCOUNTANT_ROUND):
        require_accountant_rates([first, second], [FakeAccountantRound(0, 0.5)])
    with _raises(E.DUPLICATE_DECLARATION):
        require_accountant_rates([first, first], [FakeAccountantRound(0, 0.5)])
    with _raises(E.DUPLICATE_ACCOUNTANT_ROUND):
        require_accountant_rates(
            [first], [FakeAccountantRound(0, 0.5), FakeAccountantRound(0, 0.5)]
        )
    with _raises(E.INVALID_ACCOUNTANT_INPUT):
        require_accountant_rates([first], "rounds")  # type: ignore[arg-type]


# Records contain no pseudonyms, realized roster or seeds.


def _sentinel_selection(mode: ParticipantSelectionMode = TEST):
    rng = SeededTestRandom(SENTINEL_SEED) if mode is TEST else None
    selector = PoissonParticipantSelector(0.5, mode=mode, rng=rng)
    selection = selector.select(
        round_index=1,
        eligible_pseudonyms=_pseudonyms(60, prefix=SENTINEL_PREFIX),
        disclosure=UNDISCLOSED,
    )
    return selector, rng, selection


@pytest.mark.parametrize("mode", [TEST, OPERATIONAL])
def test_records_carry_only_rate_band_and_commitment(
    mode: ParticipantSelectionMode,
) -> None:
    selector, rng, selection = _sentinel_selection(mode)
    declaration = selection.declaration
    payload = declaration.to_dict()

    assert set(payload) == {
        "accountant_sampling_rate",
        "amplification_permitted",
        "eligible_population_band",
        "procedure",
        "reason_codes",
        "roster_commitment",
        "round_index",
        "schema_version",
        "selection_mode",
        "selection_rate",
    }
    assert payload["schema_version"] == FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION
    assert payload["eligible_population_band"] == "50_to_99"
    assert payload["roster_commitment"].startswith("sha256:")
    assert json.loads(declaration.to_json()) == payload
    surfaces = [
        declaration.to_json(),
        repr(declaration),
        repr(selection),
        repr(selection.roster),
        repr(selector),
        repr(rng),
        str(selection),
    ]
    for surface in surfaces:
        assert SENTINEL_PREFIX not in surface
        assert str(SENTINEL_SEED) not in surface


def test_roster_stays_inside_the_coordinator_boundary() -> None:
    _, _, selection = _sentinel_selection()
    roster = selection.roster

    assert all(member.startswith(SENTINEL_PREFIX) for member in roster.members())
    assert roster.members()[0] in roster
    assert roster.matches_commitment(selection.declaration.roster_commitment)
    assert not roster.matches_commitment("sha256:" + "0" * 64)
    with pytest.raises(TypeError, match="^participant roster cannot be serialized$"):
        pickle.dumps(roster)
    with pytest.raises(TypeError, match="^participant roster cannot be serialized$"):
        pickle.dumps(selection)


def test_selection_pairs_only_a_declaration_with_its_own_roster() -> None:
    first = _operational_selection(round_index=1)
    second = _operational_selection(round_index=2)

    assert not second.roster.matches_commitment(first.declaration.roster_commitment)
    with _raises(E.UNVERIFIED_DECLARATION):
        ParticipantSelection(first.declaration, second.roster)
    with _raises(E.UNVERIFIED_DECLARATION):
        payload: object = first.declaration.to_dict()
        ParticipantSelection(payload, first.roster)  # type: ignore[arg-type]
    with _raises(E.UNVERIFIED_DECLARATION):
        ParticipantRoster(frozenset({"c-1"}), b"k" * 32, 0)


def test_roster_commitment_is_keyed_and_bound_to_the_round() -> None:
    clients = _pseudonyms(10)
    first = declare_unsampled_round(
        round_index=0,
        procedure=P.OPERATOR_CHOSEN,
        eligible_pseudonyms=clients,
        roster_pseudonyms=clients[:3],
    )
    second = declare_unsampled_round(
        round_index=0,
        procedure=P.OPERATOR_CHOSEN,
        eligible_pseudonyms=clients,
        roster_pseudonyms=clients[:3],
    )

    assert first.roster.members() == second.roster.members()
    assert first.declaration.roster_commitment != second.declaration.roster_commitment


def test_invalid_pseudonyms_are_rejected_without_echoing_them() -> None:
    selector = PoissonParticipantSelector(0.5)
    bad = "Jane Doe MRN 12345"
    with _raises(E.INVALID_PSEUDONYM) as caught:
        selector.select(
            round_index=0, eligible_pseudonyms=[bad], disclosure=UNDISCLOSED
        )
    assert "Jane" not in str(caught.value)
    assert "12345" not in str(caught.value)


@pytest.mark.parametrize("value", ["client-1", b"client-1", None, 12, ["a", 3]])
def test_eligible_population_must_be_a_collection_of_pseudonyms(
    value: object,
) -> None:
    selector = PoissonParticipantSelector(0.5)
    with pytest.raises(ParticipantSamplingError) as caught:
        selector.select(
            round_index=0,
            eligible_pseudonyms=value,  # type: ignore[arg-type]
            disclosure=UNDISCLOSED,
        )
    assert caught.value.code in {E.INVALID_POPULATION, E.INVALID_PSEUDONYM}


def test_duplicate_or_empty_populations_are_rejected() -> None:
    selector = PoissonParticipantSelector(0.5)
    with _raises(E.DUPLICATE_PSEUDONYM):
        selector.select(
            round_index=0,
            eligible_pseudonyms=["c-1", "c-1"],
            disclosure=UNDISCLOSED,
        )
    with _raises(E.INVALID_POPULATION):
        selector.select(round_index=1, eligible_pseudonyms=[], disclosure=UNDISCLOSED)


@pytest.mark.parametrize("round_index", [-1, True, 1.0, "1", 10**7])
def test_invalid_round_index_is_rejected(round_index: object) -> None:
    selector = PoissonParticipantSelector(0.5)
    with _raises(E.INVALID_ROUND_INDEX):
        selector.select(
            round_index=round_index,  # type: ignore[arg-type]
            eligible_pseudonyms=_pseudonyms(5),
            disclosure=UNDISCLOSED,
        )


@pytest.mark.parametrize(
    ("size", "band"),
    [
        (1, EligiblePopulationBand.UNDER_5),
        (4, EligiblePopulationBand.UNDER_5),
        (5, EligiblePopulationBand.FROM_5_TO_9),
        (10, EligiblePopulationBand.FROM_10_TO_24),
        (25, EligiblePopulationBand.FROM_25_TO_49),
        (99, EligiblePopulationBand.FROM_50_TO_99),
        (100, EligiblePopulationBand.FROM_100_TO_249),
        (250, EligiblePopulationBand.FROM_250_TO_999),
        (1000, EligiblePopulationBand.FROM_1000),
    ],
)
def test_eligible_population_is_reported_as_a_band(
    size: int, band: EligiblePopulationBand
) -> None:
    declaration = (
        PoissonParticipantSelector(0.5)
        .select(
            round_index=0,
            eligible_pseudonyms=_pseudonyms(size),
            disclosure=UNDISCLOSED,
        )
        .declaration
    )
    assert declaration.eligible_population_band is band


def test_errors_expose_stable_codes_and_field_names() -> None:
    with pytest.raises(ParticipantSamplingError) as caught:
        PoissonParticipantSelector(0.1234567)

    assert caught.value.code is E.INEXACT_RATE
    assert caught.value.field == "sampling_rate"
    assert str(caught.value) == "inexact_rate: sampling_rate"
    assert {code.value for code in AmplificationReasonCode} == {
        "poisson_selection_undisclosed",
        "full_participation_rate",
        "fixed_size_roster",
        "availability_driven_roster",
        "operator_chosen_roster",
        "ledger_discloses_participation",
        "status_discloses_participation",
        "seeded_test_randomness",
    }


@pytest.mark.integration
def test_status_driven_round_is_accounted_at_full_rate_end_to_end() -> None:
    """Cross-module flow: status disclosure -> selector -> fake accountant."""

    clients = _pseudonyms(30)
    selector = PoissonParticipantSelector(0.2, rng=random.SystemRandom())
    accountant = FakeAccountant()
    charged_declarations = []
    for round_index in range(4):
        status = build_federated_round_status(
            state=FederatedRoundState.COLLECTING,
            participant_count=12 if round_index % 2 else 0,
            completed_participant_count=0,
            required_quorum=1,
        )
        disclosure = ParticipationDisclosure.for_round_status(
            status, ledger_discloses_participation=False
        )
        declaration = selector.select(
            round_index=round_index,
            eligible_pseudonyms=clients,
            disclosure=disclosure,
        ).declaration
        accountant.charge(
            declaration,
            FakeAccountantRound(round_index, declaration.accountant_sampling_rate),
        )
        charged_declarations.append(declaration)

    assert accountant.charged_rates == [0.2, 1.0, 0.2, 1.0]
    require_accountant_rates(
        charged_declarations,
        [
            FakeAccountantRound(index, rate)
            for index, rate in enumerate(accountant.charged_rates)
        ],
    )
