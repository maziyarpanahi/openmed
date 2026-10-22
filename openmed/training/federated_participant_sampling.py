"""Secret Poisson participant selection bound to privacy accounting.

Subsampling amplification lowers the privacy cost of a federated round only
when every eligible client was included independently at the declared rate
``q`` and nobody outside the coordinator learns which clients took part. This
module owns that selection step and the declaration handed to the accountant:

* :class:`PoissonParticipantSelector` includes each eligible pseudonymous client
  independently with probability ``q`` using an injected cryptographic random
  source. The realized roster stays in a :class:`ParticipantRoster` held by the
  coordinator; it has no serialized form and a redacted ``repr``.
* :class:`ParticipantSamplingDeclaration` is the only exported record. It
  carries the round index, procedure, mode, selection rate, the rate the
  accountant must charge, controlled reason codes, an eligible-population band
  and a keyed roster commitment digest. It never carries pseudonyms, the
  realized roster, roster size or seeds.
* :func:`require_accountant_rate` and :func:`require_accountant_rates` reject
  accountant inputs whose ``sampling_rate`` differs from the declaration.

Amplification is permitted only for operational-mode Poisson selection with a
rate below one and no per-round participation disclosure. Fixed-size,
availability-driven and operator-chosen rosters, seeded test selection and any
round whose contribution ledger or status records disclose participation are
accounted at ``q = 1``. No alternative amplification bound is supported.

Inclusion is decided exactly: a rate is accepted only when it equals
``k / SAMPLING_RATE_DENOMINATOR`` for an integer ``k``, and a client is included
when a uniform integer drawn from ``[0, SAMPLING_RATE_DENOMINATOR)`` is below
``k``. The inclusion probability is therefore exactly ``k / 1_000_000`` with no
floating-point comparison.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import random
import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any, Final, NoReturn, Protocol

from .federated_status import FederatedRoundStatus

FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION: Final = (
    "openmed.training.federated_participant_sampling.v1"
)
SAMPLING_RATE_DENOMINATOR: Final = 1_000_000
MAX_ELIGIBLE_CLIENTS: Final = 100_000
MAX_ROUND_INDEX: Final = 1_000_000

_COMMITMENT_DOMAIN: Final = b"openmed.training.federated_participant_roster.v1\x00"
_COMMITMENT_KEY_BYTES: Final = 32
_MAX_TEST_SEED: Final = 2**256
_MAX_ACCOUNTANT_ROUNDS: Final = MAX_ROUND_INDEX + 1
_PSEUDONYM: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_DIGEST: Final = re.compile(r"sha256:[0-9a-f]{64}\Z")
_BUILDER_TOKEN: Final = object()


class ParticipantSelectionProcedure(str, Enum):
    """How the participants of one federated round were chosen."""

    POISSON = "poisson"
    FIXED_SIZE = "fixed_size"
    AVAILABILITY_DRIVEN = "availability_driven"
    OPERATOR_CHOSEN = "operator_chosen"

    def __str__(self) -> str:
        return self.value


class ParticipantSelectionMode(str, Enum):
    """Whether selection used operational or seeded test randomness."""

    OPERATIONAL = "operational"
    TEST = "test"

    def __str__(self) -> str:
        return self.value


class EligiblePopulationBand(str, Enum):
    """A coarse band for the number of eligible enrolled clients."""

    UNDER_5 = "under_5"
    FROM_5_TO_9 = "5_to_9"
    FROM_10_TO_24 = "10_to_24"
    FROM_25_TO_49 = "25_to_49"
    FROM_50_TO_99 = "50_to_99"
    FROM_100_TO_249 = "100_to_249"
    FROM_250_TO_999 = "250_to_999"
    FROM_1000 = "1000_or_more"

    def __str__(self) -> str:
        return self.value


_POPULATION_BANDS: Final[tuple[tuple[int, EligiblePopulationBand], ...]] = (
    (1000, EligiblePopulationBand.FROM_1000),
    (250, EligiblePopulationBand.FROM_250_TO_999),
    (100, EligiblePopulationBand.FROM_100_TO_249),
    (50, EligiblePopulationBand.FROM_50_TO_99),
    (25, EligiblePopulationBand.FROM_25_TO_49),
    (10, EligiblePopulationBand.FROM_10_TO_24),
    (5, EligiblePopulationBand.FROM_5_TO_9),
)


class AmplificationReasonCode(str, Enum):
    """Stable reasons recorded with every sampling declaration."""

    POISSON_SELECTION_UNDISCLOSED = "poisson_selection_undisclosed"
    FULL_PARTICIPATION_RATE = "full_participation_rate"
    FIXED_SIZE_ROSTER = "fixed_size_roster"
    AVAILABILITY_DRIVEN_ROSTER = "availability_driven_roster"
    OPERATOR_CHOSEN_ROSTER = "operator_chosen_roster"
    LEDGER_DISCLOSES_PARTICIPATION = "ledger_discloses_participation"
    STATUS_DISCLOSES_PARTICIPATION = "status_discloses_participation"
    SEEDED_TEST_RANDOMNESS = "seeded_test_randomness"

    def __str__(self) -> str:
        return self.value


_REASON_ORDER: Final[tuple[AmplificationReasonCode, ...]] = tuple(
    AmplificationReasonCode
)
_UNSAMPLED_REASONS: Final = {
    ParticipantSelectionProcedure.FIXED_SIZE: (
        AmplificationReasonCode.FIXED_SIZE_ROSTER
    ),
    ParticipantSelectionProcedure.AVAILABILITY_DRIVEN: (
        AmplificationReasonCode.AVAILABILITY_DRIVEN_ROSTER
    ),
    ParticipantSelectionProcedure.OPERATOR_CHOSEN: (
        AmplificationReasonCode.OPERATOR_CHOSEN_ROSTER
    ),
}


class ParticipantSamplingErrorCode(str, Enum):
    """Fixed failure codes; error text never echoes caller values."""

    INVALID_RATE = "invalid_rate"
    INEXACT_RATE = "inexact_rate"
    INVALID_MODE = "invalid_mode"
    INVALID_PROCEDURE = "invalid_procedure"
    INVALID_ROUND_INDEX = "invalid_round_index"
    ROUND_ALREADY_SELECTED = "round_already_selected"
    INVALID_POPULATION = "invalid_population"
    INVALID_PSEUDONYM = "invalid_pseudonym"
    DUPLICATE_PSEUDONYM = "duplicate_pseudonym"
    ROSTER_NOT_ELIGIBLE = "roster_not_eligible"
    INVALID_DISCLOSURE = "invalid_disclosure"
    INVALID_SEED = "invalid_seed"
    NON_CRYPTOGRAPHIC_RANDOMNESS = "non_cryptographic_randomness"
    SEEDED_RANDOMNESS_REFUSED = "seeded_randomness_refused"
    TEST_RANDOMNESS_REQUIRED = "test_randomness_required"
    RANDOMNESS_FAILED = "randomness_failed"
    UNVERIFIED_DECLARATION = "unverified_declaration"
    INVALID_ACCOUNTANT_INPUT = "invalid_accountant_input"
    ACCOUNTANT_ROUND_MISMATCH = "accountant_round_mismatch"
    ACCOUNTANT_RATE_MISMATCH = "accountant_rate_mismatch"
    MISSING_DECLARATION = "missing_declaration"
    DUPLICATE_DECLARATION = "duplicate_declaration"
    MISSING_ACCOUNTANT_ROUND = "missing_accountant_round"
    DUPLICATE_ACCOUNTANT_ROUND = "duplicate_accountant_round"

    def __str__(self) -> str:
        return self.value


class ParticipantSamplingError(ValueError):
    """Raised with a fixed code and field name when selection fails closed.

    Attributes:
        code: The stable failure code.
        field: The name of the offending argument or record field.
    """

    def __init__(self, code: ParticipantSamplingErrorCode, field: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code.value}: {field}")


def _fail(code: ParticipantSamplingErrorCode, field: str) -> NoReturn:
    raise ParticipantSamplingError(code, field)


class SeededTestRandom:
    """Deterministic randomness for synthetic tests only.

    Operational selection refuses this source. It exists so tests can assert
    exact rosters and inclusion statistics. The seed is never stored on the
    instance in readable form and never appears in its ``repr``.

    Args:
        seed: A non-negative integer below ``2**256``.

    Raises:
        ParticipantSamplingError: If ``seed`` is not such an integer.
    """

    __slots__ = ("_generator",)

    def __init__(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed < _MAX_TEST_SEED:
            _fail(ParticipantSamplingErrorCode.INVALID_SEED, "seed")
        self._generator = random.Random(seed)

    def randrange(self, stop: int) -> int:
        """Return a uniform integer in ``[0, stop)``."""

        return self._generator.randrange(stop)

    def randbytes(self, size: int) -> bytes:
        """Return ``size`` deterministic bytes."""

        return self._generator.randbytes(size)

    def __repr__(self) -> str:
        return "SeededTestRandom(<test-only>)"

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("seeded test randomness cannot be serialized")


class _RandomSource(Protocol):
    def randrange(self, stop: int) -> int: ...

    def randbytes(self, size: int) -> bytes: ...


def _resolve_random_source(
    mode: object, rng: object
) -> tuple[ParticipantSelectionMode, _RandomSource]:
    if not isinstance(mode, ParticipantSelectionMode):
        _fail(ParticipantSamplingErrorCode.INVALID_MODE, "mode")
    if mode is ParticipantSelectionMode.TEST:
        if type(rng) is not SeededTestRandom:
            _fail(ParticipantSamplingErrorCode.TEST_RANDOMNESS_REQUIRED, "rng")
        return mode, rng
    if rng is None:
        return mode, random.SystemRandom()
    if isinstance(rng, SeededTestRandom):
        _fail(ParticipantSamplingErrorCode.SEEDED_RANDOMNESS_REFUSED, "rng")
    # Only the stdlib OS-backed generator is accepted. Subclasses can override
    # draws, and plain ``random.Random`` instances are seedable Mersenne
    # Twister generators, so both are refused.
    if type(rng) is not random.SystemRandom:
        _fail(ParticipantSamplingErrorCode.NON_CRYPTOGRAPHIC_RANDOMNESS, "rng")
    return mode, rng


@dataclass(frozen=True)
class ParticipationDisclosure:
    """Whether a round's other records disclose per-round participation.

    Args:
        ledger_discloses_participation: True when contribution-ledger receipts
            or similar per-client records reveal who took part in the round.
        status_discloses_participation: True when round status records release
            participant counts for the round.

    Raises:
        ParticipantSamplingError: If either flag is not a ``bool``.
    """

    ledger_discloses_participation: bool
    status_discloses_participation: bool

    def __post_init__(self) -> None:
        for name in (
            "ledger_discloses_participation",
            "status_discloses_participation",
        ):
            if type(getattr(self, name)) is not bool:
                _fail(ParticipantSamplingErrorCode.INVALID_DISCLOSURE, name)

    @classmethod
    def for_round_status(
        cls,
        status: FederatedRoundStatus,
        *,
        ledger_discloses_participation: bool,
    ) -> ParticipationDisclosure:
        """Derive disclosure from a built federated round status.

        A status that releases an exact participant or completed-participant
        count discloses per-round participation. Counts suppressed below the
        minimum group size do not.

        Args:
            status: A status returned by ``build_federated_round_status``.
            ledger_discloses_participation: Whether the contribution ledger
                discloses participation for the same round.

        Returns:
            The combined disclosure declaration.

        Raises:
            ParticipantSamplingError: If ``status`` is not a built status or
                the ledger flag is not a ``bool``.
        """

        if type(status) is not FederatedRoundStatus:
            _fail(ParticipantSamplingErrorCode.INVALID_DISCLOSURE, "status")
        released = (
            status.participant_count is not None
            or status.completed_participant_count is not None
        )
        return cls(
            ledger_discloses_participation=ledger_discloses_participation,
            status_discloses_participation=released,
        )


class ParticipantRoster:
    """The realized roster, kept inside the coordinator trust boundary.

    Instances are created only by the selector or
    :func:`declare_unsampled_round`. They have no serialized form, refuse
    pickling and redact their ``repr``. The keyed commitment published in the
    declaration can be checked here without revealing members.
    """

    __slots__ = ("_commitment_key", "_members", "_round_index")

    def __init__(
        self,
        members: frozenset[str],
        commitment_key: bytes,
        round_index: int,
        *,
        _builder_token: object = None,
    ) -> None:
        if _builder_token is not _BUILDER_TOKEN:
            _fail(ParticipantSamplingErrorCode.UNVERIFIED_DECLARATION, "roster")
        self._members = members
        self._commitment_key = commitment_key
        self._round_index = round_index

    def __contains__(self, pseudonym: object) -> bool:
        return pseudonym in self._members

    def __len__(self) -> int:
        return len(self._members)

    def members(self) -> tuple[str, ...]:
        """Return the selected pseudonyms in sorted order.

        Returns:
            The realized roster. Callers must keep it inside the coordinator.
        """

        return tuple(sorted(self._members))

    def matches_commitment(self, digest: str) -> bool:
        """Return whether ``digest`` commits to this roster and round.

        Args:
            digest: A ``sha256:`` roster commitment from a declaration.

        Returns:
            True when the digest matches, compared in constant time.
        """

        if type(digest) is not str or _DIGEST.fullmatch(digest) is None:
            return False
        return hmac.compare_digest(self._commitment(), digest)

    def _commitment(self) -> str:
        message = _COMMITMENT_DOMAIN + json.dumps(
            {"members": sorted(self._members), "round_index": self._round_index},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        mac = hmac.new(self._commitment_key, message, hashlib.sha256)
        return "sha256:" + mac.hexdigest()

    def __repr__(self) -> str:
        return "ParticipantRoster(<within coordinator boundary>)"

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("participant roster cannot be serialized")


@dataclass(frozen=True)
class ParticipantSamplingDeclaration:
    """The exported sampling declaration consumed by privacy accounting.

    Build it with :class:`PoissonParticipantSelector` or
    :func:`declare_unsampled_round`; direct construction is refused.

    Attributes:
        round_index: The federated round this declaration covers.
        procedure: How participants were chosen.
        selection_mode: Operational or seeded test randomness.
        selection_rate: The Poisson rate applied, or ``1.0`` when unsampled.
        accountant_sampling_rate: The rate the accountant must charge.
        amplification_permitted: Whether subsampling amplification applies.
        reason_codes: Controlled reasons in canonical order.
        eligible_population_band: Coarse size band of the eligible clients.
        roster_commitment: Keyed ``sha256:`` commitment to the realized roster.
        schema_version: The declaration schema version.
    """

    round_index: int
    procedure: ParticipantSelectionProcedure
    selection_mode: ParticipantSelectionMode
    selection_rate: float
    accountant_sampling_rate: float
    amplification_permitted: bool
    reason_codes: tuple[AmplificationReasonCode, ...]
    eligible_population_band: EligiblePopulationBand
    roster_commitment: str
    schema_version: str = FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION
    _builder_token: InitVar[object | None] = None

    def __post_init__(self, _builder_token: object | None) -> None:
        if _builder_token is not _BUILDER_TOKEN:
            _fail(ParticipantSamplingErrorCode.UNVERIFIED_DECLARATION, "declaration")
        expected_permitted = self.reason_codes == (
            AmplificationReasonCode.POISSON_SELECTION_UNDISCLOSED,
        )
        expected_rate = self.selection_rate if expected_permitted else 1.0
        if (
            self.amplification_permitted is not expected_permitted
            or self.accountant_sampling_rate != expected_rate
            or self.schema_version != FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION
            or _DIGEST.fullmatch(self.roster_commitment) is None
        ):
            _fail(ParticipantSamplingErrorCode.UNVERIFIED_DECLARATION, "declaration")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic mapping with no pseudonyms or seeds.

        Returns:
            The declaration fields as JSON-compatible values.
        """

        return {
            "accountant_sampling_rate": self.accountant_sampling_rate,
            "amplification_permitted": self.amplification_permitted,
            "eligible_population_band": self.eligible_population_band.value,
            "procedure": self.procedure.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "roster_commitment": self.roster_commitment,
            "round_index": self.round_index,
            "schema_version": self.schema_version,
            "selection_mode": self.selection_mode.value,
            "selection_rate": self.selection_rate,
        }

    def to_json(self) -> str:
        """Return byte-stable JSON with a trailing newline.

        Returns:
            The serialized declaration.
        """

        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class ParticipantSelection:
    """A declaration paired with the roster it commits to.

    Only :attr:`declaration` may leave the coordinator. :attr:`roster` stays
    inside the coordinator trust boundary.

    Attributes:
        declaration: The exportable sampling declaration.
        roster: The realized roster held by the coordinator.
    """

    __slots__ = ("declaration", "roster")

    def __init__(
        self,
        declaration: ParticipantSamplingDeclaration,
        roster: ParticipantRoster,
    ) -> None:
        if (
            type(declaration) is not ParticipantSamplingDeclaration
            or type(roster) is not ParticipantRoster
            or not roster.matches_commitment(declaration.roster_commitment)
        ):
            _fail(ParticipantSamplingErrorCode.UNVERIFIED_DECLARATION, "selection")
        self.declaration = declaration
        self.roster = roster

    def __repr__(self) -> str:
        return (
            f"ParticipantSelection(declaration={self.declaration!r}, "
            "roster=<within coordinator boundary>)"
        )

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("participant roster cannot be serialized")


class PoissonParticipantSelector:
    """Select round participants by secret Poisson sampling at rate ``q``.

    Each eligible client is included independently: the selector draws one
    uniform integer from ``[0, SAMPLING_RATE_DENOMINATOR)`` per client, in
    sorted pseudonym order, and includes the client when the draw is below
    ``q * SAMPLING_RATE_DENOMINATOR``. Each round index can be selected once,
    so a coordinator cannot redraw a round until it gets a preferred roster.

    Args:
        sampling_rate: The rate ``q`` in ``(0, 1]``. It must equal ``k / 10**6``
            for an integer ``k`` so inclusion is decided exactly.
        mode: ``OPERATIONAL`` (default) or ``TEST``.
        rng: The injected random source. Operational mode accepts only a
            ``random.SystemRandom`` instance and defaults to a new one. Test
            mode requires a :class:`SeededTestRandom`.

    Raises:
        ParticipantSamplingError: If the rate, mode or random source is not
            acceptable.
    """

    def __init__(
        self,
        sampling_rate: float,
        *,
        mode: ParticipantSelectionMode = ParticipantSelectionMode.OPERATIONAL,
        rng: random.SystemRandom | SeededTestRandom | None = None,
    ) -> None:
        self._numerator = _rate_numerator(sampling_rate)
        self._mode, self._rng = _resolve_random_source(mode, rng)
        self._selected_rounds: set[int] = set()

    @property
    def sampling_rate(self) -> float:
        """The declared Poisson rate ``q``."""

        return self._numerator / SAMPLING_RATE_DENOMINATOR

    @property
    def mode(self) -> ParticipantSelectionMode:
        """Whether this selector uses operational or test randomness."""

        return self._mode

    def select(
        self,
        *,
        round_index: int,
        eligible_pseudonyms: Collection[str],
        disclosure: ParticipationDisclosure,
    ) -> ParticipantSelection:
        """Select one round's participants and declare its accounting rate.

        Args:
            round_index: The round index, from 0 to ``MAX_ROUND_INDEX``.
            eligible_pseudonyms: Unique pseudonyms of eligible enrolled clients.
            disclosure: Whether other round records disclose participation.

        Returns:
            The declaration and the coordinator-only roster.

        Raises:
            ParticipantSamplingError: If any input is invalid, the round was
                already selected, or the random source misbehaves.
        """

        index = _round_index(round_index)
        if type(disclosure) is not ParticipationDisclosure:
            _fail(ParticipantSamplingErrorCode.INVALID_DISCLOSURE, "disclosure")
        eligible = _pseudonym_set(eligible_pseudonyms, "eligible_pseudonyms")
        if index in self._selected_rounds:
            _fail(ParticipantSamplingErrorCode.ROUND_ALREADY_SELECTED, "round_index")
        self._selected_rounds.add(index)

        if self._numerator == SAMPLING_RATE_DENOMINATOR:
            members = eligible
        else:
            members = frozenset(
                pseudonym
                for pseudonym in sorted(eligible)
                if self._draw() < self._numerator
            )
        reasons = self._reasons(disclosure)
        return _build_selection(
            round_index=index,
            procedure=ParticipantSelectionProcedure.POISSON,
            mode=self._mode,
            selection_rate=self.sampling_rate,
            reasons=reasons,
            eligible_count=len(eligible),
            members=members,
            rng=self._rng,
        )

    def _draw(self) -> int:
        draw = self._rng.randrange(SAMPLING_RATE_DENOMINATOR)
        if type(draw) is not int or not 0 <= draw < SAMPLING_RATE_DENOMINATOR:
            _fail(ParticipantSamplingErrorCode.RANDOMNESS_FAILED, "rng")
        return draw

    def _reasons(
        self, disclosure: ParticipationDisclosure
    ) -> set[AmplificationReasonCode]:
        reasons: set[AmplificationReasonCode] = set()
        if self._numerator == SAMPLING_RATE_DENOMINATOR:
            reasons.add(AmplificationReasonCode.FULL_PARTICIPATION_RATE)
        if disclosure.ledger_discloses_participation:
            reasons.add(AmplificationReasonCode.LEDGER_DISCLOSES_PARTICIPATION)
        if disclosure.status_discloses_participation:
            reasons.add(AmplificationReasonCode.STATUS_DISCLOSES_PARTICIPATION)
        if self._mode is ParticipantSelectionMode.TEST:
            reasons.add(AmplificationReasonCode.SEEDED_TEST_RANDOMNESS)
        if not reasons:
            reasons.add(AmplificationReasonCode.POISSON_SELECTION_UNDISCLOSED)
        return reasons

    def __repr__(self) -> str:
        return (
            f"PoissonParticipantSelector(sampling_rate={self.sampling_rate!r}, "
            f"mode={self._mode.value!r})"
        )

    def __reduce_ex__(self, protocol: object) -> NoReturn:
        raise TypeError("participant selector cannot be serialized")


def declare_unsampled_round(
    *,
    round_index: int,
    procedure: ParticipantSelectionProcedure,
    eligible_pseudonyms: Collection[str],
    roster_pseudonyms: Collection[str],
    mode: ParticipantSelectionMode = ParticipantSelectionMode.OPERATIONAL,
    rng: random.SystemRandom | SeededTestRandom | None = None,
) -> ParticipantSelection:
    """Declare a fixed-size, availability-driven or operator-chosen roster.

    Such rosters are not secret independent samples, so the declaration is
    always accounted at ``q = 1`` and never permits amplification. The random
    source is used only for the roster commitment key.

    Args:
        round_index: The round index, from 0 to ``MAX_ROUND_INDEX``.
        procedure: Any procedure except ``POISSON``.
        eligible_pseudonyms: Unique pseudonyms of eligible enrolled clients.
        roster_pseudonyms: The chosen roster, a subset of the eligible clients.
        mode: ``OPERATIONAL`` (default) or ``TEST``.
        rng: The injected random source, validated as for the selector.

    Returns:
        The declaration and the coordinator-only roster.

    Raises:
        ParticipantSamplingError: If any input is invalid or the roster is not
            drawn from the eligible clients.
    """

    index = _round_index(round_index)
    if (
        not isinstance(procedure, ParticipantSelectionProcedure)
        or procedure not in _UNSAMPLED_REASONS
    ):
        _fail(ParticipantSamplingErrorCode.INVALID_PROCEDURE, "procedure")
    resolved_mode, source = _resolve_random_source(mode, rng)
    eligible = _pseudonym_set(eligible_pseudonyms, "eligible_pseudonyms")
    members = _pseudonym_set(roster_pseudonyms, "roster_pseudonyms")
    if not members <= eligible:
        _fail(ParticipantSamplingErrorCode.ROSTER_NOT_ELIGIBLE, "roster_pseudonyms")
    reasons = {_UNSAMPLED_REASONS[procedure]}
    if resolved_mode is ParticipantSelectionMode.TEST:
        reasons.add(AmplificationReasonCode.SEEDED_TEST_RANDOMNESS)
    return _build_selection(
        round_index=index,
        procedure=procedure,
        mode=resolved_mode,
        selection_rate=1.0,
        reasons=reasons,
        eligible_count=len(eligible),
        members=members,
        rng=source,
    )


class AccountantRoundInput(Protocol):
    """The fields an accountant round input must expose for binding checks."""

    round_index: int
    sampling_rate: float


def require_accountant_rate(
    declaration: ParticipantSamplingDeclaration,
    accountant_round: AccountantRoundInput,
) -> None:
    """Reject an accountant round whose rate differs from the declaration.

    Args:
        declaration: A declaration built by this module.
        accountant_round: Any accountant input exposing ``round_index`` and
            ``sampling_rate``.

    Raises:
        ParticipantSamplingError: If the declaration is not verified, the
            accountant input is malformed, or its round or rate differ.
    """

    _require_declaration(declaration)
    index, rate = _accountant_fields(accountant_round)
    _bind(declaration, index, rate)


def require_accountant_rates(
    declarations: Sequence[ParticipantSamplingDeclaration],
    accountant_rounds: Sequence[AccountantRoundInput],
) -> None:
    """Reject a whole accountant schedule unless every round matches.

    Every accountant round needs exactly one declaration for its index, and
    every declared round must be charged exactly once.

    Args:
        declarations: Declarations built by this module.
        accountant_rounds: Accountant inputs exposing ``round_index`` and
            ``sampling_rate``.

    Raises:
        ParticipantSamplingError: On any malformed, missing, duplicate or
            mismatched round. Nothing is accepted partially.
    """

    declared: dict[int, ParticipantSamplingDeclaration] = {}
    for declaration in _bounded_sequence(declarations, "declarations"):
        _require_declaration(declaration)
        if declaration.round_index in declared:
            _fail(ParticipantSamplingErrorCode.DUPLICATE_DECLARATION, "declarations")
        declared[declaration.round_index] = declaration
    charged: dict[int, float] = {}
    for accountant_round in _bounded_sequence(accountant_rounds, "accountant_rounds"):
        index, rate = _accountant_fields(accountant_round)
        if index in charged:
            _fail(
                ParticipantSamplingErrorCode.DUPLICATE_ACCOUNTANT_ROUND,
                "accountant_rounds",
            )
        charged[index] = rate
    for index, rate in charged.items():
        if index not in declared:
            _fail(ParticipantSamplingErrorCode.MISSING_DECLARATION, "declarations")
        _bind(declared[index], index, rate)
    if set(declared) - set(charged):
        _fail(
            ParticipantSamplingErrorCode.MISSING_ACCOUNTANT_ROUND, "accountant_rounds"
        )


def _bind(declaration: ParticipantSamplingDeclaration, index: int, rate: float) -> None:
    if index != declaration.round_index:
        _fail(ParticipantSamplingErrorCode.ACCOUNTANT_ROUND_MISMATCH, "round_index")
    if rate != declaration.accountant_sampling_rate:
        _fail(ParticipantSamplingErrorCode.ACCOUNTANT_RATE_MISMATCH, "sampling_rate")


def _require_declaration(declaration: object) -> None:
    if type(declaration) is not ParticipantSamplingDeclaration:
        _fail(ParticipantSamplingErrorCode.UNVERIFIED_DECLARATION, "declaration")


def _accountant_fields(accountant_round: object) -> tuple[int, float]:
    try:
        index = getattr(accountant_round, "round_index")
        rate = getattr(accountant_round, "sampling_rate")
    except Exception:
        _fail(ParticipantSamplingErrorCode.INVALID_ACCOUNTANT_INPUT, "accountant_round")
    if type(index) is not int or not 0 <= index <= MAX_ROUND_INDEX:
        _fail(ParticipantSamplingErrorCode.INVALID_ACCOUNTANT_INPUT, "round_index")
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 1:
        _fail(ParticipantSamplingErrorCode.INVALID_ACCOUNTANT_INPUT, "sampling_rate")
    return index, float(rate)


def _bounded_sequence(value: object, field: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _fail(ParticipantSamplingErrorCode.INVALID_ACCOUNTANT_INPUT, field)
    if len(value) > _MAX_ACCOUNTANT_ROUNDS:
        _fail(ParticipantSamplingErrorCode.INVALID_ACCOUNTANT_INPUT, field)
    return tuple(value)


def _rate_numerator(value: object) -> int:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not 0 < value <= 1
    ):
        _fail(ParticipantSamplingErrorCode.INVALID_RATE, "sampling_rate")
    numerator = round(value * SAMPLING_RATE_DENOMINATOR)
    if numerator < 1 or numerator / SAMPLING_RATE_DENOMINATOR != value:
        _fail(ParticipantSamplingErrorCode.INEXACT_RATE, "sampling_rate")
    return numerator


def _round_index(value: object) -> int:
    if type(value) is not int or not 0 <= value <= MAX_ROUND_INDEX:
        _fail(ParticipantSamplingErrorCode.INVALID_ROUND_INDEX, "round_index")
    return value


def _pseudonym_set(value: object, field: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        _fail(ParticipantSamplingErrorCode.INVALID_POPULATION, field)
    seen: set[str] = set()
    for pseudonym in value:
        if len(seen) >= MAX_ELIGIBLE_CLIENTS:
            _fail(ParticipantSamplingErrorCode.INVALID_POPULATION, field)
        if type(pseudonym) is not str or _PSEUDONYM.fullmatch(pseudonym) is None:
            _fail(ParticipantSamplingErrorCode.INVALID_PSEUDONYM, field)
        if pseudonym in seen:
            _fail(ParticipantSamplingErrorCode.DUPLICATE_PSEUDONYM, field)
        seen.add(pseudonym)
    if not seen and field == "eligible_pseudonyms":
        _fail(ParticipantSamplingErrorCode.INVALID_POPULATION, field)
    return frozenset(seen)


def _population_band(count: int) -> EligiblePopulationBand:
    for lower, band in _POPULATION_BANDS:
        if count >= lower:
            return band
    return EligiblePopulationBand.UNDER_5


def _build_selection(
    *,
    round_index: int,
    procedure: ParticipantSelectionProcedure,
    mode: ParticipantSelectionMode,
    selection_rate: float,
    reasons: set[AmplificationReasonCode],
    eligible_count: int,
    members: frozenset[str],
    rng: _RandomSource,
) -> ParticipantSelection:
    key = rng.randbytes(_COMMITMENT_KEY_BYTES)
    if type(key) is not bytes or len(key) != _COMMITMENT_KEY_BYTES:
        _fail(ParticipantSamplingErrorCode.RANDOMNESS_FAILED, "rng")
    roster = ParticipantRoster(members, key, round_index, _builder_token=_BUILDER_TOKEN)
    ordered = tuple(code for code in _REASON_ORDER if code in reasons)
    permitted = ordered == (AmplificationReasonCode.POISSON_SELECTION_UNDISCLOSED,)
    declaration = ParticipantSamplingDeclaration(
        round_index=round_index,
        procedure=procedure,
        selection_mode=mode,
        selection_rate=selection_rate,
        accountant_sampling_rate=selection_rate if permitted else 1.0,
        amplification_permitted=permitted,
        reason_codes=ordered,
        eligible_population_band=_population_band(eligible_count),
        roster_commitment=roster._commitment(),
        _builder_token=_BUILDER_TOKEN,
    )
    return ParticipantSelection(declaration, roster)


__all__ = [
    "FEDERATED_PARTICIPANT_SAMPLING_SCHEMA_VERSION",
    "MAX_ELIGIBLE_CLIENTS",
    "MAX_ROUND_INDEX",
    "SAMPLING_RATE_DENOMINATOR",
    "AccountantRoundInput",
    "AmplificationReasonCode",
    "EligiblePopulationBand",
    "ParticipantRoster",
    "ParticipantSamplingDeclaration",
    "ParticipantSamplingError",
    "ParticipantSamplingErrorCode",
    "ParticipantSelection",
    "ParticipantSelectionMode",
    "ParticipantSelectionProcedure",
    "ParticipationDisclosure",
    "PoissonParticipantSelector",
    "SeededTestRandom",
    "declare_unsampled_round",
    "require_accountant_rate",
    "require_accountant_rates",
]
