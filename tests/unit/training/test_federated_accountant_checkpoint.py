"""Synthetic, offline contracts for composed privacy-accountant checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import pytest

from openmed.risk import budget_migration
from openmed.training import federated_accountant_checkpoint as checkpoint_module
from openmed.training.federated_accountant_checkpoint import (
    ACCOUNTANT_CHECKPOINT_REASON_CODES,
    FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION,
    FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION,
    RECONCILIATION_OUTCOMES,
    AccountantCheckpointError,
    AccountantCompositionParameters,
    AccountantMechanism,
    AccountantReceipt,
    CheckpointedAccountant,
    CheckpointIntegrityKey,
    InMemoryAccountantCheckpointStore,
    StoredAccountantCheckpoint,
)

# Sentinels that must never reach a receipt, checkpoint payload, repr or error.
SENTINEL_SITE = "site-SENTINEL-ALPHA-7731"
SENTINEL_SEED = 918273645
SENTINEL_CONTRIBUTION = 0.123456789123
SECRET = b"k" * 32
KEY = CheckpointIntegrityKey(key_id="synthetic-key-1", secret=SECRET)


def round_digest(round_index: int, attempt: int = 0) -> str:
    """Digest a synthetic round receipt that carries sentinel identities."""

    receipt = {
        "round_index": round_index,
        "attempt": attempt,
        "site_ids": [SENTINEL_SITE],
        "seed": SENTINEL_SEED,
        "contribution": SENTINEL_CONTRIBUTION,
    }
    encoded = json.dumps(receipt, sort_keys=True).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


class AdditiveAccountant:
    """Basic composition: the bound is the sum of per-round epsilons."""

    accountant_id = "additive-fake"
    accountant_version = "1"

    def compose(
        self,
        mechanisms: Sequence[AccountantMechanism],
        parameters: AccountantCompositionParameters,
    ) -> float:
        return math.fsum(m.parameters["epsilon"] for m in mechanisms)


class GaussianRdpAccountant:
    """RDP-style fake: per-order sums converted with the declared delta."""

    accountant_id = "rdp-fake"
    accountant_version = "1"

    def compose(
        self,
        mechanisms: Sequence[AccountantMechanism],
        parameters: AccountantCompositionParameters,
    ) -> float:
        best = math.inf
        for order in parameters.orders:
            rdp = math.fsum(
                order
                * m.parameters["sampling_rate"] ** 2
                / (m.parameters["noise_multiplier"] ** 2)
                for m in mechanisms
            )
            best = min(best, rdp + math.log(1.0 / parameters.delta) / (order - 1))
        return best


ADDITIVE_PARAMETERS = AccountantCompositionParameters(
    composition="basic", delta=1e-5, max_epsilon=10.0
)
RDP_PARAMETERS = AccountantCompositionParameters(
    composition="rdp", delta=1e-5, max_epsilon=50.0, orders=(2, 4, 8, 16, 32)
)


def additive_round(index: int):
    return "sum_release", {"epsilon": 0.5 + 0.1 * index}


def rdp_round(index: int):
    return "subsampled_gaussian", {
        "sampling_rate": 0.1,
        "noise_multiplier": 1.1,
        "clipping_norm": 1.0,
    }


ACCOUNTANT_CASES = [
    (AdditiveAccountant, ADDITIVE_PARAMETERS, additive_round),
    (GaussianRdpAccountant, RDP_PARAMETERS, rdp_round),
]


def make_genesis(store=None, accountant=None, parameters=None):
    store = InMemoryAccountantCheckpointStore() if store is None else store
    accountant = AdditiveAccountant() if accountant is None else accountant
    parameters = ADDITIVE_PARAMETERS if parameters is None else parameters
    session = CheckpointedAccountant.genesis(
        store=store,
        accountant=accountant,
        parameters=parameters,
        integrity_key=KEY,
        genesis_id="genesis-synthetic-1",
    )
    return store, session


def restore(store, accountant=None, parameters=None, key=KEY, witness=None):
    return CheckpointedAccountant.restore(
        store=store,
        accountant=AdditiveAccountant() if accountant is None else accountant,
        parameters=ADDITIVE_PARAMETERS if parameters is None else parameters,
        integrity_key=key,
        witness=witness,
    )


def run_round(session, index, maker=additive_round, attempt=0):
    mechanism, parameters = maker(index)
    session.begin_round(
        round_digest(index, attempt), mechanism=mechanism, parameters=parameters
    )
    return session.commit_round(round_digest(index, attempt))


def assert_code(excinfo, code):
    error = excinfo.value
    assert isinstance(error, AccountantCheckpointError)
    assert error.code == code
    assert error.code in ACCOUNTANT_CHECKPOINT_REASON_CODES
    assert str(error) == f"{error.code}: {error.field}"
    assert error.__cause__ is None
    # Raised ``from None`` inside handlers; never chained to a source error.
    assert error.__context__ is None or error.__suppress_context__ is True
    for sentinel in (SENTINEL_SITE, str(SENTINEL_SEED), str(SENTINEL_CONTRIBUTION)):
        assert sentinel not in str(error)
        assert sentinel not in repr(error)


def load_document(store) -> dict:
    return json.loads(store.load().payload)


def write_raw(store, document: Mapping, *, generation=None, reseal=True):
    """Force a document into the store, optionally resealing it with the key."""

    body = dict(document)
    if reseal:
        body = checkpoint_module._seal_document(body, KEY)
    payload = checkpoint_module._canonical_json(body)
    current = store.load()
    target = current.generation if generation is None else generation
    store._record = StoredAccountantCheckpoint(generation=target, payload=payload)


def rechain(events: list[dict]) -> list[dict]:
    """Recompute indices and chain digests to forge a coherent history."""

    previous = checkpoint_module._GENESIS_PREVIOUS_DIGEST
    rebuilt = []
    for index, event in enumerate(events):
        fresh = {k: v for k, v in event.items() if k != "chain_digest"}
        fresh["index"] = index
        previous = checkpoint_module._chain_digest(previous, fresh)
        fresh["chain_digest"] = previous
        rebuilt.append(fresh)
    return rebuilt


# --- vocabulary and genesis -------------------------------------------------


def test_reason_codes_and_outcomes_are_fixed_vocabularies():
    assert isinstance(ACCOUNTANT_CHECKPOINT_REASON_CODES, frozenset)
    assert {
        "checkpoint_missing",
        "genesis_conflict",
        "integrity_mismatch",
        "generation_conflict",
        "budget_regressed",
        "stale_checkpoint",
        "history_diverged",
        "uncertain_spend_unreconciled",
    } <= ACCOUNTANT_CHECKPOINT_REASON_CODES
    assert RECONCILIATION_OUTCOMES == ("committed", "abandoned")
    assert FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION.endswith(".v1")
    assert FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION.endswith(".v1")


def test_genesis_is_explicit_recorded_and_refuses_existing_state():
    store, session = make_genesis()
    receipt = session.receipt
    assert receipt.generation == 1
    assert receipt.transition == "genesis"
    assert receipt.charged_epsilon == 0.0
    document = load_document(store)
    assert document["events"][0]["kind"] == "genesis"
    assert document["events"][0]["genesis_id"] == "genesis-synthetic-1"
    with pytest.raises(AccountantCheckpointError) as excinfo:
        make_genesis(store=store)
    assert_code(excinfo, "genesis_conflict")
    assert store.load().generation == 1


def test_restore_never_starts_a_fresh_accountant_silently():
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(InMemoryAccountantCheckpointStore())
    assert_code(excinfo, "checkpoint_missing")


def test_restore_round_trips_a_valid_checkpoint_negative_control():
    store, session = make_genesis()
    for index in range(3):
        run_round(session, index)
    restored = restore(store, witness=session.receipt)
    assert restored.receipt.to_dict() == session.receipt.to_dict()
    assert restored.generation == session.generation == 7
    assert restored.charged_epsilon == pytest.approx(0.5 + 0.6 + 0.7)


# --- crash/restart equivalence ---------------------------------------------


@pytest.mark.parametrize(("accountant_cls", "parameters", "maker"), ACCOUNTANT_CASES)
@pytest.mark.parametrize("crash_at", [0, 2, 4])
@pytest.mark.parametrize("outcome", RECONCILIATION_OUTCOMES)
def test_interrupted_restart_is_equal_or_more_conservative(
    accountant_cls, parameters, maker, crash_at, outcome
):
    rounds = 5
    _, uninterrupted = make_genesis(accountant=accountant_cls(), parameters=parameters)
    for index in range(rounds):
        run_round(uninterrupted, index, maker)
    baseline = uninterrupted.charged_epsilon
    assert uninterrupted.committed_epsilon == baseline

    store, session = make_genesis(accountant=accountant_cls(), parameters=parameters)
    for index in range(crash_at):
        run_round(session, index, maker)
    mechanism, values = maker(crash_at)
    session.begin_round(round_digest(crash_at), mechanism=mechanism, parameters=values)
    del session  # crash: the round was reserved but never committed

    restarted = CheckpointedAccountant.restore(
        store=store,
        accountant=accountant_cls(),
        parameters=parameters,
        integrity_key=KEY,
    )
    assert restarted.uncertain_rounds == 1
    restarted.reconcile_round(round_digest(crash_at), outcome=outcome)
    start = crash_at + 1
    if outcome == "abandoned":
        start = crash_at  # retry the abandoned round under a fresh attempt digest
    for index in range(start, rounds):
        attempt = 1 if (outcome == "abandoned" and index == crash_at) else 0
        run_round(restarted, index, maker, attempt=attempt)

    if outcome == "committed":
        assert restarted.charged_epsilon == pytest.approx(baseline, rel=1e-12)
        assert restarted.charged_epsilon >= baseline - 1e-12
    else:
        assert restarted.charged_epsilon > baseline
        assert restarted.committed_epsilon == pytest.approx(baseline, rel=1e-12)


def test_uncertain_spend_is_charged_retained_and_blocks_new_rounds():
    store, session = make_genesis()
    run_round(session, 0)
    session.begin_round(
        round_digest(1), mechanism="sum_release", parameters={"epsilon": 0.6}
    )
    for _ in range(3):  # repeated crashes never drop the reservation
        session = restore(store)
        assert session.uncertain_rounds == 1
        assert session.charged_epsilon == pytest.approx(1.1)
        assert session.committed_epsilon == pytest.approx(0.5)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(2), mechanism="sum_release", parameters={"epsilon": 0.1}
        )
    assert_code(excinfo, "uncertain_spend_unreconciled")
    receipt = session.reconcile_round(round_digest(1), outcome="committed")
    assert receipt.uncertain_rounds == 0
    assert receipt.committed_rounds == 2
    assert session.committed_epsilon == pytest.approx(1.1)


def test_abandoned_reconciliation_keeps_the_charge():
    store, session = make_genesis()
    session.begin_round(
        round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
    )
    before = session.charged_epsilon
    receipt = session.reconcile_round(round_digest(0), outcome="abandoned")
    assert receipt.abandoned_rounds == 1
    assert receipt.charged_epsilon == before == 0.5
    assert receipt.committed_epsilon == 0.0
    assert restore(store).charged_epsilon == 0.5


def test_reconciliation_rejects_unknown_rounds_outcomes_and_reuse():
    _, session = make_genesis()
    session.begin_round(
        round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
    )
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.reconcile_round(round_digest(9), outcome="committed")
    assert_code(excinfo, "round_unknown")
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.reconcile_round(round_digest(0), outcome="released")
    assert_code(excinfo, "outcome_invalid")
    session.reconcile_round(round_digest(0), outcome="abandoned")
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.commit_round(round_digest(0))
    assert_code(excinfo, "round_unknown")
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
        )
    assert_code(excinfo, "round_duplicate")


def test_reservation_beyond_the_limit_is_refused_without_writing():
    store, session = make_genesis()
    session.begin_round(
        round_digest(0), mechanism="sum_release", parameters={"epsilon": 10.0}
    )
    session.commit_round(round_digest(0))
    generation = store.load().generation
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(1), mechanism="sum_release", parameters={"epsilon": 0.01}
        )
    assert_code(excinfo, "budget_exceeded")
    assert store.load().generation == generation


# --- refused restores ---------------------------------------------------------


def test_older_checkpoint_payload_in_the_store_is_refused():
    store, session = make_genesis()
    run_round(session, 0)
    old_payload = store.load().payload
    run_round(session, 1)
    store._record = StoredAccountantCheckpoint(
        generation=store.load().generation, payload=old_payload
    )
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "generation_mismatch")


def test_rolled_back_store_is_refused_against_a_witness():
    store, session = make_genesis()
    run_round(session, 0)
    backup = store.load()
    run_round(session, 1)
    witness = session.receipt
    store._record = backup  # restore an older backup wholesale
    restore(store)  # without a witness the backup is internally consistent
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, witness=witness)
    assert_code(excinfo, "stale_checkpoint")


def test_wiped_store_and_new_genesis_cannot_reset_spent_budget():
    store, session = make_genesis()
    run_round(session, 0)
    witness = session.receipt
    wiped = InMemoryAccountantCheckpointStore()
    _, fresh = make_genesis(store=wiped)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(wiped, witness=witness)
    assert_code(excinfo, "stale_checkpoint")
    for index in range(5):
        run_round(fresh, 10 + index, lambda _i: ("sum_release", {"epsilon": 0.0}))
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(wiped, witness=witness)
    assert_code(excinfo, "history_diverged")


def test_truncated_history_is_refused():
    store, session = make_genesis()
    for index in range(3):
        run_round(session, index)
    witness = session.receipt
    document = load_document(store)
    truncated = dict(document, events=document["events"][:-2])

    write_raw(store, truncated, reseal=False)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "integrity_mismatch")

    write_raw(store, truncated)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "history_truncated")

    events = document["events"][:-2]
    forged = dict(
        document,
        events=events,
        generation=len(events),
        head_digest=events[-1]["chain_digest"],
        uncertain_rounds=[],
        committed_boundary={
            "abandoned_rounds": 0,
            "committed_rounds": 2,
            "chain_digest": events[-1]["chain_digest"],
            "event_index": len(events) - 1,
        },
        bound={"charged_epsilon": 1.1, "committed_epsilon": 1.1},
    )
    write_raw(store, forged, generation=len(events))
    restore(store)  # coherent but shorter: only a witness can detect it
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, witness=witness)
    assert_code(excinfo, "stale_checkpoint")


def test_reordered_history_is_refused():
    store, session = make_genesis()
    for index in range(3):
        run_round(session, index)
    witness = session.receipt
    document = load_document(store)
    events = list(document["events"])
    events[1], events[3] = events[3], events[1]
    reordered = dict(document, events=events)

    write_raw(store, reordered, reseal=False)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "integrity_mismatch")

    write_raw(store, reordered)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "history_out_of_order")

    reindexed = [dict(event, index=i) for i, event in enumerate(events)]
    write_raw(store, dict(document, events=reindexed))
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "history_chain_broken")

    # Swap two whole reserve/reconcile pairs and forge a coherent chain.
    pairs = list(document["events"])
    swapped = [pairs[0], pairs[3], pairs[4], pairs[1], pairs[2], *pairs[5:]]
    forged_events = rechain(swapped)
    forged = dict(
        document,
        events=forged_events,
        head_digest=forged_events[-1]["chain_digest"],
        committed_boundary=dict(
            document["committed_boundary"],
            chain_digest=forged_events[-1]["chain_digest"],
        ),
    )
    write_raw(store, forged)
    restore(store)  # same multiset of rounds: bound is unchanged
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, witness=witness)
    assert_code(excinfo, "history_diverged")


def test_tampered_fields_are_refused():
    store, session = make_genesis()
    run_round(session, 0)
    document = load_document(store)
    lowered = dict(document, bound={"charged_epsilon": 0.0, "committed_epsilon": 0.0})
    write_raw(store, lowered, reseal=False)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "integrity_mismatch")

    write_raw(store, document, reseal=False)
    events = document["events"]
    inflated = dict(
        document,
        events=[
            events[0],
            dict(events[1], parameters={"epsilon": 0.1}),
            events[2],
        ],
    )
    write_raw(store, inflated)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "history_chain_broken")


def test_recorded_bound_above_the_recomputed_bound_is_refused_by_the_verifier(
    monkeypatch,
):
    calls = []
    original = budget_migration.enforce_budget_migration

    def spy(before, after):
        calls.append((before, after))
        return original(before, after)

    monkeypatch.setattr(budget_migration, "enforce_budget_migration", spy)
    store, session = make_genesis()
    run_round(session, 0)
    assert calls, "transitions must run the budget-migration verifier"
    assert all(
        isinstance(item, budget_migration.BudgetLedgerSnapshot)
        for pair in calls
        for item in pair
    )
    document = load_document(store)
    write_raw(
        store, dict(document, bound={"charged_epsilon": 5.0, "committed_epsilon": 5.0})
    )
    calls.clear()
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "budget_regressed")
    assert calls


def test_drifting_accountant_cannot_lower_spent_budget():
    class Discounting(AdditiveAccountant):
        def compose(self, mechanisms, parameters):
            return super().compose(mechanisms, parameters) / 2

    class Inflating(AdditiveAccountant):
        def compose(self, mechanisms, parameters):
            return super().compose(mechanisms, parameters) * 2

    store, session = make_genesis()
    run_round(session, 0)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, accountant=Discounting())
    assert_code(excinfo, "budget_regressed")
    conservative = restore(store, accountant=Inflating())
    assert conservative.charged_epsilon == 1.0


def test_non_monotone_accountant_is_refused_on_transition():
    class Forgetful(AdditiveAccountant):
        def compose(self, mechanisms, parameters):
            return 0.5 if len(mechanisms) < 2 else 0.1

    store, session = make_genesis(accountant=Forgetful())
    run_round(session, 0)
    generation = store.load().generation
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(1), mechanism="sum_release", parameters={"epsilon": 0.5}
        )
    assert_code(excinfo, "budget_regressed")
    assert store.load().generation == generation


def test_wrong_key_accountant_or_parameters_are_refused():
    store, session = make_genesis()
    run_round(session, 0)
    other_secret = CheckpointIntegrityKey(key_id="synthetic-key-1", secret=b"x" * 32)
    other_id = CheckpointIntegrityKey(key_id="synthetic-key-2", secret=SECRET)
    for key in (other_secret, other_id):
        with pytest.raises(AccountantCheckpointError) as excinfo:
            restore(store, key=key)
        assert_code(excinfo, "integrity_mismatch")

    class OtherVersion(AdditiveAccountant):
        accountant_version = "2"

    class OtherAccountant(AdditiveAccountant):
        accountant_id = "other-fake"

    for accountant in (OtherVersion(), OtherAccountant()):
        with pytest.raises(AccountantCheckpointError) as excinfo:
            restore(store, accountant=accountant)
        assert_code(excinfo, "accountant_mismatch")

    for parameters in (
        AccountantCompositionParameters(
            composition="basic", delta=1e-6, max_epsilon=10.0
        ),
        AccountantCompositionParameters(
            composition="basic", delta=1e-5, max_epsilon=20.0
        ),
        AccountantCompositionParameters(
            composition="basic", delta=1e-5, max_epsilon=10.0, orders=(2, 4)
        ),
    ):
        with pytest.raises(AccountantCheckpointError) as excinfo:
            restore(store, parameters=parameters)
        assert_code(excinfo, "parameters_mismatch")


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("{not json", "checkpoint_malformed"),
        ('{"a":1,"a":2}', "checkpoint_malformed"),
        ('{"a":NaN}', "checkpoint_malformed"),
        ("[]", "checkpoint_malformed"),
    ],
)
def test_malformed_payloads_are_refused(payload, code):
    store, _ = make_genesis()
    store._record = StoredAccountantCheckpoint(generation=1, payload=payload)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, code)


def test_schema_and_canonical_form_are_enforced():
    store, _ = make_genesis()
    document = load_document(store)
    write_raw(store, dict(document, schema_version="openmed.other.v9"))
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "schema_unsupported")

    write_raw(store, document, reseal=False)
    pretty = json.dumps(load_document(store), indent=2)
    store._record = StoredAccountantCheckpoint(generation=1, payload=pretty)
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store)
    assert_code(excinfo, "checkpoint_malformed")


def test_witness_must_match_the_accountant():
    store, session = make_genesis()
    run_round(session, 0)
    _, other = make_genesis(
        accountant=GaussianRdpAccountant(), parameters=RDP_PARAMETERS
    )
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, witness=other.receipt)
    assert_code(excinfo, "witness_invalid")
    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(store, witness={"generation": 1})
    assert_code(excinfo, "witness_invalid")
    restored = restore(store, witness=session.receipt.to_dict())
    assert restored.generation == session.generation


# --- fencing and store failures -----------------------------------------------


def test_concurrent_writers_are_fenced_by_generation():
    store, _ = make_genesis()
    first = restore(store)
    second = restore(store)
    first.begin_round(
        round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
    )
    winner = store.load()
    with pytest.raises(AccountantCheckpointError) as excinfo:
        second.begin_round(
            round_digest(1), mechanism="sum_release", parameters={"epsilon": 0.1}
        )
    assert_code(excinfo, "generation_conflict")
    assert store.load() == winner
    with pytest.raises(AccountantCheckpointError) as excinfo:
        second.commit_round(round_digest(0))
    assert_code(excinfo, "restore_required")
    refreshed = restore(store)
    assert refreshed.uncertain_rounds == 1
    refreshed.commit_round(round_digest(0))
    with pytest.raises(AccountantCheckpointError) as excinfo:
        first.commit_round(round_digest(0))
    assert_code(excinfo, "generation_conflict")


def test_in_memory_store_compare_and_swap_semantics():
    store = InMemoryAccountantCheckpointStore()
    assert store.load() is None
    record = StoredAccountantCheckpoint(generation=1, payload="{}")
    assert store.compare_and_swap(expected_generation=0, record=record) is True
    assert store.compare_and_swap(expected_generation=0, record=record) is False
    skip = StoredAccountantCheckpoint(generation=3, payload="{}")
    assert store.compare_and_swap(expected_generation=1, record=skip) is False
    assert store.load() == record


class FlakyStore(InMemoryAccountantCheckpointStore):
    """Store that fails after or before applying the next write."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_after_write = False
        self.fail_before_write = False

    def compare_and_swap(self, *, expected_generation, record):
        if self.fail_before_write:
            self.fail_before_write = False
            raise OSError(SENTINEL_SITE)
        result = super().compare_and_swap(
            expected_generation=expected_generation, record=record
        )
        if self.fail_after_write:
            self.fail_after_write = False
            raise OSError(SENTINEL_SITE)
        return result


@pytest.mark.parametrize("landed", [True, False])
def test_ambiguous_store_failure_requires_restore_and_stays_conservative(landed):
    store = FlakyStore()
    _, session = make_genesis(store=store)
    if landed:
        store.fail_after_write = True
    else:
        store.fail_before_write = True
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
        )
    assert_code(excinfo, "store_failure")
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.commit_round(round_digest(0))
    assert_code(excinfo, "restore_required")
    restored = restore(store)
    assert restored.uncertain_rounds == (1 if landed else 0)
    assert restored.charged_epsilon == (0.5 if landed else 0.0)


def test_store_load_failure_is_a_stable_code():
    class Broken(InMemoryAccountantCheckpointStore):
        def load(self):
            raise OSError(SENTINEL_SITE)

    with pytest.raises(AccountantCheckpointError) as excinfo:
        restore(Broken())
    assert_code(excinfo, "store_failure")


@pytest.mark.parametrize(
    "result", [RuntimeError(SENTINEL_SITE), math.nan, -1.0, True, "1.0", math.inf]
)
def test_accountant_failures_are_stable_codes(result):
    class Faulty(AdditiveAccountant):
        def compose(self, mechanisms, parameters):
            if isinstance(result, Exception):
                raise result
            return result

    _, session = make_genesis(accountant=Faulty())
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(0), mechanism="sum_release", parameters={"epsilon": 0.5}
        )
    assert_code(excinfo, "accountant_failure")


# --- receipts expose no identities, seeds or contributions --------------------


@pytest.mark.parametrize(
    "parameters",
    [
        {"seed": SENTINEL_SEED},
        {"random_seed": float(SENTINEL_SEED)},
        {"site_id": SENTINEL_SITE},
        {"client_contribution": SENTINEL_CONTRIBUTION},
        {"epsilon": SENTINEL_SITE},
        {"epsilon": math.nan},
        {"epsilon": True},
        {},
    ],
)
def test_mechanism_parameters_refuse_identities_seeds_and_contributions(parameters):
    _, session = make_genesis()
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(0), mechanism="sum_release", parameters=parameters
        )
    assert_code(excinfo, "mechanism_invalid")


@pytest.mark.parametrize("mechanism", [SENTINEL_SITE, "site_gaussian", "", 7])
def test_mechanism_kind_refuses_identity_like_values(mechanism):
    _, session = make_genesis()
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(
            round_digest(0), mechanism=mechanism, parameters={"epsilon": 0.5}
        )
    assert_code(excinfo, "mechanism_invalid")


@pytest.mark.parametrize("digest", [SENTINEL_SITE, "sha256:" + "A" * 64, None])
def test_round_references_must_be_opaque_digests(digest):
    _, session = make_genesis()
    with pytest.raises(AccountantCheckpointError) as excinfo:
        session.begin_round(digest, mechanism="sum_release", parameters={"epsilon": 1})
    assert_code(excinfo, "identifier_invalid")


def test_sentinels_never_reach_receipts_payloads_or_reprs():
    store, session = make_genesis(
        accountant=GaussianRdpAccountant(), parameters=RDP_PARAMETERS
    )
    receipts = [session.receipt]
    for index in range(3):
        mechanism, values = rdp_round(index)
        receipts.append(
            session.begin_round(
                round_digest(index), mechanism=mechanism, parameters=values
            )
        )
        receipts.append(session.commit_round(round_digest(index)))
    texts = [store.load().payload, repr(session), repr(store.load()), repr(KEY)]
    for receipt in receipts:
        texts.extend([receipt.to_json(), repr(receipt)])
    for text in texts:
        for sentinel in (
            SENTINEL_SITE,
            str(SENTINEL_SEED),
            str(SENTINEL_CONTRIBUTION),
            SECRET.decode("ascii"),
        ):
            assert sentinel not in text
    receipt_text = receipts[-1].to_json()
    assert round_digest(0) not in receipt_text  # receipts omit round references


def test_receipt_shape_is_fixed_and_round_trips():
    _, session = make_genesis()
    receipt = run_round(session, 0)
    payload = receipt.to_dict()
    assert set(payload) == {
        "schema_version",
        "accountant_id",
        "accountant_version",
        "composition",
        "parameters_fingerprint",
        "generation",
        "transition",
        "head_digest",
        "checkpoint_digest",
        "committed_rounds",
        "abandoned_rounds",
        "uncertain_rounds",
        "charged_epsilon",
        "committed_epsilon",
        "remaining_epsilon",
        "delta",
        "max_epsilon",
    }
    assert payload["schema_version"] == FEDERATED_ACCOUNTANT_RECEIPT_SCHEMA_VERSION
    assert AccountantReceipt.from_dict(payload) == receipt
    assert json.loads(receipt.to_json()) == payload
    with pytest.raises(AccountantCheckpointError) as excinfo:
        AccountantReceipt.from_dict(dict(payload, extra=1))
    assert_code(excinfo, "witness_invalid")


def test_checkpoint_payload_is_canonical_versioned_json():
    store, session = make_genesis(
        accountant=GaussianRdpAccountant(), parameters=RDP_PARAMETERS
    )
    mechanism, values = rdp_round(0)
    session.begin_round(round_digest(0), mechanism=mechanism, parameters=values)
    payload = store.load().payload
    document = json.loads(payload)
    assert payload == json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    assert payload.isascii()
    assert document["schema_version"] == FEDERATED_ACCOUNTANT_CHECKPOINT_SCHEMA_VERSION
    assert document["accountant"] == {
        "accountant_id": "rdp-fake",
        "accountant_version": "1",
    }
    assert document["parameters"]["orders"] == [2, 4, 8, 16, 32]
    assert document["generation"] == 2 == len(document["events"])
    assert document["uncertain_rounds"] == [round_digest(0)]
    assert document["committed_boundary"]["committed_rounds"] == 0
    assert document["integrity"]["algorithm"] == "hmac-sha256"
    assert document["integrity"]["key_id"] == "synthetic-key-1"
    assert [event["kind"] for event in document["events"]] == ["genesis", "reserve"]


# --- value validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"composition": "basic", "delta": 0.0, "max_epsilon": 1.0},
        {"composition": "basic", "delta": 1.0, "max_epsilon": 1.0},
        {"composition": "basic", "delta": 1e-5, "max_epsilon": 0.0},
        {"composition": "basic", "delta": 1e-5, "max_epsilon": math.inf},
        {"composition": "Basic-1", "delta": 1e-5, "max_epsilon": 1.0},
        {"composition": "rdp", "delta": 1e-5, "max_epsilon": 1.0, "orders": (4, 2)},
        {"composition": "rdp", "delta": 1e-5, "max_epsilon": 1.0, "orders": (1,)},
        {"composition": "rdp", "delta": 1e-5, "max_epsilon": 1.0, "orders": (2.5,)},
    ],
)
def test_composition_parameters_are_validated(kwargs):
    with pytest.raises(AccountantCheckpointError) as excinfo:
        AccountantCompositionParameters(**kwargs)
    assert_code(excinfo, "parameters_invalid")


@pytest.mark.parametrize(
    ("key_id", "secret"),
    [("synthetic-key-1", b"short"), ("synthetic-key-1", "k" * 32), ("", SECRET)],
)
def test_integrity_key_is_validated_and_hidden(key_id, secret):
    with pytest.raises(AccountantCheckpointError) as excinfo:
        CheckpointIntegrityKey(key_id=key_id, secret=secret)
    assert_code(excinfo, "integrity_key_invalid")
    assert SECRET.decode("ascii") not in repr(KEY)
    assert "synthetic-key-1" in repr(KEY)


def test_genesis_identifier_is_validated():
    with pytest.raises(AccountantCheckpointError) as excinfo:
        CheckpointedAccountant.genesis(
            store=InMemoryAccountantCheckpointStore(),
            accountant=AdditiveAccountant(),
            parameters=ADDITIVE_PARAMETERS,
            integrity_key=KEY,
            genesis_id=SENTINEL_SITE,
        )
    assert_code(excinfo, "identifier_invalid")
