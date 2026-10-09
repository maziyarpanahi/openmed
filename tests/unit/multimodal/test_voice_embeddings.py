"""Synthetic biometric privacy negative controls, without providers or weights."""

import copy
import gc
import json
import math
import pickle
import weakref
from array import array
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pytest

from openmed.multimodal.voice_embeddings import (
    VoiceEmbeddingError,
    VoiceEmbeddingHandle,
    VoiceEmbeddingSession,
)


def refused(code, action):
    with pytest.raises(VoiceEmbeddingError) as error:
        action()
    assert error.value.reason_code == code
    assert str(error.value) == code
    assert error.value.__context__ is None


def test_similarity_is_local_non_diagnostic_and_never_identity():
    owner = VoiceEmbeddingSession()
    a, b, c = [owner.add(v) for v in ([1.0, 0.0], [0.0, 1.0], [-1.0, 0.0])]
    assert a.similarity(a).score == 1
    assert a.similarity(b).score == 0
    assert a.similarity(c).score == -1
    assert a.similarity(b).notice == "non_diagnostic_voice_similarity"
    assert a.similarity(b).reviewer_confirmation_required is True
    huge = owner.add([1e308, 1e308])
    assert math.isclose(huge.similarity(huge).score, 1)
    refused("embedding_dimensions_mismatch", lambda: a.similarity(owner.add([1])))


@pytest.mark.parametrize(
    "boundary", ["pause", "withdrawal", "cancellation", "finalization"]
)
def test_boundaries_erase_even_with_allocator_aliases_and_retained_handles(boundary):
    buffers = []

    def allocator(size):
        buffer = array("d", [0.0]) * size
        buffers.append(buffer)
        return buffer

    owner = VoiceEmbeddingSession(allocator=allocator)
    handle = owner.add([123.25, -456.75])
    receipt = owner.destroy(boundary)
    assert receipt.reason_code == boundary
    assert receipt.destroyed_count == 1
    assert receipt.handle_digests == (handle.to_evidence()["handle_digest"],)
    assert all(len(buffer) == 0 for buffer in buffers)
    assert owner.diagnostics() == {"handle_count": 0, "handle_digests": ()}
    refused("embedding_destroyed", lambda: handle.similarity(handle))
    refused("embedding_session_closed", lambda: owner.add([1]))
    assert owner.destroy(boundary).destroyed_count == 0


@pytest.mark.parametrize(
    "boundary", ["pause", "withdrawal", "cancellation", "finalization"]
)
def test_allocations_unreachable_after_end(boundary):
    allocations = []

    def allocator(size):
        buffer = array("d", [0.0]) * size
        allocations.append(weakref.ref(buffer))
        return buffer

    callbacks = []
    owner = VoiceEmbeddingSession(
        allocator=allocator, register=lambda d, cb: callbacks.append(cb)
    )
    handle = owner.add([1, 2])
    owner.destroy(boundary)
    gc.collect()
    assert all(ref() is None for ref in allocations)
    assert callbacks[0]().destroyed_count == 0
    assert handle.to_evidence()


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_serialization_and_copy_fail(protocol):
    owner = VoiceEmbeddingSession()
    handle = owner.add([827.125, -934.375])
    for value in (handle, owner):
        refused(
            "embedding_serialization_refused", lambda: pickle.dumps(value, protocol)
        )
        refused("embedding_serialization_refused", lambda: copy.copy(value))
        refused("embedding_serialization_refused", lambda: copy.deepcopy(value))
        with pytest.raises(TypeError):
            json.dumps(value)
    refused("embedding_persistence_refused", handle.persist)


def test_evidence_repr_and_diagnostics_contain_only_counts_and_random_digests(caplog):
    owner = VoiceEmbeddingSession()
    handle = owner.add([827.125, -934.375])
    second = owner.add([827.125, -934.375])
    assert handle.to_evidence() != second.to_evidence()
    assert set(handle.to_evidence()) == {"handle_digest"}
    digest = handle.to_evidence()["handle_digest"]
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    assert set(owner.diagnostics()) == {"handle_count", "handle_digests"}
    assert not hasattr(handle, "__dict__")
    safe = json.dumps(
        [handle.to_evidence(), owner.diagnostics(), asdict(owner.destroy())]
    )
    for value in ("827.125", "934.375"):
        assert value not in safe + repr(handle) + repr(owner) + caplog.text
    assert caplog.text == ""


def test_cross_session_refusal_even_after_owners_are_destroyed():
    a, b = VoiceEmbeddingSession(), VoiceEmbeddingSession()
    left, right = a.add([1]), b.add([1])
    refused("embedding_cross_session_refused", lambda: left.similarity(right))
    a.destroy()
    b.destroy()
    del a, b
    gc.collect()
    refused("embedding_cross_session_refused", lambda: left.similarity(right))
    refused("embedding_destroyed", lambda: left.similarity(left))


def test_handles_and_callbacks_do_not_keep_session_or_buffers_alive():
    callbacks, allocations = [], []

    def allocate(size):
        buffer = array("d", [0.0]) * size
        allocations.append(weakref.ref(buffer))
        return buffer

    owner = VoiceEmbeddingSession(
        allocator=allocate, register=lambda d, cb: callbacks.append(cb)
    )
    owner_ref = weakref.ref(owner)
    handle = owner.add([1, 2])
    del owner
    gc.collect()
    assert owner_ref() is None
    assert allocations[0]() is None
    assert callbacks[0]().destroyed_count == 0
    refused("embedding_destroyed", lambda: handle.similarity(handle))


def test_failed_registration_drops_owned_buffers_and_sanitizes_failure():
    buffers = []

    def allocate(size):
        result = array("d", [0]) * size
        buffers.append(result)
        return result

    def fail(digest, erase):
        raise RuntimeError("synthetic-private-payload")

    owner = VoiceEmbeddingSession(allocator=allocate, register=fail)
    refused("embedding_registration_failed", lambda: owner.add([123.5]))
    assert buffers == [array("d")]
    assert owner.diagnostics()["handle_count"] == 0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [0, 0],
        [float("nan")],
        [float("inf")],
        [10**400],
        [True],
        ["private"],
        object(),
    ],
)
def test_invalid_input_is_value_free(values):
    owner = VoiceEmbeddingSession()
    refused("embedding_vector_invalid", lambda: owner.add(values))
    assert owner.diagnostics()["handle_count"] == 0


def test_invalid_allocator_and_reuse_are_refused_without_destroying_existing_handle():
    buffer = array("d", [0])
    owner = VoiceEmbeddingSession(allocator=lambda size: buffer)
    handle = owner.add([1])
    refused("embedding_allocator_failed", lambda: owner.add([2]))
    assert handle.similarity(handle).score == 1
    owner.destroy()
    bad = array("d", [123])
    owner = VoiceEmbeddingSession(allocator=lambda size: bad)
    refused("embedding_allocator_failed", lambda: owner.add([1]))
    assert len(bad) == 0


def test_bounds_and_reentrant_retention_erasure():
    owner = VoiceEmbeddingSession(max_handles=1, max_dimensions=2)
    owner.add([1])
    refused("embedding_capacity_exceeded", lambda: owner.add([2]))
    refused("embedding_limits_invalid", lambda: VoiceEmbeddingSession(max_handles=True))
    refused(
        "embedding_vector_invalid",
        lambda: VoiceEmbeddingSession(max_dimensions=1).add([1, 2]),
    )
    owner = VoiceEmbeddingSession(register=lambda digest, erase: erase())
    refused("embedding_destroyed", lambda: owner.add([1]))
    assert owner.diagnostics()["handle_count"] == 0


def test_concurrent_similarity_and_destruction_fail_closed():
    owner = VoiceEmbeddingSession()
    handle = owner.add([1, 2, 3])

    def compare(_):
        try:
            return handle.similarity(handle).score
        except VoiceEmbeddingError as error:
            return error.reason_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = [pool.submit(compare, i) for i in range(50)]
        owner.destroy("cancellation")
        assert all(
            job.result() == pytest.approx(1) or job.result() == "embedding_destroyed"
            for job in jobs
        )
    assert owner.diagnostics()["handle_count"] == 0


def test_unregistered_handle_cannot_put_caller_payload_in_repr():
    owner = VoiceEmbeddingSession()
    refused(
        "embedding_handle_invalid",
        lambda: VoiceEmbeddingHandle(owner, "synthetic-private-payload"),
    )


def test_rng_and_allocator_failures_never_leave_sensitive_buffers(monkeypatch):
    def fail(*args):
        raise RuntimeError("synthetic-private-payload")

    owner = VoiceEmbeddingSession(allocator=fail)
    refused("embedding_allocator_failed", lambda: owner.add([123.5]))
    monkeypatch.setattr("openmed.multimodal.voice_embeddings.secrets.token_bytes", fail)
    refused("embedding_allocator_failed", lambda: owner.add([123.5]))
    assert owner.diagnostics()["handle_count"] == 0
