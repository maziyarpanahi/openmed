"""Offline embedding-owned retention hook integration, not an audio policy."""

from array import array

import pytest

from openmed.multimodal.voice_embeddings import (
    VoiceEmbeddingError,
    VoiceEmbeddingSession,
)


@pytest.mark.integration
def test_host_retention_callbacks_destroy_only_registered_embedding_artifacts():
    callbacks, buffers = {}, []

    def allocate(size):
        buffer = array("d", [0]) * size
        buffers.append(buffer)
        return buffer

    owner = VoiceEmbeddingSession(
        allocator=allocate,
        register=lambda digest, erase: callbacks.update({digest: erase}),
    )
    handles = [owner.add([1, 0]), owner.add([0, 1])]
    assert handles[0].similarity(handles[1]).score == 0
    assert set(callbacks) == set(owner.diagnostics()["handle_digests"])
    receipts = [erase() for erase in callbacks.values()]
    assert sum(r.destroyed_count for r in receipts) == 2
    assert all(r.reason_code == "embedding_destroyed" for r in receipts)
    assert all(len(b) == 0 for b in buffers)
    for handle in handles:
        with pytest.raises(VoiceEmbeddingError, match="^embedding_destroyed$"):
            handle.similarity(handle)
    assert owner.destroy("withdrawal").destroyed_count == 0
    assert all(erase().destroyed_count == 0 for erase in callbacks.values())
