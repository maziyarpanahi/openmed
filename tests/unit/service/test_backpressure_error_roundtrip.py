"""Round-trip only trusted, locally constructed synthetic exceptions."""

import copy
import pickle

import pytest

from openmed.service.backpressure import AdmissionQueue, BackpressureError


class DerivedBackpressureError(BackpressureError):
    pass


def make_error(error_type=BackpressureError):
    return error_type(
        priority="batch",
        queue_depth=8,
        queue_capacity=8,
        retry_after_seconds=0.25,
        queue_name="synthetic",
        low_watermark=2,
        max_wait_ms=250,
        reason="high_watermark",
    )


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_preserves_type_message_and_details(protocol):
    original = make_error()
    restored = pickle.loads(pickle.dumps(original, protocol=protocol))
    assert type(restored) is type(original)
    assert restored.args == original.args
    assert str(restored) == str(original)
    assert restored.to_details() == original.to_details()


@pytest.mark.parametrize("operation", [copy.copy, copy.deepcopy])
def test_copy_preserves_structured_exception(operation):
    original = make_error()
    restored = operation(original)
    assert restored is not original
    assert restored.to_details() == original.to_details()
    assert restored.args == original.args


def test_extra_state_and_notes_are_preserved():
    original = make_error()
    original.extra = {"synthetic": [1, 2]}
    if hasattr(original, "add_note"):
        original.add_note("synthetic note")
    restored = copy.deepcopy(original)
    assert restored.__dict__ == original.__dict__
    assert restored.extra is not original.extra


def test_compatible_subclass_roundtrips():
    original = make_error(DerivedBackpressureError)
    restored = pickle.loads(pickle.dumps(original))
    assert type(restored) is DerivedBackpressureError
    assert restored.to_details() == original.to_details()


def test_optional_fields_can_be_absent():
    original = BackpressureError(
        priority="batch", queue_depth=1, queue_capacity=1, retry_after_seconds=-2
    )
    restored = pickle.loads(pickle.dumps(original))
    assert restored.to_details() == original.to_details()
    assert restored.retry_after_seconds == 0
    assert "low_watermark" not in restored.to_details()


def test_queue_hysteresis_is_unchanged():
    queue = AdmissionQueue(
        queue_name="synthetic", high_watermark=2, low_watermark=0, max_wait_ms=25
    )
    queue.admit(priority="batch")
    queue.admit(priority="batch")
    with pytest.raises(BackpressureError):
        queue.admit(priority="batch")
    queue.release()
    assert queue.snapshot().shedding
    queue.release()
    assert not queue.snapshot().shedding
    queue.admit(priority="batch")
    assert queue.snapshot().depth == 1
