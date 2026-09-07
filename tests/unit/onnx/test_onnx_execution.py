"""Native RunOptions cancellation, scope isolation and deadline regressions."""

import threading
import time

import pytest

from openmed.onnx.execution import OnnxExecutionCancelled, OnnxExecutionControl


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_invalid_deadlines_are_rejected(timeout):
    with pytest.raises(ValueError):
        OnnxExecutionControl(timeout_seconds=timeout)


def test_cancelled_scope_has_safe_error_before_any_native_work():
    control = OnnxExecutionControl()
    control.cancel()
    with pytest.raises(OnnxExecutionCancelled, match="cancelled"):
        control.check()


def test_deadline_is_absolute_across_multiple_checks(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("openmed.onnx.execution.time.monotonic", lambda: now[0])
    control = OnnxExecutionControl(timeout_seconds=3)
    control.check()
    now[0] = 102
    control.check()
    now[0] = 103
    with pytest.raises(TimeoutError):
        control.check()


@pytest.mark.parametrize("mode", ["cancel", "deadline"])
def test_active_call_receives_native_termination_and_watchdog_is_joined(mode):
    pytest.importorskip("onnxruntime")
    control = OnnxExecutionControl(timeout_seconds=0.1 if mode == "deadline" else 3)
    options_seen = []

    class Session:
        def run(self, names, feed, options):
            options_seen.append(options)
            if mode == "cancel":
                control.cancel()
            deadline = time.monotonic() + 2
            while not options.terminate and time.monotonic() < deadline:
                time.sleep(0.005)
            assert options.terminate, "native run options never received termination"
            raise RuntimeError("native error with synthetic-sensitive detail")

    before = set(threading.enumerate())
    with pytest.raises(
        TimeoutError if mode == "deadline" else OnnxExecutionCancelled
    ) as exc:
        control.run(Session(), ["out"], {})
    assert "synthetic-sensitive" not in str(exc.value)
    assert options_seen[0].terminate
    assert set(threading.enumerate()) == before


def test_one_scope_cannot_cancel_another_scope_on_a_shared_session():
    pytest.importorskip("onnxruntime")
    a, b = OnnxExecutionControl(), OnnxExecutionControl()

    class Session:
        def run(self, names, feed, options):
            a.cancel()
            assert not options.terminate
            return ["b-result"]

    assert b.run(Session(), ["out"], {}) == ["b-result"]
    with pytest.raises(OnnxExecutionCancelled):
        a.check()


def test_real_ort_session_accepts_independent_run_options(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["x"], ["y"])],
        "control-fixture",
        [onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=9
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feed = {"x": np.array([3.0], dtype=np.float32)}
    options = ort.RunOptions()
    options.terminate = True
    with pytest.raises(Exception):
        session.run(["y"], feed, options)
    result = OnnxExecutionControl(timeout_seconds=3).run(session, ["y"], feed)
    assert result[0].tolist() == [3.0]
