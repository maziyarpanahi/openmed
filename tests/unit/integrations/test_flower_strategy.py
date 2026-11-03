"""Offline contract tests for the optional Flower strategy bridge."""

from __future__ import annotations

import abc
import importlib.util
import json
import logging
import subprocess
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmed.integrations import flower_strategy as bridge_module
from openmed.integrations.flower_strategy import (
    FLOWER_BRIDGE_SCHEMA_VERSION,
    FLOWER_METADATA_METRIC,
    FLOWER_REASON_CODES,
    FLOWER_TENSOR_TYPE,
    FlowerBridgeError,
    FlowerClientUpdate,
    FlowerRoundReceipt,
    OpenMedFlowerClient,
    OpenMedFlowerStrategy,
    flower_update_digest,
    policy_schema_fingerprint,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

ROOT = Path(__file__).resolve().parents[3]
SENTINEL_CLIENT_ID = "client-site-meridian-canary-7731"
SENTINEL_TENSOR = b"TENSOR-VALUE-CANARY-0.123456789"
SENTINEL_TEXT = "TENSOR-VALUE-CANARY"
SENTINEL_METRIC = "patient-meridian-canary@example.com"
MODEL_DIGEST = "sha256:" + "a" * 64


@dataclass
class FakeArray:
    """Minimal array stand-in with shape, dtype and serialized bytes."""

    shape: tuple[int, ...]
    dtype: str
    payload: bytes


def _serialize(array: FakeArray) -> bytes:
    return array.payload


def _policy(*, require_clipped: bool = True) -> FederatedUpdatePolicy:
    return FederatedUpdatePolicy(
        model_digest=MODEL_DIGEST,
        parameters=(
            FederatedParameterMetadata("adapter.lora_B.weight", (4, 2), "float32"),
            FederatedParameterMetadata("adapter.lora_A.weight", (2, 3), "float32"),
        ),
        require_clipped=require_clipped,
    )


def _arrays(seed: int = 0) -> dict[str, FakeArray]:
    return {
        "adapter.lora_A.weight": FakeArray(
            (2, 3), "float32", SENTINEL_TENSOR + bytes([seed, 1])
        ),
        "adapter.lora_B.weight": FakeArray(
            (4, 2), "float32", SENTINEL_TENSOR + bytes([seed, 2])
        ),
    }


class FakeTrainer:
    """OpenMed-aware local trainer returning named, clipped updates."""

    def __init__(
        self,
        *,
        arrays: Mapping[str, Any] | None = None,
        clipped: Any = True,
        num_examples: Any = 12,
        metrics: Mapping[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.arrays = dict(_arrays(seed) if arrays is None else arrays)
        self.clipped = clipped
        self.num_examples = num_examples
        self.metrics = dict(metrics or {})
        self.calls = 0

    def fit(self, parameters: Any, config: Mapping[str, Any]) -> FlowerClientUpdate:
        self.calls += 1
        return FlowerClientUpdate(
            arrays=self.arrays,
            num_examples=self.num_examples,
            clipped=self.clipped,
            metrics=self.metrics,
        )

    def get_parameters(self, config: Mapping[str, Any]) -> list[Any]:
        return ["initial"]

    def evaluate(
        self, parameters: Any, config: Mapping[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        return 0.25, 3, {"loss": 0.25, "site": SENTINEL_CLIENT_ID}


class FakeDelegate:
    """Strategy protocol fake that records aggregation calls."""

    def __init__(self, *, error: BaseException | None = None) -> None:
        self.error = error
        self.calls: list[tuple[int, list[Any], list[Any]]] = []
        self.forwarded: list[tuple[str, tuple[Any, ...]]] = []

    def initialize_parameters(self, client_manager: Any) -> Any:
        self.forwarded.append(("initialize_parameters", (client_manager,)))
        return "initial-parameters"

    def configure_fit(self, server_round: int, parameters: Any, manager: Any) -> Any:
        self.forwarded.append(("configure_fit", (server_round, parameters, manager)))
        return ["fit-instructions"]

    def aggregate_fit(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> tuple[Any, dict[str, Any]]:
        self.calls.append((server_round, list(results), list(failures)))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(tensors=[b"aggregate"]), {"aggregated": True}

    def configure_evaluate(
        self, server_round: int, parameters: Any, manager: Any
    ) -> Any:
        self.forwarded.append(("configure_evaluate", (server_round,)))
        return ["evaluate-instructions"]

    def aggregate_evaluate(
        self, server_round: int, results: list[Any], failures: list[Any]
    ) -> Any:
        self.forwarded.append(("aggregate_evaluate", (server_round,)))
        return 0.5, {}

    def evaluate(self, server_round: int, parameters: Any) -> Any:
        self.forwarded.append(("evaluate", (server_round,)))
        return None


def _client(trainer: FakeTrainer | None = None, **kwargs: Any) -> OpenMedFlowerClient:
    return OpenMedFlowerClient(
        trainer or FakeTrainer(), policy=_policy(), serializer=_serialize, **kwargs
    )


def _fit_res(
    arrays: list[Any],
    num_examples: int,
    metrics: dict[str, Any],
    *,
    tensor_type: str = FLOWER_TENSOR_TYPE,
) -> SimpleNamespace:
    """Simulate the framework serializing a NumPyClient fit result."""

    return SimpleNamespace(
        status=SimpleNamespace(code=0, message=SENTINEL_CLIENT_ID),
        parameters=SimpleNamespace(
            tensors=[_serialize(array) for array in arrays], tensor_type=tensor_type
        ),
        num_examples=num_examples,
        metrics=metrics,
    )


def _result(seed: int = 0, **trainer_kwargs: Any) -> tuple[Any, SimpleNamespace]:
    arrays, count, metrics = _client(FakeTrainer(seed=seed, **trainer_kwargs)).fit(
        ["global"], {}
    )
    proxy = SimpleNamespace(cid=f"{SENTINEL_CLIENT_ID}-{seed}")
    return proxy, _fit_res(arrays, count, metrics)


def _forged_metadata(
    policy: FederatedUpdatePolicy, tensors: list[bytes], **overrides: Any
) -> str:
    metadata = FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format="dense",
        parameters=policy.parameters,
        total_elements=sum(item.element_count for item in policy.parameters),
        update_digest=flower_update_digest(policy.parameters, tensors),
        clipped=True,
        policy=policy,
    )
    payload = metadata.to_dict()
    payload.update(overrides)
    return json.dumps(payload)


def _strategy(
    delegate: FakeDelegate | None = None, **kwargs: Any
) -> OpenMedFlowerStrategy:
    return OpenMedFlowerStrategy(delegate or FakeDelegate(), policy=_policy(), **kwargs)


def _assert_refused(
    strategy: OpenMedFlowerStrategy,
    delegate: FakeDelegate,
    outcome: tuple[Any, dict[str, Any]],
    code: str,
) -> None:
    assert outcome == (
        None,
        {"openmed.round_status": "refused", "openmed.reason_code": code},
    )
    assert delegate.calls == []
    receipt = strategy.transcript[-1]
    assert receipt.status == "refused"
    assert receipt.reason_code == code
    assert receipt.updates_root is None


# Valid synthetic round ------------------------------------------------------


def test_valid_synthetic_round_is_validated_then_delegated() -> None:
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    results = [_result(seed=1), _result(seed=2)]

    parameters, metrics = strategy.aggregate_fit(3, results, [])

    assert parameters.tensors == [b"aggregate"]
    assert metrics == {"aggregated": True}
    assert len(delegate.calls) == 1
    server_round, forwarded, failures = delegate.calls[0]
    assert server_round == 3
    assert forwarded == results
    assert failures == []

    (receipt,) = strategy.transcript
    assert receipt.status == "aggregated"
    assert receipt.reason_code is None
    assert receipt.result_count == 2
    assert receipt.failure_count == 0
    assert receipt.schema_fingerprint == policy_schema_fingerprint(_policy())
    assert receipt.updates_root is not None
    assert receipt.updates_root.startswith("sha256:")
    assert receipt.to_dict()["schema_version"] == FLOWER_BRIDGE_SCHEMA_VERSION


def test_updates_root_is_order_independent() -> None:
    first = _strategy()
    second = _strategy()
    one, two = _result(seed=1), _result(seed=2)
    first.aggregate_fit(1, [one, two], [])
    second.aggregate_fit(1, [two, one], [])
    assert first.transcript[0].updates_root == second.transcript[0].updates_root


def test_client_emits_policy_ordered_arrays_and_metadata_only_metrics() -> None:
    trainer = FakeTrainer(
        metrics={"loss": 0.5, "site": SENTINEL_METRIC, "steps": 7, "flag": True},
    )
    client = _client(trainer, metric_allowlist=("loss", "site", "flag"))

    arrays, count, metrics = client.fit(["global"], {"epochs": 1})

    expected = _arrays()
    assert arrays == [
        expected["adapter.lora_A.weight"],
        expected["adapter.lora_B.weight"],
    ]
    assert count == 12
    # Strings are dropped even when allowlisted; unlisted keys are dropped.
    assert set(metrics) == {FLOWER_METADATA_METRIC, "loss", "flag"}
    metadata = FederatedUpdateMetadata.from_json(
        metrics[FLOWER_METADATA_METRIC], policy=_policy()
    )
    assert metadata.clipped is True
    assert metadata.update_digest == flower_update_digest(
        _policy().parameters, [_serialize(array) for array in arrays]
    )
    assert SENTINEL_TEXT not in metrics[FLOWER_METADATA_METRIC]


def test_round_with_numpy_serialization_passes_end_to_end() -> None:
    np = pytest.importorskip("numpy")
    import io

    def save(array: Any) -> bytes:
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue()

    arrays = {
        "adapter.lora_A.weight": np.full((2, 3), 0.5, dtype=np.float32),
        "adapter.lora_B.weight": np.full((4, 2), -0.25, dtype=np.float32),
    }
    client = OpenMedFlowerClient(
        FakeTrainer(arrays=arrays), policy=_policy(), serializer=save
    )
    out, count, metrics = client.fit([], {})
    fit_res = SimpleNamespace(
        parameters=SimpleNamespace(
            tensors=[save(array) for array in out], tensor_type=FLOWER_TENSOR_TYPE
        ),
        num_examples=count,
        metrics=metrics,
    )
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    parameters, _ = strategy.aggregate_fit(1, [(object(), fit_res)], [])
    assert parameters is not None
    assert strategy.transcript[0].status == "aggregated"


# Strategy fails the round closed -------------------------------------------


def test_unknown_parameter_fails_the_round_closed() -> None:
    policy = _policy()
    tensors = [b"one", b"two", b"three"]
    parameters = [item.to_dict() for item in policy.parameters] + [
        {"name": "adapter.extra.weight", "shape": [1], "dtype": "float32"}
    ]
    forged = _forged_metadata(
        policy,
        tensors[:2],
        parameters=parameters,
        total_elements=sum(item.element_count for item in policy.parameters) + 1,
    )
    bad = (
        SimpleNamespace(cid=SENTINEL_CLIENT_ID),
        SimpleNamespace(
            parameters=SimpleNamespace(tensors=tensors, tensor_type=FLOWER_TENSOR_TYPE),
            num_examples=4,
            metrics={FLOWER_METADATA_METRIC: forged},
        ),
    )
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=1), bad], [])
    _assert_refused(strategy, delegate, outcome, "parameter_set_mismatch")


def test_tensor_digest_mismatch_fails_the_round_closed() -> None:
    proxy, fit_res = _result(seed=1)
    fit_res.parameters.tensors[0] = fit_res.parameters.tensors[0] + b"tamper"
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=2), (proxy, fit_res)], [])
    _assert_refused(strategy, delegate, outcome, "update_digest_mismatch")


def test_reordered_tensors_fail_the_digest_check() -> None:
    proxy, fit_res = _result(seed=1)
    fit_res.parameters.tensors.reverse()
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [(proxy, fit_res)], [])
    _assert_refused(strategy, delegate, outcome, "update_digest_mismatch")


def test_model_digest_mismatch_fails_the_round_closed() -> None:
    policy = _policy()
    tensors = [b"one", b"two"]
    forged = _forged_metadata(policy, tensors, model_digest="sha256:" + "b" * 64)
    fit_res = SimpleNamespace(
        parameters=SimpleNamespace(tensors=tensors, tensor_type=FLOWER_TENSOR_TYPE),
        num_examples=4,
        metrics={FLOWER_METADATA_METRIC: forged},
    )
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [(object(), fit_res)], [])
    _assert_refused(strategy, delegate, outcome, "model_digest_mismatch")


def test_missing_clipping_fails_the_round_closed() -> None:
    policy = _policy()
    tensors = [b"one", b"two"]
    forged = _forged_metadata(policy, tensors, clipped=False)
    fit_res = SimpleNamespace(
        parameters=SimpleNamespace(tensors=tensors, tensor_type=FLOWER_TENSOR_TYPE),
        num_examples=4,
        metrics={FLOWER_METADATA_METRIC: forged},
    )
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=1), (object(), fit_res)], [])
    _assert_refused(strategy, delegate, outcome, "clipping_not_declared")


@pytest.mark.parametrize(
    ("mutate", "code"),
    [
        (lambda res: res.metrics.clear(), "metadata_missing"),
        (
            lambda res: res.metrics.update({FLOWER_METADATA_METRIC: b"{}"}),
            "metadata_missing",
        ),
        (
            lambda res: res.metrics.update({FLOWER_METADATA_METRIC: "{not json"}),
            "metadata_invalid",
        ),
        (lambda res: res.parameters.tensors.pop(), "parameter_count_mismatch"),
        (
            lambda res: res.parameters.tensors.__setitem__(0, "text"),
            "invalid_tensor_payload",
        ),
        (
            lambda res: setattr(res.parameters, "tensor_type", "pickle"),
            "invalid_tensor_payload",
        ),
        (lambda res: setattr(res, "num_examples", 0), "invalid_result"),
        (lambda res: setattr(res, "num_examples", True), "invalid_result"),
        (lambda res: setattr(res, "metrics", [SENTINEL_METRIC]), "invalid_result"),
        (lambda res: delattr(res, "parameters"), "invalid_tensor_payload"),
    ],
)
def test_each_result_violation_fails_the_round_closed(mutate: Any, code: str) -> None:
    proxy, fit_res = _result(seed=1)
    mutate(fit_res)
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=2), (proxy, fit_res)], [])
    _assert_refused(strategy, delegate, outcome, code)


def test_schema_fingerprint_mismatch_fails_the_round_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strategy = _strategy()
    monkeypatch.setattr(
        bridge_module,
        "fingerprint_update_schema",
        lambda metadata: "sha256:" + "0" * 64,
    )
    delegate = strategy._delegate
    outcome = strategy.aggregate_fit(1, [_result(seed=1)], [])
    _assert_refused(strategy, delegate, outcome, "schema_fingerprint_mismatch")


def test_expected_fingerprint_must_match_the_trusted_policy() -> None:
    expected = policy_schema_fingerprint(_policy())
    assert _strategy(expected_schema_fingerprint=expected) is not None
    with pytest.raises(FlowerBridgeError) as error:
        _strategy(expected_schema_fingerprint="sha256:" + "f" * 64)
    assert error.value.code == "schema_fingerprint_mismatch"


def test_duplicate_update_fails_the_round_closed() -> None:
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=1), _result(seed=1)], [])
    _assert_refused(strategy, delegate, outcome, "duplicate_update")


@pytest.mark.parametrize(
    ("server_round", "results", "failures", "code"),
    [
        (0, None, [], "invalid_round"),
        (True, None, [], "invalid_round"),
        (1, [], [], "empty_round"),
        (1, "not-a-list", [], "invalid_result"),
        (1, None, "not-a-list", "invalid_result"),
        (1, [("only-proxy",)], [], "invalid_result"),
        (1, None, [RuntimeError(SENTINEL_CLIENT_ID)], "round_failures_present"),
    ],
)
def test_round_level_violations_fail_closed(
    server_round: Any, results: Any, failures: Any, code: str
) -> None:
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    if results is None:
        results = [_result(seed=1)]
    outcome = strategy.aggregate_fit(server_round, results, failures)
    _assert_refused(strategy, delegate, outcome, code)


def test_too_many_results_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bridge_module, "_MAX_RESULTS", 1)
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=1), _result(seed=2)], [])
    _assert_refused(strategy, delegate, outcome, "too_many_results")


def test_accepted_failures_never_reach_the_delegate() -> None:
    delegate = FakeDelegate()
    strategy = _strategy(delegate, accept_failures=True)
    failure = (SimpleNamespace(cid=SENTINEL_CLIENT_ID), _result(seed=9)[1])
    strategy.aggregate_fit(1, [_result(seed=1)], [failure, RuntimeError("x")])
    assert delegate.calls[0][2] == []
    receipt = strategy.transcript[0]
    assert receipt.status == "aggregated"
    assert receipt.failure_count == 2


def test_delegate_errors_are_contained_without_echoing_details() -> None:
    delegate = FakeDelegate(error=RuntimeError(SENTINEL_CLIENT_ID + SENTINEL_TEXT))
    strategy = _strategy(delegate)
    outcome = strategy.aggregate_fit(1, [_result(seed=1)], [])
    assert outcome[0] is None
    assert outcome[1]["openmed.reason_code"] == "delegate_aggregation_failed"
    assert strategy.transcript[-1].reason_code == "delegate_aggregation_failed"


def test_raise_on_violation_raises_a_fixed_code() -> None:
    proxy, fit_res = _result(seed=1)
    fit_res.parameters.tensors[0] = b"tampered"
    strategy = _strategy(raise_on_violation=True)
    with pytest.raises(FlowerBridgeError) as error:
        strategy.aggregate_fit(1, [(proxy, fit_res)], [])
    assert error.value.code == "update_digest_mismatch"
    assert str(error.value) == "update_digest_mismatch"
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_strategy_forwards_non_fit_calls_to_the_delegate() -> None:
    delegate = FakeDelegate()
    strategy = _strategy(delegate)
    assert strategy.initialize_parameters("manager") == "initial-parameters"
    assert strategy.configure_fit(2, "params", "manager") == ["fit-instructions"]
    assert strategy.configure_evaluate(2, "params", "manager") == [
        "evaluate-instructions"
    ]
    assert strategy.aggregate_evaluate(2, [], []) == (0.5, {})
    assert strategy.evaluate(2, "params") is None
    assert [name for name, _ in delegate.forwarded] == [
        "initialize_parameters",
        "configure_fit",
        "configure_evaluate",
        "aggregate_evaluate",
        "evaluate",
    ]


def test_strategy_rejects_unsafe_configuration() -> None:
    with pytest.raises(FlowerBridgeError) as error:
        OpenMedFlowerStrategy(FakeDelegate(), policy=_policy(require_clipped=False))
    assert error.value.code == "invalid_policy"
    with pytest.raises(FlowerBridgeError) as error:
        OpenMedFlowerStrategy(FakeDelegate(), policy={"model_digest": MODEL_DIGEST})
    assert error.value.code == "invalid_policy"
    with pytest.raises(FlowerBridgeError) as error:
        OpenMedFlowerStrategy(object(), policy=_policy())
    assert error.value.code == "invalid_delegate"
    with pytest.raises(FlowerBridgeError) as error:
        _strategy(max_transcript=0)
    assert error.value.code == "invalid_configuration"


def test_transcript_is_bounded() -> None:
    strategy = _strategy(max_transcript=2)
    for server_round in (1, 2, 3):
        strategy.aggregate_fit(server_round, [], [])
    assert [receipt.server_round for receipt in strategy.transcript] == [2, 3]


# Client refuses unsafe updates ---------------------------------------------


@pytest.mark.parametrize(
    ("trainer_kwargs", "code"),
    [
        ({"clipped": False}, "clipping_not_declared"),
        ({"clipped": 1}, "clipping_not_declared"),
        (
            {
                "arrays": {
                    **_arrays(),
                    "adapter.extra.weight": FakeArray((1,), "float32", b"x"),
                }
            },
            "parameter_not_allowlisted",
        ),
        (
            {"arrays": {"adapter.lora_A.weight": _arrays()["adapter.lora_A.weight"]}},
            "parameter_missing",
        ),
        (
            {
                "arrays": {
                    **_arrays(),
                    "adapter.lora_A.weight": FakeArray((3, 2), "float32", b"x"),
                }
            },
            "parameter_shape_mismatch",
        ),
        (
            {
                "arrays": {
                    **_arrays(),
                    "adapter.lora_A.weight": FakeArray((2, 3), "float64", b"x"),
                }
            },
            "parameter_shape_mismatch",
        ),
        (
            {"arrays": {**_arrays(), "adapter.lora_A.weight": SENTINEL_TENSOR}},
            "invalid_tensor_payload",
        ),
        ({"num_examples": 0}, "invalid_client_update"),
        ({"num_examples": True}, "invalid_client_update"),
    ],
)
def test_client_refuses_unsafe_updates(trainer_kwargs: dict[str, Any], code: str):
    client = _client(FakeTrainer(**trainer_kwargs))
    with pytest.raises(FlowerBridgeError) as error:
        client.fit(["global"], {})
    assert error.value.code == code
    assert str(error.value) == code


def test_client_refuses_non_update_return_values() -> None:
    class RawTrainer:
        def fit(self, parameters: Any, config: Any) -> Any:
            return [SENTINEL_TENSOR], 1, {}

    with pytest.raises(FlowerBridgeError) as error:
        _client(RawTrainer()).fit([], {})
    assert error.value.code == "invalid_client_update"


def test_client_contains_trainer_and_serializer_errors() -> None:
    class FailingTrainer:
        def fit(self, parameters: Any, config: Any) -> Any:
            raise RuntimeError(SENTINEL_CLIENT_ID + SENTINEL_TEXT)

    with pytest.raises(FlowerBridgeError) as error:
        _client(FailingTrainer()).fit([], {})
    assert error.value.code == "client_fit_failed"
    assert error.value.__suppress_context__ is True

    def failing_serializer(array: Any) -> bytes:
        raise ValueError(SENTINEL_TEXT)

    client = OpenMedFlowerClient(
        FakeTrainer(), policy=_policy(), serializer=failing_serializer
    )
    with pytest.raises(FlowerBridgeError) as error:
        client.fit([], {})
    assert error.value.code == "serializer_failed"

    client = OpenMedFlowerClient(
        FakeTrainer(), policy=_policy(), serializer=lambda array: "text"
    )
    with pytest.raises(FlowerBridgeError) as error:
        client.fit([], {})
    assert error.value.code == "serializer_failed"


def test_client_rejects_unsafe_configuration() -> None:
    with pytest.raises(FlowerBridgeError) as error:
        OpenMedFlowerClient(FakeTrainer(), policy=_policy(require_clipped=False))
    assert error.value.code == "invalid_policy"
    with pytest.raises(FlowerBridgeError) as error:
        OpenMedFlowerClient(object(), policy=_policy())
    assert error.value.code == "invalid_delegate"
    for allowlist in ((FLOWER_METADATA_METRIC,), ("",), (1,), "loss"):
        with pytest.raises(FlowerBridgeError) as error:
            OpenMedFlowerClient(
                FakeTrainer(), policy=_policy(), metric_allowlist=allowlist
            )
        assert error.value.code == "invalid_metric_allowlist"


def test_client_filters_evaluate_metrics_and_forwards_parameters() -> None:
    client = _client(metric_allowlist=("loss", "site"))
    assert client.get_parameters({}) == ["initial"]
    loss, count, metrics = client.evaluate([], {})
    assert (loss, count, metrics) == (0.25, 3, {"loss": 0.25})


def test_update_repr_hides_arrays_and_metrics() -> None:
    update = FlowerClientUpdate(
        arrays=_arrays(), num_examples=3, clipped=True, metrics={"s": SENTINEL_METRIC}
    )
    assert SENTINEL_TEXT not in repr(update)
    assert SENTINEL_METRIC not in repr(update)


def test_digest_helper_rejects_malformed_inputs() -> None:
    parameters = _policy().parameters
    with pytest.raises(FlowerBridgeError) as error:
        flower_update_digest(parameters, [b"one"])
    assert error.value.code == "parameter_count_mismatch"
    with pytest.raises(FlowerBridgeError) as error:
        flower_update_digest(parameters, [b"one", "two"])
    assert error.value.code == "invalid_tensor_payload"
    first = flower_update_digest(parameters, [b"a", b"bc"])
    second = flower_update_digest(parameters, [b"ab", b"c"])
    assert first != second


# Privacy: no client identifiers or tensor values ----------------------------


def test_logs_transcripts_and_errors_contain_no_identifiers_or_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    strategy = _strategy()
    strategy.aggregate_fit(1, [_result(seed=1), _result(seed=2)], [])
    proxy, tampered = _result(seed=3)
    tampered.parameters.tensors[0] = SENTINEL_TENSOR + b"tampered"
    strategy.aggregate_fit(2, [(proxy, tampered)], [RuntimeError(SENTINEL_CLIENT_ID)])
    strategy.aggregate_fit(3, [(proxy, tampered)], [])
    failing = _strategy(FakeDelegate(error=RuntimeError(SENTINEL_CLIENT_ID)))
    failing.aggregate_fit(4, [_result(seed=4)], [])
    raising = _strategy(raise_on_violation=True)
    with pytest.raises(FlowerBridgeError) as error:
        raising.aggregate_fit(5, [(proxy, tampered)], [])

    rendered = "\n".join(
        [
            caplog.text,
            json.dumps([receipt.to_dict() for receipt in strategy.transcript]),
            json.dumps([receipt.to_dict() for receipt in failing.transcript]),
            repr(strategy.transcript),
            repr(strategy),
            repr(error.value),
            str(error.value),
        ]
    )
    assert caplog.records, "bridge should log round outcomes"
    for sentinel in (SENTINEL_CLIENT_ID, SENTINEL_TEXT, SENTINEL_METRIC):
        assert sentinel not in rendered
    for receipt in strategy.transcript:
        assert set(receipt.to_dict()) == {
            "schema_version",
            "server_round",
            "status",
            "reason_code",
            "result_count",
            "failure_count",
            "schema_fingerprint",
            "updates_root",
        }


def test_reason_codes_are_closed_and_unknown_codes_are_collapsed() -> None:
    assert "update_digest_mismatch" in FLOWER_REASON_CODES
    assert FlowerBridgeError(SENTINEL_CLIENT_ID).code == "internal_error"
    assert isinstance(FlowerRoundReceipt.__dataclass_fields__, dict)


# Optional framework stays lazy ---------------------------------------------


def _fake_flwr(monkeypatch: pytest.MonkeyPatch) -> None:
    class Strategy(abc.ABC):
        @abc.abstractmethod
        def initialize_parameters(self, client_manager): ...

        @abc.abstractmethod
        def configure_fit(self, server_round, parameters, client_manager): ...

        @abc.abstractmethod
        def aggregate_fit(self, server_round, results, failures): ...

        @abc.abstractmethod
        def configure_evaluate(self, server_round, parameters, client_manager): ...

        @abc.abstractmethod
        def aggregate_evaluate(self, server_round, results, failures): ...

        @abc.abstractmethod
        def evaluate(self, server_round, parameters): ...

    class NumPyClient:
        def to_client(self) -> str:
            return "client"

    def ndarray_to_bytes(array: Any) -> bytes:
        return array.payload

    flwr = types.ModuleType("flwr")
    client = types.ModuleType("flwr.client")
    client.NumPyClient = NumPyClient
    common = types.ModuleType("flwr.common")
    common.ndarray_to_bytes = ndarray_to_bytes
    server = types.ModuleType("flwr.server")
    strategy = types.ModuleType("flwr.server.strategy")
    strategy.Strategy = Strategy
    for name, module in {
        "flwr": flwr,
        "flwr.client": client,
        "flwr.common": common,
        "flwr.server": server,
        "flwr.server.strategy": strategy,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_framework_adapters_use_the_fake_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_flwr(monkeypatch)
    delegate = FakeDelegate()
    framework_strategy = _strategy(delegate).as_flower_strategy()
    assert isinstance(framework_strategy, sys.modules["flwr.server.strategy"].Strategy)
    parameters, _ = framework_strategy.aggregate_fit(1, [_result(seed=1)], [])
    assert parameters.tensors == [b"aggregate"]
    assert framework_strategy.initialize_parameters("m") == "initial-parameters"

    framework_client = OpenMedFlowerClient(
        FakeTrainer(), policy=_policy()
    ).as_flower_client()
    assert isinstance(framework_client, sys.modules["flwr.client"].NumPyClient)
    arrays, count, metrics = framework_client.fit([], {})
    assert count == 12
    assert FLOWER_METADATA_METRIC in metrics
    assert framework_client.get_parameters({}) == ["initial"]
    assert framework_client.evaluate([], {})[0] == 0.25


def test_missing_framework_raises_install_hint(monkeypatch: pytest.MonkeyPatch):
    for name in ("flwr", "flwr.client", "flwr.common", "flwr.server.strategy"):
        monkeypatch.setitem(sys.modules, name, None)
    with pytest.raises(ImportError, match=r"operator-installed flwr"):
        _strategy().as_flower_strategy()
    client = OpenMedFlowerClient(FakeTrainer(), policy=_policy())
    with pytest.raises(ImportError, match=r"operator-installed flwr"):
        client.as_flower_client()
    with pytest.raises(ImportError, match=r"operator-installed flwr"):
        client.fit([], {})


def test_core_package_imports_without_the_framework() -> None:
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'flwr' or name.startswith('flwr.'):\n"
        "            raise ImportError('blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import openmed, openmed.integrations\n"
        "import openmed.integrations.flower_strategy as bridge\n"
        "assert not any(m == 'flwr' or m.startswith('flwr.') for m in sys.modules)\n"
        "print(bridge.FLOWER_BRIDGE_SCHEMA_VERSION)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert completed.stdout.strip() == FLOWER_BRIDGE_SCHEMA_VERSION


# Packaging: optional extra and license policy -------------------------------


def _license_policy() -> Any:
    path = ROOT / "scripts" / "release" / "check_license_policy.py"
    spec = importlib.util.spec_from_file_location("openmed_license_policy", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_flower_extra_is_an_empty_marker_and_passes_the_license_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _license_policy()
    pyproject = policy.read_pyproject(ROOT / "pyproject.toml")
    project = pyproject["project"]
    extras = project["optional-dependencies"]
    # Operators install Flower themselves; the extra declares no packages.
    assert extras["flower"] == []
    assert not any(
        policy.dependency_name(item) == "flwr"
        for items in [project["dependencies"], *extras.values()]
        for item in items
    )

    monkeypatch.setattr(policy, "installed_license_text", lambda name: None)
    results = policy.audit_pyproject(ROOT / "pyproject.toml")
    assert not [result for result in results if result.entry.group == "flower"]
    assert all(result.allowed for result in results)
