"""Deterministic server optimizer rules over aggregated adapter deltas."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.training.federated_schema_fingerprint import fingerprint_update_schema
from openmed.training.federated_server_optimizer import (
    MAX_SERVER_OPTIMIZER_ELEMENTS,
    MAX_SERVER_OPTIMIZER_PARAMETERS,
    SERVER_OPTIMIZER_ERROR_CODES,
    SERVER_OPTIMIZER_STATE_SCHEMA_VERSION,
    SERVER_UPDATE_PRIVACY_ACCOUNTING,
    SERVER_UPDATE_RECORD_SCHEMA_VERSION,
    ServerOptimizerConfig,
    ServerOptimizerError,
    ServerOptimizerRule,
    ServerParameterSchema,
    apply_server_update,
    digest_adapter_values,
    digest_aggregate_delta,
    initial_server_optimizer_state,
    restore_server_optimizer_state,
)
from openmed.training.federated_update_metadata import (
    FederatedParameterMetadata,
    FederatedUpdateMetadata,
    FederatedUpdatePolicy,
)

ROOT = Path(__file__).resolve().parents[3]
A = "adapter.lora_A.weight"
B = "adapter.lora_B.weight"
X0 = {A: (0.5, -0.25, 1.0), B: (0.0, 2.0)}
DELTAS = (
    {A: (0.1, -0.2, 0.3), B: (0.05, -0.4)},
    {A: (-0.3, 0.1, 0.0), B: (0.2, 0.25)},
    {A: (0.7, -0.05, -0.15), B: (-0.1, 0.0)},
)
CONFIGS = {
    "sgd": {"rule": "sgd", "learning_rate": 0.5},
    "momentum": {"rule": "momentum", "learning_rate": 1.0, "momentum": 0.9},
    "nesterov": {
        "rule": "momentum",
        "learning_rate": 1.0,
        "momentum": 0.9,
        "nesterov": True,
    },
    "adagrad": {"rule": "adagrad", "learning_rate": 0.1},
    "adam": {"rule": "adam", "learning_rate": 0.1},
    "yogi": {"rule": "yogi", "learning_rate": 0.1},
}
# Final global adapters after three rounds, as exact IEEE-754 float64 hex.
GOLDEN = {
    "sgd": {
        A: ["0x1.8000000000000p-1", "-0x1.4cccccccccccdp-2", "0x1.1333333333333p+0"],
        B: ["0x1.3333333333333p-4", "0x1.ecccccccccccdp+0"],
    },
    "momentum": {
        A: ["0x1.cd4fdf3b645a2p-1", "-0x1.4dd2f1a9fbe77p-1", "0x1.a9ba5e353f7cfp+0"],
        B: ["0x1.a978d4fdf3b66p-2", "0x1.64189374bc6a8p+0"],
    },
    "nesterov": {
        A: ["0x1.5c63f141205bcp+0", "-0x1.860aa64c2f838p-1", "0x1.bf27bb2fec56ep+0"],
        B: ["0x1.0c432ca57a788p-1", "0x1.4d495182a9931p+0"],
    },
    "adagrad": {
        A: ["0x1.0514d90e62cbep-1", "-0x1.1343c46430122p-2", "0x1.058e366d67b4ep+0"],
        B: ["0x1.b82780d04309ep-6", "0x1.fc5083d4f1860p+0"],
    },
    "adam": {
        A: ["0x1.2f07d6ef28742p-1", "-0x1.b934df09907bdp-2", "0x1.36192344c5a50p+0"],
        B: ["0x1.f765bf0094120p-3", "0x1.dbdc948dfa492p+0"],
    },
    "yogi": {
        A: ["0x1.2f03876cac44cp-1", "-0x1.b8a32285c3b88p-2", "0x1.35ef7541d1358p+0"],
        B: ["0x1.f6df5ba2ec8cap-3", "0x1.dbed5613d45e4p+0"],
    },
}
SENTINEL_VALUE = 0.123456789012345
SENTINEL_TEXT = "SITE-ALPHA-PATIENT-0042"


def _schema() -> ServerParameterSchema:
    return ServerParameterSchema.from_lengths({B: 2, A: 3})


def _config(name: str) -> ServerOptimizerConfig:
    return ServerOptimizerConfig(**CONFIGS[name])


def _reference(name: str) -> dict[str, list[str]]:
    """Independent textbook implementation of each server rule."""

    spec = dict(CONFIGS[name])
    rule = spec["rule"]
    lr = spec["learning_rate"]
    mu = spec.get("momentum", 0.0)
    nesterov = spec.get("nesterov", False)
    b1, b2, tau = 0.9, 0.99, 1e-3
    x = {k: list(v) for k, v in X0.items()}
    m = {k: [0.0] * len(v) for k, v in X0.items()}
    v = {k: [tau * tau] * len(vv) for k, vv in X0.items()}
    for delta in DELTAS:
        for key in sorted(x):
            for i, g in enumerate(delta[key]):
                if rule == "sgd":
                    x[key][i] = x[key][i] + lr * g
                elif rule == "momentum":
                    m[key][i] = mu * m[key][i] + g
                    direction = g + mu * m[key][i] if nesterov else m[key][i]
                    x[key][i] = x[key][i] + lr * direction
                else:
                    m[key][i] = b1 * m[key][i] + (1.0 - b1) * g
                    sq = g * g
                    if rule == "adagrad":
                        v[key][i] = v[key][i] + sq
                    elif rule == "adam":
                        v[key][i] = b2 * v[key][i] + (1.0 - b2) * sq
                    else:
                        diff = v[key][i] - sq
                        sign = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
                        v[key][i] = v[key][i] - (1.0 - b2) * sq * sign
                    denom = math.sqrt(v[key][i]) + tau
                    x[key][i] = x[key][i] + lr * (m[key][i] / denom)
    return {k: [value.hex() for value in values] for k, values in sorted(x.items())}


def _run(name: str, rounds: int = 3, *, start=None, adapter=None):
    config = _config(name)
    state = start or initial_server_optimizer_state(config, _schema())
    adapter = dict(X0) if adapter is None else adapter
    records = []
    for delta in DELTAS[state.step : state.step + rounds]:
        result = apply_server_update(state, adapter, delta)
        state, adapter = result.state, result.global_adapter
        records.append(result.record)
    return adapter, state, records


def _hex(adapter) -> dict[str, list[str]]:
    return {key: [value.hex() for value in adapter[key]] for key in sorted(adapter)}


def _code(excinfo: pytest.ExceptionInfo[ServerOptimizerError]) -> str:
    return excinfo.value.code


# Reference vectors and byte-identical determinism.


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_reference_vectors_match_golden_and_independent_formula(name: str) -> None:
    adapter, state, _ = _run(name)

    assert _hex(adapter) == GOLDEN[name]
    assert _reference(name) == GOLDEN[name]
    assert state.step == 3
    assert all(type(value) is float for values in adapter.values() for value in values)


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_repeated_runs_produce_byte_identical_state_and_records(name: str) -> None:
    first = _run(name)
    second = _run(name)

    assert first[1].to_checkpoint_bytes() == second[1].to_checkpoint_bytes()
    assert first[1].state_digest == second[1].state_digest
    assert [r.to_json() for r in first[2]] == [r.to_json() for r in second[2]]


_SUBPROCESS = """
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from tests.unit.training.test_federated_server_optimizer import CONFIGS, _hex, _run
out = {}
for name in sorted(CONFIGS):
    adapter, state, records = _run(name)
    out[name] = {
        "adapter": _hex(adapter),
        "state": state.state_digest,
        "checkpoint": hashlib.sha256(state.to_checkpoint_bytes()).hexdigest(),
        "records": [r.to_json() for r in records],
    }
print(json.dumps(out, sort_keys=True))
"""


def test_reference_vectors_are_identical_across_hash_seeds() -> None:
    outputs = []
    for seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        completed = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS, str(ROOT)],
            capture_output=True,
            check=True,
            cwd=ROOT,
            env=env,
            text=True,
        )
        outputs.append(completed.stdout)
    assert outputs[0] == outputs[1]
    decoded = json.loads(outputs[0])
    for name in CONFIGS:
        adapter, state, _ = _run(name)
        assert decoded[name]["adapter"] == GOLDEN[name]
        assert decoded[name]["state"] == state.state_digest
        expected = hashlib.sha256(state.to_checkpoint_bytes()).hexdigest()
        assert decoded[name]["checkpoint"] == expected


def test_negative_controls_change_outputs_and_digests() -> None:
    heavy = _run("momentum")
    nesterov = _run("nesterov")
    adam = _run("adam")
    yogi = _run("yogi")

    assert _hex(heavy[0]) != _hex(nesterov[0])
    assert heavy[1].state_digest != nesterov[1].state_digest
    assert _hex(adam[0]) != _hex(yogi[0])
    assert adam[1].config.config_digest != yogi[1].config.config_digest
    shifted = ServerOptimizerConfig(rule="adam", learning_rate=0.1, beta1=0.8)
    assert shifted.hyperparameter_code != adam[1].config.hyperparameter_code


def test_sgd_with_unit_learning_rate_adds_the_aggregate_delta() -> None:
    state = initial_server_optimizer_state(
        ServerOptimizerConfig(rule="sgd", learning_rate=1.0), _schema()
    )
    result = apply_server_update(state, X0, DELTAS[0])

    assert result.global_adapter[A] == tuple(
        x + d for x, d in zip(X0[A], DELTAS[0][A], strict=True)
    )
    assert result.state.step == 1


def test_inputs_are_not_mutated_and_outputs_are_immutable() -> None:
    adapter = {A: [0.5, -0.25, 1.0], B: [0.0, 2.0]}
    delta = {A: [0.1, -0.2, 0.3], B: [0.05, -0.4]}
    state = initial_server_optimizer_state(_config("adam"), _schema())
    result = apply_server_update(state, adapter, delta)

    assert adapter == {A: [0.5, -0.25, 1.0], B: [0.0, 2.0]}
    assert delta == {A: [0.1, -0.2, 0.3], B: [0.05, -0.4]}
    assert isinstance(result.global_adapter[A], tuple)
    with pytest.raises(TypeError):
        result.global_adapter[A] = (0.0, 0.0, 0.0)  # type: ignore[index]
    assert state.step == 0


# Declared hyperparameter codes.


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_hyperparameter_codes_round_trip_exactly(name: str) -> None:
    config = _config(name)
    code = config.hyperparameter_code

    assert code.isascii() and code.startswith(f"rule={config.rule.value};")
    assert ServerOptimizerConfig.from_hyperparameter_code(code) == config
    assert config.config_digest.startswith("sha256:")
    assert config.to_dict() == {
        "rule": config.rule.value,
        "hyperparameter_code": code,
        "config_digest": config.config_digest,
    }


def test_hyperparameter_code_declares_every_resolved_value() -> None:
    config = ServerOptimizerConfig(rule=ServerOptimizerRule.ADAM, learning_rate=0.1)

    assert config.hyperparameter_code == (
        "rule=adam;learning_rate=0x1.999999999999ap-4;beta1=0x1.ccccccccccccdp-1;"
        "beta2=0x1.fae147ae147aep-1;tau=0x1.0624dd2f1a9fcp-10;"
        "initial_accumulator=0x1.0c6f7a0b5ed8dp-20"
    )
    assert ServerOptimizerConfig(rule="sgd", learning_rate=0.5).hyperparameter_code == (
        "rule=sgd;learning_rate=0x1.0000000000000p-1"
    )
    assert _config("nesterov").hyperparameter_code.endswith(";nesterov=1")


@pytest.mark.parametrize(
    "code",
    [
        "rule=sgd;learning_rate=0.5",
        "rule=sgd;learning_rate=0x1.0p-1",
        "rule=sgd; learning_rate=0x1.0000000000000p-1",
        "rule=sgd;learning_rate=0x1.0000000000000p-1;momentum=0x0.0p+0",
        "learning_rate=0x1.0000000000000p-1;rule=sgd",
        "rule=unknown;learning_rate=0x1.0000000000000p-1",
        "rule=momentum;learning_rate=0x1.0000000000000p+0;momentum="
        "0x1.ccccccccccccdp-1;nesterov=2",
        "",
        "x" * 1024,
        "rule=sgd;learning_rate=0x1.0000000000000p-1" + chr(0xE9),
    ],
)
def test_noncanonical_hyperparameter_codes_are_rejected(code: str) -> None:
    with pytest.raises(ServerOptimizerError) as excinfo:
        ServerOptimizerConfig.from_hyperparameter_code(code)
    assert _code(excinfo) in {"invalid_hyperparameter_code", "invalid_rule"}
    assert code not in str(excinfo.value) or not code


@pytest.mark.parametrize(
    ("kwargs", "code", "field"),
    [
        ({"rule": "sgd", "learning_rate": 0.0}, "invalid_hyperparameter", "lr"),
        ({"rule": "sgd", "learning_rate": math.inf}, "invalid_hyperparameter", "lr"),
        ({"rule": "sgd", "learning_rate": True}, "invalid_hyperparameter", "lr"),
        ({"rule": "sgd", "learning_rate": 1e9}, "invalid_hyperparameter", "lr"),
        (
            {"rule": "sgd", "learning_rate": 1.0, "momentum": 0.9},
            "invalid_hyperparameter",
            "momentum",
        ),
        (
            {"rule": "sgd", "learning_rate": 1.0, "nesterov": True},
            "invalid_hyperparameter",
            "nesterov",
        ),
        (
            {"rule": "momentum", "learning_rate": 1.0, "momentum": 1.0},
            "invalid_hyperparameter",
            "momentum",
        ),
        (
            {"rule": "momentum", "learning_rate": 1.0, "beta1": 0.9},
            "invalid_hyperparameter",
            "beta1",
        ),
        (
            {"rule": "adagrad", "learning_rate": 0.1, "beta2": 0.99},
            "invalid_hyperparameter",
            "beta2",
        ),
        (
            {"rule": "adam", "learning_rate": 0.1, "beta2": -0.1},
            "invalid_hyperparameter",
            "beta2",
        ),
        (
            {"rule": "adam", "learning_rate": 0.1, "tau": 0.0},
            "invalid_hyperparameter",
            "tau",
        ),
        (
            {"rule": "yogi", "learning_rate": 0.1, "initial_accumulator": 1e-9},
            "invalid_hyperparameter",
            "initial_accumulator",
        ),
        ({"rule": "fedprox", "learning_rate": 0.1}, "invalid_rule", None),
    ],
)
def test_config_rejects_undeclared_or_out_of_range_hyperparameters(
    kwargs: dict, code: str, field: str | None
) -> None:
    with pytest.raises(ServerOptimizerError) as excinfo:
        ServerOptimizerConfig(**kwargs)
    assert _code(excinfo) == code
    expected_field = "learning_rate" if field == "lr" else field
    assert excinfo.value.field == expected_field
    assert str(excinfo.value) == (
        code if expected_field is None else f"{code}: {expected_field}"
    )


# Parameter schemas and bounds.


def test_parameter_schema_is_canonical_and_bounded() -> None:
    first = ServerParameterSchema.from_lengths({B: 2, A: 3})
    second = ServerParameterSchema.from_lengths({A: 3, B: 2})

    assert first.names == (A, B)
    assert first.total_elements == 5
    assert first.digest == second.digest
    assert first.digest != ServerParameterSchema.from_lengths({A: 3, B: 3}).digest
    with pytest.raises(ServerOptimizerError) as excinfo:
        ServerParameterSchema.from_lengths({})
    assert _code(excinfo) == "invalid_parameter_schema"
    for lengths in ({"bad name": 1}, {A: 0}, {A: True}, {A: 1.0}):
        with pytest.raises(ServerOptimizerError) as excinfo:
            ServerParameterSchema.from_lengths(lengths)
        assert _code(excinfo) == "invalid_parameter_schema"
        assert "bad name" not in str(excinfo.value)
    too_many = {f"p{i}": 1 for i in range(MAX_SERVER_OPTIMIZER_PARAMETERS + 1)}
    with pytest.raises(ServerOptimizerError) as excinfo:
        ServerParameterSchema.from_lengths(too_many)
    assert _code(excinfo) == "parameter_limit_exceeded"
    with pytest.raises(ServerOptimizerError) as excinfo:
        ServerParameterSchema.from_lengths({A: MAX_SERVER_OPTIMIZER_ELEMENTS + 1})
    assert _code(excinfo) == "parameter_limit_exceeded"


def test_parameter_schema_binds_validated_update_schema_fingerprint() -> None:
    parameters = (
        FederatedParameterMetadata(A, (1, 3), "float32"),
        FederatedParameterMetadata(B, (2, 1), "float32"),
    )
    policy = FederatedUpdatePolicy(
        model_digest="sha256:" + "a" * 64, parameters=parameters
    )
    metadata = FederatedUpdateMetadata(
        model_digest=policy.model_digest,
        adapter_format="dense",
        parameters=parameters,
        total_elements=5,
        update_digest="sha256:" + "b" * 64,
        clipped=True,
        policy=policy,
    )
    schema = ServerParameterSchema.from_update_metadata(metadata)

    assert schema.parameters == ((A, 3), (B, 2))
    assert schema.update_schema_fingerprint == fingerprint_update_schema(metadata)
    assert schema.digest != _schema().digest
    with pytest.raises(ServerOptimizerError) as excinfo:
        not_metadata: object = {"parameters": []}
        ServerParameterSchema.from_update_metadata(not_metadata)  # type: ignore
    assert _code(excinfo) == "invalid_parameter_schema"


# Input validation with fixed codes.


@pytest.mark.parametrize(
    ("adapter", "delta", "code"),
    [
        (X0, {A: (0.1, 0.2, 0.3)}, "schema_mismatch"),
        (X0, {**DELTAS[0], SENTINEL_TEXT: (1.0,)}, "schema_mismatch"),
        (X0, {A: (0.1, 0.2), B: (0.0, 0.0)}, "schema_mismatch"),
        (X0, {A: (0.1, math.nan, 0.3), B: (0.0, 0.0)}, "non_finite_value"),
        (X0, {A: (0.1, math.inf, 0.3), B: (0.0, 0.0)}, "non_finite_value"),
        (X0, {A: (0.1, True, 0.3), B: (0.0, 0.0)}, "invalid_values"),
        (X0, {A: "abc", B: (0.0, 0.0)}, "invalid_values"),
        (X0, [(A, (0.1, 0.2, 0.3))], "invalid_values"),
        ({A: (0.5, -0.25, 1.0), B: (0.0, math.nan)}, DELTAS[0], "non_finite_value"),
    ],
)
def test_apply_rejects_mismatched_or_invalid_inputs(adapter, delta, code) -> None:
    state = initial_server_optimizer_state(_config("adam"), _schema())
    with pytest.raises(ServerOptimizerError) as excinfo:
        apply_server_update(state, adapter, delta)
    assert _code(excinfo) == code
    assert SENTINEL_TEXT not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_apply_rejects_overflowing_updates() -> None:
    state = initial_server_optimizer_state(
        ServerOptimizerConfig(rule="sgd", learning_rate=100.0), _schema()
    )
    huge = {A: (1e308, 0.0, 0.0), B: (0.0, 0.0)}
    with pytest.raises(ServerOptimizerError) as excinfo:
        apply_server_update(state, X0, huge)
    assert _code(excinfo) == "non_finite_value"


def test_apply_rejects_non_state() -> None:
    with pytest.raises(ServerOptimizerError) as excinfo:
        apply_server_update(object(), X0, DELTAS[0])  # type: ignore[arg-type]
    assert _code(excinfo) == "invalid_state"


def test_error_codes_are_a_closed_set() -> None:
    assert "state_integrity_mismatch" in SERVER_OPTIMIZER_ERROR_CODES
    assert ServerOptimizerError("not-a-code").code == "invalid_state"


# Checkpoint, restore and restart.


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_restart_from_checkpoint_reproduces_uninterrupted_run(name: str) -> None:
    uninterrupted_adapter, uninterrupted_state, uninterrupted_records = _run(name)

    adapter, state, records = _run(name, rounds=1)
    checkpoint = state.to_checkpoint_bytes()
    lineage_digest = records[-1].state_digest
    restored = restore_server_optimizer_state(
        checkpoint,
        config=ServerOptimizerConfig.from_hyperparameter_code(
            records[-1].hyperparameter_code
        ),
        schema=_schema(),
        expected_step=1,
        expected_state_digest=lineage_digest,
    )
    assert restored.state_digest == state.state_digest
    assert restored.to_checkpoint_bytes() == checkpoint
    resumed_adapter, resumed_state, resumed_records = _run(
        name, rounds=2, start=restored, adapter=adapter
    )

    assert _hex(resumed_adapter) == _hex(uninterrupted_adapter) == GOLDEN[name]
    assert (
        resumed_state.to_checkpoint_bytes() == uninterrupted_state.to_checkpoint_bytes()
    )
    assert [r.to_json() for r in records + resumed_records] == [
        r.to_json() for r in uninterrupted_records
    ]


def test_checkpoint_bytes_are_canonical_versioned_json() -> None:
    _, state, _ = _run("adam", rounds=2)
    payload = state.to_checkpoint_bytes()
    decoded = json.loads(payload)

    assert payload.isascii()
    assert payload == json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode(
        "ascii"
    )
    assert set(decoded) == {
        "schema_version",
        "rule",
        "hyperparameter_code",
        "config_digest",
        "parameter_schema",
        "parameter_schema_digest",
        "step",
        "slots",
        "state_digest",
    }
    assert decoded["schema_version"] == SERVER_OPTIMIZER_STATE_SCHEMA_VERSION
    assert decoded["state_digest"] == state.state_digest
    assert set(decoded["slots"]) == {"first_moment", "second_moment"}
    assert set(decoded["slots"]["first_moment"]) == {A, B}


def _restore(payload: bytes, name: str = "adam", **overrides):
    kwargs = {"config": _config(name), "schema": _schema(), "expected_step": 2}
    kwargs.update(overrides)
    return restore_server_optimizer_state(payload, **kwargs)


def _resign(decoded: dict) -> bytes:
    """Rebuild a payload as a forger who recomputes only the JSON layout."""

    return json.dumps(decoded, sort_keys=True, separators=(",", ":")).encode()


def test_restore_rejects_tampered_slot_bytes() -> None:
    _, state, _ = _run("adam", rounds=2)
    decoded = json.loads(state.to_checkpoint_bytes())
    original = decoded["slots"]["second_moment"][A]
    replacement = "A" if original[0] != "A" else "B"
    decoded["slots"]["second_moment"][A] = replacement + original[1:]

    with pytest.raises(ServerOptimizerError) as excinfo:
        _restore(_resign(decoded))
    assert _code(excinfo) == "state_integrity_mismatch"


@pytest.mark.parametrize(
    ("field", "value"),
    [("step", 3), ("rule", "yogi"), ("config_digest", "sha256:" + "0" * 64)],
)
def test_restore_rejects_tampered_header_fields(field: str, value: object) -> None:
    _, state, _ = _run("adam", rounds=2)
    decoded = json.loads(state.to_checkpoint_bytes())
    decoded[field] = value

    with pytest.raises(ServerOptimizerError) as excinfo:
        _restore(_resign(decoded), expected_step=decoded["step"])
    assert _code(excinfo) == "state_integrity_mismatch"


def test_restore_rejects_valid_state_not_bound_by_lineage() -> None:
    _, bound, _ = _run("adam", rounds=2)
    other = initial_server_optimizer_state(_config("adam"), _schema())
    other = apply_server_update(other, X0, DELTAS[1]).state
    other = apply_server_update(other, X0, DELTAS[0]).state

    assert _restore(other.to_checkpoint_bytes()).step == 2
    with pytest.raises(ServerOptimizerError) as excinfo:
        _restore(other.to_checkpoint_bytes(), expected_state_digest=bound.state_digest)
    assert _code(excinfo) == "state_lineage_mismatch"
    with pytest.raises(ServerOptimizerError) as excinfo:
        _restore(other.to_checkpoint_bytes(), expected_state_digest="sha256:xyz")
    assert _code(excinfo) == "invalid_state"


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"config": ServerOptimizerConfig(rule="yogi", learning_rate=0.1)}, "rule"),
        (
            {"config": ServerOptimizerConfig(rule="adam", learning_rate=0.2)},
            "hyperparameter",
        ),
        (
            {"config": ServerOptimizerConfig(rule="adam", learning_rate=0.1, tau=0.01)},
            "hyperparameter",
        ),
        ({"schema": ServerParameterSchema.from_lengths({A: 3, B: 3})}, "schema"),
        ({"schema": ServerParameterSchema.from_lengths({A: 3, "other": 2})}, "schema"),
        ({"expected_step": 1}, "step"),
        ({"expected_step": 3}, "step"),
    ],
)
def test_restore_rejects_mismatched_state(overrides: dict, code: str) -> None:
    _, state, _ = _run("adam", rounds=2)
    with pytest.raises(ServerOptimizerError) as excinfo:
        _restore(state.to_checkpoint_bytes(), **overrides)
    assert _code(excinfo) == f"state_{code}_mismatch"


def test_restore_rejects_malformed_payloads() -> None:
    _, state, _ = _run("adam", rounds=2)
    payload = state.to_checkpoint_bytes()
    decoded = json.loads(payload)

    cases = {
        "invalid_state": [
            payload.decode(),
            b"not json",
            b"[]",
            payload.replace(b'{"', b'{ "', 1),
            payload[:-1] + b',"rule":"adam"}',
            _resign({**decoded, "site_id": SENTINEL_TEXT}),
            _resign({k: v for k, v in decoded.items() if k != "slots"}),
        ],
        "unsupported_state_version": [
            _resign({**decoded, "schema_version": "openmed.other.v9"}),
        ],
        "state_too_large": [payload + b" " * (64 * 1024 * 1024)],
    }
    for code, payloads in cases.items():
        for candidate in payloads:
            with pytest.raises(ServerOptimizerError) as excinfo:
                _restore(candidate)  # type: ignore[arg-type]
            assert _code(excinfo) == code
            assert SENTINEL_TEXT not in str(excinfo.value)
            assert excinfo.value.__cause__ is None


def test_initial_state_round_trips_and_sgd_has_no_slots() -> None:
    sgd = initial_server_optimizer_state(_config("sgd"), _schema())
    decoded = json.loads(sgd.to_checkpoint_bytes())

    assert decoded["slots"] == {}
    assert decoded["step"] == 0
    restored = restore_server_optimizer_state(
        sgd.to_checkpoint_bytes(),
        config=_config("sgd"),
        schema=_schema(),
        expected_step=0,
    )
    assert restored.state_digest == sgd.state_digest
    momentum = initial_server_optimizer_state(_config("momentum"), _schema())
    assert set(json.loads(momentum.to_checkpoint_bytes())["slots"]) == {"velocity"}


# Privacy: no per-client data, only digests and hyperparameter codes.


def test_records_contain_only_digests_codes_and_counters() -> None:
    delta = {A: (SENTINEL_VALUE, 0.0, 0.0), B: (0.0, 0.0)}
    state = initial_server_optimizer_state(_config("yogi"), _schema())
    result = apply_server_update(state, X0, delta)
    record = result.record.to_dict()

    assert set(record) == {
        "schema_version",
        "rule",
        "hyperparameter_code",
        "config_digest",
        "parameter_schema_digest",
        "step",
        "previous_state_digest",
        "state_digest",
        "input_adapter_digest",
        "aggregate_delta_digest",
        "output_adapter_digest",
        "privacy_accounting",
    }
    assert record["schema_version"] == SERVER_UPDATE_RECORD_SCHEMA_VERSION
    assert record["privacy_accounting"] == SERVER_UPDATE_PRIVACY_ACCOUNTING
    assert record["privacy_accounting"] == "post_processing_no_additional_charge"
    assert record["previous_state_digest"] == state.state_digest
    assert record["state_digest"] == result.state.state_digest
    assert record["aggregate_delta_digest"] == digest_aggregate_delta(delta, _schema())
    assert record["input_adapter_digest"] == digest_adapter_values(X0, _schema())
    assert record["output_adapter_digest"] == digest_adapter_values(
        result.global_adapter, _schema()
    )
    assert record["aggregate_delta_digest"] != digest_adapter_values(delta, _schema())
    for key, value in record.items():
        if key.endswith("_digest"):
            assert value.startswith("sha256:") and len(value) == 71
        elif key == "step":
            assert value == 1
    rendered = result.record.to_json() + json.dumps(result.state.describe())
    for leaked in (repr(SENTINEL_VALUE), SENTINEL_VALUE.hex(), A, B):
        assert leaked not in rendered
    assert json.loads(result.record.to_json()) == record


def test_state_repr_and_description_hide_vectors() -> None:
    delta = {A: (SENTINEL_VALUE, 0.0, 0.0), B: (0.0, 0.0)}
    state = initial_server_optimizer_state(_config("momentum"), _schema())
    state = apply_server_update(state, X0, delta).state

    assert repr(SENTINEL_VALUE) not in repr(state)
    assert set(state.describe()) == {
        "schema_version",
        "rule",
        "hyperparameter_code",
        "config_digest",
        "parameter_schema_digest",
        "step",
        "state_digest",
    }
    assert state.slot_values("velocity")[A][0] == SENTINEL_VALUE
    with pytest.raises(ServerOptimizerError) as excinfo:
        state.slot_values("second_moment")
    assert _code(excinfo) == "invalid_state"


def test_checkpoint_carries_only_schema_codes_digests_and_aggregate_slots() -> None:
    _, state, _ = _run("adam", rounds=1)
    decoded = json.loads(state.to_checkpoint_bytes())

    assert decoded["parameter_schema"] == {
        "parameters": [[A, 3], [B, 2]],
        "update_schema_fingerprint": None,
    }
    for slot in decoded["slots"].values():
        assert set(slot) == {A, B}
        assert all(isinstance(value, str) for value in slot.values())
    keys: list[str] = []
    pending: list[object] = [decoded]
    while pending:
        node = pending.pop()
        if isinstance(node, dict):
            keys.extend(node)
            pending.extend(node.values())
    for forbidden in ("site", "client", "participant", "path", "endpoint"):
        assert not any(forbidden in key for key in keys)
    assert SENTINEL_TEXT not in state.to_checkpoint_bytes().decode()
