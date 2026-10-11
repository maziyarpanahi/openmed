"""Offline, synthetic privacy and failure controls for summary/NLI commands."""

from __future__ import annotations

import ctypes
import importlib
import json
import logging
import os
import stat
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmed.cli.main import main
from openmed.clinical.nli import verify
from openmed.clinical.summarize import LeakageCheck, SummarizationResult
from openmed.core.pii import DeidentificationResult, PIIEntity

SUMMARY = importlib.import_module("openmed.clinical.summarize")
CLI = importlib.import_module("openmed.cli.clinical_review")
SOURCE = "Patient Casey Example presented with a cough. The synthetic admission was uncomplicated."
DEIDENTIFIED = SOURCE.replace("Casey Example", "[NAME]")
CANARY = "SYNTHETIC_PRIVATE_CANARY"


@pytest.fixture(autouse=True)
def local_flags(monkeypatch):
    from openmed.core.offline import HF_OFFLINE_ENV_VARS

    for key in HF_OFFLINE_ENV_VARS:
        monkeypatch.setenv(key, "1")


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "synthetic-note.txt"
    source.write_text(SOURCE)
    claims = tmp_path / "synthetic-claims.json"
    claims.write_text(json.dumps(["Casey Example presented with a cough."]))
    return source, claims, tmp_path / "summary.txt", tmp_path / "metadata.json"


@pytest.fixture
def local_deid(monkeypatch):
    calls = []

    def deidentify(text, **kwargs):
        calls.append((text, kwargs))
        return DeidentificationResult(
            original_text=text,
            deidentified_text=text.replace("Casey Example", "[NAME]"),
            pii_entities=[
                PIIEntity(
                    text="Casey Example",
                    label="NAME",
                    start=8,
                    end=21,
                    confidence=0.99,
                    redacted_text="[NAME]",
                )
            ],
            method="mask",
            timestamp=datetime(2026, 1, 1),
        )

    monkeypatch.setattr(SUMMARY, "deidentify", deidentify)
    return calls


def summary_args(inputs, *options):
    source, _, summary, metadata = inputs
    return [
        "summarize",
        str(source),
        "--summary-output",
        str(summary),
        "--metadata-output",
        str(metadata),
        *options,
    ]


def nli_args(inputs, *options):
    source, claims, _, _ = inputs
    return ["nli", "verify", "--source", str(source), "--claims", str(claims), *options]


def test_extractive_summary_uses_native_guard_and_separate_private_files(
    inputs, local_deid, capfd
):
    previous_logging = logging.root.manager.disable
    expected = SUMMARY.summarize(SOURCE, model="extractive")
    assert main(summary_args(inputs, "--model", "extractive", "--json")) == 0
    captured = capfd.readouterr()
    data = json.loads(captured.out)
    assert captured.err == ""
    assert data["ok"] is True and data["command"] == "summarize"
    assert inputs[2].read_text() == expected.summary
    assert json.loads(inputs[3].read_text()) == data
    assert data["data"]["leakage_check"] == expected.leakage_check.to_dict()
    assert data["data"]["template_digest"] == expected.template_digest
    assert data["data"]["backend_id"] == "deterministic-extractive"
    assert local_deid[-1][1]["config"].local_only is True
    for path in inputs[2:]:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    for text in (SOURCE, DEIDENTIFIED, expected.summary, "Casey Example"):
        assert text not in captured.out + captured.err + inputs[3].read_text()
    assert logging.root.manager.disable == previous_logging


def test_real_guard_refuses_source_identifier_from_local_backend(
    inputs, local_deid, monkeypatch, capfd
):
    def leaking(_self, text, *, mode):
        print(CANARY)
        os.write(2, CANARY.encode())
        return "Casey Example has a cough."

    import openmed.clinical.summarize_backends as backends

    monkeypatch.setattr(backends.ExtractiveSummarizerBackend, "summarize", leaking)
    assert main(summary_args(inputs, "--model", "extractive", "--json")) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "summary_leakage_rejected"
    assert CANARY not in captured.out + captured.err
    assert "Casey Example" not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])


@pytest.mark.parametrize("kind", ["mlx", "admission", "unexpected"])
def test_unavailable_summary_is_fixed_and_has_no_partial_outputs(
    kind, inputs, monkeypatch, capfd
):
    from openmed.clinical.summarize_backends import LocalSummarizerError
    from openmed.core.capabilities import MissingOptionalDependencyError

    def failed(*args, **kwargs):
        print(CANARY)
        if kind == "mlx":
            raise MissingOptionalDependencyError(
                package="mlx", feature=CANARY, extra="mlx"
            )
        if kind == "admission":
            raise LocalSummarizerError(CANARY)
        raise RuntimeError(CANARY)

    monkeypatch.setattr(SUMMARY, "summarize", failed)
    assert main(summary_args(inputs, "--json")) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == (
        "summary_failed" if kind == "unexpected" else "summary_backend_unavailable"
    )
    assert CANARY not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])


@pytest.mark.parametrize(
    "field,value",
    [
        ("backend", CANARY),
        ("template_digest", CANARY),
        ("summary", "x" * 8193),
        ("mode", CANARY),
    ],
)
def test_summary_projection_refuses_malformed_metadata(
    field, value, inputs, monkeypatch, capfd
):
    result = SummarizationResult(
        "Synthetic protected output",
        LeakageCheck(True, 1, 0),
        backend="deterministic-extractive",
        template_digest="sha256:" + "1" * 64,
    )
    object.__setattr__(result, field, value)
    monkeypatch.setattr(SUMMARY, "summarize", lambda *args, **kwargs: result)
    assert main(summary_args(inputs, "--json")) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "summary_result_invalid"
    assert CANARY not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])


@pytest.mark.parametrize("which", [2, 3, "same", "symlink"])
def test_existing_output_never_overwritten_and_reservations_cleaned(
    which, inputs, local_deid, capfd
):
    source, claims, summary, metadata = inputs
    args = summary_args(inputs, "--model", "extractive", "--json")
    if which in (2, 3):
        inputs[which].write_text(CANARY)
    elif which == "same":
        args = summary_args(
            (source, claims, summary, summary), "--model", "extractive", "--json"
        )
    else:
        metadata.symlink_to(source)
    assert main(args) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "summary_output_failed"
    if which in (2, 3):
        assert inputs[which].read_text() == CANARY
        assert not inputs[5 - which].exists()
    else:
        assert not summary.exists()
    assert source.read_text() == SOURCE


def test_partial_file_writes_are_removed(inputs, local_deid, monkeypatch, capfd):
    original = os.fdopen

    class FailingStream:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def write(self, content):
            self.wrapped.write(content[:3])
            raise OSError(CANARY)

    def opened(descriptor, mode, *args, **kwargs):
        stream = original(descriptor, mode, *args, **kwargs)
        return FailingStream(stream) if mode == "wb" else stream

    monkeypatch.setattr(os, "fdopen", opened)
    assert main(summary_args(inputs, "--model", "extractive", "--json")) == 1
    assert (
        json.loads(capfd.readouterr().out)["error"]["code"] == "summary_output_failed"
    )
    assert not any(path.exists() for path in inputs[2:])


@pytest.mark.parametrize("command", ["summary", "nli"])
@pytest.mark.parametrize(
    "kind",
    ["directory", "fifo", "symlink", "oversized", "invalid-utf8", "empty", "missing"],
)
def test_bounded_regular_input_and_fixed_errors(command, kind, inputs, tmp_path, capfd):
    path = tmp_path / "invalid-input"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "symlink":
        path.symlink_to(inputs[0])
    elif kind == "oversized":
        path.write_bytes(b"x" * 16385)
    elif kind == "invalid-utf8":
        path.write_bytes(b"\xff")
    elif kind == "empty":
        path.write_text(" ")
    changed = (path, *inputs[1:])
    args = (
        summary_args(changed, "--model", "extractive", "--json")
        if command == "summary"
        else nli_args(changed, "--backend", "heuristic", "--json")
    )
    assert main(args) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"].startswith(command + "_input_")
    assert str(path) not in captured.out + captured.err


def test_input_replacement_during_open_is_refused(inputs, monkeypatch, capfd):
    original_open = os.open
    original_path = inputs[0].with_suffix(".original")

    def swapped(path, flags, *args, **kwargs):
        if path == str(inputs[0]):
            inputs[0].rename(original_path)
            inputs[0].write_text(CANARY)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swapped)
    assert main(summary_args(inputs, "--model", "extractive", "--json")) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "summary_input_not_regular"
    assert CANARY not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])


def test_symlink_input_is_refused_without_platform_nofollow_flag(
    inputs, monkeypatch, capfd
):
    path = inputs[0].with_suffix(".link")
    path.symlink_to(inputs[0])
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    assert main(nli_args((path, *inputs[1:]), "--backend", "heuristic", "--json")) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_input_not_regular"
    assert str(path) not in captured.out + captured.err


def test_human_console_also_omits_clinical_values(inputs, local_deid, capfd):
    assert main(summary_args(inputs, "--model", "extractive")) == 0
    captured = capfd.readouterr()
    assert (
        captured.out
        == "Summary files written; qualified clinical review is required.\n"
    )
    assert captured.err == ""
    assert main(nli_args(inputs, "--backend", "heuristic")) in (0, 1)
    captured = capfd.readouterr()
    assert "Casey Example" not in captured.out + captured.err
    assert set(json.loads(captured.out)["claims"][0]) == CLI._NLI_FIELDS
    assert captured.err == ""


def test_human_error_is_fixed_and_does_not_echo_path(inputs, capfd):
    inputs[0].unlink()
    assert main(nli_args(inputs, "--backend", "heuristic")) == 2
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == "Local clinical request could not complete.\n"


@pytest.mark.parametrize(
    "argv,code",
    [
        (["summarize", CANARY, "--json"], "summary_arguments_invalid"),
        (
            ["summarize", CANARY, "--bad=" + CANARY, "--json"],
            "summary_arguments_invalid",
        ),
        (
            ["--bad=" + CANARY, "summarize", CANARY, "--json"],
            "summary_arguments_invalid",
        ),
        (["nli", "verify", "--source", CANARY, "--json"], "nli_arguments_invalid"),
        (["nli", CANARY, "--json"], "nli_arguments_invalid"),
        (
            ["--config-path", "summarize", CANARY, "--json"],
            "summary_arguments_invalid",
        ),
    ],
)
def test_invalid_arguments_never_echo_values(argv, code, capfd):
    assert main(argv) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == code
    assert CANARY not in captured.out + captured.err


@pytest.mark.parametrize(
    "model", ["remote", "https://private.example/model", "openai", CANARY]
)
def test_remote_summary_aliases_fail_before_processing(model, inputs, capfd):
    assert main(summary_args(inputs, "--model", model, "--json")) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "summary_model_invalid"
    assert model not in captured.out + captured.err


def test_unsupported_mode_has_stable_code(inputs, capfd):
    assert main(summary_args(inputs, "--mode", CANARY, "--json")) == 2
    assert (
        json.loads(capfd.readouterr().out)["error"]["code"]
        == "summary_mode_unsupported"
    )


def test_heuristic_is_explicit_and_native_projection_matches(inputs, capfd):
    claims = json.loads(inputs[1].read_text())
    expected = verify(claims, SOURCE, backend="heuristic")
    assert main(nli_args(inputs, "--backend", "heuristic", "--json")) == int(
        any(v["label"] != "entailment" for v in expected)
    )
    captured = capfd.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "ok": True,
        "command": "nli verify",
        "data": {"claims": expected},
    }
    assert SOURCE not in captured.out and claims[0] not in captured.out
    assert not any(path.exists() for path in inputs[2:])


def test_missing_default_checkpoint_does_not_fall_back(inputs, monkeypatch, capfd):
    monkeypatch.setattr(
        "openmed.clinical.nli_backends.get_default_nli_model", lambda: None
    )
    assert main(nli_args(inputs, "--json")) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_backend_unavailable"
    assert "Casey Example" not in captured.out + captured.err


@pytest.mark.parametrize(
    "backend", ["remote", "https://private.example/model", "openai", CANARY]
)
def test_remote_nli_names_never_run(backend, inputs, capfd):
    assert main(nli_args(inputs, "--backend", backend, "--json")) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_backend_invalid"
    assert backend not in captured.out + captured.err


@pytest.mark.parametrize(
    "claims",
    [
        [],
        [""],
        [1],
        {"claim": CANARY},
        ["x" * 4097],
        ["x"] * 129,
        "not-json",
        ["\ud800"],
    ],
)
def test_claims_are_bounded_string_arrays(claims, inputs, capfd):
    inputs[1].write_text(claims if claims == "not-json" else json.dumps(claims))
    assert main(nli_args(inputs, "--backend", "heuristic", "--json")) == 2
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_claims_invalid"
    assert CANARY not in captured.out + captured.err


@pytest.mark.parametrize(
    "label", ["entailment", "contradiction", "neutral", "abstention"]
)
def test_caller_local_backend_runs_offline_and_console_chatter_is_discarded(
    label, inputs, monkeypatch, capfd
):
    previous_logging = logging.root.manager.disable

    class Backend:
        backend_id = "synthetic-private-name"

        def predict(self, premise, claim):
            assert premise == SOURCE
            assert claim == json.loads(inputs[1].read_text())[0]
            print(CANARY)
            print(CANARY, file=sys.stderr)
            os.write(1, CANARY.encode())
            os.write(2, CANARY.encode())
            logging.error(CANARY)
            ctypes.CDLL(None).printf(CANARY.encode())
            return {"label": label, "score": 0.8}

    monkeypatch.setitem(
        sys.modules,
        "synthetic_local_backend",
        SimpleNamespace(create=lambda: Backend()),
    )
    assert main(
        nli_args(
            inputs, "--backend-factory", "synthetic_local_backend:create", "--json"
        )
    ) == int(label != "entailment")
    captured = capfd.readouterr()
    row = json.loads(captured.out)["data"]["claims"][0]
    assert row == {
        "claim_index": 0,
        "label": label,
        "score": 0.8,
        "backend_id": "caller-supplied-local",
        "contradicted": label == "contradiction",
        "review_required": label == "abstention",
    }
    assert CANARY not in captured.out + captured.err
    assert "synthetic-private-name" not in captured.out
    assert captured.err == ""
    assert logging.root.manager.disable == previous_logging


def test_factory_cannot_silently_select_heuristic(inputs, monkeypatch, capfd):
    from openmed.clinical.nli import HeuristicNLIBackend

    monkeypatch.setitem(
        sys.modules,
        "synthetic_local_backend",
        SimpleNamespace(create=HeuristicNLIBackend),
    )
    assert (
        main(
            nli_args(
                inputs, "--backend-factory", "synthetic_local_backend:create", "--json"
            )
        )
        == 2
    )
    assert json.loads(capfd.readouterr().out)["error"]["code"] == "nli_backend_invalid"


def test_network_attempt_from_caller_backend_is_refused(inputs, monkeypatch, capfd):
    import socket

    def factory():
        socket.create_connection(("192.0.2.1", 443))

    monkeypatch.setitem(
        sys.modules, "synthetic_local_backend", SimpleNamespace(create=factory)
    )
    assert (
        main(
            nli_args(
                inputs, "--backend-factory", "synthetic_local_backend:create", "--json"
            )
        )
        == 1
    )
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_failed"
    assert "192.0.2.1" not in captured.out + captured.err


@pytest.mark.parametrize(
    "result",
    [
        {"label": "unknown", "score": 0.5},
        {"label": "entailment", "score": float("nan")},
        {"label": "entailment", "score": True},
    ],
)
def test_malformed_backend_result_is_fixed(result, inputs, monkeypatch, capfd):
    monkeypatch.setitem(
        sys.modules,
        "synthetic_local_backend",
        SimpleNamespace(create=lambda: lambda *args: result),
    )
    assert (
        main(
            nli_args(
                inputs, "--backend-factory", "synthetic_local_backend:create", "--json"
            )
        )
        == 1
    )
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "nli_failed"
    assert captured.err == ""


@pytest.mark.parametrize("command", ("summary", "nli"))
@pytest.mark.parametrize("foreign", (False, True))
def test_provider_error_codes_and_getters_cannot_cross_cli(
    command, foreign, inputs, monkeypatch, capfd
):
    from openmed.cli._output import CliError

    class ForeignCliError(CliError):
        def __init__(self):
            Exception.__init__(self, CANARY)

        @property
        def code(self):
            raise AssertionError(CANARY)

    error = ForeignCliError() if foreign else CliError(CANARY, code=CANARY)

    def fail(*args, **kwargs):
        raise error

    if command == "summary":
        monkeypatch.setattr(SUMMARY, "summarize", fail)
        args = summary_args(inputs, "--json")
    else:
        monkeypatch.setattr(CLI, "_nli_backend", fail)
        args = nli_args(inputs, "--json")
    assert main(args) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == command + "_failed"
    assert CANARY not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])


@pytest.mark.parametrize("command", ("summary", "nli"))
def test_diagnostic_string_subclasses_cannot_masquerade_as_cli_metadata(
    command, inputs, monkeypatch, capfd
):
    allowed = "deterministic-extractive" if command == "summary" else "heuristic"

    class PrivateBackend(str):
        def __hash__(self):
            return hash(allowed)

        def __eq__(self, other):
            return other == allowed

    if command == "summary":
        result = SummarizationResult(
            "Synthetic protected output",
            LeakageCheck(True, 1, 0),
            backend="deterministic-extractive",
            template_digest="sha256:" + "1" * 64,
        )
        object.__setattr__(result, "backend", PrivateBackend(CANARY))
        monkeypatch.setattr(SUMMARY, "summarize", lambda *args, **kwargs: result)
        args = summary_args(inputs, "--json")
    else:
        nli = importlib.import_module("openmed.clinical.nli")

        values = verify(json.loads(inputs[1].read_text()), SOURCE, backend="heuristic")
        values[0]["backend_id"] = PrivateBackend(CANARY)
        monkeypatch.setattr(nli, "verify", lambda *args, **kwargs: values)
        args = nli_args(inputs, "--backend", "heuristic", "--json")
    assert main(args) == 1
    captured = capfd.readouterr()
    assert json.loads(captured.out)["error"]["code"] == command + "_result_invalid"
    assert CANARY not in captured.out + captured.err
    assert not any(path.exists() for path in inputs[2:])
