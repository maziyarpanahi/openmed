from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openmed.integrations import executable_udf


def _fake_redact(text: str) -> str:
    return (
        text.replace("Jane Roe", "[NAME]")
        .replace("John Doe", "[NAME]")
        .replace("555-0100", "[PHONE]")
        .replace("jane.roe@example.org", "[EMAIL]")
    )


def _batch_result(texts: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        items=[
            SimpleNamespace(
                success=True,
                result=SimpleNamespace(deidentified_text=_fake_redact(text)),
            )
            for text in texts
        ]
    )


def test_stdin_stdout_adapter_redacts_every_row_in_order_and_batches(
    monkeypatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    shared_loader = object()

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        calls.append((list(texts), dict(kwargs)))
        return _batch_result(list(texts))

    monkeypatch.setattr(executable_udf, "process_batch", fake_process_batch)
    input_rows = [
        "Patient Jane Roe called 555-0100",
        "No identifiers in this row",
        "Escalate John Doe",
        "",
        "Email jane.roe@example.org",
    ]
    stdin = io.StringIO("\n".join(input_rows) + "\n")
    stdout = io.StringIO()

    emitted = executable_udf.redact_tsv_stream(
        stdin,
        stdout,
        batch_size=2,
        loader=shared_loader,
        use_safety_sweep=False,
    )

    output_rows = stdout.getvalue().splitlines()
    assert emitted == len(input_rows)
    assert len(output_rows) == len(input_rows)
    assert output_rows == [
        "Patient [NAME] called [PHONE]",
        "No identifiers in this row",
        "Escalate [NAME]",
        "",
        "Email [EMAIL]",
    ]
    assert "Jane Roe" not in stdout.getvalue()
    assert "John Doe" not in stdout.getvalue()
    assert "555-0100" not in stdout.getvalue()
    assert "jane.roe@example.org" not in stdout.getvalue()
    assert [texts for texts, _ in calls] == [
        input_rows[:2],
        input_rows[2:3],
        input_rows[4:],
    ]
    assert all(kwargs["operation"] == "deidentify" for _, kwargs in calls)
    assert all(kwargs["continue_on_error"] is False for _, kwargs in calls)
    assert all(kwargs["loader"] is shared_loader for _, kwargs in calls)
    assert [kwargs["batch_size"] for _, kwargs in calls] == [2, 1, 1]
    assert all(kwargs["use_safety_sweep"] is False for _, kwargs in calls)


def test_tsv_escaping_round_trips_control_characters(monkeypatch) -> None:
    received: list[str] = []

    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        received.extend(texts)
        return _batch_result(list(texts))

    monkeypatch.setattr(executable_udf, "process_batch", fake_process_batch)
    stdout = io.StringIO()

    executable_udf.redact_tsv_stream(
        io.StringIO("Jane Roe\\tcalled\\nagain\\\\soon\n"),
        stdout,
        loader=object(),
    )

    assert received == ["Jane Roe\tcalled\nagain\\soon"]
    assert stdout.getvalue() == "[NAME]\\tcalled\\nagain\\\\soon\n"


def test_cli_failure_does_not_echo_or_log_raw_phi(monkeypatch, caplog) -> None:
    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        raise RuntimeError(f"model failed on {texts[0]}")

    monkeypatch.setattr(executable_udf, "process_batch", fake_process_batch)
    stdout = io.StringIO()
    stderr = io.StringIO()

    code = executable_udf.main(
        ["--batch-size", "2"],
        stdin=io.StringIO("Patient Jane Roe called\n"),
        stdout=stdout,
        stderr=stderr,
    )

    assert code == 1
    assert stdout.getvalue() == ""
    assert stderr.getvalue() == "executable UDF redaction failed\n"
    assert "Jane Roe" not in stderr.getvalue()
    assert "Jane Roe" not in caplog.text


def test_cardinality_mismatch_fails_closed(monkeypatch) -> None:
    def fake_process_batch(texts: list[str], **kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(items=[])

    monkeypatch.setattr(executable_udf, "process_batch", fake_process_batch)

    try:
        list(
            executable_udf.redact_tsv_lines(
                ["Jane Roe\n"],
                loader=object(),
            )
        )
    except executable_udf.ExecutableUDFError as exc:
        assert "result count" in str(exc)
        assert "Jane Roe" not in str(exc)
    else:  # pragma: no cover
        raise AssertionError("cardinality mismatch should fail closed")


def test_console_script_points_to_executable_udf_entrypoint() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert (
        'openmed-executable-udf = "openmed.integrations.executable_udf:main"'
        in pyproject
    )
