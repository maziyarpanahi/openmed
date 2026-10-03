"""Synthetic brief transport parity and privacy tests, not model evaluation."""

import argparse
import json
import logging
import stat
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from openmed.cli._output import CliError
from openmed.cli.brief import handle_brief
from openmed.service.app import create_app
from openmed.service.brief import brief_response, brief_response_schema
from openmed.service.client import OpenMedClient
from openmed.service.logging import ACCESS_LOGGER_NAME
from tests.unit.clinical.test_brief import fixture_context
from tests.unit.service.test_rest_client import SyncASGITransport

REVIEW_ID = "a" * 64


def setup_review():
    value, context = fixture_context()
    provider = lambda text, review_id: (value, context)
    expected = brief_response(
        value.original_text,
        model="extractive",
        review_id=REVIEW_ID,
        context_provider=provider,
    )
    assert expected["status"] == "needs_review"
    return value, provider, expected


def test_rest_python_mcp_and_schema_parity(caplog):
    value, provider, expected = setup_review()
    app = create_app()
    app.state.brief_context_provider = provider
    caplog.set_level(logging.INFO, logger=ACCESS_LOGGER_NAME)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post(
            "/brief",
            json={
                "text": value.original_text,
                "model": "extractive",
                "review_id": REVIEW_ID,
            },
        )
    assert response.status_code == 200
    assert response.json() == expected
    client_app = create_app()
    client_app.state.brief_context_provider = provider
    with OpenMedClient(transport=SyncASGITransport(client_app)) as client:
        assert (
            client.brief(value.original_text, model="extractive", review_id=REVIEW_ID)
            == expected
        )
    assert set(expected) == set(brief_response_schema()["required"])
    from openmed.mcp.server import openmed_brief
    from openmed.mcp.tool_registry import TOOL_REGISTRY

    actual = openmed_brief(
        value.original_text,
        model="extractive",
        review_id=REVIEW_ID,
        runtime_provider=lambda: SimpleNamespace(brief_context_provider=provider),
    )
    assert actual == expected
    assert TOOL_REGISTRY.get("openmed_brief").annotations()["readOnlyHint"] is True
    records = [
        r.openmed_access_log for r in caplog.records if r.name == ACCESS_LOGGER_NAME
    ]
    assert records and records[0]["brief_claim_count"] == 3
    assert value.original_text not in json.dumps(records)
    assert REVIEW_ID not in json.dumps(records)
    assert "dehydration" not in json.dumps(records)


@pytest.mark.parametrize(
    "change",
    [
        {"model": "https://remote.example"},
        {"profile": "unknown"},
        {"review_id": "invalid"},
        {"text": 123},
        {"text": ""},
        {"text": "é" * 9000},
        {"context": {"approved": True}},
    ],
)
def test_rest_rejects_untrusted_configuration(change):
    payload = {"text": "Synthetic note", "model": "extractive", "review_id": REVIEW_ID}
    payload.update(change)
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.post("/brief", json=payload)
    assert response.status_code == 422
    assert "Synthetic note" not in response.text


def test_missing_review_and_mismatched_provider_refuse():
    response = brief_response("Synthetic note", model="extractive", review_id=REVIEW_ID)
    assert response["refusal_reason"] == "review_required"
    value, context = fixture_context()
    response = brief_response(
        "different note",
        model="extractive",
        review_id=REVIEW_ID,
        context_provider=lambda *args: (value, context),
    )
    assert response["refusal_reason"] == "invalid_evidence"
    assert response["summary"] == ""


def cli_args(tmp_path, text):
    source = tmp_path / "source.txt"
    source.write_text(text)
    return argparse.Namespace(
        path=source,
        model="extractive",
        profile="bhc",
        review_id=REVIEW_ID,
        summary_output=tmp_path / "summary.txt",
        review_output=tmp_path / "review.json",
        json_output=True,
        command="brief",
    )


def test_cli_separates_protected_content_and_private_audit(tmp_path, capsys):
    value, provider, expected = setup_review()
    args = cli_args(tmp_path, value.original_text)
    assert handle_brief(args, context_provider=provider) == 0
    assert args.summary_output.read_text() == expected["summary"]
    audit = json.loads(args.review_output.read_text())
    assert "summary" not in audit
    assert "dehydration" not in args.review_output.read_text()
    assert "dehydration" not in capsys.readouterr().out
    for path in (args.summary_output, args.review_output):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize("collision", ["existing", "same", "symlink"])
def test_cli_never_overwrites_outputs(tmp_path, collision):
    value, provider, _ = setup_review()
    args = cli_args(tmp_path, value.original_text)
    protected = tmp_path / "protected.txt"
    protected.write_text("keep")
    if collision == "existing":
        args.review_output = protected
    elif collision == "same":
        args.review_output = args.summary_output
    else:
        args.review_output.symlink_to(protected)
    with pytest.raises(CliError, match="Brief request or output failed"):
        handle_brief(args, context_provider=provider)
    assert protected.read_text() == "keep"
    assert not args.summary_output.exists()


def test_cli_refusal_has_nonzero_gate_exit(tmp_path):
    args = cli_args(tmp_path, "Synthetic note")
    assert handle_brief(args) == 1
    assert args.summary_output.read_text() == ""
    assert (
        json.loads(args.review_output.read_text())["refusal_reason"]
        == "review_required"
    )


def test_missing_runtime_refusal_is_identical_on_all_surfaces(
    monkeypatch, tmp_path, capsys
):
    from openmed.clinical import summarize_backends
    from openmed.clinical.brief import build_clinical_brief
    from openmed.core.capabilities import MissingOptionalDependencyError
    from openmed.mcp.server import openmed_brief

    value, context = fixture_context()

    def missing_runtime(_):
        raise MissingOptionalDependencyError(
            package="private-package-detail", feature="private-feature", extra="mlx"
        )

    monkeypatch.setattr(
        summarize_backends, "resolve_summarizer_backend", lambda _: missing_runtime
    )
    expected = build_clinical_brief(value, model="mlx", context=context).to_response()
    assert expected["refusal_reason"] == "model_unavailable"
    assert expected["summary"] == ""
    provider = lambda *_: (value, context)
    app = create_app()
    app.state.brief_context_provider = provider
    payload = {"text": value.original_text, "model": "mlx", "review_id": REVIEW_ID}
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.post("/brief", json=payload)
    assert response.status_code == 200
    assert response.json() == expected
    client_app = create_app()
    client_app.state.brief_context_provider = provider
    with OpenMedClient(transport=SyncASGITransport(client_app)) as client:
        assert client.brief(**payload) == expected
    assert (
        openmed_brief(
            **payload,
            runtime_provider=lambda: SimpleNamespace(brief_context_provider=provider),
        )
        == expected
    )
    args = cli_args(tmp_path, value.original_text)
    args.model = "mlx"
    assert handle_brief(args, context_provider=provider) == 1
    assert args.summary_output.read_text() == ""
    assert json.loads(args.review_output.read_text()) == {
        key: value for key, value in expected.items() if key != "summary"
    }
    stdout = capsys.readouterr().out
    assert "model_unavailable" in stdout
    assert "private-package-detail" not in json.dumps(expected) + stdout
