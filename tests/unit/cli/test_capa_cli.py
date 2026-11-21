import json

from openmed.cli.main import main


def call(capsys, directory, action, *args):
    status = main(
        ["compliance", "capa", action, "--directory", str(directory), "--json", *args]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == f"compliance capa {action}"
    return status, payload


NEW = (
    "--id",
    "CAPA-2026-0001",
    "--severity",
    "high",
    "--source",
    "incident",
    "--capability",
    "deidentify",
    "--version",
    "4.3.0",
)


def test_capa_lifecycle_and_corrupted_ledger(tmp_path, capsys):
    assert call(capsys, tmp_path, "new", *NEW)[0] == 0
    status, result = call(
        capsys, tmp_path, "transition", "--id", NEW[1], "--state", "closed"
    )
    assert (
        status == 2
        and result["error"]["code"] == "invalid_corrective_action_transition"
    )
    assert (
        call(
            capsys, tmp_path, "transition", "--id", NEW[1], "--state", "investigating"
        )[0]
        == 0
    )
    assert (
        call(capsys, tmp_path, "list")[1]["data"]["records"][0]["state"]
        == "investigating"
    )
    assert call(capsys, tmp_path, "verify")[1]["data"]["verified"]
    path = tmp_path / "corrective-actions.json"
    ledger = json.loads(path.read_text())
    ledger["records"][0]["severity"] = "low"
    path.write_text(json.dumps(ledger))
    status, result = call(capsys, tmp_path, "verify")
    assert status == 1 and result["error"]["code"] == "invalid_corrective_action_ledger"


def test_checkout_path_refused_before_writes(tmp_path, capsys):
    (tmp_path / ".git").mkdir()
    directory = tmp_path / "nested" / "ledger"
    status, result = call(capsys, directory, "new", *NEW)
    assert status == 2 and result["error"]["code"] == "capa_repository_path"
    assert not directory.exists()
    assert call(capsys, directory, "new", "--allow-repo", *NEW)[0] == 0
