"""Exercise the public credentialed command using synthetic records."""

import json
import socket

from openmed.cli.main import main
from openmed.eval import credentialed


def _run(output):
    return main(
        [
            "eval",
            "credentialed",
            "--suite",
            "i2b2",
            "--year",
            "2014",
            "--model",
            "example/test",
            "--output",
            str(output),
        ]
    )


def test_offline_aggregate_only(tmp_path, monkeypatch):
    corpus = tmp_path / "input"
    corpus.mkdir()
    for i in range(6):
        (corpus / f"private-{i}.xml").write_text(
            '<ROOT><TEXT>Example Name</TEXT><TAGS><NAME TYPE="PATIENT" '
            'start="0" end="12"/></TAGS></ROOT>'
        )
    monkeypatch.setenv("OPENMED_I2B2_PATH", str(corpus))
    monkeypatch.setattr(socket.socket, "connect", lambda *a: 1 / 0)
    monkeypatch.setattr(
        credentialed,
        "_load_cached_model",
        lambda *args: (
            "a" * 64,
            lambda *args: [{"start": 0, "end": 12, "label": "PERSON"}],
        ),
    )
    before = set(tmp_path.rglob("*"))
    output = tmp_path / "result.json"
    assert _run(output) == 0
    assert set(tmp_path.rglob("*")) - before == {output}
    assert output.stat().st_mode & 0o077 == 0
    raw = output.read_text()
    assert not any(
        v in raw
        for v in ("Example Name", "private-", str(corpus), "fixture_id", "start")
    )
    report = json.loads(raw)
    assert report["by_label"]["PERSON"]["tp"] == 6
    assert report["revision_digest"] == "a" * 64
    assert report["dua"] == "i2b2/DBMI DUA"
    assert report["model_id"] == "example/test"
    assert _run(output) == 2
    assert output.read_text() == raw


def test_worktree_and_symlink_refused_before_model(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(
        credentialed, "_load_cached_model", lambda *args: called.append(args)
    )
    repo = tmp_path / "linked"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: /example")
    alias = tmp_path / "alias"
    alias.symlink_to(repo, target_is_directory=True)
    for target in ("inside-checkout.json", repo / "out", alias / "out"):
        assert _run(target) == 2
    assert called == []


def test_missing_model_does_not_download(tmp_path, monkeypatch, capsys):
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    monkeypatch.setattr(socket.socket, "connect", lambda *a: 1 / 0)
    assert _run(tmp_path / "out") == 2
    assert "model_not_cached" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_inference_error_contains_no_source(tmp_path, monkeypatch, capsys):
    (tmp_path / "private.xml").write_text("<ROOT><TEXT>sample</TEXT><TAGS/></ROOT>")
    monkeypatch.setenv("OPENMED_I2B2_PATH", str(tmp_path))

    def fail(*args):
        raise ValueError("private payload")

    monkeypatch.setattr(
        credentialed, "_load_cached_model", lambda *args: ("a" * 64, fail)
    )
    assert _run(tmp_path / "out") == 2
    diagnostic = capsys.readouterr().err
    assert "credentialed_evaluation_failed" in diagnostic
    assert "private" not in diagnostic
    assert not (tmp_path / "out").exists()


def test_cegs_ngrid_cli_edition_privacy_and_path_refusal(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    from openmed.eval.datasets.cegs_ngrid import CEGS_NGRID_DUA

    source = tmp_path / "synthetic.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"private-record-{i}",
                    "text": "Private Example",
                    "entities": [{"start": 0, "end": 15, "label": "PATIENT"}],
                }
            )
            for i in range(5)
        )
    )
    monkeypatch.setenv("OPENMED_CEGS_NGRID_PATH", str(source))
    seen = []

    def runner(fixture, *args):
        seen.append(True)
        return [{"start": 0, "end": 15, "label": "PERSON"}]

    monkeypatch.setattr(
        credentialed, "_load_cached_model", lambda *args: ("e" * 64, runner)
    )
    out = tmp_path / "out"
    args = [
        "eval",
        "credentialed",
        "--suite",
        "cegs-ngrid-2016",
        "--model",
        "example/test",
        "--output",
        str(out),
    ]
    assert main([*args, "--year", "2014"]) == 2
    assert not seen
    assert main(args) == 0
    assert len(seen) == 5
    raw = out.read_text()
    report = json.loads(raw)
    assert (report["corpus"], report["year"], report["dua"]) == (
        "cegs-n-grid",
        2016,
        CEGS_NGRID_DUA,
    )
    assert report["by_label"]["PERSON"]["tp"] == 5
    assert not any(
        x in raw for x in ("Private Example", "private-record-", str(source))
    )
    monkeypatch.setenv("OPENMED_CEGS_NGRID_PATH", str(Path.cwd()))
    assert main([*args[:-1], str(tmp_path / "refused")]) == 2
    assert len(seen) == 5
    assert not (tmp_path / "refused").exists()
    assert str(Path.cwd()) not in capsys.readouterr().err
