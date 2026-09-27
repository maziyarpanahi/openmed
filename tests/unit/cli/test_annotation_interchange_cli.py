"""CLI tests for annotation interchange and declarative migration."""

from __future__ import annotations

import json
from pathlib import Path

from openmed.cli.main import main

FIXTURE = Path("tests/fixtures/annotation/interchange.tsv")
PIPELINE = Path("tests/fixtures/annotation/pipeline.json")


def test_annotation_import_and_export_round_trip(
    tmp_path: Path, capsys: object
) -> None:
    envelope_path = tmp_path / "annotations.json"
    output_path = tmp_path / "annotations.tsv"

    assert (
        main(
            [
                "annotation",
                "import",
                "--input",
                str(FIXTURE),
                "--output",
                str(envelope_path),
                "--json",
            ]
        )
        == 0
    )
    import_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert import_output["data"]["count"] == 3

    assert (
        main(
            [
                "annotation",
                "export",
                "--input",
                str(envelope_path),
                "--output",
                str(output_path),
                "--json",
            ]
        )
        == 0
    )
    export_output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert export_output["data"]["state"] == "success"
    assert output_path.read_text(encoding="utf-8") == FIXTURE.read_text(
        encoding="utf-8"
    )


def test_annotation_cli_refuses_overwrite(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "exists.json"
    output.write_text("keep", encoding="utf-8")

    code = main(
        [
            "annotation",
            "import",
            "--input",
            str(FIXTURE),
            "--output",
            str(output),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert code == 1
    assert payload["error"]["code"] == "annotation_import_failed"
    assert output.read_text(encoding="utf-8") == "keep"


def test_annotation_cli_never_echoes_sensitive_path_fragments(
    tmp_path: Path, capsys: object
) -> None:
    canary = "synthetic-patient-name-9824"
    output = tmp_path / canary / "annotations.json"
    output.parent.mkdir()
    output.write_text("keep", encoding="utf-8")

    assert (
        main(
            [
                "annotation",
                "import",
                "--input",
                str(FIXTURE),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 1
    )
    assert canary not in capsys.readouterr().out  # type: ignore[attr-defined]

    output.unlink()
    assert (
        main(
            [
                "annotation",
                "import",
                "--input",
                str(FIXTURE),
                "--output",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    assert canary not in capsys.readouterr().out  # type: ignore[attr-defined]


def test_pipeline_scan_writes_report_and_safe_stub(
    tmp_path: Path, capsys: object
) -> None:
    report = tmp_path / "report.json"
    stub = tmp_path / "openmed-pipeline.json"

    code = main(
        [
            "annotation",
            "scan-pipeline",
            "--input",
            str(PIPELINE),
            "--report",
            str(report),
            "--stub",
            str(stub),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    native = json.loads(stub.read_text(encoding="utf-8"))

    assert code == 0
    assert payload["data"]["state"] == "partial"
    assert payload["data"]["automatic"] is False
    assert [item["stage"] for item in native["stages"]] == [
        "source_adaptation",
        "extraction",
        "validation",
    ]
    assert json.loads(report.read_text(encoding="utf-8"))["state"] == "partial"


def test_pipeline_scan_preflights_all_outputs_before_writing(
    tmp_path: Path, capsys: object
) -> None:
    report = tmp_path / "report.json"
    stub = tmp_path / "existing.json"
    stub.write_text("keep", encoding="utf-8")

    code = main(
        [
            "annotation",
            "scan-pipeline",
            "--input",
            str(PIPELINE),
            "--report",
            str(report),
            "--stub",
            str(stub),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    assert code == 1
    assert payload["error"]["code"] == "pipeline_scan_failed"
    assert not report.exists()
    assert stub.read_text(encoding="utf-8") == "keep"
