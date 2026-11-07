"""Offline build_model_card CLI runs with a synthetic privacy disclosure."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from openmed.core.repro_hash import build_training_provenance, write_training_provenance
from openmed.eval.model_card_privacy import (
    NO_FORMAL_DP_GUARANTEE_STATEMENT,
    privacy_accountant_statement_digest,
)
from tests.fixtures.eval.model_card_privacy import (
    REPO_ID,
    SITE_SENTINEL,
    dp_federated_disclosure,
    federated_without_dp_disclosure,
    gate_report,
    manifest_row,
    promotion_evidence,
)

pytestmark = pytest.mark.integration

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/release/build_model_card.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("build_model_card_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _inputs(tmp_path: Path) -> list[str]:
    manifest = tmp_path / "models.jsonl"
    manifest.write_text(json.dumps(manifest_row(), sort_keys=True) + "\n")
    gate = _write_json(tmp_path / "gate-report.json", gate_report().to_dict())
    provenance = write_training_provenance(
        tmp_path,
        build_training_provenance(
            rng_seeds={"python": 123, "numpy": 456},
            data_manifest_hash="sha256:" + "2" * 64,
            recipe_config_hash="sha256:" + "3" * 64,
            env_lock_digest="sha256:" + "4" * 64,
            base_model="OpenMed/OpenMed-PII-Tiny-44M",
            base_model_revision="main",
            git_sha="abc123",
            repo_id=REPO_ID,
        ),
    )
    return [
        "--manifest",
        str(manifest),
        "--repo-id",
        REPO_ID,
        "--gate-report",
        str(gate),
        "--training-provenance",
        str(provenance),
    ]


def _run(cli, args: list[str], capsys) -> tuple[int, str, str]:
    code = cli.main(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_renders_dp_disclosure_and_checks_byte_stability(
    tmp_path: Path,
    capsys,
) -> None:
    cli = _load_cli()
    disclosure = dp_federated_disclosure()
    disclosure_path = _write_json(tmp_path / "privacy.json", disclosure)
    evidence_path = _write_json(
        tmp_path / "privacy-evidence.json",
        promotion_evidence(privacy_accountant_statement_digest(disclosure)),
    )
    output = tmp_path / "card" / "README.md"
    args = [
        *_inputs(tmp_path),
        "--privacy-disclosure",
        str(disclosure_path),
        "--privacy-evidence",
        str(evidence_path),
    ]

    code, _, err = _run(cli, [*args, "--output", str(output)], capsys)
    assert code == 0, err
    markdown = output.read_text(encoding="utf-8")
    datasheet = json.loads((output.parent / "model-datasheet.json").read_text())
    assert "## Privacy Disclosure" in markdown
    assert datasheet["privacy_disclosure"]["epsilon"] == 3.0

    code, out, err = _run(cli, [*args, "--check", str(output)], capsys)
    assert (code, out, err) == (0, "", "")


def test_cli_renders_federated_without_dp_statement(tmp_path: Path, capsys) -> None:
    cli = _load_cli()
    disclosure_path = _write_json(
        tmp_path / "privacy.json", federated_without_dp_disclosure()
    )
    code, out, err = _run(
        cli,
        [*_inputs(tmp_path), "--privacy-disclosure", str(disclosure_path)],
        capsys,
    )
    assert code == 0, err
    assert f"> {NO_FORMAL_DP_GUARANTEE_STATEMENT}" in out


def test_cli_blocks_dp_claim_without_evidence_and_does_not_echo_inputs(
    tmp_path: Path,
    capsys,
) -> None:
    cli = _load_cli()
    disclosure = dp_federated_disclosure()
    disclosure["site_label"] = SITE_SENTINEL
    disclosure_path = _write_json(tmp_path / "privacy.json", disclosure)
    output = tmp_path / "README.md"

    code, out, err = _run(
        cli,
        [
            *_inputs(tmp_path),
            "--privacy-disclosure",
            str(disclosure_path),
            "--output",
            str(output),
        ],
        capsys,
    )

    assert code == 1
    assert out == ""
    assert "unknown_field" in err
    assert SITE_SENTINEL not in err
    assert str(tmp_path) not in err
    assert not output.exists()

    del disclosure["site_label"]
    _write_json(disclosure_path, disclosure)
    code, _, err = _run(
        cli,
        [*_inputs(tmp_path), "--privacy-disclosure", str(disclosure_path)],
        capsys,
    )
    assert code == 1
    assert "accountant_evidence_missing" in err


def test_cli_reports_unreadable_disclosure_without_path(
    tmp_path: Path,
    capsys,
) -> None:
    cli = _load_cli()
    missing = tmp_path / "private-dir" / "privacy.json"
    code, _, err = _run(
        cli,
        [*_inputs(tmp_path), "--privacy-disclosure", str(missing)],
        capsys,
    )
    assert code == 1
    assert "privacy_disclosure_unreadable" in err
    assert "private-dir" not in err
