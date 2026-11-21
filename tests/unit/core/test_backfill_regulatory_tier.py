"""Offline classification, idempotence and preservation of unknown rows."""

import json

from scripts.manifest.backfill_regulatory_tier import main


def test_backfill_dry_run_and_apply(tmp_path, capsys):
    manifest = tmp_path / "models.jsonl"
    pairs = [
        ("PII", "token-classification"),
        ("NER", "token-classification"),
        ("ZeroShot", "token-classification"),
        ("General", "text-generation"),
        ("Vision", "visual-question-answering"),
        ("General", "unknown"),
    ]
    rows = [
        {"repo_id": f"OpenMed/fixture-{i}", "task": task, "family": family}
        for i, (family, task) in enumerate(pairs)
    ]
    rows.append(
        {
            **rows[1],
            "repo_id": "OpenMed/specialty",
            "capability": "mental_health",
            "regulatory_tier": "restricted",
        }
    )
    manifest.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    before = manifest.read_bytes()
    args = ["--manifest", str(manifest)]
    assert main([*args, "--dry-run"]) == 0
    proposed = json.loads(capsys.readouterr().out)
    assert before == manifest.read_bytes()
    assert proposed["counts"] == {
        "assistive_review": 3,
        "non_clinical": 1,
        "restricted": 1,
    }
    assert proposed["review"] == [
        {"repo_id": "OpenMed/fixture-5", "reason": "unknown_family_task"},
        {"repo_id": "OpenMed/specialty", "reason": "existing_declaration_differs"},
    ]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out) == proposed
    output = manifest.read_bytes()
    actual = [json.loads(line) for line in output.splitlines()]
    assert [row["regulatory_tier"] for row in actual[:5]] == [
        "non_clinical",
        "assistive_review",
        "assistive_review",
        "assistive_review",
        "restricted",
    ]
    assert actual[5:] == rows[5:]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["changed"] == 0
    assert manifest.read_bytes() == output
