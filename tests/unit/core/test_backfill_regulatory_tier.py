"""Offline classification, idempotence and preservation of unknown rows."""

import json

from scripts.manifest.backfill_regulatory_tier import backfill, main


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


def test_reviewed_visual_qa_models_and_unreviewed_sibling(tmp_path):
    repos = [
        "OpenMed/Ministral-3B-Medical-v1",
        "OpenMed/Qwen2.5-VL-3B-Medical-v1",
        "OpenMed/Qwen3.5-2B-Medical-v1",
        "OpenMed/other-visual-model",
    ]
    manifest = tmp_path / "models.jsonl"
    rows = [{"repo_id": repo, "family": "General", "task": "unknown"} for repo in repos]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = backfill(manifest, dry_run=False)
    assert report["changed"] == 3
    assert report["counts"] == {"restricted": 3}
    assert report["review"] == [{"repo_id": repos[-1], "reason": "unknown_family_task"}]
    actual = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert actual == [
        *(
            dict(row, capability="visual_qa", regulatory_tier="restricted")
            for row in rows[:3]
        ),
        rows[-1],
    ]
    first = manifest.read_bytes()
    assert backfill(manifest, dry_run=False)["changed"] == 0
    assert manifest.read_bytes() == first
