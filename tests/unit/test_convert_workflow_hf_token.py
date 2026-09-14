"""GitHub Actions model-publication boundary tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
HF_TOKEN_POLICY = ROOT / "docs" / "security" / "hf-token-policy.md"
RETIRED_MODEL_WORKFLOWS = (
    WORKFLOWS / "convert-models.yml",
    WORKFLOWS / "nightly-release.yml",
)
FORBIDDEN_ACTIONS_MODEL_PUBLISH_MARKERS = (
    "hf-publish",
    "HF_WRITE_TOKEN",
    "openmed.core.hf_publish",
    "openmed.mlx.convert",
    "openmed.coreml.convert",
    "openmed.onnx.convert",
    "scripts/release/dispatch_batch.py",
    "scripts/release/orchestrate.py run",
)


def test_hosted_model_conversion_and_publish_workflows_are_removed() -> None:
    assert all(not path.exists() for path in RETIRED_MODEL_WORKFLOWS)


def test_actions_workflows_cannot_convert_or_publish_model_artifacts() -> None:
    violations: dict[str, list[str]] = {}
    workflow_paths = (*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml"))
    for path in sorted(workflow_paths):
        workflow = path.read_text(encoding="utf-8")
        markers = [
            marker
            for marker in FORBIDDEN_ACTIONS_MODEL_PUBLISH_MARKERS
            if marker in workflow
        ]
        if markers:
            violations[path.name] = markers

    assert violations == {}


def test_model_release_gate_requires_explicit_dispatch() -> None:
    workflow = (WORKFLOWS / "release-gates.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "repository_dispatch:" in workflow
    assert "\n  schedule:" not in workflow
    assert "cron:" not in workflow


def test_hf_token_policy_requires_local_manual_publication() -> None:
    policy = HF_TOKEN_POLICY.read_text(encoding="utf-8")
    compact = " ".join(policy.split())

    assert "must not store" in compact
    assert "GitHub Actions" in compact
    assert "maintainer-controlled machine" in compact
    assert "explicit local command" in compact
    assert "Revoke" in compact
    assert "org-wide write access" in compact
