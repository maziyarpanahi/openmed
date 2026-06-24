"""Conversion workflow credential policy tests."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONVERT_WORKFLOW = ROOT / ".github" / "workflows" / "convert-models.yml"
HF_TOKEN_POLICY = ROOT / "docs" / "security" / "hf-token-policy.md"


def test_convert_workflow_uses_protected_hf_publish_environment():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "publish_to_hub:" in workflow
    assert "  publish-hf:" in workflow
    assert "name: hf-publish" in workflow
    assert "HF_WRITE_TOKEN: ${{ secrets.HF_WRITE_TOKEN }}" in workflow


def test_convert_workflow_fails_publish_job_when_hf_token_is_missing():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "Require HF write token before publish" in workflow
    assert 'if [ -z "${HF_WRITE_TOKEN:-}" ]; then' in workflow
    assert "::error title=Missing HF_WRITE_TOKEN::" in workflow
    assert "exit 1" in workflow
    assert re.search(r"hf_[A-Za-z0-9]{20,}", workflow) is None
    assert 'echo "$HF_WRITE_TOKEN' not in workflow


def test_convert_workflow_publishes_downloaded_conversion_artifacts():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert "actions/download-artifact@v8" in workflow
    assert "name: mlx-model" in workflow
    assert "name: coreml-model" in workflow
    assert "python -m openmed.core.hf_publish" in workflow
    assert "--artifact-dir publish-artifacts/mlx-output" in workflow
    assert "--artifact-dir publish-artifacts/coreml-output" in workflow
    assert '--format "$FORMAT"' in workflow
    assert "--format coreml" in workflow
    assert "published-model-manifest" in workflow


def test_convert_workflow_passes_dispatch_inputs_through_safe_env_vars():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")

    assert workflow.count("MODEL_ID: ${{ github.event.inputs.model_id }}") == 4
    assert workflow.count("QUANTIZE: ${{ github.event.inputs.quantize }}") == 2
    assert workflow.count('--model "$MODEL_ID"') == 4
    assert workflow.count('case "$QUANTIZE" in') == 2
    assert workflow.count('[[ ! "$MODEL_ID" =~') == 4
    assert '--model "${{ github.event.inputs.model_id }}"' not in workflow
    assert '[ "${{ github.event.inputs.quantize }}"' not in workflow
    assert (
        'QUANTIZE_ARG="--quantize ${{ github.event.inputs.quantize }}"' not in workflow
    )


def test_convert_workflow_keeps_dispatch_inputs_out_of_shell_blocks():
    workflow = CONVERT_WORKFLOW.read_text(encoding="utf-8")
    run_blocks = re.findall(r"(?m)^\s+run: \|\n((?:^\s{8,}.*\n?)+)", workflow)

    assert run_blocks
    assert "github.event.inputs" not in "\n".join(run_blocks)


def test_hf_token_policy_documents_scope_storage_rotation_and_revocation():
    policy = HF_TOKEN_POLICY.read_text(encoding="utf-8")

    assert "org-write access" in policy
    assert "`HF_WRITE_TOKEN` secret" in policy
    assert "`hf-publish` GitHub Actions protected environment" in policy
    assert "Rotate the token every 90 days" in policy
    assert "Revoke the token" in policy
    assert "org-wide write access" in policy
