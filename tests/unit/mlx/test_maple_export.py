"""Offline tests for the pinned Maple MLX export plan."""

from __future__ import annotations

import json

import pytest

from openmed.mlx.maple_export import (
    MAPLE_SOURCE_REVISION,
    main,
    plan_maple_mlx_variants,
)


def test_plan_uses_pinned_source_and_separate_mixed_bit_directories(tmp_path):
    plan = plan_maple_mlx_variants(tmp_path / "exports", bits=[8, 4, 8])

    assert plan.source_model == "deepgrove/maple-preview"
    assert plan.source_revision == MAPLE_SOURCE_REVISION
    assert [variant.bits for variant in plan.variants] == [8, 4]
    assert [variant.format for variant in plan.variants] == [
        "mlx-8bit",
        "mlx-4bit",
    ]
    assert plan.variants[0].output_directory.name == "maple-preview-8bit-mlx"


def test_plan_directs_2bit_callers_to_published_artifact(tmp_path):
    with pytest.raises(ValueError, match="published 2-bit artifact"):
        plan_maple_mlx_variants(tmp_path, bits=[2])


@pytest.mark.parametrize("group_size", [0, 16, 256])
def test_plan_rejects_unsupported_group_size(tmp_path, group_size):
    with pytest.raises(ValueError, match="group_size"):
        plan_maple_mlx_variants(tmp_path, group_size=group_size)


def test_dry_run_cli_emits_plan_without_importing_mlx(tmp_path, capsys):
    exit_code = main(
        [
            "--output",
            str(tmp_path / "exports"),
            "--bits",
            "4",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["variants"][0]["format"] == "mlx-4bit"
    assert payload["runtime_model"] == "deepgrove/maple-preview-2bit-mlx"
