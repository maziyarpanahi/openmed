"""Complete local recipe orchestration, atomicity and CLI integration controls."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openmed.training.synthetic.dataset_recipe import (
    GeneratedArtifacts,
    RecipeGenerator,
    run_dataset_recipe,
)


def node(code="notes", generator="social_history", dependencies=None):
    return {
        "code": code,
        "generator": generator,
        "version": "1.0.0",
        "seed": 7,
        "count": 1,
        "depends_on": dependencies or [],
        "split": "train",
    }


def recipe(nodes=None, max_bytes=10_000_000):
    return {
        "schema_version": 1,
        "nodes": nodes or [node()],
        "max_records": 10,
        "max_bytes": max_bytes,
        "split_manifest_digest": "0" * 64,
    }


def stub(callback=None, validator=None):
    generate = callback or (
        lambda seed, count, dependencies: GeneratedArtifacts(
            count, {"records.json": b"synthetic"}
        )
    )
    return {
        "stub": RecipeGenerator(
            "1.0.0", generate, validator or (lambda output: None), "1" * 64
        )
    }


def test_actual_multimodal_builtin_recipe_reproduces_all_artifact_identities(tmp_path):
    value = recipe([node(), node("images", "burned_in", ["notes"])])
    first = run_dataset_recipe(value, tmp_path / "first")
    second = run_dataset_recipe(value, tmp_path / "second")
    assert first == second
    left = {
        path.relative_to(tmp_path / "first"): path.read_bytes()
        for path in (tmp_path / "first").rglob("*")
        if path.is_file()
    }
    right = {
        path.relative_to(tmp_path / "second"): path.read_bytes()
        for path in (tmp_path / "second").rglob("*")
        if path.is_file()
    }
    assert left == right
    assert any(path.suffix == ".png" for path in left)
    manifest = json.loads((tmp_path / "first" / "derivation.json").read_bytes())
    assert manifest["generator_order"] == ["notes", "images"]
    assert manifest["split_manifest_digest"] == "0" * 64
    assert manifest["runtime"]["python"] == sys.version.split()[0]
    assert manifest["runtime"]["pillow"] != "not_used"


def test_dependency_order_is_enforced_and_hooks_run_only_after_all_success(tmp_path):
    calls = []

    def generate(seed, count, dependencies):
        calls.append(tuple(dependencies))
        return GeneratedArtifacts(count, {"records.json": b"synthetic"})

    def hook(stage, manifest):
        assert calls == [(), ("a",)]
        assert not (stage / "complete.json").exists()
        assert (stage / "a/records.json").exists()
        assert (stage / "b/records.json").exists()
        (stage / "capsule.json").write_bytes(b"local capsule")

    run_dataset_recipe(
        recipe([node("b", "stub", ["a"]), node("a", "stub")]),
        tmp_path / "out",
        registry=stub(generate),
        post_generation_hooks=[hook],
    )
    assert (tmp_path / "out/capsule.json").exists()


@pytest.mark.parametrize(
    "value",
    [
        recipe([node(generator="missing")]),
        recipe([node("a", "stub", ["b"]), node("b", "stub", ["a"])]),
        recipe([node("a", "stub", ["absent"])]),
    ],
)
def test_unknown_generator_or_bad_dependencies_leave_no_destination(tmp_path, value):
    with pytest.raises(ValueError):
        run_dataset_recipe(value, tmp_path / "out", registry=stub())
    assert not (tmp_path / "out").exists()


def test_size_limit_failure_leaves_no_completed_or_partial_dataset(tmp_path):
    with pytest.raises(ValueError):
        run_dataset_recipe(
            recipe([node(generator="stub")], max_bytes=1),
            tmp_path / "out",
            registry=stub(),
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".openmed-synthetic-*"))


def test_interrupted_generator_cleans_owned_stage_and_destination(tmp_path):
    def interrupted(*args):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_dataset_recipe(
            recipe([node(generator="stub")]),
            tmp_path / "out",
            registry=stub(interrupted),
        )
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".openmed-synthetic-*"))


def test_hook_failure_or_manifest_mutation_cannot_complete(tmp_path):
    def changed(stage, manifest):
        manifest["record_count"] = 99

    with pytest.raises(ValueError):
        run_dataset_recipe(
            recipe([node(generator="stub")]),
            tmp_path / "out",
            registry=stub(),
            post_generation_hooks=[changed],
        )
    assert not (tmp_path / "out").exists()


def test_existing_destination_and_files_are_preserved(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("existing")
    with pytest.raises(ValueError):
        run_dataset_recipe(recipe(), output)
    assert (output / "keep.txt").read_text() == "existing"


def test_actual_top_level_cli_generates_dataset_and_reports_only_counts(tmp_path):
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(recipe()))
    cli = Path(sys.executable).with_name("openmed")
    result = subprocess.run(
        [
            str(cli),
            "synthetic",
            "build",
            str(recipe_file),
            str(tmp_path / "out"),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(Path.cwd()),
            "HOME": str(tmp_path),
            "OPENMED_OFFLINE": "1",
        },
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["command"] == "synthetic build"
    assert payload["data"]["complete"] is True
    assert "text" not in payload["data"]
    assert (tmp_path / "out/complete.json").exists()


def test_cli_invalid_recipe_returns_controlled_failure_and_no_output(tmp_path):
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text('{"submitted_value":"DO_NOT_ECHO"}')
    cli = Path(sys.executable).with_name("openmed")
    result = subprocess.run(
        [
            str(cli),
            "synthetic",
            "build",
            str(recipe_file),
            str(tmp_path / "out"),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "DO_NOT_ECHO" not in result.stdout + result.stderr
    assert not (tmp_path / "out").exists()


def test_hook_output_counts_toward_byte_limit(tmp_path):
    def oversized(stage, manifest):
        (stage / "capsule.json").write_bytes(b"x" * 100)

    with pytest.raises(ValueError):
        run_dataset_recipe(
            recipe([node(generator="stub")], max_bytes=20),
            tmp_path / "out",
            registry=stub(),
            post_generation_hooks=[oversized],
        )
    assert not (tmp_path / "out").exists()


def test_node_permutation_preserves_declared_recipe_and_artifacts(tmp_path):
    nodes = [node("b", "stub", ["a"]), node("a", "stub")]
    first = run_dataset_recipe(recipe(nodes), tmp_path / "one", registry=stub())
    second = run_dataset_recipe(recipe(nodes[::-1]), tmp_path / "two", registry=stub())
    assert first == second
    assert (tmp_path / "one/derivation.json").read_bytes() == (
        tmp_path / "two/derivation.json"
    ).read_bytes()


def test_cli_rejects_ambiguous_duplicate_recipe_fields(tmp_path):
    recipe_file = tmp_path / "recipe.json"
    encoded = json.dumps(recipe())
    recipe_file.write_text('{"schema_version": 99, ' + encoded[1:])
    cli = Path(sys.executable).with_name("openmed")
    result = subprocess.run(
        [
            str(cli),
            "synthetic",
            "build",
            str(recipe_file),
            str(tmp_path / "out"),
            "--json",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path.cwd()), "HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert not (tmp_path / "out").exists()
