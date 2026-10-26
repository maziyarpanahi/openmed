"""Bounded local synthetic recipe execution with transactional completion."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FILE = re.compile(r"[a-z][a-z0-9_]{0,63}\.(?:json|jsonl|png|txt)\Z")


@dataclass(frozen=True)
class GeneratedArtifacts:
    """Synthetic generator outputs and exact record accounting."""

    record_count: int
    artifacts: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if (
            isinstance(self.record_count, bool)
            or not isinstance(self.record_count, int)
            or not 0 <= self.record_count <= 10_000
        ):
            raise ValueError("generated record count must be a bounded integer")
        if not isinstance(self.artifacts, Mapping) or len(self.artifacts) > 256:
            raise ValueError("generated artifacts must be a bounded mapping")
        if any(
            not isinstance(name, str)
            or not _FILE.fullmatch(name)
            or not isinstance(payload, bytes)
            or len(payload) > 100_000_000
            for name, payload in self.artifacts.items()
        ):
            raise ValueError(
                "generated artifacts require controlled filenames and bounded bytes"
            )
        object.__setattr__(self, "artifacts", MappingProxyType(dict(self.artifacts)))


@dataclass(frozen=True)
class RecipeGenerator:
    """Explicit versioned, trusted local generator and its output validator."""

    version: str
    generate: Callable[[int, int, Mapping[str, GeneratedArtifacts]], GeneratedArtifacts]
    validate: Callable[[GeneratedArtifacts], None]
    provenance_digest: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.version, str)
            or not re.fullmatch(r"[0-9]{1,5}\.[0-9]{1,5}\.[0-9]{1,5}", self.version)
            or not callable(self.generate)
            or not callable(self.validate)
            or not isinstance(self.provenance_digest, str)
            or not _DIGEST.fullmatch(self.provenance_digest)
        ):
            raise ValueError(
                "recipe adapter requires a version, local callbacks and provenance digest"
            )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _recipe(recipe: Mapping[str, object], registry: Mapping[str, RecipeGenerator]):
    if (
        not isinstance(recipe, Mapping)
        or set(recipe)
        != {
            "schema_version",
            "nodes",
            "max_records",
            "max_bytes",
            "split_manifest_digest",
        }
        or recipe["schema_version"] != 1
        or isinstance(recipe["schema_version"], bool)
    ):
        raise ValueError("invalid versioned dataset recipe")
    for field, maximum in (("max_records", 10_000), ("max_bytes", 100_000_000)):
        value = recipe[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError("dataset recipe output limit is invalid")
    split_digest = recipe["split_manifest_digest"]
    if not isinstance(split_digest, str) or not _DIGEST.fullmatch(split_digest):
        raise ValueError("recipe requires a pinned split manifest digest")
    nodes = recipe["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 64:
        raise ValueError("recipe requires 1 to 64 generator nodes")
    indexed = {}
    total = 0
    for node in nodes:
        if not isinstance(node, dict) or set(node) != {
            "code",
            "generator",
            "version",
            "seed",
            "count",
            "depends_on",
            "split",
        }:
            raise ValueError("invalid recipe generator declaration")
        for field in ("code", "generator"):
            if not isinstance(node[field], str) or not _CODE.fullmatch(node[field]):
                raise ValueError("invalid controlled recipe generator code")
        if node["code"] in indexed:
            raise ValueError("duplicate recipe node")
        if node["generator"] not in registry or not isinstance(
            registry[node["generator"]], RecipeGenerator
        ):
            raise ValueError("recipe generator is unavailable")
        adapter = registry[node["generator"]]
        if (
            node["version"] != adapter.version
            or not isinstance(adapter.provenance_digest, str)
            or not _DIGEST.fullmatch(adapter.provenance_digest)
        ):
            raise ValueError("recipe generator version or provenance mismatch")
        for field, maximum in (("seed", 2**53), ("count", recipe["max_records"])):
            if (
                isinstance(node[field], bool)
                or not isinstance(node[field], int)
                or not 0 <= node[field] <= maximum
            ):
                raise ValueError("invalid generator seed or record count")
        if node["split"] not in {"train", "validation", "test", "negative_control"}:
            raise ValueError("invalid declared dataset split")
        dependencies = node["depends_on"]
        if (
            not isinstance(dependencies, list)
            or len(dependencies) > 64
            or not all(
                isinstance(code, str) and _CODE.fullmatch(code) for code in dependencies
            )
            or len(set(dependencies)) != len(dependencies)
        ):
            raise ValueError("invalid recipe dependency list")
        indexed[node["code"]] = {**node, "depends_on": sorted(dependencies)}
        total += node["count"]
    if total > recipe["max_records"]:
        raise ValueError("recipe exceeds aggregate record limit")
    complete = set()
    order = []
    while len(complete) != len(indexed):
        ready = sorted(
            code
            for code, node in indexed.items()
            if code not in complete and set(node["depends_on"]) <= complete
        )
        if not ready:
            raise ValueError("recipe has missing dependencies or a cycle")
        order.extend(ready)
        complete.update(ready)
    return indexed, order


def run_dataset_recipe(
    recipe: Mapping[str, object],
    output_directory: str | Path,
    *,
    registry: Mapping[str, RecipeGenerator] | None = None,
    post_generation_hooks: Sequence[Callable[[Path, Mapping[str, object]], None]] = (),
) -> dict[str, object]:
    """Execute declared generators, validate, stage and atomically mark completion.

    Hooks run only after every generator succeeds and before a completion marker
    exists. They may validate or package the staged outputs. A failed hook or
    interrupted generator removes owned staging and the empty destination
    reservation; no apparently complete dataset remains at the destination.
    """
    recipe = dict(recipe) if isinstance(recipe, Mapping) else recipe
    registry = builtin_recipe_generators() if registry is None else dict(registry)
    indexed, order = _recipe(recipe, registry)
    if len(post_generation_hooks) > 16 or not all(
        callable(hook) for hook in post_generation_hooks
    ):
        raise ValueError("provide bounded local post-generation hooks")
    destination = Path(output_directory)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ValueError("dataset destination parent is unavailable") from None
    try:
        destination.mkdir()
    except OSError:
        raise ValueError("dataset destination must not exist") from None
    try:
        stage = Path(
            tempfile.mkdtemp(prefix=".openmed-synthetic-", dir=destination.parent)
        )
    except OSError:
        destination.rmdir()
        raise ValueError("synthetic dataset staging is unavailable") from None
    results = {}
    artifact_rows = []
    total_bytes = 0
    try:
        for code in order:
            node = indexed[code]
            adapter = registry[node["generator"]]
            result = adapter.generate(
                node["seed"],
                node["count"],
                {key: results[key] for key in node["depends_on"]},
            )
            if (
                not isinstance(result, GeneratedArtifacts)
                or isinstance(result.record_count, bool)
                or result.record_count != node["count"]
                or not isinstance(result.artifacts, Mapping)
                or len(result.artifacts) > 256
            ):
                raise ValueError("generator output accounting is invalid")
            adapter.validate(result)
            node_directory = stage / code
            node_directory.mkdir()
            for filename, payload in sorted(result.artifacts.items()):
                if (
                    not isinstance(filename, str)
                    or not _FILE.fullmatch(filename)
                    or not isinstance(payload, bytes)
                ):
                    raise ValueError(
                        "generator output must use controlled filenames and bytes"
                    )
                total_bytes += len(payload)
                if total_bytes > recipe["max_bytes"]:
                    raise ValueError("recipe exceeds aggregate byte limit")
                (node_directory / filename).write_bytes(payload)
                artifact_rows.append(
                    {
                        "path": f"{code}/{filename}",
                        "byte_size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "split": node["split"],
                    }
                )
            results[code] = result
        normalized = dict(recipe)
        normalized["nodes"] = [indexed[code] for code in sorted(indexed)]
        recipe_digest = hashlib.sha256(_canonical(normalized)).hexdigest()
        manifest = {
            "schema_version": 1,
            "synthetic": True,
            "recipe_digest": recipe_digest,
            "split_manifest_digest": recipe["split_manifest_digest"],
            "generator_order": order,
            "nodes": [
                {
                    **indexed[code],
                    "provenance_digest": registry[
                        indexed[code]["generator"]
                    ].provenance_digest,
                }
                for code in order
            ],
            "artifacts": artifact_rows,
            "record_count": sum(result.record_count for result in results.values()),
            "runtime": {
                "python": platform.python_version(),
                "openmed": importlib.metadata.version("openmed"),
                "faker": importlib.metadata.version("faker"),
                "pillow": importlib.metadata.version("pillow")
                if any(node["generator"] == "burned_in" for node in indexed.values())
                else "not_used",
            },
        }
        before_hooks = _canonical(manifest)
        for hook in post_generation_hooks:
            hook(stage, manifest)
        if _canonical(manifest) != before_hooks:
            raise ValueError("post-generation hook changed derivation metadata")
        all_paths = list(stage.rglob("*"))
        if any(path.is_symlink() for path in all_paths):
            raise ValueError("post-generation output cannot include symbolic links")
        files = [path for path in all_paths if path.is_file()]
        if (
            len(files) > 2048
            or sum(path.stat().st_size for path in files) > recipe["max_bytes"]
        ):
            raise ValueError("post-generation outputs exceed declared limits")
        generated_paths = {row["path"] for row in artifact_rows}
        packaged = []
        for path in sorted(files):
            relative = path.relative_to(stage)
            if relative.as_posix() in generated_paths:
                continue
            if not _FILE.fullmatch(relative.name) or any(
                not _CODE.fullmatch(part) for part in relative.parts[:-1]
            ):
                raise ValueError("packaged artifact path must use controlled names")
            packaged.append(
                {
                    "path": relative.as_posix(),
                    "byte_size": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        manifest["post_generation_artifacts"] = packaged
        # Hooks may add capsules, but may not silently alter committed generation.
        for row in artifact_rows:
            if (
                hashlib.sha256((stage / row["path"]).read_bytes()).hexdigest()
                != row["sha256"]
            ):
                raise ValueError("post-generation hook changed a generated artifact")
        encoded = _canonical(manifest)
        (stage / "derivation.json").write_bytes(encoded)
        (stage / "complete.json").write_bytes(
            _canonical({"manifest_sha256": hashlib.sha256(encoded).hexdigest()})
        )
        try:
            os.replace(stage, destination)
        except OSError:
            # Windows cannot rename a directory onto an existing one, even the
            # empty reservation; release it (rmdir fails unless it is empty).
            destination.rmdir()
            os.replace(stage, destination)
        return {
            "recipe_digest": recipe_digest,
            "artifact_count": len(artifact_rows),
            "record_count": manifest["record_count"],
            "complete": True,
        }
    except BaseException as exc:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            destination.rmdir()
        except OSError:
            pass
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ValueError("synthetic recipe generation or validation failed") from None


def builtin_recipe_generators() -> dict[str, RecipeGenerator]:
    """Return existing local social-history and burned-in-image generator adapters."""

    def social(seed, count, dependencies):
        from openmed.training.synthetic.social_history import (
            generate_social_history_examples,
        )

        records = generate_social_history_examples(count, seed=seed)
        payload = b"".join(_canonical(record.to_dict()) + b"\n" for record in records)
        return GeneratedArtifacts(count, {"records.jsonl": payload})

    def images(seed, count, dependencies):
        from openmed.training.synthetic.burned_in import BurnedInGenerator

        if count > 32:
            raise ValueError("burned-in recipe count exceeds bounded rendering limit")
        artifacts = {}
        for index in range(count):
            example = BurnedInGenerator(seed=seed + index).generate(
                "en", background="solid"
            )
            buffer = io.BytesIO()
            example.image.save(buffer, format="PNG")
            item = example.to_training_item()
            item.pop("image")
            artifacts[f"image_{index:04d}.png"] = buffer.getvalue()
            artifacts[f"labels_{index:04d}.json"] = _canonical(item)
        return GeneratedArtifacts(count, artifacts)

    def validated(result):
        for name, payload in result.artifacts.items():
            if name.endswith(".json"):
                item = json.loads(payload)
                if item.get("is_synthetic") is not True:
                    raise ValueError("generated image labels lack synthetic markers")
            elif name.endswith(".jsonl"):
                records = [json.loads(line) for line in payload.splitlines()]
                if len(records) != result.record_count or any(
                    record.get("metadata", {}).get("synthetic") is not True
                    for record in records
                ):
                    raise ValueError("generated records lack synthetic markers")

    adapters = {}
    adapter_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for code, generate in (("social_history", social), ("burned_in", images)):
        spec = importlib.util.find_spec(f"openmed.training.synthetic.{code}")
        generator_digest = hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest()
        provenance = hashlib.sha256(
            _canonical(
                {
                    "adapter_source_digest": adapter_digest,
                    "generator_source_digest": generator_digest,
                }
            )
        ).hexdigest()
        adapters[code] = RecipeGenerator("1.0.0", generate, validated, provenance)
    return adapters
