#!/usr/bin/env python3
"""Train a pinned encoder locally; never publish or enable a default alias.

Optional training dependencies are deliberately not part of the core package:
torch, transformers, safetensors, pyarrow, and huggingface_hub. The download
phase fetches only public licensed literature/general NLI and model files.
No patient-note dataset, hosted inference, paid job, or visibility API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import urllib.request
from collections import Counter
from pathlib import Path

from openmed.training.clinical_nli import (
    BASE_MODEL,
    BASE_REVISION,
    BIONLI_FILES,
    BIONLI_LICENSE_SOURCE,
    LABELS,
    MULTINLI_REVISION,
    PARTIAL_LABEL,
    RECIPE_PATH,
    NLIPair,
    corpus_fingerprint,
    digest,
    grouped_split,
    load_bionli_csv,
    load_training_recipe,
    pair_payload,
    remove_conflicting_public_groups,
    synthetic_clinical_pairs,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _download(root: Path) -> tuple[Path, dict[str, Path]]:
    from huggingface_hub import snapshot_download

    base = root / "base"
    public = root / "public"
    public.mkdir(exist_ok=True)
    snapshot_download(
        BASE_MODEL,
        revision=BASE_REVISION,
        local_dir=base,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer*",
            "vocab.txt",
            "special_tokens_map.json",
            "README.md",
        ],
    )
    snapshot_download(
        "nyu-mll/multi_nli",
        repo_type="dataset",
        revision=MULTINLI_REVISION,
        local_dir=public / "multi_nli",
        allow_patterns=["README.md", "data/train-*.parquet"],
    )
    paths = {}
    for name, file_id in BIONLI_FILES.items():
        path = public / f"bionli-{name}.csv"
        if not path.is_file():
            url = (
                "https://drive.usercontent.google.com/download"
                f"?id={file_id}&export=download&confirm=t"
            )
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                path.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        # Validate schema before considering a download complete.
        load_bionli_csv(path)
        paths[name] = path
    return base, paths


def _public_mnli(path: Path) -> list[NLIPair]:
    import pyarrow.parquet as pq

    result = []
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    for batch in pq.ParquetFile(path).iter_batches(
        columns=["premise", "hypothesis", "label", "genre"], batch_size=4096
    ):
        for row in batch.to_pylist():
            # Fiction contains heterogeneous share-alike/regional licenses.
            # The remaining training genres use the permissive OANC license.
            if row["genre"] == "fiction" or row["label"] not in mapping:
                continue
            result.append(
                NLIPair(
                    row["premise"],
                    row["hypothesis"],
                    mapping[row["label"]],
                    "mnli:" + digest(" ".join(row["premise"].split()))[7:],
                )
            )
    # Same prompt can occur more than once. Conflicting annotation votes are
    # removed, not selected according to a desired held-out outcome.
    unique: dict[str, NLIPair] = {}
    conflicts = set()
    for pair in result:
        old = unique.get(pair.pair_digest)
        if old is not None and old.label != pair.label:
            conflicts.add(pair.pair_digest)
        unique[pair.pair_digest] = pair
    return [pair for key, pair in unique.items() if key not in conflicts]


def _balanced(pairs: list[NLIPair], limit: int, rng: random.Random) -> list[NLIPair]:
    groups: dict[str, list[NLIPair]] = {}
    for pair in pairs:
        groups.setdefault(pair.label, []).append(pair)
    selected = []
    for label in sorted(groups):
        rows = groups[label]
        rng.shuffle(rows)
        selected.extend(rows[: limit // len(groups)])
    rng.shuffle(selected)
    return selected


def prepare(root: Path, *, seed: int) -> tuple[Path, dict[str, list[NLIPair]]]:
    """Prepare source-disjoint local partitions and aggregate provenance."""

    base, public_files = _download(root)
    bionli = load_bionli_csv(public_files["train"])
    development = load_bionli_csv(public_files["development"])
    # Preserve the author's development publications exclusively for a final
    # biomedical evaluation; neither calibration nor training sees them.
    reserved_groups = {pair.source_group for pair in development}
    reserved_pairs = {pair.pair_digest for pair in development}
    bionli = [
        pair
        for pair in bionli
        if pair.source_group not in reserved_groups
        and pair.pair_digest not in reserved_pairs
    ]
    bionli, conflict_counts = remove_conflicting_public_groups(bionli)
    bionli_splits = grouped_split(bionli, seed=seed)
    mnli_splits = grouped_split(
        _public_mnli(root / "public/multi_nli/data/train-00000-of-00001.parquet"),
        seed=seed,
    )
    synthetic_splits = grouped_split(synthetic_clinical_pairs(), seed=seed)
    rng = random.Random(seed)
    splits = {
        "train": (
            _balanced(mnli_splits["train"], 24_000, rng)
            + _balanced(bionli_splits["train"], 12_000, rng)
            + synthetic_splits["train"]
        ),
        "validation": (
            _balanced(mnli_splits["validation"], 1200, rng)
            + _balanced(bionli_splits["validation"], 1200, rng)
            + synthetic_splits["validation"]
        ),
        "public_test": _balanced(mnli_splits["test"], 1500, rng),
        "biomedical_test": _balanced(development, 2000, rng),
        "synthetic_test": synthetic_splits["test"],
    }
    # Check both source groups and exact normalized pairs across partitions.
    group_owner: dict[str, str] = {}
    pair_owner: dict[str, str] = {}
    for name, pairs in splits.items():
        for pair in pairs:
            for mapping, key in (
                (group_owner, pair.source_group),
                (pair_owner, pair.pair_digest),
            ):
                if key in mapping and mapping[key] != name:
                    raise ValueError("public training/evaluation overlap")
                mapping[key] = name
    manifest = {
        "schema": "openmed.training.clinical_nli.corpus.v1",
        "seed": seed,
        "base": {
            "model_id": BASE_MODEL,
            "revision": BASE_REVISION,
            "license": "apache-2.0",
        },
        "sources": [
            {
                "id": "nyu-mll/multi_nli",
                "revision": MULTINLI_REVISION,
                "license": "OANC permissive license; fiction excluded",
                "source": f"https://huggingface.co/datasets/nyu-mll/multi_nli/blob/{MULTINLI_REVISION}/README.md",
            },
            {
                "id": "BioNLI",
                "license": "cc-by-4.0",
                "source": BIONLI_LICENSE_SOURCE,
                "authors": "Bastan, Surdeanu, Balasubramanian (2022)",
                "file_sha256": {
                    name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name, path in public_files.items()
                },
                "label_protocol": "pos=entailment; all perturbations=not_entailment (partial label)",
            },
            {
                "id": "authored-clinical-templates-v1",
                "license": "apache-2.0",
                "synthetic": True,
            },
        ],
        "restricted_data": "none; MedNLI is eval-only and was not downloaded",
        "patient_data": False,
        "biomedical_conflicting_duplicates": conflict_counts,
        "partitions": {
            name: {
                "count": len(pairs),
                "fingerprint": corpus_fingerprint(pairs),
                "source_groups": len({pair.source_group for pair in pairs}),
                "labels": dict(Counter(pair.label for pair in pairs)),
            }
            for name, pairs in splits.items()
        },
    }
    _write(root / "corpus-manifest.json", manifest)
    for name, pairs in splits.items():
        _write(root / f"{name}.json", [pair_payload(pair) for pair in pairs])
    return base, splits


def train(
    root: Path, *, seed: int, epochs: int, max_steps: int | None, recipe: dict
) -> None:
    """Train and select by validation only, saving an unpublished candidate."""

    import numpy as np
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    base, pairs = prepare(root, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    config = AutoConfig.from_pretrained(base, local_files_only=True)
    config.id2label = dict(enumerate(LABELS))
    config.label2id = {value: key for key, value in config.id2label.items()}
    config.num_labels = 3
    model = AutoModelForSequenceClassification.from_pretrained(
        base,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True,
        attn_implementation="eager",
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    settings = recipe["training"]
    batch_size = settings["batch_size"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    encoded = {}
    skipped = {}
    for name in ("train", "validation"):
        records = []
        for pair in pairs[name]:
            tokens = tokenizer(pair.premise, pair.hypothesis, truncation=False)
            if len(tokens["input_ids"]) > settings["sequence_length"]:
                continue
            records.append(
                (tokens, 3 if pair.label == PARTIAL_LABEL else LABELS.index(pair.label))
            )
        encoded[name] = records
        skipped[name] = len(pairs[name]) - len(records)
    rng = random.Random(seed)
    history = []
    best = -1.0
    step = 0
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        rows = list(encoded["train"])
        rng.shuffle(rows)
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset : offset + batch_size]
            tokens = tokenizer.pad([row[0] for row in batch], return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            labels = torch.tensor([row[1] for row in batch], device=device)
            logits = model(**tokens).logits
            logp = torch.log_softmax(logits, dim=-1)
            # A binary negative constrains the sum of contradiction + neutral,
            # without introducing incorrect three-way ground-truth labels.
            loss = torch.where(
                labels == 3,
                -torch.logsumexp(logp[:, :2], dim=-1),
                -logp.gather(1, labels.clamp_max(2)[:, None]).squeeze(1),
            ).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "epoch": epoch + 1,
                            "loss": float(loss.detach().cpu()),
                            "seconds": round(time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
            if max_steps is not None and step >= max_steps:
                break
        model.eval()
        correct = count = 0
        with torch.inference_mode():
            for offset in range(0, len(encoded["validation"]), 32):
                batch = encoded["validation"][offset : offset + 32]
                tokens = tokenizer.pad([row[0] for row in batch], return_tensors="pt")
                output = (
                    model(**{key: value.to(device) for key, value in tokens.items()})
                    .logits.argmax(-1)
                    .cpu()
                    .tolist()
                )
                for pred, (_, label) in zip(output, batch, strict=True):
                    correct += int(pred != 2 if label == 3 else pred == label)
                    count += 1
        accuracy = correct / count
        history.append(
            {"epoch": epoch + 1, "accuracy": accuracy, "count": count, "steps": step}
        )
        print(json.dumps({"validation": history[-1]}), flush=True)
        if accuracy > best:
            best = accuracy
            output = root / "candidate"
            model.save_pretrained(output)
            tokenizer.save_pretrained(output)
        _write(
            root / "training-run.json",
            {
                "seed": seed,
                "device": device,
                "epochs_requested": epochs,
                "optimizer": "AdamW",
                "learning_rate": settings["learning_rate"],
                "batch_size": batch_size,
                "max_length": settings["sequence_length"],
                "recipe_fingerprint": digest(recipe),
                "overlength_policy": "drop_without_truncation",
                "skipped_overlength": skipped,
                "validation_history": history,
                "elapsed_seconds": round(time.monotonic() - started),
                "paid_compute": False,
                "published": False,
                "reproducibility": "seeded; bitwise MPS determinism is not claimed",
            },
        )
        if max_steps is not None and step >= max_steps:
            break


def main() -> None:
    """Run a local training command with explicit output and bounded epochs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=RECIPE_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    recipe = load_training_recipe(args.config)
    args.seed = args.seed if args.seed is not None else recipe["seed"]
    args.epochs = (
        args.epochs if args.epochs is not None else recipe["training"]["epochs"]
    )
    if (
        args.epochs < 1
        or args.epochs > 10
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        parser.error("epochs must be 1..10 and max-steps must be positive")
    if args.output.exists() and (args.output / "training-run.json").exists():
        parser.error(
            "use a new output directory; existing training evidence is preserved"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    train(
        args.output,
        seed=args.seed,
        epochs=args.epochs,
        max_steps=args.max_steps,
        recipe=recipe,
    )


if __name__ == "__main__":
    main()
