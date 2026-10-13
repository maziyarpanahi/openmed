# Versioned local synthetic dataset recipes

`openmed synthetic build recipe.json new-output-directory --json` runs an actual
bounded offline recipe. Built-in adapters reuse existing social-history and
burned-in-image generators; images require the optional Pillow dependency. No
models, private datasets, generator services or publication are downloaded or
invoked. CLI output contains only digests/counts/completion status.

A version-1 recipe requires `schema_version`, `nodes`, `max_records`, `max_bytes`,
and a pinned `split_manifest_digest`. Each node requires `code`, `generator`,
`version`, `seed`, `count`, `depends_on` and `split`. Built-in generator identifiers
are `social_history` and `burned_in`, adapter version `1.0.0`. Splits are controlled
train/validation/test/negative_control declarations. Record-level split planning
and proof of group isolation remain separate; this runner preserves the supplied
split commitment and assignments rather than inventing them.

For example, declare a social-history node `notes` with seed 7, count 1 and no
dependencies, then an image node `images` with seed 7, count 1 and dependency
`notes`. Set `max_records` to 10, `max_bytes` to 10000000, and
`split_manifest_digest` to the SHA-256 digest of your locally approved split plan.
Both nodes use `split: "train"`. Supply explicit generator versions and all fields.

The following runnable toy recipe uses a demonstrative synthetic node assignment;
it is not evidence of a reviewed patient-trajectory split:

```python
import hashlib
import json
from pathlib import Path

split = {"synthetic": True, "assignments": {"notes": "train", "images": "train"}}
split_digest = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
nodes = []
for code, generator, dependencies in (
    ("notes", "social_history", []), ("images", "burned_in", ["notes"])
):
    nodes.append({"code": code, "generator": generator, "version": "1.0.0",
                  "seed": 7, "count": 1, "depends_on": dependencies, "split": "train"})
recipe = {"schema_version": 1, "nodes": nodes, "max_records": 10,
          "max_bytes": 10000000, "split_manifest_digest": split_digest}
Path("recipe.json").write_text(json.dumps(recipe))
```

```sh
openmed synthetic build recipe.json synthetic-demo-output --json
```

The output reports `complete: true`, two records and three generated artifacts.
The directory contains social-history JSONL, one PNG, its label JSON, derivation
metadata and a completion marker. Choose a new destination for each run.

The runner rejects unavailable generators, wrong versions, missing dependencies,
cycles and output-limit violations. It reserves a new destination, writes into an
owned sibling stage, validates every generator and runs local validation/capsule
hooks only after successful generation. Hooks cannot silently alter generated
artifacts or derivation metadata. Only then are derivation and completion markers
written and the stage atomically promoted. Failures and handled interruptions
clean the owned stage/reservation; existing destinations are never overwritten.
An unhandled process kill can leave a hidden abandoned stage, never a complete
official destination. Treat absence of `complete.json` as incomplete.

Manifest identities pin seeds, node versions, source digests, split inputs,
artifact digests and declared runtime versions. Reproduction assumes the same
pinned Python/Faker/Pillow/font environment; verify bytes in that environment.
Injected `RecipeGenerator` adapters and post-generation hooks are trusted local
Python callbacks, not a sandbox or remote plugin execution interface. Publication,
scientific fidelity and clinical validation remain owner-reviewed work.
