# Portable Task Packs

A task pack lets an app choose a job instead of a model repository. OpenMed
declares three packs in one manifest that Python, the Swift and Kotlin
OpenMedKit ports and the web kit all read, so every surface picks the same
default models and promises the same labels.

| Pack id | Task | Canonical labels | Notice key |
|---|---|---|---|
| `pii` | `pii_detection` | 12 direct identifiers (`ACCOUNT_NUMBER`, `DATE_OF_BIRTH`, `EMAIL`, `FIRST_NAME`, `ID_NUM`, `IP_ADDRESS`, `LAST_NAME`, `PERSON`, `PHONE`, `SSN`, `STREET_ADDRESS`, `URL`) | `openmed.notice.pii.review_required` |
| `clinical_entity` | `clinical_entity_extraction` | `CONDITION`, `DISEASE`, `PATHOLOGY` | `openmed.notice.clinical_entity.extraction_only` |
| `medication` | `medication_mention_extraction` | `CHEMICAL` | `openmed.notice.medication.extraction_only` |

Canonical labels are a floor: every candidate emits at least these labels and
may emit more. The medication pack is extraction-only. It reports medication
mentions and their offsets, never dosing advice, and it never triggers an
action. Kits must show the pack's notice with its output.

## Files

| File | Role |
|---|---|
| `openmed/core/task_packs.py` | Pack declarations (the source of truth) and the Python resolver |
| `openmed/core/schemas/json/task_pack.schema.json` | JSON Schema for the manifest; `#/$defs/task_pack` validates one pack |
| `openmed/core/data/task_packs.json` | Generated manifest that kits consume |
| `scripts/manifest/build_task_packs.py` | Offline generator with `--check` |

## Runtimes and formats

Each pack lists candidates for three runtimes. A runtime loads exactly one
model format:

| Runtime | Format | Kit |
|---|---|---|
| `apple` | `mlx-fp` | OpenMedKit for Swift |
| `android` | `onnx` | OpenMedKit for Kotlin |
| `web` | `onnx` | OpenMedKit for the web |

When no manifest row satisfies a pack's rules for a runtime, the manifest
records `"status": "no_candidate"` with an empty candidate list instead of
dropping the runtime.

## Selection rules

The generator keeps a `models.jsonl` row for a pack and runtime only when all
of these hold:

- `family` equals the pack's `family`, and `task` equals `manifest_task`.
- When `family_token` is set, one `-` or `_` separated part of the repository
  name equals it, ignoring case (for example `PharmaDetect`).
- `formats` contains the runtime's format.
- `languages` contains every pack language.
- Label coverage is `all`: the row's `canonical_labels`, normalized with
  `openmed.core.labels.normalize_label`, include every pack canonical label.
  `CHEM` therefore counts as `CHEMICAL`.
- `param_count` is known and at most `max_param_count`. A row without a
  parameter count fails the size cap.
- `license` is in the pack's permissive `licenses` list. Rows with `other`,
  a copyleft license or no license are never selected.

Matching rows are ordered by ascending `param_count`, then `repo_id`, and the
first `max_candidates` are kept with ranks starting at 1. Ports use the
candidates in rank order and fall back to the next one when a candidate is not
available on the device.

## Python resolver

```python
from openmed.core.task_packs import resolve

resolution = resolve("medication", "apple")
resolution.status        # "resolved" or "no_candidate"
resolution.repo_ids      # ordered repository ids, best first
resolution.notice_key    # "openmed.notice.medication.extraction_only"
```

`resolve` reads the bundled manifest only. It never downloads, verifies or
loads a model and makes no network call. Unknown packs and runtimes raise
`TaskPackError` with the value-free reason `unknown_pack` or
`unknown_runtime`. `load_task_packs(path)` validates a manifest against the
schema and the cross-field rules (sequential ranks, size cap, permissive
licenses, known notice keys, canonical labels) and fails closed with a reason
code such as `schema_invalid` or `rank_not_sequential`.

## Web package

The `openmed` web package exports `piiPack`. `piiPack.resolve()` returns the
pack's `web` candidates in the order `resolve("pii", "web")` returns them, with
the same fields plus each candidate's catalog `reproducibility_hash` and the
notice text. It reads bundled data only. The candidates come from this
manifest, joined at build time with the web ONNX model catalog
(`openmed/catalog`). A candidate that is not a catalog row with the same
license and parameter count fails the build, so the pack never names a model
the catalog does not ship and the main entry never carries the catalog rows.

```ts
import { piiPack } from "openmed";

const pack = await piiPack.load({ wasmPaths: "/ort/" });
const spans = await pack.extract(text);
showNotice(pack.notice);
```

`piiPack.load(options)` tries the candidates in rank order through the
OpenMed ONNX loader and falls back to the next when one is not available.
Without `allowRemoteModels: true` it loads from local files only
(`localFilesOnly: true, allowRemoteModels: false`), refuses remote ONNX
Runtime assets such as the default CDN `wasmPaths`, and never fetches; when no
candidate is local it fails with `TaskPackError` code
`local_model_unavailable`. With consent, a candidate may be downloaded, and
`candidates_unavailable` means none loaded. Package-wide offline-only mode
refuses the Hub ids with `remote_model_refused`. The default model candidate
requests the package's pinned `DEFAULT_MODEL_REVISION`. Pass `pipeline` to
use your own token-classification pipeline: `pack.extract()` then returns
exactly what `extractPii()` returns for that pipeline. Inference stays on the
device, the pack logs nothing, and its errors carry codes only.

`scripts/web/build_web_task_packs.py` writes the bundled module
`js/openmedkit-web/src/task-packs.generated.ts`; run it after regenerating
this manifest or the web catalog, and `--check` fails when the module is
stale.

## Regenerating the manifest

Change a pack in `TASK_PACK_DECLARATIONS`, or refresh `models.jsonl`, then
run:

```bash
python scripts/manifest/build_task_packs.py
python scripts/manifest/build_task_packs.py --check
```

The command is offline and byte-stable. `--check` writes nothing and exits
non-zero when the committed manifest is stale; the unit tests run the same
check.

## Versioning

The manifest carries `"schema": "openmed.task_packs"` and a semantic
`schema_version`. Readers reject any major version other than `1` and any
field their copy of the schema does not declare, so an unknown document fails
closed. Adding an optional field bumps the minor version and ships with an
updated schema; adding a required field or changing the meaning of a field
bumps the major version. Each pack also
carries its own semantic `version`, bumped when its labels, rules or notice
change. Packs will later gain regulatory-tier and intended-use fields from the
registry work.
