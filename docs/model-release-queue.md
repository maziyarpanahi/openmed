# Local Model Release Queue

`recipes/queue.yaml` is an optional, reviewable input for a maintainer-run local
model release. GitHub Actions does not run this queue automatically. The
weekday fields preserve dependency ordering and grouping; they do not define a
cron schedule.

Before running any conversion or publication command, validate and inspect the
selected rows locally:

```bash
python scripts/release/dispatch_batch.py validate --queue recipes/queue.yaml
python scripts/release/dispatch_batch.py plan \
  --queue recipes/queue.yaml \
  --weekday monday
python scripts/release/dispatch_batch.py run-batch \
  --queue recipes/queue.yaml \
  --weekday monday \
  --dry-run
```

An actual run must be initiated explicitly on maintainer-controlled hardware
after the dry-run commands, target repositories, gate commands, and available
resources have been reviewed. Rows with `publish: true` require the local token
and evidence controls in the [manual Hugging Face publication
policy](security/hf-token-policy.md).

## Queue Format

```yaml
version: 1
weekly_themes:
  monday: language-pack
  tuesday: clinical-ner
  wednesday: quantized-edge
  thursday: benchmark-refresh
  friday: sdk-release
items:
  - id: pii-french-small-v1-mlx
    weekday: monday
    theme: language-pack
    model_id: OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1
    formats:
      - mlx-fp
    publish: true
```

Required item fields:

- `id`: stable lowercase queue key used in local reports and artifact names. It
  may contain letters, digits, `.`, `_`, and `-`.
- `weekday`: one of `monday`, `tuesday`, `wednesday`, `thursday`, or `friday`.
- `theme`: the release theme for that row. It must match the `weekly_themes`
  value for the selected weekday.
- `model_id`: source model repository to convert.
- `formats`: one or more of `mlx-fp`, `mlx-8bit`, `mlx-4bit`, or `coreml`.
- `publish`: a YAML boolean controlling whether an explicitly invoked local run
  attempts publication after conversion and gates.

Optional fields:

- `depends_on_green_parent`: queue item ids that must precede an edge artifact
  by at least one day. Each parent must be a reviewed, non-edge artifact for
  the same source model.
- `gate_command`: an argument list to run after conversion and before publish.
  Use a YAML list, not a shell command string.

## Local Ordering

Monday and Tuesday rows list parent artifacts first. Wednesday rows are
reserved for edge artifacts such as MLX 8-bit and Core ML and must declare
`depends_on_green_parent`, pointing at an earlier parent row. The dispatcher
fails closed unless each referenced parent exists, is marked for publication,
uses the same `model_id`, is a non-edge artifact, and precedes the edge row by
at least one day.

Thursday and Friday remain grouping labels for benchmark and SDK work. An empty
selection performs no work. The local dispatcher records each selected model
independently so one failure remains visible, but it does not substitute for a
maintainer reviewing the complete batch before any upload.
