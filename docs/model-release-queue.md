# Model Release Queue

The scheduled model release workflow reads `recipes/queue.yaml`. The queue is the small, reviewable control plane for
daily model publication: each row names one model, the weekday theme it belongs to, which artifact formats to build, and
whether the artifact should be published after conversion.

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

- `id`: stable queue key used in workflow matrix entries and artifact names.
- `weekday`: one of `monday`, `tuesday`, `wednesday`, `thursday`, or `friday`.
- `theme`: the weekly release theme for that row.
- `model_id`: source model repository to convert.
- `formats`: one or more of `mlx-fp`, `mlx-8bit`, `mlx-4bit`, or `coreml`.
- `publish`: whether the converted artifact is pushed after conversion and gates.

Optional fields:

- `depends_on_green_parent`: queue item ids that must precede an edge artifact by at least one day.
- `gate_command`: command list to run after conversion and before publish when a gate exists.

## Weekly Ordering

The scheduled workflow runs on weekdays and selects the queue rows for the current UTC weekday. Monday and Tuesday rows
publish parent artifacts first. Wednesday rows are reserved for edge artifacts such as MLX 8-bit and CoreML and must
declare `depends_on_green_parent`, pointing at Monday or Tuesday parent rows. The dispatcher validates that those parent
rows exist and are earlier in the week before it creates the workflow matrix.

Each queued model runs as an independent matrix item with `fail-fast: false`. A failed model therefore does not cancel
the rest of the batch, but the failed item still marks the run as failed and appears by queue `id` in the logs.
