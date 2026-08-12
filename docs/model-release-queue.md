# Model Release Queue

The scheduled model release workflow reads `recipes/queue.yaml`. The queue is
the small, reviewable control plane for daily model publication: each row names
one model, the weekday theme it belongs to, which artifact formats to build,
and whether the artifact should be published after conversion. The workflow
runs at 06:17 UTC every weekday and keeps the manual single-model dispatch.

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

- `id`: stable lowercase queue key used in workflow matrix entries and artifact
  names. It may contain letters, digits, `.`, `_`, and `-`.
- `weekday`: one of `monday`, `tuesday`, `wednesday`, `thursday`, or `friday`.
- `theme`: the weekly release theme for that row. It must match the
  `weekly_themes` value for the selected weekday.
- `model_id`: source model repository to convert.
- `formats`: one or more of `mlx-fp`, `mlx-8bit`, `mlx-4bit`, or `coreml`.
- `publish`: a YAML boolean controlling whether the converted artifact is pushed
  after conversion and gates.

Optional fields:

- `depends_on_green_parent`: queue item ids that must precede an edge artifact
  by at least one day. Each parent must be a published, non-edge artifact for
  the same source model.
- `gate_command`: an argument list to run after conversion and before publish
  when a gate exists. Use a YAML list, not a shell command string.

## Weekly Ordering

The scheduled workflow selects the queue rows for the current UTC weekday.
Monday and Tuesday rows publish parent artifacts first. Wednesday rows are
reserved for edge artifacts such as MLX 8-bit and CoreML and must declare
`depends_on_green_parent`, pointing at Monday or Tuesday parent rows. Queue
curators add an edge row only after its referenced parent has passed the
available gates. Before creating the matrix, the dispatcher fails closed unless
each referenced parent exists, is published, uses the same `model_id`, is a
non-edge artifact, and precedes the edge row by at least one day.

Thursday maps to benchmark refreshes and Friday maps to the SDK release train.
Those themes may have no model-conversion rows; an empty day produces a skipped
batch rather than inventing work outside the reviewed queue.

Each queued model runs as an independent matrix item with `fail-fast: false`.
Conversion runs first, then `gate_command` when configured, then the existing HF
publish path. The write token is removed from conversion and gate environments.
A failed model therefore does not cancel the rest of the batch, but its named
matrix job still fails and surfaces the queue `id` in the workflow result.
