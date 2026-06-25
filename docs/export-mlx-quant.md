# MLX Quantized Export Certification

OpenMed MLX exports can emit INT4 artifacts, but an INT4 artifact is only
marked certified when it holds recall against its full-precision parent on the
same synthetic or approved eval fixtures.

## INT4 Export

```bash
python -m openmed.mlx.convert \
  --model OpenMed/example-token-classifier \
  --output dist/example-mlx-4bit \
  --quantize 4 \
  --quantize-group-size 64 \
  --eval-suite openmed/eval/golden/fixtures/checksum_ids.json
```

When `--eval-suite` is supplied with `--quantize 4`, conversion writes the INT4
artifact and then runs:

- the Hugging Face full-precision parent
- the emitted MLX artifact

Both runs use the same benchmark fixtures and character-weighted label recall.
The resulting `recall_delta.json` is written next to the artifact unless
`--recall-delta-report` supplies a different path.

## Report Contract

`recall_delta.json` contains:

- `format`: `mlx-4bit`
- `quantization.bits` and `quantization.group_size`
- `fp_parent_per_label_recall`
- `candidate_per_label_recall`
- `per_label`, with FP recall, INT4 recall, and delta per label
- `quant_recall_delta`, the single max recall-loss figure consumed by G4
- `limit`, currently `0.010` for INT4
- `certified`, true only when the measured delta is below the INT4 limit

The artifact `config.json` and `openmed-mlx.json` mirror the certification
fields under `quant_recall_delta`, `certified`, `recall_delta_path`, and
`quantization`. An over-budget INT4 export still writes the artifact and report,
but it records `certified: false` so the release gate can block the format.
