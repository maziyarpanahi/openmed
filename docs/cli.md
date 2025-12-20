# CLI & Automation

The `openmed` console script mirrors the Python APIs so you can analyze snippets, inspect registry metadata, and tweak
configuration without writing code. It is perfect for demos, notebooks, or CI smoke tests.

## Installation

The CLI is installed automatically with the base package:

```bash
uv pip install .
# or pip install openmed
openmed --help
```

## Analyze text from the terminal

```bash
openmed analyze \
  --model disease_detection_superclinical \
  --text "Imatinib inhibits BCR-ABL in chronic myeloid leukemia." \
  --output-format json \
  --confidence-threshold 0.6 \
  --group-entities
```

Flags:

- `--model`: registry key or HF id.
- Provide either `--text` or `--input-file`.
- `--output-format`: `dict`, `json`, `html`, or `csv`.
- `--group-entities`: toggles adjacent span merging.
- `--no-confidence`: omit confidence values from output.
- `--use-medical-tokenizer` / `--no-medical-tokenizer`: force on/off medical token remapping in the output (defaults to config/on).
- `--medical-tokenizer-exceptions`: comma-separated terms to keep intact when remapping.

## Discover models

```bash
openmed models list --include-remote   # adds Hugging Face search
openmed models info pharma_detection_superclinical
```

`models list` prints registry models (and optionally HF remote ones). `models info <key>` dumps the curated metadata so you
can confirm categories, entity types, recommended confidence, and model sizes.

## Manage configuration

```bash
openmed config show
openmed config set device cuda
openmed config set default_org OpenMed
openmed config set hf_token xxx --unset    # remove a setting
```

- Values are stored in the CLI config file (see `openmed/core/config.py` for defaults). Use `--config-path` to point at an
  alternate location when scripting CI/CD workflows.
- The CLI reuses `OpenMedConfig`, so any changes you make here apply to both CLI runs and Python imports (assuming they
  load the same file).

## Automation tips

- Wrap CLI calls in shell scripts for smoke tests. Example:

  ```bash
  openmed models list >/tmp/models.txt
  openmed analyze --model disease_detection_superclinical --text "$SAMPLE_NOTE" --output-format json > /tmp/result.json
  ```

- Use `uv run` or `pipx run` to avoid polluting system environments when scripting release pipelines.
- In GitHub Actions, you can execute the CLI after installing `.[hf]` to ensure models load before publishing.

For more complex workflows (batch jobs, structured logging, streaming), prefer the Python APIs covered in
[Analyze Text Helper](./analyze-text.md) and [ModelLoader & Pipelines](./model-loader.md).
