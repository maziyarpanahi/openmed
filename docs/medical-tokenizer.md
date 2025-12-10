# Medical-Aware Tokenizer

OpenMed can apply a medical-aware pre-tokenizer to Hugging Face **fast** tokenizers. It uses the fast `tokenizers`
backend (Bert-style) and lets you inject domain exceptions (e.g., `COVID-19`, `IL-6`, `CAR-T`) as added tokens so they
stay intact when possible.

## Python usage

```python
from openmed import analyze_text
from openmed.core import OpenMedConfig

cfg = OpenMedConfig(use_medical_tokenizer=True)

text = "IL-6-mediated cytokine storm post-CAR-T; tocilizumab 8mg/kg started."
result = analyze_text(text, model_name="oncology_detection_superclinical", config=cfg)
print(result.entities)
```

Toggle or extend exceptions at runtime:

```python
cfg = OpenMedConfig(
    use_medical_tokenizer=False,  # disable
    medical_tokenizer_exceptions=["MY-DRUG-001", "ABC-123"]
)
```

Environment overrides (take precedence over defaults):

```
OPENMED_USE_MEDICAL_TOKENIZER=0
OPENMED_MEDICAL_TOKENIZER_EXCEPTIONS="MY-DRUG-001,ABC-123"
```

## CLI usage

Use the new flags on `openmed analyze`:

```bash
openmed analyze --text "COVID-19 patient on IL-6 inhibitor" --no-medical-tokenizer
openmed analyze --text "t(8;21) AML post-CAR-T" --medical-tokenizer-exceptions "MY-TERM,ABC-001"
```

Without flags, the CLI inherits the config file and defaults to the medical tokenizer on.

## How it works

- Built with `tokenizers` (Rust) Bert-style pre-tokenizer; exceptions are injected as added tokens to reduce splitting of
  critical biomedical terms.
- Falls back silently to the model’s native tokenizer if `tokenizers` is unavailable or the tokenizer is a slow (Python)
  variant.

## Examples

See `examples/custom_tokenizer/` for runnable scripts:

- `custom_tokenize_alignment.py` – custom tokens -> model -> back-map labels.
- `eval_tokenization_comparison.py` – tables comparing WordPiece vs spaCy vs medical pre-tokenizer on hard clinical text.
- `notebooks/Medical_Tokenizer_Benchmark.ipynb` – quick latency + entity stability check with tokenizer on/off.

Run them after installing `examples/custom_tokenizer/requirements.txt`.

```bash
.venv-openmed/bin/python examples/custom_tokenizer/custom_tokenize_alignment.py
.venv-openmed/bin/python examples/custom_tokenizer/eval_tokenization_comparison.py
```

## Recommendations

- Keep the medical tokenizer enabled for clinical/biomedical text; disable if you deliberately want raw model tokenization
  (e.g., benchmarking against published baselines).
- Add project-specific exceptions for proprietary codes or trial IDs via config or CLI flags.
