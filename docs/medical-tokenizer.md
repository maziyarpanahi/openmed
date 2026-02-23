# Medical-Aware Tokenizer

OpenMed supports a medical-aware tokenizer **for output remapping**:

1) OpenMed runs the model using the model’s own Hugging Face tokenizer (WordPiece/BPE), unchanged.
2) OpenMed tokenizes the same text using a medical-friendly tokenizer (span tokens with offsets).
3) Model predictions (char spans) are remapped onto the medical tokens and merged, producing cleaner entities for UI and
   downstream pipelines.

This avoids imposing new tokens/vocabulary on the model and keeps inference behavior stable.

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

## How it works

- Uses `medical_tokenize(...)` to produce stable clinical tokens with character offsets (no model involvement).
- Uses `remap_predictions_to_tokens(...)` to project model spans back to those tokens and merge adjacent tokens with the
  same label.
- Does **not** modify the model tokenizer, model vocab, or embeddings.

## Examples

See `examples/custom_tokenizer/` for runnable scripts:

- `custom_tokenize_alignment.py` – custom tokens -> model -> back-map labels.
- `eval_tokenization_comparison.py` – tables comparing WordPiece vs spaCy vs medical pre-tokenizer on hard clinical text.
- `compare_medical_remap.py` – side-by-side comparison of OpenMed outputs with remapping on/off.
- `notebooks/Medical_Tokenizer_Benchmark.ipynb` – quick latency + entity stability check with tokenizer on/off.

Run them after installing `examples/custom_tokenizer/requirements.txt`.

```bash
.venv-openmed/bin/python examples/custom_tokenizer/custom_tokenize_alignment.py
.venv-openmed/bin/python examples/custom_tokenizer/eval_tokenization_comparison.py
```

## Recommendations

- Keep the medical tokenizer enabled for clinical/biomedical text; disable if you deliberately want raw model tokenization
  (e.g., benchmarking against published baselines).
- Add project-specific exceptions for proprietary codes or trial IDs via config values.
