# Custom Tokenizer Demos

This folder contains runnable examples showing how to:
- Keep your own clinical tokenization while still using OpenMed / Hugging Face models.
- Compare baseline WordPiece tokenization to SciSpaCy-style rules and a fast `tokenizers` port.

## Files
- `custom_tokenize_alignment.py` — minimal example: custom regex tokenizer -> OpenMed encoder -> remap predictions back to your tokens (prints before/after).
- `eval_tokenization_comparison.py` — harder clinical corpus, shows side-by-side tables of tokens + labels for:
  - HF WordPiece (model tokenizer)
  - spaCy tokenizer configured with SciSpaCy-style rules
  - Hugging Face `tokenizers` port of those rules
- `requirements.txt` — extras needed for these demos (spaCy, tokenizers). Core OpenMed deps already cover transformers/torch.

## Setup
```bash
python -m venv .venv-openmed
source .venv-openmed/bin/activate
pip install -r requirements.txt
pip install -e .   # from repo root to get openmed package
```

## Run the demos
```bash
# Alignment demo
.venv-openmed/bin/python examples/custom_tokenizer/custom_tokenize_alignment.py

# Tokenization comparison with tables
.venv-openmed/bin/python examples/custom_tokenizer/eval_tokenization_comparison.py

# Toggle medical tokenizer via CLI (uses config defaults otherwise)
.venv-openmed/bin/openmed analyze --text "COVID-19 patient on IL-6 inhibitor" --no-medical-tokenizer
```

## Notes
- Default model is `oncology_detection_superclinical` in the alignment demo (override via `OPENMED_MODEL` env var).
- The comparison script uses `oncology_detection_tiny` for speed; swap inside the script if you want higher-capacity models.
