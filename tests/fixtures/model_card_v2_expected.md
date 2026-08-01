---
license: apache-2.0
pipeline_tag: token-classification
library_name: openmed
tags:
- openmed
- medical-nlp
---

# OpenMed-PII-Chinese-BigMed-Large-560M-v1

This model card is rendered from the OpenMed model manifest. Update `models.jsonl` and rerun the publish step instead of editing this file directly.

## Manifest Summary

| Field | Value |
|---|---|
| Repository | `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` |
| Family | PII |
| Task | token-classification |
| Languages | zh |
| Tier | Large |
| Parameters | 560M (560,000,000) |
| Architecture | xlm-roberta |
| Base model | FacebookAI/xlm-roberta-large |
| Formats | pytorch |
| License | apache-2.0 |
| arXiv | [arXiv:2508.01630](https://arxiv.org/abs/2508.01630) |
| Reproducibility hash | `sha256:1111111111111111111111111111111111111111111111111111111111111111` |
| Released | 2026-06-14 |

## Benchmark

| Dataset | Micro F1 | Recall |
|---|---:|---:|
| AI4Privacy + Synthetic Chinese PII | 0.7589 | 0.7517 |

## Per-Script Evaluation

| Script | Dataset | Recall | Leakage floor |
|---|---|---:|---:|
| Han Simplified | AI4Privacy + Synthetic Chinese PII | 0.7517 | Not reported |
| Han Traditional | Not reported | Not reported | Not reported |

## Download Size by Format

| Format | Download size |
|---|---:|
| Safetensors | 2,235.723 MB |
| MLX | 2,235.717 MB |
| Core ML | Not published |
| ONNX | 1,330.432 MB |

## Tokenizer Script Coverage

| Script | UNK rate | Byte fallback rate | Tokens / grapheme | Verdict |
|---|---:|---:|---:|---|
| Han Simplified | 0.00% | 0.00% | 0.7500 | supported |
| Han Traditional | 0.00% | 0.00% | 0.7500 | supported |
| Devanagari | 0.00% | 0.00% | 0.7500 | unclaimed |
| Bengali | 0.00% | 0.00% | 0.7500 | unclaimed |
| Tamil | 0.00% | 0.00% | 0.7500 | unclaimed |
| Telugu | 0.00% | 0.00% | 0.7500 | unclaimed |
| Kannada | 0.00% | 0.00% | 0.7500 | unclaimed |
| Malayalam | 0.00% | 0.00% | 0.7500 | unclaimed |
| Gujarati | 0.00% | 0.00% | 0.7500 | unclaimed |
| Gurmukhi | 0.00% | 0.00% | 0.7500 | unclaimed |
| Odia | 0.00% | 0.00% | 0.7500 | unclaimed |

## License

Declared license: `apache-2.0`.

## Canonical Labels

`PERSON`, `DATE`, `ID_NUM`
