---
license: apache-2.0
pipeline_tag: token-classification
library_name: openmed
tags:
- openmed
- medical-nlp
---

# OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx

This model card is rendered from the OpenMed model manifest. Update `models.jsonl` and rerun the publish step instead of editing this file directly.

## Manifest Summary

| Field | Value |
|---|---|
| Repository | `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` |
| Family | PII |
| Task | token-classification |
| Languages | tr |
| Tier | Small |
| Parameters | 44M (44,000,000) |
| Architecture | deberta-v2 |
| Base model | OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1 |
| Formats | mlx-fp, mlx-8bit |
| License | apache-2.0 |
| arXiv | [arXiv:2508.01630](https://arxiv.org/abs/2508.01630) |
| Reproducibility hash | `sha256:1111111111111111111111111111111111111111111111111111111111111111` |
| Released | 2026-06-14 |

## Benchmark

| Dataset | Micro F1 | Recall |
|---|---:|---:|
| openmed-golden-pii | 0.9823 | 0.9910 |

## Canonical Labels

`PERSON`, `DATE`, `ID_NUM`
