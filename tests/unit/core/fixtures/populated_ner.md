---
license: apache-2.0
language:
- en
pipeline_tag: token-classification
tags:
- named-entity-recognition
- biomedical-nlp
- openmed
- medical
base_model: google/electra-large-discriminator
library_name: transformers
---

# 🧬 [OpenMed-NER-OncologyDetect-ElectraMed-560M](https://huggingface.co/OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M)

**OpenMed NER model · token-classification**

[![License](https://img.shields.io/badge/License-apache-2.0-blue.svg)](https://huggingface.co/OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M)
[![OpenMed](https://img.shields.io/badge/🏥-OpenMed-green)](https://huggingface.co/OpenMed)
[![arXiv](https://img.shields.io/badge/arXiv-2508.01630-b31b1b.svg)](https://arxiv.org/abs/2508.01630)

> ⚙️ This model card is generated automatically from the OpenMed model manifest. Do not edit by hand.

## 📋 Model Details

| Field | Value |
|-------|-------|
| Tier | Large |
| Parameters | 560M |
| Architecture | electra |
| Base model | google/electra-large-discriminator |
| Formats | `pytorch`, `mlx-fp` |
| Languages | en |
| License | apache-2.0 |
| Released | 2025-08-05 |

## 🏷️ Entity Labels

- `CANCER`
- `TREATMENT`

## 🚀 Quick Start

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="OpenMed/OpenMed-NER-OncologyDetect-ElectraMed-560M",
    aggregation_strategy="simple",
)

entities = ner("The liver showed signs of fatty infiltration.")
print(entities)
```

## 📊 Benchmark

| Dataset | Micro-F1 | Recall |
|---------|----------|--------|
| ONCOLOGY | 0.9063 | 0.9044 |

## 🔁 Reproducibility

- **Manifest hash**: `sha256:abc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abcd`

## 📜 Citation

```bibtex
@misc{panahi2025openmedner,
      title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets},
      author={Maziyar Panahi},
      year={2025},
      eprint={2508.01630},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.01630},
}
```

## 📄 License

Licensed under the **apache-2.0** license.
