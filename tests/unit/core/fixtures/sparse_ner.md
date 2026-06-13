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
library_name: transformers
---

# 🧬 [OpenMed-NER-AnatomyDetect-TinyMed-135M](https://huggingface.co/OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M)

**OpenMed NER model · token-classification**

[![License](https://img.shields.io/badge/License-apache-2.0-blue.svg)](https://huggingface.co/OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M)
[![OpenMed](https://img.shields.io/badge/🏥-OpenMed-green)](https://huggingface.co/OpenMed)
[![arXiv](https://img.shields.io/badge/arXiv-2508.01630-b31b1b.svg)](https://arxiv.org/abs/2508.01630)

> ⚙️ This model card is generated automatically from the OpenMed model manifest. Do not edit by hand.

## 📋 Model Details

| Field | Value |
|-------|-------|
| Tier | Tiny |
| Parameters | 135M |
| Architecture | distilbert |
| Formats | `pytorch` |
| Languages | en |
| License | apache-2.0 |
| Released | 2025-08-05 |

## 🏷️ Entity Labels

- `ORGAN`
- `TISSUE`
- `ANATOMY`

## 🚀 Quick Start

```python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="OpenMed/OpenMed-NER-AnatomyDetect-TinyMed-135M",
    aggregation_strategy="simple",
)

entities = ner("The liver showed signs of fatty infiltration.")
print(entities)
```

## 📊 Benchmark

| Dataset | Micro-F1 | Recall |
|---------|----------|--------|
| — | — | — |

## 🔁 Reproducibility

- **Manifest hash**: `sha256:eec908c76996f710aaf18b180e8668c356a22d9bea64e031ae1237718be167fb`

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
