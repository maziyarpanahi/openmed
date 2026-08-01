# PII Tokenizer Script Coverage Audit

This committed audit covers 654 PII-family models across 11 script targets. The unsupported threshold is strictly greater than 1% UNK tokens on a script claimed by the model's declared language.

- Model-script pairs above the UNK threshold: 4114
- Claimed model-script pairs marked unsupported: 76

| Model | Languages | Script | UNK | Byte fallback | Tokens/grapheme | Verdict | Threshold flag |
|---|---|---|---:|---:|---:|---|---|
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/gliner-multi-pii-v1-mlx` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1` | ar | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-BigMed-Large-560M-v1-mlx` | ar | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1` | ar | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Arabic-SnowflakeMed-Large-568M-v1-mlx` | ar | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Bengali-mSuperClinical-Large-279M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `han_simplified` | 0.00% | 71.72% | 1.812 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `han_traditional` | 0.00% | 79.22% | 2.110 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `devanagari` | 0.00% | 92.35% | 4.479 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `bengali` | 0.00% | 92.90% | 4.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `tamil` | 0.00% | 92.77% | 4.297 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `telugu` | 0.00% | 93.73% | 5.243 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `kannada` | 0.00% | 89.52% | 3.418 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `malayalam` | 0.00% | 89.87% | 3.118 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `gujarati` | 0.00% | 90.13% | 3.192 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `gurmukhi` | 0.00% | 90.09% | 3.135 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1` | en | `odia` | 0.00% | 92.90% | 4.133 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-278M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `han_simplified` | 0.00% | 71.72% | 1.812 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `han_traditional` | 0.00% | 79.22% | 2.110 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `devanagari` | 0.00% | 92.35% | 4.479 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `bengali` | 0.00% | 92.90% | 4.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `tamil` | 0.00% | 92.77% | 4.297 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `telugu` | 0.00% | 93.73% | 5.243 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `kannada` | 0.00% | 89.52% | 3.418 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `malayalam` | 0.00% | 89.87% | 3.118 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `gujarati` | 0.00% | 90.13% | 3.192 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `gurmukhi` | 0.00% | 90.09% | 3.135 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1` | en | `odia` | 0.00% | 92.90% | 4.133 | unclaimed |  |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BigMed-Large-560M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalBERT-Base-110M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Base-149M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-BioClinicalModern-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Base-110M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedBERT-Large-340M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Base-110M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-BiomedELECTRA-Large-335M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BigMed-Large-560M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-BioClinicalModern-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-ModernMed-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-NomicMed-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-QwenMed-XLarge-600M-v1` | en | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Chinese-SnowflakeMed-Large-568M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-335M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalBGE-Large-568M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Base-109M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Large-335M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalE5-Small-33M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicalLongformer-Base-149M-v1` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ClinicDischarge-Base-110M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1` | nl | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-278M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1` | nl | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BigMed-Large-560M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalBERT-Base-110M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Base-149M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BioClinicalModern-Large-395M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Base-110M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERT-Large-340M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedBERTFull-Base-110M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Base-110M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-BiomedELECTRA-Large-335M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-335M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1` | nl | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalBGE-Large-568M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Base-109M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Large-335M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalE5-Small-33M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `han_simplified` | 0.00% | 83.76% | 1.462 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `han_traditional` | 0.00% | 86.40% | 1.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `devanagari` | 0.00% | 94.01% | 2.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `bengali` | 0.00% | 94.68% | 3.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `tamil` | 0.00% | 96.18% | 5.311 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `telugu` | 0.00% | 96.18% | 5.614 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `kannada` | 0.00% | 95.58% | 5.403 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `malayalam` | 0.00% | 95.84% | 5.066 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `gujarati` | 0.00% | 95.41% | 4.479 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicalLongformer-Base-149M-v1` | nl | `odia` | 0.00% | 95.78% | 4.427 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ClinicDischarge-Base-110M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-EuroMed-Large-210M-v1` | nl | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1` | nl | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-FastClinical-Small-82M-v1-mlx` | nl | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-GTEMed-Base-149M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinical-Small-66M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-LiteClinicalU-Small-66M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1` | nl | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mClinicalE5-Large-560M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mLiteClinical-Base-135M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Base-149M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-ModernMed-Large-395M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1` | nl | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-mSuperClinical-Large-279M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-NomicMed-Large-395M-v1` | nl | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-QwenMed-XLarge-600M-v1` | nl | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1` | nl | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SnowflakeMed-Large-568M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1` | nl | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Base-184M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1` | nl | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Large-434M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1` | nl | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperClinical-Small-44M-v1-mlx` | nl | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1` | nl | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Base-125M-v1-mlx` | nl | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1` | nl | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Dutch-SuperMedical-Large-355M-v1-mlx` | nl | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-EuroMed-Large-210M-v1` | en | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-FastClinical-Small-82M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1` | fr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-278M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1` | fr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BigMed-Large-560M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalBERT-Base-110M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Base-149M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BioClinicalModern-Large-395M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Base-110M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERT-Large-340M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedBERTFull-Base-110M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Base-110M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-BiomedELECTRA-Large-335M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-335M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1` | fr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalBGE-Large-568M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Base-109M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Large-335M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalE5-Small-33M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicalLongformer-Base-149M-v1` | fr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ClinicDischarge-Base-110M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-EuroMed-Large-210M-v1` | fr | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1` | fr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-FastClinical-Small-82M-v1-mlx` | fr | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-GTEMed-Base-149M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinical-Small-66M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-LiteClinicalU-Small-66M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1` | fr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mClinicalE5-Large-560M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mLiteClinical-Base-135M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Base-149M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-ModernMed-Large-395M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1` | fr | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-mSuperClinical-Large-279M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-NomicMed-Large-395M-v1` | fr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-QwenMed-XLarge-600M-v1` | fr | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1` | fr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1` | fr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Base-184M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1` | fr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Large-434M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1` | fr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperClinical-Small-44M-v1-mlx` | fr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1` | fr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Base-125M-v1-mlx` | fr | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1` | fr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-French-SuperMedical-Large-355M-v1-mlx` | fr | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1` | de | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-278M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1` | de | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BigMed-Large-560M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalBERT-Base-110M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Base-149M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BioClinicalModern-Large-395M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Base-110M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERT-Large-340M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedBERTFull-Base-110M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Base-110M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-BiomedELECTRA-Large-335M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-335M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1` | de | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalBGE-Large-568M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Base-109M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Large-335M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalE5-Small-33M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicalLongformer-Base-149M-v1` | de | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ClinicDischarge-Base-110M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-EuroMed-Large-210M-v1` | de | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1` | de | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-FastClinical-Small-82M-v1-mlx` | de | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-GTEMed-Base-149M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinical-Small-66M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-LiteClinicalU-Small-66M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1` | de | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mClinicalE5-Large-560M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mLiteClinical-Base-135M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Base-149M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-ModernMed-Large-395M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1` | de | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-mSuperClinical-Large-279M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-NomicMed-Large-395M-v1` | de | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-QwenMed-XLarge-600M-v1` | de | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1` | de | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SnowflakeMed-Large-568M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1` | de | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Base-184M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1` | de | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Large-434M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1` | de | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperClinical-Small-44M-v1-mlx` | de | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1` | de | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Base-125M-v1-mlx` | de | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1` | de | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-German-SuperMedical-Large-355M-v1-mlx` | de | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-GTEMed-Base-149M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.521 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1` | hi | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-278M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.521 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1` | hi | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BigMed-Large-560M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `devanagari` | 18.75% | 0.00% | 0.877 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalBERT-Base-110M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Base-149M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BioClinicalModern-Large-395M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Base-110M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERT-Large-340M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedBERTFull-Base-110M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Base-110M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-BiomedELECTRA-Large-335M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `devanagari` | 8.86% | 0.00% | 1.082 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-335M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.521 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1` | hi | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalBGE-Large-568M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `devanagari` | 8.86% | 0.00% | 1.082 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Base-109M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `devanagari` | 8.86% | 0.00% | 1.082 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Large-335M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `devanagari` | 8.86% | 0.00% | 1.082 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `devanagari` | 0.00% | 92.73% | 3.014 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicalLongformer-Base-149M-v1` | hi | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `devanagari` | 18.75% | 0.00% | 0.877 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ClinicDischarge-Base-110M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `devanagari` | 0.00% | 0.00% | 1.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-EuroMed-Large-210M-v1` | hi | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `devanagari` | 0.00% | 92.73% | 3.014 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1` | hi | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `devanagari` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-FastClinical-Small-82M-v1-mlx` | hi | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-GTEMed-Base-149M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `devanagari` | 15.79% | 0.00% | 1.041 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `devanagari` | 8.86% | 0.00% | 1.082 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-LiteClinicalU-Small-66M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.521 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1` | hi | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mClinicalE5-Large-560M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.863 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mLiteClinical-Base-135M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Base-149M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-ModernMed-Large-395M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.904 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1` | hi | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-mSuperClinical-Large-279M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `devanagari` | 0.00% | 53.50% | 2.151 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-NomicMed-Large-395M-v1` | hi | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `devanagari` | 0.00% | 67.59% | 1.986 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-QwenMed-XLarge-600M-v1` | hi | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `devanagari` | 0.00% | 0.00% | 0.521 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1` | hi | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SnowflakeMed-Large-568M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `devanagari` | 0.00% | 0.00% | 1.356 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1` | hi | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Base-184M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `devanagari` | 0.00% | 0.00% | 1.356 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1` | hi | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Large-434M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `devanagari` | 0.00% | 0.00% | 1.356 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | hi | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `devanagari` | 100.00% | 0.00% | 0.342 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1-mlx` | hi | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `devanagari` | 0.00% | 92.73% | 3.014 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1` | hi | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `devanagari` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Base-125M-v1-mlx` | hi | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `devanagari` | 0.00% | 92.73% | 3.014 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1` | hi | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `devanagari` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Hindi-SuperMedical-Large-355M-v1-mlx` | hi | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1` | it | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-278M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1` | it | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BigMed-Large-560M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalBERT-Base-110M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Base-149M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BioClinicalModern-Large-395M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Base-110M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERT-Large-340M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedBERTFull-Base-110M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Base-110M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-BiomedELECTRA-Large-335M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-335M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1` | it | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalBGE-Large-568M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Base-109M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Large-335M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalE5-Small-33M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicalLongformer-Base-149M-v1` | it | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ClinicDischarge-Base-110M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-EuroMed-Large-210M-v1` | it | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1` | it | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-FastClinical-Small-82M-v1-mlx` | it | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-GTEMed-Base-149M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinical-Small-66M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-LiteClinicalU-Small-66M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1` | it | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mClinicalE5-Large-560M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mLiteClinical-Base-135M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Base-149M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-ModernMed-Large-395M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1` | it | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-mSuperClinical-Large-279M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-NomicMed-Large-395M-v1` | it | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-QwenMed-XLarge-600M-v1` | it | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1` | it | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SnowflakeMed-Large-568M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1` | it | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Base-184M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1` | it | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Large-434M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1` | it | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperClinical-Small-44M-v1-mlx` | it | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1` | it | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Base-125M-v1-mlx` | it | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1` | it | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Italian-SuperMedical-Large-355M-v1-mlx` | it | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `han_simplified` | 0.00% | 0.00% | 0.662 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `han_traditional` | 0.00% | 0.00% | 0.767 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1` | ja | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `han_simplified` | 0.00% | 0.00% | 0.662 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `han_traditional` | 0.00% | 0.00% | 0.767 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-BigMed-Large-560M-v1-mlx` | ja | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `han_simplified` | 0.00% | 65.17% | 1.113 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `han_traditional` | 0.00% | 73.12% | 1.274 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-NomicMed-Large-395M-v1` | ja | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `han_simplified` | 0.00% | 0.00% | 0.787 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `han_traditional` | 0.00% | 2.86% | 0.959 | supported |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Japanese-QwenMed-XLarge-600M-v1` | ja | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-NomicMed-Large-395M-v1` | ko | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Korean-QwenMed-XLarge-600M-v1` | ko | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-LiteClinical-Small-66M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mClinicalE5-Large-560M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mLiteClinical-Base-135M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Base-149M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-ModernMed-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-mSuperClinical-Large-279M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-NomicMed-Large-395M-v1` | en | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-278M-v1-mlx` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BigMed-Large-560M-v1-mlx` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalBERT-Base-110M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Base-149M-v1` | pt | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BioClinicalModern-Large-395M-v1` | pt | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Base-110M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERT-Large-340M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedBERTFull-Base-110M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Base-110M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-BiomedELECTRA-Large-335M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-335M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalBGE-Large-568M-v1-mlx` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Base-109M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Large-335M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalE5-Small-33M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `han_simplified` | 0.00% | 83.76% | 1.462 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `han_traditional` | 0.00% | 86.40% | 1.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `devanagari` | 0.00% | 94.01% | 2.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `bengali` | 0.00% | 94.68% | 3.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `tamil` | 0.00% | 96.18% | 5.311 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `telugu` | 0.00% | 96.18% | 5.614 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `kannada` | 0.00% | 95.58% | 5.403 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `malayalam` | 0.00% | 95.84% | 5.066 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `gujarati` | 0.00% | 95.41% | 4.479 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicalLongformer-Base-149M-v1` | pt | `odia` | 0.00% | 95.78% | 4.427 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ClinicDischarge-Base-110M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-FastClinical-Small-82M-v1-mlx` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinical-Small-66M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-LiteClinicalU-Small-66M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-mLiteClinical-Base-135M-v1-mlx` | pt | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Base-149M-v1` | pt | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-ModernMed-Large-395M-v1` | pt | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1` | pt | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-mSuperClinical-Large-279M-v1-mlx` | pt | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-NomicMed-Large-395M-v1` | pt | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-QwenMed-XLarge-600M-v1` | pt | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SnowflakeMed-Large-568M-v1-mlx` | pt | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Base-184M-v1-mlx` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Large-434M-v1-mlx` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperClinical-Small-44M-v1-mlx` | pt | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Base-125M-v1-mlx` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Portuguese-SuperMedical-Large-355M-v1-mlx` | pt | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-QwenMed-XLarge-600M-v1` | en | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SnowflakeMed-Large-568M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1` | es | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-278M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1` | es | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BigMed-Large-560M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalBERT-Base-110M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Base-149M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BioClinicalModern-Large-395M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Base-110M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERT-Large-340M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedBERTFull-Base-110M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Base-110M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-BiomedELECTRA-Large-335M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-335M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1` | es | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalBGE-Large-568M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Base-109M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Large-335M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalE5-Small-33M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicalLongformer-Base-149M-v1` | es | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ClinicDischarge-Base-110M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `telugu` | 0.00% | 99.21% | 3.629 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-EuroMed-Large-210M-v1` | es | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1` | es | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-FastClinical-Small-82M-v1-mlx` | es | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-GTEMed-Base-149M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinical-Small-66M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-LiteClinicalU-Small-66M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1` | es | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mClinicalE5-Large-560M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mLiteClinical-Base-135M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Base-149M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-ModernMed-Large-395M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1` | es | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-mSuperClinical-Large-279M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-NomicMed-Large-395M-v1` | es | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-QwenMed-XLarge-600M-v1` | es | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1` | es | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SnowflakeMed-Large-568M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1` | es | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Base-184M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1` | es | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Large-434M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1` | es | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperClinical-Small-44M-v1-mlx` | es | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1` | es | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Base-125M-v1-mlx` | es | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1` | es | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Spanish-SuperMedical-Large-355M-v1-mlx` | es | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Base-184M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Large-434M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Base-125M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-SuperMedical-Large-355M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `telugu` | 0.00% | 0.00% | 0.757 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1` | te | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-278M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `telugu` | 0.00% | 0.00% | 0.757 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1` | te | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BigMed-Large-560M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalBERT-Base-110M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Base-149M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BioClinicalModern-Large-395M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Base-110M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERT-Large-340M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedBERTFull-Base-110M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Base-110M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-BiomedELECTRA-Large-335M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-335M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `telugu` | 0.00% | 0.00% | 0.757 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1` | te | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalBGE-Large-568M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Base-109M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Large-335M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalE5-Small-33M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `telugu` | 0.00% | 94.26% | 5.729 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicalLongformer-Base-149M-v1` | te | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ClinicDischarge-Base-110M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.675 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `han_traditional` | 0.00% | 6.78% | 0.808 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `devanagari` | 0.00% | 0.00% | 1.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `bengali` | 0.00% | 88.76% | 2.438 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `tamil` | 0.00% | 92.38% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `telugu` | 0.00% | 99.21% | 3.629 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `malayalam` | 0.00% | 97.96% | 3.224 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `gujarati` | 0.00% | 99.05% | 2.877 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `gurmukhi` | 0.00% | 99.05% | 2.838 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-EuroMed-Large-210M-v1` | te | `odia` | 0.00% | 99.37% | 4.213 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `telugu` | 0.00% | 94.26% | 5.729 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1` | te | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `telugu` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-FastClinical-Small-82M-v1-mlx` | te | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-GTEMed-Base-149M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinical-Small-66M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-LiteClinicalU-Small-66M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `telugu` | 0.00% | 0.00% | 0.757 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1` | te | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mClinicalE5-Large-560M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `telugu` | 7.94% | 0.00% | 0.900 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.650 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.740 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mLiteClinical-Base-135M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Base-149M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-ModernMed-Large-395M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `telugu` | 0.00% | 0.00% | 0.957 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1` | te | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-mSuperClinical-Large-279M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-NomicMed-Large-395M-v1` | te | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `telugu` | 0.00% | 87.72% | 3.257 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-QwenMed-XLarge-600M-v1` | te | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `telugu` | 0.00% | 0.00% | 0.757 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1` | te | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SnowflakeMed-Large-568M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `telugu` | 3.79% | 0.00% | 1.886 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1` | te | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Base-184M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `telugu` | 3.79% | 0.00% | 1.886 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1` | te | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Large-434M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `telugu` | 3.79% | 0.00% | 1.886 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1` | te | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `han_simplified` | 100.00% | 0.00% | 0.100 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `han_traditional` | 100.00% | 0.00% | 0.110 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `telugu` | 100.00% | 0.00% | 0.329 | unsupported | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperClinical-Small-44M-v1-mlx` | te | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `telugu` | 0.00% | 94.26% | 5.729 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1` | te | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `telugu` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Base-125M-v1-mlx` | te | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `telugu` | 0.00% | 94.26% | 5.729 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1` | te | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `han_simplified` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `han_traditional` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `devanagari` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `bengali` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `tamil` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `telugu` | 0.00% | 0.00% | 0.000 | supported |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `kannada` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `malayalam` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `gujarati` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `gurmukhi` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Telugu-SuperMedical-Large-355M-v1-mlx` | te | `odia` | 0.00% | 0.00% | 0.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-278M-v1-mlx` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BigMed-Large-560M-v1-mlx` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalBERT-Base-110M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Base-149M-v1` | tr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BioClinicalModern-Large-395M-v1` | tr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Base-110M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERT-Large-340M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `han_simplified` | 41.67% | 0.00% | 0.750 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `han_traditional` | 53.33% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedBERTFull-Base-110M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Base-110M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `han_simplified` | 75.41% | 0.00% | 0.762 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `han_traditional` | 81.67% | 0.00% | 0.822 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `devanagari` | 100.00% | 0.00% | 0.342 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `bengali` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `tamil` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-BiomedELECTRA-Large-335M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-335M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalBGE-Large-568M-v1-mlx` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Base-109M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Large-335M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalE5-Small-33M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicalLongformer-Base-149M-v1` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `han_traditional` | 51.56% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `devanagari` | 18.75% | 0.00% | 0.877 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `bengali` | 16.42% | 0.00% | 0.918 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `tamil` | 64.52% | 0.00% | 0.419 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ClinicDischarge-Base-110M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-FastClinical-Small-82M-v1-mlx` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `han_simplified` | 50.79% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `han_traditional` | 52.38% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `devanagari` | 15.79% | 0.00% | 1.041 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `bengali` | 14.86% | 0.00% | 1.014 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `tamil` | 70.00% | 0.00% | 0.405 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinical-Small-66M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-LiteClinicalU-Small-66M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.863 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 1.027 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `tamil` | 7.94% | 0.00% | 0.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `telugu` | 7.94% | 0.00% | 0.900 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 1.284 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `malayalam` | 6.17% | 0.00% | 1.066 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 1.247 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `gurmukhi` | 4.55% | 0.00% | 0.892 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-mLiteClinical-Base-135M-v1-mlx` | tr | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Base-149M-v1` | tr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-ModernMed-Large-395M-v1` | tr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1` | tr | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-mSuperClinical-Large-279M-v1-mlx` | tr | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `han_simplified` | 0.00% | 65.17% | 1.113 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `han_traditional` | 0.00% | 73.12% | 1.274 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `devanagari` | 0.00% | 53.50% | 2.151 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `bengali` | 0.00% | 81.82% | 2.712 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `tamil` | 0.00% | 84.68% | 3.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `kannada` | 0.00% | 99.15% | 3.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `malayalam` | 0.00% | 94.96% | 3.132 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `gujarati` | 0.00% | 93.47% | 2.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `gurmukhi` | 0.00% | 93.27% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-NomicMed-Large-395M-v1` | tr | `odia` | 0.00% | 94.74% | 3.547 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.787 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `han_traditional` | 0.00% | 2.86% | 0.959 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `devanagari` | 0.00% | 67.59% | 1.986 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `bengali` | 0.00% | 77.78% | 2.219 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `tamil` | 0.00% | 72.34% | 2.541 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `telugu` | 0.00% | 87.72% | 3.257 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `kannada` | 0.00% | 83.90% | 3.060 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `malayalam` | 0.00% | 79.81% | 2.737 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `gujarati` | 0.00% | 86.63% | 2.562 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `gurmukhi` | 0.00% | 89.01% | 2.581 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-QwenMed-XLarge-600M-v1` | tr | `odia` | 0.00% | 98.87% | 3.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SnowflakeMed-Large-568M-v1-mlx` | tr | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Base-184M-v1-mlx` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Large-434M-v1-mlx` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperClinical-Small-44M-v1-mlx` | tr | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Base-125M-v1-mlx` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Turkish-SuperMedical-Large-355M-v1-mlx` | tr | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-BigMed-Large-278M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `han_simplified` | 41.27% | 0.00% | 0.787 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `han_traditional` | 42.86% | 0.00% | 0.863 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `devanagari` | 8.86% | 0.00% | 1.082 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `bengali` | 8.43% | 0.00% | 1.137 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `tamil` | 18.46% | 0.00% | 0.878 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `telugu` | 100.00% | 0.00% | 0.329 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `kannada` | 100.00% | 0.00% | 0.358 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `malayalam` | 100.00% | 0.00% | 0.316 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `gujarati` | 100.00% | 0.00% | 0.315 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `gurmukhi` | 100.00% | 0.00% | 0.311 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-335M-v1-mlx` | en | `odia` | 100.00% | 0.00% | 0.293 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-ClinicalBGE-Large-568M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.650 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.904 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.890 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.892 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.957 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.985 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.842 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 1.000 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 1.122 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-mSuperClinical-Large-279M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 1.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.662 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `han_traditional` | 0.00% | 0.00% | 0.767 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.521 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `bengali` | 0.00% | 0.00% | 0.671 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `tamil` | 0.00% | 0.00% | 0.784 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `telugu` | 0.00% | 0.00% | 0.757 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `kannada` | 0.00% | 0.00% | 0.761 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `malayalam` | 0.00% | 0.00% | 0.750 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.685 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `gurmukhi` | 0.00% | 0.00% | 0.743 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SnowflakeMed-Large-568M-v1-mlx` | en | `odia` | 0.00% | 0.00% | 0.573 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Base-184M-v1-mlx` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Large-434M-v1-mlx` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.825 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `han_traditional` | 3.03% | 0.00% | 0.904 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `devanagari` | 0.00% | 0.00% | 1.356 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `bengali` | 0.75% | 0.00% | 1.822 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `tamil` | 5.26% | 0.00% | 1.284 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `telugu` | 3.79% | 0.00% | 1.886 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `kannada` | 4.03% | 0.00% | 1.851 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `malayalam` | 3.97% | 0.00% | 1.658 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `gujarati` | 3.45% | 0.00% | 1.589 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `gurmukhi` | 6.14% | 0.00% | 1.541 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperClinical-Small-44M-v1-mlx` | en | `odia` | 4.27% | 0.00% | 1.560 | unclaimed | FLAG |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `han_simplified` | 0.00% | 84.55% | 1.538 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `han_traditional` | 0.00% | 87.30% | 1.726 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `devanagari` | 0.00% | 92.73% | 3.014 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `bengali` | 0.00% | 92.07% | 3.973 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `tamil` | 0.00% | 94.26% | 5.419 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `telugu` | 0.00% | 94.26% | 5.729 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `kannada` | 0.00% | 93.51% | 5.522 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `malayalam` | 0.00% | 93.89% | 5.171 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `gujarati` | 0.00% | 93.13% | 4.589 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `gurmukhi` | 0.00% | 97.65% | 2.878 | unclaimed |  |
| `OpenMed/OpenMed-PII-Vietnamese-SuperMedical-Large-355M-v1-mlx` | en | `odia` | 0.00% | 93.53% | 4.533 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-ai4privacy-multilingual` | de, en, es, fr, it, nl | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-mlx` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-mlx-8bit` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_simplified` | 0.00% | 0.00% | 0.613 | supported |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_traditional` | 0.00% | 13.11% | 0.836 | supported |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `devanagari` | 0.00% | 0.00% | 0.753 | supported |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `telugu` | 0.00% | 48.94% | 1.343 | supported |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_simplified` | 0.00% | 0.00% | 0.613 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_traditional` | 0.00% | 13.11% | 0.836 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `devanagari` | 0.00% | 0.00% | 0.753 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `telugu` | 0.00% | 48.94% | 1.343 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_simplified` | 0.00% | 0.00% | 0.613 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_traditional` | 0.00% | 13.11% | 0.836 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `devanagari` | 0.00% | 0.00% | 0.753 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `telugu` | 0.00% | 48.94% | 1.343 | supported |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-mlx-8bit` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_simplified` | 0.00% | 0.00% | 0.613 | supported |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `han_traditional` | 0.00% | 13.11% | 0.836 | supported |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `devanagari` | 0.00% | 0.00% | 0.753 | supported |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `telugu` | 0.00% | 48.94% | 1.343 | supported |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-multilingual-v2` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, sw, te, th, tr, xh, zu | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-mlx-8bit` | en | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-piimb-v2` | de, en, es, fr, it, nl, pt | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-nemotron-v2` | de, en, es, fr, it, nl, pt | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `han_simplified` | 0.00% | 0.00% | 0.613 | supported |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `han_traditional` | 0.00% | 13.11% | 0.836 | supported |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `devanagari` | 0.00% | 0.00% | 0.753 | supported |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `telugu` | 0.00% | 48.94% | 1.343 | supported |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-openmed-pii54-multilingual` | ar, de, en, es, fr, he, hi, id, it, ja, nl, pt, ro, te, th, tr, xh, zu | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `han_simplified` | 0.00% | 0.00% | 0.613 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `han_traditional` | 0.00% | 13.11% | 0.836 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `devanagari` | 0.00% | 0.00% | 0.753 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `bengali` | 0.00% | 5.80% | 0.945 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `tamil` | 0.00% | 46.81% | 1.270 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `telugu` | 0.00% | 48.94% | 1.343 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `kannada` | 0.00% | 37.04% | 1.209 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `malayalam` | 0.00% | 50.57% | 1.145 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `gujarati` | 0.00% | 0.00% | 0.808 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `gurmukhi` | 0.00% | 33.33% | 1.378 | unclaimed |  |
| `OpenMed/privacy-filter-piimb-fine-grained` | de, en, es, fr, it, nl, pt | `odia` | 0.00% | 46.81% | 1.880 | unclaimed |  |
