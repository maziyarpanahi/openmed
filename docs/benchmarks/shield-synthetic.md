# Synthetic SHIELD-Schema Baseline

This is a local rules control on two OpenMed-generated synthetic notes
using SHIELD's nine label names. It does not use the SHIELD public sample
or restricted corpus and does not measure a clinical model.

| Measure | Result |
|---|---:|
| Exact span F1 | 71.43% |
| Exact span recall | 55.56% |
| Character leakage | 59.91% |
| Synthetic notes | 2 |

## By Canonical Label

| Label | Recall | Leakage |
|---|---:|---:|
| `AGE` | 100.00% | 0.00% |
| `DATE` | 100.00% | 0.00% |
| `ID_NUM` | 100.00% | 0.00% |
| `LOCATION` | 0.00% | 100.00% |
| `ORGANIZATION` | 0.00% | 100.00% |
| `PERSON` | 0.00% | 100.00% |
| `PHONE` | 100.00% | 0.00% |
| `URL` | 100.00% | 0.00% |

## Evidence and Reproduction

- [Committed result JSON](shield-synthetic.report.json)
- [Committed synthetic fixture](https://github.com/maziyarpanahi/openmed/blob/master/openmed/eval/fixtures/shield_synthetic_baseline.json)
- Fixture rights: OpenMed-generated synthetic fixture; Apache-2.0
- Fixture SHA-256: `sha256:3b94709625517cfb4ed583646809607ef2d108568f56dd22989505514983d096`
- Rules model: `openmed-synthetic-regex-phi-baseline-v1`; revision `v1` on `cpu`
- Configuration revision: `v1`
- Rules script SHA-256: `sha256:937436756933dce6a5dd6bef9d3b2151f4cd6752f2eaa5dabac148a016d98f48`
- Source base commit: `2f6090a74673a45346ef3ee15c66fb8de2c1b60b`
- Reproducibility hash: `sha256:d166ab379e01e399e4fe849ca006805b2bf7010e733035dfb546a50bbe9eb558`
- Report timestamp: `2026-09-27T21:30:04.742355+00:00`

Recompute the report with:

```bash
python -m scripts.status.generate_shield_synthetic_baseline \
  --source-revision 2f6090a74673a45346ef3ee15c66fb8de2c1b60b
```

Limitations: Two OpenMed-generated synthetic notes using SHIELD label names; no SHIELD public-sample or restricted records were used. This is a rules smoke baseline, not clinical model performance or a high-recall release gate.
