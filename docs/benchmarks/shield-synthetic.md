# Synthetic SHIELD-Schema Baseline

This is a local rules control on 2 OpenMed-generated synthetic notes
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
- Rules script SHA-256: `sha256:eb20c575e3a315b6e0fcd5aa7b4714243bee68a1f9427c949f5aa6db46766c88`
- Text digests normalize UTF-8 checkout newlines to LF (`utf8-lf-v1`).
- Source base commit: `5132cb95532476e6698b6b440751e8beb553e394`
- Reproducibility hash: `sha256:2cc88b7070ede609b18fe206a6f3fae95a0fbb0013f84c51962aaa3f06afeae9`
- Report timestamp: `2026-09-28T11:30:10.614226+00:00`

Recompute the report with:

```bash
python -m scripts.status.generate_shield_synthetic_baseline \
  --source-revision 5132cb95532476e6698b6b440751e8beb553e394
```

Limitations: Two OpenMed-generated synthetic notes using SHIELD label names; no SHIELD public-sample or restricted records were used. This is a rules smoke baseline, not clinical model performance or a high-recall release gate.
