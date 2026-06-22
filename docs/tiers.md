# Device Tiers and SLOs

The device tier is the contract a checkpoint promises a device. Fitting the
declared tier is a hard gate for release checks, while model quality gates are
measured separately.

Param ranges map existing OpenMed size labels and are illustrative pending
manifest generation. The gate reads the manifest's real `param_count`, not the
tier word or the exact M-counts below.

| Tier | Param range (illustrative) | Existing sizes | Reference device | Target RAM (resident) | Target latency (p50 / p95, ~1 page) | Default format |
|------|----------------------------|----------------|------------------|----------------------|--------------------------------------|----------------|
| **Tiny** *(default mobile; incl. distilled Nano floor)* | ~10–135M (Nano 10–30M distilled; Small-44M, LiteClinical-66M) | Small/Lite | Phone / tablet (Apple Silicon), embedded | ≤ 350 MB (Nano ≤ 150 MB) | ≤ 60 ms / ≤ 150 ms (Nano ≤ 25 / ≤ 60 ms) | INT8 (MLX-8bit / CoreML) |
| **Base** *(laptop default)* | ~135–280M (Base-125/135/184M, TinyMed-135M) | Base | Laptop CPU, modest GPU | ≤ 900 MB | ≤ 150 ms / ≤ 400 ms | INT8 (FP fallback) |
| **Large** *(workstation)* | ~430–570M (SuperClinical-434M, ~568M) | Large | Workstation GPU | ≤ 4 GB | ≤ 250 ms / ≤ 800 ms | FP16 (INT8 if recall holds) |
| **Accurate / XLarge** *(server)* | ~600M+ / MoE (incl. ~1.4B privacy-filter) | XLarge / MoE | Server GPU | ≤ 8 GB (MoE) | ≤ 400 ms / ≤ 1200 ms | FP16 (INT8 if recall holds) |

## Machine-Readable Budgets

`openmed.eval.tiers.TIERS` is the machine-readable SLO source for gate
harnesses. It exposes four named rows: `Tiny`, `Base`, `Large`, and
`Accurate-XLarge`. Each row carries:

| Field | Meaning |
|---|---|
| `ram_mb_max` | Maximum resident RAM budget for the tier. |
| `p50_ms_max` | Maximum p50 latency budget for approximately one page. |
| `p95_ms_max` | Maximum p95 latency budget for approximately one page. |
| `default_format` | Default release artifact format for the tier. |

The `Large` and `Accurate-XLarge` RAM values are represented as 4096 MB and
8192 MB in code.

## Notes

- **Tier ≠ tier word across families** ("Large" = 434M for token-class NER but
  ~459–568M elsewhere; "Base" = 184M or 220M depending on family). The manifest
  carries the real `param_count`; tier words on cards are descriptive only.
- **Nano** is a **distillation target** folded into the Tiny tier, not a
  separate scheme.
- **Default selection** off the manifest's `tier`: Tiny on mobile, Base on
  laptop, Large/Accurate only when the caller opts in or recall demands it.
- The latency/RAM budgets above are the **single per-tier SLO source**; the
  §3 engine and §7 gates reference this table, not separate numbers.
