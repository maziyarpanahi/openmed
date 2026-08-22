# Cost versus cloud benchmark

OpenMed can turn a measured local `PerfReport` into a dated cost comparison
against published paid cloud tiers. The report is deterministic, contains no
source text, and keeps every cloud row linked to the price captured for it.
It is planning evidence, not a vendor quote or a guarantee of savings.

## What the report computes

The input performance report must provide `docs_per_second` and
`chars_per_document` (either at the top level or in `metadata`). The hardware
model supplies:

```json
{
  "purchase_price_usd": 1200.0,
  "useful_life_hours": 12000.0,
  "power_watts": 45.0,
  "electricity_usd_per_kwh": 0.2
}
```

For each paid cloud tier, the benchmark reports:

- cloud USD per million characters;
- local USD per million characters, amortizing purchase price over useful
  life and adding electricity;
- savings per million characters; and
- the character volume at which purchase price is recovered versus the
  cloud tier after marginal electricity cost, or `never` when that crossover
  would fall beyond the hardware's declared useful life.

The formulas use measured characters per second. Free tiers, taxes, support
plans, negotiated discounts, storage, data transfer, operator time, request
rounding, and financing are deliberately excluded. Those exclusions make the
result reproducible but require human review before a business decision.

## Run the comparison

Store the hardware model in the performance report as
`hardware_cost_model`, or pass it separately with `--hardware`. Then run:

```bash
openmed benchmark cost \
  --perf reports/mobile/perf.json \
  --prices openmed/eval/data/cloud_prices.json \
  --output-dir reports/cost
```

The command writes `cost-vs-cloud.json` and `cost-vs-cloud.md` atomically. Every
normalized semantic input contributes to the report's SHA-256 fingerprint;
unknown fields and raw text do not. Local model paths are replaced by stable
digests. Input files, price rows, strings, numeric ranges, and report sizes are
bounded before they are materialized. A price row is rejected when it lacks a
safe HTTPS source, capture date, positive normalized price, or explicit
`verify: true` marker, or when its monthly band overlaps another row for the
same provider, service, and region.

Paid rows represent marginal monthly bands. The reported cloud USD per million
characters and breakeven value are sensitivity calculations at one row's
marginal rate; they do not integrate a progressive monthly invoice across all
preceding bands. Monthly tier boundaries and lifetime breakeven volume are
therefore shown separately.

## Committed price snapshot

The bundled USD table was captured on 2026-08-21:

| Provider | Meter | Published unit | Normalized paid tiers | Source |
|---|---|---|---|---|
| AWS | Amazon Comprehend Medical NERe | 100 characters, 100-character request minimum | $0.10, $0.05, and $0.01 per 1,000 characters | [AWS pricing](https://aws.amazon.com/comprehend/medical/pricing/) |
| Azure | Azure Language Text Analytics for health, East US | 1,000 text records; each record covers up to 1,000 characters | $0.020, $0.015, $0.006, and $0.005 per 1,000 characters | [Azure Retail Prices API](https://prices.azure.com/api/retail/prices?currencyCode=%27USD%27&%24filter=meterId%20eq%20%27379462cb-a6d6-5bda-b9cb-67f00929c795%27%20and%20armRegionName%20eq%20%27eastus%27) |

AWS's page states that requests use 100-character units and its worked
production example publishes the three NERe prices. Microsoft's retail meter
publishes `$20`, `$15`, `$6`, and `$5` per 1,000 health text records across
the paid volume bands; the normalized values divide by the 1,000 characters
covered by each record. Microsoft's public pricing page also directs customers
above 10 million monthly records to contact sales, so the final retail-API row
must not be treated as a negotiated quote.

Re-capture and review the table before a release, pricing statement, or
hardware purchase. Set `verify` to true only after confirming the source,
unit, region, tier boundary, and effective date.
