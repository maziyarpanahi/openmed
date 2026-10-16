# Relaxed span and relation matching review

This guide accompanies the focused correction for [issue #4208](https://github.com/maziyarpanahi/openmed/issues/4208).
The maintainer accepted the scope for v3.2; the matching semantics and resource
tradeoffs remain subject to implementation review. No evaluation gold values,
fairness/release thresholds, or stored baselines are changed.

## Proposed scoring semantics

Count a maximum number of eligible one-to-one gold/prediction pairs. Each gold
and prediction can contribute at most one true positive. False positives and
false negatives still subtract that count from the unchanged input counts.

For relaxed spans, eligibility remains positive character overlap and equal
canonical labels. Relaxed relations retain the existing predicate: equal fixture
IDs and relation types, with positive overlap of both directed endpoints. There
is no new endpoint-label condition. Input normalization and fixture-coordinate
shifting remain unchanged. Strict span scoring and the strict relation matcher
retain their existing implementations.

The former greedy scorer preferred the largest overlap for each gold record.
That local choice can consume the only prediction eligible for a later gold
record. The proposed objective maximizes the true-positive count; it does not
maximize total overlap length or expose selected pairs. Permuting inputs can
change an internal pairing but cannot change the maximum count. For the chain
benchmark below, the old scorer counts 999 pairs and this candidate counts 1,000.
That is an intentional scoring correction requiring maintainer review.

The implementation greedily seeds a matching and then performs iterative,
multi-source augmenting-path searches over an implicit graph. A failed search
proves there is no augmenting path. Starting all unmatched roots together avoids
repeatedly scanning the same blocked component. Exhaustive 3-by-3 graphs, seeded
rectangular graphs, a 1,500-vertex alternating path, public permutation tests,
strict controls, and a real multilingual-group call exercise these properties.

## Input sizes, memory and runtime limits

The public scorers accept arbitrary iterables and currently document no count
cap. Fairness and transfer callers often score one fixture. The multilingual
harness calls `_corpus_coordinates` and scores an entire benchmark/language
group, so its aggregate size must be considered separately. That harness still
shifts documents into disjoint coordinates; this correction does not change it.

The native bundled golden loader at base
`0386c53a27951ac51f7a94b3876c69e8e7f84393` returns 228 synthetic documents and
825 gold spans: median 3 and maximum 18 spans per document. Its 66
benchmark/language groups have at most 65 gold spans. These are repository test
sizes, not a deployment distribution or a bound on prediction counts. Real
deployment input-size expectations have not been supplied.

The benchmark covers 200 and 1,000 records per side per document, plus two
5,000-per-side aggregate shapes: 5 documents of 1,000 records and 50 documents
of 100 records. Sparse, dense, permuted, alternating-chain and blocked-component
inputs are included. Aggregate span measurements call the actual multilingual
group implementation, including exact scoring and coordinate construction;
aggregate relation measurements retain separate fixture IDs.

For L gold and R predictions, the matcher uses O(L + R) auxiliary space: two
match arrays, unmatched prediction keys, a queue, visited vertices and parent
links. It never materializes candidate edges. Normalized inputs and the
multilingual harness retain their existing allocations. The earlier candidate's
approximately 29.8 MB dense 1,000-by-1,000 edge-storage probe motivated this
revision. In the current paired protocol, dense span scoring allocates 222,104
peak traced bytes on upstream and 125,104 on this candidate. Sparse and chain
cases incur a larger linear bookkeeping cost, reported below.

Worst-case time is O(L * R * min(L, R)) for repeated augmentations. A greedy
maximum or one final blocked search takes O(L * R), and favorable input order
can stop far earlier. There is no universal speedup claim. No evaluation input
cap, timeout, approximate result, or silent fallback is introduced. Workloads
beyond the measured envelope can still be expensive; deployment limits and the
scoring objective require maintainer review. The benchmark runner itself limits
each synthetic invocation to 10,000 records per side.

The eval cache source hash now includes the relation scorer and shared matcher,
so helper-only changes invalidate cached reports. Existing cache entries may
become cold misses. Fixture hashing and unordered-value normalization are
unchanged; this does not include the separate pending #4207 correction.

## Measurements

Python 3.11.15, Linux/WSL2, identical frozen development dependencies and lock
SHA-256 `0539b1737218cf6c0c5a93c3765c8490314f20171390d3cc208496748cb1a446`.
Each paired baseline/candidate subprocess uses one warm-up, three untraced
timing samples, and a separate traced allocation sample. Imports and input
construction are outside the timing and tracing interval. Process peak RSS
includes both. Independent test jobs were active on this shared host; timing
medians and ranges are diagnostic rather than isolated-host or production
guarantees. The raw command/result receipts are retained with the contribution
validation records.

Times below are seconds (median, with minimum–maximum in parentheses). Memory
values are bytes, shown as upstream → candidate. A skipped test is never counted
as a passing benchmark or validation result.

| Workload | Shape | Records/document × documents | Time, upstream → candidate | Peak traced bytes, upstream → candidate | Peak process RSS, upstream → candidate |
|---|---|---:|---|---:|---:|
| span | sparse | 200 × 1 | 0.028885 (0.028183–0.047337) → 0.000383 (0.000376–0.000393) | 20,088 → 22,616 | 168,992,768 → 169,185,280 |
| span | dense | 200 × 1 | 0.057342 (0.057047–0.058299) → 0.000352 (0.000351–0.000381) | 43,256 → 22,616 | 169,451,520 → 169,021,440 |
| span | sparse | 1000 × 1 | 0.701641 (0.667395–0.710863) → 0.001805 (0.001795–0.001875) | 78,828 → 125,104 | 169,267,200 → 169,525,248 |
| span | dense | 1000 × 1 | 1.577826 (1.555138–1.590515) → 0.001849 (0.001845–0.002557) | 222,104 → 125,104 | 169,394,176 → 169,353,216 |
| span | chain | 1000 × 1 | 0.780848 (0.689506–0.855842) → 0.750533 (0.725936–0.820834) | 78,884 → 235,404 | 169,553,920 → 169,697,280 |
| span | blocked | 1000 × 1 | 0.502751 (0.409315–0.514721) → 0.085702 (0.081407–0.099217) | 129,388 → 193,488 | 169,680,896 → 169,553,920 |
| span | permuted | 1000 × 1 | 0.775137 (0.762641–0.825117) → 0.449086 (0.405246–0.453378) | 78,828 → 125,104 | 169,299,968 → 169,570,304 |
| relation | sparse | 1000 × 1 | 0.158931 (0.153885–0.170915) → 0.000945 (0.000918–0.000962) | 72,776 → 119,216 | 170,045,440 → 169,951,232 |
| relation | dense | 1000 × 1 | 0.767036 (0.708551–0.804249) → 0.001004 (0.000917–0.001092) | 216,144 → 119,248 | 170,082,304 → 169,988,096 |
| relation | chain | 1000 × 1 | 0.171843 (0.161199–0.209030) → 0.174946 (0.171481–0.175432) | 72,924 → 230,092 | 170,504,192 → 170,668,032 |
| relation | blocked | 1000 × 1 | 0.262798 (0.255434–0.289745) → 0.081135 (0.079821–0.099596) | 124,020 → 188,176 | 170,196,992 → 170,147,840 |
| multilingual | sparse | 1000 × 5 | 19.904838 (19.678082–20.151968) → 0.462068 (0.460220–0.472306) | 2,969,585 → 2,963,513 | 172,679,168 → 176,140,288 |
| multilingual | dense | 1000 × 5 | 24.225322 (22.917504–24.334831) → 0.361515 (0.361300–0.369339) | 3,028,701 → 2,907,769 | 176,115,712 → 176,017,408 |
| multilingual | sparse | 100 × 50 | 16.859439 (16.846317–17.134350) → 0.381095 (0.362564–0.390504) | 2,980,050 → 2,973,978 | 175,456,256 → 176,566,272 |
| multilingual | dense | 100 × 50 | 21.201143 (20.226731–21.287516) → 0.371321 (0.358241–0.393359) | 2,996,022 → 2,975,834 | 175,034,368 → 175,616,000 |
| relation | sparse | 1000 × 5 | 1.756148 (1.699433–1.773008) → 0.008854 (0.008382–0.014190) | 870,680 → 615,856 | 173,060,096 → 173,035,520 |
| relation | dense | 1000 × 5 | 4.761298 (4.653536–4.761835) → 0.008653 (0.008517–0.008866) | 985,604 → 615,888 | 173,068,288 → 172,965,888 |

Every sparse, dense, permuted and blocked pair has identical baseline/candidate
counts. The two chain cases intentionally change from 999 to 1,000 true
positives, with unchanged input counts. Aggregate cases retain 5,000 true
positives on each implementation.

## Reproduce

Use the same script from this candidate with each checkout's frozen interpreter.
For example, run these from both the unpatched base and candidate roots:

```bash
uv sync --frozen --extra dev --python 3.11
uv run --frozen --extra dev python /path/to/candidate/scripts/benchmarks/relaxed_matching.py \
  --workload span --shape dense --size 1000 --repeat 3
uv run --frozen --extra dev python /path/to/candidate/scripts/benchmarks/relaxed_matching.py \
  --workload relation --shape chain --size 1000 --repeat 3
uv run --frozen --extra dev python /path/to/candidate/scripts/benchmarks/relaxed_matching.py \
  --workload multilingual --shape dense --size 1000 --documents 5 --repeat 3
```

Supported shapes are `sparse`, `dense`, `permuted`, `chain`, and `blocked`;
workloads are `span`, `relation`, and `multilingual`. The JSON result includes
all timing samples, counts, memory metrics, Python/platform, lock digest and
production source digests. The aggregate relation workload also accepts
`--documents 5`. Peak process RSS is available on Linux/macOS; traced allocations
and runtime samples remain available on other platforms.

## Production source measured

These production files were unchanged throughout all 17 paired measurements.
The final source checks verify their digests before committing the candidate.

| File under `openmed/eval` | SHA-256 |
|---|---|
| `_matching.py` | `50398cbef84c2fc42b4a2d1180249e3f95f815c63cab5d71b3fd14c95b5b85f8` |
| `cache.py` | `329a365050c5aaac1fa89454cda5d9be5e608df6213d881583216ddea625eef6` |
| `metrics.py` | `01d5e1a5e99556654494930bf0e1eabe8bcfb65e5aff3a373682491018a8feac` |
| `relation_metrics.py` | `fd7805f3de67ddc04d653a220b95655a7275db13a3cd280642f9642e72580b8e` |
