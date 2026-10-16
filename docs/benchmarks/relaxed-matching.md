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
`8f3c05b2ff388eceb2c1a46bf80be710e540650d` returns 228 synthetic documents and
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
so helper-only changes invalidate cached reports. The merged #3523 read-race
repair is preserved unchanged. Existing cache entries may
become cold misses. Fixture hashing and unordered-value normalization are
unchanged; this does not include the separate pending #4207 correction.

## Measurements

Python 3.11.15, Linux/WSL2, identical frozen development dependencies and lock
SHA-256 `f30bf2213048a701562a6ebc9e462336d5144bfe2603cedeb6b13656d8612f47`.
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
| span | sparse | 200 × 1 | 0.035015 (0.032370–0.035890) → 0.000391 (0.000348–0.000391) | 20,088 → 22,616 | 169,250,816 → 168,525,824 |
| span | dense | 200 × 1 | 0.092437 (0.077867–0.108630) → 0.000550 (0.000368–0.000667) | 43,256 → 22,616 | 169,345,024 → 169,132,032 |
| span | sparse | 1000 × 1 | 0.773294 (0.772668–0.926713) → 0.002204 (0.002012–0.002252) | 78,828 → 125,104 | 169,500,672 → 169,308,160 |
| span | dense | 1000 × 1 | 1.841533 (1.821940–2.017140) → 0.002734 (0.002553–0.002801) | 222,104 → 125,104 | 169,410,560 → 169,156,608 |
| span | chain | 1000 × 1 | 0.925821 (0.861992–1.022067) → 0.922775 (0.817853–0.962739) | 78,884 → 235,404 | 169,476,096 → 169,701,376 |
| span | blocked | 1000 × 1 | 0.502140 (0.466248–0.524758) → 0.078477 (0.076596–0.102928) | 129,388 → 193,488 | 169,308,160 → 169,504,768 |
| span | permuted | 1000 × 1 | 1.050519 (0.855075–1.094423) → 0.427318 (0.401180–0.436725) | 78,828 → 125,104 | 168,996,864 → 169,648,128 |
| relation | sparse | 1000 × 1 | 0.249806 (0.236083–0.289499) → 0.001080 (0.000965–0.001231) | 72,776 → 119,216 | 169,951,232 → 169,910,272 |
| relation | dense | 1000 × 1 | 0.873970 (0.849502–0.903666) → 0.000937 (0.000898–0.000937) | 216,144 → 119,248 | 169,996,288 → 169,975,808 |
| relation | chain | 1000 × 1 | 0.231480 (0.202086–0.273632) → 0.225624 (0.208760–0.236928) | 72,924 → 230,092 | 170,131,456 → 170,471,424 |
| relation | blocked | 1000 × 1 | 0.302654 (0.297631–0.439476) → 0.077612 (0.077432–0.090602) | 124,020 → 188,176 | 170,278,912 → 170,323,968 |
| multilingual | sparse | 1000 × 5 | 18.662433 (18.456978–20.244970) → 0.362772 (0.360572–0.429356) | 2,969,585 → 2,963,513 | 176,263,168 → 177,340,416 |
| multilingual | dense | 1000 × 5 | 22.139593 (22.055445–22.708020) → 0.370672 (0.367828–0.461319) | 3,028,701 → 2,907,769 | 176,369,664 → 176,939,008 |
| multilingual | sparse | 100 × 50 | 17.363954 (17.224625–17.672591) → 0.400669 (0.386818–0.425698) | 2,980,050 → 2,973,978 | 175,513,600 → 176,562,176 |
| multilingual | dense | 100 × 50 | 18.855012 (18.564389–18.949552) → 0.363605 (0.358451–0.383115) | 2,996,022 → 2,975,834 | 175,628,288 → 175,443,968 |
| relation | sparse | 1000 × 5 | 1.787487 (1.784171–1.792084) → 0.008858 (0.008726–0.011298) | 870,680 → 615,856 | 173,228,032 → 173,121,536 |
| relation | dense | 1000 × 5 | 4.657447 (4.591240–4.940590) → 0.008033 (0.007994–0.008080) | 985,604 → 615,888 | 173,252,608 → 173,060,096 |

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
| `cache.py` | `a00aaa0b908aef0df080724179f1c6470f5c1352b413aae233e68ee32a7b3302` |
| `metrics.py` | `01d5e1a5e99556654494930bf0e1eabe8bcfb65e5aff3a373682491018a8feac` |
| `relation_metrics.py` | `fd7805f3de67ddc04d653a220b95655a7275db13a3cd280642f9642e72580b8e` |
