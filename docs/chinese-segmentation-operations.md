# Chinese segmentation operations

[Anonymization](anonymization.md#chinese-word-segmentation) covers the default
`jieba` path and the dictionary line format. This guide covers running the
optional `pkuseg` and HanLP backends against **locally provisioned** model
assets: where those assets live, how to select and pin them, how to govern
institutional dictionaries, and how to run segmentation regressions before and
after a change.

OpenMed never downloads optional segmentation model files on your behalf. Every
asset described here is provisioned by the operator.

## Backends and declared version ranges

| Backend | Install | License | Declared range |
| --- | --- | --- | --- |
| `jieba` | included in the base package | MIT | `jieba>=0.42.1,<0.43` |
| `pkuseg` | `pip install "openmed[zh-pkuseg]"` | MIT | `pkuseg>=0.0.25,<0.1` |
| `hanlp` | `pip install "openmed[zh-hanlp]"` | Apache-2.0 | `hanlp>=2.1,<3` |

`jieba` is a base dependency, not an extra, so the default backend is always
present. It is pinned twice: the base range above, and a wider `jieba>=0.42` in
the `openmed[zh]` extra, which also carries OpenCC and pypinyin. The base range is the
binding one for the segmenter, since installing `openmed[zh]` cannot loosen it.

The ranges above are the dependency constraints declared in `pyproject.toml`;
they are the supported envelope. OpenMed does not certify
individual upstream releases inside a range, and it does not pin or verify
model-asset versions at all — asset versioning is an operator responsibility
and belongs in your own deployment manifest.

Selecting an uninstalled backend raises `ImportError` naming both the required
extra and the backend license.

## Configuration surface

| Setting | Environment variable | Default |
| --- | --- | --- |
| `chinese_segmentation_backend` | `OPENMED_CHINESE_SEGMENTATION_BACKEND` | `jieba` |
| `chinese_user_dict_path` | `OPENMED_CHINESE_USER_DICT` | unset |
| `chinese_pkuseg_domain` | `OPENMED_CHINESE_PKUSEG_DOMAIN` | `medicine` |

The backend value is lower-cased and must be one of `jieba`, `pkuseg`, or
`hanlp`; anything else raises `ValueError` during configuration validation. An
empty `chinese_pkuseg_domain` is rejected the same way.

## pkuseg model provisioning and cache layout

`chinese_pkuseg_domain` accepts **either** an upstream domain name **or** a
filesystem path. OpenMed decides which by looking the value up in
`pkuseg.config.available_models`. The full rule is this table, which is
executed against the implementation by the documentation test:

<!-- pkuseg-resolution-rule -->

| `chinese_pkuseg_domain` | Known model name | Resolution |
| --- | --- | --- |
| `medicine` | yes | `pkuseg_home/medicine` |
| `default` | yes | verbatim |
| `/srv/openmed/models/pkuseg-medicine` | no | verbatim |

"`pkuseg_home/<name>`" means `Path(pkuseg.config.pkuseg_home).expanduser() /
<name>`, and that directory must already exist or a `FileNotFoundError` is
raised telling you to download the domain model explicitly. "verbatim" means
the value is handed to pkuseg unresolved and unchecked by OpenMed.

The literal `default` is the one known name that is still passed through, which
is why it appears in the table on its own row.

The practical consequence: the shipped default `medicine` must be pre-downloaded
into `pkuseg_home` before first use, while a model directory you provisioned
yourself is used as-is:

```python
from openmed.core.config import OpenMedConfig

# Resolved under pkuseg_home; the directory must already exist.
named = OpenMedConfig(
    chinese_segmentation_backend="pkuseg",
    chinese_pkuseg_domain="medicine",
)

# Passed through verbatim to pkuseg.
provisioned = OpenMedConfig(
    chinese_segmentation_backend="pkuseg",
    chinese_pkuseg_domain="/srv/openmed/models/pkuseg-medicine",
)
```

### What fails when

Dictionaries and model assets fail at different moments, and conflating them
misfiles real incidents:

| Input | Read at | Failure surfaces as |
| --- | --- | --- |
| bundled dictionary | construction | `DictionaryIngestionError` |
| `chinese_user_dict_path` | construction | `DictionaryIngestionError` |
| pkuseg domain or path | first `segment()` | `FileNotFoundError` |
| HanLP model path | first `segment()` | `FileNotFoundError` |

Constructing a segmenter **does** touch the filesystem: every backend loads the
bundled dictionary, and your `chinese_user_dict_path` if set, during
`__init__`. So a bad dictionary path fails fast, at construction, and you can
rely on that.

Model assets are the lazy half. Resolution and loading are deferred to the
first `segment()` call, so a misprovisioned model surfaces on first use rather
than at startup. That is the case worth an explicit provision-check in your
deployment, because construction succeeding tells you nothing about it.

## HanLP model provisioning

HanLP accepts a preloaded callable tokenizer or a local model path, supplied
through `create_chinese_segmenter(..., hanlp_model=...)`. There is no
configuration field or environment variable for it, because OpenMed does not
choose HanLP model weights.

```python
from openmed.processing import create_chinese_segmenter

segmenter = create_chinese_segmenter(
    "hanlp",
    hanlp_model="/srv/openmed/models/hanlp-tok",
)
```

- Supplying neither raises `ValueError`.
- A path that does not exist raises `FileNotFoundError`.
- The path is `expanduser()`-expanded, then handed to `hanlp.load()`.
- A callable is adopted immediately; a path is loaded lazily on first
  `segment()`.

## Deployment matrix

| Backend | Model assets | Provisioned by | Network at runtime |
| --- | --- | --- | --- |
| `jieba` | none beyond the package | the package | none |
| `pkuseg`, named domain | `pkuseg_home/<name>` | operator | none |
| `pkuseg`, path | the path you pass | operator | none |
| `hanlp`, path | the path you pass | operator | none |
| `hanlp`, preloaded callable | held by the caller | operator | none |

No backend causes OpenMed to fetch anything. Air-gapped deployments therefore
need only the wheel plus whatever model directory the chosen backend resolves
to.

## Upgrade and rollback

Model assets live entirely outside the installed package — in `pkuseg_home` or
in an operator-chosen directory — so upgrades and rollbacks are directory
swaps, not reinstalls.

A loaded model is cached on the segmenter instance after the first `segment()`
call. Replacing the files under a path that a live process already loaded does
**not** take effect in that process. Roll forward or back by swapping the
directory (or repointing the configuration at a new one) and then constructing
fresh segmenters — in practice, restarting the workers.

A safe sequence:

1. Record the current boundary-F1 baseline on your own held-out set (see
   [Segmentation regressions](#segmentation-regressions)).
2. Provision the new asset **beside** the current one, under a new directory.
3. Repoint `chinese_pkuseg_domain`, or the `hanlp_model` path, at the new
   directory.
4. Re-run the regression and compare against the recorded baseline.
5. To roll back, repoint at the previous directory and restart. Because the old
   directory was never mutated, rollback needs no re-download.

Upgrading the backend *library* is a separate action, bounded by the declared
ranges above, and should be re-validated the same way.

## Institutional dictionary governance

### Format and provenance

A dictionary is a UTF-8 text file, one entry per line, in `term`,
`term frequency`, or `term frequency POS` form. Single-member `.zip` archives
are also accepted; archive metadata is checked before any member is
decompressed.

Keep institutional dictionaries **outside** the package and point
`chinese_user_dict_path` / `OPENMED_CHINESE_USER_DICT` at the file. This keeps
private terminology and restricted lexicons out of the distributed artifact,
and it makes the dictionary a versioned deployment input you can attribute,
review, and roll back independently of the OpenMed release.

Record, in your own manifest: source of each term batch, licence or internal
approval covering it, the reviewer, and the date. OpenMed deliberately does not
log dictionary contents (see [Validation](#validation-and-limits)), so it
cannot reconstruct that provenance for you.

### Merge order and precedence

Every backend loads the bundled synthetic clinical dictionary first and then
your file, but they do not consume the merged result the same way. The full
rule is this table, which is executed against the implementation by the
documentation test:

<!-- dictionary-precedence-rule -->

| Backend | Duplicate term in your file | `frequency` and `POS` columns |
| --- | --- | --- |
| `jieba` | overrides | used |
| `pkuseg` | no effect | discarded |
| `hanlp` | no effect | discarded |

`jieba` applies your entries after the bundled file and keeps their frequency
and tag, so a repeated term replaces what the bundled dictionary said about it.
`pkuseg` and HanLP reduce the merged entries to de-duplicated bare term
strings, so a repeated term changes nothing.

Practically: tune frequencies to arbitrate between competing `jieba`
segmentations, and do not expect them to move a pkuseg or HanLP boundary.

### Validation and limits

Dictionaries supplied through `chinese_user_dict_path` are read through a
bounded streaming loader that stops as soon as a limit is crossed:

| Limit | Value |
| --- | --- |
| Max compressed source bytes (`.zip`) | 16 MiB |
| Max decompressed bytes | 64 MiB |
| Max entries | 100000 |
| Max records | 200000 |
| Max bytes per entry | 4096 |
| Max characters per term | 256 |
| Max archive expansion ratio | 100.0 |

Terms are always treated as literals: Unicode control characters and
executable regular-expression constructs are rejected. Failures raise a
subclass of `DictionaryIngestionError` — `DictionarySourceError`,
`DictionaryArchiveError`, `DictionarySizeLimitError`,
`DictionaryExpansionLimitError`, `DictionaryEntryLimitError`,
`DictionaryRecordLimitError`, `DictionaryEncodingError`, or
`DictionaryEntryValidationError` — so a CI gate can distinguish a malformed
entry from an oversized upload.

Rejection logs contain only a hash of the source path, the byte size, the entry
count, and a machine-readable reason. Dictionary content and raw paths are
never logged or included in raised exceptions, so validation failures are safe
to surface in shared CI output.

Validate a candidate dictionary before deploying it:

```python
from openmed.processing import DictionaryIngestionError, load_user_dictionary

try:
    entries = load_user_dictionary("/srv/openmed/zh_terms.txt")
except DictionaryIngestionError as error:
    raise SystemExit(f"dictionary rejected: {type(error).__name__}") from error

print(f"accepted {len(entries)} entries")
```

### How each backend applies dictionary terms

The three backends do not apply terms the same way, and the difference is
visible in output boundaries:

- **`jieba`** loads the bundled dictionary into a private `jieba.Tokenizer`,
  then adds each validated entry from your file with its frequency and POS tag.
  Terms influence jieba's own DAG scoring.
- **`pkuseg`** receives the merged term list natively as pkuseg's `user_dict`.
- **HanLP** receives no term list at all. OpenMed instead applies a
  deterministic longest-match overlay to HanLP's output: matches are collected
  longest-term-first, overlapping matches are discarded leftmost-first, and the
  surviving spans force boundaries at their edges while removing any interior
  boundary.

So a term can change HanLP's boundaries **after** the model has run, whereas
under `jieba` and `pkuseg` it participates in the model's own decision.
Validate dictionary changes against each backend you deploy; do not assume a
term batch that helped one will behave identically on another.

## Segmentation regressions

Every backend routes its output through the same validation before returning,
so a regression shows up as an exception or as a boundary-quality drop, never
as silently misaligned offsets.

`validate_segmentation()` enforces that tokens are `SpanToken` instances, are
ordered and non-overlapping, carry offsets within the text, have `text` exactly
equal to the source slice, and leave no non-whitespace code point uncovered.

`segmentation_boundary_f1()` scores a candidate against a reference on interior
character boundaries. The terminal document boundary is excluded, because every
valid segmentation shares it and including it would inflate short examples.

```python
from openmed.processing import segmentation_boundary_f1
from openmed.processing.tokenization import SpanToken

text = "患者张伟因高血压入院"
gold = [
    SpanToken("患者", 0, 2),
    SpanToken("张伟", 2, 4),
    SpanToken("因", 4, 5),
    SpanToken("高血压", 5, 8),
    SpanToken("入院", 8, 10),
]
predicted = [
    SpanToken("患者", 0, 2),
    SpanToken("张伟", 2, 4),
    SpanToken("因高血压", 4, 8),
    SpanToken("入院", 8, 10),
]

assert segmentation_boundary_f1(gold, gold) == 1.0
assert segmentation_boundary_f1(gold, predicted) < 1.0
```

The names and conditions in this guide are synthetic. Build your regression set
the same way — from synthetic or fully de-identified text — because a held-out
segmentation set is a durable artifact that outlives the change it was created
for.

The shipped gate scores the default backend over a synthetic probe set and
requires a **mean boundary F1 of at least 0.90**. Mirror that shape for your own
set: fix the reference cuts, record the mean, and fail the deployment when a
model or dictionary change moves it down.

Run the segmentation suite with:

```bash
python -m pytest tests/unit/processing/test_zh_segmentation.py
```

## End-to-end conformance against installed models

The conformance suite in
`tests/unit/processing/test_zh_segmentation_conformance.py` runs every backend
against the same corpus and the same checks. It is the executable form of this
guide: if a locally provisioned model is wired up correctly, the suite passes;
if it is absent, the suite skips with a reason naming what is missing.

### Provisioning the assets

The two optional backends are opted into differently, and the asymmetry is
deliberate:

| Backend | Opt-in | Why |
| --- | --- | --- |
| `pkuseg` | none — the suite probes `pkuseg_home/medicine` | the domain name is fixed, so the location is derivable |
| `hanlp` | `OPENMED_HANLP_MODEL_PATH` | OpenMed does not choose HanLP weights, so the path must be supplied |

Do not look for an `OPENMED_PKUSEG_*` counterpart to the HanLP variable; there
is none, because pkuseg's `medicine` model has a single canonical location and
HanLP's does not. `OPENMED_HANLP_MODEL_PATH` is read by the conformance suite,
not by `OpenMedConfig` — the library itself takes the path through
`create_chinese_segmenter(..., hanlp_model=...)`.

```bash
# pkuseg: provision the medicine domain model into pkuseg_home, then just run.
python -m pytest tests/unit/processing/test_zh_segmentation_conformance.py -rs

# HanLP: point the variable at a local model directory.
OPENMED_HANLP_MODEL_PATH=/srv/openmed/models/hanlp-tok \
  python -m pytest tests/unit/processing/test_zh_segmentation_conformance.py -rs
```

The live-model cases are marked `integration`. They are not deselected by
default, so they run whenever their assets are present and skip otherwise.

Each reason the suite can emit under `-rs` names one missing asset. On a
correct install the first row never appears — `jieba` is a base dependency, so
that reason means the installation itself is broken, not that an optional extra
is absent. On a correct install with neither optional extra provisioned you see
the last three:

<!-- conformance-skip-reasons -->

| Emitted when | Reason |
| --- | --- |
| the base install is broken | `requires the core jieba dependency to be installed` |
| `openmed[zh-pkuseg]` is absent | `requires the optional openmed[zh-pkuseg] extra (MIT)` |
| `openmed[zh-hanlp]` is absent | `requires the optional openmed[zh-hanlp] extra (Apache-2.0)` |
| fewer than two backends are usable | `requires at least two installed Chinese segmentation backends with their user-supplied domain models` |

Provisioning an extra without its model asset produces a further skip naming
the model rather than the package.

A skip is therefore an inventory report, not a silent pass: each line names the
one asset to provision next.

### The shared corpus

`tests/fixtures/processing/zh_segmentation_conformance.json` is a **generator
specification**, not a flat case list. It carries 40 synthetic names, 5
conditions, and 4 word templates; expanding names against conditions yields 200
cases, cycling through the templates. The fixture's `metadata.sha256` pins the
joined case texts, and the suite recomputes that digest, so an accidental edit
to the corpus fails rather than silently changing what conformance means.

All of it is synthetic and algorithmically generated — no real clinical text.

### Running the harness yourself

`run_segmenter_conformance()` is public, so you can point the same checks at
your own corpus in your own deployment pipeline:

```python
from openmed.processing import (
    SegmentationConformanceCase,
    create_chinese_segmenter,
    run_segmenter_conformance,
)

cases = [
    SegmentationConformanceCase(
        text="患者王芳因心房颤动入院",
        gold_words=("患者", "王芳", "因", "心房颤动", "入院"),
        required_terms=("王芳",),
    ),
]

segmenter = create_chinese_segmenter("pkuseg", pkuseg_domain="medicine")
report = run_segmenter_conformance(segmenter, cases, backend="pkuseg")

if not report.ok:
    for issue in report.issues:
        print(f"{issue.check}: {issue.detail}")
    raise SystemExit("segmentation conformance failed")

print(report.to_evidence())
```

`gold_words` must concatenate to `text`; a set of cuts that does not
reconstruct the case is rejected rather than scored. `required_terms` are the
dictionary terms the backend must keep intact as single tokens.

### What gates, and what is only recorded

This distinction matters for anyone wiring the harness into a release gate.

**`report.ok` is the gate.** It is true only when the backend produced no
defect. Every defect carries a `check` name drawn from
`SEGMENTATION_CONFORMANCE_CHECKS`:

| Check | Fails when the backend |
| --- | --- |
| `protocol` | raises, or returns anything other than `SpanToken` items |
| `alignment` | emits a word that cannot be aligned to the source |
| `offsets` | reports offsets that do not match the source slice |
| `overlap` | returns overlapping tokens |
| `ordering` | returns tokens out of order |
| `coverage` | drops a non-whitespace source span |
| `dictionary` | splits a required dictionary term |

`report.checks_triggered` gives the distinct check names that fired, so a
failing gate can say which class of defect appeared.

A backend whose `segment()` returns `None` is reported as a `protocol` defect
rather than crashing the harness. A generator is accepted as a legitimate lazy
tokenizer and scored on what it yields, so a generator that yields nothing is
reported as a `coverage` defect, not a protocol one.

**`boundary_f1`, `dictionary_hit_rate`, and `chars_per_second` are not part of
`report.ok`.** A report with poor metrics and no defect is `ok`; a report with
perfect metrics and one defect is not. They are recorded on the report and in
`to_evidence()` so a release log can carry them.

That is a statement about `report.ok`, not about what the shipped suite
asserts. The suite gates two of the three separately:

<!-- conformance-metric-gating -->

| Metric | Suite asserts | Role | Determinism |
| --- | --- | --- | --- |
| `boundary_f1` | `>= SEGMENTATION_BOUNDARY_F1_FLOOR` | quality gate | deterministic |
| `dictionary_hit_rate` | `== 1.0` | quality gate | deterministic |
| `chars_per_second` | `> 0.0` | liveness only | hardware-dependent |

`chars_per_second` is asserted only to be positive, which any working backend
satisfies. That proves the timer ran; it judges nothing about the number. There
is no throughput threshold you can fail, so do not read the row as a
performance standard.

#### One metric, two thresholds

`boundary_f1` is asserted twice, at different values, because it answers two
different questions:

| Subject | Threshold | Question it answers |
| --- | --- | --- |
| the suite's in-repo reference segmenter | exactly `1.0` | did the harness itself break? |
| an installed backend | `>= 0.90` | does this backend conform? |

The reference segmenter is a fixed implementation scored over a checksum-pinned
corpus, so its result is fully determined; anything below `1.0` means the
harness or the corpus changed, not that a backend is weak. Installed backends
legitimately vary by build and by user-supplied model, so they are held to the
floor instead. Only the second row is a bar your backend must clear.

The exact-`1.0` row derives its strength from that determinism, which makes it
specifically fragile to **Unicode folding**. The corpus contains full-width
characters, and the harness compares tokens against exact source offsets. If a
normalization step that folded width or applied NFKC were ever introduced
upstream of segmentation, full-width `ＣＴ` would fold to `CT`, offsets would
shift, and a genuine offsets defect could score as a pass. No such step exists
today — OpenMed's shared input gateway strips whitespace and enforces size and
encoding limits, applies no Unicode normalization, and sits on request-entry
surfaces that segmentation never routes through. Treat this as a constraint to
preserve when changing normalization, not as a present-day defect.

`SEGMENTATION_BOUNDARY_F1_FLOOR` is **0.90**, the same floor the shipped
default backend already meets. Import it from the package, alongside the other
conformance names:

```python
from openmed.processing import SEGMENTATION_BOUNDARY_F1_FLOOR
```

Gate on the constant rather than writing `0.90` into your own pipeline, so a
future change to the floor reaches you.

`boundary_f1` is gated precisely because it is **not** hardware-dependent:
scored over a sha256-pinned corpus with a pinned backend build it is fully
deterministic. `chars_per_second` is the only number that varies with the
machine, which is why it is recorded rather than asserted.

### What the floor does and does not catch

The conformance suite validates the span protocol and dictionary-term
survival, plus a 0.90 floor that rejects character-level output; it does not
validate general segmentation quality.

That boundary is deliberate and is pinned by a test rather than left implicit.
Two stubs make it concrete. Both satisfy every conformance check, so only the
floor separates them:

| Stub | `boundary_f1` | Outcome |
| --- | --- | --- |
| emits one character per token | 0.8068 | rejected by the floor |
| emits dictionary terms, merges every other run into one blob | 0.9722 | **passes** |

So a segmenter can clear the floor while merging all non-dictionary text. The
floor was not raised to close that gap, because ~0.98 would exceed the shipped
default backend's own gate and would likely reject legitimate user-supplied
HanLP models. Treat a passing conformance run as evidence that the protocol and
your dictionary terms hold — not as evidence of good segmentation. Keep your own
held-out quality set for that.
