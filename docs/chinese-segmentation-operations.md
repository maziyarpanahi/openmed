# Chinese segmentation model operations

The default `jieba` backend and OpenMed's small synthetic clinical dictionary
ship with the package. The optional `pkuseg` and HanLP backends need model
assets that you provision: OpenMed selects, resolves, and loads them, but it
never downloads, mirrors, or updates them on your behalf.

This guide covers version support, cache layout, deployment, upgrade and
rollback, institutional dictionary governance, and the regression check to run
before promoting a model or a dictionary. For the segmentation API and the
dictionary file format, see
[PII anonymization](anonymization.md#chinese-word-segmentation).

## Supported versions and deployment requirements

| Backend | Install | Package range | License | Model asset |
|---|---|---|---|---|
| `jieba` | core dependency | `jieba>=0.42.1,<0.43` | MIT | Bundled prefix dictionary |
| `pkuseg` | `pip install "openmed[zh-pkuseg]"` | `pkuseg>=0.0.25,<0.1` | MIT | Domain model directory you provision |
| `hanlp` | `pip install "openmed[zh-hanlp]"` | `hanlp>=2.1,<3` | Apache-2.0 | Tokenizer directory you provision |

Plan for these runtime requirements:

- `jieba` builds a prefix-dictionary cache in the system temporary directory on
  first use. Keep that directory writable in read-only containers, or accept the
  rebuild cost on every start.
- `pkuseg` 0.0.25 publishes wheels only for CPython 3.5 to 3.8, so on the Python
  versions OpenMed supports it is built from the source distribution. Provide a
  C++ toolchain with NumPy and Cython at build time, or build the wheel once and
  install that artifact in the runtime image.
- HanLP installs a PyTorch stack (`torch`, `transformers`, `sentencepiece`), so
  size the image and the memory budget for it. A component can depend on further
  resources that are loaded recursively, so stage them together with the model.

Backend selection and dictionary paths come from the standard configuration:

| Setting | Environment variable | Applies to |
|---|---|---|
| `chinese_segmentation_backend` | `OPENMED_CHINESE_SEGMENTATION_BACKEND` | all backends |
| `chinese_pkuseg_domain` | `OPENMED_CHINESE_PKUSEG_DOMAIN` | `pkuseg` |
| `chinese_user_dict_path` | `OPENMED_CHINESE_USER_DICT` | all backends |

OpenMed reads no environment variable for the HanLP model location. Pass the
local path or a preloaded tokenizer through `hanlp_model`.

## Cache and storage layout

| Backend | Default root | Override | Contents |
|---|---|---|---|
| `pkuseg` | `~/.pkuseg` | `PKUSEG_HOME` | One directory per domain model |
| HanLP | `~/.hanlp`, or `%APPDATA%\hanlp` on Windows | `HANLP_HOME` | One directory per component, each holding `meta.json` |
| `jieba` | system temporary directory | — | Prefix-dictionary cache |

Keep provisioned models outside the OpenMed installation so an SDK upgrade never
rewrites them, and mount them read-only in production. Name each directory after
the exact upstream release, for example `medicine-v0.0.16`, so the running
configuration names the artifact it loaded.

## Provision a pkuseg domain model

pkuseg records its domain archives and their SHA-256 digests in
`pkuseg.config.model_urls` and `pkuseg.config.model_hash`. Fetch and verify the
archive on a connected staging host, then copy the extracted directory to the
target host:

```bash
python -c "import pkuseg; print(pkuseg.config.model_hash['medicine'])"
curl -fLO https://github.com/lancopku/pkuseg-python/releases/download/v0.0.16/medicine.zip
sha256sum medicine.zip
unzip -q medicine.zip -d /srv/openmed/zh/pkuseg
mv /srv/openmed/zh/pkuseg/medicine /srv/openmed/zh/pkuseg/medicine-v0.0.16
```

Point the configuration at the absolute directory rather than at the bare domain
name, so the loaded artifact is explicit and a rollback is a configuration
change:

```bash
export OPENMED_CHINESE_SEGMENTATION_BACKEND=pkuseg
export OPENMED_CHINESE_PKUSEG_DOMAIN=/srv/openmed/zh/pkuseg/medicine-v0.0.16
export OPENMED_CHINESE_USER_DICT=/srv/openmed/zh/dict/institution.txt
```

```python
from openmed.core.config import OpenMedConfig
from openmed.processing import create_chinese_segmenter_from_config

config = OpenMedConfig()
segmenter = create_chinese_segmenter_from_config(config)

note = "患者王芳因心房颤动入院"
tokens = segmenter.segment(note)

assert all(token.text == note[token.start : token.end] for token in tokens)
print([token.text for token in tokens])
```

A bare domain name such as `medicine` is resolved against
`pkuseg.config.pkuseg_home` and fails closed with the expected directory when
the model is not installed there; pkuseg is never allowed to download it.
Because the resolved directory reaches pkuseg as a user-supplied model, the
archive's own `medicine_dict.pkl` post-processing dictionary is not applied.
Institutional terms belong in the OpenMed user dictionary, which every backend
receives.

## Provision a HanLP tokenizer

HanLP extracts each component under `HANLP_HOME`. Load the tokenizer once on a
connected staging host, then copy the component directory, the one that contains
`meta.json`, to the target host:

```bash
export HANLP_HOME=/srv/openmed/zh/hanlp
python -c "import hanlp; hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)"
```

```python
import os

from openmed.core.config import OpenMedConfig
from openmed.processing import create_chinese_segmenter_from_config

config = OpenMedConfig()
segmenter = create_chinese_segmenter_from_config(
    config,
    hanlp_model=os.environ["OPENMED_HANLP_MODEL_DIR"],
)

note = "患者王芳因心房颤动入院"
tokens = segmenter.segment(note)

assert all(token.text == note[token.start : token.end] for token in tokens)
print([token.text for token in tokens])
```

A missing path raises `FileNotFoundError` before anything is loaded. Pass a
preloaded callable instead of a path when you need specific devices or a
multi-task model; fine-grained (`tok/fine`), coarse (`tok/coarse`), and plain
`tok` outputs are all accepted. Dictionary terms are applied to HanLP output as
a deterministic longest-match overlay.

## Deployment matrix

| Environment | Provisioning | Runtime |
|---|---|---|
| Workstation | Fetch and verify once into `~/.pkuseg` or `$HANLP_HOME` | Optional extra installed in the development environment |
| Container image | Bake the verified directory into a read-only layer, or mount it as a volume, and set the backend variables in the image | No egress; the model path is stable across releases |
| Air-gapped site | Transfer the verified archive on approved media and verify the digest again after transfer, as in [offline and air-gapped installation](offline-install.md) | No egress; promotion is a configuration change |
| CI | Keep the optional extras out of the default job; mount the models only in the job that runs the regression check | Regression scores gate the promotion |

## Upgrade and rollback

1. Provision the candidate beside the current model, for example
   `medicine-v0.0.17` next to `medicine-v0.0.16`.
2. Run the [regression check](#run-a-segmentation-regression-check) against the
   candidate directory with the dictionary that is in production.
3. Promote by repointing `OPENMED_CHINESE_PKUSEG_DOMAIN`, or the `hanlp_model`
   path, and restarting the workers. Model files are loaded lazily on the first
   `segment()` call, so a running process keeps its loaded model until it is
   restarted.
4. Roll back by repointing the same setting at the previous directory and
   restarting. Keep the previous directory until the candidate has completed a
   full reporting cycle.

Record for every promotion: the model directory name, the upstream archive URL
and digest, the dictionary revision, the regression scores, and the approver.

## Merge institutional dictionaries

Institutional terminology stays outside the package. Keep one reviewed source
file per contributing service, merge them into the file that
`OPENMED_CHINESE_USER_DICT` points at, and store a provenance record beside each
source: owner, license, source system and export date, review date, approver,
and revision. Dictionaries hold terminology only; never add patient names,
identifiers, or any other PHI.

Merge and validate as follows:

1. Normalize every source to `term`, `term frequency`, or `term frequency POS`
   lines. A `#` starts a comment.
2. Deduplicate terms across sources and keep the frequency of the owning
   service.
3. Validate the merged file before promoting it. `load_user_dictionary` is
   fail-closed: it raises a `DictionaryIngestionError` subclass carrying a rule
   name and a line number, and never the rejected entry text.

```python
from openmed.processing import load_user_dictionary

entries = load_user_dictionary("/srv/openmed/zh/dict/institution.txt")
print(len(entries))
```

Validation is bounded by defaults that callers may lower but never raise: 16 MiB
compressed and 64 MiB decompressed bytes, 100,000 entries, 200,000 records,
4 KiB per line, 256 characters per term, and an archive expansion ratio of 100.
Entries must be strict UTF-8 literals. Control characters, regular-expression
constructs, non-integer or non-positive frequencies, and POS tags longer than 32
characters or outside `A-Za-z0-9_-` are rejected. The same rules back the
mitigations in the
[multilingual ingestion threat model](security/threat-model-multilingual-ingestion.md).

Backends consume the merged file differently: `jieba` uses the term, frequency,
and POS fields, `pkuseg` forces a cut at every term, and HanLP applies the terms
as a longest-match overlay. Frequency and POS therefore change `jieba` output
only.

## Run a segmentation regression check

Keep a reviewed gold set of synthetic sentences that covers your specialties,
one JSON record per line with the agreed word cuts:

```json
{"words": ["患者", "王芳", "因", "心房颤动", "入院"]}
```

Score a candidate backend, model, or dictionary against that set before
promoting it:

```python
import json
import os
from pathlib import Path

from openmed.core.config import OpenMedConfig
from openmed.processing import (
    create_chinese_segmenter_from_config,
    segmentation_boundary_f1,
)
from openmed.processing.tokenization import SpanToken

GATE = 0.90


def gold_cases(path):
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        words = json.loads(line)["words"]
        tokens = []
        cursor = 0
        for word in words:
            tokens.append(SpanToken(word, cursor, cursor + len(word)))
            cursor += len(word)
        cases.append(("".join(words), tokens))
    return cases


segmenter = create_chinese_segmenter_from_config(
    OpenMedConfig(),
    hanlp_model=os.environ.get("OPENMED_HANLP_MODEL_DIR"),
)
cases = gold_cases(os.environ["OPENMED_ZH_GOLD_SET"])
scores = [
    segmentation_boundary_f1(gold, segmenter.segment(text)) for text, gold in cases
]
mean_f1 = sum(scores) / len(scores)

print(f"mean boundary F1 over {len(scores)} sentences: {mean_f1:.3f}")
assert mean_f1 >= GATE
```

OpenMed gates its own held-out probe set for the default backend at a mean
boundary F1 of 0.90. Start there, and raise the gate once a backend, model, and
dictionary combination has a settled baseline. Segmentation is also checked on
every call: `validate_segmentation` rejects tokens that overlap, run out of
order, or do not match the source text, and any segmentation that leaves
non-whitespace text uncovered. A mismatched or corrupted model surfaces as an
error instead of as silent span drift.
