# Multimodal dependency licenses

The `[multimodal]` and `[ocr-paddle]` extras pull document, image, and DICOM
ingestion libraries into the OpenMed process. Invariant I2 requires that
everything running in-process is permissively licensed, and that GPL,
source-available, or proprietary tools are reached only through an
out-of-process bridge in `openmed/interop/bridges/`.

This page is the human-readable view of the license record. The authoritative,
machine-checkable copy is `openmed/multimodal/_licenses.py`, enforced by
`tests/unit/multimodal/test_multimodal_licenses.py`. Adding a dependency to
either extra without recording its license fails the test suite.

Licenses below were read from each project's PyPI metadata and verified on
**2026-08-03**.

## Python dependencies

| Distribution | SPDX | Source |
| --- | --- | --- |
| easyocr | `Apache-2.0` | [PyPI](https://pypi.org/project/easyocr/) |
| markdown-it-py | `MIT` | [PyPI](https://pypi.org/project/markdown-it-py/) |
| numpy | `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | [PyPI](https://pypi.org/project/numpy/) |
| onnx | `Apache-2.0` | [PyPI](https://pypi.org/project/onnx/) |
| paddleocr | `Apache-2.0` | [PyPI](https://pypi.org/project/paddleocr/) |
| pdfplumber | `MIT` | [PyPI](https://pypi.org/project/pdfplumber/) |
| piexif | `MIT` | [PyPI](https://pypi.org/project/piexif/) |
| pikepdf | `MPL-2.0` | [PyPI](https://pypi.org/project/pikepdf/) |
| pillow | `MIT-CMU` | [PyPI](https://pypi.org/project/pillow/) |
| pydicom | `MIT` | [PyPI](https://pypi.org/project/pydicom/) |
| pytesseract | `Apache-2.0` | [PyPI](https://pypi.org/project/pytesseract/) |
| python-docx | `MIT` | [PyPI](https://pypi.org/project/python-docx/) |
| python-doctr | `Apache-2.0` | [PyPI](https://pypi.org/project/python-doctr/) |

`paddleocr` is pinned by the `ocr-paddle` extra; every other entry is pinned by
`multimodal`. The split is a packaging decision because paddlepaddle is heavy and platform-sensitive.

Every entry is on the permissive allow-list in
`scripts/release/check_license_policy.py`, so all of them may be imported
in-process. None of the multimodal extras currently requires a subprocess
bridge.

### Entries that need a note

**pydicom — MIT, verified.** Earlier roadmap revisions (sec 2.2c, 5.8, 4.6)
carried a "license unverified — confirm before bundling" caveat against
pydicom and the deid library. That caveat is resolved: pydicom publishes MIT
and is safe to bundle in-process. The record supersedes the roadmap note.

**pikepdf — MPL-2.0.** The one weak-copyleft entry. MPL-2.0 obligations attach at *file* scope: distributing a modified pikepdf source file requires
publishing that file's source, but importing pikepdf from OpenMed code imposes no obligation on OpenMed's own sources. The repository license policy lists `MPL-2.0` as allowed for this reason. If OpenMed ever vendors or patches pikepdf source, the modified files must be published under MPL-2.0.

**pillow — MIT-CMU.** Pillow publishes `MIT-CMU`, the current SPDX identifier
for the HPND-style license it has always carried. The OM-036 reviewed-license
table spells the same license `HPND`. Both are permissive and both pass the
gate; the tests compare policy outcomes rather than license strings so the two
spellings do not conflict.

**numpy — multi-license expression.** The expression covers vendored components. Every term is permissive; none is copyleft.

## OCR system binaries

`pytesseract` is a thin wrapper: it shells out to a Tesseract binary the user
installs through a system package manager. That binary is not in the Python
dependency graph, so the pyproject-driven gate cannot see it and its license is
recorded separately.

| Binary | SPDX | Engine | Source |
| --- | --- | --- | --- |
| Tesseract OCR | `Apache-2.0` | `tesseract` | [LICENSE](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE) |

OpenMed does not redistribute the Tesseract binary. Users obtain it themselves
(`brew install tesseract`, `apt-get install tesseract-ocr`), so no bundling
obligation arises; the license is recorded for downstream users who package
OpenMed together with its OCR runtime.

## Dataset caveat: TCIA collections

The Cancer Imaging Archive is the most likely source a contributor reaches for
when testing the DICOM path, and its terms do **not** follow pydicom's license.

TCIA licenses data **per collection, not archive-wide**. Individual collections
range from CC BY through to restricted terms requiring explicit permission. No
TCIA collection may be committed as a test fixture or bundled with OpenMed
without checking that specific collection's license first. Use synthetic DICOM
fixtures instead, consistent with the repository rule that committed golden
data must be synthetic.

See [TCIA data usage policies](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/).

## Scope

This page covers the `[multimodal]` and `[ocr-paddle]` extras only. The
repository-wide dependency license gate is OM-036
(`scripts/release/check_license_policy.py`), which audits every non-dev extra
and runs in CI. SBOM and supply-chain scanning are tracked separately.

## Re-verifying

Re-check a license against upstream metadata with:

```bash
curl -s https://pypi.org/pypi/pydicom/json \
  | python3 -c "import json,sys; i=json.load(sys.stdin)['info']; \
print(i.get('license_expression') or i.get('license'), \
[c for c in i['classifiers'] if c.startswith('License')])"
```

When a license changes, update the SPDX value **and** the `verified_on` date in
`openmed/multimodal/_licenses.py`, then update the table above.
