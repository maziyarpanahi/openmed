# On-device segmenter resources

Chinese and Indic artifact packs can include a compact segmenter resource set
without embedding the jieba or ICU runtimes. The same manifest descriptor and
resource files are supported by MLX, ONNX/Transformers.js, CoreML, Python, and
OpenMedKit.

Segmenter resources are opt-in. Latin and whitespace-segmented bundles do not
gain new required files or manifest fields.

## Package a segmenter

Pass one of the following values to `--segmenter` on the MLX, ONNX, CoreML, or
Transformers.js converter:

- `openmed-han-v1` packages the compact jieba-compatible Han dictionary.
- `openmed-indic-v1` packages the Devanagari break-rule table.
- `openmed-cjk-indic-v1` packages both resources.

For example:

```bash
python -m openmed.mlx.convert \
  --model OpenMed/example-zh-hi \
  --output ./openmed-mlx \
  --segmenter openmed-cjk-indic-v1
```

Each selected converter writes a `segmenter` object into its artifact manifest.
The descriptor records the segmenter id, scripts, resource paths, per-file
licenses, byte sizes, SHA-256 digests, total size, and size budget. Bundle
validators reject missing, modified, path-escaping, unlicensed, or oversized
resources.

## Size and licensing contract

The on-device resource budget is **64 KiB per bundle**. This limit is stored in
the descriptor as `size_budget_bytes: 65536` and enforced during packaging and
validation. It covers only the declared data tables; no optional runtime code is
included.

The compact Han dictionary uses the jieba-compatible dictionary format and is
recorded as `MIT`. The Indic grapheme-break table is adapted from ICU 57.1
`icu4c/source/data/brkitr/rules/char.txt` at immutable revision
`0c5873f89bf64f6bbc0a24b84f07d79b25785a42` and is recorded with the SPDX
identifier `ICU`. A combined descriptor records `MIT AND ICU`.

Every Indic or combined bundle includes `segmenter/ICU.txt` as a declared,
size-checked, digest-bound `license_notice` resource. The accompanying
`indic_rules.json` records the upstream project, exact source path and revision,
retrieval date, local modifications, and notice filename. Validators reject a
bundle whose ICU notice or provenance record is missing, modified, or
incomplete.

## Runtime behavior

Python's `ResourceSegmenter` and Swift's `OpenMedSegmenter` consume the same
descriptor. Both return UTF-8 byte offsets, use longest-match dictionary
segmentation for Han, and apply the packaged grapheme rules for Devanagari.
The Python path uses only the standard library by default; an installed jieba
runtime can be requested as an optional accelerated Han path.
