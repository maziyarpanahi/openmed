# Synthetic photo QA fixtures

This directory contains fictional label-reading and shape-counting images for
offline VLM evaluation. Every printed code, date-like value and identifier is
invented; no patient data or real product label is included.

`truth.json` records the question, expected answer, half-open pixel box and
SHA-256 of the RGB raster for every item. Regenerate the PNGs and manifest with
the pinned project Pillow version:

```bash
python tests/fixtures/multimodal/photo_qa/generate_fixtures.py
```

The test verifies that regeneration reproduces the committed manifests and
pixels, and that every box lies inside its image.
