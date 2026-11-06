# Synthetic clinical document rendering

Render generated clinical content into actual form, letter, label and fax-like PNG pages, with word geometry and synthetic PHI annotations. The renderer runs locally and accepts a seed and layout choices, not real documents or patient text. It reuses the [clinical note grammars](clinical-note-grammars.md) and source-linked [fused evidence graphs](../../multimodal/fused-evidence-graphs.md).

## Render and reproduce

Install Pillow through `pip install "openmed[multimodal]"` if it is not already available. Image dependencies are loaded when rendering is requested.

```python
from openmed.training.synthetic.document_renderer import (
    DocumentLayout,
    DocumentRenderSpec,
    ScanArtifacts,
    render_synthetic_document,
    reproduce_synthetic_document,
)

document = render_synthetic_document(
    spec=DocumentRenderSpec(
        seed=42,
        layout=DocumentLayout.FAX,
        grammar="encounter",
        artifacts=ScanArtifacts(
            skew_degrees=3,
            blur_radius=0.5,
            scan_strength=4,
            jpeg_quality=45,
        ),
    )
)
repeated = reproduce_synthetic_document(document)
assert repeated.digest == document.digest
assert repeated.pages[0].png == document.pages[0].png
assert any(word.is_phi for page in document.pages for word in page.words)
report = document.to_json()
```

`document.text`, each page's `clean_png` and `png`, and each word's mask PNGs are explicit synthetic test payloads. Use them directly for local OCR fixtures or explicitly save them in your own test-data directory. The renderer writes no files, downloads no assets and calls no network services. Ordinary representations and `to_json()` exclude text, images, font paths and backend exception details.

The same seed, specification, font bytes and rendering runtime reproduce the same output. The receipt binds the generated text, actual clean and final PNGs, font bytes, Pillow/FreeType/codec/shaping runtime and the transformation chain. `reproduce_synthetic_document` reruns rendering and compares the result; it rejects a changed runtime, font, content, pixels or receipt. These digests detect mismatch; they are not signatures or proof that external data is synthetic.

## Layout and annotations

| Choice | Executed layout |
| --- | --- |
| `FORM` | Ruled fields flowing down the left column and then the right column |
| `LETTER` | A single text column beneath a letterhead bar |
| `LABEL` | Compact rounded labels, paginated without dropping content |
| `FAX` | Gray header, dashed footer divider and a single text column |

Choose `encounter`, `discharge`, `referral`, `operative` or `nursing` as the grammar. Content consists of generated English identifiers and fictional clinical placeholders. This is a layout and corruption test source, not evidence of clinical language realism or multilingual OCR coverage. Page dimensions default to 768 × 1024, or 512 × 320 for labels; `page_size=(width, height)` overrides them within the documented limits.

Every lexical word has a half-open offset span in `document.text`, canonical labels and fact references when it overlaps a generated fact, a page index, and a global `reading_order`. Reading order visits a page's body in column order, then its footer, before the following page. Text offsets are a separate coordinate system: the protected text buffer contains the entire body followed by each page's footer text.

`clean_box` is the tight half-open rectangle around the glyph pixels actually drawn, measured from a rasterized font mask. `clean_mask_png` contains that tight grayscale mask. Each page has a real synthetic-copy stamp, a repeated generated identifier with PHI truth, and a page number. Disable the stamp or identifier overlay with `stamp=False` or `phi_overlay=False`.

Handwriting placeholders are seeded, untranscribed strokes. Their clean region and transformed quadrilateral are marked separately; they are not word annotations or handwriting-recognition gold. Disable them with `handwriting_placeholder=False`. The placeholder region records placement, not its remaining visibility after corruption.

## Scan effects and visible geometry

Effects execute in a fixed order and record actual output hashes, including identity stages:

1. Horizontal affine shear (`skew_degrees`, −8 to 8), around the vertical midpoint, with bilinear sampling and a fixed canvas.
2. Gaussian blur (`blur_radius`, 0 to 3 pixels).
3. Seeded speckles and white scanner dropout lines (`scan_strength`, 0 to 32).
4. A white occlusion rectangle (`occlusion=(x0, y0, x1, y1)`, normalized coordinates).
5. Actual JPEG encoding and decoding (`jpeg_quality`, 5 to 100, or `None`).
6. Fresh RGB PNG encoding without metadata.

The same affine transform, blur, dropouts and occlusion are applied to per-word glyph masks. `transformed_polygon` is the clean ink rectangle mapped through the shear. `support_box` bounds its warped and blurred mask on the canvas. `visible_box` and `visible_mask_png` describe the remaining support after dropout and occlusion. `visible_fraction` is the remaining mask intensity mass divided by the on-canvas pre-occlusion mass; `clipped` separately indicates geometric clipping at the canvas boundary.

Words retain their truth as `visible`, `partial`, `occluded` or `outside`. Fully hidden words and identifiers remain in the gold annotations. Clean evidence graphs contain all words; final graphs contain only words with nonempty visible support. Both graphs bind page → pixel region → generated OCR reference nodes to their own actual image digest and page coordinates. These are reference annotations, not outputs from an OCR engine.

Visibility describes geometric support, not recognition accuracy. Speckles and JPEG compression can alter contrast and create ringing outside the mask. Blur, compression, overlap and clipping can make text unreadable even when some support remains. The renderer does not certify OCR recall, clinical correctness or privacy compliance.

## Fonts and bounds

The default is Pillow's explicit embedded Aileron face, recorded as CC0-1.0 and rendered from its digested bytes. A `BurnedInFont(path=..., license=..., index=...)` can instead select an existing local font file. The renderer checks the declared permissive font license, glyph coverage and shaping requirements. It does not search platform font directories, substitute a missing font or download one. Reproduction with a custom font requires that same asset and face.

`DocumentRenderLimits` bounds pages, words, page area, mask-processing work and encoded output sizes. Defaults allow 8 pages, 384 words, 2 million pixels per page, 256 million mask-work pixels and 16 MiB per encoded payload. Hard ceilings keep overrides bounded. A word that cannot fit, pagination overflow, missing font, unsupported glyph or exceeded limit rejects the whole render with `DocumentRenderError("synthetic_document_render_rejected")`; there is no truncation or partial successful result.

These are cooperative application bounds. Native font and codec execution is not process-isolated. Use trusted local fonts and your own process boundary for untrusted assets. No dependencies, fonts, weights or datasets are bundled by this module.
