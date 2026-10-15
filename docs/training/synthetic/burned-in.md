# Burned-in PHI images

`BurnedInGenerator` renders a synthetic name, date, and medical record number
into the pixels of an image, and returns both the character-level spans and
the pixel bounding boxes for each. It is the synthetic counterpart of
screenshots and scans that have text burned into them, for training and
smoke-testing image de-identification. See the [overview](index.md) for shared
conventions.

Pillow is imported only when an image is rendered. Install it with
`pip install "openmed[multimodal]"`; without it, `generate` raises
`ImportError`, while importing `openmed.training.synthetic` still works.

## Generate one image

```python
from openmed.training.synthetic import BurnedInGenerator

generator = BurnedInGenerator(seed=3)
example = generator.generate("en")

print(example.image.size)            # (640, 320)
print(example.text.splitlines())     # name, date, MRN, one per line
for box in example.gold_boxes:
    print(box.label, box.bbox, box.text, box.start, box.end)
    assert example.text[box.start:box.end] == box.text
```

`gold_boxes` (alias `boxes`) holds one `BurnedInTextBox` per label in
`BURNED_IN_LABELS` (`PERSON`, `DATE`, `ID_NUM`). `bbox` is Pillow's half-open
`(x0, y0, x1, y1)` pixel box, and `start`/`end` index into `example.text`.
`gold_spans` carries the same offsets as the
[locale PHI](locale-phi.md) `SyntheticPhiSpan` records. The `language`
argument selects the locale used for the name and date text.

For a one-off example with default variation, use the helper:

```python
from openmed.training.synthetic import generate_burned_in_example

example = generate_burned_in_example(seed=3, language="en", canvas_size=(640, 320))
```

## Layout controls

Without controls, background, fonts, font sizes, and positions are chosen
deterministically from the seed. To pin them, pass exactly three values, in
`(PERSON, DATE, ID_NUM)` order:

```python
example = generator.generate(
    "en",
    positions=[(20, 20), (20, 80), (20, 140)],   # top-left pixel of each box
    font_sizes=[18, 18, 18],
    background="solid",
)
print(example.gold_boxes[0].bbox)
```

- `background` is one of `BACKGROUND_MODES`: `solid`, `gradient`, `phantom`.
- Constructor pools default to `DEFAULT_CANVAS_SIZE` (640 × 320),
  `DEFAULT_FONT_SIZES` (18, 22, 26) and `DEFAULT_FONT_NAMES` (three DejaVu
  fonts). If a named font is missing on the machine, Pillow's bundled default
  font is used instead and the image still renders.
- Invalid controls (wrong count, non-positive size, unknown background, text
  that cannot fit the canvas) raise `ValueError`.

## Training items

```python
item = example.to_training_item()
print(sorted(item))
# ['image', 'is_synthetic', 'labels', 'language', 'metadata', 'pixel_boxes',
#  'synthetic_source', 'text']
```

The Pillow image stays in memory in the dictionary; saving it (for example
`item["image"].save("sample.png")`) is up to the caller. `metadata` records
the seed, background mode, canvas size, and `contains_real_phi: False`.

## Seeds and determinism

The same seed and settings give the same text and boxes:

```python
a = BurnedInGenerator(seed=3).generate("en")
b = BurnedInGenerator(seed=3).generate("en")
assert a.text == b.text and a.gold_boxes == b.gold_boxes
```

## Known limits

- Exactly three labels (`PERSON`, `DATE`, `ID_NUM`); no addresses, faces,
  barcodes, or handwriting.
- Text is drawn horizontally with simple backgrounds. Real scans have noise,
  skew, compression, and layouts these images do not model.
- Values come from the locale PHI generator and Faker, so the
  [Gregorian-ASCII-date and Faker-version caveats](locale-phi.md#known-limits)
  apply here too. Rendered pixels can differ slightly between Pillow or font
  versions even when the seed is unchanged.
- The images are augmentation data, not clinical ground truth.
