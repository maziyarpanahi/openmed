# Multimodal dependency licenses

The [dependency license inventory](../security/license-inventory.md) records the
reviewed license expression for every package declared in the `multimodal` and
`ocr-paddle` extras. The offline inventory gate checks those declarations in
`pyproject.toml` and refuses an unrecorded or disallowed dependency. Run it from
the repository root when proposing a dependency change.

```bash
python scripts/licenses/inventory.py
```

[PyPI lists pydicom as MIT](https://pypi.org/project/pydicom/).
The separate [deid project also lists MIT](https://pypi.org/project/deid/).
OpenMed's multimodal extra depends on pydicom. It does not declare the separate
`deid` distribution. These sources were checked on 2026-11-07.

## OCR system binary

The Python wrapper `pytesseract` calls the separately installed Tesseract
executable. [Tesseract's source license](https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE)
is Apache-2.0. The OpenMed Python distribution does not redistribute the system
binary. Review the installed binary and its language data separately when
packaging an OCR runtime.

## Imaging data

The software dependency licenses are independent of the data they read. TCIA
lists the applicable terms for each collection. Check the collection’s [data
usage policy](https://www.cancerimagingarchive.net/data-usage-policies-and-restrictions/)
before using it in an evaluation. Use synthetic DICOM data for repository
fixtures.
