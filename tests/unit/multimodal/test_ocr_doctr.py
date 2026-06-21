import pytest
from unittest.mock import MagicMock, patch
from openmed.multimodal.ocr import run_doctr_ocr, OcrResult

class MockWord:
    def __init__(self, value, confidence, xmin, ymin, xmax, ymax):
        self.value = value
        self.confidence = confidence
        self.geometry = ((xmin, ymin), (xmax, ymax))

class MockLine:
    def __init__(self, words):
        self.words = words

class MockBlock:
    def __init__(self, lines):
        self.lines = lines

class MockPage:
    def __init__(self, dimensions, blocks):
        self.dimensions = dimensions
        self.blocks = blocks

class MockDocument:
    def __init__(self, pages):
        self.pages = pages

@patch("openmed.multimodal.ocr.ocr_predictor")
@patch("openmed.multimodal.ocr.DOCTR_AVAILABLE", True)
def test_run_doctr_ocr_mapping(mock_predictor):
    mock_word = MockWord("Clinical", 0.991234, 0.1, 0.2, 0.3, 0.4)
    mock_line = MockLine([mock_word])
    mock_block = MockBlock([mock_line])
    mock_page = MockPage((1000, 500), [mock_block])
    mock_doc = MockDocument([mock_page])
    
    mock_instance = MagicMock()
    mock_instance.return_value = mock_doc
    mock_predictor.return_value = mock_instance

    results = run_doctr_ocr(["sample_invoice.jpg"])

    assert len(results) == 1
    assert results[0].text == "Clinical"
    assert results[0].bbox == [50, 200, 150, 400]
    assert results[0].confidence == 0.9912
    assert results[0].page == 0

@patch("openmed.multimodal.ocr.DOCTR_AVAILABLE", False)
def test_run_doctr_ocr_import_error():
    with pytest.raises(ImportError) as exc_info:
        run_doctr_ocr(["sample_invoice.jpg"])
    assert "python-doctr" in str(exc_info.value)