import math
from typing import List, NamedTuple

class OcrResult(NamedTuple):
    text: str
    bbox: List[int]
    confidence: float
    page: int

try:
    from doctr.models import ocr_predictor
    DOCTR_AVAILABLE = True
except ImportError:
    DOCTR_AVAILABLE = False

def run_doctr_ocr(image_paths: List[str], engine_name: str = "doctr") -> List[OcrResult]:
    if not DOCTR_AVAILABLE:
        raise ImportError(
            "The 'python-doctr' package is required to run this engine. "
            "Please install it using: pip install python-doctr"
        )

    predictor = ocr_predictor(pretrained=True)
    doc = predictor(image_paths)
    ocr_results = []

    for page_idx, page in enumerate(doc.pages):
        page_height, page_width = page.dimensions
        
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (rel_xmin, rel_ymin), (rel_xmax, rel_ymax) = word.geometry
                    
                    abs_xmin = int(round(rel_xmin * page_width))
                    abs_ymin = int(round(rel_ymin * page_height))
                    abs_xmax = int(round(rel_xmax * page_width))
                    abs_ymax = int(round(rel_ymax * page_height))
                    
                    result_item = OcrResult(
                        text=word.value,
                        bbox=[abs_xmin, abs_ymin, abs_xmax, abs_ymax],
                        confidence=round(float(word.confidence), 4),
                        page=page_idx
                    )
                    ocr_results.append(result_item)

    return ocr_results