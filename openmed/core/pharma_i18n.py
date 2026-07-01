"""Turkish pharmaceutical NER support for OpenMed.

This module provides detection for Turkish drug brands and substances
based on the drugbase-tr lexicon.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

# Resolve data path
DATA_DIR = Path(__file__).parent / "data" / "pharma" / "tr"

def _load_lexicon(filename: str) -> Dict[str, str]:
    path = DATA_DIR / filename
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# Lazy loading to avoid overhead if not used
_BRANDS: Optional[Dict[str, str]] = None
_SUBSTANCES: Optional[Dict[str, str]] = None
_ATC_MAP: Optional[Dict[str, Any]] = None

def get_brands() -> Dict[str, str]:
    global _BRANDS
    if _BRANDS is None:
        _BRANDS = _load_lexicon("drug_brands.json")
    return _BRANDS

def get_substances() -> Dict[str, str]:
    global _SUBSTANCES
    if _SUBSTANCES is None:
        _SUBSTANCES = _load_lexicon("drug_substances.json")
    return _SUBSTANCES

def get_atc_mapping() -> Dict[str, Any]:
    global _ATC_MAP
    if _ATC_MAP is None:
        _ATC_MAP = _load_lexicon("atc_mapping.json")
    return _ATC_MAP

def extract_turkish_pharma_entities(text: str) -> List[Dict[str, Any]]:
    """Extract drug entities from Turkish text using the drugbase-tr lexicon.
    
    Args:
        text: Input text
        
    Returns:
        List of entity dictionaries with start, end, label, and metadata.
    """
    if not text:
        return []

    brands = get_brands()
    substances = get_substances()
    atc_map = get_atc_mapping()
    
    entities = []
    
    # Find word spans including Turkish characters and dashes
    words = list(re.finditer(r"[A-Za-zİıĞğÜüŞşÖöÇç0-9\-]+", text))
    
    i = 0
    while i < len(words):
        # Try bigram first (for multi-word brands like 'ADALAT CRONO')
        if i + 1 < len(words):
            bigram_text = text[words[i].start():words[i+1].end()].lower()
            if bigram_text in brands:
                canonical = brands[bigram_text]
                entities.append({
                    "text": text[words[i].start():words[i+1].end()],
                    "label": "DRUG_BRAND",
                    "start": words[i].start(),
                    "end": words[i+1].end(),
                    "confidence": 0.95,
                    "metadata": {"canonical": canonical, "atc": atc_map.get(canonical, [])}
                })
                i += 2
                continue
            elif bigram_text in substances:
                canonical = substances[bigram_text]
                entities.append({
                    "text": text[words[i].start():words[i+1].end()],
                    "label": "DRUG_SUBSTANCE",
                    "start": words[i].start(),
                    "end": words[i+1].end(),
                    "confidence": 0.95,
                    "metadata": {"canonical": canonical}
                })
                i += 2
                continue

        # Try unigram
        unigram_text = words[i].group().lower()
        if unigram_text in brands:
            canonical = brands[unigram_text]
            entities.append({
                "text": words[i].group(),
                "label": "DRUG_BRAND",
                "start": words[i].start(),
                "end": words[i].end(),
                "confidence": 0.9,
                "metadata": {"canonical": canonical, "atc": atc_map.get(canonical, [])}
            })
        elif unigram_text in substances:
            canonical = substances[unigram_text]
            entities.append({
                "text": words[i].group(),
                "label": "DRUG_SUBSTANCE",
                "start": words[i].start(),
                "end": words[i].end(),
                "confidence": 0.9,
                "metadata": {"canonical": canonical}
            })
            
        i += 1
        
    return entities
