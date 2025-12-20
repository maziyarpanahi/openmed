from openmed.processing.tokenization import medical_tokenize, remap_predictions_to_tokens, SpanToken


def test_medical_tokenize_keeps_hyphen_chain():
    text = "IL-6-mediated cytokine storm"
    tokens = medical_tokenize(text)
    assert any(t.text == "IL-6-mediated" for t in tokens)


def test_remap_predictions_merges_wordpieces_to_medical_token():
    text = "IL-6-mediated cytokine storm"
    tokens = medical_tokenize(text)

    # Simulate model outputs on wordpieces with char spans
    preds = [
        {"start": 0, "end": 2, "entity": "B-Gene_or_gene_product", "score": 0.9, "metadata": {"sentence_index": 0}},
        {"start": 3, "end": 4, "entity": "I-Gene_or_gene_product", "score": 0.8, "metadata": {"sentence_index": 0}},
        {"start": 5, "end": 13, "entity": "I-Gene_or_gene_product", "score": 0.85, "metadata": {"sentence_index": 0}},
    ]

    remapped = remap_predictions_to_tokens(preds, text, tokens)
    assert len(remapped) == 1
    assert remapped[0]["entity_group"] == "Gene_or_gene_product"
    assert remapped[0]["start"] == 0
    assert remapped[0]["end"] == 13
    assert "sentence_index" in remapped[0]["metadata"]


def test_remap_predictions_merges_adjacent_tokens_same_label():
    text = "B-cell ALL"
    tokens = [SpanToken("B-cell", 0, 6), SpanToken("ALL", 7, 10)]
    preds = [
        {"start": 0, "end": 6, "entity_group": "Cell", "score": 0.9, "metadata": {}},
        {"start": 7, "end": 10, "entity_group": "Cell", "score": 0.8, "metadata": {}},
    ]
    remapped = remap_predictions_to_tokens(preds, text, tokens, gap=1)
    assert len(remapped) == 1
    assert remapped[0]["start"] == 0
    assert remapped[0]["end"] == 10
    assert remapped[0]["entity_group"] == "Cell"

