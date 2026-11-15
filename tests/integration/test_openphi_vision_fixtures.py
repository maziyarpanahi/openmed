"""Synthetic, model-free consumption and leakage controls; no suite scoring."""

import pytest

from openmed.eval.openphi_vision_fixtures import load_openphi_vision_fixtures

ImageDraw = pytest.importorskip("PIL.ImageDraw")


@pytest.mark.integration
def test_gold_box_redaction_and_missing_critical_box_control():
    for case in load_openphi_vision_fixtures():
        for page in case.pages:
            image = page.to_image()
            draw = ImageDraw.Draw(image)
            for box in page.gold_boxes:
                x0, y0, x1, y1 = box.bbox
                # Pillow rectangles are inclusive; fixture boxes are half-open.
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=0)
            assert image.getbbox() is None

            # Negative control: omit one critical identifier. Its ink remains,
            # so a suite cannot equate successful noncritical masking with safety.
            omitted = next(box for box in page.gold_boxes if box.critical)
            image = page.to_image()
            draw = ImageDraw.Draw(image)
            for box in page.gold_boxes:
                if box is omitted:
                    continue
                x0, y0, x1, y1 = box.bbox
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=0)
            assert image.crop(omitted.bbox).getbbox() is not None


@pytest.mark.integration
def test_local_provider_input_preserves_packet_page_order():
    packet = next(
        case for case in load_openphi_vision_fixtures() if case.track == "packets"
    )
    # The pending #3205 suite can pass pixels to an offline provider without
    # passing gold text. This stand-in measures input bytes; it claims no OCR.
    inputs = [(page.page_id, page.to_image()) for page in packet.pages]
    assert [page_id for page_id, _ in inputs] == [page.page_id for page in packet.pages]
    for index, (_, image) in enumerate(inputs):
        assert image.size == packet.pages[index].size
        assert image.mode == "L"
        assert image.tobytes() == packet.pages[index].to_image().tobytes()
    assert packet.expected_packet_splits != ((0, 1), (2, 3))
