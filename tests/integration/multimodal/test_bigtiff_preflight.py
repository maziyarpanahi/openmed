"""Synthetic local stream integration; no models, pixels or network providers."""

import io

import pytest

from openmed.multimodal.bigtiff_metadata import BigTiffLimits, read_bigtiff_metadata
from openmed.multimodal.tiff_metadata import TiffMetadataError
from tests.fixtures.bigtiff_pyramid import pyramid

pytestmark = pytest.mark.integration


class ShortStream(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.bytes_read = 0

    def read(self, size):
        data = super().read(min(7, size))
        self.bytes_read += len(data)
        return data


@pytest.mark.parametrize("seekable", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
def test_bounded_stream_ownership_and_restoration(seekable, accepted):
    data, header_end = pyramid()
    stream = ShortStream(b"PREFIX" + data)
    stream.seek(6)
    if not seekable:
        stream.seekable = lambda: False
    limit = header_end if accepted else 16
    kwargs = dict(
        profile="bigtiff-ifd-chain-v1",
        source_size=len(data),
        limits=BigTiffLimits(max_header_bytes=limit),
    )
    if accepted:
        assert read_bigtiff_metadata(stream, **kwargs).total_tiles == 5
    else:
        with pytest.raises(TiffMetadataError):
            read_bigtiff_metadata(stream, **kwargs)
    assert stream.bytes_read == limit
    assert not stream.closed
    assert stream.tell() == (6 if seekable else 6 + limit)
