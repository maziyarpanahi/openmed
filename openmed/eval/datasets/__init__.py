"""Dataset adapters for public and data-use-gated eval corpora."""

from .dua_stubs import (
    DUA_GATED_CORPORA,
    DUACorpusStub,
    DUACredentialRequired,
    all_dua_stubs,
    dua_stub_for,
    load_dua_corpus,
)
from .licenses import DatasetLicense, PUBLIC_DATASET_LICENSES, license_for
from .public import (
    DatasetLoadResult,
    DatasetUnavailable,
    PUBLIC_DATASETS,
    PUBLIC_LABEL_MAPS,
    PublicDatasetAdapter,
    PublicDatasetRecord,
    PublicDatasetSpan,
    adapter_for,
    assert_no_gated_content_committed,
    load_public_dataset,
    map_public_label,
)

__all__ = [
    "DUA_GATED_CORPORA",
    "DUACorpusStub",
    "DUACredentialRequired",
    "DatasetLicense",
    "DatasetLoadResult",
    "DatasetUnavailable",
    "PUBLIC_DATASETS",
    "PUBLIC_DATASET_LICENSES",
    "PUBLIC_LABEL_MAPS",
    "PublicDatasetAdapter",
    "PublicDatasetRecord",
    "PublicDatasetSpan",
    "adapter_for",
    "all_dua_stubs",
    "assert_no_gated_content_committed",
    "dua_stub_for",
    "license_for",
    "load_dua_corpus",
    "load_public_dataset",
    "map_public_label",
]
