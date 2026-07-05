"""Build a minimal, fully synthetic TID1500-style DICOM SR object in memory.

The generated report mirrors the gold content tree committed at
``openmed/eval/golden/fixtures/dicom_sr_content.jsonl`` so the extractor can be
scored node-for-node offline. It carries synthetic PHI in the SR headers (name,
ID, birth date, referring physician, institution) so the de-identification path
can be verified with a no-PHI-in-output assertion. No real study or patient data
is used.

pydicom is imported at call time only; importers of this module stay free of the
optional imaging dependency until a builder is actually invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Synthetic identifiers used both to seed SR headers and to assert they never
# survive into flattened output. These are invented, not real.
PHI_PATIENT_NAME = "DOE^Jane"
PHI_PATIENT_ID = "MRN-778812"
PHI_BIRTH_DATE = "19770903"
PHI_REFERRING_PHYSICIAN = "Smith^Alice"
PHI_INSTITUTION = "OpenMed Synthetic Imaging Center"

PHI_TOKENS: tuple[str, ...] = (
    "Jane",
    "DOE",
    PHI_PATIENT_ID,
    PHI_BIRTH_DATE,
    "Alice",
    "Smith",
    PHI_INSTITUTION,
)

STUDY_UID = "1.2.826.0.1.3680043.10.777.1"
SERIES_UID = "1.2.826.0.1.3680043.10.777.2"
SOP_UID = "1.2.826.0.1.3680043.10.777.3"


def _code(dataset_cls: Any, value: str, scheme: str, meaning: str) -> Any:
    item = dataset_cls()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def _concept_name_code(dataset_cls: Any, sequence_cls: Any, code: Any) -> Any:
    return sequence_cls([code])


def _container(
    dataset_cls: Any,
    sequence_cls: Any,
    *,
    concept: Any,
    relationship: str | None,
    template_id: str | None = None,
) -> Any:
    node = dataset_cls()
    node.ValueType = "CONTAINER"
    node.ContinuityOfContent = "SEPARATE"
    if relationship is not None:
        node.RelationshipType = relationship
    node.ConceptNameCodeSequence = sequence_cls([concept])
    if template_id is not None:
        template = dataset_cls()
        template.MappingResource = "DCMR"
        template.TemplateIdentifier = template_id
        node.ContentTemplateSequence = sequence_cls([template])
    node.ContentSequence = sequence_cls([])
    return node


def _code_item(
    dataset_cls: Any,
    sequence_cls: Any,
    *,
    concept: Any,
    value_code: Any,
    relationship: str,
) -> Any:
    node = dataset_cls()
    node.ValueType = "CODE"
    node.RelationshipType = relationship
    node.ConceptNameCodeSequence = sequence_cls([concept])
    node.ConceptCodeSequence = sequence_cls([value_code])
    return node


def _text_item(
    dataset_cls: Any,
    sequence_cls: Any,
    *,
    concept: Any,
    text: str,
    relationship: str,
) -> Any:
    node = dataset_cls()
    node.ValueType = "TEXT"
    node.RelationshipType = relationship
    node.ConceptNameCodeSequence = sequence_cls([concept])
    node.TextValue = text
    return node


def _num_item(
    dataset_cls: Any,
    sequence_cls: Any,
    *,
    concept: Any,
    numeric_value: str,
    unit: Any,
    relationship: str,
) -> Any:
    node = dataset_cls()
    node.ValueType = "NUM"
    node.RelationshipType = relationship
    node.ConceptNameCodeSequence = sequence_cls([concept])
    measured = dataset_cls()
    measured.NumericValue = numeric_value
    measured.MeasurementUnitsCodeSequence = sequence_cls([unit])
    node.MeasuredValueSequence = sequence_cls([measured])
    return node


def build_synthetic_sr_dataset() -> Any:
    """Return an in-memory pydicom SR ``FileDataset`` matching the gold tree.

    Raises:
        ImportError: If pydicom is not installed. Callers should guard with
            ``pytest.importorskip("pydicom")``.
    """
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.sequence import Sequence
    from pydicom.uid import (
        ComprehensiveSRStorage,
        ExplicitVRLittleEndian,
    )

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = ComprehensiveSRStorage
    file_meta.MediaStorageSOPInstanceUID = SOP_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = "1.2.826.0.1.3680043.10.777.99"

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # SR object identity + modality.
    ds.SOPClassUID = ComprehensiveSRStorage
    ds.SOPInstanceUID = SOP_UID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID
    ds.Modality = "SR"
    ds.SeriesNumber = "1"
    ds.InstanceNumber = "1"
    ds.CompletionFlag = "COMPLETE"
    ds.VerificationFlag = "UNVERIFIED"
    ds.ContentDate = "20210105"
    ds.ContentTime = "101500"

    # Synthetic PHI in the SR headers (must be scrubbed before text emission).
    ds.PatientName = PHI_PATIENT_NAME
    ds.PatientID = PHI_PATIENT_ID
    ds.PatientBirthDate = PHI_BIRTH_DATE
    ds.PatientSex = "F"
    ds.ReferringPhysicianName = PHI_REFERRING_PHYSICIAN
    ds.InstitutionName = PHI_INSTITUTION
    ds.StudyDate = "20210105"
    ds.StudyTime = "100000"

    # Root CONTAINER (TID 1500 imaging measurement report).
    report_root = _container(
        Dataset,
        Sequence,
        concept=_code(Dataset, "126000", "DCM", "Imaging Measurement Report"),
        relationship=None,
        template_id="1500",
    )
    ds.ValueType = report_root.ValueType
    ds.ContinuityOfContent = report_root.ContinuityOfContent
    ds.ConceptNameCodeSequence = report_root.ConceptNameCodeSequence
    ds.ContentTemplateSequence = report_root.ContentTemplateSequence

    language = _code_item(
        Dataset,
        Sequence,
        concept=_code(
            Dataset, "121049", "DCM", "Language of Content Item and Descendants"
        ),
        value_code=_code(Dataset, "eng", "RFC5646", "English"),
        relationship="HAS CONCEPT MOD",
    )
    procedure = _code_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "121058", "DCM", "Procedure reported"),
        value_code=_code(Dataset, "169069000", "SCT", "CT of chest"),
        relationship="HAS CONCEPT MOD",
    )

    imaging_measurements = _container(
        Dataset,
        Sequence,
        concept=_code(Dataset, "126010", "DCM", "Imaging Measurements"),
        relationship="CONTAINS",
        template_id="1501",
    )
    measurement_group = _container(
        Dataset,
        Sequence,
        concept=_code(Dataset, "125007", "DCM", "Measurement Group"),
        relationship="CONTAINS",
        template_id="1502",
    )
    tracking_id = _text_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "112039", "DCM", "Tracking Identifier"),
        text="Lesion 1",
        relationship="HAS OBS CONTEXT",
    )
    finding_site = _code_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "363698007", "SCT", "Finding Site"),
        value_code=_code(Dataset, "45653009", "SCT", "Upper lobe of right lung"),
        relationship="HAS CONCEPT MOD",
    )
    long_axis = _num_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "103339001", "SCT", "Long Axis"),
        numeric_value="12.5",
        unit=_code(Dataset, "mm", "UCUM", "mm"),
        relationship="CONTAINS",
    )
    short_axis = _num_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "103340004", "SCT", "Short Axis"),
        numeric_value="8.0",
        unit=_code(Dataset, "mm", "UCUM", "mm"),
        relationship="CONTAINS",
    )
    measurement_group.ContentSequence = Sequence(
        [tracking_id, finding_site, long_axis, short_axis]
    )
    imaging_measurements.ContentSequence = Sequence([measurement_group])

    qualitative = _container(
        Dataset,
        Sequence,
        concept=_code(Dataset, "C0034375", "UMLS", "Qualitative Evaluations"),
        relationship="CONTAINS",
    )
    finding = _text_item(
        Dataset,
        Sequence,
        concept=_code(Dataset, "121071", "DCM", "Finding"),
        text="Solid pulmonary nodule, stable in size",
        relationship="CONTAINS",
    )
    qualitative.ContentSequence = Sequence([finding])

    ds.ContentSequence = Sequence(
        [language, procedure, imaging_measurements, qualitative]
    )
    return ds


def write_synthetic_sr(path: str | Path) -> Path:
    """Write the synthetic SR object to ``path`` and return it.

    Raises:
        ImportError: If pydicom is not installed.
    """
    dataset = build_synthetic_sr_dataset()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        dataset.save_as(target, enforce_file_format=True)
    except TypeError:  # pragma: no cover - older pydicom compatibility.
        dataset.save_as(target, write_like_original=False)
    return target
