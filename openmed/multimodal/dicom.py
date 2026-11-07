"""DICOM PS3.15 header de-identification.

The module imports pydicom lazily so the multimodal package remains importable
without optional imaging dependencies. Header provenance records tags and
actions only; raw PHI and original UIDs are intentionally omitted.
"""

from __future__ import annotations

import hashlib
import importlib
import math
import re
import uuid
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

from .base import ExtractedDocument, register_handler
from .exceptions import MissingDependencyError

_DICOM_INSTALL_HINT = 'Install with: pip install "openmed[multimodal]".'
_DEFAULT_UID_SALT = "openmed-dicom-uid-v1"
_PROFILE_NAME = "DICOM PS3.15 Basic Application Level Confidentiality Profile"


class DicomPixelStatus(str, Enum):
    """Describe pixel coverage without treating header processing as OCR."""

    NOT_PRESENT = "pixels_not_present"
    NOT_CLEANED = "pixels_not_cleaned"
    DECLARED_CLEAN = "pixels_declared_clean"
    CLEANED = "pixels_cleaned"


class DicomDeidentificationError(ValueError):
    """Refuse an unsafe DICOM operation with a value-free reason code."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class DicomHeaderDeidPolicy:
    """Policy knobs for DICOM header de-identification."""

    output_path: str | Path | None = None
    date_shift_days: int | None = None
    patient_key: str | bytes | None = None
    date_shift_max_days: int | None = None
    date_shift_secret: str | bytes | None = None
    uid_salt: str | bytes = _DEFAULT_UID_SALT
    keep_year: bool = False
    fail_on_unclean_pixels: bool = False
    redact_encapsulated_documents: bool = False
    document_policy: Any = None
    document_models: Any = None
    clean_descriptors: bool = False
    clean_structured_content: bool = False
    retain_longitudinal_temporal_information: bool = False
    retain_device_identity: bool = False
    retain_patient_characteristics: bool = False
    retain_uids: bool = False
    detector: Any = None


@dataclass(frozen=True)
class DicomHeaderAction:
    """Audit-safe description of one DICOM header action."""

    tag: str
    keyword: str
    vr: str
    action: str
    ps315_action: str
    location: str = "Dataset"
    value_sha256: str | None = None
    value_length: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit-safe action summary."""

        payload: dict[str, Any] = {
            "tag": self.tag,
            "keyword": self.keyword,
            "vr": self.vr,
            "action": self.action,
            "ps315_action": self.ps315_action,
            "location": self.location,
        }
        if self.value_sha256 is not None:
            payload["value_sha256"] = self.value_sha256
        if self.value_length is not None:
            payload["value_length"] = self.value_length
        return payload


@dataclass(frozen=True)
class DicomHeaderDeidResult:
    """Result returned by :func:`deidentify_dicom_headers`."""

    source_path: Path
    output_path: Path
    date_shift_days: int
    actions: tuple[DicomHeaderAction, ...]
    uid_remap_count: int
    private_tag_removed_count: int
    pixel_status: DicomPixelStatus = DicomPixelStatus.NOT_CLEANED
    profile_options: tuple[str, ...] = ()

    @property
    def action_count(self) -> int:
        """Number of header actions performed."""

        return len(self.actions)

    def to_audit_report(self) -> dict[str, Any]:
        """Return an audit-safe provenance summary."""

        action_counts = Counter(action.action for action in self.actions)
        return {
            "type": "dicom_header_deidentification",
            "profile": _PROFILE_NAME,
            "profile_version": _BASIC_PROFILE_VERSION,
            "profile_options": list(self.profile_options),
            "source_path_sha256": _hash_value(str(self.source_path)),
            "output_path_sha256": _hash_value(str(self.output_path)),
            "source_suffix": self.source_path.suffix.lower(),
            "output_suffix": self.output_path.suffix.lower(),
            "date_shift_days": self.date_shift_days,
            "longitudinal_temporal_information_modified": (
                "MODIFIED"
                if "retain_longitudinal_temporal_information" in self.profile_options
                else "REMOVED"
            ),
            "action_count": self.action_count,
            "action_counts": dict(sorted(action_counts.items())),
            "uid_remap_count": self.uid_remap_count,
            "private_tag_removed_count": self.private_tag_removed_count,
            "pixel_status": self.pixel_status.value,
            "actions": [action.to_dict() for action in self.actions],
        }


@dataclass(frozen=True)
class DicomPixelRedactionPolicy:
    """Policy knobs for DICOM burned-in pixel-text redaction."""

    output_path: str | Path | None = None
    ocr_engine: Any = None
    model_name: str | None = None
    confidence_threshold: float = 0.5
    bbox_padding: int = 1
    verify_residual: bool = True
    fail_on_residual: bool = True
    custom_recognizer: Any = None
    overlay_mode: str = "remove"
    redact_encapsulated_documents: bool = False
    document_policy: Any = None
    document_models: Any = None


@dataclass(frozen=True)
class DicomPixelFinding:
    """Audit-safe description of one OCR-projected pixel redaction."""

    frame_index: int
    bbox: tuple[int, int, int, int]
    label: str
    confidence: float
    text_sha256: str
    text_length: int
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return an audit-safe finding summary without raw OCR text."""

        payload: dict[str, Any] = {
            "frame_index": self.frame_index,
            "bbox": list(self.bbox),
            "label": self.label,
            "confidence": self.confidence,
            "text_sha256": self.text_sha256,
            "text_length": self.text_length,
        }
        if self.sources:
            payload["sources"] = list(self.sources)
        return payload


@dataclass(frozen=True)
class DicomResidualTextReport:
    """Residual OCR PHI verification report for redacted DICOM pixels."""

    frame_count: int
    residuals: tuple[DicomPixelFinding, ...] = ()

    @property
    def residual_entity_count(self) -> int:
        """Number of residual OCR-projected PHI findings."""

        return len(self.residuals)

    @property
    def passed(self) -> bool:
        """Whether residual OCR found zero PHI."""

        return self.residual_entity_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable residual report."""

        return {
            "frame_count": self.frame_count,
            "passed": self.passed,
            "residual_entity_count": self.residual_entity_count,
            "residuals": [finding.to_dict() for finding in self.residuals],
        }


@dataclass(frozen=True)
class DicomPixelRedactionResult:
    """Result returned by :func:`redact_dicom_pixels`."""

    source_path: Path
    output_path: Path
    frames_processed: int
    findings: tuple[DicomPixelFinding, ...]
    residual_report: DicomResidualTextReport
    carrier_actions: tuple[DicomHeaderAction, ...] = ()
    pixel_status: DicomPixelStatus = DicomPixelStatus.NOT_CLEANED
    _processed_document_digests: tuple[str, ...] = field(default=(), repr=False)

    @property
    def redaction_count(self) -> int:
        """Number of projected pixel boxes blacked out."""

        return len(self.findings)

    def to_audit_report(self) -> dict[str, Any]:
        """Return an audit-safe provenance summary."""

        label_counts = Counter(finding.label for finding in self.findings)
        frame_counts = Counter(finding.frame_index for finding in self.findings)
        return {
            "type": "dicom_pixel_ocr_redaction",
            "source_path_sha256": _hash_value(str(self.source_path)),
            "output_path_sha256": _hash_value(str(self.output_path)),
            "source_suffix": self.source_path.suffix.lower(),
            "output_suffix": self.output_path.suffix.lower(),
            "frames_processed": self.frames_processed,
            "redaction_count": self.redaction_count,
            "redaction_counts_by_label": dict(sorted(label_counts.items())),
            "redaction_counts_by_frame": {
                str(frame): count for frame, count in sorted(frame_counts.items())
            },
            "findings": [finding.to_dict() for finding in self.findings],
            "residual_report": self.residual_report.to_dict(),
            "carrier_actions": [action.to_dict() for action in self.carrier_actions],
            "pixel_status": self.pixel_status.value,
        }


@dataclass
class _Context:
    date_shift_days: int
    keep_year: bool
    uid_salt: bytes
    actions: list[DicomHeaderAction] = field(default_factory=list)
    uid_map: dict[str, str] = field(default_factory=dict)
    private_tag_removed_count: int = 0
    processed_document_digests: set[str] = field(default_factory=set)
    profile_options: tuple[str, ...] = ()
    detector: Any = None


# Pinned normative metadata only: DICOM PS3.15 2026d Tables E.1-1 and E.3.4-1.
# Source: https://dicom.nema.org/medical/dicom/2026d/output/chtml/part15/chapter_E.html
# No patient values, terminology descriptions, or runtime network access.
_BASIC_PROFILE_VERSION = "2026d"
_BASIC_PROFILE_DATA = """
00080050|AccessionNumber|Z
00184000|AcquisitionComments|X
00400556|AcquisitionContextDescription|X
00400555|AcquisitionContextSequence|X/Z
00080022|AcquisitionDate|X/Z
0008002A|AcquisitionDateTime|X/Z/D
00181400|AcquisitionDeviceProcessingDescription|X/D
001811BB|AcquisitionFieldOfViewLabel|D
00189424|AcquisitionProtocolDescription|X
00080032|AcquisitionTime|X/Z
00080017|AcquisitionUID|U
00404035|ActualHumanPerformersSequence|X
001021B0|AdditionalPatientHistory|X
0040A353|AddressTrial|X
00380010|AdmissionID|X
00380020|AdmittingDate|X
00081084|AdmittingDiagnosesCodeSequence|X
00081080|AdmittingDiagnosesDescription|X
00380021|AdmittingTime|X
00001000|AffectedSOPInstanceUID|X
00102110|Allergies|X
0040B034|AnnotationDateTime|X
006A0006|AnnotationGroupDescription|X
006A0005|AnnotationGroupLabel|D
006A0003|AnnotationGroupUID|D
00440004|ApprovalStatusDateTime|X
40000010|Arbitrary|X
00440104|AssertionDateTime|D
00440105|AssertionExpirationDateTime|X
04000562|AttributeModificationDateTime|D
0040A078|AuthorObserverSequence|X
22000005|BarcodeValue|X/Z
300A00C3|BeamDescription|X
300C0127|BeamHoldTransitionDateTime|D
300A00DD|BolusDescription|X
00101081|BranchOfService|X
0014407E|CalibrationDate|X
00181203|CalibrationDateTime|Z
0014407C|CalibrationTime|X
0016004D|CameraOwnerName|X
00181007|CassetteID|X
04000115|CertificateOfSigner|D
04000310|CertifiedTimestamp|X
003A020C|ChannelDerivationDescription|X
003A0203|ChannelLabel|X
00120060|ClinicalTrialCoordinatingCenterName|Z
00120082|ClinicalTrialProtocolEthicsCommitteeApprovalNumber|X
00120081|ClinicalTrialProtocolEthicsCommitteeName|D
00120020|ClinicalTrialProtocolID|D
00120021|ClinicalTrialProtocolName|Z
00120072|ClinicalTrialSeriesDescription|X
00120071|ClinicalTrialSeriesID|X
00120030|ClinicalTrialSiteID|Z
00120031|ClinicalTrialSiteName|Z
00120010|ClinicalTrialSponsorName|D
00120040|ClinicalTrialSubjectID|D
00120042|ClinicalTrialSubjectReadingID|D
00120051|ClinicalTrialTimePointDescription|X
00120050|ClinicalTrialTimePointID|Z
00400310|CommentsOnRadiationDose|X
00400280|CommentsOnThePerformedProcedureStep|X
300A02EB|CompensatorDescription|X
00209161|ConcatenationUID|U
3010000F|ConceptualVolumeCombinationDescription|Z
30100017|ConceptualVolumeDescription|Z
30100006|ConceptualVolumeUID|U
00403001|ConfidentialityConstraintOnPatientDataDescription|X
30100013|ConstituentConceptualVolumeUID|U
0008009C|ConsultingPhysicianName|Z
0008009D|ConsultingPhysicianIdentificationSequence|X
0050001B|ContainerComponentID|X
0040051A|ContainerDescription|X
00400512|ContainerIdentifier|D
00700086|ContentCreatorIdentificationCodeSequence|X
00700084|ContentCreatorName|Z/D
00080023|ContentDate|Z/D
0040A730|ContentSequence|D
00080033|ContentTime|Z/D
00080107|ContextGroupLocalVersion|D
00080106|ContextGroupVersion|D
00180010|ContrastBolusAgent|Z/D
00181042|ContrastBolusStartTime|X
00181043|ContrastBolusStopTime|X
0018A002|ContributionDateTime|X
0018A003|ContributionDescription|X
00102150|CountryOfResidence|X
21000040|CreationDate|X
21000050|CreationTime|X
0040A307|CurrentObserverTrial|X
00380300|CurrentPatientLocation|X
50XXXXXX|CurveData|X
00080025|CurveDate|X
00080035|CurveTime|X
0040A07C|CustodialOrganizationSequence|X
FFFCFFFC|DataSetTrailingPadding|X
0040A121|Date|D
0040A110|DateOfDocumentOrVerbalTransactionTrial|X
00181205|DateOfInstallation|X
00181200|DateOfLastCalibration|X
0018700C|DateOfLastDetectorCalibration|X/D
00181204|DateOfManufacture|X
00181012|DateOfSecondaryCapture|X
0040A120|DateTime|D
00181202|DateTimeOfLastCalibration|X
00189701|DecayCorrectionDateTime|D
0018937F|DecompositionDescription|X
00082111|DerivationDescription|X
21000140|DestinationAE|D
0018700A|DetectorID|X/D
3010001B|DeviceAlternateIdentifier|Z
00500020|DeviceDescription|X
3010002D|DeviceLabel|D
00181000|DeviceSerialNumber|X/Z/D
0016004B|DeviceSettingDescription|X
00181002|DeviceUID|U
04000105|DigitalSignatureDateTime|D
FFFAFFFA|DigitalSignaturesSequence|X
04000100|DigitalSignatureUID|U
00209164|DimensionOrganizationUID|U
00380030|DischargeDate|X
00380040|DischargeDiagnosisDescription|X
00380032|DischargeTime|X
300A079A|DisplacementReferenceLabel|X
0040E012|DisplayURI|X
4008011A|DistributionAddress|X
40080119|DistributionName|X
3004007F|DoseCalculationModelName|X
300A0016|DoseReferenceDescription|X
300A0013|DoseReferenceUID|U
3010006E|DosimetricObjectiveUID|U
00686226|EffectiveDateTime|D
0040A034|EffectiveStartDateTime|X
0040A035|EffectiveStopDateTime|X
00420011|EncapsulatedDocument|D
00189517|EndAcquisitionDateTime|X/D
30100037|EntityDescription|X
30100035|EntityLabel|D
30100038|EntityLongLabel|D
30100036|EntityName|X
300A0676|EquipmentFrameOfReferenceDescription|X
00120087|EthicsCommitteeApprovalEffectivenessEndDate|X
00120086|EthicsCommitteeApprovalEffectivenessStartDate|X
00102160|EthnicGroup|X
00102161|EthnicGroupCodeSequence|X
00102162|EthnicGroups|X
00189804|ExclusionStartDateTime|D
00404011|ExpectedCompletionDateTime|X
00080058|FailedSOPInstanceUIDList|U
0070031A|FiducialUID|U
00402017|FillerOrderNumberImagingServiceRequest|Z
003A032B|FilterLookupTableDescription|X
0040A023|FindingsGroupRecordingDateTrial|X
0040A024|FindingsGroupRecordingTimeTrial|X
30080054|FirstTreatmentDate|X/D
300A0196|FixationDeviceDescription|X
00340002|FlowIdentifier|D
00340001|FlowIdentifierSequence|D
3010007F|FractionationNotes|Z
300A0072|FractionGroupDescription|X
00189074|FrameAcquisitionDateTime|D
00209158|FrameComments|X
00200052|FrameOfReferenceUID|U
00340007|FrameOriginTimestamp|D
00189151|FrameReferenceDateTime|D
00189623|FunctionalSyncPulse|D
00181008|GantryID|X
00100044|GenderIdentityCodeSequence|X
00100045|GenderIdentityComment|X
00100041|GenderIdentitySequence|X
00181005|GeneratorID|X
00160076|GPSAltitude|X
00160075|GPSAltitudeRef|X
0016008C|GPSAreaInformation|X
0016008D|GPSDateStamp|X
00160088|GPSDestBearing|X
00160087|GPSDestBearingRef|X
0016008A|GPSDestDistance|X
00160089|GPSDestDistanceRef|X
00160084|GPSDestLatitude|X
00160083|GPSDestLatitudeRef|X
00160086|GPSDestLongitude|X
00160085|GPSDestLongitudeRef|X
0016008E|GPSDifferential|X
0016007B|GPSDOP|X
00160081|GPSImgDirection|X
00160080|GPSImgDirectionRef|X
00160072|GPSLatitude|X
00160071|GPSLatitudeRef|X
00160074|GPSLongitude|X
00160073|GPSLongitudeRef|X
00160082|GPSMapDatum|X
0016007A|GPSMeasureMode|X
0016008B|GPSProcessingMethod|X
00160078|GPSSatellites|X
0016007D|GPSSpeed|X
0016007C|GPSSpeedRef|X
00160079|GPSStatus|X
00160077|GPSTimeStamp|X
0016007F|GPSTrack|X
0016007E|GPSTrackRef|X
00160070|GPSVersionID|X
00700001|GraphicAnnotationSequence|D
0072000A|HangingProtocolCreationDateTime|D
00181011|HardcopyCreationDeviceID|X
00081304|HistologicalDiagnosesCodeSequence|X
0040E004|HL7DocumentEffectiveTime|X
00404037|HumanPerformerName|X
00404036|HumanPerformerOrganization|X
00880200|IconImageSequence|X
00084000|IdentifyingComments|X
00204000|ImageComments|X
00284000|ImagePresentationComments|X
00402400|ImagingServiceRequestComments|X
003A0314|ImpedanceMeasurementDateTime|D
40080300|Impressions|X
00686270|InformationIssueDateTime|D
00080015|InstanceCoercionDateTime|X
00080012|InstanceCreationDate|X/D
00080013|InstanceCreationTime|X/Z/D
00080014|InstanceCreatorUID|U
04000600|InstanceOriginStatus|X
00080081|InstitutionAddress|X
00081040|InstitutionalDepartmentName|X
00081041|InstitutionalDepartmentTypeCodeSequence|X
00080082|InstitutionCodeSequence|X/Z/D
00080080|InstitutionName|X/Z/D
00189919|InstructionPerformedDateTime|Z/D
00101050|InsurancePlanIdentification|X
30100085|IntendedFractionStartTime|X
3010004D|IntendedPhaseEndDate|X/D
3010004C|IntendedPhaseStartDate|X/D
00401011|IntendedRecipientsOfResultsIdentificationSequence|X
300A0741|InterlockDateTime|D
300A0742|InterlockDescription|D
300A0783|InterlockOriginDescription|D
40080112|InterpretationApprovalDate|X
40080113|InterpretationApprovalTime|X
40080111|InterpretationApproverSequence|X
4008010C|InterpretationAuthor|X
40080115|InterpretationDiagnosisDescription|X
40080200|InterpretationID|X
40080202|InterpretationIDIssuer|X
40080100|InterpretationRecordedDate|X
40080101|InterpretationRecordedTime|X
40080102|InterpretationRecorder|X
4008010B|InterpretationText|X
4008010A|InterpretationTranscriber|X
40080108|InterpretationTranscriptionDate|X
40080109|InterpretationTranscriptionTime|X
00180035|InterventionDrugStartTime|X
00180027|InterventionDrugStopTime|X
00083010|IrradiationEventUID|U
00402004|IssueDateOfImagingServiceRequest|X
00380011|IssuerOfAdmissionID|X
00380014|IssuerOfAdmissionIDSequence|X
00120022|IssuerOfClinicalTrialProtocolID|X
00120073|IssuerOfClinicalTrialSeriesID|X
00120032|IssuerOfClinicalTrialSiteID|X
00120041|IssuerOfClinicalTrialSubjectID|X
00120043|IssuerOfClinicalTrialSubjectReadingID|X
00120055|IssuerOfClinicalTrialTimePointID|X
00100021|IssuerOfPatientID|X
00380061|IssuerOfServiceEpisodeID|X
00380064|IssuerOfServiceEpisodeIDSequence|X
00400513|IssuerOfTheContainerIdentifierSequence|Z
00400562|IssuerOfTheSpecimenIdentifierSequence|Z
00402005|IssueTimeOfImagingServiceRequest|X
22000002|LabelText|X/Z
00281214|LargePaletteColorLookupTableUID|U
001021D0|LastMenstrualDate|X
0016004F|LensMake|X
00160050|LensModel|X
00160051|LensSerialNumber|X
0016004E|LensSpecification|X
00500021|LongDeviceDescription|X
04000404|MAC|X
0016002B|MakerNote|X
0018100B|ManufacturerDeviceClassUID|U
30100043|ManufacturerDeviceIdentifier|Z
00020003|MediaStorageSOPInstanceUID|U
00102000|MedicalAlerts|X
00101090|MedicalRecordLocator|X
00101080|MilitaryRank|X
04000550|ModifiedAttributesSequence|X
00203403|ModifiedImageDate|X
00203406|ModifiedImageDescription|X
00203405|ModifiedImageTime|X
00203401|ModifyingDeviceID|X
04000563|ModifyingSystem|D
0040B03F|MontageChannelLabel|X
0040B03B|MontageName|X
30080056|MostRecentTreatmentDate|X/D
0018937B|MultienergyAcquisitionDescription|X
003A0020|MultiplexGroupLabel|X
003A0310|MultiplexGroupUID|U
00081060|NameOfPhysiciansReadingStudy|X
00401010|NamesOfIntendedRecipientsOfResults|X
00100012|NametoUse|X
00100013|NametoUseComment|X
00081000|NetworkID|X
04000552|NonconformingDataElementValue|X
04000551|NonconformingModifiedAttributesSequence|X
0040A192|ObservationDateTrial|X
0040A032|ObservationDateTime|X/D
0040A033|ObservationStartDateTime|X
0040A402|ObservationSubjectUIDTrial|U
0040A193|ObservationTimeTrial|X
0040A171|ObservationUID|U
00102180|Occupation|X
00081072|OperatorIdentificationSequence|X/D
00081070|OperatorsName|X/Z/D
00402010|OrderCallbackPhoneNumber|X
00402011|OrderCallbackTelecomInformation|X
00402008|OrderEnteredBy|X
00402009|OrderEntererLocation|X
04000561|OriginalAttributesSequence|X
21000070|Originator|X
00120023|OtherClinicalTrialProtocolIDsSequence|X
00101000|OtherPatientIDs|X
00101002|OtherPatientIDsSequence|X
00101001|OtherPatientNames|X
60XX4000|OverlayComments|X
60XX3000|OverlayData|X
00080024|OverlayDate|X
00080034|OverlayTime|X
300A0760|OverrideDateTime|D
00281199|PaletteColorLookupTableUID|U
0040A07A|ParticipantSequence|X
0040A082|ParticipationDateTime|Z
00101040|PatientAddress|X
00101010|PatientAge|X
00100030|PatientBirthDate|Z
00101005|PatientBirthName|X
00100032|PatientBirthTime|X
00380400|PatientInstitutionResidence|X
00100050|PatientInsurancePlanCodeSequence|X
00101060|PatientMotherBirthName|X
00100010|PatientName|Z
00100101|PatientPrimaryLanguageCodeSequence|X
00100102|PatientPrimaryLanguageModifierCodeSequence|X
001021F0|PatientReligiousPreference|X
00100040|PatientSex|Z
00102203|PatientSexNeutered|X/Z
00101020|PatientSize|X
00102155|PatientTelecomInformation|X
00102154|PatientTelephoneNumbers|X
00101030|PatientWeight|X
00104000|PatientComments|X
00100020|PatientID|Z/D
300A0794|PatientSetupPhotoDescription|X
300A0650|PatientSetupUID|U
00380500|PatientState|X
00401004|PatientTransportArrangements|X
300A0792|PatientTreatmentPreparationMethodDescription|X
300A078E|PatientTreatmentPreparationProcedureParameterDescription|X
00400243|PerformedLocation|X
00400254|PerformedProcedureStepDescription|X
00400250|PerformedProcedureStepEndDate|X
00404051|PerformedProcedureStepEndDateTime|X
00400251|PerformedProcedureStepEndTime|X
00400253|PerformedProcedureStepID|X
00400244|PerformedProcedureStepStartDate|X
00404050|PerformedProcedureStepStartDateTime|X
00400245|PerformedProcedureStepStartTime|X
00400241|PerformedStationAETitle|X
00404030|PerformedStationGeographicLocationCodeSequence|X
00400242|PerformedStationName|X
00404028|PerformedStationNameCodeSequence|X
00081050|PerformingPhysicianName|X
00081052|PerformingPhysicianIdentificationSequence|X
00401102|PersonAddress|X
00401104|PersonTelecomInformation|X
00401103|PersonTelephoneNumbers|X
00401101|PersonIdentificationCodeSequence|D
0040A123|PersonName|D
00100011|PersonNamestoUseSequence|X
00081048|PhysiciansOfRecord|X
00081049|PhysiciansOfRecordIdentificationSequence|X
00081062|PhysiciansReadingStudyIdentificationSequence|X
40080114|PhysicianApprovingInterpretation|X
00402016|PlacerOrderNumberImagingServiceRequest|Z
00181004|PlateID|X
30020123|PositionAcquisitionTemplateDescription|X
30020121|PositionAcquisitionTemplateName|X
001021C0|PregnancyStatus|X
00400012|PreMedication|X
300A000E|PrescriptionDescription|X
3010007B|PrescriptionNotes|Z
30100081|PrescriptionNotesSequence|Z
00700082|PresentationCreationDate|X
00700083|PresentationCreationTime|X
00701101|PresentationDisplayCollectionUID|U
00701102|PresentationSequenceCollectionUID|U
00081302|PrimaryDiagnosisCodeSequence|X
00081301|PrincipalDiagnosisCodeSequence|X
30100061|PriorTreatmentDoseDescription|X
PRIVATE|PrivateAttributes|X
00404052|ProcedureStepCancellationDateTime|X
0044000B|ProductExpirationDateTime|X
00100015|PronounCodeSequence|X
00100016|PronounComment|X
00181030|ProtocolName|X/D
00081088|PyramidDescription|X
00200027|PyramidLabel|X
00080019|PyramidUID|U
300A0619|RadiationDoseIdentificationLabel|D
300A0623|RadiationDoseInVivoMeasurementLabel|D
300A067D|RadiationGenerationModeDescription|Z
300A067C|RadiationGenerationModeLabel|D
00181078|RadiopharmaceuticalStartDateTime|X
00181072|RadiopharmaceuticalStartTime|X
00181079|RadiopharmaceuticalStopDateTime|X
00181073|RadiopharmaceuticalStopTime|X
300C0113|ReasonForOmissionDescription|X
0040100A|ReasonForRequestedProcedureCodeSequence|X
00321030|ReasonForStudy|X
3010005C|ReasonForSuperseding|Z
04000565|ReasonForTheAttributeModification|D
00402001|ReasonForTheImagingServiceRequest|X
00401002|ReasonForTheRequestedProcedure|X
00321066|ReasonForVisit|X
00321067|ReasonForVisitCodeSequence|X
00741234|ReceivingAE|X
300A073A|RecordedRTControlPointDateTime|D
3010000B|ReferencedConceptualVolumeUID|U
0040A13A|ReferencedDateTime|D
04000402|ReferencedDigitalSignatureSequence|X
300A0083|ReferencedDoseReferenceUID|U
3010006F|ReferencedDosimetricObjectiveUID|U
30100031|ReferencedFiducialsUID|U
30060024|ReferencedFrameOfReferenceUID|U
00404023|ReferencedGeneralPurposeScheduledProcedureStepTransactionUID|U
00081140|ReferencedImageSequence|X/Z/U*
0040A172|ReferencedObservationUIDTrial|U
00380004|ReferencedPatientAliasSequence|X
00101100|ReferencedPatientPhotoSequence|X
00081120|ReferencedPatientSequence|X
00081111|ReferencedPerformedProcedureStepSequence|X/Z/D
04000403|ReferencedSOPInstanceMACSequence|X
00081155|ReferencedSOPInstanceUID|U
00041511|ReferencedSOPInstanceUIDInFile|U
00081110|ReferencedStudySequence|X/Z
300A0785|ReferencedTreatmentPositionGroupUID|U
00080092|ReferringPhysicianAddress|X
00080090|ReferringPhysicianName|Z
00080094|ReferringPhysicianTelephoneNumbers|X
00080096|ReferringPhysicianIdentificationSequence|X
00102152|RegionOfResidence|X
300600C2|RelatedFrameOfReferenceUID|U
00400275|RequestAttributesSequence|X
00321070|RequestedContrastAgent|X
00401400|RequestedProcedureComments|X
00321060|RequestedProcedureDescription|X/Z
00401001|RequestedProcedureID|X
00401005|RequestedProcedureLocation|X
00189937|RequestedSeriesDescription|X
00001001|RequestedSOPInstanceUID|U
00741236|RequestingAE|X
00321032|RequestingPhysician|X
00321033|RequestingService|X
00189185|RespiratoryMotionCompensationTechniqueDescription|X
00102299|ResponsibleOrganization|X
00102297|ResponsiblePerson|X
40084000|ResultsComments|X
40080118|ResultsDistributionListSequence|X
40080040|ResultsID|X
40080042|ResultsIDIssuer|X
00080054|RetrieveAETitle|X
300E0004|ReviewDate|Z
300E0008|ReviewerName|X/Z
300E0005|ReviewTime|Z
3006004D|ROICreatorSequence|X
3006002D|ROIDateTime|X
30060028|ROIDescription|X
30060038|ROIGenerationDescription|X
300600A6|ROIInterpreter|Z
3006004E|ROIInterpreterSequence|X
30060026|ROIName|Z
3006002E|ROIObservationDateTime|X
30060088|ROIObservationDescription|X
30060085|ROIObservationLabel|X
300A0615|RTAccessoryDeviceSlotID|Z
300A0611|RTAccessoryHolderSlotID|Z
3010005A|RTPhysicianIntentNarrative|Z
300A0006|RTPlanDate|X/D
300A0004|RTPlanDescription|X
300A0002|RTPlanLabel|D
300A0003|RTPlanName|X
300A0007|RTPlanTime|X/D
30100054|RTPrescriptionLabel|D
300A062A|RTToleranceSetLabel|D
30100056|RTTreatmentApproachLabel|X/D
3010003B|RTTreatmentPhaseUID|U
30080162|SafePositionExitDate|D
30080164|SafePositionExitTime|D
30080166|SafePositionReturnDate|D
30080168|SafePositionReturnTime|D
0038001A|ScheduledAdmissionDate|X
0038001B|ScheduledAdmissionTime|X
0038001C|ScheduledDischargeDate|X
0038001D|ScheduledDischargeTime|X
00404034|ScheduledHumanPerformersSequence|X
0038001E|ScheduledPatientInstitutionResidence|X
00400006|ScheduledPerformingPhysicianName|X
0040000B|ScheduledPerformingPhysicianIdentificationSequence|X
00400007|ScheduledProcedureStepDescription|X
00400004|ScheduledProcedureStepEndDate|X
00400005|ScheduledProcedureStepEndTime|X
00404008|ScheduledProcedureStepExpirationDateTime|X
00400009|ScheduledProcedureStepID|X
00400011|ScheduledProcedureStepLocation|X
00404010|ScheduledProcedureStepModificationDateTime|X
00400002|ScheduledProcedureStepStartDate|X
00404005|ScheduledProcedureStepStartDateTime|X
00400003|ScheduledProcedureStepStartTime|X
00400001|ScheduledStationAETitle|X
00404027|ScheduledStationGeographicLocationCodeSequence|X
00400010|ScheduledStationName|X
00404025|ScheduledStationNameCodeSequence|X
00321020|ScheduledStudyLocation|X
00321021|ScheduledStudyLocationAETitle|X
00321000|ScheduledStudyStartDate|X
00321001|ScheduledStudyStartTime|X
00321010|ScheduledStudyStopDate|X
00321011|ScheduledStudyStopTime|X
00181010|SecondaryCaptureDeviceID|X
00081303|SecondaryDiagnosesCodeSequence|X
0040B036|SegmentDefinitionDateTime|X
0072005E|SelectorAEValue|D
0072005F|SelectorASValue|D
00720061|SelectorDAValue|D
00720063|SelectorDTValue|D
00720066|SelectorLOValue|D
00720068|SelectorLTValue|D
00720065|SelectorOBValue|D
0072006A|SelectorPNValue|D
0072006C|SelectorSHValue|D
0072006E|SelectorSTValue|D
0072006B|SelectorTMValue|D
0072006D|SelectorUNValue|D
00720071|SelectorURValue|D
00720070|SelectorUTValue|D
00080021|SeriesDate|X/D
0008103E|SeriesDescription|X
0020000E|SeriesInstanceUID|U
00080031|SeriesTime|X/D
00380062|ServiceEpisodeDescription|X
00380060|ServiceEpisodeID|X
300A01B2|SetupTechniqueDescription|X
00100046|SexParametersforClinicalUseCategoryCodeSequence|X
00100042|SexParametersforClinicalUseCategoryComment|X
00100047|SexParametersforClinicalUseCategoryReference|X
00100043|SexParametersforClinicalUseCategorySequence|X
300A01A6|ShieldingDeviceDescription|X
004006FA|SlideIdentifier|X
001021A0|SmokingStatus|X
01000420|SOPAuthorizationDateTime|X
00080018|SOPInstanceUID|U
30100015|SourceConceptualVolumeUID|U
0018936A|SourceEndDateTime|D
00640003|SourceFrameOfReferenceUID|U
00340005|SourceIdentifier|D
00082112|SourceImageSequence|X/Z/U*
300A0216|SourceManufacturer|X
04000564|SourceOfPreviousValues|Z
30080105|SourceSerialNumber|X/Z
00189369|SourceStartDateTime|D
300A022C|SourceStrengthReferenceDate|D
300A022E|SourceStrengthReferenceTime|D
00380050|SpecialNeeds|X
0040050A|SpecimenAccessionNumber|X
00400602|SpecimenDetailedDescription|X
00400551|SpecimenIdentifier|D
00400610|SpecimenPreparationSequence|Z
00400600|SpecimenShortDescription|X
00400554|SpecimenUID|U
00189516|StartAcquisitionDateTime|X/D
00080055|StationAETitle|X
00081010|StationName|X/Z/D
00880140|StorageMediaFileSetUID|U
30060008|StructureSetDate|Z
30060006|StructureSetDescription|X
30060002|StructureSetLabel|D
30060004|StructureSetName|X
30060009|StructureSetTime|Z
00321040|StudyArrivalDate|X
00321041|StudyArrivalTime|X
00324000|StudyComments|X
00321050|StudyCompletionDate|X
00321051|StudyCompletionTime|X
00080020|StudyDate|Z
00081030|StudyDescription|X
00200010|StudyID|Z
00320012|StudyIDIssuer|X
0020000D|StudyInstanceUID|U
00320034|StudyReadDate|X
00320035|StudyReadTime|X
00080030|StudyTime|Z
00320032|StudyVerifiedDate|X
00320033|StudyVerifiedTime|X
00440010|SubstanceAdministrationDateTime|X
00200200|SynchronizationFrameOfReferenceUID|U
300A0054|TableTopPositionAlignmentUID|U
00182042|TargetUID|U
0040A354|TelephoneNumberTrial|X
0040DB0D|TemplateExtensionCreatorUID|U
0040DB0C|TemplateExtensionOrganizationUID|U
0040DB07|TemplateLocalVersion|X
0040DB06|TemplateVersion|X
40004000|TextComments|X
20300020|TextString|X
00100014|ThirdPersonPronounsSequence|X
0040A122|Time|D
0040A112|TimeOfDocumentCreationOrVerbalTransactionTrial|X
00181201|TimeOfLastCalibration|X
0018700E|TimeOfLastDetectorCalibration|X/D
00181014|TimeOfSecondaryCapture|X
00080201|TimezoneOffsetFromUTC|X
00880910|TopicAuthor|X
00880912|TopicKeywords|X
00880906|TopicSubject|X
00880904|TopicTitle|X
00620021|TrackingUID|U
00081195|TransactionUID|U
00185011|TransducerIdentificationSequence|X
30080024|TreatmentControlPointDate|D
30080025|TreatmentControlPointTime|D
30080250|TreatmentDate|X/D
300A00B2|TreatmentMachineName|X/Z
300A0608|TreatmentPositionGroupLabel|D
300A0609|TreatmentPositionGroupUID|U
300A0700|TreatmentSessionUID|U
30100077|TreatmentSite|X/D
300A000B|TreatmentSites|X
3010007A|TreatmentTechniqueNotes|Z
30080251|TreatmentTime|X/D
300A0736|TreatmentToleranceViolationDateTime|D
300A0734|TreatmentToleranceViolationDescription|D
0018100A|UDISequence|X
0040A124|UID|U
00700006|UnformattedTextValue|D
00181009|UniqueDeviceIdentifier|X
30100033|UserContentLabel|D
30100034|UserContentLongLabel|D
0040A352|VerbalSourceTrial|X
0040A358|VerbalSourceIdentifierCodeSequenceTrial|X
0040A030|VerificationDateTime|D
0040A088|VerifyingObserverIdentificationCodeSequence|Z
0040A075|VerifyingObserverName|D
0040A073|VerifyingObserverSequence|D
0040A027|VerifyingOrganization|D
00384000|VisitComments|X
0040B020|WaveformAnnotationSequence|X/D
003A0329|WaveformFilterDescription|X
00189371|XRayDetectorID|D
00189373|XRayDetectorLabel|X
00189367|XRaySourceID|D
"""
_BASIC_PROFILE_CATALOG = tuple(
    tuple(row.split("|")) for row in _BASIC_PROFILE_DATA.strip().splitlines()
)
_BASIC_PROFILE_ACTIONS = {
    tag: action for tag, _keyword, action in _BASIC_PROFILE_CATALOG
}
_PROFILE_OPTION_DATA = """
00184000|clean_descriptors:C
00400556|clean_descriptors:C
00400555|clean_structured_content:C
00080022|retain_longitudinal_temporal_information:C
0008002A|retain_longitudinal_temporal_information:C
00181400|clean_descriptors:C
001811BB|clean_descriptors:C
00189424|clean_descriptors:C
00080032|retain_longitudinal_temporal_information:C
00080017|retain_uids:K
001021B0|clean_descriptors:C
00380020|retain_longitudinal_temporal_information:C
00081084|clean_descriptors:C
00081080|clean_descriptors:C
00380021|retain_longitudinal_temporal_information:C
00001000|retain_uids:K
00102110|retain_patient_characteristics:C|clean_descriptors:C
0040B034|retain_longitudinal_temporal_information:C
006A0006|clean_descriptors:C
006A0005|clean_descriptors:C
006A0003|retain_uids:K
00440004|retain_longitudinal_temporal_information:C
00440104|retain_longitudinal_temporal_information:C
00440105|retain_longitudinal_temporal_information:C
04000562|retain_longitudinal_temporal_information:C
300A00C3|clean_descriptors:C
300C0127|retain_device_identity:K|retain_longitudinal_temporal_information:C
300A00DD|clean_descriptors:C
0014407E|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181203|retain_device_identity:K|retain_longitudinal_temporal_information:C
0014407C|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181007|retain_device_identity:K
04000310|retain_longitudinal_temporal_information:C
003A020C|clean_descriptors:C
003A0203|clean_descriptors:C
00120072|clean_descriptors:C
00120051|clean_descriptors:C
00400310|clean_descriptors:C
00400280|clean_descriptors:C
300A02EB|clean_descriptors:C
00209161|retain_uids:K
3010000F|clean_descriptors:C
30100017|clean_descriptors:C
30100006|retain_uids:K
30100013|retain_uids:K
0040051A|clean_descriptors:C
00080023|retain_longitudinal_temporal_information:C
0040A730|clean_structured_content:C
00080033|retain_longitudinal_temporal_information:C
00080107|retain_longitudinal_temporal_information:C
00080106|retain_longitudinal_temporal_information:C
00180010|clean_descriptors:C
00181042|retain_longitudinal_temporal_information:C
00181043|retain_longitudinal_temporal_information:C
0018A002|retain_longitudinal_temporal_information:C
0018A003|clean_descriptors:C
21000040|retain_longitudinal_temporal_information:C
21000050|retain_longitudinal_temporal_information:C
00080025|retain_longitudinal_temporal_information:C
00080035|retain_longitudinal_temporal_information:C
0040A121|retain_longitudinal_temporal_information:C
0040A110|retain_longitudinal_temporal_information:C
00181205|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181200|retain_device_identity:K|retain_longitudinal_temporal_information:C
0018700C|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181204|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181012|retain_longitudinal_temporal_information:C
0040A120|retain_longitudinal_temporal_information:C
00181202|retain_device_identity:K|retain_longitudinal_temporal_information:C
00189701|retain_longitudinal_temporal_information:C
0018937F|clean_descriptors:C
00082111|clean_descriptors:C
21000140|retain_device_identity:C
0018700A|retain_device_identity:K
00500020|retain_device_identity:K
3010002D|retain_device_identity:K
00181000|retain_device_identity:K
0016004B|clean_descriptors:C
00181002|retain_uids:K|retain_device_identity:K
04000105|retain_longitudinal_temporal_information:C
00209164|retain_uids:K
00380030|retain_longitudinal_temporal_information:C
00380040|clean_descriptors:C
00380032|retain_longitudinal_temporal_information:C
300A079A|clean_descriptors:C
3004007F|clean_descriptors:C
300A0016|clean_descriptors:C
300A0013|retain_uids:K
3010006E|retain_uids:K
00686226|retain_longitudinal_temporal_information:C
0040A034|retain_longitudinal_temporal_information:C
0040A035|retain_longitudinal_temporal_information:C
00189517|retain_longitudinal_temporal_information:C
30100037|clean_descriptors:C
30100035|clean_descriptors:C
30100038|clean_descriptors:C
30100036|clean_descriptors:C
300A0676|clean_descriptors:C
00120087|retain_longitudinal_temporal_information:C
00120086|retain_longitudinal_temporal_information:C
00102160|retain_patient_characteristics:K
00102161|retain_patient_characteristics:K
00102162|retain_patient_characteristics:K
00189804|retain_longitudinal_temporal_information:C
00404011|retain_longitudinal_temporal_information:C
00080058|retain_uids:K
0070031A|retain_uids:K
003A032B|clean_descriptors:C
0040A023|retain_longitudinal_temporal_information:C
0040A024|retain_longitudinal_temporal_information:C
30080054|retain_longitudinal_temporal_information:C
300A0196|clean_descriptors:C
3010007F|clean_descriptors:C
300A0072|clean_descriptors:C
00189074|retain_longitudinal_temporal_information:C
00209158|clean_descriptors:C
00200052|retain_uids:K
00340007|retain_longitudinal_temporal_information:C
00189151|retain_longitudinal_temporal_information:C
00189623|retain_longitudinal_temporal_information:C
00181008|retain_device_identity:K
00100045|clean_descriptors:C
00181005|retain_device_identity:K
0016008D|retain_longitudinal_temporal_information:C
0072000A|retain_longitudinal_temporal_information:C
00181011|retain_device_identity:K
00081304|clean_descriptors:C
0040E004|retain_longitudinal_temporal_information:C
00084000|clean_descriptors:C
00204000|clean_descriptors:C
00402400|clean_descriptors:C
003A0314|retain_longitudinal_temporal_information:C
40080300|clean_descriptors:C
00686270|retain_longitudinal_temporal_information:C
00080015|retain_longitudinal_temporal_information:C
00080012|retain_longitudinal_temporal_information:C
00080013|retain_longitudinal_temporal_information:C
00080014|retain_uids:K
00189919|retain_longitudinal_temporal_information:C
30100085|retain_longitudinal_temporal_information:C
3010004D|retain_longitudinal_temporal_information:C
3010004C|retain_longitudinal_temporal_information:C
300A0741|retain_longitudinal_temporal_information:C
300A0742|clean_descriptors:C
300A0783|clean_descriptors:C
40080112|retain_longitudinal_temporal_information:C
40080113|retain_longitudinal_temporal_information:C
40080115|clean_descriptors:C
40080100|retain_longitudinal_temporal_information:C
40080101|retain_longitudinal_temporal_information:C
4008010B|clean_descriptors:C
40080108|retain_longitudinal_temporal_information:C
40080109|retain_longitudinal_temporal_information:C
00180035|retain_longitudinal_temporal_information:C
00180027|retain_longitudinal_temporal_information:C
00083010|retain_uids:K
00402004|retain_longitudinal_temporal_information:C
00402005|retain_longitudinal_temporal_information:C
22000002|clean_descriptors:C
00281214|retain_uids:K
001021D0|retain_longitudinal_temporal_information:C
0016004F|retain_device_identity:K
00160050|retain_device_identity:K
00160051|retain_device_identity:K
0016004E|retain_device_identity:K
00500021|clean_descriptors:C
0016002B|clean_descriptors:C
0018100B|retain_uids:K|retain_device_identity:K
30100043|retain_device_identity:K
00020003|retain_uids:K
00102000|clean_descriptors:C
00203403|retain_longitudinal_temporal_information:C
00203405|retain_longitudinal_temporal_information:C
00203401|retain_device_identity:K
04000563|retain_device_identity:K
0040B03F|clean_descriptors:C
0040B03B|clean_descriptors:C
30080056|retain_longitudinal_temporal_information:C
0018937B|clean_descriptors:C
003A0020|clean_descriptors:C
003A0310|retain_uids:K
00081000|retain_device_identity:C
0040A192|retain_longitudinal_temporal_information:C
0040A032|retain_longitudinal_temporal_information:C
0040A033|retain_longitudinal_temporal_information:C
0040A402|retain_uids:K
0040A193|retain_longitudinal_temporal_information:C
0040A171|retain_uids:K
00102180|clean_descriptors:C
21000070|retain_device_identity:C
00080024|retain_longitudinal_temporal_information:C
00080034|retain_longitudinal_temporal_information:C
300A0760|retain_longitudinal_temporal_information:C
00281199|retain_uids:K
0040A082|retain_longitudinal_temporal_information:C
00101010|retain_patient_characteristics:K
00100040|retain_patient_characteristics:K
00102203|retain_patient_characteristics:K
00101020|retain_patient_characteristics:K
00101030|retain_patient_characteristics:K
00104000|clean_descriptors:C
300A0794|clean_descriptors:C
300A0650|retain_uids:K
00380500|retain_patient_characteristics:C|clean_descriptors:C
300A0792|clean_descriptors:C
300A078E|clean_descriptors:C
00400254|clean_descriptors:C
00400250|retain_longitudinal_temporal_information:C
00404051|retain_longitudinal_temporal_information:C
00400251|retain_longitudinal_temporal_information:C
00400244|retain_longitudinal_temporal_information:C
00404050|retain_longitudinal_temporal_information:C
00400245|retain_longitudinal_temporal_information:C
00400241|retain_device_identity:C
00404030|retain_device_identity:K
00400242|retain_device_identity:K
00404028|retain_device_identity:K
00181004|retain_device_identity:K
30020123|clean_descriptors:C
30020121|clean_descriptors:C
001021C0|retain_patient_characteristics:K
00400012|retain_patient_characteristics:C
300A000E|clean_descriptors:C
3010007B|clean_descriptors:C
30100081|clean_descriptors:C
00700082|retain_longitudinal_temporal_information:C
00700083|retain_longitudinal_temporal_information:C
00701101|retain_uids:K
00701102|retain_uids:K
00081302|clean_descriptors:C
00081301|clean_descriptors:C
30100061|clean_descriptors:C
00404052|retain_longitudinal_temporal_information:C
0044000B|retain_longitudinal_temporal_information:C
00100016|clean_descriptors:C
00181030|clean_descriptors:C
00081088|clean_descriptors:C
00200027|clean_descriptors:C
00080019|retain_uids:K
300A0619|clean_descriptors:C
300A0623|clean_descriptors:C
300A067D|clean_descriptors:C
300A067C|clean_descriptors:C
00181078|retain_longitudinal_temporal_information:C
00181072|retain_longitudinal_temporal_information:C
00181079|retain_longitudinal_temporal_information:C
00181073|retain_longitudinal_temporal_information:C
300C0113|clean_descriptors:C
0040100A|clean_descriptors:C
00321030|clean_descriptors:C
3010005C|clean_descriptors:C
04000565|clean_descriptors:C
00402001|clean_descriptors:C
00401002|clean_descriptors:C
00321066|clean_descriptors:C
00321067|clean_descriptors:C
00741234|retain_device_identity:C
300A073A|retain_longitudinal_temporal_information:C
3010000B|retain_uids:K
0040A13A|retain_longitudinal_temporal_information:C
300A0083|retain_uids:K
3010006F|retain_uids:K
30100031|retain_uids:K
30060024|retain_uids:K
00404023|retain_uids:K
00081140|retain_uids:K
0040A172|retain_uids:K
00081120|retain_uids:K
00081111|retain_uids:K
00081155|retain_uids:K
00041511|retain_uids:K
00081110|retain_uids:K
300A0785|retain_uids:K
300600C2|retain_uids:K
00400275|clean_descriptors:C
00321070|clean_descriptors:C
00401400|clean_descriptors:C
00321060|clean_descriptors:C
00189937|clean_descriptors:C
00001001|retain_uids:K
00741236|retain_device_identity:C
00321033|clean_descriptors:C
00189185|clean_descriptors:C
40084000|clean_descriptors:C
00080054|retain_device_identity:C
300E0004|retain_longitudinal_temporal_information:C
300E0005|retain_longitudinal_temporal_information:C
3006002D|retain_longitudinal_temporal_information:C
30060028|clean_descriptors:C
30060038|clean_descriptors:C
30060026|clean_descriptors:C
3006002E|retain_longitudinal_temporal_information:C
30060088|clean_descriptors:C
30060085|clean_descriptors:C
3010005A|clean_descriptors:C
300A0006|retain_longitudinal_temporal_information:C
300A0004|clean_descriptors:C
300A0002|clean_descriptors:C
300A0003|clean_descriptors:C
300A0007|retain_longitudinal_temporal_information:C
30100054|clean_descriptors:C
300A062A|clean_descriptors:C
30100056|clean_descriptors:C
3010003B|retain_uids:K
30080162|retain_longitudinal_temporal_information:C
30080164|retain_longitudinal_temporal_information:C
30080166|retain_longitudinal_temporal_information:C
30080168|retain_longitudinal_temporal_information:C
0038001A|retain_longitudinal_temporal_information:C
0038001B|retain_longitudinal_temporal_information:C
0038001C|retain_longitudinal_temporal_information:C
0038001D|retain_longitudinal_temporal_information:C
00400007|clean_descriptors:C
00400004|retain_longitudinal_temporal_information:C
00400005|retain_longitudinal_temporal_information:C
00404008|retain_longitudinal_temporal_information:C
00400011|retain_device_identity:K
00404010|retain_longitudinal_temporal_information:C
00400002|retain_longitudinal_temporal_information:C
00404005|retain_longitudinal_temporal_information:C
00400003|retain_longitudinal_temporal_information:C
00400001|retain_device_identity:C
00404027|retain_device_identity:K
00400010|retain_device_identity:K
00404025|retain_device_identity:K
00321020|retain_device_identity:K
00321021|retain_device_identity:C
00321000|retain_longitudinal_temporal_information:C
00321001|retain_longitudinal_temporal_information:C
00321010|retain_longitudinal_temporal_information:C
00321011|retain_longitudinal_temporal_information:C
00181010|retain_device_identity:K
00081303|clean_descriptors:C
0040B036|retain_longitudinal_temporal_information:C
0072005E|retain_device_identity:C
0072005F|retain_patient_characteristics:K
00720061|retain_longitudinal_temporal_information:C
00720063|retain_longitudinal_temporal_information:C
00720066|clean_descriptors:C
00720068|clean_descriptors:C
0072006C|clean_descriptors:C
0072006E|clean_descriptors:C
0072006B|retain_longitudinal_temporal_information:C
00720070|clean_descriptors:C
00080021|retain_longitudinal_temporal_information:C
0008103E|clean_descriptors:C
0020000E|retain_uids:K
00080031|retain_longitudinal_temporal_information:C
00380062|clean_descriptors:C
300A01B2|clean_descriptors:C
00100046|retain_patient_characteristics:K
00100042|retain_patient_characteristics:C|clean_descriptors:C
00100047|retain_patient_characteristics:K
00100043|retain_patient_characteristics:K
300A01A6|clean_descriptors:C
001021A0|retain_patient_characteristics:K
01000420|retain_longitudinal_temporal_information:C
00080018|retain_uids:K
30100015|retain_uids:K
0018936A|retain_longitudinal_temporal_information:C
00640003|retain_uids:K
00082112|retain_uids:K
300A0216|retain_device_identity:K
30080105|retain_device_identity:K
00189369|retain_longitudinal_temporal_information:C
300A022C|retain_longitudinal_temporal_information:C
300A022E|retain_longitudinal_temporal_information:C
00380050|retain_patient_characteristics:C
00400602|clean_descriptors:C
00400610|clean_structured_content:C
00400600|clean_descriptors:C
00400554|retain_uids:K
00189516|retain_longitudinal_temporal_information:C
00080055|retain_device_identity:C
00081010|retain_device_identity:K
00880140|retain_uids:K
30060008|retain_longitudinal_temporal_information:C
30060006|clean_descriptors:C
30060002|clean_descriptors:C
30060004|clean_descriptors:C
30060009|retain_longitudinal_temporal_information:C
00321040|retain_longitudinal_temporal_information:C
00321041|retain_longitudinal_temporal_information:C
00324000|clean_descriptors:C
00321050|retain_longitudinal_temporal_information:C
00321051|retain_longitudinal_temporal_information:C
00080020|retain_longitudinal_temporal_information:C
00081030|clean_descriptors:C
0020000D|retain_uids:K
00320034|retain_longitudinal_temporal_information:C
00320035|retain_longitudinal_temporal_information:C
00080030|retain_longitudinal_temporal_information:C
00320032|retain_longitudinal_temporal_information:C
00320033|retain_longitudinal_temporal_information:C
00440010|retain_longitudinal_temporal_information:C
00200200|retain_uids:K
300A0054|retain_uids:K
00182042|retain_uids:K
0040DB0D|retain_uids:K
0040DB0C|retain_uids:K
0040DB07|retain_longitudinal_temporal_information:C
0040DB06|retain_longitudinal_temporal_information:C
0040A122|retain_longitudinal_temporal_information:C
0040A112|retain_longitudinal_temporal_information:C
00181201|retain_device_identity:K|retain_longitudinal_temporal_information:C
0018700E|retain_device_identity:K|retain_longitudinal_temporal_information:C
00181014|retain_longitudinal_temporal_information:C
00080201|retain_longitudinal_temporal_information:C
00620021|retain_uids:K
00081195|retain_uids:K
00185011|retain_device_identity:K
30080024|retain_longitudinal_temporal_information:C
30080025|retain_longitudinal_temporal_information:C
30080250|retain_longitudinal_temporal_information:C
300A00B2|retain_device_identity:K
300A0608|clean_descriptors:C
300A0609|retain_uids:K
300A0700|retain_uids:K
30100077|clean_descriptors:C
300A000B|clean_descriptors:C
3010007A|clean_descriptors:C
30080251|retain_longitudinal_temporal_information:C
300A0736|retain_longitudinal_temporal_information:C
300A0734|clean_descriptors:C
0018100A|retain_device_identity:K
00700006|clean_descriptors:C
00181009|retain_device_identity:K
30100033|clean_descriptors:C
30100034|clean_descriptors:C
0040A030|retain_longitudinal_temporal_information:C
00384000|clean_descriptors:C
0040B020|clean_structured_content:C
003A0329|clean_descriptors:C
00189371|retain_device_identity:K
00189373|retain_device_identity:K
00189367|retain_device_identity:K
"""
_PROFILE_OPTION_ACTIONS = {
    parts[0]: dict(option.split(":") for option in parts[1:])
    for row in _PROFILE_OPTION_DATA.strip().splitlines()
    if (parts := row.split("|"))
}
_STRUCTURED_CONTENT_DATA = """
DCM|121022|TEXT|X
DCM|113795|IMAGE|D|retain_uids:K
DCM|126201|DATE|X|retain_longitudinal_temporal_information:C
DCM|130884|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|125203|TEXT|X|clean_descriptors:C
DCM|126202|TIME|X|retain_longitudinal_temporal_information:C
NCIt|C67447|TEXT|X|clean_descriptors:C
SCT|440252007|TEXT|D|clean_descriptors:C
NCDR|15|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|112050|TEXT|X|clean_descriptors:C
SCT|398164008|DATETIME|X|retain_longitudinal_temporal_information:C
SCT|398325003|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|121080|IMAGE|X|retain_uids:K
DCM|121080|WAVEFORM|X|retain_uids:K
DCM|113723|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|113720|TEXT|X|clean_descriptors:C
DCM|113724|TEXT|D|clean_descriptors:C
NCDR|76|PNAME|D
DCM|121120|COMPOSITE|X|retain_uids:K
SCT|371524004|COMPOSITE|X|retain_uids:K
SCT|371524004|TEXT|X|clean_descriptors:C
DCM|121106|TEXT|X|clean_descriptors:C
SCT|116224001|TEXT|X|clean_descriptors:C
DCM|112347|TEXT|D
DCM|121077|TEXT|X|clean_descriptors:C
DCM|111018|DATE|X/D|retain_longitudinal_temporal_information:C
DCM|111019|TIME|X/D|retain_longitudinal_temporal_information:C
DCM|122073|COMPOSITE|X|retain_uids:K
LN|11955-2|DATE|X|retain_longitudinal_temporal_information:C
DCM|121431|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|121432|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111527|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|122165|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|122105|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111536|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111702|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|121125|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111535|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|121433|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111526|DATETIME|X/D|retain_longitudinal_temporal_information:C
DCM|112363|TEXT|D
DCM|112357|UIDREF|D|retain_uids:K
DCM|112373|COMPOSITE|X|retain_uids:K
DCM|112372|COMPOSITE|X|retain_uids:K
DCM|111021|TEXT|X|clean_descriptors:C
DCM|121145|TEXT|X|clean_descriptors:C
DCM|120999|TEXT|X|retain_device_identity:K
DCM|113877|TEXT|X|retain_device_identity:K
DCM|121013|TEXT|X|retain_device_identity:K
DCM|121017|TEXT|X
DCM|121016|TEXT|X|retain_device_identity:K
DCM|121012|UIDREF|X/D|retain_uids:K|retain_device_identity:K
DCM|113880|TEXT|X/D|retain_device_identity:K
DCM|121193|TEXT|D|retain_device_identity:K
DCM|121197|TEXT|X
DCM|121196|TEXT|X|retain_device_identity:K
DCM|121198|UIDREF|X|retain_uids:K|retain_device_identity:K
DCM|122163|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|121342|IMAGE|X|retain_uids:K
DCM|122083|TEXT|X/D|clean_descriptors:C
DCM|122082|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|122081|DATETIME|X|retain_longitudinal_temporal_information:C
SCT|271921002|TEXT|X/D|clean_descriptors:C
LN|11778-8|DATE|X|retain_longitudinal_temporal_information:C
DCM|113810|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|121122|TEXT|X|retain_device_identity:K|clean_descriptors:C
DCM|128429|UIDREF|D|retain_uids:K
NCIt|C54627|NUM|X
DCM|121088|PNAME|X
LN|11951-1|TEXT|D
DCM|121021|TEXT|X
DCM|121071|TEXT|X/D|clean_descriptors:C
SCT|363698007|TEXT|D|clean_descriptors:C
DCM|112227|UIDREF|X/D|retain_uids:K
DCM|127857|DATE|D|retain_longitudinal_temporal_information:C
DCM|127858|TIME|D|retain_longitudinal_temporal_information:C
LN|11329-0|TEXT|X|clean_descriptors:C
DCM|130527|TEXT|D
DCM|113832|TEXT|D|retain_device_identity:K
DCM|125010|TEXT|X
DCM|128775|TEXT|X
DCM|112229|IMAGE|D|retain_uids:K
DCM|125201|IMAGE|X|retain_uids:K
DCM|121200|IMAGE|X|retain_uids:K
DCM|121138|IMAGE|D|retain_uids:K
DCM|122712|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|112366|COMPOSITE|X|retain_uids:K
DCM|111033|TEXT|D|clean_descriptors:C
LN|18785-6|TEXT|D|clean_descriptors:C
DCM|121154|TEXT|D
DCM|113850|PNAME|D
DCM|113605|TEXT|X
DCM|113769|UIDREF|D|retain_uids:K
DCM|110190|TEXT|X
DCM|111706|TEXT|X
DCM|111724|TEXT|X
DCM|113012|TEXT|X|clean_descriptors:C
LN|18118-0|TEXT|X|clean_descriptors:C
DCM|112371|COMPOSITE|D|retain_uids:K
DCM|112352|TEXT|D
DCM|112351|TEXT|D
DCM|111516|TEXT|X|clean_descriptors:C
DCM|121036|PNAME|X
DCM|113873|TEXT|X
DCM|111040|COMPOSITE|D|retain_uids:K
DCM|111705|TEXT|D
DCM|112361|COMPOSITE|X|retain_uids:K
DCM|112354|IMAGE|X|retain_uids:K
DCM|113815|TEXT|D|clean_descriptors:C
DCM|121110|TEXT|X|clean_descriptors:C
DCM|128425|COMPOSITE|X|retain_uids:K
DCM|128425|IMAGE|X|retain_uids:K
DCM|128425|UIDREF|X|retain_uids:K
DCM|128426|TEXT|X|clean_descriptors:C
DCM|109054|TEXT|X|clean_descriptors:C
DCM|122128|TEXT|X
DCM|121126|UIDREF|D|retain_uids:K
DCM|121114|PNAME|D
DCM|121152|PNAME|X
DCM|113871|TEXT|X
DCM|113872|TEXT|X
DCM|113870|PNAME|D
DCM|128774|TEXT|X
DCM|121009|TEXT|X
DCM|121008|PNAME|D
DCM|121173|TEXT|X|clean_descriptors:C
DCM|121020|TEXT|X
DCM|113516|TEXT|X
DCM|122075|COMPOSITE|X|retain_uids:K
DCM|121124|TEXT|X/D
DCM|122146|DATETIME|X/D|retain_longitudinal_temporal_information:C
NCDR|52|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|121065|TEXT|X/D|clean_descriptors:C
NCDR|53|TEXT|X
DCM|122177|TEXT|X|clean_descriptors:C
DCM|121019|UIDREF|X|retain_uids:K
DCM|121018|UIDREF|X|retain_uids:K
DCM|122701|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|111703|TEXT|X|clean_descriptors:C
DCM|126071|TEXT|X|clean_descriptors:C
DCM|128230|TEXT|X|clean_descriptors:C
DCM|121002|COMPOSITE|D|retain_uids:K
DCM|128436|COMPOSITE|D|retain_uids:K
DCM|128403|TEXT|D|clean_descriptors:C
DCM|128414|COMPOSITE|D|retain_uids:K
DCM|128414|IMAGE|D|retain_uids:K
DCM|113514|TEXT|X
DCM|113503|UIDREF|D|retain_uids:K
DCM|113511|TEXT|X
DCM|113512|TEXT|X
DCM|123003|DATETIME|X/D|retain_longitudinal_temporal_information:C
DCM|123004|DATETIME|X|retain_longitudinal_temporal_information:C
DCM|130507|TEXT|X|clean_descriptors:C
DCM|113513|TEXT|X
DCM|126100|COMPOSITE|X|retain_uids:K
DCM|113907|TEXT|X|clean_descriptors:C
DCM|113552|TEXT|X|clean_descriptors:C
DCM|121075|TEXT|X|clean_descriptors:C
DCM|111054|DATE|X/D|retain_longitudinal_temporal_information:C
DCM|121191|IMAGE|X/D|retain_uids:K
DCM|121214|IMAGE|D|retain_uids:K
DCM|112364|COMPOSITE|X|retain_uids:K
DCM|121121|TEXT|X
DCM|111469|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|111058|TEXT|D|clean_descriptors:C
DCM|131561|DATE|X|retain_longitudinal_temporal_information:C
DCM|131563|TEXT|X|clean_descriptors:C
DCM|131562|TIME|X|retain_longitudinal_temporal_information:C
DCM|112002|UIDREF|D|retain_uids:K
DCM|113985|UIDREF|D|retain_uids:K
DCM|121434|TEXT|X
DCM|121435|PNAME|X
DCM|121435|TEXT|X
SCT|160476009|TEXT|X|clean_descriptors:C
DCM|121233|IMAGE|D|retain_uids:K
DCM|121112|IMAGE|D|retain_uids:K
DCM|121112|WAVEFORM|X|retain_uids:K
DCM|121232|UIDREF|D|retain_uids:K
DCM|128447|COMPOSITE|X|retain_uids:K
DCM|112353|COMPOSITE|X|retain_uids:K
DCM|128444|COMPOSITE|D|retain_uids:K
DCM|111700|TEXT|X
DCM|121041|TEXT|X
DCM|121039|UIDREF|X|retain_uids:K
DCM|128416|COMPOSITE|D|retain_uids:K
SCT|398201009|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|113809|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|110119|TEXT|X|retain_device_identity:K
DCM|122173|DATETIME|X|retain_longitudinal_temporal_information:C
SCT|397898000|DATETIME|D|retain_longitudinal_temporal_information:C
DCM|109056|TEXT|X|clean_descriptors:C
DCM|111060|DATE|X/D|retain_longitudinal_temporal_information:C
DCM|110180|UIDREF|D|retain_uids:K
DCM|111061|TIME|X/D|retain_longitudinal_temporal_information:C
DCM|121033|NUM|X|retain_patient_characteristics:K
DCM|121031|DATE|X
DCM|121030|TEXT|X/D
DCM|121029|PNAME|D
DCM|121032|CODE|X|retain_patient_characteristics:K
DCM|126070|TEXT|X|clean_descriptors:C
DCM|121028|UIDREF|X|retain_uids:K
DCM|121111|TEXT|X|clean_descriptors:C
DCM|112359|COMPOSITE|X|retain_uids:K
DCM|130885|UIDREF|X|retain_uids:K
DCM|112040|UIDREF|D|retain_uids:K
LN|74711-3|TEXT|X|retain_device_identity:K
DCM|121000|CONTAINER|X|retain_device_identity:K
DCM|112356|UIDREF|D|retain_uids:K
DCM|121143|WAVEFORM|D|retain_uids:K
DCM|128470|COMPOSITE|X|retain_uids:K
DCM|128470|IMAGE|X|retain_uids:K
DCM|128470|UIDREF|X|retain_uids:K
DCM|113701|COMPOSITE|X|retain_uids:K
"""
_STRUCTURED_CONTENT_ACTIONS = {
    tuple(parts[:3]): (parts[3], dict(option.split(":") for option in parts[4:]))
    for row in _STRUCTURED_CONTENT_DATA.strip().splitlines()
    if (parts := row.split("|"))
}

_STRUCTURED_CONCEPT_TYPES: dict[tuple[str, str], set[str]] = {}
for _scheme, _value, _value_type in _STRUCTURED_CONTENT_ACTIONS:
    _STRUCTURED_CONCEPT_TYPES.setdefault((_scheme, _value), set()).add(_value_type)

_PROFILE_OPTIONS = {
    "clean_descriptors": ("113105", "Clean Descriptors Option"),
    "clean_structured_content": ("113104", "Clean Structured Content Option"),
    "retain_longitudinal_temporal_information": (
        "113107",
        "Retain Longitudinal Temporal Information Modified Dates Option",
    ),
    "retain_device_identity": ("113109", "Retain Device Identity Option"),
    "retain_patient_characteristics": (
        "113108",
        "Retain Patient Characteristics Option",
    ),
    "retain_uids": ("113110", "Retain UIDs Option"),
}

_DT_DATE_RE = re.compile(r"^(?P<date>\d{8})(?P<rest>.*)$")
_SEEDED_PHI_SCRUB_VRS = frozenset(
    {"AE", "AS", "CS", "LO", "LT", "SH", "ST", "UC", "UR", "UT"}
)
_SEEDED_PHI_SOURCE_VRS = _SEEDED_PHI_SCRUB_VRS | {"DA", "DT", "PN", "TM", "UI"}


def deidentify_dicom_headers(
    path: str | Path,
    *,
    policy: Any | None = None,
) -> DicomHeaderDeidResult:
    """De-identify DICOM headers and write a de-identified DICOM file.

    ``policy`` may be a :class:`DicomHeaderDeidPolicy`, a mapping, or any object
    exposing equivalent attributes. When no ``output_path`` is supplied, the
    source file is rewritten in place.
    """
    return _deidentify_headers(path, policy=policy)


def _deidentify_headers(
    path: str | Path,
    *,
    policy: Any | None,
    processed_document_digests: tuple[str, ...] = (),
    dataset: Any = None,
) -> DicomHeaderDeidResult:

    pydicom = _import_pydicom()
    source = Path(path)
    resolved_policy = _coerce_policy(policy)
    output_path = (
        Path(resolved_policy.output_path)
        if resolved_policy.output_path is not None
        else source
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile_options = _enabled_profile_options(resolved_policy)
    shift_days = (
        _resolve_shift_days(resolved_policy)
        if "retain_longitudinal_temporal_information" in profile_options
        else 0
    )
    from .documents_pdf import _resolve_detector

    try:
        detector = _resolve_detector(
            resolved_policy.detector
            if resolved_policy.detector is not None
            else resolved_policy.document_models
        )
    except Exception:
        raise DicomDeidentificationError("profile_detector_failed") from None
    if any(option.startswith("clean_") for option in profile_options) and (
        detector is None
    ):
        raise DicomDeidentificationError("profile_detector_required")
    context = _Context(
        date_shift_days=shift_days,
        keep_year=resolved_policy.keep_year,
        uid_salt=_bytes_value(resolved_policy.uid_salt, name="uid_salt"),
        processed_document_digests=set(processed_document_digests),
        profile_options=profile_options,
        detector=detector,
    )

    if dataset is None:
        dataset = pydicom.dcmread(source, force=True)
    pixel_status = _pixel_coverage(dataset)
    if resolved_policy.fail_on_unclean_pixels and (
        pixel_status is DicomPixelStatus.NOT_CLEANED
    ):
        raise DicomDeidentificationError("pixels_not_cleaned")
    _sanitize_carriers(dataset, context, policy=resolved_policy)
    seeded_phi_terms = _seeded_phi_scrub_terms(dataset, context)
    _invalidate_identity_claims(dataset)
    _deidentify_dataset(dataset, context, location="Dataset")
    _clear_seeded_phi_copies(
        dataset,
        seeded_phi_terms,
        context,
        location="Dataset",
    )
    _set_standard_deid_markers(dataset, context, pixel_status=pixel_status)
    _deidentify_file_meta(dataset, context)
    if hasattr(dataset, "preamble"):
        dataset.preamble = b"\0" * 128

    _save_dataset(dataset, output_path)
    return DicomHeaderDeidResult(
        source_path=source,
        output_path=output_path,
        date_shift_days=shift_days,
        actions=tuple(context.actions),
        uid_remap_count=len(context.uid_map),
        private_tag_removed_count=context.private_tag_removed_count,
        pixel_status=pixel_status,
        profile_options=profile_options,
    )


def redact_dicom_pixels(
    path: str | Path,
    *,
    policy: Any | None = None,
    output_path: str | Path | None = None,
    ocr_engine: Any = None,
    models: Any = None,
    model_name: str | None = None,
    confidence_threshold: float | None = None,
    bbox_padding: int | None = None,
    verify_residual: bool | None = None,
    fail_on_residual: bool | None = None,
    custom_recognizer: Any = None,
    lang: str | None = None,
) -> DicomPixelRedactionResult:
    """Redact burned-in PHI text from DICOM pixels using OCR bboxes.

    Header values from the source DICOM seed a per-image custom recognizer
    before header de-identification clears them. Returned reports intentionally
    exclude raw OCR/header text and carry hashes, labels, bboxes, and counts.
    """
    result, _dataset = _redact_pixels(
        path,
        policy=policy,
        output_path=output_path,
        ocr_engine=ocr_engine,
        models=models,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        bbox_padding=bbox_padding,
        verify_residual=verify_residual,
        fail_on_residual=fail_on_residual,
        custom_recognizer=custom_recognizer,
        lang=lang,
    )
    return result


def _redact_pixels(
    path: str | Path,
    *,
    policy: Any | None = None,
    output_path: str | Path | None = None,
    ocr_engine: Any = None,
    models: Any = None,
    model_name: str | None = None,
    confidence_threshold: float | None = None,
    bbox_padding: int | None = None,
    verify_residual: bool | None = None,
    fail_on_residual: bool | None = None,
    custom_recognizer: Any = None,
    lang: str | None = None,
    save: bool = True,
) -> tuple[DicomPixelRedactionResult, Any]:

    pydicom = _import_pydicom()
    source = Path(path)
    resolved_policy = _override_pixel_policy(
        _coerce_pixel_policy(policy),
        output_path=output_path,
        ocr_engine=ocr_engine,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        bbox_padding=bbox_padding,
        verify_residual=verify_residual,
        fail_on_residual=fail_on_residual,
        custom_recognizer=custom_recognizer,
    )
    destination = (
        Path(resolved_policy.output_path)
        if resolved_policy.output_path is not None
        else source
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset = pydicom.dcmread(source, force=True)
    context = _Context(date_shift_days=0, keep_year=False, uid_salt=b"unused")
    if "FloatPixelData" in dataset or "DoubleFloatPixelData" in dataset:
        raise DicomDeidentificationError("pixel_data_unsupported")
    if not isinstance(
        resolved_policy.overlay_mode, str
    ) or resolved_policy.overlay_mode not in {"remove", "burn"}:
        raise DicomDeidentificationError("invalid_overlay_mode")
    overlays = (
        _overlay_planes(dataset) if resolved_policy.overlay_mode == "burn" else ()
    )
    _sanitize_carriers(dataset, context, policy=resolved_policy)
    header_recognizer = _header_seed_recognizer(dataset)
    model = _resolve_pixel_model_name(models, resolved_policy.model_name)
    _invalidate_identity_claims(dataset)

    if "PixelData" not in dataset:
        if save:
            _save_dataset(dataset, destination)
        residual_report = DicomResidualTextReport(frame_count=0)
        return DicomPixelRedactionResult(
            source_path=source,
            output_path=destination,
            frames_processed=0,
            findings=(),
            residual_report=residual_report,
            carrier_actions=tuple(context.actions),
            pixel_status=(
                DicomPixelStatus.NOT_PRESENT
                if _pixel_coverage(dataset) is DicomPixelStatus.NOT_PRESENT
                else DicomPixelStatus.NOT_CLEANED
            ),
            _processed_document_digests=tuple(
                sorted(context.processed_document_digests)
            ),
        ), dataset

    _decompress_pixel_data(dataset)
    pixel_array = _copy_pixel_array(dataset)
    frame_views = tuple(_iter_pixel_frames(pixel_array, dataset))
    _burn_overlay_planes(frame_views, overlays, dataset)
    findings: list[DicomPixelFinding] = []

    for frame_index, frame in enumerate(frame_views):
        frame_findings = _detect_frame_pixel_findings(
            frame,
            dataset=dataset,
            frame_index=frame_index,
            policy=resolved_policy,
            model_name=model,
            header_recognizer=header_recognizer,
            lang=lang,
        )
        for finding in frame_findings:
            _blackout_bbox(frame, finding.bbox, dataset)
        findings.extend(frame_findings)

    dataset.PixelData = _pixel_bytes(pixel_array)

    residual_report = DicomResidualTextReport(frame_count=len(frame_views))
    if resolved_policy.verify_residual:
        residual_report = _residual_text_report(
            frame_views,
            dataset=dataset,
            policy=resolved_policy,
            model_name=model,
            header_recognizer=header_recognizer,
            lang=lang,
        )
        if resolved_policy.fail_on_residual and not residual_report.passed:
            raise ValueError(
                "DICOM residual OCR PHI verification failed: "
                f"{residual_report.residual_entity_count} residual findings"
            )

    root_cleaned = resolved_policy.verify_residual and residual_report.passed
    dataset.BurnedInAnnotation = "NO" if root_cleaned else "YES"
    pixel_status = (
        DicomPixelStatus.CLEANED
        if root_cleaned and _pixel_coverage(dataset) is not DicomPixelStatus.NOT_CLEANED
        else DicomPixelStatus.NOT_CLEANED
    )
    # Pixel processing alone has not cleaned the headers. Invalidate inherited
    # attestations; the combined dispatcher sets them after its header pass.
    _invalidate_identity_claims(dataset)
    if save:
        _save_dataset(dataset, destination)
    return DicomPixelRedactionResult(
        source_path=source,
        output_path=destination,
        frames_processed=len(frame_views),
        findings=tuple(findings),
        residual_report=residual_report,
        carrier_actions=tuple(context.actions),
        pixel_status=pixel_status,
        _processed_document_digests=tuple(sorted(context.processed_document_digests)),
    ), dataset


def _import_pydicom() -> Any:
    try:
        return importlib.import_module("pydicom")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="pydicom", instruction=_DICOM_INSTALL_HINT
        ) from exc


def _import_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - exercised without extra.
        raise MissingDependencyError(
            dependency="numpy", instruction=_DICOM_INSTALL_HINT
        ) from exc


def _coerce_pixel_policy(policy: Any | None) -> DicomPixelRedactionPolicy:
    if policy is None:
        return DicomPixelRedactionPolicy()
    if isinstance(policy, DicomPixelRedactionPolicy):
        return policy
    if isinstance(policy, Mapping):
        return DicomPixelRedactionPolicy(
            output_path=policy.get("pixel_output_path", policy.get("output_path")),
            ocr_engine=policy.get("ocr_engine"),
            model_name=(
                policy.get("pii_model_name")
                or policy.get("pii_model")
                or policy.get("model_name")
            ),
            confidence_threshold=_optional_float(
                policy.get("confidence_threshold"), default=0.5
            ),
            bbox_padding=_optional_int(policy.get("bbox_padding")) or 1,
            verify_residual=bool(policy.get("verify_residual", True)),
            fail_on_residual=bool(policy.get("fail_on_residual", True)),
            custom_recognizer=policy.get("custom_recognizer"),
            **_carrier_policy_fields(policy),
        )
    return DicomPixelRedactionPolicy(
        output_path=getattr(
            policy,
            "pixel_output_path",
            getattr(policy, "output_path", None),
        ),
        ocr_engine=getattr(policy, "ocr_engine", None),
        model_name=(
            getattr(policy, "pii_model_name", None)
            or getattr(policy, "pii_model", None)
            or getattr(policy, "model_name", None)
        ),
        confidence_threshold=_optional_float(
            getattr(policy, "confidence_threshold", None), default=0.5
        ),
        bbox_padding=_optional_int(getattr(policy, "bbox_padding", None)) or 1,
        verify_residual=bool(getattr(policy, "verify_residual", True)),
        fail_on_residual=bool(getattr(policy, "fail_on_residual", True)),
        custom_recognizer=getattr(policy, "custom_recognizer", None),
        **_carrier_policy_fields(policy),
    )


def _override_pixel_policy(
    policy: DicomPixelRedactionPolicy,
    *,
    output_path: str | Path | None,
    ocr_engine: Any,
    model_name: str | None,
    confidence_threshold: float | None,
    bbox_padding: int | None,
    verify_residual: bool | None,
    fail_on_residual: bool | None,
    custom_recognizer: Any,
) -> DicomPixelRedactionPolicy:
    return DicomPixelRedactionPolicy(
        output_path=output_path if output_path is not None else policy.output_path,
        ocr_engine=ocr_engine if ocr_engine is not None else policy.ocr_engine,
        model_name=model_name if model_name is not None else policy.model_name,
        confidence_threshold=(
            float(confidence_threshold)
            if confidence_threshold is not None
            else policy.confidence_threshold
        ),
        bbox_padding=(
            int(bbox_padding) if bbox_padding is not None else policy.bbox_padding
        ),
        verify_residual=(
            bool(verify_residual)
            if verify_residual is not None
            else policy.verify_residual
        ),
        fail_on_residual=(
            bool(fail_on_residual)
            if fail_on_residual is not None
            else policy.fail_on_residual
        ),
        custom_recognizer=(
            custom_recognizer
            if custom_recognizer is not None
            else policy.custom_recognizer
        ),
        overlay_mode=policy.overlay_mode,
        redact_encapsulated_documents=policy.redact_encapsulated_documents,
        document_policy=policy.document_policy,
        document_models=policy.document_models,
    )


def _coerce_policy(policy: Any | None) -> DicomHeaderDeidPolicy:
    if policy is None:
        return DicomHeaderDeidPolicy()
    if isinstance(policy, DicomHeaderDeidPolicy):
        return policy
    if isinstance(policy, Mapping):
        return DicomHeaderDeidPolicy(
            output_path=policy.get("output_path"),
            date_shift_days=_optional_int(policy.get("date_shift_days")),
            patient_key=policy.get("patient_key"),
            date_shift_max_days=_optional_int(policy.get("date_shift_max_days")),
            date_shift_secret=policy.get("date_shift_secret"),
            uid_salt=policy.get("uid_salt", _DEFAULT_UID_SALT),
            keep_year=bool(policy.get("keep_year", False)),
            fail_on_unclean_pixels=bool(policy.get("fail_on_unclean_pixels", False)),
            **_profile_policy_fields(policy),
            **_carrier_policy_fields(policy, pixel=False),
        )
    return DicomHeaderDeidPolicy(
        output_path=getattr(policy, "output_path", None),
        date_shift_days=_optional_int(getattr(policy, "date_shift_days", None)),
        patient_key=getattr(policy, "patient_key", None),
        date_shift_max_days=_optional_int(getattr(policy, "date_shift_max_days", None)),
        date_shift_secret=getattr(policy, "date_shift_secret", None),
        uid_salt=getattr(policy, "uid_salt", _DEFAULT_UID_SALT),
        keep_year=bool(getattr(policy, "keep_year", False)),
        fail_on_unclean_pixels=bool(getattr(policy, "fail_on_unclean_pixels", False)),
        **_profile_policy_fields(policy),
        **_carrier_policy_fields(policy, pixel=False),
    )


def _profile_policy_fields(policy: Any) -> dict[str, Any]:
    get = (
        policy.get
        if isinstance(policy, Mapping)
        else lambda k, d: getattr(policy, k, d)
    )
    fields = {name: get(name, False) for name in _PROFILE_OPTIONS}
    fields["detector"] = get("detector", None)
    return fields


def _enabled_profile_options(policy: DicomHeaderDeidPolicy) -> tuple[str, ...]:
    if any(type(getattr(policy, name)) is not bool for name in _PROFILE_OPTIONS):
        raise DicomDeidentificationError("invalid_profile_option")
    enabled = {name for name in _PROFILE_OPTIONS if getattr(policy, name)}
    # Explicit legacy shift parameters are an existing request to retain dates
    # with modification. Declare that option rather than silently losing them.
    if any(
        value is not None
        for value in (
            policy.date_shift_days,
            policy.patient_key,
            policy.date_shift_max_days,
            policy.date_shift_secret,
        )
    ):
        enabled.add("retain_longitudinal_temporal_information")
    return tuple(name for name in _PROFILE_OPTIONS if name in enabled)


def _carrier_policy_fields(policy: Any, *, pixel: bool = True) -> dict[str, Any]:
    get = (
        policy.get
        if isinstance(policy, Mapping)
        else lambda k, d: getattr(policy, k, d)
    )
    fields = {
        "redact_encapsulated_documents": bool(
            get("redact_encapsulated_documents", False)
        ),
        "document_policy": get("document_policy", None),
        "document_models": get("document_models", None),
    }
    if pixel:
        fields["overlay_mode"] = get("overlay_mode", "remove")
    return fields


def _pixel_coverage(dataset: Any) -> DicomPixelStatus:
    statuses = []
    if any(
        key in dataset
        for key in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")
    ):
        statuses.append(
            DicomPixelStatus.DECLARED_CLEAN
            if dataset.get("BurnedInAnnotation") == "NO"
            else DicomPixelStatus.NOT_CLEANED
        )
    for element in dataset:
        if element.VR == "SQ" and int(element.tag) != 0x00880200:
            statuses.extend(_pixel_coverage(item) for item in element.value or ())
    if DicomPixelStatus.NOT_CLEANED in statuses:
        return DicomPixelStatus.NOT_CLEANED
    if DicomPixelStatus.DECLARED_CLEAN in statuses:
        return DicomPixelStatus.DECLARED_CLEAN
    return DicomPixelStatus.NOT_PRESENT


def _sanitize_carriers(
    dataset: Any,
    context: _Context,
    *,
    policy: Any,
    location: str = "Dataset",
) -> None:
    if "PixelDataProviderURL" in dataset:
        raise DicomDeidentificationError("external_pixel_data_unsupported")
    if (
        getattr(policy, "overlay_mode", "remove") == "burn"
        and location != "Dataset"
        and any(0x6000 <= int(tag) >> 16 <= 0x60FF for tag in dataset.keys())
    ):
        raise DicomDeidentificationError("nested_overlay_burn_unsupported")
    # A retired overlay may occupy unused bits of the main PixelData rather
    # than an OverlayData element. Decode/re-encode those bits before dropping
    # the metadata, and refuse planes overlapping stored image values.
    try:
        embedded = [
            group
            for group in range(0x6000, 0x6100, 2)
            if (group, 0x0102) in dataset
            and (
                (group, 0x3000) not in dataset
                or int(dataset[(group, 0x0102)].value) != 0
                or (
                    (group, 0x0100) in dataset
                    and int(dataset[(group, 0x0100)].value) != 1
                )
            )
        ]
    except Exception:
        raise DicomDeidentificationError("embedded_overlay_not_cleanable") from None
    if embedded:
        try:
            stored = int(dataset.BitsStored)
            allocated = int(dataset.BitsAllocated)
            if any(
                not stored <= int(dataset[(group, 0x0102)].value) < allocated
                for group in embedded
            ):
                raise ValueError
            _decompress_pixel_data(dataset)
            dataset.PixelData = _pixel_bytes(_copy_pixel_array(dataset))
        except Exception:
            raise DicomDeidentificationError("embedded_overlay_not_cleanable") from None
    for tag in list(dataset.keys()):
        element = dataset[tag]
        group = int(element.tag) >> 16
        tag_int = int(element.tag)
        if (
            0x5000 <= group <= 0x50FF
            or 0x6000 <= group <= 0x60FF
            or tag_int == 0x00880200
        ):
            _record_action(
                element,
                context,
                action="remove",
                ps315_action="X",
                location=location,
                record_value=False,
            )
            del dataset[tag]
        elif tag_int == 0x00420011:
            payload = dataset.EncapsulatedDocument
            digest = (
                hashlib.sha256(payload.rstrip(b"\0")).hexdigest()
                if isinstance(payload, bytes)
                else ""
            )
            if digest in context.processed_document_digests:
                continue
            _redact_encapsulated_document(dataset, policy)
            context.processed_document_digests.add(
                hashlib.sha256(dataset.EncapsulatedDocument.rstrip(b"\0")).hexdigest()
            )
            _record_action(
                dataset[tag],
                context,
                action="replace",
                ps315_action="D",
                location=location,
                record_value=False,
            )
        elif element.VR == "SQ":
            for index, item in enumerate(element.value or ()):
                _sanitize_carriers(
                    item,
                    context,
                    policy=policy,
                    location=f"{location}.{_keyword(element)}[{index}]",
                )


def _redact_encapsulated_document(dataset: Any, policy: Any) -> None:
    if not policy.redact_encapsulated_documents:
        raise DicomDeidentificationError("encapsulated_document_requires_redaction")
    mime = dataset.get("MIMETypeOfEncapsulatedDocument")
    extension = (
        {
            "application/pdf": ".pdf",
            "text/xml": ".xml",
            "application/xml": ".xml",
        }.get(mime)
        if isinstance(mime, str)
        else None
    )
    if extension is None:
        raise DicomDeidentificationError("encapsulated_document_type_unsupported")
    original = dataset.EncapsulatedDocument
    if not isinstance(original, bytes) or not original:
        raise DicomDeidentificationError("encapsulated_document_invalid")
    # Only registered redaction handlers with an explicit in-memory byte output
    # can authorize replacement. Extraction text is never treated as redaction.
    from .base import redact_document

    source = BytesIO(original)
    source.name = "encapsulated" + extension
    try:
        # The PDF handler otherwise treats models=None as extraction followed
        # by rasterization without any detected redaction rectangles.
        from .documents_pdf import _resolve_detector

        if _resolve_detector(policy.document_models) is None:
            raise ValueError
        document_policy = dict(policy.document_policy or {})
        for key in ("output_path", "redacted_path", "destination_path"):
            document_policy.pop(key, None)
        document_policy["return_bytes"] = True
        document = redact_document(
            source,
            policy=document_policy,
            models=policy.document_models,
        )
        replacement = document.metadata.get(
            "redacted_document_bytes", document.metadata.get("redacted_pdf_bytes")
        )
        if (
            not isinstance(replacement, bytes)
            or not replacement
            or replacement.rstrip(b"\0") == original.rstrip(b"\0")
        ):
            raise ValueError
    except Exception:
        raise DicomDeidentificationError(
            "encapsulated_document_redaction_failed"
        ) from None
    dataset.EncapsulatedDocument = replacement
    dataset.EncapsulatedDocumentLength = len(replacement)


def _overlay_planes(dataset: Any) -> tuple[tuple[Any, int, int, int], ...]:
    planes = []
    np = _import_numpy()
    for group in range(0x6000, 0x6100, 2):
        if (group, 0x0010) not in dataset:
            continue
        try:
            origin = dataset[(group, 0x0050)].value
            row, column = int(origin[0]) - 1, int(origin[1]) - 1
            start_frame = (
                int(dataset.get((group, 0x0051), 1).value) - 1
                if (group, 0x0051) in dataset
                else 0
            )
            if start_frame < 0:
                raise ValueError
            if (group, 0x3000) in dataset:
                mask = dataset.overlay_array(group)
            else:
                _decompress_pixel_data(dataset)
                bits = int(dataset.BitsAllocated)
                position = int(dataset[(group, 0x0102)].value)
                if (
                    bits not in (8, 16, 32)
                    or not int(dataset.BitsStored) <= position < bits
                ):
                    raise ValueError
                if int(dataset.SamplesPerPixel) != 1:
                    raise ValueError
                endian = (
                    ">"
                    if str(dataset.file_meta.TransferSyntaxUID) == "1.2.840.10008.1.2.2"
                    else "<"
                )
                raw = np.frombuffer(
                    dataset.PixelData, dtype=np.dtype(endian + f"u{bits // 8}")
                )
                mask = ((raw >> position) & 1).reshape(
                    -1, int(dataset.Rows), int(dataset.Columns)
                )
                if row != 0 or column != 0:
                    raise ValueError
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
            planes.append((mask, row, column, start_frame))
        except Exception:
            raise DicomDeidentificationError("overlay_burn_failed") from None
    return tuple(planes)


def _burn_overlay_planes(
    frames: Sequence[Any], planes: Sequence[Any], dataset: Any
) -> None:
    np = _import_numpy()
    for masks, row, column, start_frame in planes:
        if start_frame + len(masks) > len(frames):
            raise DicomDeidentificationError("overlay_burn_failed")
        for index, mask in enumerate(masks):
            frame = frames[start_frame + index]
            height, width = frame.shape[:2]
            y0, x0 = max(0, row), max(0, column)
            y1, x1 = (
                min(height, row + mask.shape[0]),
                min(width, column + mask.shape[1]),
            )
            if y0 >= y1 or x0 >= x1:
                continue
            region = frame[y0:y1, x0:x1]
            selected = mask[y0 - row : y1 - row, x0 - column : x1 - column].astype(bool)
            limits = np.iinfo(frame.dtype)
            fill = (
                limits.min
                if dataset.PhotometricInterpretation == "MONOCHROME1"
                else limits.max
            )
            region[selected] = fill


def _optional_int(value: Any | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any | None, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _resolve_shift_days(policy: DicomHeaderDeidPolicy) -> int:
    from openmed.core.pii import _resolve_date_shift_days

    shift_days = _resolve_date_shift_days(
        date_shift_days=policy.date_shift_days,
        patient_key=policy.patient_key,
        date_shift_max_days=policy.date_shift_max_days,
        date_shift_secret=policy.date_shift_secret,
    )
    if shift_days == 0:
        raise ValueError("date_shift_days must be non-zero for DICOM de-id")
    return shift_days


def _resolve_pixel_model_name(models: Any, policy_model_name: str | None) -> str | None:
    if policy_model_name:
        return str(policy_model_name)
    if models is None:
        return None
    if isinstance(models, str):
        return models
    if isinstance(models, Mapping):
        for key in ("pii_model_name", "pii_model", "model_name", "pii"):
            value = models.get(key)
            if value:
                return str(value)
        return None
    for attr in ("pii_model_name", "pii_model", "model_name"):
        value = getattr(models, attr, None)
        if value:
            return str(value)
    return None


def _copy_pixel_array(dataset: Any) -> Any:
    numpy = _import_numpy()
    return numpy.array(dataset.pixel_array, copy=True)


def _decompress_pixel_data(dataset: Any) -> None:
    """Normalize compressed Pixel Data before editing and writing raw bytes."""
    file_meta = getattr(dataset, "file_meta", None)
    transfer_syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if transfer_syntax is None or not bool(
        getattr(transfer_syntax, "is_compressed", False)
    ):
        return
    try:
        dataset.decompress(generate_instance_uid=False)
    except Exception:
        raise ValueError(
            "Compressed DICOM Pixel Data could not be decoded with the installed "
            "pixel-data codecs"
        ) from None


def _iter_pixel_frames(pixel_array: Any, dataset: Any) -> Sequence[Any]:
    number_of_frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
    if number_of_frames > 1 and getattr(pixel_array, "ndim", 0) >= 3:
        frame_count = min(number_of_frames, int(pixel_array.shape[0]))
        return tuple(pixel_array[index] for index in range(frame_count))
    return (pixel_array,)


def _pixel_bytes(pixel_array: Any) -> bytes:
    numpy = _import_numpy()
    return numpy.ascontiguousarray(pixel_array).tobytes()


def _detect_frame_pixel_findings(
    frame: Any,
    *,
    dataset: Any,
    frame_index: int,
    policy: DicomPixelRedactionPolicy,
    model_name: str | None,
    header_recognizer: Any,
    lang: str | None,
) -> tuple[DicomPixelFinding, ...]:
    from . import ocr as ocr_mod

    languages = [lang] if lang else None
    ocr_result = ocr_mod.ocr(
        _frame_for_ocr(frame, dataset),
        engine=policy.ocr_engine,
        languages=languages,
    )
    document = ocr_result.to_document()
    entities = _detect_pixel_entities(
        document.text,
        model_name=model_name,
        confidence_threshold=policy.confidence_threshold,
        lang=lang,
        header_recognizer=header_recognizer,
        custom_recognizer=policy.custom_recognizer,
    )
    return _project_entities_to_findings(
        document,
        entities,
        frame_index=frame_index,
        frame_shape=frame.shape,
        padding=policy.bbox_padding,
    )


def _detect_pixel_entities(
    text: str,
    *,
    model_name: str | None,
    confidence_threshold: float,
    lang: str | None,
    header_recognizer: Any,
    custom_recognizer: Any,
) -> tuple[Any, ...]:
    if not text.strip():
        return ()

    result = _extract_dicom_pixel_phi(
        text,
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        lang=lang,
        custom_recognizer=custom_recognizer,
    )
    entities = list(getattr(result, "entities", ()) or ())
    if header_recognizer is not None:
        entities.extend(header_recognizer.detect_entities(text))
    return _dedupe_entities(text, entities)


def _extract_dicom_pixel_phi(
    text: str,
    *,
    model_name: str | None,
    confidence_threshold: float,
    lang: str | None,
    custom_recognizer: Any,
) -> Any:
    from openmed.core.pii import extract_pii

    kwargs: dict[str, Any] = {
        "confidence_threshold": confidence_threshold,
        "lang": lang or "en",
        "custom_recognizer": custom_recognizer,
    }
    if model_name is not None:
        kwargs["model_name"] = model_name
    return extract_pii(text, **kwargs)


def _dedupe_entities(text: str, entities: Sequence[Any]) -> tuple[Any, ...]:
    deduped: list[Any] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in entities:
        bounds = _entity_bounds(entity, text)
        if bounds is None:
            continue
        label = _entity_label(entity)
        key = (bounds[0], bounds[1], label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return tuple(deduped)


def _project_entities_to_findings(
    document: ExtractedDocument,
    entities: Sequence[Any],
    *,
    frame_index: int,
    frame_shape: Sequence[int],
    padding: int,
) -> tuple[DicomPixelFinding, ...]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    findings: list[DicomPixelFinding] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for entity in entities:
        bounds = _entity_bounds(entity, document.text)
        if bounds is None:
            continue
        surface = document.text[bounds[0] : bounds[1]]
        for span in document.spans:
            if span.bbox is None or not _intervals_overlap(
                span.start, span.end, bounds[0], bounds[1]
            ):
                continue
            bbox = _clamp_bbox(span.bbox, width=width, height=height, padding=padding)
            if bbox is None or bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            findings.append(
                DicomPixelFinding(
                    frame_index=frame_index,
                    bbox=bbox,
                    label=_entity_label(entity),
                    confidence=_entity_confidence(entity),
                    text_sha256=_hash_value(surface),
                    text_length=len(surface),
                    sources=_entity_sources(entity),
                )
            )
    return tuple(findings)


def _residual_text_report(
    frame_views: Sequence[Any],
    *,
    dataset: Any,
    policy: DicomPixelRedactionPolicy,
    model_name: str | None,
    header_recognizer: Any,
    lang: str | None,
) -> DicomResidualTextReport:
    residuals: list[DicomPixelFinding] = []
    for frame_index, frame in enumerate(frame_views):
        residuals.extend(
            _detect_frame_pixel_findings(
                frame,
                dataset=dataset,
                frame_index=frame_index,
                policy=policy,
                model_name=model_name,
                header_recognizer=header_recognizer,
                lang=lang,
            )
        )
    return DicomResidualTextReport(
        frame_count=len(frame_views),
        residuals=tuple(residuals),
    )


def _frame_for_ocr(frame: Any, dataset: Any) -> Any:
    numpy = _import_numpy()
    array = numpy.asarray(frame)
    if _is_color_frame(array, dataset):
        return _normalize_color_frame(array)

    image = _normalize_uint8(array)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        image = 255 - image
    return image


def _is_color_frame(frame: Any, dataset: Any) -> bool:
    samples = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
    photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
    return samples > 1 or photometric in {"RGB", "YBR_FULL", "YBR_FULL_422"}


def _normalize_color_frame(frame: Any) -> Any:
    numpy = _import_numpy()
    array = numpy.asarray(frame)
    if array.dtype == numpy.uint8:
        return numpy.array(array, copy=True)
    return _normalize_uint8(array)


def _normalize_uint8(array: Any) -> Any:
    numpy = _import_numpy()
    values = numpy.asarray(array)
    if values.dtype == numpy.uint8:
        return numpy.array(values, copy=True)
    values = values.astype("float32", copy=False)
    minimum = float(numpy.nanmin(values))
    maximum = float(numpy.nanmax(values))
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum == maximum:
        return numpy.zeros(values.shape, dtype=numpy.uint8)
    scaled = (values - minimum) * (255.0 / (maximum - minimum))
    return numpy.clip(scaled, 0, 255).astype(numpy.uint8)


def _blackout_bbox(
    frame: Any,
    bbox: tuple[int, int, int, int],
    dataset: Any,
) -> None:
    x0, y0, x1, y1 = bbox
    value = _pixel_black_value(frame, dataset)
    if getattr(frame, "ndim", 0) >= 3:
        frame[y0:y1, x0:x1, ...] = value
    else:
        frame[y0:y1, x0:x1] = value


def _pixel_black_value(frame: Any, dataset: Any) -> int:
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() != "MONOCHROME1":
        return 0
    bits_stored = int(
        getattr(dataset, "BitsStored", getattr(dataset, "BitsAllocated", 8)) or 8
    )
    return int((2**bits_stored) - 1)


def _clamp_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x0 = max(0, math.floor(float(bbox[0])) - padding)
    y0 = max(0, math.floor(float(bbox[1])) - padding)
    x1 = min(width, math.ceil(float(bbox[2])) + padding)
    y1 = min(height, math.ceil(float(bbox[3])) + padding)
    if x0 >= x1 or y0 >= y1:
        return None
    return (x0, y0, x1, y1)


def _intervals_overlap(
    start: int,
    end: int,
    other_start: int,
    other_end: int,
) -> bool:
    return start < other_end and end > other_start


def _entity_bounds(entity: Any, text: str) -> tuple[int, int] | None:
    start = getattr(entity, "start", None)
    end = getattr(entity, "end", None)
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 0 <= start < end <= len(text)
    ):
        return start, end

    surface = str(getattr(entity, "text", "") or "")
    if not surface:
        return None
    found = text.find(surface)
    if found < 0:
        return None
    return found, found + len(surface)


def _entity_label(entity: Any) -> str:
    return str(
        getattr(entity, "canonical_label", None)
        or getattr(entity, "entity_type", None)
        or getattr(entity, "label", None)
        or "UNKNOWN"
    )


def _entity_confidence(entity: Any) -> float:
    try:
        return float(getattr(entity, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _entity_sources(entity: Any) -> tuple[str, ...]:
    sources = getattr(entity, "sources", None)
    if sources:
        return tuple(str(source) for source in sources)
    metadata = getattr(entity, "metadata", None) or {}
    if isinstance(metadata, Mapping):
        source = metadata.get("detector") or metadata.get("source")
        if source:
            return (str(source),)
        custom = metadata.get("custom_recognizer")
        if isinstance(custom, Mapping) and custom.get("detector"):
            return (str(custom["detector"]),)
    return ("model",)


def _header_seed_recognizer(dataset: Any) -> Any:
    terms = _header_seed_terms(dataset)
    if not terms:
        return None
    from openmed.core.custom_recognizer import CustomRecognizer

    return CustomRecognizer.from_config(
        {
            "case_sensitive": False,
            "deny_terms": [
                {
                    "term": term,
                    "label": label,
                    "confidence": 1.0,
                    "id": _header_seed_rule_id(keyword, term),
                }
                for keyword, label, term in terms
            ],
        }
    )


def _header_seed_terms(dataset: Any) -> tuple[tuple[str, str, str], ...]:
    specs = (
        ("PatientName", "NAME"),
        ("OtherPatientNames", "NAME"),
        ("PatientBirthName", "NAME"),
        ("PatientMothersBirthName", "NAME"),
        ("PatientID", "ID_NUM"),
        ("OtherPatientIDs", "ID_NUM"),
        ("AccessionNumber", "ID_NUM"),
        ("AdmissionID", "ID_NUM"),
        ("StudyID", "ID_NUM"),
        ("PatientBirthDate", "DATE"),
        ("StudyDate", "DATE"),
        ("SeriesDate", "DATE"),
        ("ContentDate", "DATE"),
        ("AcquisitionDate", "DATE"),
    )
    terms: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for keyword, label in specs:
        value = getattr(dataset, keyword, None)
        for raw in _dicom_value_strings(value):
            for term in _header_value_variants(raw, keyword=keyword):
                normalized = " ".join(term.split())
                if not normalized:
                    continue
                dedupe_key = normalized.casefold()
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                terms.append((keyword, label, normalized))
    return tuple(terms)


def _dicom_value_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, bytes):
        return (value.decode("utf-8", errors="ignore"),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(item for child in value for item in _dicom_value_strings(child))
    return (str(value),)


def _header_value_variants(value: str, *, keyword: str) -> tuple[str, ...]:
    text = value.strip()
    if not text:
        return ()

    variants = [text]
    if "^" in text:
        parts = [part.strip() for part in text.split("^") if part.strip()]
        if parts:
            variants.append(" ".join(parts))
        if len(parts) >= 2:
            variants.append(f"{parts[1]} {parts[0]}")
            variants.append(f"{parts[0]} {parts[1]}")

    if keyword.lower().endswith("date"):
        variants.extend(_date_variants(text))
    return tuple(variants)


def _date_variants(value: str) -> tuple[str, ...]:
    text = value.strip()
    if not re.fullmatch(r"\d{8}", text):
        return ()
    year, month, day = text[:4], text[4:6], text[6:8]
    return (
        f"{year}-{month}-{day}",
        f"{month}/{day}/{year}",
        f"{day}/{month}/{year}",
    )


def _header_seed_rule_id(keyword: str, term: str) -> str:
    digest = _hash_value(f"{keyword}:{term}")[:12]
    return f"dicom_header_{keyword}_{digest}"


def _profile_key(element: Any) -> str:
    tag = int(element.tag)
    if element.tag.is_private:
        return "PRIVATE"
    if 0x5000 <= tag >> 16 <= 0x50FF:
        return "50XXXXXX"
    if 0x6000 <= tag >> 16 <= 0x60FF:
        return f"60XX{tag & 0xFFFF:04X}"
    return f"{tag:08X}"


def _option_action(base: str, overrides: Mapping[str, str], context: _Context) -> str:
    selected = [
        overrides[name] for name in context.profile_options if name in overrides
    ]
    # Cleaning is stricter than retaining when multiple options cover a field.
    return "C" if "C" in selected else selected[0] if selected else base


def _catalog_action(element: Any, context: _Context) -> str | None:
    key = _profile_key(element)
    base = _BASIC_PROFILE_ACTIONS.get(key)
    if base is None:
        return None
    return _option_action(base, _PROFILE_OPTION_ACTIONS.get(key, {}), context)


def _deidentify_dataset(
    dataset: Any,
    context: _Context,
    *,
    location: str,
    overrides: Mapping[int, str] | None = None,
) -> None:
    for tag in list(dataset.keys()):
        element = dataset[tag]
        tag_int = int(element.tag)
        _validate_element_vr(element)
        action = (overrides or {}).get(tag_int, _catalog_action(element, context))
        # Binary carrier replacement already happened at the shared boundary.
        if tag_int == 0x00420011:
            continue
        if element.tag.is_private or tag_int >> 16 == 0x0004:
            action = "X"
        if action is None:
            action = _unlisted_action(element, context)
            _record_unlisted(element, context, action=action, location=location)
        _apply_profile_action(dataset, element, context, action, location=location)


def _validate_element_vr(element: Any) -> None:
    if element.tag.is_private:
        return
    pydicom = _import_pydicom()
    try:
        expected = pydicom.datadict.dictionary_VR(element.tag)
    except KeyError:
        return
    if element.VR not in expected.split(" or "):
        raise DicomDeidentificationError("profile_vr_mismatch")


def _unlisted_action(element: Any, context: _Context) -> str:
    if not getattr(element, "keyword", ""):
        return "X"
    if element.VR == "SQ":
        return "K"
    if element.VR == "UN":
        return "X"
    if element.VR == "UI":
        return (
            "K"
            if _is_structural_uid(element) or "retain_uids" in context.profile_options
            else "U"
        )
    if element.VR in {"DA", "DT", "TM"}:
        return (
            "C"
            if "retain_longitudinal_temporal_information" in context.profile_options
            else "Z"
        )
    if element.VR == "PN":
        return "Z"
    if element.VR in _SEEDED_PHI_SCRUB_VRS:
        if element.VR == "CS" or _keyword(element) in {
            "SpecificCharacterSet",
            "MappingResource",
            "TemplateIdentifier",
        }:
            return "K"
        return (
            "C"
            if any(name.startswith("clean_") for name in context.profile_options)
            else "Z"
        )
    return "K"


def _record_unlisted(
    element: Any, context: _Context, *, action: str, location: str
) -> None:
    context.actions.append(
        DicomHeaderAction(
            tag=_format_tag(element.tag),
            keyword="",
            vr="",
            action="unlisted_public",
            ps315_action=action,
            location=location,
        )
    )


def _apply_profile_action(
    dataset: Any, element: Any, context: _Context, action: str, *, location: str
) -> None:
    resolved = action.split("/")[0]
    if resolved == "X":
        _record_action(
            element, context, action="remove", ps315_action=action, location=location
        )
        if element.tag.is_private:
            context.private_tag_removed_count += 1
        del dataset[element.tag]
    elif resolved == "Z":
        _clear_element(element, context, ps315_action=action, location=location)
    elif resolved == "D":
        _record_action(
            element, context, action="replace", ps315_action=action, location=location
        )
        element.value = _dummy_value(element, context)
    elif resolved == "U":
        if element.VR == "SQ":
            _deidentify_sequence(element, context, location=location)
        else:
            element.VR = "UI"
            _remap_uid_element(element, context, location=location)
    elif resolved == "C":
        if element.VR == "SQ":
            _deidentify_sequence(element, context, location=location)
        elif element.VR in {"DA", "DT"}:
            _shift_date_element(element, context, location=location)
        elif element.VR == "TM":
            # Whole-day shifting preserves relative times within the day.
            values = _dicom_value_strings(element.value)
            if any(
                value and re.fullmatch(r"(?:\d{2}){1,3}(?:\.\d{1,6})?", value) is None
                for value in values
            ):
                raise DicomDeidentificationError("profile_temporal_invalid")
            _record_action(
                element,
                context,
                action="retain_time",
                ps315_action="C",
                location=location,
            )
        else:
            if element.VR not in _SEEDED_PHI_SOURCE_VRS:
                raise DicomDeidentificationError("profile_binary_cleaning_unsupported")
            _record_action(
                element, context, action="clean", ps315_action="C", location=location
            )
            element.value = _map_dicom_values(
                element.value, lambda value: _clean_profile_text(value, context)
            )
    elif resolved == "K":
        if element.VR == "SQ":
            _deidentify_sequence(element, context, location=location)
        else:
            _record_action(
                element,
                context,
                action="keep",
                ps315_action="K",
                location=location,
                record_value=False,
            )
    else:
        raise DicomDeidentificationError("profile_action_unsupported")


def _dummy_value(element: Any, context: _Context) -> Any:
    if element.VR == "SQ":
        pydicom = _import_pydicom()
        item = pydicom.dataset.Dataset()
        if int(element.tag) == 0x0040A730:
            item.ValueType = "TEXT"
            item.RelationshipType = "CONTAINS"
            item.TextValue = "[REMOVED]"
            code = pydicom.dataset.Dataset()
            code.CodeValue = "121106"
            code.CodingSchemeDesignator = "DCM"
            code.CodeMeaning = "Comment"
            item.ConceptNameCodeSequence = [code]
        return [item]
    if element.VR == "UI":
        return _remap_uid(element.value or "empty", context)
    if element.VR in {"DA", "DT", "TM"}:
        return {"DA": "19000101", "DT": "19000101000000", "TM": "000000"}[element.VR]
    if element.VR in {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}:
        return b"\0\0"
    if element.VR in {"DS", "IS", "US", "SS", "UL", "SL", "UV", "SV", "FL", "FD", "AT"}:
        return 0
    if element.VR == "AS":
        return "000Y"
    return "ANON" if element.VR == "CS" else "[REMOVED]"


def _clean_profile_text(value: Any, context: _Context) -> str:
    text = str(value)
    if not text:
        return text
    if context.detector is None:
        raise DicomDeidentificationError("profile_detector_required")
    from .documents_pdf import _iter_entities

    try:
        result = context.detector(text)
        if result is None:
            raise ValueError
        if isinstance(result, Mapping):
            keys = [
                key for key in ("entities", "pii_entities", "spans") if key in result
            ]
            if len(keys) != 1:
                raise ValueError
            collection = result[keys[0]]
        elif hasattr(result, "entities"):
            collection = result.entities
        elif hasattr(result, "pii_entities"):
            collection = result.pii_entities
        else:
            collection = result
        if not isinstance(collection, Iterable) or isinstance(
            collection, (str, bytes, bytearray, Mapping)
        ):
            raise ValueError
        entities = _iter_entities(result)
        ranges = []
        for entity in entities:
            get = (
                entity.get
                if isinstance(entity, Mapping)
                else lambda key: getattr(entity, key, None)
            )
            start, end = get("start"), get("end")
            if (
                type(start) is not int
                or type(end) is not int
                or not 0 <= start < end <= len(text)
            ):
                raise ValueError
            ranges.append((start, end))
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        for start, end in reversed(merged):
            text = text[:start] + "[REMOVED]" + text[end:]
        return text
    except Exception:
        raise DicomDeidentificationError("profile_detector_failed") from None


def _deidentify_sequence(element: Any, context: _Context, *, location: str) -> None:
    cleaned = []
    for index, item in enumerate(element.value or ()):
        child_location = f"{location}.{_keyword(element)}[{index}]"
        overrides = None
        if (
            "clean_structured_content" in context.profile_options
            and "ValueType" in item
        ):
            keep, overrides = _clean_structured_item(
                item, context, location=child_location
            )
            if not keep:
                continue
        _deidentify_dataset(item, context, location=child_location, overrides=overrides)
        cleaned.append(item)
    element.value = cleaned


def _clean_structured_item(
    item: Any, context: _Context, *, location: str
) -> tuple[bool, dict[int, str]]:
    value_type = str(item.get("ValueType", ""))
    concepts = item.get("ConceptNameCodeSequence", ())
    if len(concepts) != 1:
        # Unknown concept structure cannot safely authorize content retention.
        return False, {}
    concept = concepts[0]
    scheme = str(concept.get("CodingSchemeDesignator", ""))
    alternatives = [
        keyword
        for keyword in ("CodeValue", "LongCodeValue", "URNCodeValue")
        if keyword in concept
    ]
    if len(alternatives) != 1:
        raise DicomDeidentificationError("profile_concept_invalid")
    code_value = concept.get(alternatives[0])
    if not isinstance(code_value, str) or not code_value:
        raise DicomDeidentificationError("profile_concept_invalid")
    key = (scheme, code_value, value_type)
    known_types = _STRUCTURED_CONCEPT_TYPES.get((scheme, code_value))
    if known_types is not None and value_type not in known_types:
        # A clinical descriptor such as Finding Site may be encoded with a
        # controlled CODE value instead of its TEXT form. Clean that code
        # sequence. Identity concepts without descriptor-cleaning semantics
        # must not fall through to a different representation (e.g. NUM).
        text_rule = _STRUCTURED_CONTENT_ACTIONS.get((scheme, code_value, "TEXT"))
        if not (
            value_type == "CODE"
            and text_rule is not None
            and text_rule[1].get("clean_descriptors") == "C"
        ):
            return False, {}
    if (
        alternatives[0] == "URNCodeValue"
        and known_types is None
        and value_type != "CONTAINER"
    ):
        return False, {}
    rule = _STRUCTURED_CONTENT_ACTIONS.get(key)
    if rule is None:
        # Retired SNOMED aliases are deliberately removed, rather than bundling
        # a restricted terminology crosswalk or missing historical identifiers.
        if scheme in {"SRT", "SNM3", "99SDM", "UMLS"} and value_type != "CONTAINER":
            return False, {}
        if value_type != "CONTAINER" and scheme not in {
            "DCM",
            "SCT",
            "LN",
            "NCIt",
            "NCDR",
            "UCUM",
            "RFC5646",
        }:
            return False, {}
        action = {
            "PNAME": "D",
            "UIDREF": "U",
            "DATE": "D",
            "DATETIME": "D",
            "TIME": "D",
            "TEXT": "C",
            "CODE": "C",
            "NUM": "K",
            "CONTAINER": "K",
            "IMAGE": "X",
            "COMPOSITE": "X",
            "WAVEFORM": "X",
        }.get(value_type, "X")
        if (
            value_type in {"DATE", "DATETIME", "TIME"}
            and "retain_longitudinal_temporal_information" in context.profile_options
        ):
            action = "C"
        if value_type == "UIDREF" and "retain_uids" in context.profile_options:
            action = "K"
    else:
        action = _option_action(rule[0], rule[1], context)
    value_keyword = {
        "TEXT": "TextValue",
        "PNAME": "PersonName",
        "UIDREF": "UID",
        "DATE": "Date",
        "DATETIME": "DateTime",
        "TIME": "Time",
        "NUM": "MeasuredValueSequence",
        "CODE": "ConceptCodeSequence",
        "IMAGE": "ReferencedSOPSequence",
        "COMPOSITE": "ReferencedSOPSequence",
        "WAVEFORM": "ReferencedSOPSequence",
    }.get(value_type)
    if action.split("/")[0] == "X":
        if value_keyword and value_keyword in item:
            _record_action(
                item[value_keyword],
                context,
                action="remove_content_item",
                ps315_action=action,
                location=location,
            )
        return False, {}
    overrides = (
        {int(item[value_keyword].tag): action}
        if value_keyword and value_keyword in item
        else {}
    )
    return True, overrides


def _seeded_phi_scrub_terms(dataset: Any, context: _Context) -> tuple[str, ...]:
    terms = {
        _normalize_seeded_phi_text(term)
        for _keyword_name, _label, term in _header_seed_terms(dataset)
    }

    def collect(source: Any) -> None:
        for tag in list(source.keys()):
            element = source[tag]
            if element.VR == "SQ":
                for item in element.value or ():
                    collect(item)
                continue
            if str(element.VR) not in _SEEDED_PHI_SOURCE_VRS:
                continue
            sensitive_source = (
                element.tag.is_private
                or _catalog_action(element, context) not in {None, "K", "C"}
                or element.VR in {"DA", "DT", "PN", "TM"}
                or _should_remap_uid(element)
            )
            if not sensitive_source:
                continue
            for raw in _dicom_value_strings(element.value):
                for variant in _header_value_variants(raw, keyword=_keyword(element)):
                    terms.add(_normalize_seeded_phi_text(variant))

    collect(dataset)
    return tuple(
        sorted((term for term in terms if len(term) >= 3), key=len, reverse=True)
    )


def _clear_seeded_phi_copies(
    dataset: Any,
    terms: Sequence[str],
    context: _Context,
    *,
    location: str,
) -> None:
    if not terms:
        return
    for tag in list(dataset.keys()):
        element = dataset[tag]
        if element.VR == "SQ":
            for index, item in enumerate(element.value or ()):
                child_location = f"{location}.{_keyword(element)}[{index}]"
                _clear_seeded_phi_copies(
                    item,
                    terms,
                    context,
                    location=child_location,
                )
            continue
        if str(element.VR) not in _SEEDED_PHI_SCRUB_VRS:
            continue
        values = _dicom_value_strings(element.value)
        if any(
            term in _normalize_seeded_phi_text(value)
            for value in values
            for term in terms
        ):
            _clear_element(element, context, ps315_action="Z", location=location)


def _normalize_seeded_phi_text(value: Any) -> str:
    text = re.sub(r"[\^=,_]+", " ", str(value))
    return " ".join(text.casefold().split())


def _invalidate_identity_claims(dataset: Any, *, root: bool = True) -> None:
    if root or "PatientIdentityRemoved" in dataset:
        dataset.PatientIdentityRemoved = "NO"
    for keyword in ("DeidentificationMethod", "DeidentificationMethodCodeSequence"):
        if keyword in dataset:
            del dataset[keyword]
    for element in dataset:
        if element.VR == "SQ":
            for item in element.value or ():
                _invalidate_identity_claims(item, root=False)


def _set_standard_deid_markers(
    dataset: Any, context: _Context, *, pixel_status: DicomPixelStatus
) -> None:
    if "DeidentificationMethodCodeSequence" in dataset:
        del dataset.DeidentificationMethodCodeSequence
    unclean = pixel_status is DicomPixelStatus.NOT_CLEANED
    dataset.PatientIdentityRemoved = "NO" if unclean else "YES"
    methods = ["PS3.15 Basic Profile 2026d"]
    methods.extend(_PROFILE_OPTIONS[name][1] for name in context.profile_options)
    if unclean:
        methods.insert(0, "OpenMed header processing; pixels not cleaned")
    dataset.DeidentificationMethod = methods
    dataset.LongitudinalTemporalInformationModified = (
        "MODIFIED"
        if "retain_longitudinal_temporal_information" in context.profile_options
        else "REMOVED"
    )
    # Incomplete pixel coverage never receives a whole-instance profile claim.
    if not unclean:
        pydicom = _import_pydicom()
        codes = [("113100", "Basic Application Confidentiality Profile")]
        codes.extend(_PROFILE_OPTIONS[name] for name in context.profile_options)
        items = []
        for value, meaning in codes:
            item = pydicom.dataset.Dataset()
            item.CodeValue = value
            item.CodingSchemeDesignator = "DCM"
            item.CodeMeaning = meaning
            items.append(item)
        dataset.DeidentificationMethodCodeSequence = items

    for tag_int, keyword, vr, ps315_action in (
        (0x00120062, "PatientIdentityRemoved", "CS", "D"),
        (0x00120063, "DeidentificationMethod", "LO", "D"),
        (0x00280303, "LongitudinalTemporalInformationModified", "CS", "D"),
        *(
            ((0x00120064, "DeidentificationMethodCodeSequence", "SQ", "D"),)
            if not unclean
            else ()
        ),
    ):
        context.actions.append(
            DicomHeaderAction(
                tag=_format_tag_int(tag_int),
                keyword=keyword,
                vr=vr,
                action="replace",
                ps315_action=ps315_action,
                location="Dataset",
            )
        )


def _deidentify_file_meta(dataset: Any, context: _Context) -> None:
    original = getattr(dataset, "file_meta", None)
    if original is None:
        return
    pydicom = _import_pydicom()
    file_meta = pydicom.dataset.FileMetaDataset()
    # Only decoding/identity essentials survive; all application-supplied meta
    # values, AE titles, addresses, private payloads and group lengths go away.
    for keyword in ("MediaStorageSOPClassUID", "TransferSyntaxUID"):
        if keyword in original:
            _validate_element_vr(original[keyword])
            setattr(file_meta, keyword, getattr(original, keyword))
    source_uid = str(
        dataset.get("SOPInstanceUID", original.get("MediaStorageSOPInstanceUID", ""))
    )
    if "SOPInstanceUID" not in dataset and "retain_uids" not in context.profile_options:
        source_uid = _remap_uid(source_uid, context)
    file_meta.MediaStorageSOPInstanceUID = source_uid
    file_meta.ImplementationClassUID = "2.25.19040426776935454670452470711936026468"
    file_meta.ImplementationVersionName = "OPENMED_DEID"
    for element in original:
        if element.keyword in {"MediaStorageSOPClassUID", "TransferSyntaxUID"}:
            continue
        _record_action(
            element,
            context,
            action="replace_file_meta",
            ps315_action="D",
            location="FileMetaDataset",
            record_value=False,
        )
    dataset.file_meta = file_meta


def _clear_element(
    element: Any,
    context: _Context,
    *,
    ps315_action: str,
    location: str,
) -> None:
    _record_action(
        element,
        context,
        action="clear",
        ps315_action=ps315_action,
        location=location,
    )
    element.value = [] if element.VR == "SQ" else ""


def _is_structural_uid(element: Any) -> bool:
    keyword = _keyword(element)
    return keyword.endswith("SOPClassUID") or keyword in {
        "TransferSyntaxUID",
        "ImplementationClassUID",
    }


def _should_remap_uid(element: Any) -> bool:
    return element.VR == "UI" and not _is_structural_uid(element)


def _remap_uid_element(element: Any, context: _Context, *, location: str) -> None:
    _record_action(
        element,
        context,
        action="replace_uid",
        ps315_action="U",
        location=location,
    )
    element.value = _map_dicom_values(
        element.value, lambda value: _remap_uid(value, context)
    )


def _remap_uid(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        raise DicomDeidentificationError("profile_uid_invalid")
    replacement = context.uid_map.get(text)
    if replacement is None:
        digest = hashlib.sha256(context.uid_salt + text.encode("utf-8")).digest()
        replacement = f"2.25.{uuid.UUID(bytes=digest[:16]).int}"
        context.uid_map[text] = replacement
    return replacement


def _shift_date_element(element: Any, context: _Context, *, location: str) -> None:
    _record_action(
        element,
        context,
        action="shift_date",
        ps315_action="C",
        location=location,
    )
    if element.VR == "DA":
        element.value = _map_dicom_values(
            element.value,
            lambda value: _shift_dicom_date(value, context),
        )
    else:
        element.value = _map_dicom_values(
            element.value,
            lambda value: _shift_dicom_datetime(value, context),
        )


def _map_dicom_values(value: Any, mapper: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [mapper(item) for item in value]
    return mapper(value)


def _shift_dicom_date(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        return text
    try:
        shifted = datetime.strptime(text, "%Y%m%d") + timedelta(
            days=context.date_shift_days
        )
        if context.keep_year:
            shifted = _replace_year_safe(shifted, int(text[:4]))
    except (TypeError, ValueError, OverflowError):
        return ""
    return shifted.strftime("%Y%m%d")


def _shift_dicom_datetime(value: Any, context: _Context) -> str:
    text = str(value).strip()
    if not text:
        return text
    if re.fullmatch(r"\d{8}(?:\d{2}){0,3}(?:\.\d{1,6})?(?:[+-]\d{4})?", text) is None:
        return ""
    match = _DT_DATE_RE.match(text)
    if match is None:
        return ""
    shifted = _shift_dicom_date(match.group("date"), context)
    return f"{shifted}{match.group('rest')}" if shifted else ""


def _replace_year_safe(date_value: datetime, year: int) -> datetime:
    try:
        return date_value.replace(year=year)
    except ValueError:
        return date_value.replace(year=year, month=2, day=28)


def _record_action(
    element: Any,
    context: _Context,
    *,
    action: str,
    ps315_action: str,
    location: str,
    record_value: bool = True,
) -> None:
    listed = _profile_key(element) in _BASIC_PROFILE_ACTIONS
    if not listed and location != "FileMetaDataset":
        record_value = False
    context.actions.append(
        DicomHeaderAction(
            tag=_format_tag(element.tag),
            keyword=_keyword(element) if listed else "",
            vr=str(element.VR) if listed else "",
            action=action,
            ps315_action=ps315_action,
            location=location,
            value_sha256=_hash_value(element.value) if record_value else None,
            value_length=len(str(element.value)) if record_value else None,
        )
    )


def _keyword(element: Any) -> str:
    keyword = getattr(element, "keyword", "")
    return str(keyword) if keyword else _format_tag(element.tag)


def _format_tag(tag: Any) -> str:
    return f"({int(tag) >> 16:04X},{int(tag) & 0xFFFF:04X})"


def _format_tag_int(tag: int) -> str:
    return f"({tag >> 16:04X},{tag & 0xFFFF:04X})"


def _hash_value(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _bytes_value(value: str | bytes, *, name: str) -> bytes:
    if isinstance(value, bytes):
        if not value:
            raise ValueError(f"{name} must be non-empty")
        return value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        if not encoded:
            raise ValueError(f"{name} must be non-empty")
        return encoded
    raise TypeError(f"{name} must be str or bytes")


def _save_dataset(dataset: Any, output_path: Path) -> None:
    try:
        dataset.save_as(output_path, enforce_file_format=True)
    except TypeError:  # pragma: no cover - older pydicom compatibility.
        dataset.save_as(output_path, write_like_original=False)


def _dicom_handler(
    path: str | Path,
    *,
    policy: Any = None,
    models: Any = None,
    lang: str | None = None,
) -> ExtractedDocument:
    pixel_result, dataset = _redact_pixels(
        path,
        policy=policy,
        models=models,
        lang=lang,
        save=False,
    )
    header_policy = _coerce_policy(policy)
    header_policy = replace(
        header_policy,
        output_path=pixel_result.output_path,
        detector=header_policy.detector
        if header_policy.detector is not None
        else models,
    )
    result = _deidentify_headers(
        pixel_result.output_path,
        policy=header_policy,
        processed_document_digests=pixel_result._processed_document_digests,
        dataset=dataset,
    )
    return ExtractedDocument(
        text="",
        metadata={
            "format": "dicom",
            "dicom_header_deid": result.to_audit_report(),
            "dicom_pixel_redaction": pixel_result.to_audit_report(),
        },
    )


register_handler(".dcm", _dicom_handler, requires_multimodal=False)


__all__ = [
    "DicomHeaderAction",
    "DicomHeaderDeidPolicy",
    "DicomHeaderDeidResult",
    "DicomPixelStatus",
    "DicomDeidentificationError",
    "deidentify_dicom_headers",
]
