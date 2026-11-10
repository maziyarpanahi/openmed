"""Six executable synthetic development controls; no clinical reviewer evidence.

Only small in-memory generated sources are used. Candidate observations are
computed by the real modality APIs. Known marker detection is not PHI recall;
ECG confirmation below is explicitly simulated for synthetic waveforms.
"""

from __future__ import annotations

import hashlib
import io
import json
import platform
from dataclasses import asdict, replace
from importlib import import_module, metadata
from pathlib import Path

from openmed.eval.suites.multimodal_clinical import (
    CandidateLineage,
    CandidateObservation,
    ReviewAsset,
    ValidationCase,
    ValidationPack,
    ValidationSlice,
    ValidationTask,
)


def digest(value: str | bytes) -> str:
    """Fingerprint synthetic fixture source or explicitly described configuration."""
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


def _observation(
    content, *, abstained=False, privacy_failure=False, provenance_failure=False
):
    return CandidateObservation(
        content=content,
        abstained=abstained,
        critical_privacy_failure=privacy_failure,
        provenance_failure=provenance_failure,
    )


def _png(image, *, format="PNG"):
    stream = io.BytesIO()
    image.save(stream, format=format)
    return stream.getvalue()


def _ecg(task):
    from openmed.multimodal.ecg.interval_candidates import (
        IntervalPolicy,
        SampleRange,
        extract_interval_candidates,
    )
    from openmed.multimodal.ecg.lead_layout import (
        AcquisitionKind,
        LeadLayout,
        LeadObservation,
    )
    from openmed.multimodal.ecg.signal_quality import (
        AcquisitionEvidence,
        CalibrationSegment,
        LeadCalibration,
        QualityLimits,
        SignalQualityError,
        SignalQualityPolicy,
        assess_signal_quality,
        waveform_fingerprint,
    )
    from openmed.multimodal.ecg.waveform import EcgLead, EcgProvenance, EcgWaveform

    values = [0.0] * 450
    for anchor in (100, 200, 300):
        for start, peak, end, amplitude in (
            (anchor - 4, anchor, anchor + 4, 1),
            (anchor - 30, anchor - 25, anchor - 20, 0.3),
            (anchor + 20, anchor + 30, anchor + 40, 0.4),
        ):
            for index in range(start, end + 1):
                values[index] += amplitude * (
                    (index - start) / (peak - start)
                    if index <= peak
                    else (end - index) / (end - peak)
                )
    payload = json.dumps(
        {"synthetic": True, "rate_hz": 100, "I": values, "II": values}
    ).encode()
    asset = ReviewAsset(media_type="application/json", payload=payload)
    wave = EcgWaveform(
        leads=tuple(
            EcgLead.from_samples(lead, values, unit="mV") for lead in ("I", "II")
        ),
        sample_rate_hz=100,
        provenance=EcgProvenance("array", asset.digest),
        start_offset_seconds=0,
    )
    quality = SignalQualityPolicy(
        version="synthetic-v1",
        evidence_digest=digest("synthetic quality thresholds"),
        required_leads=("I", "II"),
        sample_rate_hz=100,
        minimum_duration_seconds=1,
        block_samples=50,
        minimum_block_valid_fraction=0.9,
        flatline_tolerance_mv=0.001,
        rail_tolerance_mv=0.01,
        missing_fraction=QualityLimits(0.01, 0.05),
        gap_seconds=QualityLimits(0.02, 0.1),
        clipped_fraction=QualityLimits(0, 0.01),
        flatline_seconds=QualityLimits(2, 3),
        baseline_span_mv=QualityLimits(0.2, 0.4),
        roughness_mv=QualityLimits(0.3, 0.6),
        gain_relative_error=QualityLimits(0.001, 0.01),
    )
    acquisition = AcquisitionEvidence(
        source_digest=asset.digest,
        waveform_digest=waveform_fingerprint(wave),
        layout=LeadLayout.digital(("I", "II")),
        acquisition=AcquisitionKind.DIGITAL,
        observations=tuple(
            LeadObservation(
                label=lead,
                row=index,
                start_seconds=0,
                duration_seconds=4.5,
                sample_rate_hz=100,
                sample_count=450,
            )
            for index, lead in enumerate(("I", "II"))
        ),
        calibrations=tuple(
            LeadCalibration(
                lead=lead,
                expected_gain_counts_per_mv=200,
                segments=(
                    CalibrationSegment(
                        start_sample=0,
                        end_sample=450,
                        gain_counts_per_mv=200,
                        lower_rail_mv=-5,
                        upper_rail_mv=5,
                    ),
                ),
            )
            for lead in ("I", "II")
        ),
    )
    interval = IntervalPolicy(
        version="synthetic-v1",
        evidence_digest=digest("synthetic interval thresholds"),
        sample_rate_hz=100,
        boundary_low_mv=0.01,
        boundary_high_mv=0.025,
        secondary_peak_mv=0.1,
        qrs_peak_mv=0.8,
        qrs_slope_mv_per_sample=0.2,
        quiet_samples=2,
        join_samples=1,
        baseline_fraction=0.5,
        qrs_width=SampleRange(3, 15),
        secondary_width=SampleRange(5, 30),
        rr_spacing=SampleRange(60, 150),
        p_search=SampleRange(5, 35),
        t_search=SampleRange(5, 60),
        boundary_uncertainty_samples=3,
        alignment_samples=3,
        minimum_beats=3,
        minimum_leads=2,
        rate_spread_bpm=10,
        interval_spread_ms=40,
    )
    outputs = []
    for index in range(2):
        settings = replace(quality, minimum_duration_seconds=1 if index == 0 else 5)
        if task is ValidationTask.ECG_QUALITY:
            result = assess_signal_quality(wave, settings, acquisition)
            outputs.append(
                _observation(result.to_json(), abstained=not result.measurement_allowed)
            )
        else:
            try:
                result = extract_interval_candidates(
                    wave, settings, acquisition, interval, reviewer_confirmed=True
                )
            except SignalQualityError:
                report = assess_signal_quality(wave, settings, acquisition)
                outputs.append(_observation(report.to_json(), abstained=True))
                continue
            outputs.append(
                _observation(
                    json.dumps(
                        {
                            "measurements": [
                                asdict(item) for item in result.measurements
                            ],
                            "leads": [asdict(item) for item in result.leads],
                            "report": json.loads(result.to_report_json()),
                        }
                    ),
                    abstained=not result.quality.measurement_allowed,
                )
            )
    return (
        (asset,),
        tuple(outputs),
        "Synthetic triangles at 100 Hz; known timing anchors 100/200/300 samples. Confirmation is simulated. Compare duration gates 1s/5s.",
    )


def _form():
    from PIL import Image, ImageDraw

    from openmed.multimodal.evidence import EvidenceNode, NodeKind, PageGeometry
    from openmed.multimodal.vision.forms.field_grounding import (
        FormLabel,
        FormSection,
        FormValue,
        GroundingPolicy,
        LabelRelation,
        ValueKind,
        VisualForm,
        ground_form_fields,
    )
    from openmed.multimodal.vision.providers.base import Confidence

    image = Image.new("RGB", (400, 100), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 20), "Synthetic field", fill="black")
    draw.text((150, 20), "Synthetic value", fill="black")
    asset = ReviewAsset(media_type="image/png", payload=_png(image))

    def node(index, box):
        return EvidenceNode(
            node_id=f"{index:032x}",
            kind=NodeKind.PIXELS,
            source_digest=asset.digest,
            geometry=PageGeometry(page=0, width=400, height=100),
            region=box,
        )

    form = VisualForm(
        sections=(FormSection(node(1, (0, 0, 400, 100))),),
        labels=(
            FormLabel(
                evidence=node(2, (20, 20, 130, 35)),
                text="Synthetic field",
                section_id=f"{1:032x}",
                relation=LabelRelation.RIGHT,
                accepts=(ValueKind.TEXT,),
                confidence=Confidence(score=1),
            ),
        ),
        values=(
            FormValue(
                evidence=node(3, (150, 20, 260, 35)),
                kind=ValueKind.TEXT,
                text="Synthetic value",
                confidence=Confidence(score=1),
            ),
        ),
    )
    outputs = tuple(
        _observation(
            ground_form_fields(
                form,
                GroundingPolicy(
                    version="synthetic-v1",
                    maximum_gap=gap,
                    minimum_axis_overlap=0.5,
                    minimum_score=0.1,
                    ambiguity_margin=0.05,
                ),
            ).to_json()
        )
        for gap in (0.2, 0.01)
    )
    return (
        (asset,),
        outputs,
        "Generated form with one label/value. Compare maximum normalized gap 0.2/0.01. Proposals remain review-required.",
    )


def _dicom():
    import numpy as np
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    from openmed.multimodal.vision.dicom.series_sampler import (
        DisplayWindow,
        SamplingPolicy,
        sample_dicom_series,
    )

    assets = []
    for index in range(6):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID = CTImageStorage
        ds.SOPInstanceUID = f"1.2.826.0.1.3680043.10.543.{index + 1}"
        ds.SeriesInstanceUID = "1.2.826.0.1.3680043.10.543.1000"
        ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.10.543.2000"
        ds.Modality = "CT"
        ds.Rows = ds.Columns = 16
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.ImagePositionPatient = [0, 0, index * 2]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.AcquisitionNumber = 1
        ds.PixelData = (np.arange(256, dtype="<u2") + index).tobytes()
        stream = io.BytesIO()
        pydicom.dcmwrite(stream, ds, enforce_file_format=True)
        assets.append(
            ReviewAsset(media_type="application/dicom", payload=stream.getvalue())
        )
    outputs = tuple(
        _observation(
            sample_dicom_series(
                tuple(asset.payload for asset in assets),
                window=DisplayWindow(center=128, width=256),
                policy=SamplingPolicy(max_selected=count, max_gap_mm=4),
            ).to_json()
        )
        for count in (4, 2)
    )
    return (
        tuple(assets),
        outputs,
        "Six native synthetic CT frames at 2mm spacing; compare budgets of four/two frames. Coverage is not diagnostic adequacy.",
    )


def _pixel():
    import numpy as np
    from PIL import Image

    from openmed.multimodal.privacy.pixel_coverage import (
        LocalPixelDetector,
        PixelFramePlan,
        PixelInspection,
        audit_pixel_redaction,
        inventory_pixel_frames,
    )

    image = Image.new("RGB", (16, 12), (30, 30, 30))
    image.paste((255, 255, 255), (5, 3, 9, 6))
    asset = ReviewAsset(media_type="image/png", payload=_png(image))

    def inspect(frame):
        ys, xs = np.where(np.asarray(frame).min(axis=2) > 250)
        boxes = (
            ((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),)
            if len(xs)
            else ()
        )
        return PixelInspection(boxes, True)

    outputs = []
    for masks in (((5, 3, 9, 6),), ()):
        plans = tuple(
            PixelFramePlan(reference=item.reference, masks=masks)
            for item in inventory_pixel_frames(asset.payload, source_kind="raster")
        )
        result = audit_pixel_redaction(
            asset.payload,
            plans,
            source_kind="raster",
            detector=LocalPixelDetector(
                digest=digest("synthetic bright marker"),
                inspect=inspect,
                local_only=True,
            ),
            residual_detector=LocalPixelDetector(
                digest=digest("synthetic residual bright marker"),
                inspect=inspect,
                local_only=True,
            ),
        )
        outputs.append(
            _observation(
                result.to_json(),
                abstained=not result.accepted,
                privacy_failure=not result.accepted,
            )
        )
    return (
        (asset,),
        tuple(outputs),
        "Known bright-marker carrier, with a complete/omitted mask. Detector coverage is synthetic wiring, not clinical PHI sensitivity.",
    )


def _wsi():
    from PIL import Image

    from openmed.multimodal.vision.pathology.slide_tiler import (
        PyramidLevel,
        TilePolicy,
        plan_slide_tiles,
    )
    from openmed.multimodal.vision.pathology.tile_provenance import (
        TileFeatures,
        aggregate_tile_features,
    )

    image = Image.new("RGB", (8, 6))
    image.putdata([(x * 20, y * 30, 80) for y in range(6) for x in range(8)])
    asset = ReviewAsset(media_type="image/tiff", payload=_png(image, format="TIFF"))
    outputs = []
    for width in (4, 8):
        plan = plan_slide_tiles(
            asset.digest,
            8,
            6,
            (PyramidLevel(index=0, width=8, height=6, scale_numerator=1),),
            TilePolicy(
                width=width,
                height=3,
                overlap_x=0,
                overlap_y=0,
                working_bytes_per_pixel=12,
                max_inflight_tiles=2,
                budget_bytes=4096,
                reserved_bytes=128,
                min_tissue_percent=25,
            ),
        )
        features = []
        for tile in plan.selected_tiles:
            crop = image.crop(tile.bounds)
            pixels = [
                crop.getpixel((x, y))
                for y in range(crop.height)
                for x in range(crop.width)
            ]
            features.append(
                TileFeatures(
                    tile_ref=tile.tile_ref,
                    plan_digest=plan.digest,
                    source_digest=asset.digest,
                    level=tile.level,
                    native_bounds=tile.bounds,
                    source_bounds=tile.source_bounds,
                    processed_input_digest=digest(crop.tobytes()),
                    provider_digest=digest("synthetic RGB mean"),
                    feature_space_digest=digest("three-channel RGB mean"),
                    transforms=(),
                    features=tuple(
                        sum(pixel[channel] for pixel in pixels) / len(pixels)
                        for channel in range(3)
                    ),
                )
            )
        outputs.append(
            _observation(aggregate_tile_features(plan, tuple(features)).to_json())
        )
    return (
        (asset,),
        tuple(outputs),
        "Generated 8x6 RGB slide; execute 4x3/8x3 tiling and actual crop RGB means. No pathology model or clinical claims.",
    )


def development_pack(task: ValidationTask) -> ValidationPack:
    """Run one real synthetic component control with two declared configurations.

    Each pack has one source unit and is intentionally too small for publication
    of rates. Repeated execution does not create independent clinical cases.
    """
    if task in (ValidationTask.ECG_QUALITY, ValidationTask.ECG_INTERVAL):
        assets, observations, context = _ecg(task)
    else:
        assets, observations, context = {
            ValidationTask.FORM: _form,
            ValidationTask.DICOM: _dicom,
            ValidationTask.PIXEL_PHI: _pixel,
            ValidationTask.WSI: _wsi,
        }[task]()
    # Hash fixture code itself so changes to construction/configuration alter lineage.
    harness = digest(Path(__file__).read_bytes())
    component = {
        ValidationTask.ECG_QUALITY: "openmed.multimodal.ecg.signal_quality",
        ValidationTask.ECG_INTERVAL: "openmed.multimodal.ecg.interval_candidates",
        ValidationTask.FORM: "openmed.multimodal.vision.forms.field_grounding",
        ValidationTask.DICOM: "openmed.multimodal.vision.dicom.series_sampler",
        ValidationTask.PIXEL_PHI: "openmed.multimodal.privacy.pixel_coverage",
        ValidationTask.WSI: "openmed.multimodal.vision.pathology.tile_provenance",
    }[task]
    code = digest(Path(import_module(component).__file__).read_bytes())
    runtime = digest(
        json.dumps(
            {
                "python": platform.python_version(),
                "packages": {
                    name: metadata.version(name)
                    for name in ("Pillow", "numpy", "pydicom")
                },
            },
            sort_keys=True,
        )
    )
    hardware = digest(
        json.dumps(
            {
                "system": platform.system(),
                "release": platform.release(),
                "architecture": platform.machine(),
            },
            sort_keys=True,
        )
    )
    return ValidationPack(
        task=task,
        cases=(
            ValidationCase(
                assets=assets,
                context=context,
                candidates=observations,
                slice=ValidationSlice.NOMINAL,
            ),
        ),
        candidates=tuple(
            CandidateLineage(
                model_digest=digest(
                    f"{code}; synthetic algorithm {task.value}; configuration {index}"
                ),
                preprocessing_digest=digest(context + str(index)),
                runtime_digest=runtime,
                hardware_digest=hardware,
                harness_digest=harness,
                license_digest=digest("Apache-2.0 fixture source; no external weights"),
            )
            for index in range(2)
        ),
        sampling_manifest_digest=digest(
            "one generated source; convenience software control; not population sampling"
        ),
        dataset_license_digest=digest("Apache-2.0 generated synthetic control"),
    )
