"""Synthetic burned-in-PHI DICOM benchmark suite (OM-162).

This suite certifies the "zero residual PHI" claim for OpenMed's DICOM
de-identification path with a fully reproducible, no-DUA scoring corpus. It

1. deterministically generates synthetic medical images that carry PHI *both*
   in the DICOM header (patient name, MRN, dates) *and* rendered ("burned in")
   onto the pixel data at known bounding boxes, recording gold span + bbox
   truth from a fixed seed;
2. runs the real DICOM redaction path -- burned-in pixel text redaction
   (:func:`openmed.multimodal.redact_dicom_pixels`) followed by header
   de-identification (:func:`openmed.multimodal.deidentify_dicom_headers`) --
   into one final artifact; and
3. scores the result with the eval harness's leakage-first metrics
   (residual-PHI rate + character recall) and emits a
   :class:`~openmed.eval.report.BenchmarkReport` so it can feed the leaderboard
   / status page.

All PHI is synthetic (faked names, MRNs, and dates); no real images or PHI are
bundled. The corpus is generated from fully synthetic phantom images -- no TCIA
collection is sampled or committed -- so there is no data-use-agreement or
license-compatibility question to resolve at runtime.

``pydicom`` and ``Pillow`` are optional dependencies gated behind the
``multimodal`` extra. They are imported lazily inside the generator; callers who
have not installed the extra get a clean :class:`MissingDependencyError` rather
than an import-time failure, and the accompanying tests skip cleanly.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from faker import Faker

from openmed.core.labels import DATE, ID_NUM, PERSON
from openmed.eval.harness import BenchmarkFixture
from openmed.eval.metrics import (
    EvalSpan,
    compute_character_recall,
    compute_leakage_rate,
    compute_recall_slices,
)
from openmed.eval.report import BenchmarkReport
from openmed.multimodal.exceptions import MissingDependencyError

MULTIMODAL_DICOM = "multimodal_dicom"

#: Provenance is fully synthetic: phantom images generated from a seed, with
#: synthetic PHI burned in. Nothing is sampled from TCIA or any DUA-gated
#: source, so there is no restricted data committed to the repository.
CORPUS_PROVENANCE = "fully-synthetic-phantom"
CORPUS_LICENSE = "generated-synthetic"
DATA_USE_AGREEMENT_REQUIRED = False
TCIA_SAMPLED = False

#: Default deterministic seed for the generated corpus.
DEFAULT_SEED = 20240917
#: Default number of synthetic studies in the corpus.
DEFAULT_CORPUS_SIZE = 6
#: Default system name recorded in the benchmark report.
DEFAULT_SYSTEM_NAME = "openmed-dicom-redaction"

#: Device tier reported for pixel-space (image) PHI in the harness slices.
PIXEL_DEVICE = "cpu"

# Header keyword -> canonical PHI label for the header truth set.
_HEADER_PHI_LABELS: dict[str, str] = {
    "PatientName": PERSON,
    "PatientID": ID_NUM,
    "PatientBirthDate": DATE,
    "StudyDate": DATE,
}

# Keywords whose header values must be gone from the de-identified dataset for a
# clean pass. ``PatientName`` / ``PatientID`` are cleared to empty strings;
# ``PatientBirthDate`` / ``StudyDate`` are date-shifted so the *original* value
# must not survive.
_HEADER_DIRECT_IDENTIFIERS: tuple[str, ...] = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "StudyDate",
)


@dataclass(frozen=True)
class BurnedInPhiWord:
    """One burned-in PHI token with its gold pixel bbox and canonical label."""

    text: str
    label: str
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1)
    frame_index: int = 0

    def to_ocr_word(self) -> Any:
        """Return the token as an :class:`openmed.multimodal.OcrWord`."""
        from openmed.multimodal import OcrWord

        x0, y0, x1, y1 = self.bbox
        return OcrWord(
            self.text,
            (float(x0), float(y0), float(x1), float(y1)),
            0.99,
            page=self.frame_index,
        )


@dataclass(frozen=True)
class SyntheticDicomCase:
    """A generated synthetic DICOM study with header + pixel PHI truth."""

    case_id: str
    path: Path
    header_phi: Mapping[str, str]
    pixel_words: tuple[BurnedInPhiWord, ...]
    #: Concatenated burned-in pixel text used as the source string for
    #: character-offset leakage/recall scoring.
    pixel_text: str
    pixel_gold_spans: tuple[EvalSpan, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GoldBoxOcrEngine:
    """Deterministic OCR engine that reads burned-in words from known bboxes.

    The engine reports a word only when the pixels inside its gold bbox are
    still "inked" (non-zero). This lets the benchmark drive the *real* redaction
    path deterministically: on the original image every gold word is returned,
    and on a correctly redacted (blacked-out) image none survive -- exactly the
    contract the DICOM pixel-redaction path expects from an OCR backend, without
    depending on a heavy OCR model. It mirrors the fixture engine the multimodal
    tests use so the benchmark measures redaction wiring, not OCR quality.
    """

    name = "gold-box-ocr"

    def __init__(self, words: Sequence[BurnedInPhiWord]) -> None:
        self._words = tuple(words)

    def recognize(self, image: Any, *, languages: Any = None) -> Any:
        del languages
        from openmed.multimodal import OcrResult

        np = _import_numpy()
        array = np.asarray(image)
        found = []
        for word in self._words:
            x0, y0, x1, y1 = word.bbox
            region = array[y0:y1, x0:x1]
            if region.size and region.max(initial=0) > 0:
                found.append(word.to_ocr_word())
        return OcrResult(words=tuple(found), metadata={"engine": self.name})


def multimodal_dicom_metadata(
    *,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
) -> dict[str, Any]:
    """Return provenance, license, and generation metadata for the suite."""
    return {
        "suite": MULTIMODAL_DICOM,
        "corpus_provenance": CORPUS_PROVENANCE,
        "license": CORPUS_LICENSE,
        "requires_data_use_agreement": DATA_USE_AGREEMENT_REQUIRED,
        "tcia_sampled": TCIA_SAMPLED,
        "phi_kind": "synthetic",
        "seed": seed,
        "corpus_size": corpus_size,
        "redaction_path": [
            "openmed.multimodal.redact_dicom_pixels",
            "openmed.multimodal.deidentify_dicom_headers",
        ],
        "notes": (
            "Fully synthetic phantom DICOMs with faked burned-in PHI; no TCIA "
            "collection is sampled and no DUA-gated data is committed."
        ),
    }


def generate_synthetic_dicom_corpus(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
) -> list[SyntheticDicomCase]:
    """Generate a deterministic synthetic burned-in-PHI DICOM corpus.

    Each case writes a single-frame phantom DICOM with synthetic header PHI and
    a synthetic patient name / MRN / study date rendered onto the pixels at
    recorded bounding boxes. Generation is fully deterministic in *seed*.

    Args:
        output_dir: Directory to write the ``.dcm`` files into (created if
            needed).
        seed: Deterministic seed controlling the synthetic PHI and layout.
        corpus_size: Number of synthetic studies to generate.

    Returns:
        A list of :class:`SyntheticDicomCase`, one per generated study.

    Raises:
        MissingDependencyError: If ``pydicom`` or ``Pillow`` (the ``multimodal``
            extra) are not installed.
        ValueError: If *corpus_size* is not positive.
    """
    if corpus_size <= 0:
        raise ValueError("corpus_size must be a positive integer")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    faker = Faker("en_US")
    faker.seed_instance(seed)

    cases: list[SyntheticDicomCase] = []
    for index in range(corpus_size):
        case = _generate_case(out_dir, faker, index=index, seed=seed)
        cases.append(case)
    return cases


def run_multimodal_dicom(
    *,
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
    cases: Sequence[SyntheticDicomCase] | None = None,
    system_name: str = DEFAULT_SYSTEM_NAME,
    pii_model_name: str | None = None,
    device: str = "cpu",
    generated_at: str | None = None,
    date_shift_days: int = 17,
    offline: bool = True,
) -> BenchmarkReport:
    """Score the DICOM redaction path over the synthetic corpus.

    Generates (or reuses) the synthetic corpus, runs the real header + pixel
    redaction path over every case, and returns a leakage-first
    :class:`BenchmarkReport`. The pixel and header passes are both scored:

    * ``metrics["leakage"]`` -- character-weighted residual-PHI rate over the
      burned-in pixel text (``overall`` is the headline residual-PHI rate).
    * ``metrics["character_recall"]`` / ``metrics["recall_slices"]`` --
      character recall of the pixel-text detector.
    * ``metrics["header_residual_phi_rate"]`` -- fraction of header direct
      identifiers whose original value survived de-identification.

    Because every burned-in token is also present in the DICOM header, the
    pixel-redaction path's header-seeded deny recognizer detects it without a
    token-classification model. When *offline* is ``True`` (the default) the
    pixel PHI model call is short-circuited so the benchmark runs fully
    local-first, with no network or ``transformers`` dependency, and measures
    the OCR -> detect -> black-out -> residual-re-OCR wiring rather than model
    quality. Set *offline* to ``False`` to exercise the real model backend when
    it is installed.

    Args:
        output_dir: Directory for generated DICOMs. A temporary directory is
            used when omitted (and *cases* is not supplied).
        seed: Deterministic corpus seed.
        corpus_size: Number of synthetic studies.
        cases: Pre-generated corpus to reuse instead of regenerating.
        system_name: System name recorded in the benchmark report.
        pii_model_name: Actual PII checkpoint used by the pixel detector. When
            omitted, the registered default English PII model is used.
        device: Reported device tier.
        generated_at: Optional ISO timestamp for the report.
        date_shift_days: Non-zero header date shift applied by the header pass.
        offline: When ``True`` (default), short-circuit the pixel PHI model so
            detection relies on the header-seeded deny recognizer only.

    Returns:
        A :class:`BenchmarkReport` in the harness report format.

    Raises:
        MissingDependencyError: If the ``multimodal`` extra is not installed.
        ValueError: If *cases* is supplied as an empty corpus.
    """
    import tempfile

    from openmed.multimodal import (
        DicomHeaderDeidPolicy,
        deidentify_dicom_headers,
        redact_dicom_pixels,
    )

    pydicom = _import_pydicom()
    np = _import_numpy()

    temp_holder: tempfile.TemporaryDirectory[str] | None = None
    if cases is None:
        if output_dir is None:
            temp_holder = tempfile.TemporaryDirectory(prefix="om162-dicom-")
            corpus_dir: str | Path = temp_holder.name
        else:
            corpus_dir = output_dir
        corpus = generate_synthetic_dicom_corpus(
            corpus_dir, seed=seed, corpus_size=corpus_size
        )
    else:
        corpus = list(cases)
    if not corpus:
        raise ValueError("cases must contain at least one synthetic DICOM case")

    resolved_pii_model = pii_model_name or _default_pii_model_name()

    pixel_gold: list[EvalSpan] = []
    pixel_pred: list[EvalSpan] = []
    pixel_source_parts: list[str] = []
    offset = 0
    residual_pixel_findings = 0
    header_identifier_total = 0
    header_identifier_residual = 0
    final_artifact_hashes: list[str] = []

    model_context = _offline_pixel_model() if offline else contextlib.nullcontext()
    try:
        with model_context:
            for case in corpus:
                # Pixel redaction runs first so the real path can seed its deny
                # recognizer from the original headers. Header de-identification
                # then consumes that output, leaving one final artifact with
                # both surfaces de-identified. The PHI-bearing intermediate is
                # always removed.
                pixel_stage = case.path.with_name(f"{case.path.stem}.pixel-stage.dcm")
                final_out = case.path.with_name(f"{case.path.stem}.redacted.dcm")
                final_out.unlink(missing_ok=True)
                try:
                    result = redact_dicom_pixels(
                        case.path,
                        output_path=pixel_stage,
                        ocr_engine=GoldBoxOcrEngine(case.pixel_words),
                        model_name=resolved_pii_model,
                        verify_residual=True,
                        fail_on_residual=False,
                    )
                    deidentify_dicom_headers(
                        pixel_stage,
                        policy=DicomHeaderDeidPolicy(
                            output_path=final_out,
                            date_shift_days=date_shift_days,
                        ),
                    )
                except Exception:
                    final_out.unlink(missing_ok=True)
                    raise
                finally:
                    pixel_stage.unlink(missing_ok=True)

                deidentified = pydicom.dcmread(final_out)
                final_artifact_hashes.append(_sha256_file(final_out))
                for keyword in _HEADER_DIRECT_IDENTIFIERS:
                    original = case.header_phi.get(keyword)
                    if not original:
                        continue
                    header_identifier_total += 1
                    if _header_value_survives(deidentified, keyword, original):
                        header_identifier_residual += 1

                residual_pixel_findings += result.residual_report.residual_entity_count
                final_pixels = np.asarray(deidentified.pixel_array)
                frame_count = int(getattr(deidentified, "NumberOfFrames", 1) or 1)

                # Offset every case's spans into a single concatenated source
                # string so character-level leakage/recall accounting is well
                # defined.
                for span in case.pixel_gold_spans:
                    pixel_gold.append(
                        EvalSpan(
                            start=span.start + offset,
                            end=span.end + offset,
                            label=span.label,
                            text=span.text,
                            language=span.language,
                            device=PIXEL_DEVICE,
                            metadata=dict(span.metadata),
                        )
                    )

                # Credit a word only when its final gold region contains no
                # residual ink. Scoring the final pixels prevents a tiny,
                # overlapping redaction bbox from counting as a full success.
                cursor = 0
                for word in case.pixel_words:
                    start = case.pixel_text.index(word.text, cursor)
                    cursor = start + len(word.text)
                    if _word_is_fully_redacted(
                        word,
                        final_pixels,
                        frame_count=frame_count,
                    ):
                        pixel_pred.append(
                            EvalSpan(
                                start=start + offset,
                                end=start + offset + len(word.text),
                                label=word.label,
                                text=word.text,
                                language="en",
                                device=PIXEL_DEVICE,
                            )
                        )

                pixel_source_parts.append(case.pixel_text)
                offset += len(case.pixel_text) + 1  # +1 for the join separator
    finally:
        if temp_holder is not None:
            temp_holder.cleanup()

    pixel_source = "\n".join(pixel_source_parts)

    leakage = compute_leakage_rate(
        pixel_gold,
        pixel_pred,
        default_device=PIXEL_DEVICE,
        source_text=pixel_source,
    )
    recall = compute_character_recall(
        pixel_gold,
        pixel_pred,
        default_device=PIXEL_DEVICE,
        source_text=pixel_source,
    )
    recall_slices = compute_recall_slices(
        pixel_gold,
        pixel_pred,
        default_device=PIXEL_DEVICE,
        source_text=pixel_source,
    )

    header_residual_rate = (
        header_identifier_residual / header_identifier_total
        if header_identifier_total
        else 0.0
    )

    metrics: dict[str, Any] = {
        "leakage": leakage.to_dict(),
        "residual_phi_rate": leakage.overall,
        "character_recall": recall.to_dict(),
        "recall_slices": recall_slices.to_dict(),
        "pixel_residual_finding_count": residual_pixel_findings,
        "header_residual_phi_rate": header_residual_rate,
        "header_direct_identifier_count": header_identifier_total,
        "header_residual_identifier_count": header_identifier_residual,
    }

    metadata = multimodal_dicom_metadata(seed=seed, corpus_size=len(corpus))
    metadata["case_ids"] = [case.case_id for case in corpus]
    metadata["date_shift_days"] = date_shift_days
    metadata["offline"] = offline
    metadata["system_name"] = system_name
    metadata["pii_model_name"] = resolved_pii_model
    metadata["artifact_pipeline"] = "pixel-redaction-then-header-deidentification"
    metadata["final_artifact_sha256"] = final_artifact_hashes

    return BenchmarkReport(
        suite=MULTIMODAL_DICOM,
        model_name=system_name,
        device=device,
        fixture_count=len(corpus),
        metrics=metrics,
        generated_at=generated_at,
        metadata=metadata,
    )


def load_multimodal_dicom_fixtures(
    *,
    output_dir: str | Path | None = None,
    seed: int = DEFAULT_SEED,
    corpus_size: int = DEFAULT_CORPUS_SIZE,
) -> list[BenchmarkFixture]:
    """Generate the corpus and expose the pixel PHI as benchmark fixtures.

    Each fixture's ``text`` is the burned-in pixel text and ``gold_spans`` are
    the pixel PHI spans, so the corpus plugs into the generic harness. The
    on-disk DICOM path and header truth are carried in fixture metadata.
    """
    import tempfile

    if output_dir is None:
        # Keep the generated files alive for the life of the process so callers
        # can still read the DICOM paths from fixture metadata.
        holder = tempfile.mkdtemp(prefix="om162-dicom-fixtures-")
        corpus_dir: str | Path = holder
    else:
        corpus_dir = output_dir

    corpus = generate_synthetic_dicom_corpus(
        corpus_dir, seed=seed, corpus_size=corpus_size
    )
    base_metadata = multimodal_dicom_metadata(seed=seed, corpus_size=len(corpus))
    fixtures: list[BenchmarkFixture] = []
    for case in corpus:
        fixture_metadata = dict(base_metadata)
        fixture_metadata.update(
            {
                "dicom_path": str(case.path),
                "header_phi_labels": dict(_HEADER_PHI_LABELS),
                "pixel_word_count": len(case.pixel_words),
                "source_dicom_sha256": case.metadata["source_dicom_sha256"],
            }
        )
        fixtures.append(
            BenchmarkFixture(
                fixture_id=case.case_id,
                text=case.pixel_text,
                gold_spans=case.pixel_gold_spans,
                language="en",
                metadata=fixture_metadata,
            )
        )
    return fixtures


# ---------------------------------------------------------------------------
# Generation internals
# ---------------------------------------------------------------------------


def _generate_case(
    out_dir: Path,
    faker: Faker,
    *,
    index: int,
    seed: int,
) -> SyntheticDicomCase:
    np = _import_numpy()

    name = faker.name()
    mrn = f"MRN-{faker.numerify('#######')}"
    study_date = faker.date_of_birth(minimum_age=1, maximum_age=90).strftime("%Y%m%d")
    birth_date = faker.date_of_birth(minimum_age=18, maximum_age=95).strftime("%Y%m%d")

    # DICOM PatientName uses caret-delimited "Family^Given" form.
    family, _, given = name.partition(" ")
    dicom_patient_name = f"{family}^{given}" if given else family

    header_phi = {
        "PatientName": dicom_patient_name,
        "PatientID": mrn,
        "PatientBirthDate": birth_date,
        "StudyDate": study_date,
    }

    # Three burned-in lines: patient name, MRN, and the study date.
    burned_lines = [
        (name, PERSON),
        (mrn, ID_NUM),
        (_format_display_date(study_date), DATE),
    ]

    pixels, words = _render_burned_in_frame(np, burned_lines)

    path = out_dir / f"om162-synthetic-{index:03d}.dcm"
    _write_pixel_dicom(
        np,
        path,
        pixels,
        header_phi,
        uid_entropy=f"{seed}:{index}",
    )

    pixel_text = " ".join(word.text for word in words)
    pixel_gold_spans = _spans_for_words(words, pixel_text)

    return SyntheticDicomCase(
        case_id=f"om162-synthetic-{index:03d}",
        path=path,
        header_phi=header_phi,
        pixel_words=tuple(words),
        pixel_text=pixel_text,
        pixel_gold_spans=pixel_gold_spans,
        metadata={
            "seed": seed,
            "index": index,
            "source_dicom_sha256": _sha256_file(path),
        },
    )


def _render_burned_in_frame(
    np: Any,
    lines: Sequence[tuple[str, str]],
) -> tuple[Any, list[BurnedInPhiWord]]:
    """Render PHI lines onto a phantom frame; return pixels + word bboxes."""
    image_mod, draw_mod, font_mod = _import_pillow()

    width, height = 320, 200
    # Phantom "anatomy": a mid-grey ellipse so the frame is not blank.
    image = image_mod.new("L", (width, height), color=0)
    draw = draw_mod.Draw(image)
    draw.ellipse((40, 60, 280, 190), fill=90)

    font = font_mod.load_default()

    words: list[BurnedInPhiWord] = []
    y = 8
    line_height = 18
    for text, label in lines:
        x = 8
        for token in text.split(" "):
            if not token:
                continue
            # Measure the token box so the gold bbox matches the drawn pixels.
            box = draw.textbbox((x, y), token, font=font)
            draw.text((x, y), token, fill=255, font=font)
            x0 = max(int(box[0]) - 1, 0)
            y0 = max(int(box[1]) - 1, 0)
            x1 = min(int(box[2]) + 1, width)
            y1 = min(int(box[3]) + 1, height)
            words.append(
                BurnedInPhiWord(text=token, label=label, bbox=(x0, y0, x1, y1))
            )
            x = int(box[2]) + 6
        y += line_height

    pixels = np.asarray(image, dtype=np.uint8)
    return pixels, words


def _spans_for_words(
    words: Sequence[BurnedInPhiWord],
    pixel_text: str,
) -> tuple[EvalSpan, ...]:
    spans: list[EvalSpan] = []
    cursor = 0
    for word in words:
        start = pixel_text.index(word.text, cursor)
        end = start + len(word.text)
        cursor = end
        spans.append(
            EvalSpan(
                start=start,
                end=end,
                label=word.label,
                text=word.text,
                language="en",
                device=PIXEL_DEVICE,
            )
        )
    return tuple(spans)


def _write_pixel_dicom(
    np: Any,
    path: Path,
    pixels: Any,
    header_phi: Mapping[str, str],
    *,
    uid_entropy: str,
) -> Path:
    pydicom = _import_pydicom()
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CTImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid(
        entropy_srcs=[MULTIMODAL_DICOM, uid_entropy, "sop-instance"]
    )
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid(
        entropy_srcs=[MULTIMODAL_DICOM, "implementation-class"]
    )

    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = str(file_meta.MediaStorageSOPInstanceUID)
    dataset.StudyInstanceUID = generate_uid(
        entropy_srcs=[MULTIMODAL_DICOM, uid_entropy, "study-instance"]
    )
    dataset.SeriesInstanceUID = generate_uid(
        entropy_srcs=[MULTIMODAL_DICOM, uid_entropy, "series-instance"]
    )
    dataset.PatientName = header_phi["PatientName"]
    dataset.PatientID = header_phi["PatientID"]
    dataset.PatientBirthDate = header_phi["PatientBirthDate"]
    dataset.StudyDate = header_phi["StudyDate"]
    dataset.Modality = "OT"
    dataset.BurnedInAnnotation = "YES"
    dataset.Rows = int(pixels.shape[-2])
    dataset.Columns = int(pixels.shape[-1])
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    if pixels.ndim == 3:
        dataset.NumberOfFrames = int(pixels.shape[0])
    dataset.PixelData = np.ascontiguousarray(pixels).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return path


def _header_value_survives(dataset: Any, keyword: str, original: str) -> bool:
    """Return True if the *original* header value still appears after de-id."""
    value = getattr(dataset, keyword, None)
    if value is None:
        return False
    text = str(value)
    if not text:
        return False
    # Direct match of the original value (or its digits for numeric ids/dates).
    if original and original in text:
        return True
    original_digits = re.sub(r"\D", "", original)
    if original_digits and original_digits in re.sub(r"\D", "", text):
        return True
    return False


def _word_is_fully_redacted(
    word: BurnedInPhiWord,
    pixel_array: Any,
    *,
    frame_count: int,
) -> bool:
    """Return whether the final pixels contain no residual ink in a gold bbox."""
    np = _import_numpy()
    pixels = np.asarray(pixel_array)
    if word.frame_index < 0 or word.frame_index >= frame_count:
        return False
    frame = pixels[word.frame_index] if frame_count > 1 else pixels
    x0, y0, x1, y1 = word.bbox
    region = frame[y0:y1, x0:x1]
    return bool(region.size) and not bool(np.any(region))


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an artifact without exposing its content."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_pii_model_name() -> str:
    """Return the registered English PII checkpoint for pixel detection."""
    from openmed.core.model_registry import get_default_pii_model

    model_name = get_default_pii_model("en")
    if not model_name:  # pragma: no cover - registry invariant
        raise RuntimeError("no default English PII model is registered")
    return model_name


def _format_display_date(compact: str) -> str:
    """Turn a compact ``YYYYMMDD`` date into ``YYYY-MM-DD`` for display."""
    if len(compact) == 8 and compact.isdigit():
        return f"{compact[0:4]}-{compact[4:6]}-{compact[6:8]}"
    return compact


def _import_pydicom() -> Any:
    try:
        return importlib.import_module("pydicom")
    except ImportError as exc:  # pragma: no cover - exercised via extra-less env
        raise MissingDependencyError(
            dependency="pydicom",
            instruction='Install with: pip install "openmed[multimodal]".',
        ) from exc


def _import_numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:  # pragma: no cover - exercised via extra-less env
        raise MissingDependencyError(
            dependency="numpy",
            instruction='Install with: pip install "openmed[multimodal]".',
        ) from exc


def _import_pillow() -> tuple[Any, Any, Any]:
    try:
        return (
            importlib.import_module("PIL.Image"),
            importlib.import_module("PIL.ImageDraw"),
            importlib.import_module("PIL.ImageFont"),
        )
    except ImportError as exc:  # pragma: no cover - exercised via extra-less env
        raise MissingDependencyError(
            dependency="Pillow",
            instruction='Install with: pip install "openmed[multimodal]".',
        ) from exc


@contextlib.contextmanager
def _offline_pixel_model() -> Iterator[None]:
    """Short-circuit the DICOM pixel PHI model for local-first scoring.

    The multimodal DICOM redaction path calls a token-classification model on
    the OCR text before consulting the header-seeded deny recognizer. In this
    benchmark every burned-in token is also a header value, so the deny
    recognizer alone recovers it. Replacing the model call with an
    entity-free stub keeps the benchmark fully offline and deterministic while
    still exercising the real OCR -> detect -> black-out -> re-OCR pipeline.
    """
    from openmed.processing.outputs import PredictionResult

    dicom_mod = importlib.import_module("openmed.multimodal.dicom")

    def _model_free_extract(text: str, **_kwargs: Any) -> PredictionResult:
        return PredictionResult(
            text=text,
            entities=[],
            model_name="offline-header-seeded",
            timestamp="1970-01-01T00:00:00+00:00",
        )

    original = dicom_mod._extract_dicom_pixel_phi
    dicom_mod._extract_dicom_pixel_phi = _model_free_extract
    try:
        yield
    finally:
        dicom_mod._extract_dicom_pixel_phi = original


__all__ = [
    "MULTIMODAL_DICOM",
    "CORPUS_PROVENANCE",
    "CORPUS_LICENSE",
    "DATA_USE_AGREEMENT_REQUIRED",
    "TCIA_SAMPLED",
    "DEFAULT_SEED",
    "DEFAULT_CORPUS_SIZE",
    "DEFAULT_SYSTEM_NAME",
    "BurnedInPhiWord",
    "SyntheticDicomCase",
    "GoldBoxOcrEngine",
    "multimodal_dicom_metadata",
    "generate_synthetic_dicom_corpus",
    "run_multimodal_dicom",
    "load_multimodal_dicom_fixtures",
]
