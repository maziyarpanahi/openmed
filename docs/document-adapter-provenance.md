# Document Adapter Provenance

Document adapters package document-family handling for PDFs, images, OCR output,
and structured clinical documents while preserving reproducibility, licensing
boundaries, and offline operation. Each adapter release should make it clear what
was packaged, how it was validated, and what happens when the preferred adapter
or artifact is unavailable.

This guide is child work for [#1285](https://github.com/maziyarpanahi/openmed/issues/1285)
and does not close the parent epic.

## Adapter Metadata

Every packaged adapter should declare:

- Adapter name and version.
- Supported document families, MIME types, and file extensions.
- Source repository, release tag, or source commit.
- Build environment, build timestamp, and package digest.
- Runtime requirements, optional extras, and offline artifact locations.
- Validation fixture set, validation date, and validation result summary.
- License identifiers for adapter code and any packaged artifacts.

## Provenance Records

Keep provenance records with the packaged adapter rather than only in release
notes. At minimum, record the source revision, build command, build environment,
input artifacts, package hash, and validation results. Validation summaries
should reference synthetic or permissively licensed fixtures only; do not include
raw PHI or restricted datasets in release artifacts.

When a packaged adapter depends on model weights, record the exact artifact name,
version, digest, and license. If the weights are not bundled, document the
expected local path or user-supplied download step without adding a mandatory
network call to inference.

## Fallback Behavior

Adapters must fail predictably when the preferred engine, model, or optional
dependency is unavailable. Document the fallback order and user-visible behavior:

- Use a packaged local artifact when it is available and license-compatible.
- Fall back to a deterministic local parser or metadata-only extraction when the
  model artifact is absent.
- Surface an actionable error when no safe local fallback exists.
- Preserve source offsets and provenance fields when falling back.
- Never send documents to a remote service as an implicit fallback.

Fallback behavior should be covered by release validation so an offline package
does not appear healthy only because a developer machine had extra dependencies
or cached models.

## Clinical Disclaimer

Document adapters assist document processing, extraction, and de-identification
workflows. They are not diagnostic systems and must not be presented as a
replacement for clinical judgment, treatment decisions, or medical-device
approval. Any adapter documentation that mentions clinical use should preserve
this boundary.

## Licensing And Offline Packaging

Only adapter code and artifacts with compatible permissive licenses may be
bundled by default. Model weights are not bundled unless they are explicitly
packaged as permissive local artifacts with recorded provenance and hashes.

If a model has restricted, source-available, proprietary, GPL-incompatible, DUA,
UMLS, SNOMED CT, CPT, MIMIC, i2b2, or n2c2 constraints, do not bundle it in an
OpenMed offline package. Document the requirement and require users to provide
the artifact out of band.

Offline packages should avoid external network dependencies during inference,
preserve provenance metadata, validate the packaged fallback path, and include
only synthetic or permissively licensed validation data.

## Release Checklist

Before publishing a document adapter:

- Verify required metadata fields and provenance records are complete.
- Confirm packaged code and artifacts have compatible licenses.
- Validate preferred and fallback behavior in an offline environment.
- Confirm model weights are either permissive packaged artifacts or documented
  as user-supplied external inputs.
- Confirm fixtures are synthetic or permissively licensed.
- Update release notes and related documentation.
