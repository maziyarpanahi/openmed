# Offline multimodal preflight CLI

Run bounded admission before selecting a decoder or model:

```bash
openmed multimodal preflight ./synthetic-input.png
openmed multimodal preflight ./synthetic-packet --directory --profile mobile
openmed multimodal preflight ./synthetic-input.wav --format text
```

The installed CLI and optional `python -m openmed.cli.typer_app` use the same
guarded parser and implementation. The command does not load ambient configuration,
download weights, decode pixels or audio, run OCR, transcribe, infer, scan for
malware, or modify inputs. Header admission does not establish clinical validity,
benign content, complete encoding, successful redaction, or safe decoder memory use.

## Output and exit codes

JSON is the default. It is deterministic for unchanged inputs, options and directory
contents, with no timestamps. Each encountered file receives an ordinal opaque ID
such as `asset-000001`. Names, paths, raw bytes, OCR, transcripts, DICOM values,
credentials and underlying exception messages are excluded, including usage errors.
There is no option to expose filenames. IDs identify positions in this invocation;
they are not stable across changes to a directory. SHA-256 digests are content
fingerprints, not anonymization or a guarantee against identifying known files.

| Exit | Meaning |
| --- | --- |
| `0` | All inspected assets and shared aggregate checks accept; also successful help |
| `1` | Hold: an asset, collection limit, filesystem check or aggregate check failed |
| `2` | Invalid invocation or a command-level failure |

Concise text output looks like:

```text
Multimodal preflight: accept
asset-000001: accept
```

JSON `assets` retains filesystem/header failures alongside actual shared preflight
reports. `batch_asset_ids` maps shared batch positions to the corresponding opaque
IDs: files rejected before a manifest can be built are not given invented manifests.
`batch` is `null` if no manifest was admitted. A batch subreport may accept while the
overall command holds because another file failed. Use the top-level `outcome` or
exit code. Duplicate content holds the batch even when both individual files accept.

## Header coverage

| Input | Bounded evidence | Admission limitation |
| --- | --- | --- |
| PNG / JPEG | Declared width and height | No image decoding or complete encoding validation |
| Classic TIFF | First-IFD geometry | Later IFDs and EXIF are not inspected; not a multipage TIFF assessment |
| PCM / IEEE-float WAV | Declared duration, channels, sample rate | No sample decoding or complete audio validation |
| PDF | Shared page-tree geometry report | Current shared limits cannot establish raster pixel geometry, so even a readable PDF holds |
| DICOM Part 10 prefix | `DICM` signature | Prefix alone cannot establish dimensions, frames or transfer syntax; holds |

Unknown media, unsupported header formats, malformed or out-of-budget headers hold.
Extensions are ignored. The actual file digest is computed with a bounded read and
verified again by the shared preflight contract. A digest created from the same
input is an identity check, not verification against a trusted external checksum.

## Directory and read bounds

Directories require `--directory`. Traversal is depth-first, sorted by entry name
within each directory, and continues after individual file failures. Enumeration
has a separate budget including directories, links and unsupported entries. A
directory exceeding the remaining entry budget is refused without selecting an
arbitrary partial subset. Limits always hold the overall result.

| Option | Default | Hard maximum |
| --- | --- | --- |
| `--max-files` | 128 result positions | 256 |
| `--max-entries` | 4,096 entries | 10,000, plus one overflow probe |
| `--max-depth` | 16 | 32; zero inspects only the selected directory's immediate files |
| `--max-file-bytes` | Selected profile ceiling | Can only tighten that ceiling |
| `--max-total-bytes` | 256 MiB | 256 MiB |
| `--max-total-pages` | 100 | 10,000 |
| `--max-total-pixels` | 100,000,000 | 100,000,000 |
| `--max-total-frames` | 1,024 | 10,000 |
| `--max-total-duration` | 1,800 seconds | 86,400 seconds |

`--profile desktop` and `--profile mobile` select the existing versioned
[asset limit profiles](asset-limits.md). These are admission policies, not measured
device capacities. Aggregate ceilings apply independently of per-file ceilings.
Unknown applicable geometry/duration remains unknown, never zero.

Declared file bytes are reserved before any prefix, header or hash read, including
malformed inputs. Individual sources cannot read beyond their original size plus
one growth probe. Total reads across prefix/header/hash passes are capped at five
times each reserved file size plus five bytes. PNG/JPEG/TIFF/WAV header work is
limited to 64 KiB. PDF metadata parsing is limited to 8 MiB input, 10,000 objects,
the selected profile's page ceiling and 8 MiB of decompressed object streams.
At most one PDF metadata buffer is parsed at a time; original media is not retained
in the result. This is bounded metadata parsing, not a PDF renderer.

Symlinks are refused in the input path and traversal. POSIX opens each component
relative to held directory descriptors with `O_NOFOLLOW`; special files are opened
nonblocking and rejected before reads. Windows uses held ancestor handles, rejects
reparse points and denies write/delete sharing while files are read. Windows device
paths, UNC paths and alternate streams are refused. Unsupported secure-open
platforms fail closed. Use a physical path without `..` or symlink components.
Checks detect observed file/directory changes; this is not a transactional filesystem
snapshot or protection against a privileged adversary modifying the filesystem.

Validate with synthetic inputs:

```bash
python -m pytest tests/unit/cli/test_multimodal_preflight.py tests/unit/multimodal/test_batch_preflight.py -q
```
