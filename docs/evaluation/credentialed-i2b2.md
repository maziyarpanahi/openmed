# Run the credentialed i2b2 de-identification benchmark

OpenMed uses the i2b2/DBMI corpora for evaluation under approved local access.
The `i2b2` suite supports the 2006 Track 1B and 2014 i2b2/UTHealth
de-identification releases. OpenMed does not bundle or download DUA data.

## Before you run

- Obtain approved access under the applicable data-use agreement (DUA) for the
  release you intend to evaluate. Use an environment and storage location
  authorized for that corpus.
- Keep the XML corpus and a private output directory outside the repository
  checkout and any synced or shared directory. Use a separate corpus directory
  for each release.
- Install the OpenMed CLI and stage an approved local PII model, including its
  runtime dependencies, before granting the environment access to the corpus.
  Verify that the model loads on synthetic input in
  [local-only mode](../configuration.md#local-only-offline-mode).

## Run the suite locally

| Environment variable | Value |
| --- | --- |
| `OPENMED_I2B2_PATH` | Absolute path to the approved local directory that contains the release's XML files. |
| `OPENMED_I2B2_YEAR` | `2006` or `2014`. The loader defaults to `2014` when this variable is unset. |
| `OPENMED_OFFLINE` | Set to `1` for cache-only model loading and offline inference. |

In a POSIX shell, replace all placeholder paths with paths in your approved
environment. Use an existing private output directory and fresh filenames for
each run. The subshell keeps the environment variables scoped to the command.

```bash
(
  umask 077
  export OPENMED_I2B2_PATH="/path/to/approved/i2b2/2014-xml"
  export OPENMED_I2B2_YEAR=2014
  export OPENMED_OFFLINE=1

  openmed benchmark pii \
    --suite i2b2 \
    --models "/path/to/approved/local-pii-model" \
    --output "/path/to/private/benchmark-output/i2b2-2014.json" \
    >"/path/to/private/benchmark-output/i2b2-2014.log" 2>&1
)
```

For the 2006 release, change the year to `2006`, use the approved 2006 XML
directory, and choose new report and log filenames. The loader refuses a
missing, empty, or repository-local corpus directory. If the private log shows
a loader error, check your local path, XML layout, and year before retrying.

## Keep results private

Keep the source corpus, report, and captured output in approved storage outside
the checkout. Treat the full JSON and log as restricted. The report includes
per-document fixture IDs in its metadata, and failures can contain local paths
or annotation values. Never commit the dataset, per-document outputs, full
reports, or logs. Keep them out of pull requests, issue attachments, and CI
artifacts.

If the DUA permits sharing results, copy only reviewed aggregate counts and
scores, such as the overall span F1. Exclude fixture IDs, per-document metrics,
spans, snippets, and local paths. Follow the DUA's retention and disposal
requirements for local artifacts.

The [evaluation harness guide](../eval-harness.md) describes the benchmark
metrics and synthetic alternatives for contributors without approved access.
