<!-- scenario: empty -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |

_No readiness entries are registered._
<!-- scenario: complete -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `en-GB` | `complete` | `complete` | `complete` | `complete` | `complete` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 5 |
| `partial` | 0 |
| `blocked` | 0 |
| `missing_evidence` | 0 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `en-GB` | detection | `sha256:0000000000000000000000000000000000000000000000000000000000000000` |
| `en-GB` | surrogates | `sha256:1111111111111111111111111111111111111111111111111111111111111111` |
| `en-GB` | validation | `sha256:2222222222222222222222222222222222222222222222222222222222222222` |
| `en-GB` | fixtures | `sha256:3333333333333333333333333333333333333333333333333333333333333333` |
| `en-GB` | release_evidence | `sha256:4444444444444444444444444444444444444444444444444444444444444444` |
<!-- scenario: partial -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `de` | `partial` | `partial` | `complete` | `not_reported` | `not_reported` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 1 |
| `partial` | 2 |
| `blocked` | 0 |
| `missing_evidence` | 0 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `de` | validation | `sha256:5555555555555555555555555555555555555555555555555555555555555555` |
<!-- scenario: blocked -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `am` | `not_reported` | `not_reported` | `blocked` | `not_reported` | `blocked` |
| `fa` | `complete` | `not_reported` | `blocked` | `not_reported` | `not_reported` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 1 |
| `partial` | 0 |
| `blocked` | 3 |
| `missing_evidence` | 0 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `fa` | detection | `sha256:6666666666666666666666666666666666666666666666666666666666666666` |
<!-- scenario: aliased -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `en-GB` | `not_reported` | `not_reported` | `blocked` | `not_reported` | `not_reported` |
| `he` | `complete` | `partial` | `not_reported` | `not_reported` | `not_reported` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 1 |
| `partial` | 1 |
| `blocked` | 1 |
| `missing_evidence` | 0 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `he` | detection | `sha256:7777777777777777777777777777777777777777777777777777777777777777` |
<!-- scenario: missing_evidence -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: `2026.10`

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `sw` | `not_reported` | `not_reported` | `not_reported` | `missing_evidence` | `missing_evidence` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 0 |
| `partial` | 0 |
| `blocked` | 0 |
| `missing_evidence` | 2 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |

_No evidence digests are recorded._
<!-- scenario: ordering_and_escaping -->
# Language readiness matrix

Schema: `openmed.language_readiness_matrix.v1`

Version: ``` ``2026.10`-rc|1 ```

## Capability matrix

| Locale | detection | surrogates | validation | fixtures | release_evidence |
| --- | --- | --- | --- | --- | --- |
| `am` | `complete` | `not_reported` | `not_reported` | `not_reported` | `not_reported` |
| `en` | `not_reported` | `not_reported` | `not_reported` | `not_reported` | `complete` |
| `en-GB` | `not_reported` | `not_reported` | `not_reported` | `missing_evidence` | `not_reported` |
| `es-419` | `not_reported` | `not_reported` | `blocked` | `not_reported` | `not_reported` |
| `sr-Latn` | `not_reported` | `partial` | `not_reported` | `not_reported` | `not_reported` |
| `zh-Hant-TW` | `complete` | `not_reported` | `not_reported` | `not_reported` | `not_reported` |

## State totals

| State | Count |
| --- | --- |
| `complete` | 3 |
| `partial` | 1 |
| `blocked` | 1 |
| `missing_evidence` | 1 |

## Evidence digests

| Locale | Capability | Digest |
| --- | --- | --- |
| `am` | detection | `sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa` |
| `en` | release_evidence | `sha256:9999999999999999999999999999999999999999999999999999999999999999` |
| `zh-Hant-TW` | detection | `sha256:8888888888888888888888888888888888888888888888888888888888888888` |
