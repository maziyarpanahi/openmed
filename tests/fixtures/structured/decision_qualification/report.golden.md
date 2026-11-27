# Decision qualification report

## Status

| Field | Value |
| --- | --- |
| Status | qualified |
| Qualified | true |
| Human review required | true |
| Reasons | none |
| Backend | `synthetic.encoder.v1` (encoder) |
| Report schema | 1.0.0 |
| Report digest | `sha256:231ddef71a20c18ee6038bebe2da1d27a54261073414c6b612a4f5c585b8d56e` |

## Inputs

| Split | Records | Digest |
| --- | ---: | --- |
| development | 26 | `sha256:120b8639b682f251ebbffd4b8bcda4ee5da475eb91c3244d0d0cd2041de7412f` |
| evaluation | 20 | `sha256:1047f90c822c5158898f00a2a871e63895133cd567d8f75e854c576aedd7980a` |

## Policy

| Setting | Value |
| --- | ---: |
| accuracy_floor | 0.9 |
| coverage_floor | 0.5 |
| min_examples_per_option | 2 |

## Per-mode results

| Mode | Split | Total | Covered | Coverage | Accuracy | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| boolean_choice | development | 8 | 6 | 0.7500 | 1.0000 | 0.1400 |
| boolean_choice | evaluation | 6 | 4 | 0.6667 | 1.0000 | 0.1250 |
| fixed_choice | development | 10 | 6 | 0.6000 | 1.0000 | 0.1400 |
| fixed_choice | evaluation | 8 | 6 | 0.7500 | 1.0000 | 0.1400 |
| multi_label | development | 8 | 7 | 0.8750 | 1.0000 | 0.1829 |
| multi_label | evaluation | 6 | 5 | 0.8333 | 1.0000 | 0.1880 |

## Frozen thresholds

| Mode | Threshold | Value |
| --- | --- | ---: |
| boolean_choice | minimum_confidence | 0.7400 |
| boolean_choice | minimum_margin | 0.4800 |
| fixed_choice | minimum_confidence | 0.7400 |
| fixed_choice | minimum_margin | 0.4800 |
| multi_label | multi_label_threshold | 0.5600 |
