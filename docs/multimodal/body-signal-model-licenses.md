# Body-signal Model License Provenance

This page records what upstream sources say about the licenses of candidate
models for pose, audio, sleep and chest X-ray pipelines. Read it before an
adapter, example or fixture names one of these models. It is a provenance
record taken from upstream LICENSE files and model cards, not legal advice and
not an approval to use a model for any purpose.

## Repository rule

`AGENTS.md` requires core dependencies to stay permissive-license compatible.
GPL, source-available, proprietary and DUA-gated assets, including MIMIC, must
not be bundled. Restricted integrations stay behind optional user-supplied
assets or out-of-process bridges. For the models on this page that means:

- OpenMed does not bundle, download by default or pin any of these weights.
- A model whose weight license is not stated, or whose weights were trained on
  DUA-gated data, is **user-supplied only**: the user obtains the weights,
  verifies that the intended use is permitted, accepts the applicable terms and
  passes a local path. Supplying a file does not itself grant usage rights.
- A model under a copyleft or non-commercial license is not used by OpenMed
  code, examples or fixtures.
- Check the scope of the code, weight and training-data terms separately. A
  repository license may cover distributed model files, but a code-license
  badge alone does not establish the terms of every checkpoint or dataset.

The statuses below are conservative OpenMed integration decisions. Training on
restricted data is a reason to review the applicable agreement; it does not by
itself prove that the same license governs the resulting weights. "Not stated"
means no separate weight-license statement was found in the cited sources, not
that the weights are public domain or that the repository license cannot apply.

Model-license propagation into exported artifacts and app bundles is covered
in [Export Toolchains, Artifact Licenses and
Redistribution](../security/export-redistribution.md).

## Candidate models

Sources in this table were checked on 2026-11-19.

| Model | Code license | Weight license | Training-data notes | Intended use and out-of-scope notes | Status |
| --- | --- | --- | --- | --- | --- |
| torchxrayvision (chest X-ray) | `Apache-2.0` ([LICENSE](https://github.com/mlmed/torchxrayvision/blob/master/LICENSE)) | Not stated. The [README](https://github.com/mlmed/torchxrayvision/blob/master/README.md) lists pretrained weights but states no weight license. | Weights such as `densenet121-res224-mimic_nb`, `-mimic_ch` and `-chex`, and the `all` models, are trained on MIMIC-CXR or CheXpert. MIMIC-CXR is credentialed access under the [PhysioNet Credentialed Health Data License 1.5.0](https://physionet.org/content/mimic-cxr/2.1.0/). For CheXpert, consult the [Stanford dataset access route](https://aimi.stanford.edu/datasets/chexpert-chest-x-rays) and [AIMI licensing terms](https://aimi.stanford.edu/datasets-commercial-use): non-commercial access and commercial licensing are separate. | Research baselines and feature extractors. The README states no clinical intended use. | User-supplied only |
| YAMNet (audio events) | `Apache-2.0` ([TensorFlow Model Garden LICENSE](https://github.com/tensorflow/models/blob/master/LICENSE), which covers `research/`) | `Apache-2.0`, stated by a maintainer on the [audioset-users list](https://groups.google.com/g/audioset-users/c/Ly4guQnlv_o) by reference to the Model Garden license (2020-04-06). This is the weight-license evidence used here. | Trained on AudioSet ([README](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/README.md)). The AudioSet dataset is `CC-BY-4.0` and its ontology, the source of the class names in `yamnet_class_map.csv`, is `CC-BY-SA-4.0` ([download page](https://research.google.com/audioset/download.html), [ontology README](https://github.com/audioset/ontology/blob/master/README.md)). | General audio-event classification over 521 AudioSet classes. The [upstream model documentation](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/yamnet/1.md) warns that scores are uncalibrated and task-specific domain mismatch is possible; it provides no clinical-use validation. | User-supplied. Review ontology-derived class-map redistribution separately; preserve attribution and applicable CC BY-SA terms. |
| MediaPipe BlazePose GHUM 3D (pose) | `Apache-2.0` ([MediaPipe LICENSE](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)) | `Apache-2.0`, stated in the [model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) (April 2021). | The card reports 30K consented mobile-AR images and says most training images depict fitness poses. The z coordinate is up to scale, not metric. | Card lists augmented reality, pose and gesture recognition, fitness and repetition counting, and pose measurements. Out of scope: multiple people, people further than about 4 m, head not visible, metric-accurate depth, and surveillance or identity recognition. "Not intended for human life-critical decisions." | User-supplied. The listed pose-measurement uses do not establish validation for clinical gait assessment or fall detection. |
| YASA (sleep staging) | `BSD-3-Clause` ([LICENSE](https://github.com/raphaelvallat/yasa/blob/master/LICENSE)) | Not separately stated in the cited sources. The pretrained classifiers ship inside the BSD-3-Clause package; the repository license is not a separate checkpoint-specific statement. | Trained and validated on about 3,000 nights from the National Sleep Research Resource: CCSHS, CFS, CHAT, HomePAP, MESA, MrOS and SHHS ([`staging.py`](https://github.com/raphaelvallat/yasa/blob/master/src/yasa/staging.py), [yasa_classifier README](https://github.com/raphaelvallat/yasa_classifier/blob/master/README.md)). NSRR data needs an approved [Data Access and Use Agreement](https://sleepdata.org/about/data-security). | Full-night sleep staging. The docstring says predictions should always be checked by a trained visual scorer, and the classifiers were trained only on full-night recordings. It requires full inspection for naps, sleep disorders and suboptimal recordings. | User-supplied only |

## Other pose models screened

These models are common in pose and gait work. The table records why most of
them are not candidates. Sources in this table were checked on 2026-11-19.

| Model | Code license | Weight license | Notes | Status |
| --- | --- | --- | --- | --- |
| MoveNet SinglePose (Lightning, Thunder) | Distributed as model files; the card covers them. | `Apache-2.0`, stated in the [model card](https://storage.googleapis.com/movenet/MoveNet.SinglePose%20Model%20Card.pdf). | Trained on COCO Keypoint 2017 and an internal set of YouTube fitness images. Intended for fitness and AR entertainment; multiple people, surveillance and identity recognition are out of scope. | User-supplied |
| RTMPose (MMPose) | `Apache-2.0` ([LICENSE](https://github.com/open-mmlab/mmpose/blob/main/LICENSE)) | Not stated. The [RTMPose README](https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md) links weights without a separate license. | The README's COCO body weights are trained on AI Challenger and COCO (`aic-coco` in the file names). | User-supplied only |
| ViTPose | `Apache-2.0` ([LICENSE](https://github.com/ViTAE-Transformer/ViTPose/blob/main/LICENSE)) | Not stated. | The [README](https://github.com/ViTAE-Transformer/ViTPose/blob/main/README.md) initialises the backbones from MAE pretrained models, and the [MAE LICENSE](https://github.com/facebookresearch/mae/blob/main/LICENSE) is `CC-BY-NC-4.0`. Whether the non-commercial term carries into the fine-tuned weights is unresolved. | Not used |
| Ultralytics YOLO pose models | `AGPL-3.0` ([LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)) | `AGPL-3.0`, or a separately negotiated Enterprise License ([README](https://github.com/ultralytics/ultralytics/blob/main/README.md)). | Copyleft, so it cannot be bundled or imported in-process. | Not used |
| OpenPose | Academic or non-profit noncommercial research license ([LICENSE](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)) | Not separately stated in the cited license; do not infer a separate grant for downloaded weights. | Non-commercial, no redistribution, no sublicensing. | Not used |

## Keeping this page current

Upstream licenses and model cards change. Recheck every source and update its
check date before an adapter, example or fixture first names a model, and when
a model version changes. If a row's weight license becomes explicit, change its
status only after the training-data terms have been checked too.
