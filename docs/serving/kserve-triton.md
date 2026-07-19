# KServe and Triton inference backends

OpenMed can package an exported ONNX token classifier as a Triton model
repository and use a user-operated Triton or KServe endpoint as the inference
backend. Tokenization and entity decoding remain in the OpenMed process; the
remote server receives numeric model tensors and returns `logits`.

OpenMed does not install or start Triton or KServe. Operate the serving system
separately and apply the authentication, network isolation, and transport
controls required for your deployment.

## Install the client and export dependencies

Repository generation needs ONNX support. HTTP and gRPC remote inference use
the `triton` extra:

```bash
pip install 'openmed[onnx,triton]'
```

The HTTP backend follows the KServe V2 JSON protocol. The gRPC backend uses the
same protocol over OpenMed's existing gRPC and protobuf runtime.

## 1. Export and build the model repository

Start with an OpenMed ONNX token-classification artifact produced by
`openmed.onnx.convert`. The artifact must expose dynamic `batch` axes,
`input_ids`, `attention_mask`, and a `logits` output.

```python
from pathlib import Path

from openmed.onnx import convert
from openmed.service.backends import write_triton_model_repository

artifact_dir = Path("build/openmed-pii-onnx")
convert(
    "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    artifact_dir,
)

repository = write_triton_model_repository(
    artifact_dir / "model.onnx",
    "build/model-repository",
    model_name="openmed_pii",
    version=1,
    max_batch_size=8,
)
print(repository.model_dir)
```

The adapter derives input/output names, data types, and non-batch dimensions
from the ONNX graph. It creates the documented Triton layout:

```text
build/model-repository/
└── openmed_pii/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

External ONNX tensor data is copied with the graph. Absolute or parent-relative
sidecar paths are rejected, and an existing numeric model version is never
overwritten.

`config.pbtxt` uses the `onnxruntime` backend, declares `max_batch_size`, and
lists every model input and output using Triton's `TYPE_*` data types. You can
recheck a copied repository before mounting it:

```python
from openmed.service.backends import validate_triton_model_repository

validate_triton_model_repository(
    "build/model-repository",
    model_name="openmed_pii",
    version=1,
)
```

## 2. Serve the repository

Mount `build/model-repository` into your own Triton deployment and point
`--model-repository` at that mount. For KServe, configure its Triton runtime or
storage initializer to expose the same repository. Verify the V2 model-ready
endpoint before routing OpenMed traffic:

```bash
curl --fail http://triton.example:8000/v2/models/openmed_pii/ready
```

## 3. Select the remote backend through configuration

Keep the original ONNX artifact directory available to the OpenMed client. Its
tokenizer and `id2label` metadata are used locally; the ONNX graph itself runs
only on the remote server.

```python
from openmed.core import ModelLoader, OpenMedConfig

config = OpenMedConfig(
    backend="remote",
    remote_inference_endpoint="http://triton.example:8000",
    remote_inference_protocol="http",
    remote_inference_model_name="openmed_pii",
    remote_inference_model_version="1",
    remote_inference_tokenizer="build/openmed-pii-onnx",
    remote_inference_timeout_seconds=30,
    remote_inference_verify_tls=True,
)

loader = ModelLoader(config)
pipeline = loader.create_pipeline(
    "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
    task="token-classification",
    aggregation_strategy="simple",
)
entities = pipeline("Synthetic patient Alice Nguyen")
```

No pipeline call-site change is required: `OpenMedConfig.backend` selects the
remote implementation through the normal `ModelLoader.create_pipeline` path.
The remote pipeline supports one string or a batch of strings and returns the
same token-classification entity dictionary shape as the local pipeline.

The same settings can be stored in the normal OpenMed TOML configuration:

```toml
backend = "remote"
remote_inference_endpoint = "https://kserve.example"
remote_inference_protocol = "http"
remote_inference_model_name = "openmed_pii"
remote_inference_model_version = "1"
remote_inference_tokenizer = "/models/openmed-pii-onnx"
remote_inference_timeout_seconds = 30
remote_inference_verify_tls = true
```

For Triton gRPC, change the endpoint and protocol:

```python
config = OpenMedConfig(
    backend="remote",
    remote_inference_endpoint="grpcs://triton.example:8001",
    remote_inference_protocol="grpc",
    remote_inference_model_name="openmed_pii",
    remote_inference_tokenizer="/models/openmed-pii-onnx",
)
```

## Privacy and deployment boundary

The adapter never logs source text or tensor contents. The client sends token
IDs, attention masks, and any tokenizer-required token-type IDs, not the raw
text string. Those tensors can still encode sensitive content, so use only a
trusted endpoint, keep TLS verification enabled, and enforce authorization and
network policy outside OpenMed. `local_only=True` prevents tokenizer downloads;
it does not disable the explicitly selected remote inference connection.

Protocol references:

- [Triton model repository layout](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [Triton model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [KServe V2 inference protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
