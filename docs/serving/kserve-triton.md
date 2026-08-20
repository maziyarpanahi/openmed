# KServe and Triton model repositories

OpenMed can package an exported ONNX token classifier as a Triton model
repository and use a user-operated Triton or KServe endpoint as its inference
backend. Tokenization, source offsets, and entity decoding remain in the
OpenMed process. The server receives numeric tensors and returns `logits`.

OpenMed does not install or start Triton or KServe. Operate the serving system
separately and apply the authentication, network isolation, and transport
controls required for your deployment.

## Install client and export dependencies

Repository generation needs ONNX support. Remote HTTP and gRPC inference use
client libraries in the `triton` extra:

```bash
pip install 'openmed[onnx,triton]'
```

The extra contains no KServe or Triton server distribution. HTTP follows the
KServe V2 JSON protocol. gRPC uses a small, wire-compatible subset of the
KServe V2 protobuf contract and the standard
`inference.GRPCInferenceService/ModelInfer` method.

## 1. Export and build the repository

Start with an OpenMed ONNX token-classification artifact produced by
`openmed.onnx.convert`. For `max_batch_size > 0`, every graph input and output
must have a dynamic leading batch dimension. The graph must expose
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

The adapter derives tensor names, data types, and non-batch dimensions from
the ONNX graph and renders a deterministic `config.pbtxt` using Triton's
`onnxruntime` backend:

```text
build/model-repository/
└── openmed_pii/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

For a graph with external tensor data, the adapter uses Triton's documented
multi-file ONNX form and copies every referenced sidecar:

```text
build/model-repository/openmed_pii/1/
└── model.onnx/
    ├── model.onnx
    └── weights.bin
```

Absolute, parent-relative, and symlink-escaping external-data paths are
rejected. Existing numeric model versions are never overwritten, and a new
version must have the same derived schema as the existing config.

Recheck a copied repository before mounting it:

```python
from openmed.service.backends import validate_triton_model_repository

validate_triton_model_repository(
    "build/model-repository",
    model_name="openmed_pii",
    version=1,
)
```

## 2. Serve it in your infrastructure

Mount `build/model-repository` into your own Triton deployment and point
`--model-repository` at that mount. For KServe, configure its Triton runtime or
storage initializer to expose the same repository. Verify the V2 model-ready
endpoint before routing OpenMed traffic:

```bash
curl --fail https://triton.example/v2/models/openmed_pii/ready
```

Server installation, GPU tuning, autoscaling, credentials, and lifecycle
management remain outside OpenMed.

## 3. Select remote inference through configuration

Keep the original artifact or model repository available to the OpenMed
client. Its fast tokenizer and `id2label` metadata are used locally; only the
ONNX graph executes on the remote server.

```python
from openmed.core import ModelLoader, OpenMedConfig

config = OpenMedConfig(
    backend="remote",
    remote_inference_endpoint="https://triton.example",
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

No pipeline call-site change is required. `OpenMedConfig.backend` selects the
remote implementation through the normal `ModelLoader.create_pipeline` path.
The pipeline accepts one string or a batch and returns the same entity
dictionary shape and source spans as local ONNX inference.

The settings can also be stored in the normal flat TOML configuration:

```toml
backend = "remote"
remote_inference_endpoint = "https://kserve.example"
remote_inference_protocol = "http"
remote_inference_model_name = "openmed_pii"
remote_inference_model_version = "1"
remote_inference_tokenizer = "/models/openmed-pii-onnx"
remote_inference_timeout_seconds = 30.0
remote_inference_verify_tls = true
```

For Triton gRPC, use a gRPC target and protocol:

```python
config = OpenMedConfig(
    backend="remote",
    remote_inference_endpoint="grpcs://triton.example:8001",
    remote_inference_protocol="grpc",
    remote_inference_model_name="openmed_pii",
    remote_inference_tokenizer="/models/openmed-pii-onnx",
)
```

`grpc://` and bare `host:port` targets are plaintext. `grpcs://` uses TLS and
does not permit certificate verification to be disabled. HTTP deployments
should likewise keep `remote_inference_verify_tls=true`.

## Privacy and offline boundary

The adapter does not log source text, request tensors, response bodies, or raw
server errors. It sends token IDs, attention masks, and tokenizer-required
token-type IDs rather than the original string. These tensors can still encode
sensitive content, so treat the endpoint as inside the trusted PHI boundary,
use transport encryption, and enforce authentication and network policy in
your serving infrastructure.

Remote inference is explicit only: it is never auto-selected. It is also
rejected when `local_only=True` or `OPENMED_OFFLINE=1`, because either setting
promises a no-egress runtime.

Protocol references:

- [Triton model repository layout](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_repository.html)
- [Triton model configuration](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html)
- [KServe V2 inference protocol](https://kserve.github.io/website/docs/concepts/architecture/data-plane/v2-protocol)
