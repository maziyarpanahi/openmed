# ONNX test fixtures

The ONNX Runtime wrapper tests use a small invocation seam instead of committing
a binary model. The seam still receives real `OnnxTensor` inputs, so the tests
verify tensor names, shapes, element types, special-token offset handling,
softmax/argmax decoding, missing-output errors, cancellation, and disposal.

Keep future binary fixtures synthetic and free of PHI.
