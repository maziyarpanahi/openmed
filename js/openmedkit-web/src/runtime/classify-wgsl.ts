/** Batched float32 token-classification head used by the direct WebGPU path. */
export const CLASSIFY_WGSL_SOURCE = String.raw`
struct Shape {
  batch_size: u32,
  sequence_length: u32,
  hidden_size: u32,
  label_count: u32,
}

@group(0) @binding(0) var<storage, read> hidden_states: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<uniform> shape: Shape;

@compute @workgroup_size(8, 8, 1)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
  let token = id.x;
  let label = id.y;
  let batch = id.z;
  if (
    token >= shape.sequence_length ||
    label >= shape.label_count ||
    batch >= shape.batch_size
  ) {
    return;
  }

  let row = batch * shape.sequence_length + token;
  var value = bias[label];
  for (var hidden = 0u; hidden < shape.hidden_size; hidden += 1u) {
    let hidden_index = row * shape.hidden_size + hidden;
    let weight_index = hidden * shape.label_count + label;
    value += hidden_states[hidden_index] * weights[weight_index];
  }
  logits[row * shape.label_count + label] = value;
}
`;
