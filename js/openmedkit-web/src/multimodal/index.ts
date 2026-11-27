/**
 * `openmed/multimodal`: admission and typed projections of the multimodal
 * evidence packets that `openmed.multimodal.apple_contract` (Python) and
 * OpenMedKit (Swift) exchange. Packets carry coordinates, counts and digests,
 * never media bytes or transcript text. Admission adds no media decoding, no
 * network access and no clinical or disclosure decision.
 */
export {
  MULTIMODAL_ABSTENTION_CODES,
  MULTIMODAL_CLOCK_REASONS,
  MULTIMODAL_MAX_WIRE_BYTES,
  MULTIMODAL_PACKET_SCHEMA,
  MULTIMODAL_PACKET_SCHEMAS,
  MULTIMODAL_PROJECTION_KINDS,
  MULTIMODAL_PROVIDER_OUTCOMES,
  MULTIMODAL_REFUSALS,
  MultimodalContractError,
  MultimodalDigestUnavailableError,
  MultimodalPacket,
  admitMultimodalPacket,
} from "./packet";
export type {
  AdmitMultimodalPacketOptions,
  MultimodalAbstentionCode,
  MultimodalAsset,
  MultimodalClockAlignment,
  MultimodalClockReason,
  MultimodalClockState,
  MultimodalDigestProvider,
  MultimodalInterval,
  MultimodalNanosecondBound,
  MultimodalNanosecondInterval,
  MultimodalProjection,
  MultimodalProjectionKind,
  MultimodalProviderOutcome,
  MultimodalProviderResult,
  MultimodalProviderSchema,
  MultimodalRectangle,
  MultimodalRefusal,
  MultimodalRotation,
  MultimodalSpanInvalidation,
  MultimodalSpatialUnit,
  MultimodalTranscript,
  MultimodalVisual,
  MultimodalVisualKind,
} from "./packet";
