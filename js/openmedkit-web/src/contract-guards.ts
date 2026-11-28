// Compile-time checks that the hand-written public types match the generated
// contract types. Nothing imports this module, so packages that compile the
// web sources with `noUnusedLocals` (the browser extension) never see these
// unused aliases, while `npm run typecheck` here still checks them.
import type { OpenMedContractSpan } from "./generated/contract-types";
import type { OpenMedSpan } from "./types";

/** `true` when `A` and `B` are assignable to each other. */
type MutuallyAssignable<A, B> = [A] extends [B]
  ? [B] extends [A]
    ? true
    : false
  : false;
type AssertTrue<T extends true> = T;

/**
 * Fails `npm run typecheck` when the hand-written `OpenMedSpan` and the
 * generated `span.schema.json` type diverge.
 */
export type OpenMedSpanMatchesContract = AssertTrue<
  MutuallyAssignable<OpenMedSpan, OpenMedContractSpan>
>;
