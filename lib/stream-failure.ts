/** Diagnose only the known adapter guard; do not infer a transport root cause. */
export function isMissingFinishReason(error: unknown): boolean {
  return typeof error === "string"
    && /^(?:Error:\s*)?Stream ended without finish_reason\s*$/i.test(error.trim());
}

interface FailedStreamMessage {
  stopReason?: string;
  errorMessage?: string;
  content?: readonly { type: string }[];
}

/**
 * Derived from the existing transcript, not a second copy of provider data.
 * Deliberately exclude text, tool inputs, URLs, headers and arbitrary error text.
 * A missing marker says nothing about why the upstream stream ended.
 */
export function getStreamFailureDiagnostic(message: FailedStreamMessage) {
  if (message.stopReason !== "error" || !isMissingFinishReason(message.errorMessage)) return null;
  return {
    errorClass: "missing_finish_reason" as const,
    terminationKind: "finish_marker_not_observed" as const,
    receivedBlocks: message.content?.length ?? 0,
    pendingToolCalls: message.content?.filter((block) => block.type === "toolCall").length ?? 0,
  };
}
