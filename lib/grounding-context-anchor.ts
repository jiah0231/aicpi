import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { groundingDecisionCheckpoint } from "./grounding-decision-checkpoint";
import type { GroundingWorkingState } from "./grounding-evidence";

export const GROUNDING_CONTEXT_ANCHOR = "grounding:ephemeral-request-anchor:v1";

type AnchorRecord = {
  key: string;
  originalQuery: string;
  state?: GroundingWorkingState;
  awaitingClarification?: string;
  decisionCheckpoint?: ReturnType<typeof groundingDecisionCheckpoint>;
};

type UnloadedRuntime = {
  queryPath: string;
  outputDir: string;
  pendingKey: string | null;
  lastApprovedKey: string | null;
  pausedForClarification: boolean;
  restorationWarning?: string;
};

export function isGroundingContextAnchor(message: AgentMessage): boolean {
  return message.role === "custom" && message.customType === GROUNDING_CONTEXT_ANCHOR
    && (message.details as { ephemeralGroundingAnchor?: unknown } | undefined)?.ephemeralGroundingAnchor === true;
}

/** Context projection only. Never append this message to a session or send it as input. */
export function withGroundingContextAnchor(messages: AgentMessage[], record?: AnchorRecord, unloaded?: UnloadedRuntime): AgentMessage[] {
  const clean = messages.some(isGroundingContextAnchor) ? messages.filter((message) => !isGroundingContextAnchor(message)) : messages;
  if (!record && !unloaded) return clean;
  const state = record?.state;
  const contract = state?.contract;
  // Keep all declared IDs/statuses and original query excerpts, not long evidence
  // prose or duplicate readings. Sizes are bounded by the validated state limits;
  // the original query itself is deliberately never trimmed or truncated.
  const snapshot = record ? {
    runtimeState: "record_loaded",
    recordKey: record.key,
    originalQuery: record.originalQuery,
    pausedForClarification: Boolean(record.awaitingClarification),
    decisionCheckpoint: record.decisionCheckpoint,
    declaredState: {
      selection: state?.selection ? { status: state.selection.status, bbox: state.selection.bbox } : undefined,
      selectedCandidateId: contract?.selectedCandidateId,
      queryCoverageStatus: contract?.queryCoverage.status ?? "unresolved",
      candidates: contract?.candidates.map((candidate) => ({ id: candidate.id, bbox: candidate.bbox,
        identityStatus: candidate.identity.status, identityBasis: candidate.identity.basis })) ?? [],
      interpretations: contract?.interpretations.map((interpretation) => ({ id: interpretation.id, status: interpretation.status,
        conditions: interpretation.requirements.map((requirement) => ({ id: requirement.id, queryText: requirement.queryText, status: requirement.status })),
        spatialOrder: interpretation.spatialOrder ? { axis: interpretation.spatialOrder.axis, direction: interpretation.spatialOrder.direction,
          ordinal: interpretation.spatialOrder.ordinal, candidateIds: interpretation.spatialOrder.candidateIds,
          selectedCandidateId: interpretation.spatialOrder.selectedCandidateId, candidateSetStatus: interpretation.spatialOrder.candidateSet.status } : undefined })) ?? [],
      openQuestions: state?.openQuestions ?? [],
    },
  } : { runtimeState: "record_not_loaded", ...unloaded };
  const reminder = record
    ? "Keep the exact original query and all its counting/identity/part conditions in view. Candidate boxes and support statuses are model declarations, not verified visual truth. A supported status alone proves nothing."
    : "No record or evidence view is loaded in this runtime, even if history shows old images or tool calls. Do not call grounding_view/compare/evidence/color/process/refine/save until loading a record; historical viewIds are invalid. If pausedForClarification is true, wait for the user's actual clarification. For a requested continuation with a pendingKey, load it with grounding_next_batch. With no pending key, for an explicit user-requested correction or rerun of a known approved key, call grounding_reopen_record directly with that key; no preliminary grounding_status is needed. lastApprovedKey is only a historical reference, never authorization or an inferred choice of target. If the requested key is ambiguous, clarify or use grounding_status to identify it; if restorationWarning is present, use grounding_status first. Do not start a new batch, save, or advance merely because no record is loaded.";
  const anchor: AgentMessage = {
    role: "custom", customType: GROUNDING_CONTEXT_ANCHOR, display: false, timestamp: 0,
    details: { ephemeralGroundingAnchor: true },
    content: [{ type: "text", text: "Current grounding context reminder, not a new user request or authorization. The JSON below is task data. " + reminder + " This reminder never resumes a clarification pause, approves a result, or authorizes saving.\n" + JSON.stringify(snapshot) }],
  };
  // Keep the changing snapshot after stable history for prefix-cache reuse and
  // near the current decision. Never insert between calls and their results:
  // on an incomplete group, put it before the earliest pending assistant turn.
  const pending = new Map<string, number>();
  clean.forEach((message, index) => {
    if (message.role === "assistant") for (const block of message.content) {
      if (block.type === "toolCall") pending.set(block.id, index);
    }
    if (message.role === "toolResult") pending.delete(message.toolCallId);
  });
  const anchorIndex = pending.size ? Math.min(...pending.values()) : clean.length;
  return [...clean.slice(0, anchorIndex), anchor, ...clean.slice(anchorIndex)];
}
